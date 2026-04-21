import torch
import json
import numpy as np
import os
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import VoronoiNN
from collections import defaultdict

class GraphGenerator:
    def __init__(self, atom_init_path="models/weights/atom_init.json"):
        if not os.path.exists(atom_init_path):
            raise FileNotFoundError(f"Missing {atom_init_path}")
            
        with open(atom_init_path) as f:
            data = json.load(f)
            self.atom_features = {int(k): np.array(v) for k, v in data.items()}
        
        self.vnn = VoronoiNN(cutoff=8.0, allow_pathological=True)
        self.orbital_cache = {}

    # --- 1. ORBITAL FIELD MATRIX LOGIC (1056-dim) ---
    def _get_orbital_binary_vector(self, symbol):
        """Constructs 32-dim binary vector for valence orbital slots (s, p, d, f)."""
        if symbol in self.orbital_cache: return self.orbital_cache[symbol]
        vec = np.zeros(32, dtype=np.float32)
        try:
            el = Element(symbol)
            full_es = el.full_electronic_structure 
            counts = {'s':0, 'p':0, 'd':0, 'f':0}
            for _, shell, occ in full_es:
                if shell in counts: counts[shell] += occ
            
            # Fill slots sequentially
            for i in range(min(int(counts['s']), 2)): vec[i] = 1.0
            for i in range(min(int(counts['p']), 6)): vec[2+i] = 1.0
            for i in range(min(int(counts['d']), 10)): vec[8+i] = 1.0
            for i in range(min(int(counts['f']), 14)): vec[18+i] = 1.0
        except: pass
        self.orbital_cache[symbol] = vec
        return vec

    def _compute_ofm_features(self, struct, site_idx, nbrs):
        """Implements OGCNN Eq. 1: Orbital interactions weighted by distance and angle."""
        Oc = self._get_orbital_binary_vector(struct[site_idx].specie.symbol)
        interaction_matrix = np.zeros((32, 32), dtype=np.float32)
        
        for nbr in nbrs:
            On = self._get_orbital_binary_vector(struct[nbr["site_index"]].specie.symbol)
            theta = float(nbr.get("weight", 1.0)) # Solid angle
            dist = struct.get_distance(site_idx, nbr["site_index"])
            zeta = 1.0 / (dist**2) if dist > 0.1 else 100.0 # 1/r^2 weighting
            
            interaction_matrix += np.outer(Oc, On) * (theta * zeta)
            
        # Flatten and concatenate center atom's 1D vector
        return np.concatenate([interaction_matrix, Oc.reshape(32, 1)], axis=1).flatten()

    # --- 2. GAUSSIAN EXPANSION (41-dim) ---
    def _gaussian_expansion(self, distances):
        centers = np.arange(0, 8.2, 0.2)
        dist_arr = np.array(distances)
        return np.exp(-(dist_arr[..., np.newaxis] - centers)**2 / 0.2**2)

    def process_cif(self, cif_path):
        """Unified CIF processor for all GNN architectures."""
        struct = Structure.from_file(cif_path)
        n_atoms = len(struct)

        # Build Atomic Features
        basic_feas = []  # 92-dim
        full_feas = []   # 1148-dim (92 basic + 1056 OFM)
        all_nbrs = []

        for i in range(n_atoms):
            basic = self.atom_features.get(struct[i].specie.number, np.zeros(92))
            basic_feas.append(basic)
            
            try:
                nbrs = self.vnn.get_nn_info(struct, i)
            except:
                # Fallback for complex structures
                nbrs = [{"site_index": n.index, "weight": 1.0} for n in struct.get_neighbors(8.0, i)]
            
            all_nbrs.append(nbrs)
            ofm = self._compute_ofm_features(struct, i, nbrs)
            full_feas.append(np.concatenate([basic, ofm]))

        # Prepare connectivity and bond features
        src, dst, dists = [], [], []
        nbr_fea_idx = []
        for i, nbrs in enumerate(all_nbrs):
            # Sort by distance and take top 12
            nbrs = sorted(nbrs, key=lambda x: struct.get_distance(i, x["site_index"]))[:12]
            idx_list = []
            for n in nbrs:
                src.append(i); dst.append(n['site_index'])
                dists.append(struct.get_distance(i, n['site_index']))
                idx_list.append(n['site_index'])
            while len(idx_list) < 12: # Padding
                idx_list.append(0)
            nbr_fea_idx.append(idx_list)

        # Line Graph (For ALIGNN)
        center_edges = defaultdict(list)
        for idx, (s, d) in enumerate(zip(src, dst)): center_edges[d].append((idx, s))
        l_src, l_dst, l_angles = [], [], []
        for center, incoming in center_edges.items():
            for i in range(len(incoming)):
                for j in range(i + 1, len(incoming)):
                    idx_i, s_i = incoming[i]
                    idx_j, s_j = incoming[j]
                    v1 = struct[s_i].coords - struct[center].coords
                    v2 = struct[s_j].coords - struct[center].coords
                    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    l_src.append(idx_i); l_dst.append(idx_j); l_angles.append(cos_ang)

        # Triplets (For iCGCNN)
        ti, tj, tk = [], [], []
        for i, idxs in enumerate(nbr_fea_idx):
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    ti.append(i); tj.append(idxs[a]); tk.append(idxs[b])

        return {
            # ALIGNN/iCGCNN basic node features
            'x': torch.FloatTensor(np.array(basic_feas, dtype=np.float32)),
            'edge_index': torch.LongTensor([src, dst]),
            'edge_attr': torch.FloatTensor(self._gaussian_expansion(dists)),
            'line_index': torch.LongTensor([l_src, l_dst]),
            'line_attr': torch.FloatTensor(l_angles).unsqueeze(1),
            'batch': torch.zeros(n_atoms, dtype=torch.long),
            
            # OGCNN/iCGCNN specific features
            'af': torch.FloatTensor(np.array(full_feas, dtype=np.float32)), # 1148-dim
            'nf': torch.FloatTensor(self._gaussian_expansion(dists)).view(len(src), 1, -1),
            'ni': torch.LongTensor(nbr_fea_idx),
            'crys_idx': [torch.arange(n_atoms)],
            'ti': torch.LongTensor(ti), 'tj': torch.LongTensor(tj), 'tk': torch.LongTensor(tk)
        }