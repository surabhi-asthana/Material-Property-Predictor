# material-predictor-suite/models/predictor.py

import os
import sys
import json
import numpy as np
import torch
from collections import defaultdict
from pymatgen.core import Structure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.arch.icgcnn_model import iCGCNN
from models.arch.alignn_model import ALIGNNModel
from models.arch.ogcnn_model  import OGCNN5Task

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "weights")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

# 5 properties returned to frontend — same keys across all 3 models
OUTPUT_KEYS = ["formation_energy", "band_gap", "fermi_energy", "hull_distance", "magnetization"]


def gaussian_expansion(distances, dmin=0, dmax=8, step=0.2, var=0.2):
    centers = np.arange(dmin, dmax + step, step)
    return np.exp(
        -(distances[:, None] - centers[None, :]) ** 2 / var ** 2
    ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# iCGCNN — 7 heads trained, we use 5 (skip bulk, shear)
# ══════════════════════════════════════════════════════════════════════
class ICGCNNPredictor:
    def __init__(self):
        with open(os.path.join(CONFIG_DIR, "icgcnn_config.json")) as f:
            cfg = json.load(f)

        with open(cfg["atom_init_path"]) as f:
            self.atom_init = json.load(f)

        with open(cfg["scaler_path"]) as f:
            self.scalers = json.load(f)

        self.model = iCGCNN(
            orig_atom_fea_len = cfg.get("orig_atom_fea_len", 92),
            nbr_fea_len       = cfg.get("nbr_fea_len", 41),
            atom_fea_len      = cfg.get("atom_fea_len", 64),
            n_conv            = cfg.get("n_conv", 3),
            h_fea_len         = cfg.get("h_fea_len", 128),
        ).to(DEVICE)

        ckpt  = torch.load(os.path.join(WEIGHT_DIR, "icgcnn.pth"),
                           map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        print("✓ iCGCNN loaded")

    def _build_graph(self, struct, max_nbrs=12, radius=8.0):
        all_nbrs = struct.get_all_neighbors(radius, include_index=True)
        N, M     = len(struct), max_nbrs
        nbr_fea_idx = np.zeros((N, M), dtype=np.int64)
        nbr_fea     = np.zeros((N, M, 41), dtype=np.float32)

        for i, nbrs in enumerate(all_nbrs):
            for col, nbr in enumerate(sorted(nbrs, key=lambda x: x[1])[:M]):
                nbr_fea_idx[i, col] = nbr[2]
                nbr_fea[i, col]     = gaussian_expansion(np.array([nbr[1]]))[0]

        atom_fea = [
            self.atom_init.get(str(site.specie.Z), [0.0] * 92)
            for site in struct
        ]
        return (
            torch.tensor(atom_fea,    dtype=torch.float32),
            torch.tensor(nbr_fea,     dtype=torch.float32),
            torch.tensor(nbr_fea_idx, dtype=torch.long),
        )

    def predict(self, cif_path):
        struct = Structure.from_file(cif_path)
        atom_fea, nbr_fea, nbr_fea_idx = self._build_graph(struct)
        crys_idx = [torch.arange(len(struct))]

        with torch.no_grad():
            out = self.model(
                atom_fea.to(DEVICE),
                nbr_fea.to(DEVICE),
                nbr_fea_idx.to(DEVICE),
                crys_idx
            )

        def denorm(key, val):
            sc = self.scalers.get(key, {})
            return round(float(val) * sc.get("std", 1.0) + sc.get("mean", 0.0), 4)

        return {
            "formation_energy": denorm("formation",     out["formation"][0].item()),
            "band_gap":         denorm("bandgap",       out["bandgap"][0].item()),
            "fermi_energy":     denorm("fermi",         out["fermi"][0].item()),
            "hull_distance":    denorm("hull",          out["hull"][0].item()),
            "magnetization":    denorm("magnetization", out["magnetization"][0].item()),
        }


# ══════════════════════════════════════════════════════════════════════
# ALIGNN — one .pt file per property, all 5
# ══════════════════════════════════════════════════════════════════════

class ALIGNNPredictor:
    PROP_MAP = {
        "formation_energy": ("alignn_formation_energy.pt",     None),
        "band_gap":         ("alignn_band_gap.pt",       "softplus"),
        "fermi_energy":     ("alignn_fermi_energy.pt",         None),
        "hull_distance":    ("alignn_hull.pt",          None),
        "magnetization":    ("alignn_magnetization.pt", None),
    }

    def __init__(self):
        with open(os.path.join(CONFIG_DIR, "alignn_config.json")) as f:
            cfg = json.load(f)

        with open(cfg["atom_init_path"]) as f:
            self.atom_init = json.load(f)

        self.scalers = cfg.get("scalers", {})
        atom_dim     = cfg.get("atom_dim", 92)

        self.models = {}
        for prop, (fname, out_act) in self.PROP_MAP.items():
            path = os.path.join(WEIGHT_DIR, fname)
            try:
                m = ALIGNNModel(
                    atom_dim          = atom_dim,
                    hidden            = cfg.get("hidden", 128),
                    alignn_layers     = cfg.get("alignn_layers", 3),
                    gcn_layers        = cfg.get("gcn_layers", 3),
                    dropout           = 0.0,
                    output_activation = out_act,
                ).to(DEVICE)

                state = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                state = {k.replace("module.", ""): v for k, v in state.items()}
                m.load_state_dict(state)
                m.eval()
                self.models[prop] = m
                print(f"✓ ALIGNN [{prop}] loaded")

            except FileNotFoundError:
                print(f"❌ ALIGNN [{prop}] — file not found: {fname}")
            except Exception as e:
                print(f"❌ ALIGNN [{prop}] — failed: {e}")

    # def __init__(self):
    #     with open(os.path.join(CONFIG_DIR, "alignn_config.json")) as f:
    #         cfg = json.load(f)

    #     with open(cfg["atom_init_path"]) as f:
    #         self.atom_init = json.load(f)

    #     self.scalers = cfg.get("scalers", {})
    #     atom_dim     = cfg.get("atom_dim", 92)

    #     self.models = {}
    #     for prop, (fname, out_act) in self.PROP_MAP.items():
    #         m = ALIGNNModel(
    #             atom_dim          = atom_dim,
    #             hidden            = cfg.get("hidden", 128),
    #             alignn_layers     = cfg.get("alignn_layers", 3),
    #             gcn_layers        = cfg.get("gcn_layers", 3),
    #             dropout           = 0.0,
    #             output_activation = out_act,
    #         ).to(DEVICE)
    #         try:
    #             state = torch.load(weight_path, map_location="cpu", weights_only=False)
    #         except RuntimeError as e:
    #             if "hasRecord" in str(e):
    #             # Legacy pickle format — re-serialize on the fly
    #                 import pickle
    #                 with open(weight_path, "rb") as f:
    #                     state = pickle.load(f, encoding="latin1")
    #                 torch.save(state, weight_path)   # overwrite so it only happens once
    #                 print(f"  ↻ Re-saved {fname} in current PyTorch format")
    #             else:
    #                 print(f"  ✗ ALIGNN [{prop}] weight load failed: {e}")
    #                 continue

    #         if isinstance(state, dict) and "state_dict" in state:
    #             state = state["state_dict"]
    #         state = {k.replace("module.", ""): v for k, v in state.items()}
    #         m.load_state_dict(state)
    #         m.eval()
    #         self.models[prop] = m
    #         print(f"✓ ALIGNN [{prop}] loaded")


    def _build_graph(self, struct, max_nbrs=12, radius=8.0):
        from torch_geometric.data import Data

        atom_fea = torch.tensor([
            self.atom_init.get(str(site.specie.Z), [0.0] * 92)
            for site in struct
        ], dtype=torch.float)

        all_nbrs = struct.get_all_neighbors(radius, include_index=True)
        src, dst, dist, nbr_coords = [], [], [], []
        for i, nbrs in enumerate(all_nbrs):
            for nbr in sorted(nbrs, key=lambda x: x[1])[:max_nbrs]:
                src.append(i)
                dst.append(nbr[2])
                dist.append(nbr[1])
                nbr_coords.append(nbr[0].coords)

        if not src:
            src = [0]; dst = [0]; dist = [0.0]
            nbr_coords = [struct[0].coords]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(gaussian_expansion(np.array(dist)), dtype=torch.float)

        center_edges = defaultdict(list)
        for idx, (s, d) in enumerate(zip(src, dst)):
            center_edges[d].append((idx, nbr_coords[idx]))

        a_src, a_dst, angles = [], [], []
        for center, incoming in center_edges.items():
            pos_c = struct[center].coords
            vecs = [
                (nc - pos_c) / (np.linalg.norm(nc - pos_c) + 1e-8)
                for _, nc in incoming
            ]
            for i in range(len(incoming)):
                for j in range(i + 1, len(incoming)):
                    angles.append(float(np.clip(np.dot(vecs[i], vecs[j]), -1, 1)))
                    a_src.append(incoming[i][0])
                    a_dst.append(incoming[j][0])

        if a_src:
            line_index = torch.tensor([a_src, a_dst], dtype=torch.long)
            line_attr  = torch.tensor(angles, dtype=torch.float).unsqueeze(1)
        else:
            line_index = torch.zeros((2, 1), dtype=torch.long)
            line_attr  = torch.zeros((1, 1), dtype=torch.float)

        return Data(
            x=atom_fea, edge_index=edge_index, edge_attr=edge_attr,
            line_index=line_index, line_attr=line_attr,
            batch=torch.zeros(len(struct), dtype=torch.long)
        )
    # def _build_graph(self, struct, max_nbrs=12, radius=8.0):
    #     from torch_geometric.data import Data

    #     atom_fea = torch.tensor([
    #         self.atom_init.get(str(site.specie.Z), [0.0] * 92)
    #         for site in struct
    #     ], dtype=torch.float)

        # all_nbrs = struct.get_all_neighbors(radius, include_index=True)
        # src, dst, dist = [], [], []
        # for i, nbrs in enumerate(all_nbrs):
        #     for nbr in sorted(nbrs, key=lambda x: x[1])[:max_nbrs]:
        #         src.append(i); dst.append(nbr[2]); dist.append(nbr[1])
        # if not src:
        #     src = [0]; dst = [0]; dist = [0.0]

        # edge_index = torch.tensor([src, dst], dtype=torch.long)
        # edge_attr  = torch.tensor(gaussian_expansion(np.array(dist)), dtype=torch.float)

        # coords = np.array([struct[i].coords for i in range(len(struct))])
        # center_edges = defaultdict(list)
        # for idx, (s, d) in enumerate(zip(src, dst)):
        #     center_edges[d].append((idx, s))

        # a_src, a_dst, angles = [], [], []
        # for center, incoming in center_edges.items():
        #     pos_c = coords[center]
        #     vecs  = [
        #         (coords[si] - pos_c) / (np.linalg.norm(coords[si] - pos_c) + 1e-8)
        #         for _, si in incoming
        #     ]
        #     for i in range(len(incoming)):
        #         for j in range(i + 1, len(incoming)):
        #             angles.append(float(np.clip(np.dot(vecs[i], vecs[j]), -1, 1)))
        #             a_src.append(incoming[i][0])
        #             a_dst.append(incoming[j][0])
        # if a_src:
        #     line_index = torch.tensor([a_src, a_dst], dtype=torch.long)
        #     line_attr  = torch.tensor(angles, dtype=torch.float).unsqueeze(1)
        # else:
        #     line_index = torch.zeros((2, 1), dtype=torch.long)
        #     line_attr  = torch.zeros((1, 1), dtype=torch.float)

        # return Data(
        #     x=atom_fea, edge_index=edge_index, edge_attr=edge_attr,
        #     line_index=line_index, line_attr=line_attr,
        #     batch=torch.zeros(len(struct), dtype=torch.long)
        # )

    def predict(self, cif_path):
        struct = Structure.from_file(cif_path)
        data   = self._build_graph(struct).to(DEVICE)

        results = {}
        for prop, model in self.models.items():
            with torch.no_grad():
                val = model(data).item()
            sc  = self.scalers.get(prop, {})
            results[prop] = round(val * sc.get("std", 1.0) + sc.get("mean", 0.0), 4)
        return results


# ══════════════════════════════════════════════════════════════════════
# OGCNN — 5 heads, all properties in one model
# ══════════════════════════════════════════════════════════════════════
class OGCNNPredictor:
    def __init__(self):
        with open(os.path.join(CONFIG_DIR, "ogcnn_config.json")) as f:
            cfg = json.load(f)

        with open(cfg["scaler_path"]) as f:
            self.scalers = json.load(f)

        self.ofm_dim = cfg.get("orig_atom_fea_len", 1148)
        self.model   = OGCNN5Task(
            orig_atom_fea_len = self.ofm_dim,
            nbr_fea_len       = cfg.get("nbr_fea_len", 41),
            enc_hidden        = cfg.get("enc_hidden", 256),
            atom_fea_len      = cfg.get("atom_fea_len", 512),
            n_conv            = cfg.get("n_conv", 3),
            h_fea_len         = cfg.get("h_fea_len", 128),
        ).to(DEVICE)

        ckpt  = torch.load(os.path.join(WEIGHT_DIR, "ogcnn.pth"),
                           map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state)
        self.model.eval()
        print("✓ OGCNN loaded")

    def _build_graph(self, struct, max_nbrs=12, radius=8.0):
        N = len(struct)
        atom_fea    = np.zeros((N, self.ofm_dim), dtype=np.float32)
        nbr_idx_arr = np.zeros((N, max_nbrs), dtype=np.int64)
        nbr_fea_arr = np.zeros((N, max_nbrs, 41), dtype=np.float32)

        for i in range(N):
            atom_fea[i, min(struct[i].specie.Z, self.ofm_dim - 1)] = 1.0

        for i, nbrs in enumerate(struct.get_all_neighbors(8.0, include_index=True)):
            for col, nbr in enumerate(sorted(nbrs, key=lambda x: x[1])[:max_nbrs]):
                nbr_idx_arr[i, col] = nbr[2]
                nbr_fea_arr[i, col] = gaussian_expansion(np.array([nbr[1]]))[0]

        return (
            torch.tensor(atom_fea,    dtype=torch.float32),
            torch.tensor(nbr_fea_arr, dtype=torch.float32),
            torch.tensor(nbr_idx_arr, dtype=torch.long),
            [torch.arange(N)],
        )

    def predict(self, cif_path):
        struct = Structure.from_file(cif_path)
        atom_fea, nbr_fea, nbr_fea_idx, crys_idx = self._build_graph(struct)

        with torch.no_grad():
            out = self.model(
                atom_fea.to(DEVICE),
                nbr_fea.to(DEVICE),
                nbr_fea_idx.to(DEVICE),
                crys_idx
            )

        def denorm(prop, val):
            sc = self.scalers.get(prop, {})
            return round(float(val) * sc.get("std", 1.0) + sc.get("mean", 0.0), 4)

        return {
            "formation_energy": denorm("formation",     out["formation"][0].item()),
            "band_gap":         denorm("bandgap",       out["bandgap"][0].item()),
            "fermi_energy":     denorm("fermi",         out["fermi"][0].item()),
            "hull_distance":    denorm("hull",          out["hull"][0].item()),
            "magnetization":    denorm("magnetization", out["magnetization"][0].item()),
        }


# ══════════════════════════════════════════════════════════════════════
# Unified entry point called by app.py
# ══════════════════════════════════════════════════════════════════════
_predictors = {}

def _load_predictors():
    global _predictors
    if _predictors:
        return
    print("Loading models...")
    for name, cls in [("iCGCNN", ICGCNNPredictor),
                      ("ALIGNN",  ALIGNNPredictor),
                      ("OGCNN",   OGCNNPredictor)]:
        try:
            _predictors[name] = cls()
        except Exception as e:
            print(f"⚠ {name} load failed: {e}")
    print(f"Models ready: {list(_predictors.keys())}")


def predict(cif_path: str, model_name: str) -> dict:
    _load_predictors()
    if model_name not in _predictors:
        raise ValueError(f"'{model_name}' not loaded. Available: {list(_predictors.keys())}")
    return _predictors[model_name].predict(cif_path)