import torch
import torch.nn as nn

class iConvLayer(nn.Module):
    """
    Improved Convolution Layer from crystals-icgcnn.ipynb.
    Features: Two-body interactions, Three-body interactions, and Edge updates.
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len  = nbr_fea_len

        # Two-body interaction (standard gated convolution)
        self.fc_full = nn.Linear(2*atom_fea_len + nbr_fea_len, 2*atom_fea_len)

        # Three-body interaction (incorporates triplet geometry)
        self.fc_three = nn.Linear(3*atom_fea_len + 2*nbr_fea_len, 2*atom_fea_len)

        # Edge update (updates bond features during message passing)
        self.fc_edge = nn.Linear(2*atom_fea_len + nbr_fea_len, nbr_fea_len)

        self.sigmoid  = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.bn3 = nn.BatchNorm1d(nbr_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, triplet_i, triplet_j, triplet_k):
        N, M = nbr_fea_idx.shape

        # --- TWO-BODY INTERACTION ---
        atom_nbr_fea = atom_fea[nbr_fea_idx.view(-1)].view(N, M, -1)
        total = torch.cat([
            atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea,
            nbr_fea
        ], dim=2)

        gated = self.bn1(self.fc_full(total).view(-1, 2*self.atom_fea_len)).view(N, M, 2*self.atom_fea_len)
        f, c = gated.chunk(2, dim=2)
        two_body = self.bn2(torch.sum(self.sigmoid(f) * self.softplus(c), dim=1))

        # --- THREE-BODY INTERACTION ---
        three_body = torch.zeros_like(atom_fea)
        if len(triplet_i) > 0:
            ai, aj, ak = atom_fea[triplet_i], atom_fea[triplet_j], atom_fea[triplet_k]
            eij = nbr_fea[triplet_i, 0, :]
            eik = nbr_fea[triplet_i, 1 % M, :] # Simplified triplet edge indexing

            three_in = torch.cat([ai, aj, ak, eij, eik], dim=1)
            three_out = self.fc_three(three_in)
            f3, c3 = three_out.chunk(2, dim=1)
            three_body.index_add_(0, triplet_i, self.sigmoid(f3) * self.softplus(c3))

        # Combine interactions with residual connection
        out = self.softplus(atom_fea + two_body + three_body)

        # --- EDGE UPDATE ---
        edge_in = torch.cat([
            atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea,
            nbr_fea
        ], dim=2)
        new_nbr_fea = self.softplus(self.bn3(self.fc_edge(edge_in).view(-1, self.nbr_fea_len)).view(N, M, self.nbr_fea_len))

        return out, new_nbr_fea


class iCGCNN(nn.Module):
    """
    Multitask iCGCNN predicting 7 properties simultaneously.
    """
    def __init__(self, orig_atom_fea_len=92, nbr_fea_len=41, atom_fea_len=64, n_conv=3, h_fea_len=128):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([iConvLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.softplus = nn.Softplus()

        # Property heads matching the 7-property training logic
        self.head_formation     = nn.Linear(h_fea_len, 1)
        self.head_bandgap       = nn.Linear(h_fea_len, 1)
        self.head_hull          = nn.Linear(h_fea_len, 1)
        self.head_bulk          = nn.Linear(h_fea_len, 1)
        self.head_shear         = nn.Linear(h_fea_len, 1)
        self.head_fermi         = nn.Linear(h_fea_len, 1)
        self.head_magnetization = nn.Linear(h_fea_len, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, triplet_i=None, triplet_j=None, triplet_k=None):
        atom_fea = self.embedding(atom_fea)

        for conv in self.convs:
            atom_fea, nbr_fea = conv(
                atom_fea, nbr_fea, nbr_fea_idx,
                triplet_i if triplet_i is not None else torch.LongTensor([]),
                triplet_j if triplet_j is not None else torch.LongTensor([]),
                triplet_k if triplet_k is not None else torch.LongTensor([])
            )

        # Global Mean Pooling
        crys_fea = self.softplus(self.conv_to_fc(torch.stack([atom_fea[idx].mean(dim=0) for idx in crystal_atom_idx])))

        return {
            "formation":     self.head_formation(crys_fea),
            "bandgap":       self.head_bandgap(crys_fea),
            "hull":          self.head_hull(crys_fea),
            "bulk":          self.head_bulk(crys_fea),
            "shear":         self.head_shear(crys_fea),
            "fermi":         self.head_fermi(crys_fea),
            "magnetization": self.head_magnetization(crys_fea)
        }