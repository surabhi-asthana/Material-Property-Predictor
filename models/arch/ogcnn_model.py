import torch
import torch.nn as nn

# Standard properties for the 5-task multi-head output
PROPERTIES = ["formation", "magnetization", "fermi", "bandgap", "hull"]

class OGCNNConvLayer(nn.Module):
    """
    Gated Graph Convolutional Layer with Residual Connections.
    Uses Sigmoid and Softplus gates for non-linear message passing.
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        # Fully connected layer for the combined node and edge features
        self.fc_full  = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid  = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        # Gather neighbor atom features
        atom_nbr_fea = atom_fea[nbr_fea_idx.view(-1)].view(N, M, -1)

        # Concatenate center atom, neighbor atom, and bond features
        total = torch.cat([
            atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
            atom_nbr_fea,
            nbr_fea
        ], dim=2)

        # Apply gated linear unit (GLU) logic
        gated = self.bn1(
            self.fc_full(total).view(-1, 2 * self.atom_fea_len)
        ).view(N, M, 2 * self.atom_fea_len)

        f, c = gated.chunk(2, dim=2)
        # Summation over neighbors (aggregation)
        out  = self.bn2(torch.sum(self.sigmoid(f) * self.softplus(c), dim=1))
        
        # Residual connection
        return self.softplus(atom_fea + out)


class OGCNN5Task(nn.Module):
    """
    OGCNN Architecture with 5 independent output heads.
    Input: 1148-dim (92 basic + 1056 OFM features).
    Encoder: 1148 → 256 → 512.
    """
    def __init__(self, orig_atom_fea_len=1148, nbr_fea_len=41,
                 enc_hidden=256, atom_fea_len=512, n_conv=3, h_fea_len=128):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.properties   = PROPERTIES

        # Encoder: Projects the high-dimensional OFM into the latent space
        self.embedding = nn.Sequential(
            nn.Linear(orig_atom_fea_len, enc_hidden),
            nn.Softplus(),
            nn.Linear(enc_hidden, atom_fea_len),
            nn.Softplus()
        )

        # Shared GNN Convolution layers
        self.convs = nn.ModuleList([
            OGCNNConvLayer(atom_fea_len, nbr_fea_len)
            for _ in range(n_conv)
        ])

        # 5 independent heads for multi-task property prediction
        self.output_heads = nn.ModuleDict({
            prop: nn.Sequential(
                nn.Linear(atom_fea_len, h_fea_len),
                nn.Softplus(),
                nn.Linear(h_fea_len, 1)
            )
            for prop in PROPERTIES
        })

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crys_idx):
        # 1. Feature Embedding
        atom_fea = self.embedding(atom_fea)

        # 2. Graph Convolutions
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        # 3. Global Mean Pooling per Crystal
        crys_fea = torch.stack([
            atom_fea[idx.to(atom_fea.device)].mean(dim=0)
            for idx in crys_idx
        ])

        # 4. Multi-task output heads
        return {
            prop: self.output_heads[prop](crys_fea).squeeze(1)
            for prop in PROPERTIES
        }

if __name__ == "__main__":
    # Sanity check for parameter count and architecture
    model_test = OGCNN5Task()
    print(f"OGCNN5Task defined ✓")
    print(f"Total Parameters: {sum(p.numel() for p in model_test.parameters()):,}")
    print(f"Target Tasks: {PROPERTIES}")