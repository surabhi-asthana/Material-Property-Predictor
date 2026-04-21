import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool

class EdgeGatedConv(MessagePassing):
    """Gated Graph Convolution used for both Atom and Line graphs."""
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.src_gate  = nn.Linear(node_dim, edge_dim)
        self.dst_gate  = nn.Linear(node_dim, edge_dim)
        self.edge_gate = nn.Linear(edge_dim,  edge_dim)
        self.bn_edge   = nn.BatchNorm1d(edge_dim)
        self.src_msg   = nn.Linear(node_dim, node_dim)
        self.edge_msg  = nn.Linear(edge_dim,  node_dim)
        self.bn_node   = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        gate = torch.sigmoid(
            self.src_gate(x[row]) + self.dst_gate(x[col]) + self.edge_gate(edge_attr)
        )
        edge_attr_new = self.bn_edge(F.silu(gate * edge_attr))
        x_new = self.bn_node(F.silu(self.propagate(edge_index, x=x, edge_attr=edge_attr_new)))
        return x + x_new, edge_attr + edge_attr_new

    def message(self, x_j, edge_attr):
        return self.src_msg(x_j) + self.edge_msg(edge_attr)

class ALIGNNLayer(nn.Module):
    """A single ALIGNN block updating bond features then atom features."""
    def __init__(self, node_dim, edge_dim, angle_dim):
        super().__init__()
        self.line_conv = EdgeGatedConv(edge_dim,  angle_dim)
        self.atom_conv = EdgeGatedConv(node_dim,  edge_dim)

    def forward(self, x, edge_index, edge_attr, line_index, line_attr):
        edge_attr, line_attr = self.line_conv(edge_attr, line_index, line_attr)
        x, edge_attr         = self.atom_conv(x, edge_index, edge_attr)
        return x, edge_attr, line_attr

class ALIGNNModel(nn.Module):
    """
    Unified ALIGNN Model matching the 100-epoch training notebook.
    Default hidden=128 optimized for i3 processors.
    """
    def __init__(self, atom_dim, edge_dim=41, angle_dim=1,
                 hidden=128, alignn_layers=3, gcn_layers=3,
                 dropout=0.1, output_activation=None):
        super().__init__()
        self.atom_embed  = nn.Linear(atom_dim, hidden)
        self.edge_embed  = nn.Linear(edge_dim,  hidden)
        self.angle_embed = nn.Linear(angle_dim, hidden // 4)

        self.alignn_layers = nn.ModuleList([
            ALIGNNLayer(hidden, hidden, hidden // 4) for _ in range(alignn_layers)
        ])
        self.gcn_layers = nn.ModuleList([
            EdgeGatedConv(hidden, hidden) for _ in range(gcn_layers)
        ])

        layers = [
            nn.Linear(hidden, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        ]
        if output_activation == "softplus":
            layers.append(nn.Softplus())
        self.output_proj = nn.Sequential(*layers)

    def forward(self, data):
        x         = self.atom_embed(data.x)
        edge_attr = self.edge_embed(data.edge_attr)
        line_attr = self.angle_embed(data.line_attr)

        for layer in self.alignn_layers:
            x, edge_attr, line_attr = layer(x, data.edge_index, edge_attr, data.line_index, line_attr)
        for layer in self.gcn_layers:
            x, edge_attr = layer(x, data.edge_index, edge_attr)

        x = global_mean_pool(x, data.batch)
        return self.output_proj(x).squeeze(-1)