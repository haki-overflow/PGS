"""Neural models: an MLP on flat features and three GNNs on graphs.

Every model outputs one logit per sample (higher = more likely insecure).
"""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_max_pool, global_mean_pool


class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GNN(nn.Module):
    """Encoder -> `layers` message-passing layers with residuals -> mean+max pool -> head.

    GCN and GraphSAGE see bus features and topology only. GATv2 also reads branch
    features (impedance, rating, pre-outage loading and flow) in its attention.
    no_topology=True drops every edge (ablation): buses then never exchange messages.
    """

    def __init__(self, conv, d_node, d_edge, hidden=64, layers=3, dropout=0.1, use_edges=True,
                 no_topology=False):
        super().__init__()
        self.conv_type = conv
        self.no_topology = no_topology
        self.use_edges = use_edges and conv == "gat" and not no_topology
        self.enc = nn.Linear(d_node, hidden)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if conv == "gcn":
                self.convs.append(GCNConv(hidden, hidden))
            elif conv == "sage":
                self.convs.append(SAGEConv(hidden, hidden))
            elif conv == "gat":
                self.convs.append(GATv2Conv(hidden, hidden // 4, heads=4,
                                            edge_dim=d_edge if self.use_edges else None))
            else:
                raise ValueError(conv)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data):
        h = torch.relu(self.enc(data.x))
        edge_index = data.edge_index[:, :0] if self.no_topology else data.edge_index
        for conv, norm in zip(self.convs, self.norms):
            if self.use_edges:
                m = conv(h, edge_index, data.edge_attr)
            else:
                m = conv(h, edge_index)
            h = norm(h + self.drop(torch.relu(m)))
        g = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch)], dim=-1)
        return self.head(g).squeeze(-1)
