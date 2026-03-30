import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
from .components import KANFeedForward

class PhysicsInformedGraphConv(MessagePassing):
    def __init__(self, d_model, edge_feat_dim=32, use_kan=False, grid_size=8, dropout=0.1):
        super().__init__(aggr='add') 
        self.use_kan = use_kan
        
        if self.use_kan:
            self.message_mlp = KANFeedForward(d_model * 2 + edge_feat_dim, d_model, grid_size=grid_size, dropout=dropout)
        else:
            self.message_mlp = nn.Sequential(
                nn.Linear(d_model * 2 + edge_feat_dim, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
            
        self.update_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            tmp = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(tmp)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(tmp)

class GraphSYMKANEncoder(nn.Module):
    def __init__(self, config, use_kan=False):
        super().__init__()
        self.node_emb = nn.Linear(config.NODE_FEATURE_DIM, config.GNN_HIDDEN)
        
        self.convs = nn.ModuleList([
            PhysicsInformedGraphConv(
                d_model=config.GNN_HIDDEN, 
                edge_feat_dim=config.EDGE_FEATURE_DIM,
                use_kan=use_kan, 
                grid_size=config.KAN_GRID_SIZE, 
                dropout=config.DROPOUT
            ) 
            for _ in range(config.GNN_LAYERS)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(config.GNN_HIDDEN) for _ in range(config.GNN_LAYERS)])
        self.proj = nn.Linear(config.GNN_HIDDEN, config.D_MODEL)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h = norm(h + h_new)
            
        h_dense, mask = to_dense_batch(h, batch)
        return self.proj(h_dense), mask