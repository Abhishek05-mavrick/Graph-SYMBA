import re
import torch
from torch_geometric.data import Data

def build_edge_attr(src_node_feats, tgt_node_feats, edge_index, edge_feat_dim):
    edge_attrs = []
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0, i].item()
        tgt_idx = edge_index[1, i].item()
        src_feat = src_node_feats[src_idx]  # [p_type, momentum, spin, 1.0, ...]
        tgt_feat = tgt_node_feats[tgt_idx]
        # Encode pairwise physics: type difference, momentum product, spin product
        edge_feat = [
            src_feat[0],          # source particle type
            tgt_feat[0],          # target particle type
            abs(src_feat[0] - tgt_feat[0]),  # type difference (interaction kind)
            src_feat[1] * tgt_feat[1],        # momentum product (off-shell flag interaction)
            src_feat[2] * tgt_feat[2],        # spin product
            float(src_idx == tgt_idx),        # self-loop indicator
        ]
        edge_feat += [0.0] * (edge_feat_dim - len(edge_feat))
        edge_attrs.append(edge_feat)
    return torch.tensor(edge_attrs, dtype=torch.float)

def _extract_particle_symbol(particle_str):
    """Extract particle symbol before the parenthesis, then match known symbols."""
    prefix = particle_str.split('(')[0].strip() if '(' in particle_str else particle_str.strip()
    tokens = prefix.split()
    symbol = tokens[-1] if tokens else ''
    type_map = {'e': 1.0, 'A': 2.0, 'mu': 3.0, 'u': 4.0, 'd': 5.0, 's': 6.0, 'c': 7.0, 'b': 8.0, 't': 9.0, 'tt': 9.0}
    return type_map.get(symbol, 0.0)

def topology_to_pyg(topology_str: str, config):
    """Real Feynman → PyG converter with Explicit Physics Priors"""
    edges = []
    node_features = []
    vertex_blocks = str(topology_str).split('Vertex ')[1:] 
    global_node_idx = 0
    
    for block in vertex_blocks:
        raw_particles = [p.strip() for p in block.split(',') if '(' in p]
        
        current_vertex_node_indices = []
        
        for p in raw_particles:
            p_type = _extract_particle_symbol(p)
            momentum = 1.0 if 'OffShell' in p else 0.0
            spin = 0.5 if p_type in (1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) else 1.0
            
            feat = [p_type, momentum, spin, 1.0]  # 4 explicit physics features
            feat_padded = feat + [0.0] * (config.NODE_FEATURE_DIM - 4)
            node_features.append(feat_padded)
            
            current_vertex_node_indices.append(global_node_idx)
            global_node_idx += 1
            
        # Fully connect all particles interacting inside this specific vertex
        for i in current_vertex_node_indices:
            for j in current_vertex_node_indices:
                edges.append([i, j]) # Includes self-loops (i==j) for stability

    # Fallback if parsing fails (prevents DataLoader crashes)
    if not node_features:
        node_features = [[0.0] * config.NODE_FEATURE_DIM]
        edges = [[0, 0]]

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=build_edge_attr(node_features, node_features, edge_index, config.EDGE_FEATURE_DIM)
    )