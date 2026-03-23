import re
import torch
from torch_geometric.data import Data

def topology_to_pyg(topology_str: str, config):
    """Real Feynman → PyG converter with Explicit Physics Priors"""
    edges = []
    node_features = []
    
    # Split safely by the word 'Vertex' to handle the ML4SCI comma-separated format
    vertex_blocks = str(topology_str).split('Vertex ')[1:] 
    
    global_node_idx = 0
    
    for block in vertex_blocks:
        # Extract the raw particle strings (e.g., "OffShell mu(X_3)") from this vertex
        raw_particles = [p.strip() for p in block.split(',') if '(' in p]
        
        current_vertex_node_indices = []
        
        for p in raw_particles:
            # Encode particle type (expanded to handle Task 1.2 particles)
            p_type = 1.0 if 'e' in p else 2.0 if 'A' in p else 3.0 if 'mu' in p else 4.0 if 'u' in p else 0.0
            momentum = 1.0 if 'OffShell' in p else 0.0
            spin = 0.5 if ('e' in p or 'mu' in p or 'u' in p) else 1.0
            
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
        edge_attr=torch.zeros(edge_index.size(1), config.EDGE_FEATURE_DIM)
    )