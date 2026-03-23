import torch
import torch.nn as nn
from .encoder import GraphSYMKANEncoder
from .decoder import KANTransformerDecoder

class GraphSYMKANModel(nn.Module):
    def __init__(self, config, tgt_vocab_size, use_kan_in_gnn=False): 
        super(GraphSYMKANModel, self).__init__()
        
        self.graph_encoder = GraphSYMKANEncoder(config, use_kan=use_kan_in_gnn) 
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.D_MODEL)
        
        self.decoder = KANTransformerDecoder(
            config.N_LAYERS, config.D_MODEL, config.N_HEADS, 
            config.D_FF, config.DROPOUT, config.KAN_GRID_SIZE
        )
        self.output_projection = nn.Linear(config.D_MODEL, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return ~mask

    def forward(self, x, edge_index, edge_attr, batch, tgt, tgt_mask=None, tgt_padding_mask=None):
        memory, memory_mask = self.graph_encoder(x, edge_index, edge_attr, batch)
        tgt_emb = self.tgt_embedding(tgt)
        
        output = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask, 
            tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=~memory_mask
        )
        return self.output_projection(output)