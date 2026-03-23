import torch
import torch.nn as nn
from .components import RoPEMultiheadAttention, SimpleMultiheadAttention

class KANTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff=2048, dropout=0.1, grid_size=8):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = SimpleMultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.standard_ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(d_model) for _ in range(3)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        
        if memory.shape[0] != tgt.shape[0]:
            memory = memory.transpose(0, 1)
            
        tgt2, _ = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        
        tgt2 = self.standard_ff(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt

class KANTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout, grid_size):
        super().__init__()
        self.layers = nn.ModuleList([KANTransformerDecoderLayer(d_model, nhead, d_ff, dropout, grid_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.norm(tgt)