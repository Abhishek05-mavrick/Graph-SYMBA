import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- KAN BLOCKS ---
class SineKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=8, is_first=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        
        freq = torch.arange(1, grid_size + 1).float()
        if not is_first:
            freq = freq / (grid_size + 1)
        self.freq = nn.Parameter(freq)

        input_phase = torch.linspace(0, math.pi, input_dim)
        grid_phase = torch.arange(1, grid_size + 1).float() / (grid_size + 1)
        phase = input_phase.unsqueeze(-1) + grid_phase.unsqueeze(0)
        self.register_buffer('phase', phase)

        self.projection = nn.Linear(input_dim * grid_size, output_dim)
        self.gate_linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, self.input_dim)
        s = torch.sin(x_flat.unsqueeze(-1) * self.freq.view(1, 1, -1) + self.phase.unsqueeze(0))
        s_flat = s.view(-1, self.input_dim * self.grid_size)
        y = self.projection(s_flat)
        gate = torch.sigmoid(self.gate_linear(x_flat))
        return (y * gate).view(*orig_shape[:-1], self.output_dim)

class KANFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, grid_size=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SineKANLayer(d_model, d_ff, grid_size=grid_size, is_first=True),
            nn.LayerNorm(d_ff),
            nn.Dropout(dropout),
            SineKANLayer(d_ff, d_model, grid_size=grid_size),
            nn.Dropout(dropout)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# --- RoPE ATTENTION BLOCKS ---
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

def apply_rotary_pos_emb(x, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotate_half(x) * sin)

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope = RotaryPositionalEncoding(self.head_dim)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch, q_len, _ = query.shape
        _, k_len, _ = key.shape
        
        q = self.q_proj(query).view(batch, q_len, self.nhead, self.head_dim)
        k = self.k_proj(key).view(batch, k_len, self.nhead, self.head_dim)
        v = self.v_proj(value).view(batch, k_len, self.nhead, self.head_dim)
        
        cos, sin = self.rope(max(q_len, k_len), query.device)
        q = apply_rotary_pos_emb(q, cos[:, :q_len], sin[:, :q_len])
        k = apply_rotary_pos_emb(k, cos[:, :k_len], sin[:, :k_len])
            
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if attn_mask is not None and key_padding_mask is not None:
            bool_mask = (~attn_mask.unsqueeze(0).unsqueeze(0)) & (~key_padding_mask.unsqueeze(1).unsqueeze(2))
        elif attn_mask is not None:
            bool_mask = ~attn_mask.unsqueeze(0).unsqueeze(0)
        elif key_padding_mask is not None:
            bool_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            bool_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask, dropout_p=self.dropout if self.training else 0.0)
        return (self.out_proj(out.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)), None)

class SimpleMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch, q_len, _ = query.shape
        _, k_len, _ = key.shape
        
        q = self.q_proj(query).view(batch, q_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, k_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, k_len, self.nhead, self.head_dim).transpose(1, 2)
        
        if attn_mask is not None and key_padding_mask is not None:
            bool_mask = (~attn_mask.unsqueeze(0).unsqueeze(0)) & (~key_padding_mask.unsqueeze(1).unsqueeze(2))
        elif attn_mask is not None:
            bool_mask = ~attn_mask.unsqueeze(0).unsqueeze(0)
        elif key_padding_mask is not None:
            bool_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            bool_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask, dropout_p=self.dropout if self.training else 0.0)
        return (self.out_proj(out.transpose(1, 2).contiguous().view(batch, q_len, self.d_model)), None)

try:
    from performer_pytorch import Performer

    class PerformerDecoderWrapper(torch.nn.Module):
        def __init__(self, d_model, n_heads, num_layers, d_ff, dropout=0.1):
            super().__init__()
            self.performer = Performer(
                dim=d_model,
                depth=num_layers,
                heads=n_heads,
                dim_head=d_model // n_heads,
                causal=True,
                cross_attend=True,
                ff_mult=d_ff // d_model
            )
        def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
            return self.performer(tgt, context=memory, mask=~tgt_key_padding_mask if tgt_key_padding_mask is not None else None, context_mask=~memory_key_padding_mask if memory_key_padding_mask is not None else None)
except ImportError:
    pass
