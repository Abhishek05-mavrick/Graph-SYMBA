"""
Model B: Graph Encoder (TransformerConv + Positional Identity) + Transformer Decoder
Input  = Feynman diagram as PyG graph
Output = tokenised squared-amplitude

Upgrades implemented here:
1. Absolute Node Index Embeddings (solves permutation invariance per particle)
2. Random-Walk Positional Encodings (RWSE) (solves graph topography awareness)
3. TransformerConv Message Passing (solves oversmoothing via graph-attention)
4. Explicit Edge Attributes (propagator states physics momentum/mass directly)
"""
import math, time, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
import torch_geometric.transforms as T
from tqdm import tqdm
from src.train.training import run_train_epoch, run_eval_epoch, run_test_beam


D_MODEL       = 256
GNN_HIDDEN    = 256
GNN_LAYERS    = 3
N_HEADS       = 8
N_DEC_LAYERS  = 4
D_FF          = 1024
DROPOUT       = 0.1
MAX_LEN       = 512
NODE_FEAT_DIM = 64
EDGE_FEAT_DIM = 32
RWSE_DIM      = 16
MAX_NODES     = 200
BATCH         = 32
EPOCHS        = 60
LR            = 1.5e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PAD_IDX       = 1

class SimpleVocab:
    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx.get('<unk>', 0))
    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]
    def lookup_token(self, idx):
        return self.idx_to_token.get(idx, '<unk>')
    def __contains__(self, token):
        return token in self.token_to_idx
    def __len__(self):
        return len(self.token_to_idx)


class GraphTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial node features mapping
        self.node_emb = nn.Linear(NODE_FEAT_DIM, GNN_HIDDEN)
        
        # 1. Absolute Node Index Embedding (solves Permutation Invariance)
        self.idx_emb  = nn.Embedding(MAX_NODES, GNN_HIDDEN)
        
        # 2. RWSE Positional Encoding (Topology structure)
        self.rwse_emb = nn.Linear(RWSE_DIM, GNN_HIDDEN)

        # 3 & 4. TransformerConv with edge_dim matching Edge Attributes
        self.convs  = nn.ModuleList([
            TransformerConv(GNN_HIDDEN, GNN_HIDDEN, heads=4, concat=False, edge_dim=EDGE_FEAT_DIM, dropout=DROPOUT)
            for _ in range(GNN_LAYERS)
        ])
        
        self.norms  = nn.ModuleList([nn.LayerNorm(GNN_HIDDEN) for _ in range(GNN_LAYERS)])
        self.proj   = nn.Linear(GNN_HIDDEN, D_MODEL)

    def forward(self, batch_data):
        x, edge_index, batch, edge_attr = batch_data.x, batch_data.edge_index, batch_data.batch, batch_data.edge_attr
        
        # Map raw node features
        h = self.node_emb(x)
        
        # Extract dynamic local node indices within each disjoint graph
        # Utilizing `batch` tensor via fast counting logic
        counts = torch.bincount(batch)
        local_idx = torch.cat([torch.arange(c.item(), device=x.device) for c in counts])
        
        # Add Node identity and RWSE Identity explicitly natively
        h = h + self.idx_emb(local_idx)
        if hasattr(batch_data, 'rwse'):
            h = h + self.rwse_emb(batch_data.rwse)
            
        # Message passing layers with exact Edge Physics Attributes
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h = norm(h + h_new) # Applying safe skip connections to stop oversmoothing
            
        # Density batching
        h_dense, mask = to_dense_batch(h, batch)
        return self.proj(h_dense), mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(-torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AdvancedGraphTransformer(nn.Module):
    def __init__(self, tgt_vocab_size):
        super().__init__()
        self.encoder  = GraphTransformerEncoder()
        self.tgt_emb  = nn.Embedding(tgt_vocab_size, D_MODEL, padding_idx=PAD_IDX)
        self.pos_enc  = PositionalEncoding(D_MODEL)
        dec_layer = nn.TransformerDecoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, batch_first=True, norm_first=True)
        self.decoder  = nn.TransformerDecoder(dec_layer, N_DEC_LAYERS)
        self.out_proj = nn.Linear(D_MODEL, tgt_vocab_size)

    def forward(self, batch_data, tgt, tgt_mask=None, tgt_pad_mask=None):
        memory, mem_mask = self.encoder(batch_data)
        t = self.pos_enc(self.tgt_emb(tgt))
        out = self.decoder(t, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=torch.logical_not(mem_mask))
        return self.out_proj(out)


def train_domain(domain_name, data_dir, save_dir):
    print(f'\n{"="*60}')
    print(f'  Advanced Graph-Transformer — {domain_name}')
    print(f'{"="*60}')

    train_data = torch.load(data_dir / 'train_graphs.pt', weights_only=False)
    val_data   = torch.load(data_dir / 'val_graphs.pt',   weights_only=False)
    test_data  = torch.load(data_dir / 'test_graphs.pt',  weights_only=False)

    train_ds, vocab = train_data['dataset'], train_data['vocab']
    val_ds   = val_data['dataset']
    test_ds  = test_data['dataset']

    print("Computing Random Walk Positional Encodings (RWSE)...")
    rwse = T.AddRandomWalkPE(walk_length=RWSE_DIM, attr_name='rwse')
    for ds in [train_ds, val_ds, test_ds]:
        for i in range(len(ds)):
            ds[i] = rwse(ds[i])

    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}, Vocab: {len(vocab)}')

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH)

    model = AdvancedGraphTransformer(len(vocab)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4) # Standard Adam decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.3)

    best_val = float('inf')
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / f'advanced_graph_mlp_{domain_name.lower()}_best.pt'

    bos_idx = vocab.token_to_idx.get('<bos>', 2)
    eos_idx = vocab.token_to_idx.get('<eos>', 3)
    
    for ep in range(1, EPOCHS + 1):
        tr_l, tr_t = run_train_epoch(model, train_loader, criterion, optimizer, DEVICE, MAX_LEN, PAD_IDX, scheduler)
        comp_mets = (ep % 5 == 0) or (ep == EPOCHS)
        vl_l, vl_s, vl_sym = run_eval_epoch(model, val_loader, criterion, DEVICE, vocab, MAX_LEN, PAD_IDX, bos_idx, eos_idx, compute_metrics=comp_mets)
        
        if comp_mets:
            print(f'Ep {ep:3d}/{EPOCHS} | '
                  f'Train loss={tr_l:.4f} tok={tr_t:.1f}% | '
                  f'Val loss={vl_l:.4f} seq={vl_s:.1f}% sym={vl_sym:.1f}%')
        else:
            print(f'Ep {ep:3d}/{EPOCHS} | '
                  f'Train loss={tr_l:.4f} tok={tr_t:.1f}% | '
                  f'Val loss={vl_l:.4f} (skipped decoding)')
                  
        if vl_l < best_val:
            best_val = vl_l
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, weights_only=True))
    te_s_g, te_sym_g = run_eval_epoch(model, test_loader, criterion, DEVICE, vocab, MAX_LEN, PAD_IDX, bos_idx, eos_idx)[1:]
    te_s_b, te_sym_b = run_test_beam(model, test_loader, DEVICE, vocab, MAX_LEN, PAD_IDX, bos_idx, eos_idx, k=5)
    
    print(f'\n>>> {domain_name} TEST')
    print(f'  Greedy : seq_acc={te_s_g:.2f}%  sym_acc={te_sym_g:.2f}%')
    print(f'  Beam(5): seq_acc={te_s_b:.2f}%  sym_acc={te_sym_b:.2f}%')
    return te_s_g, te_sym_g, te_s_b, te_sym_b

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent.parent
    data_qed = root / 'preprocessed' / 'qed'
    data_qcd = root / 'preprocessed' / 'qcd'
    save_dir = root / 'checkpoints'

    results = {}
    if data_qed.exists() and (data_qed / 'train_graphs.pt').exists():
        results['QED'] = train_domain('QED', data_qed, save_dir)
    if data_qcd.exists() and (data_qcd / 'train_graphs.pt').exists():
        results['QCD'] = train_domain('QCD', data_qcd, save_dir)

    print('\n' + '='*60)
    print('  Advanced Graph-Transformer — Final Results')
    print('='*60)
    for d, (te_s_g, te_sym_g, te_s_b, te_sym_b) in results.items():
        print(f'  {d}:  Greedy_seq = {te_s_g:.2f}%   Beam_seq = {te_s_b:.2f}%   Beam_sym = {te_sym_b:.2f}%')
