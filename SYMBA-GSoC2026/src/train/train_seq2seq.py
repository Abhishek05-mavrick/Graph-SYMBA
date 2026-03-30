"""
Model A: Seq2Seq Transformer Baseline
Input  = Feynman diagram as text sequence
Output = squared-amplitude as text sequence
Reports sequence accuracy and sympy accuracy using autoregressive greedy decoding.
"""
import math, time
import sympy
from sympy.parsing.sympy_parser import parse_expr
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D_MODEL       = 256
N_HEADS       = 8
N_ENC_LAYERS  = 4
N_DEC_LAYERS  = 4
D_FF          = 1024
DROPOUT       = 0.1
MAX_LEN       = 512
BATCH         = 32
EPOCHS        = 35
LR            = 3e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PAD_IDX       = 1

# ---------------------------------------------------------------------------
# Vocabulary and Dataset
# ---------------------------------------------------------------------------
class SimpleVocab:
    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.token_to_idx.get('<unk>', 0))
    def __contains__(self, token):
        return token in self.token_to_idx
    def __len__(self):
        return len(self.token_to_idx)

import re
def tokenize_target(text: str) -> list:
    t = re.sub(r'\s+', '', str(text))
    t = re.sub(r'gamma_{\+MOMENTUM_\[ID\],\.\.\.}', ' <GAMMA> ', t)
    t = re.sub(r's_{\d+}', ' <S> ', t)
    t = re.sub(r't_{\d+}', ' <T> ', t)
    t = re.sub(r'u_{\d+}', ' <U> ', t)
    for sym in ['+', '-', '*', ',', '^', '%', '}', '(', ')']:
        t = t.replace(sym, f' {sym} ')
    t = re.sub(r'\b\w+_\w\b', lambda m: f' {m.group(0)} ', t)
    return [tok for tok in re.sub(r' {2,}', ' ', t).split(' ') if tok]

def build_vocab(series, is_target=False):
    vocab = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}
    for item in series:
        toks = tokenize_target(item) if is_target else str(item).split()
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    return SimpleVocab(vocab)

class TextDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.df = df
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src_toks = str(row['topology']).split()
        tgt_toks = tokenize_target(row['squared_amplitude'])

        src_idx = [self.src_vocab['<bos>']] + [self.src_vocab[t] for t in src_toks][:MAX_LEN-2] + [self.src_vocab['<eos>']]
        tgt_idx = [self.tgt_vocab['<bos>']] + [self.tgt_vocab[t] for t in tgt_toks][:MAX_LEN-2] + [self.tgt_vocab['<eos>']]

        return torch.tensor(src_idx, dtype=torch.long), torch.tensor(tgt_idx, dtype=torch.long)

def collate_fn(batch):
    src = [item[0] for item in batch]
    tgt = [item[1] for item in batch]
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX)
    return src, tgt

# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(-torch.arange(0, d_model, 2).float() * math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ---------------------------------------------------------------------------
# Seq2Seq Transformer
# ---------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, D_MODEL, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, D_MODEL, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(D_MODEL)

        self.transformer = nn.Transformer(
            d_model=D_MODEL, nhead=N_HEADS,
            num_encoder_layers=N_ENC_LAYERS, num_decoder_layers=N_DEC_LAYERS,
            dim_feedforward=D_FF, dropout=DROPOUT, batch_first=True
        )
        self.out_proj = nn.Linear(D_MODEL, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        s = self.pos_enc(self.src_emb(src))
        t = self.pos_enc(self.tgt_emb(tgt))
        out = self.transformer(s, t, src_mask=src_mask, tgt_mask=tgt_mask,
                               src_key_padding_mask=src_pad_mask,
                               tgt_key_padding_mask=tgt_pad_mask)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
import threading

def check_symbolic_equivalence(p_toks, t_toks):
    """Check if two token sequences represent symbolically equivalent expressions."""
    def toks_to_str(l):
        parts = []
        for t in l:
            tok = str(t)
            if tok in ('<bos>', '<eos>', '<pad>', '<unk>'):
                continue
            # Map physics tokens to safe SymPy variable names
            tok = tok.replace('<S>', 's_var')
            tok = tok.replace('<T>', 't_var')
            tok = tok.replace('<U>', 'u_var')
            tok = tok.replace('<GAMMA>', 'gamma_var')
            tok = tok.replace('SQUARE', '**2')
            tok = tok.replace('INDEX_', 'idx_')
            tok = tok.replace('MOMENTUM_', 'p_')
            tok = tok.replace('<', '').replace('>', '')
            parts.append(tok)
        # Join with spaces so SymPy can parse multi-char variable names
        return ' '.join(parts)

    p, t = toks_to_str(p_toks), toks_to_str(t_toks)
    if not p.strip() or not t.strip(): return False

    # Normalized string comparison (strip all whitespace)
    p_norm = re.sub(r'\s+', '', p)
    t_norm = re.sub(r'\s+', '', t)
    if p_norm == t_norm: return True

    # SymPy structural equivalence check with timeout
    result = [False]
    def _check():
        try:
            local_dict = {}
            expr_p = parse_expr(p, local_dict=local_dict)
            expr_t = parse_expr(t, local_dict=local_dict)
            # Try cheap check first: expand and compare
            if sympy.expand(expr_p - expr_t) == 0:
                result[0] = True
                return
            # Fall back to simplify (slower but catches more)
            result[0] = (sympy.simplify(expr_p - expr_t) == 0)
        except Exception:
            pass

    worker = threading.Thread(target=_check)
    worker.daemon = True
    worker.start()
    worker.join(5.0)  # 5 second timeout
    return result[0]

def causal_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

def run_train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, c_tok, n_tok = 0.0, 0, 0

    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        src_pad = torch.eq(src, PAD_IDX)
        tgt_pad = torch.eq(tgt_in, PAD_IDX)
        tgt_mask = causal_mask(tgt_in.size(1), device)

        logits = model(src, tgt_in, tgt_mask=tgt_mask, src_pad_mask=src_pad, tgt_pad_mask=tgt_pad)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(-1)
        mask = torch.ne(tgt_out, PAD_IDX)
        c_tok += torch.eq(preds[mask], tgt_out[mask]).sum().item()
        n_tok += mask.sum().item()

    return total_loss / len(loader), (c_tok / n_tok * 100 if n_tok else 0)

@torch.no_grad()
def greedy_decode_seq2seq(model, src, bos_idx, eos_idx, pad_idx, max_len, device):
    """Autoregressive greedy decode for seq2seq transformer."""
    model.eval()
    src_pad = torch.eq(src, pad_idx)
    memory = model.transformer.encoder(model.pos_enc(model.src_emb(src)), src_key_padding_mask=src_pad)
    batch_size = src.size(0)
    
    ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len - 1):
        tgt_mask = causal_mask(ys.size(1), device)
        tgt_pad_mask = torch.eq(ys, pad_idx)
        t = model.pos_enc(model.tgt_emb(ys))
        out = model.transformer.decoder(t, memory, tgt_mask=tgt_mask,
                                         tgt_key_padding_mask=tgt_pad_mask,
                                         memory_key_padding_mask=src_pad)
        logits = model.out_proj(out[:, -1, :])
        next_tok = logits.argmax(-1)
        next_tok = next_tok.masked_fill(finished, pad_idx)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_idx)
        if finished.all():
            break
    return ys[:, 1:]  # strip bos

def run_eval_epoch(model, loader, criterion, device, tgt_vocab, bos_idx, eos_idx, compute_metrics=True):
    model.eval()
    total_loss, c_seq, c_sym, n_seq = 0.0, 0, 0, 0
    
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        
        with torch.no_grad():
            src_pad = torch.eq(src, PAD_IDX)
            tgt_pad = torch.eq(tgt_in, PAD_IDX)
            tgt_mask = causal_mask(tgt_in.size(1), device)
            logits = model(src, tgt_in, tgt_mask=tgt_mask, src_pad_mask=src_pad, tgt_pad_mask=tgt_pad)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            
            if not compute_metrics:
                continue
            
            # Autoregressive decoding for honest seq_acc
            preds = greedy_decode_seq2seq(model, src, bos_idx, eos_idx, PAD_IDX, MAX_LEN, device)
            
            for b in range(preds.shape[0]):
                p_len = (preds[b] == eos_idx).nonzero(as_tuple=True)[0]
                p_len = p_len[0].item() if len(p_len) > 0 else preds.shape[1]
                t_len = (tgt_out[b] == eos_idx).nonzero(as_tuple=True)[0]
                t_len = t_len[0].item() if len(t_len) > 0 else tgt_out.shape[1]
                
                p_seq, t_seq = preds[b, :p_len], tgt_out[b, :t_len]
                
                if p_len == t_len and torch.equal(p_seq, t_seq):
                    c_seq += 1
                else:
                    ptoks = [tgt_vocab.idx_to_token.get(i.item(), '') for i in p_seq]
                    ttoks = [tgt_vocab.idx_to_token.get(i.item(), '') for i in t_seq]
                    if check_symbolic_equivalence(ptoks, ttoks):
                        c_sym += 1
                n_seq += 1
                
    return (total_loss / len(loader), 
            c_seq / n_seq * 100 if n_seq else 0,
            (c_seq + c_sym) / n_seq * 100 if n_seq else 0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def train_domain(domain_name, data_dir, save_dir):
    print(f'\n{"="*60}')
    print(f'  Seq-to-Seq Transformer — {domain_name}')
    print(f'{"="*60}')

    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df   = pd.read_csv(data_dir / 'val.csv')
    test_df  = pd.read_csv(data_dir / 'test.csv')

    # Build vocab from TRAIN set only (no data leakage)
    src_vocab = build_vocab(train_df['topology'], False)
    tgt_vocab = build_vocab(train_df['squared_amplitude'], True)
    
    print(f'Src vocab: {len(src_vocab)}, Tgt vocab: {len(tgt_vocab)}')

    train_ds = TextDataset(train_df, src_vocab, tgt_vocab)
    val_ds   = TextDataset(val_df, src_vocab, tgt_vocab)
    test_ds  = TextDataset(test_df, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, collate_fn=collate_fn)

    bos_idx, eos_idx = 2, 3  # matches SPECIAL_SYMBOLS ordering

    model = Seq2SeqTransformer(len(src_vocab), len(tgt_vocab)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float('inf')
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / f'seq2seq_{domain_name.lower()}_best.pt'

    for ep in range(1, EPOCHS + 1):
        tr_l, tr_t = run_train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        comp_mets = (ep % 5 == 0) or (ep == EPOCHS)
        vl_l, vl_s, vl_sym = run_eval_epoch(model, val_loader, criterion, DEVICE, tgt_vocab, bos_idx, eos_idx, compute_metrics=comp_mets)
        scheduler.step()
        
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
    te_l, te_s, te_sym = run_eval_epoch(model, test_loader, criterion, DEVICE, tgt_vocab, bos_idx, eos_idx)
    
    print(f'\n>>> {domain_name} TEST')
    print(f'  Greedy : seq_acc={te_s:.2f}%  sym_acc={te_sym:.2f}%')
    return te_s, te_sym

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent.parent
    data_qed = root / 'preprocessed' / 'qed'
    save_dir = root / 'checkpoints'

    results = {}
    if data_qed.exists() and (data_qed / 'train.csv').exists():
        results['QED'] = train_domain('QED', data_qed, save_dir)

    print('\n' + '='*60)
    print('  Seq2Seq Transformer Baseline — Final Results')
    print('='*60)
    for d, (seq, sym) in results.items():
        print(f'  {d}:  seq_acc = {seq:.2f}%   sym_acc = {sym:.2f}%')
