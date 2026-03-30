import torch
import sympy
from sympy.parsing.sympy_parser import parse_expr
import threading

def causal_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

def check_symbolic_equivalence(p_toks, t_toks, vocab):
    """Check if two token sequences represent symbolically equivalent expressions."""
    import re

    def toks_to_str(l):
        parts = []
        for t in l:
            token = vocab.lookup_token(t) if isinstance(t, int) else str(t)
            if token in ('<bos>', '<eos>', '<pad>', '<unk>', '<BOS>', '<EOS>', '<PAD>', '<UNK>'):
                continue
            # Map physics tokens to safe SymPy variable names
            token = token.replace('<S>', 's_var').replace('<s>', 's_var')
            token = token.replace('<T>', 't_var').replace('<t>', 't_var')
            token = token.replace('<U>', 'u_var').replace('<u>', 'u_var')
            token = token.replace('<GAMMA>', 'gamma_var').replace('<gamma>', 'gamma_var')
            token = token.replace('SQUARE', '**2')
            token = token.replace('INDEX_', 'idx_')
            token = token.replace('MOMENTUM_', 'p_')
            token = token.replace('<', '').replace('>', '')
            parts.append(token)
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

@torch.no_grad()
def greedy_decode_batch(model, batch_data, bos_idx, eos_idx, pad_idx, max_len, device):
    model.eval()
    memory, mem_mask = model.encoder(batch_data)
    mem_key_pad = torch.logical_not(mem_mask)
    batch_size = memory.size(0)
    
    ys = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len - 1):
        tgt_mask = causal_mask(ys.size(1), device)
        t = model.pos_enc(model.tgt_emb(ys))
        out = model.decoder(t, memory, tgt_mask=tgt_mask,
                            memory_key_padding_mask=mem_key_pad)
        logits = model.out_proj(out[:, -1, :])
        next_tok = logits.argmax(-1)
        next_tok = next_tok.masked_fill(finished, pad_idx)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_idx)
        if finished.all():
            break
    return ys[:, 1:]  # strip bos

@torch.no_grad()
def beam_search_single(model, batch_data, bos_idx, eos_idx, max_len, k, device):
    model.eval()
    memory, mem_mask = model.encoder(batch_data)  # [1, N, D]
    mem_key_pad = torch.logical_not(mem_mask)
    
    beams = [(0.0, [bos_idx])]  # (log_prob, token_list)
    
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_idx:
                candidates.append((score, seq))
                continue
            
            ys = torch.tensor([seq], dtype=torch.long, device=device)
            tgt_mask = causal_mask(ys.size(1), device)
            t = model.pos_enc(model.tgt_emb(ys))
            out = model.decoder(t, memory, tgt_mask=tgt_mask,
                                memory_key_padding_mask=mem_key_pad)
            log_probs = torch.log_softmax(model.out_proj(out[0, -1]), dim=-1)
            top_probs, top_ids = log_probs.topk(k)
            for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
                candidates.append((score + prob, seq + [idx]))
        
        # Sort strictly by raw log probability sum
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:k]
        if all(s[-1] == eos_idx for _, s in beams):
            break
            
    return beams[0][1][1:]  # best beam, strip bos

