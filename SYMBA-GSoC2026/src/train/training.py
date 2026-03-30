import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import greedy_decode_batch, beam_search_single, check_symbolic_equivalence, causal_mask


def run_train_epoch(model, loader, criterion, optimizer, device, max_len, pad_idx, scheduler=None):
    model.train()
    total_loss, c_tok, n_tok = 0.0, 0, 0

    for batch in tqdm(loader, leave=False):
        batch   = batch.to(device)
        tgt_in  = batch.y[:, :-1]
        tgt_out = batch.y[:, 1:]
        tgt_mask = causal_mask(tgt_in.size(1), device)
        tgt_pad  = torch.eq(tgt_in, pad_idx)

        logits = model(batch, tgt_in, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad)
        loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(-1)
        mask  = torch.ne(tgt_out, pad_idx)
        c_tok += torch.eq(preds[mask], tgt_out[mask]).sum().item()
        n_tok += mask.sum().item()

    return total_loss / len(loader), (c_tok / n_tok * 100 if n_tok else 0)


def run_eval_epoch(model, loader, criterion, device, vocab, max_len, pad_idx, bos_idx, eos_idx, compute_metrics=True):
    model.eval()
    total_loss, c_seq, c_sym, n_seq = 0.0, 0, 0, 0

    for batch in tqdm(loader, leave=False):
        batch   = batch.to(device)
        tgt_in  = batch.y[:, :-1]
        tgt_out = batch.y[:, 1:]

        with torch.no_grad():
            tgt_mask = causal_mask(tgt_in.size(1), device)
            tgt_pad  = torch.eq(tgt_in, pad_idx)
            logits   = model(batch, tgt_in, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad)
            loss     = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()

            if not compute_metrics:
                continue

            preds = greedy_decode_batch(model, batch, bos_idx, eos_idx, pad_idx, max_len, device)

            for b in range(preds.shape[0]):
                p_eos = (preds[b] == eos_idx).nonzero(as_tuple=True)[0]
                p_len = p_eos[0].item() if len(p_eos) > 0 else preds.shape[1]
                t_eos = (tgt_out[b] == eos_idx).nonzero(as_tuple=True)[0]
                t_len = t_eos[0].item() if len(t_eos) > 0 else tgt_out.shape[1]

                p_seq = preds[b, :p_len]
                t_seq = tgt_out[b, :t_len]

                if p_len == t_len and torch.equal(p_seq, t_seq):
                    c_seq += 1
                else:
                    ptoks = [vocab.idx_to_token.get(i.item(), '') for i in p_seq]
                    ttoks = [vocab.idx_to_token.get(i.item(), '') for i in t_seq]
                    # Pass indices even, but for flexibility we changed check_symbolic_equivalence to use token lookup
                    if check_symbolic_equivalence(p_seq.tolist(), t_seq.tolist(), vocab):
                        c_sym += 1
                n_seq += 1

    return (
        total_loss / len(loader),
        c_seq / n_seq * 100 if n_seq else 0,
        (c_seq + c_sym) / n_seq * 100 if n_seq else 0,
    )


def run_test_beam(model, loader, device, vocab, max_len, pad_idx, bos_idx, eos_idx, k=5):
    from torch_geometric.data import Batch

    model.eval()
    c_seq, c_sym, n_seq = 0, 0, 0

    for batch in tqdm(loader, leave=False):
        data_list = batch.to_data_list()

        for sample in data_list:
            sample       = sample.to(device)
            single_batch = Batch.from_data_list([sample])


            tgt_out = sample.y[0, 1:]

            best_seq        = beam_search_single(model, single_batch, bos_idx, eos_idx, max_len, k, device)
            best_seq_tensor = torch.tensor(best_seq, device=device)

            t_eos = (tgt_out == eos_idx).nonzero(as_tuple=True)[0]
            t_len = t_eos[0].item() if len(t_eos) > 0 else len(tgt_out)
            t_seq = tgt_out[:t_len]

            p_eos = (best_seq_tensor == eos_idx).nonzero(as_tuple=True)[0]
            p_len = p_eos[0].item() if len(p_eos) > 0 else len(best_seq_tensor)
            p_seq = best_seq_tensor[:p_len]

            if p_len == t_len and torch.equal(p_seq, t_seq):
                c_seq += 1
            else:
                ptoks = [vocab.idx_to_token.get(i.item(), '') for i in p_seq]
                ttoks = [vocab.idx_to_token.get(i.item(), '') for i in t_seq]
                if check_symbolic_equivalence(p_seq.tolist(), t_seq.tolist(), vocab):
                    c_sym += 1
            n_seq += 1

    return (
        c_seq / n_seq * 100 if n_seq else 0,
        (c_seq + c_sym) / n_seq * 100 if n_seq else 0,
    )