import re
import torch
from .topology_parser import topology_to_pyg

class SymbolicVocab:
    def __init__(self, tokens, special_symbols, bos_idx, pad_idx, eos_idx, unk_idx, sep_idx):
        self.token_list = special_symbols + sorted(list(tokens))
        self.token_to_idx = {token: idx for idx, token in enumerate(self.token_list)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.bos_idx, self.pad_idx, self.eos_idx, self.unk_idx, self.sep_idx = bos_idx, pad_idx, eos_idx, unk_idx, sep_idx
    def encode(self, tokens): return [self.token_to_idx.get(t, self.unk_idx) for t in tokens]
    def decode(self, ids): return [self.idx_to_token.get(i, "<UNK>") for i in ids]
    def __len__(self): return len(self.token_list)

class GraphPhysicsDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, build_vocab=True):
        self.entries = df.to_dict('records')
        self.config = config
        self.tgt_vocab = None
        
        if build_vocab:
            all_tokens = set()
            for sq in df['squared_amplitude']:
                all_tokens.update(re.findall(r'\w+|[\+\-\*/\(\)\^]', str(sq)))
            self.tgt_vocab = SymbolicVocab(all_tokens, config.SPECIAL_SYMBOLS, 2, 0, 3, 1, 4)
    
    def __len__(self): return len(self.entries)
    
    def __getitem__(self, idx):
        row = self.entries[idx]
        graph = topology_to_pyg(row.get('topology', ''), self.config)
        tokens = re.findall(r'\w+|[\+\-\*/\(\)\^]', str(row['squared_amplitude']))
        ids = [self.tgt_vocab.bos_idx] + self.tgt_vocab.encode(tokens) + [self.tgt_vocab.eos_idx]
        padded = ids[:self.config.MAX_LENGTH] + [self.tgt_vocab.pad_idx] * max(0, self.config.MAX_LENGTH - len(ids))
        graph.y = torch.tensor(padded, dtype=torch.long)
        return graph