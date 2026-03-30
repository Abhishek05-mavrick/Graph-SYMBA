import re, time
from pathlib import Path
from collections import Counter
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

PROJECT_ROOT = Path.cwd()
RAW_QED_DIR = PROJECT_ROOT / 'QED data'
OUT_QED_DIR = PROJECT_ROOT / 'preprocessed' / 'qed'

def parse_txt_file(filepath: Path) -> list:
    rows = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    for line in content.splitlines():
        line = line.strip()
        if not line.startswith('Interaction:'): continue
        parts = line[len('Interaction:'):].strip().split(' : ', 3)
        if len(parts) == 4:
            rows.append(dict(interaction=parts[0].strip(), topology=parts[1].strip(),
                             amplitude=parts[2].strip(), squared_amplitude=parts[3].strip()))
    return rows

def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    all_rows = []
    for fp in raw_dir.glob('*.txt'):
        if 'Copy' not in fp.name:
            all_rows.extend(parse_txt_file(fp))
    return pd.DataFrame(all_rows)

def clean_and_normalize(text: str) -> str:
    text = str(text)
    text = re.sub(r'\b[ijkl]_\d+\b', 'MOMENTUM_[ID]', text)
    text = re.sub(r'\b(?![ps]_)\w+\d+\b', 'INDEX_[ID]', text)
    text = re.sub(r'\\\\', ' ', text)
    text = re.sub(r'%', ' ', text)
    text = re.sub(r'\b\w+_{', ' ', text)
    text = re.sub(r'Prop', '', text)
    text = re.sub(r'int\{', '', text)
    text = re.sub(r'([+\-*/^()])', r' \1 ', text)
    text = re.sub(r',', ' , ', text)
    text = re.sub(r'(\w)_(\w+\d+)\(X\)\^\(\*\)', r'\1_ANTI_PARTICLE_\2', text)
    text = re.sub(r'(\w)_(\w+\d+)\(X\)', r'\1_PARTICLE_\2', text)
    return re.sub(r'\s+', ' ', text).strip()

def split_df(df, ratios=(0.8, 0.1, 0.1), seed=42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df); n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
    return df.iloc[:n_train], df.iloc[n_train:n_train + n_val], df.iloc[n_train + n_val:]

SPECIAL_SYMBOLS = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

class SimpleVocab:
    def __init__(self, token_to_idx):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {v: k for k, v in token_to_idx.items()}
    def __getitem__(self, token): return self.token_to_idx.get(token, self.token_to_idx.get('<unk>', 0))
    def lookup_indices(self, tokens): return [self[t] for t in tokens]
    def lookup_token(self, idx): return self.idx_to_token.get(idx, '<unk>')
    def __len__(self): return len(self.token_to_idx)

def tokenize_target(text: str) -> list:
    t = re.sub(r'\s+', '', str(text))
    # Group strict physics symbols first
    t = re.sub(r'gamma_{\+MOMENTUM_\[ID\],\.\.\.}', ' <GAMMA> ', t)
    t = re.sub(r's_{\d+}', ' <S> ', t)
    t = re.sub(r't_{\d+}', ' <T> ', t)
    t = re.sub(r'u_{\d+}', ' <U> ', t)
    
    # Operators
    for sym in ['+', '-', '*', ',', '^', '%', '}', '(', ')']:
        t = t.replace(sym, f' {sym} ')
        
    t = re.sub(r'\b\w+_\w\b', lambda m: f' {m.group(0)} ', t)
    return [tok for tok in re.sub(r' {2,}', ' ', t).split(' ') if tok]

NODE_FEATURE_DIM = 64
EDGE_FEATURE_DIM = 32

def _extract_particle_symbol(particle_str):
    """Extract particle symbol before the parenthesis, then match known symbols."""
    # Get the part before '(' e.g. "OffShell e" from "OffShell e(X_1)"
    prefix = particle_str.split('(')[0].strip() if '(' in particle_str else particle_str.strip()
    # Match the last word-token which is the particle name
    tokens = prefix.split()
    symbol = tokens[-1] if tokens else ''
    # Explicit lookup — no substring matching
    type_map = {'e': 1.0, 'A': 2.0, 'mu': 3.0, 'u': 4.0, 'd': 5.0, 's': 6.0, 'c': 7.0, 'b': 8.0, 't': 9.0, 'tt': 9.0}
    return type_map.get(symbol, 0.0)

def _build_edge_attr(node_features, edge_index, edge_feat_dim):
    """Compute physics-informed pairwise edge features."""
    edge_attrs = []
    for i in range(edge_index.size(1)):
        src_idx = edge_index[0, i].item()
        tgt_idx = edge_index[1, i].item()
        src_feat = node_features[src_idx]
        tgt_feat = node_features[tgt_idx]
        edge_feat = [
            src_feat[0],                          # source particle type
            tgt_feat[0],                          # target particle type
            abs(src_feat[0] - tgt_feat[0]),       # type difference
            src_feat[1] * tgt_feat[1],            # momentum product
            src_feat[2] * tgt_feat[2],            # spin product
            float(src_idx == tgt_idx),            # self-loop indicator
            float(src_feat[0] == tgt_feat[0]),    # same-type indicator
            float(src_feat[1] == tgt_feat[1]),    # same-momentum indicator
        ]
        edge_feat += [0.0] * (edge_feat_dim - len(edge_feat))
        edge_attrs.append(edge_feat)
    return torch.tensor(edge_attrs, dtype=torch.float)

def topology_to_pyg(topology_str: str):
    edges, node_features = [], []
    vertex_blocks = str(topology_str).split('Vertex ')[1:]
    gidx = 0
    for block in vertex_blocks:
        raw_particles = [p.strip() for p in block.split(',') if '(' in p]
        vidx = []
        for p in raw_particles:
            p_type = _extract_particle_symbol(p)
            momentum = 1.0 if 'OffShell' in p else 0.0
            spin = 0.5 if p_type in (1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0) else 1.0  # fermions vs bosons
            node_features.append([p_type, momentum, spin, 1.0] + [0.0] * (NODE_FEATURE_DIM - 4))
            vidx.append(gidx)
            gidx += 1
        for i in vidx:
            for j in vidx: edges.append([i, j])
    if not node_features:
        node_features, edges = [[0.0] * NODE_FEATURE_DIM], [[0, 0]]
    x = torch.tensor(node_features, dtype=torch.float)
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = _build_edge_attr(node_features, ei, EDGE_FEATURE_DIM)
    return Data(x=x, edge_index=ei, edge_attr=edge_attr)

MAX_LENGTH = 512

def build_graph_dataset(df, vocab):
    topo_cache, target_cache = {}, {}
    graphs = []
    for topo in df['topology'].unique(): topo_cache[topo] = topology_to_pyg(topo)
    for sq in df['squared_amplitude'].unique():
        toks = tokenize_target(sq)
        ids = [BOS_IDX] + vocab.lookup_indices(toks) + [EOS_IDX]
        ids = ids[:MAX_LENGTH] + [PAD_IDX] * max(0, MAX_LENGTH - len(ids))
        target_cache[sq] = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        g = topo_cache[row['topology']]
        graphs.append(Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, y=target_cache[row['squared_amplitude']]))
    return graphs

if __name__ == '__main__':
    qed_raw = load_raw_data(RAW_QED_DIR)
    for c in ['topology', 'squared_amplitude']: qed_raw[c] = qed_raw[c].apply(clean_and_normalize)
    train_df, val_df, test_df = split_df(qed_raw)
    OUT_QED_DIR.mkdir(parents=True, exist_ok=True)
    
    counter = Counter()
    for sq in train_df['squared_amplitude']: counter.update(tokenize_target(sq))
    t2i = {s: i for i, s in enumerate(SPECIAL_SYMBOLS)}
    idx = len(SPECIAL_SYMBOLS)
    for tok, _ in counter.most_common():
        if tok not in t2i:
            t2i[tok] = idx; idx += 1
    vocab = SimpleVocab(t2i)
    
    for split, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
        graphs = build_graph_dataset(df, vocab)
        torch.save({'dataset': graphs, 'vocab': vocab}, OUT_QED_DIR / f'{split}_graphs.pt')
        df.to_csv(OUT_QED_DIR / f'{split}.csv', index=False)
    print(f'Done preprocessing QED data. Total records: {len(qed_raw)}')
