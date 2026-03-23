import torch
import pandas as pd
import tqdm
from torch_geometric.data import Data
from data.tokeniser import TargetTokenizer
from data.topology_parser import topology_to_pyg

def process_and_save_graphs(input_csv_path, output_pt_path, config):
    print("Loading raw data...")
    df = pd.read_csv(input_csv_path)
    
    # 1. Initialize the new Target-Only Tokenizer
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]
    tokenizer = TargetTokenizer(df, special_symbols, UNK_IDX=0)
    
    # 2. Build the vocabulary
    tgt_vocab = tokenizer.build_tgt_vocab()
    
    processed_graphs = []
    
    print("Converting textual topology to PyG Graphs & Tokenizing targets...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # A. ENCODER TRACK: Convert the textual topology to a PyG Graph object
        graph_data = topology_to_pyg(row['topology'], config)
        
        # B. DECODER TRACK: Tokenize the Squared Amplitude String
        raw_target = str(row['squared_amplitude'])
        tokens = tokenizer.tgt_tokenize(raw_target)
        
        # Convert text tokens to integer IDs
        token_ids = [tgt_vocab["<bos>"]] + [tgt_vocab[t] for t in tokens] + [tgt_vocab["<eos>"]]
        
        # Pad sequence to MAX_LENGTH
        padded_ids = token_ids[:config.MAX_LENGTH] + [tgt_vocab["<pad>"]] * max(0, config.MAX_LENGTH - len(token_ids))
        
        # Attach the target sequence to the PyG Graph object
        graph_data.y = torch.tensor(padded_ids, dtype=torch.long)
        
        processed_graphs.append(graph_data)
        
   
    print(f"Saving {len(processed_graphs)} graph objects to {output_pt_path}...")
    torch.save({
        'dataset': processed_graphs,
        'vocab': tgt_vocab
    }, output_pt_path)
    print("Data processing complete!")
