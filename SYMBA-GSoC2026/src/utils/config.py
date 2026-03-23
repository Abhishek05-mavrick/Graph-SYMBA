import torch

class ProjectConfig:
    def __init__(self):
        # Model Specs
        self.D_MODEL = 512
        self.N_LAYERS = 6
        self.N_HEADS = 8
        self.DROPOUT = 0.1
        self.D_FF = 2048
        
        # GNN Encoder Specs
        self.GNN_LAYERS = 4
        self.GNN_HIDDEN = 512
        self.NODE_FEATURE_DIM = 64
        self.EDGE_FEATURE_DIM = 32
        self.KAN_GRID_SIZE = 8
        
        # Sequence Specs
        self.MAX_LENGTH = 512
        self.INDEX_TOKEN_POOL_SIZE = 100
        self.TERM_TOKEN_POOL_SIZE = 50
        
        # Training Params
        self.EPOCHS = 150
        self.LR = 5e-5
        self.BATCH_SIZE = 16
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.SPECIAL_SYMBOLS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]
        self.TO_REPLACE = True
        self.PRECOMPUTE_DATA = True