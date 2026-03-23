import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, pad_idx):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE
        self.pad_idx = pad_idx
        
       
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.LR, 
            weight_decay=1e-4
        )

    def _get_causal_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return ~mask

    def run_epoch(self, dataloader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        correct_tokens = 0
        total_tokens = 0

        # Use PyG's native batching
        loop = tqdm(dataloader, leave=False)
        for batch in loop:
            batch = batch.to(self.device)
            
            # Autoregressive shift: 
            # Input to decoder: <bos> Token1 Token2
            # Target to predict: Token1 Token2 <eos>
            tgt_input = batch.y[:, :-1]
            tgt_expected = batch.y[:, 1:]
            
            # Create masks
            seq_len = tgt_input.size(1)
            tgt_mask = self.model.generate_square_subsequent_mask(seq_len).to(self.device)
            tgt_padding_mask = (tgt_input == self.pad_idx)

            with torch.set_grad_enabled(is_train):
                # Forward pass
                logits = self.model(
                    x=batch.x, 
                    edge_index=batch.edge_index, 
                    edge_attr=batch.edge_attr, 
                    batch=batch.batch, 
                    tgt=tgt_input, 
                    tgt_mask=tgt_mask, 
                    tgt_padding_mask=tgt_padding_mask
                )
                
                # Reshape for CrossEntropyLoss: [Batch * Seq_Len, Vocab_Size]
                logits_flat = logits.reshape(-1, logits.shape[-1])
                tgt_expected_flat = tgt_expected.reshape(-1)
                
                loss = self.criterion(logits_flat, tgt_expected_flat)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            # Metrics Tracking
            total_loss += loss.item()
            
            # Calculate Token Accuracy (ignoring padding)
            predictions = logits_flat.argmax(dim=-1)
            mask = (tgt_expected_flat != self.pad_idx)
            correct_tokens += (predictions[mask] == tgt_expected_flat[mask]).sum().item()
            total_tokens += mask.sum().item()
            
            loop.set_description(f"{'Train' if is_train else 'Val'} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0.0
        return avg_loss, accuracy

    def train(self, epochs):
        print(f"Starting Training on {self.device} for {epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.run_epoch(self.train_loader, is_train=True)
            val_loss, val_acc = self.run_epoch(self.val_loader, is_train=False)
            
            print(f"Epoch {epoch:02d} | "
                  f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.2f}%) | "
                  f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.2f}%)")
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_graph_symkan.pth")