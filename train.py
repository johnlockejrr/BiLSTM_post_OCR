import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

class HebrewDataset(Dataset):
    def __init__(self, data_path: str, char_to_idx: Dict[str, int], max_len: int = 100):
        self.data = []
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        target_text = item['target']
        
        # Convert to indices
        input_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in input_text]
        target_indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in target_text]
        
        # Pad sequences
        input_indices = input_indices[:self.max_len] + [self.char_to_idx['<PAD>']] * (self.max_len - len(input_indices))
        target_indices = target_indices[:self.max_len] + [self.char_to_idx['<PAD>']] * (self.max_len - len(target_indices))
        
        return {
            'input': torch.tensor(input_indices),
            'target': torch.tensor(target_indices)
        }

class HebrewOCRCorrector(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.layer_norm1(lstm_out)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        attn_out = self.layer_norm1(attn_out + lstm_out)  # Residual connection
        
        # Output layers
        out = self.fc1(attn_out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.layer_norm2(out)
        logits = self.fc2(out)  # (batch_size, seq_len, vocab_size)
        
        return logits

def create_vocab(data_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create vocabulary from the dataset."""
    chars = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            chars.update(item['input'])
            chars.update(item['target'])
    
    # Add special tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    chars.update(special_tokens)
    
    # Create mappings
    char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char

def calculate_accuracy(predictions, targets, pad_idx):
    """Calculate character-level accuracy."""
    # Remove padding from consideration
    mask = (targets != pad_idx)
    correct = ((predictions == targets) * mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0

def find_learning_rate(model, train_loader, criterion, device, init_value=1e-8, final_value=10., beta=0.98):
    """Find optimal learning rate using the learning rate finder technique."""
    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer = optim.Adam(model.parameters(), lr=lr)
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    
    for data in tqdm(train_loader, desc="Finding learning rate"):
        batch_num += 1
        # Get data
        input_text = data['input'].to(device)
        target_text = data['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_text)
        loss = criterion(output.view(-1, output.size(-1)), target_text.view(-1))
        
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        
        # Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
            
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    
    return log_lrs, losses

def plot_lr_finder(log_lrs, losses, save_path='lr_finder.png'):
    """Plot learning rate finder results."""
    plt.figure(figsize=(10, 6))
    plt.plot(log_lrs, losses)
    plt.xlabel('log10(lr)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, train_loader, criterion, optimizer, device, clip_norm=1.0):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_grad_norm = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc="Training")
    for data in progress_bar:
        # Get data
        input_text = data['input'].to(device)
        target_text = data['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(input_text)
        loss = criterion(output.view(-1, output.size(-1)), target_text.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        total_grad_norm += grad_norm.item()
        
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(output, dim=-1)
        accuracy = calculate_accuracy(predictions, target_text, criterion.ignore_index)
        
        # Update metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'grad_norm': f'{grad_norm:.4f}'
        })
    
    return total_loss / num_batches, total_accuracy / num_batches, total_grad_norm / num_batches

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating")
        for data in progress_bar:
            # Get data
            input_text = data['input'].to(device)
            target_text = data['target'].to(device)
            
            # Forward pass
            output = model(input_text)
            loss = criterion(output.view(-1, output.size(-1)), target_text.view(-1))
            
            # Calculate accuracy
            predictions = torch.argmax(output, dim=-1)
            accuracy = calculate_accuracy(predictions, target_text, criterion.ignore_index)
            
            # Update metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    return total_loss / num_batches, total_accuracy / num_batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--model_path', type=str, default='best_model.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.001)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--use_lr_finder', action='store_true')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    args = parser.parse_args()

    # Initialize wandb only if requested
    if args.use_wandb:
        try:
            wandb.init(project="hebrew-ocr-correction")
            wandb.config.update(args)
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create vocabulary
    char_to_idx, idx_to_char = create_vocab(os.path.join(args.data_dir, 'train.jsonl'))
    vocab_size = len(char_to_idx)

    # Load data
    train_dataset = HebrewDataset(os.path.join(args.data_dir, 'train.jsonl'), char_to_idx)
    val_dataset = HebrewDataset(os.path.join(args.data_dir, 'val.jsonl'), char_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = HebrewOCRCorrector(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    # Initialize loss criterion
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])

    # --- LR Finder logic ---
    if args.use_lr_finder:
        print("Finding optimal learning rate...")
        log_lrs, losses = find_learning_rate(model, train_loader, criterion, device)
        # Find the learning rate with the steepest loss decrease
        min_grad = None
        best_lr = args.learning_rate
        for i in range(1, len(losses)):
            grad = (losses[i] - losses[i-1]) / (log_lrs[i] - log_lrs[i-1])
            if min_grad is None or grad < min_grad:
                min_grad = grad
                best_lr = 10 ** log_lrs[i]
        print(f"Optimal learning rate found: {best_lr:.6g}")
        args.learning_rate = best_lr
        # Re-initialize model and optimizer with the new learning rate
        model = HebrewOCRCorrector(
            vocab_size=vocab_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay
        )
        print(f"Continuing training with learning rate: {args.learning_rate:.6g}\n")

    # --- Training loop continues as before ---
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.max_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.max_epochs}")
        for i, batch in enumerate(pbar):
            input_seq = batch['input'].to(device)
            target_seq = batch['target'].to(device)
            
            output = model(input_seq)
            
            # Reshape for loss calculation
            output = output.view(-1, vocab_size)
            target = target_seq.view(-1)
            
            loss = criterion(output, target)
            loss = loss / args.gradient_accumulation_steps  # Normalize loss
            loss.backward()
            
            # Calculate accuracy
            pred = output.argmax(dim=-1)
            mask = target != char_to_idx['<PAD>']
            train_correct += (pred[mask] == target[mask]).sum().item()
            train_total += mask.sum().item()
            
            train_loss += loss.item() * args.gradient_accumulation_steps  # Scale loss back
            
            # Update weights after accumulating gradients
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar
            if pbar.n % 10 == 0:  # Update every 10 batches
                pbar.set_postfix({
                    'loss': f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    'acc': f"{train_correct/train_total:.4f}"
                })
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Evaluating")
            for batch in pbar:
                input_seq = batch['input'].to(device)
                target_seq = batch['target'].to(device)
                
                output = model(input_seq)
                
                # Reshape for loss calculation
                output = output.view(-1, vocab_size)
                target = target_seq.view(-1)
                
                loss = criterion(output, target)
                
                # Calculate accuracy
                pred = output.argmax(dim=-1)
                mask = target != char_to_idx['<PAD>']
                val_correct += (pred[mask] == target[mask]).sum().item()
                val_total += mask.sum().item()
                
                val_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{val_correct/val_total:.4f}"
                })
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Print metrics
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save model and configuration
            torch.save(model.state_dict(), args.model_path)
            config = {
                'vocab_size': vocab_size,
                'embedding_dim': args.embedding_dim,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
                'char_to_idx': char_to_idx,
                'idx_to_char': idx_to_char
            }
            with open(args.model_path.replace('.pt', '_config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Log to wandb only if enabled
        if args.use_wandb:
            try:
                wandb.log({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            except Exception as e:
                print(f"Warning: Could not log to wandb: {e}")
                print("Continuing without wandb logging...")

if __name__ == "__main__":
    main()