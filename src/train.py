import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from tqdm import tqdm

from src.dataset import get_train_val_loaders
from src.model import Tiny3DCNN, get_resnet3d_10, get_resnet3d_18

def train_model(model_name="resnet18", epochs=15, batch_size=4, lr=1e-4, weight_decay=1e-2, val_size=0.25, seed=42):
    """
    Main training loop for ccRCC 3D CT risk stratification.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 1. Load Data
    train_loader, val_loader = get_train_val_loaders(
        clinical_csv="data/clinical_data.csv",
        processed_dir="data/processed",
        batch_size=batch_size,
        val_size=val_size,
        seed=seed
    )
    
    # 2. Instantiate Model
    if model_name.lower() == "tiny_cnn":
        model = Tiny3DCNN()
    elif model_name.lower() == "resnet10":
        model = get_resnet3d_10()
    else:
        model = get_resnet3d_18()
        
    model = model.to(device)
    
    # 3. Loss & Optimizer
    # BCEWithLogitsLoss is numerically stable and includes Sigmoid internally
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Cosine annealing learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Track metrics
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": []
    }
    
    best_val_loss = float("inf")
    best_val_auc = 0.0
    
    os.makedirs("data", exist_ok=True)
    best_model_path = "data/best_model.pth"
    
    print(f"\nStarting training for {epochs} epochs using {model_name}...")
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)  # shape [B, 1]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
            
        train_loss /= len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels_dev = labels.to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels_dev)
                val_loss += loss.item() * images.size(0)
                
                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs).cpu().numpy()
                
                all_labels.extend(labels.numpy())
                all_preds.extend(probs)
                
        val_loss /= len(val_loader.dataset)
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds).squeeze()
        
        # Calculate binary predictions (thresh=0.5)
        binary_preds = (all_preds >= 0.5).astype(int)
        
        # Calculate validation metrics
        val_acc = accuracy_score(all_labels, binary_preds)
        
        # ROC AUC requires at least one positive and one negative sample in the validation set
        try:
            val_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            val_auc = 0.5  # Fallback if validation set only contains one class
            
        scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        
        print(f"Epoch {epoch:02d} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")
        
        # Save best model based on validation loss & AUC
        if val_loss < best_val_loss or (val_loss <= best_val_loss and val_auc > best_val_auc):
            best_val_loss = val_loss
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "model_name": model_name
            }, best_model_path)
            print(f"[*] Best model saved! (Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f})")
            
    # Save training history to JSON
    with open("data/training_history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining completed. Best Val Loss: {best_val_loss:.4f} | Best Val AUC: {best_val_auc:.4f}")
    return history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train 3D CNN / ResNet for ccRCC staging risk stratification.")
    parser.add_argument("--model", type=str, default="resnet18", choices=["tiny_cnn", "resnet10", "resnet18"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_model(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
