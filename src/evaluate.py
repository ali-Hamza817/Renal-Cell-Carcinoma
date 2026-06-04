import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import pandas as pd

from src.dataset import get_train_val_loaders
from src.model import Tiny3DCNN, get_resnet3d_10, get_resnet3d_18

def evaluate_best_model():
    """
    Loads the best saved checkpoint, evaluates it on the validation set,
    prints metrics, and generates beautiful diagnostic plots.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "data/best_model.pth"
    history_path = "data/training_history.json"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at {checkpoint_path}. Please run training first.")
        return
        
    print(f"Loading best model checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    
    # 1. Instantiate Model and load weights
    if model_name == "tiny_cnn":
        model = Tiny3DCNN()
    elif model_name == "resnet10":
        model = get_resnet3d_10()
    else:
        model = get_resnet3d_18()
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 2. Get Validation Loader
    _, val_loader = get_train_val_loaders(
        clinical_csv="data/clinical_data.csv",
        processed_dir="data/processed",
        batch_size=2,  # Smaller batch size for evaluation safety
        val_size=0.25,
        seed=42
    )
    
    # 3. Generate Predictions
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_preds.extend(probs)
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds).squeeze()
    
    binary_preds = (all_preds >= 0.5).astype(int)
    
    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        roc_auc = 0.5
        
    print("\n" + "="*50)
    print("           MODEL EVALUATION SUMMARY           ")
    print("="*50)
    print(f"Model Architecture:    {model_name}")
    print(f"Best Epoch:            {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation Loss:       {checkpoint.get('val_loss', 0.0):.4f}")
    print(f"Accuracy:              {acc*100:.2f}%")
    print(f"Precision (High Risk): {precision:.4f}")
    print(f"Recall (High Risk):    {recall:.4f}")
    print(f"F1-Score (High Risk):  {f1:.4f}")
    print(f"ROC AUC:               {roc_auc:.4f}")
    print("="*50)
    
    # Save metrics to a file
    metrics_summary = {
        "model_name": model_name,
        "best_epoch": checkpoint.get("epoch"),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc)
    }
    with open("data/evaluation_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    # 5. Diagnostic Plotting (Sleek Dark Theme)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f172a')  # Slate-900 background
    
    # A. Plot Loss & AUC Curves from Training History
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
            
        epochs_range = range(1, len(history["train_loss"]) + 1)
        ax1 = axes[0]
        ax1.set_facecolor('#1e293b')  # Slate-800 panel
        ax1.plot(epochs_range, history["train_loss"], color='#38bdf8', linewidth=2.5, label='Train Loss')  # Sky-400
        ax1.plot(epochs_range, history["val_loss"], color='#f43f5e', linewidth=2.5, linestyle='--', label='Val Loss')  # Rose-500
        ax1.set_xlabel('Epochs', fontsize=12, color='#cbd5e1')
        ax1.set_ylabel('Loss', fontsize=12, color='#cbd5e1')
        ax1.set_title('Training and Validation Loss', fontsize=14, color='white', fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.3)
        ax1.legend(facecolor='#0f172a', edgecolor='none')
        
        # Overlay AUC on a secondary y-axis
        ax1_auc = ax1.twinx()
        ax1_auc.plot(epochs_range, history["val_auc"], color='#10b981', linewidth=2.0, label='Val AUC')  # Emerald-500
        ax1_auc.set_ylabel('Validation ROC AUC', fontsize=12, color='#10b981')
        ax1_auc.tick_params(axis='y', labelcolor='#10b981')
        ax1_auc.legend(loc='lower left', facecolor='#0f172a', edgecolor='none')
        
    # B. Plot ROC Curve
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')  # Slate-800 panel
    ax2.plot(fpr, tpr, color='#8b5cf6', linewidth=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')  # Violet-500
    ax2.plot([0, 1], [0, 1], color='#64748b', linestyle=':', linewidth=1.5, label='Random Guess')  # Slate-500
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12, color='#cbd5e1')
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12, color='#cbd5e1')
    ax2.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, color='white', fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='lower right', facecolor='#0f172a', edgecolor='none')
    
    plt.suptitle(f"ccRCC Staging Risk Stratification - {model_name.upper()} Baseline", fontsize=18, color='white', fontweight='bold')
    plt.tight_layout()
    
    out_img_path = "data/evaluation_metrics.png"
    plt.savefig(out_img_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=200)
    plt.close()
    
    print(f"\n✅ Diagnostics and ROC curve saved to: {out_img_path}")

if __name__ == "__main__":
    evaluate_best_model()
