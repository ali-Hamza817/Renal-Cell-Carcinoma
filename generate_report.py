import os
import numpy as np
import torch
import pandas as pd
from src.model import get_resnet3d_18


def load_model(checkpoint_path: str = os.path.join('data', 'best_model.pth')):
    """Load the trained ResNet‑18 model from checkpoint."""
    model = get_resnet3d_18()
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def load_volume(npy_path: str) -> torch.Tensor:
    """Load a pre‑processed .npy volume and return a 5‑D tensor (B, C, D, H, W)."""
    vol = np.load(npy_path)
    # Ensure channel dimension
    vol = np.expand_dims(vol, axis=0)  # (1, D, H, W)
    tensor = torch.from_numpy(vol).float()
    return tensor.unsqueeze(0)  # (1, 1, D, H, W)


def predict_risk(model, volume_tensor: torch.Tensor) -> float:
    """Return probability of high‑risk (sigmoid of logit)."""
    with torch.no_grad():
        logit = model(volume_tensor)
        prob = torch.sigmoid(logit).item()
    return prob


def main():
    processed_dir = os.path.join('data', 'processed')
    model = load_model()
    results = []
    for fname in os.listdir(processed_dir):
        if not fname.endswith('.npy'):
            continue
        patient_id = fname.replace('.npy', '')
        vol_path = os.path.join(processed_dir, fname)
        prob = predict_risk(model, load_volume(vol_path))
        risk_pred = 'High' if prob >= 0.5 else 'Low'
        results.append({'PatientID': patient_id, 'Probability': prob, 'PredictedRisk': risk_pred})
    # Save predictions
    pred_df = pd.DataFrame(results)
    pred_path = os.path.join('data', 'prediction_results.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"DONE: Predictions written to {pred_path}\n")
    # If clinical labels are available, compute accuracy/AUC
    clinical_path = os.path.join('data', 'clinical_data.csv')
    if os.path.exists(clinical_path):
        clinical_df = pd.read_csv(clinical_path)
        merged = pd.merge(pred_df, clinical_df[['PatientID', 'RiskLabel']], on='PatientID')
        # Convert label to 0/1 (assuming 0=Low, 1=High)
        merged['TrueRisk'] = merged['RiskLabel']
        merged['PredLabel'] = (merged['Probability'] >= 0.5).astype(int)
        acc = (merged['TrueRisk'] == merged['PredLabel']).mean()
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(merged['TrueRisk'], merged['Probability'])
        print(f"Overall accuracy on the 16-patient subset: {acc:.2f}")
        print(f"Overall ROC-AUC: {auc:.3f}")
    else:
        print("Clinical data not found – cannot compute metrics.")

if __name__ == '__main__':
    main()
