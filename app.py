import streamlit as st
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from src.model import get_resnet3d_18

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def load_model(checkpoint_path: str):
    """Load the ResNet-18 model from the checkpoint."""
    model = get_resnet3d_18()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_volume(npy_path: str) -> torch.Tensor:
    """Load a pre‑processed .npy volume and return a torch tensor.
    The volume is expected to be shape (depth, height, width) = (128,128,128).
    """
    vol = np.load(npy_path)
    # Ensure channel dimension
    vol = np.expand_dims(vol, axis=0)  # (1, D, H, W)
    tensor = torch.from_numpy(vol).float()
    return tensor.unsqueeze(0)  # (1, 1, D, H, W)

def predict_risk(model, volume_tensor):
    """Return a probability between 0 and 1 for the high‑risk class."""
    with torch.no_grad():
        logits = model(volume_tensor)
        prob = torch.sigmoid(logits).item()
    return prob

def show_slices(volume: np.ndarray, patient_id: str):
    """Create three orthogonal slice plots and display them with Streamlit."""
    depth, height, width = volume.shape
    axial = volume[depth // 2, :, :]
    coronal = volume[:, height // 2, :]
    sagittal = volume[:, :, width // 2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"CT slices – Patient {patient_id}", color='white')
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
    axes[0].imshow(axial, cmap='gray')
    axes[0].set_title('Axial', color='white')
    axes[0].axis('off')
    axes[1].imshow(coronal, cmap='gray')
    axes[1].set_title('Coronal', color='white')
    axes[1].axis('off')
    axes[2].imshow(sagittal, cmap='gray')
    axes[2].set_title('Sagittal', color='white')
    axes[2].axis('off')
    st.pyplot(fig)

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="ccRCC Risk Predictor", layout="centered")
st.title("🩺 ccRCC Metastatic‑Risk Prediction")
st.markdown("---")

# Locate processed data
processed_dir = os.path.join('data', 'processed')
if not os.path.isdir(processed_dir):
    st.error(f"Processed data folder not found: {processed_dir}")
    st.stop()

# List available patient IDs
patient_files = [f for f in os.listdir(processed_dir) if f.endswith('.npy')]
patient_ids = [os.path.splitext(f)[0] for f in patient_files]
if not patient_ids:
    st.error("No processed .npy files found.")
    st.stop()

selected_id = st.selectbox("Select a patient", patient_ids)

# Load volume and display slices
npy_path = os.path.join(processed_dir, f"{selected_id}.npy")
volume_np = np.load(npy_path)
show_slices(volume_np, selected_id)

# Load model (once) and predict
model_path = os.path.join('data', 'best_model.pth')
if not os.path.isfile(model_path):
    st.warning("Model checkpoint not found – prediction unavailable.")
else:
    model = load_model(model_path)
    prob = predict_risk(model, load_volume(npy_path))
    risk_level = "High" if prob >= 0.5 else "Low"
    st.metric(label="Risk probability", value=f"{prob:.3f}")
    st.success(f"Predicted risk: **{risk_level}** (threshold 0.5)")

st.caption("*Model was trained on a 16‑patient subset (8 Low, 8 High) and serves as a baseline. "
             "For research use, retrain on the full cohort or adjust hyper‑parameters.*")
