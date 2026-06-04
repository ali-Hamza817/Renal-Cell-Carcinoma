import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def visualize_patient_slices(patient_id, processed_dir="data/processed"):
    """
    Loads a preprocessed 3D volume from data/processed/{patient_id}.npy,
    extracts the middle axial, coronal, and sagittal slices, and plots them.
    Saves the visualization plot to data/processed/visualization_{patient_id}.png.
    """
    npy_path = os.path.join(processed_dir, f"{patient_id}.npy")
    if not os.path.exists(npy_path):
        print(f"Error: Processed file not found for patient {patient_id} at {npy_path}")
        return
        
    # Load 3D volume [depth, height, width]
    volume = np.load(npy_path)
    print(f"Loaded volume for patient {patient_id} with shape: {volume.shape}")
    
    depth, height, width = volume.shape
    
    # Extract middle slices
    axial_slice = volume[depth // 2, :, :]
    coronal_slice = volume[:, height // 2, :]
    sagittal_slice = volume[:, :, width // 2]
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"3D CT Volume Slices - Patient: {patient_id}", fontsize=16, color='white')
    
    # Sleek dark theme styling
    fig.patch.set_facecolor('#0f172a')
    for ax in axes:
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
    axes[0].imshow(axial_slice, cmap='gray')
    axes[0].set_title("Axial (Z-axis cut)", color='white', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(coronal_slice, cmap='gray', aspect=height/depth)
    axes[1].set_title("Coronal (Y-axis cut)", color='white', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(sagittal_slice, cmap='gray', aspect=width/depth)
    axes[2].set_title("Sagittal (X-axis cut)", color='white', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    out_img_path = os.path.join(processed_dir, f"visualization_{patient_id}.png")
    plt.savefig(out_img_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
    plt.close()
    
    print(f"DONE: Saved slice visualization to {out_img_path}")

if __name__ == "__main__":
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        print("No processed files found in data/processed/. Please run preprocessing first.")
        sys.exit(1)
        
    # Get first available patient ID
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith(".npy")]
    if len(sys.argv) > 1:
        patient_id = sys.argv[1]
    else:
        patient_id = processed_files[0].replace(".npy", "")
        
    visualize_patient_slices(patient_id, processed_dir)
