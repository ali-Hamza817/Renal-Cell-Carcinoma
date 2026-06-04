import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class RCC3DDataset(Dataset):
    """
    Custom PyTorch Dataset for loading 3D preprocessed ccRCC CT scans and risk labels.
    """
    def __init__(self, clinical_df, processed_dir="data/processed", patient_ids=None, transform=False):
        self.clinical_df = clinical_df.set_index("PatientID")
        self.processed_dir = processed_dir
        self.transform = transform
        
        # If specific patient IDs are not provided, load all available in processed_dir
        if patient_ids is None:
            available_files = [f for f in os.listdir(processed_dir) if f.endswith(".npy")]
            self.patient_ids = [f.replace(".npy", "") for f in available_files]
        else:
            self.patient_ids = [pid for pid in patient_ids if os.path.exists(os.path.join(processed_dir, f"{pid}.npy"))]
            
    def __len__(self):
        return len(self.patient_ids)
        
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        npy_path = os.path.join(self.processed_dir, f"{pid}.npy")
        
        # Load preprocessed volume [depth, height, width]
        volume = np.load(npy_path)
        
        # 3D Data Augmentation
        if self.transform:
            volume = self._augment(volume)
        
        # Standardize to fixed shape (depth, height, width) = (128, 128, 128)
        target_depth, target_h, target_w = 128, 128, 128
        d, h, w = volume.shape
        # Pad depth if needed
        if d < target_depth:
            pad = target_depth - d
            volume = np.pad(volume, ((0, pad), (0, 0), (0, 0)), mode='constant')
        elif d > target_depth:
            start = (d - target_depth) // 2
            volume = volume[start:start + target_depth, :, :]
        # Pad/ crop height
        if h < target_h:
            pad = target_h - h
            volume = np.pad(volume, ((0, 0), (0, pad), (0, 0)), mode='constant')
        elif h > target_h:
            start = (h - target_h) // 2
            volume = volume[:, start:start + target_h, :]
        # Pad/ crop width
        if w < target_w:
            pad = target_w - w
            volume = np.pad(volume, ((0, 0), (0, 0), (0, pad)), mode='constant')
        elif w > target_w:
            start = (w - target_w) // 2
            volume = volume[:, :, start:start + target_w]
        
        # Add channel dimension: [1, depth, height, width]
        volume = np.expand_dims(volume, axis=0)
        
        # Convert to tensor
        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        
        # Get label from clinical DataFrame
        label = self.clinical_df.loc[pid, "RiskLabel"]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return volume_tensor, label_tensor
        
    def _augment(self, volume):
        """
        Applies basic 3D data augmentation to prevent overfitting:
        - Random flips (axial, sagittal, coronal axes)
        - Random 90-degree rotations
        - Random Gaussian noise addition
        """
        # Random flips along spatial dimensions
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0)  # depth-wise (axial)
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1)  # height-wise (coronal)
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2)  # width-wise (sagittal)
            
        # Random 90-degree rotation along a random spatial plane
        if np.random.rand() > 0.5:
            axes = np.random.choice([0, 1, 2], size=2, replace=False)
            k = np.random.choice([1, 2, 3])
            volume = np.rot90(volume, k=k, axes=axes)
            
        # Add slight random Gaussian noise
        if np.random.rand() > 0.5:
            noise_std = np.random.uniform(0.01, 0.03)
            noise = np.random.normal(0, noise_std, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0.0, 1.0)
            
        return volume.copy()

def get_train_val_loaders(clinical_csv="data/clinical_data.csv", processed_dir="data/processed", batch_size=4, val_size=0.25, seed=42):
    """
    Splits available data into train and validation sets, and returns PyTorch Dataloaders.
    """
    if not os.path.exists(clinical_csv):
        raise FileNotFoundError(f"Clinical data file not found at {clinical_csv}")
        
    df = pd.read_csv(clinical_csv)
    
    # Get intersection of patients who are in clinical_data AND have processed npy files
    processed_pids = {f.replace(".npy", "") for f in os.listdir(processed_dir) if f.endswith(".npy")}
    matched_df = df[df["PatientID"].isin(processed_pids)].copy()
    
    if len(matched_df) == 0:
        raise ValueError("No patients overlap between clinical records and preprocessed 3D volumes.")
        
    patient_ids = matched_df["PatientID"].tolist()
    labels = matched_df["RiskLabel"].tolist()
    
    # Perform stratified train/validation split
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        patient_ids, labels, test_size=val_size, stratify=labels, random_state=seed
    )
    
    print(f"Dataset split results:")
    print(f"  Total patients: {len(matched_df)}")
    print(f"  Train set: {len(train_ids)} patients (Low Risk={train_labels.count(0)}, High Risk={train_labels.count(1)})")
    print(f"  Val set: {len(val_ids)} patients (Low Risk={val_labels.count(0)}, High Risk={val_labels.count(1)})")
    
    # Create datasets
    train_dataset = RCC3DDataset(matched_df, processed_dir, train_ids, transform=True)
    val_dataset = RCC3DDataset(matched_df, processed_dir, val_ids, transform=False)
    
    # Create dataloaders
    # Note: Using pin_memory=True for efficient GPU data transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=False,  # CPU execution – pin_memory not needed
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True
    )
    
    return train_loader, val_loader
