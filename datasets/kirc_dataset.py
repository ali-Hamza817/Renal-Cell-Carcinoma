# kirc_dataset module

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class KIRCDataset(Dataset):
    """Dataset for ccRCC patients.

    Each item returns:
        volume: torch.Tensor of shape (1, D, H, W)
        clinical: torch.Tensor of shape (C,)  # optional
        label: int (0=Low, 1=High)
    """
    def __init__(self, processed_dir: str, clinical_csv: str, split: str = "train",
                 transform=None, clinical_features=None, label_map=None):
        self.processed_dir = processed_dir
        self.clinical_df = pd.read_csv(clinical_csv)
        self.split = split
        self.transform = transform
        # Default clinical columns
        self.clinical_features = clinical_features or ["age", "gender", "tumor_stage", "tumor_grade"]
        # Map stage to binary label if not provided
        self.label_map = label_map or {"I":0, "II":0, "III":1, "IV":1}
        # Load patient list for the split (simple 80/10/10 split based on index)
        patient_ids = self.clinical_df["patient_id"].tolist()
        n = len(patient_ids)
        idxs = list(range(n))
        if split == "train":
            self.ids = [patient_ids[i] for i in idxs[:int(0.8*n)]]
        elif split == "val":
            self.ids = [patient_ids[i] for i in idxs[int(0.8*n):int(0.9*n)]]
        else:
            self.ids = [patient_ids[i] for i in idxs[int(0.9*n):]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        # Load volume
        vol_path = os.path.join(self.processed_dir, f"{pid}.npy")
        volume = np.load(vol_path).astype(np.float32)
        volume = np.expand_dims(volume, axis=0)  # (1, D, H, W)
        volume = torch.from_numpy(volume)
        # Clinical features
        row = self.clinical_df[self.clinical_df["patient_id"] == pid].iloc[0]
        clinical_vals = []
        for col in self.clinical_features:
            val = row[col]
            # Simple encoding: numeric stays, categorical to int
            if isinstance(val, str):
                val = 0 if val.lower() in ["female", "i", "ii"] else 1
            clinical_vals.append(float(val))
        clinical = torch.tensor(clinical_vals, dtype=torch.float32)
        # Label from stage
        stage = str(row["tumor_stage"]).split()[0]
        label = self.label_map.get(stage, 0)
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            volume = self.transform(volume)
        return volume, clinical, label
