import os
import numpy as np
import pydicom
from scipy.ndimage import zoom
from tqdm import tqdm

def load_dicom_series(series_dir):
    """
    Loads all DICOM slices from a directory, sorts them by spatial position,
    and returns a 3D volume in Hounsfield Units (HU).
    """
    slices = []
    for file_name in os.listdir(series_dir):
        file_path = os.path.join(series_dir, file_name)
        # Check if the file is a valid DICOM file
        try:
            slices.append(pydicom.dcmread(file_path))
        except Exception:
            continue
            
    if not slices:
        raise ValueError(f"No valid DICOM slices found in {series_dir}")
        
    # Sort slices by their Z coordinate (ImagePositionPatient[2])
    # Some older files may not have ImagePositionPatient, in which case we fall back to SliceLocation
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        try:
            slices.sort(key=lambda x: float(x.SliceLocation))
        except AttributeError:
            print("Warning: Slices could not be physically sorted. Sorting by filename.")
            
    # Extract 3D volume
    volume = np.stack([s.pixel_array for s in slices])
    volume = volume.astype(np.float32)
    
    # Convert to Hounsfield Units (HU)
    # HU = PixelValue * RescaleSlope + RescaleIntercept
    for i, s in enumerate(slices):
        slope = getattr(s, 'RescaleSlope', 1.0)
        intercept = getattr(s, 'RescaleIntercept', 0.0)
        
        # Apply scaling to the slice
        if slope != 1.0 or intercept != 0.0:
            volume[i] = volume[i] * slope + intercept
            
    return volume

def resize_3d_volume(volume, target_shape=(64, 128, 128)):
    """
    Resizes a 3D volume to target_shape using trilinear interpolation.
    """
    current_shape = volume.shape
    zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
    
    # Order=1 specifies trilinear interpolation (good balance of speed/quality)
    resized_volume = zoom(volume, zoom_factors, order=1, prefilter=False)
    return resized_volume

def preprocess_ct_scans(raw_dir="data/raw_dicom", processed_dir="data/processed", target_shape=(64, 128, 128)):
    """
    Loads raw DICOM files, window-clips HU to soft tissue range, normalizes,
    resizes to target 3D shape, and saves as numpy files.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(raw_dir):
        print(f"Raw directory {raw_dir} does not exist. Please run downloader first.")
        return
        
    patients = os.listdir(raw_dir)
    print(f"Starting preprocessing for {len(patients)} patients...")
    
    success_count = 0
    
    for pid in tqdm(patients, desc="Preprocessing Patients"):
        patient_path = os.path.join(raw_dir, pid)
        if not os.path.isdir(patient_path):
            continue
            
        # Find series folder(s)
        series_folders = os.listdir(patient_path)
        if not series_folders:
            continue
            
        # Use the first series folder found
        series_dir = os.path.join(patient_path, series_folders[0])
        
        try:
            # 1. Load DICOMs and convert to HU
            volume = load_dicom_series(series_dir)
            
            # 2. Apply window clipping for soft tissue (kidneys/tumors)
            # Typically kidneys/soft tissue HU ranges between -100 and +400 HU.
            # Air is -1000 HU, bone is +700+ HU.
            min_hu, max_hu = -100.0, 400.0
            volume = np.clip(volume, min_hu, max_hu)
            
            # 3. Min-Max normalize to [0, 1]
            volume = (volume - min_hu) / (max_hu - min_hu)
            
            # 4. Resize to target shape (64, 128, 128)
            volume_resized = resize_3d_volume(volume, target_shape)
            
            # 5. Save preprocessed array
            out_file = os.path.join(processed_dir, f"{pid}.npy")
            np.save(out_file, volume_resized)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error preprocessing patient {pid}: {e}")
            
    print(f"Preprocessing completed. Successfully preprocessed {success_count}/{len(patients)} patients.")

if __name__ == "__main__":
    preprocess_ct_scans()
