# preprocessing module

def load_ct_series(dicom_dir: str):
    """Load a DICOM series given a directory path.
    Returns a list of pydicom Dataset objects sorted by InstanceNumber.
    """
    import os, pydicom
    files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')]
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda s: int(s.InstanceNumber))
    return slices

def build_volume(slices, target_spacing=(1.0, 1.0, 1.0)):
    """Convert a list of DICOM slices into a 3‑D NumPy volume.
    Performs resampling to the given isotropic spacing.
    """
    import numpy as np, scipy.ndimage
    # Stack pixel arrays
    volume = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
    # Compute original spacing from DICOM metadata
    spacing = np.array([float(slices[0].SliceThickness),
                        float(slices[0].PixelSpacing[0]),
                        float(slices[0].PixelSpacing[1])])
    zoom = spacing / np.array(target_spacing)
    volume_resampled = scipy.ndimage.zoom(volume, zoom, order=1)
    return volume_resampled
