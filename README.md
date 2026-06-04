# Renal Cell Carcinoma (ccRCC) Risk Stratification

**Project goal**: Build a reproducible end‑to‑end AI pipeline that predicts the risk of metastatic progression for clear‑cell renal cell carcinoma (ccRCC) patients from a single CT scan and basic clinical information.

---

## Overview
This repository implements the full research workflow described in the accompanying thesis:
- **Data download** – pulls CT series from the Cancer Imaging Archive (TCIA) and clinical metadata from GDC.
- **Pre‑processing** – converts DICOM series to a normalized 3‑D NumPy volume (`[128, 128, 128]`).
- **Slice visualisation** – generates axial, coronal and sagittal slice PNGs for quick inspection.
- **Model training** – trains a 3‑D ResNet‑18 (or lightweight TinyCNN) on the pre‑processed volumes.
- **Evaluation** – reports validation loss, accuracy and ROC‑AUC.

All steps run on CPU (GPU optional) and are orchestrated by `run_pipeline.py`.

---

## Repository structure
```
Renal-Cell-Carcinoma/
│   README.md               # <-- you are here
│   requirements.txt         # Python dependencies
│   run_pipeline.py          # Orchestrator
│   .gitignore               # Ignored files/folders
│
├───data/
│   │   clinical_data.csv   # Clinical table with PatientID and RiskLabel
│   │   training_history.json
│   │   best_model.pth       # Model checkpoint (best val loss)
│   │
│   └───processed/          # 3‑D .npy volumes (128³) + visualisation PNGs
│
├───src/
│   │   __init__.py
│   │   data_download.py    # TCIA & GDC download utilities
│   │   preprocess.py       # DICOM → NumPy conversion & normalization
│   │   dataset.py          # PyTorch Dataset & DataLoaders
│   │   model.py            # 3‑D CNN / ResNet definitions
│   │   train.py            # Training loop
│   │   evaluate.py         # Evaluation utilities
│
├───visualize_slices.py     # Stand‑alone slice visualiser (CLI)
├───test.py                 # Minimal sanity‑check script
├───test_gdc.py             # GDC API sanity check
└───test_tcia.py            # TCIA API sanity check
```

---

## Setup & installation
1. **Python** – Tested with Python 3.14 (the bundled interpreter in `C:\Python314`).
2. **Dependencies** – Install via `pip`:
   ```bash
   C:\Python314\python.exe -m pip install -r requirements.txt
   ```
   Required packages include `torch`, `numpy`, `pandas`, `pydicom`, `matplotlib`, `tqdm`, `scikit-learn`, etc.
3. **Git** – Ensure you have write access to the remote repository (`https://github.com/ali-Hamza817/Renal-Cell-Carcinoma.git`).

---

## Running the pipeline
The easiest entry point is the orchestrator:
```bash
C:\Python314\python.exe -u run_pipeline.py
```
It will sequentially execute:
1. **Data download** – creates `data/raw_dicom/` and `data/clinical_data.csv`.
2. **Pre‑processing** – generates `data/processed/*.npy`.
3. **Slice visualisation** – produces `data/processed/visualization_<PatientID>.png`.
4. **Training** – trains the model (default 5 epochs, ResNet‑18) and saves `data/best_model.pth` and `data/training_history.json`.

You can also invoke individual steps manually – see each script’s `--help` output.

---

## Results (baseline)
- **Best validation loss**: 0.6954
- **Best validation AUC**: 0.75
- **Validation accuracy**: 0.5 (balanced low‑high risk set)

These numbers are from the 5‑epoch CPU run on the curated 16‑patient subset (8 Low‑Risk, 8 High‑Risk). They serve as a baseline for further experiments (e.g., multi‑modal fusion, deeper networks, GPU acceleration).

---

## Re‑use & extension
- **GPU acceleration** – set `device = torch.device('cuda')` in `src/train.py` if a CUDA‑enabled GPU is available.
- **More data** – increase the patient pool by editing `data_download.py` parameters.
- **Multi‑modal** – add clinical features or omics data by extending `RCC3DDataset`.
- **Hyper‑parameter tuning** – use Optuna (already in `requirements.txt`).

---

## Contributing
Feel free to open issues or submit pull requests. Please keep the following in mind:
- Follow the existing code style (PEP 8, docstrings).
- Update the `README` and documentation when adding new functionality.
- Add tests in the `test_*.py` files for new modules.

---

## License
This project is released under the **MIT License**. See the `LICENSE` file for details.

---

*Prepared for submission alongside the research manuscript.*
