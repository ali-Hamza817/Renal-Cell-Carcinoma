import os
import zipfile
import requests
import json
import pandas as pd
from tqdm import tqdm
import time

def get_clinical_data():
    """
    Queries the GDC API to retrieve clinical data for TCGA-KIRC patients.
    Saves the data to data/clinical_data.csv.
    """
    print("Querying GDC API for clinical data...")
    url = "https://api.gdc.cancer.gov/cases"
    
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": ["TCGA-KIRC"]
                }
            }
        ]
    }
    
    fields = [
        "case_id",
        "submitter_id",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_grade",
        "diagnoses.days_to_death",
        "diagnoses.days_to_last_follow_up",
        "demographic.vital_status",
        "demographic.days_to_death",
        "demographic.gender",
        "demographic.age_at_index"
    ]
    
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": "1000"  # TCGA-KIRC has about ~500 cases
    }
    
    # Retry mechanism for robust API calls
    for attempt in range(5):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            print(f"GDC API clinical data query attempt {attempt + 1} failed: {e}")
            if attempt == 4:
                raise e
            time.sleep(2)
            
    hits = data.get("data", {}).get("hits", [])
    records = []
    
    for hit in hits:
        patient_id = hit.get("submitter_id")
        gender = hit.get("demographic", {}).get("gender", "unknown")
        vital_status = hit.get("demographic", {}).get("vital_status", "unknown")
        age = hit.get("demographic", {}).get("age_at_index", None)
        
        # In TCGA, a patient can have multiple diagnosis records (rare, but possible)
        # We grab the first valid pathologic stage and grade
        stage = "unknown"
        grade = "unknown"
        days_to_death = hit.get("demographic", {}).get("days_to_death", None)
        days_to_followup = None
        
        diagnoses = hit.get("diagnoses", [])
        if diagnoses:
            # Look for non-null stage/grade
            for diag in diagnoses:
                if "ajcc_pathologic_stage" in diag and diag["ajcc_pathologic_stage"]:
                    stage = diag["ajcc_pathologic_stage"]
                if "tumor_grade" in diag and diag["tumor_grade"]:
                    grade = diag["tumor_grade"]
                if "days_to_last_follow_up" in diag and diag["days_to_last_follow_up"]:
                    days_to_followup = diag["days_to_last_follow_up"]
                if "days_to_death" in diag and diag["days_to_death"] and days_to_death is None:
                    days_to_death = diag["days_to_death"]
                    
        records.append({
            "PatientID": patient_id,
            "Gender": gender,
            "Age": age,
            "Stage": stage,
            "Grade": grade,
            "VitalStatus": vital_status,
            "DaysToDeath": days_to_death,
            "DaysToFollowup": days_to_followup
        })
        
    df = pd.DataFrame(records)
    # Filter out records where stage is unknown/not reported
    df = df[~df["Stage"].isin(["unknown", "Not Reported", "not reported", None])]
    
    # Map stages to binary labels
    # Stage I / Stage II -> Low Risk (0)
    # Stage III / Stage IV -> High Risk (1)
    stage_map = {
        "Stage I": 0, "Stage IA": 0, "Stage IB": 0,
        "Stage II": 0, "Stage IIA": 0, "Stage IIB": 0,
        "Stage III": 1, "Stage IIIA": 1, "Stage IIIB": 1, "Stage IIIC": 1,
        "Stage IV": 1, "Stage IVA": 1, "Stage IVB": 1, "Stage IVC": 1
    }
    df["RiskLabel"] = df["Stage"].map(stage_map)
    # Drop rows where stage mapping is not defined
    df = df.dropna(subset=["RiskLabel"])
    df["RiskLabel"] = df["RiskLabel"].astype(int)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/clinical_data.csv", index=False)
    print(f"Saved {len(df)} clinical records to data/clinical_data.csv")
    return df

def get_tcia_series():
    """
    Queries the TCIA API to find CT series for the TCGA-KIRC collection.
    """
    print("Querying TCIA REST API for series metadata...")
    # Create a session with retry strategy for robustness
    session = requests.Session()
    retry_adapter = requests.adapters.HTTPAdapter(max_retries=5)
    session.mount('https://', retry_adapter)
    
    url = "https://services.cancerimagingarchive.net/nbia-api/services/v4/getSeries"
    params = {
        "Collection": "TCGA-KIRC",
        "Modality": "CT",
        "format": "json"
    }
    
    for attempt in range(5):
        try:
            response = session.get(url, params=params, timeout=60)  # longer timeout
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            print(f"TCIA API series query attempt {attempt + 1} failed: {e}")
            if attempt == 4:
                raise e
            time.sleep(5)  # backoff
    
    print(f"Retrieved metadata for {len(data)} CT series from TCIA.")
    return data

def download_series_subset(clinical_df, series_list, num_patients=16):
    """
    Selects a balanced subset of patients and downloads their CT scans.
    """
    # Map PatientID in clinical_df to series
    patient_series = {}
    for s in series_list:
        pid = s.get("PatientID")
        if pid:
            if pid not in patient_series:
                patient_series[pid] = []
            patient_series[pid].append(s)
            
    # Get patients who have clinical records AND at least one CT scan series
    matching_patients = set(clinical_df["PatientID"]).intersection(set(patient_series.keys()))
    print(f"Found {len(matching_patients)} patients with both clinical staging and CT scans.")
    
    matched_df = clinical_df[clinical_df["PatientID"].isin(matching_patients)].copy()
    
    # Let's separate into Low Risk (0) and High Risk (1)
    low_risk_pids = matched_df[matched_df["RiskLabel"] == 0]["PatientID"].tolist()
    high_risk_pids = matched_df[matched_df["RiskLabel"] == 1]["PatientID"].tolist()
    
    print(f"Pool size: Low Risk (Stage I/II) = {len(low_risk_pids)} patients, High Risk (Stage III/IV) = {len(high_risk_pids)} patients")
    
    # We want to select a balanced subset, e.g., 8 low risk and 8 high risk patients
    half_num = num_patients // 2
    selected_low = low_risk_pids[:half_num]
    selected_high = high_risk_pids[:half_num]
    selected_pids = selected_low + selected_high
    
    print(f"Selected {len(selected_pids)} patients for download ({len(selected_low)} Low Risk, {len(selected_high)} High Risk).")
    
    download_dir = "data/raw_dicom"
    os.makedirs(download_dir, exist_ok=True)
    
    downloaded_count = 0
    
    for pid in selected_pids:
        # Find the best CT series for this patient
        # To avoid downloading scout scans or multi-phase series that are too large/small,
        # we aim for a series with a slice count (ImageCount) between 50 and 300 slices.
        series_candidates = patient_series[pid]
        best_series = None
        
        # Sort candidates so that series with ImageCount between 80 and 200 are preferred
        # because they are typical single-phase contrast-enhanced abdominal CT scans.
        for s in sorted(series_candidates, key=lambda x: abs(int(x.get("ImageCount", 0)) - 120)):
            best_series = s
            break
            
        if not best_series:
            print(f"No suitable series found for patient {pid}, skipping.")
            continue
            
        series_uid = best_series.get("SeriesInstanceUID")
        slice_count = best_series.get("ImageCount")
        risk = matched_df[matched_df["PatientID"] == pid]["RiskLabel"].values[0]
        risk_str = "Low Risk" if risk == 0 else "High Risk"
        
        patient_dir = os.path.join(download_dir, pid)
        series_dir = os.path.join(patient_dir, series_uid)
        
        if os.path.exists(series_dir) and len(os.listdir(series_dir)) > 0:
            print(f"Patient {pid} ({risk_str}, Series {series_uid}, {slice_count} slices) already downloaded. Skipping.")
            downloaded_count += 1
            continue
            
        print(f"Downloading Patient {pid} ({risk_str}) | Series: {series_uid} | Slices: {slice_count}")
        
        # Download ZIP file containing DICOM slices from TCIA getImage endpoint
        download_url = "https://services.cancerimagingarchive.net/nbia-api/services/v4/getImage"
        params = {"SeriesInstanceUID": series_uid}
        
        zip_path = f"data/temp_{series_uid}.zip"

        success = False
        # Use a session with retries for download
        download_session = requests.Session()
        retry_adapter = requests.adapters.HTTPAdapter(max_retries=5)
        download_session.mount('https://', retry_adapter)

        for attempt in range(5):
            try:
                with download_session.get(download_url, params=params, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                success = True
                break
            except Exception as e:
                print(f"Download attempt {attempt + 1} failed for patient {pid}: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                time.sleep(10)

        if not success:
            print(f"[ERROR] Failed to download series for patient {pid} after 5 attempts.")
            continue
            
        # Extract files
        try:
            os.makedirs(series_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(series_dir)
            print(f"[SUCCESS] Successfully extracted patient {pid} to {series_dir}")
            downloaded_count += 1
        except Exception as e:
            print(f"[ERROR] Error extracting ZIP for patient {pid}: {e}")
        finally:
            if os.path.exists(zip_path):
                # Try to remove with retries to handle antivirus/system locks on Windows
                for remove_attempt in range(5):
                    try:
                        os.remove(zip_path)
                        break
                    except Exception as remove_err:
                        if remove_attempt < 4:
                            time.sleep(1)
                        else:
                            print(f"[WARNING] Could not delete temporary zip {zip_path}: {remove_err}")
                
    print(f"Completed download process. Successfully downloaded {downloaded_count}/{len(selected_pids)} patients.")

if __name__ == "__main__":
    # Create the data directory
    os.makedirs("data", exist_ok=True)
    
    # 1. Download clinical data
    clinical_df = get_clinical_data()
    
    # 2. Get series list
    series_list = get_tcia_series()
    
    # 3. Download a balanced cohort of 16 patients (8 Low Risk, 8 High Risk)
    download_series_subset(clinical_df, series_list, num_patients=16)
