import requests
import json

url = "https://services.cancerimagingarchive.net/services/v4/TCIA/query/getSeries"
params = {
    "Collection": "TCGA-KIRC",
    "Modality": "CT",
    "format": "json"
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    print(f"Retrieved {len(data)} series:")
    for series in data[:5]:
        print(json.dumps(series, indent=2))
except Exception as e:
    print("Error:", e)
