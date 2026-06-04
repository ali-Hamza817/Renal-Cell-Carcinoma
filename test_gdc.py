import requests
import json

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
    "size": "50"
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    hits = data.get("data", {}).get("hits", [])
    print(f"Retrieved {len(hits)} records:")
    for hit in hits[:5]:
        print(json.dumps(hit, indent=2))
except Exception as e:
    print("Error:", e)
