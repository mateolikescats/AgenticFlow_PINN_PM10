import pandas as pd
import json

print("=== INSPECTING DUPLICATES ===")
try:
    with open("datos_oficiales_pm25.json", "r", encoding="utf-8") as f:
        data_pm = json.load(f)
    df_pm = pd.DataFrame(data_pm)
    print(f"PM2.5 - Total rows: {len(df_pm)}")
    print(f"PM2.5 - Unique stations: {df_pm['id'].nunique()}")
    print(f"PM2.5 - Duplicate rows (exact): {df_pm.duplicated().sum()}")
    print(f"PM2.5 - Duplicates by (id, timestamp): {df_pm.duplicated(subset=['id', 'timestamp']).sum()}")
except Exception as e:
    print("Error reading PM2.5 data:", e)

try:
    with open("datos_meteorologicos_viento.json", "r", encoding="utf-8") as f:
        data_w = json.load(f)
    df_w = pd.DataFrame(data_w)
    print(f"\nWind - Total rows: {len(df_w)}")
    print(f"Wind - Unique stations: {df_w['id'].nunique()}")
    print(f"Wind - Duplicate rows (exact): {df_w.duplicated().sum()}")
    print(f"Wind - Duplicates by (id, timestamp): {df_w.duplicated(subset=['id', 't']).sum() if 't' in df_w.columns else df_w.duplicated(subset=['id', 'timestamp']).sum()}")
except Exception as e:
    print("Error reading Wind data:", e)
