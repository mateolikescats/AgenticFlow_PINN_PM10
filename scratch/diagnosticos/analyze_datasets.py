import json
import pandas as pd

print("=== DATASETS DETAILED ANALYSIS ===")

# 1. Load PM2.5 raw data
try:
    with open("datos_oficiales_pm25.json", "r", encoding="utf-8") as f:
        pm_data = json.load(f)
    df_pm = pd.DataFrame(pm_data)
    print(f"PM2.5 Dataset:")
    print(f"  - Total records: {len(df_pm)}")
    print(f"  - Unique station IDs: {sorted(df_pm['id'].unique())} (Total: {df_pm['id'].nunique()})")
    print(f"  - Unique coordinates (lat, lon): {df_pm[['latitud', 'longitud']].drop_duplicates().shape[0]}")
    print(f"  - Min timestamp: {df_pm['timestamp'].min()} ({pd.to_datetime(df_pm['timestamp'].min(), unit='s', errors='coerce')})")
    print(f"  - Max timestamp: {df_pm['timestamp'].max()} ({pd.to_datetime(df_pm['timestamp'].max(), unit='s', errors='coerce')})")
except Exception as e:
    print("Error reading PM2.5:", e)

# 2. Load Wind data
try:
    with open("datos_meteorologicos_viento.json", "r", encoding="utf-8") as f:
        wind_data = json.load(f)
    df_w = pd.DataFrame(wind_data)
    print(f"\nWind Dataset:")
    print(f"  - Total records: {len(df_w)}")
    # Clean ID to check match (e.g. "W-3.0" -> 3)
    def clean_id(val):
        try:
            return int(float(str(val).replace("W-", "")))
        except:
            return str(val)
    df_w['clean_id'] = df_w['id'].apply(clean_id)
    print(f"  - Unique station IDs: {sorted(df_w['clean_id'].unique())} (Total: {df_w['clean_id'].nunique()})")
    print(f"  - Unique coordinates (lat, lon): {df_w[['latitud', 'longitud']].drop_duplicates().shape[0]}")
    print(f"  - Unique t (normalized time): {sorted(df_w['t'].unique())}")
except Exception as e:
    print("Error reading Wind:", e)

# 3. Compare stations
if 'df_pm' in locals() and 'df_w' in locals():
    pm_stations = set(df_pm['id'].unique())
    wind_stations = set(df_w['clean_id'].unique())
    
    intersection = pm_stations.intersection(wind_stations)
    only_pm = pm_stations - wind_stations
    only_wind = wind_stations - pm_stations
    
    print(f"\nStation Overlap Analysis:")
    print(f"  - Stations present in both: {sorted(list(intersection))}")
    print(f"  - Stations only in PM2.5: {sorted(list(only_pm))}")
    print(f"  - Stations only in Wind: {sorted(list(only_wind))}")
