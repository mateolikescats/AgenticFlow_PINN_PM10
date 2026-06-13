import json
import pandas as pd

with open("output_predictions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Filter unique stations by latitude/longitude to see the representative state
# (since output_predictions.json contains values for multiple timestamps, let's look at the latest ones, timestamp=1.0)
df_latest = df[df["timestamp"] == 1.0].copy()

# Sort by PM2.5 concentration descending
df_latest["v_mag"] = (df_latest["pred_viento_vx_m_s"]**2 + df_latest["pred_viento_vy_m_s"]**2 + df_latest["pred_viento_vz_m_s"]**2)**0.5
print(df_latest[["latitud", "longitud", "pred_pm25_ug_m3", "pred_emision_S_ug_m3_s", "pred_viento_vx_m_s", "pred_viento_vy_m_s", "v_mag"]].sort_values(by="pred_pm25_ug_m3", ascending=False).to_string())
