import json
import numpy as np

with open("datos_siata_temporal.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lats = np.array([entry["latitud"] for entry in data])
lons = np.array([entry["longitud"] for entry in data])
xs = np.array([entry["x"] for entry in data])
zs = np.array([entry["z"] for entry in data])

# Check correlation between lons and xs
corr_x = np.corrcoef(lons, xs)[0, 1]
# Check correlation between lats and zs
corr_z = np.corrcoef(lats, zs)[0, 1]

print(f"Correlation between Longitude and x: {corr_x}")
print(f"Correlation between Latitude and z: {corr_z}")

# If they are linear, fit a line: val = slope * geo + intercept
if abs(corr_x) > 0.99:
    slope_x, intercept_x = np.polyfit(lons, xs, 1)
    print(f"x = {slope_x} * lon + {intercept_x}")
if abs(corr_z) > 0.99:
    slope_z, intercept_z = np.polyfit(lats, zs, 1)
    print(f"z = {slope_z} * lat + {intercept_z}")
