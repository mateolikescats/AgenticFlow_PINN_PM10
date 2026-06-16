import json
import pandas as pd

with open("output_predictions.json", "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print("Columns in df:", df.columns)
print("Trajectory in row 0:", type(df.loc[0, 'trajectory']), df.loc[0, 'trajectory'][:2] if 'trajectory' in df.columns else "N/A")
