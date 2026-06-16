import json
import math

def check_file(path):
    print(f"Checking {path}...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Recursive function to check for NaN/Inf/None
        def inspect(val, keys=[]):
            if val is None:
                print(f"  - None value found at: {' -> '.join(map(str, keys))}")
                return 1
            elif isinstance(val, float):
                if math.isnan(val):
                    print(f"  - NaN found at: {' -> '.join(map(str, keys))}")
                    return 1
                elif math.isinf(val):
                    print(f"  - Inf found at: {' -> '.join(map(str, keys))}")
                    return 1
            elif isinstance(val, dict):
                count = 0
                for k, v in val.items():
                    count += inspect(v, keys + [k])
                return count
            elif isinstance(val, list):
                count = 0
                for i, v in enumerate(val):
                    count += inspect(v, keys + [i])
                return count
            return 0

        errors = inspect(data)
        if errors == 0:
            print("  No NaN, Inf, or None values found.")
        else:
            print(f"  Found {errors} invalid value(s).")
    except Exception as e:
        print("  Error reading file:", e)

check_file("output_predictions.json")
check_file("output_sources.json")
