with open("reporte/mapa_3d_interactivo.html", "r", encoding="utf-8") as f:
    html = f.read()

import re
match = re.search(r"const trajectoriesData = (\[.*?\]);", html)
if match:
    import json
    data = json.loads(match.group(1))
    print("Number of trajectories in HTML:", len(data))
    print("Trajectory 0 in HTML:", data[0]['points'][:3])
else:
    print("Could not find trajectoriesData in HTML!")
