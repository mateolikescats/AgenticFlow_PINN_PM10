import json

def list_sources():
    with open('output_sources.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("| Nombre del Foco | Tipo | Coordenadas (Lat, Lon) | Elevación [m] | Emisión Estimada S [ug/(m³·s)] |")
    print("| --- | --- | --- | --- | --- |")
    for d in data:
        name = d.get("name", "Desconocido")
        stype = d.get("type", "Desconocido").capitalize()
        lat = d.get("latitud", 0.0)
        lon = d.get("longitud", 0.0)
        elev = d.get("elevacion", 0.0)
        s_val = d.get("emision_S_ug_m3_s", 0.0)
        print(f"| {name} | {stype} | `{lat:.5f}, {lon:.5f}` | {elev:.1f} | {s_val:.6f} |")

if __name__ == "__main__":
    list_sources()
