import json
import urllib.request
import urllib.parse

def enrich_data():
    file_path = "datos_siata_temporal.json"
    print(f"Leyendo {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Identificar estaciones únicas para minimizar peticiones a la API
    stations = {}
    for entry in data:
        st_id = entry.get("id")
        lat = entry.get("latitud")
        lon = entry.get("longitud")
        if st_id and lat and lon and st_id not in stations:
            stations[st_id] = {"lat": lat, "lon": lon}

    print(f"Se encontraron {len(stations)} estaciones únicas.")

    # Preparar coordenadas para la petición en lote
    ids = list(stations.keys())
    lats = [stations[st_id]["lat"] for st_id in ids]
    lons = [stations[st_id]["lon"] for st_id in ids]

    lats_str = ",".join(map(str, lats))
    lons_str = ",".join(map(str, lons))

    url = f"https://elevation-api.open-meteo.com/v1/elevation?latitude={lats_str}&longitude={lons_str}"
    
    print("Consultando API de Elevación Open-Meteo...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            elevations = res_data.get("elevation", [])
            
        if len(elevations) != len(ids):
            print("Error: El número de elevaciones retornadas no coincide con las estaciones.")
            return

        # Mapear alturas devueltas a cada estación
        station_elevations = {ids[i]: elevations[i] for i in range(len(ids))}
        print("Alturas obtenidas con éxito. Ejemplo:")
        for st_id in list(station_elevations.keys())[:5]:
            print(f" - Estación {st_id}: {station_elevations[st_id]} m")

        # Escalar las alturas para la PINN (adimensionalizar z a [0, 1])
        # El Valle de Aburrá va aproximadamente desde los 1300 m hasta los 2900 m
        min_alt = 1300.0
        max_alt = 3000.0

        # Actualizar el JSON original
        for entry in data:
            st_id = entry.get("id")
            if st_id in station_elevations:
                alt = station_elevations[st_id]
                # Guardar altura real en metros
                entry["altura_metros"] = alt
                # Escalar z entre 0 y 1 para la PINN
                z_scaled = (alt - min_alt) / (max_alt - min_alt)
                entry["z"] = round(max(0.0, min(1.0, z_scaled)), 4)

        # Guardar archivo actualizado
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✅ ¡Enriquecimiento completado! Archivo {file_path} actualizado con alturas de sensores reales.")

    except Exception as e:
        print(f"❌ Error al consultar la API: {e}")
        print("Sugerencia: Si falla la conexión de red en el agente, ejecuta este script directamente en tu terminal ejecutando: python scratch/enrich_elevation.py")

if __name__ == "__main__":
    enrich_data()
