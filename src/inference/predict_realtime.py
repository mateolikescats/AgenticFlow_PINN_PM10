import os
import sys
import json
import subprocess
import pandas as pd

# Asegurar que podemos importar desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.siata_scraper import SiataOfficialNetwork

def run_realtime_pipeline():
    print("======================================================================")
    print("INICIANDO INGESTA Y INFERENCIA EN TIEMPO REAL (ULTIMAS 96 HORAS)")
    print("======================================================================")

    # 1. Ejecutar el Scraper para obtener datos actualizados
    print("\n[PASO 1] Ejecutando scraper para descargar datos oficiales de SIATA...")
    scraper = SiataOfficialNetwork()
    scraper.run_pipeline()

    if not os.path.exists("data/datos_oficiales_pm25.json"):
        print("[ERROR] No se pudo generar data/datos_oficiales_pm25.json. Abortando.")
        return

    # 2. Ejecutar el Preprocesador para alinear, limpiar y escalar
    print("\n[PASO 2] Ejecutando preprocesador para escalar coordenadas y variables...")
    try:
        subprocess.run([sys.executable, "src/data/preprocessing.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ejecutando el preprocesador: {e}")
        return

    if not os.path.exists("data/datos_siata_temporal.json"):
        print("[ERROR] No se pudo generar data/datos_siata_temporal.json. Abortando.")
        return

    # 3. Filtrar datos de las últimas 48 horas de registros disponibles
    print("\n[PASO 3] Filtrando mediciones correspondientes a las últimas 48 horas...")
    try:
        # Cargar los datos oficiales crudos para obtener las marcas de tiempo reales (Unix timestamp)
        with open("data/datos_oficiales_pm25.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        df_raw = pd.DataFrame(raw_data)

        # Encontrar el rango de tiempo de los datos
        time_min = df_raw["timestamp"].min()
        time_max = df_raw["timestamp"].max()
        time_range = time_max - time_min
        
        # El corte es 96 horas antes de la última medición disponible
        cutoff_timestamp = time_max - (96 * 3600)
        
        # Filtrar registros crudos de las últimas 96 horas
        df_raw_latest = df_raw[df_raw["timestamp"] >= cutoff_timestamp].copy()
        print(f"Ultima medicion disponible en SIATA: {pd.to_datetime(time_max, unit='s')}")
        print(f"Corte de 96 horas antes: {pd.to_datetime(cutoff_timestamp, unit='s')}")
        print(f"Encontrados {len(df_raw_latest)} registros crudos en este rango.")

        if df_raw_latest.empty:
            print("[WARN] No hay registros en las últimas 96 horas. Usando los últimos 10 registros como fallback.")
            df_raw_latest = df_raw.sort_values(by="timestamp", ascending=False).head(10)

        # Cargar datos preprocesados para obtener la elevación real por ID de estación
        with open("data/datos_siata_temporal.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        df_processed = pd.DataFrame(processed_data)

        # Crear mapeo de id -> elevacion
        station_elev = df_processed.groupby("id")["elevacion_real"].first().to_dict()

        # Obtener el último registro por cada estación para no repetir coordenadas en input_points
        df_raw_latest_unique = df_raw_latest.sort_values(by="timestamp").drop_duplicates(subset=["id"], keep="last")
        print(f"Estaciones activas únicas identificadas: {len(df_raw_latest_unique)}")

        # 4. Generar data/input_points.json para predict.jl
        input_points = []
        for _, row in df_raw_latest_unique.iterrows():
            st_id = row["id"]
            # Obtener elevación de la estación procesada (o usar promedio de 1500.0 como fallback)
            elev = station_elev.get(st_id, station_elev.get(int(st_id) if isinstance(st_id, float) else st_id, 1500.0))
            
            # Calcular tiempo escalado t entre [0, 1]
            t_scaled = (row["timestamp"] - time_min) / time_range if time_range > 0 else 0.0

            input_points.append({
                "latitud": float(row["latitud"]),
                "longitud": float(row["longitud"]),
                "elevacion": float(elev),
                "timestamp": float(t_scaled)
            })

        with open("data/input_points.json", "w") as f:
            json.dump(input_points, f, indent=4)
        print("[OK] Archivo 'data/input_points.json' generado con éxito con las coordenadas de las estaciones.")

    except Exception as e:
        print(f"[ERROR] Procesando filtros de tiempo y coordenadas: {e}")
        return

    # 5. Ejecutar la inferencia de la iPINN en Julia
    print("\n[PASO 4] Ejecutando predict.jl en Julia para inferir emisiones y vientos...")
    try:
        subprocess.run(["julia", "src/inference/predict.jl"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ejecutando la predicción en Julia: {e}")
        return

    # 6. Mostrar resumen de resultados
    if os.path.exists("data/output_predictions.json"):
        with open("data/output_predictions.json", "r") as f:
            preds = json.load(f)
        print("\n======================================================================")
        print("RESUMEN DE PREDICCIONES GENERADAS EN TIEMPO REAL:")
        print("======================================================================")
        for p in preds[:5]: # Mostrar los primeros 5 como resumen
            print(f"- Estacion en Lat: {p['latitud']:.4f}, Lon: {p['longitud']:.4f}")
            print(f"  * PM2.5 Estimado: {p['pred_pm25_ug_m3']:.2f} ug/m3")
            print(f"  * Viento Estimado: [{p['pred_viento_vx_m_s']:.2f}, {p['pred_viento_vy_m_s']:.2f}] m/s")
            print(f"  * Emision Originada (S): {p['pred_emision_S_ug_m3_s']:.6f} ug/(m3*s)")
        print(f"\n... y {len(preds) - 5} predicciones mas. Resultados completos en 'data/output_predictions.json'.")
        print("======================================================================")
    else:
        print("[ERROR] No se encontró data/output_predictions.json.")

if __name__ == "__main__":
    run_realtime_pipeline()
