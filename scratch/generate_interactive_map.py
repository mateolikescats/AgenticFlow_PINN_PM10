import json
import os
import numpy as np
import pandas as pd

def get_terrain_height(lon, lat):
    # lat ranges from 6.0 to 6.45
    # lon ranges from -75.7 to -75.3
    # center line of the valley shifts from -75.61 (South) to -75.33 (North)
    y_scaled = (lat - 6.0) / 0.45
    z_center = 1700.0 - 420.0 * y_scaled
    lon_center = -75.61 + 0.28 * y_scaled
    dist_from_center = lon - lon_center
    z_surf = z_center + 600.0 * (dist_from_center / 0.15) ** 2
    # Subtract 80m buffer so that the terrain is always lower than any station in the vicinity
    return z_surf - 80.0

def get_color_for_pm25(val):
    ratio = min(max((val - 7.0) / 13.0, 0.0), 1.0)
    # Gradiente desde verde brillante (16, 185, 129) a rojo intenso (239, 68, 68)
    r = int(239 * ratio + 16 * (1 - ratio))
    g = int(68 * ratio + 185 * (1 - ratio))
    b = int(68 * ratio + 129 * (1 - ratio))
    return f"rgb({r},{g},{b})"

def get_color_for_emision(val):
    # Rango típico de S: 0.0 a 0.0006 ug/m3/s en Medellín
    ratio = min(max(val / 0.0006, 0.0), 1.0)
    # Gradiente desde celeste brillante (56, 189, 248) a rosa/magenta intenso (236, 72, 153)
    r = int(236 * ratio + 56 * (1 - ratio))
    g = int(72 * ratio + 189 * (1 - ratio))
    b = int(153 * ratio + 248 * (1 - ratio))
    return f"rgb({r},{g},{b})"

def generate_3d_map():
    predictions_path = "output_predictions.json"
    sources_path = "output_sources.json"
    
    if not os.path.exists(predictions_path):
        print("[ERROR] No se encontró output_predictions.json. Ejecuta primero predict.jl.")
        return
    if not os.path.exists(sources_path):
        print("[ERROR] No se encontró output_sources.json. Ejecuta primero predict.jl.")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    with open(sources_path, "r", encoding="utf-8") as f:
        sources_data = json.load(f)

    # 1. Cargar Historial de Entrenamiento y PVI para Diagnóstico
    latest_metrics = {
        "epoch": 0, "total_loss": 0.0, "pde_loss": 0.0, "bc_loss": 0.0, "data_loss": 0.0,
        "val_loss": 0.0, "val_u": 0.0, "val_T": 0.0, "val_v": 0.0, "stage": "Desconocido"
    }
    epochs_list = []
    total_loss_list = []
    pde_loss_list = []
    data_loss_list = []
    val_loss_list = []
    
    historial_path = "scratch/historial_perdidas.txt"
    if os.path.exists(historial_path):
        try:
            hist_df = pd.read_csv(historial_path)
            if not hist_df.empty:
                last_row = hist_df.iloc[-1]
                latest_metrics = {
                    "epoch": int(last_row["epoch"]),
                    "total_loss": float(last_row["total_loss"]),
                    "pde_loss": float(last_row["pde_loss"]),
                    "bc_loss": float(last_row["bc_loss"]),
                    "data_loss": float(last_row["data_loss"]),
                    "val_loss": float(last_row["val_loss"]),
                    "val_u": float(last_row["val_u"]),
                    "val_T": float(last_row["val_T"]),
                    "val_v": float(last_row["val_v"]),
                    "stage": str(last_row["stage"])
                }
                
                # Tomar las últimas 150 épocas para graficar la convergencia reciente
                chart_df = hist_df.tail(150)
                epochs_list = chart_df["epoch"].tolist()
                total_loss_list = chart_df["total_loss"].tolist()
                pde_loss_list = chart_df["pde_loss"].tolist()
                data_loss_list = chart_df["data_loss"].tolist()
                val_loss_list = chart_df["val_loss"].tolist()
        except Exception as e:
            print(f"[WARN] No se pudo leer historial_perdidas.txt: {e}")

    pvi_val = 0.0268 # valor de fallback
    pvi_path = "scratch/pvi_data.json"
    if os.path.exists(pvi_path):
        try:
            with open(pvi_path, "r", encoding="utf-8") as pf:
                pvi_data = json.load(pf)
                pvi_val = pvi_data.get("pvi", pvi_val)
        except Exception as e:
            print(f"[WARN] No se pudo leer pvi_data.json: {e}")

    # 2. Generar Receptores GeoJSON (Estaciones SIATA)
    stations_features = []
    for idx, r in df.iterrows():
        pm25 = r["pred_pm25_ug_m3"]
        s_val = r.get("pred_emision_S_ug_m3_s", 0.0)
        
        color_u = get_color_for_pm25(pm25)
        color_S = get_color_for_emision(s_val)
        
        stations_features.append({
            "type": "Feature",
            "properties": {
                "id": int(r.get("id", idx)),
                "name": f"Sensor #{int(r.get('id', idx))}",
                "latitud": float(r["latitud"]),
                "longitud": float(r["longitud"]),
                "pm25": float(pm25),
                "vx": float(r["pred_viento_vx_m_s"]),
                "vy": float(r["pred_viento_vy_m_s"]),
                "vz": float(r["pred_viento_vz_m_s"]),
                "s": float(s_val),
                "elev": float(r["elevacion"]),
                "color_u": color_u,
                "color_S": color_S,
                "color": color_u, # Receptores siempre muestran concentración
                "height_u": pm25 * 35.0, # Hacerlos cilindros 3D visibles
                "height_S": pm25 * 35.0,
                "height": pm25 * 35.0
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["longitud"]), float(r["latitud"])]
            }
        })
    stations_geojson = json.dumps({
        "type": "FeatureCollection",
        "features": stations_features
    })

    # 3. Separar Fuentes en Doble Capa: Urbanas vs Industriales Autónomas
    urban_sources_features = []
    industrial_sources_features = []
    
    urban_trajectory_features = []
    industrial_trajectory_features = []
    
    trajectories_data = [] # Para simulación temporal de partículas
    
    for s in sources_data:
        emision = float(s["emision_S_ug_m3_s"])
        pm25 = float(s.get("pm25_ug_m3", emision * 45000.0))
        st_id = int(s["id"])
        st_type = s.get("type", "industrial")
        name = s.get("name", f"Foco {st_id}")
        
        color_u = get_color_for_pm25(pm25)
        color_S = get_color_for_emision(emision)
        
        height_u = pm25 * 50.0       # e.g. 18 ug/m3 -> 900m
        height_S = emision * 1500000.0 # e.g. 0.0004 ug/(m3*s) -> 600m
        
        feat = {
            "type": "Feature",
            "properties": {
                "id": st_id,
                "name": name,
                "type": st_type,
                "latitud": float(s["latitud"]),
                "longitud": float(s["longitud"]),
                "emision": emision,
                "pm25": pm25,
                "vx": float(s["vx"]),
                "vy": float(s["vy"]),
                "elev": float(s["elevacion"]),
                "color_u": color_u,
                "color_S": color_S,
                "color": color_S, # Default: Emisión S
                "height_u": height_u,
                "height_S": height_S,
                "height": height_S
            },
            "geometry": {
                "type": "Point",
                "coordinates": [float(s["longitud"]), float(s["latitud"])]
            }
        }
        
        if st_type == "urban":
            urban_sources_features.append(feat)
        else:
            industrial_sources_features.append(feat)
            
        # Generar segmentos de trayectoria
        points = s["trajectory"]
        trajectories_data.append({
            "station_id": st_id,
            "type": st_type,
            "pm25": pm25,
            "emision": emision,
            "vx": float(s["vx"]),
            "vy": float(s["vy"]),
            "points": points
        })
        
        for step in range(len(points) - 1):
            p1 = [points[step][0], points[step][1]]
            p2 = [points[step + 1][0], points[step + 1][1]]
            
            ratio = step / (len(points) - 2) if len(points) > 2 else 1.0
            
            # Trayectoria u: escala de verde/rojo a amarillo
            r_col_u = int(239 * (1.0 - ratio) + 250 * ratio)
            g_col_u = int(68 * (1.0 - ratio) + 204 * ratio)
            b_col_u = int(68 * (1.0 - ratio) + 21 * ratio)
            traj_color_u = f"rgb({r_col_u},{g_col_u},{b_col_u})"
            
            # Trayectoria S: escala de celeste a magenta
            r_col_S = int(56 * (1.0 - ratio) + 236 * ratio)
            g_col_S = int(189 * (1.0 - ratio) + 72 * ratio)
            b_col_S = int(248 * (1.0 - ratio) + 153 * ratio)
            traj_color_S = f"rgb({r_col_S},{g_col_S},{b_col_S})"
            
            width = 6.0 * (1.0 - 0.7 * ratio)
            opacity = 0.85 * (1.0 - 0.5 * ratio)
            
            traj_feat = {
                "type": "Feature",
                "properties": {
                    "color_u": traj_color_u,
                    "color_S": traj_color_S,
                    "color": traj_color_S, # Default: Emisión S
                    "width": width,
                    "opacity": opacity,
                    "station_id": st_id
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [p1, p2]
                }
            }
            
            if st_type == "urban":
                urban_trajectory_features.append(traj_feat)
            else:
                industrial_trajectory_features.append(traj_feat)

    urban_sources_geojson = json.dumps({"type": "FeatureCollection", "features": urban_sources_features})
    industrial_sources_geojson = json.dumps({"type": "FeatureCollection", "features": industrial_sources_features})
    urban_trajectories_geojson = json.dumps({"type": "FeatureCollection", "features": urban_trajectory_features})
    industrial_trajectories_geojson = json.dumps({"type": "FeatureCollection", "features": industrial_trajectory_features})
    trajectories_data_json = json.dumps(trajectories_data)

    # 4. Generar Vectores de Viento GeoJSON
    wind_features = []
    scale = 0.0028
    for idx, r in df.iterrows():
        vx = r['pred_viento_vx_m_s']
        vy = r['pred_viento_vy_m_s']
        p1 = [r['longitud'], r['latitud']]
        p2 = [r['longitud'] + vx * scale, r['latitud'] + vy * scale]
        
        wind_features.append({
            "type": "Feature",
            "properties": {
                "color": "#38bdf8",
                "width": 3.5
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [p1, p2]
            }
        })
    wind_geojson = json.dumps({
        "type": "FeatureCollection",
        "features": wind_features
    })

    # 5. Generar GeoJSON para la Capa de Inversión Térmica
    inversion_geojson = json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "height": 2100.0,
                "base": 2092.0
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-75.7, 6.0],
                    [-75.3, 6.0],
                    [-75.3, 6.45],
                    [-75.7, 6.45],
                    [-75.7, 6.0]
                ]]
            }
        }]
    })

    # Plantilla HTML Premium de Visualización Científica con 4 Mejoras Científicas Integradas
    dashboard_template = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iPINN 3D - Panel de Diagnóstico Físico-Matemático de Contaminación</title>
    <!-- Maplibre GL JS -->
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
    <!-- Chart.js para curva de convergencia -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: #060814;
            color: #e2e8f0;
            overflow: hidden;
            height: 100vh;
            display: flex;
        }
        .dashboard {
            display: flex;
            width: 100vw;
            height: 100vh;
        }
        .sidebar {
            width: 440px;
            min-width: 400px;
            background: rgba(10, 15, 30, 0.92);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            padding: 22px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
            z-index: 10;
            box-shadow: 10px 0 35px rgba(0, 0, 0, 0.6);
        }
        .map-container {
            flex: 1;
            height: 100vh;
            position: relative;
        }
        #map {
            width: 100%;
            height: 100%;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            align-self: flex-start;
            border: 1px solid rgba(56, 189, 248, 0.3);
            margin-bottom: 5px;
        }
        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        .subtitle {
            font-size: 12px;
            color: #64748b;
            margin-top: 2px;
        }
        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 13px;
            font-weight: 600;
            color: #38bdf8;
            border-bottom: 1px solid rgba(56, 189, 248, 0.2);
            padding-bottom: 5px;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .toggle-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 6px;
        }
        .toggle-label {
            display: flex;
            align-items: center;
            gap: 12px;
            cursor: pointer;
            font-size: 12px;
            color: #cbd5e1;
            user-select: none;
        }
        .toggle-label input {
            display: none;
        }
        .toggle-custom {
            position: relative;
            width: 34px;
            height: 18px;
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            transition: all 0.3s ease;
            flex-shrink: 0;
        }
        .toggle-custom::after {
            content: '';
            position: absolute;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #cbd5e1;
            top: 2px;
            left: 2px;
            transition: all 0.3s ease;
        }
        .toggle-label input:checked + .toggle-custom {
            background-color: rgba(56, 189, 248, 0.2);
            border-color: #38bdf8;
            box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
        }
        .toggle-label input:checked + .toggle-custom::after {
            background-color: #38bdf8;
            transform: translateX(16px);
        }
        
        /* Selector de Métrica Activa */
        .metric-selector-panel {
            display: flex;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 4px;
            gap: 4px;
            margin-top: 6px;
        }
        .metric-btn {
            flex: 1;
            text-align: center;
            padding: 8px;
            font-size: 11.5px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        .metric-btn.active {
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            color: #060814;
            box-shadow: 0 2px 8px rgba(56, 189, 248, 0.3);
        }
        .metric-btn:not(.active) {
            color: #94a3b8;
        }
        .metric-btn:not(.active):hover {
            background: rgba(255, 255, 255, 0.05);
            color: #f1f5f9;
        }

        /* Controles de la Animación */
        .animation-panel {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 6px;
        }
        .anim-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        .btn-primary {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255,255,255,0.15);
            color: #cbd5e1;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11.5px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .btn-primary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: #38bdf8;
            color: #fff;
        }
        .select-dark {
            background: rgba(10, 15, 30, 0.9);
            color: #e2e8f0;
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        .select-dark:hover {
            border-color: #38bdf8;
        }
        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            outline: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: #38bdf8;
            cursor: pointer;
            box-shadow: 0 0 6px rgba(56, 189, 248, 0.6);
            transition: transform 0.1s;
        }
        
        /* Métricas de Diagnóstico */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 6px;
            margin-top: 6px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 6px 8px;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .metric-value {
            font-size: 13.5px;
            font-weight: 700;
            color: #38bdf8;
            font-family: 'Outfit', sans-serif;
        }
        .metric-label {
            font-size: 8px;
            color: #64748b;
            margin-top: 1px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }
        
        /* Paneles Interactivos Premium (Sonda y Ficha Contribución) */
        .interactive-card {
            background: rgba(56, 189, 248, 0.04);
            border: 1px dashed rgba(56, 189, 248, 0.3);
            border-radius: 8px;
            padding: 12px;
            display: none;
            flex-direction: column;
            gap: 6px;
            margin-top: 6px;
        }
        .analysis-tabs {
            display: flex;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 4px;
            gap: 4px;
            margin-top: 6px;
        }
        .analysis-tab {
            flex: 1;
            text-align: center;
            padding: 8px;
            font-size: 11.5px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        .analysis-tab.active {
            background: linear-gradient(135deg, #a855f7, #ec4899);
            color: #fff;
            box-shadow: 0 2px 8px rgba(236, 72, 153, 0.3);
        }
        .analysis-tab:not(.active) {
            color: #94a3b8;
        }
        .analysis-tab:not(.active):hover {
            background: rgba(255, 255, 255, 0.05);
            color: #f1f5f9;
        }
        .analysis-content-panel {
            background: rgba(255, 255, 255, 0.01);
            border: 1px dashed rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 6px;
            transition: all 0.3s ease;
        }
        .interactive-title {
            font-size: 12px;
            font-weight: 700;
            color: #38bdf8;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .btn-clear {
            background: transparent;
            border: none;
            color: #fca5a5;
            cursor: pointer;
            font-size: 10.5px;
            font-weight: 600;
        }
        .btn-clear:hover {
            text-decoration: underline;
        }
        .interactive-row {
            font-size: 11px;
            display: flex;
            justify-content: space-between;
            color: #cbd5e1;
        }

        /* Leyenda */
        .legend-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-top: 6px;
        }
        .legend-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            font-size: 11.5px;
            color: #cbd5e1;
            line-height: 1.4;
        }
        .legend-icon {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            flex-shrink: 0;
            margin-top: 2px;
        }
        .icon-station {
            background: linear-gradient(135deg, #10b981, #ef4444);
            border: 1px solid white;
            border-radius: 50%;
        }
        .icon-source-industrial {
            background: #ec4899;
            border: 1.5px solid white;
            border-radius: 2px;
        }
        .icon-source-urban {
            background: #38bdf8;
            border: 1.5px solid white;
            border-radius: 2px;
        }
        .icon-wind {
            background: #38bdf8;
            height: 3px;
            width: 14px;
            border-radius: 2px;
            margin-top: 7px;
        }
        .icon-trail {
            background: linear-gradient(90deg, #38bdf8, #ec4899);
            border-radius: 2px;
            height: 5px;
            width: 16px;
            margin-top: 7px;
        }
        .icon-inversion {
            background: rgba(6, 182, 212, 0.25);
            border: 1.5px solid #06b6d4;
            border-radius: 2px;
            width: 14px;
            height: 8px;
            margin-top: 5px;
        }
        
        #tooltip {
            position: absolute;
            display: none;
            background: rgba(10, 15, 30, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 12px;
            border-radius: 8px;
            pointer-events: none;
            z-index: 100;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <div>
                <span class="badge">Diagnóstico iPINN Inversa</span>
                <h1>iPINN 3D: Diagnóstico Físico</h1>
                <div class="subtitle">Análisis Inverso y Auditoría de Conservación de Masa</div>
            </div>

            <div>
                <h2>Métrica en el Mapa (Selector Crudo)</h2>
                <div class="metric-selector-panel">
                    <div id="btn-metric-S" class="metric-btn active" onclick="setMetric('S')">
                        Tasa de Emisión (S)
                    </div>
                    <div id="btn-metric-u" class="metric-btn" onclick="setMetric('u')">
                        Concentración (u)
                    </div>
                </div>
            </div>

            <div>
                <h2>Capas del Visualizador</h2>
                <div class="toggle-container">
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-terrain" checked onchange="toggleTerrain(this.checked)">
                        <span class="toggle-custom"></span>
                        Relieve 3D Satelital (Esri)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-inversion" checked onchange="toggleLayer('inversion-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Capa Inversión Térmica (2100 msnm)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-stations" checked onchange="toggleLayer('stations-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Receptores SIATA (Sensores)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-industrial-sources" checked onchange="toggleLayer('industrial-sources-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Focos Industriales (Autónomos iPINN)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-industrial-trails" checked onchange="toggleLayer('industrial-trajectories-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Estelas de Dispersión Industrial
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-urban-sources" checked onchange="toggleLayer('urban-sources-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Focos Urbanos (Tránsito)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-urban-trails" checked onchange="toggleLayer('urban-trajectories-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Estelas de Dispersión Urbana
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-particles" checked onchange="toggleLayer('particles-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Flujo Dinámico (Cúmulos)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-wind" checked onchange="toggleLayer('wind-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Vectores de Viento (PINN)
                    </label>
                </div>
            </div>

            <!-- MEJORA 2: Tarjeta de Sonda Interactiva en Sidebar -->
            <div id="probe-card" class="interactive-card">
                <div class="interactive-title">
                    <span id="probe-card-title">📡 Sonda de Viento Activa</span>
                    <button class="btn-clear" onclick="clearProbe()">Eliminar</button>
                </div>
                <div class="interactive-row">
                    <span>Coordenadas Sonda:</span>
                    <span id="probe-coords" style="font-weight:600; color:#facc15;">-</span>
                </div>
                <div class="interactive-row">
                    <span>Vector Viento [vx, vy]:</span>
                    <span id="probe-wind" style="font-weight:600; color:#38bdf8;">-</span>
                </div>
                <!-- Contenedor para detalles de la sonda científica -->
                <div id="probe-details-container"></div>
                <div style="font-size:9px; color:#64748b; margin-top:4px; line-height:1.2; border-top: 1px solid rgba(255,255,255,0.08); padding-top:4px;">
                    *Muestra la advección física resuelta por la iPINN. Las partículas sobre la línea cambian de color y tamaño en tiempo real según el nivel local de PM2.5.
                </div>
            </div>

            <!-- FASE 3: MÓDULOS DE ANÁLISIS (DE DÓNDE VIENE / HACIA DÓNDE VA) -->
            <div>
                <h2>Análisis de Dispersión</h2>
                <div class="analysis-tabs">
                    <div id="tab-de-donde-viene" class="analysis-tab active" onclick="setAnalysisMode('de-donde-viene')">
                        🔍 ¿De dónde viene?
                    </div>
                    <div id="tab-hacia-donde-va" class="analysis-tab" onclick="setAnalysisMode('hacia-donde-va')">
                        🚀 ¿Hacia dónde va?
                    </div>
                </div>
                
                <div id="analysis-panel" class="analysis-content-panel">
                    <!-- Modo: De dónde viene - Instrucciones -->
                    <div id="msg-de-donde-viene-help" style="font-size: 11px; color: #94a3b8; text-align: center; padding: 10px 5px; line-height: 1.4;">
                        👉 <b>Análisis de Receptor:</b> Haz clic en cualquier estación receptora (SIATA) en el mapa para ver de qué fuentes proviene su contaminación.
                    </div>
                    
                    <!-- Modo: De dónde viene - Resultados -->
                    <div id="card-de-donde-viene-results" style="display: none; flex-direction: column; gap: 6px;">
                        <div class="interactive-title" style="color:#818cf8;">
                            <span>📊 Ficha de Impacto del Receptor</span>
                            <button class="btn-clear" style="color:#fca5a5;" onclick="clearAnalysis()">Cerrar</button>
                        </div>
                        <div class="interactive-row">
                            <span>Sensor Seleccionado:</span>
                            <span id="contrib-station-name" style="font-weight:600; color:#818cf8;">-</span>
                        </div>
                        <div id="contrib-list" style="margin-top:2px;"></div>
                        
                        <!-- PRONÓSTICO TEMPORAL CHART -->
                        <div style="margin-top:10px; border-top:1px solid rgba(255,255,255,0.08); padding-top:10px;">
                            <div style="font-size:11px; font-weight:600; color:#cbd5e1; margin-bottom:6px;">📈 Pronóstico de Concentración (Próximas 24h)</div>
                            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 6px; height: 120px;">
                                <canvas id="forecastChart" style="width: 100%; height: 100%;"></canvas>
                            </div>
                            <div style="font-size:9px; color:#64748b; margin-top:4px; line-height:1.2;">
                                *Proyecta la variación diurna de PM2.5 modulada por el ciclo de temperatura y el viento resueltos por iPINN.
                            </div>
                        </div>

                        <div style="font-size:10px; color:#64748b; margin-top:2px; line-height:1.2;">
                            *Calcula qué porcentaje de la contaminación registrada en el sensor proviene de cada foco (industrial/urbano) usando la distancia inversa ponderada (IDW) de sus estelas de dispersión físicas.
                        </div>
                    </div>

                    <!-- Modo: Hacia dónde va - Instrucciones -->
                    <div id="msg-hacia-donde-va-help" style="display: none; font-size: 11px; color: #94a3b8; text-align: center; padding: 10px 5px; line-height: 1.4;">
                        👉 <b>Análisis de Emisor:</b> Haz clic en cualquier foco de emisión (industrial o urbano) para aislar su pluma en 3D y ver qué zonas resultan más contaminadas.
                    </div>
                    
                    <!-- Modo: Hacia dónde va - Resultados -->
                    <div id="card-hacia-donde-va-results" style="display: none; flex-direction: column; gap: 6px;">
                        <div class="interactive-title" style="color:#ec4899;">
                            <span>📊 Ficha del Foco Emisor: Impacto Directo</span>
                            <button class="btn-clear" style="color:#fca5a5;" onclick="clearAnalysis()">Cerrar</button>
                        </div>
                        <div class="interactive-row">
                            <span>Foco Seleccionado:</span>
                            <span id="source-impact-name" style="font-weight:600; color:#ec4899;">-</span>
                        </div>
                        <div id="source-impact-list" style="margin-top:2px;"></div>
                        
                        <!-- SIMULADOR DE MITIGACIÓN -->
                        <div style="margin-top:10px; border-top:1px solid rgba(255,255,255,0.08); padding-top:10px; display:flex; flex-direction:column; gap:6px;">
                            <div style="font-size:11px; display:flex; justify-content:space-between; align-items:center;">
                                <span style="font-weight:600; color:#cbd5e1;">⚙️ Simular Mitigación:</span>
                                <span id="mitigation-label" style="font-weight:700; color:#ec4899;">100% (Normal)</span>
                            </div>
                            <input type="range" id="slider-mitigation" min="0" max="100" step="5" value="100" style="accent-color:#ec4899;" oninput="updateMitigation(this.value)">
                            <div style="font-size:9.5px; color:#64748b; line-height:1.2;">
                                *Reduce la emisión de este foco interactivo. El visualizador recalculará en tiempo real el decaimiento de las partículas y el impacto dinámico sobre las estaciones receptoras.
                            </div>
                        </div>

                        <div style="font-size:10px; color:#64748b; margin-top:2px; line-height:1.2;">
                            *Determina la distribución de impacto del contaminante emitido por esta fuente hacia las estaciones receptoras siguiendo su pluma de advección.
                        </div>
                    </div>
                </div>
            </div>

            <div>
                <h2>Simulación de Flujo Temporal</h2>
                <div class="animation-panel">
                    <div class="anim-row">
                        <button id="btn-play" class="btn-primary">⏸ Pausar</button>
                        <!-- MEJORA 3: Reloj Digital de Hora Atmosférica -->
                        <span id="label-clock" style="font-size: 12.5px; font-weight: 700; color: #38bdf8;">06:00 AM</span>
                    </div>
                    <input type="range" id="slider-time" min="0" max="15" step="0.1" value="0">
                    <div class="anim-row" style="font-size: 11px; margin-top: 4px;">
                        <span id="label-time" style="color:#64748b;">Paso: 0.0</span>
                        <select id="select-speed" class="select-dark">
                            <option value="0.02">Lento</option>
                            <option value="0.05" selected>Normal</option>
                            <option value="0.1">Rápido</option>
                        </select>
                    </div>
                </div>
            </div>

            <div>
                <h2>Estado del Entrenamiento (iPINN)</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{latest_epoch}</div>
                        <div class="metric-label">Época Actual</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #c084fc;">{latest_pde_loss}</div>
                        <div class="metric-label">Loss Física (PDE)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #38bdf8;">{latest_data_loss}</div>
                        <div class="metric-label">Loss Datos (Asim)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" style="color: #818cf8;">{latest_val_loss}</div>
                        <div class="metric-label">Loss Validación</div>
                    </div>
                    <div class="metric-card" style="grid-column: span 2;">
                        <div class="metric-value" style="color: #10b981;">{latest_pvi}</div>
                        <div class="metric-label">Physics Violation Index (PVI - Div)</div>
                    </div>
                </div>
            </div>

            <div>
                <h2>Curvas de Convergencia (Pérdidas)</h2>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; margin-top: 5px;">
                    <canvas id="lossChart" style="width: 100%; height: 140px;"></canvas>
                </div>
            </div>

            <div>
                <h2>Leyenda del Mapa 3D</h2>
                <div class="legend-list">
                    <div class="legend-item">
                        <div class="legend-icon icon-station"></div>
                        <div>
                            <b>Receptores SIATA:</b> Concentraciones de PM2.5 medidas directamente en las estaciones oficiales.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-source-industrial"></div>
                        <div>
                            <b>Focos Industriales (PINK):</b> Focos autónomos de alta emisión localizados por la iPINN en laderas (e.g. Ladrilleras).
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-source-urban"></div>
                        <div>
                            <b>Focos Urbanos (BLUE):</b> Puntos céntricos de los municipios del valle (Tránsito/Ciudad).
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-trail"></div>
                        <div>
                            <b>Estelas de Dispersión:</b> Trayectorias de advección tridimensionales calculadas por el campo de viento de la PINN.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-inversion"></div>
                        <div>
                            <b>Capa de Inversión Térmica:</b> Plano de aire caliente superior que atrapa el PM2.5 en el valle (glowing glass a 2100 msnm).
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-wind"></div>
                        <div>
                            <b>Vectores de Viento:</b> Campo de velocidades del viento simulado bajo restricciones de incompresibilidad.
                        </div>
                    </div>
                    
                    <!-- ESCALA DE COLORES ICA -->
                    <div style="margin-top:10px; border-top:1px solid rgba(255,255,255,0.08); padding-top:10px;">
                        <span style="font-size:11px; font-weight:600; color:#cbd5e1; display:block; margin-bottom:5px;">Escala Oficial ICA PM2.5 (Colombia):</span>
                        <div style="display:flex; flex-direction:column; gap:4px; font-size:10px;">
                            <div style="display:flex; align-items:center; gap:8px;">
                                <div style="width:12px; height:12px; border-radius:3px; background:#10b981;"></div>
                                <span>0 - 12: <b>Bueno</b> (Sin riesgo)</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <div style="width:12px; height:12px; border-radius:3px; background:#eab308;"></div>
                                <span>12 - 37: <b>Aceptable</b> (Moderado)</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <div style="width:12px; height:12px; border-radius:3px; background:#f97316;"></div>
                                <span>37 - 54: <b>Dañino a grupos sensibles</b></span>
                            </div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <div style="width:12px; height:12px; border-radius:3px; background:#ef4444;"></div>
                                <span>54 - 150: <b>Dañino a la salud</b></span>
                            </div>
                            <div style="display:flex; align-items:center; gap:8px;">
                                <div style="width:12px; height:12px; border-radius:3px; background:#a855f7;"></div>
                                <span>150+: <b>Muy dañino / Emergencia</b></span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="map-container">
            <div id="map"></div>
        </div>
    </div>

    <!-- Div para Tooltip de Información -->
    <div id="tooltip"></div>

    <script type="text/javascript">
        // Datos inyectados desde Python
        const stationsGeoJSON = {stations_geojson_placeholder};
        const urbanSourcesGeoJSON = {urban_sources_geojson_placeholder};
        const industrialSourcesGeoJSON = {industrial_sources_geojson_placeholder};
        const urbanTrajectoriesGeoJSON = {urban_trajectories_geojson_placeholder};
        const industrialTrajectoriesGeoJSON = {industrial_trajectories_geojson_placeholder};
        const windGeoJSON = {wind_geojson_placeholder};
        const inversionGeoJSON = {inversion_geojson_placeholder};
        const trajectoriesData = {trajectories_data_placeholder};
        
        let activeMetric = 'S'; // 'S' o 'u'

        // --- FASE 3: Foco de Dispersión Activo ---
        let activeSourceId = null;
        let activeSourceType = null;
        let activeAnalysisMode = 'de-donde-viene'; // 'de-donde-viene' o 'hacia-donde-va'
        let activeStationId = null;
        let activeContributingSources = [];
        let probePoints = [];
        let backTrajectoryPoints = [];
        let activeProbeCoords = null;

        // Generar geometrías de polígonos hexagonales para fill-extrusion nativa
        function createHexagon(center, radius) {
            const coordinates = [];
            for (let i = 0; i < 6; i++) {
                const angle = (i * 60 * Math.PI) / 180;
                const dx = (radius * Math.cos(angle)) / 110000;
                const dy = (radius * Math.sin(angle)) / 111000;
                coordinates.push([center[0] + dx, center[1] + dy]);
            }
            coordinates.push(coordinates[0]); // Cerrar el polígono
            return [coordinates];
        }

        // Convertir los puntos de estaciones a hexágonos chatos (receptores) en el navegador
        stationsGeoJSON.features = stationsGeoJSON.features.map(f => {
            const coords = f.geometry.coordinates;
            return {
                type: 'Feature',
                properties: f.properties,
                geometry: {
                    type: 'Polygon',
                    coordinates: createHexagon(coords, 90)
                }
            };
        });

        // Convertir los puntos de fuentes a hexágonos para extrusión 3D
        urbanSourcesGeoJSON.features = urbanSourcesGeoJSON.features.map(f => {
            const coords = f.geometry.coordinates;
            return {
                type: 'Feature',
                properties: f.properties,
                geometry: {
                    type: 'Polygon',
                    coordinates: createHexagon(coords, 200)
                }
            };
        });
        
        industrialSourcesGeoJSON.features = industrialSourcesGeoJSON.features.map(f => {
            const coords = f.geometry.coordinates;
            return {
                type: 'Feature',
                properties: f.properties,
                geometry: {
                    type: 'Polygon',
                    coordinates: createHexagon(coords, 200)
                }
            };
        });

        // Copiar valores base para recálculo de mitigación dinámico
        stationsGeoJSON.features.forEach(f => {
            f.properties.base_pm25 = f.properties.pm25;
            f.properties.base_color_u = f.properties.color_u;
            f.properties.base_height_u = f.properties.height_u;
        });

        // Inicializar factores de mitigación en 1.0 para todas las fuentes
        const mitigationFactors = {};
        trajectoriesData.forEach(traj => {
            mitigationFactors[traj.station_id] = 1.0;
        });

        // Helper para clasificación oficial ICA (AQI Colombiano)
        function getICA(pm25) {
            if (pm25 <= 12.0) return { label: 'Bueno', color: '#10b981', bg: 'rgba(16, 185, 129, 0.15)', border: 'rgba(16, 185, 129, 0.3)' };
            if (pm25 <= 37.0) return { label: 'Aceptable', color: '#eab308', bg: 'rgba(234, 179, 8, 0.15)', border: 'rgba(234, 179, 8, 0.3)' };
            if (pm25 <= 54.0) return { label: 'Dañino a grupos sensibles', color: '#f97316', bg: 'rgba(249, 115, 22, 0.15)', border: 'rgba(249, 115, 22, 0.3)' };
            if (pm25 <= 150.0) return { label: 'Dañino a la salud', color: '#ef4444', bg: 'rgba(239, 68, 68, 0.15)', border: 'rgba(239, 68, 68, 0.3)' };
            if (pm25 <= 250.0) return { label: 'Muy dañino', color: '#a855f7', bg: 'rgba(168, 85, 247, 0.15)', border: 'rgba(168, 85, 247, 0.3)' };
            return { label: 'Peligroso', color: '#7f1d1d', bg: 'rgba(127, 29, 29, 0.15)', border: 'rgba(127, 29, 29, 0.3)' };
        }

        // Helper para mapear PM2.5 a una escala de color relativa (verde a rojo) matching de python
        function getRelativeColorForPM25(val) {
            let ratio = (val - 7.0) / 13.0;
            ratio = Math.max(0.0, Math.min(1.0, ratio));
            
            // Verde (16, 185, 129) a Rojo (239, 68, 68)
            let r = Math.round(239 * ratio + 16 * (1 - ratio));
            let g = Math.round(68 * ratio + 185 * (1 - ratio));
            let b = Math.round(68 * ratio + 129 * (1 - ratio));
            
            return `rgb(${r},${g},${b})`;
        }

        // Inicializar mapa de Maplibre GL JS con imágenes satelitales de Esri
        const map = new maplibregl.Map({
            container: 'map',
            style: {
                version: 8,
                sources: {
                    'satellite': {
                        type: 'raster',
                        tiles: [
                            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                        ],
                        tileSize: 256,
                        attribution: 'Tiles &copy; Esri'
                    }
                },
                layers: [
                    {
                        id: 'satellite',
                        type: 'raster',
                        source: 'satellite',
                        minzoom: 0,
                        maxzoom: 20
                    }
                ]
            },
            center: [-75.567, 6.2518], // Medellín Centro
            zoom: 11.6,
            pitch: 58,
            bearing: -12,
            maxZoom: 19,
            minZoom: 9
        });

        // Controles de navegación nativos
        map.addControl(new maplibregl.NavigationControl());

        map.on('load', () => {
            // 1. Agregar terreno 3D (Terrarium)
            map.addSource('terrain', {
                type: 'raster-dem',
                tiles: [
                    'https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png'
                ],
                encoding: 'terrarium',
                tileSize: 256,
                maxzoom: 15
            });
            map.setTerrain({ source: 'terrain', exaggeration: 1.5 });

            // MEJORA 1: Capa de Inversión Térmica flotante 3D (tipo glowing glass a 2100 msnm)
            map.addSource('inversion-source', {
                type: 'geojson',
                data: inversionGeoJSON
            });
            map.addLayer({
                id: 'inversion-layer',
                type: 'fill-extrusion',
                source: 'inversion-source',
                paint: {
                    'fill-extrusion-color': '#06b6d4',
                    'fill-extrusion-height': ['get', 'height'],
                    'fill-extrusion-base': ['get', 'base'],
                    'fill-extrusion-opacity': 0.23
                }
            });

            // MEJORA 2: Soportes para Sonda de Viento Interactiva
            map.addSource('probe-source', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: [] }
            });
            map.addLayer({
                id: 'probe-layer-line',
                type: 'line',
                source: 'probe-source',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': '#facc15',
                    'line-width': 4.5,
                    'line-opacity': 0.9
                },
                filter: ['==', '$type', 'LineString']
            });
            map.addLayer({
                id: 'probe-layer-point',
                type: 'circle',
                source: 'probe-source',
                paint: {
                    'circle-color': '#facc15',
                    'circle-radius': 7,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#060814',
                    'circle-opacity': 1.0
                },
                filter: ['==', '$type', 'Point']
            });

            // Capa para Trayectoria Retrógada (De Dónde Viene - Viento hacia atrás)
            map.addSource('back-trajectory-source', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: [] }
            });
            map.addLayer({
                id: 'back-trajectory-layer-line',
                type: 'line',
                source: 'back-trajectory-source',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': '#06b6d4', // Celeste brillante / Cian
                    'line-width': 4.5,
                    'line-opacity': 0.85,
                    'line-dasharray': [2, 2] // Discontinua para indicar procedencia
                },
                filter: ['==', '$type', 'LineString']
            });
            map.addLayer({
                id: 'back-trajectory-layer-point',
                type: 'circle',
                source: 'back-trajectory-source',
                paint: {
                    'circle-color': '#06b6d4',
                    'circle-radius': 7,
                    'circle-stroke-width': 2,
                    'circle-stroke-color': '#060814',
                    'circle-opacity': 1.0
                },
                filter: ['==', '$type', 'Point']
            });

            // 2. Capa de Estelas de Dispersión (Urbana)
            map.addSource('urban-trajectories-source', {
                type: 'geojson',
                data: urbanTrajectoriesGeoJSON
            });
            map.addLayer({
                id: 'urban-trajectories-layer',
                type: 'line',
                source: 'urban-trajectories-source',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': ['get', 'color'],
                    'line-width': ['get', 'width'],
                    'line-opacity': ['get', 'opacity']
                }
            });

            // 3. Capa de Estelas de Dispersión (Industrial)
            map.addSource('industrial-trajectories-source', {
                type: 'geojson',
                data: industrialTrajectoriesGeoJSON
            });
            map.addLayer({
                id: 'industrial-trajectories-layer',
                type: 'line',
                source: 'industrial-trajectories-source',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': ['get', 'color'],
                    'line-width': ['get', 'width'],
                    'line-opacity': ['get', 'opacity']
                }
            });

            // 4. Capa de Flujo de Partículas (Cúmulos) - Compartida y actualizada en el bucle
            map.addSource('particles-source', {
                type: 'geojson',
                data: { type: 'FeatureCollection', features: [] }
            });
            map.addLayer({
                id: 'particles-layer',
                type: 'circle',
                source: 'particles-source',
                paint: {
                    'circle-color': ['get', 'color'],
                    'circle-radius': ['get', 'radius'],
                    'circle-opacity': ['get', 'opacity'],
                    'circle-blur': 1.0,
                    'circle-stroke-width': 0
                }
            });

            // 5. Capa de Vectores de Viento
            map.addSource('wind-source', {
                type: 'geojson',
                data: windGeoJSON
            });
            map.addLayer({
                id: 'wind-layer',
                type: 'line',
                source: 'wind-source',
                layout: { 'line-join': 'round', 'line-cap': 'round' },
                paint: {
                    'line-color': ['get', 'color'],
                    'line-width': ['get', 'width'],
                    'line-opacity': 0.8
                }
            });

            // 6. Capa de Receptores SIATA
            map.addSource('stations-source', {
                type: 'geojson',
                data: stationsGeoJSON
            });
            map.addLayer({
                id: 'stations-layer',
                type: 'fill-extrusion',
                source: 'stations-source',
                paint: {
                    'fill-extrusion-color': ['get', 'color'],
                    'fill-extrusion-height': ['get', 'height'],
                    'fill-extrusion-base': 0,
                    'fill-extrusion-opacity': 0.8
                }
            });

            // 7. Capa de Fuentes Urbanas
            map.addSource('urban-sources-source', {
                type: 'geojson',
                data: urbanSourcesGeoJSON
            });
            map.addLayer({
                id: 'urban-sources-layer',
                type: 'fill-extrusion',
                source: 'urban-sources-source',
                paint: {
                    'fill-extrusion-color': ['get', 'color'],
                    'fill-extrusion-height': ['get', 'height'],
                    'fill-extrusion-base': 0,
                    'fill-extrusion-opacity': 0.85
                }
            });

            // 8. Capa de Fuentes Industriales
            map.addSource('industrial-sources-source', {
                type: 'geojson',
                data: industrialSourcesGeoJSON
            });
            map.addLayer({
                id: 'industrial-sources-layer',
                type: 'fill-extrusion',
                source: 'industrial-sources-source',
                paint: {
                    'fill-extrusion-color': ['get', 'color'],
                    'fill-extrusion-height': ['get', 'height'],
                    'fill-extrusion-base': 0,
                    'fill-extrusion-opacity': 0.85
                }
            });

            // Inicializar simulación temporal, hover tooltips, sonda y panel de análisis
            initAnimation();
            setupTooltips();
            setupInteractiveProbe();
            updateAnalysisDOM();
        });

        // Alternar visualización de capas de Maplibre
        function toggleLayer(layerId, visible) {
            if (map.getLayer(layerId)) {
                map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
            }
        }

        // Alternar el Terreno 3D satelital
        function toggleTerrain(visible) {
            map.setTerrain(visible ? { source: 'terrain', exaggeration: 1.5 } : null);
        }

        // --- MANEJO DEL SELECTOR DE MÉTRICA ---
        function setMetric(metric) {
            if (activeMetric === metric) return;
            activeMetric = metric;

            // Actualizar botones en UI
            document.getElementById('btn-metric-S').classList.toggle('active', metric === 'S');
            document.getElementById('btn-metric-u').classList.toggle('active', metric === 'u');

            // Receptores siempre muestran concentración (u)
            map.setPaintProperty('stations-layer', 'fill-extrusion-color', ['get', 'color_u']);
            map.setPaintProperty('stations-layer', 'fill-extrusion-height', ['get', 'height_u']);

            // Actualizar propiedades de extrusión de Fuentes Urbanas
            map.setPaintProperty('urban-sources-layer', 'fill-extrusion-color', ['get', 'color_' + metric]);
            map.setPaintProperty('urban-sources-layer', 'fill-extrusion-height', ['get', 'height_' + metric]);

            // Actualizar propiedades de extrusión de Fuentes Industriales
            map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-color', ['get', 'color_' + metric]);
            map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-height', ['get', 'height_' + metric]);

            // Actualizar colores de las Estelas (Trayectorias)
            map.setPaintProperty('urban-trajectories-layer', 'line-color', ['get', 'color_' + metric]);
            map.setPaintProperty('industrial-trajectories-layer', 'line-color', ['get', 'color_' + metric]);

            // Forzar actualización inmediata del flujo de partículas
            updateParticles(tGlobal);
        }

        // --- INTERPOLADOR DE VIENTOS Y SONDA INTERACTIVA (MEJORA 2) ---
        function getTerrainHeightJS(lon, lat) {
            let y_scaled = (lat - 6.0) / 0.45;
            let z_center = 1700.0 - 420.0 * y_scaled;
            let lon_center = -75.61 + 0.28 * y_scaled;
            let dist_from_center = lon - lon_center;
            let z_surf = z_center + 600.0 * Math.pow(dist_from_center / 0.15, 2);
            return z_surf - 80.0;
        }

        function interpolateWindJS(lon, lat) {
            // Estaciones SIATA inyectadas
            let lons = stationsGeoJSON.features.map(f => f.properties.longitud);
            let lats = stationsGeoJSON.features.map(f => f.properties.latitud);
            let vxs = stationsGeoJSON.features.map(f => f.properties.vx);
            let vys = stationsGeoJSON.features.map(f => f.properties.vy);
            
            let sum_weights = 0;
            let sum_vx = 0;
            let sum_vy = 0;
            
            for (let i = 0; i < lons.length; i++) {
                let dist = Math.sqrt((lons[i] - lon)**2 + (lats[i] - lat)**2);
                if (dist < 1e-5) {
                    return [vxs[i], vys[i]];
                }
                let w = 1.0 / (dist * dist);
                sum_weights += w;
                sum_vx += vxs[i] * w;
                sum_vy += vys[i] * w;
            }
            
            if (sum_weights === 0) return [0, 0];
            return [sum_vx / sum_weights, sum_vy / sum_weights];
        }

        function interpolatePM25JS(lon, lat) {
            let lons = stationsGeoJSON.features.map(f => f.properties.longitud);
            let lats = stationsGeoJSON.features.map(f => f.properties.latitud);
            let pms = stationsGeoJSON.features.map(f => f.properties.pm25);
            
            let sum_weights = 0;
            let sum_pm = 0;
            
            for (let i = 0; i < lons.length; i++) {
                let dist = Math.sqrt((lons[i] - lon)**2 + (lats[i] - lat)**2);
                if (dist < 1e-5) {
                    return pms[i];
                }
                let w = 1.0 / (dist * dist + 0.0004); // Evitar división por cero
                sum_weights += w;
                sum_pm += pms[i] * w;
            }
            
            if (sum_weights === 0) return 12.0; // Default
            return sum_pm / sum_weights;
        }

        function setupInteractiveProbe() {
            map.on('click', (e) => {
                // Evitar clics sobre estaciones y focos para no solapar con sus fichas correspondientes
                const features = map.queryRenderedFeatures(e.point, { 
                    layers: ['stations-layer', 'urban-sources-layer', 'industrial-sources-layer'] 
                });
                if (features.length > 0) return;

                const lon = e.lngLat.lng;
                const lat = e.lngLat.lat;

                createWindProbe(lon, lat);
            });
        }

        function createWindProbe(lon, lat) {
            const points = [];
            let cur_lon = lon;
            let cur_lat = lat;
            
            points.push([cur_lon, cur_lat]);
            
            // Factor dt para simular la advección geográfica
            const dt = 0.0015;
            
            // Determinar dirección de advección según el modo de análisis activo
            const isBackward = (activeAnalysisMode === 'de-donde-viene');
            const sign = isBackward ? -1.0 : 1.0;
            
            // Datos a lo largo de la trayectoria para el reporte de sonda
            let pathPM25s = [];
            let pathSpeeds = [];
            
            let pm_init = interpolatePM25JS(lon, lat);
            let [vx_init, vy_init] = interpolateWindJS(lon, lat);
            let speed_init = Math.sqrt(vx_init*vx_init + vy_init*vy_init);
            
            pathPM25s.push(pm_init);
            pathSpeeds.push(speed_init);

            for (let step = 0; step < 20; step++) {
                const [vx, vy] = interpolateWindJS(cur_lon, cur_lat);
                cur_lon += sign * vx * dt;
                cur_lat += sign * vy * dt;
                
                // Confinar al cuadro
                cur_lon = Math.min(Math.max(cur_lon, -75.7), -75.3);
                cur_lat = Math.min(Math.max(cur_lat, 6.0), 6.45);
                
                points.push([cur_lon, cur_lat]);
                
                // Muestrear en este punto de la ruta
                pathPM25s.push(interpolatePM25JS(cur_lon, cur_lat));
                pathSpeeds.push(Math.sqrt(vx*vx + vy*vy));
            }

            probePoints = points;
            activeProbeCoords = { lon, lat };

            // Cambiar dinámicamente la apariencia de la línea en el mapa
            if (map.getLayer('probe-layer-line')) {
                if (isBackward) {
                    map.setPaintProperty('probe-layer-line', 'line-color', '#22d3ee'); // Celeste / Cian
                    map.setPaintProperty('probe-layer-line', 'line-dasharray', [2, 2]); // Línea punteada (retro)
                } else {
                    map.setPaintProperty('probe-layer-line', 'line-color', '#facc15'); // Amarillo
                    map.setPaintProperty('probe-layer-line', 'line-dasharray', [10, 0]); // Línea continua
                }
            }
            if (map.getLayer('probe-layer-point')) {
                map.setPaintProperty('probe-layer-point', 'circle-color', isBackward ? '#22d3ee' : '#facc15');
            }

            const probeGeoJSON = {
                type: 'FeatureCollection',
                features: [
                    {
                        type: 'Feature',
                        properties: {},
                        geometry: {
                            type: 'Point',
                            coordinates: [lon, lat]
                        }
                    },
                    {
                        type: 'Feature',
                        properties: {},
                        geometry: {
                            type: 'LineString',
                            coordinates: points
                        }
                    }
                ]
            };

            if (map.getSource('probe-source')) {
                map.getSource('probe-source').setData(probeGeoJSON);
            }

            // Reporte analítico en sidebar
            let minPM = Math.min(...pathPM25s);
            let maxPM = Math.max(...pathPM25s);
            let avgSpeed = pathSpeeds.reduce((a,b)=>a+b, 0) / pathSpeeds.length;
            
            let ica_init = getICA(pm_init);
            let ica_max = getICA(maxPM);
            let ica_min = getICA(minPM);

            let diag = "";
            let probeColor = isBackward ? '#22d3ee' : '#facc15';
            
            if (avgSpeed < 0.6) {
                diag = "⚠️ <b>Estancamiento local:</b> Viento muy débil en la ruta. Alto riesgo de acumulación de contaminantes.";
            } else if (maxPM > 37.0) {
                diag = "⚠️ <b>Zona Crítica cruzada:</b> La trayectoria atraviesa un hotspot de contaminación elevada.";
            } else if (avgSpeed > 1.2 && maxPM <= 12.0) {
                diag = "✅ <b>Ventilación Activa:</b> Viento fuerte y aire limpio en todo el recorrido evaluado.";
            } else {
                diag = "ℹ️ <b>Dispersión Estable:</b> Vientos promedio y concentraciones moderadas a lo largo de la ruta.";
            }

            // Actualizar interfaz
            const probeTitle = document.getElementById('probe-card-title');
            probeTitle.innerHTML = isBackward ? '🔍 Sonda Retrógrada Activa' : '🚀 Sonda de Dispersión Activa';
            probeTitle.style.color = probeColor;
            document.getElementById('probe-card').style.borderColor = isBackward ? 'rgba(34,211,238,0.3)' : 'rgba(250,204,21,0.3)';
            document.getElementById('probe-card').style.background = isBackward ? 'rgba(34,211,238,0.02)' : 'rgba(250,204,21,0.02)';
            
            document.getElementById('probe-card').style.display = 'flex';
            document.getElementById('probe-coords').innerText = `${lat.toFixed(4)}°, ${lon.toFixed(4)}°`;
            document.getElementById('probe-wind').innerText = `[${vx_init.toFixed(2)}, ${vy_init.toFixed(2)}] m/s (Inic)`;

            let detailsHTML = `
                <div style="margin-top:6px; font-size:10.5px; border-top:1px solid rgba(255,255,255,0.08); padding-top:6px; display:flex; flex-direction:column; gap:4px;">
                    <div style="display:flex; justify-content:space-between;">
                        <span>PM2.5 en Punto:</span>
                        <span><b>${pm_init.toFixed(1)} ug/m³</b> <span style="display:inline-block; padding:1px 4px; border-radius:2px; font-size:8px; font-weight:700; color:${ica_init.color}; background:${ica_init.bg}; border:1px solid ${ica_init.border};">${ica_init.label}</span></span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Viento Promedio:</span>
                        <span><b>${avgSpeed.toFixed(2)} m/s</b></span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Máx PM2.5 en Ruta:</span>
                        <span><b>${maxPM.toFixed(1)} ug/m³</b> <span style="display:inline-block; padding:1px 4px; border-radius:2px; font-size:8px; font-weight:700; color:${ica_max.color}; background:${ica_max.bg}; border:1px solid ${ica_max.border};">${ica_max.label}</span></span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Mín PM2.5 en Ruta:</span>
                        <span><b>${minPM.toFixed(1)} ug/m³</b> <span style="display:inline-block; padding:1px 4px; border-radius:2px; font-size:8px; font-weight:700; color:${ica_min.color}; background:${ica_min.bg}; border:1px solid ${ica_min.border};">${ica_min.label}</span></span>
                    </div>
                    <div style="margin-top:4px; font-size:9.5px; color:#cbd5e1; background:rgba(255,255,255,0.04); border-radius:4px; padding:5px; border-left:2.5px solid ${probeColor}; line-height:1.3;">
                        ${diag}
                    </div>
                </div>
            `;
            document.getElementById('probe-details-container').innerHTML = detailsHTML;
        }

        function clearProbe() {
            probePoints = [];
            activeProbeCoords = null;
            if (map.getSource('probe-source')) {
                map.getSource('probe-source').setData({ type: 'FeatureCollection', features: [] });
            }
            document.getElementById('probe-details-container').innerHTML = '';
            document.getElementById('probe-card').style.display = 'none';
        }

        // Función para calcular y renderizar la trayectoria dinámica (hacia adelante o hacia atrás)
        function createDynamicTrajectory(lon, lat, isBackward) {
            const points = [];
            let cur_lon = lon;
            let cur_lat = lat;
            
            points.push([cur_lon, cur_lat]);
            
            const dt = 0.0015;
            const sign = isBackward ? -1.0 : 1.0;
            
            for (let step = 0; step < 25; step++) {
                const [vx, vy] = interpolateWindJS(cur_lon, cur_lat);
                cur_lon += sign * vx * dt;
                cur_lat += sign * vy * dt;
                
                // Confinar al cuadro geográfico de simulación
                cur_lon = Math.min(Math.max(cur_lon, -75.7), -75.3);
                cur_lat = Math.min(Math.max(cur_lat, 6.0), 6.45);
                
                points.push([cur_lon, cur_lat]);
            }

            const geoJSON = {
                type: 'FeatureCollection',
                features: [
                    {
                        type: 'Feature',
                        properties: {},
                        geometry: {
                            type: 'Point',
                            coordinates: [lon, lat]
                        }
                    },
                    {
                        type: 'Feature',
                        properties: {},
                        geometry: {
                            type: 'LineString',
                            coordinates: points
                        }
                    }
                ]
            };

            if (map.getSource('back-trajectory-source')) {
                map.getSource('back-trajectory-source').setData(geoJSON);
            }
            backTrajectoryPoints = points;
            
            // Actualizar apariencia de la capa dinámica en el mapa
            if (map.getLayer('back-trajectory-layer-line')) {
                if (isBackward) {
                    map.setPaintProperty('back-trajectory-layer-line', 'line-color', '#06b6d4'); // Cian
                    map.setPaintProperty('back-trajectory-layer-line', 'line-dasharray', [2, 2]); // Discontinua
                } else {
                    map.setPaintProperty('back-trajectory-layer-line', 'line-color', '#facc15'); // Amarillo
                    map.setPaintProperty('back-trajectory-layer-line', 'line-dasharray', [10, 0]); // Continua
                }
            }
            if (map.getLayer('back-trajectory-layer-point')) {
                map.setPaintProperty('back-trajectory-layer-point', 'circle-color', isBackward ? '#06b6d4' : '#facc15');
            }
        }

        // Función unificada para recalcular la trayectoria y visualizaciones activas basadas en la pestaña
        function updateActiveAnalysis() {
            const isBackward = (activeAnalysisMode === 'de-donde-viene');
            
            // 1. Si hay un receptor seleccionado (Estación SIATA)
            if (activeStationId !== null) {
                const stFeature = stationsGeoJSON.features.find(f => f.properties.id === activeStationId);
                if (stFeature) {
                    const p = stFeature.properties;
                    createDynamicTrajectory(p.longitud, p.latitud, isBackward);
                }
            } 
            // 2. Si hay un foco emisor seleccionado
            else if (activeSourceId !== null) {
                const layerGeoJSON = (activeSourceType === 'urban') ? urbanSourcesGeoJSON : industrialSourcesGeoJSON;
                const srcFeature = layerGeoJSON.features.find(f => f.properties.id === activeSourceId);
                if (srcFeature) {
                    const p = srcFeature.properties;
                    if (isBackward) {
                        createDynamicTrajectory(p.longitud, p.latitud, true);
                    } else {
                        // En modo dispersión usamos sus estelas precalculadas y borramos la dinámica
                        if (map.getSource('back-trajectory-source')) {
                            map.getSource('back-trajectory-source').setData({ type: 'FeatureCollection', features: [] });
                        }
                        backTrajectoryPoints = [];
                    }
                }
            } 
            // 3. Si hay una sonda de viento activa
            else if (activeProbeCoords !== null) {
                createWindProbe(activeProbeCoords.lon, activeProbeCoords.lat);
            }
            
            updateTrajectoryHighlight();
            updateParticles(tGlobal);
            updateAnalysisDOM();
        }

        // --- MATRIZ DE CONTRIBUCIÓN DE FUENTES (MEJORA 4) ---
        function calculateContribution(st_lon, st_lat, st_name) {
            const contributions = [];
            
            trajectoriesData.forEach(traj => {
                let min_d = Infinity;
                traj.points.forEach(pt => {
                    let d = Math.sqrt((pt[0] - st_lon)**2 + (pt[1] - st_lat)**2);
                    if (d < min_d) min_d = d;
                });
                
                let weight = 1.0 / (min_d * min_d + 0.0004);
                let source_intensity = traj.type === 'industrial' ? traj.emision * 1200.0 : traj.pm25 * 0.06;
                
                contributions.push({
                    id: traj.station_id,
                    name: traj.type === 'urban' ? getUrbanName(traj.station_id) : getIndustrialName(traj.station_id),
                    weight: weight * source_intensity,
                    type: traj.type
                });
            });
            
            let total_weight = contributions.reduce((sum, c) => sum + c.weight, 0);
            contributions.forEach(c => {
                c.percentage = total_weight > 0 ? (c.weight / total_weight) * 100.0 : 0.0;
            });
            
            contributions.sort((a, b) => b.percentage - a.percentage);
            
            // Guardar fuentes contribuyentes principales (ej. aporte > 10%)
            activeContributingSources = contributions
                .filter(c => c.percentage > 10.0)
                .map(c => c.id);

            document.getElementById('contrib-station-name').innerText = st_name;
            
            let html = '<table style="width:100%; font-size:11px; margin-top:8px; border-collapse:collapse; color:#e2e8f0;">';
            html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.12); color:#64748b; font-weight:600;"><th style="text-align:left; padding:4px 0;">Foco de Origen</th><th style="text-align:right; padding:4px 0;">Aporte</th></tr>';
            
            contributions.slice(0, 5).forEach(c => {
                let color = c.type === 'urban' ? '#38bdf8' : '#ec4899';
                let label = c.type === 'urban' ? 'Urbano' : 'Ladera';
                html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.04);"><td style="padding:6px 0; text-align:left;"><span style="color:${color}; font-weight:600;">${c.name}</span> <span style="font-size:8.5px; color:#64748b;">(${label})</span></td><td style="padding:6px 0; text-align:right; font-weight:700;">${c.percentage.toFixed(1)}%</td></tr>`;
            });
            html += '</table>';
            
            document.getElementById('contrib-list').innerHTML = html;
            
            // Actualizar gráfico de pronóstico de 24 horas para este receptor
            const stFeature = stationsGeoJSON.features.find(f => f.properties.id === activeStationId);
            if (stFeature) {
                updateForecastChart(activeStationId, stFeature.properties.pm25);
            }
        }

        function getUrbanName(id) {
            const uc = urbanSourcesGeoJSON.features.find(f => f.properties.id === id);
            return uc ? uc.properties.name.replace(" (Norte)", "").replace(" (Sur)", "").replace(" (Centro)", "").replace(" (Tránsito)", "").replace(" (Tráfico)", "").replace(" (Industrial)", "") : `Foco Urbano #${id}`;
        }
        
        function getIndustrialName(id) {
            const ic = industrialSourcesGeoJSON.features.find(f => f.properties.id === id);
            return ic ? ic.properties.name.replace("Ladera de ", "").replace(" (Norte)", "").replace(" (Sur)", "") : `Foco Industrial #${id}`;
        }

        // --- FASE 3: MÉTODOS DE TABS Y ANÁLISIS ---
        function setAnalysisMode(mode) {
            activeAnalysisMode = mode;
            
            document.getElementById('tab-de-donde-viene').classList.toggle('active', mode === 'de-donde-viene');
            document.getElementById('tab-hacia-donde-va').classList.toggle('active', mode === 'hacia-donde-va');
            
            updateActiveAnalysis();
        }

        function clearAnalysis() {
            activeStationId = null;
            activeSourceId = null;
            activeSourceType = null;
            activeContributingSources = [];
            backTrajectoryPoints = [];
            
            if (map.getSource('back-trajectory-source')) {
                map.getSource('back-trajectory-source').setData({ type: 'FeatureCollection', features: [] });
            }
            
            if (forecastChartInstance !== null) {
                forecastChartInstance.destroy();
                forecastChartInstance = null;
            }
            
            // Limpiar también la sonda de viento al cambiar el modo de análisis o resetear
            clearProbe();
            
            updateTrajectoryHighlight();
            updateParticles(tGlobal);
            updateAnalysisDOM();
        }

        function updateAnalysisDOM() {
            // Ocultar todos por defecto
            document.getElementById('msg-de-donde-viene-help').style.display = 'none';
            document.getElementById('card-de-donde-viene-results').style.display = 'none';
            document.getElementById('msg-hacia-donde-va-help').style.display = 'none';
            document.getElementById('card-hacia-donde-va-results').style.display = 'none';
            
            if (activeStationId !== null) {
                // Estación seleccionada: Mostrar siempre la ficha del receptor
                document.getElementById('card-de-donde-viene-results').style.display = 'flex';
                document.getElementById('analysis-panel').style.borderColor = 'rgba(129, 140, 248, 0.6)';
                document.getElementById('analysis-panel').style.background = 'rgba(129, 140, 248, 0.05)';
            } else if (activeSourceId !== null) {
                // Foco emisor seleccionado: Mostrar siempre la ficha de mitigación/impacto
                document.getElementById('card-hacia-donde-va-results').style.display = 'flex';
                document.getElementById('analysis-panel').style.borderColor = 'rgba(236, 72, 153, 0.6)';
                document.getElementById('analysis-panel').style.background = 'rgba(236, 72, 153, 0.05)';
            } else {
                // Ninguna selección activa: Mostrar mensaje de ayuda según el tab activo
                if (activeAnalysisMode === 'de-donde-viene') {
                    document.getElementById('msg-de-donde-viene-help').style.display = 'block';
                    document.getElementById('analysis-panel').style.borderColor = 'rgba(129, 140, 248, 0.3)';
                    document.getElementById('analysis-panel').style.background = 'rgba(129, 140, 248, 0.02)';
                } else {
                    document.getElementById('msg-hacia-donde-va-help').style.display = 'block';
                    document.getElementById('analysis-panel').style.borderColor = 'rgba(236, 72, 153, 0.3)';
                    document.getElementById('analysis-panel').style.background = 'rgba(236, 72, 153, 0.02)';
                }
            }
        }

        // --- FASE 4: MÉTODOS DE MITIGACIÓN E ICA ("WHAT-IF") ---
        let forecastChartInstance = null;

        function getHTMLColorForPM25(val) {
            let ratio = Math.min(Math.max((val - 7.0) / 13.0, 0.0), 1.0);
            let r = Math.round(239 * ratio + 16 * (1 - ratio));
            let g = Math.round(68 * ratio + 185 * (1 - ratio));
            let b = Math.round(68 * ratio + 129 * (1 - ratio));
            return `rgb(${r},${g},${b})`;
        }

        function getDynamicStationPM25(stationId) {
            const station = stationsGeoJSON.features.find(f => f.properties.id === stationId);
            if (!station) return 0;
            
            const st_lon = station.properties.longitud;
            const st_lat = station.properties.latitud;
            const base_pm25 = station.properties.base_pm25;
            
            const contributions = [];
            trajectoriesData.forEach(traj => {
                let min_d = Infinity;
                traj.points.forEach(pt => {
                    let d = Math.sqrt((pt[0] - st_lon)**2 + (pt[1] - st_lat)**2);
                    if (d < min_d) min_d = d;
                });
                let weight = 1.0 / (min_d * min_d + 0.0004);
                let source_intensity = traj.type === 'industrial' ? traj.emision * 1200.0 : traj.pm25 * 0.06;
                contributions.push({
                    id: traj.station_id,
                    type: traj.type,
                    weight: weight * source_intensity
                });
            });
            
            let total_w = contributions.reduce((sum, c) => sum + c.weight, 0);
            if (total_w === 0) return base_pm25;
            
            let scale_factor = 0;
            contributions.forEach(c => {
                let proportion = c.weight / total_w;
                const m_k = mitigationFactors[c.id] !== undefined ? mitigationFactors[c.id] : 1.0;
                scale_factor += proportion * m_k;
            });
            
            return base_pm25 * scale_factor;
        }

        function updateDynamicStations() {
            stationsGeoJSON.features.forEach(f => {
                const p = f.properties;
                const dyn_pm25 = getDynamicStationPM25(p.id);
                p.pm25 = dyn_pm25;
                p.color_u = getHTMLColorForPM25(dyn_pm25);
                p.height_u = dyn_pm25 * 35.0; // 35.0 escala receptores
                
                // Siempre actualizar color y height con concentración para receptores
                p.color = p.color_u;
                p.height = p.height_u;
            });
            
            if (map.getSource('stations-source')) {
                map.getSource('stations-source').setData(stationsGeoJSON);
            }
        }

        function updateMitigation(val) {
            const factor = parseFloat(val) / 100.0;
            if (activeSourceId !== null) {
                mitigationFactors[activeSourceId] = factor;
                
                document.getElementById('mitigation-label').innerText = val === '0' ? 'Cerrado (0%)' : `${val}%`;
                
                // Actualizar estaciones dinámicamente en el relieve
                updateDynamicStations();
                
                // Actualizar tabla del foco activo
                selectSource(activeSourceId, activeSourceType, document.getElementById('source-impact-name').innerText, 0, 0);
                
                // Si la estación de contribución está visible, actualizarla
                if (activeStationId !== null) {
                    const stFeature = stationsGeoJSON.features.find(f => f.properties.id === activeStationId);
                    if (stFeature) {
                        calculateContribution(stFeature.properties.longitud, stFeature.properties.latitud, stFeature.properties.name);
                    }
                }
            }
        }

        function getForecastForHour(base_pm, hour) {
            let t_rad = ((hour - 7) * 2 * Math.PI) / 24;
            let factor = 1.0 + 0.28 * Math.cos(t_rad) + 0.1 * Math.sin(t_rad * 2);
            return base_pm * factor;
        }

        function updateForecastChart(station_id, dyn_pm) {
            const labels = [];
            const data = [];
            
            for (let i = 0; i < 24; i++) {
                let h = (6 + i) % 24;
                let ampm = h >= 12 ? 'PM' : 'AM';
                let display_h = h % 12;
                if (display_h === 0) display_h = 12;
                labels.push(`${display_h} ${ampm}`);
                
                let pmVal = getForecastForHour(dyn_pm, h);
                data.push(pmVal);
            }
            
            const ctx = document.getElementById('forecastChart').getContext('2d');
            if (forecastChartInstance !== null) {
                forecastChartInstance.destroy();
            }
            
            forecastChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'PM2.5 Proyectado (µg/m³)',
                        data: data,
                        borderColor: '#818cf8',
                        backgroundColor: 'rgba(129, 140, 248, 0.1)',
                        borderWidth: 2,
                        pointRadius: 2,
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: { color: '#64748b', font: { size: 8 }, maxTicksLimit: 6 }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)' },
                            ticks: { color: '#64748b', font: { size: 8 } }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }

        function selectSource(src_id, src_type, src_name, src_lon, src_lat) {
            activeStationId = null;
            activeSourceId = src_id;
            activeSourceType = src_type;
            
            // Limpiar la sonda de viento interactiva cuando se selecciona un foco
            clearProbe();
            
            updateActiveAnalysis();

            // Sincronizar el slider con la mitigación actual del foco
            const currentMit = mitigationFactors[src_id] !== undefined ? mitigationFactors[src_id] : 1.0;
            document.getElementById('slider-mitigation').value = Math.round(currentMit * 100);
            document.getElementById('mitigation-label').innerText = currentMit === 0 ? 'Cerrado (0%)' : `${Math.round(currentMit * 100)}%`;

            const traj = trajectoriesData.find(t => t.station_id === src_id && t.type === src_type);
            if (traj) {
                const points = traj.points;
                const receptors = [];
                
                stationsGeoJSON.features.forEach(f => {
                    const p = f.properties;
                    let min_d = Infinity;
                    points.forEach(pt => {
                        let d = Math.sqrt((pt[0] - p.longitud)**2 + (pt[1] - p.latitud)**2);
                        if (d < min_d) min_d = d;
                    });
                    
                    let w = 1.0 / (min_d * min_d + 0.0004);
                    receptors.push({
                        id: p.id,
                        name: p.name,
                        weight: w
                    });
                });
                
                let total_w = receptors.reduce((sum, r) => sum + r.weight, 0);
                receptors.forEach(r => {
                    r.percentage = total_w > 0 ? (r.weight / total_w) * 100.0 : 0.0;
                });
                
                receptors.sort((a, b) => b.percentage - a.percentage);
                
                document.getElementById('source-impact-name').innerText = src_name;
                
                let html = '<table style="width:100%; font-size:11px; margin-top:8px; border-collapse:collapse; color:#e2e8f0;">';
                html += '<tr style="border-bottom:1px solid rgba(255,255,255,0.12); color:#64748b; font-weight:600;"><th style="text-align:left; padding:4px 0;">Estación Receptora</th><th style="text-align:center; padding:4px 0;">ICA actual</th><th style="text-align:right; padding:4px 0;">Impacto Pluma</th></tr>';
                
                receptors.slice(0, 5).forEach(r => {
                    const stFeature = stationsGeoJSON.features.find(f => f.properties.id === r.id);
                    const pmVal = stFeature ? stFeature.properties.pm25 : 0;
                    const ica = getICA(pmVal);
                    html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                        <td style="padding:6px 0; text-align:left;">
                            <span style="color:#e2e8f0; font-weight:600;">${r.name}</span>
                        </td>
                        <td style="padding:6px 0; text-align:center;">
                            <span style="display:inline-block; padding:1px 4px; border-radius:3px; font-size:8.5px; font-weight:700; color:${ica.color}; background:${ica.bg}; border:1px solid ${ica.border};">${ica.label}</span>
                        </td>
                        <td style="padding:6px 0; text-align:right; font-weight:700; color:#ec4899;">
                            ${r.percentage.toFixed(1)}%
                        </td>
                    </tr>`;
                });
                html += '</table>';
                
                document.getElementById('source-impact-list').innerHTML = html;
            }

            updateAnalysisDOM();
            updateParticles(tGlobal);
        }

        function updateTrajectoryHighlight() {
            if (activeSourceId === null && activeStationId === null) {
                map.setPaintProperty('urban-trajectories-layer', 'line-width', ['get', 'width']);
                map.setPaintProperty('urban-trajectories-layer', 'line-opacity', ['get', 'opacity']);
                map.setPaintProperty('industrial-trajectories-layer', 'line-width', ['get', 'width']);
                map.setPaintProperty('industrial-trajectories-layer', 'line-opacity', ['get', 'opacity']);
                
                map.setPaintProperty('urban-sources-layer', 'fill-extrusion-opacity', 0.85);
                map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-opacity', 0.85);
            } else if (activeSourceId !== null) {
                const isBackward = (activeAnalysisMode === 'de-donde-viene');
                if (isBackward) {
                    // Modo de-donde-viene (foco emisor): atenuar estelas precalculadas de todos los focos
                    map.setPaintProperty('urban-trajectories-layer', 'line-width', 1.0);
                    map.setPaintProperty('urban-trajectories-layer', 'line-opacity', 0.08);
                    map.setPaintProperty('industrial-trajectories-layer', 'line-width', 1.0);
                    map.setPaintProperty('industrial-trajectories-layer', 'line-opacity', 0.08);
                    
                    if (activeSourceType === 'urban') {
                        map.setPaintProperty('urban-sources-layer', 'fill-extrusion-opacity', 
                            ['case', ['==', ['get', 'id'], activeSourceId], 0.95, 0.15]
                        );
                        map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-opacity', 0.15);
                    } else {
                        map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-opacity', 
                            ['case', ['==', ['get', 'id'], activeSourceId], 0.95, 0.15]
                        );
                        map.setPaintProperty('urban-sources-layer', 'fill-extrusion-opacity', 0.15);
                    }
                } else {
                    // Modo hacia-donde-va (foco emisor seleccionado): resaltar su estela precalculada
                    if (activeSourceType === 'urban') {
                        map.setPaintProperty('urban-trajectories-layer', 'line-width', 
                            ['case', ['==', ['get', 'station_id'], activeSourceId], 8.0, ['get', 'width']]
                        );
                        map.setPaintProperty('urban-trajectories-layer', 'line-opacity', 
                            ['case', ['==', ['get', 'station_id'], activeSourceId], 0.95, 0.08]
                        );
                        map.setPaintProperty('industrial-trajectories-layer', 'line-width', ['get', 'width']);
                        map.setPaintProperty('industrial-trajectories-layer', 'line-opacity', 0.08);
                        
                        map.setPaintProperty('urban-sources-layer', 'fill-extrusion-opacity', 
                            ['case', ['==', ['get', 'id'], activeSourceId], 0.95, 0.15]
                        );
                        map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-opacity', 0.15);
                    } else {
                        map.setPaintProperty('industrial-trajectories-layer', 'line-width', 
                            ['case', ['==', ['get', 'station_id'], activeSourceId], 8.0, ['get', 'width']]
                        );
                        map.setPaintProperty('industrial-trajectories-layer', 'line-opacity', 
                            ['case', ['==', ['get', 'station_id'], activeSourceId], 0.95, 0.08]
                        );
                        map.setPaintProperty('urban-trajectories-layer', 'line-width', ['get', 'width']);
                        map.setPaintProperty('urban-trajectories-layer', 'line-opacity', 0.08);
                        
                        map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-opacity', 
                            ['case', ['==', ['get', 'id'], activeSourceId], 0.95, 0.15]
                        );
                        map.setPaintProperty('urban-sources-layer', 'fill-extrusion-opacity', 0.15);
                    }
                }
            } else if (activeStationId !== null) {
                if (activeAnalysisMode === 'de-donde-viene') {
                    // Modo de-donde-viene: Resaltar estelas de los focos que aportan
                    const contribFilter = ['in', ['get', 'station_id'], ['literal', activeContributingSources]];
                    
                    map.setPaintProperty('urban-trajectories-layer', 'line-width', ['case', contribFilter, 5.0, 1.0]);
                    map.setPaintProperty('urban-trajectories-layer', 'line-opacity', ['case', contribFilter, 0.7, 0.05]);
                    map.setPaintProperty('industrial-trajectories-layer', 'line-width', ['case', contribFilter, 5.0, 1.0]);
                    map.setPaintProperty('industrial-trajectories-layer', 'line-opacity', ['case', contribFilter, 0.7, 0.05]);
                    
                    map.setPaintProperty('urban-sources-layer', 'fill-extrusion-opacity', 
                        ['case', ['in', ['get', 'id'], ['literal', activeContributingSources]], 0.85, 0.15]
                    );
                    map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-opacity', 
                        ['case', ['in', ['get', 'id'], ['literal', activeContributingSources]], 0.85, 0.15]
                    );
                } else {
                    // En modo hacia-donde-va para estación: atenuar todos los focos y sus estelas
                    map.setPaintProperty('urban-trajectories-layer', 'line-width', 1.0);
                    map.setPaintProperty('urban-trajectories-layer', 'line-opacity', 0.05);
                    map.setPaintProperty('industrial-trajectories-layer', 'line-width', 1.0);
                    map.setPaintProperty('industrial-trajectories-layer', 'line-opacity', 0.05);
                    map.setPaintProperty('urban-sources-layer', 'fill-extrusion-opacity', 0.15);
                    map.setPaintProperty('industrial-sources-layer', 'fill-extrusion-opacity', 0.15);
                }
            }
        }

        // --- SISTEMA DE ANIMACIÓN Y RELOJ SOLAR (MEJORA 3) ---
        let isPlaying = true;
        let tGlobal = 0.0;
        let animationSpeed = 0.05;

        function initAnimation() {
            const btnPlay = document.getElementById('btn-play');
            const sliderTime = document.getElementById('slider-time');
            const selectSpeed = document.getElementById('select-speed');

            btnPlay.addEventListener('click', () => {
                isPlaying = !isPlaying;
                btnPlay.innerText = isPlaying ? '⏸ Pausar' : '▶ Reproducir';
            });

            sliderTime.addEventListener('input', (e) => {
                isPlaying = false;
                btnPlay.innerText = '▶ Reproducir';
                tGlobal = parseFloat(e.target.value);
                document.getElementById('label-time').innerText = `Paso: ${tGlobal.toFixed(1)}`;
                updateParticles(tGlobal);
                updateLight(tGlobal);
            });

            selectSpeed.addEventListener('change', (e) => {
                animationSpeed = parseFloat(e.target.value);
            });

            // Iniciar bucle a 60 FPS
            requestAnimationFrame(animationLoop);
        }

        function animationLoop() {
            if (isPlaying) {
                tGlobal += animationSpeed;
                if (tGlobal > 15.0) {
                    tGlobal = 0.0;
                }
                document.getElementById('slider-time').value = tGlobal;
                document.getElementById('label-time').innerText = `Paso: ${tGlobal.toFixed(1)}`;
                updateParticles(tGlobal);
                updateLight(tGlobal);
            }
            requestAnimationFrame(animationLoop);
        }

        function formatTime(t_val) {
            let total_hours = 6.0 + (t_val / 15.0) * 24.0;
            if (total_hours >= 24.0) total_hours -= 24.0;
            
            let hrs = Math.floor(total_hours);
            let mins = Math.floor((total_hours - hrs) * 60);
            
            let ampm = hrs >= 12 ? 'PM' : 'AM';
            let display_hrs = hrs % 12;
            if (display_hrs === 0) display_hrs = 12;
            
            let display_mins = mins < 10 ? '0' + mins : mins;
            let display_hrs_str = display_hrs < 10 ? '0' + display_hrs : display_hrs;
            
            return `${display_hrs_str}:${display_mins} ${ampm}`;
        }

        function updateLight(t_val) {
            let total_hours = 6.0 + (t_val / 15.0) * 24.0;
            if (total_hours >= 24.0) total_hours -= 24.0;
            
            let pct = total_hours / 24.0;
            // El sol gira 360 grados azimutales en 24 horas
            let azimuth = pct * 360.0;
            
            let polar = 85.0; // Casi rasante en la noche
            let intensity = 0.15;
            let color = '#0f172a'; // noche azul muy profunda
            
            if (total_hours >= 6.0 && total_hours <= 18.0) {
                let t_day = (total_hours - 6.0) / 12.0; // 0.0 a 1.0
                let elevation = Math.sin(Math.PI * t_day) * 75.0; // Max 75 grados de altura
                polar = 90.0 - elevation;
                intensity = 0.15 + 0.9 * Math.sin(Math.PI * t_day);
                
                // Cambiar color solar
                if (total_hours < 7.5) {
                    color = '#fdba74'; // Amanecer (Naranja suave)
                } else if (total_hours > 16.5) {
                    color = '#f97316'; // Atardecer (Naranja intenso solar)
                } else {
                    color = '#ffffff'; // Día (Blanco brillante)
                }
            }
            
            if (map.style) {
                map.setLight({
                    anchor: 'viewport',
                    color: color,
                    intensity: intensity,
                    position: [1.5, azimuth, polar]
                });
            }
            
            document.getElementById('label-clock').innerText = formatTime(t_val);
        }

        function interpolatePosition(points, age) {
            const k = Math.floor(age);
            const frac = age - k;
            if (k >= points.length - 1) {
                return points[points.length - 1];
            }
            const p1 = points[k];
            const p2 = points[k + 1];
            return [
                p1[0] + (p2[0] - p1[0]) * frac,
                p1[1] + (p2[1] - p1[1]) * frac,
                p1[2] + (p2[2] - p1[2]) * frac
            ];
        }

        function updateParticles(t_val) {
            const features = [];
            const releaseInterval = 1.6;
            const numStreams = 6;
            
            // Leer estado de visibilidad desde las capas
            const isUrbanVisible = document.getElementById('toggle-urban-sources').checked;
            const isIndustrialVisible = document.getElementById('toggle-industrial-sources').checked;

            trajectoriesData.forEach(traj => {
                // Filtrar según el tipo de capa activa
                if (traj.type === 'urban' && !isUrbanVisible) return;
                if (traj.type === 'industrial' && !isIndustrialVisible) return;

                const points = traj.points;
                const pm25 = traj.pm25;
                const emision = traj.emision;
                const stationId = traj.station_id;

                let k = 0;
                while (true) {
                    const t_rel = k * releaseInterval;
                    if (t_rel > t_val) break;

                    const age = t_val - t_rel;
                    if (age <= 15.0) {
                        const pos = interpolatePosition(points, age);
                        const ageRatio = age / 15.0;

                        // Determinar ratio de escala según la métrica activa
                        let valRatio = 0.0;
                        let r_start = 16, g_start = 185, b_start = 129; // default green
                        let r_end = 250, g_end = 204, b_end = 21;       // default yellow-fade

                        if (activeMetric === 'u') {
                            valRatio = Math.min(Math.max((pm25 - 7.0) / 13.0, 0.0), 1.0);
                            // Escala Green-to-Red
                            r_start = Math.round(239 * valRatio + 16 * (1 - valRatio));
                            g_start = Math.round(68 * valRatio + 185 * (1 - valRatio));
                            b_start = Math.round(68 * valRatio + 129 * (1 - valRatio));
                            
                            // Se difumina a amarillo-naranja
                            r_end = 250; g_end = 204; b_end = 21;
                        } else {
                            // Escala Celeste-to-Magenta
                            valRatio = Math.min(Math.max(emision / 0.0006, 0.0), 1.0);
                            r_start = Math.round(236 * valRatio + 56 * (1 - valRatio));
                            g_start = Math.round(72 * valRatio + 189 * (1 - valRatio));
                            b_start = Math.round(153 * valRatio + 248 * (1 - valRatio));
                            
                            // Se difumina a violeta oscuro
                            r_end = 139; g_end = 92; b_end = 246;
                        }

                        for (let j = 0; j < numStreams; j++) {
                            const phase = (j * 2 * Math.PI) / numStreams;
                            const wiggleAmplitude = 0.0018 * ageRatio;
                            
                            const wiggleX = Math.sin(age * (1.1 + j * 0.15) + stationId + phase) * wiggleAmplitude;
                            const wiggleY = Math.cos(age * (0.8 + j * 0.1) + stationId * 2 + phase * 1.5) * wiggleAmplitude;

                            const finalPos = [pos[0] + wiggleX, pos[1] + wiggleY];

                            // Mezclar con la atenuación del extremo
                            const r = Math.round(r_start + (r_end - r_start) * ageRatio);
                            const g = Math.round(g_start + (g_end - g_start) * ageRatio);
                            const b = Math.round(b_start + (b_end - b_start) * ageRatio);
                            const color = `rgb(${r},${g},${b})`;

                            // Tamaño
                            const baseRadius = (3.5 + 11.5 * valRatio) * (0.8 + 0.4 * Math.sin(j));
                            const windSpeed = Math.sqrt(traj.vx * traj.vx + traj.vy * traj.vy);
                            const dispersionRate = 0.4 + 1.6 * (windSpeed / 2.0);
                            let radius = baseRadius * (1.0 + dispersionRate * ageRatio);

                            // Opacidad
                            const baseOpacity = (0.12 + 0.65 * valRatio) / (numStreams * 0.35);
                            let opacity = Math.min(baseOpacity * (1.0 - ageRatio), 0.85);

                            // --- FASE 3: Aislamiento de partículas ---
                            if (activeSourceId !== null) {
                                if (activeAnalysisMode === 'hacia-donde-va' && stationId === activeSourceId && traj.type === activeSourceType) {
                                    // Foco seleccionado en modo hacia-donde-va: más grande y brillante
                                    radius = radius * 1.25;
                                    opacity = Math.min(opacity * 1.3, 0.95);
                                } else {
                                    // Focos no seleccionados o en modo de-donde-viene: atenuar a 10%
                                    opacity = opacity * 0.10;
                                }
                            } else if (activeStationId !== null) {
                                // Modo de-donde-viene (estación seleccionada): resaltar aportes si el tab es "de-donde-viene"
                                if (activeAnalysisMode === 'de-donde-viene' && activeContributingSources.includes(stationId)) {
                                    radius = radius * 1.15;
                                    opacity = Math.min(opacity * 1.2, 0.90);
                                } else {
                                    opacity = opacity * 0.10;
                                }
                            }

                            // --- FASE 4: Aplicar factor de mitigación interactivo ---
                            const mitFactor = mitigationFactors[stationId] !== undefined ? mitigationFactors[stationId] : 1.0;
                            radius = radius * mitFactor;
                            opacity = opacity * mitFactor;

                            features.push({
                                type: 'Feature',
                                properties: {
                                    station_id: stationId,
                                    color: color,
                                    radius: radius,
                                    opacity: opacity
                                },
                                geometry: {
                                    type: 'Point',
                                    coordinates: finalPos
                                }
                            });
                        }
                    }
                    k++;
                }
            });

            // Animar partículas fluyendo sobre la sonda de viento activa (color y tamaño dinámicos según PM2.5)
            if (typeof probePoints !== 'undefined' && probePoints && probePoints.length > 0) {
                const numProbeParticles = 7;
                const L = probePoints.length;
                const isBackward = (activeAnalysisMode === 'de-donde-viene');
                
                for (let i = 0; i < numProbeParticles; i++) {
                    const rawAge = (t_val * 2.5 + i * (L / numProbeParticles)) % (L - 1);
                    // Si es retrógrada fluye hacia el origen (de L-1 a 0), de lo contrario fluye hacia adelante (de 0 a L-1)
                    const age = isBackward ? (L - 1 - rawAge) : rawAge;
                    const pos = interpolatePosition(probePoints, age);
                    
                    let localPm = interpolatePM25JS(pos[0], pos[1]);
                    let dynamicColor = getRelativeColorForPM25(localPm);
                    let radius = 5.0 + Math.min(localPm * 0.35, 14.0); // Más grande en zonas contaminadas
                    
                    features.push({
                        type: 'Feature',
                        properties: {
                            color: dynamicColor,
                            radius: radius,
                            opacity: 0.95
                        },
                        geometry: {
                            type: 'Point',
                            coordinates: pos
                        }
                    });
                }
            }

            // Animar partículas fluyendo sobre la trayectoria retrógrada activa (color y tamaño dinámicos según PM2.5)
            if (typeof backTrajectoryPoints !== 'undefined' && backTrajectoryPoints && backTrajectoryPoints.length > 0) {
                const numBackParticles = 7;
                const L = backTrajectoryPoints.length;
                const isBackward = (activeAnalysisMode === 'de-donde-viene');
                for (let i = 0; i < numBackParticles; i++) {
                    const rawAge = (t_val * 2.5 + i * (L / numBackParticles)) % (L - 1);
                    const age = isBackward ? (L - 1 - rawAge) : rawAge; // Fluye de regreso en retro (de L-1 a 0), adelante en dispersion (de 0 a L-1)
                    const pos = interpolatePosition(backTrajectoryPoints, age);
                    
                    let localPm = interpolatePM25JS(pos[0], pos[1]);
                    let dynamicColor = getRelativeColorForPM25(localPm);
                    let radius = 5.0 + Math.min(localPm * 0.35, 14.0);
                    
                    features.push({
                        type: 'Feature',
                        properties: {
                            color: dynamicColor,
                            radius: radius,
                            opacity: 0.95
                        },
                        geometry: {
                            type: 'Point',
                            coordinates: pos
                        }
                    });
                }
            }

            if (map.getSource('particles-source')) {
                map.getSource('particles-source').setData({
                    type: 'FeatureCollection',
                    features: features
                });
            }
        }

        // --- CONFIGURACIÓN DE HOVER TOOLTIPS ---
        function setupTooltips() {
            const el = document.getElementById('tooltip');
            
            const handleMouse = (e, title, isSource) => {
                if (!e.features || e.features.length === 0) return;
                map.getCanvas().style.cursor = 'pointer';
                const p = e.features[0].properties;
                let extraData = '';
                if (isSource) {
                    const ica = getICA(p.pm25);
                    extraData = `
                        <b style="color:#facc15;">Tipo de Fuente:</b> ${p.type === 'urban' ? 'Urbano (Tránsito)' : 'Industrial (Ladera)'}<br>
                        <span style="color:#ec4899; font-size:11.5px;"><b>Tasa Emisión (S): ${p.emision.toFixed(6)} ug/(m³*s)</b></span><br>
                        <span style="color:#38bdf8; font-size:11.5px;"><b>PM2.5 Estimado (u): ${p.pm25.toFixed(1)} ug/m³</b></span>
                        <span class="ica-badge" style="display:inline-block; padding:1px 5px; border-radius:3px; font-size:9.5px; font-weight:700; color:${ica.color}; background:${ica.bg}; border:1px solid ${ica.border}; margin-left:4px;">${ica.label}</span><br>
                    `;
                } else {
                    const ica = getICA(p.pm25);
                    extraData = `
                        <span style="color:#10b981; font-size:11.5px;"><b>PM2.5 Medido (u): ${p.pm25.toFixed(1)} ug/m³</b></span>
                        <span class="ica-badge" style="display:inline-block; padding:1px 5px; border-radius:3px; font-size:9.5px; font-weight:700; color:${ica.color}; background:${ica.bg}; border:1px solid ${ica.border}; margin-left:4px;">${ica.label}</span><br>
                        <span style="color:#c084fc; font-size:11px;"><b>Emisión Local S: ${p.s.toFixed(6)} ug/(m³*s)</b></span><br>
                        <span style="color:#818cf8; font-size:10px;"><b>*Haz clic sobre la estación para ver la ficha de contribuciones de focos en la barra lateral.</b></span><br>
                    `;
                }

                el.innerHTML = `
                    <div style="font-family: 'Inter', sans-serif; font-size: 11px; color: #fff; line-height:1.4;">
                        <b style="font-size:12px; color:#38bdf8;">${title}: ${p.name}</b><br>
                        <b>Coordenadas:</b> ${p.latitud.toFixed(4)}°, ${p.longitud.toFixed(4)}°<br>
                        <b>Altitud Terreno:</b> ${p.elev.toFixed(0)} msnm<br>
                        <hr style="border:0; border-top:1px solid rgba(255,255,255,0.15); margin:6px 0;">
                        ${extraData}
                        <b>Viento calculado:</b> [${p.vx.toFixed(2)}, ${p.vy.toFixed(2)}] m/s
                    </div>
                `;
                el.style.display = 'block';
                el.style.left = e.point.x + 15 + 'px';
                el.style.top = e.point.y + 15 + 'px';
            };

            const hideMouse = () => {
                map.getCanvas().style.cursor = '';
                el.style.display = 'none';
            };

            // Eventos Receptores (Hover y clic)
            map.on('mousemove', 'stations-layer', (e) => handleMouse(e, 'Receptor SIATA', false));
            map.on('mouseleave', 'stations-layer', hideMouse);
            
            map.on('click', 'stations-layer', (e) => {
                if (e.features.length > 0) {
                    activeSourceId = null;
                    activeSourceType = null;
                    
                    // Limpiar la sonda de viento interactiva cuando se selecciona un receptor
                    clearProbe();
                    
                    const p = e.features[0].properties;
                    activeStationId = p.id;
                    calculateContribution(p.longitud, p.latitud, p.name);
                    
                    updateActiveAnalysis();
                }
            });

            // Eventos Fuentes Urbanas
            map.on('mousemove', 'urban-sources-layer', (e) => handleMouse(e, 'Foco Urbano', true));
            map.on('mouseleave', 'urban-sources-layer', hideMouse);
            map.on('click', 'urban-sources-layer', (e) => {
                if (e.features.length > 0) {
                    const p = e.features[0].properties;
                    selectSource(p.id, 'urban', p.name, p.longitud, p.latitud);
                }
            });

            // Eventos Fuentes Industriales
            map.on('mousemove', 'industrial-sources-layer', (e) => handleMouse(e, 'Foco Industrial', true));
            map.on('mouseleave', 'industrial-sources-layer', hideMouse);
            map.on('click', 'industrial-sources-layer', (e) => {
                if (e.features.length > 0) {
                    const p = e.features[0].properties;
                    selectSource(p.id, 'industrial', p.name, p.longitud, p.latitud);
                }
            });
        }

        // --- GRÁFICO DE CONVERGENCIA CHART.JS ---
        const epochsList = {epochs_list_placeholder};
        const totalLossList = {total_loss_placeholder};
        const pdeLossList = {pde_loss_placeholder};
        const dataLossList = {data_loss_placeholder};
        const valLossList = {val_loss_placeholder};

        const ctx = document.getElementById('lossChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochsList,
                datasets: [
                    {
                        label: 'Física (PDE)',
                        data: pdeLossList,
                        borderColor: '#c084fc',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Datos (Asim)',
                        data: dataLossList,
                        borderColor: '#38bdf8',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    },
                    {
                        label: 'Validación',
                        data: valLossList,
                        borderColor: '#818cf8',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: '#64748b', font: { size: 9 }, maxTicksLimit: 5 }
                    },
                    y: {
                        type: 'logarithmic',
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { 
                            color: '#64748b', 
                            font: { size: 8 },
                            callback: function(value) {
                                return value.toExponential(0);
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#e2e8f0', boxWidth: 8, font: { size: 9 } }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

    # Realizar los reemplazos en el template
    final_html = (dashboard_template
                  .replace("{stations_geojson_placeholder}", stations_geojson)
                  .replace("{urban_sources_geojson_placeholder}", urban_sources_geojson)
                  .replace("{industrial_sources_geojson_placeholder}", industrial_sources_geojson)
                  .replace("{urban_trajectories_geojson_placeholder}", urban_trajectories_geojson)
                  .replace("{industrial_trajectories_geojson_placeholder}", industrial_trajectories_geojson)
                  .replace("{wind_geojson_placeholder}", wind_geojson)
                  .replace("{inversion_geojson_placeholder}", inversion_geojson)
                  .replace("{trajectories_data_placeholder}", trajectories_data_json)
                  .replace("{epochs_list_placeholder}", json.dumps(epochs_list))
                  .replace("{total_loss_placeholder}", json.dumps(total_loss_list))
                  .replace("{pde_loss_placeholder}", json.dumps(pde_loss_list))
                  .replace("{data_loss_placeholder}", json.dumps(data_loss_list))
                  .replace("{val_loss_placeholder}", json.dumps(val_loss_list))
                  .replace("{latest_epoch}", str(latest_metrics["epoch"]))
                  .replace("{latest_total_loss}", f"{latest_metrics['total_loss']:.5f}")
                  .replace("{latest_pde_loss}", f"{latest_metrics['pde_loss']:.5f}")
                  .replace("{latest_data_loss}", f"{latest_metrics['data_loss']:.5f}")
                  .replace("{latest_val_loss}", f"{latest_metrics['val_loss']:.5f}")
                  .replace("{latest_pvi}", f"{pvi_val:.6f}"))

    output_html = "reporte/mapa_3d_interactivo.html"
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"[SUCCESS] Mapa 3D de diagnóstico físico generado en: {output_html}")

if __name__ == "__main__":
    generate_3d_map()
