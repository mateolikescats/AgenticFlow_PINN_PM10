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
                "color": color_S, # Default: Emisión S
                "height_u": 25.0,
                "height_S": 25.0,
                "height": 25.0
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

    # Plantilla HTML Premium de Visualización Científica
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

            <div>
                <h2>Simulación de Flujo Temporal</h2>
                <div class="animation-panel">
                    <div class="anim-row">
                        <button id="btn-play" class="btn-primary">⏸ Pausar</button>
                        <span id="label-time" style="font-size: 12px; font-weight: 600; color: #38bdf8;">Paso: 0.0</span>
                    </div>
                    <input type="range" id="slider-time" min="0" max="15" step="0.1" value="0">
                    <div class="anim-row" style="font-size: 11px;">
                        <span>Velocidad de simulación:</span>
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
                        <div class="legend-icon icon-wind"></div>
                        <div>
                            <b>Vectores de Viento:</b> Campo de velocidades del viento simulado bajo restricciones de incompresibilidad.
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
        const trajectoriesData = {trajectories_data_placeholder};
        
        let activeMetric = 'S'; // 'S' o 'u'

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

            // Inicializar simulación temporal y hover tooltips
            initAnimation();
            setupTooltips();
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

            // Actualizar propiedades de extrusión de Receptores
            map.setPaintProperty('stations-layer', 'fill-extrusion-color', ['get', 'color_' + metric]);
            map.setPaintProperty('stations-layer', 'fill-extrusion-height', ['get', 'height_' + metric]);

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

        // --- SISTEMA DE ANIMACIÓN DE PARTÍCULAS (CÚMULOS) ---
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
            }
            requestAnimationFrame(animationLoop);
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
                            const radius = baseRadius * (1.0 + dispersionRate * ageRatio);

                            // Opacidad
                            const baseOpacity = (0.12 + 0.65 * valRatio) / (numStreams * 0.35);
                            const opacity = Math.min(baseOpacity * (1.0 - ageRatio), 0.85);

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
                map.getCanvas().style.cursor = 'pointer';
                if (e.features.length > 0) {
                    const p = e.features[0].properties;
                    
                    let extraData = '';
                    if (isSource) {
                        extraData = `
                            <b style="color:#facc15;">Tipo de Fuente:</b> ${p.type === 'urban' ? 'Urbana (Tránsito)' : 'Industrial (Ladera)'}<br>
                            <span style="color:#ec4899; font-size:11.5px;"><b>Tasa Emisión (S): ${p.emision.toFixed(6)} ug/(m³*s)</b></span><br>
                            <span style="color:#38bdf8; font-size:11.5px;"><b>PM2.5 Estimado (u): ${p.pm25.toFixed(1)} ug/m³</b></span><br>
                        `;
                    } else {
                        extraData = `
                            <span style="color:#10b981; font-size:11.5px;"><b>PM2.5 Medido (u): ${p.pm25.toFixed(1)} ug/m³</b></span><br>
                            <span style="color:#c084fc; font-size:11px;"><b>Emisión Local S: ${p.s.toFixed(6)} ug/(m³*s)</b></span><br>
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
                }
            };

            const hideMouse = () => {
                map.getCanvas().style.cursor = '';
                el.style.display = 'none';
            };

            // Eventos Receptores
            map.on('mousemove', 'stations-layer', (e) => handleMouse(e, 'Receptor SIATA', false));
            map.on('mouseleave', 'stations-layer', hideMouse);

            // Eventos Fuentes Urbanas
            map.on('mousemove', 'urban-sources-layer', (e) => handleMouse(e, 'Foco Urbano', true));
            map.on('mouseleave', 'urban-sources-layer', hideMouse);

            // Eventos Fuentes Industriales
            map.on('mousemove', 'industrial-sources-layer', (e) => handleMouse(e, 'Foco Industrial', true));
            map.on('mouseleave', 'industrial-sources-layer', hideMouse);
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
