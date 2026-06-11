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

def interpolate_wind(lon, lat, stations_df):
    lons = stations_df['longitud'].values
    lats = stations_df['latitud'].values
    vx_vals = stations_df['pred_viento_vx_m_s'].values
    vy_vals = stations_df['pred_viento_vy_m_s'].values
    vz_vals = stations_df['pred_viento_vz_m_s'].values
    
    dists = np.sqrt((lons - lon)**2 + (lats - lat)**2)
    if np.any(dists < 1e-5):
        idx = np.argmin(dists)
        return vx_vals[idx], vy_vals[idx], vz_vals[idx]
    else:
        weights = 1.0 / (dists ** 2)
        w_sum = np.sum(weights)
        vx = np.sum(vx_vals * weights) / w_sum
        vy = np.sum(vy_vals * weights) / w_sum
        vz = np.sum(vz_vals * weights) / w_sum
        return vx, vy, vz

def generate_3d_map():
    predictions_path = "output_predictions.json"
    if not os.path.exists(predictions_path):
        print("[ERROR] No se encontró output_predictions.json. Ejecuta primero predict_realtime.py.")
        return

    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # Coordenadas geográficas y físicas de límites del Valle
    lat_min, lat_max = 6.0, 6.45
    lon_min, lon_max = -75.7, -75.3

    # Calcular base del terreno para los soportes verticales
    z_ground = []
    for _, r in df.iterrows():
        g_h = get_terrain_height(r['longitud'], r['latitud'])
        # Asegurar al menos 60 metros de espacio para que no queden enterrados
        z_ground.append(min(r['elevacion'] - 60.0, g_h))
    df['z_ground'] = z_ground

    # Calcular las Trayectorias de Dispersión (Smoke Trails) para pasarlas como JSON a JS
    trajectories = []
    dt = 0.012  # Paso temporal de simulación visual
    for idx, r in df.iterrows():
        curr_lon = r['longitud']
        curr_lat = r['latitud']
        curr_elev = r['elevacion']
        
        path = [[curr_lon, curr_lat, curr_elev]]
        
        # Integrar la trayectoria de partículas usando el campo de viento de la PINN
        for step in range(15):
            vx, vy, vz = interpolate_wind(curr_lon, curr_lat, df)
            curr_lon += vx * dt
            curr_lat += vy * dt
            curr_elev += vz * 12000.0 * dt  # Escalado vertical visible
            
            # Limitar a la caja de visualización
            curr_lon = np.clip(curr_lon, lon_min, lon_max)
            curr_lat = np.clip(curr_lat, lat_min, lat_max)
            
            path.append([curr_lon, curr_lat, curr_elev])
            
        trajectories.append({
            "station_id": int(r.get("id", idx)),
            "pm25": float(r["pred_pm25_ug_m3"]),
            "path": path
        })

    # Serializar los datos para incrustarlos en el script de JS
    stations_json = df.to_json(orient='records')
    trajectories_json = json.dumps(trajectories)

    # Envolver en una plantilla HTML Premium con Maplibre GL JS y Deck.gl
    dashboard_html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iPINN 3D - Visualizador de Calidad del Aire y Vientos</title>
    <!-- Maplibre GL JS y Deck.gl -->
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
    <script src="https://unpkg.com/deck.gl@8.9.0/dist.min.js"></script>
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
            width: 420px;
            min-width: 380px;
            background: rgba(10, 15, 30, 0.88);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            padding: 25px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 22px;
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
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        .subtitle {
            font-size: 12.5px;
            color: #64748b;
            margin-top: 2px;
        }
        h2 {
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: #38bdf8;
            border-bottom: 1px solid rgba(56, 189, 248, 0.2);
            padding-bottom: 5px;
            margin-top: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        p.description {
            font-size: 12.5px;
            line-height: 1.5;
            color: #94a3b8;
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
            font-size: 12.5px;
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
        .legend-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .legend-item {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            font-size: 12px;
            color: #cbd5e1;
            line-height: 1.4;
        }
        .legend-icon {
            width: 16px;
            height: 16px;
            border-radius: 50%;
            flex-shrink: 0;
            margin-top: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .icon-station {
            background: linear-gradient(135deg, #10b981, #ef4444);
            border: 1.5px solid white;
            box-shadow: 0 0 5px rgba(239, 68, 68, 0.4);
            border-radius: 2px; /* Representa cilindros */
            width: 12px;
            height: 16px;
        }
        .icon-wind {
            background: #38bdf8;
            height: 4px;
            width: 16px;
            border-radius: 2px;
        }
        .icon-trail {
            background: linear-gradient(90deg, #ef4444, #facc15);
            border-radius: 4px;
            height: 6px;
            width: 18px;
            margin-top: 5px;
        }
        .icon-terrain {
            background: rgba(255, 255, 255, 0.1);
            border: 1.5px solid #22c55e;
            border-radius: 3px;
        }
        .icon-stalk {
            width: 4px;
            height: 16px;
            background: transparent;
            border-left: 2px dashed rgba(255, 255, 255, 0.5);
            border-radius: 0;
            margin-left: 6px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 8px 10px;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .metric-value {
            font-size: 14px;
            font-weight: 700;
            color: #f1f5f9;
            font-family: 'Outfit', sans-serif;
        }
        .metric-label {
            font-size: 8.5px;
            color: #64748b;
            margin-top: 2px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .control-tip {
            background: rgba(56, 189, 248, 0.04);
            border-left: 3px solid #38bdf8;
            padding: 10px;
            border-radius: 0 8px 8px 0;
            font-size: 11px;
            color: #94a3b8;
            line-height: 1.4;
        }
        .control-tip b {
            color: #38bdf8;
        }
        #tooltip {
            position: absolute;
            display: none;
            background: rgba(10, 15, 30, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 12px;
            border-radius: 8px;
            pointer-events: none;
            z-index: 100;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        /* Personalizar barra de desplazamiento */
        .sidebar::-webkit-scrollbar {
            width: 5px;
        }
        .sidebar::-webkit-scrollbar-track {
            background: transparent;
        }
        .sidebar::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
        }
        .sidebar::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.15);
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <div>
                <span class="badge">iPINN Satellite 3D</span>
                <h1>iPINN 3D: Valle de Aburrá</h1>
                <div class="subtitle">Simulación 3D Georreferenciada sobre Relieve Satelital</div>
            </div>

            <div>
                <h2>¿Qué es esta simulación?</h2>
                <p class="description">
                    Este mapa interactivo simula en 3D real el comportamiento de vientos y partículas contaminantes (<b>PM2.5</b>) integrando la física del modelo <b>iPINN</b> con datos reales de la red SIATA sobre la topografía satelital del Valle de Aburrá.
                </p>
            </div>

            <div>
                <h2>Capas Visuales</h2>
                <div class="toggle-container">
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-terrain" checked onchange="toggleTerrain(this.checked)">
                        <span class="toggle-custom"></span>
                        Relieve 3D Satelital (Esri)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-stations" checked onchange="toggleLayer('stations-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Estaciones 3D (PM2.5)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-trails" checked onchange="toggleLayer('trajectories-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Estelas de Dispersión
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-wind" checked onchange="toggleLayer('wind-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Vectores de Viento (3D)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-stalks" checked onchange="toggleLayer('stalks-layer', this.checked)">
                        <span class="toggle-custom"></span>
                        Soportes de Altura
                    </label>
                </div>
            </div>

            <div>
                <h2>Leyenda del Mapa 3D</h2>
                <div class="legend-list">
                    <div class="legend-item">
                        <div class="legend-icon icon-terrain"></div>
                        <div>
                            <b>Relieve 3D Satelital:</b> Geografía en 3D real del cañón (montañas y laderas) cubierta con imágenes satelitales de alta resolución de Esri.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-station"></div>
                        <div>
                            <b>Estaciones 3D (Cilindros):</b> El <b>diámetro</b> y la <b>altura</b> indican el PM2.5 predicho. El color varía de verde (limpio) a rojo (contaminado).
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-trail"></div>
                        <div>
                            <b>Estelas de Dispersión:</b> Senderos de partículas que muestran <b>hacia dónde sopla el viento</b> (rojo en origen denso, amarillo al diluirse).
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-wind"></div>
                        <div>
                            <b>Vectores de Viento:</b> Líneas celestes que representan la magnitud y dirección del viento calculado por la PINN.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-stalk"></div>
                        <div>
                            <b>Soportes de Altura:</b> Líneas guía para apreciar a qué altitud vuela el viento y se miden las partículas sobre el terreno.
                        </div>
                    </div>
                </div>
            </div>

            <div>
                <h2>Métricas y Parámetros</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">0.025999</div>
                        <div class="metric-label">PVI (Física Error)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">96 Horas</div>
                        <div class="metric-label">Ventana Temporal</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">18</div>
                        <div class="metric-label">Estaciones Activas</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">Inversa S</div>
                        <div class="metric-label">Cálculo Fuentes</div>
                    </div>
                </div>
            </div>

            <div class="control-tip">
                <b>Navegación 3D:</b><br>
                • <b>Rotar Cámara:</b> Mantén presionado <b>clic izquierdo</b> (o clic derecho) y arrastra.<br>
                • <b>Zoom:</b> Rueda del ratón.<br>
                • <b>Desplazar (Pan):</b> Mantén presionado <b>Ctrl + Clic izquierdo</b> y arrastra.<br>
                • <b>Detalles:</b> Pasa el cursor sobre un cilindro de estación para ver su ficha técnica detallada.
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
        const stationsData = {stations_json};
        const trajectoriesData = {trajectories_json};

        // Inicializar mapa de Maplibre GL JS con imágenes de Esri
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
            center: [-75.567, 6.2518], // Medellín
            zoom: 11.8,
            pitch: 55,
            bearing: -15,
            maxZoom: 16,
            minZoom: 9
        });

        // Controles de navegación nativos
        map.addControl(new maplibregl.NavigationControl());

        map.on('load', () => {
            // 1. Agregar terreno 3D (Mapzen Terrarium en AWS - Cobertura Global y Libre)
            map.addSource('terrain', {
                type: 'raster-dem',
                tiles: [
                    'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
                ],
                encoding: 'terrarium',
                tileSize: 256,
                maxzoom: 15
            });
            map.setTerrain({ source: 'terrain', exaggeration: 1.4 });

            // 2. Generar datos para los soportes verticales (Stalks)
            const stalkData = [];
            stationsData.forEach(d => {
                // Línea desde base hasta la estación
                stalkData.push({
                    from: [d.longitud, d.latitud, d.z_ground],
                    to: [d.longitud, d.latitud, d.elevacion]
                });
                // Línea desde la estación hasta el vector de viento (+80m)
                stalkData.push({
                    from: [d.longitud, d.latitud, d.elevacion],
                    to: [d.longitud, d.latitud, d.elevacion + 80]
                });
            });

            // Capa de Soportes (Deck.gl LineLayer)
            const stalkLayer = new deck.MapboxLayer({
                id: 'stalks-layer',
                type: deck.LineLayer,
                data: stalkData,
                getSourcePosition: d => d.from,
                getTargetPosition: d => d.to,
                getColor: [255, 255, 255, 90],
                getWidth: 2.2,
                pickable: false
            });
            map.addLayer(stalkLayer);

            // 3. Capa de Trayectorias de Dispersión (Deck.gl PathLayer)
            const pathLayer = new deck.MapboxLayer({
                id: 'trajectories-layer',
                type: deck.PathLayer,
                data: trajectoriesData,
                getPath: d => d.path,
                getColor: d => {
                    // Color naranja/amarillo con cierta transparencia
                    return [251, 146, 60, 180];
                },
                getWidth: 12,
                widthMinPixels: 2.8,
                shadowEnabled: false,
                pickable: false
            });
            map.addLayer(pathLayer);

            // 4. Capa de Vectores de Viento (Deck.gl LineLayer)
            const windLayer = new deck.MapboxLayer({
                id: 'wind-layer',
                type: deck.LineLayer,
                data: stationsData,
                getSourcePosition: d => [d.longitud, d.latitud, d.elevacion + 80],
                getTargetPosition: d => [
                    d.longitud + d.pred_viento_vx_m_s * 0.0035,
                    d.latitud + d.pred_viento_vy_m_s * 0.0035,
                    d.elevacion + 80 + d.pred_viento_vz_m_s * 5
                ],
                getColor: [56, 189, 248, 220],
                getWidth: 4,
                pickable: false
            });
            map.addLayer(windLayer);

            // 5. Capa de Estaciones (Deck.gl ColumnLayer - Cilindros Extruidos en 3D)
            const columnLayer = new deck.MapboxLayer({
                id: 'stations-layer',
                type: deck.ColumnLayer,
                data: stationsData,
                diskResolution: 16,
                radius: 180,
                extruded: true,
                elevationScale: 40, // Escala la altura según el PM2.5
                getPosition: d => [d.longitud, d.latitud, d.elevacion],
                getElevation: d => d.pred_pm25_ug_m3,
                getFillColor: d => {
                    const val = d.pred_pm25_ug_m3;
                    const ratio = Math.min(Math.max((val - 7) / 13, 0), 1);
                    // Degradado de Verde a Rojo
                    return [
                        Math.round(250 * ratio + 16 * (1 - ratio)),
                        Math.round(50 * ratio + 185 * (1 - ratio)),
                        Math.round(50 * ratio + 129 * (1 - ratio)),
                        210
                    ];
                },
                pickable: true,
                onHover: info => {
                    const el = document.getElementById('tooltip');
                    if (info.object) {
                        const d = info.object;
                        el.innerHTML = `
                            <div style="font-family: 'Inter', sans-serif; font-size: 11px; color: #fff; line-height:1.4;">
                                <b style="font-size:12px; color:#38bdf8;">Estación SIATA</b><br>
                                <b>Coordenadas:</b> ${d.latitud.toFixed(4)}°, ${d.longitud.toFixed(4)}°<br>
                                <b>Altitud:</b> ${d.elevacion.toFixed(0)} msnm<br>
                                <hr style="border:0; border-top:1px solid rgba(255,255,255,0.15); margin:6px 0;">
                                <span style="color:#ff5a5f; font-size:11.5px;"><b>PM2.5 Predicho: ${d.pred_pm25_ug_m3.toFixed(2)} ug/m³</b></span><br>
                                <b>Viento:</b> [${d.pred_viento_vx_m_s.toFixed(2)}, ${d.pred_viento_vy_m_s.toFixed(2)}, ${d.pred_viento_vz_m_s.toFixed(2)}] m/s<br>
                                <span style="color:#38bdf8;"><b>Emisión Inversa (S): ${d.pred_emision_S_ug_m3_s.toFixed(6)} ug/(m³*s)</b></span>
                            </div>
                        `;
                        el.style.display = 'block';
                        el.style.left = info.x + 15 + 'px';
                        el.style.top = info.y + 15 + 'px';
                    } else {
                        el.style.display = 'none';
                    }
                }
            });
            map.addLayer(columnLayer);
        });

        // Alternar visualización de las capas Deck.gl
        function toggleLayer(layerId, visible) {
            if (map.getLayer(layerId)) {
                map.setLayoutProperty(layerId, 'visibility', visible ? 'visible' : 'none');
            }
        }

        // Alternar el Terreno 3D satelital
        function toggleTerrain(visible) {
            map.setTerrain(visible ? { source: 'terrain', exaggeration: 1.4 } : null);
        }
    </script>
</body>
</html>
"""

    # Realizar el reemplazo de las variables inyectadas de Python en JS
    final_html = dashboard_html.replace("{stations_json}", stations_json).replace("{trajectories_json}", trajectories_json)

    output_html = "reporte/mapa_3d_interactivo.html"
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"[SUCCESS] Mapa 3D satelital interactivo premium generado en: {output_html}")

if __name__ == "__main__":
    generate_3d_map()
