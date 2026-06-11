import json
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.offline as op

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

def interpolate_pm25(lon_grid, lat_grid, stations_df):
    lons = stations_df['longitud'].values
    lats = stations_df['latitud'].values
    vals = stations_df['pred_pm25_ug_m3'].values
    
    interpolated = np.zeros_like(lon_grid)
    for i in range(lon_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            lg = lon_grid[i, j]
            lt = lat_grid[i, j]
            dists = np.sqrt((lons - lg)**2 + (lats - lt)**2)
            if np.any(dists < 1e-5):
                interpolated[i, j] = vals[np.argmin(dists)]
            else:
                # Inverse Distance Weighting
                weights = 1.0 / (dists ** 2)
                interpolated[i, j] = np.sum(vals * weights) / np.sum(weights)
    return interpolated

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

    # Generar terreno parabólico que simula la geografía real del Valle de Aburrá (Sur-Norte e inclinado)
    lon_grid = np.linspace(lon_min, lon_max, 70)
    lat_grid = np.linspace(lat_min, lat_max, 70)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    # Calcular Z del terreno usando el modelo geográfico
    Z_surface = get_terrain_height(LON, LAT)
    
    # Calcular mapa de calor espacial de PM2.5 sobre el terreno
    PM25_grid = interpolate_pm25(LON, LAT, df)

    # Inicializar la figura Plotly
    fig = go.Figure()

    # 1. Agregar superficie 3D del terreno pintada como mapa de calor espacial de PM2.5
    fig.add_trace(go.Surface(
        x=LON,
        y=LAT,
        z=Z_surface,
        surfacecolor=PM25_grid,
        colorscale='YlOrRd', # Colores de contaminación (Amarillo -> Naranja -> Rojo)
        opacity=0.35,
        showscale=True,
        colorbar=dict(
            title=dict(text="PM2.5 en Terreno (ug/m³)", font=dict(color='white')),
            x=-0.08, # Colorbar a la izquierda
            tickfont=dict(color='white')
        ),
        hoverinfo='skip',
        name='Topografía del Valle'
    ))

    # Calcular base del terreno para los soportes verticales
    z_ground = []
    for _, r in df.iterrows():
        g_h = get_terrain_height(r['longitud'], r['latitud'])
        # Asegurar al menos 60 metros de espacio para que no queden enterrados
        z_ground.append(min(r['elevacion'] - 60.0, g_h))
    df['z_ground'] = z_ground

    # 2. Agregar soportes verticales (stalks) desde el terreno hasta la estación y el cono
    stalk_x = []
    stalk_y = []
    stalk_z = []
    for _, r in df.iterrows():
        stalk_x.extend([r['longitud'], r['longitud'], r['longitud'], None])
        stalk_y.extend([r['latitud'], r['latitud'], r['latitud'], None])
        stalk_z.extend([r['z_ground'], r['elevacion'], r['elevacion'] + 80.0, None])

    fig.add_trace(go.Scatter3d(
        x=stalk_x,
        y=stalk_y,
        z=stalk_z,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.35)', width=2),
        hoverinfo='skip',
        name='Soportes de Estaciones'
    ))

    # 3. Agregar conos 3D para vectores de viento (elevados a +80m para evitar solapamientos)
    fig.add_trace(go.Cone(
        x=df['longitud'],
        y=df['latitud'],
        z=df['elevacion'] + 80.0,
        u=df['pred_viento_vx_m_s'],
        v=df['pred_viento_vy_m_s'],
        w=df['pred_viento_vz_m_s'],
        sizemode='absolute',
        sizeref=2.5,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(
            title=dict(text="Viento (m/s)", font=dict(color='white')),
            x=0.82,
            tickfont=dict(color='white')
        ),
        hoverinfo='skip',
        name='Vector Viento'
    ))

    # 4. Calcular e integrar las Trayectorias de Dispersión (Smoke Trails)
    dt = 0.015  # Paso temporal de simulación visual
    for idx, r in df.iterrows():
        curr_lon = r['longitud']
        curr_lat = r['latitud']
        curr_elev = r['elevacion']
        
        path_x = [curr_lon]
        path_y = [curr_lat]
        path_z = [curr_elev]
        
        # Integrar la trayectoria de partículas usando el campo de viento de la PINN
        for step in range(12):
            vx, vy, vz = interpolate_wind(curr_lon, curr_lat, df)
            curr_lon += vx * dt
            curr_lat += vy * dt
            curr_elev += vz * 12000.0 * dt  # Escalado vertical visible
            
            # Limitar a la caja de visualización
            curr_lon = np.clip(curr_lon, lon_min, lon_max)
            curr_lat = np.clip(curr_lat, lat_min, lat_max)
            
            path_x.append(curr_lon)
            path_y.append(curr_lat)
            path_z.append(curr_elev)
            
        fig.add_trace(go.Scatter3d(
            x=path_x,
            y=path_y,
            z=path_z,
            mode='lines',
            line=dict(
                color=list(range(len(path_x), 0, -1)),  # Gradiente inverso: rojo denso en origen, amarillo diluido al final
                colorscale='YlOrRd',
                width=3.5
            ),
            hoverinfo='skip',
            showlegend=False,
            name='Trayectorias de Dispersión'
        ))

    # 5. Agregar puntos de las estaciones (Scatter3d) coloreados por PM2.5 y dimensionados dinámicamente
    hover_texts = []
    for _, r in df.iterrows():
        text = (
            f"<b>Estación Monitoreo</b><br>"
            f"Lat: {r['latitud']:.4f}° | Lon: {r['longitud']:.4f}°<br>"
            f"Altitud: {r['elevacion']:.0f} msnm (Fondo: {r['z_ground']:.0f} msnm)<br>"
            f"<span style='color:#FF5A5F;'><b>PM2.5 Predicho: {r['pred_pm25_ug_m3']:.2f} ug/m3</b></span><br>"
            f"Viento: [{r['pred_viento_vx_m_s']:.2f}, {r['pred_viento_vy_m_s']:.2f}, {r['pred_viento_vz_m_s']:.2f}] m/s<br>"
            f"<span style='color:#00B4D8;'><b>Emisión Inversa (S): {r['pred_emision_S_ug_m3_s']:.6f} ug/(m3*s)</b></span>"
        )
        hover_texts.append(text)

    marker_sizes = df['pred_pm25_ug_m3'] * 0.8 + 6

    fig.add_trace(go.Scatter3d(
        x=df['longitud'],
        y=df['latitud'],
        z=df['elevacion'],
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=df['pred_pm25_ug_m3'],
            colorscale='YlOrRd',
            showscale=False,  # La escala de PM2.5 ya está representada en el mapa de calor terrestre
            line=dict(width=2, color='white')
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Estaciones (PM2.5)'
    ))

    # 6. Configuración del diseño Premium en modo oscuro
    fig.update_layout(
        paper_bgcolor='rgb(10, 15, 30)',
        plot_bgcolor='rgb(10, 15, 30)',
        scene=dict(
            xaxis=dict(
                title=dict(text='Longitud', font=dict(color='white')),
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=False,
                zerolinecolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='rgb(203, 213, 224)'),
            ),
            yaxis=dict(
                title=dict(text='Latitud', font=dict(color='white')),
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=False,
                zerolinecolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='rgb(203, 213, 224)'),
            ),
            zaxis=dict(
                title=dict(text='Altitud (msnm)', font=dict(color='white')),
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=False,
                zerolinecolor='rgba(255, 255, 255, 0.2)',
                tickfont=dict(color='rgb(203, 213, 224)'),
                range=[1100, 2800]
            ),
            aspectratio=dict(x=1, y=1.2, z=0.7),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.9)
            )
        ),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True
    )

    plot_div = op.plot(fig, output_type='div', include_plotlyjs='cdn')

    # Envolver en una plantilla HTML Premium autodescriptiva usando string normal
    dashboard_template = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iPINN 3D - Visualizador de Calidad del Aire y Vientos</title>
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
            background: rgba(10, 15, 30, 0.85);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            padding: 30px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 24px;
            z-index: 10;
            box-shadow: 10px 0 35px rgba(0, 0, 0, 0.6);
        }
        .map-container {
            flex: 1;
            height: 100vh;
            background-color: #0a0f1e;
            position: relative;
        }
        .map-container > div {
            width: 100% !important;
            height: 100% !important;
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
            font-size: 26px;
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
            font-size: 15px;
            font-weight: 600;
            color: #38bdf8;
            border-bottom: 1px solid rgba(56, 189, 248, 0.2);
            padding-bottom: 6px;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        p.description {
            font-size: 13px;
            line-height: 1.6;
            color: #94a3b8;
        }
        .toggle-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 8px;
        }
        .toggle-label {
            display: flex;
            align-items: center;
            gap: 12px;
            cursor: pointer;
            font-size: 13px;
            color: #cbd5e1;
            user-select: none;
        }
        .toggle-label input {
            display: none;
        }
        .toggle-custom {
            position: relative;
            width: 36px;
            height: 20px;
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            transition: all 0.3s ease;
            flex-shrink: 0;
        }
        .toggle-custom::after {
            content: '';
            position: absolute;
            width: 14px;
            height: 14px;
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
            gap: 12px;
        }
        .legend-item {
            display: flex;
            align-items: flex-start;
            gap: 14px;
            font-size: 12.5px;
            color: #cbd5e1;
            line-height: 1.5;
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
            background: linear-gradient(135deg, #facc15, #ef4444);
            border: 2px solid white;
            box-shadow: 0 0 5px rgba(239, 68, 68, 0.5);
        }
        .icon-wind {
            background: #3b82f6;
            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
            border-radius: 0;
        }
        .icon-trail {
            background: linear-gradient(90deg, #ef4444, #facc15);
            border-radius: 4px;
            height: 6px;
            width: 18px;
            margin-top: 6px;
        }
        .icon-terrain {
            background: rgba(239, 68, 68, 0.15);
            border: 1.5px solid rgba(239, 68, 68, 0.5);
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
            gap: 10px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .metric-value {
            font-size: 15px;
            font-weight: 700;
            color: #f1f5f9;
            font-family: 'Outfit', sans-serif;
        }
        .metric-label {
            font-size: 9px;
            color: #64748b;
            margin-top: 3px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .control-tip {
            background: rgba(56, 189, 248, 0.04);
            border-left: 3px solid #38bdf8;
            padding: 12px;
            border-radius: 0 8px 8px 0;
            font-size: 11.5px;
            color: #94a3b8;
            line-height: 1.5;
        }
        .control-tip b {
            color: #38bdf8;
        }
        /* Personalizar barra de desplazamiento */
        .sidebar::-webkit-scrollbar {
            width: 6px;
        }
        .sidebar::-webkit-scrollbar-track {
            background: transparent;
        }
        .sidebar::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        .sidebar::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <div>
                <span class="badge">iPINN Inversa v1.2</span>
                <h1>iPINN 3D: Valle de Aburrá</h1>
                <div class="subtitle">Predicción Física y Estimación de Emisiones en Tiempo Real</div>
            </div>

            <div>
                <h2>¿Qué es esta simulación?</h2>
                <p class="description">
                    Este mapa interactivo muestra la simulación tridimensional del viento y la dispersión de partículas contaminantes (<b>PM2.5</b>) en el cañón del Valle de Aburrá, Colombia. El modelo ha sido entrenado usando <b>Redes Neuronales Informadas por la Física (PINN)</b>, combinando las ecuaciones de conservación de masa de fluidos con mediciones de la red SIATA.
                </p>
            </div>

            <div>
                <h2>Capas Visuales</h2>
                <div class="toggle-container">
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-terrain" checked onchange="toggleLayer('Topografía del Valle', this.checked)">
                        <span class="toggle-custom"></span>
                        Mapa de Calor (Terreno)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-stations" checked onchange="toggleLayer('Estaciones (PM2.5)', this.checked)">
                        <span class="toggle-custom"></span>
                        Estaciones (PM2.5)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-trails" checked onchange="toggleLayer('Trayectorias de Dispersión', this.checked)">
                        <span class="toggle-custom"></span>
                        Trayectorias (Estelas Pluma)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-wind" checked onchange="toggleLayer('Vector Viento', this.checked)">
                        <span class="toggle-custom"></span>
                        Campo de Vientos (Conos)
                    </label>
                    <label class="toggle-label">
                        <input type="checkbox" id="toggle-stalks" checked onchange="toggleLayer('Soportes de Estaciones', this.checked)">
                        <span class="toggle-custom"></span>
                        Soportes de Altura (Líneas)
                    </label>
                </div>
            </div>

            <div>
                <h2>Leyenda del Mapa 3D</h2>
                <div class="legend-list">
                    <div class="legend-item">
                        <div class="legend-icon icon-terrain"></div>
                        <div>
                            <b>Mapa de Calor (Terreno):</b> Colorea el relieve según la concentración de PM2.5 (Amarillo = Bajo, Rojo = Alto). Revela de un vistazo qué tan grandes y dónde se acumulan los focos de contaminación.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-trail"></div>
                        <div>
                            <b>Trayectorias de Dispersión:</b> Estelas con gradiente que nacen en cada estación. Indican <b>hacia dónde se dirige</b> la contaminación (del rojo denso en origen, al amarillo al diluirse por el viento).
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-station"></div>
                        <div>
                            <b>Estaciones (PM2.5):</b> Esferas de medición. El <b>diámetro</b> indica la cantidad de PM2.5 detectada.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-wind"></div>
                        <div>
                            <b>Campo de Vientos:</b> Conos orientados en la dirección del flujo de aire en altura predicho por la física del modelo.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-stalk"></div>
                        <div>
                            <b>Soportes de Altura:</b> Líneas que conectan la estación con el suelo para evaluar la altitud en el relieve.
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
                • <b>Rotar:</b> Clic izquierdo y arrastrar.<br>
                • <b>Zoom:</b> Rueda del ratón.<br>
                • <b>Desplazar (Pan):</b> Clic derecho y arrastrar.<br>
                • <b>Detalles:</b> Pasa el cursor sobre una esfera para ver la ficha técnica y la <b>emisión inversa (S)</b>.
            </div>
        </div>
        <div class="map-container">
            {plot_div}
        </div>
    </div>

    <!-- Script de Control para ocultar/mostrar capas vía API de Plotly -->
    <script type="text/javascript">
        function toggleLayer(layerName, visible) {
            var gd = document.querySelector('.plotly-graph-div');
            if (!gd || !gd.data) {
                setTimeout(function() { toggleLayer(layerName, visible); }, 100);
                return;
            }
            var update = { visible: visible ? true : false };
            var indices = [];
            for (var i = 0; i < gd.data.length; i++) {
                if (gd.data[i].name === layerName) {
                    indices.push(i);
                }
            }
            if (indices.length > 0) {
                Plotly.restyle(gd, update, indices);
            }
        }
    </script>
</body>
</html>
"""

    dashboard_html = dashboard_template.replace("{plot_div}", plot_div)
    
    output_html = "reporte/mapa_3d_interactivo.html"
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    print(f"[SUCCESS] Mapa 3D interactivo premium generado en: {output_html}")

if __name__ == "__main__":
    generate_3d_map()
