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

    # Inicializar la figura Plotly
    fig = go.Figure()

    # 1. Agregar superficie 3D del terreno con malla de rejilla de alta tecnología (holograma)
    fig.add_trace(go.Surface(
        x=LON,
        y=LAT,
        z=Z_surface,
        colorscale='darkmint',
        opacity=0.25,
        showscale=False,
        hoverinfo='skip',
        name='Topografía del Valle',
        contours=dict(
            x=dict(show=True, highlight=False, color='rgba(255, 255, 255, 0.15)'),
            y=dict(show=True, highlight=False, color='rgba(255, 255, 255, 0.15)'),
            z=dict(show=False)
        )
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

    # 4. Agregar puntos de las estaciones (Scatter3d) coloreados por PM2.5 y dimensionados dinámicamente
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

    # El tamaño de la esfera depende linealmente de la concentración de PM2.5 para resaltar los focos
    marker_sizes = df['pred_pm25_ug_m3'] * 0.8 + 6

    fig.add_trace(go.Scatter3d(
        x=df['longitud'],
        y=df['latitud'],
        z=df['elevacion'],
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=df['pred_pm25_ug_m3'],
            colorscale='YlOrRd', # Colores cálidos y vibrantes para contaminación
            showscale=True,
            colorbar=dict(
                title=dict(text="PM2.5 (ug/m³)", font=dict(color='white')),
                x=0.96,
                tickfont=dict(color='white')
            ),
            line=dict(width=2, color='white') # Borde blanco de alto contraste para visibilidad
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Estaciones (PM2.5)'
    ))

    # 5. Configuración del diseño Premium en modo oscuro
    fig.update_layout(
        paper_bgcolor='rgb(10, 15, 30)', # Fondo azul oscuro premium
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
                range=[1100, 2800] # Límites del cañón optimizados
            ),
            aspectratio=dict(x=1, y=1.2, z=0.7), # Proporciones acotadas del cañón
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.9) # Vista angular ideal
            )
        ),
        legend=dict(
            font=dict(color='white'),
            x=0.02,
            y=0.98
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        autosize=True
    )

    # Generar el fragmento HTML del gráfico
    plot_div = op.plot(fig, output_type='div', include_plotlyjs='cdn')

    # Envolver en una plantilla HTML Premium autodescriptiva
    dashboard_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>iPINN 3D - Visualizador de Calidad del Aire y Vientos</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #060814;
            color: #e2e8f0;
            overflow: hidden;
            height: 100vh;
            display: flex;
        }}
        .dashboard {{
            display: flex;
            width: 100vw;
            height: 100vh;
        }}
        .sidebar {{
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
        }}
        .map-container {{
            flex: 1;
            height: 100vh;
            background-color: #0a0f1e;
            position: relative;
        }}
        .map-container > div {{
            width: 100% !important;
            height: 100% !important;
        }}
        .badge {{
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
        }}
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 26px;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }}
        .subtitle {{
            font-size: 12.5px;
            color: #64748b;
            margin-top: 2px;
        }}
        h2 {{
            font-family: 'Outfit', sans-serif;
            font-size: 15px;
            font-weight: 600;
            color: #38bdf8;
            border-bottom: 1px solid rgba(56, 189, 248, 0.2);
            padding-bottom: 6px;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        p.description {{
            font-size: 13px;
            line-height: 1.6;
            color: #94a3b8;
        }}
        .legend-list {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: flex-start;
            gap: 14px;
            font-size: 12.5px;
            color: #cbd5e1;
            line-height: 1.5;
        }}
        .legend-icon {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            flex-shrink: 0;
            margin-top: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .icon-station {{
            background: linear-gradient(135deg, #facc15, #ef4444);
            border: 2px solid white;
            box-shadow: 0 0 5px rgba(239, 68, 68, 0.5);
        }}
        .icon-wind {{
            background: #3b82f6;
            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
            border-radius: 0;
        }}
        .icon-terrain {{
            background: rgba(16, 185, 129, 0.2);
            border: 1.5px solid rgba(16, 185, 129, 0.6);
            border-radius: 3px;
        }}
        .icon-stalk {{
            width: 4px;
            height: 16px;
            background: transparent;
            border-left: 2px dashed rgba(255, 255, 255, 0.5);
            border-radius: 0;
            margin-left: 6px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .metric-value {{
            font-size: 15px;
            font-weight: 700;
            color: #f1f5f9;
            font-family: 'Outfit', sans-serif;
        }}
        .metric-label {{
            font-size: 9px;
            color: #64748b;
            margin-top: 3px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .control-tip {{
            background: rgba(56, 189, 248, 0.04);
            border-left: 3px solid #38bdf8;
            padding: 12px;
            border-radius: 0 8px 8px 0;
            font-size: 11.5px;
            color: #94a3b8;
            line-height: 1.5;
        }}
        .control-tip b {{
            color: #38bdf8;
        }}
        /* Personalizar barra de desplazamiento */
        .sidebar::-webkit-scrollbar {{
            width: 6px;
        }}
        .sidebar::-webkit-scrollbar-track {{
            background: transparent;
        }}
        .sidebar::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }}
        .sidebar::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <div>
                <span class="badge">iPINN Inversa v1.1</span>
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
                <h2>Leyenda del Mapa 3D</h2>
                <div class="legend-list">
                    <div class="legend-item">
                        <div class="legend-icon icon-station"></div>
                        <div>
                            <b>Estaciones (PM2.5):</b> Esferas coloreadas por nivel de contaminación. El <b>diámetro</b> indica la concentración (mayor tamaño = más PM2.5).
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-wind"></div>
                        <div>
                            <b>Campo de Vientos:</b> Conos azules flotantes orientados en la dirección del viento predicho por la física del modelo en altura.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-stalk"></div>
                        <div>
                            <b>Soportes de Altura:</b> Líneas que conectan la estación con el suelo, indicando su elevación real sobre el nivel del mar.
                        </div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-icon icon-terrain"></div>
                        <div>
                            <b>Relieve del Valle:</b> Topografía simplificada del cañón. Se observa cómo la altitud desciende hacia el Norte a medida que el río avanza.
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
</body>
</html>
"""

    output_html = "reporte/mapa_3d_interactivo.html"
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(dashboard_html)
    print(f"[SUCCESS] Mapa 3D interactivo premium generado en: {output_html}")

if __name__ == "__main__":
    generate_3d_map()
