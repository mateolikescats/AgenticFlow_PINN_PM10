import json
import os
import numpy as np
import plotly.graph_objects as go
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
            x=0.85,
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
                x=1.02,
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
        title={
            'text': "<b>iPINN 3D: Predicciones Físicas y Emisión Inversa (Últimas 96 Horas)</b>",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'color': 'white', 'family': 'Outfit, Inter'}
        },
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
        margin=dict(l=0, r=0, b=0, t=50)
    )

    output_html = "reporte/mapa_3d_interactivo.html"
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"[SUCCESS] Mapa 3D interactivo premium generado en: {output_html}")

if __name__ == "__main__":
    generate_3d_map()
