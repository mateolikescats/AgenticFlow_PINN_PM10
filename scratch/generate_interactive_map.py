import json
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd

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

    # Generar terreno parabólico que simule el cañón del Valle de Aburrá (Sur-Norte)
    # x es el ancho (Oeste-Este), y es el largo (Sur-Norte)
    lon_grid = np.linspace(lon_min, lon_max, 60)
    lat_grid = np.linspace(lat_min, lat_max, 60)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    X_scaled = 2.0 * (LON - lon_min) / (lon_max - lon_min) - 1.0
    
    # Altura del terreno: río a 1400m en el centro (x_scaled=0), laderas subiendo hasta 2600m
    # Z(x, y) = 1400 + 1200 * x^2
    Z_surface = 1400.0 + 1000.0 * (X_scaled ** 2)

    # Inicializar la figura Plotly
    fig = go.Figure()

    # 1. Agregar superficie 3D del terreno (Valle de Aburrá) con colores terrestres suaves
    fig.add_trace(go.Surface(
        x=LON,
        y=LAT,
        z=Z_surface,
        colorscale='Greens',
        opacity=0.3,
        showscale=False,
        hoverinfo='skip',
        name='Topografía del Valle'
    ))

    # 2. Agregar conos 3D para vectores de viento (pred_viento_vx_m_s, pred_viento_vy_m_s, pred_viento_vz_m_s)
    # go.Cone dibuja flechas/conos 3D
    fig.add_trace(go.Cone(
        x=df['longitud'],
        y=df['latitud'],
        z=df['elevacion'],
        u=df['pred_viento_vx_m_s'],
        v=df['pred_viento_vy_m_s'],
        w=df['pred_viento_vz_m_s'],
        sizemode='absolute',
        sizeref=1.5,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(
            title=dict(text="Viento (m/s)", font=dict(color='white')),
            x=0.88,
            tickfont=dict(color='white')
        ),
        hoverinfo='skip',
        name='Vector Viento'
    ))

    # 3. Agregar puntos de las estaciones (Scatter3d) coloreados por PM2.5
    # Texto de hover personalizado y elegante
    hover_texts = []
    for _, r in df.iterrows():
        text = (
            f"<b>Estación Monitoreo</b><br>"
            f"Lat: {r['latitud']:.4f}° | Lon: {r['longitud']:.4f}°<br>"
            f"Alt: {r['elevacion']:.0f} msnm<br>"
            f"<span style='color:#E53E3E;'><b>PM2.5 Predicho: {r['pred_pm25_ug_m3']:.2f} ug/m3</b></span><br>"
            f"Viento: [{r['pred_viento_vx_m_s']:.2f}, {r['pred_viento_vy_m_s']:.2f}, {r['pred_viento_vz_m_s']:.2f}] m/s<br>"
            f"<span style='color:#319795;'><b>Emisión Inversa (S): {r['pred_emision_S_ug_m3_s']:.6f} ug/(m3*s)</b></span>"
        )
        hover_texts.append(text)

    fig.add_trace(go.Scatter3d(
        x=df['longitud'],
        y=df['latitud'],
        z=df['elevacion'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['pred_pm25_ug_m3'],
            colorscale='YlOrRd', # Colores cálidos y vibrantes para contaminación
            showscale=True,
            colorbar=dict(
                title=dict(text="PM2.5 (ug/m³)", font=dict(color='white')),
                x=1.02,
                tickfont=dict(color='white')
            ),
            line=dict(width=1, color='black')
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Estaciones (PM2.5)'
    ))

    # 4. Configuración del diseño Premium en modo oscuro
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
                backgroundcolor='rgb(10, 15, 30)',
                gridcolor='rgb(45, 55, 72)',
                showbackground=True,
                zerolinecolor='white',
                tickfont=dict(color='rgb(203, 213, 224)'),
            ),
            yaxis=dict(
                title=dict(text='Latitud', font=dict(color='white')),
                backgroundcolor='rgb(10, 15, 30)',
                gridcolor='rgb(45, 55, 72)',
                showbackground=True,
                zerolinecolor='white',
                tickfont=dict(color='rgb(203, 213, 224)'),
            ),
            zaxis=dict(
                title=dict(text='Altitud (msnm)', font=dict(color='white')),
                backgroundcolor='rgb(10, 15, 30)',
                gridcolor='rgb(45, 55, 72)',
                showbackground=True,
                zerolinecolor='white',
                tickfont=dict(color='rgb(203, 213, 224)'),
                range=[1300, 3000]
            ),
            aspectratio=dict(x=1, y=1.2, z=0.6), # Proporciones acotadas del cañón
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.1) # Vista angular ideal
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
