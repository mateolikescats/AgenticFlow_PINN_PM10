import subprocess
import json
import os
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.mixture import GaussianMixture

# --- Schema and Tool for Julia PINN Execution ---
class ExecuteJuliaPINNInput(BaseModel):
    epochs: int = Field(..., description="Número de epochs para el optimizador Adam de la PINN.")
    learning_rate: float = Field(..., description="Tasa de aprendizaje (learning rate) para el entrenamiento.")

class ExecuteJuliaPINNTool(BaseTool):
    name: str = "Execute Julia PINN Tool"
    description: str = "Ejecuta el script numérico de Julia que resuelve la Ecuación de Boussinesq mediante Redes Neuronales Informadas por Física."
    args_schema: Type[BaseModel] = ExecuteJuliaPINNInput

    def _run(self, epochs: int, learning_rate: float) -> str:
        # Escribir configuración temporal para que Julia la lea
        config = {"epochs": epochs, "learning_rate": learning_rate}
        with open("pinn_config.json", "w") as f:
            json.dump(config, f)
            
        print(f"Lanzando proceso de Julia con {epochs} epochs y LR={learning_rate}...")
        try:
            # Ejecutar script en Julia (CLI Bridge)
            # Timeout de 5 minutos por seguridad
            result = subprocess.run(
                [r"C:\Users\arnod\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe", "src/pinn/train_interpolative.jl"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=600
            )
            
            # Recolectar salida
            output = f"Exit Code: {result.returncode}\n\n"
            output += "=== STDOUT ===\n" + result.stdout + "\n"
            
            if result.stderr:
                output += "=== STDERR ===\n" + result.stderr + "\n"
                
            return output
        except subprocess.TimeoutExpired:
            return "Error: El proceso de Julia excedió el tiempo límite (Timeout de 5 minutos)."
        except Exception as e:
            return f"Error en la ejecución de Julia: {str(e)}"

# --- Schema and Tool for Spatiotemporal Clustering (GMM) ---
class ClusteringInput(BaseModel):
    num_components: int = Field(..., description="Número esperado de nubes/cúmulos de contaminación a buscar.")

class SpatiotemporalClusteringTool(BaseTool):
    name: str = "Spatiotemporal GMM Clustering Tool"
    description: str = "Aplica Gaussian Mixture Models (GMM) a los datos de PM2.5 a lo largo del tiempo para identificar distintas nubes de contaminación en movimiento."
    args_schema: Type[BaseModel] = ClusteringInput
    
    def _run(self, num_components: int) -> str:
        if not os.path.exists("datos_siata_temporal.json"):
            return "Error: No se encontró datos_siata_temporal.json"
            
        try:
            with open("datos_siata_temporal.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Extraer características espaciotemporales pesadas por concentración
            # Filtramos solo puntos con contaminación significativa
            df = pd.DataFrame(data)
            df_filtered = df[df["pm25"] > 20.0].copy()
            
            if len(df_filtered) < num_components:
                return "Error: No hay suficientes puntos contaminados para formar el número de cúmulos solicitados."
                
            # Features: latitud, longitud, altitud(z), tiempo(t)
            X = df_filtered[['latitud', 'longitud', 'z', 't']].values
            weights = df_filtered['pm25'].values
            
            # Aplicar GMM
            gmm = GaussianMixture(n_components=num_components, random_state=42)
            # En la práctica se pesa por pm25, pero sklearn GMM no acepta weights en fit directamente fácilmente,
            # así que entrenamos sobre la distribución geométrica
            gmm.fit(X)
            
            df_filtered['cluster'] = gmm.predict(X)
            
            report = f"Clustering Espaciotemporal Completado con GMM ({num_components} componentes).\n\n"
            for i in range(num_components):
                cluster_data = df_filtered[df_filtered['cluster'] == i]
                mean_pm25 = cluster_data['pm25'].mean()
                mean_t = cluster_data['t'].mean()
                report += f"Cúmulo {i}: {len(cluster_data)} mediciones | Promedio PM2.5: {mean_pm25:.2f} | Tiempo medio: {mean_t:.2f}\n"
                
            return report
        except Exception as e:
            return f"Error en clustering: {str(e)}"

# --- New Tool for Geospatial Valle Query ---
class GeospatialValleQueryInput(BaseModel):
    x_coordinate: float = Field(..., description="Coordenada x adimensionalizada del Valle (-1.0 a 1.0) a interrogar.")

class GeospatialValleQueryTool(BaseTool):
    name: str = "Geospatial Valle Query Tool"
    description: str = "Resuelve y cruza una coordenada adimensional x (-1.0 a 1.0) con puntos geográficos reales del Valle de Aburrá."
    args_schema: Type[BaseModel] = GeospatialValleQueryInput
    
    def _run(self, x_coordinate: float) -> str:
        x = x_coordinate
        if x < -1.0 or x > 1.0:
            return f"Coordenada x = {x} fuera del dominio [-1.0, 1.0]"
            
        if -1.0 <= x < -0.6:
            return (
                f"Coordenada x = {x:.2f} corresponde a la Ladera Occidental de Medellín (San Javier y Belén). "
                "Área residencial con topografía empinada, con alta vulnerabilidad a la acumulación de contaminantes "
                "cuando la altura de mezcla desciende y se bloquea la ventilación transversal."
            )
        elif -0.6 <= x < -0.2:
            return (
                f"Coordenada x = {x:.2f} corresponde al Corredor Industrial del Sur (Sabaneta y La Estrella). "
                "Zona caracterizada por una concentración densa de fábricas pesadas, fundidoras de metales y "
                "plantas de asfalto, representando una de las fuentes estacionarias más potentes en el cañón."
            )
        elif -0.2 <= x < 0.2:
            return (
                f"Coordenada x = {x:.2f} corresponde al Fondo del Cañón / Centro de Medellín. "
                "Zona comercial de alta congestión. Cruzada por la Autopista Norte e importantes avenidas. "
                "Es el punto crítico de emisiones vehiculares móviles (transporte público y camiones de carga de diésel)."
            )
        elif 0.2 <= x < 0.6:
            return (
                f"Coordenada x = {x:.2f} corresponde al Corredor Industrial del Norte / Zona de Itagüí. "
                "Concentración masiva de industrias de manufactura, textileras y plantas de procesamiento. "
                "Registra niveles elevados y constantes de emisiones industriales basales."
            )
        else: # 0.6 <= x <= 1.0
            return (
                f"Coordenada x = {x:.2f} corresponde a la Ladera Oriental de Medellín (Manrique y Villa Hermosa). "
                "Ladera residencial asimétrica con relieve empinado. Registra bajas emisiones locales, pero una "
                "alta susceptibilidad al estancamiento de partículas debido al flujo ascendente de laderas (vientos anabáticos)."
            )

# --- New Tool for Writing Standalone LaTeX Forensic Report ---
class WriteLatexForensicReportInput(BaseModel):
    report_content: str = Field(..., description="Contenido completo en código LaTeX del reporte forense standalone.")

class WriteLatexForensicReportTool(BaseTool):
    name: str = "Write Latex Forensic Report Tool"
    description: str = "Crea y guarda el reporte forense standalone final en formato LaTeX en la ruta reporte/reporte_forense.tex."
    args_schema: Type[BaseModel] = WriteLatexForensicReportInput
    
    def _run(self, report_content: str) -> str:
        report_dir = "reporte"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "reporte_forense.tex")
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            return f"✅ Reporte forense standalone en LaTeX guardado exitosamente en: {report_path}"
        except Exception as e:
            return f"❌ Error guardando el reporte en LaTeX: {str(e)}"
