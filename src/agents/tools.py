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
