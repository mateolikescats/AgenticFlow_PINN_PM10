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

    def _save_run_to_memory(self, memory_file: str, history: list, epochs: int, requested_lr: float, started_lr: float, final_lr: float, status: str, final_loss: float):
        import datetime
        new_run = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epochs": epochs,
            "requested_lr": requested_lr,
            "started_lr": started_lr,
            "final_lr": final_lr,
            "status": status,
            "final_loss": final_loss
        }
        history.append(new_run)
        try:
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f"[MLOps Warning] No se pudo guardar en la memoria: {e}", flush=True)

    def _generate_loss_plot(self, losses: list, output_path: str = "reporte/curva_perdida.png"):
        if not losses:
            print("[MLOps Warning] No hay pérdidas registradas para generar la gráfica.", flush=True)
            return
        try:
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(losses) + 1), losses, label="Pérdida (Loss)", color="#2B6CB0", linewidth=2)
            plt.xlabel("Época / Iteración")
            plt.ylabel("Pérdida")
            plt.title("Curva de Convergencia del Entrenamiento PINN (Adam + L-BFGS)")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"[MLOps Success] Grafica de perdida guardada exitosamente en: {output_path}", flush=True)
        except Exception as e:
            print(f"[MLOps Warning] No se pudo generar la gráfica de pérdida: {e}", flush=True)

    def _run(self, epochs: int, learning_rate: float) -> str:
        memory_file = "mlops_memory.json"
        history = []
        if os.path.exists(memory_file):
            try:
                with open(memory_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception as e:
                print(f"[MLOps Warning] No se pudo leer la memoria: {e}", flush=True)

        current_lr = learning_rate
        current_epochs = epochs
        max_retries = 3
        retry_count = 0
        execution_summary = []

        # Analizar fallos previos
        failed_rates = []
        best_successful_lr = None
        best_loss = float('inf')
        
        for run in history:
            status = run.get("status")
            final_loss = run.get("final_loss")
            started_lr = run.get("started_lr")
            if status in ["nan", "stalled", "failed"]:
                if started_lr is not None:
                    failed_rates.append(started_lr)
            elif status == "success" and final_loss is not None:
                try:
                    loss_val = float(final_loss)
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_successful_lr = started_lr
                except (ValueError, TypeError):
                    pass

        # Si el learning rate actual solicitado (o mayor) falló en el pasado
        if failed_rates:
            min_failed_lr = min(failed_rates)
            if current_lr >= min_failed_lr:
                if best_successful_lr is not None and best_successful_lr < min_failed_lr:
                    current_lr = best_successful_lr
                else:
                    current_lr = min(0.01, min_failed_lr / 2.0)
                msg = f"[MLOps Meta-Learning] AJUSTE PREVENTIVO: Se solicitó LR={learning_rate}, pero fallos anteriores con LR >= {min_failed_lr} obligan a usar un LR seguro de {current_lr}."
                print(msg, flush=True)
                execution_summary.append(msg)

        started_lr = current_lr
        has_nan = False
        has_stalled = False

        while retry_count < max_retries:
            # Escribir configuración temporal para que Julia la lea
            config = {"epochs": current_epochs, "learning_rate": current_lr}
            with open("pinn_config.json", "w") as f:
                json.dump(config, f)
                
            msg = f"[RUN] [Intento {retry_count + 1}/{max_retries}] Lanzando Julia con {current_epochs} epochs y LR={current_lr}..."
            print(msg, flush=True)
            execution_summary.append(msg)
            
            try:
                # Usar Popen para streaming en tiempo real
                julia_path = r"C:\Users\arnod\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"
                if not os.path.exists(julia_path):
                    julia_path = "julia"
                
                process = subprocess.Popen(
                    [julia_path, "src/pinn/train_interpolative.jl"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                last_losses = []
                has_nan = False
                has_stalled = False
                stdout_captured = []
                
                # Bucle de lectura de salida en tiempo real
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break  # EOF
                    
                    stdout_captured.append(line)
                    stripped_line = line.strip()
                    print(f"[Julia]: {stripped_line}", flush=True)
                    
                    # Parsear logs de época
                    if "[EPOCH_LOG]" in stripped_line:
                        try:
                            # Formato: [EPOCH_LOG] Epoch: 12 | Loss: 0.4285
                            parts = stripped_line.split("|")
                            epoch_part = parts[0]
                            loss_part = parts[1]
                            
                            epoch_num = int(epoch_part.split(":")[-1].strip())
                            loss_str = loss_part.split(":")[-1].strip()
                            
                            if "nan" in loss_str.lower():
                                has_nan = True
                                print(f"[WARNING] [MLOps Control] DIVERGENCIA DETECTADA! (Loss = NaN) en Epoch {epoch_num}. Deteniendo proceso...", flush=True)
                                process.terminate()
                                break
                            
                            loss_val = float(loss_str)
                            last_losses.append(loss_val)
                            
                            # Verificar estancamiento (si el loss no disminuye en 15 epochs)
                            if len(last_losses) >= 15:
                                diff = last_losses[-10] - last_losses[-1]
                                if diff < 1e-5 and last_losses[-1] > 0.05:
                                    has_stalled = True
                                    print(f"[WARNING] [MLOps Control] ESTANCAMIENTO DETECTADO! (Pérdida estancada en {loss_val}) en Epoch {epoch_num}. Deteniendo proceso...", flush=True)
                                    process.terminate()
                                    break
                        except Exception:
                            pass
                
                process.wait()
                
                # Evaluar resultado del intento
                if has_nan:
                    execution_summary.append(f"[FAIL] Intento {retry_count + 1} falló por colapso a NaN.")
                    current_lr /= 2.0  # Disminuir learning rate
                    retry_count += 1
                    continue
                elif has_stalled:
                    execution_summary.append(f"[FAIL] Intento {retry_count + 1} falló por estancamiento de pérdida.")
                    current_lr /= 2.0  # Intentar con un paso más pequeño
                    retry_count += 1
                    continue
                elif process.returncode != 0:
                    err_msg = f"[FAIL] Intento {retry_count + 1} falló con código de salida {process.returncode}."
                    execution_summary.append(err_msg)
                    retry_count += 1
                    continue
                else:
                    success_msg = f"[SUCCESS] Entrenamiento completado con éxito en el intento {retry_count + 1}. Loss final: {last_losses[-1] if last_losses else 'Desconocido'}"
                    execution_summary.append(success_msg)
                    print(success_msg, flush=True)
                    
                    # Guardar éxito en memoria
                    final_loss = last_losses[-1] if last_losses else None
                    self._save_run_to_memory(memory_file, history, epochs, learning_rate, started_lr, current_lr, "success", final_loss)
                    
                    # Generar gráfica de pérdida
                    self._generate_loss_plot(last_losses)
                    
                    # Retornar el reporte final
                    report = "\n".join(execution_summary) + "\n\n=== LOGS DEL ENTRENAMIENTO ===\n" + "".join(stdout_captured)
                    return report
                    
            except Exception as e:
                err_msg = f"[FAIL] Error ejecutando Julia en intento {retry_count + 1}: {str(e)}"
                execution_summary.append(err_msg)
                print(err_msg, flush=True)
                retry_count += 1
                
        # Si agotamos los intentos
        status = "nan" if has_nan else ("stalled" if has_stalled else "failed")
        self._save_run_to_memory(memory_file, history, epochs, learning_rate, started_lr, current_lr, status, None)
        
        # Generar gráfica de pérdida de los intentos fallidos si se recopilaron pérdidas
        if 'last_losses' in locals() and last_losses:
            self._generate_loss_plot(last_losses)
            
        fail_report = f"[FAIL] Todos los {max_retries} intentos de entrenamiento fallaron.\n" + "\n".join(execution_summary)
        return fail_report

# --- Schema and Tool for Spatiotemporal Clustering (GMM) ---
class ClusteringInput(BaseModel):
    num_components: int = Field(..., description="Número esperado de nubes/cúmulos de contaminación a buscar.")

class SpatiotemporalClusteringTool(BaseTool):
    name: str = "Spatiotemporal GMM Clustering Tool"
    description: str = "Aplica Gaussian Mixture Models (GMM) a las predicciones de PM2.5 y emisión (S) de las últimas 48 horas para identificar distintas nubes de contaminación."
    args_schema: Type[BaseModel] = ClusteringInput
    
    def _run(self, num_components: int) -> str:
        if not os.path.exists("output_predictions.json"):
            return "Error: No se encontró el archivo de predicciones 'output_predictions.json'. Por favor ejecuta primero predict_realtime.py para generar las predicciones de las últimas 48 horas."
            
        try:
            with open("output_predictions.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                
            df = pd.DataFrame(data)
            
            # Mapear las columnas de predicciones a las esperadas por el análisis GMM
            df["pm25"] = df["pred_pm25_ug_m3"]
            df["t"] = df["timestamp"]
            # Adimensionalizar elevación a z en [0, 1] (elev_min=1400.0, elev_max=3000.0)
            df["z"] = (df["elevacion"] - 1400.0) / 1600.0
            
            # Calcular x e y adimensionales para los centroides (lon_min=-75.7, lon_max=-75.3, lat_min=6.0, lat_max=6.45)
            lon_min, lon_max = -75.7, -75.3
            lat_min, lat_max = 6.0, 6.45
            df["x"] = 2.0 * (df["longitud"] - lon_min) / (lon_max - lon_min) - 1.0
            df["y"] = 2.0 * (df["latitud"] - lat_min) / (lat_max - lat_min) - 1.0
            
            # Filtrar por el percentil 50 de PM2.5 para capturar áreas con contaminación por encima del promedio
            # Esto evita fallar si no hay valores por encima de 20 ug/m3 (por ejemplo, en días limpios)
            threshold = df["pm25"].median()
            df_filtered = df[df["pm25"] >= threshold].copy()
            
            if len(df_filtered) < num_components:
                return "Error: No hay suficientes puntos con contaminación para formar el número de cúmulos solicitados."
                
            # Features: latitud, longitud, altitud(z), tiempo(t)
            X = df_filtered[['latitud', 'longitud', 'z', 't']].values
            
            # Aplicar GMM
            gmm = GaussianMixture(n_components=num_components, random_state=42)
            gmm.fit(X)
            
            df_filtered['cluster'] = gmm.predict(X)
            
            report = f"Clustering Espaciotemporal Completado con GMM ({num_components} componentes) usando las predicciones de las últimas 48 horas.\n\n"
            for i in range(num_components):
                cluster_data = df_filtered[df_filtered['cluster'] == i]
                mean_pm25 = cluster_data['pm25'].mean()
                mean_t = cluster_data['t'].mean()
                mean_x = cluster_data['x'].mean()
                mean_y = cluster_data['y'].mean()
                mean_S = cluster_data['pred_emision_S_ug_m3_s'].mean()
                report += (
                    f"Cúmulo {i}: {len(cluster_data)} estaciones | "
                    f"Promedio PM2.5: {mean_pm25:.2f} ug/m3 | "
                    f"Tiempo medio (t): {mean_t:.2f} | "
                    f"Centroide PINN (x: {mean_x:.2f}, y: {mean_y:.2f}) | "
                    f"Emisión Promedio (S): {mean_S:.6f} ug/(m3*s)\n"
                )
                
            return report
        except Exception as e:
            return f"Error en clustering: {str(e)}"

# --- New Tool for Geospatial Valle Query ---
class GeospatialValleQueryInput(BaseModel):
    x_coordinate: float = Field(..., description="Coordenada x adimensionalizada del Valle (-1.0 a 1.0) (Este-Oeste).")
    y_coordinate: float = Field(..., description="Coordenada y adimensionalizada del Valle (-1.0 a 1.0) (Sur-Norte).")

class GeospatialValleQueryTool(BaseTool):
    name: str = "Geospatial Valle Query Tool"
    description: str = "Resuelve y cruza coordenadas adimensionales x (Este-Oeste) e y (Sur-Norte) con puntos geográficos reales del Valle de Aburrá."
    args_schema: Type[BaseModel] = GeospatialValleQueryInput
    
    def _run(self, x_coordinate: float, y_coordinate: float) -> str:
        x = x_coordinate
        y = y_coordinate
        if x < -1.0 or x > 1.0 or y < -1.0 or y > 1.0:
            return f"Coordenadas fuera del dominio [-1.0, 1.0]: x={x}, y={y}"
            
        # Determinar zona en Eje Y (Sur-Norte)
        if y < -0.3:
            zone_y = "región Sur (Sabaneta, La Estrella, Caldas o Itagüí)"
            desc_y = "el corredor industrial sur, caracterizado por fundidoras, textileras y alta congestión de carga pesada."
        elif -0.3 <= y < 0.3:
            zone_y = "región Central (Medellín y Envigado)"
            desc_y = "la zona de mayor densidad de tráfico vehicular urbano, automóviles particulares y el centro comercial de la ciudad."
        else:
            zone_y = "región Norte (Bello, Copacabana, Girardota o Barbosa)"
            desc_y = "el corredor norte, que concentra termoeléctricas, industrias manufactureras pesadas y flujos de salida del valle."
            
        # Determinar zona en Eje X (Este-Oeste)
        if x < -0.4:
            zone_x = "Ladera Occidental (San Javier, Belén, Robledo)"
            desc_x = "la vertiente residencial occidental, altamente vulnerable al estancamiento de partículas debido a la topografía empinada y vientos débiles."
        elif -0.4 <= x < 0.4:
            zone_x = "Fondo del Cañón (Cerca al río Medellín)"
            desc_x = "el eje plano y bajo del valle, donde se canalizan los vientos y se concentran las emisiones móviles por autopistas."
        else:
            zone_x = "Ladera Oriental (Manrique, Villa Hermosa, Poblado)"
            desc_x = "la vertiente oriental asimétrica, propensa a la recirculación de plumas contaminantes secundarias por corrientes anabáticas."
            
        return (
            f"La coordenada (x={x:.2f}, y={y:.2f}) se ubica geográficamente en la intersección de la {zone_x} y la {zone_y}. "
            f"Esta ubicación corresponde a {desc_x} en confluencia con {desc_y}"
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
            return f"[SUCCESS] Reporte forense standalone en LaTeX guardado exitosamente en: {report_path}"
        except Exception as e:
            return f"[ERROR] Error guardando el reporte en LaTeX: {str(e)}"

# --- New Tool for Physical Audit (verify_physics.jl) ---
class AuditPhysicsTool(BaseTool):
    name: str = "Audit Physics Tool"
    description: str = "Ejecuta el script de auditoría física verify_physics.jl en Julia para calcular el PVI (Physics Violation Index) y la divergencia máxima del viento sobre las predicciones de las últimas 48 horas."

    def _run(self) -> str:
        try:
            julia_path = r"C:\Users\arnod\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"
            if not os.path.exists(julia_path):
                julia_path = "julia"
                
            print("[MLOps] Ejecutando auditoría física en Julia...", flush=True)
            process = subprocess.run(
                [julia_path, "src/pinn/verify_physics.jl"],
                capture_output=True,
                text=True,
                check=True
            )
            
            pvi_file = "scratch/pvi_data.json"
            if os.path.exists(pvi_file):
                with open(pvi_file, "r") as f:
                    data = json.load(f)
                return (
                    f"=== RESULTADO DE AUDITORÍA FÍSICA ===\n"
                    f"- Physics Violation Index (PVI - Divergencia Media Absoluta del Viento): {data.get('pvi'):.6f}\n"
                    f"- Divergencia Máxima Absoluta: {data.get('max_div'):.6f}\n"
                    f"El script de Julia finalizó correctamente. Los datos de divergencia se exportaron a {pvi_file}."
                )
            else:
                return f"Auditoría finalizada, pero no se encontró scratch/pvi_data.json. Salida:\n{process.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Error ejecutando verify_physics.jl: {e.stderr}\nStdout:\n{e.stdout}"
        except Exception as e:
            return f"Error inesperado al auditar física: {str(e)}"
