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
                process = subprocess.Popen(
                    ["julia", "src/pinn/train_interpolative.jl"],
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
        fail_report = f"[FAIL] Todos los {max_retries} intentos de entrenamiento fallaron.\n" + "\n".join(execution_summary)
        return fail_report

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
            with open("datos_siata_temporal.json", "r") as f:
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
