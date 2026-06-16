import os
import json
import subprocess
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

class BayesianOptimizer:
    """
    Ligero y robusto optimizador bayesiano para calibración de hiperparámetros en PINNs.
    Utiliza Gaussian Process Regressor de scikit-learn con Expected Improvement (EI).
    """
    def __init__(self, n_trials=5, xi=0.01):
        self.n_trials = n_trials
        self.xi = xi
        # Rango de parámetros:
        # lr_log ∈ [-4.0, -1.0] -> 1e-4 a 1e-1
        # epochs ∈ [100.0, 300.0]
        self.bounds = np.array([
            [-4.0, -1.0],  # log10(lr)
            [100.0, 300.0]  # epochs
        ])
        
        # Historial de muestras
        self.X_sample = []
        self.y_sample = []
        
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )
        
    def objective(self, lr_log, epochs):
        """Evalúa la función objetivo corriendo el script de Julia."""
        lr = 10**float(lr_log)
        epochs_int = int(np.round(epochs))
        
        print(f"\n--- Evaluando Trial: LR={lr:.5f} (10^{lr_log:.2f}), Epochs={epochs_int} ---")
        
        # Escribir configuración temporal (fijando 300 it. de L-BFGS para excelente precisión)
        config = {"epochs": epochs_int, "learning_rate": lr, "lbfgs_iters": 300}
        with open("models/pinn_config.json", "w") as f:
            json.dump(config, f)
            
        try:
            julia_path = "julia"
            result = subprocess.run(
                [julia_path, "--project=.", "src/pinn/train_interpolative.jl"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=600
            )
            
            if result.returncode != 0:
                print(f"⚠️ Error en ejecución de Julia (código {result.returncode}):")
                print(result.stderr)
                return 1e6 # Retornar penalización alta en caso de error
                
            # Leer pérdida resultante
            if os.path.exists("models/pesos_pinn_boussinesq.json"):
                with open("models/pesos_pinn_boussinesq.json", "r") as f:
                    data = json.load(f)
                    loss = float(data.get("loss", 1e6))
                    print(f"✅ Trial Completado con Éxito. Loss Resultante: {loss:.5f}")
                    return loss
            else:
                print("⚠️ models/pesos_pinn_boussinesq.json no fue generado.")
                return 1e6
        except subprocess.TimeoutExpired:
            print("⚠️ Timeout excedido en ejecución de Julia.")
            return 1e6
        except Exception as e:
            print(f"⚠️ Excepción durante el trial: {e}")
            return 1e6

    def expected_improvement(self, X):
        """Calcula el Expected Improvement para un conjunto de candidatos X."""
        X = np.atleast_2d(X)
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        
        # Valor óptimo actual (mínimo de y)
        current_best = np.min(self.y_sample)
        
        # Z score para minimización: buscamos reducir mu por debajo del current_best
        improvement = current_best - mu - self.xi
        Z = improvement / sigma
        
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        # EI es 0 si no hay mejora esperada
        ei[sigma <= 0.0] = 0.0
        return ei

    def propose_next(self):
        """Propone el siguiente punto a evaluar maximizando la adquisición EI."""
        # Generar cuadrícula densa aleatoria de candidatos en el espacio de búsqueda
        candidates = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(2000, 2)
        )
        
        ei = self.expected_improvement(candidates)
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def run(self):
        print("==== Iniciando Optimización Bayesiana de la iPINN ====")
        
        # 1. Muestreo inicial aleatorio (2 puntos para inicializar el GP)
        np.random.seed(42)
        initial_points = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(2, 2)
        )
        
        for pt in initial_points:
            loss = self.objective(pt[0], pt[1])
            self.X_sample.append(pt)
            self.y_sample.append(loss)
            
        # 2. Bucle de optimización bayesiana activa
        for trial in range(self.n_trials - 2):
            # Ajustar proceso gaussiano con la data actual
            X_train = np.array(self.X_sample)
            y_train = np.array(self.y_sample)
            
            self.gp.fit(X_train, y_train)
            
            # Proponer siguiente punto
            next_pt = self.propose_next()
            
            loss = self.objective(next_pt[0], next_pt[1])
            
            self.X_sample.append(next_pt)
            self.y_sample.append(loss)
            
        # 3. Reportar los mejores resultados encontrados
        X_all = np.array(self.X_sample)
        y_all = np.array(self.y_sample)
        best_idx = np.argmin(y_all)
        best_lr_log, best_epochs = X_all[best_idx]
        best_lr = 10**best_lr_log
        best_epochs_int = int(np.round(best_epochs))
        
        print("\n==========================================")
        print("🎉 ¡Optimización Bayesiana Finalizada! 🎉")
        print(f"Mejor Loss: {y_all[best_idx]:.5f}")
        print(f"Mejor Learning Rate: {best_lr:.6f} (10^{best_lr_log:.2f})")
        print(f"Mejor Epochs: {best_epochs_int}")
        print("==========================================\n")
        
        # Escribir la mejor configuración final para el entrenamiento definitivo
        best_config = {"epochs": best_epochs_int, "learning_rate": best_lr, "lbfgs_iters": 300}
        with open("models/pinn_config.json", "w") as f:
            json.dump(best_config, f)
            
        return best_config

if __name__ == "__main__":
    # Corrida rápida de prueba si se ejecuta directamente
    # n_trials=3 para verificación rápida en el entorno de desarrollo
    opt = BayesianOptimizer(n_trials=3)
    # Por seguridad, si no queremos correr el script completo, podemos importar o instanciar.
    # Pero si se corre por terminal, se optimiza.
    opt.run()
