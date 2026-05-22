import sys
import os
import json
import datetime

# Asegurar que el directorio raíz del proyecto está en el path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.tools import ExecuteJuliaPINNTool

def main():
    print("=== Iniciando prueba de Memoria de Hiperparámetros (Meta-Learning) ===")
    
    memory_file = "mlops_memory.json"
    
    # 1. Inyectar un historial simulado con un fallo (NaN) previo usando LR = 0.8
    mock_history = [
        {
            "timestamp": datetime.datetime.now().isoformat(),
            "epochs": 5,
            "requested_lr": 0.8,
            "started_lr": 0.8,
            "final_lr": 0.8,
            "status": "nan",  # Marcado como fallo crítico de tipo NaN
            "final_loss": None
        }
    ]
    
    try:
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(mock_history, f, indent=4)
        print("Inyectado fallo simulado en la memoria (LR=0.8, status=nan).")
    except Exception as e:
        print(f"Error escribiendo mock history: {e}")
        return

    tool = ExecuteJuliaPINNTool()

    # 2. Corrida de Prueba: Volvemos a pedir LR = 0.8
    # El sistema debe ver en la memoria que >=0.8 causó NaN, e iniciar directamente con un LR seguro de 0.01.
    print("\n--- CORRIDA DE PRUEBA: Solicitando de nuevo LR = 0.8 (Debería aplicar ajuste preventivo) ---")
    try:
        report = tool._run(epochs=5, learning_rate=0.8)
        print("\n=== REPORTE GENERADO ===")
        print(report)
    except Exception as e:
        print(f"Error en Corrida de Prueba: {e}")

    # 3. Mostrar memoria consolidada final
    if os.path.exists(memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"\n[Verificación] Contenido final de {memory_file}:")
            print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
