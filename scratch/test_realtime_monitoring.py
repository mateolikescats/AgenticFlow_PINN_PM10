import sys
import os

# Asegurar que el directorio raíz del proyecto está en el path de Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.tools import ExecuteJuliaPINNTool

def main():
    print("=== Iniciando prueba de ExecuteJuliaPINNTool con Monitoreo en Tiempo Real ===")
    tool = ExecuteJuliaPINNTool()
    
    # Probamos con 5 epochs y learning rate de 0.01 para una prueba rápida
    try:
        report = tool._run(epochs=50, learning_rate=0.01)
        print("\n=== REPORTE GENERADO POR LA TOOL ===")
        print(report)
        print("====================================")
    except Exception as e:
        print(f"Ocurrió un error en la ejecución de la prueba: {e}")

if __name__ == "__main__":
    main()
