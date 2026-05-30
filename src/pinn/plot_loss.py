import json
import os
import matplotlib.pyplot as plt

def main():
    json_path = "historial_perdida.json"
    output_path = "reporte/curva_perdida.png"
    
    if not os.path.exists(json_path):
        print(f"Error: No se encontró {json_path}")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        history = json.load(f)
        
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    phases = history.get("phases", [])
    
    if not epochs:
        print("Error: El historial de pérdidas está vacío.")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar curva de entrenamiento (SIATA Train)
    ax.plot(epochs, train_loss, label="Pérdida de Entrenamiento (SIATA Train)", color="#2B6CB0", linewidth=2.5, marker='o', markersize=4)
    
    # Encontrar la transición entre fases (Adam a L-BFGS)
    lbfgs_start = None
    for i, phase in enumerate(phases):
        if phase == "L-BFGS":
            lbfgs_start = epochs[i]
            break
            
    # Marcar fases con color de fondo y líneas divisorias
    if lbfgs_start is not None:
        ax.axvline(x=lbfgs_start - 0.5, color="#718096", linestyle=":", linewidth=2, alpha=0.8)
        # Sombreado de fondos de las fases
        ax.axvspan(epochs[0] - 0.5, lbfgs_start - 0.5, alpha=0.1, color="#3182CE", label="Fase 1: Adam")
        ax.axvspan(lbfgs_start - 0.5, epochs[-1] + 0.5, alpha=0.1, color="#E53E3E", label="Fase 2: L-BFGS")
        
    ax.set_xlabel("Iteración / Época Total", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel("Pérdida", fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title("Curva de Convergencia PINN Termodinámica", fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
    
    # Configurar marcas del eje X automáticas de enteros para evitar superposición
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    
    ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=10, loc="best")
    ax.grid(True, which="major", linestyle="--", alpha=0.5)
    
    # Añadir caja de información con las métricas finales
    info_text = f"Pérdida Train Final: {train_loss[-1]:.2f}"
    ax.text(0.7, 0.7, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7FAFC', edgecolor='#E2E8F0', alpha=0.95))
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"¡Gráfica guardada exitosamente en {output_path}!")

if __name__ == "__main__":
    main()
