import re
import matplotlib.pyplot as plt

def parse_log():
    log_path = r"C:\Users\Cristian\.gemini\antigravity\brain\6963f278-b3d1-4bb9-aff6-0333bd3e8373\.system_generated\tasks\task-390.log"
    epochs = []
    losses = []
    val_losses = []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "[EPOCH_LOG]" in line:
                    match = re.search(r"Epoch:\s*(\d+)\s*\|\s*Loss:\s*([\d\.]+)\s*\|\s*Val Loss:\s*([\d\.]+)", line)
                    if match:
                        epoch = int(match.group(1))
                        loss = float(match.group(2))
                        val_loss = float(match.group(3))
                        
                        epochs.append(epoch)
                        losses.append(loss)
                        val_losses.append(val_loss)
    except Exception as e:
        print(f"Error abriendo o leyendo el log: {e}")
        return None

    return epochs, losses, val_losses

def generate_reports():
    parsed = parse_log()
    if not parsed or not parsed[0]:
        print("No se pudieron parsear datos del log.")
        return

    epochs, losses, val_losses = parsed

    # Generar la grafica en escala logaritmica
    output_image = r"C:\Users\Cristian\.gemini\antigravity\brain\6963f278-b3d1-4bb9-aff6-0333bd3e8373\curva_perdida_cancelada_log.png"
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, label="Loss Entrenamiento (Fisica + Datos)", color="#E53E3E", linewidth=1.5)
        plt.plot(epochs, val_losses, label="Loss Validacion (Sensores)", color="#3182CE", linewidth=1.5)
        
        plt.yscale('log') # Aplicar escala logaritmica en el eje Y
        plt.xlabel("Epoca (Epoch)")
        plt.ylabel("Perdida (Loss) - Escala Logaritmica")
        plt.title("Curva de Perdida del Entrenamiento Cancelado (Escala Logaritmica - Epoca 4283)")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_image, dpi=300)
        plt.close()
        print(f"Grafica logaritmica guardada exitosamente en: {output_image}")
    except Exception as e:
        print(f"Error generando la grafica logaritmica: {e}")

if __name__ == "__main__":
    generate_reports()
