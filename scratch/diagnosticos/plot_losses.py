import matplotlib.pyplot as plt
import os
import re

log_path = r"C:\Users\Cristian\.gemini\antigravity\brain\6963f278-b3d1-4bb9-aff6-0333bd3e8373\.system_generated\tasks\task-390.log"
output_path = r"C:\Users\Cristian\.gemini\antigravity\brain\6963f278-b3d1-4bb9-aff6-0333bd3e8373\curva_perdida.png"

epochs = []
losses = []
val_losses = []

with open(log_path, 'r', encoding='utf-8') as f:
    for line in f:
        if "[EPOCH_LOG]" in line:
            # Format: [EPOCH_LOG] Epoch: 1 | Loss: 4712.916497151764 | Val Loss: 2.253978208371288
            parts = line.strip().split('|')
            if len(parts) >= 3:
                try:
                    epoch_str = parts[0].split(':')[-1].strip()
                    loss_str = parts[1].split(':')[-1].strip()
                    val_loss_str = parts[2].split(':')[-1].strip()
                    
                    epochs.append(int(epoch_str))
                    losses.append(float(loss_str))
                    val_losses.append(float(val_loss_str))
                except ValueError:
                    continue

# Plotting
plt.figure(figsize=(10, 6))

# Use linear scale as requested
plt.plot(epochs, losses, label="Pérdida Entrenamiento (Física + Datos)", color="#E53E3E", linewidth=2)
plt.plot(epochs, val_losses, label="Pérdida Validación (Solo Datos SIATA)", color="#3182CE", linewidth=2)

plt.xlabel("Época / Iteración", fontsize=12)
plt.ylabel("Pérdida (Escala Lineal)", fontsize=12)
plt.title("Curva de Convergencia PINN - Experimento de 10,000 Puntos", fontsize=14, fontweight='bold')
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

# Save
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Success: Plot saved to {output_path}")
