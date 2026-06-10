import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Crear directorio de reportes si no existe
os.makedirs("reporte", exist_ok=True)

def plot_convergence():
    csv_path = "scratch/historial_perdidas.txt"
    if not os.path.exists(csv_path):
        print(f"[WARNING] No se encontró {csv_path}. Omitiendo gráfico de convergencia.")
        return

    print("Graficando curvas de convergencia...")
    df = pd.read_csv(csv_path)
    
    if len(df) == 0:
        print("[WARNING] El archivo de pérdidas está vacío.")
        return

    # Usar un estilo elegante
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Paleta de colores armoniosa (tipo HSL)
    colors = {
        'total_loss': '#2c3e50',  # Oscuro
        'pde_loss': '#e74c3c',    # Rojo (física)
        'bc_loss': '#f39c12',     # Naranja (fronteras)
        'data_loss': '#2ecc71',   # Verde (datos entrenamiento)
        'val_loss': '#3498db'     # Azul (validación espacial)
    }

    # Graficar líneas
    ax.plot(df['epoch'], df['total_loss'], label='Pérdida Total (Objetivo)', color=colors['total_loss'], alpha=0.9, linewidth=1.8)
    ax.plot(df['epoch'], df['pde_loss'], label='Residuo PDE (Física)', color=colors['pde_loss'], alpha=0.8, linewidth=1.5)
    ax.plot(df['epoch'], df['bc_loss'], label='Residuo BC (Fronteras)', color=colors['bc_loss'], alpha=0.7, linewidth=1.2, linestyle='--')
    ax.plot(df['epoch'], df['data_loss'], label='Pérdida de Datos (Train)', color=colors['data_loss'], alpha=0.8, linewidth=1.5)
    
    if 'val_loss' in df.columns:
        ax.plot(df['epoch'], df['val_loss'], label='Pérdida de Validación (LOSO)', color=colors['val_loss'], alpha=0.9, linewidth=1.8)

    # Identificar cambio de etapa (Adam a L-BFGS)
    if 'stage' in df.columns:
        lbfgs_epochs = df[df['stage'] == 'L-BFGS']['epoch']
        if len(lbfgs_epochs) > 0:
            split_epoch = lbfgs_epochs.iloc[0]
            ax.axvline(x=split_epoch, color='#7f8c8d', linestyle=':', linewidth=1.5)
            ax.text(split_epoch, ax.get_ylim()[0] * 1.5, ' Ajuste L-BFGS', rotation=90, color='#7f8c8d', fontsize=9, verticalalignment='bottom')

    # Configuración de escala logarítmica
    ax.set_yscale('log')
    ax.set_xlabel('Época de Entrenamiento', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Valor de Pérdida (Escala Log)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Convergencia del Modelo PINN Termodinámico 3D', fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e8f0', framealpha=0.9)
    ax.tick_params(colors='#4a5568')
    ax.grid(True, which="both", ls="-", color='#e2e8f0', alpha=0.5)

    plt.tight_layout()
    output_img = "reporte/curvas_convergencia.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"[OK] Gráfico de convergencia guardado en {output_img}")

def plot_pvi():
    json_path = "scratch/pvi_data.json"
    if not os.path.exists(json_path):
        print(f"[WARNING] No se encontró {json_path}. Omitiendo gráfico de divergencia.")
        return

    print("Graficando mapas de divergencia del viento (PVI)...")
    with open(json_path, "r") as f:
        data = json.load(f)

    nx, ny, nz = data["nx"], data["ny"], data["nz"]
    x = np.array(data["x"])
    y = np.array(data["y"])
    z = np.array(data["z"])
    pvi = data["pvi"]
    
    # Reconstruir array 3D
    div = np.array(data["divergence"]).reshape((nx, ny, nz))
    
    # Determinar cortes a graficar (z = 0.1, 0.5, 0.9)
    # Buscamos los índices en el vector z que estén más cerca de estos valores
    idx_bottom = np.argmin(np.abs(z - 0.1))
    idx_mid = np.argmin(np.abs(z - 0.5))
    idx_top = np.argmin(np.abs(z - 0.9))
    
    slices = [
        (idx_bottom, f"Fondo del Valle (z = {z[idx_bottom]:.2f})"),
        (idx_mid, f"Altura Media (z = {z[idx_mid]:.2f})"),
        (idx_top, f"Inversión Térmica / Techo (z = {z[idx_top]:.2f})")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    
    # Encontrar límites simétricos para la divergencia para centrar la escala de colores en 0
    max_val = max(np.max(np.abs(div[:, :, idx_bottom])), 
                  np.max(np.abs(div[:, :, idx_mid])), 
                  np.max(np.abs(div[:, :, idx_top])))
    # Evitar escala de cero
    max_val = max(max_val, 1e-4)

    for i, (idx, title) in enumerate(slices):
        ax = axes[i]
        # Transponer para graficar x en eje horizontal e y en eje vertical
        slice_data = div[:, :, idx].T
        
        im = ax.imshow(
            slice_data, 
            extent=[-1.0, 1.0, -1.0, 1.0], 
            origin='lower', 
            cmap='RdBu_r', 
            vmin=-max_val, 
            vmax=max_val
        )
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Ancho del Valle (x)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Largo del Valle (y)', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Añadir barra de colores común a la derecha
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Divergencia del Viento ($\\nabla \\cdot \\vec{v}$)', fontsize=11, fontweight='bold', labelpad=15)
    
    fig.suptitle(f"Auditoría Física: Campo de Divergencia del Viento 3D\nPhysics Violation Index (PVI) = {pvi:.6f} | Masa conservada si PVI ≈ 0", 
                 fontsize=14, fontweight='bold', y=0.98)

    output_img = "reporte/mapa_divergencia_pvi.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico de PVI guardado en {output_img}")

if __name__ == "__main__":
    plot_convergence()
    plot_pvi()
