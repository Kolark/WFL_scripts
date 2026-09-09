from lcoe import calculate_lcoe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Parámetros iniciales
department_name = "antioquia"
fcap_init = 29.0

# 2. Definir los rangos para la malla (Mapeo de Capacidad vs N° Turbinas)
cap_vals = np.linspace(10, 1000, 50)      # Ejemplo: Capacidad de 10 a 500 MW
turb_vals = np.arange(1, 101, 1)          # Ejemplo: 1 a 50 turbinas

C, N = np.meshgrid(cap_vals, turb_vals)

# Función auxiliar para evaluar la malla
def compute_lcoe_matrix(department, C_mesh, N_mesh, fcap_val):
    Z = np.zeros(C_mesh.shape)
    for i in range(C_mesh.shape[0]):
        for j in range(C_mesh.shape[1]):
            lcoe_res = calculate_lcoe(department, C_mesh[i, j], int(N_mesh[i, j]), FCap=fcap_val)
            Z[i, j] = lcoe_res["result"]["LCOE"]
    return Z

# 3. Configurar la figura de Matplotlib y subplots
fig = plt.figure(figsize=(10, 8))

# Subplot principal en 3D (dejamos espacio abajo para el slider)
ax = fig.add_axes([0.1, 0.2, 0.8, 0.75], projection='3d')

# Subplot para el Slider
ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])

# 4. Cálculo inicial y primer dibujado
Z = compute_lcoe_matrix(department_name, C, N, fcap_init)

surf = [ax.plot_surface(C, N, Z, cmap='viridis', edgecolor='none')]
ax.set_xlabel('Capacity')
ax.set_ylabel('N° Turbines')
ax.set_zlabel('LCOE')
ax.set_title(f'LCOE Surface - {department_name}')

# Colorbar para la escala de LCOE
cbar = fig.colorbar(surf[0], ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('LCOE')

# 5. Crear el Slider de FCap (de 0 a 100, paso 1)
slider_fcap = Slider(
    ax=ax_slider,
    label='FCap (%)',
    valmin=0.0,
    valmax=100.0,
    valinit=fcap_init,
    valstep=1.0
)

# 6. Función de actualización cuando el slider cambia
def update(val):
    fcap_current = slider_fcap.val
    
    # Calcular nuevos valores
    Z_new = compute_lcoe_matrix(department_name, C, N, fcap_current)
    
    # Limpiar superficie anterior y redibujar
    surf[0].remove()
    surf[0] = ax.plot_surface(C, N, Z_new, cmap='viridis', edgecolor='none')
    
    # Reajustar el rango del eje Z dinámicamente si los valores cambian dramáticamente
    ax.set_zlim(Z_new.min(), Z_new.max())
    
    fig.canvas.draw_idle()

# Conectar el evento del slider
slider_fcap.on_changed(update)

plt.show()