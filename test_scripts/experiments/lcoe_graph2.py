from lcoe import calculate_lcoe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 1. Configuración de parámetros y rangos iniciales
department_name = "La Guajira"
fcap_init = 29.0
cap_min, cap_max = 10, 500  # Rango de Capacidad en MW
cap_init = 250.0            # Valor inicial del slider de capacidad

cap_vals = np.linspace(cap_min, cap_max, 30)
turb_vals = np.arange(1, 51, 2)
C, N = np.meshgrid(cap_vals, turb_vals)

# 2. Funciones de cálculo para la malla (3D) y para una capacidad específica (2D)


def compute_lcoe_matrix(department, C_mesh, N_mesh, fcap_val):
    Z = np.zeros(C_mesh.shape)
    for i in range(C_mesh.shape[0]):
        for j in range(C_mesh.shape[1]):
            lcoe_res = calculate_lcoe(
                department, C_mesh[i, j],
                int(N_mesh[i, j]),
                FCap=fcap_val)
            Z[i, j] = lcoe_res["result"]["LCOE"]
    return Z


def compute_lcoe_vector(department, cap_val, turb_array, fcap_val):
    z_line = np.zeros(len(turb_array))
    for k, n_t in enumerate(turb_array):
        lcoe_res = calculate_lcoe(department, cap_val, int(n_t), FCap=fcap_val)
        z_line[k] = lcoe_res["result"]["LCOE"]
    return z_line


# 3. Configuración de la figura y subplots
fig = plt.figure(figsize=(14, 7))

# Subplot 1: Superficie 3D
ax1 = fig.add_axes([0.05, 0.25, 0.42, 0.68], projection='3d')
# Subplot 2: Curva 2D
ax2 = fig.add_axes([0.55, 0.25, 0.40, 0.68])

# Subplots para los Sliders
ax_slider_fcap = fig.add_axes([0.15, 0.10, 0.70, 0.03])
ax_slider_cap = fig.add_axes([0.15, 0.03, 0.70, 0.03])

# 4. Cálculos e inicialización de gráficos
Z = compute_lcoe_matrix(department_name, C, N, fcap_init)
z_line = compute_lcoe_vector(department_name, cap_init, turb_vals, fcap_init)

# Dibuja la superficie 3D
surf = [ax1.plot_surface(C, N, Z, cmap='viridis',
                         edgecolor='none', alpha=0.85)]
ax1.set_xlabel('Capacity')
ax1.set_ylabel('N° Turbines')
ax1.set_zlabel('LCOE')
ax1.set_title(f'Superficie LCOE 3D ({department_name})')

# Destaca el corte actual de Capacidad en el gráfico 3D
cap_line_3d, = ax1.plot(
    [cap_init] * len(turb_vals), turb_vals, z_line,
    color='red', linewidth=3, label=f'Corte Cap = {cap_init:.0f}'
)

# Dibuja la curva 2D (N° Turbines vs LCOE)
line_2d, = ax2.plot(turb_vals, z_line, color='firebrick',
                    marker='o', linewidth=2)
ax2.set_xlabel('N° Turbines')
ax2.set_ylabel('LCOE')
ax2.set_title(f'LCOE vs N° Turbinas (Capacidad = {cap_init:.0f} MW)')
ax2.grid(True, linestyle='--', alpha=0.6)

# Bar de color para el gráfico 3D
cbar = fig.colorbar(surf[0], ax=ax1, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label('LCOE')

# 5. Definición de los Sliders
slider_fcap = Slider(
    ax=ax_slider_fcap,
    label='FCap (%)',
    valmin=0.0,
    valmax=100.0,
    valinit=fcap_init,
    valstep=1.0
)

slider_cap = Slider(
    ax=ax_slider_cap,
    label='Capacity (MW)',
    valmin=cap_min,
    valmax=cap_max,
    valinit=cap_init,
    valstep=1.0
)

# 6. Función de actualización cuando cambian los sliders


def update(val):
    current_fcap = slider_fcap.val
    current_cap = slider_cap.val

    # Recomputar matrices y vectores
    Z_new = compute_lcoe_matrix(department_name, C, N, current_fcap)
    z_line_new = compute_lcoe_vector(
        department_name, current_cap, turb_vals, current_fcap)

    # Actualizar superficie 3D
    surf[0].remove()
    surf[0] = ax1.plot_surface(
        C, N, Z_new, cmap='viridis', edgecolor='none', alpha=0.85)

    # Actualizar la línea de corte en el 3D
    cap_line_3d.set_data([current_cap] * len(turb_vals), turb_vals)
    cap_line_3d.set_3d_properties(z_line_new)

    # Actualizar la curva 2D
    line_2d.set_ydata(z_line_new)
    ax2.set_title(
        f'LCOE vs N° Turbinas (Capacidad = {current_cap:.0f} MW, FCap = {current_fcap:.0f}%)')
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw_idle()


# Conectar ambos sliders
slider_fcap.on_changed(update)
slider_cap.on_changed(update)

plt.show()
