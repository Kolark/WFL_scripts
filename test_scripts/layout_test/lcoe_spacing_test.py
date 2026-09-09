import geopandas as gpd
import pandas as pd
from wind_dataset.wind_dataset import WindDataset
from wake_models.wake_metrics import get_park_cf, get_park_loss
from layout_optimization import WindFarmGAOptimizer
from tqdm import tqdm
import input_data
from lcoe import calculate_lcoe
geometries = input_data.geometries
funcs = input_data.max_multivar_funcs
wind_data = input_data.wind_ds
wake_model = input_data.wake_model
import numpy as np


import numpy as np


def generar_cuadricula(centro_x, centro_y, n, tamano_lado):
    """Genera una cuadrícula de n x n centrada en (centro_x, centro_y)

    ocupando un tamaño total fijo (tamano_lado x tamano_lado).

    Parámetros:
    - centro_x, centro_y: Coordenadas del centro de la cuadrícula.
    - n: Número de puntos por lado (si n = 1, ubica un único punto en el
    centro).
    - tamano_lado: Ancho y alto total de la cuadrícula en metros (o unidades del
    mapa).

    Retorna:
    - lista_x: Array 1D con las coordenadas X.
    - lista_y: Array 1D con las coordenadas Y.
    """
    if n == 1:
        return np.array([centro_x]), np.array([centro_y])

    # Calcular el rango centrado en (0, 0)
    mitad = tamano_lado / 2.0
    offset = np.linspace(-mitad, mitad, n)

    # Trasladar al centro real
    rango_x = centro_x + offset
    rango_y = centro_y + offset

    # Generar la malla 2D y aplanar
    grid_x, grid_y = np.meshgrid(rango_x, rango_y)

    return grid_x.ravel(), grid_y.ravel()


# --- Ejemplo de uso ---
# cx, cy = 5151051.701693241, 2949564.2492115423  # Punto central
cx, cy = 5063388.524091, 2826067.503566  # Punto central
n = 10                # Cuadrícula de 5x5 puntos
spacing = 32.0        # Separación de 2 unidades entre puntos


xs, ys = generar_cuadricula(cx, cy, n, spacing)
wind_data.fill_cache(wind_data.ds.rio.bounds())
#==============================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- 1. Definición de Parámetros ---
n_vals = np.arange(1, 21, 1)  # n de 1 a 20
tamano_lado = 6.4 * 100  # Tamaño total fijo del parque en metros (ejemplo: 2 km)

lcoe_list = []
cf_list = []

# --- 2. Simulación variando solo n ---
for n in tqdm(n_vals, desc="Evaluando n"):
    # Generar cuadrícula de tamaño fijo
    xs, ys = generar_cuadricula(cx, cy, n, tamano_lado)

    # Obtener datos de viento y modelo de estela
    ws, wd = wind_data.get_wind_data_from_cache(xs, ys)
    _, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
        xs, ys, ws, wd, wind_data.ts
    )

    # Métricas energéticas
    generated_energy = aep_wake_turbines.sum()
    max_aep = (
        wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * len(xs)
    )

    cf = (generated_energy / max_aep) * 100.0  # Porcentaje (%)
    potencia_mW = (
        (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
        / 1000.0
    )

    # Cálculo de LCOE
    lcoe_res = calculate_lcoe("la guajira", potencia_mW, len(xs), FCap=cf)
    lcoe_val = lcoe_res["result"]["LCOE"]

    # Guardar resultados
    cf_list.append(cf)
    lcoe_list.append(lcoe_val)

# Convertir a arrays
N_arr = np.array(n_vals)
CF_arr = np.array(cf_list)
LCOE_arr = np.array(lcoe_list)

# --- 3. Visualización 2D (X: n, Y: LCOE, Color: CF) ---
fig, ax = plt.subplots(figsize=(10, 6))

# Línea de tendencia continua entre los puntos
ax.plot(N_arr, LCOE_arr, color="gray", linestyle="--", alpha=0.5, zorder=1)

# Puntos coloreados según el Capacity Factor (CF)
scatter = ax.scatter(
    N_arr,
    LCOE_arr,
    c=CF_arr,
    cmap="viridis",
    s=70,
    edgecolors="black",
    linewidths=0.7,
    zorder=2,
)

# Barra de colores para el CF
cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
cbar.set_label("Capacity Factor (CF %)", fontsize=11)

# Configuración de ejes y retícula
ax.set_xlabel("Número de Puntos por Lado ($n$)", fontsize=11)
ax.set_ylabel("LCOE ($/MWh)", fontsize=11)
ax.set_title(
    f"LCOE vs. Puntos por Lado ($n$) — Parque Fijo ({tamano_lado:.0f}m x {tamano_lado:.0f}m)",
    fontsize=12,
)

ax.set_xticks(N_arr)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
#==================================================

# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm

# # --- 1. Definición de Rangos de Parámetros ---
# n_vals = np.arange(1, 21, 1)  # n de 1 a 20
# mult_vals = np.arange(10, 150, 5)  # multiplicadores de 10 a 100 (de 5 en 5)
# base_tamano = 6.4

# # Contenedores para almacenar el barrido
# n_list = []
# mult_list = []
# cf_list = []
# lcoe_list = []

# # --- 2. Simulación de Barrido Bidimensional ---
# for mult in tqdm(mult_vals, desc="Evaluando Multiplicadores de Tamaño"):
#     tamano_lado = base_tamano * mult
#     for n in n_vals:
#         # Generar cuadrícula fija con el tamaño actual
#         xs, ys = generar_cuadricula(cx, cy, n, tamano_lado)

#         # Obtener datos de viento y modelo de estela
#         ws, wd = wind_data.get_wind_data_from_cache(xs, ys)
#         _, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
#             xs, ys, ws, wd, wind_data.ts
#         )

#         # Métricas energéticas
#         generated_energy = aep_wake_turbines.sum()
#         max_aep = (
#             wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * len(xs)
#         )

#         cf = (generated_energy / max_aep) * 100.0  # En porcentaje (%)
#         potencia_mW = (
#             (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
#             / 1000.0
#         )

#         # Cálculo de LCOE
#         lcoe_res = calculate_lcoe("la guajira", potencia_mW, len(xs), FCap=cf)
#         lcoe_val = lcoe_res["result"]["LCOE"]

#         # Guardar resultados
#         n_list.append(n)
#         mult_list.append(mult)
#         cf_list.append(cf)
#         lcoe_list.append(lcoe_val)

# # Convertir a arrays de NumPy
# N_arr = np.array(n_list)
# MULT_arr = np.array(mult_list)
# CF_arr = np.array(cf_list)
# LCOE_arr = np.array(lcoe_list)

# # Reestructurar a matrices 2D (Filas: multiplicadores, Columnas: n)
# shape_2d = (len(mult_vals), len(n_vals))
# N_grid = N_arr.reshape(shape_2d)
# TAMANO_grid = (MULT_arr * base_tamano).reshape(shape_2d)
# CF_grid = CF_arr.reshape(shape_2d)
# LCOE_grid = LCOE_arr.reshape(shape_2d)

# # --- 3. Visualización 3D (X: n, Y: Tamaño de Lado, Z: LCOE, Color: CF) ---
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection="3d")

# # Normalización de color mapeada al CF
# norm = plt.Normalize(CF_grid.min(), CF_grid.max())
# colors = plt.cm.viridis(norm(CF_grid))

# # Plot de la superficie 3D
# surf = ax.plot_surface(
#     N_grid,
#     TAMANO_grid,
#     LCOE_grid,
#     facecolors=colors,
#     rstride=1,
#     cstride=1,
#     linewidth=0.3,
#     antialiased=True,
#     shade=False,
# )

# # Barra de colores para la escala del CF
# mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
# mappable.set_array(CF_grid)
# cbar = fig.colorbar(
#     mappable, ax=ax, shrink=0.6, aspect=12, pad=0.1, label="Capacity Factor (%)"
# )

# # Configuración de etiquetas y visualización
# ax.set_xlabel("Puntos por Lado ($n$)", labelpad=10)
# ax.set_ylabel("Tamaño de Lado del Parque (m)", labelpad=10)
# ax.set_zlabel("LCOE ($/MWh)", labelpad=10)
# ax.set_title("Superficie 3D: LCOE vs. Layout (n, Tamaño de Lado) y Capacity Factor (Color)")

# # Ángulo de vista
# ax.view_init(elev=25, azim=135)

# plt.tight_layout()
# plt.show()
#============
# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm

# # --- 1. Definición de Rangos de Parámetros ---
# n_vals = np.arange(1, 21, 1)  # n de 1 a 20
# mult_vals = np.arange(10, 101, 5)  # multiplicadores de 10 a 100 (de 5 en 5)
# base_tamano = 6.4

# # Contenedores para almacenar el barrido
# n_list = []
# mult_list = []
# cf_list = []
# lcoe_list = []

# # --- 2. Simulación de Barrido Bidimensional ---
# for mult in tqdm(mult_vals, desc="Evaluando Multiplicadores de Tamaño"):
#     tamano_lado = base_tamano * mult
#     for n in n_vals:
#         # Generar cuadrícula fija con el tamaño actual
#         xs, ys = generar_cuadricula(cx, cy, n, tamano_lado)

#         # Obtener datos de viento y modelo de estela
#         ws, wd = wind_data.get_wind_data_from_cache(xs, ys)
#         _, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
#             xs, ys, ws, wd, wind_data.ts
#         )

#         # Métricas energéticas
#         generated_energy = aep_wake_turbines.sum()
#         max_aep = (
#             wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * len(xs)
#         )

#         cf = (generated_energy / max_aep) * 100.0  # En porcentaje (%)
#         potencia_mW = (
#             (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
#             / 1000.0
#         )

#         # Cálculo de LCOE
#         lcoe_res = calculate_lcoe("la guajira", potencia_mW, len(xs), FCap=cf)
#         lcoe_val = lcoe_res["result"]["LCOE"]

#         # Guardar resultados
#         n_list.append(n)
#         mult_list.append(mult)
#         cf_list.append(cf)
#         lcoe_list.append(lcoe_val)

# # Convertir a arrays de NumPy
# N_arr = np.array(n_list)
# MULT_arr = np.array(mult_list)
# CF_arr = np.array(cf_list)
# LCOE_arr = np.array(lcoe_list)

# # Reestructurar a matrices 2D (Filas: multiplicadores/tamaños, Columnas: n)
# shape_2d = (len(mult_vals), len(n_vals))
# N_grid = N_arr.reshape(shape_2d)
# TAMANO_grid = (MULT_arr * base_tamano).reshape(shape_2d)
# CF_grid = CF_arr.reshape(shape_2d)
# LCOE_grid = LCOE_arr.reshape(shape_2d)

# # --- 3. Cálculo de Mínimos por Fila (para cada tamaño de lado) ---
# min_indices_by_tamano = np.argmin(LCOE_grid, axis=1)  # Mínimo a lo largo de n (columnas)

# min_tamano = TAMANO_grid[np.arange(len(mult_vals)), min_indices_by_tamano]
# min_n = N_grid[np.arange(len(mult_vals)), min_indices_by_tamano]
# min_lcoe = LCOE_grid[np.arange(len(mult_vals)), min_indices_by_tamano]

# # --- 4. Visualización 3D ---
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection="3d")

# # Normalización de color mapeada al CF
# norm = plt.Normalize(CF_grid.min(), CF_grid.max())
# colors = plt.cm.viridis(norm(CF_grid))

# # Plot de la superficie 3D
# surf = ax.plot_surface(
#     N_grid,
#     TAMANO_grid,
#     LCOE_grid,
#     facecolors=colors,
#     rstride=1,
#     cstride=1,
#     linewidth=0.3,
#     antialiased=True,
#     shade=False,
#     alpha=0.85,
# )

# # Trazar la trayectoria del mínimo LCOE por cada tamaño de lado
# ax.plot(
#     min_n,
#     min_tamano,
#     min_lcoe,
#     color="red",
#     linewidth=3,
#     marker="o",
#     markersize=5,
#     label="Mínimo LCOE por Tamaño de Lado",
#     zorder=10,
# )

# # Barra de colores
# mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
# mappable.set_array(CF_grid)
# cbar = fig.colorbar(
#     mappable, ax=ax, shrink=0.6, aspect=12, pad=0.1, label="Capacity Factor (%)"
# )

# # Configuración de etiquetas y visualización
# ax.set_xlabel("Puntos por Lado ($n$)", labelpad=10)
# ax.set_ylabel("Tamaño de Lado del Parque (m)", labelpad=10)
# ax.set_zlabel("LCOE ($/MWh)", labelpad=10)
# ax.set_title(
#     "Superficie 3D: LCOE vs. Layout con Mínimo LCOE por Tamaño de Lado"
# )

# ax.legend(loc="upper left")
# ax.view_init(elev=25, azim=135)

# plt.tight_layout()
# plt.show()
#==
ws, wd = wind_data.get_wind_data_from_cache(xs, ys)
aep_ideal_turbines, aep_wake_turbines, veff_sum = (
    wake_model.calc_wake_on_turbines_detailed(xs, ys, ws, wd, wind_data.ts)
)
generated_energy = aep_wake_turbines.sum()
max_aep = wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * len(xs)


print("generated energy", generated_energy)
print("max aep", max_aep)
cf = generated_energy / max_aep
potencia_mW = (aep_wake_turbines/wind_data.ts[:, np.newaxis]).mean(axis=0).sum() / 1000
print("cf", cf)
lcoe = calculate_lcoe("la guajira", potencia_mW, len(xs), FCap=cf*100.0)
print(lcoe["result"]["LCOE"])
import matplotlib.pyplot as plt
plt.scatter(xs, ys)
plt.show()
