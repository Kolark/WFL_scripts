import geopandas as gpd
import pandas as pd
from wind_dataset.wind_dataset import WindDataset
from wake_models.wake_metrics import get_park_cf, get_park_loss
from layout_optimization import WindFarmGAOptimizer
from tqdm import tqdm
import input_data
from lcoe import calculate_lcoe
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

geometries = input_data.geometries
funcs = input_data.max_multivar_funcs
wind_data = input_data.wind_ds
wake_model = input_data.wake_model
import numpy as np
from shapely.plotting import plot_polygon
wind_data.fill_cache(wind_data.ds.rio.bounds())
# ==============================================================================

from layout_optimization import GAConfig, GeneticAlgorithm
from layout_optimization.ga.raster_utils import get_raster_pixel_coords

gdf = gpd.read_file("/home/felipe/Desktop/Trabajo/WFL_scripts/data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)
# e_10m_36527
# geometry = gdf.geometry.iloc[36526]
geometry = gdf.geometry.iloc[36662]

candidate_x, candidate_y = get_raster_pixel_coords(
    geometry, pixel_size=wake_model.aerogenerator.rotor_diameter_m
)


def fitness_func(gene):
    x_turbines = candidate_x[gene]
    y_turbines = candidate_y[gene]

    ws, wd = wind_data.get_wind_data_from_cache(x_turbines, y_turbines)
    _, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
        x_turbines, y_turbines, ws, wd, wind_data.ts
    )

    # Métricas energéticas
    generated_energy = aep_wake_turbines.sum()
    max_aep = (
        wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * len(x_turbines)
    )

    cf = (generated_energy / max_aep) * 100.0  # En porcentaje (%)
    potencia_mW = (
        (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
        / 1000.0
    )

    # Cálculo de LCOE
    lcoe_res = calculate_lcoe("la guajira", potencia_mW, len(x_turbines), FCap=cf)
    lcoe_val = lcoe_res["result"]["LCOE"]

    return -lcoe_val

def init_bool_with_prob(sol_per_pop: int, num_genes: int) -> np.ndarray:

    p_true = 0.05
    p_false = 1.0 - p_true

    return np.random.choice(
        a=[False, True], size=(sol_per_pop, num_genes), p=[p_false, p_true]
    )

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.widgets import Slider


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def plot_interactive_history(ax, history, radius=0.5, slider_ax=None, **scatter_kwargs):
    fig = ax.get_figure()

    if slider_ax is None:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height - 0.1])
        slider_ax = fig.add_axes([pos.x0, pos.y0, pos.width, 0.03])

    # Convertir el radio a tamaño de punto (s) para ax.scatter
    # scatter usa el área en puntos al cuadrado (points^2)
    radius_in_points = ax.transData.transform((radius, 0))[0] - ax.transData.transform((0, 0))[0]
    s_size = (2 * radius_in_points) ** 2

    # Configuración de estilos por defecto para aros vacíos
    kwargs = {
        "s": s_size,
        "facecolors": "none",
        "edgecolors": "red",
        "linewidths": 1.5
    }
    kwargs.update(scatter_kwargs)

    container = {"scat": None}

    def render_frame(idx):
        if container["scat"] is not None:
            container["scat"].remove()

        x_curr, y_curr = history[idx]
        container["scat"] = ax.scatter(x_curr, y_curr, **kwargs)

    render_frame(0)

    slider = Slider(
        ax=slider_ax,
        label="Paso de Tiempo",
        valmin=0,
        valmax=len(history) - 1,
        valinit=0,
        valstep=1,
    )

    def update(val):
        render_frame(int(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    return container, slider


np.random.seed(42)
config = GAConfig(
    num_generations=1000,
    sol_per_pop=10,
    num_parents_mating=5,
    K_tournament=5,
    mutation_percent_genes=10,
    keep_elitism=5,
    x=candidate_x,
    y=candidate_y,
)
ga = GeneticAlgorithm(
    config,
    init_func=init_bool_with_prob,
    fitness_func=fitness_func,
    num_genes=len(candidate_x),
)


print(len(candidate_x))
print(np.count_nonzero(ga.population[0]))
best_solution, best_fitness = ga.run()

import matplotlib.pyplot as plt
fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

ax2.plot(ga.best_fitness_history)
ax3.plot(ga.best_amount_history)

print(best_solution)

final_x = candidate_x[best_solution]
final_y = candidate_y[best_solution]
plot_polygon(geometry, ax=ax, add_points=False)

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

# Your existing scatter plot
# ax.scatter(final_x, final_y)


# 4. Ensure equal aspect ratio so circles do not look stretched like ellipses
ax.set_aspect("equal", adjustable="datalim")
ws, wd = wind_data.get_wind_data_from_cache(final_x, final_y)
_, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
    final_x, final_y, ws, wd, wind_data.ts
)

# Métricas energéticas
generated_energy = aep_wake_turbines.sum()
max_aep = (
    wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * len(final_x)
)

cf = (generated_energy / max_aep) * 100.0  # En porcentaje (%)
potencia_mW = (
    (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
    / 1000.0
)

# Cálculo de LCOE
lcoe_res = calculate_lcoe("la guajira", potencia_mW, len(final_x), FCap=cf)
lcoe_val = lcoe_res["result"]["LCOE"]

print("CF final", cf)
print("LCOE final", lcoe_val)

scat, slider = plot_interactive_history(
    ax, list(zip(ga.best_x, ga.best_y)), radius=6.4 * 5, edgecolors="blue"
)

plt.show()


###POR HACER.

## COMBINAR EL CRITERIO DE QUITAR PUNTOS MUY CERCANOS
## CON LA DE AGREGAR CROMOSOMAS CON MAS PROBABILIDAD EN PUNTOS MAS ALEJADOS.
## HACER UN CROSSOVER ESPACIAL.
## Averiguar por que el capacity factor me daba tan alto.