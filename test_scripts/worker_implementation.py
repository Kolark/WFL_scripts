import geopandas as gpd
import pandas as pd
from wind_dataset.wind_dataset import WindDataset
from layout_optimization.gp_optimize import GridGPOptimizer
from layout_optimization.fitness_funcs import  make_multivar_fitness, make_mvfitness_max_turbines
from wake_models import WakeModel, obtener_aerogenerador
from monitor import Monitor
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from wake_models.wake_metrics import get_park_cf, get_park_loss
# ========WINDDATA===========
# wind_ds = WindDataset(
#     nc_path="data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc",
#     easting="easting",
#     northing="northing",
#     ws="WS",
# )


wind_ds = WindDataset(
    nc_path="data/netcdf/merra_processed.nc",
    easting="Easting",
    northing="Northing",
    ws="WS",
    x_dim="Easting",
    y_dim="Northing",
    time_dim="time",
)


# ========GEOMETRY===========

shp_path = "data/eolico_10_m/Zona_F_Eolico_10m.shp"

gdf = gpd.read_file(shp_path)
if gdf.crs != wind_ds.crs:
    gdf = gdf.to_crs(wind_ds.crs)
geometry = gdf.geometry.iloc[33934]
gdf = gdf.set_crs(epsg=9377)
# ========TURBINE===========
modelo = "Aeolos H-5kW"
# modelo = "Vestas V100/2000"
aerogenerador = obtener_aerogenerador(modelo)
# ========PARAMS============
wake_k = 0.05
thrust_ct = 0.8
max_wake_distance_D = 20.0
# ========WAKEMODEL=========
wake_model = WakeModel(
    aerogenerator=aerogenerador,
    wake_k=wake_k,
    thrust_ct=thrust_ct,
    max_wake_distance_D=max_wake_distance_D,
)

# fitness_fn = make_mvfitness_max_turbines(
#     aep_w=0.0, park_cf_w=10.0, parkloss_w=10.0, max_turbines=18
# )
fitness_fn = make_multivar_fitness(aep_w=0.0, park_cf_w=1.0, parkloss_w=1.0)
monitor = Monitor()

gp_optimizer = GridGPOptimizer(
    wind_data=wind_ds,
    wake_model=wake_model,
    fitness_func=fitness_fn,
    monitor=monitor
)

best_result = gp_optimizer.optimize(
    geometry=geometry,
    n_calls=100,
    n_initial_points=20,
    acq_func="EI",
    random_state=42,
    n_points=1000,
)


def add_scatter_history_slider(ax, x_history, y_history):
    fig = ax.get_figure()
    fig.subplots_adjust(bottom=0.25)

    initial_index = 0

    # Combinar X e Y en una matriz de Nx2 para el scatter inicial
    initial_offsets = np.column_stack(
        (x_history[initial_index], y_history[initial_index])
    )
    scat = ax.scatter(
        initial_offsets[:, 0], initial_offsets[:, 1], color="blue", alpha=0.7
    )

    all_x = np.concatenate(x_history)
    all_y = np.concatenate(y_history)
    ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
    ax.set_ylim(all_y.min() - 1, all_y.max() + 1)
    ax.set_title(f"Optimización - Iteración Mejorada: {initial_index}")
    ax.grid(True)

    slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])

    history_slider = Slider(
        ax=slider_ax,
        label="Iteración ",
        valmin=0,
        valmax=len(x_history) - 1,
        valinit=initial_index,
        valfmt="%d",
        valstep=1,
    )

    def update(val):
        idx = int(history_slider.val)

        # Actualizar las posiciones del scatter plot
        offsets = np.column_stack((x_history[idx], y_history[idx]))
        scat.set_offsets(offsets)

        ax.set_title(f"Optimización - Iteración Mejorada: {idx}")
        fig.canvas.draw_idle()

    history_slider.on_changed(update)
    return history_slider


fig = plt.figure()
ax = fig.add_subplot(121)
plot_polygon(polygon=geometry, ax=ax, add_points=False)
# Add a single subplot that fills the canvas

max_aep = best_result.p_nominal_kW * best_result.n_turbinas * best_result.n_horas
total_aep_ideal = best_result.aep_ideal.sum()
total_aep_wake = best_result.aep_wake.sum()
cf = get_park_cf(total_aep_wake, max_aep)
pl = get_park_loss(total_aep_ideal, total_aep_wake)

print("n_horas", best_result.n_horas)
print("max_aep", max_aep)
print("total_aep_ideal", total_aep_ideal)
print("total_aep_wake", total_aep_wake)
print("cf", cf)
print("pl", pl)

x_data, y_data = monitor.get_best_xy()
slider_activo = add_scatter_history_slider(ax, x_data, y_data)
ax2 = fig.add_subplot(122)
ax2.plot(monitor.fitness_history)
plt.show()
