from layout_optimization.ga.crossover import two_point_crossover, checkerboard_crossover
from layout_optimization.ga.selection import tournament_selection
from layout_optimization.ga.mutation import mutacion_experimental
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from layout_optimization.ga.raster_utils import get_raster_pixel_coords
from layout_optimization import GAConfig, GeneticAlgorithm
import geopandas as gpd
from wind_dataset.wind_dataset import WindDataset
import input_data
from lcoe import calculate_lcoe
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

geometries = input_data.geometries
funcs = input_data.max_multivar_funcs

wake_model = input_data.wake_model

# ==============================================================================


wind_data = WindDataset(
    nc_path="/home/felipe/Desktop/Trabajo/Data/Newdataq/wind_data_10m(1).nc",
    easting="Easting",
    northing="Northing",
    ws="WS",
    time_dim="time",
)
wind_data.fill_cache(wind_data.ds.rio.bounds())


gdf = gpd.read_file(
    "/home/felipe/Desktop/Trabajo/WFL_scripts/data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)

geo_str = "mg"
geometry = input_data.geometries[geo_str]
geom_department = input_data.geom_departamentos[geo_str]

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
        wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() *
        len(x_turbines))

    cf = (generated_energy / max_aep) * 100.0  # En porcentaje (%)
    potencia_mW = (
        (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
        / 1000.0
    )

    # Cálculo de LCOE
    lcoe_res = calculate_lcoe(
        geom_department, potencia_mW, len(x_turbines),
        FCap=cf)
    lcoe_val = lcoe_res["result"]["LCOE"]

    return -lcoe_val


def init_bool_with_prob(sol_per_pop: int, num_genes: int) -> np.ndarray:

    p_true = 0.01
    p_false = 1.0 - p_true

    return np.random.choice(
        a=[False, True], size=(sol_per_pop, num_genes), p=[p_false, p_true]
    )


def plot_interactive_history(
    ax,
    history,
    inner_radius=0.25,
    outer_radius=0.5,
    slider_ax=None,
    center_kwargs=None,
    inner_kwargs=None,
    outer_kwargs=None,
):
    fig = ax.get_figure()

    if slider_ax is None:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height - 0.1])
        slider_ax = fig.add_axes([pos.x0, pos.y0, pos.width, 0.03])

    # Default styles
    style_center = {
        "s": 20,
        "c": "black",
        "marker": "o",
        "zorder": 3,
    }
    style_inner = {
        "facecolor": "none",
        "edgecolor": "blue",
        "linewidth": 1.2,
        "linestyle": "--",
        "zorder": 2,
    }
    style_outer = {
        "facecolor": "none",
        "edgecolor": "red",
        "linewidth": 1.5,
        "zorder": 1,
        "alpha": 0.1
    }

    if center_kwargs:
        style_center.update(center_kwargs)
    if inner_kwargs:
        style_inner.update(inner_kwargs)
    if outer_kwargs:
        style_outer.update(outer_kwargs)

    container = {"patches_outer": None,
                 "patches_inner": None, "scat_center": None}

    def render_frame(idx):
        # Remove previous frame elements
        if container["patches_outer"] is not None:
            container["patches_outer"].remove()
        if container["patches_inner"] is not None:
            container["patches_inner"].remove()
        if container["scat_center"] is not None:
            container["scat_center"].remove()

        x_curr, y_curr = history[idx]

        # Create Circle patches locked to data coordinates
        outer_circles = [
            Circle((x, y), radius=outer_radius) for x, y in zip(x_curr, y_curr)
        ]
        inner_circles = [
            Circle((x, y), radius=inner_radius) for x, y in zip(x_curr, y_curr)
        ]

        # Wrap in PatchCollection so Matplotlib scales them with data/zoom
        collection_outer = PatchCollection(
            outer_circles, match_original=False, **style_outer)
        collection_inner = PatchCollection(
            inner_circles, match_original=False, **style_inner)

        container["patches_outer"] = ax.add_collection(collection_outer)
        container["patches_inner"] = ax.add_collection(collection_inner)

        # Center point (fixed display size dot)
        container["scat_center"] = ax.scatter(x_curr, y_curr, **style_center)

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
    pixel_size=wake_model.aerogenerator.rotor_diameter_m,
    checkboard_size=10
)


ga = GeneticAlgorithm(
    config,
    init_func=init_bool_with_prob,
    fitness_func=fitness_func,
    num_genes=len(candidate_x),
    mutation_func=mutacion_experimental,
    selection_func=tournament_selection,
    crossover_func=checkerboard_crossover,
    polygon=geometry
)


fig, (ax1) = plt.subplots(1, 1, figsize=(10, 4))
x, y = ga.x, ga.y
# ax1.scatter(ga.x, ga.y, marker='s', s=6.4, color='skyblue', edgecolor='navy', linewidth=1.5)
# # ax1.scatter(ga.x, ga.y, c=ga.checkerboard_mask, marker='s', s=200, color='skyblue', edgecolor='navy', linewidth=1.5)
# plt.show()

# Define side length in actual data units (e.g., 0.4 units wide and tall)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from shapely.plotting import plot_polygon
side_length = 6.4
# Create rectangles centered at each (x, y)
rects = [Rectangle((cx - side_length/2, cy - side_length/2), side_length, side_length)
         for cx, cy in zip(x, y)]

pc = PatchCollection(rects, cmap='plasma', edgecolor='black', alpha=0.8)
ax1.add_collection(pc)
c = np.ones_like(ga.checkerboard_mask)
c[3900:8132] = 0
pc.set_array(ga.checkerboard_mask)

# print(len(ga.checkerboard_mask))
# Set limits so all shapes are visible
ax1.set_xlim(min(x) - 1, max(x) + 1)
ax1.set_ylim(min(y) - 1, max(y) + 1)
ax1.set_aspect('equal') # Optional: keeps square aspect ratio
plot_polygon(geometry, ax=ax1, add_points=False, edgecolor="red")

plt.show()