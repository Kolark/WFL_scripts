from pprint import pprint
from layout_optimization.ga.crossover import two_point_crossover, checkerboard_crossover
from layout_optimization.ga.selection import tournament_selection
from layout_optimization.ga.mutation import mutacion_experimental
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from layout_optimization.ga.raster_utils import get_raster_pixel_coords
from layout_optimization import GAConfig, GeneticAlgorithm
from shapely.plotting import plot_polygon
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
wind_data.fill_cache(wind_data.ds.rio.bounds())
# ==============================================================================


gdf = gpd.read_file(
    "/home/felipe/Desktop/Trabajo/WFL_scripts/data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)

print(gdf.columns)

# e_10m_36527
# geometry = gdf.geometry.iloc[36526]
# geometry = gdf.geometry.iloc[36662]
# e_10m_38725
# geometry = gdf.geometry.iloc[38724]
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

    p_true = 0.001
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
    num_generations=10,
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

# fig, ax = plt.subplots(figsize=(10, 4))
# ax.scatter(ga.x, ga.y, c=ga.checkerboard_mask)
# plt.show()
# ====
best_solution, best_fitness = ga.run()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
ax, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
ax6.axis("off")

ax2.plot(ga.best_fitness_history)
ax3.plot(ga.best_amount_history)

print(best_solution)

final_x = candidate_x[best_solution]
final_y = candidate_y[best_solution]
plot_polygon(geometry, ax=ax, add_points=False)


# Your existing scatter plot
# ax.scatter(final_x, final_y)


# 4. Ensure equal aspect ratio so circles do not look stretched like ellipses
ax.set_aspect("equal", adjustable="datalim")
ws, wd = wind_data.get_wind_data_from_cache(final_x, final_y)
aep_ideal_turbines, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
    final_x, final_y, ws, wd, wind_data.ts)

# Métricas energéticas
generated_energy = aep_wake_turbines.sum()
ideal = aep_ideal_turbines.sum()
max_aep = (
    wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * len(final_x)
)

cf = (generated_energy / max_aep) * 100.0  # En porcentaje (%)
potencia_mW = (
    (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
    / 1000.0
)
print("potenciaMW", potencia_mW)
pl = get_park_loss(aep_ideal=ideal, aep_wakeloss=generated_energy)

# Cálculo de LCOE
lcoe_res = calculate_lcoe(geom_department, potencia_mW, len(final_x), FCap=cf)

maxpot = wake_model.aerogenerator.p_nominal_kW * len(final_x) / 1000
lcoe_test1 = calculate_lcoe(geom_department, maxpot, len(final_x), FCap=cf)
lcoe_test2 = calculate_lcoe(
    geom_department, maxpot * 2, len(final_x),
    FCap=100.0)
lcoe_test3 = calculate_lcoe(
    geom_department, potencia_mW*10, len(final_x)//2, FCap=cf)
lcoe_val = lcoe_res["result"]["LCOE"]
print("="*15, geo_str, "="*15)
print("n turbinas", len(final_x))
print("CF final", f"{cf:3f}%")
print("PL", pl)
print("LCOE final", lcoe_val)
pprint(lcoe_res)
print(
    "LCOE IRREAL", lcoe_test1["result"]["LCOE"],
    lcoe_test2["result"]["LCOE"],
    lcoe_test3["result"]["LCOE"])
print("="*32)

SEED = 42
np.random.seed(SEED)

original_x = np.array(final_x, copy=True)
original_y = np.array(final_y, copy=True)
n_initial = len(original_x)

history_indices = []
history_removed_count = []
history_lcoe = []
history_cf = []
history_pl = []

remaining_indices = list(range(n_initial))

while len(remaining_indices) >= 1:
    current_x = original_x[remaining_indices]
    current_y = original_y[remaining_indices]
    n_curr = len(remaining_indices)
    removed_count = n_initial - n_curr

    ws, wd = wind_data.get_wind_data_from_cache(current_x, current_y)
    aep_ideal_turbines, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
        current_x, current_y, ws, wd, wind_data.ts)

    generated_energy = aep_wake_turbines.sum()
    ideal = aep_ideal_turbines.sum()
    max_aep = wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * n_curr

    cf = (generated_energy / max_aep) * 100.0
    potencia_mW = (
        (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
        / 1000.0
    )
    pl = get_park_loss(aep_ideal=ideal, aep_wakeloss=generated_energy)

    lcoe_res = calculate_lcoe("la guajira", potencia_mW, n_curr, FCap=cf)
    lcoe_val = lcoe_res["result"]["LCOE"]

    history_indices.append(remaining_indices.copy())
    history_removed_count.append(removed_count)
    history_lcoe.append(lcoe_val)
    history_cf.append(cf)
    history_pl.append(pl)

    if len(remaining_indices) > 1:
        idx_to_remove = np.random.choice(len(remaining_indices))
        remaining_indices.pop(idx_to_remove)
    else:
        break

min_radius = wake_model.aerogenerator.rotor_diameter_m * 5.0

u_cand_x = candidate_x[~best_solution]
u_cand_y = candidate_y[~best_solution]

cand_x = np.array(u_cand_x, copy=True)
cand_y = np.array(u_cand_y, copy=True)
n_candidates = len(cand_x)

add_history_indices = []
add_history_count = []
add_history_lcoe = []
add_history_cf = []
add_history_pl = []

available_pool = list(range(n_candidates))
active_indices = []

first_idx = np.random.choice(available_pool)
active_indices.append(first_idx)

while len(available_pool) > 0:
    curr_x = cand_x[active_indices]
    curr_y = cand_y[active_indices]

    xs = np.concatenate((curr_x, final_x))
    ys = np.concatenate((curr_y, final_y))
    n_curr = len(xs)
    ws, wd = wind_data.get_wind_data_from_cache(xs, ys)
    aep_ideal_turbines, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
        xs, ys, ws, wd, wind_data.ts)

    generated_energy = aep_wake_turbines.sum()
    ideal = aep_ideal_turbines.sum()
    max_aep = wake_model.aerogenerator.p_nominal_kW * wind_data.ts.sum() * n_curr

    cf = (generated_energy / max_aep) * 100.0
    potencia_mW = (
        (aep_wake_turbines / wind_data.ts[:, np.newaxis]).mean(axis=0).sum()
        / 1000.0
    )
    pl = get_park_loss(aep_ideal=ideal, aep_wakeloss=generated_energy)

    lcoe_res = calculate_lcoe("la guajira", potencia_mW, n_curr, FCap=cf)
    lcoe_val = lcoe_res["result"]["LCOE"]

    add_history_indices.append(active_indices.copy())
    add_history_count.append(n_curr)
    add_history_lcoe.append(lcoe_val)
    add_history_cf.append(cf)
    add_history_pl.append(pl)

    valid_candidates = []
    for cand_idx in available_pool:
        if cand_idx in active_indices:
            continue
        dist = np.hypot(cand_x[cand_idx] - xs, cand_y[cand_idx] - ys)
        if np.all(dist >= min_radius):
            valid_candidates.append(cand_idx)

    if not valid_candidates:
        print(
            f"Saturación alcanzada: No es posible agregar más turbinas respetando r = {min_radius} m.")
        print(f"Total turbinas ubicadas: {n_curr}")
        break

    next_idx = np.random.choice(valid_candidates)
    active_indices.append(next_idx)

n_remove_steps = n_initial
n_add_steps = len(add_history_count)


def state_for_delta(delta):
    if delta <= 0:
        k = -delta
        active_idx = history_indices[k]
        removed_idx = sorted(set(range(n_initial)) - set(active_idx))
        xs_active = original_x[active_idx]
        ys_active = original_y[active_idx]
        xs_removed = original_x[removed_idx]
        ys_removed = original_y[removed_idx]
        lcoe = history_lcoe[k]
        count = n_initial - history_removed_count[k]
    else:
        m = delta - 1
        active_idx = add_history_indices[m]
        xs_active = np.concatenate((cand_x[active_idx], final_x))
        ys_active = np.concatenate((cand_y[active_idx], final_y))
        xs_removed = np.array([])
        ys_removed = np.array([])
        lcoe = add_history_lcoe[m]
        count = add_history_count[m]
    return xs_active, ys_active, xs_removed, ys_removed, lcoe, count


delta_range = list(range(-(n_remove_steps - 1), n_add_steps + 1))
delta_lcoe = [state_for_delta(d)[4] for d in delta_range]

ax4.clear()
plot_polygon(geometry, ax=ax4, add_points=False)
ax4.set_aspect("equal", adjustable="datalim")

xs0, ys0, xr0, yr0, lcoe0, count0 = state_for_delta(0)

scat_active = ax4.scatter(
    xs0, ys0, c="crimson", label="Turbinas Activas", zorder=3)
scat_removed = ax4.scatter(
    xr0, yr0, c="gray", alpha=0.3, label="Quitadas", zorder=2)
ax4.legend(loc="upper right")
ax4.set_title(f"Turbinas Activas: {count0} | LCOE: {lcoe0:.2f}")

slider_ax4 = ax4.inset_axes([0.15, 0.02, 0.70, 0.03])
slider_delta = Slider(
    ax=slider_ax4,
    label="Δ Turbinas ",
    valmin=delta_range[0],
    valmax=delta_range[-1],
    valinit=0,
    valstep=1,
)

ax5.clear()
ax5.plot(delta_range, delta_lcoe, marker="o", linewidth=1.5, color="tab:blue")
ax5.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
ax5.set_title("Evolución de LCOE al Quitar/Agregar Turbinas")
ax5.set_xlabel("Δ Turbinas respecto a la solución base")
ax5.set_ylabel("LCOE ($/MWh)")
ax5.grid(True, linestyle="--", alpha=0.6)

current_point_ax5, = ax5.plot(
    [0], [lcoe0], marker="o", color="red", markersize=8, zorder=5
)


def update_delta(val):
    delta = int(slider_delta.val)
    xs_active, ys_active, xs_removed, ys_removed, lcoe, count = state_for_delta(
        delta)

    scat_active.set_offsets(np.c_[xs_active, ys_active])
    if len(xs_removed) > 0:
        scat_removed.set_offsets(np.c_[xs_removed, ys_removed])
    else:
        scat_removed.set_offsets(np.empty((0, 2)))

    current_point_ax5.set_data([delta], [lcoe])
    ax4.set_title(f"Turbinas Activas: {count} | LCOE: {lcoe:.2f}")
    fig.canvas.draw_idle()


slider_delta.on_changed(update_delta)

scat, slider = plot_interactive_history(
    ax, list(zip(ga.best_x, ga.best_y)),
    inner_radius=wake_model.aerogenerator.rotor_diameter_m * 5,
    outer_radius=wake_model.aerogenerator.rotor_diameter_m * 20.0)

plt.show()