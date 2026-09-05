import geopandas as gpd
import pandas as pd
from wind_dataset.wind_dataset import WindDataset
from layout_optimization.grid_search import GridSearchOptimizer
from layout_optimization.fitness_funcs import make_target_aep_fitness, make_target_aep_multivar_fitness
from wake_models import WakeModel, obtener_aerogenerador
from plot_utils import plot_result
from data_export import export_points_to_gpkg
from tqdm import tqdm
import input_data
from wake_models.wake_metrics import get_park_cf, get_park_loss
# ========PLOT RESULTS=====
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
# gs_geometries = [input_data.geometry_mb, input_data.geometry_mg]

# geometries = input_data.geometries

# geometries = {
#     "sb": input_data.geometries["sb"],
#     "sg": input_data.geometries["sg"],
#     "mb": input_data.geometries["mb"],
#     "mg": input_data.geometries["mg"],
# }
geometries = input_data.geometries
funcs = input_data.basic_min_multivar_funcs
from itertools import product

# Store all combinations in a list variable
total_experiments = list(
    product(geometries.items(), funcs.items())
)
results = []
from monitor import Monitor
for (g_name, g), (fn_name, fn) in tqdm(total_experiments):
    gso = GridSearchOptimizer(
        wind_data=input_data.wind_ds,
        wake_model=input_data.wake_model,
        fitness_func=fn,
    )

    best_result = gso.optimize(geometry=g).copy()
    max_aep = (
        best_result.p_nominal_kW
        * best_result.n_turbinas
        * best_result.n_horas
    )
    total_aep_ideal = best_result.aep_ideal.sum()
    total_aep_wake = best_result.aep_wake.sum()

    results.append(
        {
            "polygon": g_name,
            "fn": fn_name,
            "fitness": best_result.fitness,
            "config": gso.best_params,
            "aep_wake": best_result.aep_wake.sum(),
            "park_cf": get_park_cf(total_aep_wake, max_aep),
            "park_loss": get_park_loss(
                total_aep_ideal, total_aep_wake
            ),
            "x_positions": [best_result.x.tolist()],
            "y_positions": [best_result.y.tolist()],
            "best_result_x":best_result.x.tolist(),
            "best_result_y":best_result.y.tolist(),
            "n_turbinas": best_result.n_turbinas,
        }
    )


pd.DataFrame(results).to_csv("results/gridsearch_result.csv")
