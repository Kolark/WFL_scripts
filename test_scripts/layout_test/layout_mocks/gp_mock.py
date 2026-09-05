import geopandas as gpd
import pandas as pd
from layout_optimization.gp_optimize import GridGPOptimizer
from tqdm import tqdm
import input_data
from wake_models.wake_metrics import get_park_cf, get_park_loss


config_1 = {
    "n_calls": 100,
    "n_initial_points": 10,
    "acq_func": "EI",
    "random_state": 42,
    "n_points": 1000,
}

config_2 = {
    "n_calls": 75,
    "n_initial_points": 20,
    "acq_func": "EI",
    "random_state": 42,
    "n_points": 2000,
}


gp_configs = {
    "config_1": config_1,
    "config_2": config_2,
}

geometries = input_data.geometries
funcs = input_data.min_multivar_funcs

results = []

from itertools import product

# Store all combinations in a list variable
total_experiments = list(
    product(geometries.items(), funcs.items(), gp_configs.items())
)
total_iterations = len(geometries) * len(funcs) * len(gp_configs)
import warnings
warnings.filterwarnings('ignore')
for (g_name, g), (fn_name, fn), (c_name, config) in tqdm(total_experiments):
    gp_optimizer = GridGPOptimizer(
        wind_data=input_data.wind_ds,
        wake_model=input_data.wake_model,
        fitness_func=fn,
        gp_args=config
    )

    res = gp_optimizer.optimize(geometry=g).copy()

    max_aep = (
        res.p_nominal_kW
        * res.n_turbinas
        * res.n_horas
    )
    total_aep_ideal = res.aep_ideal.sum()
    total_aep_wake = res.aep_wake.sum()

    results.append(
        {
            "polygon": g_name,
            "fn": fn_name,
            "config": c_name,
            "fitness": res.fitness,
            "aep_wake": res.aep_wake.sum(),
            "park_cf": get_park_cf(total_aep_wake, max_aep),
            "park_loss": get_park_loss(
                total_aep_ideal, total_aep_wake
            ),
            "x_positions": gp_optimizer.x_history,
            "y_positions": gp_optimizer.y_history,
            "fitness_history": gp_optimizer.fitness_history,
            "best_result_x":res.x.tolist(),
            "best_result_y":res.y.tolist(),
            "n_turbinas": res.n_turbinas,
        }
    )

pd.DataFrame(results).to_csv("results/gp_result.csv")
