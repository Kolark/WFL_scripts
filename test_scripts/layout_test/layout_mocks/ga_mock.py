import geopandas as gpd
import pandas as pd
from wind_dataset.wind_dataset import WindDataset
from wake_models.wake_metrics import get_park_cf, get_park_loss
from wake_models import WakeModel, obtener_aerogenerador
from plot_utils import plot_result, draw_squares_around_points
from layout_optimization import WindFarmGAOptimizer
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from tqdm import tqdm
import input_data

# ga_config_1 = {
#     "num_generations": 310,
#     "sol_per_pop": 10,
#     "num_parents_mating": 5,
#     "K_tournament": 5,
#     "parent_selection_type": "tournament",
#     "crossover_type": "two_points",
#     "mutation_type": "random",
#     "mutation_percent_genes": 10,
# }

ga_configs = {
    "ga_config_2": {
        "num_generations": 310,
        "sol_per_pop": 10,
        "num_parents_mating": 5,
        "K_tournament": 5,
        "parent_selection_type": "tournament",
        "crossover_type": "two_points",
        "mutation_type": "inversion",
        "mutation_percent_genes": 10,
        "keep_elitism": 2,
    },
    # "ga_config_3": {
    #     "num_generations": 310,
    #     "sol_per_pop": 5,
    #     "num_parents_mating": 5,
    #     "K_tournament": 5,
    #     "parent_selection_type": "tournament",
    #     "crossover_type": "two_points",
    #     "mutation_type": "inversion",
    #     "mutation_percent_genes": 10,
    #     "keep_elitism": 2,
    # },
}

geometries = input_data.geometries
funcs = input_data.max_multivar_funcs

results = []
from itertools import product

# Store all combinations in a list variable
total_experiments = list(
    product(geometries.items(), funcs.items(), ga_configs.items())
)

for (g_name, g), (fn_name, fn), (c_name, config) in tqdm(total_experiments):
    try:
        wfgao = WindFarmGAOptimizer(
            wind_data=input_data.wind_ds,
            wake_model=input_data.wake_model,
            fitness_func=fn
        )

        best_solution, best_fitness = wfgao.optimize(
            geometry=g, ga_args=config
        )
        gene_result = wfgao.evaluate_gene(best_solution).copy()
        max_aep = (
            gene_result.p_nominal_kW
            * gene_result.n_turbinas
            * gene_result.n_horas
        )
        total_aep_ideal = gene_result.aep_ideal.sum()
        total_aep_wake = gene_result.aep_wake.sum()

        results.append(
            {
                "polygon": g_name,
                "fn": fn_name,
                "config": c_name,
                "fitness": best_fitness,
                "aep_wake": gene_result.aep_wake.sum(),
                "park_cf": get_park_cf(total_aep_wake, max_aep),
                "park_loss": get_park_loss(
                    total_aep_ideal, total_aep_wake
                ),
                "history": wfgao.history,
                "fitness_history": wfgao.fitness_history,
                "x_positions": wfgao.x.tolist(),
                "y_positions": wfgao.y.tolist(),
                "spacing":wfgao.spacing,
                "best_result_x":gene_result.x.tolist(),
                "best_result_y":gene_result.y.tolist(),
                "n_turbinas": gene_result.n_turbinas,
                "error": None
            }
        )
    except Exception as error:
        results.append(
            {
                "polygon": g_name,
                "fn": fn_name,
                "config": c_name,
                "fitness": None,
                "aep_wake": None,
                "park_cf": None,
                "park_loss": None,
                "history": None,
                "fitness_history": None,
                "x_positions": None,
                "y_positions": None,
                "spacing": None,
                "best_result_x": None,
                "best_result_y": None,
                "n_turbinas": None,
                "error": error,
            }
        )

pd.DataFrame(results).to_csv("results/ga_result.csv")
