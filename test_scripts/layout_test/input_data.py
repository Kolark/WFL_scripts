import geopandas as gpd
import pandas as pd
from wind_dataset.wind_dataset import WindDataset
from layout_optimization.fitness_funcs import (
    make_max_multivar_fn,
    make_min_multivar_fn,
    make_max_mv_target_turbines,
    make_min_mv_target_turbines,
    make_max_tmax_turbines,
    make_min_tmax_turbines,
    make_custom_min_multivar
)
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
# ========WINDDATA===========
# wind_ds = WindDataset(
#     nc_path="data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc",
#     easting="easting",
#     northing="northing",
#     ws="WS",
# )

wind_ds = WindDataset(
    nc_path="/home/felipe/Desktop/Trabajo/wrf_data/final_nc_process/wind_data_10m.nc",
    easting="Easting",
    northing="Northing",
    ws="WS",
    time_dim="time"
)
# ========GEOMETRY===========

gdf = gpd.read_file("data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)

geometries = {
    "sb": gdf.geometry.iloc[33601], # e_10m_33602,
    "sg": gdf.geometry.iloc[36598], # e_10m_36599,
    "mb": gdf.geometry.iloc[34267], #e_10m_34268##BAD,
    "mg": gdf.geometry.iloc[36661], #e_10m_36662##BUENO,
    "bb": gdf.geometry.iloc[33929], # e_10m_33930,
    "bg": gdf.geometry.iloc[36643], # e_10m_36644,
}

geom_departamentos = {
    "sb": "atlantico",
    "sg": "la guajira",
    "mb": "la guajira",
    "mg": "la guajira",
    "bb": "atlantico",
    "bg": "la guajira",
}

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

# ========GA_TEST=======


max_multivar_funcs = {
    "max_fit_1": make_max_multivar_fn(1.0, 1.0, 1.0),
    # "max_fit_2": make_max_multivar_fn(0.0, 5.0, 0.0),
    # "max_fit_3": make_max_multivar_fn(0.0, 5.0, 5.0),
    # "max_fit_4": make_max_mv_target_turbines(1.0, 1.0, 1.0, 20),
    # "max_fit_5": make_max_mv_target_turbines(0.0, 5.0, 0.0, 20),
    # "max_fit_6": make_max_mv_target_turbines(0.0, 5.0, 5.0, 20),
    # "max_fit_7": make_max_tmax_turbines(1.0, 1.0, 1.0, 200),
    # "max_fit_8": make_max_tmax_turbines(5.0, 0.0, 1.0, 200),
    # "max_fit_9": make_max_tmax_turbines(0.0, 5.0, 1.0, 200)
}


min_multivar_funcs = {
    "min_fit_1": make_min_multivar_fn(1.0, 1.0, 1.0),
    "min_fit_2": make_min_multivar_fn(0.0, 5.0, 0.0),
    "min_fit_3": make_min_multivar_fn(0.0, 5.0, 5.0),
    "min_fit_4": make_min_mv_target_turbines(1.0, 1.0, 1.0, 20),
    "min_fit_5": make_min_mv_target_turbines(0.0, 5.0, 0.0, 20),
    "min_fit_6": make_min_mv_target_turbines(0.0, 5.0, 5.0, 20),
    "min_fit_7": make_min_tmax_turbines(1.0, 1.0, 1.0, 200),
    "min_fit_8": make_min_tmax_turbines(5.0, 0.0, 1.0, 200),
    "min_fit_9": make_min_tmax_turbines(0.0, 5.0, 1.0, 200),
}


basic_min_multivar_funcs = {
    "best_aep": make_custom_min_multivar(1.0, 0.0, 0.0),
    "best_cf":  make_custom_min_multivar(0.0, 1.0, 0.0),
    "best_pl":  make_custom_min_multivar(0.0, 0.0, 1.0),
}
