import numpy as np
from wind_dataset import WindDataset
from wind_dataset import WD_deprecated
from wake_models import calc_deprecated, obtener_aerogenerador_deprecated
from wake_models import WakeModel
from wake_models import obtener_aerogenerador
import time
import pandas as pd
from wake_models import get_park_cf, get_park_loss, get_power, net_aep_turbine,net_power_turbine

def create_coordinate_points_with_noise(
    n, origin_x=0.0, origin_y=0.0, spacing=1000.0, noise_std=100.0, seed=42
):
    if seed is not None:
        np.random.seed(seed)

    x_line = origin_x + np.arange(n) * spacing
    y_line = origin_y + np.arange(n) * spacing

    x_coords = np.tile(x_line, n)
    y_coords = np.repeat(y_line, n)

    x_noise = np.random.normal(0, noise_std, size=x_coords.shape)
    y_noise = np.random.normal(0, noise_std, size=y_coords.shape)

    x_coords_noisy = x_coords + x_noise
    y_coords_noisy = y_coords + y_noise

    return x_coords_noisy, y_coords_noisy

def get_bounds(x, y):
    minx = np.min(x)
    miny = np.min(y)
    maxx = np.max(x)
    maxy = np.max(y)

    return (minx, miny, maxx, maxy)

def print_result(method, n, wd_duration, wake_duration, ideal_result, wake_result):
    print(f" {method} [{n}] total:({n**2}) time_wd:[{wd_duration:.6f}] time_wake:[{wake_duration:.6f}] ideal:[{ideal_result:.6f}] wake:[{wake_result:.6f}]")
###############################################################################

max_grid_n = 30

benchmark_results = []
spacing = 1000
noise = 100
all_positions = [
    create_coordinate_points_with_noise(
        (i + 1),
        4577830.736126768,
        2688426.840796647,
        spacing=spacing,
        noise_std=noise,
    )
    for i in range(max_grid_n)
]
path_to_save = f"results/benchmarks/wake_benchmark_{spacing}_{noise}_v5.csv"
wake_k = 0.05
thrust_ct = 0.8
max_wake_distance_D = 20.0

#############################NEW METHOD DATA###################################
aerogenerador = obtener_aerogenerador("Vestas V100/2000")
wind_ds = WindDataset(
    nc_path="data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc",
    easting="easting",
    northing="northing",
    ws="WS",
    chunks = "auto"
)
bounds = get_bounds(all_positions[-1][0], all_positions[-1][1])
print(bounds)
wind_ds.fill_cache(bounds)
wake_model = WakeModel(
    aerogenerator=aerogenerador,
    wake_k=wake_k,
    thrust_ct=thrust_ct,
    max_wake_distance_D=max_wake_distance_D,
)

###########################--SERIAL--##########################################

for i in range(max_grid_n):
    positions = all_positions[i]
    x = positions[0]
    y = positions[1]
    #Time wind data
    start_get_wind_data = time.perf_counter()
    ws, wd = wind_ds.get_wind_data_from_cache(x, y)
    end_get_wind_data = time.perf_counter()

    #time calc_wake
    start_calc_wake = time.perf_counter()
    aep_ideal_turbines, aep_wake_turbines, veff_sum = wake_model.calc_wake_on_turbines(
            x=x,
            y=y,
            ws=ws.astype("float64"),
            wd=wd,
        )
    end_calc_wake = time.perf_counter()

    aep_ideal_total = aep_ideal_turbines.sum()
    aep_wake_total = aep_wake_turbines.sum()
    veff_total = veff_sum.sum()

    calc_wake_duration = (end_calc_wake - start_calc_wake)
    get_wd_duration = (end_get_wind_data- start_get_wind_data)

    park_loss = get_park_loss(aep_ideal_total, aep_wake_total)
    energia_max_parque_kWh = aerogenerador.p_nominal_kW * len(x) * ws.shape[0]
    park_cf = get_park_cf(aep_wake_total, energia_max_parque_kWh)

    result = {
        "method":"SERIAL",
        "time_calc_wake": calc_wake_duration,
        "time_get_wind_data": get_wd_duration,
        "n_grid":(i+1),
        "total_turbines":(i+1)**2,
        "ideal_total": aep_ideal_total,
        "wake_total": aep_wake_total,
        "veff_total": veff_total,
        "park_loss": park_loss,
        "capacity_factor_parque": park_cf,
    }
    benchmark_results.append(result)
    print_result("SERIAL", (i+1), get_wd_duration, calc_wake_duration, aep_ideal_total, aep_wake_total)

pd.DataFrame(benchmark_results).to_csv(path_to_save)
print("SAVED BENCHMARK RESULTS")
print("="*20)

###########################--PARALLEL--########################################

for i in range(max_grid_n):
    positions = all_positions[i]
    x = positions[0]
    y = positions[1]
    #Time wind data
    start_get_wind_data = time.perf_counter()
    ws, wd = wind_ds.get_wind_data_from_cache(x, y)
    end_get_wind_data = time.perf_counter()

    #time calc_wake
    start_calc_wake = time.perf_counter()
    aep_ideal_turbines, aep_wake_turbines, veff_sum = wake_model.calc_wake_on_turbines_parallel(
            x=x,
            y=y,
            ws=ws.astype("float64"),
            wd=wd,
        )
    end_calc_wake = time.perf_counter()

    aep_ideal_total = aep_ideal_turbines.sum()
    aep_wake_total = aep_wake_turbines.sum()
    veff_total = veff_sum.sum()

    calc_wake_duration = (end_calc_wake - start_calc_wake)
    get_wd_duration = (end_get_wind_data- start_get_wind_data)

    park_loss = get_park_loss(aep_ideal_total, aep_wake_total)
    energia_max_parque_kWh = aerogenerador.p_nominal_kW * len(x) * ws.shape[0]
    park_cf = get_park_cf(aep_wake_total, energia_max_parque_kWh)

    result = {
        "method":"PARALLEL",
        "time_calc_wake": calc_wake_duration,
        "time_get_wind_data": get_wd_duration,
        "n_grid":(i+1),
        "total_turbines":(i+1)**2,
        "ideal_total": aep_ideal_total,
        "wake_total": aep_wake_total,
        "veff_total": veff_total,
        "park_loss": park_loss,
        "capacity_factor_parque": park_cf,
    }
    benchmark_results.append(result)
    print_result("PARALLEL", (i+1), get_wd_duration, calc_wake_duration, aep_ideal_total, aep_wake_total)

pd.DataFrame(benchmark_results).to_csv(path_to_save)
print("SAVED BENCHMARK RESULTS")
print("="*20)


###########################--CALC_WAKE_WRF--###################################

wind_ds_deprecated = WD_deprecated(
    nc_path="data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc",
    easting="easting",
    northing="northing",
    ws="WS",
    wd="WDIR",
    chunks = "auto"
)
wind_ds_deprecated.fill_cache(bounds=bounds)
# wind_ds_deprecated.fill_cache(bounds=wind_ds_deprecated.ds.rio.bounds())
modelo = "Vestas V100/2000"
aerogenerador_deprecated = obtener_aerogenerador_deprecated(modelo)

for i in range(max_grid_n):
    positions = all_positions[i]
    valid_x = positions[0]
    valid_y = positions[1]

    # Time wind data
    start_get_wind_data = time.perf_counter()
    ws_deprecated, wd_deprecated = (
        wind_ds_deprecated.get_wind_data_from_cache(valid_x, valid_y)
    )
    end_get_wind_data = time.perf_counter()

    start_calc_wake = time.perf_counter()
    res_dep = calc_deprecated(
            valid_x,
            valid_y,
            ws_deprecated,
            wd_deprecated,
            aerogenerador.rotor_diameter_m,
            wake_k,
            thrust_ct,
            max_wake_distance_D,
            aerogenerador_deprecated,
            wdir_is_met_from=True
        )
    end_calc_wake = time.perf_counter()

    calc_wake_duration = (end_calc_wake - start_calc_wake)
    get_wd_duration = (end_get_wind_data- start_get_wind_data)

    result = {
        "method":"CALC_WAKE_WRF",
        "time_calc_wake": calc_wake_duration,
        "time_get_wind_data": get_wd_duration,
        "n_grid":(i+1),
        "total_turbines":(i+1)**2,
        "ideal_total": res_dep["__aep_ideal_total_kWh"],
        "wake_total": res_dep["__aep_wake_total_kWh"],
        "veff_total": res_dep["veff_sum"],
        "park_loss": res_dep["park_loss"],
        "capacity_factor_parque": res_dep["capacity_factor_parque"],
    }
    benchmark_results.append(result)
    print_result("CALC_WAKE_WRF", (i+1), get_wd_duration, calc_wake_duration, res_dep["__aep_ideal_total_kWh"], res_dep["__aep_wake_total_kWh"])

pd.DataFrame(benchmark_results).to_csv(path_to_save)
print("SAVED BENCHMARK RESULTS")
print("="*20)
