import numpy as np
from wind_dataset import WindDataset
from wind_dataset.wind_dataset import WindDataset, TimeScale
from wake_models import WakeModel, obtener_aerogenerador
import xarray as xr
import numpy as np

import calendar
import numpy as np

import numpy as np


def generate_centered_grid(minx, miny, maxx, maxy, n, s):
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0

    offsets = (np.arange(n) - (n - 1) / 2.0) * s

    grid_x = center_x + offsets
    grid_y = center_y + offsets

    X, Y = np.meshgrid(grid_x, grid_y)

    xs = X.ravel()
    ys = Y.ravel()

    return xs, ys

# Output: array([720, 744, 720, 744, 744, 672, 744])
# print(xr.load_dataset("data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc"))
# ========WINDDATA===========
wind_ds = WindDataset(
    nc_path="data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc",
    easting="easting",
    northing="northing",
    ws="WS",
    time_dim="time"
)
wind_ds_monthly = WindDataset(
    nc_path="data/netcdf/monthly_averaged_dataset.nc",
    easting="Easting",
    northing="Northing",
    ws="WS",
    time_dim="time",
    time_scale=TimeScale.Monthly
)
years = len(wind_ds_monthly.get_time_values())/12

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

minx, miny, maxx, maxy = wind_ds.ds.rio.bounds()

rng = np.random.default_rng(seed=42)
num_points = 5
# points = rng.uniform([minx, miny], [maxx, maxy], size=(num_points, 2))

# xs = points[:, 0]
# ys = points[:, 1]

xs, ys = generate_centered_grid(
    minx=minx,
    miny=miny+300000,
    maxx=maxx,
    maxy=maxy,
    n=2,
    s=5.0 * aerogenerador.rotor_diameter_m,
)
shear_alpha = 0.14
ref = 10.0
alpha = float(shear_alpha)
hub = 12.0
height_factor = (hub / ref) ** alpha
print(len(xs))

# wind_ds.fill_cache(wind_ds.ds.rio.bounds())
# ws_cache_h, wd_cache_h = wind_ds.get_wind_data_from_cache(x=xs, y=ys)

# aep_ideal_turbines_h, aep_wake_turbines_h, veff_sum_h = (
#     wake_model.calc_wake_on_turbines_parallel(xs, ys, ws_cache_h*height_factor, wd_cache_h)
# )
# print(aep_ideal_turbines_h.sum(), aep_wake_turbines_h.sum(), veff_sum_h.sum())

wind_ds_monthly.fill_cache(wind_ds_monthly.ds.rio.bounds())
ws_cache, wd_cache = wind_ds_monthly.get_wind_data_from_cache(x=xs, y=ys)
aep_ideal_turbines, aep_wake_turbines, veff_sum = (
    wake_model.calc_wake_on_turbines_parallel(xs, ys, ws_cache*height_factor, wd_cache, ts=wind_ds_monthly.ts)
)

# print(ws_cache, ws_cache*height_factor)
# import matplotlib.pyplot as plt
# plt.plot(ws_cache_h.flatten())
# plt.plot(ws_cache.flatten())
# plt.show()
print(aep_ideal_turbines.sum()/years, aep_wake_turbines.sum()/years, veff_sum.sum())
