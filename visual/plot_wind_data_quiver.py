"""
Grafica la velocidad del viento y su direccion para un tiempo t, de algunos
puntos, y permite ver como cambian a traves del tiempo.

"""

import numpy as np
from wind_dataset import WindDataset
from plot_utils import plot_wind_points_with_slider
wind_ds = WindDataset(
    nc_path="/home/felipe/Desktop/Trabajo/Data/corregidos/wind_data_100m_D01.nc",
    easting="Easting",
    northing="Northing",
    ws="WS",
    x_dim="x",
    y_dim="y"
)
minx, miny, maxx, maxy = wind_ds.ds.rio.bounds()

height = abs(miny-maxy)
width = abs(minx-maxx)
# Range from [0,0.5)
multiplier = 0.0
minx += width*multiplier
miny += height*multiplier
maxx -= width*multiplier
maxy -= height*multiplier

rng = np.random.default_rng(seed=42)
num_points = 3000
points = rng.uniform([minx, miny], [maxx, maxy], size=(num_points, 2))

wind_ds.fill_cache(bounds=(minx,miny,maxx,maxy))
ws, wd = wind_ds.get_wind_data_from_cache(x=points[:,0],y=points[:,1])
plot_wind_points_with_slider(ws, wd, points, pixel_bounds=(minx,miny,maxx,maxy))
