"""
Grafica la velocidad del viento y su direccion para un tiempo t.
Permite modificar t con un slider para ver como cambian en el tiempo.
"""
import geopandas as gpd
import numpy as np
from wind_dataset import WindDataset, TimeScale
from plot_utils import plot_wind_data_with_slider
# wind_ds = WindDataset(
#     nc_path="data/netcdf/merra_processed.nc",
#     easting="Easting",
#     northing="Northing",
#     ws="WS",
#     x_dim="Easting",
#     y_dim="Northing",
#     time_dim="time"
# )

# wind_ds = WindDataset(
#     nc_path="data/netcdf/4_final.nc",
#     # nc_path="data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc",
#     easting="Easting",
#     northing="Northing",
#     ws="WS",
#     # chunks="auto",
# )

wind_ds = WindDataset(
    nc_path="data/netcdf/monthly_averaged_dataset.nc",
    easting="Easting",
    northing="Northing",
    ws="WS",
    time_dim="time",
    time_scale=TimeScale.Monthly,
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

ws, wd, new_bounds = wind_ds.filter_by_bounds(minx,miny,maxx,maxy)

ws_point, wd_point = wind_ds.get_wind_data(
    [4947111.376452917], [2551943.728471596]
)

wind_ds.fill_cache(new_bounds)
ws_point_cache, wd_point_cache = wind_ds.get_wind_data_from_cache(
    [4947111.376452917], [2551943.728471596]
)

print(ws_point.flatten())
print(ws_point_cache.flatten())
print(ws.shape)
print(wd.shape)
gdf = gpd.read_file("data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)
geometry = gdf.geometry.iloc[32043]
print("ws nan",np.count_nonzero(np.isnan(ws)))
print("wd nan",np.count_nonzero(np.isnan(wd)))
plot_wind_data_with_slider(ws, new_bounds, wd, geometry=geometry)
