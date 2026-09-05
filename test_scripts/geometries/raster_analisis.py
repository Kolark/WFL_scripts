import geopandas as gpd
import math
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry.base import BaseGeometry
import matplotlib.pyplot as plt
gdf = gpd.read_file("data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)
fig, ax = plt.subplots(figsize=(8, 6))

diametro = 6.4
p_nominal = 5
t_max = 1000/p_nominal

x_min = 5
x_max = 20

def get_polygon_pixel_coords(
    polygon: BaseGeometry, pixel_size: float, CRS: str = "EPSG:3857"
):
    minx, miny, maxx, maxy = polygon.bounds
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    transform = rasterio.Affine(pixel_size, 0, minx, 0, -pixel_size, maxy)
    raster_array = rasterize(
        shapes=[polygon],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
    )

    rows, cols = np.where(raster_array == 1)

    if len(rows) == 0:
        return [polygon.centroid.x],[polygon.centroid.y]

    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return xs, ys

def get_cantidad(x, current_area, current_diametro):
    return current_area / (current_diametro * x) ** 2

def get_x(t, current_area, current_diametro):
    return math.sqrt(current_area / t) / current_diametro

def get_q(xtmax, current_area, current_diametro):
    if get_cantidad(x_max, current_area, current_diametro) <= t_max:
        p = max(xtmax - x_min, 0) / (x_max - x_min)
        v = x_min + p * (xtmax - x_min)
        return max(x_min, v)
    else:
        return max(x_max, xtmax - 5)

# 1 obtener el xtmax teorico,
# 2 obtener la cantidad real
# 3. guardar la cantidad real
# 4. ver cuantos estan por debajo y cuantos por arriba.
print(len(gdf))
print(gdf["id_poligon"].iloc[8516])
xs = []
qs = []
cantidades =[]
i = 0
for g in gdf.geometry:
    area = g.area
    xtmax = get_x(t_max, area, diametro)
    qvalue = get_q(xtmax, area, diametro)
    # print(xtmax, qvalue)
    cantidad = len(
        get_polygon_pixel_coords(g, pixel_size=max(xtmax - 1, 1) * diametro)[0]
    )
    cantidad_raw = len(
        get_polygon_pixel_coords(g, pixel_size=xtmax * diametro)[0]
    )
    # Quiero saber para que geometrias que segun la ecuacion tengan mas de 200
    # con x > 20.0, si la cantidad real es menor que 200
    if xtmax > 20.0 and cantidad_raw < 200:
        print(i, xtmax, get_cantidad(xtmax, area, diametro), cantidad, cantidad_raw)
    i+=1
    xs.append(xtmax)
    qs.append(qvalue)
    cantidades.append(cantidad)


# ax.scatter(x=xs, y= cantidades, s=0.25)
# ax.scatter(x=qs, y= cantidades, s=0.25)
# plt.show()
# cantidades = np.array(cantidades)
# countover = np.count_nonzero(cantidades >= 200)
# countunder = np.count_nonzero(cantidades < 200)
# print(np.argwhere(cantidades < 200))
# print(countunder, countover)
