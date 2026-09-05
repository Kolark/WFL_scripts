import numpy as np
from shapely.geometry.base import BaseGeometry
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider

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

from shapely import unary_union
gdf = gpd.read_file("data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)
# geometry = unary_union([gdf.geometry.iloc[36643],gdf.geometry.iloc[36633]])
geometry = unary_union([gdf.geometry.iloc[8516]])


diametro = 30
p_nominal = 5
t_max = 1000/p_nominal
area = geometry.area
x_min = 5
x_max = 20

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

fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.3)

base_x = np.arange(2.0, 30, 0.5)
base_x_max = max(base_x)
base_x_min = min(base_x)

cantidad_real = [len(get_polygon_pixel_coords(geometry, pixel_size=v*diametro)[0]) for v in base_x]
realline, = ax.plot(base_x, cantidad_real)

cantidad_teorico = [get_cantidad(v, area, diametro) for v in base_x]
teorline, = ax.plot(base_x, cantidad_teorico)

y_max = max(cantidad_real)
y_min = min(cantidad_real)

tmaxline, = ax.plot([base_x_min, base_x_max], [t_max, t_max])
xminline, = ax.plot([x_min, x_min], [y_min, y_max])
xmaxline, = ax.plot([x_max, x_max], [y_min, y_max])

xtmax = get_x(t_max, area, diametro)
xtmax_point = ax.scatter(x=[xtmax], y=[t_max], marker="x")

qvalue = get_q(xtmax, area, diametro)
qpoint = ax.scatter(qvalue, get_cantidad(qvalue, area, diametro))

diams = np.arange(5,100,1)
q_values = [get_q(get_x(t_max, area, d), area, d) for d in diams]
ax.plot(
    q_values,
    [get_cantidad(q, area, d) for q, d in zip(q_values, diams)],
)
ax_diam_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
diameter_slider = Slider(
    ax=ax_diam_slider,
    label='Diameter',
    valmin=5.0,
    valmax=100.0,
    valinit=diametro
)

def update(val):
    global area
    current_diametro = diameter_slider.val
    current_area = area
    cantidad_real = [len(get_polygon_pixel_coords(geometry, pixel_size=v*current_diametro)[0]) for v in base_x]
    cantidad_teorico = [get_cantidad(v, current_area, current_diametro) for v in base_x]

    realline.set_ydata(cantidad_real)
    teorline.set_ydata(cantidad_teorico)

    xtmax = get_x(t_max, current_area, current_diametro)
    qvalue = get_q(xtmax, current_area, current_diametro)

    xtmax_point.set_offsets([[xtmax, t_max]])
    qpoint.set_offsets([[qvalue, get_cantidad(qvalue, current_area, current_diametro)]])
    print("qvalue", qvalue, "cantidad real", len(get_polygon_pixel_coords(geometry, pixel_size=qvalue*current_diametro)[0]))
    fig.canvas.draw_idle()

diameter_slider.on_changed(update)

plt.show()
