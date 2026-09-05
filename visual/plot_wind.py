"""
Check visual para los datos del viento.
1. Gráfica de la velocidad del viento para todo el dominio 2 en la 1era hora.
2. Gráfica de la velocidad del viento para una subregion del dominio 2 a traves
 del tiempo (con un slider).
3. Gráfica del punto y valor correspondiente de la velocidad del viento para
esa coordenada.
4. Gráfica (quiver) de la dirección del viento en algunas coordenadas aleatorias.
5. La grafica debe estar en la orientación correcta
"""

from wind_dataset import WindDataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# =====================WindDataset==loading data===============================

wind_ds = WindDataset(
    nc_path="data/netcdf/ws_wd_ctm12_chunked_uvnorm.nc",
    easting="easting",
    northing="northing",
    ws="WS",
)


def shrink_bounds(bounds, s=0.5):
    bx, by, tx, ty = bounds
    dx, dy = (tx - bx) * (1 - s) / 2, (ty - by) * (1 - s) / 2
    return bx + dx, by + dy, tx - dx, ty - dy


print("Getting all wind data.")
bounds = wind_ds.ds.rio.bounds()
sub_bounds = shrink_bounds(bounds=bounds, s=0.65)
ws, wd, _ = wind_ds.filter_by_bounds(*bounds)

wind_ds.fill_cache(sub_bounds)
sub_ws = wind_ds._cache["ws"]
sub_wd = wind_ds._cache["wd"]


def plot_ws(ax, fig, ws_t0, bounds, cmap, edgecolor):
    ny, nx = ws_t0.shape
    c_minx, c_miny, c_maxx, c_maxy = bounds
    x_regular = np.linspace(c_minx, c_maxx, nx + 1)
    y_regular = np.linspace(c_miny, c_maxy, ny + 1)
    X_regular, Y_regular = np.meshgrid(x_regular, y_regular)

    mesh = ax.pcolormesh(
        X_regular,
        Y_regular,
        ws_t0,
        cmap=cmap,
        shading="flat",
        # edgecolor=edgecolor,
        # linewidth=0.3,
        alpha=0.5,
    )
    fig.colorbar(mesh, ax=ax, label="Wind Speed (WS) en t=0")

    return mesh


def crear_onclick(fig, ax, punto, get_title_func):

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        click_x = event.xdata
        click_y = event.ydata

        print("CLICK", click_x, click_y)
        punto.set_data([click_x], [click_y])
        ax.set_title(get_title_func(click_x, click_y))
        fig.canvas.draw_idle()

    return onclick


# ===================1.Plot all wind data======================================
fig, ax = plt.subplots(figsize=(15, 15))

all_ws_plot = plot_ws(
    ax=ax, fig=fig, ws_t0=ws[0], bounds=bounds, cmap="plasma", edgecolor="red"
)
sub_ws_plot = plot_ws(
    ax=ax,
    fig=fig,
    ws_t0=sub_ws[0],
    bounds=sub_bounds,
    cmap="viridis",
    edgecolor="blue",
)
(punto_seleccionado,) = ax.plot(
    [],
    [],
    "ro",
    markersize=8,
    markeredgecolor="black",
    label="Punto seleccionado",
)
ax_slider = plt.axes([0.25, 0.0, 0.8, 0.02])
slider = Slider(
    ax_slider, "Time Step", 0, ws.shape[0] - 1, valinit=0, valfmt="%d"
)


def get_title(xs, ys):
    ws, wd = wind_ds.get_wind_data_from_cache([xs], [ys])
    return f"Coord [{xs:.1f}, {ys:.1f}], ws [{ws[0]}] , wd [{wd[0]}]"


minx, miny, maxx, maxy = bounds

rows, cols = ws.shape[1], ws.shape[2]
cell_w = (maxx - minx) / cols
cell_h = (maxy - miny) / rows

col_centers = np.linspace(minx + cell_w / 2, maxx - cell_w / 2, cols)
row_centers = np.linspace(miny + cell_h / 2, maxy - cell_h / 2, rows)
sampled_cols = np.linspace(0, cols - 1, 20, dtype=int)
sampled_rows = np.linspace(0, rows - 1, 20, dtype=int)

xs = col_centers[sampled_cols]
ys = row_centers[sampled_rows]


def _sample_quiver(t: int):
    sampled_dir = wd[t][np.ix_(sampled_rows, sampled_cols)]  # (10, 10, 2)
    u = sampled_dir[:, :, 0]
    v = sampled_dir[:, :, 1]
    return u, v


u0, v0 = _sample_quiver(0)
quiver = ax.quiver(
    xs,
    ys,
    u0,
    v0,
    alpha=0.9,
)


def update(val: float) -> None:
    idx = int(slider.val)
    sub_ws_plot.set_array(sub_ws[idx].ravel())
    u, v = _sample_quiver(idx)
    quiver.set_UVC(u, v)
    fig.canvas.draw_idle()


onclick = crear_onclick(fig, ax, punto_seleccionado, get_title)
slider.on_changed(update)
fig.canvas.mpl_connect("button_press_event", onclick)
ax.set_aspect("equal", adjustable="box")
plt.show()
