import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm, colors as mcolors
import numpy.typing as npt
import numpy as np
from shapely.plotting import plot_polygon
def plot_wind_data_with_slider(
    var_data: npt.NDArray,
    pixel_bounds: tuple,
    dir_data: npt.NDArray,
    geometry=None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    minx, miny, maxx, maxy = pixel_bounds
    vmax = float(var_data.max())
    vmin = float(var_data.min())

    im = ax.imshow(
        var_data[0],
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=(minx, maxx, miny, maxy),
        origin="lower",
    )
    fig.colorbar(im, ax=ax, label="var_data")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - (xmax - xmin) * 0.2, xmax + (xmax - xmin) * 0.2)
    ax.set_ylim(ymin - (ymax - ymin) * 0.2, ymax + (ymax - ymin) * 0.2)
    title = ax.set_title(f"Time: {0}")

    rows, cols = var_data.shape[1], var_data.shape[2]
    cell_w = (maxx - minx) / cols
    cell_h = (maxy - miny) / rows

    col_centers = np.linspace(minx + cell_w / 2, maxx - cell_w / 2, cols)
    row_centers = np.linspace(miny + cell_h / 2, maxy - cell_h / 2, rows)
    sampled_cols = np.linspace(0, cols - 1, 20, dtype=int)
    sampled_rows = np.linspace(0, rows - 1, 20, dtype=int)

    xs = col_centers[sampled_cols]
    ys = row_centers[sampled_rows]
    qx, qy = np.meshgrid(xs, ys)

    def _sample_quiver(t: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        sampled_dir = dir_data[t][np.ix_(sampled_rows, sampled_cols)]  # (10, 10, 2)
        sampled_mag = var_data[t][np.ix_(sampled_rows, sampled_cols)]  # (10, 10)
        u = sampled_dir[:, :, 0]
        v = sampled_dir[:, :, 1]
        return u, v, sampled_mag

    u0, v0, mag0 = _sample_quiver(0)
    # quiver = ax.quiver(qx, qy, u0 * mag0, v0 * mag0, color="white", alpha=0.8, scale=vmax * 10)

    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, "Time Step", 0, var_data.shape[0] - 1, valinit=0, valfmt="%d")

    def update(val: float) -> None:
        idx = int(slider.val)
        im.set_array(var_data[idx])
        title.set_text(f"Time: {idx}")
        u, v, mag = _sample_quiver(idx)
        # quiver.set_UVC(u * mag, v * mag)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    if geometry is not None:
        plot_polygon(geometry, ax=ax, add_points=False, color="red")
    plt.show()


def plot_wind_points_with_slider(
    var_data: npt.NDArray,
    dir_data: npt.NDArray,
    points: npt.NDArray,  # (n_points, 2) in same CRS as bounds
    pixel_bounds: tuple,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    minx, miny, maxx, maxy = pixel_bounds
    vmax = float(var_data.max())
    vmin = float(var_data.min())

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    title = ax.set_title(f"Time: {0}")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    def _build_quiver(t: int):
        mag = var_data[t]           # (n_points,)
        u = dir_data[t, :, 0] * mag
        v = dir_data[t, :, 1] * mag
        colors = cmap(norm(mag))    # (n_points, 4) RGBA
        return u, v, colors

    u0, v0, c0 = _build_quiver(0)
    quiver = ax.quiver(
        points[:, 0], points[:, 1], u0, v0,
        color=c0,
        scale=vmax * 10,
        alpha=0.9,
    )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="var_data")

    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, "Time Step", 0, var_data.shape[0] - 1, valinit=0, valfmt="%d")

    def update(val: float) -> None:
        idx = int(slider.val)
        u, v, colors = _build_quiver(idx)
        quiver.set_UVC(u, v)
        quiver.set_color(colors)
        title.set_text(f"Time: {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
