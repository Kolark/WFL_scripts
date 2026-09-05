import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from shapely.plotting import plot_polygon

import matplotlib.pyplot as plt
import matplotlib.patches as patches


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_squares_around_points(x_coords, y_coords, square_size, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    offset = square_size / 2

    for x, y in zip(x_coords, y_coords):
        square = patches.Rectangle(
            (x - offset, y - offset),
            square_size,
            square_size,
            linewidth=1.5,
            edgecolor="blue",
            facecolor="none",
        )
        ax.add_patch(square)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)

    return ax


def plot_result(geometry, x, y, title, colors, ax=None, cmap="viridis"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    plot_polygon(
        geometry,
        ax=ax,
        add_points=False,
        facecolor="lightblue",
        edgecolor="blue",
        alpha=0.4,
        label="Farm Boundary",
    )

    sc = ax.scatter(
        x,
        y,
        c=colors,
        cmap=cmap,
        marker="o",
        s=100,
        zorder=3,
        label="Turbines",
    )

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("X Coordinate (m)")
    ax.set_ylabel("Y Coordinate (m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    fig = ax.get_figure()
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Turbine Value/Hue")

    if ax is None:
        plt.show()
