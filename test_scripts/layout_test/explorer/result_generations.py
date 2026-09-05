import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

# ga_results = pd.read_csv("/home/felipe/Desktop/Trabajo/Wind Farm Layout/results/ga_result_spacing_curve.csv")
ga_results = pd.read_csv("/home/felipe/Desktop/Trabajo/WFL_scripts/results/ga_result_1.csv")
# print(ga_results)
gdf = gpd.read_file("/home/felipe/Desktop/Trabajo/Wind Farm Layout/data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)

geometry_sb = gdf.geometry.iloc[33601] # e_10m_33602
geometry_sg = gdf.geometry.iloc[36598] # e_10m_36599
geometry_mb = gdf.geometry.iloc[34267]  # e_10m_34268##BAD
geometry_mg = gdf.geometry.iloc[36661]  # e_10m_36662##BUENO
geometry_bb = gdf.geometry.iloc[33929] # e_10m_33930
geometry_bg = gdf.geometry.iloc[36643] # e_10m_36644


def plot_interactive_dataset(history, x_pos, y_pos, polygon, spacing):
    """Plots a static baseline alongside a dynamic curve controlled by a Slider.

    Parameters:
    - x_data: 1D array/list for the x-axis.
    - static_data_list: List/array of y-values that do NOT change.
    - dynamic_datasets: List of 1D arrays/lists representing different
                        graph states corresponding to slider index positions.
    """
    # 1. Create main figure and split layout for plot + slider
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)  # Leave room at the bottom for slider
    ###
    patches = [
        Rectangle(
            xy=(x - spacing / 2, y - spacing / 2),
            width=spacing,
            height=spacing,
        )
        for x, y in zip(x_pos, y_pos)
    ]

    # 3. Bundle patches into a Collection for fast rendering
    pc = PatchCollection(
        patches, facecolor="skyblue", edgecolor="dodgerblue", alpha=0.7
    )
    ax.add_collection(pc)
    ###

    if polygon is not None:
        x_poly, y_poly = polygon.exterior.xy
        ax.plot(
            x_poly, y_poly, color="red", linewidth=2, label="Original Polygon"
        )

    initial_idx = 0
    mask = np.array(history[initial_idx]).astype(bool)

    final_x = x_pos[mask]
    final_y = y_pos[mask]
    scatter = ax.scatter(final_x, final_y, marker="x")


    # 4. Style the plot area
    ax.set_title("Evolution of algo")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")

    # 5. Define slider position axes: [left, bottom, width, height]
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])

    # 6. Instantiate the integer index Slider
    index_slider = Slider(
        ax=ax_slider,
        label="Index State",
        valmin=0,
        valmax=len(history) - 1,
        valinit=initial_idx,
        valstep=1,  # Snap to discrete integer indices
    )

    # 7. Define callback function triggered by slider changes
    def update(val):
        idx = int(index_slider.val)  # Get current slider index integer

        # Update ONLY the dynamic line data
        mask = np.array(history[idx]).astype(bool)
        final_x = x_pos[mask]
        final_y = y_pos[mask]
        ax.set_title(f"Evolution of algo {len(final_x)}")
        new_offsets = np.column_stack((final_x, final_y))

        # Update both X and Y coordinates at once
        scatter.set_offsets(new_offsets)

        # Redraw the canvas
        fig.canvas.draw_idle()

    # 8. Attach callback to slider event
    index_slider.on_changed(update)

    plt.show()


import ast
configs = ga_results["config"].unique()
def show_ga_history(geom, polygon, config, fn):

    filtered_df = ga_results[
        (ga_results["polygon"] == polygon) & (ga_results["config"] == config) & (ga_results["fn"] == fn)
    ]
    print(filtered_df["history"].values)
    cromosome = ast.literal_eval(filtered_df["history"].values[0])
    xs = np.array(ast.literal_eval(filtered_df["x_positions"].values[0]))
    ys = np.array(ast.literal_eval(filtered_df["y_positions"].values[0]))
    spacing = filtered_df["spacing"].values[0]
    print("SPACING", spacing)
    plot_interactive_dataset(
        history=cromosome, x_pos=xs, y_pos=ys, spacing=spacing*6.4, polygon=geom
    )
show_ga_history(geometry_mb, "geometry_mb", configs[0], "max_fit_2")
