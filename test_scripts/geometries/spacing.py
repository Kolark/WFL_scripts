import geopandas as gpd
import math
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry.base import BaseGeometry
import matplotlib.pyplot as plt
import math

gdf = gpd.read_file("data/eolico_10_m/Zona_F_Eolico_10m.shp")
gdf = gdf.to_crs(epsg=9377)

diametro = 6.4
p_nominal = 5
t_max = math.floor(1000/p_nominal)

x_min = 5
x_max = 20
k = 1000

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

def x(t, area, D):
    return math.sqrt(area/(t*D*D))

def get_polygon_pixel_count(
    polygon: BaseGeometry, pixel_size: float, CRS: str = "EPSG:3857"
) -> int:
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

    count = int(np.count_nonzero(raster_array == 1))

    # Optional fallback: If the polygon is too small to cover a pixel center,
    # default to 1 pixel (representing the centroid fallback from your original code).
    return max(count, 1) if count == 0 and not polygon.is_empty else count

def get_spacing(geometry, xmin, xmax, k, D, tmax):
    if(get_polygon_pixel_count(polygon=geometry, pixel_size=xmax*D) >= tmax):
        return xmax, 0.0
    else:
        xk = x(t=k, area=geometry.area, D=D)
        if xk > x_min:
            return xk, 1.0
        else:
            return xmin, 2.0

# def get_spacing(geometry, xmin, xmax, k, D, tmax):
#     if(get_polygon_pixel_count(polygon=geometry, pixel_size=xmax*D) >= tmax):
#         return xmax, 0.0
#     else:
#         xtmax = x(t_max, geometry.area, D=D)
#         e = ((xtmax-xmin)**2)/(xmax-xmin)
#         return xmin + e, 1.0
        # return max(x_min,x(k, geometry.area, D=D)), False

# geometry
geometryA = gdf.geometry.iloc[33929] #e_10m_33930
# geometryB = gdf.geometry.iloc[36663] #e_10m_36664


# fig, ax = plt.subplots(figsize=(8, 6))

# diametros = np.arange(5, 100, 0.25)

# spacings = []
# vs = []
# t_results = []
# for d in diametros:
#     s, v = get_spacing(geometry=geometryA, xmin=x_min, xmax=x_max, k=k, D=d, tmax=t_max)
#     spacings.append(s)
#     vs.append(v)
#     t_results.append(get_polygon_pixel_count(polygon=geometryA, pixel_size=s*d))

# print(geometryA.area)
# scatter = ax.scatter(diametros, t_results, c=vs)
# # scatter = ax.scatter(spacings, t_results, c=diametros)

# # Infinite horizontal line at y = 20
# ax.axhline(y=t_max, color='r', linestyle='--')

# # Infinite vertical line at x = 2
# ax.axvline(x=5, color='b', linestyle='-')
# ax.axvline(x=20, color='b', linestyle='-')
# fig.colorbar(scatter, ax=ax, label='Color Scale Value')
# plt.show()
#============================================
import matplotlib.pyplot as plt
import numpy as np

# --- Data Collection (existing loop) ---
diametros = np.arange(5, 100, 0.25)
spacings = []
vs = []
t_results = []

for d in diametros:
    s, v = get_spacing(geometry=geometryA, xmin=x_min, xmax=x_max, k=k, D=d, tmax=t_max)
    spacings.append(s)
    vs.append(v)
    t_results.append(get_polygon_pixel_count(polygon=geometryA, pixel_size=s * d))

# Convert list results to numpy arrays for easier plane generation
spacings = np.array(spacings)
t_results = np.array(t_results)
vs = np.array(vs)

# --- 3D Plotting ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D Scatter plot (X: Diametro, Y: Spacing, Z: Pixel Count, C: v)
scatter = ax.scatter(diametros, spacings, t_results, c=vs, cmap='viridis', s=25, edgecolor='k', linewidth=0.3)

# 1. Equivalent of axhline(y=t_max): Horizontal Plane at Z = t_max
d_grid, s_grid = np.meshgrid(
    np.linspace(diametros.min(), diametros.max(), 10),
    np.linspace(spacings.min(), spacings.max(), 10)
)
z_grid = np.full_like(d_grid, t_max)
ax.plot_surface(d_grid, s_grid, z_grid, color='r', alpha=0.3)

# 2. Equivalent of axvline(x=5 and x=20): Vertical Planes at X = 5 and X = 20
s_grid_v, z_grid_v = np.meshgrid(
    np.linspace(spacings.min(), spacings.max(), 10),
    np.linspace(t_results.min(), t_results.max(), 10)
)

for x_val in [5, 20]:
    x_grid_v = np.full_like(s_grid_v, x_val)
    ax.plot_surface(x_grid_v, s_grid_v, z_grid_v, color='b', alpha=0.2)

# Axis Labels
ax.set_xlabel('Diámetro (D)', labelpad=10)
ax.set_ylabel('Spacing (s)', labelpad=10)
ax.set_zlabel('Pixel Count (t)', labelpad=10)

# Colorbar for 'v'
cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
cbar.set_label('Color Scale Value (v)')

plt.tight_layout()
plt.show()

#=============================================================================
# import matplotlib.pyplot as plt
# import numpy as np

# # --- Parameter Grids ---
# diametros = np.arange(5, 100, 1)      # D values
# x_values = np.linspace(x_min, x_max, 50)  # x values from 5 to 20

# # Create 2D meshgrid
# D_grid, X_grid = np.meshgrid(diametros, x_values)

# # Initialize output matrices matching grid dimensions
# Z_grid = np.zeros_like(D_grid) # Pixel count t
# V_grid = np.zeros_like(D_grid) # Color value v

# # Evaluate function over the entire (D, X) parameter space
# for i in range(D_grid.shape[0]):
#     for j in range(D_grid.shape[1]):
#         d_val = D_grid[i, j]
#         x_val = X_grid[i, j]
        
#         # Calculate pixel count using x * d directly
#         pixel_size = x_val * d_val
#         Z_grid[i, j] = get_polygon_pixel_count(polygon=geometryA, pixel_size=pixel_size)
        
#         # Determine v status logic
#         s_calc, v_calc = get_spacing(geometry=geometryA, xmin=x_min, xmax=x_max, k=k, D=d_val, tmax=t_max)
#         V_grid[i, j] = v_calc

# # --- 3D Plotting ---
# fig = plt.figure(figsize=(11, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Map colors using V_grid on the surface
# norm = plt.Normalize(V_grid.min(), V_grid.max())
# colors = plt.cm.viridis(norm(V_grid))

# # 3D Surface Plot (X: Diametro, Y: Spacing x, Z: Pixel Count)
# surf = ax.plot_surface(
#     D_grid, X_grid, Z_grid, 
#     facecolors=colors, 
#     rstride=1, cstride=1, 
#     antialiased=True, 
#     alpha=0.85
# )

# # 1. Reference Plane at Z = t_max
# z_plane = np.full_like(D_grid, t_max)
# ax.plot_surface(D_grid, X_grid, z_plane, color='r', alpha=0.35)

# # Axis Labels
# ax.set_xlabel('Diámetro (D)', labelpad=10)
# ax.set_ylabel('Spacing Factor (x)', labelpad=10)
# ax.set_zlabel('Pixel Count (t)', labelpad=10)

# # Colorbar for 'v'
# mappable = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
# mappable.set_array(V_grid)
# cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7)
# cbar.set_label('Color Scale Value (v)')

# plt.tight_layout()
# plt.show()