import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


def plot_probabilidad_encendido_2d(
    x_activos: np.ndarray,
    y_activos: np.ndarray,
    x_bounds: tuple,
    y_bounds: tuple,
    d_max: float,
    modo: str = "suave",
    temperatura: float = 0.1,
    grid_res: int = 200,
    ax=None,
):
    """Genera y grafica el mapa continuo de probabilidad de encendido (heatmap)

    alrededor de las turbinas activas existentes.

    Parámetros:
    -----------
    x_activos, y_activos : np.ndarray
        Coordenadas de las turbinas encendidas actualmente.
    x_bounds, y_bounds : tuple (min, max)
        Límites del terreno/mapa a evaluar (ej: (0, 1000), (0, 1000)).
    d_max : float
        Distancia umbral a partir de la cual la prioridad se estanca
        (saturación).
    modo : str ("uniforme" o "suave")
        Modo de asignación de probabilidades ('uniforme' o 'suave').
    temperatura : float
        Temperatura para el modo 'suave'.
    grid_res : int
        Resolución de la malla de evaluación (ej: 200x200).
    ax : matplotlib.axes.Axes, opcional
        Eje donde graficar. Si es None, crea una figura nueva.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # 1. Crear la malla de puntos (Grid 2D)
    gx = np.linspace(x_bounds[0], x_bounds[1], grid_res)
    gy = np.linspace(y_bounds[0], y_bounds[1], grid_res)
    GX, GY = np.meshgrid(gx, gy)
    grid_coords = np.column_stack((GX.ravel(), GY.ravel()))

    coords_activas = np.column_stack((x_activos, y_activos))

    if len(coords_activas) == 0:
        # Si no hay turbinas activas, la probabilidad es uniforme en todo el mapa
        probs_grid = np.ones(grid_coords.shape[0]) / grid_coords.shape[0]
    else:
        # 2. Distancia mínima desde cada punto de la malla a la turbina activa más cercana
        distancias = cdist(grid_coords, coords_activas, metric="euclidean")
        dist_minima = np.min(distancias, axis=1)

        # 3. Aplicar Clipping / Capping con d_max
        dist_clipeada = np.minimum(dist_minima, d_max)

        # 4. Calcular probabilidades
        if modo == "uniforme":
            suma_dist = np.sum(dist_clipeada)
            probs_grid = (
                dist_clipeada / suma_dist
                if suma_dist > 0
                else np.ones_like(dist_clipeada)
            )

        elif modo == "suave":
            exp_dist = np.exp(
                (dist_clipeada - np.max(dist_clipeada)) / temperatura
            )
            probs_grid = exp_dist / np.sum(exp_dist)

    # 5. Reestructurar las probabilidades a la forma de la matriz 2D (grid_res, grid_res)
    P_2D = probs_grid.reshape(grid_res, grid_res)

    # 6. Plotear Mapa de Calor y Turbinas
    contour = ax.contourf(
        GX,
        GY,
        P_2D,
        levels=50,
        cmap="YlOrRd",  # Amarillo (baja prob) -> Rojo (alta prob)
        extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
    )

    # Dibujar las turbinas activas
    if len(coords_activas) > 0:
        ax.scatter(
            x_activos,
            y_activos,
            c="cyan",
            edgecolors="black",
            linewidths=1.5,
            s=60,
            zorder=3,
            label="Turbinas Activas",
        )

    # Formato visual
    plt.colorbar(contour, ax=ax, label="Densidad de Probabilidad de Encendido")
    ax.set_title(
        f"Mapa de Probabilidades (Modo: '{modo}', $d_{{max}}$={d_max})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")

    return ax

def eliminate_points_mask_fast(x, y, mask, r):
    """Eliminates conflicting points in-place using boolean masking (1 to 0)

    instead of costly np.delete allocations.
    """
    final_mask = mask.copy()
    valid_indices = np.where(final_mask)[0]

    if len(valid_indices) <= 1:
        return final_mask

    # Extract coordinates for initial valid points
    pts = np.column_stack((x[valid_indices], y[valid_indices]))

    # Compute distance matrix once
    dist_matrix = cdist(pts, pts, metric='euclidean')

    # Self-distance set to infinity so points don't conflict with themselves
    np.fill_diagonal(dist_matrix, np.inf)

    # Track active status locally (1 = active, 0 = eliminated)
    active = np.ones(len(pts), dtype=bool)

    while True:
        # Find conflicts only among currently active points (< r)
        in_conflict = active & np.any((dist_matrix < r) & active, axis=1)

        if not np.any(in_conflict):
            break

        # Calculate sum of distances for active points
        # Set self-distance to 0 temporarily for sum calculation
        np.fill_diagonal(dist_matrix, 0)
        sums = np.sum(np.where(active, dist_matrix, 0), axis=1)
        np.fill_diagonal(dist_matrix, np.inf)

        # Select conflicting point with the SMALLEST sum of distances
        conflicting_indices = np.where(in_conflict)[0]
        idx_to_remove = conflicting_indices[np.argmin(
            sums[conflicting_indices])]

        # "Delete" by toggling active state from True (1) to False (0)
        active[idx_to_remove] = False

        # Invalidate its distances so it never conflicts again
        dist_matrix[idx_to_remove, :] = np.inf
        dist_matrix[:, idx_to_remove] = np.inf

    # Update final mask using active indices
    final_mask[valid_indices[~active]] = False
    return final_mask

def eliminate_points_step_by_step(x, y, r):
    """
    Iteratively removes points in proximity conflict (< r) with the smallest sum of distances.
    Returns:
        history: list of active points array at each iteration.
        removed_history: list of points removed up to each step.
    """
    pts = np.column_stack((x, y))
    history = [pts.copy()]
    removed_history = [[]]
    removed_so_far = []

    while len(pts) > 1:
        dist_matrix = cdist(pts, pts, metric='euclidean')

        # Mask diagonal (self-distance) with infinity to check conflicts
        np.fill_diagonal(dist_matrix, np.inf)
        in_conflict = np.any(dist_matrix < r, axis=1)

        # Stop condition: No points in conflict
        if not np.any(in_conflict):
            break

        # Re-fill diagonal with 0 to calculate correct sum of distances
        np.fill_diagonal(dist_matrix, 0)
        sums = np.sum(dist_matrix, axis=1)

        # Get indices of points in conflict
        conflicting_indices = np.where(in_conflict)[0]

        # Select conflicting point with the SMALLEST sum of distances
        idx_to_remove = conflicting_indices[np.argmin(
            sums[conflicting_indices])]

        # Record and remove
        removed_so_far.append(pts[idx_to_remove].copy())
        pts = np.delete(pts, idx_to_remove, axis=0)

        history.append(pts.copy())
        removed_history.append(np.array(removed_so_far.copy()))

    return history, removed_history

# -------------------------------------------------------------
# Example Data: N points and radius r
# -------------------------------------------------------------
np.random.seed(42)
N = 100
s = 20
x = np.random.uniform(0, s, N)
y = np.random.uniform(0, s, N)
mask = np.ones(len(x), dtype=np.bool)
r = 2.5

final_mask = eliminate_points_mask_fast(x, y, mask, r)
history, removed_history = eliminate_points_step_by_step(x, y, r)

# -------------------------------------------------------------
# Matplotlib Interactive Figure with Slider
# -------------------------------------------------------------
fig, (ax, ax2) = plt.subplots(1,2, figsize=(7, 7))
plt.subplots_adjust(bottom=0.2)

def draw_step(step):
    ax.clear()
    pts = history[step]
    rems = removed_history[step]
    ax.scatter(x[final_mask], y[final_mask], s=100, zorder=5, color='green', marker="+")
    # 1. Plot remaining points
    ax.scatter(pts[:, 0], pts[:, 1], color='crimson', s=80, zorder=4, label=f'Active Points ({len(pts)})')
    
    # 2. Draw radius circles around active points
    for p in pts:
        circ = patches.Circle((p[0], p[1]), radius=r, edgecolor='crimson',
                              facecolor='crimson', alpha=0.12, linestyle='--', linewidth=1.2, zorder=2)
        ax.add_patch(circ)
        
    # 3. Plot removed points as grey 'X's
    if len(rems) > 0:
        ax.scatter(rems[:, 0], rems[:, 1], color='gray', marker='x', s=80, linewidth=2, zorder=3, label='Eliminated Points')
        
    ax.set_title(f"Iteration {step} / {len(history)-1} | Remaining Points: {len(pts)}", fontsize=12, fontweight='bold')
    ax.set_xlim(-r, s + r)
    ax.set_ylim(-r, s + r)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

# Initial render
draw_step(0)

# Create Slider Widget
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Step', 0, len(history) - 1, valinit=0, valfmt='%d')

def update(val):
    step = int(slider.val)
    draw_step(step)
    fig.canvas.draw_idle()

slider.on_changed(update)

plot_probabilidad_encendido_2d(
    x_activos=x[final_mask], 
    y_activos=y[final_mask],
    x_bounds=(0, s),
    y_bounds=(0, s),
    d_max=20.0,  # Distancia límite de saturación
    modo="suave",  # 'suave' o 'uniforme'
    temperatura= 5.5,  # Controla la nitidez del pico de mayor distancia
    grid_res=250,
    ax=ax2,
)

plt.show()