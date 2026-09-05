import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from scipy.spatial.distance import cdist

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
x = np.random.uniform(0, 100, N)
y = np.random.uniform(0, 100, N)
r = 2.5

history, removed_history = eliminate_points_step_by_step(x, y, r)

# -------------------------------------------------------------
# Matplotlib Interactive Figure with Slider
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
plt.subplots_adjust(bottom=0.2)

def draw_step(step):
    ax.clear()
    pts = history[step]
    rems = removed_history[step]
    
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
    ax.set_xlim(-r, 100 + r)
    ax.set_ylim(-r, 100 + r)
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
plt.show()