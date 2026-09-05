"""
Visor de resultados del Algoritmo Genético — Wind Farm Layout
================================================================
Script plano de matplotlib (sin Streamlit). Escoge polygon/fn/config
manualmente editando las variables de abajo, y corre el script.

Ejecutar con:
    uv run --with pandas --with numpy --with matplotlib --with shapely python ga_results_viewer.py
"""

import ast
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from shapely.plotting import plot_polygon

# ----------------------------------------------------------------------------
# Config — edita esto manualmente
# ----------------------------------------------------------------------------
CSV_PATH = "/home/felipe/Desktop/Trabajo/WFL_scripts/results/ga_result_1.csv"

POLYGON_SEL = "mg"   # <-- cambia aquí
FN_SEL = "max_fit_7"              # <-- cambia aquí
CONFIG_SEL = "ga_config_2"      # <-- cambia aquí

# TODO: reemplaza esto por tu diccionario real {polygon_name: shapely.Polygon}.
# Por ejemplo, si ya lo tienes en un pickle:
#   import pickle
#   GEOMETRIES = pickle.load(open("/ruta/a/geometrias.pkl", "rb"))
# GEOMETRIES: dict = {}
import geopandas as gpd
gdf = gpd.read_file("/home/felipe/Desktop/Trabajo/Wind Farm Layout/data/eolico_10_m/Zona_F_Eolico_10m.shp")
GEOMETRIES: dict =  {
    "sb": gdf.geometry.iloc[33601], # e_10m_33602,
    "sg": gdf.geometry.iloc[36598], # e_10m_36599,
    "mb": gdf.geometry.iloc[34267], #e_10m_34268##BAD,
    "mg": gdf.geometry.iloc[36661], #e_10m_36662##BUENO,
    "bb": gdf.geometry.iloc[33929], # e_10m_33930,
    "bg": gdf.geometry.iloc[36643], # e_10m_36644,
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def parse_array(s: str) -> np.ndarray:
    """
    Convierte strings de arreglos guardados en el CSV a np.ndarray.
    Soporta:
      - formato Python normal con comas: "[0, 1, 0]"
      - repr de numpy separado por espacios (incluye 2D con saltos de línea)
      - repr de numpy>=2.0 con escalares tipo np.float64(0.123) / np.int64(1)
    """
    s = s.strip()

    # numpy>=2.0 stringifica escalares como np.float64(0.123) / np.int64(1),
    # lo que rompe ast.literal_eval (son llamadas a función, no literales).
    # Los "desenvolvemos" a solo el número: np.float64(0.123) -> 0.123
    s = re.sub(r"np\.\w+\(([^()]+)\)", r"\1", s)

    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # Inserta comas entre números/corchetes separados solo por espacios
        s2 = re.sub(r"(?<=[\d\.\]eE\+\-])\s+(?=[\d\.\-\+\[])", ",", s)
        s2 = re.sub(r",\s*,", ",", s2)
        val = ast.literal_eval(s2)
    return np.array(val, dtype=float)


# ----------------------------------------------------------------------------
# Carga y selección de fila
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df_f = df[
    (df["polygon"] == POLYGON_SEL)
    & (df["fn"] == FN_SEL)
    & (df["config"] == CONFIG_SEL)
]

if len(df_f) == 0:
    raise ValueError(
        f"No hay filas para polygon={POLYGON_SEL!r}, fn={FN_SEL!r}, config={CONFIG_SEL!r}.\n"
        f"Valores disponibles:\n"
        f"  polygon: {sorted(df['polygon'].unique())}\n"
        f"  fn: {sorted(df['fn'].unique())}\n"
        f"  config: {sorted(df['config'].unique())}"
    )
if len(df_f) > 1:
    print(f"Aviso: hay {len(df_f)} filas para esta combinación; se usa la primera.")

row = df_f.iloc[0]

print(f"AEP (wake):  {row['aep_wake']:.3f}")
print(f"Park CF:     {row['park_cf']:.4f}")
print(f"Park Loss:   {row['park_loss']:.4f}")
print(f"N turbinas:  {int(row['n_turbinas'])}")

# ----------------------------------------------------------------------------
# Parseo de arreglos
# ----------------------------------------------------------------------------
fitness_history = parse_array(row["fitness_history"])
history = parse_array(row["history"])  # (n_gen, n_posiciones_candidatas)
x_positions = parse_array(row["x_positions"])
y_positions = parse_array(row["y_positions"])
best_x = parse_array(row["best_result_x"])
best_y = parse_array(row["best_result_y"])

if history.ndim == 1:
    history = history.reshape(1, -1)

n_gen = history.shape[0]

# ----------------------------------------------------------------------------
# Figura: fitness (izquierda) + layout (derecha), con slider de generación
# ----------------------------------------------------------------------------
fig, (ax_fit, ax_layout) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.2)

fig.suptitle(f"polygon={POLYGON_SEL} | fn={FN_SEL} | config={CONFIG_SEL}")

# --- Panel de fitness ---
ax_fit.plot(fitness_history, color="tab:blue")
ax_fit.set_xlabel("Generación")
ax_fit.set_ylabel("Fitness")
ax_fit.grid(alpha=0.3)
vline = ax_fit.axvline(n_gen - 1, color="red", linestyle="--", alpha=0.7)

# --- Panel de layout ---
geom = GEOMETRIES.get(POLYGON_SEL)
if geom is not None:
    plot_polygon(geom, ax=ax_layout, add_points=False, facecolor="none", edgecolor="black")
else:
    print(f"Aviso: no se encontró geometría para polygon={POLYGON_SEL!r} en GEOMETRIES.")

ax_layout.scatter(x_positions, y_positions, s=15, color="lightgray", label="Candidatas")

mask0 = history[-1].astype(int).astype(bool)
active_scatter = ax_layout.scatter(
    x_positions[mask0], y_positions[mask0],
    s=40, color="tab:green", label=f"Activas (gen {n_gen - 1})",
)

ax_layout.scatter(
    best_x, best_y, s=70, facecolors="none", edgecolors="tab:red",
    linewidths=1.5, label="best_result",
)

ax_layout.set_aspect("equal")
ax_layout.legend(loc="best", fontsize=8)

# --- Slider de generación (modula history -> layout) ---
if n_gen > 1:
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Generación", 0, n_gen - 1, valinit=n_gen - 1, valstep=1)

    def update(val):
        gen_idx = int(slider.val)
        mask = history[gen_idx].astype(int).astype(bool)
        if len(mask) == len(x_positions):
            active_scatter.set_offsets(np.column_stack([x_positions[mask], y_positions[mask]]))
            active_scatter.set_label(f"Activas (gen {gen_idx})")
            ax_layout.legend(loc="best", fontsize=8)
        vline.set_xdata([gen_idx, gen_idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)
else:
    print("Solo hay una generación registrada en 'history' para esta fila.")

plt.show()