import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.colors import ListedColormap
from wake_models import WakeModel, obtener_aerogenerador

# --- 1. Data & Calculations setup ---
hourly_ideal = np.load("/home/felipe/Desktop/Trabajo/WFL_scripts/test_scripts/experiments/ideal_kWh_hourly.npy")
monthly_ideal = np.load("/home/felipe/Desktop/Trabajo/WFL_scripts/test_scripts/experiments/ideal_kWh_monthly.npy")

n_horas_hourly = 78881
n_horas_monthly = 78888

turbine_availability  = 0.95
electrical_loss = 0.02
blade_soiling_erosion_loss = 0.01

factor = turbine_availability * (
    1.0-electrical_loss) * (1.0-blade_soiling_erosion_loss)

modelo = "Aeolos H-5kW"
aerogenerador = obtener_aerogenerador(modelo)

max_energy_hourly = aerogenerador.p_nominal_kW * n_horas_hourly
max_energy_monthly = aerogenerador.p_nominal_kW * n_horas_monthly

# --- 2. Figure and Layout Setup ---
fig = plt.figure(figsize=(14, 8))
plt.subplots_adjust(top=0.88, bottom=0.25, left=0.08, right=0.92, wspace=0.3)

# Setup custom colormap with BLACK for masked/filtered out values (-1)
cmap_base = plt.cm.viridis.copy()
cmap_cf = plt.cm.viridis.copy()
cmap_cf.set_under('black')  # Filtered values (-1) render as black

# --- 3. Axes Creation ---
# Tab 1 Axes (Base)
ax_base_h = fig.add_subplot(1, 2, 1)
ax_base_m = fig.add_subplot(1, 2, 2)

# Tab 2 Axes (CF)
ax_cf_h = fig.add_subplot(1, 2, 1)
ax_cf_m = fig.add_subplot(1, 2, 2)

# Set up initial images for TAB 1 (Base)
im_base_h = ax_base_h.imshow(hourly_ideal, cmap=cmap_base, origin="lower")
ax_base_h.set_title("Hourly Ideal Energy (kWh)")
cbar_bh = fig.colorbar(im_base_h, ax=ax_base_h, fraction=0.046, pad=0.04)

im_base_m = ax_base_m.imshow(monthly_ideal, cmap=cmap_base, origin="lower")
ax_base_m.set_title("Monthly Ideal Energy (kWh)")
cbar_bm = fig.colorbar(im_base_m, ax=ax_base_m, fraction=0.046, pad=0.04)

# Set up initial images for TAB 2 (CF)
# Initial parameter values
init_loss = 0.0
init_min_cf = 0.0
init_max_cf = 1.0


def compute_filtered_cf(ideal_data, max_energy, loss, cf_min, cf_max):
    cf = (ideal_data * factor * (1.0 - loss)) / max_energy
    # Mask out-of-bounds values by setting them to -1 (rendered black)
    filtered_cf = np.where((cf >= cf_min) & (cf <= cf_max), cf, -1.0)
    return filtered_cf


cf_h_data = compute_filtered_cf(hourly_ideal, max_energy_hourly, init_loss, init_min_cf, init_max_cf)
cf_m_data = compute_filtered_cf(monthly_ideal, max_energy_monthly, init_loss, init_min_cf, init_max_cf)

im_cf_h = ax_cf_h.imshow(cf_h_data, cmap=cmap_cf, vmin=0.0, vmax=1.0, origin="lower")
ax_cf_h.set_title("Hourly Capacity Factor (CF)")
cbar_cf_h = fig.colorbar(im_cf_h, ax=ax_cf_h, fraction=0.046, pad=0.04)

im_cf_m = ax_cf_m.imshow(cf_m_data, cmap=cmap_cf, vmin=0.0, vmax=1.0, origin="lower")
ax_cf_m.set_title("Monthly Capacity Factor (CF)")
cbar_cf_m = fig.colorbar(im_cf_m, ax=ax_cf_m, fraction=0.046, pad=0.04)

# Initially hide TAB 2 (CF)
ax_cf_h.set_visible(False)
ax_cf_m.set_visible(False)
cbar_cf_h.ax.set_visible(False)
cbar_cf_m.ax.set_visible(False)

# --- 4. Interactive Widgets Setup ---

# Tab Radio Selector at the top
ax_tab = plt.axes([0.40, 0.91, 0.20, 0.06], facecolor="lightgray")
tab_selector = RadioButtons(ax_tab, ("Base", "CF"), active=0)

# Sliders at the bottom for Tab 2
ax_loss = plt.axes([0.15, 0.14, 0.70, 0.03])
ax_cf_min = plt.axes([0.15, 0.09, 0.70, 0.03])
ax_cf_max = plt.axes([0.15, 0.04, 0.70, 0.03])

s_loss = Slider(ax_loss, "Loss", 0.0, 1.0, valinit=init_loss, valfmt="%.2f")
s_cf_min = Slider(ax_cf_min, "CF Min Limit", 0.0, 1.0, valinit=init_min_cf, valfmt="%.2f")
s_cf_max = Slider(ax_cf_max, "CF Max Limit", 0.0, 1.0, valinit=init_max_cf, valfmt="%.2f")

# Hide sliders by default on Tab 1
ax_loss.set_visible(False)
ax_cf_min.set_visible(False)
ax_cf_max.set_visible(False)

# --- 5. Callback Functions ---

def update_cf(val):
    loss = s_loss.val
    cf_min = s_cf_min.val
    cf_max = s_cf_max.val
    
    new_cf_h = compute_filtered_cf(hourly_ideal, max_energy_hourly, loss, cf_min, cf_max)
    new_cf_m = compute_filtered_cf(monthly_ideal, max_energy_monthly, loss, cf_min, cf_max)
    
    im_cf_h.set_data(new_cf_h)
    im_cf_m.set_data(new_cf_m)
    fig.canvas.draw_idle()

s_loss.on_changed(update_cf)
s_cf_min.on_changed(update_cf)
s_cf_max.on_changed(update_cf)


def switch_tab(label):
    if label == "Base":
        # Show Base, Hide CF
        ax_base_h.set_visible(True)
        ax_base_m.set_visible(True)
        cbar_bh.ax.set_visible(True)
        cbar_bm.ax.set_visible(True)
        
        ax_cf_h.set_visible(False)
        ax_cf_m.set_visible(False)
        cbar_cf_h.ax.set_visible(False)
        cbar_cf_m.ax.set_visible(False)
        
        ax_loss.set_visible(False)
        ax_cf_min.set_visible(False)
        ax_cf_max.set_visible(False)

    elif label == "CF":
        # Hide Base, Show CF
        ax_base_h.set_visible(False)
        ax_base_m.set_visible(False)
        cbar_bh.ax.set_visible(False)
        cbar_bm.ax.set_visible(False)
        
        ax_cf_h.set_visible(True)
        ax_cf_m.set_visible(True)
        cbar_cf_h.ax.set_visible(True)
        cbar_cf_m.ax.set_visible(True)
        
        ax_loss.set_visible(True)
        ax_cf_min.set_visible(True)
        ax_cf_max.set_visible(True)
        
    fig.canvas.draw_idle()

tab_selector.on_clicked(switch_tab)

plt.show()