import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file('test_data/geometries.geojson')
gdf = gdf.to_crs(epsg=9377)
print(gdf)


# --- 5. Plot All Geometries with Differentiation ---
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the base geometries (all of them)
gdf.plot(ax=ax, cmap='Set3', edgecolor='black', alpha=0.5)

# Filter for the chosen geometries to overlay them with distinct styling

gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.5, hatch='//', label='Chosen')

# Add text labels for ID and Area on all geometries
for idx, row in gdf.iterrows():
    centroid = row['geometry'].centroid
    label = f"ID: {row['id']}\nArea: {row['area']:.1f} m²"

    # If the geometry is one of the chosen ones, make the text bold/red

    bbox_color = "mistyrose"
    text_color = "darkred"

    ax.annotate(
        text=label,
        xy=(centroid.x, centroid.y),
        ha="center",
        va="center",
        fontsize=9,
        color=text_color,
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc=bbox_color,
            ec="gray",
            lw=0.5,
            alpha=0.9,
        ),
    )

plt.title('Geometries with Chosen Indices Highlighted (Red/Hatched)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
