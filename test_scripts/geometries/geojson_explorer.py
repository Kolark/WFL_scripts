import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file('test_data/Poligono_de_estudio/poligono_de_estudio.shp')
gdf = gdf.to_crs(epsg=9377)

# --- 2. Explode MultiPolygon into individual Polygons ---
gdf = gdf.explode(index_parts=False).reset_index(drop=True)

# --- 3. Format columns and calculate Area ---
gdf['id'] = gdf.index + 1
gdf['area'] = gdf.geometry.area
gdf = gdf[['geometry', 'id', 'area']]

# --- 4. Define Chosen Indices ---
# Note: Python uses 0-based indexing. Index 0 corresponds to ID 1, etc.
chosen_indices = [4, 5, 10 ,13, 22]
# chosen_indices = [5, 22]

# --- 5. Plot All Geometries with Differentiation ---
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the base geometries (all of them)
gdf.plot(ax=ax, cmap='Set3', edgecolor='black', alpha=0.5)
print(gdf)
# Filter for the chosen geometries to overlay them with distinct styling
chosen_gdf = gdf.iloc[chosen_indices]
chosen_gdf.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2.5, hatch='//', label='Chosen')

# Add text labels for ID and Area on all geometries
for idx, row in gdf.iterrows():
    centroid = row['geometry'].centroid
    label = f"ID: {row['id']}\nArea: {row['area']:.1f} m²"
    
    # If the geometry is one of the chosen ones, make the text bold/red
    is_chosen = idx in chosen_indices
    bbox_color = "mistyrose" if is_chosen else "white"
    text_color = "darkred" if is_chosen else "black"
    
    ax.annotate(text=label, xy=(centroid.x, centroid.y), 
                ha='center', va='center', fontsize=9, color=text_color, weight='bold' if is_chosen else 'normal',
                bbox=dict(boxstyle="round,pad=0.3", fc=bbox_color, ec="gray", lw=0.5, alpha=0.9))

plt.title('Geometries with Chosen Indices Highlighted (Red/Hatched)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# --- 6. Export ONLY Chosen Geometries to GeoJSON ---
chosen_gdf.to_file('test_data/geometries.geojson', driver='GeoJSON')