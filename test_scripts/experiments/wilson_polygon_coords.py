import geopandas as gpd

# Load the file
gdf = gpd.read_file(
    "/home/felipe/Desktop/Trabajo/Data/T_EOLICO_Final/T_EOLICO_Final/Zona_F_Eolico_100m.shp"
)

# Target IDs
polygon_ids = [
    "e_100_10907",
    "e_100_8834",
    "e_100_2779",
    "e_100_5860",
    "e_100_7175",
]

# Filter rows (replace 'id' with your actual column name if different)
id_col = "id_poligon"
subset = gdf[gdf[id_col].isin(polygon_ids)].copy()

# Reproject to EPSG:4326 to get Latitude and Longitude
subset_wgs84 = subset.to_crs(epsg=4326)

for (idx, row), (_, row_wgs84) in zip(
    subset.iterrows(), subset_wgs84.iterrows()
):
    poly_id = row[id_col]

    # Native CRS calculations for bounds/dimensions
    geom = row.geometry
    centroid = geom.centroid
    minx, miny, maxx, maxy = geom.bounds
    width = maxx - minx
    height = maxy - miny

    # Geographic Centroid (WGS84)
    centroid_wgs84 = row_wgs84.geometry.centroid
    lon, lat = centroid_wgs84.x, centroid_wgs84.y

    print(f"ID: {poly_id}")
    print(f"  Centroid: X = {centroid.x:.4f}, Y = {centroid.y:.4f}")
    print(f"  Latitude:  {lat:.6f}")
    print(f"  Longitude: {lon:.6f}")
    print(
        f"  Bounds Size: Width = {width:.4f}, Height = {height:.4f} (native CRS units)"
    )
    print(
        f"  Extents: MinX={minx:.4f}, MinY={miny:.4f}, MaxX={maxx:.4f}, MaxY={maxy:.4f}"
    )
    print("-" * 50)