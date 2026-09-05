import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import MultiPoint
from tqdm import tqdm
import input_data
from wake_models.wake_metrics import get_park_cf, get_park_loss
from layout_optimization.grid_search import GridSearchOptimizer
from layout_optimization.fitness_funcs import make_min_multivar_fn
from monitor import Monitor

geometries = {
    "geometry_sb": input_data.geometries["geometry_sb"],
    "geometry_sg": input_data.geometries["geometry_sg"],
    "geometry_mb": input_data.geometries["geometry_mb"],
    "geometry_mg": input_data.geometries["geometry_mg"],
}

total_iterations = len(geometries)
records = []

with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for i, (g_name, g) in enumerate(geometries.items()):
        m = Monitor()
        gso = GridSearchOptimizer(
            wind_data=input_data.wind_ds,
            wake_model=input_data.wake_model,
            fitness_func=make_min_multivar_fn(0.0, 1.0, 0.0),
            monitor=m,
        )
        _ = gso.optimize(geometry=g).copy()

        for r, (angle, long, lat) in zip(m.results_history, gso.params_combinations):
            max_aep = (
                r.p_nominal_kW
                * r.n_turbinas
                * r.n_horas
            )
            total_aep_ideal = r.aep_ideal.sum()
            total_aep_wake = r.aep_wake.sum()

            # Creamos la colección de puntos (MultiPoint) con todas las turbinas
            points = list(zip(r.x.tolist(), r.y.tolist()))

            records.append(
                {
                    "angle": angle,
                    "long": long,
                    "lat": lat,
                    "polygon": g_name,
                    "fitness": r.fitness,
                    "aep_wake": total_aep_wake,
                    "park_cf": get_park_cf(total_aep_wake, max_aep),
                    "park_loss": get_park_loss(
                        total_aep_ideal, total_aep_wake
                    ),
                    "geometry": MultiPoint(
                        points
                    ),  # Una sola geometría por parque
                }
            )
        pbar.update(1)

# Conversión a GeoDataFrame
gdf_results = GeoDataFrame(records, crs="EPSG:9377")

# Exportar a GeoJSON
gdf_results.to_file("results/gridsearch_result.geojson", driver="GeoJSON")

# Opción comentada para exportar como GeoPackage (GPKG):
# gdf_results.to_file("results/gridsearch_result.gpkg", driver="GPKG", layer="turbines")
