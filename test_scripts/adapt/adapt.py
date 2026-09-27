import geopandas as gpd
from lcoe import calculate_lcoe
from wake_models import WakeModel, obtener_aerogenerador
import numpy as np
from wind_dataset import WindDataset, DatasetParams
from wake_models import WakeModel

import xarray as xr

ds_params = DatasetParams(
    nc_path="/home/felipe/Desktop/Trabajo/Data/Newdataq/wind_data_10m(1).nc",
    easting_coord="Easting", northing_coord="Northing", ws_var="WS",
    time_dim="time", time_res="monthly")

wind_ds = WindDataset(params=ds_params)

# ========PARAMS============
wake_k = 0.075
thrust_ct = 0.8
max_wake_distance_D = 20.0
# ========WAKEMODEL=========
modelo = "Aeolos H-5kW"
# modelo = "Vestas V100/2000"
aerogenerador = obtener_aerogenerador(modelo)

wake_model = WakeModel(
    aerogenerator=aerogenerador,
    wake_k=wake_k,
    thrust_ct=thrust_ct,
    max_wake_distance_D=max_wake_distance_D,
)

wind_ds.fill_cache(wind_ds.ds.rio.bounds())
wind_ds.set_cache_height_factor(12.0)
wind_ds.set_cache_density_factor()

x = np.array([5017056.6878    , 5017075.8878    , 5017095.0878    ,
       5017115.8878    , 5016960.6878    , 5016979.8878    ,
       5016999.0878    , 5017018.2878    , 5017037.4878    ,
       5017056.6878    , 5017077.4878    , 5016897.81917085,
       5016922.2878    , 5016941.4878    , 5016960.6878    ,
       5016979.8878    , 5016999.0878    , 5017018.2878    ,
       5017037.4878    , 5017058.9505417 , 5016883.8878    ,
       5016903.0878    , 5016922.2878    , 5016941.4878    ,
       5016960.6878    , 5016979.8878    , 5016999.0878    ,
       5017018.2878    , 5017037.4878    , 5017061.95642915,
       5017082.2878    , 5016864.6878    , 5016882.75642915,
       5016903.0878    , 5016845.4878    , 5016863.55642915,
       5016883.8878    , 5016826.2878    , 5016845.4878    ,
       5016864.6878    , 5016885.4878    ])
y = np.array([2768609.5373    , 2768604.7373    , 2768601.5373    ,
       2768599.9373    , 2768649.5373    , 2768647.9373    ,
       2768647.9373    , 2768647.9373    , 2768647.9373    ,
       2768646.3373    , 2768644.7373    , 2768691.60592915,
       2768692.7373    , 2768692.7373    , 2768692.7373    ,
       2768692.7373    , 2768692.7373    , 2768692.7373    ,
       2768692.7373    , 2768690.4745583 , 2768743.9373    ,
       2768739.1373    , 2768737.5373    , 2768737.5373    ,
       2768737.5373    , 2768737.5373    , 2768737.5373    ,
       2768737.5373    , 2768737.5373    , 2768738.66867085,
       2768737.5373    , 2768788.7373    , 2768781.20592915,
       2768779.1373    , 2768833.5373    , 2768826.00592915,
       2768823.9373    , 2768878.3373    , 2768873.5373    ,
       2768870.3373    , 2768868.7373    ])

res = {
    "x": x,
    "y": y,
    "n_turbinas": len(x),
    "start_idx": 4,
    "end_idx": 100,
    "years" : 8,
    "lcoe" : -1
}

from ubicar_turbinas.results_formatter import get_detailed_result
from pprint import pprint
from data_export import export_geojson_park
from dataclasses import asdict
dr = get_detailed_result(x=x, y=y, wake_model=wake_model, wind_ds=wind_ds)
# try:
# except Exception as e:
#     print(e)

export_geojson_park(
                    x,
                    y,
                    metadata=asdict(dr),
                    name=f"adapt",
                    output_dir="/home/felipe/Desktop/Trabajo/WFL_scripts/results/adapt",
                )

#===
# ws, wd = wind_ds.get_wind_data_from_cache(x, y)

# ws = ws[4:-4]
# wd = wd[4:-4]
# ts = wind_ds.ts[4:-4]
# # print(wind_ds.ds[wind_ds.time_dim].values[4:-4])
# _, aep_wake_turbines, _ = wake_model.calc_wake_on_turbines_detailed(
#     x, y, ws, wd, ts
# )

# # Métricas energéticas
# generated_energy = aep_wake_turbines.sum()
# max_aep = (
#     wake_model.aerogenerator.p_nominal_kW * ts.sum() *
#     len(x))

# cf = (generated_energy / max_aep) * 100.0  # En porcentaje (%)
# potencia_mW = (
#     (aep_wake_turbines / ts[:, np.newaxis]).mean(axis=0).sum()
#     / 1000.0
# )

# print("max_aep", max_aep)
# print("cf", cf)
# print("pot mw", potencia_mW)
# print(wind_ds.ds)
# # Cálculo de LCOE
# lcoe_res = calculate_lcoe(
#     "la guajira", potencia_mW, len(x),
#     FCap=cf)
# lcoe_val = lcoe_res["result"]["LCOE"]

