import dash
from dash import dcc, html, Input, Output
import geopandas as gpd
import plotly.graph_objects as go
import input_data

# 1. Cargar la capa de MultiPoints desde el GeoJSON
gdf_turbines = gpd.read_file("results/gridsearch_result.geojson")

geometries = {
    "geometry_sb": input_data.geometries["geometry_sb"],
    "geometry_sg": input_data.geometries["geometry_sg"],
    "geometry_mb": input_data.geometries["geometry_mb"],
    "geometry_mg": input_data.geometries["geometry_mg"],
}

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "padding": "20px", "maxWidth": "1000px", "margin": "0 auto"},
    children=[
        html.H2("Visualizador de Disposición de Turbinas (GridSearch)"),
        
        html.Div(
            style={"display": "flex", "gap": "20px", "marginBottom": "20px"},
            children=[
                html.Div([
                    html.Label("Polígono:"),
                    dcc.Dropdown(
                        id="select-polygon",
                        options=[{"label": k, "value": k} for k in geometries.keys()],
                        value="geometry_sb",
                        clearable=False,
                    ),
                ], style={"flex": "1"}),
                
                html.Div([
                    html.Label("Métrica:"),
                    dcc.Dropdown(
                        id="select-metric",
                        options=[
                            {"label": "AEP Wake", "value": "aep_wake"},
                            {"label": "Park CF", "value": "park_cf"},
                            {"label": "Park Loss", "value": "park_loss"},
                        ],
                        value="aep_wake",
                        clearable=False,
                    ),
                ], style={"flex": "1"}),
                
                html.Div([
                    html.Label("Criterio:"),
                    dcc.Dropdown(
                        id="select-criterion",
                        options=[
                            {"label": "Valor más Alto (Max)", "value": "max"},
                            {"label": "Valor más Bajo (Min)", "value": "min"},
                        ],
                        value="max",
                        clearable=False,
                    ),
                ], style={"flex": "1"}),
            ],
        ),
        
        # Tarjeta para métricas y parámetros numéricos
        html.Div(
            id="metrics-display",
            style={
                "padding": "15px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "8px",
                "border": "1px solid #e9ecef",
                "marginBottom": "20px",
            },
        ),
        
        dcc.Graph(id="layout-map", style={"height": "600px"}),
    ]
)


@app.callback(
    [Output("layout-map", "figure"), Output("metrics-display", "children")],
    [
        Input("select-polygon", "value"),
        Input("select-metric", "value"),
        Input("select-criterion", "value"),
    ],
)
def update_map(selected_polygon, selected_metric, selected_criterion):
    df_poly = gdf_turbines[gdf_turbines["polygon"] == selected_polygon]

    if df_poly.empty:
        return go.Figure(), "No hay datos disponibles para este polígono."

    # Obtener el registro que tiene el valor min/max objetivo
    target_val = (
        df_poly[selected_metric].max()
        if selected_criterion == "max"
        else df_poly[selected_metric].min()
    )

    # Seleccionamos la primera fila coincidente
    selected_run = df_poly[df_poly[selected_metric] == target_val].iloc[0]

    # Extraer métricas y parámetros
    aep_wake_val = selected_run["aep_wake"]
    park_cf_val = selected_run["park_cf"]
    park_loss_val = selected_run["park_loss"]
    
    # Parámetros agregados recientemente (usamos .get() o acceso directo)
    angle_val = selected_run.get("angle", "N/A")
    long_val = selected_run.get("long", "N/A")
    lat_val = selected_run.get("lat", "N/A")

    fig = go.Figure()

    # 1. Dibujar el Polígono de la parcela
    poly_geom = geometries[selected_polygon]
    geoms = poly_geom.geoms if hasattr(poly_geom, "geoms") else [poly_geom]
    for geom in geoms:
        x_poly, y_poly = geom.exterior.xy
        fig.add_trace(
            go.Scatter(
                x=list(x_poly),
                y=list(y_poly),
                fill="toself",
                mode="lines",
                name=f"Límite ({selected_polygon})",
                line=dict(color="#2b5c8f", width=2),
                fillcolor="rgba(43, 92, 143, 0.1)",
            )
        )

    # 2. Extraer las coordenadas del MultiPoint
    multipoint_geom = selected_run.geometry
    x_coords = [pt.x for pt in multipoint_geom.geoms]
    y_coords = [pt.y for pt in multipoint_geom.geoms]
    labels = [f"T-{i+1}" for i in range(len(x_coords))]

    # Dibujar las turbinas con información extendida en el hover
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=10, color="#d9534f", symbol="circle"),
            name="Turbinas",
            customdata=[[angle_val, long_val, lat_val]] * len(x_coords),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "X: %{x:.2f}<br>"
                "Y: %{y:.2f}<br>"
                "<b>Ángulo:</b> %{customdata[0]}°<br>"
                "<b>Longitud / Dist. X:</b> %{customdata[1]}<br>"
                "<b>Latitud / Dist. Y:</b> %{customdata[2]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Layout Turbinas - {selected_polygon} ({selected_metric.upper()} {selected_criterion.upper()})"
        ),
        xaxis_title="Coordenada X",
        yaxis_title="Coordenada Y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    # Renderizar el panel informativo de métricas + parámetros de la grilla
    metrics_text = html.Div(
        children=[
            html.H4("Resultados y Configuración Seleccionada:", style={"marginTop": 0}),
            html.Div(
                style={"display": "flex", "gap": "40px"},
                children=[
                    html.Div([
                        html.H5("Métricas del Parque", style={"marginBottom": "8px", "color": "#2b5c8f"}),
                        html.P([html.Strong("AEP Wake: "), f"{aep_wake_val:,.2f} kWh"]),
                        html.P([html.Strong("Park Capacity Factor (CF): "), f"{park_cf_val * 100:.2f}%"]),
                        html.P([html.Strong("Park Loss: "), f"{park_loss_val * 100:.2f}%"]),
                    ]),
                    html.Div([
                        html.H5("Parámetros de Grilla", style={"marginBottom": "8px", "color": "#2b5c8f"}),
                        html.P([html.Strong("Angle: "), f"{angle_val}°" if isinstance(angle_val, (int, float)) else f"{angle_val}"]),
                        html.P([html.Strong("Long: "), f"{long_val}"]),
                        html.P([html.Strong("Lat: "), f"{lat_val}"]),
                    ]),
                ]
            )
        ]
    )

    return fig, metrics_text


if __name__ == "__main__":
    app.run(debug=True)