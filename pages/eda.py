"""All exploratory views share one explicit filtered cohort."""
import math

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, callback, dcc, html
import numpy as np

from e3_dashboard.data import filter_observations, load_bundle
from e3_dashboard import plots

dash.register_page(__name__, path="/eda", name="EDA", title="EDA · Volatilidad BTC")
VARIABLES = {
    "last_close": "Último cierre conocido", "last_open": "Última apertura conocida",
    "last_high": "Último máximo conocido", "last_low": "Último mínimo conocido",
    "last_volume": "Último volumen conocido", "known_return6": "Último retorno conocido",
    "known_floor": "Piso observable de σ₇",
}


def _chart(graph_id):
    return html.Div([dcc.Graph(id=graph_id, config={"displaylogo": False}),
                     html.P(id=graph_id + "-caption", className="figure-insight")])


def layout():
    rows = load_bundle()["observations"]
    dates = sorted(r["target_date"] for r in rows)
    largest = max(float(r["target"]) for r in rows)
    upper = math.ceil(largest * 1000) / 1000
    return dbc.Container([
        html.Header([html.P("EDA · EXPLORACIÓN DE DATOS", className="eyebrow"), html.H1("Una selección, todas las vistas"),
                     html.P("Explora el objetivo observado y la información disponible antes de cada fecha. Los filtros cambian todos los gráficos e interpretaciones de esta página.", className="page-lead")], className="page-header"),
        html.Section([
            dbc.Row([
                dbc.Col([html.Label("Cohortes", htmlFor="eda-partitions", className="control-label"),
                         dcc.Dropdown(id="eda-partitions", options=[{"label": "Entrenamiento E2", "value": "training"}, {"label": "Calibración E2", "value": "calibration"}, {"label": "Prueba histórica", "value": "evaluation"}], value=["training", "calibration", "evaluation"], multi=True, clearable=True)], md=4),
                dbc.Col([html.Label("Fechas del objetivo", htmlFor="eda-dates", className="control-label"),
                         dcc.DatePickerRange(id="eda-dates", min_date_allowed=dates[0], max_date_allowed=dates[-1], start_date=dates[0], end_date=dates[-1], display_format="DD/MM/YYYY", start_date_placeholder_text="Inicio", end_date_placeholder_text="Fin")], md=4),
                dbc.Col([html.Label("Variable disponible al origen", htmlFor="eda-variable", className="control-label"),
                         dcc.Dropdown(id="eda-variable", options=[{"label": label, "value": key} for key, label in VARIABLES.items()], value="last_close", clearable=False)], md=4),
            ], className="g-3"),
            html.Div([html.Label("Rango de σ₇ observado", htmlFor="eda-range", className="control-label"),
                      dcc.RangeSlider(id="eda-range", min=0, max=upper, step=upper / 1000, value=[0, upper], marks={0: "0", upper / 2: f"{upper / 2:.3f}", upper: f"{upper:.3f}"}, tooltip={"placement": "bottom", "always_visible": False})], className="range-control"),
        ], className="filter-panel"),
        html.Div(id="eda-n", className="cohort-summary", role="status", **{"aria-live": "polite"}),
        html.Div(id="eda-insight", className="insight-panel"),
        dbc.Row([dbc.Col(_chart("eda-series"), lg=8), dbc.Col(_chart("eda-hist"), lg=4)], className="g-4 chart-row"),
        dbc.Row([dbc.Col(_chart("eda-feature"), lg=7), dbc.Col(_chart("eda-partitions-figure"), lg=5)], className="g-4 chart-row"),
        html.Section([html.P("DEPENDENCIA TEMPORAL", className="eyebrow"), html.H2("Qué permanece entre días"),
                      html.P(id="eda-dependence-note", className="muted"),
                      dbc.Row([dbc.Col(_chart("eda-acf"), md=6), dbc.Col(_chart("eda-pacf"), md=6)], className="g-4")], className="section-gap"),
        html.Section([html.P("CALIDAD Y CAMBIO DE DISTRIBUCIÓN", className="eyebrow"), html.H2("Leer la muestra antes de modelar"), html.Div(id="eda-quality")], className="content-card section-gap"),
        html.P("La fecha del objetivo es t; las variables mostradas proceden de t−1. Los filtros de EDA no modifican el ranking histórico de Modelos.", className="page-note"),
    ], fluid=True, className="page-container")


@callback(
    Output("eda-n", "children"), Output("eda-insight", "children"),
    Output("eda-series", "figure"), Output("eda-hist", "figure"), Output("eda-feature", "figure"),
    Output("eda-partitions-figure", "figure"), Output("eda-acf", "figure"), Output("eda-pacf", "figure"),
    Output("eda-dependence-note", "children"), Output("eda-quality", "children"),
    Output("eda-series-caption", "children"), Output("eda-hist-caption", "children"),
    Output("eda-feature-caption", "children"), Output("eda-partitions-figure-caption", "children"),
    Output("eda-acf-caption", "children"), Output("eda-pacf-caption", "children"),
    Input("eda-partitions", "value"), Input("eda-dates", "start_date"), Input("eda-dates", "end_date"),
    Input("eda-range", "value"), Input("eda-variable", "value"),
)
def update_eda(partitions, start, end, value_range, variable):
    rows = filter_observations(partitions=partitions or [], start=start, end=end, value_range=value_range)
    field = variable if variable in VARIABLES else "last_close"
    if rows:
        values = np.asarray([r["target"] for r in rows], dtype=float)
        q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
        summary = [html.Strong(f"{len(rows):,}".replace(",", ".")), " observaciones · ", rows[0]["target_date"], " a ", rows[-1]["target_date"]]
        insight = [html.Strong("Lo que muestra tu selección. "),
                   f"La mediana de σ₇ es {median:.5f}; el 80 % central está entre {q10:.5f} y {q90:.5f}. "
                   "El rango de volatilidad cambia con el periodo y debe conservarse al interpretar un pronóstico."]
        feature_values = [float(row[field]) for row in rows]
        cohort_labels = {"training": "Entrenamiento", "calibration": "Calibración", "evaluation": "Prueba"}
        cohort_medians = [f"{label}: {float(np.median([row['target'] for row in rows if row['partition'] == cohort])):.5f}"
                          for cohort, label in cohort_labels.items() if any(row["partition"] == cohort for row in rows)]
        captions = [
            f"σ₇ observado varía de {float(values.min()):.5f} a {float(values.max()):.5f}. Los cortes visibles conservan los huecos de la selección y las purgas.",
            f"La mediana es {median:.5f}; el 80 % central está entre {q10:.5f} y {q90:.5f}. La cola de esta selección importa al evaluar errores extremos.",
            f"{VARIABLES[field]} varía de {min(feature_values):.5g} a {max(feature_values):.5g}. Es información conocida en t−1; su asociación con σ₇ no demuestra causalidad.",
            "Medianas de la selección · " + "; ".join(cohort_medians) + ". Comparar periodos requiere atender a su distribución, no sólo al error promedio.",
            f"Selección global: {len(rows)} fechas. La ACF usa sólo el segmento diario continuo indicado arriba; los objetivos de siete retornos comparten información entre días.",
            f"Selección global: {len(rows)} fechas. La PACF resume dependencia parcial en el mismo segmento; sus bandas son referencias descriptivas, no un protocolo confirmatorio.",
        ]
    else:
        summary = [html.Strong("0"), " observaciones · cambia los filtros para recuperar una muestra."]
        insight = "La selección está vacía; no se calculan resúmenes o diagnósticos ficticios."
        captions = ["No hay observaciones para interpretar con estos filtros."] * 6
    acf, pacf, dependence = plots.dependency_panel(rows)
    quality = plots.quality_summary(rows)
    if isinstance(quality, dict):
        labels = {"duplicate_dates": "Fechas duplicadas", "invalid_dates": "Fechas inválidas",
                  "nonfinite_values": "Campos ausentes o no finitos", "invalid_last_bars": "Últimas barras OHLCV inválidas",
                  "calendar_breaks": "Saltos del calendario seleccionado", "exact_last_feature_duplicates": "Últimas barras exactamente repetidas"}
        quality_children = [html.P(quality.get("insight", "")),
                            html.Dl([element for key, label in labels.items() for element in (html.Dt(label), html.Dd(str(quality.get(key, "—"))))], className="data-card-list"),
                            html.P("Los casi duplicados y el ruido del objetivo requieren auditorías específicas; estos conteos de la selección no los certifican.", className="figure-insight")]
    elif isinstance(quality, (list, tuple)):
        quality_children = html.Ul([html.Li(str(item)) for item in quality])
    else:
        quality_children = html.P(str(quality))
    figures = [plots.target_series(rows), plots.target_histogram(rows), plots.feature_explorer(rows, field), plots.partition_comparison(rows), acf, pacf]
    for figure in figures:
        figure.update_layout(meta={"filtered_n": len(rows), "scope": "shared_eda_selection"})
    return (summary, insight, figures[0], figures[1], figures[2], figures[3], figures[4], figures[5],
            dependence, quality_children, *captions)
