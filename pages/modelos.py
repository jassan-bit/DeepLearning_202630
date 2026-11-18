"""Saved model evidence and explicit, verified CPU prediction."""
import base64
import binascii
from datetime import datetime, timezone
import math
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update
from dash.dash_table.Format import Format, Scheme
import numpy as np
import plotly.graph_objects as go

from e3_dashboard.data import get_example, load_bundle, prediction_for
from e3_dashboard.history import load_training_history
from e3_dashboard import plots

dash.register_page(__name__, path="/modelos", name="Modelos", title="Modelos · Volatilidad BTC")
ROOT = Path(__file__).resolve().parents[1]
E3 = "CNN-LSTM-atencion-aditiva E3"
METHOD_NAMES = {
    "paired_circular_block_bootstrap": "Bootstrap temporal por bloques",
    "mean_RMSE_influence_Bartlett_HAC": "Influencia de RMSE + HAC",
    "paired_seed_t": "t pareada entre semillas",
    "paired_seed_t_test": "t pareada entre semillas",
}


def _live_models():
    try:
        from e3_dashboard.inference import available_models
        return available_models()
    except (ImportError, OSError, ValueError, RuntimeError):
        return []


def _number(value, digits=6):
    if value is None or value == "":
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _column(name, key, precision=6, text=False):
    result = {"name": name, "id": key, "type": "text" if text else "numeric"}
    if not text:
        result["format"] = Format(precision=precision, scheme=Scheme.fixed, nully="—")
    return result


def _table(table_id, rows, columns, page_size=10, **kwargs):
    return dash_table.DataTable(
        id=table_id, data=rows, columns=columns,
        sort_action="native", filter_action="native", page_size=page_size,
        style_table={"overflowX": "auto"},
        style_cell={"fontFamily": "inherit", "fontSize": 13, "padding": "12px", "textAlign": "right", "minWidth": "110px", "maxWidth": "240px", "whiteSpace": "normal"},
        style_cell_conditional=[{"if": {"column_id": columns[0]["id"]}, "textAlign": "left", "minWidth": "220px", "fontWeight": "600"}],
        style_header={"backgroundColor": "#eaf1f5", "color": "#12354a", "fontWeight": "700", "borderBottom": "2px solid #0f7f87"},
        style_data={"borderBottom": "1px solid #eaf1f5", "backgroundColor": "white"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#f7fafb"}],
        css=[{"selector": ".dash-spreadsheet td div", "rule": "line-height:1.5;"}],
        **kwargs,
    )


def layout():
    bundle = load_bundle()
    catalogue = sorted([dict(row) for row in bundle["catalogue"]], key=lambda row: row["RMSE_mean"])
    live = _live_models()
    by_id = {row["id"]: row for row in live}
    for row in catalogue:
        row["serving_mode"] = "En vivo + histórico" if row["model"] in by_id else "Histórico precalculado"
        row["onnx_cpu_median_ms"] = by_id.get(row["model"], {}).get("roundtrip", {}).get("cpu_median_ms")
    examples = bundle["examples"]
    live_default = next((row["id"] for row in live if row["id"] == E3), live[0]["id"] if live else None)
    example_default = examples[0]["id"] if examples else None
    origin_default = examples[0]["target_date"] if examples else None
    card = (ROOT / "MODEL_CARD.md").read_text() if (ROOT / "MODEL_CARD.md").exists() else "La ficha del modelo está en preparación."
    mcs = [dict(row) for row in bundle.get("mcs", [])]
    for row in mcs:
        for key in ("MSE diario medio", "p secuencial de exclusión"):
            if key in row:
                row[key] = float(row[key]) if row[key] not in (None, "") else None
    mcs_columns = [_column("Modelo", "Modelo", text=True), _column("MSE diario medio", "MSE diario medio", 9),
                   _column("MCS90%", "MCS90%", text=True), _column("p secuencial de exclusión", "p secuencial de exclusión", 4)]
    try:
        history = load_training_history()
        epoch_curve = plots.training_curve(history)
        epoch_description = f"Semilla {history['seed']}: {len(history['history'])} épocas registradas; mejor época por validación: {history['best_epoch']}."
    except (OSError, ValueError, RuntimeError):
        epoch_curve = plots.figure_empty("La historia de entrenamiento no está disponible o no se pudo verificar.")
        epoch_description = "No se muestra una curva sin su historia verificada."
    contrasts = []
    for row in bundle.get("contrasts", []):
        contrasts.append({
            "rival": row.get("rival"),
            "method": METHOD_NAMES.get(row.get("method"), str(row.get("method", "")).replace("_", " ")),
            "gain_RMSE": float(row["gain_RMSE"]),
            "p_raw": float(row["p_raw"]) if row.get("p_raw") else None,
            "p_holm": float(row["p_holm"]) if row.get("p_holm") else None,
            "gain_ci95": row.get("gain_ci95", ""),
            "paired_Cohen_dz": float(row["paired_Cohen_dz"]) if row.get("paired_Cohen_dz") else None,
        })
    columns = [
        _column("Sistema", "model", text=True), _column("Modo", "serving_mode", text=True), _column("Familia", "family", text=True),
        _column("Predictores", "R", 0), _column("RMSE media", "RMSE_mean"),
        _column("DE entre semillas", "RMSE_sample_SD"), _column("MAE", "MAE_mean"),
        _column("sMAPE (%)", "sMAPE_percent", 3), _column("R² medio", "R2_mean", 4),
        _column("Cobertura 90 % (%)", "coverage90_percent", 3), _column("Anchura media", "mean_interval_width90"),
        _column("Parámetros", "parameters", 0), _column("FLOPs core mínimos", "core_flops_min", 0),
        _column("Latencia CPU ONNX mediana (ms)", "onnx_cpu_median_ms", 4),
    ]
    return dbc.Container([
        html.Header([html.P("MODELOS · EVIDENCIA Y PRONÓSTICOS", className="eyebrow"), html.H1("Comparar antes de predecir"),
                     html.P("El ranking conserva las 608 fechas históricas y todas las recetas. Examina los errores de una trayectoria y prueba sólo los modelos disponibles para inferencia verificada.", className="page-lead")], className="page-header"),
        html.Section([
            html.Div([html.Div([html.P("REFERENCIA HISTÓRICA", className="eyebrow"), html.H2("17 sistemas · una misma cohorte")]), html.Span("Media de ejecuciones individuales", className="soft-badge")], className="section-heading split-heading"),
            html.P("Las redes incluyen diez semillas; los tres clásicos, una trayectoria. La DE entre semillas no es un intervalo de predicción. Puedes ordenar y filtrar la vista sin cambiar la familia estadística.", className="muted"),
            _table("leaders-table", catalogue, columns, page_size=17, sort_by=[{"column_id": "RMSE_mean", "direction": "asc"}], cell_selectable=True),
            html.P("— indica un dato no disponible. Las latencias ONNX son del predictor 271801: preprocesamiento e inferencia caliente, un hilo y lote uno; excluyen explicación. Los FLOPs core conservan su alcance de origen. Los costes ausentes no se rellenan con cero.", className="figure-insight"),
        ], className="content-card"),
        html.Section([
            html.Div([html.Div([html.P("DIAGNÓSTICOS", className="eyebrow"), html.H2("Dónde se equivoca cada receta")]), html.Div([html.Label("Modelo para examinar", htmlFor="model-select", className="control-label"), dcc.Dropdown(id="model-select", options=[{"label": row["model"], "value": row["model"]} for row in catalogue], value=E3, clearable=False)], className="model-selector")], className="section-heading split-heading"),
            html.Div(id="model-summary", className="insight-panel"),
            dbc.Row([dbc.Col(dcc.Graph(id="model-forecast", config={"displaylogo": False}), lg=8), dbc.Col(dcc.Graph(id="model-scatter", config={"displaylogo": False}), lg=4)], className="g-4 chart-row"),
            dbc.Row([dbc.Col(dcc.Graph(id="model-residuals", config={"displaylogo": False}), lg=8), dbc.Col(dcc.Graph(id="model-residual-acf", config={"displaylogo": False}), lg=4)], className="g-4 chart-row"),
            html.P("Estas curvas usan una trayectoria representativa fijada por identificador: semilla 271801 en redes, una trayectoria en clásicos. Sus errores no sustituyen la media de diez ejecuciones del ranking.", className="figure-insight"),
        ], className="section-gap"),
        html.Section([
            html.Div([html.Div([html.P("PREDICCIÓN", className="eyebrow"), html.H2("Una ventana de entrada, un pronóstico")]), html.Span(f"{len(live)} modelos habilitados" if live else "Preparación de artefactos", className="soft-badge")], className="section-heading split-heading"),
            html.P("Se necesitan siete días consecutivos de OHLCV anteriores a la fecha objetivo. No se ajustan pesos, estadísticas o calibradores con tu archivo.", className="muted"),
            dbc.Row([
                dbc.Col([
                    html.Label("Modelo disponible", htmlFor="live-model", className="control-label"),
                    dcc.Dropdown(id="live-model", options=[{"label": row["label"], "value": row["id"]} for row in live], value=live_default, clearable=False, placeholder="Aún no hay un artefacto habilitado"),
                ], md=7),
                dbc.Col([html.Label("Fecha objetivo (UTC)", htmlFor="live-origin", className="control-label"),
                         dcc.DatePickerSingle(id="live-origin", date=origin_default, max_date_allowed=datetime.now(timezone.utc).date().isoformat(), display_format="DD/MM/YYYY", placeholder="Fecha objetivo")], md=5),
            ], className="g-3"),
            html.Div([
                html.Label("Origen de los datos", htmlFor="live-source", className="control-label"),
                dbc.RadioItems(id="live-source", options=[{"label": "Ejemplo verificado", "value": "example"}, {"label": "Mi archivo CSV", "value": "upload"}], value="example", inline=True),
            ], className="source-control"),
            html.Div([
                html.Label("Ventana histórica de ejemplo", htmlFor="live-example", className="control-label"),
                dcc.Dropdown(id="live-example", options=[{"label": row["label"], "value": row["id"]} for row in examples], value=example_default, clearable=False),
                html.P(id="example-description", className="figure-insight"),
            ], id="example-control"),
            html.Div([
                dcc.Upload(id="live-upload", children=html.Div([html.Strong("Elegir archivo CSV"), html.Span(" o arrastrarlo aquí · máximo 100 KiB")]), accept=".csv,text/csv", multiple=False, max_size=102400, className="upload-area"),
                html.P(id="upload-description", className="figure-insight"),
                html.P("Columnas: date, open, high, low, close, volume. Siete filas ordenadas, sin huecos ni el día objetivo.", className="figure-insight"),
            ], id="upload-control", style={"display": "none"}),
            dbc.Button("Calcular pronóstico", id="predict-button", color="primary", className="predict-button", disabled=not bool(live)),
            html.Div(id="prediction-result", className="prediction-output", role="status", **{"aria-live": "polite"}),
        ], className="content-card section-gap"),
        html.Section([
            html.P("COMPARACIÓN ESTADÍSTICA", className="eyebrow"), html.H2("Un resultado favorable con alcance definido"),
            html.P("Los 48 contrastes posteriores rechazan sus nulas tras Holm. La diferencia de entrenamiento, el calendario conocido y los requisitos pendientes impiden dar por cerrada R3. La selección por validación y la comparación de prueba mantienen alcances distintos.", className="muted"),
            dbc.Accordion([
                dbc.AccordionItem([
                    html.P("El MCS auxiliar al 90 % conserva diez sistemas, incluida E3. Usa MSE diario medio entre semillas: no declara un ganador único, equivalencia o una ganancia de RMSE mayor que el margen."),
                    _table("mcs-table", mcs, mcs_columns, page_size=17),
                ], title="Conjunto de modelos retenidos · MCS90"),
                dbc.AccordionItem([
                    html.P("Los IC son marginales; el tamaño de efecto entre semillas condiciona el calendario. Los tres procedimientos tienen ámbitos distintos y requieren justificar su pertinencia para la afirmación."),
                    _table("contrasts-table", contrasts, [_column("Rival", "rival", text=True), _column("Procedimiento", "method", text=True), _column("Ganancia RMSE", "gain_RMSE"), _column("p crudo", "p_raw", 7), _column("p Holm", "p_holm", 7), _column("IC95 % marginal", "gain_ci95", text=True), _column("Cohen dz entre semillas", "paired_Cohen_dz", 3)]),
                ], title="Pruebas por rival · 48 contrastes completos"),
                dbc.AccordionItem([
                    html.P("La frontera de Pareto exige latencias, memoria y desempeño bajo el mismo entorno. El catálogo conserva costes de perfiles distintos; sólo se compararán aquí mediciones de un mismo perfil, sin atribuir a otros modelos un límite de memoria no medido."),
                    html.Div(id="pareto-panel"),
                ], title="Eficiencia y costes comparables"),
                dbc.AccordionItem([
                    html.P("Las oclusiones de la entrada actual permiten explorar sensibilidades locales. La interpretación global y su comparación con el mejor benchmark requieren un estudio con los mismos casos y controles de fidelidad y estabilidad entre semillas; ese estudio sigue pendiente."),
                ], title="Interpretación global y comparación explicativa"),
            ], start_collapsed=True),
        ], className="section-gap"),
        html.Section([
            html.Div([html.Div([html.P("REFERENCIA E3", className="eyebrow"), html.H2("CNN-LSTM con atención temporal")]), dbc.Button("Abrir ficha del modelo", id="model-card-open", outline=True, color="primary")], className="section-heading split-heading"),
            dbc.Row([
                dbc.Col([html.H3("La salida positiva aditiva"), html.P("Una CNN causal y una LSTM resumen la ventana; la atención por contenido agrega los estados. La salida suma un exceso positivo al piso observable de σ₇. «Aditiva» describe la salida, no otro algoritmo de atención."),
                         html.Ol([html.Li("Transformar la ventana con las estadísticas guardadas de ajuste."), html.Li("Codificar los siete días y agregar sus estados con atención."), html.Li("Sumar el exceso positivo al piso observable."), html.Li("Devolver σ₇ en unidades originales, con incertidumbre sólo cuando exista calibración verificada.")])], md=6),
                dbc.Col([html.H3("Qué sustenta y qué falta"), html.P("La referencia evaluada tiene 1210 parámetros y diez semillas. Se distingue del primer mecanismo de penalización/saturación de 1211 y de la cabeza de momentos de 1236 parámetros."),
                         html.P("El primer piloto fue desfavorable al mecanismo modificado. En 18 fechas de volatilidad realizada alta, la aditiva tuvo más RMSE que M00 proyectado; es un diagnóstico posterior, no una regla disponible al pronosticar."),
                         html.P("Faltan revisión de novedad, defensa del aporte, potencia, tres conjuntos adicionales, ablaciones y evaluación ampliada. La atención por sí sola no es una explicación causal.", className="muted")], md=6),
            ], className="g-4"),
            html.H3("Evolución durante el ajuste"),
            html.P(epoch_description, className="figure-insight"),
            dcc.Graph(id="model-training-curve", figure=epoch_curve, config={"displaylogo": False}),
            html.P("La pérdida de ajuste es el MSE normalizado observado durante las actualizaciones; no es el riesgo del modelo final sobre todo el entrenamiento. La validación usa RMSE en unidades originales de σ₇, en otro eje: su separación vertical no es una prueba de sobreajuste. Las Tablas A/B y la curva frente al tamaño de muestra siguen pendientes.", className="figure-insight"),
            html.A("Ver formulación, curvas y estudios del modelo ↗", href="https://jassan-bit.github.io/DeepLearning_202630/entregable3/sections/06_modelo_original.html", target="_blank", rel="noopener noreferrer", className="text-link"),
        ], className="content-card section-gap"),
        dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Ficha del modelo E3"), close_button=True), dbc.ModalBody(dcc.Markdown(card, link_target="_blank"))], id="model-card-modal", size="xl", scrollable=True, is_open=False),
    ], fluid=True, className="page-container")


@callback(Output("model-select", "value"), Input("leaders-table", "active_cell"), State("leaders-table", "derived_viewport_data"), prevent_initial_call=True)
def select_table_model(cell, rows):
    if cell and rows and 0 <= cell.get("row", -1) < len(rows):
        return rows[cell["row"]]["model"]
    return no_update


@callback(Output("model-summary", "children"), Output("model-forecast", "figure"), Output("model-residuals", "figure"), Output("model-scatter", "figure"), Output("model-residual-acf", "figure"), Input("model-select", "value"))
def update_model(model_id):
    bundle = load_bundle()
    try:
        chosen = prediction_for(model_id)
    except (KeyError, ValueError):
        empty = plots.figure_empty("Seleccione un modelo del catálogo.")
        return "Modelo no disponible.", empty, empty, empty, empty
    dates, targets = bundle["predictions"]["dates"], bundle["predictions"]["targets"]
    prediction, q = chosen["predictions"], chosen.get("q")
    errors = np.asarray(prediction) - np.asarray(targets)
    rmse, mae = float(np.sqrt(np.mean(errors ** 2))), float(np.mean(np.abs(errors)))
    interval_note = "Intervalo nominal heredado del 90 %, con límites temporales." if q is not None else "Este predictor no tiene un intervalo calibrado."
    summary = [html.Strong(model_id), html.Span(f" · semilla {chosen['seed']} · {len(dates)} fechas"), html.Br(),
               f"RMSE de esta trayectoria: {rmse:.6f} · MAE: {mae:.6f}. {interval_note}"]
    return (summary, plots.forecast_figure(dates, targets, prediction, q),
            plots.residual_figure(dates, targets, prediction), plots.scatter_actual_forecast(dates, targets, prediction),
            plots.residual_acf(dates, targets, prediction))


@callback(Output("example-control", "style"), Output("upload-control", "style"), Input("live-source", "value"))
def choose_source(source):
    return ({"display": "none"} if source == "upload" else {}), ({} if source == "upload" else {"display": "none"})


@callback(Output("live-origin", "date"), Output("example-description", "children"), Input("live-example", "value"), prevent_initial_call=False)
def choose_example(example_id):
    if not example_id:
        return no_update, "No hay un ejemplo disponible."
    try:
        example = get_example(example_id)
    except (KeyError, ValueError):
        return no_update, "El ejemplo no está disponible."
    return example["target_date"], f"{example['label']}. La ventana termina el día anterior al objetivo; no se elige por su error."


@callback(Output("upload-description", "children"), Input("live-upload", "filename"))
def upload_description(filename):
    return f"Archivo seleccionado: {filename}" if filename else "No se ha seleccionado un archivo."


def _explanation_component(explanation, status):
    if not explanation:
        return html.P(status or "La explicación local de este artefacto sigue pendiente de verificación.", className="figure-insight")
    if explanation.get("figure"):
        return dcc.Graph(figure=explanation["figure"], config={"displaylogo": False})
    if explanation.get("message"):
        return html.P(explanation["message"], className="figure-insight")
    values = explanation.get("values", explanation.get("attributions"))
    if isinstance(values, list) and values and isinstance(values[0], dict):
        if all("effect" in row for row in values):
            labels = [str(row.get("label", row.get("lag", ""))) for row in values]
            effects = [float(row["effect"]) for row in values]
            if all(math.isfinite(value) for value in effects):
                figure = go.Figure(go.Bar(x=labels, y=effects, marker_color=["#0f7f87" if value >= 0 else "#b26a12" for value in effects], hovertemplate="%{x}: %{y:.6f} σ₇<extra></extra>"))
                figure.update_layout(template="plotly_white", title="Sensibilidad al sustituir cada día de entrada", xaxis_title="Día anterior al objetivo", yaxis_title="Cambio en el pronóstico σ₇", margin={"l": 70, "r": 20, "t": 60, "b": 50}, showlegend=False)
                return [dcc.Graph(figure=figure, config={"displaylogo": False}),
                        html.P("Cada barra mide el efecto de sustituir ese día por una referencia obtenida sólo de entrenamiento. Es una sensibilidad por oclusión, no una explicación causal ni una descomposición aditiva garantizada.", className="figure-insight"),
                        html.P("La fidelidad explicativa y su estabilidad entre semillas siguen pendientes de evaluación.", className="figure-insight")]
    if values is not None:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return html.P(status or "La explicación local no está disponible para esta salida.", className="figure-insight")
        if array.ndim == 2 and array.shape == (7, 5) and np.isfinite(array).all():
            figure = go.Figure(go.Heatmap(z=array, x=["Apertura", "Máximo", "Mínimo", "Cierre", "Volumen"], y=[f"t−{day}" for day in range(7, 0, -1)], colorscale=[[0, "#eaf1f5"], [1, "#0f7f87"]], colorbar={"title": "Atribución"}))
            figure.update_layout(template="plotly_white", title="Sensibilidad local por variable y día", margin={"l": 55, "r": 20, "t": 55, "b": 45})
            return [dcc.Graph(figure=figure, config={"displaylogo": False}), html.P(explanation.get("description", status or "Atribución del caso actual; no es una explicación causal."), className="figure-insight")]
    return html.P(status or "La explicación local no está disponible para esta salida.", className="figure-insight")


def _prediction_error(message):
    return html.Div(dbc.Alert(message, color="warning"), role="alert")


@callback(Output("prediction-result", "children"), Input("predict-button", "n_clicks"), State("live-model", "value"), State("live-source", "value"), State("live-example", "value"), State("live-upload", "contents"), State("live-origin", "date"), prevent_initial_call=True)
def run_prediction(clicks, model_id, source, example_id, contents, target_date):
    if not clicks:
        return no_update
    try:
        if not model_id:
            raise ValueError("Selecciona un modelo habilitado para predecir.")
        if source == "upload":
            if not contents or "," not in contents:
                raise ValueError("Selecciona un archivo CSV de siete días.")
            encoded = contents.split(",", 1)[1]
            if len(encoded) > 140000:
                raise ValueError("El archivo supera el límite de 100 KiB.")
            try:
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > 102400:
                    raise ValueError("El archivo supera el límite de 100 KiB.")
                csv_text = raw.decode("utf-8-sig")
            except (binascii.Error, UnicodeDecodeError):
                raise ValueError("El archivo debe ser un CSV válido en UTF-8.") from None
        else:
            csv_text = get_example(example_id)["csv_text"]
        from e3_dashboard.inference import predict_csv
        result = predict_csv(model_id, csv_text, target_date)
    except (ValueError, KeyError) as error:
        return _prediction_error(str(error))
    except (ImportError, OSError, RuntimeError):
        return _prediction_error("La inferencia de este modelo no está disponible en este momento. Puedes seguir explorando sus resultados históricos.")
    except Exception:
        return _prediction_error("No se pudo completar el pronóstico. Revisa la ventana de entrada e inténtalo de nuevo.")
    interval = result.get("interval")
    forecast_cards = [html.Div([html.Small("PRONÓSTICO σ₇"), html.Strong(f"{result['prediction']:.6f}"), html.Span(f"Fecha objetivo: {target_date}")], className="forecast-value")]
    if interval is not None:
        nominal = result.get("nominal_coverage")
        forecast_cards.append(html.Div([html.Small(f"INTERVALO NOMINAL {nominal:.0%}" if nominal is not None else "INTERVALO"), html.Strong(f"[{interval[0]:.6f}, {interval[1]:.6f}]"), html.Span("La cobertura depende de los datos y del cambio de distribución.")], className="forecast-interval"))
    else:
        forecast_cards.append(html.Div([html.Small("INCERTIDUMBRE"), html.Strong("Sin intervalo calibrado"), html.Span("No se sustituye por la dispersión entre semillas.")], className="forecast-interval"))
    output = [html.Div(forecast_cards, className="forecast-grid"), html.P(f"{result.get('model_id', model_id)} · semilla {result.get('seed', '—')} · {result['latency_ms']:.2f} ms en CPU", className="figure-insight"),
              html.P("Tiempo de esta consulta: incluye carga si procede, preprocesamiento e inferencia y excluye la explicación. El resultado corresponde a la última consulta calculada; vuelve a calcular tras cambiar la entrada.", className="figure-insight")]
    if result.get("ood"):
        output.append(dbc.Alert(result.get("ood_message", "La ventana se aleja del rango observado en entrenamiento. Interpreta el pronóstico como extrapolación."), color="warning"))
    elif result.get("ood_message"):
        output.append(html.P(result["ood_message"], className="figure-insight"))
    output.extend([html.H3("Explicación del caso"), html.Div(_explanation_component(result.get("explanation"), result.get("explanation_status")))])
    return output


@callback(Output("model-card-modal", "is_open"), Input("model-card-open", "n_clicks"), State("model-card-modal", "is_open"), prevent_initial_call=True)
def show_model_card(clicks, opened):
    return not opened if clicks else opened


@callback(Output("pareto-panel", "children"), Input("model-select", "value"))
def update_pareto(_model):
    comparable = [row for row in _live_models()
                  if row.get("roundtrip", {}).get("cpu_median_ms") is not None
                  and row["roundtrip"].get("RMSE_ONNX") is not None]
    if len(comparable) < 2:
        return dbc.Alert("Aún no hay al menos dos mediciones de latencia con un mismo perfil que permitan trazar una frontera comparable.", color="light")
    scopes = {row["roundtrip"].get("latency_scope") for row in comparable}
    if len(scopes) != 1:
        return dbc.Alert("Las mediciones disponibles no comparten un mismo perfil; la frontera comparable sigue pendiente.", color="light")
    latency = lambda row: row["roundtrip"]["cpu_median_ms"]
    error = lambda row: row["roundtrip"]["RMSE_ONNX"]
    figure = go.Figure(go.Scatter(x=[latency(row) for row in comparable], y=[error(row) for row in comparable], text=[row["id"] for row in comparable], mode="markers+text", textposition="top center", marker={"size": 11, "color": "#0f7f87"}))
    nondominated = [r for r in comparable if not any(latency(s) <= latency(r) and error(s) <= error(r) and (latency(s) < latency(r) or error(s) < error(r)) for s in comparable)]
    nondominated.sort(key=latency)
    figure.add_scatter(x=[latency(r) for r in nondominated], y=[error(r) for r in nondominated], mode="lines", line={"color": "#12354a", "dash": "dot"}, name="Frontera en este perfil")
    figure.update_layout(template="plotly_white", xaxis_title="Latencia CPU mediana (ms) · menor es mejor", yaxis_title="RMSE del predictor exportado · menor es mejor", title="Cuatro artefactos ONNX · semilla 271801", margin={"l": 65, "r": 25, "t": 65, "b": 60}, showlegend=False)
    return [dcc.Graph(figure=figure, config={"displaylogo": False}), html.P(f"Se muestran {len(comparable)} predictores sobre 608 fechas: un hilo CPU, lote uno, 10 calentamientos y 100 llamadas. La latencia incluye preprocesamiento e inferencia caliente y excluye explicación. No es el ranking medio de diez semillas ni una frontera de toda la familia; faltan los otros 13 sistemas.", className="figure-insight")]
