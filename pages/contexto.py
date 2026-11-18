"""Problem, sources and scope for non-specialist readers."""
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html

from e3_dashboard.data import load_bundle

dash.register_page(__name__, path="/", name="Contexto", title="Contexto · Volatilidad BTC")
ROOT = Path(__file__).resolve().parents[1]


def stat(value, label, detail):
    return html.Div([html.Strong(value), html.Span(label), html.Small(detail)], className="stat-card")


def layout():
    bundle = load_bundle()
    metadata = bundle.get("metadata", {})
    observations = bundle["observations"]
    counts = {key: sum(row["partition"] == key for row in observations)
              for key in ("training", "calibration", "evaluation")}
    dates = sorted(row["target_date"] for row in observations)
    statement = (ROOT / "DATA_STATEMENT.md").read_text() if (ROOT / "DATA_STATEMENT.md").exists() else "La declaración de datos se incorpora con esta versión."
    steps = [
        ("01", "Datos y EDA", "Ventanas, objetivo y calidad"),
        ("02", "Literatura", "Antecedentes y limitaciones"),
        ("03", "Benchmark", "Top 10 y baselines"),
        ("04", "Modelo E3", "CNN-LSTM con atención"),
        ("05", "Evaluación", "Errores y comparación posterior"),
        ("06", "Confirmación", "Potencia y conjuntos adicionales pendientes"),
    ]
    return dbc.Container([
        html.Section([
            html.Div([
                html.P("PROYECTO DE DEEP LEARNING", className="eyebrow"),
                html.H1(["Entender la volatilidad.", html.Br(), "Explorar la evidencia."]),
                html.P("¿Cuánto puede variar Bitcoin durante el próximo día? "
                       "Explora los datos, compara 17 sistemas y examina sus pronósticos con sus límites.", className="hero-copy"),
                html.Div([
                    dbc.Button("Explorar los datos", href="/eda", color="primary", className="me-2"),
                    dbc.Button("Comparar modelos", href="/modelos", outline=True, color="primary"),
                ], className="hero-actions"),
            ], className="hero-content"),
            html.Div([
                html.P("EL OBJETIVO", className="eyebrow"),
                html.Div("σ₇", className="target-symbol"),
                html.H2("Volatilidad diaria"),
                html.P("Desviación poblacional de siete retornos logarítmicos de cierre, sin anualizar."),
                html.Span("Horizonte fijo: un día", className="soft-badge"),
            ], className="target-card"),
        ], className="hero-grid"),
        html.Div([
            stat("17", "sistemas comparados", "10 del Top 10 · 6 referencias · E3"),
            stat(str(counts["evaluation"]), "fechas de prueba", "1 ene. 2025 — 31 ago. 2026"),
            stat("10", "semillas por red", "Cada clásico conserva una trayectoria"),
            stat("1 día", "hasta el objetivo", "Sólo historia disponible al origen"),
        ], className="stats-grid"),
        dbc.Row([
            dbc.Col(html.Section([
                html.P("POR QUÉ IMPORTA", className="eyebrow"), html.H2("Variabilidad, no dirección"),
                html.P("Un pronóstico alto de σ₇ señala mayor variación esperada. No dice si el precio subirá o bajará. "
                       "La pregunta del proyecto es qué métodos reducen el error y en qué condiciones dejan de hacerlo."),
                html.P(["La comparación conservada reúne ", html.Strong("608 fechas comunes"),
                        ", verificadas junto con sus objetivos y predicciones. Esta cifra describe el tamaño del experimento, no una validación prospectiva." ]),
                html.A("Consultar datos y método de E2 ↗", href="https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html", target="_blank", rel="noopener noreferrer", className="text-link"),
            ], className="content-card h-100"), md=7),
            dbc.Col(html.Section([
                html.P("PREGUNTAS DEL PROYECTO", className="eyebrow"), html.H2("Qué buscamos contrastar"),
                html.Ul([
                    html.Li("¿Qué error obtiene cada receta sobre las mismas fechas?"),
                    html.Li("¿Cuánto cambia el resultado al variar la inicialización?"),
                    html.Li("¿La ventaja se conserva en otros datos y condiciones?"),
                ], className="question-list"),
                html.P("E3 añade un exceso positivo a un piso observable de σ₇. La contribución de esa regla debe aislarse mediante ablaciones; su novedad y confirmación siguen abiertas.", className="muted"),
            ], className="content-card h-100"), md=5),
        ], className="g-4 section-gap"),
        html.Section([
            html.Div([html.P("CÓMO SE LEE EL PROYECTO", className="eyebrow"), html.H2("Del dato a una conclusión defendible")], className="section-heading"),
            html.Ol([html.Li([html.Span(number, className="pipeline-number"), html.Strong(title), html.Small(detail)])
                     for number, title, detail in steps], className="pipeline", **{"aria-label": "Etapas metodológicas del proyecto"}),
        ], className="section-gap"),
        dbc.Row([
            dbc.Col(html.Section([
                html.P("FICHA DE DATOS", className="eyebrow"), html.H2("BTCUSDT · días UTC"),
                html.Dl([
                    html.Dt("Fuente"), html.Dd("Binance Spot BTCUSDT; fuentes y huellas conservadas en el informe."),
                    html.Dt("Ventana de entrada"), html.Dd("Siete días de apertura, máximo, mínimo, cierre y volumen, conocidos antes del día objetivo."),
                    html.Dt("Cohortes auditadas"), html.Dd(f'{counts["training"]} de entrenamiento E2 · {counts["calibration"]} de calibración E2 · {counts["evaluation"]} de prueba.'),
                    html.Dt("Fechas del objetivo"), html.Dd(f"{dates[0]} — {dates[-1]}"),
                    html.Dt("Tipo de problema"), html.Dd("Pronóstico temporal de una variable continua."),
                    html.Dt("E3"), html.Dd("532 ejemplos de ajuste y 116 de validación; no tiene intervalos calibrados en esta receta."),
                ], className="data-card-list"),
                dbc.Button("Leer declaración de datos", id="data-statement-open", outline=True, color="primary", size="sm"),
            ], className="content-card h-100"), md=7),
            dbc.Col(html.Section([
                html.P("ALCANCE DE LA EVIDENCIA", className="eyebrow"), html.H2("Qué sigue abierto"),
                html.Ul([
                    html.Li("El calendario de prueba ya era conocido; repetirlo no lo convierte en una reserva ciega."),
                    html.Li("E2 y E3 conservan tratamientos de entrenamiento distintos."),
                    html.Li("Faltan potencia previa, tres conjuntos adicionales y la adecuación completa de E3."),
                    html.Li("La inferencia, las explicaciones y la incertidumbre sólo se habilitan cuando están verificadas para el artefacto elegido."),
                ], className="question-list"),
                html.P("La revisión de fuentes/licencias, la novedad, la defensa académica y los usos permitidos conservan el estado que declara el informe.", className="muted"),
            ], className="content-card h-100"), md=5),
        ], className="g-4 section-gap"),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Declaración de datos y privacidad"), close_button=True),
            dbc.ModalBody(dcc.Markdown(statement, link_target="_blank")),
        ], id="data-statement-modal", size="xl", scrollable=True, is_open=False),
        html.P(f'Corte documental: {metadata.get("cut_date_local", "17 de noviembre de 2026")}. Las cifras enlazan a evidencia histórica guardada.', className="page-note"),
    ], fluid=True, className="page-container")


@callback(Output("data-statement-modal", "is_open"), Input("data-statement-open", "n_clicks"), State("data-statement-modal", "is_open"), prevent_initial_call=True)
def show_data_statement(clicks, opened):
    return not opened if clicks else opened
