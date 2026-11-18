"""E3 multipage dashboard. Training modules are never imported by this app."""
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import html
from flask import jsonify, send_file


ROOT = Path(__file__).resolve().parent
app = dash.Dash(
    __name__, use_pages=True, external_stylesheets=["/assets/vendor/flatly-5.3.6.min.css"],
    assets_path_ignore=["vendor"],
    suppress_callback_exceptions=True, title="Volatilidad BTC · E3",
    update_title="Actualizando…",
)
server = app.server
server.config["MAX_CONTENT_LENGTH"] = 512 * 1024


@server.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="La solicitud supera el tamaño permitido."), 413


def navigation():
    return dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([html.Span("σ₇", className="brand-symbol"),
                             html.Span("Volatilidad BTC", className="brand-name")], href="/"),
            dbc.Nav([
                dbc.NavLink("Contexto", href="/", active="exact"),
                dbc.NavLink("EDA", href="/eda", active="exact"),
                dbc.NavLink("Modelos", href="/modelos", active="exact"),
            ], className="main-nav", navbar=True),
            html.Span("Deep Learning · E3", className="course-label"),
        ], fluid=True), className="site-nav", color="dark", dark=True,
    )


app.layout = html.Div([
    html.A("Ir al contenido", href="#main-content", className="skip-link"),
    navigation(),
    html.Main(dash.page_container, id="main-content", className="site-main"),
    html.Footer(dbc.Container([
        html.Div([
            html.Strong("Jassan Arteta · Deep Learning 2026-30"),
            html.Span("Evidencia al 17 nov. 2026 · dashboard E3 v1", className="footer-version"),
        ], className="footer-heading"),
        html.P("Las cargas se procesan para responder y no se guardan en esta aplicación. "
               "Los pronósticos describen volatilidad, no la dirección del precio.", className="footer-note"),
        html.Div([
            html.A("Informe y evidencia", href="https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html", target="_blank", rel="noopener noreferrer"),
            html.A("Ficha del modelo", href="/model-card", target="_blank", rel="noopener noreferrer"),
            html.A("Datos y privacidad", href="/data-statement", target="_blank", rel="noopener noreferrer"),
        ], className="footer-links"),
    ], fluid=True), className="site-footer"),
])


@server.get("/healthz")
def health():
    from e3_dashboard.data import catalogue
    try:
        count = len(catalogue())
    except (OSError, ValueError, KeyError):
        return jsonify(status="unavailable", service="e3-dashboard"), 503
    return jsonify(status="ok", service="e3-dashboard", catalogue_systems=count)


@server.get("/model-card")
@server.get("/MODEL_CARD.md")
def model_card():
    return send_file(ROOT / "MODEL_CARD.md", mimetype="text/plain; charset=utf-8")


@server.get("/data-statement")
@server.get("/DATA_STATEMENT.md")
def data_statement():
    return send_file(ROOT / "DATA_STATEMENT.md", mimetype="text/plain; charset=utf-8")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
