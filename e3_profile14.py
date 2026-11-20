"""Opt-in entrypoint: frozen five-predictor UI plus a historical fourteen view.

Run gunicorn e3_profile14:server. Importing app.py itself retains five models.
"""
import os
import re
from dash import html
from app import app, server
from e3_dashboard_v3.routes import blueprint
from e3_dashboard_v3 import runtime

server.register_blueprint(blueprint)
app.layout.children.insert(0, html.Div([
    html.Strong("Perfil histórico adicional · 20 nov. 2026: "),
    html.A("14 predictores verificados", href="/perfil-14"),
    html.Span(". Las vistas originales mantienen sus cinco predictores y sus versiones; V3 sigue experimental y no está habilitada."),
], role="status", style={"padding": "0.8rem 1.2rem", "background": "#e9f2fb", "color": "#153955"}))

original_health = server.view_functions["health"]


def health_with_profile():
    response = original_health()
    if response.status_code != 200:
        return response
    document = response.get_json()
    document.update(optional_profile=runtime.profile()["profile"],
                    profile_predictors=len(runtime.catalogue()),
                    profile_sha256=runtime.sha(runtime.PROFILE.read_bytes()),
                    original_UI_predictors=5, V3_enabled=False)
    from flask import jsonify
    return jsonify(document)


server.view_functions["health"] = health_with_profile
