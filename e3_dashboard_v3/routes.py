"""Supplementary profile page and bounded JSON endpoints."""
import csv
from datetime import date, timedelta
import io
import json
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file
from e3_dashboard.inference import InputError
from . import runtime

blueprint = Blueprint("historical_profile14", __name__)


@blueprint.get("/perfil-14")
def page():
    return send_file(Path(__file__).with_name("profile.html"))


@blueprint.get("/api/profile14/catalogue")
def catalogue():
    return jsonify(profile=runtime.profile()["profile"], models=runtime.catalogue(),
                   V3_enabled=False, new_training_runs=0, new_calibration_fits=0)


@blueprint.get("/api/profile14/example.csv")
def example():
    return send_file(Path(__file__).with_name("example.csv"), mimetype="text/csv")


@blueprint.post("/api/profile14/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) != {"model_id", "csv", "target_date"}:
        return jsonify(error="Envía model_id, csv y target_date."), 400
    try:
        result = runtime.predict(body["model_id"], body["csv"], body["target_date"])
    except InputError as error:
        return jsonify(error=str(error)), 400
    except (OSError, KeyError, TypeError, RuntimeError, ValueError):
        return jsonify(error="No se pudo verificar o ejecutar el perfil de inferencia."), 503
    return jsonify(result)
