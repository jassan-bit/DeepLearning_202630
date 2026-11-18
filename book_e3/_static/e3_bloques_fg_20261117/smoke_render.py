"""Exercise a deployed E3 dashboard through its real browser interface.

The client never imports the application, trains a model, or uploads user data.
Public targets must be Render HTTPS hosts; explicit loopback mode is a local
preflight and can never produce a public-deployment success claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[2]
GRAPH_IDS = ("eda-series", "eda-hist", "eda-feature", "eda-partitions-figure", "eda-acf", "eda-pacf")


class NoHealthRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def same_origin(first, second):
    try:
        a, b = urlsplit(first), urlsplit(second)
        default = lambda p: p.port or (443 if p.scheme == "https" else 80)
        return (a.scheme, a.hostname, default(a)) == (b.scheme, b.hostname, default(b))
    except ValueError:
        return False


def validate_url(value, allow_local=False):
    """Restrict the smoke client to a public Render origin or explicit loopback."""
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Use únicamente el origen de la URL, sin credenciales, ruta, consulta o fragmento.")
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("La URL contiene un puerto inválido.") from None
    host = (parsed.hostname or "").lower()
    if allow_local and host in {"127.0.0.1", "localhost", "::1"} and parsed.scheme in {"http", "https"}:
        return value.rstrip("/"), False
    if parsed.scheme != "https" or port not in (None, 443) or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.onrender\.com", host):
        raise ValueError("La prueba pública requiere https://nombre-del-servicio.onrender.com; use --allow-local sólo para loopback.")
    return value.rstrip("/"), True


def validate_forecast(shown, reference, tolerance):
    if not all(math.isfinite(x) for x in (shown, reference, tolerance)) or tolerance <= 0:
        raise ValueError("Pronóstico, referencia o tolerancia no finitos o inválidos.")
    if shown < 0 or abs(shown - reference) > tolerance:
        raise ValueError("El pronóstico mostrado no coincide con el predictor congelado en la tolerancia declarada.")
    return abs(shown - reference)


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Use un límite positivo y finito.")
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_references():
    registry = json.loads((ROOT / "artifacts/e3_dashboard_v1/manifest.json").read_bytes())
    examples = json.loads((ROOT / "data/e3_dashboard_v1/examples.json").read_bytes())
    predictions = json.loads((ROOT / "data/e3_dashboard_v1/predictions.json").read_bytes())
    example = next(x for x in examples if x["id"] == "evaluation-0")
    index = predictions["dates"].index(example["target_date"])
    models = []
    for model in registry["models"]:
        path = ROOT / "artifacts/e3_dashboard_v1" / model["slug"] / "metadata.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != model["metadata_sha256"]:
            raise ValueError("No se pudo verificar un metadato local de referencia.")
        metadata = json.loads(raw)
        models.append({"id": model["id"], "seed": metadata["seed"], "onnx_sha256": metadata["model_sha256"],
                       "has_interval": metadata["calibration_q"] is not None,
                       "prediction": predictions["models"][model["id"]]["predictions"][index]})
    if len(models) != 4:
        raise ValueError("El guion requiere el registro versionado completo de cuatro ONNX.")
    return example, models


def wake_service(base, seconds, report):
    """Measure readiness wait; prior idleness and a true cold start are unknown."""
    started = time.perf_counter()
    deadline = started + seconds
    attempts = []
    opener = build_opener(NoHealthRedirect())
    report["readiness"] = {"attempts": attempts, "budget_seconds": seconds,
                           "is_proven_cold_start_measurement": False}
    while time.perf_counter() < deadline:
        attempt = {"at_utc": utc_now()}
        try:
            remaining = max(0.1, deadline - time.perf_counter())
            with opener.open(Request(base + "/healthz", headers={"User-Agent": "E3-deployment-smoke/1"}),
                             timeout=min(10.0, remaining)) as response:
                attempt["HTTP"] = response.status
                health = json.loads(response.read(65537))
            if health.get("status") == "ok" and health.get("service") == "e3-dashboard" and health.get("catalogue_systems") == 17:
                attempts.append(attempt)
                report["readiness"].update({"wait_ms": (time.perf_counter() - started) * 1000, "health": health})
                return
            attempt["error"] = "El healthcheck no acredita el servicio y catálogo esperados."
        except HTTPError as error:
            attempt["HTTP"] = error.code
            if 300 <= error.code < 400:
                attempt["error"] = "Redirección rechazada; se exige el origen configurado."
                attempts.append(attempt)
                raise RuntimeError("El healthcheck redirige; no se acredita la URL Render configurada.") from None
        except (URLError, OSError, ValueError, KeyError):
            attempt["error"] = "Healthcheck aún no disponible o inválido."
        attempts.append(attempt)
        time.sleep(min(2.0, max(0.0, deadline - time.perf_counter())))
    report["readiness"]["wait_ms"] = (time.perf_counter() - started) * 1000
    raise RuntimeError("El servicio no quedó listo dentro del presupuesto de espera declarado.")


def browser_smoke(base, args, report):
    from playwright.sync_api import sync_playwright
    example, models = load_references()
    report["case"] = {"id": example["id"], "target_date": example["target_date"],
                      "csv_sha256": example["csv_sha256"], "selection": "fixed evaluation-0; not performance selected"}
    report["models"], report["pages"], report["asset_fingerprints"] = [], [], []
    report["browser_errors"], report["console_errors"], report["HTTP_5xx_after_ready"] = [], [], []
    report["blocked_cross_origin_requests"] = []
    with sync_playwright() as pw:
        options = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if args.chromium_path:
            options["executable_path"] = args.chromium_path
        browser = pw.chromium.launch(**options)
        report["client"]["chromium"] = browser.version
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, locale="es-CO", timezone_id="America/Bogota",
                                      service_workers="block")
        def restrict_origin(route):
            target = urlsplit(route.request.url)
            if target.scheme in {"http", "https"} and not same_origin(route.request.url, base):
                report["blocked_cross_origin_requests"].append({"scheme": target.scheme, "host": target.hostname})
                route.abort("blockedbyclient")
            else:
                route.continue_()
        context.route("**/*", restrict_origin)
        page = context.new_page()
        page.set_default_timeout(args.navigation_seconds * 1000)
        page.on("pageerror", lambda error: report["browser_errors"].append(str(error)))
        page.on("console", lambda message: report["console_errors"].append(message.text) if message.type == "error" else None)
        page.on("response", lambda response: report["HTTP_5xx_after_ready"].append({"url": response.url, "HTTP": response.status})
                if response.status >= 500 and urlsplit(response.url).netloc == urlsplit(base).netloc else None)
        try:
            for route in ("/", "/eda", "/modelos"):
                started = time.perf_counter()
                response = page.goto(base + route, wait_until="networkidle")
                if response is None or response.status != 200:
                    raise RuntimeError("Una página no respondió HTTP 200.")
                if not same_origin(page.url, base) or not same_origin(response.url, base):
                    raise RuntimeError("La navegación salió del origen configurado.")
                page.locator("main h1").wait_for()
                if route == "/eda":
                    page.wait_for_function("""ids => ids.every(id => {
                        const g=document.querySelector('#'+id+' .js-plotly-plot');
                        return g && g.layout && g.layout.meta && g.layout.meta.filtered_n===1316;
                    })""", arg=list(GRAPH_IDS))
                    page.locator("#eda-partitions .Select-clear-zone").click()
                    page.wait_for_function("""ids => ids.every(id => {
                        const g=document.querySelector('#'+id+' .js-plotly-plot');
                        return g && g.layout && g.layout.meta && g.layout.meta.filtered_n===0 && g.data.length===0;
                    })""", arg=list(GRAPH_IDS))
                if route == "/modelos":
                    page.wait_for_function("document.querySelectorAll('#leaders-table td[data-dash-column=\"model\"]').length===17")
                elapsed = (time.perf_counter() - started) * 1000
                if elapsed > args.navigation_seconds * 1000:
                    raise RuntimeError("Una página superó el presupuesto completo de navegación y callback.")
                row = {"path": route, "HTTP": response.status, "elapsed_ms_including_callbacks": elapsed,
                       "title": page.locator("main h1").inner_text()}
                if args.screenshots_dir:
                    destination = Path(args.screenshots_dir) / ({"/": "contexto.png", "/eda": "eda.png", "/modelos": "modelos.png"}[route])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(destination), full_page=True, animations="disabled")
                    raw = destination.read_bytes()
                    row["screenshot"] = {"name": destination.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                report["pages"].append(row)
            for route, local in (("/model-card", ROOT / "MODEL_CARD.md"), ("/data-statement", ROOT / "DATA_STATEMENT.md"),
                                 ("/assets/vendor/flatly-5.3.6.min.css", ROOT / "assets/vendor/flatly-5.3.6.min.css")):
                response = context.request.get(base + route, max_redirects=0)
                observed = hashlib.sha256(response.body()).hexdigest()
                expected = hashlib.sha256(local.read_bytes()).hexdigest()
                report["asset_fingerprints"].append({"path": route, "HTTP": response.status, "sha256": observed,
                                                     "expected_sha256": expected, "matches": observed == expected})
                if response.status != 200 or observed != expected or not same_origin(response.url, base):
                    raise RuntimeError("Un documento o recurso público difiere de la versión local revisada.")
            for model in models:
                row = {"model": model["id"], "seed": model["seed"], "local_reference_onnx_sha256": model["onnx_sha256"],
                       "status": "running"}
                report["models"].append(row)
                field = page.locator("#live-model input").first
                field.fill(model["id"])
                field.press("Enter")
                page.wait_for_function("date => document.querySelector('#live-origin input').value===date", arg="01/01/2025")
                started = time.perf_counter()
                def callback_response(response):
                    return (response.url.endswith("/_dash-update-component") and response.request.method == "POST" and
                            response.request.post_data_json.get("output") == "prediction-result.children")
                with page.expect_response(callback_response, timeout=args.prediction_seconds * 1000) as pending:
                    page.locator("#predict-button").click()
                if pending.value.status != 200:
                    raise RuntimeError("El callback de predicción no respondió HTTP 200.")
                page.wait_for_function("model => document.querySelector('#prediction-result').innerText.includes(model)", arg=model["id"],
                                       timeout=args.prediction_seconds * 1000)
                page.locator("#prediction-result .forecast-value strong").wait_for()
                page.locator("#prediction-result .js-plotly-plot").wait_for()
                shown = float(page.locator("#prediction-result .forecast-value strong").inner_text())
                difference = validate_forecast(shown, model["prediction"], args.tolerance)
                explanation = page.locator("#prediction-result .js-plotly-plot").evaluate(
                    "g => ({type:g.data[0].type,labels:Array.from(g.data[0].x),effects:Array.from(g.data[0].y)})")
                if explanation["type"] != "bar" or len(explanation["effects"]) != 7 or not all(math.isfinite(x) for x in explanation["effects"]):
                    raise RuntimeError("La explicación del predictor no tiene siete sensibilidades finitas.")
                text = page.locator("#prediction-result").inner_text()
                interval_ok = ("INTERVALO NOMINAL 90%" in text) if model["has_interval"] else ("Sin intervalo calibrado" in text)
                if not interval_ok or not re.search(r"no(?: es)? una explicación causal", text):
                    raise RuntimeError("La incertidumbre o los límites de la explicación no aparecen con el alcance esperado.")
                cpu_latency = re.search(r"· ([0-9.]+) ms en CPU", text)
                if cpu_latency is None:
                    raise RuntimeError("No aparece la latencia CPU con su alcance declarado.")
                elapsed = (time.perf_counter() - started) * 1000
                if elapsed > args.prediction_seconds * 1000:
                    raise RuntimeError("Una predicción con explicación superó el presupuesto completo declarado.")
                row.update({"status": "passed", "HTTP": pending.value.status, "displayed_prediction_six_decimals": shown,
                            "frozen_prediction": model["prediction"], "absolute_difference": difference,
                            "interval_state": "nominal_90_percent" if model["has_interval"] else "explicitly_uncalibrated",
                            "displayed_CPU_ms_excluding_explanation": float(cpu_latency.group(1)),
                            "explanation": explanation, "elapsed_ms_including_network_rendering_and_explanation": elapsed})
            if report["browser_errors"] or report["console_errors"] or report["HTTP_5xx_after_ready"] or report["blocked_cross_origin_requests"]:
                raise RuntimeError("Se detectó una excepción de navegador, un error de consola o HTTP 5xx tras quedar listo.")
        finally:
            context.close()
            browser.close()


def run(args):
    base, public = validate_url(args.url, args.allow_local)
    if public and not args.deployment_commit:
        raise ValueError("Registra el commit de despliegue declarado a partir del log de Render.")
    if args.deployment_commit and not re.fullmatch(r"[a-fA-F0-9]{40}", args.deployment_commit):
        raise ValueError("El commit declarado debe contener exactamente 40 dígitos hexadecimales.")
    report = {"schema_version": 1, "started_at_utc": utc_now(), "status": "running", "base_url": base,
              "scope": "public_Render_browser_smoke" if public else "local_preflight_not_public_deployment",
              "render_public_smoke_passed": False, "operator_declared_deployment_commit": args.deployment_commit or None,
              "remote_commit_identity_verified": False,
              "identity_note": "The current health endpoint has no commit attestation; match Render deployment logs separately. Asset fingerprints and prediction parity are checked, not a full remote source proof.",
              "thresholds_declared_before_measurement": {"readiness_seconds": args.wake_seconds,
                  "navigation_and_callbacks_seconds": args.navigation_seconds, "prediction_with_explanation_seconds": args.prediction_seconds,
                  "absolute_forecast_tolerance_original_sigma7": args.tolerance},
              "client": {"python": platform.python_version(), "smoke_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "new_fits": 0, "new_calibrations": 0, "errors": [],
              "limits": ["One fixed example per live model, not a new scientific evaluation or metric update.",
                         "No remote peak RAM, concurrency, uptime, or proven cold-start assessment.",
                         "Occlusion availability does not certify explanatory fidelity or seed stability.",
                         "This automated check is not independent human reproduction, R1, R2, or R3 certification."]}
    try:
        wake_service(base, args.wake_seconds, report)
        browser_smoke(base, args, report)
        report["status"] = "passed"
        report["render_public_smoke_passed"] = public
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)[:1500]})
        for model in report.get("models", []):
            if model["status"] == "running":
                model["status"] = "failed"
    report["finished_at_utc"] = utc_now()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--url", required=True, help="Public Render origin, e.g. https://service.onrender.com")
    result.add_argument("--allow-local", action="store_true", help="Explicit loopback preflight; never certifies Render")
    result.add_argument("--deployment-commit", default="", help="Operator-declared 40-character deployment SHA; not server attestation")
    result.add_argument("--output", required=True)
    result.add_argument("--screenshots-dir")
    result.add_argument("--chromium-path", help="Optional development Chromium executable; CI uses Playwright's pinned browser")
    result.add_argument("--wake-seconds", type=positive_seconds, default=180.0)
    result.add_argument("--navigation-seconds", type=positive_seconds, default=30.0)
    result.add_argument("--prediction-seconds", type=positive_seconds, default=30.0)
    result.add_argument("--tolerance", type=positive_seconds, default=1e-6)
    return result


if __name__ == "__main__":
    options = parser().parse_args()
    try:
        receipt = run(options)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(json.dumps({"status": receipt["status"], "scope": receipt["scope"],
                      "render_public_smoke_passed": receipt["render_public_smoke_passed"], "output": str(options.output)}))
    raise SystemExit(0 if receipt["status"] == "passed" else 1)
