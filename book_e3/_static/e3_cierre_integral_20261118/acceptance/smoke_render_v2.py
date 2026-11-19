"""Browser acceptance for five frozen CPU predictors and the E3 interval sidecar.

Prepared with declared AI assistance. This client loads no application or model
runtime. Public success requires the service-reported source commit to agree
with the operator's deployment commit; that agreement is not an attestation.
Playwright is imported only when the browser check is actually requested.
"""
from __future__ import annotations

import csv
from datetime import date
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[3]
LEGACY_PATH = ROOT / "experiments/e3_bloques_fg_20261117/smoke_render.py"
E3_ID = "CNN-LSTM-atencion-aditiva E3"
EXPECTED_MODELS = {"MLP16", "LSTM32", "GRU32", E3_ID, "TSMixer"}
SHA40 = re.compile(r"[0-9a-fA-F]{40}")
SHA64 = re.compile(r"[0-9a-f]{64}")


def _legacy_module():
    specification = importlib.util.spec_from_file_location("_e3_historical_smoke_v1", LEGACY_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


legacy = _legacy_module()
validate_url = legacy.validate_url
validate_forecast = legacy.validate_forecast
same_origin = legacy.same_origin
utc_now = legacy.utc_now


def _component(value):
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or Path(value).name != value or "/" in value or "\\" in value):
        raise ValueError("Ruta inválida en una referencia de aceptación.")
    return value


def _read(path, references, root, expected_sha=None, expected_bytes=None):
    if path.is_symlink():
        raise ValueError("Las referencias de aceptación no admiten enlaces simbólicos.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and (not SHA64.fullmatch(expected_sha) or digest != expected_sha):
        raise ValueError("Una referencia no coincide con su SHA-256 declarado.")
    if expected_bytes is not None and (type(expected_bytes) is not int or len(raw) != expected_bytes):
        raise ValueError("Una referencia no coincide con su tamaño declarado.")
    references[str(path.relative_to(root))] = {"sha256": digest, "bytes": len(raw)}
    return raw


def _json(path, references, root, **expected):
    return json.loads(_read(path, references, root, **expected))


def load_references(root=ROOT):
    """Read all five frozen references without importing the application or ORT.

    The historical four point references stay unchanged. TSMixer's independently
    exported fixed window is bound to its model/metadata and compared with the
    same seven raw OHLCV rows. The posterior interval is bound to the exact E3
    predictor, metadata, checkpoint, units and seed before it is expected in UI.
    """
    root = Path(root).resolve()
    references = {}
    examples = _json(root / "data/e3_dashboard_v1/examples.json", references, root)
    matches = [x for x in examples if x["id"] == "evaluation-0"]
    if len(matches) != 1 or matches[0]["target_date"] != "2025-01-01":
        raise ValueError("Falta el caso fijo de aceptación 2025-01-01.")
    example = dict(matches[0])
    if hashlib.sha256(example["csv_text"].encode()).hexdigest() != example["csv_sha256"]:
        raise ValueError("La huella del CSV del caso fijo no coincide.")
    window = list(csv.DictReader(io.StringIO(example["csv_text"])))
    if len(window) != 7:
        raise ValueError("El caso fijo debe contener siete barras.")
    raw_window = [[float(row[k]) for k in ("open", "high", "low", "close", "volume")] for row in window]
    predictions = _json(root / "data/e3_dashboard_v1/predictions.json", references, root)
    index = predictions["dates"].index(example["target_date"])
    models, seen = [], set()
    for relative, required_count in (("artifacts/e3_dashboard_v1", 4), ("artifacts/e3_serving_extensions_v2", 1)):
        artifact_root = root / relative
        registry = _json(artifact_root / "manifest.json", references, root)
        if registry.get("schema_version") != 1 or len(registry["models"]) != required_count:
            raise ValueError("El registro de inferencia no tiene el contrato esperado.")
        for entry in registry["models"]:
            if entry["id"] in seen:
                raise ValueError("El catálogo contiene identificadores duplicados.")
            seen.add(entry["id"])
            folder = artifact_root / _component(entry["slug"])
            if folder.is_symlink():
                raise ValueError("La carpeta de un modelo no puede ser un enlace.")
            metadata = _json(folder / "metadata.json", references, root, expected_sha=entry["metadata_sha256"])
            if metadata["model"] != entry["id"] or metadata.get("schema_version") != 1:
                raise ValueError("Un metadato no corresponde al identificador registrado.")
            _read(folder / "model.onnx", references, root, expected_sha=metadata["model_sha256"])
            q = metadata["calibration_q"]
            if q is not None and (type(q) not in (int, float) or not math.isfinite(q) or q < 0):
                raise ValueError("Cuantil de intervalo inválido.")
            if entry["id"] == "TSMixer":
                source = root / "experiments/e3_cierre_integral_20261118/serving/export_capabilities/tsmixer/cpu_examples.json"
                tsmixer = _json(source, references, root)
                if (tsmixer["model"] != entry["id"] or tsmixer["seed"] != metadata["seed"]
                        or tsmixer["model_sha256"] != metadata["model_sha256"]
                        or tsmixer["metadata_sha256"] != entry["metadata_sha256"]):
                    raise ValueError("La referencia TSMixer pertenece a otro predictor.")
                cases = [x for x in tsmixer["cases"] if x["target_date"] == example["target_date"]]
                if len(cases) != 1 or cases[0]["raw_ohlcv"] != raw_window:
                    raise ValueError("La referencia TSMixer no usa la misma ventana de aceptación.")
                prediction = cases[0]["expected_frozen_prediction_sigma7"]
            else:
                prediction = predictions["models"][entry["id"]]["predictions"][index]
            if type(prediction) not in (int, float) or not math.isfinite(prediction) or prediction < 0:
                raise ValueError("Pronóstico congelado inválido.")
            models.append({"id": entry["id"], "seed": metadata["seed"],
                           "onnx_sha256": metadata["model_sha256"], "metadata_sha256": entry["metadata_sha256"],
                           "prediction": prediction, "has_interval": q is not None, "q": q,
                           "interval_kind": "nominal_90_percent" if q is not None else "explicitly_uncalibrated",
                           "_metadata": metadata})
    if seen != EXPECTED_MODELS:
        raise ValueError("La aceptación requiere exactamente los cinco predictores declarados.")
    calibration_root = root / "artifacts/e3_serving_calibration_v2"
    registry = _json(calibration_root / "manifest.json", references, root)
    if registry.get("schema_version") != 1 or len(registry["models"]) != 1:
        raise ValueError("El registro de intervalos posteriores no tiene el contrato esperado.")
    entry = registry["models"][0]
    if entry["model_id"] != E3_ID:
        raise ValueError("El intervalo posterior esperado corresponde sólo al predictor E3.")
    profile = _json(calibration_root / _component(entry["file"]), references, root,
                    expected_sha=entry["sha256"], expected_bytes=entry["bytes"])
    model = next(x for x in models if x["id"] == E3_ID)
    metadata = model["_metadata"]
    if (profile.get("schema_version") != 1 or profile["model_id"] != model["id"]
            or profile["seed"] != model["seed"] or profile["model_sha256"] != model["onnx_sha256"]
            or profile["base_metadata_sha256"] != model["metadata_sha256"]
            or profile["checkpoint_sha256"] != metadata["scientific_source"]["checkpoint_sha256"]
            or profile["output_scale"] != metadata["output_scale"]
            or profile["output_unit"] != metadata["output_unit"]
            or profile["nominal_coverage"] != .9
            or profile["profile"] != "posterior_absolute_residual_90_v2"):
        raise ValueError("El intervalo posterior no está vinculado al predictor E3 congelado.")
    q = profile["q_sigma7"]
    residuals = profile["absolute_calibration_residuals_sigma7"]
    rank = math.ceil((len(residuals)+1)*profile["nominal_coverage"])
    if (type(q) not in (int, float) or not math.isfinite(q) or q < 0
            or not residuals or not all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in residuals)
            or profile["calibration_n"] != len(residuals) or rank != profile["order_rank"]
            or rank > len(residuals) or sorted(residuals)[rank-1] != q
            or profile["refit_model"] is not False or profile["refit_quantile"] is not False
            or profile["point_forecasts_changed"] is not False):
        raise ValueError("El intervalo posterior no conserva el cuantil histórico declarado.")
    cutoff = date.fromisoformat(profile["applicable_after_target_date"])
    if cutoff.isoformat() != profile["calibration_last_target"] or date.fromisoformat(example["target_date"]) <= cutoff:
        raise ValueError("El intervalo posterior no puede aplicarse al caso fijo.")
    model.update(has_interval=True, q=q, interval_kind="posterior_90_percent",
                 interval_profile=profile["profile"], coverage_statement=profile["coverage_statement"])
    for model in models:
        model.pop("_metadata")
    example["_reference_files"] = references
    return example, models


def validate_health(health, expected_commit="", public=False):
    """Validate service identity as reported by the endpoint, not an attestation."""
    if not isinstance(health, dict) or health.get("status") != "ok" or health.get("service") != "e3-dashboard" or health.get("catalogue_systems") != 17:
        raise ValueError("El healthcheck no identifica el servicio y los 17 sistemas esperados.")
    if expected_commit and (not isinstance(expected_commit, str) or not SHA40.fullmatch(expected_commit)):
        raise ValueError("El commit declarado debe contener 40 dígitos hexadecimales.")
    if public and not expected_commit:
        raise ValueError("La aceptación pública requiere --deployment-commit.")
    reported = health.get("source_commit")
    field = health.get("source_commit_source")
    if reported is None and not public and not expected_commit:
        return {"source_commit": None, "source_commit_source": field,
                "service_reported_commit_matches_operator": False, "remote_commit_identity_verified": False,
                "identity_basis": "no commit supplied by local service", "cryptographic_attestation": False}
    if not isinstance(reported, str) or not SHA40.fullmatch(reported) or field not in {"RENDER_GIT_COMMIT", "APP_SOURCE_COMMIT"}:
        raise ValueError("El servicio no reporta un commit válido de despliegue y su campo de origen.")
    reported = reported.lower()
    matches = bool(expected_commit) and reported == expected_commit.lower()
    if expected_commit and not matches:
        raise ValueError("El commit reportado por /healthz no coincide con el despliegue declarado.")
    return {"source_commit": reported, "source_commit_source": field,
            "service_reported_commit_matches_operator": matches,
            "remote_commit_identity_verified": False,
            "identity_basis": "service-reported environment value; operator-match only",
            "cryptographic_attestation": False}


def validate_explanation(explanation):
    labels = explanation.get("labels", [])
    effects = explanation.get("effects", [])
    normalized = [str(value).replace("−", "-").replace("–", "-") for value in labels]
    if (explanation.get("type") != "bar" or normalized != [f"t-{n}" for n in range(7,0,-1)]
            or len(effects) != 7 or not all(type(x) in (int,float) and math.isfinite(x) for x in effects)):
        raise ValueError("La explicación debe mostrar las siete oclusiones finitas t−7 a t−1.")
    return True


def validate_interval_text(text, model, tolerance):
    kind = model["interval_kind"]
    if kind == "explicitly_uncalibrated":
        if "Sin intervalo calibrado" not in text:
            raise ValueError("Falta el estado explícito sin intervalo calibrado.")
        return {"kind": kind}
    label = "INTERVALO POSTERIOR NOMINAL 90%" if kind == "posterior_90_percent" else "INTERVALO NOMINAL 90%"
    if label not in text or (kind == "posterior_90_percent" and model["coverage_statement"] not in text):
        raise ValueError("Falta la etiqueta del intervalo o el alcance histórico esperado.")
    match = re.search(r"\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*\]", text)
    if match is None:
        raise ValueError("No aparecen los dos extremos del intervalo.")
    shown = [float(match.group(1)), float(match.group(2))]
    expected = [max(0.,model["prediction"]-model["q"]), model["prediction"]+model["q"]]
    for actual, target in zip(shown, expected):
        validate_forecast(actual, target, tolerance)
    return {"kind": kind, "displayed_bounds": shown, "frozen_expected_bounds": expected,
            "nominal_coverage": .9, "is_future_coverage_guarantee": False}


def wake_service(base, seconds, report, *, expected_commit="", public=False):
    started = time.perf_counter()
    deadline = started + seconds
    attempts = []
    opener = build_opener(legacy.NoHealthRedirect())
    report["readiness"] = {"attempts": attempts, "budget_seconds": seconds, "is_proven_cold_start_measurement": False}
    while time.perf_counter() < deadline:
        attempt = {"at_utc": utc_now()}
        try:
            with opener.open(Request(base+"/healthz",headers={"User-Agent":"E3-deployment-smoke/2"}), timeout=min(10.,max(.1,deadline-time.perf_counter()))) as response:
                attempt["HTTP"] = response.status
                raw = response.read(65537)
                if len(raw)>65536:
                    raise ValueError("Healthcheck demasiado grande.")
                health = json.loads(raw)
            if attempt["HTTP"] == 200 and isinstance(health,dict) and health.get("status") == "ok":
                attempts.append(attempt)
                report["readiness"].update(wait_ms=(time.perf_counter()-started)*1000,health=health)
                report["deployment_identity"] = validate_health(health,expected_commit,public)
                return
            attempt["error"] = "El servicio aún no reporta disponibilidad."
        except HTTPError as error:
            attempt["HTTP"] = error.code
            if 300 <= error.code < 400:
                attempts.append(attempt)
                raise RuntimeError("Se rechazó la redirección del healthcheck.") from None
        except (URLError,OSError,json.JSONDecodeError):
            attempt["error"] = "Healthcheck no disponible o JSON inválido."
        attempts.append(attempt)
        time.sleep(min(2.,max(0.,deadline-time.perf_counter())))
    report["readiness"]["wait_ms"] = (time.perf_counter()-started)*1000
    raise RuntimeError("El servicio no quedó listo dentro del presupuesto declarado.")


def browser_smoke(base,args,report,example,models):
    from playwright.sync_api import sync_playwright
    report.update(models=[],pages=[],asset_fingerprints=[],browser_errors=[],console_errors=[],
                  HTTP_5xx_after_ready=[],blocked_cross_origin_requests=[])
    with sync_playwright() as pw:
        options = {"headless":True,"args":["--no-sandbox","--disable-dev-shm-usage"]}
        if args.chromium_path:
            options["executable_path"] = args.chromium_path
        browser = pw.chromium.launch(**options)
        report["client"]["chromium"] = browser.version
        context = browser.new_context(viewport={"width":1440,"height":1000},locale="es-CO",timezone_id="America/Bogota",service_workers="block")
        def restrict_origin(route):
            from urllib.parse import urlsplit
            target = urlsplit(route.request.url)
            if target.scheme in {"http","https"} and not same_origin(route.request.url,base):
                report["blocked_cross_origin_requests"].append({"scheme":target.scheme,"host":target.hostname})
                route.abort("blockedbyclient")
            else:
                route.continue_()
        context.route("**/*",restrict_origin)
        page = context.new_page()
        page.set_default_timeout(args.navigation_seconds*1000)
        page.on("pageerror",lambda error:report["browser_errors"].append(str(error)))
        page.on("console",lambda message:report["console_errors"].append(message.text) if message.type=="error" else None)
        page.on("response",lambda response:report["HTTP_5xx_after_ready"].append({"url":response.url,"HTTP":response.status}) if response.status>=500 and same_origin(response.url,base) else None)
        try:
            for route in ("/","/eda","/modelos"):
                started = time.perf_counter()
                response = page.goto(base+route,wait_until="networkidle")
                if response is None or response.status!=200 or not same_origin(page.url,base) or not same_origin(response.url,base):
                    raise RuntimeError("Una ruta no respondió HTTP 200 en el origen configurado.")
                page.locator("main h1").wait_for()
                if route=="/eda":
                    page.wait_for_function("""ids=>ids.every(id=>{const g=document.querySelector('#'+id+' .js-plotly-plot');return g&&g.layout&&g.layout.meta&&g.layout.meta.filtered_n===1316;})""",arg=list(legacy.GRAPH_IDS))
                    page.locator("#eda-partitions .Select-clear-zone").click()
                    page.wait_for_function("""ids=>ids.every(id=>{const g=document.querySelector('#'+id+' .js-plotly-plot');return g&&g.layout&&g.layout.meta&&g.layout.meta.filtered_n===0&&g.data.length===0;})""",arg=list(legacy.GRAPH_IDS))
                if route=="/modelos":
                    page.wait_for_function('document.querySelectorAll(\'#leaders-table td[data-dash-column="model"]\').length===17')
                elapsed = (time.perf_counter()-started)*1000
                if elapsed>args.navigation_seconds*1000:
                    raise RuntimeError("Una ruta superó el presupuesto completo de navegación y callbacks.")
                row = {"path":route,"HTTP":response.status,"elapsed_ms_including_callbacks":elapsed,"title":page.locator("main h1").inner_text()}
                if args.screenshots_dir:
                    destination = Path(args.screenshots_dir)/{"/":"contexto.png","/eda":"eda.png","/modelos":"modelos.png"}[route]
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    page.screenshot(path=str(destination),full_page=True,animations="disabled")
                    raw = destination.read_bytes()
                    row["screenshot"] = {"name":destination.name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
                report["pages"].append(row)
            for route,relative in (("/model-card","MODEL_CARD.md"),("/data-statement","DATA_STATEMENT.md"),("/assets/vendor/flatly-5.3.6.min.css","assets/vendor/flatly-5.3.6.min.css")):
                response = context.request.get(base+route,max_redirects=0)
                observed = hashlib.sha256(response.body()).hexdigest()
                expected = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
                report["asset_fingerprints"].append({"path":route,"HTTP":response.status,"sha256":observed,"expected_sha256":expected,"matches":observed==expected})
                if response.status!=200 or observed!=expected or not same_origin(response.url,base):
                    raise RuntimeError("Un documento o recurso público difiere de la versión local revisada.")
            for model in models:
                row = {"model":model["id"],"seed":model["seed"],"local_reference_onnx_sha256":model["onnx_sha256"],"local_reference_metadata_sha256":model["metadata_sha256"],"status":"running"}
                report["models"].append(row)
                field = page.locator("#live-model input").first
                field.fill(model["id"])
                field.press("Enter")
                page.wait_for_function("date=>document.querySelector('#live-origin input').value===date",arg="01/01/2025")
                started = time.perf_counter()
                def callback_response(response):
                    return (response.url.endswith("/_dash-update-component") and response.request.method=="POST" and response.request.post_data_json.get("output")=="prediction-result.children")
                with page.expect_response(callback_response,timeout=args.prediction_seconds*1000) as pending:
                    page.locator("#predict-button").click()
                if pending.value.status!=200:
                    raise RuntimeError("El callback de predicción no respondió HTTP 200.")
                page.wait_for_function("model=>document.querySelector('#prediction-result').innerText.includes(model)",arg=model["id"],timeout=args.prediction_seconds*1000)
                page.locator("#prediction-result .forecast-value strong").wait_for()
                page.locator("#prediction-result .js-plotly-plot").wait_for()
                shown = float(page.locator("#prediction-result .forecast-value strong").inner_text())
                difference = validate_forecast(shown,model["prediction"],args.tolerance)
                explanation = page.locator("#prediction-result .js-plotly-plot").evaluate("g=>({type:g.data[0].type,labels:Array.from(g.data[0].x),effects:Array.from(g.data[0].y)})")
                validate_explanation(explanation)
                text = page.locator("#prediction-result").inner_text()
                interval = validate_interval_text(text,model,args.tolerance)
                if not re.search(r"no(?: es)? una explicación causal",text):
                    raise RuntimeError("Falta el límite causal de la explicación por oclusión.")
                cpu_latency = re.search(r"· ([0-9.]+) ms en CPU",text)
                if cpu_latency is None or not math.isfinite(float(cpu_latency.group(1))):
                    raise RuntimeError("No aparece una latencia CPU finita con su alcance declarado.")
                elapsed = (time.perf_counter()-started)*1000
                if elapsed>args.prediction_seconds*1000:
                    raise RuntimeError("Una predicción y su explicación superaron el presupuesto completo declarado.")
                row.update(status="passed",HTTP=pending.value.status,displayed_prediction_six_decimals=shown,
                           frozen_prediction=model["prediction"],absolute_difference=difference,interval_state=model["interval_kind"],
                           interval=interval,displayed_CPU_ms_excluding_explanation=float(cpu_latency.group(1)),
                           explanation=explanation,elapsed_ms_including_network_rendering_and_explanation=elapsed)
            if any(report[key] for key in ("browser_errors","console_errors","HTTP_5xx_after_ready","blocked_cross_origin_requests")):
                raise RuntimeError("El navegador, los callbacks o las solicitudes de red registraron errores.")
        finally:
            context.close()
            browser.close()


def run(args):
    base,public = validate_url(args.url,args.allow_local)
    commit = args.deployment_commit or ""
    if (public and not commit) or (commit and not SHA40.fullmatch(commit)):
        raise ValueError("Declara el commit completo de 40 dígitos del despliegue para una prueba pública.")
    report = {"schema_version":2,"started_at_utc":utc_now(),"status":"running","base_url":base,
              "scope":"public_Render_browser_smoke" if public else "local_preflight_not_public_deployment",
              "render_public_smoke_passed":False,"public_deployment_smoke_passed":False,
              "operator_declared_deployment_commit":commit.lower() or None,
              "remote_commit_identity_verified":False,
              "identity_note":"Service-reported source_commit must match the operator declaration. No cryptographic attestation or proof of all remote source files is claimed.",
              "thresholds_declared_before_measurement":{"readiness_seconds":args.wake_seconds,"navigation_and_callbacks_seconds":args.navigation_seconds,"prediction_with_explanation_seconds":args.prediction_seconds,"absolute_forecast_and_interval_tolerance_original_sigma7":args.tolerance},
              "client":{"python":platform.python_version(),"smoke_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"historical_smoke_script_sha256":hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest(),"historical_helpers_reused_dynamically":True},
              "new_fits":0,"new_calibrations":0,"errors":[],
              "limits":["One fixed window per live model; not a new scientific evaluation.","No remote peak RAM, concurrency, uptime or proven cold-start measurement.","Occlusion availability does not certify explanatory fidelity or seed stability.","Automated acceptance is not independent human reproduction or R1/R2/R3 certification."]}
    try:
        example,models = load_references()
        report["case"] = {"id":example["id"],"target_date":example["target_date"],"csv_sha256":example["csv_sha256"],"selection":"fixed evaluation-0; not performance selected"}
        report["reference_files"] = example["_reference_files"]
        report["expected_live_models"] = [model["id"] for model in models]
        wake_service(base,args.wake_seconds,report,expected_commit=commit,public=public)
        browser_smoke(base,args,report,example,models)
        if len(report.get("pages",[]))!=3 or len(report.get("models",[]))!=5 or any(model.get("status")!="passed" for model in report["models"]):
            raise RuntimeError("La ejecución no acredita las tres rutas y cinco predicciones completas.")
        report["status"] = "passed"
        report["render_public_smoke_passed"] = public
        report["public_deployment_smoke_passed"] = public
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type":type(error).__name__,"message":str(error)[:1500]})
        for model in report.get("models",[]):
            if model.get("status")=="running":
                model["status"] = "failed"
    report["finished_at_utc"] = utc_now()
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    return report


def parser():
    result = legacy.parser()
    result.description = __doc__
    return result


if __name__=="__main__":
    options = parser().parse_args()
    try:
        receipt = run(options)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(json.dumps({"status":receipt["status"],"scope":receipt["scope"],"render_public_smoke_passed":receipt["render_public_smoke_passed"],"output":options.output}))
    raise SystemExit(0 if receipt["status"]=="passed" else 1)
