"""Colab coordinator: evaluate one frozen E3 checkpoint and publish to GitHub."""
from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
from getpass import getpass
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

REPO = "jassan-bit/DeepLearning_202630"
SOURCE_COMMIT = "cf891213f1acb6967aef5e0857fa0ac63768f836"
RESULT_COMMIT = "1ca029e869a517cb8de704e573c78ca1e0cbad3c"
E2_COMMIT = "f65b14d38e38f8d5fd80c717d71f2abd9ea708e2"
BRANCH = "e3-evaluacion-aditiva-v2"
PREFIX = "runs/evaluacion_aditiva_v2"
SNAPSHOT_SHA = "5fe65596b99b334de197f3fc08811514c5ee6ea48b8447318057fea1b7cf9be1"
NPZ_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
EXTRA_SOURCES = {
    "src/__init__.py": "c7b20782a739f0118bbfbab216b4a3fb323edef4cb5c5eaf7f5cf482aaa18d74",
    "notebooks/preparar_e3_desarrollo_runtime.py": "9b4ef359b2407f197d9841e345f87ea769c72a7aec871b07c9563d9f852d22d0",
    "notebooks/preparar_e3_innovacion_runtime.py": "de3025c5f040cab4f69e1640494423a13a5640eb68999ec343897dfb507cd5e2",
    "requirements-e3-innovation-gpu.lock.txt": "85f80fa265764d6aa213ff4b375f946eb187acb58cb915311f3a5268122224d6",
}
OUTPUT_NAMES = ("predictions.csv", "comparison.csv", "summary.json")
COMMIT_IDENTITY = {"name": "E3 development storage", "email": "e3@users.noreply.github.com"}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def download(commit, relative, target, expected):
    target = Path(target)
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise RuntimeError("No se admiten enlaces simbólicos: " + str(target))
    if target.exists():
        if not target.is_file() or digest(target.read_bytes()) != expected:
            raise RuntimeError("El archivo local difiere; se conserva: " + str(target))
        return target.read_bytes()
    request = Request(f"https://raw.githubusercontent.com/{REPO}/{commit}/{relative}",
                      headers={"User-Agent": "E3-frozen-evaluation"})
    with urlopen(request, timeout=60) as response:
        payload = response.read(20 * 1024 * 1024 + 1)
    if len(payload) > 20 * 1024 * 1024 or digest(payload) != expected:
        raise RuntimeError("Descarga no coincide con el SHA256 fijado: " + relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
    os.replace(temporary, target)
    return payload


class NoApiRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        raise HTTPError(request.full_url, code, "API redirect rejected", headers, response)


def api_error_details(error, token):
    """Expose bounded validation fields only, never HTTP bodies or credentials."""
    try:
        record = json.loads(error.read(8192))
    except (ValueError, UnicodeError, OSError):
        return ""
    if not isinstance(record, dict):
        return ""
    values = []
    if isinstance(record.get("message"), str):
        values.append(record["message"])
    errors = record.get("errors", [])
    for item in errors[:4] if isinstance(errors, list) else []:
        if isinstance(item, dict):
            values.extend(f"{key}={item[key]}" for key in ("resource", "field", "code")
                          if isinstance(item.get(key), str))
    value = " | ".join(values)
    if token:
        value = value.replace(token, "[redacted]")
    value = re.sub(r"(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", "[redacted]", value)
    return " ".join(value.split())[:600]


class GitHub:
    def __init__(self, token):
        self.token = token

    def api(self, method, path, value=None, *, missing=False):
        body = None if value is None else json.dumps(value).encode()
        request = Request(f"https://api.github.com/repos/{REPO}/{path}", data=body,
                          method=method, headers={"Authorization": "Bearer " + self.token,
                          "Accept": "application/vnd.github+json", "Content-Type": "application/json",
                          "User-Agent": "E3-frozen-evaluation", "X-GitHub-Api-Version": "2022-11-28"})
        try:
            with build_opener(NoApiRedirect()).open(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as error:
            if missing and error.code == 404:
                return None
            detail = api_error_details(error, self.token) if error.code in (400, 409, 422) else ""
            suffix = "; " + detail if detail else ""
            raise RuntimeError(f"GitHub rechazó {method} {path}: HTTP {error.code}{suffix}") from None
        except URLError:
            raise RuntimeError(f"No se confirmó la solicitud GitHub: {method} {path}") from None

    def branch(self):
        ref = self.api("GET", "git/ref/heads/" + BRANCH, missing=True)
        if ref is None:
            return None, None, {}
        head = ref["object"]["sha"]
        commit = self.api("GET", "git/commits/" + head)
        tree_sha = commit["tree"]["sha"]
        tree = self.api("GET", "git/trees/" + tree_sha + "?recursive=1")
        if tree.get("truncated"):
            raise RuntimeError("El árbol remoto está incompleto")
        files = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
        allowed = {PREFIX + "/" + n for n in (*OUTPUT_NAMES, "plan.json", "publication.json")}
        if set(files) - allowed:
            raise RuntimeError("La rama contiene archivos ajenos a esta evaluación")
        return head, tree_sha, files

    def blob(self, sha):
        record = self.api("GET", "git/blobs/" + sha)
        if record.get("encoding") != "base64":
            raise RuntimeError("Codificación remota inesperada")
        return base64.b64decode(record["content"], validate=False)

    def publish(self, entries, head=None, tree_sha=None):
        tree_entries = []
        for path, payload in entries.items():
            blob = self.api("POST", "git/blobs", {
                "encoding": "base64", "content": base64.b64encode(payload).decode()})
            tree_entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        body = {"tree": tree_entries}
        if tree_sha is not None:
            body["base_tree"] = tree_sha
        tree = self.api("POST", "git/trees", body)
        commit = self.api("POST", "git/commits", {
            "message": "E3: evaluación descriptiva del checkpoint aditivo congelado",
            "tree": tree["sha"], "parents": [RESULT_COMMIT if head is None else head],
            "author": dict(COMMIT_IDENTITY), "committer": dict(COMMIT_IDENTITY)})
        if head is None:
            ref = self.api("POST", "git/refs", {"ref": "refs/heads/" + BRANCH, "sha": commit["sha"]})
        else:
            current = self.api("GET", "git/ref/heads/" + BRANCH)
            if current["object"]["sha"] != head:
                raise RuntimeError("La rama cambió durante la evaluación; no se sobrescribe")
            ref = self.api("PATCH", "git/refs/heads/" + BRANCH, {"sha": commit["sha"], "force": False})
        if ref["object"]["sha"] != commit["sha"]:
            raise RuntimeError("GitHub no confirmó el commit esperado")
        return commit["sha"]


def _token():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        try:
            from google.colab import userdata
            token = (userdata.get("GITHUB_TOKEN") or "").strip()
        except Exception:
            token = getpass("GITHUB_TOKEN (entrada oculta): ").strip()
    if not token:
        raise RuntimeError("Falta GITHUB_TOKEN con permiso Contents: write")
    return token


def validate_payloads(payloads, plan, inputs):
    """Admit complete results for either first upload or interrupted-upload recovery."""
    def require(condition):
        if not condition:
            raise RuntimeError("Resultados incompletos o distintos del plan; se conservan sin publicar")

    require(set(payloads) == set(OUTPUT_NAMES))
    summary = json.loads(payloads["summary.json"])
    manifest = json.loads((inputs / "manifest.json").read_bytes())
    expected = {"status": "completed", "scope": "historical_descriptive", "variant": plan["model"],
        "frozen_checkpoint_sha256": plan["checkpoint_sha256"], "checkpoint_sha256": plan["checkpoint_sha256"],
        "source_result_commit": RESULT_COMMIT, "result_source_commit": RESULT_COMMIT,
        "E2_source_commit": E2_COMMIT, "data_npz_sha256": NPZ_SHA, "evaluation_n": 608,
        "source_run_id": manifest["run_id"], "source_hashes": manifest["source_hashes"],
        "no_training": True, "rows_dropped": 0, "statistical_tests_performed": False,
        "R1_R3_verified": False, "winner_declared": False, "best_epoch_frozen_before_evaluation": 38,
        "seed": 20261116}
    require(all(summary.get(k) == v for k, v in expected.items()))
    input_names = ("snapshot_manifest.json", "manifest.json", "data_admission.json",
        "additive_positive_head/best.pt", "additive_positive_head/validation_predictions.csv",
        "current_cohort_arrays.npz", *plan["e2_input_hashes"])
    require(summary["input_hashes"] == {n: digest((inputs / n).read_bytes()) for n in input_names})
    require(summary["artifact_sha256"] == {n: digest(payloads[n]) for n in OUTPUT_NAMES if n != "summary.json"})
    roundtrip = summary["validation_roundtrip"]
    require(roundtrip["examples"] == 116 and roundtrip["passed_before_test_members_opened"] is True)
    require(math.isfinite(roundtrip["maximum_absolute_error"]) and 0 <= roundtrip["maximum_absolute_error"] <= 1e-7)
    training = summary["training"]
    require(all(training[k] == 0 for k in ("new_model_fits", "new_scaler_fits", "optimizer_steps",
                                         "bias_matching_calls", "calibration_fits")))
    predictions = list(csv.DictReader(io.StringIO(payloads["predictions.csv"].decode())))
    dates = json.loads((inputs / "e2_partition.json").read_bytes())["final_partition"]["evaluation"]["sample_IDs"]
    require(len(predictions) == 608 and [r["date"] for r in predictions] == dates)
    for row in predictions:
        require(all(math.isfinite(float(row[k])) and float(row[k]) >= 0
                    for k in ("target_sigma7", "prediction_sigma7", "known_floor")))
    errors = [float(r["prediction_sigma7"]) - float(r["target_sigma7"]) for r in predictions]
    rmse = math.sqrt(math.fsum(e * e for e in errors) / 608)
    mae = math.fsum(abs(e) for e in errors) / 608
    require(math.isclose(rmse, summary["evaluation"]["RMSE"], rel_tol=1e-12, abs_tol=1e-15))
    require(math.isclose(mae, summary["evaluation"]["MAE"], rel_tol=1e-12, abs_tol=1e-15))
    comparison = list(csv.DictReader(io.StringIO(payloads["comparison.csv"].decode())))
    original = list(csv.DictReader(io.StringIO((inputs / "e2_benchmark_A2.csv").read_text())))
    require(len(comparison) == 17 and [r["model"] for r in comparison[:16]] == [r["model"] for r in original])
    require(all(float(row[k]) == float(old[k]) for row, old in zip(comparison[:16], original)
                for k in ("RMSE", "MAE")))
    require(all(int(row["predictors"]) == int(old["R"]) for row, old in zip(comparison[:16], original)))
    require(all(int(row["evaluation_examples"]) == 608 for row in comparison))
    require(comparison[-1]["model"] == "CNN-LSTM-atencion_aditiva_v2"
            and int(comparison[-1]["predictors"]) == 1
            and float(comparison[-1]["RMSE"]) == summary["evaluation"]["RMSE"]
            and float(comparison[-1]["MAE"]) == summary["evaluation"]["MAE"])
    return summary


def local_payloads(output, plan_bytes, inputs, execution_plan):
    if not output.exists() or (output.is_dir() and not any(output.iterdir())):
        return None
    if (output.is_symlink() or not output.is_dir() or not execution_plan.is_file()
            or execution_plan.read_bytes() != plan_bytes
            or {p.name for p in output.iterdir()} != set(OUTPUT_NAMES)
            or any((output / n).is_symlink() for n in OUTPUT_NAMES)):
        raise RuntimeError("Hay resultados locales parciales o de otro plan; se conservan sin repetir la evaluación")
    payloads = {n: (output / n).read_bytes() for n in OUTPUT_NAMES}
    validate_payloads(payloads, json.loads(plan_bytes), inputs)
    return payloads


def run(workspace=None):
    workspace = Path("/content/e3_evaluacion_aditiva_v2") if workspace is None else Path(workspace)
    source, inputs = workspace / "source", workspace / "inputs"
    output = workspace / "output"
    snapshot = json.loads(download(RESULT_COMMIT, "runs/innovacion_v2/snapshot_manifest.json",
                                  inputs / "snapshot_manifest.json", SNAPSHOT_SHA))
    for name in ("data_admission.json", "manifest.json", "additive_positive_head/best.pt",
                 "additive_positive_head/validation_predictions.csv"):
        download(RESULT_COMMIT, "runs/innovacion_v2/" + name, inputs / name,
                 snapshot["files"][name]["sha256"])
    manifest = json.loads((inputs / "manifest.json").read_bytes())
    hashes = {**manifest["source_hashes"], **EXTRA_SOURCES}
    for path, sha in hashes.items():
        download(SOURCE_COMMIT, path, source / path, sha)
    data_path = "book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz"
    download("af3f1223ad5899ed63016d2af2d09bae8470ab00", data_path,
             inputs / "current_cohort_arrays.npz", NPZ_SHA)
    e2_inputs = {
        "e2_partition.json": ("book_e2/_static/e2_eda_current_v1/partition_audit.json",
                              "69ac07eebcfad0e946d2edaaf433871e4045696b95035e7b5252cf3409a7e02f"),
        "e2_benchmark_A2.csv": ("book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv",
                                "879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1"),
        "e2_data_admission.json": ("book_e2/_static/e2_benchmark_final_20261113/data_admission.json",
                                   "1d430f16a4d363055800fb4ec62cc4a7bc9062a131e3a792324b8440e5cebd9c"),
    }
    for name, (path, sha) in e2_inputs.items():
        download(E2_COMMIT, path, inputs / name, sha)
    worker = Path(__file__).with_name("worker.py")
    if worker.is_symlink() or not worker.is_file():
        raise RuntimeError("Falta el trabajador verificado junto al bootstrap")
    plan = {"version": "e3-frozen-additive-evaluation-v1", "source_commit": SOURCE_COMMIT,
        "source_hashes": hashes, "worker_sha256": digest(worker.read_bytes()),
        "frozen_result_commit": RESULT_COMMIT, "snapshot_sha256": SNAPSHOT_SHA,
        "checkpoint_sha256": snapshot["files"]["additive_positive_head/best.pt"]["sha256"],
        "scaler_admission_sha256": snapshot["files"]["data_admission.json"]["sha256"],
        "npz_sha256": NPZ_SHA, "e2_commit": E2_COMMIT,
        "e2_input_hashes": {n: h for n, (_, h) in e2_inputs.items()},
        "model": "additive_positive_head", "checkpoint_epoch": 38,
        "fit_rows_e3": 532, "fit_rows_e2_neural": 655, "e3_seeds": 1, "neural_e2_seeds": 6,
        "evaluation_n": 608, "evaluation_start": "2025-01-01", "evaluation_end": "2026-08-31",
        "training_performed": False, "scaler_refit": False, "checkpoint_selection_on_test": False,
        "scope": "retrospective descriptive comparison, unequal training sizes and seed counts; no R1/R3 claim"}
    plan_bytes = canonical(plan)
    github = GitHub(_token())
    try:
        head, tree_sha, files = github.branch()
        if head is None:
            github.publish({PREFIX + "/plan.json": plan_bytes})
            head, tree_sha, files = github.branch()
        if PREFIX + "/plan.json" not in files or github.blob(files[PREFIX + "/plan.json"]) != plan_bytes:
            raise RuntimeError("El plan remoto difiere; se conserva la evaluación existente")
        complete = PREFIX + "/publication.json"
        if complete in files:
            receipt = json.loads(github.blob(files[complete]))
            if receipt["plan_sha256"] != digest(plan_bytes) or set(receipt["files"]) != set(OUTPUT_NAMES):
                raise RuntimeError("El recibo remoto no corresponde al plan")
            payloads = {}
            for name, record in receipt["files"].items():
                payload = github.blob(files[PREFIX + "/" + name])
                if digest(payload) != record["sha256"] or len(payload) != record["bytes"]:
                    raise RuntimeError("Los resultados remotos difieren del recibo")
                payloads[name] = payload
            validate_payloads(payloads, plan, inputs)
            print("La evaluación ya estaba terminada; se conservan sus resultados.")
        else:
            if set(files) != {PREFIX + "/plan.json"}:
                raise RuntimeError("Hay resultados sin recibo completo; se conservan sin sobrescribir")
            execution_plan = workspace / "execution_plan.json"
            payloads = local_payloads(output, plan_bytes, inputs, execution_plan)
            if payloads is None:
                if execution_plan.exists():
                    if execution_plan.is_symlink() or execution_plan.read_bytes() != plan_bytes:
                        raise RuntimeError("El plan local difiere; no se inicia evaluación")
                else:
                    with execution_plan.open("xb") as stream:
                        stream.write(plan_bytes)
                helper_path = source / "notebooks/preparar_e3_innovacion_runtime.py"
                spec = importlib.util.spec_from_file_location("e3_frozen_runtime", helper_path)
                helper = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(helper)
                previous = Path("/content/e3_momentos_github/runtime")
                runtime_dir = previous if (previous / "runtime_identity.json").is_file() else workspace / "runtime"
                python, environment, _ = helper.preparar_entorno(runtime_dir, device="cuda")
                environment = {k: v for k, v in environment.items() if not re.search(
                    r"(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL|API_KEY)(?:_|$)", k.upper())}
                print("Checkpoint y receta fijados en GitHub antes de evaluar.", flush=True)
                subprocess.run([str(python), str(worker), "--source-root", str(source),
                                "--inputs", str(inputs), "--output", str(output), "--device", "cuda"],
                               env=environment, check=True)
                payloads = {name: (output / name).read_bytes() for name in OUTPUT_NAMES}
                validate_payloads(payloads, plan, inputs)
            else:
                print("Se recuperan los resultados locales completos para publicarlos sin repetir inferencia.")
            publication = {"version": "e3-frozen-evaluation-publication-v1",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": digest(plan_bytes), "files": {
                    n: {"sha256": digest(p), "bytes": len(p)} for n, p in payloads.items()}}
            entries = {PREFIX + "/" + n: p for n, p in payloads.items()}
            entries[complete] = canonical(publication)
            head = github.publish(entries, head, tree_sha)
        print("\nComparación descriptiva sobre las mismas 608 fechas:")
        print(f"{'Modelo':<39} {'RMSE':>12} {'MAE':>12} {'Predictores':>12}")
        for row in csv.DictReader(io.StringIO(payloads["comparison.csv"].decode())):
            print(f"{row['model']:<39} {float(row['RMSE']):>12.9f} "
                  f"{float(row['MAE']):>12.9f} {row['predictors']:>12}")
        print("E2 neuronal: media de seis métricas por semilla; E3: una semilla congelada.")
        print("Entrenamiento diferente: E3 532 ejemplos; redes E2 655. No es una prueba de superioridad.")
        url = f"https://github.com/{REPO}/tree/{head}/{PREFIX}"
        print("Resultados guardados en GitHub:", url)
        return url
    finally:
        github.token = ""
