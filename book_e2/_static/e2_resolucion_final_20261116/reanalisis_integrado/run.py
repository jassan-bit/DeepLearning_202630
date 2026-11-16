"""Offline integration of existing E2 reanalyses, with no fitting or forwards.

Extract the complete distribution, install requirements.txt, then run python
run.py.  Each native module runs on a fresh extracted copy.  Native archives,
historical scientific results and earlier deliverables are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
import zipfile

VERSION = "e2-integrated-existing-reanalysis-v1"
IDS = ("D02", "D03", "D04", "D06", "D07", "D08", "D09", "D10",
       "negative_controls", "conditional_inference", "input_dependence", "HPO_descriptive", "Scopus7500")
SCRIPTS = {**{i: "run.py" for i in IDS if i not in ("input_dependence", "HPO_descriptive")},
           "input_dependence": "replay_shortcut_statistics.py", "HPO_descriptive": "replay_descriptive.py"}
TERMINAL = {"completed", "failed_execution", "failed_result_validation", "failed_preparation",
            "blocked_inputs", "blocked_dependencies", "not_run_budget_exhausted"}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def write(path, value, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if replace:
        temp = path.with_suffix(path.suffix + ".pending")
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(text)


def inside(root, name, *, file=True):
    part = PurePosixPath(name)
    if not name or "\\" in name or part.is_absolute() or ".." in part.parts or str(part) != name:
        raise ValueError("Unsafe relative package path: " + str(name))
    result = root.joinpath(*part.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes package")
    if any(p.is_symlink() for p in (result, *result.parents) if p.is_relative_to(root)):
        raise ValueError("Symlinked inputs are not admitted")
    if file and not result.is_file():
        raise ValueError("Missing regular input: " + name)
    return result


def record(root, path):
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def inventory(root):
    return [record(root, p) for p in sorted(root.rglob("*"))
            if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts]


def verify_archive(root, task):
    path = inside(root, task["archive"])
    if path.stat().st_size != task["archive_bytes"] or sha(path) != task["archive_sha256"]:
        raise ValueError("Native archive differs from its pinned bytes: " + task["id"])
    return path


def safe_extract(path, output):
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = member.filename.rstrip("/")
            inside(output, name, file=False)
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Symlink member in native archive")
        archive.extractall(output)


def catalog(root):
    data = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    if data.get("version") != VERSION or tuple(t["id"] for t in data["tasks"]) != IDS:
        raise ValueError("Wrong catalogue version or thirteen-task schedule")
    for task in data["tasks"]:
        expected = ["--inventory"] if task["id"] == "D02" else []
        if task["script"] != SCRIPTS[task["id"]] or task["arguments"] != expected:
            raise ValueError("Unapproved command: fitting and forward modes are excluded")
        inside(root, task["archive"], file=False)
        if task["root_directory"]:
            inside(root, task["root_directory"], file=False)
    return data


def observed_runtime():
    packages = {}
    for name in ("numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "executable": sys.executable,
            "system": platform.system(), "release": platform.release(), "machine": platform.machine(),
            "os_release": platform.freedesktop_os_release() if sys.platform == "linux" else {},
            "packages": packages}


def expected_result(task, module_root):
    """Inspect the module's real outputs, not just its process return code."""
    identifier = task["id"]
    result = json.loads(inside(module_root, task["result_json"]).read_text(encoding="utf-8"))
    details = {"native_result_status": result.get("status"), "scope": task["scope"]}
    for key, value in task.get("required_equal", {}).items():
        cursor = result
        for part in key.split("."):
            cursor = cursor[part]
        if cursor != value:
            raise ValueError(identifier + ": result field differs: " + key)
        details[key] = cursor
    for key, maximum in task.get("required_maximum", {}).items():
        value = float(result[key])
        if not math.isfinite(value) or value > maximum:
            raise ValueError(identifier + ": numeric verification bound exceeded: " + key)
        details[key] = value
    if identifier == "D02":
        if len(result["models"]) != 13 or result["scope"]["new_model_fits"] != 0 or "smoke_cpu" in result:
            raise ValueError("D02 must verify thirteen sources without a fixture forward")
        details.update(models_verified=13, new_model_forwards=0)
    if identifier in ("D04", "D07", "D09"):
        if not result.get("checks") or any(c["correcto"] is not True for c in result["checks"]):
            raise ValueError("A documentary schema/integrity check failed")
        details["schema_integrity_checks"] = len(result["checks"])
        details["scientific_audit_acceptance_inferred_from_exit_code"] = False
    if identifier == "D03":
        details["cohorts"] = {"training": 655, "calibration": 53, "evaluation": 608}
    if identifier == "conditional_inference":
        tables = []
        for name in ("conditional_OOF_test_comparisons_39.csv", "finite_loss_variance_components_15.csv"):
            actual = inside(module_root, "output/" + name)
            reference = inside(module_root, "reference/" + name)
            if actual.read_bytes() != reference.read_bytes():
                raise ValueError("Inference table is not byte-identical to its own reference")
            tables.append({"name": name, "sha256": sha(actual)})
        details["tables_byte_equal_to_reference"] = tables
    return details


def artifact_records(root, before):
    actual = {row["path"]: row for row in inventory(root)}
    return [row for path, row in actual.items() if before.get(path) != row]


def report(output, state):
    write(output / "execution_report.json", state, replace=True)
    with (output / "task_statuses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "title", "status", "returncode", "elapsed_seconds", "scope"))
        writer.writeheader()
        writer.writerows({k: t.get(k, "") for k in writer.fieldnames} for t in state["tasks"])
    rows = []
    for task in state["tasks"]:
        links = " ".join('<a href="' + html.escape(a["path"]) + '">' + html.escape(Path(a["path"]).name) + '</a>'
                         for a in task.get("generated_artifacts", []) if a["path"].endswith((".html", ".csv", ".json", ".svg")))
        rows.append("<tr><td>" + html.escape(task["id"]) + "</td><td>" + html.escape(task["status"]) +
                    "</td><td>" + html.escape(task["scope"]) + "</td><td>" + links + "</td></tr>")
    page = ('<!doctype html><html lang="es"><meta charset="utf-8"><title>Reanálisis integrado E2</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:30px auto}td,th{border:1px solid #ccc;padding:8px;vertical-align:top}table{border-collapse:collapse}td a{display:block}</style>'
            '<h1>Reanálisis integrado de resultados existentes</h1><p>Estado técnico: ' + html.escape(state["status"]) + '</p>'
            '<p>Los comandos y sus comprobaciones no certifican ausencia de fuga, adecuación de todos los modelos ni entrenamiento integral. '
            'No se ejecutan ajustes, forwards, HPO o selección de arquitecturas.</p>'
            '<p>El cierre Scopus7500 sustituye las solicitudes del formulario bibliográfico histórico incluido en el módulo HPO. '
            'No se requieren nuevas exportaciones Scopus.</p>'
            '<table><thead><tr><th>Módulo</th><th>Estado terminal</th><th>Alcance</th><th>Salidas</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
            '<p><a href="execution_report.json">Recibos, hashes y diferencias numéricas por módulo</a> · '
            '<a href="task_statuses.csv">Estados CSV</a></p></html>')
    (output / "informe.html").write_text(page, encoding="utf-8")


def run(args):
    root = Path(__file__).resolve().parent
    data = catalog(root)
    observed = observed_runtime()
    archives, input_errors = {}, []
    for task in data["tasks"]:
        try:
            archives[task["id"]] = verify_archive(root, task)
        except (OSError, ValueError) as error:
            input_errors.append(str(error))
    if args.check:
        print(json.dumps({"status": "verified" if not input_errors else "blocked_inputs", "archives": len(archives),
                          "errors": input_errors, "runtime": observed, "new_model_fits": 0, "new_model_forwards": 0}))
        return 0 if not input_errors else 2
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0:
        raise ValueError("A positive finite local time limit is required")
    output = args.output
    if output is None:
        output = root.parent / "resultado_reanalisis_integrado"
        number = 2
        while output.exists():
            output = root.parent / ("resultado_reanalisis_integrado_" + str(number))
            number += 1
    output = output.absolute()
    if output.resolve().is_relative_to(root.resolve()) or output.exists():
        raise ValueError("Choose a new output directory outside the immutable package")
    output.mkdir(parents=True)
    missing = {k: {"required": v, "observed": observed["packages"][k]}
               for k, v in data["runtime_packages"].items() if observed["packages"][k] != v}
    state = {"version": VERSION, "scope": "integrated_existing_reanalysis", "status": "running",
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "runner_sha256": sha(__file__),
             "catalog_sha256": sha(root / "catalog.json"), "runtime": observed,
             "input_errors": input_errors, "dependency_mismatches": missing,
             "new_model_fits": 0, "new_model_forwards": 0, "full_historical_campaign_retrained": False,
             "all_book_tables_and_figures_regenerated": False,
             "scientific_audit_acceptance_inferred_from_task_success": False,
             "Scopus_scope": "closed7500; historical recovery forms are superseded",
             "tasks": [{"id": t["id"], "title": t["title"], "scope": t["scope"], "status": "pending"} for t in data["tasks"]]}
    start = time.monotonic()
    for spec, task in zip(data["tasks"], state["tasks"], strict=True):
        if input_errors or missing:
            task["status"] = "blocked_inputs" if input_errors else "blocked_dependencies"
            continue
        remaining = args.cap_seconds - (time.monotonic() - start)
        if remaining < 2:
            task["status"] = "not_run_budget_exhausted"
            continue
        task_start = time.monotonic()
        work = output / "modules" / spec["id"]
        log = work / "worker.log"
        try:
            safe_extract(archives[spec["id"]], work / "extracted")
            module = work / "extracted" / spec["root_directory"] if spec["root_directory"] else work / "extracted"
            script = inside(module, spec["script"])
            before = {r["path"]: r for r in inventory(module)}
            command = [sys.executable, "-I", str(script), *spec["arguments"]]
            if spec["id"] == "input_dependence":
                command.extend(["--output", str(module / "replay_result.json")])
            task["command"] = command
            environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = "2"
            with log.open("x", encoding="utf-8") as stream:
                try:
                    completed = subprocess.run(command, cwd=module, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                               timeout=max(0.1, args.cap_seconds - (time.monotonic() - start) - 1))
                    task["returncode"] = completed.returncode
                except subprocess.TimeoutExpired:
                    task["returncode"] = 124
            if task["returncode"] != 0:
                task["status"] = "failed_execution"
            else:
                try:
                    task["result_verification"] = expected_result(spec, module)
                    task["status"] = "completed"
                except (OSError, ValueError, KeyError, TypeError) as error:
                    task["status"] = "failed_result_validation"
                    task["error"] = str(error)
            changed = artifact_records(module, before)
            task["generated_artifacts"] = [{**r, "path": (module / r["path"]).relative_to(output).as_posix()} for r in changed]
            task["native_archive_sha256"] = spec["archive_sha256"]
            task["log_sha256"] = sha(log)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            task["status"] = "failed_preparation"
            task["error"] = str(error)
        finally:
            task["elapsed_seconds"] = time.monotonic() - task_start
            report(output, state)
    if any(t["status"] not in TERMINAL for t in state["tasks"]):
        raise RuntimeError("A task was left without a terminal status")
    completed = sum(t["status"] == "completed" for t in state["tasks"])
    state.update(status="completed_declared_reanalysis_scope" if completed == len(IDS) else "incomplete_with_terminal_statuses",
                 completed_tasks=completed, total_tasks=len(IDS), elapsed_wall_seconds=time.monotonic() - start,
                 completed_at_utc=datetime.now(timezone.utc).isoformat())
    report(output, state)
    print(json.dumps({"status": state["status"], "completed_tasks": completed, "total_tasks": len(IDS),
                      "output": str(output), "new_model_fits": 0, "new_model_forwards": 0,
                      "full_historical_campaign_retrained": False}))
    return 0 if completed == len(IDS) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify archive hashes only, without numerical reanalysis")
    parser.add_argument("--output", type=Path, help="Fresh directory outside the package")
    parser.add_argument("--cap-seconds", type=float, default=600, help="Local elapsed limit; not a reset of the project budget")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
