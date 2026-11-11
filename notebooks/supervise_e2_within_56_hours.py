"""Resume only clean session-budget pauses; never extend the authorized 56-hour total."""

import hashlib
import json
import math
from pathlib import Path

MAX_TOTAL_SECONDS = 56 * 3600.0
CONTROLLER_COMMIT = "8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9"
BRANCH = "campaigns/e2-cuda-crop-11-v2"


def checked_report(output):
    output = Path(output)
    config = json.loads((output / "campaign.json").read_text())
    status = json.loads((output / "status.json").read_text())
    plan = config["plan"]
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if config["plan_sha256"] != digest or status["plan_sha256"] != digest:
        raise ValueError("campaign_or_report_plan_mismatch")
    if status["remote_branch"] != BRANCH or plan["remote_storage"]["branch"] != BRANCH:
        raise ValueError("unexpected_campaign_branch")
    budget = status["budget"]
    if budget["plan_sha256"] != digest:
        raise ValueError("budget_plan_mismatch")
    values = [plan["legacy_cache"]["spent_seconds"], budget["campaign_budget_seconds"],
              budget["charged_wall_seconds"], budget["remaining_campaign_seconds"],
              status["combined_charged_seconds"]]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("invalid_budget_metadata")
    legacy, limit, charged, remaining, combined = values
    if (not math.isclose(combined, legacy + charged, rel_tol=0, abs_tol=1e-5)
            or not math.isclose(remaining, max(0., limit-charged), rel_tol=0, abs_tol=1e-5)):
        raise ValueError("inconsistent_budget_metadata")
    count = status["completed_jobs"]
    if (type(count) is not int or type(budget["completed_jobs"]) is not int
            or count < 0 or budget["completed_jobs"] != count):
        raise ValueError("invalid_completed_result_counter")
    return status, legacy + limit


def supervise(namespace, invoke):
    """Wait for the current invocation, then resume session pauses one at a time.

    Existing immutable records and budget enforcement are owned by the same
    frozen controller. No budget extension, signal, kill or error retry occurs.
    """
    previous_count = None
    resumptions = 0
    while True:
        process = namespace.get("E2_OPT_ACTIVE_PROCESS")
        if process is None:
            raise RuntimeError("No hay un controlador conocido; conserva la sesión preparada.")
        if process.poll() is None:
            print("Supervisor: esperando el cierre normal del controlador actual, sin interrumpirlo.", flush=True)
            return_code = process.wait()
        else:
            return_code = process.poll()
        if return_code != 0:
            print("Supervisor detenido: el controlador salió con código", return_code, flush=True)
            return {"stop_reason": "controller_nonzero_exit", "return_code": return_code,
                    "automatic_resumptions": resumptions}
        status, total = checked_report(namespace["OPTIMIZATION_OUTPUT"])
        if total > MAX_TOTAL_SECONDS + 1e-5:
            raise ValueError("combined_budget_exceeds_authorized_56_hours")
        state = status["status"]
        if state != "paused_session_budget":
            print("Supervisor detenido con estado:", state, flush=True)
            if state == "paused_campaign_budget":
                print("No se amplía el límite. El saldo puede ser insuficiente para reservar otro ajuste.", flush=True)
            return {"stop_reason": state, "completed_jobs": status["completed_jobs"],
                    "automatic_resumptions": resumptions}
        if status.get("pause_reason") != "session_budget":
            raise ValueError("session_pause_reason_mismatch")
        count = status["completed_jobs"]
        if previous_count is not None and count <= previous_count:
            raise RuntimeError("session_resume_made_no_new_progress; automatic_retry_stopped")
        if status["budget"]["remaining_campaign_seconds"] <= 0:
            print("Supervisor detenido: sin saldo de campaña.", flush=True)
            return {"stop_reason": "no_remaining_campaign_budget", "completed_jobs": count,
                    "automatic_resumptions": resumptions}
        previous_count = count
        resumptions += 1
        print("Supervisor: reanudación automática", resumptions,
              "tras pausa de sesión; límite acumulado", total/3600,
              "horas; resultados conservados", count, flush=True)
        invoke(namespace)


def run_prepared_colab(namespace):
    import urllib.request
    if namespace.get("OPTIMIZATION_COMMIT") != CONTROLLER_COMMIT:
        raise RuntimeError("Se requiere el controlador vigente y la misma sesión preparada.")
    required = ("OPTIMIZATION_OUTPUT", "E2_OPT_ACTIVE_PROCESS", "GPU_PYTHON", "GITHUB_ENV")
    if any(name not in namespace for name in required):
        raise RuntimeError("Faltan variables del entrenamiento; no reinicies el kernel.")
    url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
           "91a7bc55e34fab6313e573f7fe11dda15706982a/"
           "notebooks/entregable2_optimizacion_11_origenes.ipynb")
    raw = urllib.request.urlopen(url, timeout=30).read()
    if hashlib.sha256(raw).hexdigest() != "43b45a6405a696341c8727384904de42ca7250c529a92766320f094cd512278d":
        raise RuntimeError("notebook_source_hash_mismatch")
    code = "".join(json.loads(raw)["cells"][17]["source"])
    if not code.startswith("RUN_OPTIMIZED_SELECTION = False"):
        raise RuntimeError("unexpected_frozen_controller_cell")
    code = code.replace("RUN_OPTIMIZED_SELECTION = False", "RUN_OPTIMIZED_SELECTION = True", 1)
    compiled = compile(code, "seccion8_reanudacion_automatica_sin_ampliar", "exec")
    print("Supervisor: sólo reanuda pausas de sesión; no amplía las 56 horas ni evita cierres de Colab.", flush=True)
    result = supervise(namespace, lambda ns: exec(compiled, ns))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    run_prepared_colab(globals())
