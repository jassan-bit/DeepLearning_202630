"""Colab entrypoint: ask for explicit total hours, register, then resume saved work."""


def resume_e2_after_budget_pause(namespace):
    import hashlib
    import json
    import math
    from pathlib import Path
    import subprocess
    import urllib.request

    required = ("WORKSPACE", "GPU_PYTHON", "OPTIMIZATION_SOURCE", "OPTIMIZATION_COMMIT",
                "REPO", "GITHUB_ENV", "RUN_SEARCH", "RETRY_LOST_ONCE")
    if any(key not in namespace for key in required):
        raise RuntimeError("Faltan variables: usa la misma sesión preparada de Colab, sin reiniciarla.")
    for prefix in ("E2_OPT", "E2_RECOVERY_V4", "E2_RECOVERY_V3", "E2_GUIDE"):
        process = namespace.get(prefix + "_ACTIVE_PROCESS")
        if process is not None and process.poll() is None:
            raise RuntimeError("Hay un controlador activo: espera su cierre.")
    markers = ("run_e2_cuda_search.py", "run_e2_cuda_recovery_", "e2_guide_schedule.py", "run_e2_optimized_selection")
    processes = subprocess.run(["ps", "-eo", "args="], check=True, capture_output=True, text=True).stdout
    if any(any(marker in row for marker in markers) for row in processes.splitlines()):
        raise RuntimeError("Se encontró otro controlador; no se modificó el presupuesto.")
    if not namespace["RUN_SEARCH"] or namespace["RETRY_LOST_ONCE"]:
        raise RuntimeError("Conserva RUN_SEARCH=True y RETRY_LOST_ONCE=False.")
    if not all(namespace.get(key, False) for key in
            ("SOURCE_READY", "PLAN_READY", "REMOTE_READY", "GPU_READY", "SYNTHETIC_READY", "DATA_READY")):
        raise RuntimeError("Completa la preparación 1–7 antes de autorizar una ampliación.")
    if namespace["OPTIMIZATION_COMMIT"] != "8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9":
        raise RuntimeError("Esta celda requiere el controlador vigente con corrección JSON.")
    print("La campaña pausó por presupuesto. El límite inicial combinado era 48 horas.", flush=True)
    print("Indica un NUEVO LÍMITE TOTAL: por ejemplo, 56 añade 8 horas al inicial; no significa 56 horas adicionales.", flush=True)
    print("Sólo se ampliará el tiempo permitido; modelos, once orígenes, semillas y resultados se conservan.", flush=True)
    answer = input("Nuevo límite TOTAL en horas (vacío o 0 = cancelar): ").strip()
    if not answer:
        print("Cancelado: sin cambios de presupuesto ni entrenamiento.")
        return
    hours = float(answer.replace(",", "."))
    if hours == 0:
        print("Cancelado: sin cambios de presupuesto ni entrenamiento.")
        return
    if not math.isfinite(hours) or hours <= 48:
        raise ValueError("Escribe un límite total mayor que 48, o deja vacío para cancelar.")
    helper_url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
        "7d0b7be4f9bba068d3fd2d405921d17bb78bb7df/scripts/extend_e2_optimized_budget.py")
    raw = urllib.request.urlopen(helper_url, timeout=30).read()
    if hashlib.sha256(raw).hexdigest() != "1a253aef710edb832a61f2cbc9e14ed659dc2fd9939ebe614b31350ca591be41":
        raise RuntimeError("La descarga no coincide con la versión fijada del ampliador.")
    helper_path = Path(namespace["WORKSPACE"]) / "extend_e2_optimized_budget.py"
    helper_path.write_bytes(raw)
    print("Registrando la ampliación autorizada antes de reanudar...", flush=True)
    result = subprocess.run([str(namespace["GPU_PYTHON"]), "-u", str(helper_path),
        "--controller-root", str(namespace["OPTIMIZATION_SOURCE"]),
        "--engine-root", str(namespace["REPO"]),
        "--controller-commit", namespace["OPTIMIZATION_COMMIT"],
        "--total-hours", str(hours), "--confirmed-by-user"],
        env=namespace["GITHUB_ENV"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=600)
    output = result.stdout
    for key in ("GITHUB_TOKEN", "GH_TOKEN"):
        secret = namespace["GITHUB_ENV"].get(key)
        if secret:
            output = output.replace(secret, "[REDACTED]")
    print(output, flush=True)
    if result.returncode:
        raise RuntimeError("No se confirmó la ampliación. No se lanzó entrenamiento; conserva este diagnóstico.")
    notebook_url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"
        "91a7bc55e34fab6313e573f7fe11dda15706982a/notebooks/entregable2_optimizacion_11_origenes.ipynb")
    notebook_raw = urllib.request.urlopen(notebook_url, timeout=30).read()
    if hashlib.sha256(notebook_raw).hexdigest() != "43b45a6405a696341c8727384904de42ca7250c529a92766320f094cd512278d":
        raise RuntimeError("El notebook descargado no coincide con la versión fijada.")
    notebook = json.loads(notebook_raw)
    code = "".join(notebook["cells"][17]["source"])
    if not code.startswith("RUN_OPTIMIZED_SELECTION = False"):
        raise RuntimeError("La sección de reanudación no coincide con la versión fijada.")
    code = code.replace("RUN_OPTIMIZED_SELECTION = False", "RUN_OPTIMIZED_SELECTION = True", 1)
    print("Ampliación confirmada. Reanudando desde las corridas guardadas...", flush=True)
    exec(compile(code, "seccion_8_reanudada_con_presupuesto_registrado", "exec"), namespace)


resume_e2_after_budget_pause(globals())
