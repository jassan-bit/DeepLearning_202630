"""One Colab cell: inspect the sealed interruption, request a new TOTAL, recover once."""


def run_prepared_colab(namespace):
    import hashlib
    import json
    import math
    from pathlib import Path
    import re
    import signal
    import subprocess
    import threading
    import uuid

    required = ("WORKSPACE", "GPU_PYTHON", "REPO", "GITHUB_ENV", "RUN_SEARCH",
                "RETRY_LOST_ONCE", "OPTIMIZATION_SOURCE", "OPTIMIZATION_COMMIT",
                "OPTIMIZATION_OUTPUT", "CAMPAIGN_OUTPUT", "DATA_CACHE",
                "SOURCE_READY", "PLAN_READY", "REMOTE_READY", "GPU_READY",
                "SYNTHETIC_READY", "DATA_READY", "E2_OPT_RECOVERY_COMMIT")
    if any(name not in namespace for name in required):
        raise RuntimeError("Conserva la sesión recién preparada; faltan variables de E2.")
    if (not namespace["RUN_SEARCH"] or namespace["RETRY_LOST_ONCE"]
            or not all(namespace[name] for name in ("SOURCE_READY", "PLAN_READY", "REMOTE_READY",
                                                    "GPU_READY", "SYNTHETIC_READY", "DATA_READY"))):
        raise RuntimeError("Las comprobaciones 1–7 deben estar completas; conserva RETRY_LOST_ONCE=False.")
    if namespace["OPTIMIZATION_COMMIT"] != "8e3ab6cfb1b3fc3c804f0c914fa5d92b661058d9":
        raise RuntimeError("Se requiere el controlador optimizado congelado.")
    for prefix in ("E2_OPT", "E2_GUIDE", "E2_RECOVERY_V3", "E2_RECOVERY_V4"):
        process = namespace.get(prefix + "_ACTIVE_PROCESS")
        if process is not None and process.poll() is None:
            raise RuntimeError("Hay un controlador activo; no se lanzó otro.")
    markers = ("run_e2_cuda_search.py", "run_e2_cuda_recovery_", "e2_guide_schedule.py",
               "run_e2_optimized_selection", "run_e2_optimized_recovery_v1.py")
    rows = subprocess.run(["ps", "-eo", "pid=,args="], check=True,
                          capture_output=True, text=True).stdout
    if any(marker in rows for marker in markers):
        raise RuntimeError("Se encontró otro proceso de entrenamiento; no se lanzó otro.")
    commit = namespace["E2_OPT_RECOVERY_COMMIT"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Falta el commit inmutable de recuperación.")
    source = Path(namespace["WORKSPACE"]) / "optimized-recovery-20261112-r1"
    environment = namespace["GITHUB_ENV"]
    def checked(args, **kwargs):
        return subprocess.run([str(v) for v in args], env=environment,
                              text=True, check=True, **kwargs)
    if not source.exists():
        source.mkdir(parents=True)
        checked(["git", "init", source])
        checked(["git", "-C", source, "remote", "add", "origin",
                 "https://github.com/jassan-bit/DeepLearning_202630.git"])
        checked(["git", "-C", source, "config", "core.sparseCheckout", "true"])
        checked(["git", "-C", source, "config", "remote.origin.promisor", "true"])
        checked(["git", "-C", source, "config", "remote.origin.partialclonefilter", "blob:none"])
        paths = ["/scripts/e2_optimized_reconciliation_v1.py", "/scripts/e2_optimized_recovery_git_v1.py",
                 "/scripts/run_e2_optimized_recovery_v1.py",
                 "/notebooks/recover_e2_colab_loss_20261112.py",
                 "/experiments/e2_optimized_recovery_20261112/registration.json"]
        checked(["git", "-C", source, "sparse-checkout", "set", "--no-cone", "--stdin"],
                input="\n".join(paths) + "\n")
        checked(["git", "-C", source, "fetch", "--filter=blob:none", "--depth=1", "origin", commit])
        checked(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
    actual = checked(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    if actual != commit:
        raise RuntimeError("El checkout de recuperación pertenece a otra versión.")
    registration = json.loads((source / "experiments/e2_optimized_recovery_20261112/registration.json").read_bytes())
    for path, expected in registration["implementation_sha256"].items():
        if hashlib.sha256((source / path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("La implementación de recuperación cambió.")
    command = [namespace["GPU_PYTHON"], "-u", source / "scripts/run_e2_optimized_recovery_v1.py",
        "--engine-root", namespace["REPO"], "--optimization-root", namespace["OPTIMIZATION_SOURCE"],
        "--output", namespace["OPTIMIZATION_OUTPUT"], "--legacy-cache", namespace["CAMPAIGN_OUTPUT"],
        "--data-root", namespace["DATA_CACHE"], "--source-root-itransformer",
        Path(namespace["REPO"]) / "work/author_sources/iTransformer", "--recovery-commit", commit]
    diagnostics = Path(namespace["WORKSPACE"]) / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    inspection_path = diagnostics / ("optimized-recovery-inspection-" + uuid.uuid4().hex + ".json")
    print("Inspeccionando historial y presupuesto; todavía no se entrena.", flush=True)
    inspected = subprocess.run([str(v) for v in (*command, "--inspect", "--report", inspection_path)],
        env=environment, text=True, capture_output=True)
    if not inspection_path.is_file():
        raise RuntimeError("La inspección no produjo informe; revisa su código de salida " + str(inspected.returncode))
    inspection = json.loads(inspection_path.read_bytes())
    namespace["E2_OPT_RECOVERY_INSPECTION"] = inspection
    print(json.dumps(inspection, ensure_ascii=False, indent=2), flush=True)
    if inspection["status"] == "already_recovered_use_normal_controller":
        print("Este ajuste ya fue recuperado. Continúa con la sección 8 del notebook; no se repetirá la recuperación.")
        return inspection
    if inspection["status"] != "inspection_only":
        raise RuntimeError("La inspección no permitió preparar la recuperación.")
    if inspection["recovery_authorization_exists"]:
        total = inspection["registered_total_hours"]
        print("Se conserva el límite total ya registrado:", total, "horas.", flush=True)
    else:
        print("El límite actual es 56 horas acumuladas. El saldo conservador no permite otra reserva completa.")
        print("Indica un NUEVO LÍMITE TOTAL: 57 añade 1 hora; 64 añade 8 horas. No garantiza completar E2.")
        answer = input("Nuevo límite TOTAL en horas (vacío o 0 = cancelar): ").strip()
        if not answer or answer == "0":
            print("Cancelado: no se amplió presupuesto ni se inició recuperación.")
            return inspection
        try:
            total = float(answer.replace(",", "."))
        except ValueError:
            raise RuntimeError("Escribe un número de horas totales.") from None
        if not math.isfinite(total) or total <= 56:
            raise RuntimeError("El nuevo TOTAL debe superar 56; no se cambió el presupuesto.")
    report_path = diagnostics / ("optimized-recovery-report-" + uuid.uuid4().hex + ".json")
    log_path = diagnostics / ("optimized-recovery-log-" + uuid.uuid4().hex + ".txt")
    argv = [str(x) for x in (*command, "--execute", "--total-hours", total, "--report", report_path)]
    process = subprocess.Popen(argv, cwd=source, env=environment, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    namespace["E2_OPT_ACTIVE_PROCESS"] = process
    namespace["E2_OPT_LOG_PATH"] = log_path
    namespace["E2_OPT_RECOVERY_REPORT"] = report_path
    def redact(text):
        for key in ("GITHUB_TOKEN", "GH_TOKEN"):
            secret = environment.get(key)
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text
    def stream():
        with log_path.open("x", encoding="utf-8") as log:
            for line in process.stdout:
                safe = redact(line)
                log.write(safe)
                log.flush()
                print(safe, end="", flush=True)
    thread = threading.Thread(target=stream, daemon=True)
    thread.start()
    print("Controlador de recuperación PID:", process.pid, flush=True)
    print("Primero registra la autorización; luego recupera una vez y continúa el plan.", flush=True)
    interrupted = False
    while True:
        try:
            code = process.wait()
            break
        except KeyboardInterrupt:
            if not interrupted:
                interrupted = True
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                print("Interrupción enviada; esperando cierre del trabajador y del diario.", flush=True)
    thread.join(timeout=10)
    namespace["E2_OPT_RETURN_CODE"] = code
    print("Código de salida:", code, "Informe:", report_path, flush=True)
    if report_path.is_file():
        print(redact(report_path.read_text()), flush=True)
    return code


if __name__ == "__main__":
    run_prepared_colab(globals())
