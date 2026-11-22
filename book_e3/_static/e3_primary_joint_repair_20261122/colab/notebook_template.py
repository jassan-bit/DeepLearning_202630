"""Build the separately published CPU notebook after the source commit exists.

The source packet includes this generator. Resolved notebooks and launchers are
published afterward so their source and bootstrap hashes have no circularity.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import re
from pathlib import PurePosixPath

SOURCE_COMMIT = "__STATISTICS_SOURCE_COMMIT__"
MANIFEST_SHA256 = "__STATISTICS_MANIFEST_SHA256__"
BOOTSTRAP_SHA256 = "__STATISTICS_BOOTSTRAP_SHA256__"


def notebook(source_commit=SOURCE_COMMIT, manifest_sha256=MANIFEST_SHA256,
             bootstrap_sha256=BOOTSTRAP_SHA256, published_checkpoint=None):
    for value, pattern, unresolved in ((source_commit, r"[0-9a-f]{40}", SOURCE_COMMIT),
                                      (manifest_sha256, r"[0-9a-f]{64}", MANIFEST_SHA256),
                                      (bootstrap_sha256, r"[0-9a-f]{64}", BOOTSTRAP_SHA256)):
        if value != unresolved and not re.fullmatch(pattern, value):
            raise ValueError("Use resolved immutable source pins")
    if published_checkpoint is not None:
        if set(published_checkpoint) != {"commit", "path", "sha256"}:
            raise ValueError("Published checkpoint needs commit, repository path and SHA256")
        path = PurePosixPath(published_checkpoint["path"])
        if (not re.fullmatch(r"[0-9a-f]{40}", published_checkpoint["commit"]) or
            not re.fullmatch(r"[0-9a-f]{64}", published_checkpoint["sha256"]) or
            path.is_absolute() or ".." in path.parts or path.as_posix() != published_checkpoint["path"]):
            raise ValueError("Use immutable, safe published checkpoint pins")
    cells = []
    def md(source):
        cells.append(dict(cell_type="markdown", metadata={}, source=source))
    def code(source):
        ast.parse(source)
        cells.append(dict(cell_type="code", metadata={}, source=source, execution_count=None, outputs=[]))
    md("""# E3 · batería primaria y decisión conjunta en CPU

Este cuaderno ejecuta la validación estadística sobre datos sintéticos con la
primaria exacta: promedio del RMSE de cada semilla. Conserva el pareado temporal,
la familia de rivales y la decisión de cuatro paneles del protocolo publicado.
El dictamen se limita a sus escenarios y presupuesto; la evidencia de mercado
requiere una evaluación posterior. El remuestreo temporal conserva juntas las
filas de sistemas y semillas; la dispersión entre semillas se informa aparte
y no convierte semillas × fechas en observaciones independientes.

Instala únicamente NumPy y SciPy en un entorno aislado, verifica las fuentes y
mantiene la simulación desactivada hasta que actives su celda. El protocolo fija
escenarios, niveles, réplicas y coste; los controles del cuaderno limitan cada
sesión y no cambian esas reglas.

Después de cada bloque, exporta y descarga el ZIP. Conserva ese archivo fuera
de Colab: permite continuar después de una sustitución de la VM, cuando se
pierde `/content`. Las réplicas completas se comprueban antes de reutilizarlas.
No se necesita Drive, un token ni una clave para este flujo.
""")
    md("""## 1. Instalar y verificar las fuentes

Ejecuta esta celda. Un intérprete anterior compatible se comprueba sin instalar
paquetes en él. Una VM nueva recibe su propio entorno numérico. Esta operación
registra su coste de instalación por separado del Monte Carlo.
""")
    code('''from pathlib import Path
import hashlib, importlib.util, json, re, time, urllib.request

STATISTICS_SOURCE_COMMIT = ''' + repr(source_commit) + '''
STATISTICS_MANIFEST_SHA256 = ''' + repr(manifest_sha256) + '''
STATISTICS_BOOTSTRAP_SHA256 = ''' + repr(bootstrap_sha256) + '''
assert re.fullmatch(r"[0-9a-f]{40}", STATISTICS_SOURCE_COMMIT), "Usar la edición publicada con pins completos"
assert re.fullmatch(r"[0-9a-f]{64}", STATISTICS_MANIFEST_SHA256)
assert re.fullmatch(r"[0-9a-f]{64}", STATISTICS_BOOTSTRAP_SHA256)
STATISTICS_WORK = (Path("/content") if Path("/content").is_dir() else Path.cwd()) / "e3-primary-joint-validation-cpu"
setup_start = time.perf_counter()
bootstrap_url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" +
    STATISTICS_SOURCE_COMMIT + "/experiments/e3_primary_joint_repair_20261122/colab/bootstrap_cpu.py")
with urllib.request.urlopen(bootstrap_url, timeout=90) as response:
    bootstrap_raw = response.read(262145)
assert len(bootstrap_raw) <= 262144
assert hashlib.sha256(bootstrap_raw).hexdigest() == STATISTICS_BOOTSTRAP_SHA256
bootstrap_file = STATISTICS_WORK / ("bootstrap-statistics-" + STATISTICS_SOURCE_COMMIT + ".py")
assert not any(p.is_symlink() for p in (bootstrap_file, *bootstrap_file.parents))
if bootstrap_file.exists():
    assert bootstrap_file.read_bytes() == bootstrap_raw
else:
    bootstrap_file.parent.mkdir(parents=True, exist_ok=True)
    with bootstrap_file.open("xb") as stream:
        stream.write(bootstrap_raw)
bootstrap_spec = importlib.util.spec_from_file_location("e3_pinned_statistics_bootstrap", bootstrap_file)
statistics_bootstrap = importlib.util.module_from_spec(bootstrap_spec)
bootstrap_spec.loader.exec_module(statistics_bootstrap)
candidate_python = None
for previous in (globals().get("VALIDATION_SETTINGS"), globals().get("SETTINGS")):
    if isinstance(previous, dict) and previous.get("source_commit") in {
        STATISTICS_SOURCE_COMMIT, "db74afd0763596f71bcbafb4b9150a44a2e83339",
        "c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c"}:
        candidate = previous.get("python")
        if isinstance(candidate, (str, Path)) and Path(candidate).is_file():
            try:
                statistics_bootstrap.check_python(candidate)
                candidate_python = str(candidate)
                break
            except RuntimeError:
                pass
VALIDATION_SETTINGS = statistics_bootstrap.setup(STATISTICS_SOURCE_COMMIT,
    STATISTICS_MANIFEST_SHA256, STATISTICS_WORK / "cache", existing_python=candidate_python)
STATISTICS_SOURCE = Path(VALIDATION_SETTINGS["source_root"])
SOFTWARE_SETUP_WALL_SECONDS = time.perf_counter() - setup_start
print(json.dumps({"cost_scope":"software_setup", "wall_seconds":SOFTWARE_SETUP_WALL_SECONDS,
    "monte_carlo_executed":False, "model_training_executed":False}))

def statistics_command(args):
    return statistics_bootstrap.command(VALIDATION_SETTINGS, args)

def statistics_result(args):
    result = statistics_command(args)
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("Falta el resultado JSON del comando estadístico; conservar el caso")
''')
    md("""## 2. Preparar o restaurar el caso registrado

Elige una de las dos acciones. `PREPARE_VALIDATION_CASE=True` congela el protocolo
publicado. `RESTORE_VALIDATION_CHECKPOINT=True` importa un ZIP que exportaste
anteriormente y comprueba identidad de código, protocolo y resultados. Conserva
el mismo nombre al reanudar un caso existente.

La importación y lectura son portables entre Python 3.12 y 3.13. Añadir réplicas
exige la misma cohorte numérica registrada: versión mayor/menor de Python,
NumPy/SciPy y backend. Un ZIP completo puede revisarse en otra cohorte sin
volver a simular. Un caso incompleto necesita su cohorte original para continuar;
una ejecución separada en otra versión conserva sus propios resultados y no
los mezcla con el ledger anterior.
""")
    code('''PREPARE_VALIDATION_CASE = False
RESTORE_VALIDATION_CHECKPOINT = False
VALIDATION_LABEL = "registered-primary-joint-v1"
RESTORE_ZIP_PATH = ""  # Si queda vacío, la importación habilitada abre el selector de archivos.

assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", VALIDATION_LABEL)
assert not (PREPARE_VALIDATION_CASE and RESTORE_VALIDATION_CHECKPOINT), "Elegir preparación o restauración"
STATISTICS_CASE = STATISTICS_WORK / "cases" / VALIDATION_LABEL
STATISTICS_PROTOCOL = STATISTICS_SOURCE / "joint_validation/protocol.json"
if RESTORE_VALIDATION_CHECKPOINT:
    assert not STATISTICS_CASE.exists(), "Conservar el caso existente; usa sus celdas de estado y continuación"
    if RESTORE_ZIP_PATH:
        archive = Path(RESTORE_ZIP_PATH)
        assert archive.is_file() and not archive.is_symlink()
    else:
        from google.colab import files
        uploaded = files.upload()
        assert len(uploaded) == 1, "Importar exactamente un ZIP exportado"
        uploaded_raw = next(iter(uploaded.values()))
        assert len(uploaded_raw) <= 64 * 1024 * 1024
        archive = STATISTICS_WORK / "imports" / (hashlib.sha256(uploaded_raw).hexdigest() + ".zip")
        assert not any(p.is_symlink() for p in (archive, *archive.parents))
        if archive.exists():
            assert archive.read_bytes() == uploaded_raw
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            with archive.open("xb") as stream:
                stream.write(uploaded_raw)
    print(json.dumps({"imported_archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}))
    assert not any(p.is_symlink() for p in (STATISTICS_CASE, *STATISTICS_CASE.parents))
    STATISTICS_CASE.parent.mkdir(parents=True, exist_ok=True)
    statistics_command(["restore", "--archive", str(archive), "--output", str(STATISTICS_CASE),
                        "--source-commit", STATISTICS_SOURCE_COMMIT])
if PREPARE_VALIDATION_CASE:
    if STATISTICS_CASE.exists():
        statistics_command(["status", "--case", str(STATISTICS_CASE)])
    else:
        statistics_command(["freeze", "--output", str(STATISTICS_CASE),
            "--protocol", str(STATISTICS_PROTOCOL), "--source-commit", STATISTICS_SOURCE_COMMIT])
if not PREPARE_VALIDATION_CASE and not RESTORE_VALIDATION_CHECKPOINT:
    print(json.dumps({"status":"sources_ready_no_case_mutation", "monte_carlo_executed":False}))
''')
    md("""## 3. Ejecutar un bloque o continuar

Activa `RUN_STATISTICAL_VALIDATION=True`. Cada llamada conserva réplicas
completas y continúa las faltantes. `MAX_REPLICATES_THIS_CALL` y
`SECONDS_BUDGET_THIS_CALL` acotan esta llamada; el límite acumulado y las
condiciones de aceptación permanecen fijados por el protocolo.

Un presupuesto agotado produce un resultado incompleto con su incertidumbre.
Conserva esos resultados y exporta el checkpoint para continuar o revisarlo.
""")
    code('''RUN_STATISTICAL_VALIDATION = False
MAX_REPLICATES_THIS_CALL = 8
SECONDS_BUDGET_THIS_CALL = 300

if RUN_STATISTICAL_VALIDATION:
    assert STATISTICS_CASE.is_dir(), "Preparar o restaurar primero el caso registrado"
    assert type(MAX_REPLICATES_THIS_CALL) is int and MAX_REPLICATES_THIS_CALL > 0
    assert isinstance(SECONDS_BUDGET_THIS_CALL, (int, float)) and SECONDS_BUDGET_THIS_CALL > 0
    current_state = statistics_result(["status", "--case", str(STATISTICS_CASE)])
    if current_state.get("status") == "completed":
        print("El caso ya está completo. Revisa y exporta sus resultados; no se añaden réplicas.")
    else:
        statistics_command(["run", "--case", str(STATISTICS_CASE), "--resume",
            "--max-replicates", str(MAX_REPLICATES_THIS_CALL),
            "--seconds-budget", str(SECONDS_BUDGET_THIS_CALL)])
        statistics_command(["status", "--case", str(STATISTICS_CASE)])
else:
    print(json.dumps({"status":"statistical_validation_not_activated", "monte_carlo_executed":False}))
''')
    md("""## 4. Estado, exportación y descarga

Ejecuta esta celda después de cada bloque. El ZIP conserva las reglas, el ledger
y los resultados completos o parciales; su importación comprueba hashes y
rutas. La descarga local permite sobrevivir a una sustitución de la VM.
Los costes de Monte Carlo proceden del ledger; la instalación se mide aparte.
""")
    code('''EXPORT_VALIDATION_CHECKPOINT = True
DOWNLOAD_VALIDATION_CHECKPOINT = True
EXPORTED_VALIDATION_CHECKPOINT = None

if STATISTICS_CASE.is_dir():
    statistics_command(["status", "--case", str(STATISTICS_CASE)])
    if EXPORT_VALIDATION_CHECKPOINT:
        import datetime
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        EXPORTED_VALIDATION_CHECKPOINT = STATISTICS_WORK / "exports" / (VALIDATION_LABEL + "-" + stamp + ".zip")
        assert not any(p.is_symlink() for p in (EXPORTED_VALIDATION_CHECKPOINT, *EXPORTED_VALIDATION_CHECKPOINT.parents))
        EXPORTED_VALIDATION_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        statistics_command(["export", "--case", str(STATISTICS_CASE),
                            "--output", str(EXPORTED_VALIDATION_CHECKPOINT)])
        print(json.dumps({"export_sha256":hashlib.sha256(EXPORTED_VALIDATION_CHECKPOINT.read_bytes()).hexdigest(),
                          "path":str(EXPORTED_VALIDATION_CHECKPOINT)}))
        if DOWNLOAD_VALIDATION_CHECKPOINT:
            from google.colab import files
            files.download(str(EXPORTED_VALIDATION_CHECKPOINT))
else:
    print("Aún no hay un caso preparado o restaurado. La instalación no ejecutó simulaciones.")
''')
    md("""## Alcance y recuperación opcional de pesos

El dictamen conserva las condiciones y escenarios del protocolo. Una decisión
favorable aquí constituye evidencia sobre la batería estadística; la
superioridad del modelo requiere sus observaciones reservadas y contrastes de
mercado. Los pilotos y decisiones adversas anteriores conservan su alcance.

La recuperación GET-only de checkpoints originales se entrega como
[E3_V5_FrozenWeights_PrivateRecovery.ipynb](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/""" + source_commit + """/experiments/e3_primary_joint_repair_20261122/recovery/E3_V5_FrozenWeights_PrivateRecovery.ipynb),
un flujo separado con su propio registro de
coste y acceso privado cuando corresponda. Este cuaderno numérico mantiene
únicamente las dependencias de validación.

Asistencia de IA: implementación y revisión declaradas en los archivos del
paquete. Los archivos exportados son resultados de simulación y registros de
software; no contienen una firma, sustentación o aprobación docente.
""")
    # Results are published after source commit P. The final notebook T can pin
    # their later commit R without introducing R into the immutable source P.
    config_cell = [cell for cell in cells if cell["cell_type"] == "code"][1]
    config = config_cell["source"]
    config = ("PUBLISHED_CHECKPOINT = " + repr(published_checkpoint or {}) + "\n" +
              "LOAD_PUBLISHED_CHECKPOINT = " + repr(published_checkpoint is not None) + "\n" + config)
    config = config.replace('assert not (PREPARE_VALIDATION_CASE and RESTORE_VALIDATION_CHECKPOINT), "Elegir preparación o restauración"',
        'assert sum((LOAD_PUBLISHED_CHECKPOINT, PREPARE_VALIDATION_CASE, RESTORE_VALIDATION_CHECKPOINT)) <= 1, "Elegir consulta publicada, preparación o restauración"')
    published_flow = '''if LOAD_PUBLISHED_CHECKPOINT:
    if STATISTICS_CASE.exists():
        print("Se conserva el caso existente y se consulta su estado; no se sobrescribe con el publicado.")
        statistics_command(["status", "--case", str(STATISTICS_CASE)])
    else:
        published_url = ("https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" +
                         PUBLISHED_CHECKPOINT["commit"] + "/" + PUBLISHED_CHECKPOINT["path"])
        published_archive = STATISTICS_WORK / "imports" / (PUBLISHED_CHECKPOINT["sha256"] + ".zip")
        assert not any(p.is_symlink() for p in (published_archive, *published_archive.parents))
        if published_archive.exists():
            published_raw = published_archive.read_bytes()
        else:
            with urllib.request.urlopen(published_url, timeout=90) as response:
                published_raw = response.read(64 * 1024 * 1024 + 1)
        assert len(published_raw) <= 64 * 1024 * 1024
        assert hashlib.sha256(published_raw).hexdigest() == PUBLISHED_CHECKPOINT["sha256"], "ZIP publicado distinto"
        if not published_archive.exists():
            published_archive.parent.mkdir(parents=True, exist_ok=True)
            with published_archive.open("xb") as stream:
                stream.write(published_raw)
        assert not any(p.is_symlink() for p in (STATISTICS_CASE, *STATISTICS_CASE.parents))
        STATISTICS_CASE.parent.mkdir(parents=True, exist_ok=True)
        statistics_command(["restore", "--archive", str(published_archive), "--output", str(STATISTICS_CASE),
                            "--source-commit", STATISTICS_SOURCE_COMMIT])
        published_state = statistics_result(["status", "--case", str(STATISTICS_CASE)])
        assert published_state.get("status") == "completed", "Conservar el checkpoint: el estudio publicado aún está incompleto"
        print("Estudio publicado completo importado. Su dictamen se conserva; no se ejecutaron réplicas nuevas.")
'''
    config = config.replace("if RESTORE_VALIDATION_CHECKPOINT:\n", published_flow + "if RESTORE_VALIDATION_CHECKPOINT:\n", 1)
    config = config.replace("if not PREPARE_VALIDATION_CASE and not RESTORE_VALIDATION_CHECKPOINT:",
                            "if not LOAD_PUBLISHED_CHECKPOINT and not PREPARE_VALIDATION_CASE and not RESTORE_VALIDATION_CHECKPOINT:")
    ast.parse(config)
    config_cell["source"] = config
    if published_checkpoint is not None:
        target = next(cell for cell in cells if cell["cell_type"] == "markdown" and cell["source"].startswith("## 2."))
        target["source"] = ("## 2. Consultar primero el estudio publicado\n\n"
            "La consulta publicada está activada: descarga el ZIP fijado por commit y SHA256, "
            "verifica el caso y conserva su dictamen completo. No repite las simulaciones. "
            "Si existe un caso local, muestra su estado y conserva sus archivos.\n\n" +
            target["source"].split("\n", 1)[1])
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        colab=dict(name="E3_Primaria_y_Decision_Conjunta_CPU.ipynb", provenance=[]),
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python"),
        e3_source_pins=dict(source_commit=source_commit, manifest_sha256=manifest_sha256,
                            bootstrap_sha256=bootstrap_sha256),
        e3_published_checkpoint=published_checkpoint,
        e3_monte_carlo_default=False, e3_market_training_enabled=False,
        e3_market_acquisition_enabled=False, e3_credentials_required=False,
        e3_package_scope="Primary endpoint and joint-decision statistical validation"))


def write(path, **pins):
    value = notebook(**pins)
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--bootstrap-sha256", default=BOOTSTRAP_SHA256)
    parser.add_argument("--published-checkpoint-commit")
    parser.add_argument("--published-checkpoint-path")
    parser.add_argument("--published-checkpoint-sha256")
    args = parser.parse_args()
    published = None
    values = (args.published_checkpoint_commit, args.published_checkpoint_path, args.published_checkpoint_sha256)
    if any(values):
        if not all(values):
            parser.error("Published checkpoint requires all three pins")
        published = dict(commit=values[0], path=values[1], sha256=values[2])
    write(args.output, source_commit=args.source_commit, manifest_sha256=args.manifest_sha256,
          bootstrap_sha256=args.bootstrap_sha256, published_checkpoint=published)
