"""Build the pinned, output-free exploratory moment pilot notebook for Colab."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/entregable3_momentos_colab.ipynb"
PLAN = "experiments/e3_piloto_momentos_20261116/plan.json"
SOURCE_FILES = (
    "src/__init__.py", "src/original_model.py", "src/e3_development.py",
    "src/e3_innovation_model.py", "src/e3_innovation.py", "src/e3_moment_model.py",
    "src/e3_moment_training.py", "scripts/e3_github_storage.py",
    "scripts/e3_moment_github.py", "scripts/e3_run_moments.py",
    "notebooks/preparar_e3_desarrollo_runtime.py", "notebooks/preparar_e3_innovacion_runtime.py",
    "requirements-e3-innovation-gpu.lock.txt", PLAN,
)


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or source_commit == "0" * 40:
        raise ValueError("An explicit existing 40-character source commit is required")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    cells = [cell("markdown", """
# E3 · Piloto exploratorio · CNN-LSTM con atención y cabeza de momentos

Se conserva la base CNN causal → LSTM → atención temporal. Se comparan **tres redes de 1236 parámetros**: cabeza de momentos con dos pérdidas auxiliares, la misma cabeza con MSE sola y una media directa con los mismos auxiliares. La aditiva v2 se conserva como referencia por sus predicciones verificadas.

1. En **Entorno de ejecución → Cambiar tipo de entorno**, selecciona una GPU. Se registra la GPU realmente asignada.
2. En **Secretos** (llave), habilita `GITHUB_TOKEN` con permiso de escritura de contenidos en `jassan-bit/DeepLearning_202630`. Su valor no se imprime.
3. Ejecuta las celdas en orden o pulsa **Ejecutar todas**. La instalación inicial puede tardar varios minutos y no reinicia el kernel.
4. Para reconectar, usa este mismo notebook y las mismas carpetas. El registro remoto conserva corrida y presupuesto; repetir celdas no crea una segunda corrida.

**Hasta 40 épocas por red y 600 segundos adicionales de cálculo contabilizado entre las tres**, incluyendo ajuste de estadísticas, inicialización, validación y checkpoints. Se detiene en los puntos de control del contador; una operación ya iniciada puede añadir sobremarcha registrada. Instalación, descargas y publicación pueden añadir tiempo de reloj. Se conservan los 556,212659362 segundos consumidos previamente y el límite global de 1200.

Los resultados son de desarrollo, con una sola semilla y validación ya expuesta. No constituyen novedad acreditada, preregistro confirmatorio ni superioridad frente al Top 10. La envolvente de momentos no es un intervalo de predicción. El guardado usa **GitHub**.
"""), cell("code", f'''
from pathlib import Path, PurePosixPath
import hashlib
import importlib.util
import json
import os
import re
import sys
import urllib.request

REPOSITORY = "jassan-bit/DeepLearning_202630"
SOURCE_COMMIT = "{source_commit}"
RESULTS_BRANCH = "e3-desarrollo-resultados-momentos"
RESULTS_PREFIX = "runs/momentos_v1"
RESULTS_URL = f"https://github.com/{{REPOSITORY}}/tree/{{RESULTS_BRANCH}}/{{RESULTS_PREFIX}}"
WORKSPACE = Path("/content/e3_momentos_github")
SOURCE_DIR = WORKSPACE / "source"
RUNTIME_DIR = WORKSPACE / "runtime"
RUN_DIR = WORKSPACE / "run"
PLAN_PATH = SOURCE_DIR / "{PLAN}"

def download_pinned(relative_path, commit, target, digest):
    if not re.fullmatch(r"[0-9a-f]{{40}}", commit) or not re.fullmatch(r"[0-9a-f]{{64}}", digest):
        raise RuntimeError("Commit o SHA-256 inválido")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != relative_path or "\\\\" in relative_path:
        raise RuntimeError("Ruta de fuente no admitida")
    target = Path(target)
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise RuntimeError("No se admiten enlaces simbólicos en la descarga")
    if target.exists():
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError("El archivo local difiere de la fuente fijada; se conserva: " + str(target))
        return digest
    url = f"https://raw.githubusercontent.com/{{REPOSITORY}}/{{commit}}/{{relative_path}}"
    with urllib.request.urlopen(url, timeout=60) as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError("Los bytes descargados no coinciden: " + relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
    os.replace(temporary, target)
    return digest

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el módulo fijado")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
'''), cell("markdown", "## Descargar y comprobar el código fijado\n\nCada archivo se verifica antes de importarlo y de iniciar cada etapa científica."),
        cell("code", "SOURCE_FILE_SHA256 = " + json.dumps(hashes, indent=4, ensure_ascii=False)),
        cell("code", '''
for relative_path, digest in SOURCE_FILE_SHA256.items():
    download_pinned(relative_path, SOURCE_COMMIT, SOURCE_DIR / relative_path, digest)

def assert_fixed_sources():
    for relative_path, digest in SOURCE_FILE_SHA256.items():
        path = SOURCE_DIR / relative_path
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("La fuente fijada cambió; no se inicia ajuste: " + relative_path)

assert_fixed_sources()
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
print("Código fijado:", SOURCE_COMMIT)
print("Destino de resultados (se crea al inicializar):", RESULTS_URL)
'''), cell("markdown", """
## Acceso a GitHub

El token queda en memoria del coordinador. El proceso científico recibe un entorno filtrado. Un error de autenticación o una publicación no confirmada detiene la ejecución antes del siguiente bloque de entrenamiento.
"""), cell("code", '''
GITHUB_AUTH_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
if not GITHUB_AUTH_TOKEN:
    try:
        from google.colab import userdata
        GITHUB_AUTH_TOKEN = (userdata.get("GITHUB_TOKEN") or "").strip()
    except Exception:
        from getpass import getpass
        GITHUB_AUTH_TOKEN = getpass("Token de GitHub (entrada oculta): ").strip()
if not GITHUB_AUTH_TOKEN:
    raise RuntimeError("Falta GITHUB_TOKEN; no se inicia la corrida")
print("Credencial recibida en memoria; su valor no se muestra.")
'''), cell("markdown", """
## Preparar el entorno y los datos

Se usa Python 3.12.12, PyTorch 2.8.0+cu126, NumPy 2.3.5 y SciPy 1.17.0 en un proceso aislado. El notebook registra la GPU real; la verificación local previa del software se hizo con CPU y datos artificiales. Descargar el NPZ verifica sus bytes; el trabajador sólo admite los tres miembros `training_*` previstos, sin cargar calibración o prueba.
"""), cell("code", '''
assert_fixed_sources()
runtime_helper = load_module("e3_moment_runtime", SOURCE_DIR / "notebooks/preparar_e3_innovacion_runtime.py")
GPU_PYTHON, RUN_ENV, GPU_RUNTIME = runtime_helper.preparar_entorno(RUNTIME_DIR, device="cuda")
PLAN = json.loads(PLAN_PATH.read_text())
DATA_PATH = WORKSPACE / "inputs/current_cohort_arrays.npz"
DATA_SHA256 = PLAN["data"]["sha256"]
download_pinned(PLAN["data"]["path"], PLAN["data"]["commit"], DATA_PATH, DATA_SHA256)
GITHUB_HELPER = load_module("e3_moment_github_runtime", SOURCE_DIR / "scripts/e3_moment_github.py")
print("Datos verificados:", DATA_SHA256)
'''), cell("markdown", """
## Crear o recuperar la misma corrida

**La inicialización ya usa datos de ajuste y consume presupuesto.** Antes se confirma el registro remoto único. Se comprueba que v1/v2 permanecen cerrados; la referencia aditiva se obtiene sólo de textos y predicciones verificadas, sin cargar sus pesos. Las tres redes nuevas parten del mismo codificador y estadísticas de ajuste.
"""), cell("code", '''
assert_fixed_sources()
GITHUB_ARGS = dict(token=GITHUB_AUTH_TOKEN, npz_path=DATA_PATH, expected_sha256=DATA_SHA256,
                   run_dir=RUN_DIR, plan_path=PLAN_PATH, code_root=SOURCE_DIR,
                   python=GPU_PYTHON, environment=RUN_ENV, device="cuda")
prepared = GITHUB_HELPER.prepare_github_moments(**GITHUB_ARGS)
print(json.dumps(prepared, ensure_ascii=False, indent=2))
'''), cell("markdown", """
## Entrenar, validar y guardar

Se alterna una época por receta activa. Las mejores épocas se eligen por RMSE de la misma validación de desarrollo. Cada bloque de doce épocas de modelo se publica y confirma antes del siguiente; la instalación y la publicación añaden tiempo de reloj al contador científico. Una corrida terminada o agotada no se reinicia al repetir esta celda.
"""), cell("code", '''
assert_fixed_sources()
execution = GITHUB_HELPER.run_github_moments(**GITHUB_ARGS, max_chunks=40, stop_after_epochs=12)
print(json.dumps(execution, ensure_ascii=False, indent=2))
print("Resultados y checkpoints:", RESULTS_URL)
'''), cell("markdown", """
## Revisar resultados exploratorios

Menor RMSE es mejor. La tabla muestra las tres recetas nuevas y las referencias. Las salidas auxiliares de la versión MSE sola no se presentan como momentos identificados. Una corrida incompleta no supera el criterio de continuación.
"""), cell("code", '''
from html import escape
from IPython.display import HTML, display

summary_path = RUN_DIR / "summary.json"
if not summary_path.is_file():
    print("Todavía no hay resumen; revisa el estado de las celdas anteriores.")
else:
    summary = json.loads(summary_path.read_text())
    table = []
    for name, row in summary.get("models", {}).items():
        metrics = row.get("metrics") or row
        def number(value):
            return "—" if value is None else format(float(value), ".8f")
        table.append("<tr>" + "".join("<td>" + escape(str(value)) + "</td>" for value in (
            name, row.get("parameters", "—"), row.get("best_epoch", "—"),
            number(metrics.get("RMSE")), number(metrics.get("MAE")),
            row.get("complete", "referencia"))) + "</tr>")
    display(HTML("<table><thead><tr><th>Receta</th><th>Parámetros</th><th>Época elegida</th>"
                 "<th>RMSE σ₇</th><th>MAE σ₇</th><th>Completa</th></tr></thead><tbody>"
                 + "".join(table) + "</tbody></table>"))
    print("Estado:", summary.get("status"))
    print("Comparación completa:", summary.get("comparison_complete", False))
    print("Heurística exploratoria superada:", summary.get("exploratory_continuation_gate_passed", False))
    print("La novedad y la superioridad estadística siguen sin acreditarse.")
'''), cell("code", '''
import csv
try:
    import plotly.graph_objects as go
except ImportError:
    print("Las curvas completas están guardadas como CSV en GitHub.")
else:
    figure = go.Figure()
    for name in ("moment_joint", "moment_mean_only", "direct_moment_aux"):
        path = RUN_DIR / name / "curves.csv"
        if path.is_file():
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                figure.add_trace(go.Scatter(x=[int(r["epoch"]) for r in rows],
                    y=[float(r["validation_RMSE_original"]) for r in rows],
                    mode="lines+markers", name=name))
    if figure.data:
        summary_file = RUN_DIR / "summary.json"
        if summary_file.is_file():
            reference = json.loads(summary_file.read_text()).get("models", {}).get("frozen_v2_additive", {})
            reference_rmse = (reference.get("metrics") or {}).get("RMSE")
            if reference_rmse is not None:
                figure.add_hline(y=float(reference_rmse), line_dash="dot",
                                 annotation_text="Aditiva v2: referencia fija")
        figure.update_layout(title="Validación exploratoria de desarrollo · una semilla",
                             xaxis_title="Época", yaxis_title="RMSE de σ₇ (unidades originales)")
        figure.show()
    else:
        print("Aún no hay épocas guardadas para dibujar.")
'''), cell("markdown", """
## Decisión posterior

La heurística fija exige al candidato al menos **2 % menos RMSE que el mejor de los cinco controles**, con las tres recetas completas. Es un criterio exploratorio, distinto del margen confirmatorio y de la revisión de novedad. Una mejor curva no acredita por sí sola R1 ni R3.

Después se documentarán honestamente resultados, fallos y decisiones. La revisión bibliográfica y humana, Faithful y los requisitos confirmatorios completos siguen pendientes; no se retrofecha el protocolo. Los nuevos temas se integran en los capítulos 6 y 7 del libro y sus anexos, conservando los capítulos principales 1–16.
""")]
    for index, item in enumerate(cells):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"moment-cell-{index}", "exec")
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                "kernelspec": {"display_name": "Python 3", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def pinned_commit(notebook):
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_COMMIT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Notebook source pin missing")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        old = json.loads(args.output.read_text())
        if old != build(args.source_commit or pinned_commit(old)):
            raise ValueError("Notebook or pinned source hashes changed")
        print("Pinned moment notebook verified; empty outputs; no execution or credentials")
    else:
        if not args.source_commit:
            parser.error("--source-commit is required to generate")
        args.output.write_text(json.dumps(build(args.source_commit), ensure_ascii=False, indent=1) + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
