"""Build twelve independently executable E2 archives from immutable evidence.

This is the maintainer's packaging command, not the professor's entry point.
Each extracted archive has its own run.py and included inputs. No campaign,
market fitting, statistical bootstrap or checkpoint recovery is launched here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "204a8543234aac1614e0b76914bd914e28567eb9"
PUBLIC = ROOT / "book_e2/_static/entregables_independientes_v1"
PANEL = "book_e2/_static/e2_final_audit_inputs_20261114/e2_f2a_input.json"
CPU_REQUIREMENTS = '''--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.8.0+cpu ; platform_system != "Darwin"
torch==2.8.0 ; platform_system == "Darwin"
numpy==2.3.5
pandas==2.2.3
einops==0.8.1
reformer-pytorch==1.4.4
'''
NAMES = {
 "D01":("01_informe_jupyter_book.zip","Informe Jupyter Book completo y fuentes"),
 "D02":("02_codigo_modelos.zip","Código y ejecución de las trece redes"),
 "D03":("03_datos_contrato.zip","Datos, Data Card y contrato"),
 "D04":("04_preregistro_auditoria.zip","Preregistro y auditoría de datos"),
 "D05":("05_ficha_ruta_F2a.zip","Ficha metodológica F2a y evidencia estadística"),
 "D06":("06_benchmark_comparaciones.zip","Benchmark, métricas y comparaciones"),
 "D07":("07_registro_experimentos.zip","Registro experimental y fichas de modelo"),
 "D08":("08_techo_incertidumbre.zip","Techo, incertidumbre y errores"),
 "D09":("09_arquitecturas_no_aplica.zip","Justificación de NAS no aplicable"),
 "D10":("10_paper_guia_replica.zip","Paper guía y réplica parcial"),
 "D11":("11_reproducibilidad.zip","Lista de reproducibilidad y evidencia"),
 "D12":("12_presentacion_defensa.zip","Presentación de defensa y notas del orador"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def copy(root, source, destination=None):
    source = Path(source)
    original = source if source.is_absolute() else ROOT/source
    if not original.is_file() or original.is_symlink():
        raise ValueError("Required regular source missing: "+str(original))
    destination = destination or str(source)
    target = root/destination
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original,target)
    return destination


def entry(root, source, title, *, folder="inputs", name=None):
    relative = copy(root,source,folder+"/"+(name or Path(source).name))
    return {"title":title,"path":relative,"sha256":sha(root/relative),"bytes":(root/relative).stat().st_size}


def config(root, key, *, documents=(), tables=(), json_files=(), scope=""):
    record={"id":key,"title":NAMES[key][1],"source_commit":SOURCE,"scope":scope,
            "documents":list(documents),"tables":list(tables),"json_files":list(json_files)}
    dump(root/"package.json",record)
    return record


def documents(root, pairs):
    return [entry(root,"book_e2/sections/"+name,title,folder="documents") for name,title in pairs]


def reporting_bundle(root, key):
    copy(root,"scripts/e2_standalone_reports.py","run.py")
    docs=[]; tables=[]; data=[]
    def j(path,title):data.append(entry(root,path,title))
    def t(path,title):tables.append(entry(root,path,title))
    if key=="D04":
        docs=documents(root,[("09_protocolo_registrado.md","Preregistro y decisiones"),("10_auditoria_datos.md","Contrato y auditoría ejecutada")])
        for p,title in [
            ("book_e2/_static/reduced_protocol_20261112/protocol.json","Protocolo registrado"),
            ("book_e2/_static/e2_benchmark_final_20261113/final_freeze.json","Congelación anterior al scoring"),
            ("book_e2/_static/e2_benchmark_final_20261113/data_admission.json","Admisión de datos"),
            ("book_e2/_static/e2_point1_review_v1/margin_lineage.json","Linaje de márgenes"),
            ("book_e2/_static/e2_data_training_uncertainty_current_v1/targeted_data_guard_checks.json","Cinco guardas sintéticas: recibo original")]:j(p,title)
    elif key=="D05":
        docs=documents(root,[("07_ruta_temporal.md","Ruta F2a vigente"),("13_agregacion_estadistica.md","Unidades, efectos y familias")])
        j("book_e2/_static/e2_eda_current_v1/partition_audit.json","Partición y soportes temporales")
        for p,title in [
            ("book_e2/_static/e2_route_current_v1/contrast_battery_map.csv","Mapa de pruebas por contraste"),
            ("book_e2/_static/e2_f2a_audit_20261113/tests.csv","Diagnósticos F2a publicados"),
            ("book_e2/_static/e2_agregacion_entrega_current_v1/a3_primary_contrasts_readout.csv","Veinte pares primarios y 41 campos"),
            ("book_e2/_static/e2_f2a_s8_20261113/dm_hln_hac14_all120.csv","DM sobre otra pérdida: MSE diaria")]:t(p,title)
    elif key=="D07":
        docs=documents(root,[("11_hpo_equidad.md","Selección, fichas, entrenamiento y costes")])
        for p,title in [
            ("book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv","Trece recetas y resultados individuales"),
            ("book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv","Clásicos y redes en A2"),
            ("book_e2/_static/reduced_protocol_20261112/hpo_configurations_registered_bytes_v2.csv","Dominio HPO registrado"),
            ("book_e2/_static/e2_efficiency_closure_20261113/phase_resources.csv","Recursos instrumentados por fase")]:t(p,title)
        j("book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register_provenance.json","Fuentes y ámbitos del registro")
        j("book_e2/_static/e2_benchmark_final_20261113/benchmark_analysis.json","Resultados y diagnósticos originales")
        j("experiments/e2_tide_published_r0_20261112/configurations.json","Extensión TiDE publicada")
    elif key=="D09":
        docs=documents(root,[("14_arquitecturas.md","NAS no aplicable al catálogo actual"),("44_arquitecturas_publicadas.md","Fuentes y adaptaciones registradas")])
        j("book_e2/_static/e2_agregacion_entrega_current_v1/rubric_and_deliverables.json","Aplicabilidad y rúbrica original")
        j("book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json","Trece cuerpos definidos")
        j("experiments/e2_tide_published_r0_20261112/protocol_extension.json","Definición estructural TiDE R0-v2")
    elif key=="D11":
        docs=documents(root,[("08_reproducibilidad.md","Entorno, ética y reproducción"),("61_reproduccion_consolidada.md","Apéndice B y evidencia de entrega")])
        t("book_e2/_static/e2_agregacion_entrega_current_v1/appendix_B_current.csv","Los 21 literales de reproducibilidad")
        j("book_e2/_static/e2_agregacion_entrega_current_v1/appendix_B_current.json","Estados, evidencia y límites")
        t("book_e2/_static/e2_reproduction_ablations_20261113/reproduction_by_model.csv","Recibos de las 81 reproducciones")
        j("book_e2/_static/e2_reproduction_ablations_20261113/historical_clean_runtime.json","Entorno limpio observado")
        j("book_e2/_static/e2_delivery_reproduction_20261114/delivery_audit.json","Auditoría CPU publicada")
        copy(root,"book_e2/_static/e2_repro_prereg_current_v1/environment_observed.lock.txt","inputs/environment_observed.lock.txt")
    return config(root,key,documents=docs,tables=tables,json_files=data,scope="Informe ejecutable desde entradas incluidas; estados científicos originales preservados.")


def analysis_bundle(root,key):
    copy(root,"scripts/e2_standalone_analysis.py","run.py")
    copy(root,PANEL,"inputs/e2_f2a_input.json")
    copy(root,"book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv","inputs/benchmark_A2.csv")
    copy(root,"book_e2/_static/e2_agregacion_entrega_current_v1/a3_primary_contrasts_readout.csv","inputs/a3_primary_contrasts_readout.csv")
    copy(root,"requirements-e2-f2a-audit.lock.txt","requirements_analysis.txt")
    for script in ["e2_reduced_final_analysis.py","e2_reduced_protocol.py"]:
        copy(root,"scripts/"+script,"code/scripts/"+script)
    docs=documents(root,[("06_benchmark_final.md","Benchmark e interpretación"),("13_agregacion_estadistica.md","Unidades y familias")])
    if key=="D08":
        docs+=documents(root,[("12_incertidumbre_errores.md","Techo, incertidumbre y errores"),("60_calibracion_temporal_condicional.md","Políticas temporales y límites")])
        for path in [
            "book_e2/_static/e2_error_ceiling_final_audit_20261114/taxonomy_protocol.json",
            "book_e2/_static/e2_error_ceiling_final_audit_20261114/all_6080_date_model_labels.csv",
            "book_e2/_static/e2_conditional_calibration_20261114/model_metrics.csv",
            "book_e2/_static/e2_conditional_calibration_20261114/conditional_metrics.csv",
            "book_e2/_static/e2_conditional_calibration_20261114/conditional_coverage_CI.csv",
            "book_e2/_static/e2_calibration_ensemble_20261114/weights.csv",
            "book_e2/_static/e2_calibration_ensemble_20261114/summary.json",
            "results/eda_development/daily_volatility_development.csv",
        ]:copy(root,path,"inputs/"+Path(path).name)
        for script in ["e2_conditional_calibration.py","e2_error_ceiling_final_audit.py"]:
            copy(root,"scripts/"+script,"code/scripts/"+script)
        for source,dest in [("experiments/e2_conditional_calibration_20261114/analysis_plan.json","plans/calibration.json"),("experiments/e2_error_ceiling_final_audit_20261114/analysis_plan.json","plans/errors.json")]:copy(root,source,dest)
    if key=="D12":
        docs+=documents(root,[("47_entrega_consolidada.md","Entregables y defensa"),("12_incertidumbre_errores.md","Incertidumbre y resultados adversos")])
    return config(root,key,documents=docs,scope="Reconstrucción aritmética desde 81 trayectorias incluidas; sin pesos o entrenamiento por defecto.")


CODE_WRAPPER = '''from pathlib import Path
import json, html, runpy, sys
ROOT=Path(__file__).resolve().parent
dest=None
if "--train" not in sys.argv and "--output" not in sys.argv:
    folder=ROOT/"output";folder.mkdir(exist_ok=True)
    dest=folder/"resultado.json";i=2
    while dest.exists():
        dest=folder/("resultado_"+str(i)+".json");i+=1
    sys.argv.extend(["--output",str(dest)])
try:
    runpy.run_path(str(ROOT/"code/scripts/e2_standalone_code.py"),run_name="__main__")
except SystemExit as exc:
    if exc.code not in (0,None):raise
if dest and dest.is_file():
    record=json.loads(dest.read_text())
    (ROOT/"output/informe.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><title>Código E2</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto}pre{white-space:pre-wrap}</style><h1>Ejecución independiente del código</h1><p>Los forwards usan entradas artificiales y pesos iniciales; verifican implementación, no precisión de mercado. Entrenamiento CUDA es una opción separada.</p><pre>'+html.escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre></html>',encoding="utf-8")
'''


def code_bundle(root):
    selected=subprocess.check_output(["git","ls-tree","-r","--name-only",SOURCE,"--","scripts","tests","experiments","third_party"],cwd=ROOT,text=True).splitlines()
    for path in selected:
        copy(root,path,"code/"+path)
    copy(root,"scripts/e2_standalone_code.py","code/scripts/e2_standalone_code.py")
    metadata=runpy.run_path(str(ROOT/"scripts/e2_standalone_code.py"))["METADATA_SHA256"]
    required=[*metadata,"book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json",
              "book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv",
              "results/eda_development/daily_market_development.csv","results/eda_development/daily_volatility_development.csv"]
    for path in required:copy(root,path,"code/"+path)
    for path in ROOT.glob("requirements*.txt"):copy(root,path,"code/"+path.name)
    design=json.loads((ROOT/"experiments/e2_completion_20261106/design.json").read_text())
    models={"TSMixer","TiDE","TimeMixer","TimesNet","PatchTST","iTransformer"}
    for row in design["author_sources"]:
        if row["model"] not in models:continue
        source=Path("/workspace/work/e2architecture_sources")/row["model"]/row["path"]
        if sha(source)!=row["sha256"] or source.stat().st_size!=row["bytes"]:
            raise ValueError("Author source identity changed: "+str(source))
        copy(root,source,"authors/"+row["model"]+"/"+row["path"])
    (root/"run.py").write_text(CODE_WRAPPER,encoding="utf-8")
    (root/"requirements_cpu.txt").write_text(CPU_REQUIREMENTS,encoding="utf-8")
    copy(root,"requirements-e2-training-gpu.lock.txt","requirements_cuda.txt")
    return config(root,"D02",scope="Código completo, 76 fuentes de autores y recetas congeladas. Default: forwards CPU reales sobre fixtures; reentrenamiento CUDA local opt-in.")


def dataset_bundle(root):
    values=runpy.run_path(str(ROOT/"scripts/e2_standalone_dataset.py"))
    copy(root,"scripts/e2_standalone_dataset.py","run.py")
    for name,row in values["SOURCES"].items():copy(root,row["repository_path"],"inputs/"+name)
    return config(root,"D03",scope="CSV diarios originales de desarrollo; reconstruye calidad y σ7. Descarga oficial verificada opcional; no incluye X final privado.")


def paper_bundle(root,extra):
    values=runpy.run_path(str(ROOT/"scripts/e2_standalone_paper.py"))
    copy(root,"scripts/e2_standalone_paper.py","run.py")
    for name,(path,expected) in values["SOURCES"].items():
        copy(root,path,"inputs/"+name)
        if sha(root/"inputs"/name)!=expected:raise ValueError("Paper evidence changed")
    for path,expected in values["ORIGINAL_FILES"].items():
        copy(root,path,"code/"+path)
        if sha(root/"code"/path)!=expected:raise ValueError("Paper driver changed")
    for path in extra.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:continue
        rel=path.relative_to(extra).as_posix()
        if rel.startswith("authors/"):copy(root,path,rel)
        elif path.name in ("ETTh1.csv","LICENSE.ETDataset.txt","acquisition.json"):copy(root,path,"inputs/"+path.name)
    copy(root,"book_e2/sections/05_paper_guia.md","inputs/paper_guide.md")
    (root/"requirements_training.txt").write_text(CPU_REQUIREMENTS,encoding="utf-8")
    (root/"requirements-training.txt").write_text(CPU_REQUIREMENTS,encoding="utf-8")
    return config(root,"D10",scope="Reanálisis agregado original de cinco semillas sin dependencias; reproducción CPU entrenada opcional con ETTh1 y autores incluidos.")


def readme(root,key):
    default={"D01":"Extrae y valida el libro completo y sus fuentes en output/. Abre output/site/intro.html; para navegación servida, python run.py --serve.",
             "D02":"Verifica fuentes y ejecuta forwards CPU de las 13 redes con entradas artificiales. Es comprobación de implementación, no evaluación con pesos entrenados.",
             "D03":"Reconstruye las 716 etiquetas de desarrollo desde los cierres incluidos, verifica calidad/cohortes y produce un informe y CSV.",
             "D06":"Recalcula RMSE, MAE, sMAPE, R², cobertura y anchura de los 81 predictores desde sus pronósticos guardados; comprueba A2 y genera tabla, gráfico e informe.",
             "D08":"Reconstruye métricas/cobertura originales y 6.080 categorías de error; compara con los resultados guardados. Incluye las políticas e IC publicados.",
             "D10":"Recalcula media, DE, IC y discrepancias de las cinco semillas ETTh1; reconstruye curvas/figuras e informe. No entrena por defecto.",
             "D12":"Genera 14 diapositivas HTML autónomas, notas y el benchmark reconstruido. Abre output/presentacion.html, usa flechas y Ctrl+P para imprimir/guardar PDF."}.get(key,"Valida los esquemas de las entradas incluidas y genera un informe HTML navegable, tablas/JSON y exportaciones originales. No necesita el sitio ni otros paquetes de entrega.")
    text="# "+key+" · "+NAMES[key][1]+"\n\nEste ZIP es independiente. Descomprímelo completo en una carpeta nueva. No necesitas clonar el repositorio, acceder al sitio del curso ni descargar otro entregable.\n\n## Ejecutar\n\nSe recomienda Python 3.12. Windows: `py -3.12 run.py`. Linux/macOS: `python3 run.py`. También se incluyen `ejecutar.cmd` y `ejecutar.sh`.\n\n"
    if key=="D02":
        text+="Antes del primer forward, crea un venv e instala sus dependencias: `python -m venv .venv`, activa el entorno y ejecuta `python -m pip install -r requirements_cpu.txt`. Con ese entorno activado, usa `python run.py` también en Windows. Los lanzadores detectan una .venv local. La instalación necesita Internet; después, el inventario y los forwards usan sólo archivos locales. Sin instalar librerías, `python run.py --inventory` verifica las recetas y fuentes. No mezcles el lock CPU y CUDA.\n\n"
    text+="## Qué produce\n\n"+default+"\n\nLas salidas se escriben dentro del paquete; no modifican los datos científicos incluidos. D03 usa `output/dataset_audit.html`; los otros informes usan `output/informe.html`, y D01 ofrece el sitio completo en `output/site/intro.html`.\n\n## Alcance y opciones\n\n"
    if key=="D01":
        text+="El snapshot científico/documental corresponde al commit "+SOURCE+". Incluye 70 páginas HTML y fuentes completas del libro, con MathJax local y licencias. Necesita aproximadamente 400 MB libres al extraer. `--serve --port 8765` inicia un servidor sólo en 127.0.0.1; Ctrl+C lo detiene. Para reconstruir el libro: instala `output/source/requirements-book.lock.txt` y ejecuta `python run.py --build`. El build recompila el informe, no repite entrenamientos.\n"
    elif key=="D02":
        text+="`python run.py --smoke-cpu` usa pesos iniciales y un fixture [1,7,5]; no afirma precisión Bitcoin. Incluye código original de EDA, modelos, baselines, entrenamiento, estadísticas, robustez, incertidumbre, explicaciones y tests bajo code/. Para un reentrenamiento real opcional, usa Linux/WSL2 y CUDA: instala requirements_cuda.txt en otro venv y ejecuta `python run.py --train --model MLP16 --seed 271801 --device cuda:0 --output nuevo_ajuste`. Emplea 655 ejemplos train, 53 held-out y las épocas congeladas; no abre el controlador ni el presupuesto de la campaña. Esa opción no se ejecutó al preparar el ZIP. No incluye checkpoints originales ni el tensor X de prueba; los resultados del benchmark se reconstruyen sin ellos en D06.\n"
    elif key=="D03":
        text+="Incluye los dos CSV diarios de desarrollo, la Data Card, contrato y manifiestos. CSV diario y tensor de ventanas no son el mismo objeto. No incluye ni inventa los tensores finales X/escalares de test. Descarga opcional: `python run.py --download --archive BTCUSDT-1m-2023-01.zip`; requiere Internet y verifica CHECKSUM/hash oficiales. Sin --archive, la descarga selecciona los 24 meses de desarrollo; no se inicia por defecto.\n"
    elif key=="D06":
        text+="Default: biblioteca estándar, sin red, GPU o pesos. Los IC t y p se conservan como valores publicados y no se recalculan silenciosamente. Para repetir bootstrap/t/sign-flip originales: instala requirements_analysis.txt y ejecuta `python run.py --full-primary --draws 4999 --output reanalisis_original`. La opción recalcula pruebas con el programa original; puede tardar y no entrena modelos. Para el refinamiento usa --draws 20000.\n"
    elif key=="D08":
        text+="Default: biblioteca estándar y datos incluidos. Reproducir frecuencias/cobertura no demuestra cobertura universal ni un techo Bayes. Para ejecutar los programas estadísticos completos originales, instala requirements_analysis.txt y ejecuta `python run.py --full-uncertainty --output reanalisis_completo`. Son remuestreos CPU y pueden tardar; se conservan planes/semillas y no se ajustan nuevos pesos de ensemble.\n"
    elif key=="D09":
        text+="La búsqueda de 1.000 arquitecturas/dos fases NO APLICA al catálogo actual de cuerpos definidos, incluida TiDE R0-v2. Este archivo ejecutable valida y documenta la justificación; no contiene ni simula una NAS. El HPO sigue registrado por separado.\n"
    elif key=="D10":
        text+="Incluye ETTh1 original sin modificar, su licencia/atribución, 23 fuentes de autores verificadas y el driver original. Default no necesita NumPy/Torch; los NPZ históricos privados no se sustituyen por agregados. `python run.py --validate-training` comprueba todo sin entrenar. Para una repetición nueva CPU: instala requirements_training.txt y ejecuta `python run.py --train --seed 2019 --output nueva_replica`; sin --seed entrena las cinco semillas y puntúa. Puede tardar horas. El registro local nuevo se distingue del preregistro histórico. Esa repetición no se ejecutó al preparar este paquete.\n"
    elif key=="D11":
        text+="Los 21 ítems conservan sus estados: 9 completos dentro de su alcance, 10 parciales y 2 NAS no aplicables. El runner valida coherencia y exporta las evidencias; no declara todos los criterios aprobados ni reentrena los 81 modelos.\n"
    elif key=="D12":
        text+="Las diapositivas y notas están preparadas para la defensa; su generación no acredita que la presentación oral ya haya ocurrido. No necesitan PowerPoint, CDN ni el sitio del curso. El contenido mantiene los resultados negativos, unidades y límites del estudio.\n"
    else:
        text+="Las verificaciones son de integridad y coherencia semántica del documento/registro, con datos reales incluidos. No sustituyen controles experimentales pendientes ni repiten entrenamientos. Los estados parciales quedan visibles.\n"
    text+="\n## Procedencia\n\nFuente documental: `"+SOURCE+"`. Los archivos originales conservan sus bytes y los programas nuevos son adaptadores de entrega. `source_manifest.json` enumera rutas, tamaños y SHA-256 de los archivos del paquete; `package.json` describe el componente. Los resultados locales no se confunden con otra campaña científica. No se incluyen credenciales, .git, Colab o rutas privadas necesarias para ejecutar. Las licencias de autores y datasets incluidos se conservan junto a sus fuentes.\n"
    (root/"LEEME.md").write_text(text,encoding="utf-8")
    (root/"ejecutar.sh").write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nif [ -x .venv/bin/python ]; then\n  exec .venv/bin/python run.py "$@"\nfi\nexec python3 run.py "$@"\n',encoding="utf-8")
    (root/"ejecutar.cmd").write_bytes(b'@echo off\r\ncd /d "%~dp0"\r\nif exist ".venv\\Scripts\\python.exe" (\r\n  ".venv\\Scripts\\python.exe" run.py %*\r\n) else (\r\n  py -3.12 run.py %*\r\n)\r\nif errorlevel 1 pause\r\n')


def finalize_package(root,key):
    readme(root,key)
    records=[]
    for path in sorted(root.rglob("*")):
        if not path.is_file():continue
        records.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)})
    dump(root/"source_manifest.json",{"version":"e2-independent-component-v1","id":key,"source_commit":SOURCE,"files":records,"outputs_in_manifest":False})
    target=PUBLIC/NAMES[key][0]
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():continue
            name=path.relative_to(root).as_posix()
            info=zipfile.ZipInfo(key+"/"+name,(1980,1,1,0,0,0))
            info.external_attr=(0o100755 if name.endswith(".sh") else 0o100644)<<16
            info.compress_type=zipfile.ZIP_STORED if name=="payload.tar.xz" else zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes(),compresslevel=9)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip():raise ValueError("ZIP CRC error")
    return {"id":key,"title":NAMES[key][1],"file":target.name,"bytes":target.stat().st_size,"sha256":sha(target),"root_directory":key,"files":len(records)+1,"default_command":"python run.py","requires_installation_for_default":key=="D02","scientific_source_commit":SOURCE}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-payload",type=Path,default=Path('/tmp/e2-standalone-book-payload.tar.xz'))
    parser.add_argument("--paper-extra",type=Path,default=Path('/tmp/e2-paper-standalone-inputs'))
    parser.add_argument("--staging",type=Path)
    parser.add_argument("--only",choices=sorted(NAMES))
    args=parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True)
    staging=args.staging or Path(tempfile.mkdtemp(prefix="e2-independent-packages-"))
    staging.mkdir(parents=True,exist_ok=True)
    index=[]
    for key in ([args.only] if args.only else sorted(NAMES)):
        root=staging/key
        if root.exists():raise ValueError("Use a fresh staging directory: "+str(root))
        root.mkdir()
        if key=="D01":
            copy(root,"scripts/e2_standalone_book.py","run.py")
            copy(root,args.book_payload,"payload.tar.xz")
            config(root,key,scope="Libro completo offline, MathJax local y fuente exacta; rebuild opcional.")
        elif key=="D02":code_bundle(root)
        elif key=="D03":dataset_bundle(root)
        elif key in {"D04","D05","D07","D09","D11"}:reporting_bundle(root,key)
        elif key in {"D06","D08","D12"}:analysis_bundle(root,key)
        elif key=="D10":paper_bundle(root,args.paper_extra)
        item=finalize_package(root,key);index.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    combined=index
    if args.only and (PUBLIC/"index.json").is_file():
        old=json.loads((PUBLIC/"index.json").read_text())["packages"]
        combined=sorted([r for r in old if r["id"]!=args.only]+index,key=lambda r:r["id"])
    if len(combined)==12:
        dump(PUBLIC/"index.json",{"version":"e2-independent-deliverables-v1","source_commit":SOURCE,"packages":combined,"every_archive_has_own_run_py":True,"cross_package_dependency":False,"default_requires_website_or_git_clone":False,"neural_market_training_executed_during_packaging":False})
        with (PUBLIC/"SHA256SUMS").open("w",encoding="utf-8") as stream:
            for row in combined:stream.write(row["sha256"]+"  "+row["file"]+"\n")
    if not args.only:
        Path('/tmp/e2-independent-packages-staging-path').write_text(str(staging))
    print(json.dumps({"staging":str(staging),"packages":len(index),"total_bytes":sum(r["bytes"] for r in index)}))


if __name__=="__main__":
    main()
