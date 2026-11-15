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
import io
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import tempfile
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = None  # Set only by the maintainer's explicit --source-commit.
SOURCE_BLOBS = {}
BOOK_INFO = {}
PUBLIC = ROOT / "book_e2/_static/entregables_independientes_v2_20261115"
CORRECTIONS = ROOT / "book_e2/_static/e2_corrections_20261115"
CORRECTION_AREAS = {
    "D04": ("data_audit", "controls", "cpu_diagnostic"),
    "D05": ("delivery", "statistics", "inference_statistics", "data_audit", "frozen_inference", "cpu_diagnostic", "controls"),
    "D06": ("delivery", "statistics", "inference_statistics", "frozen_inference", "cpu_diagnostic"),
    "D07": ("hpo", "frozen_inference", "cpu_diagnostic"),
    "D11": ("delivery", "statistics", "inference_statistics", "data_audit", "frozen_inference", "cpu_diagnostic", "controls", "hpo"),
    "D12": ("delivery", "statistics", "inference_statistics", "data_audit", "cpu_diagnostic", "controls", "hpo"),
}
PUBLIC_CORRECTION_SUFFIXES = {".json", ".jsonl", ".csv", ".md", ".txt", ".log", ".svg", ".png", ".html", ".py", ".npz"}
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


def verified_repository_source(path):
    """Fail if a required working-tree source differs from the announced C1."""
    path=Path(path)
    try: relative=path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Snapshot-bound input is outside the C1 repository; external caches need an explicit independent binding: "+str(path)) from exc
    expected=SOURCE_BLOBS.get(relative)
    if expected is None:raise ValueError("Required input is absent from the announced source commit: "+relative)
    raw=path.read_bytes()
    actual=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
    if actual!=expected:raise ValueError("Working-tree bytes differ from source commit: "+relative)
    return path


def copy(root, source, destination=None, *, snapshot_bound=True):
    source = Path(source)
    original = source if source.is_absolute() else ROOT/source
    if not original.is_file() or original.is_symlink():
        raise ValueError("Required regular source missing: "+str(original))
    if snapshot_bound:verified_repository_source(original)
    destination = destination or str(source)
    target = root/destination
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(original,target)
    return destination


def entry(root, source, title, *, folder="inputs", name=None):
    relative = copy(root,source,folder+"/"+(name or Path(source).name))
    return {"title":title,"path":relative,"sha256":sha(root/relative),"bytes":(root/relative).stat().st_size}


def config(root, key, *, documents=(), tables=(), json_files=(), scope=""):
    additions = correction_entries(root, key)
    record={"id":key,"title":NAMES[key][1],"source_commit":SOURCE,"scope":scope,
            "scientific_engine_commit":"1e8745497b484518f766b6da5c19be8362d9add7",
            "documents":list(documents) + additions["documents"],
            "tables":list(tables) + additions["tables"],
            "json_files":list(json_files) + additions["json_files"],
            "corrections":additions["areas"],
            "corrections_are_posthoc":True,
            "original_A2_replaced":False,
            "full_original_training_replay_claimed":False}
    dump(root/"package.json",record)
    return record


def correction_entries(root, key):
    """Include public correction outputs, preserving each area's namespace."""
    result = {"documents":[], "tables":[], "json_files":[], "areas":[]}
    for area in CORRECTION_AREAS.get(key, ()):
        directory = CORRECTIONS / area
        records = []
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                relative = path.relative_to(directory)
                if not path.is_file() or path.is_symlink() or path.suffix.lower() not in PUBLIC_CORRECTION_SUFFIXES:
                    continue
                if any(part in {"__pycache__", ".git", "work", "weights", "checkpoints", "market_raw"} for part in relative.parts):
                    continue
                destination = "corrections/" + area + "/" + relative.as_posix()
                copy(root,path,destination)
                item={"title":"Corrección " + area + " · " + relative.as_posix(),
                      "path":destination,"bytes":path.stat().st_size,"sha256":sha(path),
                      "correction_area":area,
                      "partial_execution":(path.parent/"failure_report.json").is_file(),
                      "render_in_report":not any(part in {"sources", "executed_source", "executed_sources"} for part in relative.parts)}
                records.append(item)
                kind = {".json":"json_files", ".csv":"tables", ".md":"documents"}.get(path.suffix.lower())
                if kind and item["render_in_report"]: result[kind].append(item)
        result["areas"].append({"area":area,"files":records,
            "availability":"included" if records else "not_available_in_this_snapshot",
            "interpretation":"Recorded evidence with its own status; inclusion is not methodological approval."})
    return result


CORRECTION_REPLAY_WRAPPER = '''"""Portable numerical replay of an included immutable correction plan.

The original program bytes and repository-relative inputs are retained. This
adapter calls the registered execute function; it does not register a new plan,
invoke Git, download inputs, train models or rewrite the original evidence.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, shutil, sys
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    if args.output.exists():raise ValueError("Use a new correction-replay output directory")
    sys.path.insert(0,str(ROOT/"source/scripts"))
    records=[]
    for item in config["programs"]:
        source=ROOT/item["program"]
        if digest(source)!=item["program_sha256"]:raise ValueError("Registered program changed")
        plan=ROOT/item["plan"]
        if digest(plan)!=item["plan_sha256"]:raise ValueError("Registered plan changed")
        destination=args.output/item["id"]
        spec=importlib.util.spec_from_file_location("e2_correction_replay_"+item["id"],source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        if item.get("kind","effects_rq2") in {"inference_statistics","cpu_diagnostic_statistics"}:
            summary=module.execute(argparse.Namespace(plan=plan,predictions_output=ROOT/item["predictions"],latency_output=ROOT/item["latency"],output=destination))
        else:
            destination.mkdir(parents=True)
            shutil.copyfile(plan,destination/"analysis_plan.json")
            summary=module.execute(destination)
        comparisons=[]
        for check in item["comparison_CSVs"]:
            with (ROOT/check["reference"]).open(encoding="utf-8",newline="") as f:expected=list(csv.DictReader(f))
            with (destination/check["name"]).open(encoding="utf-8",newline="") as f:actual=list(csv.DictReader(f))
            if len(expected)!=len(actual):raise ValueError("Replay CSV row count differs: "+check["name"])
            maxdiff=0.0
            for left,right in zip(expected,actual):
                if set(left)!=set(right):raise ValueError("Replay CSV columns differ")
                for key,value in left.items():
                    if value==right[key]:continue
                    try:a,b=float(value),float(right[key])
                    except ValueError:raise ValueError("Replay categorical value differs: "+key)
                    if not (math.isfinite(a) and math.isfinite(b)) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):raise ValueError("Replay numerical value differs: "+key)
                    maxdiff=max(maxdiff,abs(a-b))
            comparisons.append({"file":check["name"],"rows":len(actual),"maximum_absolute_difference":maxdiff})
        records.append({"program":item["id"],"original_program_bytes_unchanged":True,"summary":summary,"comparisons":comparisons})
    report={"status":"verified","scope":"Numerical replay of included posthoc correction plans; no new registration or model fitting", "source_commit":config["source_commit"],"records":records}
    (args.output/"replay_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
'''


def corrections_reanalysis(root, key):
    """Package supported replays with exact executed source and mirrored inputs."""
    if key not in {"D05", "D06", "D11", "D12"}: return
    plan_path = CORRECTIONS / "statistics/analysis_plan.json"
    if not plan_path.is_file(): return
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("version") != "e2-effects-rq2-closure-v1": return
    relative = plan["script"]["repository_path"]
    original = ROOT / relative
    if not original.is_file() or sha(original) != plan["script"]["sha256"]:
        candidates = list((CORRECTIONS / "statistics").rglob(Path(relative).name))
        original = next((p for p in candidates if p.is_file() and sha(p) == plan["script"]["sha256"]), None)
        if original is None: raise ValueError("Executed correction source unavailable at its registered SHA")
    program = "source/" + relative
    copy(root, original, "reanalysis/" + program)
    for record in plan["inputs"].values():
        path = ROOT / record["repository_path"]
        if not path.is_file() or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Correction replay input changed: " + record["repository_path"])
        copy(root,path,"reanalysis/source/"+record["repository_path"])
    copy(root,plan_path,"reanalysis/plans/statistics_analysis_plan.json")
    comparisons=[]
    for name in ["standardized_effects_20.csv", "paired_seed_gains_120.csv", "causal_regime_labels_608.csv", "rq2_primary_Holm10.csv", "rq2_block_sensitivity.csv"]:
        path=CORRECTIONS/"statistics"/name
        if not path.is_file():raise ValueError("Completed replay reference unavailable: "+name)
        copy(root,path,"reanalysis/references/statistics/"+name)
        comparisons.append({"name":name,"reference":"references/statistics/"+name})
    programs=[{"id":"statistics","kind":"effects_rq2","program":program,"program_sha256":sha(original),
               "plan":"plans/statistics_analysis_plan.json","plan_sha256":sha(plan_path),"comparison_CSVs":comparisons}]
    receipt_path=CORRECTIONS/"inference_statistics/execution_receipt.json"
    scipy_version=None
    if receipt_path.is_file():
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
        analysis_plan=CORRECTIONS/"inference_statistics/statistical_plan.json"
        if sha(analysis_plan)!=receipt["plan_sha256"]:raise ValueError("Inference-statistics plan differs from its execution receipt")
        binding=json.loads(analysis_plan.read_text(encoding="utf-8"))
        executed=ROOT/"scripts/e2_corrections_inference_statistics.py"
        if sha(executed)!=receipt["script_sha256"]:
            executed=next((p for p in (CORRECTIONS/"inference_statistics").rglob("*.py") if sha(p)==receipt["script_sha256"]),None)
            if executed is None:raise ValueError("Executed inference-statistics program unavailable at its recorded SHA")
        copy(root,executed,"reanalysis/source/scripts/e2_corrections_inference_statistics.py")
        for relative,expected in binding["inputs"].items():
            path=ROOT/relative
            if sha(path)!=expected:raise ValueError("Inference-statistics input changed: "+relative)
            copy(root,path,"reanalysis/source/"+relative)
        stages={}
        for stage,filename,field in [("predictions","inference_report.json","inference_report_sha256"),("latency","latency_report.json","latency_report_sha256")]:
            report_path=next((p for p in (CORRECTIONS/"frozen_inference").rglob(filename) if sha(p)==receipt[field]),None)
            if report_path is None:raise ValueError("Completed frozen-inference input report missing for replay: "+stage)
            report=json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status")!="passed":raise ValueError("Replay cannot use a failed/partial inference stage")
            destination="source/replay_inputs/"+stage
            copy(root,report_path,"reanalysis/"+destination+"/"+filename)
            for name,record in report["all_output_files"].items():
                if Path(name).name!=name:raise ValueError("Unsafe frozen-report input name")
                path=report_path.parent/name
                if sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:raise ValueError("Frozen replay input changed: "+name)
                copy(root,path,"reanalysis/"+destination+"/"+name)
            stages[stage]=destination
        copy(root,analysis_plan,"reanalysis/plans/inference_statistics_statistical_plan.json")
        comparisons=[]
        for name in ["delta_g_Holm13.csv","validation_test_Holm13.csv","RQ3_cpu_IUT_Holm10.csv"]:
            copy(root,CORRECTIONS/"inference_statistics"/name,"reanalysis/references/inference_statistics/"+name)
            comparisons.append({"name":name,"reference":"references/inference_statistics/"+name})
        programs.append({"id":"inference_statistics","kind":"inference_statistics",
                         "program":"source/scripts/e2_corrections_inference_statistics.py","program_sha256":sha(executed),
                         "plan":"plans/inference_statistics_statistical_plan.json","plan_sha256":sha(analysis_plan),
                         "predictions":stages["predictions"],"latency":stages["latency"],"comparison_CSVs":comparisons})
        scipy_version=receipt["scipy"]
    cpu_receipt_path=CORRECTIONS/"cpu_diagnostic/statistics/results/execution_receipt.json"
    if cpu_receipt_path.is_file():
        receipt=json.loads(cpu_receipt_path.read_text(encoding="utf-8"))
        candidates=[p for p in (CORRECTIONS/"cpu_diagnostic/statistics").glob("statistical_plan*.json")
                    if sha(p)==receipt["statistical_plan_sha256"]]
        if len(candidates)!=1:raise ValueError("Completed CPU diagnostic has no unique byte-bound statistical plan")
        cpu_plan_path=candidates[0]
        binding=json.loads(cpu_plan_path.read_text(encoding="utf-8"))
        if binding["program_sha256"]!=receipt["program_sha256"]:raise ValueError("CPU statistics programs disagree with their execution receipt")
        for name,expected in binding["program_sha256"].items():
            if Path(name).name!=name:raise ValueError("Unsafe CPU diagnostic source name")
            path=ROOT/"scripts"/name
            if sha(path)!=expected:raise ValueError("Executed CPU diagnostic source changed: "+name)
            target=root/"reanalysis/source/scripts"/name
            if target.is_file() and sha(target)!=expected:raise ValueError("Numerical replays require incompatible source versions: "+name)
            copy(root,path,"reanalysis/source/scripts/"+name)
        relative=Path(binding["inference_plan_path"])
        if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe CPU inference-plan path")
        path=ROOT/relative
        if sha(path)!=binding["inference_plan_sha256"] or sha(path)!=receipt["inference_plan_sha256"]:
            raise ValueError("CPU inference plan changed from the completed statistical study")
        copy(root,path,"reanalysis/source/"+relative.as_posix())
        stages={}
        for stage,filename,field,status in [
                ("predictions","inference_report.json","inference_report_sha256","completed_CPU_diagnostic_with_projection_alerts"),
                ("latency","latency_report.json","latency_report_sha256","completed_CPU_latency_diagnostic")]:
            candidates=[p for p in (CORRECTIONS/"cpu_diagnostic").rglob(filename) if sha(p)==receipt[field]]
            if len(candidates)!=1:raise ValueError("Completed CPU diagnostic input report is not uniquely available: "+stage)
            report_path=candidates[0]
            report=json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("status")!=status or report.get("stage")!=stage or not report.get("states_unchanged")
                    or report.get("original_parity_gate_passed") is not False
                    or report.get("statistical_plan_sha256")!=sha(cpu_plan_path)
                    or report.get("spec_sha256")!=binding["inference_plan_sha256"]):
                raise ValueError("CPU replay requires its complete separate study and retained failed original gate")
            if stage=="predictions" and (report.get("predictors_completed")!=78 or report.get("calibration_comparisons_completed")!=78):
                raise ValueError("CPU replay cannot filter projection alerts or use an incomplete cohort")
            destination="source/replay_inputs/cpu_"+stage
            copy(root,report_path,"reanalysis/"+destination+"/"+filename)
            for name,record in report["all_output_files"].items():
                if Path(name).name!=name:raise ValueError("Unsafe CPU-report input name")
                path=report_path.parent/name
                if sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:raise ValueError("CPU replay input changed: "+name)
                copy(root,path,"reanalysis/"+destination+"/"+name)
            stages[stage]=destination
        copy(root,cpu_plan_path,"reanalysis/plans/cpu_diagnostic_statistical_plan.json")
        comparisons=[]
        for name in ["CPU_delta_g_Holm13.csv","CPU_cost_only_Holm10.csv","CPU_latency_block_means_1000.csv"]:
            copy(root,cpu_receipt_path.parent/name,"reanalysis/references/cpu_diagnostic/"+name)
            comparisons.append({"name":name,"reference":"references/cpu_diagnostic/"+name})
        programs.append({"id":"cpu_diagnostic_statistics","kind":"cpu_diagnostic_statistics",
                         "program":"source/scripts/e2_corrections_cpu_diagnostic_statistics.py",
                         "program_sha256":binding["program_sha256"]["e2_corrections_cpu_diagnostic_statistics.py"],
                         "plan":"plans/cpu_diagnostic_statistical_plan.json","plan_sha256":sha(cpu_plan_path),
                         "predictions":stages["predictions"],"latency":stages["latency"],"comparison_CSVs":comparisons,
                         "scope":"Separate posterior CPU-only gap/cost diagnosis; no original GPU parity, original714 IUT or original RQ3 closure"})
        scipy_version=receipt["software"]["scipy"]
        if receipt["software"]["numpy"]!=plan["software"]["numpy"]:raise ValueError("Replay software versions require separate environments")
    dump(root/"reanalysis/config.json",{"version":"e2-portable-correction-replays-v1","source_commit":SOURCE,"programs":programs})
    (root/"reanalysis/run.py").write_text(CORRECTION_REPLAY_WRAPPER,encoding="utf-8")
    numpy_version=plan["software"]["numpy"]
    requirements="numpy=="+numpy_version+"\n"+("scipy=="+scipy_version+"\n" if scipy_version else "")
    (root/"requirements_corrections.txt").write_text(requirements,encoding="utf-8")


def documents(root, pairs):
    return [entry(root,"book_e2/sections/"+name,title,folder="documents") for name,title in pairs]


def current_delivery(name, fallback):
    names=[name]
    if name.startswith("a3_primary_contrasts_"):
        names.extend(["a3_primary_contrasts_readout.csv","a3_primary_contrasts_current.csv"])
    for current in dict.fromkeys(names):
        candidate=CORRECTIONS/"delivery"/current
        if candidate.is_file():return candidate
    return fallback


HPO_REPLAY_WRAPPER = '''"""Replay the included HPO metadata analysis; no acquisition or fitting."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, shutil, sys
ROOT=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("Use a fresh HPO-replay output directory")
    config=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    program=ROOT/"code/e2_hpo_coverage_closure.py"
    evidence=ROOT.parent/"corrections/hpo"
    if digest(program)!=config["program_sha256"] or digest(evidence/"analysis_plan.json")!=config["plan_sha256"]:raise ValueError("Bound HPO program/plan changed")
    args.output.mkdir(parents=True)
    shutil.copytree(evidence/"sources",args.output/"sources")
    shutil.copyfile(evidence/"analysis_plan.json",args.output/"analysis_plan.json")
    initial=evidence/"analysis_plan_initial_guard_failure.json"
    if initial.is_file():shutil.copyfile(initial,args.output/initial.name)
    spec=importlib.util.spec_from_file_location("e2_hpo_metadata_replay",program)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    summary=module.execute(args.output)
    comparisons=[]
    for name,expected in [("configurations_78.csv",78),("opportunities_234.csv",234),("model_reconciliation_13.csv",13),("parameter_coverage_65.csv",65)]:
        with (args.output/name).open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
        if len(rows)!=expected:raise ValueError("Replayed HPO table has wrong design count: "+name)
        identical=digest(args.output/name)==digest(evidence/name)
        if not identical:raise ValueError("Replayed HPO table differs from the included reference: "+name)
        comparisons.append({"file":name,"rows":len(rows),"byte_identical_to_included":identical})
    record={"status":"verified","scope":"Registered HPO metadata replay with original source/inputs; no Git, network, new HPO or fitting", "summary":summary,"tables":comparisons}
    (args.output/"portable_replay_report.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
    print(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
'''


def auxiliary_reanalyses(root, key):
    """Self-contained optional checks with scopes narrower than a full replay."""
    if key == "D04":
        script=verified_repository_source(ROOT/"scripts/e2_standalone_dataset.py")
        values=runpy.run_path(str(script))
        copy(root,script,"audit_data/run.py")
        for name,row in values["SOURCES"].items():
            copy(root,row["repository_path"],"audit_data/inputs/"+name)
    if key in {"D07","D11","D12"} and (CORRECTIONS/"hpo/analysis_plan.json").is_file():
        program=ROOT/"scripts/e2_hpo_coverage_closure.py"
        expected="c668e5853562298829a259b5ebd705dce8f9f359e3e2dffa1dbc272ad0fa12ea"
        if sha(program)!=expected:raise ValueError("Executed HPO metadata program differs from its registered bytes")
        copy(root,program,"hpo_reanalysis/code/e2_hpo_coverage_closure.py")
        dump(root/"hpo_reanalysis/config.json",{"program_sha256":expected,"plan_sha256":sha(CORRECTIONS/"hpo/analysis_plan.json"),"source_commit":SOURCE})
        (root/"hpo_reanalysis/run.py").write_text(HPO_REPLAY_WRAPPER,encoding="utf-8")
        (root/"requirements_hpo_reanalysis.txt").write_text("numpy==2.3.5\nmatplotlib==3.10.8\n",encoding="utf-8")


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
            (current_delivery("a3_primary_contrasts_current.csv","book_e2/_static/e2_agregacion_entrega_current_v1/a3_primary_contrasts_readout.csv"),"Veinte pares primarios y campos disponibles"),
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
        t(current_delivery("appendix_B_current.csv","book_e2/_static/e2_agregacion_entrega_current_v1/appendix_B_current.csv"),"Los 21 literales de reproducibilidad")
        j(current_delivery("appendix_B_current.json","book_e2/_static/e2_agregacion_entrega_current_v1/appendix_B_current.json"),"Estados, evidencia y límites")
        t("book_e2/_static/e2_reproduction_ablations_20261113/reproduction_by_model.csv","Recibos de las 81 reproducciones")
        j("book_e2/_static/e2_reproduction_ablations_20261113/historical_clean_runtime.json","Entorno limpio observado")
        j("book_e2/_static/e2_delivery_reproduction_20261114/delivery_audit.json","Auditoría CPU publicada")
        copy(root,"book_e2/_static/e2_repro_prereg_current_v1/environment_observed.lock.txt","inputs/environment_observed.lock.txt")
    return config(root,key,documents=docs,tables=tables,json_files=data,scope="Informe ejecutable desde entradas incluidas; estados científicos originales preservados.")


def analysis_bundle(root,key):
    copy(root,"scripts/e2_standalone_analysis.py","run.py")
    copy(root,PANEL,"inputs/e2_f2a_input.json")
    copy(root,"book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv","inputs/benchmark_A2.csv")
    copy(root,current_delivery("a3_primary_contrasts_current.csv","book_e2/_static/e2_agregacion_entrega_current_v1/a3_primary_contrasts_readout.csv"),"inputs/a3_primary_contrasts_readout.csv")
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


def code_bundle(root, authors_root):
    snapshot=subprocess.check_output(["git","archive","--format=tar",SOURCE,"scripts","tests","experiments","third_party"],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(snapshot),mode="r:") as archive:
        for member in archive:
            if member.isdir(): continue
            if any(part.lower().startswith(("e3_", "e3-", "run_e3_", "test_e3_")) for part in Path(member.name).parts):
                continue  # E3 files are outside this E2 deliverable projection.
            if not member.isfile() or ".." in Path(member.name).parts:
                raise ValueError("Only regular source files may enter the code snapshot")
            target=root/"code"/member.name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,target.open("wb") as stream:shutil.copyfileobj(source,stream)
    copy(root,"scripts/e2_standalone_code.py","code/scripts/e2_standalone_code.py")
    metadata=runpy.run_path(str(ROOT/"scripts/e2_standalone_code.py"))["METADATA_SHA256"]
    required=[*metadata,"book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json",
              "book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv",
              "results/eda_development/daily_market_development.csv","results/eda_development/daily_volatility_development.csv"]
    for path in required:copy(root,path,"code/"+path)
    for path in ROOT.glob("requirements*.txt"):
        if "e3" not in path.name.lower():copy(root,path,"code/"+path.name)
    design=json.loads((ROOT/"experiments/e2_completion_20261106/design.json").read_text())
    models={"TSMixer","TiDE","TimeMixer","TimesNet","PatchTST","iTransformer"}
    for row in design["author_sources"]:
        if row["model"] not in models:continue
        source=authors_root/row["model"]/row["path"]
        if sha(source)!=row["sha256"] or source.stat().st_size!=row["bytes"]:
            raise ValueError("Author source identity changed: "+str(source))
        copy(root,source,"authors/"+row["model"]+"/"+row["path"],snapshot_bound=False)
    (root/"run.py").write_text(CODE_WRAPPER,encoding="utf-8")
    (root/"requirements_cpu.txt").write_text(CPU_REQUIREMENTS,encoding="utf-8")
    copy(root,"requirements-e2-training-gpu.lock.txt","requirements_cuda.txt")
    return config(root,"D02",scope="Código completo, 76 fuentes de autores y recetas congeladas. Default: forwards CPU reales sobre fixtures; reentrenamiento CUDA local opt-in.")


def dataset_bundle(root):
    values=runpy.run_path(str(verified_repository_source(ROOT/"scripts/e2_standalone_dataset.py")))
    copy(root,"scripts/e2_standalone_dataset.py","run.py")
    for name,row in values["SOURCES"].items():copy(root,row["repository_path"],"inputs/"+name)
    copy(root,"scripts/download_e2_known_market.py","code/scripts/download_e2_known_market.py")
    copy(root,"book/_static/data/test_raw_reserve_20250101_20260831.json",
         "code/book/_static/data/test_raw_reserve_20250101_20260831.json")
    return config(root,"D03",scope="CSV diarios originales de desarrollo; reconstruye calidad y σ7. Descarga oficial verificada opcional; no incluye X final privado.")


def paper_bundle(root,extra):
    values=runpy.run_path(str(verified_repository_source(ROOT/"scripts/e2_standalone_paper.py")))
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
        if rel.startswith("authors/"):copy(root,path,rel,snapshot_bound=False)
        elif path.name in ("ETTh1.csv","LICENSE.ETDataset.txt","acquisition.json"):copy(root,path,"inputs/"+path.name,snapshot_bound=False)
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
        text+="El snapshot científico/documental corresponde al commit "+SOURCE+". Incluye "+str(BOOK_INFO["html_pages"])+" páginas HTML y la proyección de fuentes del libro, con MathJax local y licencias. El manifiesto inventaría "+str(BOOK_INFO["expanded_bytes"])+" bytes antes de extraer; una reconstrucción necesita espacio adicional para su salida. `--serve --port 8765` inicia un servidor sólo en 127.0.0.1; Ctrl+C lo detiene. Para reconstruir el libro: instala `output/source/requirements-book.lock.txt` y ejecuta `python run.py --build`. El build recompila el informe, no repite entrenamientos.\n"
    elif key=="D02":
        text+="`python run.py --smoke-cpu` usa pesos iniciales y un fixture [1,7,5]; no afirma precisión Bitcoin. Incluye código original de EDA, modelos, baselines, entrenamiento, estadísticas, robustez, incertidumbre, explicaciones y tests bajo code/. Para un reentrenamiento real opcional, usa Linux/WSL2 y CUDA: instala requirements_cuda.txt en otro venv y ejecuta `python run.py --train --model MLP16 --seed 271801 --device cuda:0 --output nuevo_ajuste`. Emplea 655 ejemplos train, 53 held-out y las épocas congeladas; no abre el controlador ni el presupuesto de la campaña. Esa opción no se ejecutó al preparar el ZIP. No incluye checkpoints originales ni el tensor X de prueba; los resultados del benchmark se reconstruyen sin ellos en D06.\n"
    elif key=="D03":
        text+="Incluye los dos CSV diarios de desarrollo, la Data Card, contrato y manifiestos. CSV diario y tensor de ventanas no son el mismo objeto. No incluye ni inventa los tensores finales X/escalares de test. Descarga opcional: `python run.py --download --archive BTCUSDT-1m-2023-01.zip`; requiere Internet y verifica CHECKSUM/hash oficiales. Sin --archive, la descarga selecciona los 24 meses de desarrollo; no se inicia por defecto.\n"
        text+="\nPara obtener las veinte fuentes originales del test conocido de enero de 2025 a agosto de 2026: `python code/scripts/download_e2_known_market.py --output datos_test`. Sólo necesita Python estándar y una red que permita data.binance.vision. Verifica el manifiesto histórico, tamaños y SHA-256; no cambia datos ni inicia ajustes. Para verificar una descarga ya completa: añade `--validate-only`. La descarga no reconstruye por sí sola todos los tensores del benchmark ni declara cerrada una auditoría cuyo test siga sin estar disponible.\n"
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
    if CORRECTION_AREAS.get(key):
        text+="\n## Correcciones posteriores incluidas\n\nLas carpetas corrections/ conservan el espacio de nombres de cada área: planes, recibos, tablas, fuentes y estados. Los informes predeterminados verifican esas entradas y muestran sus resultados; no vuelven a ajustar redes, ejecutar controles o remuestrear. Los resultados adversos, no estimables o pendientes se conservan. Estas ampliaciones no sustituyen A2 ni acreditan una defensa oral.\n"
    if (root/"reanalysis/run.py").is_file():
        text+="\n## Repetir los suplementos estadísticos incluidos\n\nInstala requirements_corrections.txt en un entorno separado y ejecuta `python reanalysis/run.py --output nueva_repeticion`. Se incluyen los programas originales exactos, planes fechados y todos sus insumos en sus rutas relativas. El adaptador llama a execute, sin Git, tokens, descarga, nuevo registro o ajustes de modelos. Compara las tablas numéricas con las referencias incluidas. reanalysis/config.json enumera los programas realmente disponibles. La repetición de efectos/RQ2 y el diagnóstico complementario CPU, si está incluido, conservan sus propios alcances. El suplemento CPU no acredita paridad CUDA, ausencia de sobreajuste original ni la IUT/RQ3 original; los dos fallos de paridad permanecen visibles. No se promete reproducir toda la campaña.\n"
    if (root/"audit_data/run.py").is_file():
        text+="\n## Reauditar el desarrollo incluido\n\n`python audit_data/run.py --output output/data_reaudit` ejecuta el verificador original de datos con Python estándar y sus seis insumos incluidos, sin Git ni otros ZIP. Reconstruye σ₇ y comprueba calendarios y soporte de desarrollo. No verifica por sí solo los trece ports, los 81 checkpoints o el contenido X del test ausente; los recibos de la auditoría actual y sus límites se conservan por separado en corrections/data_audit/.\n"
    if (root/"hpo_reanalysis/run.py").is_file():
        text+="\n## Repetir el análisis de metadatos HPO\n\nInstala requirements_hpo_reanalysis.txt y ejecuta `python hpo_reanalysis/run.py --output nueva_revision_hpo`. El adaptador copia el plan y sources incluidos a una salida nueva y llama al execute original byte por byte, sin Git, red, recuperación de datos o nuevos ajustes. Verifica las tablas frente a sus referencias. No uses --recover-only ni --register-plan para este replay: conservar el registro previo es parte del alcance.\n"
    text+="\n## Procedencia\n\nFuente documental: `"+SOURCE+"`. Los archivos originales conservan sus bytes y los programas nuevos son adaptadores de entrega. `source_manifest.json` enumera rutas, tamaños y SHA-256 de los archivos del paquete; `package.json` describe el componente. Los resultados locales no se confunden con otra campaña científica. No se incluyen credenciales, .git, Colab o rutas privadas necesarias para ejecutar. Las licencias de autores y datasets incluidos se conservan junto a sus fuentes.\n"
    (root/"LEEME.md").write_text(text,encoding="utf-8")
    (root/"ejecutar.sh").write_text('#!/bin/sh\nset -eu\ncd "$(dirname "$0")"\nif [ -x .venv/bin/python ]; then\n  exec .venv/bin/python run.py "$@"\nfi\nexec python3 run.py "$@"\n',encoding="utf-8")
    (root/"ejecutar.cmd").write_bytes(b'@echo off\r\ncd /d "%~dp0"\r\nif exist ".venv\\Scripts\\python.exe" (\r\n  ".venv\\Scripts\\python.exe" run.py %*\r\n) else (\r\n  py -3.12 run.py %*\r\n)\r\nif errorlevel 1 pause\r\n')


def finalize_package(root,key):
    auxiliary_reanalyses(root,key)
    corrections_reanalysis(root,key)
    package=json.loads((root/"package.json").read_text(encoding="utf-8"))
    if (root/"audit_data/run.py").is_file():
        package["optional_data_reaudit"]={"entrypoint":"audit_data/run.py","standard_library_only":True,
            "scope":"Included development CSV/target/support checks; not a full thirteen-port or test-X audit"}
    if (root/"hpo_reanalysis/run.py").is_file():
        package["optional_hpo_reanalysis"]={"entrypoint":"hpo_reanalysis/run.py","requirements":"requirements_hpo_reanalysis.txt",
            "original_program_bytes_unchanged":True,"new_HPO_or_fitting":False}
    if (root/"reanalysis/run.py").is_file():
        package["optional_correction_reanalysis"]={"entrypoint":"reanalysis/run.py","config":"reanalysis/config.json",
            "original_program_bytes_unchanged":True,"git_or_external_checkout_required":False,
            "programs":[r["id"] for r in json.loads((root/"reanalysis/config.json").read_text())["programs"]],
            "scope":"Recorded posthoc analytical replays from included data; no model fitting or new predictor inference"}
    dump(root/"package.json",package)
    readme(root,key)
    records=[]
    for path in sorted(root.rglob("*")):
        if not path.is_file():continue
        records.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)})
    dump(root/"source_manifest.json",{"version":"e2-independent-component-v1","id":key,"source_commit":SOURCE,"files":records,"outputs_in_manifest":False})
    target=PUBLIC/NAMES[key][0]
    descriptor, temporary_name=tempfile.mkstemp(prefix="."+key+"-",suffix=".zip",dir=PUBLIC)
    __import__('os').close(descriptor)
    temporary=Path(temporary_name)
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():continue
            name=path.relative_to(root).as_posix()
            info=zipfile.ZipInfo(key+"/"+name,(1980,1,1,0,0,0))
            info.external_attr=(0o100755 if name.endswith(".sh") else 0o100644)<<16
            info.compress_type=zipfile.ZIP_STORED if name=="payload.tar.xz" else zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes(),compresslevel=9)
    if temporary.stat().st_size>=100_000_000:
        temporary.unlink();raise ValueError("Standalone archive exceeds GitHub's 100 MB file limit: "+key)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip():raise ValueError("ZIP CRC error")
    temporary.replace(target)
    return {"id":key,"title":NAMES[key][1],"file":target.name,"bytes":target.stat().st_size,"sha256":sha(target),"root_directory":key,"files":len(records)+1,"default_command":"python run.py","requires_installation_for_default":key=="D02","scientific_source_commit":SOURCE}


def main():
    global SOURCE, SOURCE_BLOBS, BOOK_INFO, CORRECTIONS, PUBLIC
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit",required=True,help="Exact committed C1 source snapshot; no implicit HEAD")
    parser.add_argument("--book-payload",type=Path)
    parser.add_argument("--paper-extra",type=Path)
    parser.add_argument("--authors-root",type=Path,default=Path('/workspace/work/e2architecture_sources'))
    parser.add_argument("--corrections-root",type=Path,default=CORRECTIONS)
    parser.add_argument("--public-output",type=Path,default=PUBLIC)
    parser.add_argument("--staging",type=Path)
    parser.add_argument("--only",choices=sorted(NAMES))
    args=parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}",args.source_commit):parser.error("--source-commit requires a full 40-character SHA")
    SOURCE=subprocess.check_output(["git","rev-parse",args.source_commit+"^{commit}"],cwd=ROOT,text=True).strip()
    if SOURCE!=args.source_commit:parser.error("Source commit did not resolve to the exact supplied SHA")
    tree=subprocess.check_output(["git","ls-tree","-rz",SOURCE],cwd=ROOT)
    SOURCE_BLOBS={}
    for record in tree.split(b"\0"):
        if not record:continue
        metadata,name=record.split(b"\t",1)
        mode,kind,oid=metadata.split()
        if kind==b"blob":SOURCE_BLOBS[name.decode("utf-8")]=oid.decode("ascii")
    CORRECTIONS=args.corrections_root.resolve();PUBLIC=args.public_output.resolve()
    if args.only in (None,"D01") and args.book_payload is None:parser.error("D01 requires --book-payload generated for this exact source commit")
    if args.only in (None,"D10") and args.paper_extra is None:parser.error("D10 requires --paper-extra with the verified ETTh1/author cache")
    if args.only in (None,"D01"):
        with tarfile.open(args.book_payload,"r:xz") as archive:
            stream=archive.extractfile("bundle_manifest.json")
            if stream is None:raise ValueError("D01 payload has no bundle manifest")
            with stream:payload_manifest=json.load(stream)
        if payload_manifest.get("source_commit")!=SOURCE:
            raise ValueError("D01 payload source differs from the exact --source-commit")
        BOOK_INFO={"html_pages":payload_manifest["html_validation"]["html_pages"],
                   "expanded_bytes":sum(record["bytes"] for record in payload_manifest["files"].values())}
    if args.only and (PUBLIC/"index.json").is_file():
        prior=json.loads((PUBLIC/"index.json").read_text(encoding="utf-8"))
        if prior.get("source_commit")!=SOURCE or any(row.get("scientific_source_commit")!=SOURCE for row in prior.get("packages",[])):
            raise ValueError("--only cannot mix source snapshots; build all twelve or use a fresh --public-output")
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
            copy(root,args.book_payload,"payload.tar.xz",snapshot_bound=False)
            config(root,key,scope="Libro completo offline, MathJax local y fuente exacta; rebuild opcional.")
        elif key=="D02":code_bundle(root,args.authors_root)
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
