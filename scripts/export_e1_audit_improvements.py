"""Publish an explicit allowlist of aggregate audit improvements only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/e1_audit_improvements_20261104"
DESTINATION = ROOT / "book/_static/e1_audit_improvements_20261104"
PROTOCOL = ROOT / "experiments/e1_audit_improvements_20261104/protocol.json"
FILES = {
 "descriptive": ["README.md","summary.json","code_manifest_before_analysis.json",
  "annual_run_metrics.csv","annual_metrics_summary.csv","distribution_by_period.csv",
  "variance_components.csv","development_bin_thresholds.json","severity_run_metrics.csv",
  "severity_metrics_summary.csv","residual_bins_by_seed.csv","residual_bins_summary.csv",
  "monthly_prediction_aggregates.csv","difficult_months.csv","annual_rmse.png",
  "variance_components.png","monthly_predictions_D7.png","residual_variance_D7.png"],
 "inference": ["README.md","summary.json","paired_comparisons.csv","wilcoxon_sensitivity.csv",
  "dm_tests.csv","ljung_box.csv","date_bootstrap.csv","design_sensitivity.csv",
  "relative_differences.csv","paired_qq.png","date_bootstrap_intervals.png"],
 "planning": ["README.md","registration.json","summary.json","advanced_margin_scenarios.csv",
  "operational_threshold_scenarios.csv","academic_joint_power_scenarios.csv",
  "academic_planning_summary.json","academic_margin_README.md"],
 "traceability": ["README.md","summary.json","epoch_summary_by_month.csv","epoch_policy_summary.csv"],
 "reproducibility": ["README.md","summary.json","replication_receipt.json",
  "private_bundle_receipt.json","independent_replica_audit.json"],
}
AUDITS = ["revision_e1_final_inference_20261104", "revision_e1_audit_descriptives_20261104",
          "revision_e1_margin_planning_20261104", "revision_e1_academic_policy_20261104"]
LITERATURE_FILES = ["report.md","summary.json","search_log.json","candidate_assessment.csv",
    "candidate_assessment.json","source_code_manifest.json","dataset_comparability.csv",
    "dataset_comparability.json","manifest.json"]
CODE = ["analyze_e1_audit_descriptives.py","analyze_e1_final_inference.py","plan_e1_advanced_margin.py",
        "plan_e1_academic_benchmark.py",
        "summarize_e1_epoch_policy.py","restore_e1_private_artifacts.py",
        "verify_e1_final_inference.py","verify_e1_audit_descriptives.py","verify_e1_margin_planning.py",
        "verify_e1_academic_policy.py",
        "draw_e1_architectures.py","draw_e1_cv_calendar.py","export_e1_audit_improvements.py"]
DENIED_COLUMNS = {"y_true","y_pred","persistence_pred","residual","residuals","error",
                  "open","high","low","close","volume","sigma_7","prediction"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def check_file(path):
    if path.name.endswith((".pt",".pth",".tar.gz",".zip",".npy",".npz",".parquet")):
        raise ValueError("A weight, archive or vector file is outside the public scope")
    if path.suffix == ".csv":
        columns = set(pd.read_csv(path,nrows=0).columns)
        if DENIED_COLUMNS & columns:
            raise ValueError("Individual market/forecast columns cannot be published: "+str(path))
        if path.name == "monthly_prediction_aggregates.csv":
            frame=pd.read_csv(path)
            if (not {"phase","month","n_dates"} <= set(frame)
                    or not frame.month.astype(str).str.fullmatch(r"\d{4}-\d{2}").all()
                    or not frame.n_dates.between(28,31).all()):
                raise ValueError("Monthly predictions must remain whole-month aggregates of 28–31 dates")
        if path.name == "residual_bins_by_seed.csv":
            frame=pd.read_csv(path)
            identities={"history_days","model","seed","predicted_bin","n_dates","stochastic_training","status"}
            if not identities <= set(frame):
                raise ValueError("Residual bin metadata is incomplete")
            small=frame.n_dates.lt(10)
            if frame.loc[small,list(set(frame)-identities)].notna().any().any():
                raise ValueError("Residual bins with fewer than ten dates must suppress their statistics")


def validate_summary(folder):
    summary=json.loads((folder/"summary.json").read_text())
    for name,expected in summary.get("output_sha256",summary.get("outputs_sha256",{})).items():
        path=folder/name
        if digest(path)!=expected:
            raise ValueError("Recorded output changed: "+str(path))
    for name,expected in summary.get("source_sha256",{}).items():
        if digest(ROOT/name)!=expected:
            raise ValueError("Recorded source changed: "+name)
    if summary.get("protocol_sha256") and summary["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Supplementary protocol changed")
    for key in ["new_model_fits","new_project_fits","new_checkpoint_forward_calls",
                "checkpoint_forward_calls","raw_archives_opened","raw_market_archives_opened"]:
        if summary.get(key,0) != 0:
            raise ValueError("This supplement must not train, predict again or open raw archives")
    return summary


def export():
    for folder in FILES:
        validate_summary(SOURCE/folder)
    for name in AUDITS:
        report=json.loads((ROOT/"references/audits"/(name+".json")).read_text())
        if report.get("status")!="PASS":
            raise ValueError("Independent verification must pass before publication: "+name)
    repro=json.loads((SOURCE/"reproducibility/summary.json").read_text())
    if repro.get("status")!="PASS":
        raise ValueError("Private isolated restoration must pass before publication")
    inputs={}
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".e1_audit_summary_",dir=DESTINATION.parent) as temporary:
        build=Path(temporary)
        def copy(source,relative):
            source=Path(source);check_file(source)
            dest=build/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
            inputs[source.relative_to(ROOT).as_posix()]=digest(source)
        copy(PROTOCOL,"protocol.json")
        copy(PROTOCOL.with_name("academic_margin_policy.json"),"academic_margin_policy.json")
        copy(ROOT/"experiments/e1_next_study_pilot_template.json","next_study_pilot_DRAFT.json")
        copy(SOURCE/"inference_registration.json","inference_registration.json")
        for folder,names in FILES.items():
            for name in names:copy(SOURCE/folder/name,f"{folder}/{name}")
        old_note=build/"planning/README.md"
        old_note.write_text(
            "> Actualización: tras confirmar el contexto académico se definió un criterio operativo futuro del 5 % "
            "frente a ambas referencias. Véanse academic_margin_README.md y academic_margin_policy.json. "
            "El texto siguiente conserva el alcance histórico de los primeros escenarios; no es una política "
            "previa al test ya observado.\n\n"+old_note.read_text())
        # These are operational floors for a future design, not altered results
        # of the original hypothetical normal-difference power calculation.
        planning=pd.read_csv(SOURCE/"planning/advanced_margin_scenarios.csv")
        for method in ["R_normal_approximation_ceiling","R_noncentral_t_normal_difference"]:
            planning[method+"_with_guide_min5"]=planning[method].clip(lower=5)
            planning[method+"_with_rank_min6"]=planning[method].clip(lower=6)
        planning["scope_execution_minimum"]="Future execution requires max(guide minimum,dimensioned R); current pilot not prospectively completed"
        planning.to_csv(build/"planning/advanced_execution_minimums.csv",index=False,lineterminator="\n")
        (build/"planning/execution_minimums_note.md").write_text(
            "# Mínimos de ejecución y R teórico\n\n"
            "El R de la t no central es una cantidad teórica bajo los supuestos declarados. "
            "El diseño ejecutable debe respetar **R ≥ max(5, R dimensionado)**; si usa pruebas por rangos, "
            "**R ≥ max(6, R dimensionado)**. Un R teórico de 3 no elimina el mínimo de la guía ni acredita potencia real. "
            "La plantilla de un piloto futuro sigue como borrador no ejecutado.\n")
        lit=ROOT/"references/literature_e1/recovery_20261104"
        for name in LITERATURE_FILES:copy(lit/name,f"literature/{name}")
        for name in AUDITS:
            for suffix in [".json",".md"]:
                copy(ROOT/"references/audits"/(name+suffix),f"verification/{name}{suffix}")
        for name in CODE:copy(ROOT/"scripts"/name,f"code/{name}")
        copy(ROOT/"BUILD.md","BUILD.md")
        for name in ["cierre_auditoria_e1_20261104.json","cierre_auditoria_e1_20261104.csv"]:
            copy(ROOT/"references/audits"/name,name)
        (build/"README.md").write_text(
            "# Mejoras concretas de la auditoría E1\n\n"
            "Complemento exploratorio calculado después de conocer el test, a partir de pronósticos ya guardados. "
            "No se entrenaron modelos, ejecutaron nuevos pronósticos ni abrieron ZIP de mercado. "
            "MLP/D7 sigue siendo el baseline elegido antes del test.\n\n"
            "Se publican únicamente métricas agregadas, pruebas, tablas por año/mes/rango, diagramas y "
            "metadatos de trazabilidad. Los pesos, las predicciones individuales, los derivados diarios, "
            "el paquete privado de réplica y el PDF del curso no se copian al sitio. "
            "La carpeta reproducibility contiene recibos y una auditoría de restauración, no el paquete.\n\n"
            "El criterio operativo académico futuro es una mejora superior al 5 % frente a MLP/D7 y persistencia; "
            "los escenarios de potencia son hipotéticos y no acreditan un piloto prospectivo o utilidad financiera. "
            "El borrador next_study_pilot_DRAFT.json no ha sido ejecutado. "
            "El corpus incluido mantiene cinco textos completos: los candidatos nuevos con código no se "
            "convierten en papers leídos ni cierran el respaldo aplicado de MLP/CNN1D.\n\n"
            "Los IC por semillas, diferencias temporales y bootstrap tienen estimandos distintos; no son "
            "confirmaciones independientes. Los p-valores e intervalos conservan sus límites y multiplicidad.\n")
        for path in build.rglob("*"):
            if path.is_file():check_file(path)
        manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "publication_scope":"explicit aggregate-only allowlists; no new vectors, prices, checkpoints or archives",
            "source_sha256":inputs,"new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_archive_reads":0,
            "files_sha256":{p.relative_to(build).as_posix():digest(p) for p in build.rglob("*") if p.is_file()}}
        write_json(build/"public_allowlist_manifest.json",manifest)
        if DESTINATION.exists():shutil.rmtree(DESTINATION)
        shutil.copytree(build,DESTINATION)
    return {"status":"aggregate_export_verified","files":len(manifest["files_sha256"])+1,
            "new_model_fits":0,"new_checkpoint_forward_calls":0,"new_market_archive_reads":0}


if __name__ == "__main__":print(json.dumps(export()))
