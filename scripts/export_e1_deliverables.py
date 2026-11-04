"""Build a public delivery index from existing records, without running models."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"book/_static/e1_deliverables_20261104"
OLD=ROOT/"results/dl_r2_cv_20261104"
NEW=ROOT/"results/filipe_multihistory_20261104"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    freeze_path=ROOT/"experiments/filipe_multihistory_20261104/freeze.json"
    freeze=read(freeze_path)
    plan_path=ROOT/"experiments/filipe_multihistory_20261104/protocol.json"
    plan=read(plan_path)
    policy_path=ROOT/"experiments/e1_audit_improvements_20261104/academic_margin_policy.json"
    policy=read(policy_path)
    initial_path=ROOT/"experiments/dl_benchmark_20261104/protocol_initial.json"
    initial=read(initial_path)
    audit_path=NEW/"test/raw_test_audit.json"
    raw_audit=read(audit_path)
    supplement_path=ROOT/"experiments/e1_audit_improvements_20261104/protocol.json"
    supplement=read(supplement_path)
    inputs={}
    for path in [freeze_path,plan_path,policy_path,initial_path,audit_path,supplement_path,
                 ROOT/"references/decisiones_e1_20261102.json"]:
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    entries=[]
    for label,path,key in [
        ("Primer protocolo de implementación; posterior al EDA",initial_path,"registered_at_utc"),
        ("Extensión de cuatro entradas registrada",plan_path,"registered_at_utc"),
        ("Selección y épocas cerradas antes del test",freeze_path,"closed_at_utc"),
        ("Primer acceso de modelado a los ZIP de test",audit_path,"first_raw_access_at_utc"),
        ("Suplemento inferencial posterior a inspeccionar test",supplement_path,"registered_at_utc"),
        ("Criterio académico futuro; posterior al test actual",policy_path,"recorded_at_utc")]:
        value=read(path)[key]
        local=datetime.fromisoformat(value).astimezone(ZoneInfo("America/Bogota"))
        entries.append({"event":label,"timestamp_utc":value,"timestamp_bogota":local.isoformat(),
                        "source":path.relative_to(ROOT).as_posix(),"source_sha256":sha(path)})
    declaration={
        "document_role":"Versioned declaration compiled from existing records; not a new preregistration",
        "compiled_at_bogota":datetime.now(ZoneInfo("America/Bogota")).isoformat(),
        "problem":{"type":4,"input_features":["open","high","low","close","volume"],
                   "input_history_days":[7,14,21,28],"target_return_days":7,"target_ddof":0,"forecast_horizon_days":1},
        "primary_metric":{"name":"RMSE","scale":"Original unannualized sigma7 units after inverse target scaling",
                          "aggregation":"Pooled evaluation dates per model/history/seed, then mean and sample SD across seeds"},
        "secondary_metrics":["MAE","MSE","R2","MAPE with strictly positive targets and reported coverage","MASE with permitted-training calendar lag1 scale"],
        "original_operational_margin":{"relative_RMSE_gain":.05,"reference":"Deterministic persistence on the same eligible dates",
            "timing":"Documented before the first project DL fit and reserved-test evaluation, after development EDA",
            "limit":"Not a validated financial tolerance; not power for proving a gain above5percent; not automatically a margin between two networks"},
        "future_academic_margin":{"relative_RMSE_gain":policy["minimum_relative_improvement"],
            "references":["Frozen neural baseline MLP/D7","Deterministic persistence"],
            "estimand":"1 - E[seed RMSE advanced]/E[seed RMSE reference]; boundary difference=.95*RMSEreference-RMSEadvanced",
            "timing":"Defined after the current test, for future independent comparisons only",
            "financial_utility_claim":False,"policy_source":policy_path.relative_to(ROOT).as_posix()},
        "units":{"observation":"Daily UTC outcome; adjacent sigma7 labels share six returns",
            "training_randomness":"Ten paired training seeds4101..4110, conditional on the observed calendar",
            "independent_market_series":1,"market":"Binance Spot BTCUSDT",
            "temporal_uncertainty":"Shared chronological dates and temporal blocks; days/months/seeds do not create independent markets"},
        "partition":{"initial_training":"2023-01-01..2023-12-31","forward_validation":"2024-01-01..2024-12-31",
            "final_development_fit":"Permitted2023-2024 outcomes with D-day purge before test",
            "evaluated_test":"2025-01-01..2026-08-31","unassigned":"September-October2026",
            "calendar_days":{"initial":365,"validation":366,"test":608,"total":1339},
            "calendar_percentages":{"initial":27.26,"validation":27.33,"test":45.41},
            "split_rule":"Chronological deterministic, no shuffled partition; no partition seed applies",
            "embargo":"Equal to maximum covariate lag D under guide6.4",
            "CV":"Actual timeseries-cv daily forward indices grouped into twelve monthly training/validation blocks",
            "test":"120 models frozen before first modeling access; twenty reporting months, zero refits"},
        "baseline_selection":freeze["selection"]["baseline"],
        "chronology":entries,
        "limits":["No claim of registration before EDA or before the historical market periods",
                  "The final supplement is post-hoc, not prospective confirmation",
                  "The future academic criterion was not predeclared for the completed test",
                  "A prospective pilot and new evaluation remain pending for a future study"],
        "inherited_next_delivery":"Keep source, target, metric and dependence definitions explicit; any new model designed from this test needs another independent evaluation.",
        "source_sha256":inputs}
    write_json(DEST/"declaration_and_chronology.json",declaration)

    # Metadata only: no checkpoint bytes, scaler values, targets or forecasts.
    cv_rows=[]
    for directory,history_filter in [(OLD,7),(NEW,None)]:
        runs=pd.read_csv(directory/"runs.csv")
        runs=runs.loc[runs.variant.eq("relative_window")]
        if history_filter is not None:runs=runs.loc[runs.history_days.eq(history_filter)]
        folds=pd.read_csv(directory/"fold_metrics.csv")
        folds=folds.loc[folds.variant.eq("relative_window")]
        if history_filter is not None:folds=folds.loc[folds.history_days.eq(history_filter)]
        for row in folds.to_dict("records"):
            match=runs.loc[runs.model.eq(row["model"]) & runs.seed.eq(row["seed"]) & runs.history_days.eq(row["history_days"])]
            if len(match)!=1:raise ValueError("Each CV segment must identify a single trajectory")
            config_path=directory/"runs"/match.run_id.iloc[0]/row["fold_id"]/"configuration.json"
            config=read(config_path)
            cv_rows.append({"model":row["model"],"history_days":int(row["history_days"]),"trajectory_seed":int(row["seed"]),
                "fold_month":row["fold_id"],"width":int(config["width"]),"learning_rate":config["learning_rate"],
                "selected_epochs":int(config["best_epoch"]),"parameters":int(row["parameters"]),
                "inner_initialization_seed":int(config["inner_initialization_seed"]),
                "refit_initialization_seed":int(config["refit_initialization_seed"]),
                "optimizer":"AdamW","batch_size":32,"dropout":.1,"weight_decay":.0001,
                "configuration_sha256":sha(config_path),"full_configuration_delivery":"private verified artifact bundle"})
    if len(cv_rows)!=1440:raise ValueError("Expected1440 primary CV model-month segments")
    cv=pd.DataFrame(cv_rows).sort_values(["history_days","model","trajectory_seed","fold_month"])
    if cv.duplicated(["model","history_days","trajectory_seed","fold_month"]).any():raise ValueError("Duplicate CV configuration")
    cv.to_csv(DEST/"cv_configuration_catalog.csv",index=False,lineterminator="\n")
    final_rows=[]
    for job in freeze["final_jobs"]:
        directory=NEW/"final_fit"/f"D{job['history_days']}_{job['model']}_s{job['seed']}"
        config=read(directory/"configuration.json")
        final_rows.append({"model":job["model"],"history_days":job["history_days"],"seed":job["seed"],
            "width":job["width"],"learning_rate":job["learning_rate"],"epochs":job["epochs"],"parameters":job["parameters"],
            "initialization_seed":config["initialization_seed"],"train_samples":config["train_samples"],
            "train_first_day_utc":config["train_first_day_utc"],"train_last_day_utc":config["train_last_day_utc"],
            "optimizer":"AdamW","batch_size":32,"dropout":.1,"weight_decay":.0001,
            "configuration_sha256":sha(directory/"configuration.json"),"full_configuration_delivery":"private verified artifact bundle"})
    if len(final_rows)!=120:raise ValueError("Expected120 final closed configurations")
    pd.DataFrame(final_rows).to_csv(DEST/"final_configuration_catalog.csv",index=False,lineterminator="\n")
    for name in ["requirements-book.lock.txt","requirements-cv.txt","requirements-training.lock.txt","requirements-eda.lock.txt"]:
        (DEST/"dependencies").mkdir(exist_ok=True)
        shutil.copyfile(ROOT/name,DEST/"dependencies"/name)
        inputs[name]=sha(ROOT/name)
    products=[
        ("8.1","Informe en Jupyter Book","Informe entregado","book/intro.md","Capítulos1–8 y referencias; contenido y evidencias disponibles"),
        ("8.2","Código completo versionado","Código y réplica privada verificados","BUILD.md","La réplica verifica evidencia guardada; no se afirma repetir todos los ajustes en otra máquina"),
        ("8.3","Dataset o instrucciones y scripts","Adquisición y trazabilidad documentadas","data/README.md","Brutos y derivados completos se conservan localmente; publicación de resúmenes"),
        ("8.4","Síntesis y flujo de literatura","Tabla y flujo entregados; respaldo aplicado parcial","references/literature_e1/study_synthesis_six_columns_20261103.csv","MLP independiente y CNN1D simple:0/5 textos completos del corpus; candidatos de código no son nuevas inclusiones"),
        ("8.5","Comparación, baseline e inferencia","Resultados entregados con límites","book/sections/05_benchmark.md","Comparación final suplementaria post-hoc; baselineMLP/D7 no reseleccionado"),
        ("8.6","Dependencias, semillas y configuraciones","Metadatos y catálogos entregados","book/_static/e1_deliverables_20261104/final_configuration_catalog.csv","Pesos y escaladores completos en paquete privado; no identidad universal entre versiones/hardware"),
        ("8.7","Declaración previa de métrica, margen, independencia y partición","Declaración versionada con cronología explícita","book/_static/e1_deliverables_20261104/declaration_and_chronology.json","Registro anterior al test, posterior al EDA; criterio académico avanzado posterior al test y solo futuro")]
    rows=[dict(zip(["item","product","status","evidence","limit"],row)) for row in products]
    with (DEST/"delivery_manifest.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    write_json(DEST/"delivery_manifest.json",{"compiled_at_bogota":datetime.now(ZoneInfo("America/Bogota")).isoformat(),
        "guide":"E1, página33, siete productos; producto documentado no equivale a cierre científico de todos sus requisitos",
        "new_project_fits":0,"new_forecasts":0,"raw_market_archive_reads":0,"products":rows,
        "CV_configuration_records":1440,"final_configuration_records":120,
        "publication_scope":"Configuration metadata, dependencies, declarations and aggregate results only; no weights, scaler values, market series or individual forecasts"})
    write_json(DEST/"public_manifest.json",{"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "script_sha256":sha(__file__),"source_sha256":inputs,
        "files_sha256":{p.relative_to(DEST).as_posix():sha(p) for p in DEST.rglob("*") if p.is_file() and p.name!="public_manifest.json"}})
    print(json.dumps({"delivery_items":7,"CV_configuration_records":1440,"final_configuration_records":120,
                      "new_project_fits":0,"new_forecasts":0,"raw_archive_reads":0}))


if __name__=="__main__":main()
