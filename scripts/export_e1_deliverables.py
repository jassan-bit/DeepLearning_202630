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
D7_STUDY="e1_filipe_confirmatory_d7_20261105"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def registered_d7_delivery():
    """Read only the completed study's verified public aggregate allowlist."""
    directory=ROOT/"book/_static"/D7_STUDY
    if not directory.exists():
        return None
    manifest_path=directory/"public_manifest.json"
    if directory.is_symlink() or manifest_path.is_symlink():
        raise ValueError("Expected a regular registered D7 public export")
    manifest=read(manifest_path)
    if manifest.get("study_id")!=D7_STUDY:
        raise ValueError("Wrong registered D7 public manifest")
    private_names={"predictions.csv","preprocessing.json","splits.json",
                   "history_inner.csv","history_refit.csv"}
    inputs={manifest_path.relative_to(ROOT).as_posix():sha(manifest_path)}
    for name,expected in manifest["files_sha256"].items():
        relative=Path(name)
        path=directory/relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())
                or relative.name in private_names
                or relative.suffix.lower() not in {".json",".csv",".md",".png",".svg"}):
            raise ValueError("Unexpected private or unsafe D7 public artifact: "+name)
        if sha(path)!=expected:
            raise ValueError("Changed registered D7 public aggregate: "+name)
        inputs[path.relative_to(ROOT).as_posix()]=expected

    def public(name):
        if name not in manifest["files_sha256"]:
            raise ValueError("Missing registered D7 public record: "+name)
        return read(directory/name)

    plan=public("protocol.json")
    lock=public("locked_design.json")
    overview=public("execution_overview.json")
    analysis=public("analysis/summary.json")
    audit=public("verification/revision_filipe_confirmatory_d7_20261105.json")
    phases={phase:public(phase+"/execution_summary.json")
            for phase in ("variance_extension","definitive")}
    n=lock["common_R"]
    if (type(n) is not int or n<30 or plan["study_id"]!=D7_STUDY
            or analysis["status"]!="definitive_conditional_seed_analysis_complete"
            or analysis["R_fixed_before_definitive"]!=n
            or analysis["fresh_definitive_trajectories"]!=3*n
            or analysis["primary_comparisons"]!=3
            or audit.get("status")!="PASS" or audit.get("phase")!="definitive"):
        raise ValueError("The registered D7 study must be complete and independently verified")
    if (lock["all25_pilot_seeds_excluded_from_definitive"] is not True
            or any(analysis[key] is not False for key in
                ["pilot_seeds_in_definitive","test2025_2026_read","new_market_generalization_claim",
                 "historical_inference_relabelled_confirmatory","advanced_gain_boundary_tested",
                 "achieved_or_joint_power_claim"])
            or overview["variance_extension"]!={"seeds_per_model":20,"trajectories":60,"fits":1440}
            or overview["definitive"]!={"seeds_per_model":n,"trajectories":3*n,"fits":72*n}
            or any(overview[key] is not False for key in
                ["new_temporal_evaluation","old_benchmark_repaired","original_test_reopened","achieved_power_claim"])):
        raise ValueError("Preserve the registered D7 inferential and data-access boundaries")
    for phase,seeds in [("variance_extension",20),("definitive",n)]:
        execution=phases[phase]
        if (execution["seeds_per_model"]!=seeds or execution["trajectories"]!=3*seeds
                or execution["monthly_segments"]!=36*seeds
                or execution["project_fits"]!=72*seeds
                or execution["test_files_opened"]!=0 or execution["raw_archives_opened"]!=0
                or execution["publication_precedes_every_trajectory"] is not True):
            raise ValueError("Incomplete or out-of-scope registered D7 phase: "+phase)
    if "configuration_catalog.csv" not in manifest["files_sha256"]:
        raise ValueError("Missing public registered D7 configuration catalogue")
    with (directory/"configuration_catalog.csv").open(newline="") as stream:
        configs=list(csv.DictReader(stream))
    keys=[(r["phase"],r["model"],r["trajectory_seed"],r["fold_month"]) for r in configs]
    counts={phase:sum(r["phase"]==phase for r in configs) for phase in phases}
    if (len(keys)!=len(set(keys)) or len(configs)!=(60+3*n)*12
            or counts!={"variance_extension":720,"definitive":36*n}
            or any(int(r["history_days"])!=7 for r in configs)
            or overview["configuration_records"]!=len(configs)):
        raise ValueError("Wrong public registered D7 configuration catalogue")
    events=[]
    for label,name,key in [
        ("Registro del estudio D7 condicional", "protocol.json", "registered_at_utc"),
        ("Protocolo D7 publicado antes de la ampliación", "preregistration_git_receipt.json", "published_at_utc"),
        ("Primera trayectoria de ampliación D7", "variance_extension/execution_summary.json", "first_new_trajectory_started_at_utc"),
        ("Ampliación D7 completada", "variance_extension/execution_summary.json", "completed_at_utc"),
        ("R D7 definitivo fijado", "locked_design.json", "locked_at_utc"),
        ("R D7 publicado antes de las corridas definitivas", "locked_design_publication_receipt.json", "published_at_utc"),
        ("Primera trayectoria definitiva D7", "definitive/execution_summary.json", "first_new_trajectory_started_at_utc"),
        ("Definitiva D7 completada", "definitive/execution_summary.json", "completed_at_utc"),
        ("Análisis definitivo D7", "analysis/summary.json", "analyzed_at_utc"),
        ("Auditoría independiente definitiva D7", "verification/revision_filipe_confirmatory_d7_20261105.json", "audited_at_utc")]:
        path=directory/name
        value=public(name)[key]
        events.append({"event":label,"timestamp_utc":value,
            "timestamp_bogota":datetime.fromisoformat(value).astimezone(ZoneInfo("America/Bogota")).isoformat(),
            "source":path.relative_to(ROOT).as_posix(),"source_sha256":sha(path)})
    summary={"study_id":D7_STUDY,"scope":plan["scope"],
        "public_manifest":manifest_path.relative_to(ROOT).as_posix(),
        "public_manifest_sha256":sha(manifest_path),"common_R":n,
        "variance_extension":{"seeds_per_model":20,"trajectories":60,"project_fits":1440,
                              "configuration_records":720},
        "definitive":{"seeds_per_model":n,"trajectories":3*n,"project_fits":72*n,
                      "configuration_records":36*n},
        "configuration_records":len(configs),"primary_comparisons":3,
        "all25_pilot_seeds_excluded_from_definitive":True,
        "definitive_repetitions_performed":True,"independent_verification":"PASS",
        "analysis_status":analysis["status"],"new_independent_market_evaluation":False,
        "historical_benchmark_repaired":False,"original_test_reopened":False,
        "achieved_or_joint_power_claim":False,"advanced_gain_boundary_tested":False}
    return {"summary":summary,"chronology":events,"source_sha256":inputs}


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
    pilot_path=ROOT/"experiments/e1_filipe_pilot_20261105/protocol.json"
    pilot_execution_path=ROOT/"results/e1_filipe_pilot_20261105/execution_summary.json"
    pilot_analysis_path=ROOT/"results/e1_filipe_pilot_20261105/analysis/summary.json"
    pilot,pilot_execution,pilot_analysis=map(read,[pilot_path,pilot_execution_path,pilot_analysis_path])
    scope_path=ROOT/"references/audits/statistical_scope_e1_20261105.json"
    scope=read(scope_path)
    inputs={}
    for path in [freeze_path,plan_path,policy_path,initial_path,audit_path,supplement_path,
                 ROOT/"references/decisiones_e1_20261102.json",
                 ROOT/"references/literature_e1/recovery_20261105/selection_update.json",
                 ROOT/"references/literature_e1/recovery_20261105/primary_source.json",
                 pilot_path,pilot_execution_path,pilot_analysis_path,scope_path]:
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
    d7=registered_d7_delivery()
    if d7:
        entries.extend(d7["chronology"])
        inputs.update(d7["source_sha256"])
    literature_path=DEST/"literature_extension.json"
    literature=read(literature_path) if literature_path.exists() else None
    if literature:
        if literature["bibliographic_references"]!=39 or literature["BTC_full_text_studies"]!=6:
            raise ValueError("Preserve the39-reference literature extension and six-text BTC corpus")
        for name,expected in literature["source_sha256"].items():
            if sha(ROOT/name)!=expected:
                raise ValueError("Changed literature delivery source: "+name)
        inputs.update(literature["source_sha256"])
    for label,path,key in [
        ("Registro del nuevo piloto de aleatoriedad; calendario2024 ya conocido",pilot_path,"registered_at_utc"),
        ("Primera trayectoria nueva del piloto",pilot_execution_path,"first_new_trajectory_started_at_utc"),
        ("Análisis del piloto bajo código registrado antes de sus ajustes",pilot_analysis_path,"analyzed_at_utc")]:
        value=read(path)[key]
        entries.append({"event":label,"timestamp_utc":value,
                        "timestamp_bogota":datetime.fromisoformat(value).astimezone(ZoneInfo("America/Bogota")).isoformat(),
                        "source":path.relative_to(ROOT).as_posix(),"source_sha256":sha(path)})
    entries.sort(key=lambda event:datetime.fromisoformat(event["timestamp_utc"]))
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
        "new_registered_pilot":{"scope":"Prospective only for new training randomizations on the known2024 calendar; not a repair of the old benchmark",
            "new_trajectories":pilot_execution["trajectories"],"seeds_per_model_history":5,
            "pilot_seeds":pilot["seeds"],"definitive_new_seeds_not_executed":True,
            "common_point_SD_R":pilot_analysis["common_point_SD_R"],
            "common_recommended_upper_SD_R":pilot_analysis["common_recommended_upper_SD_R"],
            "achieved_power_claim":False,"new_independent_market_evaluation":False},
        "statistical_scope":{"assessment_source":scope_path.relative_to(ROOT).as_posix(),
            "assessment_status":scope["status"],
            "future_confirmation_status":scope["future_confirmation"]["status"],
            "new_evaluation_cohort":None,"advanced_model":None,
            "historical_supplement_is_confirmatory":False,
            "baseline_pilot_powers_advanced_claim":False},
        "limits":["No claim of registration before EDA or before the historical market periods",
                  "The final supplement is post-hoc, not prospective confirmation",
                  "The future academic criterion was not predeclared for the completed test",
                  "The new baseline pilot precedes only its own fits; definitive repetitions and an independent future market evaluation remain unperformed"],
        "inherited_next_delivery":"Keep source, target, metric and dependence definitions explicit; any new model designed from this test needs another independent evaluation.",
        "source_sha256":inputs}
    if literature:
        declaration["literature_extension"]=literature
    if d7:
        declaration["registered_d7_study"]=d7["summary"]
        declaration["new_registered_pilot"]["status_scope"]="Historical four-history, twelve-pair pilot design; subsequent D7 study is recorded separately"
        declaration["limits"][-1]="The original four-history pilot and its R302/1680 planning remain historical; the separately registered D7 definitive comparison is conditional on the same known2024 calendar, without independent market evaluation"
    write_json(DEST/"declaration_and_chronology.json",declaration)
    public_scope={**scope,
        "compiled_at_utc":datetime.now(timezone.utc).isoformat(),
        "assessment_source_sha256":sha(scope_path),
        "source_sha256":{name:sha(ROOT/name) for name in scope["source_records"]}}
    if d7:
        public_scope["registered_d7_study"]=d7["summary"]
        public_scope["historical_assessment_preserved"]=True
    write_json(DEST/"statistical_scope.json",public_scope)
    scope_lines=["| Evidencia | Momento y unidad | Afirmación respaldada | Límite |",
                 "|---|---|---|---|"]
    for row in scope["evidence_rows"]:
        scope_lines.append("| "+" | ".join(row[key] for key in
            ["evidence","timing_scope","supported_claim","unsupported_claim"])+" |")
    if d7 and not any(row["evidence"].startswith("Estudio D7") for row in scope["evidence_rows"]):
        scope_lines.append("| Estudio D7 registrado | Semillas nuevas; calendario2024 conocido; R="+str(d7["summary"]["common_R"])+" fijado y publicado antes de la definitiva | Tres contrastes entre recetas fijas, condicionales a la aleatoriedad de entrenamiento | No nueva evaluación de mercado, reparación histórica, potencia lograda o mejora avanzada >5% |")
    (DEST/"statistical_scope.md").write_text("\n".join(scope_lines)+"\n")

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
        ("8.4","Síntesis y flujo de literatura","39 referencias; seis textos BTC y respaldo metodológico con niveles explícitos","references/literature_e1/journal_update_20261105/methodological_sources.json","T1 original de revista FNN, T2 código autor Conv1D, T3 identidad editorial con original pendiente; no se añaden estudios BTC. Dimensión CNN de S8, recetas exactas, recurrencia amplia e indexación individual conservan límites"),
        ("8.5","Comparación, baseline e inferencia","Resultados y piloto nuevo entregados con límites","book/sections/05_benchmark.md","Benchmark histórico conservado; nuevo piloto dimensiona futuras semillas, sin potencia alcanzada ni réplica de mercado"),
        ("8.6","Dependencias, semillas y configuraciones","Metadatos originales y 720 configuraciones piloto entregados","book/_static/e1_filipe_pilot_20261105/configuration_catalog.csv","Pesos y escaladores completos en paquete privado; no identidad universal entre versiones/hardware"),
        ("8.7","Declaración previa de métrica, margen, independencia y partición","Cronología original y registro anterior al nuevo piloto","book/_static/e1_deliverables_20261104/declaration_and_chronology.json","Piloto registrado antes de sus ajustes; no previo al benchmark histórico, a los datos de2024 ni al test ya examinado")]
    rows=[dict(zip(["item","product","status","evidence","limit"],row)) for row in products]
    if d7:
        updates={
            "8.5":("Benchmark histórico y estudio D7 registrado entregados con alcances separados",
                   "book/_static/"+D7_STUDY+"/analysis/summary.json",
                   "Definitiva con R="+str(d7["summary"]["common_R"])+" semillas nuevas por modelo; tres contrastes predefinidos sobre2024 conocido, sin generalización temporal nueva ni potencia lograda"),
            "8.6":("Catálogos históricos y configuraciones de ambas fases D7 entregados",
                   "book/_static/"+D7_STUDY+"/configuration_catalog.csv",
                   "Metadatos por fase; pesos, escaladores y predicciones individuales privados"),
            "8.7":("Cronología histórica y dos publicaciones previas de D7 documentadas",
                   "book/_static/"+D7_STUDY+"/tables/chronology.md",
                   "Protocolo antes de ampliación y R antes de definitiva; calendario previamente conocido, sin reescribir el benchmark")}
        for row in rows:
            if row["item"] in updates:
                row["status"],row["evidence"],row["limit"]=updates[row["item"]]
    with (DEST/"delivery_manifest.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    delivery={"compiled_at_bogota":datetime.now(ZoneInfo("America/Bogota")).isoformat(),
        "guide":"E1, página33, siete productos; producto documentado no equivale a cierre científico de todos sus requisitos",
        "execution_counter_scope":"The zero execution counters below describe this metadata export only; completed training is recorded separately by historical pilot and registered D7 phase",
        "new_project_fits":0,"new_forecasts":0,"raw_market_archive_reads":0,"products":rows,
        "CV_configuration_records":1440,"final_configuration_records":120,
        "literature_review":"39 references; six BTC full texts plus methodological T1 journal full text, T2 author Conv1D code and T3 publication identity only. No increase in BTC model frequencies, source S8 CNN dimensionality or individual index verification claims",
        "registered_pilot":{"new_trajectories":60,"fits":1440,"public_configuration_records":720,
            "common_point_SD_R":pilot_analysis["common_point_SD_R"],
            "common_recommended_upper_SD_R":pilot_analysis["common_recommended_upper_SD_R"],
            "definitive_repetitions_performed":False,"achieved_power_claim":False},
        "publication_scope":"Configuration metadata, dependencies, declarations and aggregate results only; no weights, scaler values, market series or individual forecasts"}
    if literature:
        delivery["literature_extension"]=literature
    if d7:
        delivery["registered_d7_study"]=d7["summary"]
    write_json(DEST/"delivery_manifest.json",delivery)
    write_json(DEST/"public_manifest.json",{"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "script_sha256":sha(__file__),"source_sha256":inputs,
        "files_sha256":{p.relative_to(DEST).as_posix():sha(p) for p in DEST.rglob("*") if p.is_file() and p.name!="public_manifest.json"}})
    print(json.dumps({"delivery_items":7,"CV_configuration_records":1440,"final_configuration_records":120,
                      "new_project_fits":0,"new_forecasts":0,"raw_archive_reads":0}))


if __name__=="__main__":main()
