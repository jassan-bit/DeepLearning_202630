"""Independent standard-library QA of archived evidence, without model execution.

AI assistance declared. This checker reads existing CSV/JSON/source files only;
it does not import model code, train, infer, recalibrate or resample.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    if not path.is_file() and path.with_suffix(path.suffix + ".gz").is_file():
        with gzip.open(path.with_suffix(path.suffix + ".gz"), "rt", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load(path):
    return json.loads(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify(delivery, repository_source=None, publication_root=None, book_root=None, book_base=None):
    result = {"status": "passed_scoped_independent_AI_evidence_review",
              "scope": "Archived sources, arithmetic, chronology scopes and editorial contracts; not R1/R3 accreditation",
              "new_forecasting_model_fits": 0, "new_market_inference": 0,
              "new_statistical_resamples": 0, "R1_accredited": False, "R3_accredited": False}
    pilot = delivery / "pilot"
    archive = pilot / "archive"
    payload = archive / "payload"
    objects = load(pilot / "source_objects.json")
    manifest = load(archive / "snapshot_manifest.json")
    assert objects["results_commit"] == "8eaa581bc34b615a8507c9e384a3e4be29084720"
    for name, source in objects["files"].items():
        data = (archive / name).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == source["git_blob_sha"], name
        if name.startswith("payload/"):
            reference = manifest["files"][name.removeprefix("payload/")]
            assert len(data) == reference["bytes"]
            assert hashlib.sha256(data).hexdigest() == reference["sha256"]
    receipt = load(payload / "pilot_receipt.json")
    plan = load(payload / "plan.json")
    assert hashlib.sha256(canonical(plan)).hexdigest() == receipt["plan_sha256"]
    assert plan["identity"]["source_commit"] == "8f65eac34aaf49293924ae8328c235cedbdd2866"
    assert (plan["fit_rows"], plan["selection_rows"], plan["purged_rows"], plan["assessment_rows"]) == (532,54,7,55)
    assert receipt["completed_fits"] == receipt["required_fits"] == 30
    assert receipt["market_test_arrays_opened"] == receipt["market_calibration_arrays_opened"] == 0
    assert receipt["additional_datasets_used"] == 0
    phases = receipt["fit_resource_receipts"]
    assert len(phases) == 30
    assert all(r["status"] == "completed" and r["optimizer_steps"] > 0 and not r["untrained_checkpoint_accepted"] for r in phases.values())
    assert max(r["deadline_overrun_seconds"] for r in phases.values()) == 0
    for name, expected in receipt["artifact_sha256"].items():
        assert fingerprint(payload / name) == expected
    groups = {}
    for row in records(payload / "assessment_predictions.csv"):
        groups.setdefault((row["name"], row["seed"]), []).append(row)
    assert len(groups) == 17 and sum(map(len, groups.values())) == 935
    common_dates = None
    individual = {}
    target_by_date = {}
    for key, rows in groups.items():
        rows.sort(key=lambda row: row["date"])
        dates = [row["date"] for row in rows]
        assert len(rows) == len(set(dates)) == 55
        common_dates = dates if common_dates is None else common_dates
        assert dates == common_dates
        for row in rows:
            actual = float(row["target"])
            assert math.isfinite(actual) and math.isfinite(float(row["prediction"]))
            assert row["date"] not in target_by_date or actual == target_by_date[row["date"]]
            target_by_date[row["date"]] = actual
        errors = [float(row["prediction"]) - float(row["target"]) for row in rows]
        individual[key] = math.sqrt(math.fsum(e*e for e in errors)/len(errors))
    recipes = {name: math.fsum(score for (model,_),score in individual.items() if model == name)/sum(model==name for model,_ in individual)
               for name in {model for model,_ in individual}}
    assert len(recipes) == 7
    saved = {r["name"]:float(r["mean_RMSE"]) for r in records(payload / "recipe_summary.csv")}
    for name, score in saved.items():
        assert math.isclose(recipes[name],score,rel_tol=1e-12,abs_tol=1e-15)
    best = min((name for name in recipes if name != "atom_crps"), key=recipes.get)
    gain = 1-recipes["atom_crps"]/recipes[best]
    assert best == "atom_mse_only" and math.isclose(gain,-0.02704311988801411,rel_tol=1e-12)
    assert gain < plan["heuristic"]["minimum_relative_RMSE_gain"] == .02
    assert receipt["continuation_heuristic_passed"] is False
    assert not any(receipt[k] for k in ("p_values_computed","R1_accredited","R3_accredited","confirmatory_claim_allowed"))
    ordered = sorted(recipes,key=recipes.get)
    assert ordered.index("atom_crps")+1 == 5
    assert recipes["atom_crps"] > recipes["floor_plus_mean_excess"]
    floor = {r["date"]:float(r["prediction"]) for (name,_),rows in groups.items() if name=="floor_only" for r in rows}
    assert all(float(r["prediction"])+1e-12 >= floor[r["date"]] for rows in groups.values() for r in rows)
    result["pilot"] = {"source_objects_verified":len(objects["files"]),"artifact_hashes_verified":len(receipt["artifact_sha256"]),
                       "completed_original_fits":30,"trajectories":17,"rows":935,"dates":55,
                       "candidate_RMSE_rank_of_7":5,"candidate_worse_than_best_control_percent":-100*gain,
                       "candidate_worse_than_floor_plus_mean_percent":100*(recipes["atom_crps"]/recipes["floor_plus_mean_excess"]-1),
                       "original_heuristic_unchanged":True}
    adequacy = delivery / "adequacy"
    inputs = load(adequacy / "archive/input_manifest.json")
    for name, expected in inputs["files_sha256"].items():
        assert fingerprint(adequacy / "archive" / name) == expected
    audit = load(adequacy / "results/adequacy_receipt.json")
    assert audit["input_manifest_sha256"] == fingerprint(adequacy / "archive/input_manifest.json")
    assert audit["source_provenance"] == inputs["provenance"]
    assert audit["script_sha256"] == fingerprint(adequacy / "audit_existing_predictions.py")
    for name, expected in audit["output_sha256"].items():
        assert fingerprint(adequacy / "results" / name) == expected
    assert audit["source_provenance"]["historical_residual_predictions"]["commit"] == "510c3bddfdc9926322e3210ac61ee1f5eeffad2e"
    assert audit["configuration"]["formal_adequacy_pass_fail"] is False
    assert not any(audit[k] for k in ("R1_accredited","R3_accredited","new_forecast_model_fits","new_forecast_model_inference","new_statistical_resamples"))
    historic = records(adequacy / "archive/historical_e3_predictions.csv")
    assert len(historic) == 6080 and len({row["seed"] for row in historic}) == 10
    assert len({row["date"] for row in historic}) == 608
    resources = records(adequacy / "archive/resources_supplied.csv")
    job_keys = {(r["dataset"],r["model"],r["seed"],r["kind"]) for r in resources}
    assert len(resources) == 900 and len(job_keys) == 796
    unknown = sum(r["unclosed_crash_compute_unknown"] == "True" for r in resources)
    assert unknown == 3
    result["adequacy"] = {"inputs_verified":len(inputs["files_sha256"]),"output_hashes_verified":len(audit["output_sha256"]),
                          "historical_prediction_rows":6080,"historical_dates":608,"historical_seeds":10,
                          "extended_resource_phase_rows":900,"extended_unique_jobs":796,"unclosed_crash_compute_unknown_rows":unknown,
                          "maximum_deadline_overrun_includes_downtime_seconds":max(float(r["deadline_overrun_seconds"] or 0) for r in resources),
                          "cohorts_kept_separate":True,"no_formal_adequacy_or_prospective_power_claim":True}
    panels = inputs["provenance"].get("extended_candidate_residual_panels",{})
    if panels:
        assert set(panels) == {"D00","D01","D02","D03"}
        source_metrics = {(r["dataset"],r["seed"]):r for r in records(adequacy/"archive/per_seed_candidate_clean.csv") if r["split"]=="test"}
        total = 0
        for dataset, source in panels.items():
            assert source["result_commit"] == "16a4504f6fb9208c7d6f89b65017194fa20470da"
            assert source["filter"] == {"scenario":"unperturbed","split":"test"}
            with gzip.open(adequacy/"archive"/source["archive_file"],"rb") as stream:
                assert hashlib.sha256(stream.read()).hexdigest() == source["extracted_uncompressed_sha256"]
            panel = records(adequacy/"archive"/source["archive_file"])
            assert len(panel) == source["extracted_rows"]
            seeds = {r["seed"] for r in panel}
            assert len(seeds) == 10
            for seed in seeds:
                rows = sorted((r for r in panel if r["seed"]==seed),key=lambda r:r["date"])
                assert len(rows) == len({r["date"] for r in rows})
                assert all(r["dataset"] == dataset and float(r["prediction_sigma7"])+1e-12>=float(r["known_floor"]) for r in rows)
                errors = [float(r["prediction_sigma7"])-float(r["target_sigma7"]) for r in rows]
                rmse = math.sqrt(math.fsum(e*e for e in errors)/len(errors))
                mae = math.fsum(abs(e) for e in errors)/len(errors)
                assert math.isclose(rmse,float(source_metrics[dataset,seed]["RMSE"]),rel_tol=1e-12,abs_tol=1e-15)
                assert math.isclose(mae,float(source_metrics[dataset,seed]["MAE"]),rel_tol=1e-12,abs_tol=1e-15)
            total += len(panel)
        assert total == 18080
        historical_tests = records(adequacy/"results/historical_residual_diagnostics.csv")
        extended_tests = records(adequacy/"results/extended_candidate_residual_diagnostics.csv")
        for family, adjusted_key, size in [(historical_tests,"Holm40_asymptotic_diagnostic_p",40),(extended_tests,"Holm160_asymptotic_diagnostic_p",160)]:
            assert len(family)==size
            previous=0.
            for rank,row in enumerate(sorted(family,key=lambda r:float(r["raw_asymptotic_p"]))):
                previous=max(previous,min(1.,(size-rank)*float(row["raw_asymptotic_p"])))
                assert math.isclose(previous,float(row[adjusted_key]),rel_tol=1e-12,abs_tol=1e-15)
                assert row["R3_superiority_test"] == "False" and row["formal_adequacy_pass_fail"] == ""
        result["adequacy"].update(extended_candidate_clean_test_trajectories=40,extended_saved_prediction_rows=18080,
                                  per_seed_metrics_independently_recomputed=80,
                                  diagnostic_family_sizes=[40,160],separate_Holm_families_recomputed=True,
                                  global_200_diagnostic_FWER_not_claimed=True)
    contribution = load(delivery / "contribution/contribution_declaration.json")
    assert not any(contribution[k] for k in ("R1_accredited","R2_student_defense_observed","R3_accredited","novelty_absence_proved","bad_pilot_refutes_novelty","new_model_fits_executed_by_this_consolidation","new_model_inference","old_796_jobs_relabelled_as_V3","old_packets_modified"))
    assert contribution["AI_assistance_declared"] is True
    result["contribution"]={"candidate_not_promoted":True,"pilot_failure_not_novelty_refutation":True,"AI_substitution_kept":True}
    inventory = delivery / "inventory"
    source_manifest = load(inventory / "source_manifest.json")
    with zipfile.ZipFile(inventory / "sources.zip") as zipped:
        assert set(zipped.namelist()) == {r["archive_name"] for r in source_manifest["sources"]} | {"source_manifest.json"}
        assert zipped.read("source_manifest.json") == (inventory / "source_manifest.json").read_bytes()
        for source in source_manifest["sources"]:
            data = zipped.read(source["archive_name"])
            assert len(data) == source["bytes"]
            assert hashlib.sha256(data).hexdigest() == source["sha256"]
            assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() == source["git_blob_sha"]
    designs = records(inventory / "design_inventory.csv")
    neural = [r for r in designs if r["category"] == "neural"]
    assert len(designs) == 22 and len(neural) == 15
    assert sum(r["evidence_state"] == "market_fitted" for r in neural) == 12
    assert sum(r["evidence_state"] == "synthetic_only" for r in neural) == 3
    selections = records(inventory / "selection_opportunities.csv")
    assert len(selections) == 183
    assert sum(r["scope"] == "extended_audit_796" for r in selections) == 168
    ledger = records(inventory / "attempt_ledger.csv")
    assert len(ledger) == len({r["job_id"] for r in ledger}) == 844
    assert sum(r["job_type"] == "neural_fit" for r in ledger) == 824
    assert sum(r["job_type"] == "ARIMA_fit" for r in ledger) == 4
    assert sum(r["job_type"] == "no_optimizer_control" for r in ledger) == 16
    reconciled = load(inventory / "reconciliation.json")
    assert reconciled["preregistration_claimed"] is False
    assert reconciled["Nd_model"]["global_all_thoughts_count_certified"] is False
    assert reconciled["Nd_selection"]["single_all_selection_opportunities_count_certified"] is False
    assert reconciled["Nd_model"]["documented_entry_interval_if_that_sketch_is_distinct"] == [22,23]
    result["inventory"] = {"source_records_verified":len(source_manifest["sources"]),"implemented_neural_recipes":15,
                           "market_fitted_neural_recipes":12,"synthetic_only_neural_recipes":3,
                           "documented_entries":22,"unresolved_sketch_may_add_one":True,
                           "HPO_trial_jobs_all_declared_models":183,"completed_job_records":844,
                           "neural_fit_jobs":824,"ARIMA_fit_jobs":4,"no_optimizer_jobs":16,
                           "no_exhaustive_informal_design_or_physical_restart_count_claimed":True}
    dashboard = load(delivery / "dashboard/verification.json")
    edited_directory = delivery / "dashboard/edited"
    if not edited_directory.is_dir():
        edited_directory = publication_root or delivery.parents[1]
    for name, expected in dashboard["edited_files_sha256"].items():
        assert fingerprint(edited_directory / name) == expected
    if repository_source is not None:
        base = load(delivery / "dashboard/base_verification.json")
        for row in base["files"]:
            data=(repository_source/row["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest()==row["base_sha256"]
            assert hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==row["base_git_blob_sha"]
        for name, expected in dashboard["unchanged_artifact_bundle_sha256"].items():
            assert fingerprint(repository_source/name)==expected
        for label in dashboard["unchanged_callback_and_helpers"]:
            name,function = label.split(":",1)
            before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((repository_source/name).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((edited_directory/name).read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            assert before[function]==after[function],label
    result["dashboard"]={"edited_file_hashes_verified":5,"runtime_and_artifact_proofs_rechecked":repository_source is not None,
                         "callbacks_helpers":21,"artifact_bundle_files":27}
    release = load(delivery / "release/network_observations.json")
    assert release["render_public_smoke_passed"] is False and release["zenodo_published_verified"] is False
    assert all("Tunnel connection failed: 403" in r["error"] for r in release["observations"])
    result["network"]={"proxy_block_distinguished_from_origin_failure":True,"public_Render_or_DOI_success_not_fabricated":True}
    if book_root is not None:
        labels = {"sections/06_modelo_original.md":"(e3-contribution-evidence-close-20261120)=",
                  "sections/07_diseno_riguroso.md":"(e3-design-inventory-reconciled)=",
                  "sections/08_verificacion_estadistica.md":"(e3-v3-pilot-completed-20261120)="}
        for name,label in labels.items():
            assert label in (book_root/name).read_text(),name
        assert "(e3-existing-adequacy-close-20261120)=" in (book_root/"sections/08_verificacion_estadistica.md").read_text()
        names = ["intro.md","sections/06_modelo_original.md","sections/07_diseno_riguroso.md",
                 "sections/08_verificacion_estadistica.md","sections/09_dashboard_predictivo.md",
                 "sections/10_despliegue_render.md","sections/11_informe_tecnico.md",
                 "sections/12_lista_entregables.md","sections/14_evaluacion_calificacion.md",
                 "sections/15_integridad_academica.md",
                 "sections/16_conclusiones.md"]
        result["book_review"] = {"manual_scientific_scope_review":"completed_before_publication",
                                  "reviewed_files_sha256":{name:fingerprint(book_root/name) for name in names},
                                  "new_evidence_labels_verified":4,
                                  "build_and_live_deployment_are_separate_publisher_checks":True}
        if book_base is not None:
            protected = 0
            for source in (book_base/"_static").rglob("*"):
                if source.is_file():
                    assert source.read_bytes() == (book_root/"_static"/source.relative_to(book_base/"_static")).read_bytes()
                    protected += 1
            result["book_review"]["preexisting_static_files_byte_identical"] = protected
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root","--delivery",dest="delivery",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-source","--repository-source",dest="repository_source",type=Path,
                        help="Optional unmodified baseline checkout of commit29f9a26; not the newly edited publication checkout")
    parser.add_argument("--repo-root",type=Path,help="Publication checkout containing the five edited dashboard root files")
    parser.add_argument("--book-root",type=Path)
    parser.add_argument("--book-base",type=Path,help="Optional baseline book to compare every existing static download")
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    result=verify(args.delivery,args.repository_source,args.repo_root,args.book_root,args.book_base)
    if args.receipt:
        args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,sort_keys=True,allow_nan=False))
