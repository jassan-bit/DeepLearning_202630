"""Campaign orchestration with mandatory GitHub barriers and one-shot test use.

Imports perform no fitting, acquisition or label access. Planning software
fixtures are explicit and can never qualify for final protocol closure.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import io
import gzip
import hashlib
import math
import os
import shutil
from pathlib import Path
import re
import time
import uuid
import zipfile

import numpy as np
from scipy.stats import beta

from . import common as c, data, models, power, training

VERSION = "e3-r3-controller-1"
SEED_BANK_SHA256 = "137fe45eca358081307c041ab31dd94efb23f533202438e2eb6888e4e79836ee"
PANELS = ("F00", "F01", "F02", "F03")
DEVELOPMENT = ("fit", "selection", "planning", "calibration")


class ControllerError(RuntimeError):
    """Fixed campaign diagnostic, never a credential-bearing HTTP response."""


def _require(condition, code):
    if not condition:
        raise ControllerError(code)


def _spec_file(spec, base):
    _require(isinstance(spec, dict) and isinstance(spec.get("path"), str) and isinstance(spec.get("sha256"), str), "hash_bound_evidence_file_required")
    path = c.safe(spec["path"] if Path(spec["path"]).is_absolute() else Path(base)/spec["path"])
    _require(path.is_file() and c.sha(path) == spec["sha256"], "evidence_file_hash_differs")
    return path


def _sources(source_root, source_commit):
    root = c.safe(source_root)
    _require(isinstance(source_commit, str) and bool(re.fullmatch(r"[0-9a-f]{40}", source_commit)), "immutable_source_commit_required")
    verified = c.verify_sources(root)
    manifest=c.read_json(root/"SOURCE_MANIFEST.json")["files"]
    required=("engine/controller.py","engine/common.py","engine/data.py","engine/models.py","engine/training.py","engine/storage.py","engine/statistics.py",
              "configs/CANDIDATE_IDENTITY.json","configs/FINAL_SEED_BANK.json","vendor/SOURCE_MANIFEST.json")
    _require(all(name in manifest for name in required),"executed_sources_missing_from_manifest")
    executed={"controller.py":Path(__file__),"common.py":Path(c.__file__),"data.py":Path(data.__file__),"models.py":Path(models.__file__),
              "training.py":Path(training.__file__)}
    _require(all(c.sha(root/"engine"/name)==c.sha(path) for name,path in executed.items()),"manifest_does_not_match_executed_sources")
    _require(c.sha(root/"vendor/SOURCE_MANIFEST.json")==models.verify_sources()["manifest_sha256"],"executed_vendor_sources_differ")
    anchor = c.read_json(root/"PUBLIC_SOURCE_ANCHOR.json")
    _require(anchor.get("source_commit") == source_commit and anchor.get("manifest_sha256") == verified["manifest_sha256"], "public_source_anchor_differs")
    return root, verified


def _panel_paths(panel_dirs):
    if isinstance(panel_dirs, (list, tuple)):
        panel_dirs = dict(zip(PANELS, panel_dirs)) if len(panel_dirs) == 4 else {}
    _require(isinstance(panel_dirs, dict) and set(panel_dirs) == set(PANELS), "exactly_four_identified_panels_required")
    return {panel: c.safe(panel_dirs[panel]) for panel in PANELS}


def _partition_metadata(admission):
    return {part: {"sha256": admission["partition_sha256"][part], "count": admission["partition_counts"][part],
                   "dates": admission["partition_date_ranges"][part], "support": admission["partition_support_bounds"][part]}
            for part in DEVELOPMENT if part in admission["partition_paths"]}


def _admissions(panel_dirs, *, allow_fixture=False):
    paths = _panel_paths(panel_dirs)
    records = {}
    for panel, path in paths.items():
        record = data.read_admission(path)
        _require(allow_fixture or record["status"] == "development_admitted", "software_fixture_cannot_admit_real_campaign")
        if record["status"] != "software_fixture":
            data.verify_evidence(record["rights"], "approved")
            data.verify_evidence(record["exposure"], "audited")
        records[panel] = record
    return paths, records



def source_identity_sha256(provenance):
    """Hash only source fields known before the final protocol is closed.

    Raw-file/version hashes arrive after acquisition and are bound separately;
    including them here would create a circular future-data requirement.
    """
    keys=("provider","instrument","calendar")
    _require(isinstance(provenance,dict) and all(isinstance(provenance.get(key),str) and provenance[key] for key in keys),
             "provider_instrument_calendar_source_identity_required")
    return c.digest({key:provenance[key] for key in keys})


def _progress(event,plan,*,job=None,status=None,remote_commit=None):
    """Emit only fixed identity/progress fields, never credentials or arrays."""
    record={"event":event,"case_id":plan["case_id"],"jobs_total":len(plan["jobs"])}
    if job is not None:
        record.update(job_id=job["id"],panel=job["panel"],system=job["model"],seed=job["seed"])
    if status is not None:record["status"]=status
    if remote_commit is not None:record["remote_commit"]=remote_commit
    print(json.dumps(record,ensure_ascii=False,allow_nan=False),flush=True)


def default_stress_plans():
    """Prospective sensor stress definitions; no data-dependent tuning."""
    values={
        "input_noise":{"version":"close-preserving-OHLV-lognoise-1","kind":"ohlv_log_measurement_noise","log_sd":.001,"seed":20261122011,
                       "preserve_close":True,"coherent_OHLC_projection":True,"frozen_before_test":True,
                       "assumption":"multiplicative measurement error on O/H/L/volume; close and true labels/history remain exact; all methods share each disturbed panel"},
        "missingness":{"version":"causal-open-volume-carry-1","kind":"causal_open_volume_carry_forward","rate":.10,"seed":20261122012,
                       "preserve_close":True,"first_row_preserved":True,"frozen_before_test":True,
                       "assumption":"independent missing O/volume entries after first known window row; causal carry-forward and OHLC clipping; close/labels remain exact"}}
    return {key:{**value,"scenario_sha256":c.digest(value)} for key,value in values.items()}


def _stress_plans(protocol):
    plans=protocol.get("robustness_stress_plans")
    _require(isinstance(plans,dict) and set(plans)=={"input_noise","missingness"},"both_frozen_sensor_stress_plans_required")
    for name,spec in plans.items():
        _require(isinstance(spec,dict) and spec.get("scenario_sha256")==c.digest({k:v for k,v in spec.items() if k!="scenario_sha256"})
                 and spec.get("frozen_before_test") is True and spec.get("preserve_close") is True and type(spec.get("seed")) is int,
                 "stress_scenario_identity_or_freeze_invalid")
        if name=="input_noise":
            _require(spec.get("kind")=="ohlv_log_measurement_noise" and type(spec.get("log_sd")) in (int,float)
                     and 0<spec["log_sd"]<=.1 and spec.get("coherent_OHLC_projection") is True,"invalid_noise_sensor_scenario")
        else:
            _require(spec.get("kind")=="causal_open_volume_carry_forward" and type(spec.get("rate")) in (int,float)
                     and 0<spec["rate"]<1 and spec.get("first_row_preserved") is True,"invalid_missingness_sensor_scenario")
    return plans


def stress_partition(partition, spec, panel):
    """Perturb observed OHLV causally; preserve close, labels and true floor."""
    _require(panel in PANELS,"unknown_stress_panel")
    raw=np.asarray(partition["raw"],dtype=float).copy()
    rng=np.random.default_rng(np.random.SeedSequence([spec["seed"],PANELS.index(panel)]))
    if spec["kind"]=="ohlv_log_measurement_noise":
        noise=np.exp(rng.normal(0.,spec["log_sd"],size=(len(raw),7,4)))
        raw[:,:,[0,1,2,4]]*=noise
        raw[:,:,1]=np.maximum.reduce((raw[:,:,1],raw[:,:,0],raw[:,:,3]))
        raw[:,:,2]=np.minimum.reduce((raw[:,:,2],raw[:,:,0],raw[:,:,3]))
    elif spec["kind"]=="causal_open_volume_carry_forward":
        missing=rng.random((len(raw),7,2))<spec["rate"]
        missing[:,0,:]=False
        for step in range(1,7):
            for channel,index in ((0,0),(1,4)):
                raw[:,step,index]=np.where(missing[:,step,channel],raw[:,step-1,index],raw[:,step,index])
        raw[:,:,0]=np.clip(raw[:,:,0],raw[:,:,2],raw[:,:,1])
    else:
        raise ControllerError("unsupported_frozen_stress_transform")
    _require(np.array_equal(raw[:,:,3],np.asarray(partition["raw"])[:,:,3]) and np.isfinite(raw).all(),"stress_changed_close_or_became_nonfinite")
    return {**partition,"raw":raw}


def _method_admission(source_root, spec, settings):
    """Verify actual archived validation files and recompute finite-MC gates.

    A passing report needs the same exact R/T/B/tuning and the complete
    single-panel temporal family. Application to four panels is declared,
    never manufactured by copying a marginal Monte Carlo ledger four times.
    Cross-panel joint power belongs to the separate full-family power study.
    Actual frozen plan/runner/replicate bytes and finite-MC bounds are checked.
    """
    report_path = _spec_file(spec, source_root)
    report = c.read_json(report_path)
    expected = {"status": "PASSED_PREDECLARED_VALIDATION", "admitted":True,"protocol_closure_permitted":True,"family_rivals": 20,
                "validation_scope":"single_panel_temporal_family", "panels_per_replicate":1,"final_panels_application":4,
                "all_three_methods": True, "complete_and_partial_nulls": True,
                "extreme_tail_validated": True, "coverage_validated": True}
    _require(all(report.get(k) == v for k,v in expected.items()), "temporal_methods_not_admitted_for_full_family")
    _require(settings in report.get("validated_settings", []), "temporal_validation_does_not_cover_final_settings")
    statistics_sha=report.get("statistics_source_sha256",report.get("statistics_sha256"))
    _require(statistics_sha == c.sha(Path(source_root)/"engine/statistics.py"), "temporal_validation_statistics_source_differs")
    evidence = report.get("evidence", {})
    files = {key: _spec_file(evidence.get(key), report_path.parent) for key in ("plan", "freeze", "runner", "replicates")}
    plan, freeze = c.read_json(files["plan"]), c.read_json(files["freeze"])
    _require(freeze.get("plan_sha256") == c.sha(files["plan"]) and freeze.get("runner_sha256") == c.sha(files["runner"])
             and freeze.get("statistics_sha256") == statistics_sha and plan.get("frozen_before_execution") is True,
             "temporal_validation_frozen_identity_differs")
    _require(plan.get("n_rivals") == 20 and plan.get("panels_per_replicate") == 1 and plan.get("n_fixed_seeds") == settings["R"]
             and plan.get("n_dates") == settings["T"] and plan.get("bootstrap_reps") == settings["B"]
             and plan.get("block_length") == settings["block_length"] and plan.get("hac_lags") == settings["hac_lags"],
             "temporal_validation_plan_scope_differs")
    _require(all(plan.get(k)==settings[k] for k in ("reference_reps","reference_seed","horizon","margin","alpha") if k in settings),
             "temporal_validation_reference_or_endpoint_differs")
    scenarios = plan.get("scenarios", [])
    _require(any(s.get("null_rivals") == 20 and s.get("boundary_null") == "epa" for s in scenarios)
             and any(0 < s.get("null_rivals", 0) < 20 for s in scenarios)
             and any(s.get("boundary_null") == "practical" for s in scenarios)
             and any(s.get("overlap", 1) > 1 for s in scenarios), "temporal_validation_required_scenarios_missing")
    uniform_n = plan.get("replicates_per_stochastic_scenario")
    _require(type(uniform_n) is int and uniform_n > 0, "temporal_validation_replication_count_missing")
    groups={scenario["id"]:{"replicates":set(),"false":0,"mse_coverage":0,"rmse_gain_coverage":0,
                              "per_rival_power":[0]*20,"joint_power":0} for scenario in scenarios}
    scenario_specs={scenario["id"]:scenario for scenario in scenarios}
    _require(len(groups)==len(scenarios),"validation_scenarios_duplicated")
    n_by_scenario={key:uniform_n for key in groups}
    components=["replicates"]
    source_map=None
    report_type=report.get("report_type","single_study")
    _require(report_type in ("single_study","provenance_composite"),"unsupported_temporal_validation_report_type")
    if report_type=="provenance_composite":
        for key in ("prior_replicates","prior_report","scenario_source_map"):
            files[key]=_spec_file(evidence.get(key),report_path.parent)
        source_map=c.read_json(files["scenario_source_map"])
        _require(source_map.get("frozen_before_focal_execution") is True and source_map.get("no_pooling") is True
                 and source_map.get("statistics_source_sha256")==statistics_sha
                 and freeze.get("scenario_source_map_sha256")==c.sha(files["scenario_source_map"]),
                 "composite_validation_source_map_not_frozen")
        declared=plan.get("replicates_per_scenario")
        _require(isinstance(declared,dict) and set(declared)==set(groups)
                 and all(type(value) is int and value>0 for value in declared.values()),"composite_validation_scenario_sizes_invalid")
        mapping=source_map.get("scenario_sources",{})
        focal=source_map.get("focal_scenario")
        _require(set(mapping)==set(groups) and focal in groups
                 and mapping[focal].get("component")=="replicates"
                 and all(mapping[key].get("component")=="prior_replicates" for key in groups if key!=focal)
                 and all(mapping[key].get("replicates")==declared[key] for key in groups),
                 "composite_validation_provenance_map_invalid")
        excluded=source_map.get("preserved_but_excluded",{})
        _require(excluded.get("prior_replicates")==[focal] and not excluded.get("replicates"),
                 "composite_validation_exclusion_map_invalid")
        parent=c.read_json(files["prior_report"])
        _require(source_map.get("prior_report_sha256")==c.sha(files["prior_report"])
                 and plan.get("focal_scenario")==focal,"composite_prior_report_or_focal_identity_differs")
        _require(parent.get("status")=="INCONCLUSIVE_BOUNDED_VALIDATION" and parent.get("admitted") is False
                 and parent.get("statistics_source_sha256")==statistics_sha,
                 "composite_original_inconclusive_verdict_must_be_preserved")
        parent_ledger=_spec_file(parent.get("evidence",{}).get("replicates"),files["prior_report"].parent)
        _require(c.sha(parent_ledger)==c.sha(files["prior_replicates"]),"composite_prior_archive_not_bound_to_original_report")
        parent_plan_path=_spec_file(parent.get("evidence",{}).get("plan"),files["prior_report"].parent)
        parent_plan=c.read_json(parent_plan_path)
        parent_freeze_path=_spec_file(parent.get("evidence",{}).get("freeze"),files["prior_report"].parent)
        parent_runner_path=_spec_file(parent.get("evidence",{}).get("runner"),files["prior_report"].parent)
        parent_freeze=c.read_json(parent_freeze_path)
        _require(parent_freeze.get("plan_sha256")==c.sha(parent_plan_path)
                 and parent_freeze.get("runner_sha256")==c.sha(parent_runner_path)
                 and parent_freeze.get("statistics_sha256")==statistics_sha,
                 "composite_original_frozen_bundle_identity_differs")
        parent_n=parent_plan.get("replicates_per_stochastic_scenario")
        _require(type(plan.get("seed")) is int and type(parent_plan.get("seed")) is int and plan["seed"]!=parent_plan["seed"],
                 "composite_focal_DGP_seed_not_independent_from_prior")
        _require(type(parent_n) is int and parent_n>0 and all(declared[key]==parent_n for key in groups if key!=focal),
                 "composite_reused_scenario_size_differs_from_original")
        _require({row.get("id") for row in parent_plan.get("scenarios",[])}==set(groups)
                 and all(next(row for row in parent_plan["scenarios"] if row["id"]==key)==scenario_specs[key] for key in groups),
                 "composite_prior_scenario_definitions_changed")
        n_by_scenario=declared
        components.append("prior_replicates")
    verified_ledgers={}
    def ledger_rows():
        total=0
        for component in components:
            raw_path=files[component]
            plaintext_sha=evidence[component].get("uncompressed_sha256")
            if plaintext_sha is None and component=="replicates":
                plaintext_sha=report.get("replicates_uncompressed_sha256",report.get("raw_results_sha256"))
            is_gzip=raw_path.suffix==".gz"
            _require(not is_gzip or isinstance(plaintext_sha,str) and bool(re.fullmatch(r"[0-9a-f]{64}",plaintext_sha)),
                     "compressed_validation_plaintext_hash_required")
            uncompressed=hashlib.sha256();selected_count=0;excluded_rows=set()
            with (gzip.open(raw_path,"rb") if is_gzip else raw_path.open("rb")) as stream:
                for line in stream:
                    total+=len(line)
                    _require(total<=512*1024*1024 and len(line)<=1024*1024,"validation_ledger_decompression_exceeds_bound")
                    uncompressed.update(line)
                    if not line.strip():continue
                    row=json.loads(line)
                    if source_map is not None:
                        key=row.get("scenario")
                        _require(key in groups,"composite_ledger_scenario_unknown")
                        selected=source_map["scenario_sources"][key]["component"]==component
                        if not selected:
                            _require(key in source_map["preserved_but_excluded"].get(component,[]) and component=="prior_replicates"
                                     and type(row.get("replicate")) is int and 0<=row["replicate"]<parent_n,
                                     "composite_unmapped_or_unplanned_ledger_rows")
                            excluded_key=(key,row["replicate"])
                            _require(excluded_key not in excluded_rows,"composite_excluded_original_replicate_duplicated")
                            excluded_rows.add(excluded_key)
                            continue
                    selected_count+=1
                    yield row
            _require(plaintext_sha is None or uncompressed.hexdigest()==plaintext_sha,"validation_plaintext_hash_differs")
            if source_map is not None and component=="prior_replicates":
                _require(excluded_rows=={(focal,index) for index in range(parent_n)},"composite_original_inconclusive_rows_not_preserved")
            verified_ledgers[component]={"sha256":c.sha(raw_path),"uncompressed_sha256":uncompressed.hexdigest(),
                                         "selected_rows":selected_count,"preserved_excluded_rows":len(excluded_rows)}
    try:
        for row in ledger_rows():
            key=row.get("scenario")
            _require(key in groups and row.get("panel") in (None,"single_temporal_panel")
                     and type(row.get("replicate")) is int and 0<=row["replicate"]<n_by_scenario[key],
                     "validation_raw_scenario_or_replicate_unknown")
            bucket=groups[key]
            _require(row["replicate"] not in bucket["replicates"],"validation_raw_duplicate_replicate")
            _require(set(row.get("method_false_reject",{}))==set(power.METHODS) and len(row.get("per_rival_gate",[]))==20
                     and all(isinstance(x,bool) for x in row["per_rival_gate"])
                     and isinstance(row.get("false_reject_any_null"),bool)
                     and all(isinstance(row.get(k),bool) for k in ("coverage_ci99_mse","coverage_ci99_gain")),
                     "temporal_validation_raw_family_incomplete")
            # Verify the reported complete-family decision from the raw
            # finite-reference tail counts, not a caller's pass boolean.
            vectors=row.get("test_vectors",{})
            names=("raw_p","adjusted_p","status","mc_tail_count","mc_reps","mc_p_rule")
            _require(all(isinstance(vectors.get(name),list) and len(vectors[name])==120 for name in names),
                     "validation_ledger_full_120_tail_vectors_required")
            raw_ps=[]
            for i in range(120):
                status=vectors["status"][i]
                count,reps=vectors["mc_tail_count"][i],vectors["mc_reps"][i]
                expected_reps=settings["B"] if (i//2)%3==1 else settings.get("reference_reps",49999)
                _require(status=="ok" and type(count) is int and type(reps) is int and 0<=count<=reps and reps==expected_reps
                         and vectors["mc_p_rule"][i]=="absolute_tail_plus_one","validation_MC_tail_identity_invalid")
                raw_p=(count+1)/(reps+1)
                _require(isinstance(vectors["raw_p"][i],(float,int)) and abs(vectors["raw_p"][i]-raw_p)<=1e-12,
                         "validation_MC_tail_p_mismatch")
                raw_ps.append(raw_p)
            from .statistics import holm
            adjusted=holm(raw_ps)
            _require(np.allclose(adjusted,vectors["adjusted_p"],rtol=0,atol=1e-12),"validation_Holm_adjustment_differs")
            scenario=scenario_specs[key]
            if scenario["null_rivals"]:
                label=0 if scenario["boundary_null"]=="epa" else 1
                flags={method:any(adjusted[rival*6+mi*2+label]<settings.get("alpha",.05)
                                  for rival in range(scenario["null_rivals"]))
                       for mi,method in enumerate(power.METHODS)}
            else:flags={method:False for method in power.METHODS}
            _require(flags==row["method_false_reject"] and any(flags.values())==row["false_reject_any_null"],
                     "validation_false_rejection_counter_differs_from_raw_p")
            _require(all(not gate or all(adjusted[rival*6+i]<settings.get("alpha",.05) for i in range(6))
                         for rival,gate in enumerate(row["per_rival_gate"])),
                     "validation_power_gate_contradicts_adjusted_p")
            for label,truth_key in (("mse","truth_mse"),("gain","truth_gain")):
                interval=row.get("ci99_"+label);truth=row.get(truth_key)
                covered=bool(isinstance(interval,list) and len(interval)==2 and isinstance(truth,(float,int))
                             and all(isinstance(v,(float,int)) and math.isfinite(v) for v in interval)
                             and interval[0]<=truth<=interval[1])
                _require(covered==row["coverage_ci99_"+label],"validation_coverage_counter_differs_from_interval")
                bucket["mse_coverage" if label=="mse" else "rmse_gain_coverage"]+=int(covered)
            _require(isinstance(row.get("joint_gate"),bool) and row["joint_gate"]==all(row["per_rival_gate"]),
                     "validation_joint_power_counter_differs")
            bucket["replicates"].add(row["replicate"])
            bucket["false"]+=int(row["false_reject_any_null"])
            bucket["joint_power"]+=int(row["joint_gate"])
            bucket["per_rival_power"]=[a+int(b) for a,b in zip(bucket["per_rival_power"],row["per_rival_gate"])]
    except (OSError,EOFError,ValueError,TypeError,KeyError):
        raise ControllerError("validation_ledger_corrupt_or_invalid") from None
    strong_alternative_seen=False
    for scenario in scenarios:
        bucket=groups[scenario["id"]]
        n=n_by_scenario[scenario["id"]]
        _require(bucket["replicates"]==set(range(n)),"temporal_validation_raw_replicates_incomplete")
        if scenario.get("null_rivals",0)>0:
            k=bucket["false"]
            upper=1. if k==n else float(beta.ppf(.975,k+1,n-k))
            _require(upper<=.05,"temporal_validation_raw_FWER_not_demonstrated")
        else:
            strong_alternative_seen=True
            _require(all(k>0 and float(beta.ppf(.025,k,n-k+1))>=.90 for k in bucket["per_rival_power"]),
                     "temporal_validation_strong_alternative_power_not_demonstrated")
        for name in ("mse_coverage","rmse_gain_coverage"):
            k=bucket[name]
            lower=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
            _require(lower>=.95,"temporal_validation_raw_coverage_not_demonstrated")
    _require(strong_alternative_seen,"temporal_validation_strong_alternative_missing")

    return {"path": str(report_path), "sha256": c.sha(report_path),
            "report_type":report_type,"replicates_per_scenario":n_by_scenario,"verified_ledgers":verified_ledgers,
            "verified_bundle": {k:{"path":str(p),"sha256":c.sha(p)} for k,p in files.items()}}


def _admit_final(source_root, protocol, protocol_sha, power_receipt):
    final = protocol.get("final", {})
    _require(final.get("closed") is True, "final_protocol_not_closed")
    R,T,B = (final.get(k) for k in ("R", "T", "B"))
    _require(type(R) is int and 10 <= R <= 4091 and type(T) is int and T > 0 and type(B) is int and B > 0, "final_dimensioning_R_T_B_required")
    _stress_plans(protocol)
    adequacy=protocol.get("adequacy",{})
    adequacy_path=Path(adequacy.get("config_path","configs/adequacy.json"))
    if not adequacy_path.is_absolute():adequacy_path=Path(source_root)/adequacy_path
    _require(adequacy_path.is_file() and adequacy.get("config_sha256")==c.sha(adequacy_path),"adequacy_threshold_configuration_not_frozen")
    settings = protocol["statistical_settings"]
    _require(settings.get("bootstrap_reps") == B, "final_bootstrap_setting_differs_from_power")
    targets = final.get("test_panels")
    _require(isinstance(targets, dict) and set(targets) == set(PANELS), "fixed_four_panel_test_calendars_required")
    for target in targets.values():
        _require(isinstance(target, dict) and target.get("count") == T and isinstance(target.get("start"), str)
                 and isinstance(target.get("end"), str) and target["start"] <= target["end"], "fixed_test_calendar_invalid")
        dt.date.fromisoformat(target["start"])
        dt.date.fromisoformat(target["end"])
        _require(target.get("reserve_mode") in ("documented_historical_reserve","external_custodian","post_protocol"),
                 "fixed_authentic_reserve_mode_required")
    validated_settings={"R":R,"T":T,"B":B,"block_length":settings["block_length"],"hac_lags":settings["hac_lags"],
                        "horizon":settings.get("horizon",1),"margin":settings.get("margin",.02)}
    validated_settings.update({k:settings[k] for k in ("reference_reps","reference_seed","alpha") if k in settings})
    method = _method_admission(source_root, final.get("method_admission") or protocol.get("method_validation"), validated_settings)
    spec = final.get("power_receipt")
    _require(isinstance(spec, dict), "frozen_power_receipt_required")
    path = _spec_file(spec, source_root)
    if power_receipt is not None:
        supplied = c.safe(power_receipt)
        _require(c.sha(supplied) == c.sha(path), "supplied_power_receipt_differs")
    result = c.read_json(path)
    _require(result.get("status") == "DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP" and result.get("temporal_calibration_verified") is True,
             "complete_conditional_empirical_power_receipt_required")
    _require([result.get("final_"+k) for k in ("R","T","B")] == [R,T,B], "power_sample_size_differs")
    _require(result.get("methods_sha256")==c.sha(Path(source_root)/"engine/statistics.py"),"power_methods_source_differs")
    fixed = result.get("fixed_before_validation", {})
    _require(c.digest(fixed) == result.get("fixed_before_validation_sha256") and fixed.get("cell") == {"R":R,"T":T,"B":B}, "power_winner_not_hash_bound")
    validation = result.get("validation_result", {})
    _require(validation.get("cell") == fixed["cell"] and validation.get("meets_power_target") is True and not validation.get("incomplete")
             and validation.get("minimum_per_contrast_simultaneous_mc_lower_bound", 0) >= .90, "power_validation_not_demonstrated")
    _require(bool(result.get("planning_evidence", {}).get("verified_files")), "power_planning_evidence_missing")
    for item in result["planning_evidence"]["verified_files"]:
        _spec_file(item, path.parent)
    design_path=Path(source_root)/"configs/power_design.json"
    design=c.read_json(design_path)
    _require(result.get("design_sha256")==c.digest(design),"power_design_identity_differs")
    _require(design.get("power_decision_rule")=="finite-B-count-plus-one-actual-Holm-v1"
             and result.get("operational_rule")==design["power_decision_rule"]
             and validation.get("operational_rule")==design["power_decision_rule"],"power_operational_finite_MC_rule_differs")
    _require(design.get("inner_block_length")==settings["block_length"] and design.get("hac_lags")==settings["hac_lags"]
             and design.get("horizon")==settings.get("horizon",1) and design.get("margin_relative_rmse")==settings.get("margin",.02),"power_statistical_rule_differs")
    _require(all(design.get(k)==settings[k] for k in ("reference_reps","reference_seed","alpha") if k in settings),"power_reference_bootstrap_identity_differs")
    alternatives=design.get("alternatives_relative_rmse_gain",[])
    _require(len(alternatives)>0 and {s.get("alternative_relative_rmse_gain") for s in validation.get("scenarios",[])}==set(alternatives),"power_declared_alternatives_incomplete")
    expected_contrasts={(p,m) for p in PANELS for m in models.METHODS[1:]}
    for scenario in validation.get("scenarios", []):
        contrasts = scenario.get("contrasts", [])
        _require({(x.get("panel"),x.get("rival")) for x in contrasts}==expected_contrasts,"power_contrast_family_duplicated_or_incomplete")
        for contrast in contrasts:
            k,n=contrast.get("successes"),contrast.get("trials")
            _require(type(k) is int and type(n) is int and 0<=k<=n and n>=design["mc_validation_reps"],"power_raw_counter_invalid")
            tail=design["mc_confidence_error"]/(80*len(alternatives))
            lower=0. if k==0 else float(beta.ppf(tail/2,k,n-k+1))
            _require(lower>=.90 and contrast.get("simultaneous_mc_lower_bound",1.)<=lower+1e-12,"power_raw_counter_does_not_establish_point90")
        _require(len(contrasts) == 80 and all(x.get("simultaneous_mc_lower_bound",0) >= .90 for x in contrasts)
                 and isinstance(scenario.get("joint_statistical_rule_power"), (int,float)), "power_all_80_contrasts_and_joint_report_required")
    _require(bool(validation.get("scenarios")), "power_meaningful_alternative_evidence_missing")
    return {"R":R,"T":T,"B":B,"power":{"path":str(path),"sha256":c.sha(path)},"method_admission":method,"protocol_sha256":protocol_sha}



def _prospective_final_support(protocol,admissions):
    """Reject an impossible reserved calendar before any final fit starts."""
    targets=protocol.get("final",{}).get("test_panels",{})
    for panel in PANELS:
        record=admissions[panel]
        origin=record.get("provenance",{}).get("bar_index_origin")
        _require(isinstance(origin,str) and record.get("provenance",{}).get("bar_index_convention")=="zero_based_complete_observed_calendar_sessions_v1",
                 "final_preparation_requires_common_observed_bar_index_origin")
        dt.date.fromisoformat(origin)
        ends=[]
        for part in DEVELOPMENT:
            dates=record.get("partition_date_ranges",{}).get(part)
            _require(isinstance(dates,list) and len(dates)==2 and all(isinstance(value,str) for value in dates),
                     "final_preparation_requires_development_date_bounds")
            _require(origin<=dates[0],"development_bar_index_origin_after_endpoint")
            ends.append(dates[1])
        target=targets.get(panel,{})
        _require(isinstance(target.get("start"),str) and target["start"]>max(ends),
                 "planned_reserved_test_not_after_all_development_dates")


def prepare_case(source_root, case_root, panel_dirs, stage, source_commit, protocol_path, power_receipt=None, *, software_fixture=False):
    """Freeze a new plan; read development partitions only; never fit or test."""
    _require(stage in ("planning", "final"), "unknown_campaign_stage")
    _require(not software_fixture or stage == "planning", "fixtures_never_qualify_for_final_campaign")
    root, sources = _sources(source_root, source_commit)
    protocol_path = c.safe(protocol_path)
    protocol, protocol_sha = c.read_json(protocol_path), c.sha(protocol_path)
    identity_path = root/"configs/CANDIDATE_IDENTITY.json"
    _require(protocol.get("candidate_recipe") == models.CANDIDATE and protocol.get("candidate_identity_sha256") == c.sha(identity_path), "frozen_candidate_identity_differs")
    _require(protocol.get("family", {}).get("method_order") == list(models.METHODS), "complete_canonical_family_required")
    paths, admissions = _admissions(panel_dirs, allow_fixture=software_fixture)
    _require(all(set(DEVELOPMENT).issubset(a["partition_paths"]) and "test" not in a["partition_paths"] for a in admissions.values()), "prepare_requires_four_development_only_admissions")
    final = _admit_final(root, protocol, protocol_sha, power_receipt) if stage == "final" else None
    if final is not None:_prospective_final_support(protocol,admissions)
    bank_path = _spec_file(protocol.get("seed_bank"), root)
    _require(c.sha(bank_path) == SEED_BANK_SHA256, "historical_frozen_seed_bank_identity_differs")
    bank = c.read_json(bank_path)["seed_ids"]
    _require(len(bank) == 4096 and len(set(bank)) == 4096, "unique_frozen_seed_bank_4096_required")
    count = protocol.get("planning_seed_count",5) if stage == "planning" else final["R"]
    _require(type(count) is int and (count == 5 if stage == "planning" else 10 <= count <= 4091), "fixed_seed_prefix_or_planning_tail_required")
    seeds = bank[4091:4096] if stage == "planning" else bank[:count]
    if final is not None:
        receipt = c.read_json(final["power"]["path"])
        _require(receipt.get("final_seed_ids") == seeds, "power_seed_ids_differ_from_prefix")
    training_config = dict(protocol.get("training", {}))
    allowed = {"total_budget_seconds", "epoch_reservation_seconds", "budget_stop_policy", "override_historical_hyperparameters"}
    _require(software_fixture or set(training_config).issubset(allowed), "prospective_hyperparameter_substitution_refused")
    _require(training_config.get("override_historical_hyperparameters",False) is False, "historical_hyperparameters_must_be_preserved")
    panels = []
    for panel, admission in admissions.items():
        for part in DEVELOPMENT:
            data.load_partition(paths[panel], part)  # integrity/causality only, no exploratory summaries
        metadata=paths[panel]/"recovery_metadata.json"
        _require(software_fixture or metadata.is_file(),"genuine_development_requires_metadata_only_recovery_bundle")
        recovery={"path":f"receipts/dataset_metadata/{panel}.json","sha256":c.sha(metadata)} if metadata.exists() else None
        if recovery is not None:
            replay=c.read_json(metadata)
            _require(replay.get("kind")=="development_metadata_replay_v1" and c.safe(replay.get("original_panel_dir","."))==paths[panel]
                     and replay.get("original_admission",{}).get("sha256")==c.sha(paths[panel]/"admission.json"),
                     "development_recovery_metadata_does_not_match_admission")
        panels.append({"id":panel,"panel_dir":str(paths[panel]),"data_sha256":admission["admission_sha256"],
                       "development":_partition_metadata(admission),"provenance":admission["provenance"],
                       "rights":admission["rights"],"exposure":admission["exposure"],"recovery_metadata":recovery})
    dependencies=[]
    if final is not None:
        power_result=c.read_json(final["power"]["path"])
        for item in power_result.get("planning_evidence",{}).get("receipts",[]):
            receipt_path=_spec_file(item,Path(final["power"]["path"]).parent)
            planning_root=receipt_path.parent.parent
            prior=c.read_json(planning_root/"plan.json");receipt=c.read_json(receipt_path)
            _require(prior.get("stage")=="planning" and prior.get("source_commit")==source_commit
                     and receipt.get("case_id")==prior.get("case_id") and receipt.get("plan_sha256")==c.digest(prior),
                     "power_planning_dependency_case_identity_invalid")
            dep={"case_root":str(planning_root),"case_id":prior["case_id"],"plan_sha256":c.digest(prior),"source_commit":source_commit}
            if dep not in dependencies:dependencies.append(dep)
        _require(bool(dependencies),"final_recovery_requires_real_planning_case_dependency")
    case = c.safe(case_root)
    with c.case_lock(case):
        _require(not (case/"plan.json").exists(), "case_already_prepared_use_resume")
        case_id = "r3-engine-" + ("plan-" if stage == "planning" else "final-") + uuid.uuid4().hex
        jobs = []
        for panel in panels:
            for mi, model in enumerate(models.METHODS):
                for si, seed in enumerate(seeds if model in models.NEURAL else seeds[:1]):
                    jobs.append({"id":f'{panel["id"]}-m{mi:02d}-s{si:04d}',"panel":panel["id"],"model":model,"seed":seed,
                                 "protocol_sha256":protocol_sha,"training":training_config,
                                 "identity":{"case_id":case_id,"stage":stage,"source_commit":source_commit,
                                             "source_manifest_sha256":sources["manifest_sha256"],"data_admission_sha256":panel["data_sha256"],
                                             "candidate_identity_sha256":c.sha(identity_path),"software_fixture":software_fixture}})
        plan = {"version":VERSION,"case_id":case_id,"stage":stage,"source_commit":source_commit,
                "original_case_root":str(case),"original_protocol_path":str(protocol_path),"dependency_cases":dependencies,
                "source_manifest_sha256":sources["manifest_sha256"],"candidate_identity_sha256":c.sha(identity_path),
                "protocol_sha256":protocol_sha,"data_panels":panels,"seeds":seeds,"seed_bank_sha256":c.sha(bank_path),
                "family":list(models.METHODS),"jobs":jobs,"final_admission":final,"software_fixture":software_fixture,
                "created_at_utc":c.now(),"test_opened":False,"R3_accredited":False}
        for panel in panels:
            if panel["recovery_metadata"] is not None:
                payload=(Path(panel["panel_dir"])/"recovery_metadata.json").read_bytes()
                _require(hashlib.sha256(payload).hexdigest()==panel["recovery_metadata"]["sha256"],"recovery_metadata_changed_while_preparing")
                c.atomic(case/panel["recovery_metadata"]["path"],payload,immutable=True)
        c.atomic(case/"protocol.json", protocol_path.read_bytes(), immutable=True)
        c.write_json(case/"source_identity.json", {**sources,"source_commit":source_commit}, immutable=True)
        c.write_json(case/"plan.json", plan, immutable=True)
        c.write_json(case/"session.json", {"session_open":False,"stage":"prepared","at_utc":c.now()})
        c.write_json(case/"status.json", {"status":"PREPARED_NO_FITS","jobs":len(jobs),"test_opened":False,"R3_accredited":False})
        if final is not None:
            c.atomic(case/"receipts/power.json",Path(final["power"]["path"]).read_bytes(),immutable=True)
        return {"status":"PREPARED_NO_FITS","case_id":case_id,"plan_sha256":c.digest(plan),"jobs":len(jobs),"neural_fits":4*16*len(seeds),"deterministic_systems":20,"source_commit":source_commit}


def _validate_jobs(plan):
    expected={(panel,model,seed) for panel in PANELS for model in models.METHODS
              for seed in (plan["seeds"] if model in models.NEURAL else plan["seeds"][:1])}
    jobs=plan.get("jobs",[])
    _require(plan.get("stage") in ("planning","final") and (plan.get("stage")!="final" or not plan.get("software_fixture")),
             "fixture_or_stage_cannot_qualify_final_case")
    _require(len(jobs)==len(expected) and len({j.get("id") for j in jobs})==len(jobs)
             and {(j.get("panel"),j.get("model"),j.get("seed")) for j in jobs}==expected,
             "campaign_job_cartesian_family_incomplete_or_duplicated")
    _require(all(j.get("protocol_sha256")==plan["protocol_sha256"] and j.get("identity",{}).get("case_id")==plan["case_id"]
                 and j.get("identity",{}).get("source_commit")==plan["source_commit"] for j in jobs), "campaign_job_identity_differs")


def _load_case(source_root, case_root, panel_dirs=None):
    case = c.safe(case_root)
    plan = c.read_json(case/"plan.json")
    _validate_jobs(plan)
    _, sources = _sources(source_root, plan["source_commit"])
    _require(sources["manifest_sha256"] == plan["source_manifest_sha256"], "campaign_source_manifest_differs")
    _require(c.sha(case/"protocol.json") == plan["protocol_sha256"] and plan["family"] == list(models.METHODS), "campaign_protocol_or_family_differs")
    bank=c.read_json(Path(source_root)/"configs/FINAL_SEED_BANK.json")["seed_ids"]
    _require(plan.get("seed_bank_sha256")==SEED_BANK_SHA256 and c.sha(Path(source_root)/"configs/FINAL_SEED_BANK.json")==SEED_BANK_SHA256
             and plan["seeds"]==(bank[4091:4096] if plan["stage"]=="planning" else bank[:len(plan["seeds"])])
             and (plan["stage"]!="final" or 10<=len(plan["seeds"])<=4091), "case_seeds_differ_from_fixed_tail_or_prefix")
    if panel_dirs is None:
        return case, plan, None, None
    paths, admissions = _admissions(panel_dirs,allow_fixture=plan["software_fixture"])
    for panel in plan["data_panels"]:
        actual = _partition_metadata(admissions[panel["id"]])
        _require(all(actual.get(part) == expected for part,expected in panel["development"].items()), "development_partition_identity_differs")
        _require(admissions[panel["id"]]["provenance"] == panel["provenance"] and admissions[panel["id"]]["rights"]==panel["rights"]
                 and admissions[panel["id"]]["exposure"]==panel["exposure"], "development_provenance_or_owner_evidence_differs")
    return case, plan, paths, admissions


def _receipt(case, plan, remote):
    _require(isinstance(remote,dict) and remote.get("exists") is True and remote.get("remote_verified") is True
             and remote.get("source_commit") == plan["source_commit"] and remote.get("case_id") == plan["case_id"]
             and remote.get("plan_sha256") == c.digest(plan), "authenticated_remote_case_required")
    marker = case/".github_backup_receipt.json"
    _require((marker.is_file() and c.read_json(marker).get("commit") == remote.get("head")) or (case/".github_pending").is_dir(),
             "stale_or_unreceipted_local_case_restore_first")


def _public(case, admissions):
    """Bounded metadata only; complete prediction CSVs stay encrypted.

    The deterministic policy is independent of outcomes. All full derived
    files remain in the encrypted private manifest and complete export ZIP.
    """
    if admissions is None:
        final_metadata=case/"receipts/final_admissions.json"
        if final_metadata.exists():
            admissions=c.read_json(final_metadata)
        else:
            admissions={p["id"]:{"rights":p["rights"]} for p in c.read_json(case/"plan.json")["data_panels"]}
    if not all(a.get("rights",{}).get("allow_derived_publication") is True for a in admissions.values()):
        return {}
    maximum_public_bytes=2*1024*1024
    maximum_receipt_bytes=256*1024
    maximum_inventory_bytes=1024*1024
    receipts=("status.json","source_identity.json","selected_manifest.json","training_seal.json",
              "evaluation_receipt.json","analysis_receipt.json")
    artifacts={}
    for folder in ("predictions","metrics","receipts"):
        base=case/folder
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in (".csv",".json",".jsonl",".tsv",".npy",".npz"):
                    artifacts[path.relative_to(case).as_posix()]=path
    for name in receipts:
        if (case/name).exists():artifacts[name]=case/name
    inventory=[{"path":name,"sha256":c.sha(path),"bytes":path.stat().st_size,
                "location":"encrypted_checkpoint_shards_and_complete_export"}
               for name,path in sorted(artifacts.items())]
    inventory_raw=c.canonical(inventory)
    index={"version":"bounded-public-metadata-1","policy":"fixed receipts under byte caps; no full prediction CSVs or outcome-dependent row selection",
           "full_prediction_rows_preserved":True,"private_artifact_count":len(inventory),
           "private_inventory_sha256":c.digest(inventory),"private_inventory_bytes":len(inventory_raw),
           "full_inventory_location":"authenticated encrypted checkpoint manifest; reconstruct these fixed derived prefixes",
           "full_results_location":"encrypted GitHub shards and export_case ZIP; independent backup secret required",
           "maximum_public_plain_bytes":maximum_public_bytes,"maximum_receipt_bytes":maximum_receipt_bytes,
           "artifact_inventory":inventory if len(inventory_raw)<=maximum_inventory_bytes else None,
           "inventory_omitted_only_for_public_byte_bound":len(inventory_raw)>maximum_inventory_bytes}
    index_bytes=c.canonical(index)+b"\n"
    files={"receipts/private_artifact_inventory.json":index_bytes}
    used=len(index_bytes)
    for name in receipts:
        path=case/name
        if path.exists() and path.stat().st_size<=maximum_receipt_bytes and used+path.stat().st_size<=maximum_public_bytes:
            files["receipts/"+name]=path
            used+=path.stat().st_size
    _require(used<=maximum_public_bytes,"bounded_public_metadata_policy_exceeded")
    return files


def _backup(case, plan, store, admissions, stage):
    _require(store is not None and all(callable(getattr(store,k,None)) for k in ("backup","status","restore")), "github_store_required_no_local_training_bypass")
    try:
        had_pending = (case/".github_pending").exists()
        result = store.backup(case, public_files=_public(case,admissions), stage=stage)
        if had_pending:
            _require(not (case/".github_pending").exists(), "pending_snapshot_not_reconciled")
            result = store.backup(case, public_files=_public(case,admissions), stage=stage)
        _require(isinstance(result,dict) and result.get("remote_verified") is True and result.get("source_commit") == plan["source_commit"]
                 and result.get("case_id") == plan["case_id"] and result.get("plan_sha256") == c.digest(plan), "backup_receipt_not_verified")
        remote = store.status()
        _receipt(case, plan, remote)
        _require(remote["head"] == result.get("commit"), "backup_remote_head_differs")
        _progress("github_checkpoint_verified",plan,status="remote_verified",remote_commit=result["commit"])
        return result
    except Exception as error:
        code = str(error) if type(error).__name__ == "StorageError" and re.fullmatch(r"[a-z][a-z0-9_]*(?: status=[1-5][0-9]{2})?",str(error)) else "storage_diagnostic_unavailable"
        failures = c.read_json(case/"failures.json") if (case/"failures.json").exists() else []
        failures.append({"at_utc":c.now(),"stage":stage,"kind":"remote_durability_failure","error_type":type(error).__name__,"diagnostic":code})
        c.write_json(case/"failures.json",failures)
        raise training.DurabilityError("github_backup_failed_stop_before_further_jobs") from None


def _restore_if_missing(case_root, store):
    case = c.safe(case_root)
    if not (case/"plan.json").exists():
        _require(not case.exists(), "fresh_restore_destination_required_for_missing_case")
        _require(store is not None, "github_store_required_for_restore")
        result = store.restore(case)
        _require(isinstance(result,dict) and result.get("remote_verified") is True, "no_authenticated_remote_case_to_restore")


def _uncertain_compute(case):
    history=c.read_json(case/"recovery_history.json") if (case/"recovery_history.json").exists() else []
    _require(isinstance(history,list),"recovery_history_invalid")
    return any(r.get("unobserved_remote_compute_possible") is True for r in history)


def run_case(source_root, case_root, panel_dirs, store, *, activate=False, max_jobs=None, checkpoint_interval_seconds=60.):
    """Explicitly fit/resume; stop synchronously on every failed durability barrier."""
    _require(activate is True, "training_requires_explicit_author_activation")
    _require(max_jobs is None or type(max_jobs) is int and max_jobs > 0, "invalid_job_batch_limit")
    _restore_if_missing(case_root,store)
    case, plan, paths, admissions = _load_case(source_root,case_root,panel_dirs)
    with c.case_lock(case):
        if plan["stage"] == "final":
            _admit_final(source_root,c.read_json(case/"protocol.json"),plan["protocol_sha256"],None)
        _require(not (case/"training_seal.json").exists(), "sealed_training_cannot_be_modified")
        # Reacquired data are hash checked before the first fit in this process.
        arrays = {panel:{part:data.load_partition(path,part) for part in ("fit","selection")} for panel,path in paths.items()}
        remote = store.status() if store is not None else None
        if remote is not None and remote.get("exists"):
            _receipt(case,plan,remote)
        c.write_json(case/"session.json",{"session_open":True,"stage":"training","at_utc":c.now()})
        _backup(case,plan,store,admissions,"before_any_fit_or_resume")
        completed, executed, skipped = [], 0, 0
        for job in plan["jobs"]:
            directory = case/"jobs"/job["id"]
            if (directory/"done.json").exists():
                completed.append(training.verify_job(job,directory))
                skipped += 1
                _progress("job_complete_reused",plan,job=job,status="complete_verified")
                continue
            if max_jobs is not None and executed >= max_jobs:
                continue
            _backup(case,plan,store,admissions,"before_job_"+job["id"])
            _progress("job_started_or_resumed",plan,job=job,status="training")
            def checkpoint(job_dir,event):
                _require(c.safe(job_dir) == directory,"checkpoint_callback_job_differs")
                _backup(case,plan,store,admissions,str(event.get("kind","checkpoint")))
            try:
                done = training.fit_job(job,arrays[job["panel"]]["fit"],arrays[job["panel"]]["selection"],directory,
                                        on_checkpoint=checkpoint,checkpoint_interval_seconds=checkpoint_interval_seconds)
                completed.append(done)
                executed += 1
                training.export_job_resources(directory)
                saved=_backup(case,plan,store,admissions,"after_job_"+job["id"])
                _progress("job_complete_backed_up",plan,job=job,status="complete_verified",remote_commit=saved["commit"])
            except training.DurabilityError:
                raise
            except BaseException as error:
                failures = c.read_json(case/"failures.json") if (case/"failures.json").exists() else []
                failures.append({"at_utc":c.now(),"job_id":job["id"],"kind":"job_failed","error_type":type(error).__name__,"adverse_or_incomplete_preserved":True})
                c.write_json(case/"failures.json",failures)
                c.write_json(case/"session.json",{"session_open":False,"stage":"failed","at_utc":c.now()})
                _backup(case,plan,store,admissions,"job_failure_saved")
                raise
        result = {"status":"TRAINING_COMPLETE" if len(completed)==len(plan["jobs"]) else "TRAINING_PARTIAL", "jobs_total":len(plan["jobs"]),
                  "jobs_completed_verified":len(completed),"jobs_executed_this_call":executed,"jobs_skipped_verified":skipped,
                  "test_opened":False,"R3_accredited":False,"unobserved_remote_compute_possible":_uncertain_compute(case),
                  "charged_seconds_are_lower_bounds_if_unobserved_remote_compute":_uncertain_compute(case)}
        c.write_json(case/"status.json",result)
        c.write_json(case/"session.json",{"session_open":False,"stage":result["status"],"at_utc":c.now()})
        result["backup"] = _backup(case,plan,store,admissions,"training_call_finished")
        return result


def _completed(case,plan):
    return {j["id"]:training.verify_job(j,case/"jobs"/j["id"]) for j in plan["jobs"]}


def _rows(job,partition,prediction,done):
    n = len(partition["y"])
    _require(all(len(prediction[k]) == n for k in ("raw_sigma7","projected_sigma7","floor")), "prediction_length_differs")
    known = np.asarray(partition.get("known_floor_A",np.asarray(prediction["floor"])**2),dtype=float)
    rfit = done["preprocessor"]["candidate"]["preprocessor"]["reference_rms"]
    _require(np.isfinite(known).all() and np.all(known >= 0) and math.isfinite(rfit) and rfit>0, "causal_instrument_invalid")
    result = []
    for i in range(n):
        row = {"panel":job["panel"],"system":job["model"],"seed":str(job["seed"]) if job["model"] in models.NEURAL else "deterministic",
               "date":str(partition["dates"][i]),"y":float(partition["y"][i]),"prediction":float(prediction["projected_sigma7"][i]),
               "known_floor_A":float(known[i]),"rfit":float(rfit),"raw_prediction":float(prediction["raw_sigma7"][i])}
        _require(all(math.isfinite(row[k]) for k in ("y","prediction","known_floor_A","rfit","raw_prediction")), "nonfinite_prediction_refused")
        result.append(row)
    return result


def _predict_partition(case,plan,paths,done,part):
    partitions = {panel:data.load_partition(path,part) for panel,path in paths.items()}
    rows = []
    for job in plan["jobs"]:
        result = training.predict_job(job,partitions[job["panel"]],case/"jobs"/job["id"])
        rows.extend(_rows(job,partitions[job["panel"]],result,done[job["id"]]))
    path = case/"predictions"/(part+"_predictions.csv")
    c.write_csv(path,rows)
    return path


def predict_planning(source_root, case_root, panel_dirs, store):
    """Predict dedicated development planning only, after every job verifies."""
    case,plan,paths,admissions = _load_case(source_root,case_root,panel_dirs)
    _require(plan["stage"] == "planning", "planning_case_required")
    with c.case_lock(case):
        _receipt(case,plan,store.status())
        done = _completed(case,plan)
        path = case/"predictions/planning_predictions.csv"
        receipt_path = path.with_suffix(".receipt.json")
        if receipt_path.exists():
            receipt = c.read_json(receipt_path)
            _require(receipt.get("predictions_sha256") == c.sha(path) and receipt.get("plan_sha256") == c.digest(plan), "existing_planning_export_differs")
            _backup(case,plan,store,admissions,"planning_export_recovered")
            return {"predictions":str(path),"receipt":str(receipt_path),"reused":True}
        _backup(case,plan,store,admissions,"before_planning_predictions")
        path = _predict_partition(case,plan,paths,done,"planning")
        data_manifest = {"panels":{panel:{"admission_sha256":a["admission_sha256"],"planning_sha256":a["partition_sha256"]["planning"],
                                                     "support_audit":a["support_audit"]} for panel,a in admissions.items()}}
        manifest_path = case/"receipts/planning_data.json"
        c.write_json(manifest_path,data_manifest,immutable=True)
        fit_receipts=[]
        for job in plan["jobs"]:
            result = done[job["id"]]
            fit_receipts.append({"phase":"planning","panel":job["panel"],"system":job["model"],
                                 "seed":str(job["seed"]) if job["model"] in models.NEURAL else "deterministic", "success":True,
                                 "active_seconds":result["active_seconds_charged"],"compute_uncertain":result["compute_uncertain"],
                                 "active_seconds_is_charge_not_exact_elapsed":bool(result["compute_uncertain"]),
                                 "checkpoint":{"path":str(case/"jobs"/job["id"]/result["checkpoint_file"]),"sha256":result["checkpoint_sha256"]}})
            attempts_path=case/"jobs"/job["id"]/"attempts.json"
            if attempts_path.exists():
                for attempt in c.read_json(attempts_path):
                    fit_receipts.append({"phase":"planning","panel":job["panel"],"system":job["model"],"seed":str(job["seed"]) if job["model"] in models.NEURAL else "deterministic",
                                         "success":False,"active_seconds":0.,"ledger_event":attempt,
                                         "accounting":"charge/observed attempt already included in cumulative completed receipt; do not sum again",
                                         "not_an_additional_independent_fit":True})
        receipt = {"kind":"development_planning_predictions","partition":"planning","final_test_opened":False,
                   "synthetic":plan["software_fixture"],"is_fixture":plan["software_fixture"],"case_id":plan["case_id"],"plan_sha256":c.digest(plan),
                   "predictions_sha256":c.sha(path),"seed_bank_sha256":plan["seed_bank_sha256"],"protocol_sha256":plan["protocol_sha256"],
                   "methods_sha256":c.sha(Path(source_root)/"engine/statistics.py"),"planning_partition_purged":True,"instruments_available_at_origin":True,
                   "protocol_file":{"path":str(case/"protocol.json"),"sha256":plan["protocol_sha256"]},
                   "methods_file":{"path":str(c.safe(source_root)/"engine/statistics.py"),"sha256":c.sha(Path(source_root)/"engine/statistics.py")},
                   "source_manifest_file":{"path":str(c.safe(source_root)/"SOURCE_MANIFEST.json"),"sha256":plan["source_manifest_sha256"]},
                   "planning_data_file":{"path":str(manifest_path),"sha256":c.sha(manifest_path)},"fit_receipts":fit_receipts}
        c.write_json(receipt_path,receipt,immutable=True)
        _backup(case,plan,store,admissions,"planning_predictions_saved")
        return {"predictions":str(path),"receipt":str(receipt_path),"reused":False}


def seal_training(source_root, case_root, panel_dirs, store):
    """Freeze all selected states and development calibrators before test."""
    case,plan,paths,admissions = _load_case(source_root,case_root,panel_dirs)
    _require(plan["stage"] == "final", "final_case_required_to_seal")
    with c.case_lock(case):
        _admit_final(source_root,c.read_json(case/"protocol.json"),plan["protocol_sha256"],None)
        _receipt(case,plan,store.status())
        done = _completed(case,plan)
        selected = {"version":VERSION,"case_id":plan["case_id"],"plan_sha256":c.digest(plan),"protocol_sha256":plan["protocol_sha256"],
                    "jobs":[{"id":j["id"],"model":j["model"],"panel":j["panel"],"seed":j["seed"],
                             "checkpoint_file":done[j["id"]]["checkpoint_file"],"checkpoint_sha256":done[j["id"]]["checkpoint_sha256"],
                             "selected_state_sha256":done[j["id"]]["selected_state_sha256"]} for j in plan["jobs"]]}
        c.write_json(case/"selected_manifest.json",selected,immutable=True)
        prediction_files = {}
        for part in ("fit","selection","calibration"):
            path = case/"predictions"/(part+"_predictions.csv")
            identity_path = case/"receipts"/(part+"_prediction_identity.json")
            if identity_path.exists():
                previous=c.read_json(identity_path)
                _require(previous.get("sha256") == c.sha(path) and previous.get("selected_manifest_sha256") == c.sha(case/"selected_manifest.json"), "calibration_or_diagnostic_cache_differs")
            else:
                _predict_partition(case,plan,paths,done,part)
                c.write_json(identity_path,{"sha256":c.sha(path),"selected_manifest_sha256":c.sha(case/"selected_manifest.json")},immutable=True)
            prediction_files[part] = {"path":str(path),"sha256":c.sha(path)}
        calibrator_path=case/"metrics/calibrators.json"
        from .assessment import fit_calibrators
        fitted_calibrators=fit_calibrators(prediction_files["calibration"]["path"],output_path=calibrator_path)
        _require(len(fitted_calibrators.get("calibrators",[]))==len(plan["jobs"]) and all(item.get("status")=="ok" for item in fitted_calibrators["calibrators"]),
                 "every_selected_system_needs_sufficient_frozen_calibration")
        seal={"version":VERSION,"case_id":plan["case_id"],"plan_sha256":c.digest(plan),"protocol_sha256":plan["protocol_sha256"],
              "selected_manifest_sha256":c.sha(case/"selected_manifest.json"),"calibrators_sha256":c.sha(calibrator_path),
              "prediction_files":prediction_files,"test_opened":False,"unobserved_remote_compute_possible":_uncertain_compute(case)}
        c.write_json(case/"training_seal.json",seal,immutable=True)
        backup=_backup(case,plan,store,admissions,"selected_states_and_calibrators_sealed_before_test")
        return {"status":"TRAINING_AND_CALIBRATION_SEALED","selected_manifest":str(case/"selected_manifest.json"),"seal":str(case/"training_seal.json"),"backup":backup}


def _final_panels(case,plan,panel_dirs):
    """Bind newly released test metadata without changing the frozen plan."""
    paths, admissions = _admissions(panel_dirs,allow_fixture=False)
    seal = c.read_json(case/"training_seal.json")
    protocol=c.read_json(case/"protocol.json")
    for panel in PANELS:
        admission=admissions[panel]
        frozen=next(p for p in plan["data_panels"] if p["id"]==panel)
        # Custodian final preparation may be a test-only new directory. If it
        # includes dev partitions they must match exactly; it must never replace
        # the development identities frozen before training.
        for part,meta in _partition_metadata(admission).items():
            _require(meta==frozen["development"].get(part),"released_panel_changes_development_identity")
        _require(all(frozen["provenance"].get(key) is not None and admission["provenance"].get(key)==frozen["provenance"][key]
                     for key in ("provider","instrument","calendar")),"released_test_source_instrument_or_calendar_differs")
        origin=frozen["provenance"].get("bar_index_origin")
        _require(isinstance(origin,str) and admission["provenance"].get("bar_index_origin")==origin
                 and frozen["provenance"].get("bar_index_convention")=="zero_based_complete_observed_calendar_sessions_v1"
                 and admission["provenance"].get("bar_index_convention")==frozen["provenance"]["bar_index_convention"],
                 "released_test_bar_index_origin_differs_from_development")
        dt.date.fromisoformat(origin)
        auth=admission.get("final_authorization") or {}
        _require(auth.get("development_bar_index_origin")==origin,"final_authorization_bar_index_origin_differs")
        _require(auth.get("authorize_test_preparation") is True and auth.get("protocol_sha256")==plan["protocol_sha256"]
                 and auth.get("selected_checkpoints_sha256")==seal["selected_manifest_sha256"]
                 and auth.get("sources_methods_size_frozen") is True, "final_custodian_authorization_not_bound")
        data.verify_evidence(auth,"authorized")
        target=protocol["final"]["test_panels"][panel]
        _require(source_identity_sha256(admission["provenance"])==target.get("source_identity_sha256"),
                 "released_source_identity_differs_from_frozen_protocol")
        reserve_inventory=_spec_file(auth.get("reserve_inventory"),case)
        exposure_form=admission.get("exposure",{}).get("form",{})
        _require(exposure_form.get("reserve_mode")==target.get("reserve_mode")
                 and target.get("reserve_mode") in ("documented_historical_reserve","external_custodian","post_protocol"),
                 "released_reserve_mode_differs_from_fixed_protocol")
        _require(c.sha(reserve_inventory)==target.get("exposure_evidence_sha256")==exposure_form.get("exposure_inventory_sha256"),
                 "reserve_inventory_evidence_differs_from_frozen_protocol")
        _require(admission["partition_counts"].get("test")==target["count"]
                 and admission["partition_date_ranges"].get("test")==[target["start"],target["end"]],"released_test_calendar_or_count_differs")
        _require(admission["partition_support_bounds"]["test"][0] > max(v["support"][1] for v in frozen["development"].values()), "released_test_support_overlaps_development")
    return paths,admissions


def _saved_predictions(case,seal):
    result={}
    for part,record in seal["prediction_files"].items():
        path=case/"predictions"/(part+"_predictions.csv")
        _require(part in DEVELOPMENT and c.sha(path)==record["sha256"],"saved_development_prediction_hash_differs")
        result[part]={"path":str(path),"sha256":record["sha256"]}
    return result


def _cached_final_prediction(case,plan,job,done,store,admissions,eval_id,partition_factory,scenario=None):
    folder=case/"evaluation" if scenario is None else case/"evaluation"/scenario
    cache=folder/(job["id"]+".json")
    pending=folder/(job["id"]+".pending.json")
    condition=scenario or "nominal"
    if cache.exists():
        saved=c.read_json(cache)
        _require(saved.get("eval_id")==eval_id and saved.get("condition")==condition
                 and saved.get("checkpoint_sha256")==done["checkpoint_sha256"] and saved.get("rows_sha256")==c.digest(saved.get("rows")),
                 "evaluation_cache_identity_or_hash_differs")
        return saved["rows"]
    _require(not pending.exists(),"uncertain_final_forward_refuse_automatic_requery_preserve_evidence")
    c.write_json(pending,{"eval_id":eval_id,"job_id":job["id"],"condition":condition,"checkpoint_sha256":done["checkpoint_sha256"]},immutable=True)
    _backup(case,plan,store,admissions,"before_single_final_forward_"+job["id"]+"_"+condition)
    try:
        partition=partition_factory()
        prediction=training.predict_job(job,partition,case/"jobs"/job["id"])
        # The close-preserving scenarios have the same true known floor and
        # label. Apply that common projection explicitly to every adapter.
        prediction["projected_sigma7"]=np.maximum(prediction["raw_sigma7"],np.sqrt(partition["known_floor_A"]))
        rows=_rows(job,partition,prediction,done)
    except BaseException as error:
        failures=c.read_json(case/"failures.json") if (case/"failures.json").exists() else []
        failures.append({"kind":"final_query_failed_or_uncertain","job_id":job["id"],"condition":condition,"eval_id":eval_id,
                         "error_type":type(error).__name__,"at_utc":c.now(),"automatic_requery_allowed":False})
        c.write_json(case/"failures.json",failures)
        _backup(case,plan,store,admissions,"failed_final_query_evidence_preserved")
        raise
    c.write_json(cache,{"eval_id":eval_id,"condition":condition,"checkpoint_sha256":done["checkpoint_sha256"],"rows":rows,"rows_sha256":c.digest(rows)},immutable=True)
    _backup(case,plan,store,admissions,"single_final_prediction_cached_"+job["id"]+"_"+condition)
    return rows


def evaluate_once(source_root, case_root, panel_dirs, store, *, activate=False):
    """Run or export the same evaluation, refusing uncertain repeated forwards."""
    _require(activate is True,"final_test_requires_explicit_author_activation")
    case,plan,_,_=_load_case(source_root,case_root)
    _require(plan["stage"]=="final","final_case_required")
    with c.case_lock(case):
        protocol=c.read_json(case/"protocol.json")
        stress_plans=_stress_plans(protocol)
        _admit_final(source_root,protocol,plan["protocol_sha256"],None)
        _receipt(case,plan,store.status())
        done=_completed(case,plan)
        seal=c.read_json(case/"training_seal.json")
        _require(seal.get("plan_sha256")==c.digest(plan) and seal.get("selected_manifest_sha256")==c.sha(case/"selected_manifest.json")
                 and seal.get("calibrators_sha256")==c.sha(case/"metrics/calibrators.json"),"selected_states_or_calibrators_changed_since_seal")
        actual=[{"id":j["id"],"model":j["model"],"panel":j["panel"],"seed":j["seed"],"checkpoint_file":done[j["id"]]["checkpoint_file"],
                 "checkpoint_sha256":done[j["id"]]["checkpoint_sha256"],"selected_state_sha256":done[j["id"]]["selected_state_sha256"]} for j in plan["jobs"]]
        _require(c.read_json(case/"selected_manifest.json")["jobs"]==actual,"selected_checkpoints_changed_before_test")
        calibrators=c.read_json(case/"metrics/calibrators.json")
        _require(calibrators.get("sha256")==c.digest({k:v for k,v in calibrators.items() if k!="sha256"}),"calibrator_semantic_hash_differs")
        receipt_path=case/"evaluation_receipt.json"
        if receipt_path.exists():
            previous=c.read_json(receipt_path)
            if previous.get("status")=="completed":
                identity_keys=("case_id","plan_sha256","protocol_sha256","selected_manifest_sha256","calibrators_file_sha256","calibrators_sha256",
                               "calibrators_frozen_before_test","selected_checkpoints_verified_before_test","final_admissions_file_sha256","stress_plans_sha256","panels")
                _require(previous.get("case_id")==plan["case_id"] and previous.get("plan_sha256")==c.digest(plan)
                         and previous.get("protocol_sha256")==plan["protocol_sha256"] and previous.get("selected_manifest_sha256")==seal["selected_manifest_sha256"]
                         and previous.get("calibrators_file_sha256")==seal["calibrators_sha256"] and previous.get("calibrators_sha256")==calibrators["sha256"]
                         and previous.get("final_admissions_file_sha256")==c.sha(case/"receipts/final_admissions.json")
                         and previous.get("stress_plans_sha256")==c.digest(stress_plans)
                         and all(key in previous for key in identity_keys) and previous.get("eval_id")==c.digest({k:previous[k] for k in identity_keys}),
                         "completed_evaluation_identity_differs")
                path=case/"predictions/test_predictions.csv"
                _require(previous.get("predictions_sha256")==c.sha(path),"completed_evaluation_export_changed")
                stress_files={}
                for name,record in previous.get("stress_prediction_files",{}).items():
                    stress_path=case/"predictions"/("test_"+name+"_predictions.csv")
                    _require(name in stress_plans and c.sha(stress_path)==record["sha256"],"saved_stress_prediction_hash_differs")
                    _require(record.get("scenario_sha256")==stress_plans[name]["scenario_sha256"] and record.get("selected_manifest_sha256")==seal["selected_manifest_sha256"]
                             and record.get("eval_id")==previous["eval_id"] and record.get("protocol_sha256")==plan["protocol_sha256"]
                             and record.get("no_refit") is True and record.get("completed") is True,"saved_stress_receipt_identity_differs")
                    stress_files[name]={**record,"path":str(stress_path)}
                _require(set(stress_files)==set(stress_plans),"completed_stress_predictions_incomplete")
                _backup(case,plan,store,None,"completed_evaluation_reconciled_no_new_queries")
                return {"status":"EVALUATION_ALREADY_COMPLETED_REUSED","evaluation_receipt":str(receipt_path),"prediction_files":{**_saved_predictions(case,seal),"test":{"path":str(path),"sha256":c.sha(path)}},"calibrators":str(case/"metrics/calibrators.json"),"stress_prediction_files":stress_files,"no_new_forwards":True}
        paths,admissions=_final_panels(case,plan,panel_dirs)
        c.write_json(case/"receipts/final_admissions.json",admissions,immutable=True)
        identity={"case_id":plan["case_id"],"plan_sha256":c.digest(plan),"protocol_sha256":plan["protocol_sha256"],
                  "selected_manifest_sha256":seal["selected_manifest_sha256"],"calibrators_file_sha256":seal["calibrators_sha256"],
                  "calibrators_sha256":calibrators.get("sha256"),"calibrators_frozen_before_test":True,"selected_checkpoints_verified_before_test":True,
                  "final_admissions_file_sha256":c.sha(case/"receipts/final_admissions.json"),"stress_plans_sha256":c.digest(stress_plans),
                  "panels":{p:{"admission_sha256":a["admission_sha256"],"test_sha256":a["partition_sha256"]["test"]} for p,a in admissions.items()}}
        eval_id=c.digest(identity)
        evaluation={**identity,"eval_id":eval_id,"sealed":True,"evaluation_sealed":True,"status":"started","test_opening_authorized":True}
        if receipt_path.exists():
            _require(c.read_json(receipt_path)==evaluation,"evaluation_identity_already_used_with_other_inputs")
        else:
            c.write_json(receipt_path,evaluation,immutable=True)
        # This immutable identity and the selected/calibrator seal must be
        # remotely readable before the first test payload is loaded.
        _backup(case,plan,store,admissions,"evaluation_identity_committed_before_first_test_load")
        all_rows=[]
        stress_rows={name:[] for name in stress_plans}
        loaded={}
        disturbed={}
        def partition_for(panel,scenario=None):
            if panel not in loaded:loaded[panel]=data.load_partition(paths[panel],"test",allow_test=True)
            if scenario is None:return loaded[panel]
            key=(panel,scenario)
            if key not in disturbed:disturbed[key]=stress_partition(loaded[panel],stress_plans[scenario],panel)
            return disturbed[key]
        for job in plan["jobs"]:
            all_rows.extend(_cached_final_prediction(case,plan,job,done[job["id"]],store,admissions,eval_id,
                                                    lambda:partition_for(job["panel"])))
            for scenario in stress_plans:
                stress_rows[scenario].extend(_cached_final_prediction(case,plan,job,done[job["id"]],store,admissions,eval_id,
                                                                     lambda name=scenario:partition_for(job["panel"],name),scenario=scenario))
        stress_files={}
        for scenario,rows in stress_rows.items():
            stress_path=case/"predictions"/("test_"+scenario+"_predictions.csv")
            c.write_csv(stress_path,rows)
            stress_files[scenario]={"path":str(stress_path),"sha256":c.sha(stress_path),"scenario_sha256":stress_plans[scenario]["scenario_sha256"],
                                    "selected_manifest_sha256":seal["selected_manifest_sha256"],"eval_id":eval_id,"protocol_sha256":plan["protocol_sha256"],
                                    "no_refit":True,"completed":True}
        path=case/"predictions/test_predictions.csv"
        c.write_csv(path,all_rows)
        completed={**evaluation,"status":"completed","predictions_sha256":c.sha(path),"all_jobs_cached":True,"stress_prediction_files":stress_files,"completed_at_utc":c.now(),"R3_accredited":False,
                   "unobserved_remote_compute_possible":_uncertain_compute(case)}
        c.write_json(receipt_path,completed)
        c.write_json(case/"status.json",{"status":"FINAL_EVALUATED_ANALYSIS_PENDING","eval_id":eval_id,"test_opened":True,"R3_accredited":False})
        backup=_backup(case,plan,store,admissions,"final_evaluation_completed_saved")
        return {"status":"FINAL_EVALUATED_ANALYSIS_PENDING","evaluation_receipt":str(receipt_path),"prediction_files":{**_saved_predictions(case,seal),"test":{"path":str(path),"sha256":c.sha(path)}},"calibrators":str(case/"metrics/calibrators.json"),"stress_prediction_files":stress_files,"backup":backup,"R3_accredited":False}


def _work_path(path, workspace_root):
    path=c.safe(path);workspace=c.safe(workspace_root)
    _require(path.is_relative_to(workspace) and path!=workspace,"recovery_write_outside_authorized_workspace")
    return path


def _preserve_existing(path,workspace_root):
    path=_work_path(path,workspace_root)
    if not path.exists():return None
    retained=c.safe(workspace_root)/"retained"/(path.name+"-"+uuid.uuid4().hex)
    retained.parent.mkdir(parents=True,exist_ok=True)
    if path.is_dir():
        with c.case_lock(path):os.rename(path,retained)
    else:os.rename(path,retained)
    return str(retained)


def _install_recovery_file(target,payload,expected_sha,workspace_root):
    """Install original bytes at the original path, never rebase an identity."""
    target=c.safe(target)
    _require(hashlib.sha256(payload).hexdigest()==expected_sha,"recovery_evidence_payload_hash_differs")
    if target.exists() and target.is_file() and c.sha(target)==expected_sha:return {"path":str(target),"reused":True}
    target=_work_path(target,workspace_root)
    preserved=_preserve_existing(target,workspace_root)
    c.atomic(target,payload,immutable=True)
    return {"path":str(target),"sha256":expected_sha,"preserved_previous":preserved}


def archive_evidence(source_root,case_root,evidence_path,workspace_root):
    """Archive every power result as original bytes, before remote sync."""
    case,plan,_,_=_load_case(source_root,case_root)
    _require(plan["stage"]=="planning","power_evidence_archive_requires_planning_case")
    source=_work_path(evidence_path,workspace_root)
    obj=c.read_json(source)
    _require(obj.get("status") in ("NOT_DIMENSIONED","DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP")
             and all(key in obj for key in ("final_R","final_T","final_B")),"power_result_evidence_required")
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest();relative="receipts/power/"+digest+".json"
    entry={"original_path":str(source),"archive_path":relative,"sha256":digest,"kind":"power_result"}
    with c.case_lock(case):
        c.atomic(case/relative,raw,immutable=True)
        registry_path=case/"receipts/recovery_evidence.json"
        registry=c.read_json(registry_path) if registry_path.exists() else {"version":"original-evidence-locations-1","artifacts":[]}
        _require(registry.get("version")=="original-evidence-locations-1" and isinstance(registry.get("artifacts"),list),"invalid_recovery_evidence_registry")
        if entry not in registry["artifacts"]:registry["artifacts"].append(entry)
        c.write_json(registry_path,registry)
    return {"status":"ORIGINAL_POWER_EVIDENCE_ARCHIVED","archive":relative,"sha256":digest,
            "registry":"receipts/recovery_evidence.json","requires_sync_before_remote_durability_claim":True,"new_forwards":0}


def restore_case(source_root,case_root,store,workspace_root,*,planning_store=None,expected_dependency=None):
    """Restore authenticated state and its dependencies at ORIGINAL paths.

    This stage performs no market request, label construction or model query.
    Metadata-only dataset replay is separate from explicit network reacquisition.
    Existing unlocked directories are retained, not overwritten or discarded.
    """
    workspace=c.safe(workspace_root);target=_work_path(case_root,workspace)
    _,source_identity=_sources(source_root,store.source_commit)
    temporary_parent=workspace/(".restore-stage-"+uuid.uuid4().hex)
    temporary_parent.mkdir(parents=True,exist_ok=False);temporary=temporary_parent/"case"
    preserved=None
    try:
        remote=store.restore(temporary)
        _require(isinstance(remote,dict) and remote.get("remote_verified") is True,"no_authenticated_case_available_for_restore")
        plan=c.read_json(temporary/"plan.json")
        _validate_jobs(plan)
        _require(plan.get("source_manifest_sha256")==source_identity["manifest_sha256"]
                 and c.sha(temporary/"protocol.json")==plan["protocol_sha256"],"restored_source_or_protocol_identity_differs")
        _require(plan.get("original_case_root")==str(target) and plan.get("source_commit")==store.source_commit,
                 "restore_original_case_path_required_no_identity_rebase")
        if expected_dependency is not None:
            _require(plan.get("case_id")==expected_dependency.get("case_id") and c.digest(plan)==expected_dependency.get("plan_sha256")
                     and plan.get("stage")=="planning","restored_planning_dependency_identity_differs")
        dependencies=[]
        for dep in plan.get("dependency_cases",[]):
            _require(plan["stage"]=="final" and planning_store is not None,"planning_store_required_for_final_restore_dependency")
            dep_target=_work_path(dep["case_root"],workspace)
            _require(dep_target!=target and dep.get("source_commit")==store.source_commit,"invalid_or_cyclic_restore_dependency")
            dependencies.append(restore_case(source_root,dep_target,planning_store,workspace,expected_dependency=dep))
        preserved=_preserve_existing(target,workspace)
        target.parent.mkdir(parents=True,exist_ok=True)
        os.rename(temporary,target)
        # Protocol and power evidence paths are part of the immutable protocol;
        # restoring the exact files avoids mutating their JSON or hash identity.
        installed=[]
        protocol_raw=(target/"protocol.json").read_bytes()
        _require(hashlib.sha256(protocol_raw).hexdigest()==plan["protocol_sha256"],"restored_protocol_bytes_differ")
        installed.append(_install_recovery_file(plan["original_protocol_path"],protocol_raw,plan["protocol_sha256"],workspace))
        protocol=c.read_json(target/"protocol.json")
        if plan["stage"]=="final":
            power_spec=protocol["final"]["power_receipt"]
            installed.append(_install_recovery_file(power_spec["path"],(target/"receipts/power.json").read_bytes(),power_spec["sha256"],workspace))
        registry_path=target/"receipts/recovery_evidence.json"
        if registry_path.exists():
            registry=c.read_json(registry_path)
            _require(registry.get("version")=="original-evidence-locations-1","invalid_recovery_evidence_registry")
            # Restore only the most recently archived version at each original
            # location; all earlier versions remain preserved in their archive.
            latest={item["original_path"]:item for item in registry["artifacts"]}
            for original,item in latest.items():
                relative=Path(item["archive_path"])
                _require(not relative.is_absolute() and ".." not in relative.parts and relative.parts[:2]==("receipts","power"),"unsafe_recovery_evidence_archive_path")
                installed.append(_install_recovery_file(original,(target/relative).read_bytes(),item["sha256"],workspace))
        metadata=[]
        for panel in plan["data_panels"]:
            binding=panel.get("recovery_metadata")
            if binding is None:
                _require(plan.get("software_fixture") is True,"real_restored_panel_missing_replay_metadata")
                continue
            bundle=_spec_file(binding,target)
            metadata.append(data.restore_development_metadata(bundle,workspace,expected_bundle_sha256=binding["sha256"]))
        result={"status":"CASE_AND_ORIGINAL_EVIDENCE_RESTORED","case_root":str(target),"case_id":plan["case_id"],
                "plan_sha256":c.digest(plan),"remote":remote,"preserved_existing_case":preserved,
                "dependencies":dependencies,"installed_original_evidence":installed,"dataset_metadata":metadata,
                "raw_market_arrays_restored":False,"market_acquisition_executed":False,"new_forwards":0,
                "next":"Explicitly reacquire original data bytes from their pinned metadata, verify admission/NPZ hashes, then resume."}
        c.write_json(target/"receipts/recovery_installation.json",result)
        return result
    finally:
        shutil.rmtree(temporary_parent,ignore_errors=True)


def sync_case(source_root,case_root,store,*,panel_dirs=None,stage="saved_analysis_and_exports_synced"):
    """Persist saved analysis/exports without any label load or model call."""
    case,plan,_,admissions=_load_case(source_root,case_root,panel_dirs)
    with c.case_lock(case):
        remote=store.status()
        if remote.get("exists"):_receipt(case,plan,remote)
        return {"status":"SAVED_CASE_REMOTE_VERIFIED","backup":_backup(case,plan,store,admissions,stage),"new_forwards":0}


def status_case(source_root,case_root,panel_dirs=None,store=None):
    case,plan,_,_= _load_case(source_root,case_root,panel_dirs)
    complete=sum((case/"jobs"/j["id"]/"done.json").exists() for j in plan["jobs"])
    return {"version":VERSION,"case_id":plan["case_id"],"stage":plan["stage"],"plan_sha256":c.digest(plan),"jobs":len(plan["jobs"]),
            "completion_markers":complete,"completion_markers_are_not_verification":True,"training_sealed":(case/"training_seal.json").exists(),
            "evaluation":c.read_json(case/"evaluation_receipt.json") if (case/"evaluation_receipt.json").exists() else None,
            "remote":store.status() if store is not None else None,"R3_accredited":False}


def export_case(source_root,case_root,output,*,panel_dirs=None,store=None):
    """Export saved derived evidence without forwards or raw/checkpoint data."""
    case,plan,_,admissions=_load_case(source_root,case_root,panel_dirs)
    files={}
    for folder in ("predictions","metrics","receipts"):
        if (case/folder).exists():
            for path in (case/folder).rglob("*"):
                if path.is_file() and path.suffix in (".json",".csv"):
                    files[path.relative_to(case).as_posix()]=path
    for name in ("status.json","selected_manifest.json","training_seal.json","evaluation_receipt.json","analysis_receipt.json"):
        if (case/name).exists():files[name]=case/name
    output=c.safe(output)
    _require(not output.is_relative_to(case),"export_destination_must_be_outside_case")
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):archive.writestr(name,path.read_bytes())
    c.atomic(output,raw.getvalue())
    if store is not None:_backup(case,plan,store,admissions,"saved_results_exported_no_new_queries")
    return {"status":"SAVED_RESULTS_EXPORTED","path":str(output),"sha256":c.sha(output),"files":len(files),"new_forwards":0,"raw_inputs_included":False}

# Short names for API clients; all resolve to the same guarded implementation.
status = status_case
export = export_case
