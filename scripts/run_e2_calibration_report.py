"""Report the two preregistered calibration groups without pooling seed panels.

The historical input is the immutable published section-30 gzip. The new input
is a hash-explicit optimized-variance summary and its durable worker/replay
receipts. No fitting or simulation occurs on import. The only MC implementation
admitted is e2_calibrated_planning.py SHA256
d64e3f872d94419f8a5010d00a0b55759d604b5e5ee044842630586b5632e5ba,
frozen in source commit d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

from run_e2_reference_variance import json_new, median_epoch_count, replay_vectors

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-calibration-report-v1"
CALIBRATION_SOURCE_SHA256 = "d64e3f872d94419f8a5010d00a0b55759d604b5e5ee044842630586b5632e5ba"
COMPUTE_SOURCE_COMMIT = "d24c7d1f78038284f2bad179f6a7c8cb9cc0bdb2"
COMPUTE_REGISTRATION_SHA256 = "3b54ac44f5aed956ac98fbc31c95dfcb70c1116b6cca65a8269ff11cd843ac91"
OLD_REGISTRATION_COMMIT = "cf68db00f4d7c3626bdb517bf922265a3b229ea4"
OLD_REGISTRATION_SHA256 = "cc10ddba82f0c59cfcd8caf70e713481ec0701e71860862ed13ead0df0bf4797"
OLD_GZIP_SHA256 = "dae4c8f96be104014ee5c21cf422a4617c5e3047b18ef04c0701123be44eeb35"
OLD_DECODED_SHA256 = "ce80590b807773c9dd97d5f1daed5851d4793123d7e918060ae144e7906b27fe"
OLD_MODELS = ("CNN-1D + LSTM + atención temporal", "GRU32")
NEW_MODELS = ("TimeMixer",) + OLD_MODELS
OLD_SEEDS = tuple(range(271901,271906))
NEW_SEEDS = tuple(range(273001,273006))
OLD_VERSION = "e2-preselection-reference-variance-v1"
NEW_VERSION = "e2-optimized-preselection-reference-variance-v1"
HISTORICAL_PROTOCOL_SHA256 = "e8da0f0098acd24a58664b83d0ba499b43aa9d597a36c52da6b3325c2c3c7bec"
DATA_SOURCE_SHA256 = {
    "results/eda_development/daily_market_development.csv":"bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171",
    "results/eda_development/daily_volatility_development.csv":"20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a"}
CORE_SOURCE_SHA256 = {
    "scripts/e2_development_models.py":"5f9fd7e57dfb63d9f6fb3cb97dc6a7fa47aeb3ccad29c0836d73f70a87978d90",
    "scripts/e2_staged_training.py":"b66ac8d2beb3748f4934e55e99514f53c52eada4a4dbd2e4b597d080c9f4722b",
    "scripts/e2_temporal_training.py":"3e62a12135b47414ac11038f5bd931076c2c12c7fdd8edcbd3b46eae4dc90ac9",
    "scripts/e2_training_models.py":"46be0644e0ed9d966163b17f171092e81b6a82dd734efc7954d6104f5ee5360b"}
RUNNER_SOURCE_SHA256 = {
    "old":"012c0db66a4cb03ae7c50e4e80ffc369961e4740eec468d359485c64e9a9b501",
    "new":"7488d7aeb436ffc8fdbee388bd2d05bfe7ea5df0ba3ae1f9456ac67e79f79923"}
EXPECTED_DATES = tuple((datetime(2024,11,8,tzinfo=timezone.utc)+timedelta(days=i)).isoformat()
                       for i in range(53))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _hex(value,length=64):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{"+str(length)+r"}",value) is not None


def _read_regular(path,expected_sha256):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not _hex(expected_sha256):
        raise ValueError("A regular hash-explicit input is required")
    raw = path.read_bytes()
    if _digest(raw) != expected_sha256:
        raise ValueError("Input SHA256 differs from its immutable expectation: "+path.name)
    return raw


def _dates(values):
    if not isinstance(values,list) or len(values) != 53:
        raise ValueError("Exactly53 registered daily dates are required")
    canonical = []
    for value in values:
        stamp = datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        canonical.append(stamp.astimezone(timezone.utc).isoformat())
    if tuple(canonical) != EXPECTED_DATES:
        raise ValueError("Evaluation must be every UTC day2024-11-08..2024-12-30 in order")
    return canonical


def _values(values):
    if (not isinstance(values,list) or len(values) != 53 or any(type(value) not in (int,float)
            or not math.isfinite(value) or value < 0 for value in values)):
        raise ValueError("All53 original-unit values must be finite and nonnegative")
    return [float(value) for value in values]


def _source_provenance(worker,group):
    if group == "old":
        sources = worker.get("source_sha256",{})
        runner = "scripts/run_e2_reference_variance.py"
    else:
        sources = worker.get("source_sha256_before",{})
        if (worker.get("source_hashes_unchanged") is not True
                or sources != worker.get("source_sha256_after")):
            raise ValueError("New worker sources changed during fitting")
        runner = "scripts/run_e2_optimized_variance.py"
    expected = {**CORE_SOURCE_SHA256,runner:RUNNER_SOURCE_SHA256[group]}
    if any(sources.get(path) != sha for path,sha in expected.items()):
        raise ValueError("Worker model/trainer source provenance differs from the frozen recipe")
    receipt = worker.get("data_receipt",{})
    if (receipt.get("admitted_end") != "2024-12-30"
            or receipt.get("source_protocol_sha256") != HISTORICAL_PROTOCOL_SHA256
            or receipt.get("source_sha256") != DATA_SOURCE_SHA256
            or receipt.get("source_values_parsed_only_through_admission") is not True
            or receipt.get("external_2025_2026_files_opened") != 0):
        raise ValueError("Worker development data provenance differs from registration")


def _worker_provenance(worker,name,seed,phase,group,commit,sha):
    expected_phase = "epoch_selection" if phase == "selection" else "fresh_refit"
    expected_version = OLD_VERSION if group == "old" else NEW_VERSION
    if (not isinstance(worker,dict) or worker.get("status") != "completed"
            or worker.get("version") != expected_version or worker.get("model") != name
            or worker.get("seed") != seed or worker.get("phase") != expected_phase
            or worker.get("registration_commit") != commit
            or worker.get("registration_file_sha256") != sha):
        raise ValueError("Completed worker identity or registration differs from its panel row")
    if (worker.get("fresh_optimizer") is not True or any(worker.get(key) is not False for key in
            ("hpo_executed","nas_search_executed","selected_hpo_recipe","weights_reused",
             "final_2025_2026_access","replacement_attempt"))):
        raise ValueError("Only fresh fixed-reference non-HPO development workers are admitted")
    if group == "new":
        policy = worker.get("runtime_policy",{})
        if (policy.get("registration_commit") != COMPUTE_SOURCE_COMMIT
                or policy.get("registration_sha256") != COMPUTE_REGISTRATION_SHA256
                or worker.get("torch_num_threads") != policy.get("threads")
                or worker.get("torch_num_interop_threads") != 1):
            raise ValueError("New worker runtime proof belongs to another registration")
    _source_provenance(worker,group)
    metadata = worker.get("model_metadata",{})
    if (metadata.get("model") != name or metadata.get("initialization_seed") != seed
            or metadata.get("pilot_fixed_not_hpo") is not True or metadata.get("weights_reused") is not False
            or metadata.get("input_shape") != [7,5]):
        raise ValueError("Fixed-reference model metadata differs from its identity")
    pattern = metadata.get("pilot_pattern")
    if name == "GRU32":
        if pattern is not None: raise ValueError("GRU32 cannot acquire a NAS architecture")
    elif (not isinstance(pattern,dict) or pattern.get("widths") != [64,64,64]
            or pattern.get("dropout_pattern") != [0,1,0] or pattern.get("is_reference") is not True):
        raise ValueError("Only the declared reference NAS architecture is admitted")
    train_dates = worker.get("training_split",{}).get("dates",[])
    cutoff = "2024-06-25" if phase == "selection" else "2024-10-31"
    if (not train_dates or str(train_dates[-1])[:10] != cutoff
            or any(str(day)[:10] > cutoff for day in train_dates)):
        raise ValueError("Training dates cross the registered support cutoff")
    checkpoint = worker.get("checkpoint",{})
    if (checkpoint.get("safe_load_weights_only") is not True
            or checkpoint.get("optimizer_state_saved") is not False
            or not _hex(checkpoint.get("sha256")) or not _hex(checkpoint.get("state_sha256"))):
        raise ValueError("Safe inference checkpoint evidence is required")
    if phase == "selection":
        validation = worker.get("validation_split") or {}
        days = validation.get("dates",[])
        if (not days or str(days[0])[:10] != "2024-07-03" or str(days[-1])[:10] != "2024-09-23"
                or worker.get("outer_labels_used_for_epoch_selection") is not False):
            raise ValueError("Epoch selection validation differs from registration")
    elif (worker.get("validation_split") is not None
            or worker.get("epoch_selection_inputs_passed_to_refit") is not False
            or worker.get("observation_after_durable_prediction") is not True
            or worker.get("frozen_fingerprint_invariant") is not True):
        raise ValueError("Refits require frozen prediction-before-observation evidence")


def extract_panel(summary,group="old",*,registration_sha256=COMPUTE_REGISTRATION_SHA256,
                  registration_commit=COMPUTE_SOURCE_COMMIT):
    """Accept an entire fixed group, never the surviving seeds/models of it."""
    if group not in ("old","new"): raise ValueError("Choose separate old or new calibration group")
    names,seeds = (OLD_MODELS,OLD_SEEDS) if group == "old" else (NEW_MODELS,NEW_SEEDS)
    commit,sha = ((OLD_REGISTRATION_COMMIT,OLD_REGISTRATION_SHA256) if group == "old"
                  else (registration_commit,registration_sha256))
    if group == "new" and (commit != COMPUTE_SOURCE_COMMIT or sha != COMPUTE_REGISTRATION_SHA256):
        raise ValueError("New panel must refer to the frozen compute registration commit and bytes")
    if (not isinstance(summary,dict) or summary.get("version") != (OLD_VERSION if group == "old" else NEW_VERSION)
            or summary.get("registration_commit") != commit or summary.get("registration_file_sha256") != sha):
        raise ValueError("Panel identity differs from its immutable registered campaign")
    if any(summary.get(key) is not False for key in
           ("HPO_executed","NAS_search_executed","final_2025_2026_access","final_R_set","final_power_claimed")):
        raise ValueError("A fixed-reference development panel without final-inference claims is required")
    if summary.get("replacement_attempts") != 0 or summary.get("preselection_reference_only") is not True:
        raise ValueError("Replaced attempts or selected recipes cannot enter this pilot analysis")
    if not isinstance(summary.get("models"),dict) or set(summary["models"]) - set(NEW_MODELS):
        raise ValueError("Unexpected model labels in the pilot summary")
    panel = {"status":"ready","group":group,"model_names":list(names),"seed_ids":list(seeds),
        "dates":list(EXPECTED_DATES),"predictions":[],"truths":None,
        "excluded_models":["TimeMixer"] if group == "old" else [],"blocking_reasons":[],
        "selection_seed_survivors_used":False,"partial_model_survivors_used":False,
        "registration_commit":commit,"registration_sha256":sha}
    for name in names:
        model = summary["models"].get(name,{})
        for phase in ("selection","refit"):
            rows = model.get(phase,[])
            if not isinstance(rows,list): raise ValueError("Each phase must contain an explicit row list")
            ids = [row.get("seed") for row in rows]
            if len(set(ids)) != len(ids) or any(type(seed) is not int or seed not in seeds for seed in ids):
                raise ValueError("Duplicate or foreign seed identities in the panel")
            if len(rows) != 5 or set(ids) != set(seeds) or any(row.get("status") != "completed" for row in rows):
                panel["blocking_reasons"].append(name+":"+phase+": five complete registered attempts required")
    if group == "new" and summary.get("paired_3x5_complete") is not True:
        panel["blocking_reasons"].append("New primary three-model campaign is incomplete")
    if panel["blocking_reasons"]:
        panel.update(status="blocked_incomplete_panel",predictions=None,truths=None)
        return panel
    if group == "new" and (summary.get("source_hashes_unchanged") is not True
            or summary.get("source_sha256_before") != summary.get("source_sha256_after")):
        raise ValueError("New campaign source snapshot changed")
    for name in names:
        model = summary["models"][name]
        epochs = median_epoch_count(model["selection"],seeds=seeds)
        if summary.get("fixed_epoch_counts",{}).get(name) != epochs:
            raise ValueError("Refit epoch count differs from its frozen five-seed median")
        predictions = {}
        for phase in ("selection","refit"):
            for row in model[phase]:
                if row.get("model") != name or row.get("phase") != ("epoch_selection" if phase == "selection" else "fresh_refit"):
                    raise ValueError("Panel row belongs to another model or phase")
                worker = row.get("worker_profile")
                _worker_provenance(worker,name,row["seed"],phase,group,commit,sha)
                if group == "new" and (worker["source_sha256_before"] != summary["source_sha256_before"]
                        or worker.get("runtime_policy") != summary.get("runtime_policy")):
                    raise ValueError("Workers must use the one frozen campaign source/runtime policy")
                if phase == "refit":
                    if worker.get("fixed_epoch_count") != epochs:
                        raise ValueError("A refit changed its frozen epoch count")
                    vectors = worker.get("evaluation",{})
                    _dates(vectors.get("dates"))
                    values,truth = _values(vectors.get("predictions")),_values(vectors.get("truths"))
                    if panel["truths"] is None: panel["truths"] = truth
                    if truth != panel["truths"]: raise ValueError("All models and seeds must share identical observations")
                    predictions[row["seed"]] = values
        panel["predictions"].append([predictions[seed] for seed in seeds])
    return panel


def load_frozen_analysis():
    """Verify source bytes before importing any NumPy/SciPy simulation code."""
    path = ROOT/"scripts/e2_calibrated_planning.py"
    _read_regular(path,CALIBRATION_SOURCE_SHA256)
    spec = importlib.util.spec_from_file_location("_e2_frozen_calibrated_planning",path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_group(panel,delta,*,validate_only=False):
    """Run only the registered200k/200k/200k, four-R design for a complete group."""
    if panel["status"] != "ready": return {**panel,"analysis":None,"simulations_executed":False}
    if validate_only:
        return {**{key:value for key,value in panel.items() if key not in ("predictions","truths")},
                "analysis":None,"simulations_executed":False,"validation_only":True,
                "panel_values_json_sha256":_digest(json.dumps(
                    {key:panel[key] for key in ("predictions","truths")},
                    sort_keys=True,allow_nan=False).encode())}
    module = load_frozen_analysis()
    config = module.CalibrationConfig()
    analysis = module.calibrated_power_analysis(panel["predictions"],panel["truths"],delta,m=153,
        dates=panel["dates"],model_names=panel["model_names"],seed_ids=panel["seed_ids"],config=config)
    if analysis.get("registered_design_match") is not True:
        raise RuntimeError("Calibration result does not match the preregistered MC design")
    return {**{key:value for key,value in panel.items() if key not in ("predictions","truths")},
            "analysis":analysis,"simulations_executed":True,"calibration_source_sha256":CALIBRATION_SOURCE_SHA256}


def _contained(root,relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts: raise ValueError("Receipt path escapes its campaign")
    path = Path(root)/relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Receipt symlink/path escapes its campaign")
    return path


def verify_new_replay_files(summary,root):
    """Bind embedded completed receipts to the durable worker and replay files."""
    checked = []
    for name in NEW_MODELS:
        for phase in ("selection","refit"):
            for row in summary["models"][name][phase]:
                path = _contained(root,row["worker_result_file"])
                worker = json.loads(path.read_text(encoding="utf-8"))
                if worker != row["worker_profile"]: raise ValueError("Worker receipt differs from the frozen summary")
                entry = {"model":name,"seed":row["seed"],"phase":phase,"worker_sha256":_digest(path.read_bytes())}
                checkpoint = worker["checkpoint"]
                checkpoint_path = _contained(path.parent,checkpoint["relative_path"])
                checkpoint_raw = _read_regular(checkpoint_path,checkpoint["sha256"])
                entry.update(checkpoint_sha256=checkpoint["sha256"],checkpoint_bytes=len(checkpoint_raw),
                             safe_load_assertion_from_frozen_runner=True)
                if phase == "refit":
                    events = _contained(path.parent,"replay/events.jsonl")
                    _read_regular(events,worker["replay_ledger_sha256"])
                    vectors = replay_vectors(events,name+"/seed"+str(row["seed"]),list(EXPECTED_DATES))
                    if vectors != worker["evaluation"]: raise ValueError("Durable replay differs from the paired evaluation")
                    entry["replay_sha256"] = worker["replay_ledger_sha256"]
                checked.append(entry)
    return checked


def run_report(old_variance_gzip,registration_path,output,*,new_summary=None,
               expected_new_summary_sha256=None,validate_only=False):
    """Create separate machine-readable group reports; never refit predictors."""
    output = Path(output)
    if output.exists() or output.is_symlink(): raise FileExistsError("Calibration output must be new")
    _read_regular(ROOT/"scripts/e2_calibrated_planning.py",CALIBRATION_SOURCE_SHA256)
    for relative,sha in {**CORE_SOURCE_SHA256,
            "scripts/run_e2_reference_variance.py":RUNNER_SOURCE_SHA256["old"],
            "scripts/run_e2_optimized_variance.py":RUNNER_SOURCE_SHA256["new"]}.items():
        _read_regular(ROOT/relative,sha)
    registered = json.loads(_read_regular(registration_path,COMPUTE_REGISTRATION_SHA256))
    if registered.get("version") != "e2-compute-calibration-registration-v1":
        raise ValueError("Frozen compute calibration registration required")
    delta = registered["calibration"]["delta"]
    raw_old = _read_regular(old_variance_gzip,OLD_GZIP_SHA256)
    decoded_old = gzip.decompress(raw_old)
    if _digest(decoded_old) != OLD_DECODED_SHA256: raise ValueError("Historical gzip payload differs from publication")
    old_panel = extract_panel(json.loads(decoded_old),"old")
    new_panel = {"status":"not_available","group":"new","analysis":None,"simulations_executed":False,
                 "blocking_reasons":["The new optimized variance summary was not supplied"]}
    replay_checks = []
    if new_summary is not None:
        if expected_new_summary_sha256 is None: raise ValueError("The new summary needs its explicit SHA256")
        summary = json.loads(_read_regular(new_summary,expected_new_summary_sha256))
        new_panel = extract_panel(summary,"new")
        if new_panel["status"] == "ready":
            replay_checks = verify_new_replay_files(summary,Path(new_summary).parent)
    # All admission/provenance checks precede either large simulation group.
    output.mkdir(parents=True)
    request = {"version":VERSION,"registration_commit":COMPUTE_SOURCE_COMMIT,
        "calibration_source_commit":COMPUTE_SOURCE_COMMIT,
        "calibration_source_sha256":CALIBRATION_SOURCE_SHA256,"registration_sha256":COMPUTE_REGISTRATION_SHA256,
        "old_variance_gzip_sha256":OLD_GZIP_SHA256,"old_decoded_sha256":OLD_DECODED_SHA256,
        "new_summary_sha256":expected_new_summary_sha256,"delta":delta,"groups_pooled":False,
        "new_replay_receipts":replay_checks,"model_fits_executed":0,"final_R_set":False,
        "validate_only":validate_only}
    json_new(output/"request.json",request)
    groups = {}
    for label,panel in (("old",old_panel),("new",new_panel)):
        result = run_group(panel,delta,validate_only=validate_only)
        json_new(output/(label+"_calibration.json"),result)
        groups[label] = result
    report = {**request,"status":"separate_groups_validated_only" if validate_only else "separate_conditional_groups_reported","groups":groups,
        "final_R":None,"achieved_final_power":None,
        "limits":["Historical two-model diagnostic and new three-model pilot are separate groups.",
                  "No incomplete seed set or surviving new-model subset is analyzed.",
                  "The frozen module reports conditional empirical-DGP Monte Carlo guarantees only."]}
    json_new(output/"report.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-variance-gzip",type=Path,required=True)
    parser.add_argument("--registration",type=Path,required=True)
    parser.add_argument("--new-summary",type=Path)
    parser.add_argument("--expected-new-summary-sha256")
    parser.add_argument("--validate-only",action="store_true",
                        help="Verify provenance, complete panels and replay receipts without importing MC dependencies")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = run_report(args.old_variance_gzip,args.registration,args.output,
        new_summary=args.new_summary,expected_new_summary_sha256=args.expected_new_summary_sha256,
        validate_only=args.validate_only)
    print(json.dumps({"status":result["status"],"groups":{key:value["status"] for key,value in result["groups"].items()},
                      "final_R":None},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
