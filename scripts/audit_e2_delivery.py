"""Audit the published E2 delivery using Python's standard library only.

This command reconstructs the arithmetic of the frozen benchmark and verifies
bound reports. It never fits models, generates forecasts, downloads files or
claims to repeat the producing A100 tensor comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

VERSION = "e2-delivery-readonly-audit-v1"
INPUT_SHA = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
A2_SHA = "879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1"
REPRODUCTION_SUMMARY_SHA = "c430ac4ff165370382a98921fb31e7af6dc34c04dfbba64a81b992f95b8ff340"
RESULTS_COMMIT = "d604ede6bf0978e14a1b1d7bafaae6250fb8ed07"
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
TOP10 = NEURAL[:10]
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
SEEDS = tuple(range(271801, 271807))
MANIFEST_FOLDERS = (
    "e2_benchmark_final_20261113", "e2_reproduction_ablations_20261113",
    "e2_final_audit_inputs_20261114", "e2_f2a_audit_20261113", "e2_f2a_s8_20261113",
    "e2_flops_final_audit_20261114", "e2_error_ceiling_final_audit_20261114",
    "e2_temporal_final_audit_20261114", "e2_literature_final_audit_20261114",
)
SUPPLEMENT_FOLDERS = ("e2_calibration_ensemble_20261114",
                      "e2_conditional_calibration_20261114",
                      "e2_literature_delivery_closure_20261114_v2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def verify_manifest(folder):
    """Reject altered/missing files and paths escaping the manifest directory."""
    folder = Path(folder).resolve()
    manifest = read_json(folder/"manifest.json")
    entries = manifest.get("files") or manifest.get("files_sha256") or manifest.get("output_sha256")
    require(isinstance(entries, dict) and bool(entries), "Empty/unsupported manifest: "+str(folder))
    for name, descriptor in entries.items():
        path = (folder/name).resolve()
        require(path.is_relative_to(folder), "Manifest path escapes directory: "+name)
        require(path.is_file(), "Manifest file missing: "+str(path))
        expected = descriptor["sha256"] if isinstance(descriptor, dict) else descriptor
        require(digest(path) == expected, "SHA256 mismatch: "+str(path))
        if isinstance(descriptor, dict) and "bytes" in descriptor:
            require(path.stat().st_size == descriptor["bytes"], "Byte count mismatch: "+str(path))
    return {"folder": folder.name, "manifest_sha256": digest(folder/"manifest.json"),
            "verified_files": len(entries)}


def reconstruct_panel(payload):
    dates = [(date(2025, 1, 1)+timedelta(days=i)).isoformat() for i in range(608)]
    require(payload["dates"] == dates, "The common consecutive 608-day calendar changed")
    require(payload["mask"] == [True]*608 and all(type(x) is bool for x in payload["mask"]),
            "The original common eligibility mask changed")
    y = payload["y_true"]
    require(len(y) == 608 and all(math.isfinite(v) for v in y), "Invalid frozen targets")
    scores, by_model = {}, defaultdict(list)
    for row in payload["predictors"]:
        key = row["model"], str(row["seed"])
        require(key not in scores, "Duplicate frozen predictor")
        p = row["prediction_original"]
        require(len(p) == 608 and all(math.isfinite(v) for v in p), "Invalid frozen forecasts")
        score = math.sqrt(math.fsum((a-b)**2 for a, b in zip(p, y))/608)
        scores[key] = score
        by_model[row["model"]].append((str(row["seed"]), score))
        require(len(row["calibration_dates"]) == len(row["calibration_predictions"]) ==
                len(row["calibration_targets"]) == 53, "Incomplete saved calibration panel")
    require(set(by_model) == set(NEURAL+CLASSICAL) and len(scores) == 81, "Incomplete 16-system panel")
    for model in NEURAL:
        require({s for s, _ in by_model[model]} == {str(s) for s in SEEDS}, "Six neural seeds changed")
    for model in CLASSICAL:
        require([s for s, _ in by_model[model]] == ["deterministic"], "Classical seed replicated or changed")
    means = {model: math.fsum(v for _, v in rows)/len(rows) for model, rows in by_model.items()}
    return scores, means


def verify_A2(rows, means, tolerance=5e-15):
    require(len(rows) == 16 and {r["model"] for r in rows} == set(means), "A2 system set changed")
    differences = {}
    for row in rows:
        model = row["model"]
        require(int(row["n_common_dates"]) == 608, "A2 calendar denominator changed")
        require(int(row["R"]) == (6 if model in NEURAL else 1), "A2 replicate count changed")
        difference = abs(float(row["RMSE"])-means[model])
        require(difference <= tolerance, "A2 is not mean individual-seed RMSE: "+model)
        differences[model] = difference
    return differences


def verify_recorded_reproduction(folder, original_scores):
    folder = Path(folder)
    rows, ablations = read_csv(folder/"reproduction.csv"), read_csv(folder/"ablation_fits.csv")
    status, source = read_json(folder/"status.json"), read_json(folder/"source_evidence.json")
    require(len(rows) == 81 and len(ablations) == 18 and status["completed"] == 99,
            "The completed reproduction/ablation counts changed")
    keys = set()
    for row in rows:
        key = row["model"], row["seed"]
        require(key in original_scores and key not in keys, "Incomplete/duplicate reproduction")
        keys.add(key)
        require(row["reproduction_passed"] == "True" and float(row["max_abs_calibration53"]) == 0 and
                float(row["max_abs_test608"]) == 0, "Recorded reproduction did not pass")
        require(abs(float(row["rmse_original"])-original_scores[key]) <= 5e-15,
                "Recorded reproduction RMSE differs from frozen forecast arithmetic")
        if row["model"] in NEURAL:
            require(row["state_tensor_bitwise_equal"] == "True", "Recorded neural state comparison failed")
        else:
            require(row["classical_parameters_exactly_equal"] == "True", "Recorded classical comparison failed")
    conditions = defaultdict(set)
    for row in ablations:
        require(row["model"] == "LSTM-autoencoder con predictor", "Unexpected ablated model")
        require(row["HPO_repeated"] == "False" and row["same_frozen_remaining_phase_epochs"] == "True",
                "Ablation changed the frozen remaining-phase schedule")
        require(row["seed"] not in conditions[row["condition"]], "Duplicate ablation seed")
        conditions[row["condition"]].add(row["seed"])
    require(len(conditions) == 3 and all(s == {str(v) for v in SEEDS} for s in conditions.values()),
            "Three six-seed ablations required")
    jobs = source["verified_jobs"]
    require(source["results_commit"] == RESULTS_COMMIT, "Recorded reproduction source commit changed")
    require(len(jobs) == 99 and len({r["attempt_id"] for r in jobs}) == 99,
            "Missing/duplicate 99-job provenance")
    require(Counter(r["kind"] for r in jobs) == {"reproduction": 81, "ablation": 18},
            "Provenance kind counts changed")
    require({r["attempt_id"] for r in jobs} == {r["attempt_id"] for r in rows+ablations},
            "Provenance does not bind the published fits")
    manifest, receipt = read_json(folder/"producer_manifest.json"), read_json(folder/"producer_receipt.json")
    require(manifest["receipt_sha256"] == hashlib.sha256(canonical(receipt)).hexdigest(),
            "Producer summary receipt binding changed")
    raw_summary = folder/"producer_summary.json"
    require(digest(raw_summary) == REPRODUCTION_SUMMARY_SHA, "Frozen producer summary changed")
    require(receipt["artifacts_sha256"] == {"summary.json": digest(raw_summary)},
            "Producer summary artifact binding changed")
    budget = status["budget"]
    require(budget["approved_total_hours"] == 84 and not budget["budget_reset"] and
            not budget["unresolved_reservations"], "Final campaign budget state changed")
    return {"recorded_reproduction_fits": 81, "recorded_ablation_fits": 18,
            "recorded_max_prediction_difference": 0,
            "source_results_commit": source["results_commit"],
            "producer_tensor_checks_reexecuted_by_this_command": False,
            "original_budget_closed": True}


def verify_primary_refinement(folder):
    folder = Path(folder)
    rows = read_csv(folder/"primary_refinement_20000.csv")
    expected = {(m, b) for m in TOP10 for b in ("GRU32", "Persistence")}
    require(len(rows) == 20 and {(r["model"], r["baseline"]) for r in rows} == expected,
            "Primary family changed")
    for row in rows:
        require(int(row["draws"]) == 20000 and int(row["block_days"]) == 14,
                "Refinement settings changed")
        require(float(row["delta"]) == 0.00025772842664981106 and
                row["original_result_replaced"] == "False", "Primary margin/history changed")
        require(float(row["p_Holm"]) == 1 and row["reject_at_FWER05"] == "False",
                "Published primary refinement decision changed")
    checks = read_csv(folder/"original_4999_exact_checks.csv")
    require(len(checks) == 20 and all(r["exact_original_4999_reproduction"] == "True" for r in checks),
            "Original Monte Carlo verification missing")
    return {"primary_family": 20, "refinement_draws": 20000,
            "refinement_Holm_rejections": 0, "bootstrap_reexecuted_by_this_command": False}


def audit(root, require_supplements=False):
    started = time.perf_counter()
    root = Path(root).resolve()
    static = root/"book_e2/_static"
    manifests = [verify_manifest(static/name) for name in MANIFEST_FOLDERS]
    missing_supplements = []
    for name in SUPPLEMENT_FOLDERS:
        if (static/name/"manifest.json").is_file():
            manifests.append(verify_manifest(static/name))
        else:
            missing_supplements.append(name)
    require(not require_supplements or not missing_supplements,
            "Required delivery supplements missing: "+", ".join(missing_supplements))
    panel_path = static/"e2_final_audit_inputs_20261114/e2_f2a_input.json"
    a2_path = static/"e2_benchmark_final_20261113/benchmark_A2.csv"
    require(digest(panel_path) == INPUT_SHA and digest(a2_path) == A2_SHA, "Frozen identity changed")
    scores, means = reconstruct_panel(read_json(panel_path))
    differences = verify_A2(read_csv(a2_path), means)
    reproduction = verify_recorded_reproduction(static/"e2_reproduction_ablations_20261113", scores)
    primary = verify_primary_refinement(static/"e2_temporal_final_audit_20261114")
    return {"version": VERSION, "status": "verified", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "readonly_frozen_delivery_arithmetic_and_bound_reports_not_new_model_reproduction",
            "runtime": {"python": platform.python_version(), "executable": sys.executable,
                        "platform": platform.platform(), "dependencies": "Python standard library only"},
            "manifest_checks": manifests, "verified_manifest_files": sum(m["verified_files"] for m in manifests),
            "required_supplements_enforced": require_supplements,
            "missing_optional_supplement_manifests": missing_supplements,
            "frozen_prediction_sha256": INPUT_SHA, "original_A2_sha256": A2_SHA,
            "calendar_days": 608, "original_predictors": 81, "systems": 16,
            "recomputed_max_A2_RMSE_difference": max(differences.values()),
            "per_model_A2_RMSE_difference": differences,
            "recorded_model_reproduction": reproduction, "recorded_primary_refinement": primary,
            "methodological_approval_claimed": False, "new_model_fits": 0,
            "new_model_inference": False, "network_access": False,
            "historical_blinding_recovered": False, "a_priori_delta_g_recovered": False,
            "elapsed_seconds": time.perf_counter()-started}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-supplements", action="store_true",
                   help="Fail if the final conditional-calibration, ensemble or literature manifest is missing")
    a = p.parse_args()
    result = audit(a.root, a.require_supplements)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("status", "systems", "original_predictors", "calendar_days",
                        "verified_manifest_files", "recomputed_max_A2_RMSE_difference", "new_model_fits")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
