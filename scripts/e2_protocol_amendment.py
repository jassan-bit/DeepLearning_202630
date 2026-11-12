"""Check the registered E2 amendment without loading prices or starting fits.

The registration receipt binds the protocol and implementation bytes. A separate
Git receipt points to the immutable registration commit, avoiding a circular
self-hash. Neither a protocol nor a JSON freeze attestation proves that private
weights exist; execution and material verification remain explicit pending gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

try:
    from . import e2_amended_calendar as calendar
    from . import e2_amended_data_guard as data_guard
except ImportError:
    import e2_amended_calendar as calendar
    import e2_amended_data_guard as data_guard

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = "experiments/e2_protocol_amendment_20261112"
PROTOCOL_PATH = DIRECTORY + "/protocol.json"
REGISTRATION_PATH = DIRECTORY + "/registration.json"
GIT_RECEIPT_PATH = DIRECTORY + "/git_registration.json"
PROTOCOL_ID = "e2-resource-temporal-amendment-v1"
PARENT = "75dff4ffea3c5d5a275eab6a0ac3872ea1147346"
DELTA = .00025772842664981106
_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _path(root, relative):
    root, relative = Path(root).resolve(), Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts, "Use a repository-relative metadata path")
    value = root / relative
    _require(value.resolve().is_relative_to(root) and not any(p.is_symlink() for p in (value, *value.parents) if p.is_relative_to(root)),
             "Registered metadata cannot escape the checkout or use symlinks")
    return value


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _json(root, path):
    return json.loads(_path(root, path).read_text(encoding="utf-8"))


def _instant(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "Registration times must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def read_protocol(root=ROOT):
    value = _json(root, PROTOCOL_PATH)
    validate_protocol(value, root=root)
    return value


def validate_protocol(value, *, root=ROOT):
    _require(value.get("protocol_id") == PROTOCOL_ID and value.get("parent_source_commit") == PARENT,
             "Use the versioned prospective amendment and its actual parent")
    _instant(value["prepared_at_utc"])
    catalogue = _json(root, "experiments/e2_published_architectures_20261112/catalogue.json")
    _require(value.get("systems") == catalogue["systems"], "The mandatory catalogue must contain the same sixteen systems")
    neural = [row["model_id"] for row in catalogue["systems"] if row["role"] != "classical_comparator"]
    top10 = [row["model_id"] for row in catalogue["systems"] if row["role"] == "Top10"]
    _require(value.get("neural_systems") == neural and len(neural) == 13, "Thirteen neural systems are required")
    _require(value.get("classical_systems") == ["Persistence", "SeasonalNaive", "ARIMA"], "Only the three required comparators belong to this scope")
    calendar.compile_calendar(value)
    _require(value.get("contract", {}).get("delta") == DELTA
             and value["contract"]["input_shape"] == [7, 5], "Preserve the D7 and practical-margin contract")
    hpo = value["hpo"]
    _require(hpo.get("trials") == 18 and hpo.get("optimization_seeds") == [271201, 271202]
             and hpo.get("max_epochs") == 200 and hpo.get("patience_epochs") == 20,
             "Do not silently change the paired HPO opportunities or stopping caps")
    _require(hpo.get("selection_scope") == "LOCAL at each registered anchor, never pooled over later anchors",
             "HPO cannot use future anchor outcomes")
    nas = value["nas"]
    _require(nas.get("models") == ["TiDE"] and nas.get("origin") == "2024-01-01"
             and nas.get("repeated_per_outer_fold") is False, "Only one initial TiDE NAS is registered")
    _require(nas.get("phase1_architectures") == 1000 and nas.get("phase1_seed") == 271301
             and nas.get("phase1_epochs") == 50 and nas.get("phase2_epochs") == 200
             and nas.get("phase2_seeds") == [271401, 271402, 271403, 271404, 271405],
             "Keep the full NAS domain, two phases and distinct confirmation seeds")
    _require(nas.get("k_options") == [20, 40] and nas.get("k_before_pilot") is None,
             "A pilot-derived k must not be invented at registration")
    _require(value.get("workload_upper_bounds") == calendar.selection_inventory(40),
             "The registered workload arithmetic differs from the common schedule")
    inference = value["inference"]
    expected = {(model, baseline) for model in top10 for baseline in ("GRU32", "Persistence")}
    contrasts = inference.get("contrasts", [])
    _require(len(contrasts) == 20 and {(r["model"], r["baseline"]) for r in contrasts} == expected
             and all(r.get("delta") == DELTA and r.get("null") == "gain <= delta"
                     and r.get("alternative") == "gain > delta"
                     and r.get("gain") == "mean_seed_RMSE_baseline - mean_seed_RMSE_model" for r in contrasts),
             "The twenty primary paired contrasts must be fixed before power planning")
    _require(inference.get("family_wise_alpha") == .05 and inference.get("family_size") == 20
             and inference.get("planning_true_gain") == 2*DELTA
             and inference.get("primary_temporal_block_days") == 14,
             "Margin-null inference and planning effect/temporal blocks changed")
    power = value["power"]
    _require(power.get("R") is None and power.get("minimum_R") == 6 and power.get("maximum_R") == 30
             and power.get("pilot_models") == ["TimeMixer", "PatchTST", "GRU32"]
             and power.get("seed_pool") == list(range(271801, 271831)),
             "R remains a pilot-derived gate, not a copied historical repetition count")
    data = value["data_roles"]
    _require(data.get("no_existing_historical_virgin_test_certified") is True
             and data["historical_608"].get("already_evaluated") is True
             and data["historical_60"].get("already_evaluated") is True,
             "Already-evaluated dates cannot be renamed as an untouched final test")
    future = data["independent_final"]
    _require(future.get("start") is None and future.get("end") is None
             and future.get("calendar_days") == 100 and future.get("automatic_extension") is False
             and future.get("future_weight_training_end") == "2026-08-24"
             and future.get("calibration_start") == "2026-09-01" and future.get("calibration_end") == "2026-10-30",
             "The future cohort must await a real freeze and keep its declared training/calibration separation")
    budget = value["budget"]
    _require(budget.get("authorized_total_hours") == 64 and budget.get("authorized_total_seconds") == 230400
             and budget.get("authorization_history_total_hours") == [48, 56, 64]
             and budget.get("fresh_budget_granted_by_this_amendment") is False
             and budget.get("remaining_seconds") is None and budget.get("state") == "reconciliation_required",
             "Registration cannot restart or enlarge the cumulative budget, or invent its current balance")
    _require(value.get("new_market_training_executed") is False and value.get("private_colab_session_modified") is False,
             "This metadata check cannot claim market execution or a modified private session")
    for relative, expected_hash in value["source_metadata_sha256"].items():
        _require(digest(_path(root, relative)) == expected_hash, "Registered parent evidence changed: " + relative)
    return {"protocol_id": PROTOCOL_ID, "valid": True, "market_payloads_opened": 0}


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)


def git_registration(root=ROOT, *, required=False):
    try:
        proof = _json(root, GIT_RECEIPT_PATH)
        commit = proof["registration_commit"]
        _require(isinstance(commit, str) and _SHA.fullmatch(commit), "A full immutable registration commit is required")
        stamp = _git(root, "show", "-s", "--format=%cI", commit).decode().strip()
        _require(_instant(stamp) == _instant(proof["committed_at_utc"]), "Git registration time differs from its actual commit")
        _require(_git(root, "show", commit + ":" + PROTOCOL_PATH) == _path(root, PROTOCOL_PATH).read_bytes(),
                 "The immutable registration commit contains a different protocol")
        _require(_git(root, "show", commit + ":" + REGISTRATION_PATH) == _path(root, REGISTRATION_PATH).read_bytes(),
                 "The immutable registration commit contains a different source receipt")
        headers = _git(root, "cat-file", "-p", commit).decode().split("\n\n", 1)[0].splitlines()
        _require("parent " + PARENT in headers, "The amendment commit has a different scientific parent")
        _require(proof.get("protocol_sha256") == digest(_path(root, PROTOCOL_PATH)), "Git proof refers to a different protocol hash")
        return {"verified": True, "registration_commit": commit, "committed_at_utc": _instant(stamp).isoformat()}
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        if required:
            raise ValueError("The actual immutable Git registration must be verified before independent final binding") from error
        return {"verified": False, "reason": "local preparation or registration commit unavailable; execution remains gated"}


def verify_registration(root=ROOT, *, require_git=False):
    protocol = read_protocol(root)
    receipt = _json(root, REGISTRATION_PATH)
    _require(receipt.get("protocol_id") == PROTOCOL_ID
             and receipt.get("protocol_sha256") == digest(_path(root, PROTOCOL_PATH))
             and receipt.get("parent_source_commit") == PARENT, "Registration receipt identity changed")
    required = {PROTOCOL_PATH, DIRECTORY + "/exposure_registry.json", DIRECTORY + "/selection_budget_policy.json",
                DIRECTORY + "/calendar.csv", "scripts/e2_protocol_amendment.py",
                "scripts/e2_amended_calendar.py", "scripts/e2_amended_data_guard.py", "scripts/e2_amended_metrics.py"}
    _require(required.issubset(receipt.get("files_sha256", {})), "The registration must bind all protocol and guard sources")
    for name, expected in receipt["files_sha256"].items():
        _require(digest(_path(root, name)) == expected, "Registered source bytes changed: " + name)
    _require(_path(root, PROTOCOL_PATH).read_bytes() == _path(root, "book_e2/_static/protocol_amendment_20261112/protocol.json").read_bytes(),
             "The public protocol must match the registered source")
    return {"protocol_id": PROTOCOL_ID, "protocol_sha256": receipt["protocol_sha256"],
            "source_bytes_verified": len(receipt["files_sha256"]),
            "git_registration": git_registration(root, required=require_git),
            "execution_ready": False, "pending_gates": [g for g in protocol["execution_gates"] if g["state"] == "pending"],
            "workload_upper_bounds": calendar.selection_inventory(40),
            "market_training_started": False, "private_colab_modified": False}


def validate_stage_context(model, stage, origin, *, selected_at=None, train_label_end=None,
                           architecture_source_origin=None, architecture_id=None, root=ROOT):
    """Validate scheduling only; passing this check does not authorize fitting."""
    protocol = read_protocol(root)
    _require(model in [r["model_id"] for r in protocol["systems"]], "Unknown mandatory system")
    if stage in ("nas1", "nas2"):
        _require(model == "TiDE" and origin == "2024-01-01", "Full NAS is allowed only for initial TiDE selection")
    elif stage == "hpo":
        _require(model in protocol["neural_systems"] and origin in calendar.HPO_ANCHORS,
                 "HPO is confined to the three common neural anchors")
        if model == "TiDE":
            calendar.validate_nas_dependency(origin, winner_origin=architecture_source_origin,
                                             winner_architecture_id=architecture_id)
    elif stage == "classical_hpo":
        _require(model in ("ARIMA", "SeasonalNaive") and origin in calendar.HPO_ANCHORS,
                 "The deterministic baselines use only their declared candidate rules and anchors")
    elif stage in ("outer_refit", "historical_refit"):
        expected_origins = calendar.OUTER_ORIGINS if stage == "outer_refit" else ("2025-01-01",)
        _require(origin in expected_origins, "Refit origin differs from the common calendar")
        calendar.validate_forecast_context(origin, selected_at, train_label_end)
        if model == "TiDE":
            _require(architecture_source_origin == "2024-01-01" and isinstance(architecture_id, str) and architecture_id,
                     "Refits require the actual initial NAS winner receipt, not the reference body")
    else:
        raise ValueError("Unknown amended selection/refit stage")
    return {"valid_context": True, "model": model, "stage": stage, "origin": origin,
            "execution_authorized_by_this_metadata_check": False}


def validate_bound_cohort_registration(binding, binding_commit, binding_path, *, root=ROOT, now_utc=None):
    """Require a real prior Git binding in addition to a metadata attestation.

    This still does not certify private artifact existence; the executing worker
    must verify all actual checkpoint/calibrator bundles separately.
    """
    report = verify_registration(root, require_git=True)
    data_guard.validate_prospective_binding(binding, report["protocol_sha256"], now_utc=now_utc)
    _require(isinstance(binding_commit, str) and _SHA.fullmatch(binding_commit), "The final cohort needs its own immutable binding commit")
    _path(root, binding_path)  # validates the relative path; no market read
    registered_binding = json.loads(_git(root, "show", binding_commit + ":" + binding_path))
    _require(registered_binding == binding, "The real binding commit contains a different cohort")
    committed = _instant(_git(root, "show", "-s", "--format=%cI", binding_commit).decode().strip())
    start = _instant(binding["start"] + "T00:00:00+00:00")
    _require(_instant(report["git_registration"]["committed_at_utc"])
             <= _instant(binding["freeze_attestation"]["amendment_registered_at_utc"])
             <= _instant(binding["freeze_attestation"]["frozen_at_utc"])
             <= _instant(binding["bound_at_utc"]) <= committed < start,
             "The protocol, actual freeze and actual cohort binding must precede the first target date")
    return {"git_chronology_verified": True, "protocol_sha256": report["protocol_sha256"],
            "binding_commit": binding_commit, "private_material_verification": "still required on the worker"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify registration without any training")
    parser.add_argument("--require-git", action="store_true", help="Require the immutable registration commit and its real timestamp")
    args = parser.parse_args(argv)
    print(json.dumps(verify_registration(require_git=args.require_git), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
