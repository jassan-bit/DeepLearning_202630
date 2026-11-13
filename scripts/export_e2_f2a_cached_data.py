"""Export existing E2 targets and predictions without fitting or inference.

The original scientific output is read only. All validation finishes before
creating the separate export directory or contacting the backup service.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
VERSION = "e2-f2a-cached-input-export-v1"
PIN_SHA = {
    "freeze": "c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4",
    "result": "625f72776d69cc7a11766c48d30a43e761e79023864f8e46d1ee62062e2f8461",
    "admission": "1d430f16a4d363055800fb4ec62cc4a7bc9062a131e3a792324b8440e5cebd9c",
    "cache": "9563ef557e180f34ee37dce3b1480c38aaefd7063df21c031644973310e28deb",
    "loader": "693842a92386cd3c49ed3e10a9296428e900010df7ae7db373a80d625e851867",
    "backup": "f4e039ebf8c71a835127a36202ebcdbcd9e43456e35ecef3de390320b8a479ef",
}
CALIBRATION_TARGET_SHA256 = "3401d6557cac39166c887b8595dae84342f72ce64e58f2d781ecb5710867deb7"
FINAL_DATES = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
CALIBRATION_DATES = [(date(2024, 11, 8) + timedelta(days=i)).isoformat() for i in range(53)]
MAX_METADATA_BYTES = 4 * 1024**2


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_pinned(path, expected_sha):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("A regular existing source file is required")
    raw = path.read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError("Existing source SHA256 mismatch: " + path.name)
    return raw


def _regular_json(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("A regular existing JSON source is required")
    return json.loads(path.read_bytes())


def _inside(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Source artifact path escapes its original root")
    return path


def validate_source(source_root):
    source_root = Path(source_root)
    head = subprocess.check_output(["git", "--no-optional-locks", "-C", str(source_root),
                                    "rev-parse", "HEAD"], text=True).strip()
    if head != ENGINE_COMMIT:
        raise ValueError("The preserved original E source commit is required")
    changed = subprocess.run(["git", "--no-optional-locks", "-C", str(source_root),
        "diff", "--no-ext-diff", "--quiet", ENGINE_COMMIT, "--", "scripts"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if changed.returncode:
        raise ValueError("The preserved original E source files changed")
    read_pinned(source_root / "scripts/e2_reduced_final_data.py", PIN_SHA["loader"])
    read_pinned(source_root / "scripts/e2_reduced_github_backup.py", PIN_SHA["backup"])


def _original_module(source_root, name):
    sys.dont_write_bytecode = True
    scripts = str(Path(source_root).resolve() / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != Path(scripts, name + ".py"):
        raise ValueError("An original E module was shadowed")
    return module


def derive_targets(source_root, development_root, cache_csv):
    """Use original E's population-sigma7 construction, without a predictor."""
    import pandas as pd
    loader = _original_module(source_root, "e2_reduced_final_data")
    development, receipt = loader.load_development_bridge(development_root)
    evaluation = pd.read_csv(io.BytesIO(Path(cache_csv).read_bytes()), index_col="market_day_utc")
    evaluation.index = pd.DatetimeIndex(pd.to_datetime(evaluation.index, utc=True))
    frame = pd.DataFrame(development.features, index=development.calendar, columns=loader.FEATURES)
    panel = loader._panel_from_frames(frame, evaluation)
    calendar = pd.date_range(FINAL_DATES[0], FINAL_DATES[-1], tz="UTC")
    support = panel.support.loc[calendar]
    support_sha = sha(canonical({"dates": [str(d.date()) for d in support.index],
        "columns": list(support.columns), "mask": support.to_numpy(dtype=bool).tolist()}))
    return (calendar.strftime("%Y-%m-%d").tolist(), panel.targets_at(calendar).tolist(),
            support["common"].to_numpy(dtype=bool).tolist(), receipt, support_sha)


def _finite_vector(values, size):
    if (not isinstance(values, list) or len(values) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("An existing finite aligned prediction/target vector is required")
    return values


def build_bundle(source_root, output_root, development_root=None):
    """Validate all existing inputs in memory; write nothing and run no model."""
    source_root, output_root = Path(source_root), Path(output_root)
    validate_source(source_root)
    final_root = output_root / "final"
    score_root = final_root / "score-worker-source-recovery-v1"
    freeze = json.loads(read_pinned(final_root / "final_freeze.json", PIN_SHA["freeze"]))
    result = json.loads(read_pinned(score_root / "result.json", PIN_SHA["result"]))
    if result.get("status") != "completed" or result.get("retuning_performed") is not False:
        raise ValueError("The original completed score is required")
    scientific = score_root / "scientific_outputs"
    admission_raw = read_pinned(scientific / "data_admission.json", PIN_SHA["admission"])
    admission = json.loads(admission_raw)
    cache_root = final_root / "private-known-E1-data"
    cache = cache_root / "known_E1_daily.csv"
    read_pinned(cache, PIN_SHA["cache"])
    cache_receipt = cache_root / "known_E1_data_receipt.json"
    if _regular_json(cache_receipt) != admission:
        raise ValueError("The cached daily receipt differs from the completed score admission")
    if admission.get("daily_market_sha256") != PIN_SHA["cache"]:
        raise ValueError("The original daily cache identity changed")
    dates, targets, mask, dev_receipt, support_sha = derive_targets(
        source_root, source_root if development_root is None else development_root, cache)
    _finite_vector(targets, 608)
    if (dates != FINAL_DATES or len(mask) != 608 or any(type(x) is not bool for x in mask)
            or not all(mask) or admission.get("common_eligible_dates") != dates
            or admission.get("common_eligible_count") != 608
            or admission.get("support_sha256") != support_sha
            or admission.get("development_bridge") != dev_receipt):
        raise ValueError("Derived targets differ from the original data/calendar admission")
    artifact_hashes = result["artifacts_sha256"]
    analysis_path = "scientific_outputs/benchmark_analysis.json"
    analysis = json.loads(read_pinned(_inside(score_root, analysis_path), artifact_hashes[analysis_path]))
    if analysis.get("common_mask_sha256") != sha(canonical(mask)):
        raise ValueError("The original scoring mask changed")
    expected_rows = {row["model"]: row for row in analysis["table_A2"]}
    frozen_models = freeze["models"]
    if len(frozen_models) != 81 or len(expected_rows) != 16:
        raise ValueError("Exactly 81 original predictors and 16 systems are required")
    predictors, seen, grouped_rmse, sources = [], set(), {}, {}
    for row in frozen_models:
        name, seed = row["model"], row["seed"]
        identity = (name, seed)
        if identity in seen:
            raise ValueError("Duplicate frozen predictor identity")
        seen.add(identity)
        slug = hashlib.sha256(name.encode()).hexdigest()[:16]
        relative = "scientific_outputs/predictions_" + slug + "_" + str(seed) + ".json"
        pred = json.loads(read_pinned(_inside(score_root, relative), artifact_hashes[relative]))
        if (pred.get("dates") != dates or pred.get("model") != name or pred.get("seed") != seed
                or pred.get("checkpoint_sha256") != row["checkpoint_sha256"]
                or pred.get("mask_sha256") != sha(canonical(mask))):
            raise ValueError("Saved predictions do not match their frozen predictor")
        predictions = _finite_vector(pred["prediction_original"], 608)
        rmse = math.sqrt(sum((p-y)**2 for p, y in zip(predictions, targets)) / 608)
        grouped_rmse.setdefault(name, []).append(rmse)
        calibrator_path = _inside(final_root, row["calibrator"])
        calibrator = json.loads(read_pinned(calibrator_path, row["calibrator_sha256"]))
        if (calibrator.get("predictor_sha256") != row["checkpoint_sha256"]
                or calibrator.get("dates") != CALIBRATION_DATES
                or calibrator.get("calibration_targets_sha256") != CALIBRATION_TARGET_SHA256
                or calibrator.get("sha256") != sha(canonical({k: v for k, v in calibrator.items() if k != "sha256"}))):
            raise ValueError("The original calibrator identity changed")
        worker_root = calibrator_path.parent
        calibration_name = "predictions.json" if seed == "deterministic" else "calibration_predictions.json"
        calibration_path = worker_root / calibration_name
        calibration = _regular_json(calibration_path)
        cp = _finite_vector(calibration["prediction_original"], 53)
        cy = _finite_vector(calibration["target_original"], 53)
        if (calibration.get("dates") != CALIBRATION_DATES
                or sha(canonical(cp)) != calibrator["calibration_predictions_sha256"]
                or sha(canonical(cy)) != CALIBRATION_TARGET_SHA256):
            raise ValueError("Saved calibration records differ from the frozen calibrator")
        if seed != "deterministic" and (calibration.get("model") != name
                or calibration.get("seed") != seed
                or calibration.get("checkpoint_sha256") != row["checkpoint_sha256"]
                or calibration.get("same_predictor_as_final") is not True):
            raise ValueError("Calibration did not use the frozen final neural predictor")
        if seed == "deterministic":
            original_worker = _regular_json(worker_root / "result.json")
            if (original_worker.get("checkpoint_sha256") != row["checkpoint_sha256"]
                    or original_worker.get("model") != name
                    or original_worker.get("status") != "completed"):
                raise ValueError("Classical calibration does not match its frozen checkpoint")
            if original_worker.get("artifacts_sha256", {}).get(calibration_name) != sha(calibration_path.read_bytes()):
                raise ValueError("Classical calibration artifact hash changed")
        q = calibrator["q"]
        if type(q) not in (int, float) or not math.isfinite(q) or q < 0:
            raise ValueError("A finite frozen interval quantile is required")
        rank = math.ceil((53 + 1) * .9)
        if (calibrator.get("n") != 53 or calibrator.get("rank") != rank
                or abs(sorted(abs(y-p) for y, p in zip(cy, cp))[rank-1] - q) > 1e-12):
            raise ValueError("Saved calibration residuals do not reproduce the frozen quantile")
        predictors.append({"model": name, "seed": seed,
            "checkpoint_sha256": row["checkpoint_sha256"], "prediction_original": predictions,
            "calibrator_q": q, "calibration_dates": CALIBRATION_DATES,
            "calibration_targets": cy, "calibration_predictions": cp})
        sources[relative] = artifact_hashes[relative]
        sources[str(calibration_path.relative_to(final_root))] = sha(calibration_path.read_bytes())
    for name, values in grouped_rmse.items():
        original = expected_rows[name]["RMSE_per_seed"]
        if len(values) != len(original) or any(abs(a-b) > 1e-10 for a, b in zip(values, original)):
            raise ValueError("Derived truth does not reproduce all 81 original seed RMSE values")
    if set(grouped_rmse) != set(expected_rows):
        raise ValueError("The original system set changed")
    return {"version": VERSION, "dates": dates, "y_true": targets, "mask": mask,
        "predictors": predictors, "original_analysis": analysis,
        "provenance": {"engine_commit": ENGINE_COMMIT, "pinned_source_sha256": dict(PIN_SHA),
            "source_artifacts_sha256": sources, "support_sha256": support_sha,
            "targets_sha256": sha(canonical(targets)), "calibration_targets_sha256": CALIBRATION_TARGET_SHA256,
            "all_81_original_seed_RMSE_verified": True, "RMSE_absolute_tolerance": 1e-10,
            "scientific_fit": False, "model_inference": False, "benchmark_repeated": False,
            "raw_market_data_included": False}}


def _immutable_write(path, raw):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("The export destination must not use symlinks")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("An existing export differs; preserve it for reconciliation")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def publish_export(bundle, destination, client):
    """Publish only the already validated derived bundle to a new runs path."""
    raw = canonical(bundle) + b"\n"
    if len(raw) >= MAX_METADATA_BYTES:
        raise ValueError("The derived input JSON exceeds the 4 MiB metadata limit")
    destination = Path(destination)
    attempt_id = "f2a-" + sha(raw)[:24] + "-posthoc"
    receipt = {"version": VERSION, "kind": "posthoc_input_export", "status": "completed",
        "attempt_id": attempt_id, "scientific_fit": False, "benchmark_repeated": False,
        "model_inference": False, "raw_market_data_included": False,
        "artifacts_sha256": {"e2_f2a_input.json": sha(raw)}}
    _immutable_write(destination / "e2_f2a_input.json", raw)
    _immutable_write(destination / "export_receipt.json", canonical(receipt) + b"\n")
    client._branch_head()  # Existing results branch only; never initialize a destination.
    client._checked = True  # Skip the uploader's mutating preflight transport probe.
    return client.persist_result(destination, receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    source, output, destination = map(Path, (args.source_root, args.output_root, args.destination))
    if destination.resolve().is_relative_to(output.resolve()) or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Export writes must remain outside the scientific output and original source")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use the existing Python 3.12 runtime; no runtime installation is allowed")
    token = sys.stdin.readline().rstrip("\r\n")
    if not token or any(c.isspace() for c in token):
        raise ValueError("A GitHub token must arrive privately through stdin")
    bundle = build_bundle(source, output)
    original_backup = _original_module(source, "e2_reduced_github_backup")
    result = publish_export(bundle, destination, original_backup.GitHubBackup(token))
    print(json.dumps({k: result[k] for k in ("status", "attempt_id", "remote_commit", "manifest_path")}))
    return result


if __name__ == "__main__":
    main()
