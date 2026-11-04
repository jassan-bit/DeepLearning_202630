"""Close all configurations using development CV before final fit or test access.

This script reads only stored 2024 development predictions/configurations and
metadata. It never opens a raw market archive. The resulting freeze is immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filipe_multihistory_20261104"
OLD = ROOT / "results/dl_r2_cv_20261104"
PROTOCOL = ROOT / "experiments/filipe_multihistory_20261104/protocol.json"
FREEZE = PROTOCOL.with_name("freeze.json")
MODELS = ("MLP", "LSTM", "CNN1D")
HISTORIES = (7, 14, 21, 28)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_configuration(rows, tolerance=1e-12):
    """Use CV RMSE, then registered smaller-D/parameter/name tie breaks."""
    rows = list(rows)
    if not rows or any(not np.isfinite(row["cv_rmse_mean"]) for row in rows):
        raise ValueError("Finite development selection scores required")
    minimum = min(row["cv_rmse_mean"] for row in rows)
    ties = [row for row in rows if abs(row["cv_rmse_mean"]-minimum) <= tolerance]
    return min(ties, key=lambda row: (row["history_days"], row["parameters"], row["model"]))


def prepare_freeze():
    plan = json.loads(PROTOCOL.read_text())
    execution = json.loads((OUT / "execution_summary.json").read_text())
    if (not execution.get("all_registered_trajectories_complete")
            or execution.get("trajectories") != 90
            or execution.get("completed_project_fits") != 2160
            or execution.get("test_files_opened") != 0):
        raise ValueError("All 90 new development trajectories must finish before closure")
    if (OUT / "test/predictions.csv").exists() or (OUT / "test/raw_access_started.json").exists():
        raise ValueError("Test access/scoring must not precede configuration closure")
    context = json.loads((OUT / "code_manifest_before_fitting.json").read_text())
    if context["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Registered development protocol changed")
    for name, expected in context["code_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Signed training code changed: " + name)
    for name, expected in plan["sources"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered input changed: " + name)

    previous = pd.read_csv(OLD / "predictions.csv")
    previous = previous.loc[previous.variant.eq("relative_window")].copy()
    current = pd.read_csv(OUT / "predictions.csv")
    predictions = pd.concat([previous, current], ignore_index=True)
    expected_dates = pd.date_range("2024-01-01", "2024-12-31").strftime("%Y-%m-%d").tolist()
    jobs, scores, source_configs = [], [], {}
    reference = None
    for history in HISTORIES:
        for model in MODELS:
            seed_rmse = []
            parameters = 80*history+33 if model == "MLP" else (5025 if model == "LSTM" else 1057)
            for seed in plan["seeds"]:
                frame = predictions.loc[
                    predictions.history_days.eq(history) & predictions.model.eq(model)
                    & predictions.seed.eq(seed)].sort_values("outcome_day")
                if (len(frame) != 366 or frame.outcome_day.tolist() != expected_dates
                        or not frame.variant.eq("relative_window").all()
                        or not np.isfinite(frame[["y_true", "y_pred", "persistence_pred"]].to_numpy()).all()
                        or (frame.y_true < 0).any() or frame.run_id.nunique() != 1):
                    raise ValueError(f"Invalid or incomplete development trajectory {history}/{model}/{seed}")
                labels = frame[["y_true", "persistence_pred"]].to_numpy()
                if reference is None:
                    reference = labels
                elif not np.allclose(labels, reference, rtol=0, atol=1e-14):
                    raise ValueError("All development settings must share labels and reference")
                rmse = float(np.sqrt(np.mean((frame.y_true.to_numpy()-frame.y_pred.to_numpy())**2)))
                seed_rmse.append(rmse)
                origin = OLD if history == 7 else OUT
                path = origin / "runs" / str(frame.run_id.iloc[0]) / "2024-12/configuration.json"
                config = json.loads(path.read_text())
                registered = plan["models"][model]
                epochs = int(config["best_epoch"])
                if (config["model"] != model or int(config["history_days"]) != history
                        or int(config["trajectory_seed"]) != seed
                        or int(config["width"]) != registered["width"]
                        or float(config["learning_rate"]) != registered["learning_rate"]
                        or not 1 <= epochs <= plan["training"]["max_epochs"]):
                    raise ValueError("December development configuration differs from registration")
                relative = path.relative_to(ROOT).as_posix()
                source_configs[relative] = digest(path)
                jobs.append({"history_days": history, "model": model, "seed": seed,
                             "width": registered["width"], "learning_rate": registered["learning_rate"],
                             "epochs": epochs, "parameters": parameters,
                             "epoch_source_configuration": relative,
                             "epoch_source_configuration_sha256": source_configs[relative]})
            scores.append({"model": model, "history_days": history, "parameters": parameters,
                           "cv_rmse_mean": float(np.mean(seed_rmse)),
                           "cv_rmse_sample_sd": float(np.std(seed_rmse, ddof=1)), "seeds": len(seed_rmse)})
    if len(predictions) != 43920 or len(jobs) != 120:
        raise ValueError("The complete four-history development experiment is required")
    selected = {model: select_configuration(row for row in scores if row["model"] == model)
                for model in MODELS}
    baseline = select_configuration(scores)
    code_paths = list(dict.fromkeys(plan["training_code"] + [
        "scripts/close_filipe_multihistory.py", "scripts/load_filipe_reserved_test.py",
        "scripts/final_fit_filipe_multihistory.py", "scripts/analyze_filipe_multihistory.py"]))
    cv_paths = [OUT / name for name in ("predictions.csv", "fold_metrics.csv", "runs.csv", "execution_summary.json")]
    cv_paths += [OLD / "predictions.csv", OLD / "fold_metrics.csv", OLD / "runs.csv"]
    analysis = OUT / "analysis_cv/summary.json"
    if not analysis.is_file():
        raise ValueError("Complete the predeclared CV analysis before closing configuration")
    analysis_summary = json.loads(analysis.read_text())
    if (analysis_summary.get("phase") != "cv" or analysis_summary.get("neural_forecasts") != 43920
            or not analysis_summary.get("bds")
            or analysis_summary["bds"].get("planned_hypotheses") != 2880):
        raise ValueError("The full four-history CV/BDS analysis is required before closure")
    return {
        "schema_version": 1, "status": "configuration_closed_before_reserved_test_access",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol_sha256": digest(PROTOCOL),
        "code_sha256": {name: digest(ROOT / name) for name in code_paths},
        "source_cv_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in cv_paths},
        "analysis_cv_sha256": {path.relative_to(ROOT).as_posix(): digest(path)
                               for path in sorted(analysis.parent.rglob("*")) if path.is_file()},
        "source_epoch_configuration_sha256": source_configs,
        "selection": {"rule": plan["selection_policy"], "selected_by_model": selected,
                      "baseline": baseline, "all_settings": scores,
                      "persistence_cv_rmse": float(np.sqrt(np.mean((reference[:,0]-reference[:,1])**2)))},
        "final_epoch_rule": plan["final_fit_policy"]["epochs"],
        "final_fit_policy": plan["final_fit_policy"], "final_jobs": jobs,
        "test_policy": plan["test_policy"], "raw_root": "/workspace/binance-spot-data",
        "reserve_manifest_sha256": plan["test_policy"]["raw_reserve_manifest_sha256"],
        "test_data_read_before_this_freeze": False,
        "test_selection_allowed": False,
        "limitations": plan["statistical_limits"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Validate an existing immutable freeze")
    args = parser.parse_args()
    if args.validate:
        freeze = json.loads(FREEZE.read_text())
        if freeze["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Closed protocol changed")
        for key in ("code_sha256", "source_cv_sha256", "analysis_cv_sha256", "source_epoch_configuration_sha256"):
            for name, expected in freeze[key].items():
                if digest(ROOT / name) != expected:
                    raise ValueError("Closed artifact changed: " + name)
        print(json.dumps({"status": "freeze_verified", "sha256": digest(FREEZE), "jobs": len(freeze["final_jobs"])}))
        return
    if FREEZE.exists():
        raise FileExistsError("An existing freeze must never be overwritten")
    freeze = prepare_freeze()
    with FREEZE.open("x") as stream:
        stream.write(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": freeze["status"], "final_jobs": len(freeze["final_jobs"]),
                      "baseline_selected_on_CV": freeze["selection"]["baseline"], "sha256": digest(FREEZE)}))


if __name__ == "__main__":
    main()
