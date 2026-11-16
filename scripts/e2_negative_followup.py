"""Registered, bounded replication of two already inspected negative controls.

This posterior diagnostic preserves the original campaign and control family.
It never selects seeds, recipes, permutations or stopping from test scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "book_e2/_static/e2_closure_followup_20261116/audit"
WORK = ROOT / "work/e2-negative-followup-20261116"
ORIGINAL = ROOT / "book_e2/_static/e2_scientific_closure_20261116/controls"
MODELS = ("TSMixer", "iTransformer")
PERMUTATIONS = (2026111601, 2026111602, 2026111603)
INIT_SEED = 271801
DELTA = .00025772842664981106


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError("Immutable evidence already differs: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(raw)


def utc():
    return datetime.now(timezone.utc).isoformat()


def require_budget(args):
    import e2_corrections_budget as budget
    with budget.locked_ledger(args.ledger) as (rows, _):
        matching = [r for r in budget.summarize(rows)["open_reservations"].values()
                    if r["phase"] == args.phase]
    if len(matching) != 1:
        raise RuntimeError("Exactly one matching active cumulative-budget reservation is required")


def prepare(args):
    import e2_corrections_controls as engine
    spec = json.loads((ORIGINAL / "specification.json").read_text())
    paths = [ROOT / "scripts/e2_corrections_controls.py", Path(__file__),
             ROOT / "scripts/e2_final_models_mlp.py", ROOT / "scripts/e2_final_models_attention.py",
             ROOT / "scripts/e2_reduced_fit.py", ROOT / "scripts/e2_published_registry.py",
             ORIGINAL / "specification.json"]
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    records = {}
    for model in MODELS:
        number = engine.MODELS.index(model) + 1
        path = ORIGINAL / ("negative_" + str(number).zfill(2) + ".json")
        record = json.loads(path.read_text())
        bindings[str(path.relative_to(ROOT))] = sha(path)
        records[model] = {"model": model, "original_elapsed_seconds": record["elapsed_seconds"],
                          "original_initial_state_sha256": record["phase_records"]["supervised"]["initial_state_sha256"],
                          "recipe": spec["negative"]["recipes"][model]}
    trace = {
        "reviewed_at_utc": utc(), "review_type": "static source and artifact provenance; no new forward or fit",
        "scope": list(MODELS), "sources_sha256": bindings,
        "checks": [
            {"id": "N1", "status": "PASS_static", "finding": "Original supervised optimizer receives only Xtrain655 and block-permuted ytrain655; fixed epochs; no validation early stop", "source": "scripts/e2_corrections_controls.py:447-484"},
            {"id": "N2", "status": "PASS_static", "finding": "Target scale is the true training-only mean, invariant to a label permutation; no test target enters scaling", "source": "scripts/e2_corrections_controls.py:115-124"},
            {"id": "N3", "status": "PASS_static", "finding": "TSMixer forward receives X only; six residual blocks and scalar bridge use the same D7 window/moments", "source": "scripts/e2_final_models_mlp.py:182-191"},
            {"id": "N4", "status": "PASS_static", "finding": "iTransformer forward receives X only; normalized five variable tokens plus same-window moments; no target/date/calendar", "source": "scripts/e2_final_models_attention.py:314-341"},
            {"id": "N5", "status": "LIMITATION", "finding": "Body, hyperparameters and epochs were selected with real development targets before these controls; the complete HPO procedure is not rerun under permutation"},
            {"id": "N6", "status": "LIMITATION", "finding": "sigma7(t) contains seven returns from closes t-7..t; six returns are known in the legitimate input ending t-1. Same-window moments can encode legitimate partial target information; retained association alone is not evidence of leakage."}
        ],
        "conclusion": "No explicit future-target or test-label path found in reviewed source. Static checks do not establish chance equivalence or certify absence of leakage."
    }
    write(PUBLIC / "trace_review.json", trace)
    plan = {
        "version": "e2-negative-followup-v1", "prepared_at_utc": utc(),
        "status": "frozen_before_additional_fits", "posterior": True,
        "selection_reason": "TSMixer and iTransformer selected for investigation after the original single-permutation results; follow-up is diagnostic, not a new confirmatory family",
        "models": list(MODELS), "initialization_seed": INIT_SEED,
        "label_permutation_seeds": list(PERMUTATIONS), "label_block_days": 7,
        "permutation_algorithm": "unchanged equal-length contiguous calendar blocks; never cross missing-date runs; no rejection of fixed rows or score-based redraw",
        "original_specification": "book_e2/_static/e2_scientific_closure_20261116/controls/specification.json",
        "sources_sha256": bindings, "trace_sha256": sha(PUBLIC / "trace_review.json"),
        "source_bindings_original_specification": spec["sources_sha256"],
        "arrays_path": spec["arrays_path"], "arrays_sha256": spec["arrays_sha256"],
        "records": records, "training_cohort": 655, "evaluation_cohort": 608,
        "HPO": False, "calibration_or_test_selection": False, "fixed_recipes": True,
        "original_primary_modified": False, "new_blind_test_claimed": False,
        "bootstrap": {"draws": 10000, "calendar_block_days": 14, "seed": 2026111604,
                      "scope": "calendar uncertainty conditional on the three fixed new predictors per model; not uncertainty across all permutations, HPO or training partitions"},
        "equivalence": {"endpoint": "mean of three RMSE gains versus true-training-mean constant on the fixed608-day test", "margin": DELTA,
                        "margin_source": "inherited registered domain RMSE delta; not selected from these results", "test": "approximate centered calendar-block bootstrap TOST, Holm over the two model endpoints", "alpha": .05,
                        "interpretation": "Any equivalence applies only to the finite registered mean RMSE endpoint; it does not establish correlation-equivalence or global absence of leakage"},
        "association": "Spearman rho and95% calendar-block intervals reported descriptively for each fit; no invented correlation tolerance",
        "audit_acceptance": "Do not certify original13-model audit from this targeted2-model study. Record explicit near-constant-RMSE equivalence only if prespecified TOST+Holm succeeds; otherwise retain inconclusive/adverse result.",
        "failure_policy": "All six scheduled fits attempted in fixed order unless time cap/failure blocks the run; failures retained, no implicit retry or substitute seed; incomplete design yields no equivalence decision",
        "budget": {"cap_seconds": 800, "unit": "scientific worker wall seconds, cumulative84h account", "projection": "3*(original TSMixer wall+original iTransformer wall), with substantial operational cap; no guarantee"}
    }
    write(args.plan, plan)
    print(json.dumps({"event": "negative_followup_plan_prepared", "plan": str(args.plan), "sha256": sha(args.plan), "fits": 0}), flush=True)


def admitted(plan_path):
    import e2_corrections_controls as engine
    plan = json.loads(plan_path.read_text())
    for relative, expected in plan["sources_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    if sha(PUBLIC / "trace_review.json") != plan["trace_sha256"]:
        raise ValueError("Trace review changed")
    spec_path = ROOT / plan["original_specification"]
    spec, data = engine.admitted(spec_path)
    return plan, spec, data


def key(model, permutation):
    return model.replace("-", "_") + "_" + str(permutation)


def run(args):
    require_budget(args)
    import numpy as np
    import torch
    import e2_corrections_controls as engine
    import e2_staged_training as historical
    from e2_reduced_fit import _builder
    plan, spec, data = admitted(args.plan)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    x = torch.tensor(data["train"]["X"], dtype=torch.float32)
    for seed in plan["label_permutation_seeds"]:
        for model_name in plan["models"]:
            started = time.monotonic()
            name = key(model_name, seed)
            result_path = PUBLIC / "results" / (name + ".json")
            job = WORK / name
            if (job / "request.json").exists() or result_path.exists():
                raise FileExistsError("Follow-up attempt already exists; no implicit retry: " + name)
            write(job / "request.json", {"plan_sha256": sha(args.plan), "started_at_utc": utc(),
                                        "model": model_name, "label_permutation_seed": seed, "phase": args.phase})
            try:
                permuted, permutation = engine.permute_blocks(data["train"]["y_scaled"], data["train"]["dates"], 7, seed)
                y = torch.tensor(permuted, dtype=torch.float32)
                recipe = plan["records"][model_name]["recipe"]
                trial = recipe["trial"]
                config = historical.TrainConfig(learning_rate=trial["learning_rate"], weight_decay=trial["weight_decay"],
                    batch_size=trial["batch_size"], max_epochs=200, patience=20, seed=INIT_SEED,
                    warmup_fraction=trial.get("warmup_fraction", 0))
                model = _builder(model_name, trial, INIT_SEED, json.loads(engine.SOURCE_ROOTS.read_text()))
                if sum(p.numel() for p in model.parameters()) != recipe["parameters"]:
                    raise ValueError("Frozen parameter count differs")
                phase = engine.phase_cpu(model, x, y, config, recipe["fixed_epochs"]["supervised"], data["target_scale"])
                if phase["initial_state_sha256"] != plan["records"][model_name]["original_initial_state_sha256"]:
                    raise ValueError("Initialization differs from the original negative control")
                fit_finished_at = utc()
                checkpoint = job / "diagnostic_model.pt"
                torch.save({"state_dict": historical.clone_state(model), "model": model_name, "seed": INIT_SEED,
                            "plan_sha256": sha(args.plan), "label_permutation_seed": seed,
                            "permutation_sha256": engine.array_hash(permutation)}, checkpoint)
                predictions = []
                with torch.inference_mode():
                    for row in data["test"]["X"]:
                        value = model(torch.tensor(row[None], dtype=torch.float32)).reshape(-1)
                        if len(value) != 1 or not bool(torch.isfinite(value).all()):
                            raise ValueError("Nonfinite/malformed diagnostic prediction")
                        predictions.append(float(value.item()) * data["target_scale"])
                payload = PUBLIC / "predictions" / (name + ".npz")
                payload.parent.mkdir(parents=True, exist_ok=True)
                if payload.exists():
                    raise FileExistsError("Prediction payload exists")
                np.savez_compressed(payload, dates=data["test"]["dates"].astype(str), truth=data["test"]["y"],
                                    prediction=np.asarray(predictions), permutation=permutation)
                record = {"status": "completed", "model": model_name, "initialization_seed": INIT_SEED,
                    "label_permutation_seed": seed, "plan_sha256": sha(args.plan), "fit_finished_at_utc": fit_finished_at,
                    "permutation_sha256": engine.array_hash(permutation), "permuted_labels_sha256": engine.array_hash(permuted),
                    "permutation_fixed_rows": int(np.sum(permutation == np.arange(655))), "phase": phase,
                    "checkpoint_path": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha(checkpoint),
                    "prediction_path": str(payload.relative_to(ROOT)), "prediction_sha256": sha(payload),
                    "runtime": {"torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
                                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                    "elapsed_seconds": time.monotonic() - started, "original_primary_modified": False}
                write(result_path, record)
                print(json.dumps({"event": "negative_followup_fit_completed", "model": model_name, "permutation": seed, "seconds": record["elapsed_seconds"]}), flush=True)
            except Exception as exc:
                write(PUBLIC / "failures" / (name + ".json"), {"status": "failed", "error": str(exc), "error_type": type(exc).__name__, "elapsed_seconds": time.monotonic() - started, "plan_sha256": sha(args.plan)})
                raise
    summarize(args, plan, data)


def summarize(args, plan, data):
    import numpy as np
    from scipy.stats import rankdata
    import e2_corrections_controls as engine
    config = plan["bootstrap"]
    indices = engine.calendar_indices(data["test"]["dates"], config["draws"], config["calendar_block_days"], config["seed"])
    if (indices < 0).any():
        raise ValueError("Complete contiguous test calendar required")
    truth = data["test"]["y"]
    reference_error = (data["target_scale"] - truth) ** 2
    rows, models = [], []
    for model in plan["models"]:
        gains, gain_draws = [], []
        for seed in plan["label_permutation_seeds"]:
            name = key(model, seed)
            record_path = PUBLIC / "results" / (name + ".json")
            record = json.loads(record_path.read_text())
            payload = ROOT / record["prediction_path"]
            if record["plan_sha256"] != sha(args.plan) or sha(payload) != record["prediction_sha256"]:
                raise ValueError("Result/prediction provenance differs")
            with np.load(payload, allow_pickle=False) as saved:
                prediction = saved["prediction"].copy()
                if not np.array_equal(saved["truth"], truth) or not np.array_equal(saved["dates"].astype("datetime64[D]"), data["test"]["dates"]):
                    raise ValueError("Truth/date payload differs")
            errors = (prediction - truth) ** 2
            gain = float(np.sqrt(reference_error.mean()) - np.sqrt(errors.mean()))
            bootstrap_gains = np.sqrt(reference_error[indices].mean(1)) - np.sqrt(errors[indices].mean(1))
            ranks_p, ranks_y = rankdata(prediction), rankdata(truth)
            rho = float(np.corrcoef(ranks_p, ranks_y)[0, 1]) if ranks_p.std() and ranks_y.std() else None
            rs = []
            for start in range(0, len(indices), 250):
                ix = indices[start:start + 250]
                a, b = rankdata(prediction[ix], axis=1), rankdata(truth[ix], axis=1)
                a -= a.mean(1, keepdims=True)
                b -= b.mean(1, keepdims=True)
                den = np.sqrt((a*a).sum(1)*(b*b).sum(1))
                rs.extend(np.divide((a*b).sum(1), den, out=np.full(len(ix), np.nan), where=den > 0).tolist())
            finite = np.asarray(rs)[np.isfinite(rs)]
            rows.append({"model": model, "permutation_seed": seed, "gain_RMSE_vs_training_mean": gain,
                         "gain_CI95_conditional": np.quantile(bootstrap_gains, [.025, .975]).tolist(),
                         "Spearman_rho": rho, "rho_CI95_conditional": None if not len(finite) else np.quantile(finite, [.025, .975]).tolist(),
                         "valid_rho_draws": len(finite), "fixed_rows": record["permutation_fixed_rows"]})
            gains.append(gain)
            gain_draws.append(bootstrap_gains)
        mean_gain, draws = float(np.mean(gains)), np.mean(gain_draws, axis=0)
        p_lower = engine.centered_bootstrap_p(draws, mean_gain, -DELTA)
        p_upper = (1 + int(np.sum(DELTA + draws - mean_gain <= mean_gain))) / (len(draws) + 1)
        models.append({"model": model, "registered_new_permutations": len(gains), "mean_gain_RMSE_fixed_three": mean_gain,
                       "range_gain_over_fixed_three": [float(min(gains)), float(max(gains))],
                       "CI90_calendar_conditional": np.quantile(draws, [.05, .95]).tolist(),
                       "p_TOST_lower": p_lower, "p_TOST_upper": p_upper, "p_TOST": max(p_lower, p_upper)})
    for row, adjusted in zip(models, engine.holm([r["p_TOST"] for r in models])):
        row["p_TOST_Holm2"] = adjusted
        row["equivalence_fixed_mean_RMSE_endpoint"] = adjusted <= .05
    summary = {"version": plan["version"], "plan_sha256": sha(args.plan), "completed_fits": len(rows), "planned_fits": 6,
               "rows": rows, "models": models, "bootstrap": config, "margin": DELTA,
               "original13_model_audit_accepted": False,
               "interpretation": "Targeted posterior diagnostic; any finite-three mean-RMSE equivalence is conditional on these trained functions and dates. Three permutations do not identify full permutation/HPO variability or establish correlation-equivalence. No significance or equivalence result alone proves/excludes leakage.",
               "original_primary_modified": False}
    write(PUBLIC / "summary.json", summary)
    print(json.dumps({"event": "negative_followup_summary", "fits": len(rows), "full_audit_accepted": False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "run"):
        child = sub.add_parser(mode)
        child.add_argument("--plan", type=Path, default=PUBLIC / "plan.json")
        if mode == "run":
            child.add_argument("--ledger", type=Path, required=True)
            child.add_argument("--phase", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.mode](args)


if __name__ == "__main__":
    main()
