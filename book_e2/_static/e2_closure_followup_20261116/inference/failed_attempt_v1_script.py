"""Conditional HPO/test comparison and finite loss-variance decomposition.

No fitting, model loading, new forecasts, or changes to the original benchmark.
The registration command only inventories inputs. Scientific execution requires
the exact registered script/input hashes and a live shared-budget reservation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "book_e2/_static/e2_closure_followup_20261116/inference"
HPO = ROOT / "book_e2/_static/e2_corrections_20261115/hpo/sources"
INPUTS = ROOT / "book_e2/_static/e2_final_audit_inputs_20261114"
MODELS = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
SEEDS = tuple(range(271801, 271807))
FOLDS = ("2024-01-01", "2024-07-01", "2024-10-01")
VERSION = "e2-conditional-inference-closure-v2"
COMPONENTS = ("model_fixed_catalogue", "initialization_nested_in_model", "calendar",
              "model_by_calendar", "initialization_by_calendar_remainder")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def record(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
            "bytes": path.stat().st_size}


def inventory():
    winners = {}
    selected = {}
    sources = {}
    for path in sorted((HPO / "selections").glob("*.json")):
        row = json.loads(path.read_text())
        model = row["model_id"]
        require(model in MODELS and model not in winners, "Unexpected/duplicate winner")
        winners[model] = row["winner_configuration_id"]
        sources["selection:" + model] = record(path)
    require(set(winners) == set(MODELS), "All thirteen recorded winners required")
    for path in sorted((HPO / "metadata/jobs").glob("*/result.json")):
        row = json.loads(path.read_text())
        job = row.get("job", {})
        model, configuration = job.get("model_id"), job.get("configuration_id")
        if model in winners and configuration == winners[model]:
            key = model + ":" + job["fold_origin"]
            require(key not in selected and job["fold_origin"] in FOLDS,
                    "Duplicate or unexpected winning fold")
            require(len(row["prediction_dates"]) == len(row["predictions_original"]) == 83,
                    "Winning fold must contain83 dated forecasts")
            selected[key] = {"model": model, "fold_origin": job["fold_origin"],
                             "configuration": configuration, "hpo_seed": job["seed"],
                             "cache_reuse": bool(row.get("cache_reuse", False)),
                             "checkpoint_sha256": row.get("selected_checkpoint", {}).get("sha256"),
                             **record(path)}
            sources["OOF:" + key] = record(path)
    require(len(selected) == 39, "All39 winning fold trajectories required")
    for name, path in {
        "original_test": INPUTS / "e2_f2a_input.json",
        "development_targets": INPUTS / "daily_volatility_development.csv",
        "A2": ROOT / "book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv",
        "original_analysis": ROOT / "book_e2/_static/e2_benchmark_final_20261113/benchmark_analysis.json",
    }.items():
        sources[name] = record(path)
    require(sources["original_test"]["sha256"] ==
            "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187",
            "Original A100 forecast bundle changed")
    return sources, selected


def register(directory):
    require(not directory.exists(), "Register in a fresh directory")
    sources, selected = inventory()
    plan = {
        "version": VERSION, "registered_at_utc": utc(), "script_sha256": sha(__file__),
        "scope": "Posterior exploratory analysis, after the original benchmark and earlier diagnostics, before these calculations",
        "sources": sources, "selected_fold_records": selected,
        "models": list(MODELS), "test_seed_ids": list(SEEDS), "fold_origins": list(FOLDS),
        "draws": 5000, "primary_block_days": 14, "sensitivity_block_days": [7, 28],
        "random_seed_base": 202611161501, "family_alpha": 0.05,
        "period_comparison": {
            "effect": "mean of6 original final-test608 individual RMSEs minus pooled249 OOF RMSE of the recorded winner's3 fold-specific predictors",
            "OOF": "83 dates per fold; single HPO initialization per fold; conditional on recorded configuration selection and stopping",
            "test": "same original A100 trajectories as A2; equal6seed average; original chronology and weights unchanged",
            "H0": "conditional risk difference=0", "H1": "conditional risk difference differs from0",
            "resampling": "Independent noncircular moving-block draws within each83-date OOF fold and within608-date test; dates shared across models in each period. Independently sample6 optimization indices with replacement inside each test model; no invented seed pairing across architectures or periods.",
            "p": "(1+count(abs(Tstar-Tobserved)>=abs(Tobserved)))/(B+1)",
            "CI": "95percent marginal percentile intervals; Holm13 for primary block14 only",
            "sensitivity": "blocks7/28; raw sensitivity p-values and marginal intervals are nondecisory",
            "decision": "Reject equality when primary Holm13 p<=0.05; otherwise equality/consistency/equivalence not established",
            "not_claimed": ["same predictor in both periods", "independent HPO validation of selected recipe", "absence of selection optimism", "causal attribution to overfitting or drift", "prospective confirmation"],
        },
        "variance_decomposition": {
            "data": "squared errors of balanced13models x6initializations x608dates; volatility units to the fourth power for variance of squared loss",
            "method": "Exact orthogonal finite ANOVA projection under uniform empirical weighting, no mixed-model distribution fit",
            "components": list(COMPONENTS),
            "definitions": {
                "model_fixed_catalogue": "Var_m(mean_st L_mst)",
                "initialization_nested_in_model": "mean_ms[(mean_t L_mst - mean_st L_mst)^2]",
                "calendar": "Var_t(mean_ms L_mst)",
                "model_by_calendar": "mean_mt[(mean_s L_mst - mean_st L_mst - mean_ms L_mst + mean_mst L_mst)^2]",
                "initialization_by_calendar_remainder": "mean_mst[(L_mst - mean_t L_mst - mean_s L_mst + mean_st L_mst)^2]",
            },
            "uncertainty": "For each bootstrap replicate, jointly resample608dates by noncircular moving blocks and independently resample6initializations inside each model, recompute all orthogonal projections; fixed catalogue remains fixed.",
            "CI": "95percent marginal percentile intervals for finite component magnitude and fraction; no component significance tests",
            "not_claimed": ["population random-effect variance", "variance across markets", "variance across new train partitions", "HPO selection uncertainty", "independent daily residuals"],
            "numerical_QA": "Direct orthogonal projection sum equals empirical total loss variance; nonnegative component magnitudes; independently verify additive planted tensor before real calculations",
        },
        "budget": {"phase": "followup-inference", "maximum_reserved_seconds": 600,
                   "ledger": "book_e2/_static/e2_closure_followup_20261116/compute_ledger.jsonl"},
        "invariants": {"new_fits": 0, "new_model_forwards": 0, "original_A2_RQ1_changed": False,
                       "calibration_used_as_HPO_validation": False, "NAS": False},
        "limits": [
            "All inference is conditional on one known BTC chronology and the existing fitted predictors.",
            "OOF and test have different training sizes, parameters and calendars; the contrast cannot identify the cause of a difference.",
            "OOF folds selected the winner; this bootstrap does not replay selection, and no extra initialization variance exists for the OOF single-seed trajectories.",
            "Temporal block inference is approximate under suitable weak dependence and local stationarity; observed drift is retained and block sensitivity cannot guarantee validity.",
            "Finite orthogonal components describe loss variation in a fixed catalogue, not a fitted random-effects model or algorithm-population variance.",
            "Nonrejection is not equality, consistency, equivalence, adequate fit, or absence of overfitting.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__},
    }
    directory.mkdir(parents=True)
    write_json(directory / "analysis_plan.json", plan)
    print(json.dumps({"plan": str(directory / "analysis_plan.json"),
                      "sha256": sha(directory / "analysis_plan.json"),
                      "script_sha256": plan["script_sha256"]}))


def moving(rng, n, length):
    starts = rng.integers(0, n - length + 1, size=(n + length - 1) // length)
    return (starts[:, None] + np.arange(length)).reshape(-1)[:n]


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    ranked = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    result = np.empty_like(values)
    result[order] = np.minimum(1, ranked)
    return result


def decompose(loss):
    grand = loss.mean()
    model = loss.mean(axis=(1, 2))
    seed = loss.mean(axis=2)
    calendar = loss.mean(axis=(0, 1))
    model_calendar = loss.mean(axis=1)
    a = model - grand
    b = seed - model[:, None]
    c = calendar - grand
    d = model_calendar - model[:, None] - calendar[None, :] + grand
    e = loss - seed[:, :, None] - model_calendar[:, None, :] + model[:, None, None]
    components = np.asarray([np.mean(a*a), np.mean(b*b), np.mean(c*c),
                             np.mean(d*d), np.mean(e*e)])
    total = float(np.mean((loss-grand)**2))
    require(total > 0 and np.isfinite(components).all(), "Degenerate/nonfinite loss decomposition")
    relative_error = abs(components.sum()-total) / total
    require(relative_error < 1e-10 and (components >= 0).all(), "Orthogonal projection QA failed")
    return components, components/total, total, float(relative_error)


def budget_admission(plan):
    path = ROOT / plan["budget"]["ledger"]
    lines = path.read_bytes().splitlines(keepends=True)
    previous, active = None, {}
    for raw in lines:
        row = json.loads(raw)
        require(row["previous_record_sha256"] == previous, "Budget ledger chain is invalid")
        previous = hashlib.sha256(raw).hexdigest()
        if row["event"] == "reserve":
            active[row["reservation_id"]] = row
        elif row["event"] == "finish":
            active.pop(row["reservation_id"])
    matches = [(key, row) for key, row in active.items()
               if row["phase"] == plan["budget"]["phase"]]
    require(len(matches) == 1 and matches[0][1]["cap_seconds"] <= 600,
            "Scientific execution requires one live shared-budget reservation <=600s")
    return {"reservation_id": matches[0][0], "reserve_record": matches[0][1],
            "ledger_head_at_admission_sha256": previous}


def load_arrays(plan):
    payload = json.loads((ROOT / plan["sources"]["original_test"]["path"]).read_text())
    require(len(payload["dates"]) == 608 and payload["mask"] == [True]*608, "Incomplete original608")
    truth = np.asarray(payload["y_true"], float)
    rows = {(r["model"], r["seed"]): r for r in payload["predictors"]}
    test = np.asarray([[rows[m, s]["prediction_original"] for s in SEEDS] for m in MODELS])
    require(test.shape == (13, 6, 608) and np.isfinite(test).all(), "Original78 predictions incomplete")
    development = {r["market_day_utc"][:10]: float(r["sigma_7"])
                   for r in csv.DictReader((ROOT / plan["sources"]["development_targets"]["path"]).open())
                   if r["sigma_7"]}
    oof, admission = np.empty((13, 3, 83)), []
    reference_dates = {}
    for mi, model in enumerate(MODELS):
        for fi, fold in enumerate(FOLDS):
            source = plan["selected_fold_records"][model + ":" + fold]
            row = json.loads((ROOT / source["path"]).read_text())
            dates = row["prediction_dates"]
            require(len(set(dates)) == 83 and sorted(dates) == dates, "OOF dates not unique/ordered")
            require(all((np.datetime64(dates[i+1])-np.datetime64(dates[i])) == np.timedelta64(1, 'D')
                        for i in range(82)), "Do not compress calendar gaps")
            if fi in reference_dates:
                require(dates == reference_dates[fi], "OOF calendars differ across models")
            reference_dates[fi] = dates
            y = np.array([development[d] for d in dates])
            prediction = np.asarray(row["predictions_original"], float)
            require(np.isfinite(prediction).all() and np.isfinite(y).all(), "Invalid OOF numerical values")
            oof[mi, fi] = (prediction-y)**2
            require(abs(float(np.sqrt(oof[mi, fi].mean()))-row["rmse_original"]) < 1e-12,
                    "OOF reconstruction does not match recorded RMSE")
            admission.append({"model": model, "fold_origin": fold, "configuration": source["configuration"],
                              "seed": source["hpo_seed"], "dates": 83, "first_date": dates[0],
                              "last_date": dates[-1], "source_sha256": source["sha256"],
                              "selected_checkpoint_sha256": source["checkpoint_sha256"],
                              "cache_reuse": source["cache_reuse"]})
    return oof, (test-truth[None, None, :])**2, admission


def execute(directory, expected):
    started = time.monotonic()
    path = directory / "analysis_plan.json"
    require(sha(path) == expected, "Exact registered plan SHA required")
    plan = json.loads(path.read_text())
    require(plan["version"] == VERSION and sha(__file__) == plan["script_sha256"], "Registered script changed")
    require(inventory() == (plan["sources"], plan["selected_fold_records"]), "Registered source changed")
    budget = budget_admission(plan)
    output = directory / "result_v1"
    require(not output.exists(), "Results are append-only; use a newly registered plan for reruns")
    output.mkdir()
    write_json(output / "execution_admission.json", {"started_at_utc": utc(), "plan_sha256": expected,
               "script_sha256": sha(__file__), "budget": budget})
    planted = np.arange(3.)[:, None, None]+np.arange(4.)[None, :, None]+np.arange(9.)[None, None, :]
    qa_values, _, _, _ = decompose(planted)
    require(qa_values[3] < 1e-20 and qa_values[4] < 1e-20, "Planted additive tensor QA failed")
    oof, loss, admissions = load_arrays(plan)
    write_csv(output / "winning_OOF_admission_39.csv", admissions)
    oof_rmse = np.sqrt(oof.mean(axis=(1, 2)))
    test_rmse = np.sqrt(loss.mean(axis=2)).mean(axis=1)
    observed = test_rmse-oof_rmse
    components, fractions, total, error = decompose(loss)
    rows, variance_rows, samples = [], [], {}
    max_error = error
    for length in [14, 7, 28]:
        rng = np.random.default_rng(plan["random_seed_base"] + length)
        differences = np.empty((plan["draws"], 13))
        vc = np.empty((plan["draws"], 5))
        vf = np.empty_like(vc)
        for b in range(plan["draws"]):
            oof_indices = np.stack([moving(rng, 83, length) for _ in range(3)])
            sampled_oof = np.take_along_axis(oof, oof_indices[None, :, :], axis=2)
            test_indices = moving(rng, 608, length)
            seed_indices = rng.integers(0, 6, size=(13, 6))
            sampled_loss = loss[np.arange(13)[:, None], seed_indices][:, :, test_indices]
            differences[b] = np.sqrt(sampled_loss.mean(axis=2)).mean(axis=1)-np.sqrt(sampled_oof.mean(axis=(1, 2)))
            vc[b], vf[b], _, err = decompose(sampled_loss)
            max_error = max(max_error, err)
        raw_p = (1+(np.abs(differences-observed[None, :]) >= np.abs(observed[None, :])).sum(axis=0))/(plan["draws"]+1)
        adjusted = holm(raw_p) if length == 14 else [None]*13
        for i, model in enumerate(MODELS):
            ci = np.quantile(differences[:, i], [.025, .975])
            rows.append({"model": model, "block_days": length, "OOF_pooled249_RMSE": float(oof_rmse[i]),
                         "original_test608_mean_individual_RMSE": float(test_rmse[i]),
                         "test_minus_OOF_RMSE": float(observed[i]), "CI95_lower": float(ci[0]),
                         "CI95_upper": float(ci[1]), "p_two_sided_raw": float(raw_p[i]),
                         "p_Holm13": None if adjusted[i] is None else float(adjusted[i]),
                         "primary_family": length == 14,
                         "equality_rejected_primary": bool(adjusted[i] <= .05) if length == 14 else None,
                         "equivalence_established": False, "same_predictor_across_periods": False,
                         "new_fits": 0})
        for i, name in enumerate(COMPONENTS):
            ci, fci = np.quantile(vc[:, i], [.025, .975]), np.quantile(vf[:, i], [.025, .975])
            variance_rows.append({"component": name, "block_days": length,
                                  "finite_variance_component": float(components[i]),
                                  "component_CI95_lower": float(ci[0]), "component_CI95_upper": float(ci[1]),
                                  "fraction_of_total": float(fractions[i]), "fraction_CI95_lower": float(fci[0]),
                                  "fraction_CI95_upper": float(fci[1]), "total_loss_variance": total,
                                  "population_random_effect_component": False})
        samples[f"period_difference_block{length}"] = differences
        samples[f"variance_components_block{length}"] = vc
        samples[f"variance_fractions_block{length}"] = vf
        print(json.dumps({"completed_block_days": length, "draws": plan["draws"],
                          "elapsed_seconds": time.monotonic()-started}), flush=True)
    write_csv(output / "conditional_OOF_test_comparisons_39.csv", rows)
    write_csv(output / "finite_loss_variance_components_15.csv", variance_rows)
    np.savez_compressed(output / "bootstrap_draws.npz", **samples)
    summary = {
        "version": VERSION, "plan_sha256": expected, "completed_at_utc": utc(),
        "elapsed_seconds": time.monotonic()-started, "models": 13, "OOF_folds": 3,
        "OOF_dates_per_model": 249, "test_dates": 608, "test_predictors": 78,
        "period_primary_rejections_Holm13": sum(r["equality_rejected_primary"] for r in rows if r["primary_family"]),
        "primary_rejected_models": [r["model"] for r in rows if r["primary_family"] and r["equality_rejected_primary"]],
        "finite_decomposition_max_relative_identity_error": max_error,
        "finite_decomposition_not_population_random_effect_fit": True,
        "HPO_validation_comparison_executed": True,
        "independent_post_selection_validation_established": False,
        "equivalence_or_consistency_established": False,
        "cause_of_period_difference_identified": False,
        "original_A2_RQ1_changed": False, "new_fits": 0, "new_model_forwards": 0,
        "limitations": plan["limits"],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {"version": VERSION, "plan_sha256": expected,
               "files": [record(p) for p in sorted(output.iterdir()) if p.is_file()]})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["prepare", "run"])
    p.add_argument("--directory", type=Path, default=BASE)
    p.add_argument("--plan-sha")
    args = p.parse_args()
    if args.operation == "prepare":
        register(args.directory)
    else:
        require(args.plan_sha is not None, "--plan-sha is required")
        execute(args.directory, args.plan_sha)


if __name__ == "__main__":
    main()
