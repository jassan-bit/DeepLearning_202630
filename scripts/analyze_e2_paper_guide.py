#!/usr/bin/env python3
"""Analyse the external E2 paper replication and its preregistered pilot.

The published number is a rounded fixed reference, not an original sample with
known variance. Seed intervals condition on the retained test calendar. Temporal
intervals resample forecast origins jointly, retaining horizons and channels.
No Torch, market CSV, fitting or selection occurs in this module.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

SEEDS = (2019, 2020, 2021, 2022, 2023)
PILOT_MODELS = ("TimeMixer", "iTransformer", "PatchTST")
REFERENCE_MSE = 0.375
REFERENCE_MAE = 0.400
ALIGNMENT_TOLERANCE = 1e-5


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False,
                                    indent=2, allow_nan=False) + "\n")


def finite_array(value, name, ndim=None):
    value = np.asarray(value, dtype=np.float64)
    if not value.size or not np.isfinite(value).all():
        raise ValueError(name + " must contain finite nonempty values")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(name + " has the wrong number of dimensions")
    return value


def seed_mean_interval(values, confidence=0.95):
    """Student interval for training randomness on the same finite calendar."""
    values = finite_array(values, "seed metrics", 1)
    if len(values) < 2:
        raise ValueError("At least two independently fitted seeds are required")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(student_t.ppf((1 + confidence) / 2, len(values) - 1)
                 * sd / math.sqrt(len(values)))
    return {"mean": mean, "sample_sd": sd, "n_seeds": len(values),
            "ci95": [mean - half, mean + half],
            "method": "Student-t over fitted-seed metrics; conditional fixed retained calendar",
            "assumption": "Independent initialization/training randomizations; approximate normality of seed metrics",
            "does_not_estimate_market_or_original_paper_variance": True}


def replication_summary(mse_by_seed, mae_by_seed, condition_scope=None):
    mse = seed_mean_interval(mse_by_seed)
    mae = seed_mean_interval(mae_by_seed)
    margin = 0.05 * REFERENCE_MSE
    delta_ci = [v - REFERENCE_MSE for v in mse["ci95"]]
    compatible = delta_ci[0] >= -margin and delta_ci[1] <= margin
    # The paper rounds to three decimals. This sensitivity never changes the
    # registered decision against its displayed reference 0.375.
    rounded = []
    for reference in [REFERENCE_MSE - 0.0005, REFERENCE_MSE + 0.0005]:
        ci = [v - reference for v in mse["ci95"]]
        rounded.append({"fixed_reference": reference, "difference_ci95": ci,
                        "within_registered_absolute_margin": ci[0] >= -margin and ci[1] <= margin})
    return {"reference": {"paper": "TimeMixer", "dataset": "ETTh1", "context": 96,
                           "horizon": 96, "channels": 7, "table": "13", "physical_PDF_page": 17,
                           "MSE": REFERENCE_MSE, "MAE": REFERENCE_MAE,
                           "publication_dispersion_of_selected_row": "NR",
                           "reference_treated_as_fixed_rounded_value": True},
            "MSE": {**mse, "difference_from_paper": mse["mean"] - REFERENCE_MSE,
                    "relative_difference_percent": 100 * (mse["mean"] / REFERENCE_MSE - 1),
                    "difference_ci95": delta_ci},
            "MAE": {**mae, "difference_from_paper": mae["mean"] - REFERENCE_MAE,
                    "relative_difference_percent": 100 * (mae["mean"] / REFERENCE_MAE - 1),
                    "difference_ci95": [v - REFERENCE_MAE for v in mae["ci95"]],
                    "role": "secondary descriptive; not selected after observing the MSE decision"},
            "registered_success_rule": {"metric": "mean seed MSE", "absolute_margin": margin,
                                        "relative_tolerance": 0.05,
                                        "rule": "Entire conditional 95% difference interval contained in [-0.01875,0.01875]",
                                        "numerical_compatibility": bool(compatible),
                                        "does_not_by_itself_establish_identical_historical_row_conditions": True},
            "reference_rounding_sensitivity": rounded,
            "condition_scope": condition_scope or "Partial author-preset replication; historical paper tail coverage not independently established",
            "Bitcoin_margin_used": False,
            "no_two_population_significance_test": "Original row variance and individual predictions are unavailable"}


def circular_block_indices(n, block, rng):
    if isinstance(n, bool) or isinstance(block, bool) or int(n) != n or int(block) != block:
        raise ValueError("Calendar length and block length must be positive integers")
    n, block = int(n), int(block)
    if n < 2 or not 1 <= block <= n:
        raise ValueError("Block length must be between one and calendar length")
    starts = rng.integers(0, n, size=math.ceil(n / block))
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def sparse_to_calendar(values, query_indices, calendar_length):
    """Map sparse queries onto their original hourly calendar, never compress it."""
    values = finite_array(values, "sparse values")
    if values.ndim == 1:
        values = values[None, :]
    indices = np.asarray(query_indices)
    if (indices.ndim != 1 or len(indices) != values.shape[-1] or
            not np.issubdtype(indices.dtype, np.integer) or
            len(np.unique(indices)) != len(indices) or np.any(indices < 0) or
            np.any(indices >= calendar_length)):
        raise ValueError("Sparse query indices must be unique original calendar positions")
    dense = np.full((*values.shape[:-1], calendar_length), np.nan)
    dense[..., indices] = values
    return dense


def bootstrap_distribution(values, statistic, block=192, repetitions=4999, seed=20261106):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 2 or np.isinf(values).any():
        raise ValueError("A finite or sparse-NaN calendar array is required")
    if repetitions < 99:
        raise ValueError("At least 99 bootstrap draws are required")
    rng = np.random.default_rng(seed)
    result = np.empty(repetitions, dtype=np.float64)
    for b in range(repetitions):
        indices = circular_block_indices(values.shape[-1], block, rng)
        result[b] = statistic(values[..., indices], indices)
    finite = result[np.isfinite(result)]
    if len(finite) < 0.95 * repetitions:
        raise ValueError("Insufficient bootstrap draws with estimable strata/query coverage")
    return finite


def centered_upper_p(observed, bootstrap_values, null_boundary=0.0):
    """Approximate one-sided frontier bootstrap test, with finite-draw correction."""
    boot = finite_array(bootstrap_values, "bootstrap values", 1)
    return float((1 + np.count_nonzero(boot - observed >= observed - null_boundary)) / (len(boot) + 1))


def interval_record(observed, boot, null_boundary=0.0):
    return {"effect": float(observed), "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "p_one_sided_unadjusted": centered_upper_p(observed, boot, null_boundary),
            "null_boundary": null_boundary, "finite_bootstrap_draws": len(boot),
            "interval_kind": "nominal percentile interval; not a simultaneous-family interval"}


def holm(pvalues):
    p = finite_array(pvalues, "p-values", 1)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    result = np.empty_like(p)
    result[order] = np.minimum(1.0, adjusted)
    return result


def region_gap(squared_error, high_mask):
    errors = finite_array(squared_error, "squared error", 2)
    mask = np.asarray(high_mask, dtype=bool)
    if mask.shape != (errors.shape[1],) or not mask.any() or mask.all():
        raise ValueError("Both preregistered regions must contain origins")
    if np.any(errors < 0):
        raise ValueError("Squared errors cannot be negative")
    high = np.sqrt(errors[:, mask].mean(axis=1))
    normal = np.sqrt(errors[:, ~mask].mean(axis=1))
    return {"effect": float(np.mean(high - normal)), "high_RMSE_mean": float(high.mean()),
            "rest_RMSE_mean": float(normal.mean()), "per_seed_gap": (high - normal).tolist(),
            "n_high_origins": int(mask.sum()), "n_rest_origins": int((~mask).sum())}


def analyse_limitation(diagnostics, blocks=(192, 96, 384), repetitions=4999, seed=20261106):
    """Diagnostics maps each model to per-seed arrays, with one shared calendar."""
    if set(diagnostics) != set(PILOT_MODELS):
        raise ValueError("Exactly the three registered Top10 models are required")
    first = diagnostics["TimeMixer"]
    queries = np.asarray(first["query_indices"], dtype=np.int64)
    n = int(first["calendar_length"])
    masks = np.asarray(first["high_mask"], dtype=bool)
    dense = {}
    no_marks = {}
    errors = {}
    for name, data in diagnostics.items():
        if (int(data["calendar_length"]) != n or not np.array_equal(data["query_indices"], queries)
                or not np.array_equal(data["high_mask"], masks)):
            raise ValueError("Pilot models must share queries, calendar and train-defined regime")
        dense[name] = sparse_to_calendar(data["batch_delta"], queries, n)
        no_marks[name] = sparse_to_calendar(data["no_marks_delta"], queries, n)
        errors[name] = finite_array(data["val_squared_error_by_origin"], name + " errors", 2)
        if errors[name].shape[-1] != n:
            raise ValueError("Pilot error calendar length differs")
    results = {"state": "executed_pilot_conditional_on_learned_weights", "calendar_length": n,
               "query_origins": len(queries), "training_seeds": list(SEEDS),
               "no_new_independent_market_series": True, "calendar_input_used_only_in_external_ETTh1_task": True,
               "BTC_D7_omits_calendar_so_this_bug_is_not_automatically_a_BTC_limitation": True,
               "simple_marker_correction_is_not_an_original_E3_architecture": True,
               "bootstrap": {"unit": "hourly forecast origin; retain complete horizons/channels and paired seed trajectories",
                             "method": "circular moving-block bootstrap on the original dense hourly calendar",
                             "primary_block_hours": blocks[0], "sensitivity_blocks_hours": list(blocks[1:]),
                             "draws": repetitions, "seed": seed, "seeds_resampled_as_markets": False,
                             "assumptions_and_limits": "Approximate weak temporal dependence; finite roughly14.5 primary blocks; no exact coverage claim under distribution shift",
                             "sparse_query_calendar_not_compressed": True}, "by_block": {}}
    for block in blocks:
        l1 = []
        mean_stat = lambda x, index: float(np.nanmean(x))
        for name, array in dense.items():
            observed = float(np.nanmean(array))
            boot = bootstrap_distribution(array, mean_stat, block, repetitions, seed)
            l1.append({"comparison": name + " batch sensitivity", "model": name,
                       **interval_record(observed, boot, ALIGNMENT_TOLERANCE),
                       "per_seed_sensitivity": np.nanmean(array, axis=1).tolist()})
        contrasts = []
        for control in ("iTransformer", "PatchTST"):
            array = dense["TimeMixer"] - dense[control]
            observed = float(np.nanmean(array))
            boot = bootstrap_distribution(array, mean_stat, block, repetitions, seed)
            contrasts.append({"comparison": "TimeMixer minus " + control,
                              **interval_record(observed, boot)})
        # Family L1 is the registered guide-model boundary plus two controls.
        principal = [next(r for r in l1 if r["model"] == "TimeMixer"), *contrasts]
        for record, adjusted in zip(principal, holm([r["p_one_sided_unadjusted"] for r in principal])):
            record["p_one_sided_Holm_family3"] = float(adjusted)
        l2 = []
        for name, err in errors.items():
            point = region_gap(err, masks)
            def statistic(x, indices):
                mask = masks[indices]
                if not mask.any() or mask.all(): return np.nan
                return float(np.mean(np.sqrt(x[:, mask].mean(axis=1)) - np.sqrt(x[:, ~mask].mean(axis=1))))
            boot = bootstrap_distribution(err, statistic, block, repetitions, seed)
            l2.append({"model": name, **point, **interval_record(point["effect"], boot)})
        for record, adjusted in zip(l2, holm([r["p_one_sided_unadjusted"] for r in l2])):
            record["p_one_sided_Holm_family3"] = float(adjusted)
            record["positive_gap_supported"] = bool(adjusted < .05 and record["ci95"][0] > 0)
        results["by_block"][str(block)] = {
            "L1_batch_sensitivity": l1, "L1_paired_control_contrasts": contrasts,
            "L1_calendar_disabled_same_weights": {k: {"mean_absolute_prediction_change": float(np.nanmean(v)),
                                                       "per_seed": np.nanmean(v, axis=1).tolist()}
                                                   for k, v in no_marks.items()},
            "L2_high_input_variability": l2,
            "scope": "Exploratory validation pilot; not a full benchmark or population-wide architectural/family claim"}
    return results


def array_alias(data, names):
    for name in names:
        if name in data: return np.asarray(data[name])
    raise ValueError("Missing array field: " + "/".join(names))


def find_run(root, model, seed, pilot=False):
    root = Path(root)
    choices = ([root/model/f"seed_{seed}", root/model/str(seed), root/f"{model}_seed_{seed}"]
               if pilot else [root/f"seed_{seed}", root/str(seed), root/"TimeMixer"/f"seed_{seed}"])
    found = [p for p in choices if p.is_dir()]
    if len(found) != 1:
        raise ValueError(f"Expected one private run for {model}, seed {seed}; found {len(found)}")
    return found[0]


def load_replication(root, expected_design_sha256=None):
    rows, errors, receipts = [], [], []
    origins_ref = None
    targets_ref = None
    root = Path(root)
    score_directory = root/"replication_scores" if (root/"replication_scores").is_dir() else root
    completed_path = score_directory/"completed.json"
    if not completed_path.is_file():
        raise ValueError("All registered seeds must have frozen scores before analysis")
    completed = json.loads(completed_path.read_text())
    if expected_design_sha256 and completed.get("design_sha256") != expected_design_sha256:
        raise ValueError("Scores use another registered design")
    score_receipts = {r["seed"]:r for r in completed.get("rows",[])}
    if set(score_receipts) != set(SEEDS): raise ValueError("Incomplete five-seed score receipt")
    for seed in SEEDS:
        direct = [root/f"seed_{seed}_test.npz", root/"replication_scores"/f"seed_{seed}_test.npz"]
        choices = [p for p in direct if p.is_file()]
        if not choices:
            run = find_run(root, "TimeMixer", seed)
            choices = [p for p in [run/"score.npz", run/"predictions.npz", run/"test_predictions.npz"] if p.is_file()]
        if len(choices) != 1: raise ValueError("Expected exactly one scored prediction file per seed")
        path = choices[0]
        if digest(path) != score_receipts[seed].get("prediction_npz_sha256"):
            raise ValueError("Frozen scored prediction bytes differ")
        with np.load(path, allow_pickle=False) as arrays:
            pred = finite_array(array_alias(arrays, ["pred", "preds", "predictions"]), "predictions", 3)
            target = finite_array(array_alias(arrays, ["target", "targets", "true"]), "targets", 3)
            origins = array_alias(arrays, ["origins", "forecast_origins"])
        if pred.shape != target.shape or pred.shape != (2688, 96, 7):
            raise ValueError("Original preset needs retained test shape [2688,96,7]")
        if origins.shape != (2688,) or not np.array_equal(origins, np.arange(11520, 11520 + 2688)):
            raise ValueError("Retained original test origins differ")
        if origins_ref is not None and (not np.array_equal(origins, origins_ref) or not np.array_equal(target, targets_ref)):
            raise ValueError("Seeds must evaluate identical targets and calendar")
        origins_ref, targets_ref = origins, target
        # Preserve the author's float32 aggregate metric; use float64 only for
        # the separate per-origin bootstrap sufficient statistics.
        source_error = pred.astype(np.float32)-target.astype(np.float32)
        err = pred - target
        rows.append({"seed": seed, "MSE": float(np.square(source_error).mean()), "MAE": float(np.abs(source_error).mean()),
                     "retained_test_origins": len(origins), "nominal_test_origins": 2785,
                     "omitted_tail_origins": 97, "scale": "training-standardized seven-channel ETTh1"})
        errors.append(np.square(err).mean(axis=(1, 2)))
        receipts.append({"file": path.name, "seed": seed, "sha256": digest(path)})
    return rows, np.stack(errors), receipts


def load_diagnostics(root, expected_design_sha256=None):
    result, receipts = {}, []
    for name in PILOT_MODELS:
        arrays_by_seed = []
        for seed in SEEDS:
            choices = [Path(root)/name/f"seed_{seed}"/"diagnostics.npz",
                       Path(root)/f"{name}_seed_{seed}"/"diagnostics.npz"]
            files = [p for p in choices if p.is_file()]
            if len(files) != 1: raise ValueError(f"Missing/ambiguous diagnostics: {name}/{seed}")
            path = files[0]
            receipt_path=path.with_name("receipt.json")
            if not receipt_path.is_file():raise ValueError("Missing learned-pilot diagnostic receipt")
            receipt=json.loads(receipt_path.read_text())
            if (receipt.get("model")!=name or receipt.get("seed")!=seed or
                    digest(path)!=receipt.get("diagnostics_sha256") or
                    (expected_design_sha256 and receipt.get("design_sha256")!=expected_design_sha256)):
                raise ValueError("Pilot diagnostic receipt/hash/registered design differs")
            with np.load(path, allow_pickle=False) as z: arrays_by_seed.append({k: z[k] for k in z.files})
            receipts.append({"model": name, "seed": seed, "file": path.name, "sha256": digest(path)})
        first = arrays_by_seed[0]
        for item in arrays_by_seed[1:]:
            for key in ["query_indices", "high_mask", "variability", "high_threshold"]:
                if not np.array_equal(item[key], first[key]): raise ValueError("Pilot cohorts/threshold differ between seeds")
        result[name] = {"query_indices": first["query_indices"], "calendar_length": len(first["high_mask"]),
                        "high_mask": first["high_mask"], "high_threshold": float(first["high_threshold"]),
                        "batch_delta": np.stack([a["batch_delta"] for a in arrays_by_seed]),
                        "no_marks_delta": np.stack([a["no_marks_delta"] for a in arrays_by_seed]),
                        "val_squared_error_by_origin": np.stack([a["val_squared_error_by_origin"] for a in arrays_by_seed])}
    return result, receipts


def validate_analysis_design(design):
    """Refuse statistical choices that differ from the pre-result contract."""
    analysis=design["analysis"]; criterion=design["replication_criterion"]
    expected={"bootstrap_replicates":4999,"bootstrap_seed":20261106,
              "block_lengths":[192,96,384],"alignment_tolerance":1e-5,
              "alignment_queries":512,"historical_bank_size":127}
    if any(analysis.get(k)!=v for k,v in expected.items()):
        raise ValueError("Statistical analysis differs from the registered calendar/block/query protocol")
    if (criterion.get("reference_value")!=REFERENCE_MSE or
            criterion.get("absolute_tolerance")!=.01875 or
            criterion.get("relative_tolerance")!=.05 or
            criterion.get("MAE_secondary_reference")!=REFERENCE_MAE or
            criterion.get("not_Bitcoin_delta") is not True or
            criterion.get("strict_paper_row_identity_confirmed") is not False):
        raise ValueError("Replication criterion or its documented original-row uncertainty differs")
    if (design["replication"].get("seeds")!=list(SEEDS) or
            design["pilot"].get("seeds")!=list(SEEDS) or
            set(design["pilot"].get("models",[]))!=set(PILOT_MODELS)):
        raise ValueError("The full registered seeds and three pilot models are required")


def plots(output, replication, limitations):
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e2-paper-guide")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if replication:
        row = replication["MSE"]
        fig, ax = plt.subplots(figsize=(8, 3))
        ci = row["difference_ci95"]
        ax.axvspan(-.01875, .01875, alpha=.15, color="tab:green", label="Margen registrado ±5 %")
        ax.errorbar([row["difference_from_paper"]], [0], xerr=[[row["difference_from_paper"]-ci[0]], [ci[1]-row["difference_from_paper"]]], fmt="o", capsize=5)
        ax.axvline(0, color="black", linewidth=.8); ax.set(yticks=[], xlabel="MSE media réplica − referencia publicada 0,375", title="IC95 % entre semillas, condicionado al calendario retenido")
        ax.legend(); fig.tight_layout(); fig.savefig(output/"replication_discrepancy.png", dpi=160); plt.close(fig)
    if limitations:
        rows = limitations["by_block"]["192"]["L1_batch_sensitivity"]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for i, r in enumerate(rows):
            lo, hi = r["ci95"];ax.errorbar(r["effect"], i, xerr=[[max(0,r["effect"]-lo)], [max(0,hi-r["effect"])]], fmt="o", capsize=4)
        ax.set(yticks=range(len(rows)), yticklabels=[r["model"] for r in rows], xlabel="Cambio absoluto medio de predicción B1 frente a banco B128", title="Piloto aprendido: dependencia del batch, IC95 % temporal")
        ax.axvline(ALIGNMENT_TOLERANCE, color="grey", linestyle="--");fig.tight_layout();fig.savefig(output/"pilot_batch_invariance.png",dpi=160);plt.close(fig)
        rows=limitations["by_block"]["192"]["L2_high_input_variability"]
        fig,ax=plt.subplots(figsize=(8,3.6))
        for i,r in enumerate(rows):
            lo,hi=r["ci95"];ax.errorbar(r["effect"],i,xerr=[[max(0,r["effect"]-lo)],[max(0,hi-r["effect"])]],fmt="o",capsize=4)
        ax.axvline(0,color="black",linewidth=.8);ax.set(yticks=range(len(rows)),yticklabels=[r["model"] for r in rows],xlabel="RMSE alta variación observable − resto",title="Piloto en validación: heterogeneidad de error, IC95 % temporal")
        fig.tight_layout();fig.savefig(output/"pilot_regime_error.png",dpi=160);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--design",required=True);p.add_argument("--replication-root");p.add_argument("--pilot-root")
    p.add_argument("--output-dir",required=True);p.add_argument("--bootstrap-draws",type=int,default=4999)
    args=p.parse_args(argv)
    if args.bootstrap_draws != 4999: raise SystemExit("The registered public analysis requires 4999 bootstrap draws")
    design_path=Path(args.design);design=json.loads(design_path.read_text());validate_analysis_design(design)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    replicas=limitations=None;inputs=[]
    if args.replication_root:
        rows,errors,receipts=load_replication(args.replication_root,digest(design_path));inputs.extend(receipts)
        replicas=replication_summary([r["MSE"] for r in rows],[r["MAE"] for r in rows])
        replicas["temporal_uncertainty_separate_from_registered_seed_success"]={}
        for b in (192,96,384):
            boot=bootstrap_distribution(errors,lambda x,index:float(x.mean()),b,4999,20261106)
            replicas["temporal_uncertainty_separate_from_registered_seed_success"][str(b)]={"MSE":float(errors.mean()),"ci95":np.quantile(boot,[.025,.975]).tolist(),"unit":"forecast origin, full96x7error retained; fixed fitted seeds"}
        write_json(out/"replication_summary.json",replicas)
        with (out/"replication_seed_metrics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if args.pilot_root:
        diagnostics,receipts=load_diagnostics(args.pilot_root,digest(design_path));inputs.extend(receipts)
        limitations=analyse_limitation(diagnostics,repetitions=4999)
        limitations["high_variability_threshold_train_only"]=diagnostics["TimeMixer"]["high_threshold"]
        write_json(out/"limitation_summary.json",limitations)
    if replicas is None and limitations is None: raise SystemExit("At least one completed private experiment input is required")
    plots(out,replicas,limitations)
    hashes={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="provenance.json"}
    write_json(out/"provenance.json",{"generator":"scripts/analyze_e2_paper_guide.py","generator_sha256":digest(__file__),"design_sha256":digest(design_path),"private_input_receipts":inputs,"output_hashes":hashes,"individual_forecasts_exported":False,"weights_exported":False,"dataset_CSV_exported":False,"new_Bitcoin_results":0,"scope":"External ETTh1 partial replication and exploratory learned pilot; not the Bitcoin benchmark"})
    print(json.dumps({"replication_analyzed":replicas is not None,"pilot_analyzed":limitations is not None,"new_Bitcoin_results":0}))


if __name__=="__main__":main()
