#!/usr/bin/env python3
"""Post hoc residual taxonomy and predictability references from frozen forecasts.

No fitting, model inference, new weight selection or primary benchmark mutation.
The automatic labeling rules are recorded before generating the labels. This is
explicitly post hoc: the original targets and scores were already known.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

VERSION = "e2-error-ceiling-posthoc-audit-v1"
INPUT_SHA = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
DEV_SHA = "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a"
MODELS = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer")
SEEDS = tuple(range(271801, 271807))
CATEGORIES = ("small_reference_band", "moderate_underestimate", "moderate_overestimate",
              "large_underestimate", "large_overestimate")
LABELS = ("Banda pequeña", "Subestimación moderada", "Sobreestimación moderada",
          "Subestimación grande", "Sobreestimación grande")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def validate_payload(payload):
    dates = pd.to_datetime(payload["dates"], utc=True)
    y = np.asarray(payload["y_true"], dtype=float)
    if (len(dates) != 608 or len(set(dates)) != 608 or y.shape != (608,)
            or dates[0] != pd.Timestamp("2025-01-01", tz="UTC")
            or dates[-1] != pd.Timestamp("2026-08-31", tz="UTC")
            or not np.all(np.diff(dates.asi8) == pd.Timedelta(days=1).value)):
        raise ValueError("Expected the fixed daily 608-date chronology")
    if not np.isfinite(y).all() or not np.asarray(payload["mask"], dtype=bool).all():
        raise ValueError("Nonfinite targets or changed common mask")
    predictors = {}
    for p in payload["predictors"]:
        key = (p["model"], p["seed"])
        pred = np.asarray(p["prediction_original"], dtype=float)
        if key in predictors or pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError("Duplicate, malformed or nonfinite forecast")
        predictors[key] = pred
    for model in MODELS:
        if {s for m, s in predictors if m == model} != set(SEEDS):
            raise ValueError("Incomplete main-model seed grid")
    baseline = [p for (m, _), p in predictors.items() if m == "Persistence"]
    if len(baseline) != 1:
        raise ValueError("Expected one Persistence forecast")
    forecasts = np.asarray([[predictors[(m, s)] for s in SEEDS] for m in MODELS])
    original = {r["model"]: r for r in payload["original_analysis"]["table_A2"]}
    for model, pred in zip(MODELS, forecasts):
        score = float(np.sqrt(np.mean((pred - y) ** 2, axis=1)).mean())
        if abs(score - original[model]["RMSE"]) > 1e-10:
            raise ValueError(f"Original mean-seed RMSE mismatch: {model}")
    return dates, y, forecasts, baseline[0]


def development_thresholds(frame):
    dates = pd.to_datetime(frame["market_day_utc"], utc=True)
    values = frame["sigma_7"].to_numpy(dtype=float)
    if dates.duplicated().any() or not set(dates.dt.year).issubset({2023, 2024}):
        raise ValueError("Development must be unique and restricted to 2023–2024")
    finite = np.isfinite(values)
    if np.any(values[finite] < 0):
        raise ValueError("Volatility must be nonnegative")
    consecutive = np.diff(dates.astype("int64")) == pd.Timedelta(days=1).value
    adjacent = finite[1:] & finite[:-1] & consecutive
    changes = np.abs(np.diff(values))[adjacent]
    if len(changes) < 100:
        raise ValueError("Insufficient consecutive finite development observations")
    low, high = np.quantile(changes, [.5, .9], method="linear")
    q25, q75 = np.quantile(values[finite], [.25, .75], method="linear")
    if not 0 < low < high or not q25 < q75:
        raise ValueError("Degenerate reference thresholds")
    return {"source_years": [2023, 2024], "finite_target_days": int(finite.sum()),
            "consecutive_finite_change_pairs": int(len(changes)),
            "absolute_daily_change_q50": float(low), "absolute_daily_change_q90": float(high),
            "target_q25": float(q25), "target_q75": float(q75),
            "quantile_method": "linear", "missing_observations_bridged": False,
            "scope": "Development reference scale, not an irreducible noise floor"}


def classify_errors(errors, low, high):
    errors = np.asarray(errors, dtype=float)
    if not np.isfinite(errors).all() or not 0 < low < high:
        raise ValueError("Finite errors and ordered positive thresholds required")
    a = np.abs(errors)
    categories = np.zeros(errors.shape, dtype=np.int8)
    categories[(a > low) & (a <= high) & (errors < 0)] = 1
    categories[(a > low) & (a <= high) & (errors >= 0)] = 2
    categories[(a > high) & (errors < 0)] = 3
    categories[(a > high) & (errors >= 0)] = 4
    return categories


def stratified_sample(strata, sample_size=100, seed=272101):
    """Proportional allocation: one per stratum, then largest remainders."""
    strata = np.asarray(strata)
    groups = {str(k): np.flatnonzero(strata == k) for k in sorted(set(strata))}
    if not len(groups) <= sample_size <= len(strata):
        raise ValueError("Sample size cannot represent all strata")
    counts = {k: 1 for k in groups}
    remaining = sample_size - len(groups)
    capacity = {k: len(v) - 1 for k, v in groups.items()}
    total = sum(capacity.values())
    ideal = {k: remaining * v / total if total else 0. for k, v in capacity.items()}
    for k in groups:
        counts[k] += math.floor(ideal[k])
    missing = sample_size - sum(counts.values())
    order = sorted(groups, key=lambda k: (-(ideal[k] - math.floor(ideal[k])), k))
    for k in order[:missing]:
        counts[k] += 1
    rng = np.random.default_rng(seed)
    selected, design = [], []
    for k, population in groups.items():
        n = counts[k]
        selected.extend(int(i) for i in rng.choice(population, n, replace=False))
        design.append({"stratum": k, "population_dates": len(population), "sample_dates": n,
                       "inclusion_probability": n / len(population),
                       "design_weight": len(population) / n})
    return np.asarray(sorted(selected), dtype=int), design


def block_indices(rng, n, draws, block):
    if not 1 <= block <= n or draws < 1:
        raise ValueError("Invalid block/draws")
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def bootstrap_profiles(profiles, draws=10000, block=14, seed=272102):
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim != 2 or not np.isfinite(profiles).all() or draws < 10000:
        raise ValueError("Finite date profiles and at least 10000 draws required")
    rng = np.random.default_rng(seed)
    replicates = np.empty((draws, profiles.shape[1]))
    for start in range(0, draws, 128):
        stop = min(start + 128, draws)
        idx = block_indices(rng, len(profiles), stop - start, block)
        replicates[start:stop] = profiles[idx].mean(axis=1)
    return np.quantile(replicates, [.025, .975], axis=0).T


def permutation_entropy(values, order=3, delay=1):
    """Bandt–Pompe entropy; omit incomplete windows, stable time-order tie rule."""
    values = np.asarray(values, dtype=float)
    if order < 2 or delay < 1:
        raise ValueError("Order >= 2 and delay >= 1 required")
    patterns, ties = Counter(), 0
    for start in range(len(values) - (order - 1) * delay):
        window = values[start:start + order * delay:delay]
        if not np.isfinite(window).all():
            continue
        ties += int(len(set(window)) < order)
        patterns[tuple(np.argsort(window, kind="stable"))] += 1
    n = sum(patterns.values())
    if n == 0:
        raise ValueError("No complete ordinal windows")
    p = np.asarray(list(patterns.values()), dtype=float) / n
    h = float(-np.sum(p * np.log(p)))
    return {"order": order, "delay": delay, "complete_windows": n,
            "distinct_patterns": len(patterns), "possible_patterns": math.factorial(order),
            "entropy_nats": h, "normalized_entropy": h / math.lgamma(order + 1),
            "windows_with_ties": ties, "ties_policy": "stable temporal order",
            "scope": "Ordinal complexity; no conversion to RMSE or Bayes risk"}


def gamma_reference(best, baseline, reference):
    denominator = baseline - reference
    raw = (best - reference) / denominator if denominator else None
    valid = bool(denominator > 0 and reference <= best <= baseline)
    return {"best_RMSE": best, "baseline_RMSE": baseline, "reference_RMSE": reference,
            "raw_algebraic_gamma": raw, "gamma_remaining_fraction": raw if valid else None,
            "valid_order_for_ceiling_gap": valid, "score_orientation": "s=-RMSE",
            "formula": "(RMSE_best-RMSE_reference)/(RMSE_baseline-RMSE_reference)",
            "status": "conditional_reference_only_not_identified_Bayes_ceiling" if valid
                      else "not_interpretable_as_ceiling_gap"}


def protocol(thresholds, draws, plan_path):
    return {"version": VERSION, "registered_at": datetime.now(ZoneInfo("America/Bogota")).isoformat(),
            "status": "posthoc_rules_recorded_before_automatic_label_generation",
            "test_and_original_scores_already_known": True,
            "original_preregistration_claimed": False, "manual_human_annotation_claimed": False,
            "scientific_fit": False, "model_inference": False, "weight_selection": False,
            "input_sha256": INPUT_SHA, "development_sha256": DEV_SHA,
            "script_sha256": sha256(__file__), "models": list(MODELS), "seeds": list(SEEDS),
            "published_plan_sha256": sha256(plan_path),
            "published_plan_path": "experiments/e2_error_ceiling_final_audit_20261114/analysis_plan.json",
            "published_plan_commit": "4361275cbc52e2f6362349557b471cda110bbefc",
            "thresholds": thresholds,
            "taxonomy_unit": "Date/model error of arithmetic mean of six frozen seed forecasts",
            "seed_guardrail": "Keep six individual-seed labels and seed agreement separately",
            "original_A2_metric_preserved": "Mean of individual seed RMSE, not RMSE of mean forecast",
            "categories_in_order": list(CATEGORIES),
            "rules": ["e=mean(frozen seed forecasts)-y; a=abs(e)",
                      "small iff a<=Q50", "moderate_under iff Q50<a<=Q90,e<0",
                      "moderate_over iff Q50<a<=Q90,e>=0",
                      "large_under iff a>Q90,e<0", "large_over iff a>Q90,e>=0"],
            "category_scope": "Exclusive residual descriptions, not causal explanations or expert assessment",
            "sample": {"dates_per_model": 100, "common_sample": True, "seed": 272101,
                       "strata": "year x observed target regime using development Q25/Q75",
                       "allocation": "One per stratum plus proportional residual capacity and largest remainders",
                       "error_dependent_selection": False, "individual_seed_cases_per_model": 600},
            "intervals": {"method": "circular moving block percentile bootstrap",
                          "draws": draws, "block_days_primary": 14,
                          "sensitivity_block_days": [7, 28], "seed": 272102,
                          "scope": "Marginal 95% temporal sensitivity conditional on frozen predictors; not finite-calendar counting uncertainty",
                          "assumptions": "Short dependence and approximate stationarity; structural breaks limit extrapolation"},
            "ensemble": "Uniform ten main models, matched seed IDs, no outcome-based weight tuning",
            "entropy": {"orders": [3, 4, 5], "delay": 1, "doi": "10.1103/PhysRevLett.88.174102"},
            "ideal_reference": "Zero forecast error is mathematical bound, not identified attainable/Bayes ceiling",
            "gamma_bootstrap_seed": 272103}


def analyze(input_path, development_path, output, draws=10000, plan_path=None):
    input_path, development_path, output = map(Path, (input_path, development_path, output))
    if draws < 10000 or sha256(input_path) != INPUT_SHA or sha256(development_path) != DEV_SHA:
        raise ValueError("Require 10000 draws and exact frozen source hashes")
    payload = json.loads(input_path.read_text())
    dates, y, forecasts, baseline = validate_payload(payload)
    development = pd.read_csv(development_path)
    thresholds = development_thresholds(development)
    plan_path = Path(plan_path) if plan_path else Path(__file__).resolve().parents[1] / "experiments/e2_error_ceiling_final_audit_20261114/analysis_plan.json"
    published_plan = json.loads(plan_path.read_text())
    planned = published_plan["taxonomy"]["development_thresholds"]
    if (published_plan["input_sha256"] != INPUT_SHA or
            published_plan["sample"]["seed"] != 272101 or
            published_plan["uncertainty"]["draws"] != draws or
            published_plan["taxonomy"]["seed_concordance"] != list(SEEDS) or
            abs(planned["L_abs_change_sigma7_Q50"] - thresholds["absolute_daily_change_q50"]) > 1e-16 or
            abs(planned["H_abs_change_sigma7_Q90"] - thresholds["absolute_daily_change_q90"]) > 1e-16):
        raise ValueError("Published post hoc analysis plan does not match implementation")
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "taxonomy_protocol.json"
    proposed = protocol(thresholds, draws, plan_path)
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if any(old.get(k) != v for k, v in proposed.items() if k != "registered_at"):
            raise FileExistsError("Existing labeling protocol changed: use explicit new output/version")
    else:
        dump(receipt_path, proposed)
    receipt = json.loads(receipt_path.read_text())
    # Labels are generated only after a protocol receipt exists.
    low, high = thresholds["absolute_daily_change_q50"], thresholds["absolute_daily_change_q90"]
    errors = forecasts.mean(axis=1) - y
    seed_errors = forecasts - y
    labels = classify_errors(errors, low, high)
    seed_labels = classify_errors(seed_errors, low, high)
    regimes = np.where(y <= thresholds["target_q25"], "low",
                       np.where(y <= thresholds["target_q75"], "middle", "high"))
    strata = np.asarray([f"{d.year}:{r}" for d, r in zip(dates, regimes)])
    selected, design = stratified_sample(strata)
    chosen = set(selected.tolist())
    design_by_stratum = {r["stratum"]: r for r in design}
    dump(output / "sampling_design.json", {"sample_dates": 100, "strata": design,
         "dates_sha256": hashlib.sha256("\n".join(dates[i].date().isoformat() for i in selected).encode()).hexdigest(),
         "common_sample_for_all_models": True, "error_dependent_selection": False})
    profiles = np.stack([(labels == c).T for c in range(5)], axis=2).reshape(608, -1).astype(float)
    cis = {block: bootstrap_profiles(profiles, draws, block).reshape(10, 5, 2)
           for block in (14, 7, 28)}
    rows, sample_seeds, frequencies, coverage = [], [], [], []
    for m, model in enumerate(MODELS):
        for i, date in enumerate(dates):
            ds = design_by_stratum[strata[i]]
            rows.append({"model": model, "date": date.date().isoformat(),
                "category": CATEGORIES[int(labels[m, i])], "observed_target_regime": regimes[i],
                "stratum": strata[i], "signed_error_mean_forecast": float(errors[m, i]),
                "absolute_error_mean_forecast": float(abs(errors[m, i])),
                "mean_absolute_individual_seed_error": float(np.abs(seed_errors[m, :, i]).mean()),
                "seed_error_SD_ddof1": float(seed_errors[m, :, i].std(ddof=1)),
                "seed_category_agreement": float((seed_labels[m, :, i] == labels[m, i]).mean()),
                "audit_sample": i in chosen, "sample_inclusion_probability": ds["inclusion_probability"],
                "sample_design_weight_if_selected": ds["design_weight"] if i in chosen else None})
            if i in chosen:
                for s, seed in enumerate(SEEDS):
                    sample_seeds.append({"model": model, "date": rows[-1]["date"], "seed": seed,
                        "category": CATEGORIES[int(seed_labels[m, s, i])],
                        "signed_error": float(seed_errors[m, s, i]), "stratum": strata[i],
                        "date_design_weight": ds["design_weight"]})
        for c, category in enumerate(CATEGORIES):
            weighted = sum(design_by_stratum[strata[i]]["design_weight"]
                           for i in selected if labels[m, i] == c) / len(y)
            row = {"model": model, "category": category,
                "calendar_count_exact": int((labels[m] == c).sum()),
                "calendar_fraction_exact": float((labels[m] == c).mean()),
                "audit_sample_count": int((labels[m, selected] == c).sum()),
                "audit_sample_weighted_fraction": weighted,
                "individual_seed_fraction_exact": float((seed_labels[m] == c).mean())}
            for block in (14, 7, 28):
                row[f"CI95_block{block}_low"] = float(cis[block][m, c, 0])
                row[f"CI95_block{block}_high"] = float(cis[block][m, c, 1])
            frequencies.append(row)
        coverage.append({"model": model, "labelled_dates": 608, "audit_distinct_dates": 100,
            "audit_individual_seed_cases": 600, "mean_seed_category_agreement": float((seed_labels[m] == labels[m]).mean()),
            "dates_seed_sign_disagreement": int(((seed_errors[m].min(axis=0) < 0) & (seed_errors[m].max(axis=0) > 0)).sum()),
            "RMSE_mean_forecast_diagnostic": float(np.sqrt(np.mean(errors[m] ** 2)))})
    labelled = pd.DataFrame(rows)
    labelled.to_csv(output / "all_6080_date_model_labels.csv", index=False)
    labelled.loc[labelled.audit_sample].to_csv(output / "audit_100_dates_per_main_model.csv", index=False)
    pd.DataFrame(sample_seeds).to_csv(output / "audit_600_seed_cases_per_main_model.csv", index=False)
    pd.DataFrame(frequencies).to_csv(output / "error_category_frequencies.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "model_annotation_coverage.csv", index=False)
    shared = []
    for a, b in itertools.combinations(range(10), 2):
        la, lb = labels[a] >= 3, labels[b] >= 3
        union, joint = int((la | lb).sum()), int((la & lb).sum())
        shared.append({"model_a": MODELS[a], "model_b": MODELS[b], "dates": len(y),
            "large_error_a_dates": int(la.sum()), "large_error_b_dates": int(lb.sum()),
            "shared_large_error_dates": joint, "shared_fraction_all_dates": joint / len(y),
            "large_error_Jaccard": joint / union if union else None,
            "residual_Pearson_correlation": float(np.corrcoef(errors[a], errors[b])[0, 1])})
    pd.DataFrame(shared).to_csv(output / "shared_error_diversity.csv", index=False)
    entropy = [{"period": period, **permutation_entropy(values, order)}
               for period, values in (("development_2023_2024", development.sigma_7.to_numpy()),
                                      ("known_test_2025_2026", y)) for order in (3, 4, 5)]
    pd.DataFrame(entropy).to_csv(output / "permutation_entropy.csv", index=False)
    original_rmse = np.sqrt(np.mean(seed_errors ** 2, axis=2)).mean(axis=1)
    winner = "LSTM-autoencoder con predictor"  # Original already public winner, not re-selected.
    best_losses = seed_errors[MODELS.index(winner)] ** 2
    base_losses = (baseline - y) ** 2
    ensemble = forecasts.mean(axis=0)
    ensemble_losses = (ensemble - y) ** 2
    best, base = float(original_rmse[MODELS.index(winner)]), float(np.sqrt(base_losses.mean()))
    ensemble_score = float(np.sqrt(ensemble_losses.mean(axis=1)).mean())
    empirical, ideal = gamma_reference(best, base, ensemble_score), gamma_reference(best, base, 0.)
    ideal["status"] = "zero_error_mathematical_bound_only_not_attainable_Bayes_ceiling"
    rng, boot = np.random.default_rng(272103), {k: [] for k in ("best", "baseline", "ensemble", "gamma_ideal")}
    for start in range(0, draws, 128):
        idx = block_indices(rng, len(y), min(128, draws - start), 14)
        b = np.sqrt(best_losses[:, idx].mean(axis=2)).mean(axis=0)
        p = np.sqrt(base_losses[idx].mean(axis=1))
        e = np.sqrt(ensemble_losses[:, idx].mean(axis=2)).mean(axis=0)
        for key, value in (("best", b), ("baseline", p), ("ensemble", e), ("gamma_ideal", b / p)):
            boot[key].extend(value.tolist())
    scores = {"version": VERSION, "input_sha256": INPUT_SHA, "n_dates": 608,
        "original_winner_unchanged": winner, "primary_statistic_unchanged": "mean individual-seed RMSE",
        "original_best_RMSE": best, "Persistence_RMSE": base, "uniform_ten_matched_seed_RMSE": ensemble_score,
        "uniform_all_60_forecast_mean_RMSE_diagnostic": float(np.sqrt(np.mean((ensemble.mean(axis=0) - y) ** 2))),
        "ensemble_RMSE_per_seed": np.sqrt(ensemble_losses.mean(axis=1)).tolist(),
        "empirical_ensemble_reference": empirical, "ideal_zero_error_reference": ideal,
        "temporal_CI95_block14": {key: np.quantile(val, [.025, .975]).tolist() for key, val in boot.items()},
        "Bayes_risk_identified": False, "irreducible_noise_floor_estimated": False,
        "ensemble_weights": {m: .1 for m in MODELS}, "weights_tuned_to_outcomes": False,
        "forecast_generation": "Arithmetic of saved predictions; no new model inference",
        "reference_warning": "Ensemble is worse than best and baseline: not a valid ceiling; algebraic gamma is not interpreted",
        "original_primary_superiority_or_equivalence_claim_changed": False,
        "test_scope": "Retrospective, already known since E1; not independent confirmation"}
    dump(output / "ceiling_gamma_references.json", scores)
    dump(output / "scientific_sources.json", {
        "ordinal_entropy_primary": {"authors": "Christoph Bandt; Bernd Pompe", "year": 2002,
            "title": "Permutation Entropy: A Natural Complexity Measure for Time Series",
            "journal": "Physical Review Letters", "volume": 88, "article": 174102,
            "doi": "10.1103/PhysRevLett.88.174102",
            "primary_url": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.174102",
            "verified_url": "https://pubmed.ncbi.nlm.nih.gov/12005759/",
            "supports": "Ordinal complexity descriptor, not a conversion to Bayes error",
            "verified_at": receipt["registered_at"]},
        "block_bootstrap_primary": {"authors": "Hans R. Künsch", "year": 1989,
            "title": "The Jackknife and the Bootstrap for General Stationary Observations",
            "doi": "10.1214/aos/1176347265",
            "scope": "Dependence-aware resampling under stationarity assumptions"}})
    write_report(output, receipt, scores, frequencies, coverage, entropy)
    manifest = {"version": VERSION, "created_at": datetime.now(ZoneInfo("America/Bogota")).isoformat(),
        "input_sha256": INPUT_SHA, "development_sha256": DEV_SHA, "script_sha256": sha256(__file__),
        "published_plan_sha256": sha256(plan_path),
        "published_plan_commit": "4361275cbc52e2f6362349557b471cda110bbefc",
        "scientific_fit": False, "model_inference": False, "benchmark_repeated": False,
        "primary_result_mutated": False, "annotated_main_models": 10, "date_model_labels": len(rows),
        "distinct_audit_dates_per_model": 100, "audit_seed_cases_per_model": 600,
        "bootstrap_draws_per_interval": draws,
        "files": {p.name: sha256(p) for p in sorted(output.iterdir())
                  if p.is_file() and p.name != "manifest.json"}}
    dump(output / "manifest.json", manifest)
    return manifest


def write_report(output, receipt, scores, frequencies, coverage, entropy):
    t = receipt["thresholds"]
    lines = [
        "# Taxonomía de errores y referencias de predictibilidad", "",
        "Este complemento se calculó después de conocer el test y sus resultados originales. "
        "Las reglas se conservaron antes de generar sus etiquetas automáticas. No es un prerregistro "
        "del experimento original ni una revisión humana. Sólo utiliza pronósticos guardados y la "
        "volatilidad de desarrollo 2023–2024: no hubo entrenamiento, inferencia de modelos ni ajuste de pesos.", "",
        f"Reglas registradas en {receipt['registered_at']} (America/Bogota). "
        f"Exportación congelada SHA-256: {INPUT_SHA}.", "",
        "## Categorías y muestra auditable", "",
        "La unidad principal es una fecha por modelo; se usa la media aritmética de sus seis pronósticos "
        "congelados. Es un predictor agregado diagnóstico y no sustituye la tabla A2, cuya estadística "
        "sigue siendo la media de los RMSE individuales. Se publican además las etiquetas por semilla y "
        "su concordancia para hacer visible la cancelación de errores en la media.", "",
        f"Con e = pronóstico − observado, los cortes son a = {t['absolute_daily_change_q50']:.12g} "
        f"y b = {t['absolute_daily_change_q90']:.12g}: Q50 y Q90 de los "
        f"{t['consecutive_finite_change_pairs']} cambios absolutos diarios consecutivos del objetivo "
        "de desarrollo. No se atraviesan huecos. Son escalas descriptivas, no márgenes confirmatorios "
        "de relevancia ni ruido irreducible.", "",
        "| Categoría exclusiva | Regla |", "|---|---|",
        "| Banda pequeña | abs(e) ≤ a |",
        "| Subestimación moderada | a < abs(e) ≤ b y e < 0 |",
        "| Sobreestimación moderada | a < abs(e) ≤ b y e ≥ 0 |",
        "| Subestimación grande | abs(e) > b y e < 0 |",
        "| Sobreestimación grande | abs(e) > b y e ≥ 0 |", "",
        "Se etiquetaron las 608 fechas de cada uno de los diez modelos: **6.080 casos fecha-modelo**. "
        "La muestra conserva **100 fechas distintas por modelo**, con las seis semillas por fecha "
        "(**600 errores individuales por modelo**). Son casos con residuos continuos; no se afirma que "
        "cada uno sea un fallo grave. La selección común se estratificó por año y régimen observado "
        "del objetivo, con Q25/Q75 calculados sólo en desarrollo y semilla 272101. No depende de los errores. "
        "Estos regímenes son descriptivos, no entradas de predicción ni causas económicas.", "",
        "| Modelo | Fechas etiquetadas | Fechas auditables | Casos por semilla | Concordancia de etiquetas |",
        "|---|---:|---:|---:|---:|"]
    for r in coverage:
        lines.append(f"| {r['model']} | 608 | 100 | 600 | {r['mean_seed_category_agreement']:.1%} |")
    lines.extend(["",
        "Las frecuencias del calendario completo son recuentos exactos de este conjunto finito. "
        "Los IC del 95% son intervalos marginales de sensibilidad temporal: 10.000 remuestreos "
        "de bloques circulares de 14 días, con sensibilidad a 7 y 28 días. Suponen dependencia "
        "corta y estacionariedad aproximada; no son incertidumbre de contar estos 608 días ni "
        "garantizan cobertura simultánea de todas las categorías. Los cambios de régimen limitan la extrapolación.", "",
        "| Modelo | Banda pequeña | Subest. moderada | Sobreest. moderada | Subest. grande | Sobreest. grande |",
        "|---|---:|---:|---:|---:|---:|"])
    for model in MODELS:
        cells = [f"{r['calendar_fraction_exact']:.1%} [{r['CI95_block14_low']:.1%}, {r['CI95_block14_high']:.1%}]"
                 for r in frequencies if r["model"] == model]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.extend(["",
        "Los archivos detallados conservan los pesos de inclusión N_h/n_h, frecuencias muestrales "
        "ponderadas, la concordancia de semillas y correlación/intersección/Jaccard de errores grandes "
        "para los 45 pares de modelos. Las categorías describen residuos; no sustituyen una "
        "taxonomía semántica anotada por expertos.", "",
        "## Referencias de predictibilidad y distancia γ", "",
        "Se calculó entropía de permutación de Bandt y Pompe (2002; "
        "[DOI 10.1103/PhysRevLett.88.174102](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.88.174102)), "
        "con órdenes 3, 4 y 5 y retardo 1, sin atravesar faltantes. Es complejidad ordinal, no "
        "una conversión a error Bayes, un porcentaje de predictibilidad o un RMSE mínimo alcanzable.", "",
        "| Periodo | Orden | Ventanas | Entropía normalizada |", "|---|---:|---:|---:|"])
    for r in entropy:
        lines.append(f"| {r['period']} | {r['order']} | {r['complete_windows']} | {r['normalized_entropy']:.6f} |")
    lines.extend(["",
        f"El ensemble uniforme de los diez modelos, emparejado por semilla y sin ajustar pesos, "
        f"alcanza RMSE medio **{scores['uniform_ten_matched_seed_RMSE']:.10f}**. "
        f"SAE conserva **{scores['original_best_RMSE']:.10f}** y Persistence "
        f"**{scores['Persistence_RMSE']:.10f}**. El ensemble resulta peor: **no constituye un techo "
        "empírico válido**. Se conserva el resultado negativo, sin buscar una combinación más favorable.", "",
        "Con s = −RMSE, la fórmula de la guía se orienta así:", "",
        r"$$\gamma = \frac{RMSE_{mejor}-RMSE_{referencia}}{RMSE_{base}-RMSE_{referencia}}.$$",
        "",
        f"Con el ensemble, el cociente algebraico es "
        f"{scores['empirical_ensemble_reference']['raw_algebraic_gamma']:.6f}; el denominador es negativo "
        "y **no se interpreta como fracción pendiente**. Frente a la cota matemática ideal RMSE = 0, "
        f"**γ_ideal = {scores['ideal_zero_error_reference']['gamma_remaining_fraction']:.6f}**, "
        f"IC temporal de 14 días [{scores['temporal_CI95_block14']['gamma_ideal'][0]:.6f}, "
        f"{scores['temporal_CI95_block14']['gamma_ideal'][1]:.6f}]. Es la fracción del error de Persistence "
        "que permanece respecto a una predicción perfecta; **no estima la brecha frente a un techo real alcanzable**. "
        "El riesgo Bayes y el ruido irreducible permanecen sin identificar.", "",
        "El complemento documenta las referencias ensayadas, la distancia condicionada y sus límites; "
        "no acredita un techo de desempeño identificado. No modifica la ausencia de superioridad práctica "
        "confirmada ni convierte el test conocido desde E1 en prueba independiente.", "",
        "## Archivos auditables", "",
        "- taxonomy_protocol.json: reglas, fecha, umbrales, semillas y alcance post hoc.",
        "- all_6080_date_model_labels.csv: etiquetas del calendario completo.",
        "- audit_100_dates_per_main_model.csv y audit_600_seed_cases_per_main_model.csv: muestra y semillas.",
        "- error_category_frequencies.csv: frecuencias e IC de 7, 14 y 28 días.",
        "- shared_error_diversity.csv: solapamiento y correlación de los 45 pares.",
        "- permutation_entropy.csv y ceiling_gamma_references.json: referencias y límites de γ.",
        "- manifest.json: hashes y declaración de ausencia de ajuste/inferencia."])
    (output / "proposed_chapter.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--development", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--draws", type=int, default=10000)
    p.add_argument("--plan", type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.input, a.development, a.output, a.draws, a.plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
