"""Post hoc standardized effects and causal-regime RQ2 from frozen predictions.

This program never imports a model framework, loads weights, fits a model,
changes A2, or alters the original RQ1 hypothesis family. Register its immutable
plan before execution. RQ2 conditions on the six existing fitted trajectories;
the separate standardized-effect intervals describe optimization seeds only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "book_e2/_static/e2_final_audit_inputs_20261114"
SPEC = ROOT / "book_e2/_static/e2_point1_review_v1/rq_specification.json"
DEFAULT_OUTPUT = ROOT / "book_e2/_static/e2_corrections_20261115/statistics"
VERSION = "e2-effects-rq2-closure-v1"
TOP10 = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
         "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer")
NEURAL = TOP10 + ("MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
SEEDS = tuple(range(271801, 271807))
DELTA = 0.00025772842664981106
Q75 = 0.024247163097617024
PINNED_INPUTS = {
    "e2_f2a_input.json": "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187",
    "daily_volatility_development.csv": "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def software_record():
    return {"python": platform.python_version(), "numpy": importlib.metadata.version("numpy"),
            "platform": platform.platform(), "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "thread_environment": {name: os.environ.get(name)
                                   for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def input_records():
    paths = {name: INPUTS / name for name in PINNED_INPUTS}
    paths["rq_specification.json"] = SPEC
    paths["benchmark_A2.csv"] = ROOT / "book_e2/_static/e2_benchmark_final_20261113/benchmark_A2.csv"
    result = {}
    for name, path in paths.items():
        require(path.is_file(), f"Missing input {path}")
        digest = sha256(path)
        if name in PINNED_INPUTS:
            require(digest == PINNED_INPUTS[name], f"Frozen input changed: {name}")
        result[name] = {"repository_path": path.relative_to(ROOT).as_posix(),
                        "bytes": path.stat().st_size, "sha256": digest}
    return result


def register_plan(output, *, draws=20000):
    require(type(draws) is int and draws >= 20000, "At least 20,000 draws required")
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "analysis_plan.json").exists(), "Plan exists; do not overwrite a registration")
    spec = json.loads(SPEC.read_text())
    require(spec["RQ2"]["q75_2023"] == Q75 and spec["common_estimand"]["delta"] == DELTA,
            "Historical regime threshold or practical margin changed")
    require(tuple(spec["common_estimand"]["secondary_candidates"]) == TOP10,
            "Secondary candidate catalogue changed")
    parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    value = {
        "version": VERSION, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_role": "posthoc_plan_after_A2_and_before_this_new_statistical_execution",
        "historical_preregistration_claimed": False, "source_parent_commit": parent,
        "script": {"repository_path": Path(__file__).relative_to(ROOT).as_posix(),
                   "sha256": sha256(__file__)}, "inputs": input_records(),
        "software": software_record(), "draws": draws, "alpha_family": 0.05,
        "minimum_valid_fraction": 0.95,
        "effects": {
            "pairs": [[model, baseline] for model in TOP10 for baseline in ("GRU32", "Persistence")],
            "gain": "reference RMSE minus candidate RMSE, per seed; raw gains, not gain-minus-delta",
            "effect": "paired Cohen dz = mean(raw six-seed gains)/sample_SD(raw gains,ddof=1)",
            "CI": "95 percent percentile bootstrap of six paired seed gains",
            "seed_resampling": True, "calendar_resampling": False,
            "random_seed_base": 202611150100, "inference_scope": "optimization conditional on fixed calendar",
            "zero_variance_policy": "undefined point effect/CI; invalid bootstrap draws counted and disclosed",
            "deterministic_reference": "Persistence has one fixed trajectory reused in six gains; only candidate optimization variability is sampled",
            "new_effect_p_values": False,
        },
        "RQ2": {
            "candidates": list(TOP10), "reference": "GRU32", "delta": DELTA,
            "regime": "sigma7(t-1)>q75_2023", "q75_2023": Q75,
            "first_lag_source": "development CSV sigma7 on2024-12-31; later lags are preceding dated y_true",
            "effect": "(GRU normal RMSE minus candidate normal RMSE)-(GRU high RMSE minus candidate high RMSE)",
            "metric_aggregation": "compute RMSE per trajectory inside each sampled regime, then average six RMSEs",
            "H0": "erosion<=delta", "H1": "erosion>delta",
            "bootstrap": "ordinary non-circular moving blocks from entire ordered608-date calendar; uniform starts0..N-L; concatenate and truncate toN",
            "retain_regime_membership_with_sampled_date": True,
            "fixed_seed_trajectories": True, "resample_optimization_seeds": False,
            "primary_block_days": 14, "diagnostic_block_days": [7, 28],
            "random_seed_base": 202611150200,
            "null_frontier": "Tnull_star=delta+(Tstar-Tobserved)",
            "p_greater": "(1+count(Tnull_star>=Tobserved))/(Bvalid+1)",
            "CI": "95 percent marginal percentile interval of valid bootstrap effects",
            "multiplicity": "Holm10 at0.05 for block14 only; sensitivity p-values unadjusted and nondecisory",
            "invalid_replicate": "absent normal or high stratum; no redraw, no zero imputation",
            "support_report": "dated regime labels, maximal runs, overlapping eligible block counts, valid/invalid replicas",
        },
        "invariants": {"original_A2_unchanged": True, "original_RQ1_unchanged": True,
                       "new_model_fits": 0, "new_model_inference": False,
                       "new_market_data": False, "RQ3_executed": False},
        "limitations": [
            "The analysis is exploratory after A2 on the already known E1 calendar, not restored prospective confirmation.",
            "One BTC series and fixed fitted trajectories; no market-population or HPO-selection uncertainty.",
            "Moving-block inference is approximate under suitable weak dependence/local stationarity;14 days is not universally adequate.",
            "Only six optimization pairs inform dz; its bootstrap can be unstable and undefined samples are disclosed.",
            "No demonstrated RQ2-specific power; non-rejection does not establish robustness or equivalence.",
            "A causal regime available at issue time is distinct from grouping by realized same-day target.",
        ],
    }
    write_json(output / "analysis_plan.json", value)
    return value


def causal_regime(targets, first_prior_sigma, *, threshold=Q75):
    targets = np.asarray(targets, dtype=np.float64)
    require(targets.ndim == 1 and len(targets) > 0 and np.isfinite(targets).all(), "Invalid regime targets")
    lag = np.r_[float(first_prior_sigma), targets[:-1]]
    require(np.isfinite(lag).all(), "Nonfinite causal regime inputs")
    return lag, lag > threshold


def load_data():
    payload = json.loads((INPUTS / "e2_f2a_input.json").read_text())
    dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
    require(payload["dates"] == dates and payload["mask"] == [True] * 608,
            "Requires the complete original ordered common calendar")
    y = np.asarray(payload["y_true"], dtype=np.float64)
    require(y.shape == (608,) and np.isfinite(y).all(), "Invalid targets")
    rows = {}
    for row in payload["predictors"]:
        key = (row["model"], row["seed"])
        require(key not in rows, "Duplicate model/seed")
        values = np.asarray(row["prediction_original"], dtype=np.float64)
        require(values.shape == y.shape and np.isfinite(values).all(), "Invalid prediction trajectory")
        require(len(row["checkpoint_sha256"]) == 64, "Checkpoint hash required")
        rows[key] = values
    expected = {(m, s) for m in NEURAL for s in SEEDS} | {(m, "deterministic") for m in CLASSICAL}
    require(set(rows) == expected, "The81-predictor frozen registry must be complete")
    prediction = {model: np.stack([rows[model, seed] for seed in SEEDS]) for model in NEURAL}
    prediction.update({model: rows[model, "deterministic"][None, :] for model in CLASSICAL})
    development = list(csv.DictReader((INPUTS / "daily_volatility_development.csv").open()))
    previous = [row for row in development if row["market_day_utc"][:10] == "2024-12-31"]
    require(len(previous) == 1, "First issue-time lag is missing or duplicated")
    lag, high = causal_regime(y, float(previous[0]["sigma_7"]))
    return y, prediction, dates, lag, high


def standardized_effect(gains, *, draws, seed, minimum_valid_fraction=0.95):
    gains = np.asarray(gains, dtype=np.float64)
    require(gains.shape == (6,) and np.isfinite(gains).all(), "Exactly six finite paired gains required")
    sd = float(gains.std(ddof=1))
    observed = float(gains.mean() / sd) if np.ptp(gains) > 0 and sd > 0 else None
    sampled = gains[np.random.default_rng(seed).integers(0, 6, size=(draws, 6))]
    deviations = sampled.std(axis=1, ddof=1)
    valid = (np.ptp(sampled, axis=1) > 0) & (deviations > 0)
    values = sampled[valid].mean(axis=1) / deviations[valid]
    valid_count = int(valid.sum())
    estimable = observed is not None and valid_count >= minimum_valid_fraction * draws
    ci = np.quantile(values, [0.025, 0.975]).tolist() if estimable else [None, None]
    return {"mean_gain_RMSE": float(gains.mean()), "SD_seed_gains": sd,
            "paired_Cohen_dz": observed, "dz_CI95_lower": ci[0], "dz_CI95_upper": ci[1],
            "status": "estimable_conditional_seed_effect" if estimable else "not_estimable",
            "bootstrap_draws": draws, "valid_draws": valid_count, "invalid_draws": draws-valid_count,
            "bootstrap_seed": seed, "n_seed_pairs": 6,
            "scope": "six optimization pairs conditional on the608-date calendar; no temporal uncertainty",
            "CI_method": "paired-seed percentile bootstrap; undefined SD-zero draws excluded and counted"}


def moving_block_counts(n, block_days, draws, seed, *, chunk_size=256):
    require(1 <= block_days <= n and draws > 0 and chunk_size > 0, "Invalid block/count dimensions")
    rng = np.random.default_rng(seed)
    k = math.ceil(n / block_days)
    for start in range(0, draws, chunk_size):
        count = min(chunk_size, draws-start)
        starts = rng.integers(0, n-block_days+1, size=(count, k))
        indices = (starts[:, :, None] + np.arange(block_days)).reshape(count, -1)[:, :n]
        weights = np.zeros((count, n), dtype=np.float64)
        np.add.at(weights, (np.arange(count)[:, None], indices), 1)
        yield weights


def regime_statistic(squared_errors, high, counts):
    """Return each candidate erosion vs last model, averaged after per-seed RMSE."""
    squared_errors = np.asarray(squared_errors, dtype=np.float64)
    high, counts = np.asarray(high, dtype=bool), np.asarray(counts, dtype=np.float64)
    require(squared_errors.ndim == 3 and squared_errors.shape[-1] == len(high), "Invalid loss geometry")
    require(counts.ndim == 2 and counts.shape[1] == len(high), "Invalid calendar weights")
    scores = []
    for membership in (~high, high):
        weights = counts * membership[None, :]
        denominator = weights.sum(axis=1)
        sums = np.einsum("bt,mst->bms", weights, squared_errors, optimize=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            scores.append(np.sqrt(sums / denominator[:, None, None]).mean(axis=2))
    normal, elevated = scores
    gain_normal = normal[:, -1, None] - normal[:, :-1]
    gain_high = elevated[:, -1, None] - elevated[:, :-1]
    return gain_normal-gain_high, gain_normal, gain_high


def one_sided_summary(observed, samples, *, delta, minimum_valid_fraction):
    samples = np.asarray(samples, dtype=np.float64)
    valid = np.isfinite(samples)
    values = samples[valid]
    estimable = np.isfinite(observed) and len(values) >= minimum_valid_fraction * len(samples)
    if estimable:
        null_samples = delta + (values-observed)
        exceedances = int(np.count_nonzero(null_samples >= observed))
        p = float((1+exceedances) / (len(values)+1))
        ci = np.quantile(values, [0.025, 0.975]).tolist()
    else:
        exceedances, p, ci = None, None, [None, None]
    return {"erosion_RMSE": float(observed) if np.isfinite(observed) else None,
            "CI95_lower": ci[0], "CI95_upper": ci[1], "p_raw": p,
            "valid_draws": int(valid.sum()), "invalid_draws": int((~valid).sum()),
            "null_exceedances": exceedances,
            "status": "estimable" if estimable else "not_estimable"}


def holm10(rows):
    require(len(rows) == 10, "RQ2 has exactly ten candidates")
    ordered = sorted((1.0 if row["p_raw"] is None else row["p_raw"], i)
                     for i, row in enumerate(rows))
    previous = 0.0
    for rank, (pvalue, i) in enumerate(ordered):
        previous = max(previous, min(1.0, (10-rank)*pvalue))
        rows[i]["p_Holm10"] = previous if rows[i]["p_raw"] is not None else None
        rows[i]["reject_FWER05"] = rows[i]["p_raw"] is not None and previous <= 0.05
    return rows


def support_record(dates, high, lag, blocks):
    runs = []
    start = 0
    for stop in range(1, len(high)+1):
        if stop == len(high) or high[stop] != high[start]:
            runs.append({"regime": "high" if high[start] else "normal", "start": dates[start],
                         "end": dates[stop-1], "days": stop-start})
            start = stop
    block_support = {}
    for block in blocks:
        members = np.convolve(high.astype(int), np.ones(block, dtype=int), mode="valid")
        block_support[str(block)] = {
            "all_overlapping_blocks": len(members), "blocks_with_high_dates": int((members > 0).sum()),
            "blocks_with_normal_dates": int((members < block).sum()),
            "blocks_with_both_regimes": int(((members > 0) & (members < block)).sum()),
            "counts_are_not_independent_block_replicates": True}
    return {"calendar_dates": len(dates), "high_dates": int(high.sum()),
            "normal_dates": int((~high).sum()), "threshold": Q75,
            "first_prior_sigma7": float(lag[0]), "maximal_calendar_runs": runs,
            "overlapping_candidate_block_support": block_support}


def execute(output):
    plan_path = output / "analysis_plan.json"
    require(plan_path.is_file(), "Register a plan first with --register-plan")
    plan = json.loads(plan_path.read_text())
    require(plan["version"] == VERSION and plan["script"]["sha256"] == sha256(__file__),
            "Program differs from registered bytes; preserve plan and register a new output")
    require(plan["inputs"] == input_records(), "Registered input bytes changed")
    require(not (output / "execution_receipt.json").exists(), "Execution exists; never overwrite results")
    started = datetime.now(timezone.utc).isoformat()
    y, prediction, dates, lag, high = load_data()
    draws, minimum = plan["draws"], plan["minimum_valid_fraction"]
    seed_metrics = {name: np.sqrt(((values-y[None, :])**2).mean(axis=1))
                    for name, values in prediction.items()}
    effects, seed_rows = [], []
    for i, (candidate, reference) in enumerate(plan["effects"]["pairs"]):
        baseline = seed_metrics[reference]
        gains = baseline-seed_metrics[candidate]
        require(gains.shape == (6,), "Deterministic baseline must broadcast, not create fitted replicas")
        effects.append({"candidate": candidate, "reference": reference, "delta_RMSE": DELTA,
                        **standardized_effect(gains, draws=draws,
                            seed=plan["effects"]["random_seed_base"]+i, minimum_valid_fraction=minimum)})
        seed_rows.extend({"candidate": candidate, "reference": reference, "seed": seed,
                          "candidate_RMSE": float(seed_metrics[candidate][j]),
                          "reference_RMSE": float(baseline[0 if len(baseline) == 1 else j]),
                          "raw_gain_RMSE": float(gains[j]),
                          "reference_is_single_deterministic_trajectory": len(baseline) == 1}
                         for j, seed in enumerate(SEEDS))
    write_csv(output / "standardized_effects_20.csv", effects)
    write_csv(output / "paired_seed_gains_120.csv", seed_rows)
    write_csv(output / "causal_regime_labels_608.csv", [
        {"date": day, "lag_date": (date.fromisoformat(day)-timedelta(days=1)).isoformat(),
         "sigma7_lag1": float(lag[i]), "q75_2023": Q75, "high_at_issue_time": bool(high[i])}
        for i, day in enumerate(dates)])
    squared = np.stack([(prediction[name]-y[None, :])**2 for name in (*TOP10, "GRU32")])
    point, normal, elevated = regime_statistic(squared, high, np.ones((1, len(y))))
    blocks = [plan["RQ2"]["primary_block_days"], *plan["RQ2"]["diagnostic_block_days"]]
    all_rows = []
    for block in blocks:
        samples = np.concatenate([regime_statistic(squared, high, weights)[0]
                                  for weights in moving_block_counts(len(y), block, draws,
                                      plan["RQ2"]["random_seed_base"]+block)])
        rows = []
        for i, candidate in enumerate(TOP10):
            rows.append({"candidate": candidate, "reference": "GRU32", "delta_RMSE": DELTA,
                         "gain_normal_RMSE": float(normal[0, i]), "gain_high_RMSE": float(elevated[0, i]),
                         **one_sided_summary(point[0, i], samples[:, i], delta=DELTA,
                                             minimum_valid_fraction=minimum),
                         "block_days": block, "bootstrap_draws": draws,
                         "bootstrap_seed": plan["RQ2"]["random_seed_base"]+block,
                         "primary_block": block == 14, "seed_resampling": False,
                         "scope": "exploratory calendar inference conditional on six frozen trajectories"})
        if block == 14:
            holm10(rows)
            write_csv(output / "rq2_primary_Holm10.csv", rows)
        all_rows.extend(rows)
    write_csv(output / "rq2_block_sensitivity.csv", all_rows)
    support = support_record(dates, high, lag, blocks)
    write_json(output / "rq2_support.json", support)
    primary = [row for row in all_rows if row["primary_block"]]
    summary = {"version": VERSION, "effect_pairs": len(effects),
               "effects_with_estimable_CI": sum(row["dz_CI95_lower"] is not None for row in effects),
               "RQ2_primary_pairs": len(primary), "RQ2_estimable_pairs": sum(row["status"] == "estimable" for row in primary),
               "RQ2_rejections_Holm10": sum(row["reject_FWER05"] for row in primary),
               "RQ2_rejected_candidates": [row["candidate"] for row in primary if row["reject_FWER05"]],
               "high_dates": support["high_dates"], "normal_dates": support["normal_dates"],
               "bootstrap_draws": draws, "primary_block_days": 14,
               "sensitivity_blocks": [7, 28], "A2_replaced": False, "RQ1_replaced": False,
               "new_model_fits": 0, "new_model_inference": False,
               "interpretation": "RQ2 rejection supports erosion greater than delta; non-rejection does not establish robustness or equivalence.",
               "limitations": plan["limitations"]}
    write_json(output / "summary.json", summary)
    final_inputs = input_records()
    require(final_inputs == plan["inputs"], "Input bytes changed during execution")
    write_json(output / "execution_receipt.json", {
        "version": VERSION, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha256(plan_path), "script_sha256": sha256(__file__),
        "inputs_reverified_after_execution": final_inputs, "software": software_record(),
        "new_model_fits": 0, "new_model_inference": False, "original_A2_unchanged": True})
    readme = """# Efectos estandarizados y contraste de erosión por régimen

El plan se guardó antes de este cálculo, después de conocer A2. Es una ampliación
exploratoria; conserva los pronósticos, A2 y las veinte decisiones RQ1 originales.

`standardized_effects_20.csv` aporta d de Cohen pareado e IC percentiles entre
las seis ganancias brutas por semilla. Su incertidumbre es de optimización,
condicionada al calendario; no es un efecto de 608 fechas independientes.
Persistence conserva una trayectoria fija: no se crean seis ajustes clásicos.
`paired_seed_gains_120.csv` conserva los valores que permiten auditarlo.

`rq2_primary_Holm10.csv` contrasta erosión mayor que δ frente a GRU32. Calcula
RMSE por trayectoria dentro de cada régimen, después promedia las seis RMSE.
Se remuestrea todo el calendario en bloques móviles no circulares, conservando
las etiquetas causales de régimen y los predictores fijos. La decisión corresponde
sólo al bloque 14 y Holm10; 7/28 son sensibilidad descriptiva, sin elegir el p menor.
`causal_regime_labels_608.csv` y `rq2_support.json` documentan disponibilidad,
soporte y bloques solapados; esos bloques no son observaciones independientes.

Los IC son marginales. La aproximación requiere dependencia suficientemente
corta y una longitud de bloque adecuada; no se afirma calibración exacta ante
cualquier cambio de régimen. No hay potencia RQ2 demostrada. No rechazo no
significa robustez, equivalencia ni ausencia de erosión. Los campos no estimables
se conservan; no se sustituyen por cero ni se generan sorteos nuevos para ocultarlos.

Reproducción: desde el repositorio, con Python y NumPy instalados, registre en
una carpeta NUEVA y ejecute sin modificar los bytes registrados:

```sh
python scripts/e2_effects_rq2_closure.py --output /tmp/e2-effects-rq2-repeat --register-plan
python scripts/e2_effects_rq2_closure.py --output /tmp/e2-effects-rq2-repeat --execute
```

Plan, recibo, hashes de entradas/programa y versiones permiten auditar la ejecución.
Los tiempos de registro cambian al repetir; las semillas y resultados son reproducibles.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
             for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", {"version": VERSION, "files": files,
                                         "scope": "posthoc_saved-prediction_statistical_supplement_no_fits"})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=20000)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register-plan", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = register_plan(args.output, draws=args.draws) if args.register_plan else execute(args.output)
    print(json.dumps(result if args.execute else {"registered": str(args.output / "analysis_plan.json"),
                                                "draws": result["draws"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
