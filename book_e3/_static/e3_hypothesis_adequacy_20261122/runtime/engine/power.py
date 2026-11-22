"""Development-only empirical planning; no model fitting or final-label access.

This module distinguishes a conditional empirical-DGP power calculation from
validation of a temporal test and from an unconditional power guarantee.  The
outer simulation resamples *crossed* fitted-seed curves and calendar blocks
shared only within coincident-date groups; R*T cells are never iid.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import inspect
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import beta


VERSION = "development-grouped-calendar-empirical-power-v4"
OPERATIONAL_RULE = "finite-B-count-plus-one-actual-Holm-v1"
METHODS = ("dm_hln_hac", "stationary_bootstrap", "gw_conditional")
SYSTEM_IDS = ("input_only_mse", "FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
              "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer", "MLP16", "LSTM32", "GRU32",
              "Persistence", "SeasonalNaive", "ARIMA", "additive_positive_head", "floor_only", "floor_plus_mean_excess", "global_mse")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)


def resampling_budget(*, rivals: int = 20, panels: int = 4, alpha: float = .05,
                      expected_tail_exceedances: int = 100,
                      mc_p_rule: str = "absolute_tail_plus_one") -> dict[str, Any]:
    """Resolution and tail sampling requirements, explicitly not power.

    Absolute-tail two-sided +1 MC p has minimum 1/(B+1). The legacy
    doubled smaller-tail rule instead has minimum 2/(B+1). The additional
    adverse screen has 3*20*3 bilateral EPA slots and sets the larger B.
    """
    if rivals < 1 or panels < 1 or not 0 < alpha < 1 or expected_tail_exceedances < 1:
        raise ValueError("invalid multiplicity/tail design")
    if mc_p_rule not in ("absolute_tail_plus_one", "doubled_smaller_tail_plus_one"):
        raise ValueError("unrecognized Monte Carlo tail rule")
    positive = 6 * rivals
    adverse = 3 * rivals * max(0, panels - 1)
    family = max(positive, adverse)
    threshold = alpha / family
    numerator = 1 if mc_p_rule == "absolute_tail_plus_one" else 2
    return {"positive_slots_per_panel": positive, "additional_adverse_slots": adverse,
            "smallest_family_threshold": threshold,
            "mc_p_rule": mc_p_rule, "p_resolution_numerator": numerator,
            "minimum_B_for_two_sided_resolution": math.ceil(numerator / threshold) - 1,
            "expected_tail_exceedances": expected_tail_exceedances,
            "B_for_requested_expected_tail_exceedances_diagnostic_only": math.ceil(numerator * expected_tail_exceedances / threshold),
            "interpretation": "resolution is necessary arithmetic; expected exceedances are a precision diagnostic, not a universal B gate or a power certificate"}


def seed_prefix(design: Mapping[str, Any], count: int) -> list[int]:
    path = Path(design["seed_bank"]["path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if sha256(path) != design["seed_bank"]["sha256"]:
        raise ValueError("frozen seed bank checksum mismatch")
    bank = json.loads(path.read_text())
    ids = bank["seed_ids"]
    capacity = min(design["seed_bank"].get("planning_indices", [len(ids)]))
    if count < 1 or count > capacity or len(ids) != len(set(ids)):
        raise ValueError("seed count outside unique frozen bank capacity")
    return ids[:count]


def planning_seed_ids(design: Mapping[str, Any], count: int) -> list[int]:
    """Preallocated planning-only tail, disjoint from the final bank prefix."""
    # seed_prefix verifies the archived bank before the tail is consulted.
    seed_prefix(design, 1)
    path = Path(design["seed_bank"]["path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    bank = json.loads(path.read_text())["seed_ids"]
    indices = design["seed_bank"].get("planning_indices", [])
    if count < 1 or count > len(indices) or any(i < 0 or i >= len(bank) for i in indices):
        raise ValueError("planning seed count exceeds preallocated bank tail")
    return [bank[i] for i in indices[:count]]


def budget(design: Mapping[str, Any], *, R: int, T: int,
           fit_receipts: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Exact fit counts and labeled arithmetic estimates, never measured cost.

    System budgets include all auxiliary stages, notably SAE pretraining and
    prediction. Measured consumption includes failed/retried attempts in the
    receipts and is not inferred from a historical single candidate fit.
    """
    if R < 1 or T < 1:
        raise ValueError("positive R/T required")
    panels = len(design["panels"])
    neural = sum(item["kind"] == "neural" for item in design["systems"])
    arima = sum(item["kind"] == "arima" for item in design["systems"])
    analytic = sum(item["kind"] == "analytic" for item in design["systems"])
    fits = panels * neural * R
    nominal_seconds = fits * float(design["budget"]["nominal_total_neural_seconds_per_system_seed"])
    rows = panels * (neural * R + arima + analytic) * T
    byte_assumption = design["budget"].get("checkpoint_bytes_per_neural_fit")
    storage = None if byte_assumption is None else int(fits * byte_assumption + rows * design["budget"]["prediction_csv_bytes_per_row"])
    timed = [r for r in fit_receipts if isinstance(r.get("active_seconds"), (int, float))]
    uncertain = [r for r in timed if r.get("compute_uncertain") or r.get("active_seconds_is_charge_not_exact_elapsed")]
    exact = [r for r in timed if not r.get("compute_uncertain") and not r.get("active_seconds_is_charge_not_exact_elapsed")]
    measured = sum(float(r["active_seconds"]) for r in exact) if exact else None
    rate = design["budget"].get("compute_price_per_hour")
    limits = design["budget"]["limits"]
    failures = []
    if fits > limits["max_neural_fits"]:
        failures.append("neural_fit_limit")
    if nominal_seconds / 3600 > limits["max_nominal_neural_hours"]:
        failures.append("nominal_compute_limit")
    if storage is not None and storage > limits["max_storage_bytes"]:
        failures.append("storage_limit")
    return {"R": R, "T": T, "panels": panels, "neural_systems": neural,
            "neural_fits": fits, "arima_fits": panels * arima,
            "analytic_trajectories": panels * analytic,
            "deterministic_independent_seed_replicates": 0,
            "nominal_neural_active_seconds": nominal_seconds,
            "nominal_neural_active_hours": nominal_seconds / 3600,
            "nominal_compute_money": None if rate is None else nominal_seconds / 3600 * rate,
            "prediction_rows": rows, "estimated_storage_bytes": storage,
            "storage_assumption_bytes_per_neural_fit": byte_assumption,
            "measured_planning_active_seconds": measured,
            "measured_planning_timed_attempts": len(exact),
            "planning_active_seconds_charged_total": sum(float(r["active_seconds"]) for r in timed) if timed else None,
            "planning_cost_uncertain_attempts": len(uncertain),
            "planning_observed_interruption_partial_seconds": sum(float(r.get("aborted_seconds_observed", 0.)) for r in fit_receipts),
            "planning_receipt_attempts": len(fit_receipts),
            "measured_time_is_total_for_observed_attempts_only": True,
            "within_declared_known_limits": not failures,
            "limit_failures": failures,
            "unpriced_or_excluded": ["ARIMA active time", "data acquisition and licensing", "inference/export", "power Monte Carlo CPU time", "GitHub transfers/storage", "Colab waiting and interruptions", "unobserved retries/recovery", "additional HPO outside frozen system budget"],
            "caveat": "nominal ceilings and storage assumptions are not measured consumption; SAE phases share each system-seed total budget"}


def _verify_file(item: Mapping[str, Any], base: Path) -> dict[str, str]:
    path = Path(item["path"])
    if not path.is_absolute():
        path = base / path
    if not path.is_file() or sha256(path) != item["sha256"]:
        raise ValueError(f"provenance file missing or checksum mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": item["sha256"]}


def _paths(planning_paths: Mapping[str, Any] | Sequence[str | Path]) -> tuple[list[Path], list[Path]]:
    if isinstance(planning_paths, Mapping):
        return ([Path(x) for x in planning_paths.get("predictions", [])],
                [Path(x) for x in planning_paths.get("receipts", [])])
    paths = [Path(planning_paths)] if isinstance(planning_paths, (str, Path)) else [Path(x) for x in planning_paths]
    return paths, [p.with_suffix(".receipt.json") for p in paths]


def load_planning(planning_paths: Mapping[str, Any] | Sequence[str | Path],
                  design: Mapping[str, Any], *, allow_synthetic_fixture: bool = False) -> dict[str, Any]:
    """Validate complete, hash-bound, development-only prediction/fit receipts.

    Receipt declarations do not prove a custodian's honesty; actual files,
    source/protocol/method identities and every completed checkpoint are
    checked. Fixture/synthetic data cannot qualify for final dimensioning.
    Deterministic predictions are stored once with seed='deterministic'.
    """
    csvs, receipts = _paths(planning_paths)
    if not csvs or len(csvs) != len(receipts):
        raise ValueError("supply matching nonempty predictions and receipts paths")
    names = [x["id"] for x in design["systems"]]
    neural = [x["id"] for x in design["systems"] if x["kind"] == "neural"]
    deterministic = [x for x in names if x not in neural]
    panels = list(design["panels"])
    if set(names) != set(SYSTEM_IDS) or len(names) != 21 or len(neural) != 16 or len(deterministic) != 5 or len(panels) != 4:
        raise ValueError("dimensioning requires the complete 21-system, 4-panel identity (16 neural, 5 deterministic)")
    if names[0] != design["candidate"] or "global_mse" not in names:
        raise ValueError("candidate first and global_mse primary rival required")
    records: dict[tuple[str, str, str, str], tuple[float, float, float, float]] = {}
    attempts, verified, receipt_hashes = [], [], []
    successful: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for csv_path, receipt_path in zip(csvs, receipts):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("kind") != "development_planning_predictions" or receipt.get("partition") != "planning":
            raise ValueError("only dedicated development planning partition is allowed")
        fixture_only = allow_synthetic_fixture and receipt.get("synthetic") is True and receipt.get("is_fixture") is True
        if receipt.get("final_test_opened") is not False or (not fixture_only and (receipt.get("synthetic") is not False or receipt.get("is_fixture", False))):
            raise ValueError("final/exposed-test, synthetic or fixture data cannot dimension a final protocol")
        if receipt.get("predictions_sha256") != sha256(csv_path):
            raise ValueError("planning predictions checksum mismatch")
        if receipt.get("seed_bank_sha256") != design["seed_bank"]["sha256"]:
            raise ValueError("planning seed bank differs from frozen bank")
        if not receipt.get("protocol_sha256") or not receipt.get("methods_sha256"):
            raise ValueError("protocol and methods identities required")
        for key in ("protocol_file", "methods_file", "source_manifest_file", "planning_data_file"):
            if key not in receipt:
                raise ValueError(f"receipt missing {key}")
            verified.append(_verify_file(receipt[key], receipt_path.parent))
        if receipt["protocol_file"]["sha256"] != receipt["protocol_sha256"] or receipt["methods_file"]["sha256"] != receipt["methods_sha256"]:
            raise ValueError("receipt identity hashes inconsistent")
        if not fixture_only and receipt["methods_sha256"] != sha256(Path(__file__).with_name("statistics.py")):
            raise ValueError("planning receipt methods differ from the executed temporal implementation")
        if receipt.get("planning_partition_purged") is not True or receipt.get("instruments_available_at_origin") is not True:
            raise ValueError("purged planning split and causal instruments must be evidenced")
        for job in receipt.get("fit_receipts", []):
            if job.get("phase") != "planning" or not isinstance(job.get("active_seconds"), (int, float)) or job["active_seconds"] < 0:
                raise ValueError("every attempt needs planning phase and actual nonnegative active_seconds")
            attempts.append(dict(job))
            if job.get("success") is True:
                if "checkpoint" not in job:
                    raise ValueError("successful planning jobs need actual hash-bound checkpoint/state")
                verified.append(_verify_file(job["checkpoint"], receipt_path.parent))
                key = (str(job["panel"]), str(job["system"]), str(job["seed"]))
                if key in successful:
                    raise ValueError("duplicate successful planning fit; retain attempts without duplicating completed jobs")
                successful[key] = job
        receipt_hashes.append({"path": str(receipt_path), "sha256": sha256(receipt_path)})
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"panel", "system", "seed", "date", "y", "prediction", "known_floor_A", "rfit"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"planning CSV requires columns {sorted(required)}")
            for row in reader:
                try:
                    dt.date.fromisoformat(row["date"])
                except ValueError as exc:
                    raise ValueError("planning endpoint date must use canonical ISO YYYY-MM-DD") from exc
                key = (row["panel"], row["system"], row["seed"], row["date"])
                if key in records:
                    raise ValueError("duplicate planning prediction key")
                if row["panel"] not in panels or row["system"] not in names:
                    raise ValueError("unknown planning panel/system")
                values = tuple(float(row[c]) for c in ("y", "prediction", "known_floor_A", "rfit"))
                if not np.isfinite(values).all() or values[2] < 0 or values[3] <= 0:
                    raise ValueError("nonfinite predictions or invalid causal floor/fit scale")
                if values[0] + 1e-10 < math.sqrt(values[2]) or values[1] + 1e-10 < math.sqrt(values[2]):
                    raise ValueError("label/prediction violates frozen known-floor support")
                records[key] = values
    # Genuine complete curves are required for every system, not just candidate.
    seed_sets = []
    date_sets: dict[str, set[str]] = {}
    for panel in panels:
        panel_dates = None
        for system in names:
            seeds = {key[2] for key in records if key[:2] == (panel, system)}
            if system in neural:
                if len(seeds) < design["minimum_planning_seeds"]:
                    raise ValueError(f"at least {design['minimum_planning_seeds']} genuine neural planning seeds required: {panel}/{system}")
                seed_sets.append(seeds)
            elif seeds != {"deterministic"}:
                raise ValueError(f"deterministic system must be stored once: {panel}/{system}")
            for seed in seeds:
                if (panel, system, seed) not in successful:
                    raise ValueError("predictions lack verified successful planning job")
                dates = {k[3] for k in records if k[:3] == (panel, system, seed)}
                if panel_dates is None:
                    panel_dates = dates
                elif panel_dates != dates:
                    raise ValueError("planning endpoints differ across methods/seeds within panel")
        date_sets[panel] = panel_dates or set()
    if not seed_sets or any(s != seed_sets[0] for s in seed_sets):
        raise ValueError("all neural methods/panels need the same paired seed bank prefix")
    pilot_seeds = [str(s) for s in planning_seed_ids(design, len(seed_sets[0]))]
    if set(pilot_seeds) != seed_sets[0]:
        raise ValueError("planning neural seed IDs must be frozen preallocated common bank tail, never ranked/replaced")
    # Calendar overlap is not a requirement of R3. Pair blocks only where the
    # *actual dates* coincide; never create shared shocks by relative position.
    minimum_dates = design.get("minimum_within_panel_planning_dates", design.get("minimum_common_planning_dates", 80))
    if any(len(v) < minimum_dates for v in date_sets.values()):
        raise ValueError("insufficient chronological planning dates within a panel")
    groups: list[dict[str, Any]] = []
    for pi, panel in enumerate(panels):
        for group in groups:
            intersection = group["date_set"] & date_sets[panel]
            if len(intersection) >= minimum_dates:
                group["indices"].append(pi)
                group["date_set"] = intersection
                break
        else:
            groups.append({"indices": [pi], "date_set": set(date_sets[panel])})
    # Rectangular storage uses the shortest group prefix, chosen by dates
    # before reading any losses. It carries no across-group simultaneity claim.
    n_dates = min(len(g["date_set"]) for g in groups)
    panel_dates: dict[str, list[str]] = {}
    group_evidence = []
    for group in groups:
        dates = sorted(group["date_set"])[:n_dates]
        for pi in group["indices"]:
            panel_dates[panels[pi]] = dates
        group_evidence.append({"panel_indices": group["indices"], "panels": [panels[i] for i in group["indices"]],
                               "actual_dates": dates, "available_coincident_dates": len(group["date_set"])})
    losses = np.empty((len(panels), len(names), len(pilot_seeds), n_dates))
    targets = np.empty((len(panels), n_dates))
    instruments = np.empty((len(panels), n_dates, 2))
    for pi, panel in enumerate(panels):
        panel_rfit = None
        for ti, date in enumerate(panel_dates[panel]):
            reference = None
            for si, system in enumerate(names):
                for ri, seed in enumerate(pilot_seeds):
                    record = records[(panel, system, seed if system in neural else "deterministic", date)]
                    target, prediction, floor, rfit = record
                    if reference is None:
                        reference = (target, floor, rfit)
                    elif not np.allclose(reference, (target, floor, rfit), rtol=0, atol=1e-10):
                        raise ValueError("label, known floor or fit scale disagrees across paired predictions")
                    losses[pi, si, ri, ti] = (prediction - target) ** 2
            targets[pi, ti] = reference[0]
            if panel_rfit is None:
                panel_rfit = reference[2]
            elif not math.isclose(panel_rfit, reference[2], rel_tol=0., abs_tol=1e-10):
                raise ValueError("rfit must be one adjustment-only scale per panel")
            instruments[pi, ti] = (1., math.sqrt(reference[1]) / reference[2])
        if np.linalg.matrix_rank(instruments[pi]) < 2:
            raise ValueError("GW instrument has no development variation")
    if np.any(losses.mean(axis=(2, 3)) <= 0):
        raise ValueError("zero empirical risk cannot define relative practical alternatives")
    return {"losses": losses, "y": targets, "instruments": instruments,
            "systems": names, "panels": panels, "planning_seed_ids": pilot_seeds,
            "dates": list(range(n_dates)), "panel_dates": panel_dates,
            "calendar_groups": [g["indices"] for g in groups], "fit_receipts": attempts,
            "joint_calendar_coupling": "shared actual-date blocks within groups; independent calendar draws between groups, conditional empirical-DGP assumption; paired training seeds shared",
            "evidence": {"receipts": receipt_hashes, "verified_files": verified,
                         "csvs": [{"path": str(p), "sha256": sha256(p)} for p in csvs],
                         "relative_array_length": n_dates, "relative_array_positions_are_not_shared_calendar_shocks": True,
                         "calendar_groups": group_evidence,
                         "panel_dates": panel_dates,
                         "omitted_dates_by_panel": {p: len(v)-n_dates for p, v in date_sets.items()},
                         "joint_calendar_coupling_assumed_between_groups": "independent draws, not empirically identified cross-calendar covariance",
                         "panel_dates_before_joint_alignment": {p: len(v) for p, v in date_sets.items()},
                         "full_loss_shape": list(losses.shape)}}


def dependence_summary(planning: Mapping[str, Any]) -> dict[str, Any]:
    """Joint empirical covariance diagnostics, not a stationarity certificate."""
    losses = planning["losses"]
    d = losses[:, 1:] - losses[:, :1]
    panel_rival = d.shape[0] * d.shape[1]
    time_mean = d.mean(axis=2).reshape(panel_rival, d.shape[-1])
    seed_mean = d.mean(axis=3).reshape(panel_rival, d.shape[2])
    def cov(a: np.ndarray) -> list[list[float]]:
        return np.atleast_2d(np.cov(a, ddof=1)).tolist()
    calendar_cov = cov(time_mean)
    groups = planning.get("calendar_groups", [list(range(d.shape[0]))])
    memberships = {pi: gi for gi, group in enumerate(groups) for pi in group}
    for i in range(panel_rival):
        for j in range(panel_rival):
            if memberships[i // d.shape[1]] != memberships[j // d.shape[1]]:
                calendar_cov[i][j] = None
    return {"contrast_order": [f"{p}/{s}" for p in planning["panels"] for s in planning["systems"][1:]],
            "calendar_covariance_of_seed_mean_loss_differences": calendar_cov,
            "calendar_covariance_null_means_not_identified_across_disjoint_groups": True,
            "calendar_groups": groups,
            "training_seed_covariance_of_calendar_mean_loss_differences": cov(seed_mean),
            "calendar_lag1_correlation": [float(np.corrcoef(x[:-1], x[1:])[0, 1]) if np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else None for x in time_mean],
            "calendar_first_second_half_means": [[float(x[:len(x)//2].mean()), float(x[len(x)//2:].mean())] for x in time_mean],
            "seed_variance_identifiable_only_from_observed_fitted_curves": True,
            "stationarity_verified": False,
            "training_calendar_independence_assumed_for_crossed_resampling": True}


def alternative_losses(losses: np.ndarray, relative_rmse_gain: float) -> tuple[np.ndarray, list[list[float]]]:
    """Nonnegative joint-loss DGP with specified aggregate-RMSE gain.

    Keep candidate losses; scale each rival's *entire* loss surface by one
    fixed development-only factor. This preserves empirical dependence
    patterns up to that deterministic scale. It does not fit an alternative
    forecasting model or establish a market alternative as true.
    """
    if not 0 < relative_rmse_gain < 1:
        raise ValueError("alternative gain must be in (0,1)")
    means = losses.mean(axis=(2, 3))
    if np.any(means <= 0):
        raise ValueError("positive empirical risks required")
    factors = means[:, :1] / ((1 - relative_rmse_gain) ** 2 * means)
    factors[:, 0] = 1.
    return losses * factors[:, :, None, None], factors.tolist()


def _calendar_indices(length: int, target: int, block_length: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(target, dtype=np.int32)
    out[0] = rng.integers(length)
    for t in range(1, target):
        out[t] = rng.integers(length) if rng.random() < 1 / block_length else (out[t-1] + 1) % length
    return out


def crossed_draw(planning: Mapping[str, Any], losses: np.ndarray, *, R: int, T: int,
                 block_length: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared seed-vector sample and calendar blocks within real-date groups.

    Deterministic systems have equal seed traces and therefore remain one
    trajectory when expanded to the analyzer's rectangular tensor. Nonnegative
    losses map to y+sqrt(loss), satisfying the known floor when y does.
    """
    seed_idx = rng.integers(losses.shape[2], size=R)
    sampled = np.empty((losses.shape[0], losses.shape[1], R, T))
    target = np.empty((losses.shape[0], T))
    z = np.empty((losses.shape[0], T, planning["instruments"].shape[-1]))
    for group in planning.get("calendar_groups", [list(range(losses.shape[0]))]):
        calendar = _calendar_indices(losses.shape[3], T, block_length, rng)
        for pi in group:
            sampled[pi] = np.take(np.take(losses[pi], seed_idx, axis=1), calendar, axis=2)
            target[pi] = planning["y"][pi, calendar]
            z[pi] = planning["instruments"][pi, calendar]
    return target[:, None, None, :] + np.sqrt(sampled), target, z


def _holm(values: Sequence[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted.tolist()


def statistical_rule(results: Sequence[Mapping[str, Any]], margin: float = .02) -> dict[str, Any]:
    """Actual statistical conjunction used inside *every* outer replication.

    Every contrast must pass all 3 distinct methods, EPA and practical tests,
    family adjustment, effect>=.2, and positive difference CI. We additionally
    require the gain interval lower bound above the declared practical margin.
    Historical mean individual RMSE must also be favorable: pooled MSE
    superiority alone does not establish that different estimand's ordering.
    The adverse additional-panel family is re-adjusted jointly, not panelwise.
    External data adequacy and robustness gates are not simulated or certified.
    """
    if not results:
        raise ValueError("panel analyses required")
    rivals = [c["rival"] for c in results[0]["comparisons"]]
    alpha = float(results[0].get("family", {}).get("alpha", .05))
    if not 0 < alpha <= .05 or any(float(r.get("family", {}).get("alpha", .05)) != alpha for r in results):
        raise ValueError("panels require the same prespecified family alpha <=.05")
    passing = np.zeros((len(results), len(rivals)), dtype=bool)
    historical_favorable = np.zeros_like(passing)
    adverse_tests, adverse_directions, adverse_methods = [], [], []
    for pi, result in enumerate(results):
        if [c["rival"] for c in result["comparisons"]] != rivals:
            raise ValueError("complete common ordered family required in every panel")
        metrics = result.get("metrics", {})
        candidate_metric = metrics.get(result.get("candidate"), {}).get("mean_seed_rmse")
        for ri, c in enumerate(result["comparisons"]):
            rival_metric = metrics.get(c["rival"], {}).get("mean_seed_rmse")
            historical_favorable[pi, ri] = bool(
                candidate_metric is not None and rival_metric is not None
                and math.isfinite(candidate_metric) and math.isfinite(rival_metric)
                and candidate_metric < rival_metric)
            test_ok = all(c["methods"][m][null]["status"] == "ok" and c["methods"][m][null]["adjusted_p"] < alpha
                          for m in METHODS for null in ("epa", "practical"))
            ci_gain, ci_diff = c.get("ci95_relative_rmse_gain"), c.get("ci95_mse_difference")
            ci_conservative_gain = c.get("ci99_relative_rmse_gain", ci_gain)
            ci_conservative_diff = c.get("ci99_mse_difference", ci_diff)
            passing[pi, ri] = bool(historical_favorable[pi, ri] and test_ok and c["mean_mse_difference"] > 0 and c["practical_mean"] > 0
                                   and c.get("temporal_effect_cohen_d") is not None and c["temporal_effect_cohen_d"] >= .2
                                   and ci_diff is not None and ci_diff[0] > 0
                                   and ci_gain is not None and ci_gain[0] > margin
                                   and ci_conservative_diff is not None and ci_conservative_diff[0] > 0
                                   and ci_conservative_gain is not None and ci_conservative_gain[0] > margin)
            if pi:
                for method in METHODS:
                    test = c["methods"][method]["epa"]
                    adverse_tests.append(test.get("mc_p_upper", test["raw_p"]) if test["status"] == "ok" else 1.)
                    adverse_directions.append(c["mean_mse_difference"] < 0)
                    adverse_methods.append(method)
    adverse_adjusted = _holm(adverse_tests) if adverse_tests else []
    adverse = any(negative and p < alpha and method in ("dm_hln_hac", "stationary_bootstrap")
                  for p, negative, method in zip(adverse_adjusted, adverse_directions, adverse_methods))
    gw_conditional_adverse = any(negative and p < alpha and method == "gw_conditional"
                                for p, negative, method in zip(adverse_adjusted, adverse_directions, adverse_methods))
    additional = np.all(passing[1:].sum(axis=0) >= min(2, len(results)-1)) if len(results) > 1 else True
    historical_main = bool(historical_favorable[0].all())
    historical_additional = bool(np.all(historical_favorable[1:].sum(axis=0) >= min(2, len(results)-1))) if len(results) > 1 else True
    return {"contrast_pass": passing,
            "family_alpha": alpha,
            "historical_mean_seed_rmse_favorable": historical_favorable,
            "historical_main_all_mean_rmse_favorable": historical_main,
            "historical_additional_mean_rmse_two_of_three": historical_additional,
            "additional_statistical_two_of_three": bool(additional),
            "all_contrasts_pass": bool(passing.all()),
            "joint_statistical_rule_pass": bool(passing[0].all() and additional and historical_main and historical_additional and not adverse),
            "additional_significant_inferiority": bool(adverse),
            "additional_gw_conditional_rejection_with_negative_sample_mean": bool(gw_conditional_adverse)}


def binomial_interval(successes: int, trials: int, alpha: float) -> list[float]:
    if not 0 <= successes <= trials or trials < 1 or not 0 < alpha < 1:
        raise ValueError("invalid binomial interval inputs")
    return [0. if successes == 0 else float(beta.ppf(alpha / 2, successes, trials-successes+1)),
            1. if successes == trials else float(beta.ppf(1-alpha / 2, successes+1, trials-successes))]


def conservative_bootstrap_guard(result: Mapping[str, Any], *, B: int,
                                 family_error: float) -> dict[str, Any]:
    """Conservative binomial tail upper bounds inside the planning rule.

    V3 exposes actual counts/reps for all three procedures and absolute-tail
    +1 p=(1+k)/(N+1). Bound that binomial tail directly; no doubled tail.
    Legacy doubled smaller bootstrap tails retain their two-tail guard.
    Allocate error over all MC tests and reapply the full 120-slot Holm rule.
    This guard is a planning sensitivity stricter than raw finite-B tests;
    it does not replace temporal bootstrap calibration under the market DGP.
    """
    if not 0 < family_error < 1:
        raise ValueError("bootstrap tail guard error must be in (0,1)")
    # Copy JSON-compatible statistical output; preserve every raw p-value.
    guarded = json.loads(json.dumps(result))
    all_tests = [c["methods"][m][n] for c in guarded["comparisons"] for m in METHODS for n in ("epa", "practical")]
    mc_tests = sum(t.get("mc_p_rule") == "absolute_tail_plus_one" for t in all_tests)
    allocation_tests = max(2 * len(guarded["comparisons"]), mc_tests)
    tests = []
    for comparison in guarded["comparisons"]:
        for method in METHODS:
            for null in ("epa", "practical"):
                test = comparison["methods"][method][null]
                if test["status"] == "ok" and test.get("mc_p_rule") == "absolute_tail_plus_one":
                    count, trials = int(test["mc_tail_count"]), int(test["mc_reps"])
                    raw = float(test["raw_p"])
                    if not 0 <= count <= trials or trials < 1 or not math.isclose(raw, (count+1)/(trials+1), rel_tol=1e-10):
                        raise ValueError("MC tail count/rule/reps do not match reported raw p")
                    upper = 1. if count == trials else float(beta.ppf(1-family_error/allocation_tests, count+1, trials-count))
                    test["mc_p_upper"] = max(raw, upper)
                    test["tail_guard_family_error"] = family_error
                elif method == "stationary_bootstrap" and test["status"] == "ok":
                    raw = float(test["raw_p"])
                    k = max(0, min(B, int(round(raw * (B+1)/2 - 1))))
                    upper = 1. if raw >= 1. or k == B else min(1., 2 * float(beta.ppf(1-family_error/(2*allocation_tests), k+1, B-k)))
                    test["mc_p_upper"] = max(raw, upper)
                    test["tail_guard_family_error"] = family_error
                    test["inferred_minimum_tail_exceedances"] = k
                tests.append(test)
    adjusted = _holm([t.get("mc_p_upper", t["raw_p"]) for t in tests])
    for test, value in zip(tests, adjusted):
        test["adjusted_p"] = value
    return guarded


def joint_power_bounds(contrasts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Dependence-free intersection bounds from marginal event probabilities.

    Every-positive contrast implies the full statistical rule (including no
    adverse additional result). The converse does not hold because the rule
    allows two of three additional panels. Hence its lower bound transfers,
    but the all-positive upper bound is NOT an upper bound for the R3 rule.
    Monte Carlo intervals provide uncertainty on these marginal estimates.
    """
    p = [float(c["power"]) for c in contrasts]
    lo = [float(c.get("simultaneous_mc_lower_bound", c.get("tail_guard_error_adjusted_mc_lower_bound"))) for c in contrasts]
    hi = [float(c["simultaneous_mc_interval"][1]) for c in contrasts]
    if not p:
        raise ValueError("nonempty marginal contrasts required")
    lower = max(0., 1-sum(1-x for x in p))
    conservative_lower = max(0., 1-sum(1-x for x in lo))
    return {"all_positive_contrast_intersection": {"frechet_union_lower": lower, "frechet_upper": min(p),
                                                  "simultaneous_mc_lower": conservative_lower,
                                                  "simultaneous_mc_upper": min(hi)},
            "full_statistical_rule": {"lower_from_all_positive_subset": lower,
                                       "simultaneous_mc_lower_from_all_positive_subset": conservative_lower,
                                       "upper": 1.,
                                       "reason": "all-positive intersection is sufficient, not necessary, for main-all plus two-of-three additional rule; its minimum marginal upper does not transfer"},
            "unknown_across_calendar_groups_dependence_not_assumed_identified": True}


def estimate_power(planning: Mapping[str, Any], design: Mapping[str, Any], cell: Mapping[str, int],
                   *, reps: int, seed: int, analyzer: Callable[..., Mapping[str, Any]],
                   deadline: float | None = None) -> dict[str, Any]:
    """Monte Carlo of the complete statistical rule under each fixed alternative.

    Counter outcomes are *joint per-contrast conjunctions*, not marginal DM
    rejection probabilities or a normal-quantile sample-size approximation.
    Small fixtures can call this function; dimension() separately requires a
    complete real-family receipt and independent temporal validation evidence.
    """
    if reps < 1:
        raise ValueError("positive Monte Carlo repetitions required")
    streams = np.random.SeedSequence(seed).spawn(len(design["alternatives_relative_rmse_gain"]))
    scenarios = []
    count_contrasts = len(planning["panels"]) * (len(planning["systems"])-1)
    ci_alpha = design["mc_confidence_error"] / (count_contrasts * len(streams))
    for gain, stream in zip(design["alternatives_relative_rmse_gain"], streams):
        if gain <= design["margin_relative_rmse"]:
            raise ValueError("power alternatives must exceed practical margin; boundary power cannot be .90")
        losses, factors = alternative_losses(planning["losses"], gain)
        rng = np.random.default_rng(stream)
        successes = np.zeros((len(planning["panels"]), len(planning["systems"])-1), dtype=int)
        joint, all_contrasts, adverse = 0, 0, 0
        diagnostic_successes = np.zeros_like(successes)
        diagnostic_joint = 0
        tail_count_ledger = []
        first_tail_diagnostic = None
        for iteration in range(reps):
            if deadline is not None and time.monotonic() >= deadline:
                return {"cell": dict(cell), "mc_seed": seed, "incomplete": True,
                        "reason": "predeclared Monte Carlo wall budget exhausted",
                        "completed_scenarios": scenarios,
                        "current_alternative": gain, "completed_reps_current_alternative": iteration,
                        "partial_contrast_success_counts": successes.tolist(),
                        "minimum_per_contrast_simultaneous_mc_lower_bound": 0., "meets_power_target": False}
            predictions, y, z = crossed_draw(planning, losses, R=cell["R"], T=cell["T"],
                                            block_length=design["outer_block_length"], rng=rng)
            reference_kwargs = {k: design[k] for k in ("reference_reps", "reference_seed", "alpha") if k in design}
            results = [analyzer(predictions[pi], y[pi], planning["systems"],
                                bootstrap_reps=cell["B"], block_length=design["inner_block_length"],
                                hac_lags=design["hac_lags"], instruments=z[pi],
                                margin=design["margin_relative_rmse"],
                                seed=int(rng.integers(0, 2**63-1)), horizon=design["horizon"], **reference_kwargs)
                       for pi in range(len(planning["panels"]))]
            tail_error = design["bootstrap_tail_guard_family_error"]
            # The actual final rule uses the finite-B count+1 p-values and
            # Holm. Replacing p-values by binomial tail upper bounds would
            # define a DIFFERENT rule; at B8191 it may never reject even when
            # k=0. Keep that precision sensitivity strictly diagnostic.
            diagnostic_results = [conservative_bootstrap_guard(r, B=cell["B"], family_error=tail_error) for r in results]
            diagnostic_rule = statistical_rule(diagnostic_results, design["margin_relative_rmse"])
            diagnostic_successes += diagnostic_rule["contrast_pass"]
            diagnostic_joint += diagnostic_rule["joint_statistical_rule_pass"]
            tail_count_ledger.append([
                [[t.get("mc_tail_count"), t.get("mc_reps"), t["status"]]
                 for c in panel["comparisons"] for method in METHODS for null in ("epa", "practical")
                 for t in [c["methods"][method][null]]]
                for panel in results])
            if first_tail_diagnostic is None:
                first_tail_diagnostic = [
                    [{"rival": c["rival"], "method": method, "null": null,
                      "raw_p": t["raw_p"], "mc_p_upper": t.get("mc_p_upper"),
                      "mc_tail_count": t.get("mc_tail_count"), "mc_reps": t.get("mc_reps"),
                      "status": t["status"]}
                     for c in panel["comparisons"] for method in METHODS for null in ("epa", "practical")
                     for t in [c["methods"][method][null]]]
                    for panel in diagnostic_results]
            rule = statistical_rule(results, design["margin_relative_rmse"])
            successes += rule["contrast_pass"]
            joint += rule["joint_statistical_rule_pass"]
            all_contrasts += rule["all_contrasts_pass"]
            adverse += rule["additional_significant_inferiority"]
        contrasts = []
        for pi, panel in enumerate(planning["panels"]):
            for ri, rival in enumerate(planning["systems"][1:]):
                k = int(successes[pi, ri])
                interval = binomial_interval(k, reps, ci_alpha)
                contrasts.append({"panel": panel, "rival": rival, "successes": k, "trials": reps,
                                  "power": k/reps, "simultaneous_mc_interval": interval,
                                  "simultaneous_mc_lower_bound": interval[0]})
        scenarios.append({"alternative_relative_rmse_gain": gain,
                          "alternative_primary_mse_ratio": (1-gain)**2,
                          "fixed_loss_scaling_factors": factors, "contrasts": contrasts,
                          "joint_statistical_rule_power": joint/reps,
                          "joint_point_identification_scope": planning.get("joint_calendar_coupling", "shared calendar fixture"),
                          "joint_power_dependence_free_bounds": joint_power_bounds(contrasts),
                          "joint_statistical_rule_mc_interval": binomial_interval(joint, reps, design["mc_confidence_error"]),
                          "joint_all_80_positive_contrasts_power": all_contrasts/reps,
                          "additional_inferiority_screen_rejection_rate": adverse/reps,
                          "finite_mc_tail_diagnostics": {
                              "used_for_continuation": False,
                              "reason": "Replacing finite-B p by CP tail upper bounds defines a stricter different rule; the actual inner bootstrap randomness is included in the outer power experiment.",
                              "diagnostic_family_error_per_panel": tail_error,
                              "strict_tail_upper_contrast_successes": diagnostic_successes.tolist(),
                              "strict_tail_upper_joint_power": diagnostic_joint/reps,
                              "count_ledger_order": {"panels": planning["panels"], "rivals": planning["systems"][1:],
                                                     "methods": list(METHODS), "nulls": ["epa", "practical"],
                                                     "cell_fields": ["mc_tail_count", "mc_reps", "status"]},
                              "replicate_tail_count_ledger": tail_count_ledger,
                              "first_prespecified_replicate_tail_intervals": first_tail_diagnostic}})
    worst = min(c["simultaneous_mc_lower_bound"] for s in scenarios for c in s["contrasts"])
    return {"cell": dict(cell), "mc_seed": seed, "mc_reps_per_alternative": reps,
            "operational_rule": OPERATIONAL_RULE,
            "scenarios": scenarios, "minimum_per_contrast_simultaneous_mc_lower_bound": worst,
            "meets_power_target": worst >= design["target_per_contrast_power"],
            "scope": "conditional empirical development loss DGP and declared disjoint-calendar coupling; fixed-seed temporal rule nested in grouped seed/calendar draws; joint point not identified across disjoint calendars; dependence-free bounds also reported"}


def _calibration(design: Mapping[str, Any], analyzer: Callable[..., Any],
                 cell: Mapping[str, int] | None = None) -> tuple[bool, str]:
    spec = design.get("temporal_validation_report")
    if not spec:
        return False, "full-family temporal calibration evidence not supplied"
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file() or sha256(path) != spec["sha256"]:
        return False, "temporal validation report missing/checksum mismatch"
    report = json.loads(path.read_text())
    expected = {"family_rivals": 20, "all_three_methods": True,
                "complete_and_partial_nulls": True, "extreme_tail_validated": True,
                "coverage_validated": True, "status": "PASSED_PREDECLARED_VALIDATION"}
    if any(report.get(k) != v for k, v in expected.items()):
        return False, "available temporal validation does not establish the full declared scope"
    settings = {"R": cell["R"], "T": cell["T"], "B": cell["B"],
                "block_length": design["inner_block_length"], "hac_lags": design["hac_lags"],
                "horizon": design["horizon"], "margin": design["margin_relative_rmse"]} if cell is not None else None
    if settings is not None:
        settings.update({k: design[k] for k in ("reference_reps", "reference_seed", "alpha") if k in design})
    if settings is None or settings not in report.get("validated_settings", []):
        return False, "temporal validation does not cover the exact selected R/T/B and temporal settings"
    source = inspect.getsourcefile(analyzer)
    if not source or report.get("statistics_source_sha256") != sha256(source):
        return False, "temporal validation does not match the executed statistical source"
    # Reuse the same evidence parser as the final controller. A self-authored
    # report carrying successful booleans cannot replace frozen plans, actual
    # complete replicate ledgers and recomputed error/coverage bounds. Import
    # here avoids a definition-time cycle (controller imports this module).
    from .controller import _method_admission, ControllerError
    try:
        _method_admission(Path(__file__).resolve().parents[1], spec, settings)
    except (ControllerError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        return False, "temporal validation evidence bundle rejected: " + str(exc)
    return True, "full declared temporal validation evidence bundle verified"


def dimension(planning_paths: Mapping[str, Any] | Sequence[str | Path],
              design: Mapping[str, Any] | str | Path, *, output: str | Path | None = None,
              analyzer: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Select with one Monte Carlo stream; validate the fixed winner once.

    Inputs can be absent: return NOT_DIMENSIONED with final R/T/B = null,
    budget bounds and the missing evidence, rather than inventing a design.
    When inputs exist this can be computationally expensive; it never trains.
    A failed winner is retained; a second winner is never selected after seeing
    validation outcomes. Changing the grid/DGP creates a new versioned study.
    """
    if isinstance(design, (str, Path)):
        config_path = Path(design)
        design = json.loads(config_path.read_text())
    design = dict(design)
    if analyzer is None:
        from .statistics import analyze
        analyzer = analyze
    analyzer_source = inspect.getsourcefile(analyzer)
    methods_sha = sha256(analyzer_source) if analyzer_source and Path(analyzer_source).is_file() else None
    result: dict[str, Any] = {"version": VERSION, "design_sha256": _identity(design),
                              "methods_sha256": methods_sha,
                              "source_identity": {"analyzer_source_file": str(analyzer_source) if analyzer_source else None,
                                                  "analyzer_source_sha256": methods_sha,
                                                  "power_source_sha256": sha256(__file__)},
                              "operational_rule": OPERATIONAL_RULE,
                              "status": "NOT_DIMENSIONED", "final_R": None, "final_T": None,
                              "final_B": None, "final_seed_ids": None, "r3_accredited": False,
                              "reasons": [], "validation_attempted": False,
                              "resampling_requirements": resampling_budget(alpha=design.get("alpha", .05)),
                              "reference_budget_R10_T252_not_dimensioned": budget(design, R=10, T=252)}
    calibration_now, calibration_reason = _calibration(design, analyzer)
    result["temporal_validation_gate_before_planning"] = {"approved": calibration_now, "reason": calibration_reason,
                                                          "report": design.get("temporal_validation_report")}
    try:
        if design["target_per_contrast_power"] < .90 or design["minimum_planning_seeds"] < 5:
            raise ValueError("at least .90 per-contrast power and five genuine planning seeds required")
        if design.get("power_decision_rule", OPERATIONAL_RULE) != OPERATIONAL_RULE or design.get("tail_mc_uncertainty_role", "diagnostic_only") != "diagnostic_only":
            raise ValueError("power design must match the actual finite-B final rule; tail uncertainty sensitivity is diagnostic")
        if methods_sha is None:
            raise ValueError("executed analyzer needs an actual source file identity")
        seed_prefix(design, max(design["grid"]["R"]))
        planning = load_planning(planning_paths, design)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        result["reasons"].append(str(exc))
        if output is not None:
            _write(output, result)
        return result
    result["planning_evidence"] = planning["evidence"]
    if output is not None and Path(output).is_file():
        prior = json.loads(Path(output).read_text())
        if "selection_results" in prior:
            if prior.get("design_sha256") == result["design_sha256"] and prior.get("planning_evidence") == planning["evidence"]:
                return prior
            raise ValueError("existing power study identity differs; use a new output/version and retain the previous result")
    result["empirical_dependence"] = dependence_summary(planning)
    result["observed_planning_cost"] = budget(design, R=len(planning["planning_seed_ids"]), T=len(planning["dates"]), fit_receipts=planning["fit_receipts"])
    requirement = resampling_budget(alpha=design.get("alpha", .05), expected_tail_exceedances=design["expected_tail_exceedances"], mc_p_rule=design.get("mc_p_rule", "absolute_tail_plus_one"))
    result["resampling_requirements"] = requirement
    cells, excluded = [], []
    for R, T, B in itertools.product(design["grid"]["R"], design["grid"]["T"], design["grid"]["B"]):
        cell = {"R": int(R), "T": int(T), "B": int(B)}
        cost = budget(design, R=R, T=T)
        reason = []
        if R < 10:
            reason.append("at_least_10_final_seeds")
        if B < requirement["minimum_B_for_two_sided_resolution"]:
            reason.append("bootstrap_two_sided_family_resolution")
        cost["bootstrap_expected_exceedances_per_smallest_tail"] = B * requirement["smallest_family_threshold"] / requirement["p_resolution_numerator"]
        cost["bootstrap_tail_precision_method"] = "actual finite-B count+1/Holm decision; CP upper-tail sensitivity diagnostic only; simultaneous outer-MC power confidence bounds"
        if design["inner_block_length"] > T/3 or design["hac_lags"] >= T//3:
            reason.append("temporal_settings_incompatible_with_T")
        if B*T > design["budget"]["limits"]["max_bootstrap_index_cells_per_call"]:
            reason.append("bootstrap_memory_work_limit")
        if not cost["within_declared_known_limits"]:
            reason.extend(cost["limit_failures"])
        if reason:
            excluded.append({"cell": cell, "reasons": reason, "budget": cost})
        else:
            cells.append((cell, cost))
    # The objective and tie-breaking order are frozen before any MC outcome.
    cells.sort(key=lambda item: (item[1]["nominal_neural_active_seconds"], item[1]["prediction_rows"], item[0]["B"]))
    result["excluded_grid_cells"] = excluded
    calls = len(cells) * design["mc_selection_reps"] * len(design["alternatives_relative_rmse_gain"]) * len(design["panels"])
    result["selection_analyzer_calls"] = calls
    validation_calls = design["mc_validation_reps"] * len(design["alternatives_relative_rmse_gain"]) * len(design["panels"]) if cells else 0
    result["validation_analyzer_calls_upper_bound"] = validation_calls
    result["total_analyzer_calls_upper_bound"] = calls + validation_calls
    result["selection_bootstrap_index_cells_upper_bound"] = sum(c["B"] * c["T"] for c, _ in cells) * design["mc_selection_reps"] * len(design["alternatives_relative_rmse_gain"]) * len(design["panels"])
    result["largest_bootstrap_index_cells_per_call"] = max((c["B"]*c["T"] for c, _ in cells), default=0)
    if calls > design["budget"]["limits"]["max_selection_analyzer_calls"]:
        result["reasons"].append("power simulation exceeds predeclared selection compute limit; no cells silently removed")
        if output is not None:
            _write(output, result)
        return result
    if calls + validation_calls > design["budget"]["limits"].get("max_total_analyzer_calls", calls+validation_calls):
        result["reasons"].append("power simulation exceeds predeclared selection plus independent-validation call budget")
        if output is not None:
            _write(output, result)
        return result
    study_seed = np.random.SeedSequence(design["mc_seed"])
    streams = study_seed.spawn(len(cells) + 1)
    summaries, winner = [], None
    mc_start = time.monotonic()
    deadline = mc_start + design["budget"]["limits"]["max_power_mc_wall_seconds"]
    for index, (cell, cost) in enumerate(cells):
        mc_seed = int(streams[index].generate_state(1, dtype=np.uint64)[0])
        estimate = estimate_power(planning, design, cell, reps=design["mc_selection_reps"], seed=mc_seed, analyzer=analyzer, deadline=deadline)
        estimate["budget"] = cost
        summaries.append(estimate)
        if estimate.get("incomplete"):
            result["reasons"].append("predeclared Monte Carlo wall budget exhausted; partial study preserved")
            break
        if estimate["meets_power_target"]:
            winner = estimate
            break
    result["selection_results"] = summaries
    if winner is None:
        result["reasons"].append("no feasible grid cell demonstrated >=0.90 per contrast under all declared meaningful alternatives")
    else:
        winner_record = {"cell": winner["cell"], "design_sha256": result["design_sha256"],
                         "planning_evidence_sha256": _identity(planning["evidence"]),
                         "selection_results_sha256": _identity(summaries),
                         "validation_stream_spawn_key": list(streams[-1].spawn_key)}
        result["fixed_before_validation"] = winner_record
        result["fixed_before_validation_sha256"] = _identity(winner_record)
        if output is not None:
            # Persist the winner *before* querying the independent MC stream.
            _write(str(output) + ".winner_before_validation.json", winner_record)
        validation_seed = int(streams[-1].generate_state(1, dtype=np.uint64)[0])
        result["validation_attempted"] = True
        result["validation_result"] = estimate_power(planning, design, winner["cell"], reps=design["mc_validation_reps"], seed=validation_seed, analyzer=analyzer, deadline=deadline)
        calibrated, reason = _calibration(design, analyzer, winner["cell"])
        result["temporal_calibration_verified"] = calibrated
        if not result["validation_result"]["meets_power_target"]:
            result["reasons"].append("fixed winner failed independent empirical-DGP validation; no replacement winner selected")
        if not calibrated:
            result["reasons"].append(reason)
        if result["validation_result"]["meets_power_target"] and calibrated:
            cell = winner["cell"]
            result.update(status="DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP", final_R=cell["R"], final_T=cell["T"], final_B=cell["B"],
                          final_seed_ids=seed_prefix(design, cell["R"]), final_budget=winner["budget"])
    result["measured_power_mc_wall_seconds"] = time.monotonic()-mc_start
    result["limitations"] = ["Empirical DGP finite seed support cannot establish unseen training-tail behavior.",
                             "Calendar groups use coincident actual dates; independent calendar coupling across groups is a simulation assumption, not identified cross-calendar dependence. Joint points are conditional and dependence-free bounds are reported.",
                             "Calendar weak stationarity/mixing, representative development, adequate block length and crossed seed/calendar resampling are assumptions.",
                             "Inner temporal inference conditions on the sampled fitted models; outer seed variation is a planning model, not an unconditional temporal theorem.",
                             "All alternatives impose specified >2% aggregate-RMSE gains; no finite design guarantees .90 power for every effect arbitrarily close to the margin.",
                             "Reported joint power concerns the implemented statistical rule; source rights, panel adequacy, model robustness and all R3 evidence remain external gates.",
                             "No Monte Carlo, normal-quantile bound, seed minimum or resampling minimum accredits R3."]
    if output is not None:
        _write(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True)
    parser.add_argument("--predictions", nargs="*", default=[])
    parser.add_argument("--receipts", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    paths = {"predictions": args.predictions, "receipts": args.receipts} if args.receipts else args.predictions
    result = dimension(paths, args.design, output=args.output)
    print(json.dumps({"status": result["status"], "final_R": result["final_R"], "final_T": result["final_T"], "final_B": result["final_B"], "reasons": result["reasons"]}, indent=2))
    return 0 if result["status"] == "DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP" else 2


if __name__ == "__main__":
    raise SystemExit(main())
