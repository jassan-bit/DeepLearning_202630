"""Descriptive, aggregate-only reports of the registered finite HPO search.

Associations across twelve configurations are neither causal parameter
importance nor an independent validation of the selected recipe. No data loader,
model fitting, extra configuration, or hypothesis test belongs in this module.
"""
from __future__ import annotations

from collections import defaultdict
import atexit
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import spearmanr


NEURAL_SUPPORT = {
    "learning_rate": {"kind": "continuous", "scale": "log10", "bounds": [1e-4, .003]},
    "weight_decay": {"kind": "continuous", "scale": "log10", "bounds": [1e-6, .01]},
    "dropout": {"kind": "continuous", "scale": "linear", "bounds": [0., .3]},
    "batch_size": {"kind": "categorical", "levels": [16, 32, 64]},
}


def _finite(value, *, nonnegative=False):
    return (isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value)) and (not nonnegative or value >= 0))


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _axes(candidates, model):
    if model == "ARIMA":
        return {"order": {"kind": "categorical", "levels": [r["parameters"]["order"] for r in candidates],
                          "meaning": "Registered (p,d,q) recipes; not a continuous learning-rate axis"}}
    if model == "Ridge":
        values = [r["parameters"]["alpha"] for r in candidates]
        if not all(_finite(v) and v > 0 for v in values):
            raise ValueError("Ridge candidates require positive finite alpha")
        return {"alpha": {"kind": "continuous", "scale": "log10", "bounds": [min(values), max(values)],
                          "support_origin": "endpoints of the explicitly registered finite alpha grid"}}
    if any(set(r["parameters"]) != set(NEURAL_SUPPORT) for r in candidates):
        raise ValueError("Neural reports require the four registered optimization axes")
    return NEURAL_SUPPORT


def _position(value, axis):
    if axis["kind"] == "categorical":
        return {"kind": "categorical", "value": value, "declared_levels": axis["levels"],
                "in_random_support": value in axis["levels"], "support_fraction": None}
    low, high = axis["bounds"]
    transform = math.log10 if axis["scale"] == "log10" else float
    if not _finite(value) or (axis["scale"] == "log10" and value <= 0):
        raise ValueError("A continuous HPO parameter is outside its mathematical domain")
    width = transform(high)-transform(low)
    return {"kind": "continuous", "value": float(value), "scale": axis["scale"],
            "bounds": list(axis["bounds"]), "in_random_support": low <= value <= high,
            "support_fraction": (transform(value)-transform(low))/width if width else None,
            "support_fraction_clipped": False}


def build_hpo_report(candidates, rankings, *, model, expected_candidates=12):
    """Return all registered candidates, observed ranks and exploratory curves.

    ``rankings`` is ``primary_recipe.json['ranking']`` from the final development
    recipe search. A missing attempt or nonfinite score remains explicit. The
    score is mean seed-wise RMSE; no post-selection confidence interval is made.
    """
    candidates = [dict(r) for r in candidates if r.get("model") == model]
    if len(candidates) != expected_candidates or expected_candidates != 12:
        raise ValueError("Exactly twelve registered candidates per tunable model are required")
    registry = {r["candidate_id"]: r for r in candidates}
    if len(registry) != len(candidates):
        raise ValueError("Duplicate candidate identity")
    ranks = {}
    for row in rankings:
        identity = row["candidate_id"]
        if identity not in registry or identity in ranks:
            raise ValueError("Unknown or duplicate ranked candidate")
        ranks[identity] = dict(row)
    axes = _axes(candidates, model)
    rows = []
    for candidate in candidates:
        identity = candidate["candidate_id"]
        result = ranks.get(identity, {"status": "not_attempted"})
        status = result.get("status", "not_attempted")
        score = result.get("mean_seed_RMSE")
        admissible = status == "passed" and _finite(score, nonnegative=True)
        if status == "passed" and not admissible:
            status = "failed_nonfinite_or_invalid_score"
        rows.append({"candidate_id": identity, "parameters": candidate["parameters"],
                     "status": status, "valid_score": admissible,
                     "mean_seed_RMSE": float(score) if admissible else None,
                     "latency_ms": float(result["latency_ms"]) if _finite(result.get("latency_ms"), nonnegative=True) else None,
                     "parameters_total": result.get("parameters_total"),
                     "rank_observed": None,
                     "source_reference_point": candidate.get("source_reference_point") is True,
                     "failure_count": len(result.get("failures", [])),
                     "assigned_compute_seconds": candidate.get("assigned_compute_seconds"),
                     "positions": {key: _position(candidate["parameters"][key], axis) for key, axis in axes.items()}})
    valid = [r for r in rows if r["valid_score"]]
    calendars = {ranks[r["candidate_id"]].get("dates_sha256") for r in valid
                 if ranks[r["candidate_id"]].get("dates_sha256") is not None}
    if len(calendars) > 1:
        raise ValueError("HPO comparison requires the same full validation calendar")
    def key(row):
        latency = row["latency_ms"] if row["latency_ms"] is not None else math.inf
        total = row["parameters_total"] if _finite(row["parameters_total"], nonnegative=True) else math.inf
        return row["mean_seed_RMSE"], latency, total, row["candidate_id"]
    ordered = sorted(valid, key=key)
    for rank, row in enumerate(ordered, 1):
        row["rank_observed"] = rank
    curves = {}
    for axis_name, specification in axes.items():
        observations = [{"candidate_id": r["candidate_id"], "parameter": r["parameters"][axis_name],
                         "mean_seed_RMSE": r["mean_seed_RMSE"],
                         "source_reference_point": r["source_reference_point"]} for r in valid]
        if specification["kind"] == "categorical":
            by_level = defaultdict(list)
            for obs in observations:
                by_level[json.dumps(obs["parameter"], sort_keys=True)].append(obs["mean_seed_RMSE"])
            groups = []
            for level in specification["levels"]:
                values = by_level[json.dumps(level, sort_keys=True)]
                groups.append({"level": level, "valid_candidates": len(values),
                               "mean_candidate_RMSE": float(np.mean(values)) if values else None})
            curves[axis_name] = {"axis": specification, "observations": observations,
                                "level_summaries": groups,
                                "association_status": "categorical_descriptive_only", "Spearman_rho": None}
        else:
            x = [o["parameter"] for o in observations]; y = [o["mean_seed_RMSE"] for o in observations]
            estimable = len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0
            curves[axis_name] = {"axis": specification, "observations": observations,
                                "association_status": "estimated_exploratory" if estimable else "not_estimable",
                                "not_estimable_reason": None if estimable else "Fewer than three valid candidates, constant parameter, or constant score",
                                "Spearman_rho": float(spearmanr(x, y).statistic) if estimable else None}
    return _json_value({"schema_version": 1, "model": model, "expected_candidates": 12,
                       "observed_valid_candidates": len(valid), "recorded_attempts": len(ranks),
                       "status": "complete_registered_finite_search" if len(valid) == 12 else "incomplete_or_failed_search",
                       "all_candidates": rows, "best_observed": ordered[0] if ordered else None,
                       "winner_tie_break": "lower mean-seed RMSE, lower latency, lower parameter count, lexicographic candidate ID; absent costs remain unknown",
                       "curves": curves, "score": "development mean per-seed RMSE in original sigma7 units",
                       "inference": {"causal_hyperparameter_importance": False, "p_values_or_significance_claims": False,
                                     "global_optimum_or_held_out_winner_performance_proved": False,
                                     "limitation": "Only twelve jointly varying configurations, including one source reference; marginal associations are confounded and descriptive. Failed or missing attempts do not earn a zero score. A best observed candidate in a partial search does not close HPO."},
                       "model_fits_performed_by_report": 0})


def write_hpo_report(report, output_dir):
    """Write aggregate JSON and standalone SVG curves, returning artifact paths."""
    if "MPLCONFIGDIR" not in os.environ:
        cache = tempfile.TemporaryDirectory(prefix="e2-hpo-mpl-")
        os.environ["MPLCONFIGDIR"] = cache.name
        os.environ.setdefault("XDG_CACHE_HOME", cache.name)
        atexit.register(cache.cleanup)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path, figure_path = output/"hpo_report.json", output/"hpo_curves.svg"
    if json_path.exists() or figure_path.exists():
        raise ValueError("Closed HPO report artifacts cannot be overwritten")
    rows = report["all_candidates"]; curves = report["curves"]
    fig, panels = plt.subplots(1+len(curves), 1, figsize=(8, 3*(1+len(curves))), squeeze=False)
    ax = panels[0, 0]
    for index, row in enumerate(rows):
        if row["valid_score"]:
            ax.scatter(index, row["mean_seed_RMSE"], marker="D" if row["source_reference_point"] else "o", color="#175586")
    ax.set_xticks(range(len(rows)), [r["candidate_id"].rsplit("-", 1)[-1] for r in rows], rotation=45)
    ax.set_ylabel("Mean seed RMSE (sigma7)")
    ax.set_title(report["model"]+": "+report["status"]+"; missing scores are not zero", fontsize=10)
    for panel, (name, curve) in zip(panels[1:, 0], curves.items()):
        obs = curve["observations"]; spec = curve["axis"]
        if spec["kind"] == "continuous":
            for row in obs:
                panel.scatter(row["parameter"], row["mean_seed_RMSE"], marker="D" if row["source_reference_point"] else "o", color="#175586")
            if spec["scale"] == "log10": panel.set_xscale("log")
            rho = curve["Spearman_rho"]
            panel.set_title("Exploratory Spearman rho="+("NE" if rho is None else f"{rho:.3f}")+"; association, no causal importance", fontsize=10)
        else:
            groups = curve["level_summaries"]
            for index, group in enumerate(groups):
                if group["mean_candidate_RMSE"] is not None:
                    panel.scatter(index, group["mean_candidate_RMSE"], color="#175586")
            panel.set_xticks(range(len(groups)), [str(g["level"]) for g in groups], rotation=45)
            panel.set_title("Mean of observed candidate scores per level; jointly varied axes confound comparison", fontsize=9)
        panel.set_xlabel(name); panel.set_ylabel("Mean seed RMSE (sigma7)")
    fig.tight_layout()
    fig.savefig(figure_path, format="svg", metadata={"Date": None})
    plt.close(fig)
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    return [json_path, figure_path]
