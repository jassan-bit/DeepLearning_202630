"""Versioned E2 calendar metadata: three HPO anchors, one initial TiDE NAS.

Only dates and workload bounds are calculated here. No market source, trainer,
price, target, checkpoint, campaign balance or final-test payload is opened.
Historical dates retain their exploratory status; this calendar creates no new
independent test set and does not migrate a sealed controller.
"""
from __future__ import annotations

import argparse
from calendar import monthrange
from copy import deepcopy
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Any, Mapping

VERSION = "e2-amended-calendar-v1"
DAY = timedelta(days=1)
HPO_ANCHORS = ("2024-01-01", "2024-07-01", "2024-10-01")
OUTER_ORIGINS = tuple(f"2024-{month:02d}-01" for month in range(1, 11))
NAS_ORIGIN = HPO_ANCHORS[0]
POWER_MODELS = ("TimeMixer", "PatchTST", "GRU32")
POWER_SEEDS = tuple(range(271701, 271706))
INNER_OFFSETS = {"train_cutoff_offset_days": -98,
                 "validation_start_offset_days": -90,
                 "validation_end_offset_days": -8}
DEFAULT_CALENDAR = {
    "hpo_anchors": list(HPO_ANCHORS),
    "outer_development_origins": list(OUTER_ORIGINS),
    "nas_origin": NAS_ORIGIN,
    "inner": dict(INNER_OFFSETS),
    "embargo_days": 7,
    "recipe_rule": "latest_hpo_anchor_at_or_before_origin",
    "historical_final_origin": "2025-01-01",
    "historical_final_end": "2026-08-31",
    "power_pilot": {"origin": "2024-11-08", "end": "2024-12-30",
                    "train_label_end": "2024-10-31",
                    "models": list(POWER_MODELS), "seeds": list(POWER_SEEDS)},
}


def _day(value: str | date) -> date:
    """Accept a calendar day, never silently truncate a timestamp/time zone."""
    if isinstance(value, datetime):
        raise ValueError("Supply a UTC calendar date, not a timestamp")
    if type(value) is date:
        return value
    if (not isinstance(value, str) or len(value) != 10
            or value[4:5] != "-" or value[7:8] != "-"):
        raise ValueError("Use an explicit ISO UTC calendar date YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Invalid ISO UTC calendar date") from error
    if parsed.isoformat() != value:
        raise ValueError("Use a canonical ISO UTC calendar date")
    return parsed


def _config(protocol: Mapping[str, Any] | None = None) -> dict:
    """Validate an optional full protocol or calendar against this amendment."""
    if protocol is None:
        return deepcopy(DEFAULT_CALENDAR)
    if not isinstance(protocol, Mapping):
        raise TypeError("Calendar metadata must be a mapping")
    config = protocol.get("calendar", protocol)
    if not isinstance(config, Mapping):
        raise TypeError("The protocol calendar must be a mapping")
    for key in ("hpo_anchors", "outer_development_origins", "nas_origin",
                "embargo_days", "recipe_rule", "historical_final_origin", "historical_final_end"):
        if config.get(key) != DEFAULT_CALENDAR[key]:
            raise ValueError("Calendar differs from the registered three-anchor amendment: " + key)
    inner = config.get("inner")
    if not isinstance(inner, Mapping) or any(inner.get(k) != v for k, v in INNER_OFFSETS.items()):
        raise ValueError("Inner training, validation or embargo offsets changed")
    power = config.get("power_pilot")
    if (not isinstance(power, Mapping) or power.get("origin") != "2024-11-08"
            or power.get("end") != "2024-12-30"):
        raise ValueError("The disjoint development power-pilot calendar changed")
    # The metadata schema may call this field cutoff or train_label_end. Both
    # mean the declared maximum, not a claim about the last quality-eligible row.
    for name in ("train_label_end", "train_cutoff", "refit_train_end"):
        if name in power and power[name] != "2024-10-31":
            raise ValueError("Power-pilot fitting must end no later than its registered cutoff")
    for name in ("models", "model_names"):
        if name in power and tuple(power[name]) != POWER_MODELS:
            raise ValueError("The prospective power pilot cannot reintroduce the optional hybrid")
    for name in ("seeds", "seed_ids", "seed_values"):
        if name in power and tuple(power[name]) != POWER_SEEDS:
            raise ValueError("The five independent power-pilot seed IDs changed")
    return deepcopy(dict(config))


def boundaries(origin: str | date) -> dict:
    """Compile both seven-day bridges under the existing D7/sigma7 contract."""
    origin = _day(origin)
    inner_cutoff = origin - 98 * DAY
    validation_start, validation_end = origin - 90 * DAY, origin - 8 * DAY
    return {
        "origin": origin.isoformat(),
        "inner_train_cutoff": inner_cutoff.isoformat(),
        "inner_validation_start": validation_start.isoformat(),
        "inner_validation_end": validation_end.isoformat(),
        "inner_validation_calendar_days": (validation_end - validation_start).days + 1,
        "inner_bridge_start": (inner_cutoff + DAY).isoformat(),
        "inner_bridge_end": (validation_start - DAY).isoformat(),
        "refit_train_cutoff": validation_end.isoformat(),
        "bridge_start": (origin - 7 * DAY).isoformat(),
        "bridge_end": (origin - DAY).isoformat(),
        "bridge_calendar_days": 7,
    }


def selection_anchor(origin: str | date) -> str:
    """Use only a recipe selected at an anchor available by the forecast origin."""
    origin = _day(origin)
    anchors = [anchor for anchor in HPO_ANCHORS if _day(anchor) <= origin]
    if not anchors:
        raise ValueError("No registered recipe is available before the first HPO anchor")
    return anchors[-1]


def validate_forecast_context(origin: str | date, selected_at: str | date,
                              train_label_end: str | date) -> dict:
    """Reject future/stale recipes and labels inside the target-overlap embargo."""
    origin, selected_at, train_label_end = map(_day, (origin, selected_at, train_label_end))
    expected = selection_anchor(origin)
    if selected_at.isoformat() != expected:
        raise ValueError("Use the latest registered HPO anchor at or before this forecast origin")
    cutoff = origin - 8 * DAY
    if train_label_end > cutoff:
        raise ValueError("Training labels intrude into the seven-day embargo or the forecast period")
    return {"origin": origin.isoformat(), "selection_anchor": expected,
            "train_label_end": train_label_end.isoformat(),
            "maximum_train_label_end": cutoff.isoformat(),
            "status": "chronology_validated_from_metadata_only"}


def validate_nas_dependency(hpo_origin: str | date, *, winner_origin: str | date | None = None,
                            winner_architecture_id: str | None = None) -> dict:
    """Later TiDE HPO uses the frozen initial winner, never another NAS search."""
    anchor = _day(hpo_origin).isoformat()
    if anchor not in HPO_ANCHORS:
        raise ValueError("NAS/HPO dependency requires a registered HPO anchor")
    if anchor == NAS_ORIGIN:
        if winner_origin is not None or winner_architecture_id is not None:
            raise ValueError("Initial reference HPO must precede the first NAS winner")
        return {"hpo_anchor": anchor, "body_for_hpo": "registered_reference_architecture",
                "next_stage": "single_exhaustive_TiDE_NAS", "repeat_nas": False}
    if (winner_origin is None or _day(winner_origin).isoformat() != NAS_ORIGIN
            or not isinstance(winner_architecture_id, str) or not winner_architecture_id.strip()):
        raise ValueError("Later HPO requires the identified architecture winner frozen at the initial NAS origin")
    return {"hpo_anchor": anchor, "body_for_hpo": "frozen_initial_TiDE_NAS_winner",
            "winner_origin": NAS_ORIGIN, "winner_architecture_id": winner_architecture_id,
            "repeat_nas": False}


def selection_inventory(k: int = 40) -> dict:
    """Prospective bounds, excluding cache reuse, pilots, refits and analysis.

    Values of k below the registered default are cost scenarios only; this
    function neither chooses k nor authorizes changing a sealed NAS protocol.
    """
    if type(k) is not int or not 10 <= k <= 40:
        raise ValueError("Inventory scenarios require an integer k in [10,40]")
    hpo = 13 * 18 * 2 * len(HPO_ANCHORS)
    sae_extra = 18 * 2 * len(HPO_ANCHORS) * 2
    nas1, nas2 = 1000, 5 * (k + 1)
    return {
        "scope": "prospective_selection_upper_bound_before_compatible_reuse",
        "neural_systems": 13, "hpo_anchors": len(HPO_ANCHORS),
        "trials_per_neural_anchor": 18, "hpo_seeds": 2,
        "neural_hpo_logical_fits": hpo,
        "TiDE_nas_searches": 1, "nas_phase1_architectures": 1000,
        "nas_phase1_logical_fits": nas1, "nas_phase2_k": k,
        "nas_phase2_seeds": 5, "nas_phase2_reference_upper_bound": 1,
        "nas_phase2_logical_fit_upper_bound": nas2,
        "neural_selection_logical_fit_upper_bound": hpo + nas1 + nas2,
        "SAE_additional_auxiliary_optimizer_stages": sae_extra,
        "neural_selection_optimizer_stage_upper_bound": hpo + sae_extra + nas1 + nas2,
        "ARIMA_candidate_fits": 18 * len(HPO_ANCHORS),
        "seasonal_naive_validation_rules": 3 * len(HPO_ANCHORS),
        "naive_optimizer_fits": 0,
        "registered_k_default": 40, "inventory_does_not_authorize_k_change": True,
        "observed_completed_fits": None, "compatible_reuse_deducted": False,
        "wall_hours_estimated": None,
        "excluded_work": ["resource/noise pilots", "power pilot", "monthly selected-recipe refits",
                          "historical/final evaluation refits", "ablations", "robustness",
                          "uncertainty, explanation and inference", "clean reproduction"],
    }


def compile_calendar(protocol: Mapping[str, Any] | None = None) -> dict:
    """Return flat rows for three HPO and ten outer-calendar CSV exports."""
    _config(protocol)
    hpo_rows = []
    for anchor in HPO_ANCHORS:
        hpo_rows.append({**boundaries(anchor), "neural_systems": 13,
            "trials_per_neural": 18, "optimization_seeds": 2,
            "neural_hpo_logical_fits": 13 * 18 * 2,
            "ARIMA_candidate_fits": 18, "run_initial_TiDE_NAS": anchor == NAS_ORIGIN,
            "TiDE_HPO_body": "reference_before_initial_NAS" if anchor == NAS_ORIGIN
                else "frozen_initial_NAS_winner",
            "TiDE_architecture_origin": NAS_ORIGIN,
            "data_role": "known_development_selection"})
    outer_rows = []
    for text in OUTER_ORIGINS:
        origin = _day(text)
        end = origin.replace(day=monthrange(origin.year, origin.month)[1])
        outer_rows.append({"origin": text, "end": end.isoformat(),
            "calendar_days": (end - origin).days + 1,
            "selection_anchor": selection_anchor(origin),
            "train_label_cutoff": (origin - 8 * DAY).isoformat(),
            "bridge_start": (origin - 7 * DAY).isoformat(),
            "bridge_end": (origin - DAY).isoformat(),
            "repeat_HPO": False, "repeat_NAS": False,
            "fresh_weights_and_train_only_scalers": True,
            "data_role": "known_historical_exploratory_evaluation"})
    power_origin, power_end = _day("2024-11-08"), _day("2024-12-30")
    last_validation_end = _day(hpo_rows[-1]["inner_validation_end"])
    if last_validation_end >= power_origin:
        raise ValueError("HPO validation and the power pilot overlap")
    return {
        "version": VERSION,
        "date_system": "UTC daily calendar dates; missing days never compressed",
        "hpo_plan_rows": hpo_rows, "outer_plan_rows": outer_rows,
        "historical_final": {"origin": "2025-01-01", "end": "2026-08-31",
            "selection_anchor": selection_anchor("2025-01-01"),
            "train_label_cutoff": "2024-12-24", "bridge_start": "2024-12-25",
            "bridge_end": "2024-12-31", "repeat_HPO": False, "repeat_NAS": False,
            "data_role": "known_608_dates_exploratory_not_an_independent_test"},
        "power_pilot": {"origin": power_origin.isoformat(), "end": power_end.isoformat(),
            "calendar_days": (power_end - power_origin).days + 1,
            "train_label_cutoff": "2024-10-31", "bridge_start": "2024-11-01",
            "bridge_end": "2024-11-07", "selection_anchor": "2024-10-01",
            "models": list(POWER_MODELS), "seeds": list(POWER_SEEDS),
            "logical_fits": len(POWER_MODELS) * len(POWER_SEEDS),
            "last_HPO_validation_end": last_validation_end.isoformat(),
            "labels_used_for_HPO_or_NAS_selection": False,
            "data_role": "known_development_power_pilot_only"},
        "nas_policy": {"model": "TiDE", "origin": NAS_ORIGIN,
            "sequence": ["reference_HPO", "1000_architecture_screening", "k_plus_reference_confirmation"],
            "later_HPO_requires_identified_frozen_initial_winner": True,
            "full_NAS_repeated_per_outer_fold": False},
        "known_dates_become_unseen_by_this_amendment": False,
        "final_independent_cohort_defined_here": False,
        "R_fixed_here": False, "market_data_read": False,
        "private_Colab_session_modified": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, help="Read protocol JSON metadata only")
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol else None
    print(json.dumps(compile_calendar(protocol), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
