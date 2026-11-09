"""Enumerate E2 search plans and count budgets without model/data execution.

Only the Python standard library is used. This module does not implement model
factories, read market data, fit predictors, or certify a statistical power target.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Any

SCHEMA_VERSION = 1
PLANNER_VERSION = "e2-search-plan-v1"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _unique(values: list[Any], name: str) -> None:
    if len({json.dumps(v, sort_keys=True, allow_nan=False) for v in values}) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


def _domain(values: Any, name: str, *, empty: bool = False) -> list[Any]:
    if not isinstance(values, list) or (not empty and not values):
        raise ValueError(f"{name} must be a nonempty list")
    for v in values:
        if type(v) not in (str, bool, int, float):
            raise ValueError(f"{name} only accepts scalar JSON values")
        if type(v) in (int, float):
            _finite(v, name)
    _unique(values, name)
    return values


def _log_range(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} requires [positive lower, upper]")
    low, high = [_finite(v, name) for v in value]
    if not 0 < low < high:
        raise ValueError(f"{name} requires 0 < lower < upper")


def validate_protocol(protocol: dict[str, Any]) -> None:
    if (not isinstance(protocol, dict) or type(protocol.get("schema_version")) is not int
            or protocol["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Unsupported or missing protocol schema_version")
    models = protocol.get("models")
    refs = protocol.get("neural_reference_profiles")
    if not isinstance(models, list) or len(models) != 11:
        raise ValueError("The frozen catalogue requires eleven candidate profiles")
    if not isinstance(refs, list) or len(refs) != 3:
        raise ValueError("Three separate neural reference profiles are required")
    names, ids = [], []
    for i, model in enumerate(models + refs):
        if not isinstance(model, dict):
            raise ValueError("Model profiles must be objects")
        name, recipe = model.get("model"), model.get("search_recipe_id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Every model needs a nonempty name")
        if not isinstance(recipe, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", recipe):
            raise ValueError("Every model needs a safe nonempty search_recipe_id")
        names.append(name)
        ids.append(recipe)
        if type(model.get("nas")) is not bool:
            raise ValueError("NAS must be explicitly true or false")
        if i >= len(models) and model["nas"]:
            raise ValueError("Frozen E1 reference architectures cannot enter this NAS")
        aux = _integer(model.get("auxiliary_phases", 0), "auxiliary_phases")
        if aux not in (0, 2):
            raise ValueError("This protocol supports zero or two auxiliary phases")
        if i >= len(models) and aux:
            raise ValueError("E1 reference profiles cannot have auxiliary phases")
        _integer(model.get("same_window_moments"), "same_window_moments")
        hpo = model.get("hpo")
        if not isinstance(hpo, dict):
            raise ValueError("Every profile requires an hpo object")
        values = _domain(hpo.get("dropout_values"), "dropout_values", empty=model["nas"])
        for p in values:
            if not 0 <= _finite(p, "dropout") < 1:
                raise ValueError("Dropout must be in [0,1)")
        axes = hpo.get("extra_axes")
        if not isinstance(axes, dict):
            raise ValueError("extra_axes must be an object")
        reserved = {"trial_id", "recipe_id", "learning_rate", "weight_decay", "batch_size",
                    "dropout", "optimization_seeds", "architecture_reference"}
        for key, values in axes.items():
            if not isinstance(key, str) or not key or key in reserved:
                raise ValueError("Extra HPO axes need nonreserved names")
            _domain(values, key)
    _unique(names, "model names")
    _unique(ids, "recipe ids")

    common = protocol.get("common_hpo", {})
    trials = _integer(common.get("trials"), "trials", 1)
    if trials != 18:
        raise ValueError("This version fixes eighteen HPO opportunities alongside eighteen ARIMA orders")
    _integer(common.get("seed"), "HPO generation seed")
    zeros = _integer(common.get("zero_weight_decay_trials"), "zero WD trials")
    if not 0 < zeros < trials:
        raise ValueError("WD mixture must include both zero and positive trials")
    _log_range(common.get("learning_rate"), "learning_rate")
    _log_range(common.get("weight_decay_positive"), "weight_decay_positive")
    for value in _domain(common.get("batch_sizes"), "batch_sizes"):
        _integer(value, "batch size", 2)
    opt_seeds = _domain(common.get("optimization_seeds"), "optimization_seeds")
    for seed in opt_seeds:
        _integer(seed, "optimization seed")

    nas = protocol.get("nas", {})
    if nas.get("widths") != [16, 32, 64, 128, 256]:
        raise ValueError("NAS widths differ from the guide's fixed space")
    if nas.get("dropout_rate") != 0.2:
        raise ValueError("NAS dropout rate must be 0.2")
    if nas.get("reference") != {"widths": [64, 64, 64], "dropout_pattern": [0, 1, 0]}:
        raise ValueError("NAS reference must match the declared guide reference")
    phase1 = _integer(nas.get("phase1_seed"), "NAS phase1 seed")
    phase2 = _domain(nas.get("phase2_seeds"), "phase2_seeds")
    if len(phase2) < 5:
        raise ValueError("NAS phase2 needs at least five distinct seeds")
    for seed in phase2:
        _integer(seed, "NAS phase2 seed")
    _unique(opt_seeds + [phase1] + phase2, "optimization/NAS seed sets")
    ks = _domain(nas.get("k_options"), "k_options")
    for k in ks:
        _integer(k, "NAS finalist count", 10)
        if k > 999:
            raise ValueError("NAS finalist count cannot exceed 999 plus reference")

    settings = protocol.get("budget", {})
    for key in ("selection_origins", "outer_cv_origins", "final_selection_origins",
                "neural_references", "min_final_seeds", "max_final_seeds", "noise_pilot_architectures",
                "noise_pilot_seeds", "power_pilot_models", "power_pilot_seeds"):
        _integer(settings.get(key), key, 1)
    if settings["selection_origins"] != settings["outer_cv_origins"] + settings["final_selection_origins"]:
        raise ValueError("Selection origins must equal outer CV plus final selection origins")
    if settings["neural_references"] != len(refs):
        raise ValueError("Budget/reference profile counts disagree")
    if (settings["min_final_seeds"] < 6 or settings["max_final_seeds"] < settings["min_final_seeds"]
            or settings["noise_pilot_seeds"] < 5 or settings["power_pilot_seeds"] < 5):
        raise ValueError("Pilots need five seeds; this protocol requires at least six final seeds")
    if not 3 <= settings["noise_pilot_architectures"] <= 1000:
        raise ValueError("Noise pilot needs three anchors and at most 1000 unique architectures")
    if settings["power_pilot_models"] not in (2, 3):
        raise ValueError("The guide's power pilot uses two or three representative models")
    power_names = settings.get("power_pilot_model_names")
    if not isinstance(power_names, list) or len(power_names) != settings["power_pilot_models"]:
        raise ValueError("Name every power-pilot model before counting its cost")
    if any(not isinstance(name, str) for name in power_names):
        raise ValueError("Power-pilot model names must be strings")
    _unique(power_names, "power-pilot model names")
    by_name = {m["model"]: m for m in models + refs}
    if any(name not in by_name or by_name[name].get("auxiliary_phases", 0) for name in power_names):
        raise ValueError("Power pilot requires known non-SAE models, including references when declared")
    expected_profiles = sum(not m["nas"] for m in models + refs)
    if settings.get("profiling_fixed_neural_stages") != expected_profiles:
        raise ValueError("Profile every fixed-body neural model once; NAS timings reuse the noise pilot")
    _integer(settings.get("profiling_classical_fits"), "classical profiling fits", 1)
    for key, required in (("classical_hpo_systems", 2), ("classical_hpo_trials_each", 18),
                          ("classical_refit_systems", 2)):
        if key in settings and settings[key] != required:
            raise ValueError(f"{key} differs from this planner's declared classical search")

    # Every declared optimization pool is separate from the other pools. Pure
    # configuration generation uses its own independent deterministic seed.
    all_seed_values = list(opt_seeds) + [phase1] + list(phase2)
    for key, count_key in (("noise_pilot_seed_values", "noise_pilot_seeds"),
                           ("power_pilot_seed_values", "power_pilot_seeds"),
                           ("final_seeds", "max_final_seeds")):
        if key in settings:
            pool = _domain(settings[key], key)
            if len(pool) != settings[count_key]:
                raise ValueError(f"{key} count differs from its declared budget")
            for seed in pool:
                _integer(seed, key)
            all_seed_values.extend(pool)
    if "profiling_seed" in settings:
        all_seed_values.append(_integer(settings["profiling_seed"], "profiling seed"))
    _unique(all_seed_values, "all declared optimization/pilot/profile/final seed pools")

    if "max_epochs" in common or "phase1_epochs" in nas or "phase2_epochs" in nas:
        full_epochs = _integer(common.get("max_epochs"), "full HPO epoch cap", 1)
        short_epochs = _integer(nas.get("phase1_epochs"), "short NAS epoch cap", 1)
        nas_full_epochs = _integer(nas.get("phase2_epochs"), "full NAS epoch cap", 1)
        if not short_epochs < nas_full_epochs == full_epochs:
            raise ValueError("Short epoch cap must be smaller than the common full epoch cap")
        if "patience_epochs" in common and _integer(common["patience_epochs"], "patience", 1) > short_epochs:
            raise ValueError("Patience cannot exceed the short-stage epoch cap")
    hardware = protocol.get("hardware")
    if hardware is not None:
        short_seconds = _integer(hardware.get("short_stage_wall_limit_seconds"), "short wall cap", 1)
        full_seconds = _integer(hardware.get("full_stage_wall_limit_seconds"), "full wall cap", 1)
        if short_seconds >= full_seconds:
            raise ValueError("Short wall cap must be strictly smaller than full wall cap")
        if "fit_process_memory_limit_bytes" in hardware and "cgroup_memory_bytes" in hardware:
            if not 0 < _integer(hardware["fit_process_memory_limit_bytes"], "fit memory cap", 1) < _integer(hardware["cgroup_memory_bytes"], "cgroup memory", 1):
                raise ValueError("Fit memory cap must leave headroom below the cgroup limit")


def _profile(protocol: dict[str, Any], model: str | dict[str, Any]) -> dict[str, Any]:
    name = model if isinstance(model, str) else model.get("model")
    matches = [m for m in protocol["models"] + protocol["neural_reference_profiles"] if m["model"] == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown model: {name}")
    return matches[0]


def _balanced(values: list[Any], count: int, rng: random.Random) -> list[Any]:
    # Shuffle the category order so the extra occurrence, when needed, is not
    # systematically assigned to the first declared category.
    categories = list(values)
    rng.shuffle(categories)
    result = [categories[i % len(categories)] for i in range(count)]
    rng.shuffle(result)
    return result


def _stratified_log(bounds: list[float], count: int, rng: random.Random) -> list[float]:
    low, high = map(math.log, bounds)
    values = [math.exp(low + (high-low) * ((i+rng.random())/count)) for i in range(count)]
    rng.shuffle(values)
    return values


def generate_hpo(protocol: dict[str, Any], model: str | dict[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    profile = _profile(protocol, model)
    common = protocol["common_hpo"]
    count = common["trials"]
    token = f"{PLANNER_VERSION}:{common['seed']}:{profile['search_recipe_id']}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(token).digest(), "big"))
    rates = _stratified_log(common["learning_rate"], count, rng)
    zeros = common["zero_weight_decay_trials"]
    decays = [0.0] * zeros + _stratified_log(common["weight_decay_positive"], count-zeros, rng)
    rng.shuffle(decays)
    axes = {"batch_size": _balanced(common["batch_sizes"], count, rng)}
    if not profile["nas"]:
        axes["dropout"] = _balanced(profile["hpo"]["dropout_values"], count, rng)
    for key in sorted(profile["hpo"]["extra_axes"]):
        axes[key] = _balanced(profile["hpo"]["extra_axes"][key], count, rng)
    rows = []
    for i in range(count):
        row = {"trial_id": f"{profile['search_recipe_id']}:hpo-{i:03d}",
               "recipe_id": profile["search_recipe_id"], "learning_rate": rates[i],
               "weight_decay": decays[i], "optimization_seeds": list(common["optimization_seeds"])}
        row.update({key: values[i] for key, values in axes.items()})
        if profile["nas"]:
            row["architecture_reference"] = {
                "widths": list(protocol["nas"]["reference"]["widths"]),
                "dropout_pattern": list(protocol["nas"]["reference"]["dropout_pattern"]),
                "dropout_rate": protocol["nas"]["dropout_rate"]}
        rows.append(row)
    return rows


def enumerate_architectures(protocol: dict[str, Any], model: str | dict[str, Any]) -> list[dict[str, Any]]:
    validate_protocol(protocol)
    profile = _profile(protocol, model)
    if not profile["nas"]:
        return []
    rows = []
    for widths in itertools.product(protocol["nas"]["widths"], repeat=3):
        for pattern in itertools.product((0, 1), repeat=3):
            rows.append({
                "architecture_id": f"{profile['search_recipe_id']}:w{'-'.join(map(str, widths))}:b{''.join(map(str, pattern))}",
                "recipe_id": profile["search_recipe_id"], "widths": list(widths),
                "dropout_pattern": list(pattern), "dropout_rate": protocol["nas"]["dropout_rate"],
                "is_reference": list(widths) == [64, 64, 64] and list(pattern) == [0, 1, 0]})
    return rows


def _phases(supervised: int, auxiliary: int) -> dict[str, int]:
    return {"supervised": supervised, "auxiliary": auxiliary, "total": supervised + auxiliary}


def noise_pilot_architectures(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    """Common width/pattern pilot: reference, extremes and stratified random rows."""
    validate_protocol(protocol)
    space = [(w, b) for w in itertools.product(protocol["nas"]["widths"], repeat=3)
             for b in itertools.product((0, 1), repeat=3)]
    anchors = [((64, 64, 64), (0, 1, 0)), ((16, 16, 16), (0, 0, 0)),
               ((256, 256, 256), (1, 1, 1))]
    remaining = [pair for pair in space if pair not in anchors]
    token = f"{PLANNER_VERSION}:{protocol['common_hpo']['seed']}:noise-pilot-patterns".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(token).digest(), "big"))
    n = protocol["budget"]["noise_pilot_architectures"] - len(anchors)
    selected = list(anchors)
    for i in range(n):
        left, right = len(remaining)*i//n, len(remaining)*(i+1)//n
        selected.append(remaining[rng.randrange(left, right)])
    return [{"pattern_id": f"w{'-'.join(map(str, w))}:b{''.join(map(str, b))}",
             "widths": list(w), "dropout_pattern": list(b), "dropout_rate": 0.2,
             "is_reference": (w, b) == anchors[0]}
            for w, b in selected]


def budget(protocol: dict[str, Any], k: int, R: int) -> dict[str, Any]:
    """Worst-case counts; reference is added even if it might be a finalist.

    Pilots/profiles are global F=1, not repeated at every selection origin.
    Diagnostic, robustness, ablation and calibration-refit costs are excluded.
    """
    validate_protocol(protocol)
    if k not in protocol["nas"]["k_options"]:
        raise ValueError("k is not one of the preregistered scenarios")
    b = protocol["budget"]
    _integer(R, "final seeds", b["min_final_seeds"])
    if R > b["max_final_seeds"]:
        raise ValueError("R exceeds the declared resource cap")
    models = protocol["models"] + protocol["neural_reference_profiles"]
    n = len(models)
    nas_models = [m for m in models if m["nas"]]
    n_nas = len(nas_models)
    aux = sum(m.get("auxiliary_phases", 0) for m in models)
    aux_nas = sum(m.get("auxiliary_phases", 0) for m in nas_models)
    f, t, s = b["selection_origins"], protocol["common_hpo"]["trials"], len(protocol["common_hpo"]["optimization_seeds"])
    r2 = len(protocol["nas"]["phase2_seeds"])
    noise_a, noise_s = b["noise_pilot_architectures"], b["noise_pilot_seeds"]
    full = {
        "hpo": _phases(n*t*s*f, aux*t*s*f),
        "nas_phase2_upper_bound": _phases(n_nas*(k+1)*r2*f, aux_nas*(k+1)*r2*f),
        "neural_refits": _phases(n*f*R, aux*f*R),
        "noise_pilot_full_global": _phases(n_nas*noise_a*noise_s, aux_nas*noise_a*noise_s),
        "fixed_neural_resource_profiles_global": _phases(b["profiling_fixed_neural_stages"], 0),
        "power_pilot_non_sae_models_global": _phases(b["power_pilot_models"]*b["power_pilot_seeds"], 0),
    }
    short = {
        "nas_phase1": _phases(n_nas*1000*f, aux_nas*1000*f),
        "noise_pilot_short_global": _phases(n_nas*noise_a, aux_nas*noise_a),
    }
    return {
        "scenario": {"selection_origins": f, "outer_cv_origins": b["outer_cv_origins"],
                     "final_selection_origins": b["final_selection_origins"], "k": k,
                     "final_seeds_R": R, "nas_models": n_nas, "neural_models_with_references": n,
                     "resource_scenario_not_final_seed_decision": True},
        "full_training_phases": full,
        "short_training_phases": short,
        "full_phase_total": sum(v["total"] for v in full.values()),
        "short_phase_total": sum(v["total"] for v in short.values()),
        "classical_adjustments": {"ridge_hpo": 18*f, "arima_hpo": 18*f,
                                  "deterministic_refits": 2*f,
                                  "resource_profiling": b["profiling_classical_fits"],
                                  "total": 36*f + 2*f + b["profiling_classical_fits"],
                                  "seasonal_period_scores_not_fits": 3*f,
                                  "persistence_parameter_fits": 0},
        "scope": "Search, global pilots/reference profiles and refits only. Counts are upper bounds, not observed work.",
        "excluded": ["diagnostic controls", "robustness", "ablations", "calibration-specific refits",
                     "interpretability", "environment setup"],
        "limits": ["Power pilot names declared non-SAE models; no auxiliary cost is hidden.",
                   "Fixed-body profiles are supervised stages; NAS profiling reuses the noise pilot.",
                   "R is a resource scenario and does not establish statistical power.",
                   "Full and short phases have different budgets and must not be added as equivalent fits.",
                   "Hardware time/memory/monetary estimates remain unmeasured.",
                   "All ten outer origins and the final selection origin are charged separately."]}


def make_manifest(protocol: dict[str, Any], k: int, R: int) -> dict[str, Any]:
    validate_protocol(protocol)
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "planner_version": PLANNER_VERSION, "schema_version": SCHEMA_VERSION,
        "protocol_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "execution": {"models_imported": False, "market_data_read": False, "fits_executed": 0},
        "seed_pools": {"configuration_generation": protocol["common_hpo"]["seed"],
            "hpo_optimization": list(protocol["common_hpo"]["optimization_seeds"]),
            "nas_phase1": protocol["nas"]["phase1_seed"],
            "nas_phase2": list(protocol["nas"]["phase2_seeds"]),
            "noise_pilot": protocol["budget"].get("noise_pilot_seed_values"),
            "fixed_body_profiling": protocol["budget"].get("profiling_seed"),
            "power_pilot": protocol["budget"].get("power_pilot_seed_values"),
            "final_resource_scenario": protocol["budget"].get("final_seeds", [])[:R]},
        "hpo": {m["model"]: generate_hpo(protocol, m) for m in protocol["models"] + protocol["neural_reference_profiles"]},
        "nas": {m["model"]: enumerate_architectures(protocol, m) for m in protocol["models"] if m["nas"]},
        "noise_pilot_architecture_patterns": noise_pilot_architectures(protocol),
        "classical_domains": {
            "ridge_alpha": [10 ** (-6 + 9*i/17) for i in range(18)],
            "arima_orders": [list(order) for order in itertools.product(range(3), range(2), range(3))],
            "seasonal_naive_periods": [7, 14, 28], "persistence_lag": 1},
        "budget": budget(protocol, k, R),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New JSON manifest file; existing files are refused")
    parser.add_argument("--k", type=int)
    parser.add_argument("--final-seeds", type=int)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    k = args.k if args.k is not None else protocol["nas"]["k_options"][0]
    r = args.final_seeds if args.final_seeds is not None else protocol["budget"]["max_final_seeds"]
    if args.output is not None and args.output.exists():
        raise FileExistsError("Refusing to replace an existing plan manifest")
    manifest = make_manifest(protocol, k, r)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"planner_version": PLANNER_VERSION, "fits_executed": 0,
                      "output": str(args.output) if args.output else None,
                      "budget": manifest["budget"]}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

