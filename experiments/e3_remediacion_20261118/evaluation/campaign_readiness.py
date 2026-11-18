"""Reproduce the historical training-equity and local entry-point audit, without fits.

This inspects bytes and ASTs; a parsed constructor is not an executed training
recipe. Missing paths describe this sparse checkout, not the complete remote
repository. The verified historical plans remain unchanged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "experiments/e3_estadistica_posterior_20261117/inputs.py"
E3 = "CNN-LSTM-atencion-aditiva E3"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_audit(relative, entrypoint):
    path = ROOT / relative
    result = {"path": relative, "required_entrypoint": entrypoint,
              "exists_in_this_checkout": path.is_file(), "executed": False}
    if path.is_file():
        tree = ast.parse(path.read_text(), filename=relative)
        definitions = [x.name for x in tree.body if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
        result.update(sha256=sha(path), syntax_valid=True, entrypoint_defined=entrypoint in definitions)
        if entrypoint not in definitions:
            raise ValueError("Expected entrypoint changed: " + relative)
    return result


def build(folder):
    spec = importlib.util.spec_from_file_location("frozen_inputs_readiness", FROZEN)
    inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inputs)
    verified = inputs.load_inputs(folder)
    old = json.loads((folder / "e2_plan.json").read_text())
    candidate = json.loads((folder / "e3_plan.json").read_text())
    old_summary = json.loads((folder / "e2_summary.json").read_text())
    candidate_summary = json.loads((folder / "e3_summary.json").read_text())
    models = old["models"]
    classics = old["deterministic_references"]
    if len(models) != 13 or classics != ["Persistence", "SeasonalNaive", "ARIMA"] or set([E3] + models + classics) != set(verified["model_order"]):
        raise ValueError("The exact seventeen-system family is required; Ridge is excluded")
    if old_summary["equal_training_budget_claimed"] or candidate_summary["equal_training_budget_claimed"]:
        raise ValueError("Frozen claims changed; audit the source before proceeding")
    source_specs = {
        "registry": ("scripts/e2_published_registry.py", "build_train_model"),
        "historical_builder": ("scripts/e2_training_models.py", "build_train_model"),
        "CNN_bodies": ("scripts/e2_recipe_models.py", "build_model"),
        "SAE": ("scripts/e2_published_sae.py", "build_sae"),
        "periodic": ("scripts/e2_published_periodic.py", "build_periodic"),
        "PatchTST": ("scripts/e2_published_patchtst.py", "build_patchtst"),
        "classical": ("scripts/e2_required_classical_training.py", "select_classical"),
        "fixed_CPU_replay": ("scripts/e2_cpu_fixed_recipe_reproduction.py", "timing_projection"),
        "fixed_CUDA_replay": ("scripts/e2_reproduce_from_raw.py", "prepare"),
        "historical_E3_additive": ("src/e3_innovation_model.py", "InnovationModel"),
        "distinct_development_candidate": ("src/original_model.py", "OriginalModel"),
    }
    source_rows = {key: source_audit(*value) for key, value in source_specs.items()}
    metadata_paths = [
        "experiments/e2_published_architectures_20261112/architecture_policy.json",
        "book_e2/_static/search_protocol_20261109/protocol.json",
        "book_e2/_static/current_implementation_20261108/source_design.json",
        "book_e2/_static/e2_corrections_20261115/hpo/sources/selections",
    ]
    dependencies = [{"path": p, "exists_in_this_checkout": (ROOT / p).exists(),
                     "scope": "local_sparse_checkout_presence_not_remote_availability"} for p in metadata_paths]
    recipes = []
    for name in models:
        jobs = [j for j in old["jobs"] if j["model"] == name]
        if len(jobs) != 4 or {j["seed"] for j in jobs} != set(range(271807, 271811)):
            raise ValueError("Historical four-seed extension differs: " + name)
        recipe = jobs[0]
        body = "SAE" if name == "LSTM-autoencoder con predictor" else "periodic" if name in ("TimesNet", "TimeMixer") else "PatchTST" if name == "PatchTST" else "historical_builder"
        recipes.append({"model": name, "role": "neural_rival", "registry": "scripts/e2_published_registry.py:build_train_model",
            "body_source_key": body, "historical_parameters": recipe["parameter_count"],
            "historical_fixed_phase_epochs": recipe["fixed_epochs"], "historical_trial": recipe["trial"],
            "new_fits": 0, "future_common_training_executed": False,
            "future_gate": "exact recipe/author-source/parameter admission + common split/runtime/HPO/fit ledger; no import or fit executed here"})
    for name in classics:
        recipes.append({"model": name, "role": "deterministic_rival", "body_source_key": "classical",
            "saved_trajectories": 1, "new_fits": 0, "future_common_training_executed": False,
            "future_gate": "select only on shared development data; forecast before observing target; retain every failure; no artificial seed replication"})
    recipes.append({"model": E3, "role": "frozen_historical_candidate", "body_source_key": "historical_E3_additive",
        "variant": candidate["architecture"]["variant"], "historical_parameters": candidate["architecture"]["trainable_parameters"],
        "new_fits": 0, "future_common_training_executed": False,
        "future_gate": "decide additive historical candidate versus distinct OriginalModel before protocol freeze; separate identities, properties and provenance"})
    return {"schema_version": 1, "status": "audit_not_new_training_or_preregistration", "date": "2026-11-18",
        "source_review_commit": "73d8958c50a69fbea46deb828f03f89385318789", "input_provenance": verified["provenance"],
        "audit_script_sha256": sha(Path(__file__)), "source_entrypoints": source_rows,
        "local_metadata_dependencies": dependencies, "recipes": recipes,
        "historical_equity": {"same_test_calendar": old["data"]["evaluation_start"] == candidate["data"]["evaluation_start"] and old["data"]["evaluation_end"] == candidate["data"]["evaluation_end"],
            "E2_training_examples": old["data"]["training_examples"], "E2_training_end": old["data"]["training_end"],
            "E2_calibration_examples": old["data"]["calibration_examples"],
            "E3_fit_examples": candidate["data"]["fit_count"], "E3_fit_end": candidate["data"]["fit_end"],
            "E3_validation_examples": candidate["data"]["validation_count"], "E3_calibration_allowed": candidate["data"]["calibration_members_allowed"],
            "E2_extension_budget": old["budget"], "E3_budget": candidate["budget"],
            "equal_training_budget_claimed": False, "common_training_or_calibration_treatment_verified": False,
            "ten_seed_extension_is_not_equal_training_or_power": True},
        "unresolved_gates": [
            "Choose the exact candidate identity and falsifiable property; additive 1210-parameter forecasts cannot inherit a distinct architecture's theorem or novelty evidence.",
            "Admit exactly three additional metadata-selected datasets with source rights, immutable hashes, causal OHLCV checks and task calendars; none are admitted by this audit.",
            "Bind identical development/calibration/evaluation information sets, target, window, purge and train-only scaling policies for all seventeen systems; architecture-specific preprocessing must be explicit.",
            "Choose fixed-recipe external validation versus a newly budgeted tuning campaign in advance. Fixed-recipe repetition is not reproduction of historical HPO; new HPO must share compute access, total worker-time accounting and search opportunity including candidate bias matching and TiDE NAS.",
            "Select classical orders and seasonal lag using only the shared inner development data; deterministic comparators have one causal trajectory, not ten stochastic fits.",
            "Admit exact source trees, dependencies and per-architecture timing receipts on the selected CPU/GPU; do not infer thirteen-model feasibility from one CNN. Count failed/aborted jobs and never reset old ledgers.",
            "Run a genuinely separate development pilot to justify future variance/alternatives/margins, all three procedures and joint power; seal the full protocol before confirmatory outputs.",
            "Freeze every checkpoint/scaler/calibrator and signed admission manifest before admitting new evaluation labels; keep an exposure log because the historical calendar is already known.",
            "Predefine robustness, recalibration/uncertainty and subgroup analyses on suitable non-test data and budgets, then run the frozen analyses; the current audit performs none of them.",
        ], "new_model_fits": 0, "new_model_inferences": 0, "datasets_admitted": 0,
        "dynamic_recipe_execution_verified": False, "future_campaign_ready": False, "R3_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.data)
    content = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != content:
        raise ValueError("Preserve different output and choose a new destination")
    args.output.write_bytes(content)
    print(json.dumps({"recipes": len(result["recipes"]), "source_entrypoints_parsed": len(result["source_entrypoints"]),
        "new_model_fits": 0, "future_campaign_ready": False, "output": str(args.output)}))


if __name__ == "__main__":
    main()
