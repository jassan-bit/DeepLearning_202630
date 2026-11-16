"""Exploratory initialization-only comparison; never fits or changes acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "book_e2/_static/e2_closure_followup_20261116/audit/initialization"
FOLLOWUP = PUBLIC.parent
ORIGINAL = ROOT / "book_e2/_static/e2_scientific_closure_20261116/controls"
MODELS = ("TSMixer", "iTransformer")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError("Immutable evidence differs: " + str(path))
        return
    path.write_bytes(raw)


def prepare(args):
    import e2_corrections_controls as engine
    paths = [Path(__file__), ROOT / "scripts/e2_negative_followup.py", FOLLOWUP / "plan.json",
             FOLLOWUP / "summary.json", ORIGINAL / "negative_summary.json", ORIGINAL / "specification.json",
             ROOT / "scripts/e2_corrections_controls.py", ROOT / "scripts/e2_reduced_fit.py",
             ROOT / "scripts/e2_final_models_attention.py", ROOT / "scripts/e2_final_models_mlp.py"]
    initial = {}
    for model in MODELS:
        record_path = ORIGINAL / ("negative_" + str(engine.MODELS.index(model)+1).zfill(2) + ".json")
        paths.append(record_path)
        record = json.loads(record_path.read_text())
        initial[model] = record["phase_records"]["supervised"]["initial_state_sha256"]
    plan = {"version": "e2-negative-initialization-v1", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "frozen_before_initialization_forwards", "posterior_exploratory": True, "models": list(MODELS),
            "initialization_seed": 271801, "expected_initial_state_sha256": initial,
            "bindings_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "cohort": "same608 real test windows with original training-only scaling",
            "procedure": "Construct each frozen body/recipe using original fixed seed, verify exact initial state hash, no optimizer, then eval forward once per test window",
            "reported_endpoints": ["RMSE", "RMSE gain versus true-training-mean constant", "Spearman rho", "descriptive comparison with original single and fixed new three label-permuted fits"],
            "hypothesis_tests": False, "new_margin_or_null": False, "new_criterion_for_original_audit": False,
            "interpretation": "Association before supervised fitting may motivate an inductive-bias explanation; this two-function comparison does not establish a causal mechanism or remove a leakage concern by itself",
            "new_fits": 0, "original_primary_modified": False, "budget_cap_seconds": 60}
    write(args.plan, plan)
    print(json.dumps({"event": "initialization_diagnostic_prepared", "sha256": sha(args.plan), "new_fits": 0}), flush=True)


def run(args):
    import e2_corrections_budget as budget
    with budget.locked_ledger(args.ledger) as (rows, _):
        matching = [r for r in budget.summarize(rows)["open_reservations"].values() if r["phase"] == args.phase]
    if len(matching) != 1:
        raise RuntimeError("One matching scientific reservation required")
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    import e2_corrections_controls as engine
    import e2_staged_training as historical
    from e2_reduced_fit import _builder
    plan = json.loads(args.plan.read_text())
    for relative, expected in plan["bindings_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Frozen input/source differs: " + relative)
    spec, data = engine.admitted(ORIGINAL / "specification.json")
    followup = json.loads((FOLLOWUP / "summary.json").read_text())
    old = json.loads((ORIGINAL / "negative_summary.json").read_text())
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    truth = data["test"]["y"]
    reference_rmse = float(np.sqrt(np.mean((truth-data["target_scale"])**2)))
    rows = []
    for name in plan["models"]:
        started = time.monotonic()
        trial = spec["negative"]["recipes"][name]["trial"]
        model = _builder(name, trial, plan["initialization_seed"], json.loads(engine.SOURCE_ROOTS.read_text()))
        model.to("cpu")
        before = historical.state_sha256(model)
        if before != plan["expected_initial_state_sha256"][name]:
            raise ValueError("Exact original initial state not recovered: " + name)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for row in data["test"]["X"]:
                value = model(torch.tensor(row[None], dtype=torch.float32)).reshape(-1)
                if len(value) != 1 or not bool(torch.isfinite(value).all()):
                    raise ValueError("Malformed/nonfinite initialization forward")
                predictions.append(float(value.item()) * data["target_scale"])
        after = historical.state_sha256(model)
        if before != after:
            raise ValueError("State/buffers changed during initialization-only evaluation")
        prediction = np.asarray(predictions)
        rmse = float(np.sqrt(np.mean((prediction-truth)**2)))
        rho = float(spearmanr(prediction, truth).statistic) if prediction.std() else None
        destination = PUBLIC / (name + "_predictions.npz")
        if destination.exists():
            raise FileExistsError("Prior initialization payload exists")
        np.savez_compressed(destination, prediction=prediction, truth=truth, dates=data["test"]["dates"].astype(str))
        original = next(r for r in old["rows"] if r["model"] == name)
        rows.append({"model": name, "initialization_seed": plan["initialization_seed"], "initial_state_sha256": before,
                     "final_state_sha256": after, "new_fits": 0, "RMSE": rmse,
                     "gain_RMSE_vs_training_mean": reference_rmse-rmse, "Spearman_rho": rho,
                     "original_single_permutation": {"gain_RMSE_vs_training_mean": original["gain_RMSE_vs_training_mean"], "Spearman_rho": original["Spearman_rho"]},
                     "new_three_permutations": [r for r in followup["rows"] if r["model"] == name],
                     "predictions_path": str(destination.relative_to(ROOT)), "predictions_sha256": sha(destination),
                     "elapsed_seconds": time.monotonic()-started})
    write(PUBLIC / "result.json", {"plan_sha256": sha(args.plan), "rows": rows, "new_fits": 0,
                                  "original_audit_acceptance_changed": False, "original_primary_modified": False,
                                  "interpretation": plan["interpretation"], "no_hypothesis_tests_or_new_null": True})
    print(json.dumps({"event": "initialization_diagnostic_completed", "models": len(rows), "new_fits": 0}), flush=True)


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
