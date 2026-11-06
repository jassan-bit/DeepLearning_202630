"""Memorize a tiny synthetic batch before any final E2 market-data training.

This is a software sanity check, not a Bitcoin pilot, HPO candidate or accuracy
result. Its synthetic optimization recipe and tolerances are fixed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from e2_final_models import MLP_CONV, ATTENTION_SSM, build_model, optimizer_groups


SEED = 202611067
LEARNING_RATES = (.003, .001, .01)
MAXIMUM_STEPS = 120
LOSS_RATIO_LIMIT = .05
ABSOLUTE_MSE_LIMIT = .001


def check_one(name, sources):
    trials = []
    for rate in LEARNING_RATES:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        model = build_model(name, candidate={"dropout": 0.}, source_roots=sources)
        generator = torch.Generator().manual_seed(SEED+1)
        x = torch.randn(4, 7, 5, generator=generator)
        y = torch.tensor([.35, .8, 1.35, 1.9], dtype=torch.float32)
        groups, _ = optimizer_groups(model, 0.)
        optimizer = torch.optim.AdamW(groups, lr=rate)
        model.train()
        start = time.perf_counter()
        with torch.no_grad():
            initial = float((model(x)-y).square().mean())
        losses = []; updates = 0
        for step in range(1, MAXIMUM_STEPS+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            loss = (model(x)-y).square().mean()
            if not torch.isfinite(loss):
                break
            losses.append(float(loss.detach()))
            # Check the weights that actually attained the criterion. Taking
            # one more Adam step first can overshoot a memorized toy batch.
            if step >= 10 and losses[-1] <= min(ABSOLUTE_MSE_LIMIT, initial*LOSS_RATIO_LIMIT):
                break
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            updates += 1
        model.train()
        with torch.no_grad():
            final_train = float((model(x)-y).square().mean())
            model.eval(); final_eval = float((model(x)-y).square().mean())
        passed = (np.isfinite(final_train) and final_train <= ABSOLUTE_MSE_LIMIT
                  and final_train <= initial*LOSS_RATIO_LIMIT)
        trials.append({"learning_rate": rate, "initial_training_MSE": initial,
                       "final_training_MSE": final_train, "final_eval_MSE": final_eval,
                       "steps": updates, "loss_checks": len(losses), "elapsed_seconds": time.perf_counter()-start,
                       "passed": bool(passed)})
        if passed:
            return {"model": name, "status": "passed", "trials": trials,
                    "criterion": "Memorize four distinct targets in training mode; eval loss is separately disclosed, including BatchNorm running-statistic limitations"}
    return {"model": name, "status": "failed", "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh preflight receipt; retain failures rather than overwriting them")
    sources = json.loads(args.source_roots_json.read_text())
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    results = []
    for name in (*MLP_CONV, *ATTENTION_SSM):
        result = check_one(name, sources); results.append(result)
        print(json.dumps({"model": name, "status": result["status"],
                          "synthetic_steps": sum(t["steps"] for t in result["trials"])}), flush=True)
    status = "passed" if all(r["status"] == "passed" for r in results) else "failed"
    receipt = {"status": status, "at_utc": datetime.now(timezone.utc).isoformat(),
               "generator": Path(__file__).name,
               "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
               "models": results, "toy_seed": SEED, "batch_size": 4,
               "learning_rates_tried_in_fixed_order": list(LEARNING_RATES),
               "maximum_steps_per_toy_attempt": MAXIMUM_STEPS,
               "absolute_training_MSE_limit": ABSOLUTE_MSE_LIMIT,
               "training_MSE_ratio_limit": LOSS_RATIO_LIMIT,
               "threads": 1, "market_data_read": False, "market_model_fits": 0,
               "scope": "Implementation memorization sanity only; no generalization, HPO or power evidence"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
    raise SystemExit(status != "passed")


if __name__ == "__main__":
    main()
