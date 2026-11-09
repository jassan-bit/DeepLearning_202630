"""Exhaustive synthetic conformance of NAS configurations, not a model search.

No market inputs or optimizers are accepted. Numeric runs test every width/bit
configuration under ONE registered HPO row. They do not evaluate the complete
Cartesian product of all HPO rows and architectures. Meta runs retain defaults
and check geometry/autograd, never numerical values.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
from e2_nas_factory import MODEL_IDS, build_nas_model, build_from_plan
import e2_search_plan as planner


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conformance_trial(protocol, name):
    rows = planner.generate_hpo(protocol, name)
    # num_kernels2 is an explicit registered HPO option, not a silently resized
    # backbone. Full six-kernel extremes are checked separately and reported.
    if name == "TimesNet":
        rows = [row for row in rows if row["num_kernels"] == 2]
    return rows[0]


def evaluate_case(name, architecture, *, protocol, mode, fixture, trial=None):
    widths, bits = architecture["widths"], architecture["dropout_pattern"]
    start = time.perf_counter()
    torch.manual_seed(281109)
    if mode == "meta":
        with torch.device("meta"):
            model = build_nas_model(name, widths, bits)
            x = torch.empty(fixture.shape, device="meta", requires_grad=True)
    elif mode == "numeric":
        model = build_from_plan(protocol, name, architecture, trial)
        x = fixture.detach().clone().requires_grad_(True)
    else:
        raise ValueError("Explicit meta or numeric conformance mode required")
    model.train()
    prediction = model(x)
    if prediction.shape != (len(x),):
        raise AssertionError("Incorrect scalar prediction geometry")
    if mode == "numeric" and (not bool(torch.isfinite(prediction).all())
                              or not bool((prediction > 0).all())):
        raise AssertionError("Nonfinite or nonpositive synthetic prediction")
    prediction.sum().backward()
    if x.grad is None or x.grad.shape != x.shape:
        raise AssertionError("Input autograd connection missing")
    if mode == "numeric" and not bool(torch.isfinite(x.grad).all()):
        raise AssertionError("Nonfinite input gradient")
    count, connected = 0, 0
    for label, parameter in model.named_parameters():
        count += parameter.numel()
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or parameter.grad.shape != parameter.shape:
            raise AssertionError("Missing gradient: "+label)
        connected += parameter.numel()
        if mode == "numeric" and not bool(torch.isfinite(parameter.grad).all()):
            raise AssertionError("Nonfinite parameter gradient: "+label)
    row = {"architecture_id": architecture["architecture_id"], "widths": widths,
           "dropout_pattern": bits, "mode": mode,
           "parameter_count": count, "connected_trainable_parameters": connected,
           "output_shape": list(prediction.shape), "input_gradient_shape": list(x.grad.shape),
           "all_parameter_gradients_connected": True,
           "numerical_finiteness_checked": mode == "numeric",
           "registered_hpo_trial_id": trial["trial_id"] if trial else None,
           "structure_options": model.architecture_metadata["structure_hpo_options"],
           "seconds": time.perf_counter()-start}
    del model, x, prediction
    return row


def run(protocol_path, names, mode, output, *, threads=1):
    if output.exists():
        raise FileExistsError("Conformance output directory must be new")
    protocol = json.loads(protocol_path.read_text())
    planner.validate_protocol(protocol)
    if any(name not in MODEL_IDS for name in names) or len(set(names)) != len(names):
        raise ValueError("Choose unique declared NAS families")
    torch.set_num_threads(threads)
    fixture = torch.randn((1, 7, 5), generator=torch.Generator().manual_seed(281108))
    output.mkdir(parents=True)
    source = Path(__file__).resolve().parent
    files = list(source.glob("e2_nas_*.py")) + [source/"e2_search_plan.py", Path(__file__)]
    source_hashes = {p.name: _sha(p) for p in sorted(files)}
    records, failures = {}, []
    ledger_path = output/"cases.jsonl"
    with ledger_path.open("x") as ledger:
        try:
            for name in names:
                trial = conformance_trial(protocol, name) if mode == "numeric" else None
                architectures = planner.enumerate_architectures(protocol, name)
                counts = []
                for i, architecture in enumerate(architectures):
                    row = evaluate_case(name, architecture, protocol=protocol,
                                        mode=mode, fixture=fixture, trial=trial)
                    ledger.write(json.dumps({"model": name, **row}, allow_nan=False)+"\n")
                    ledger.flush()
                    counts.append(row["parameter_count"])
                    if (i+1)%100 == 0:
                        print(json.dumps({"model": name, "mode": mode, "checked": i+1}), flush=True)
                records[name] = {"checked": len(counts), "expected": 1000,
                                 "parameter_count_min": min(counts),
                                 "parameter_count_max": max(counts),
                                 "registered_hpo_trial": trial,
                                 "numerical_finiteness_checked": mode == "numeric"}
                gc.collect()
        except Exception as error:
            failures.append({"type": type(error).__name__, "message": str(error)})
    summary = {"version": "e2-nas-conformance-v1", "mode": mode,
               "status": "complete" if not failures and len(records)==len(names) else "failed",
               "synthetic_only": True, "market_data_read": False,
               "optimizer_steps": 0, "hpo_search_executed": False, "nas_search_executed": False,
               "fixture": "one fixed artificial [1,7,5] window; training mode for dropout/normalization",
               "fixture_sha256": hashlib.sha256(fixture.numpy().tobytes()).hexdigest(),
               "runtime": {"torch": torch.__version__, "python": platform.python_version(),
                           "device": "meta" if mode=="meta" else "cpu", "threads": threads},
               "protocol_file_sha256": _sha(protocol_path), "source_sha256": source_hashes,
               "models": records, "failures": failures, "ledger_sha256": _sha(ledger_path),
               "limits": ["Conformance is not optimization, predictive performance or power.",
                          "Numeric coverage uses one registered HPO row per family.",
                          "Meta checks do not establish finiteness or data-dependent numerical correctness.",
                          "Not the Cartesian product of1000 architectures and18 HPO rows."]}
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    if failures:
        raise RuntimeError(json.dumps(failures))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("meta", "numeric"))
    parser.add_argument("--model", action="append", choices=list(MODEL_IDS))
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads not in (1, 2, 4):
        parser.error("Use1,2 or4 CPU threads")
    summary = run(args.protocol, args.model or list(MODEL_IDS), args.mode, args.output,
                  threads=args.threads)
    print(json.dumps({"status": summary["status"], "models": list(summary["models"]),
                      "optimizer_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()

