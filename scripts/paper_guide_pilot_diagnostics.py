#!/usr/bin/env python3
"""Private validation diagnostics for the learned external ETTh1 pilot.

Each invocation handles one fixed model/seed to avoid cross-repository module
caches. Queries use only original validation windows; context-bank windows are
entirely within training. These are implementation/regime probes, not Bitcoin
benchmark results or an original E3 model.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def input_variability(dataset):
    """Train-standardized RMS first difference of the observed L96×7 history."""
    values = np.asarray(dataset.values, dtype=np.float64)
    # A supervised window's input starts at its dataset row index. Do not add
    # the otherwise possible final histories whose future target is incomplete.
    per_hour = np.square(np.diff(values, axis=0)).mean(axis=1)
    sums = np.concatenate(([0.0], np.cumsum(per_hour)))
    starts = np.arange(len(dataset))
    return np.sqrt((sums[starts + dataset.seq_len - 1] - sums[starts]) / (dataset.seq_len - 1))


def batch_tensors(dataset, indices, device="cpu", omit_marks=False):
    import torch
    samples = [dataset[int(i)] for i in indices]
    x = torch.from_numpy(np.stack([a[0] for a in samples])).to(device)
    marks = None if omit_marks else torch.from_numpy(np.stack([a[2] for a in samples])).to(device)
    y = np.stack([a[1][-dataset.pred_len:] for a in samples])
    future = None if omit_marks else torch.from_numpy(np.stack([a[3] for a in samples])).to(device)
    return x, marks, future, y


def forward(model, data):
    import torch
    x, marks, future, _ = data
    with torch.no_grad():
        output = model(x, marks, None, future)
    if isinstance(output, tuple): output = output[0]
    return output[:, -96:, :].detach().cpu().numpy()


def evaluate_pilot(model, train, validation, queries=512, bank_size=127, batch_size=128,
                   frozen_validation_predictions=None):
    """Return private per-origin diagnostics with a common, fixed calendar."""
    import torch
    if batch_size != 128 or bank_size != 127 or queries != 512:
        raise ValueError("Registered diagnostic protocol is B128/127 bank/512 queries")
    if len(validation) != 2785 or len(train) != 8449:
        raise ValueError("Diagnostic calendar must be the original L96/H96 ETTh1 split")
    model.eval()
    query_indices = np.linspace(0, len(validation)-1, queries, dtype=np.int64)
    bank_indices = np.linspace(0, len(train)-1, bank_size, dtype=np.int64)
    if int(train.forecast_origins[-1] + train.pred_len) > int(validation.forecast_origins[0]):
        raise ValueError("Context bank contains an origin/target beyond the permitted training boundary")
    train_variability = input_variability(train)
    threshold = float(np.quantile(train_variability, .75))
    variability = input_variability(validation)
    high = variability > threshold
    err = np.empty(len(validation), dtype=np.float64)
    # This is the registered pilot inference convention, not the original
    # replication's drop_last test cohort. Keep every validation origin.
    if frozen_validation_predictions is not None:
        pred, target, origins = frozen_validation_predictions
        if (pred.shape != (len(validation),96,7) or target.shape != pred.shape or
                not np.array_equal(origins, validation.forecast_origins)):
            raise ValueError("Frozen pilot validation predictions/calendar differ")
        err[:] = np.square(pred.astype(np.float64)-target).mean(axis=(1,2))
    else:
        for start in range(0, len(validation), batch_size):
            idx = np.arange(start, min(start + batch_size, len(validation)))
            data = batch_tensors(validation, idx)
            pred = forward(model, data)
            if pred.shape != data[3].shape: raise ValueError("Pilot model output/target shape mismatch")
            err[idx] = np.square(pred.astype(np.float64)-data[3]).mean(axis=(1,2))
    bank = batch_tensors(train, bank_indices)
    deltas, no_marks, mse_b1, mse_bank = [], [], [], []
    for q in query_indices:
        query = batch_tensors(validation, [q])
        combined = (torch.cat([query[0], bank[0]], dim=0),
                    torch.cat([query[1], bank[1]], dim=0),
                    torch.cat([query[2], bank[2]], dim=0), None)
        alone = forward(model, query)[0]
        grouped = forward(model, combined)[0]
        no_calendar_query = (query[0], None, None, query[3])
        no_calendar_grouped = (combined[0], None, None, None)
        alone_nc = forward(model, no_calendar_query)[0]
        grouped_nc = forward(model, no_calendar_grouped)[0]
        deltas.append(np.abs(alone.astype(np.float64)-grouped).mean())
        no_marks.append(np.abs(alone_nc.astype(np.float64)-grouped_nc).mean())
        mse_b1.append(np.square(alone.astype(np.float64)-query[3][0]).mean())
        mse_bank.append(np.square(grouped.astype(np.float64)-query[3][0]).mean())
    return {"query_indices": query_indices, "bank_train_indices": bank_indices,
            "query_forecast_origins": validation.forecast_origins[query_indices],
            "bank_train_forecast_origins": train.forecast_origins[bank_indices],
            "batch_delta": np.asarray(deltas), "no_marks_delta": np.asarray(no_marks),
            "query_MSE_B1": np.asarray(mse_b1), "query_MSE_bank": np.asarray(mse_bank),
            "val_squared_error_by_origin": err, "variability": variability,
            "high_threshold": np.asarray(threshold), "high_mask": high,
            "val_forecast_origins": validation.forecast_origins}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",required=True);parser.add_argument("--data",required=True)
    parser.add_argument("--source-root",required=True);parser.add_argument("--control-source-root")
    parser.add_argument("--run-dir",required=True);parser.add_argument("--model",choices=["TimeMixer","iTransformer","PatchTST"],required=True)
    parser.add_argument("--seed",type=int,required=True);parser.add_argument("--output-dir",required=True)
    args=parser.parse_args(argv)
    if args.seed not in [2019,2020,2021,2022,2023]: raise SystemExit("Seed is outside the registered pilot")
    design=json.loads(Path(args.design).read_text())
    from analyze_e2_paper_guide import validate_analysis_design
    validate_analysis_design(design)
    source=args.source_root if args.model=="TimeMixer" else args.control_source_root
    if not source: raise SystemExit("A pinned source root is required for the selected model")
    import torch
    from e2_paper_data import load_etth1,fit_scaler,make_split
    from run_e2_paper_replication import load_fitted_model,registration_payload,verify_registration
    registered_inputs=registration_payload(Path(args.design),design,Path(args.data),Path(args.source_root),
                                          Path(args.control_source_root) if args.control_source_root else None)
    verify_registration(Path(args.run_dir).resolve().parents[2],registered_inputs)
    model,setup=load_fitted_model(Path(args.run_dir),Path(source))
    torch.set_num_threads(int(setup["training"]["threads"]))
    if (setup.get("phase")!="pilot" or int(setup.get("seed",-1))!=args.seed or
            setup.get("model_name")!=args.model):
        raise SystemExit("Only the completed registered pilot checkpoint/seed is admissible")
    raw=load_etth1(args.data,design["dataset"]["sha256"])
    # Use the exact saved train-only scaler, not a new fit on validation.
    scaler=setup.get("scaler") or json.loads((Path(args.run_dir)/"scaler.json").read_text())
    if scaler != fit_scaler(raw["values"]): raise SystemExit("Pilot scaler differs from the original training-only prefix")
    model_args=setup["model_args"]
    train=make_split(raw["values"],raw["dates"],"train",seq_len=96,pred_len=96,label_len=int(model_args.get("label_len",0)),scaler=scaler)
    validation=make_split(raw["values"],raw["dates"],"val",seq_len=96,pred_len=96,label_len=int(model_args.get("label_len",0)),scaler=scaler)
    completed=json.loads((Path(args.run_dir)/"completed.json").read_text())
    frozen_path=Path(args.run_dir)/"validation.npz"
    if digest(frozen_path)!=completed.get("validation_predictions_sha256"):
        raise SystemExit("Frozen pilot validation prediction hash differs")
    with np.load(frozen_path,allow_pickle=False) as z:
        frozen=(z["predictions"],z["targets"],z["origins"])
    result=evaluate_pilot(model,train,validation,frozen_validation_predictions=frozen)
    out=Path(args.output_dir)/args.model/f"seed_{args.seed}";out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"diagnostics.npz",**result)
    receipt={"model":args.model,"seed":args.seed,"phase":"learned_validation_pilot_diagnostics",
             "design_sha256":digest(args.design),"generator_sha256":digest(__file__),
             "checkpoint_sha256":digest(Path(args.run_dir)/"checkpoint.pth"),
             "source_dataset_sha256":raw["source_sha256"],"diagnostics_sha256":digest(out/"diagnostics.npz"),
             "nominal_validation_origins":len(validation),"queried_origins":512,"bank_training_origins":127,
             "high_variability_threshold_train_q75":float(result["high_threshold"]),
             "bank_never_contains_future_prices_relative_to_validation_queries":True,
             "no_test_targets_scored":True,"new_Bitcoin_results":0,
             "calendar_disabled_probe_uses_same_weights_no_retraining":True,
             "calendar_disabled_probe_not_a_claim_of_forecast_improvement":True}
    (out/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"model":args.model,"seed":args.seed,"validation_diagnostics_completed":True,"test_scores":0,"new_Bitcoin_results":0}))


if __name__=="__main__":main()
