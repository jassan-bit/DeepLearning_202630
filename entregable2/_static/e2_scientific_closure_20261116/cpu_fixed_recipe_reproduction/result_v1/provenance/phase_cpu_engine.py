"""Post-benchmark bounded controls; never changes the frozen primary analysis.

Real fits require an audited cohort, an immutable prepared specification and an
open reservation in e2_corrections_budget.py. CPU neural diagnostics use a new
explicit backend, not the production CUDA entrypoint or a synthetic-test flag.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "book_e2/_static/e2_corrections_20261115/controls"
WORK = ROOT / "work/e2-corrections-controls"
ARCHITECTURES = ROOT / "book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json"
REGISTER = ROOT / "book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv"
SOURCE_ROOTS = Path("/workspace/work/e2architecture_sources/source_roots.json")
ENGINE = "1e8745497b484518f766b6da5c19be8362d9add7"
SEED = 271801
DELTA = .00025772842664981106
MODELS = ["FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32"]
SAE = MODELS[3]
VERSION = "e2-corrections-controls-v1"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError("Existing control artifact differs: " + str(path))
        return
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require_budget(phase):
    import e2_corrections_budget as budget
    with budget.locked_ledger(budget.LEDGER) as (records, _):
        state = budget.summarize(records)
    matching = [row for row in state["open_reservations"].values() if row["phase"] == phase]
    if len(matching) != 1:
        raise RuntimeError("Run this scientific command inside the one matching budget reservation: " + phase)
    return matching[0]


def array_hash(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode() + canonical(list(array.shape)) + array.tobytes()).hexdigest()


def choose_key(arrays, choices):
    found = [key for key in choices if key in arrays]
    if not found:
        raise ValueError("Missing array; expected one of " + repr(choices))
    return found[0]


def load_cohorts(path, key_map=None):
    import numpy as np
    result, mapping = {}, {}
    with np.load(path, allow_pickle=False) as source:
        for stage, aliases, count in (("train", ("train", "training"), 655), ("cal", ("cal", "calibration"), 53),
                                      ("test", ("test", "evaluation"), 608)):
            row, keys = {}, {}
            patterns = {"X": ("X", "X_scaled", "x_scaled"),
                        "relative": ("relative", "X_relative", "relative_X", "relative_window"),
                        "y": ("y_original", "y", "target_original"),
                        "y_scaled": ("y_scaled", "target_scaled"), "dates": ("dates", "ids"),
                        "raw": ("raw", "raw_windows", "X_raw")}
            for field, endings in patterns.items():
                explicit = next(((key_map or {}).get(alias, {}).get(ending)
                                 for alias in aliases for ending in (field,) + endings
                                 if (key_map or {}).get(alias, {}).get(ending)), None)
                key = explicit or choose_key(source, [prefix + "_" + end for prefix in aliases for end in endings])
                if key not in source:
                    raise ValueError("Missing array declared by auditor: " + key)
                row[field] = source[key].copy(); keys[field] = key
            row["dates"] = row["dates"].astype("datetime64[D]")
            if row["X"].shape != (count, 7, 5) or row["relative"].shape != (count, 7, 5) or row["raw"].shape != (count, 7, 5):
                raise ValueError("Incorrect cohort geometry: " + stage)
            if row["y"].shape != (count,) or row["y_scaled"].shape != (count,) or len(row["dates"]) != count:
                raise ValueError("Incorrect target/date geometry: " + stage)
            if not all(np.isfinite(row[key]).all() for key in ("X", "relative", "raw", "y", "y_scaled")):
                raise ValueError("Nonfinite audited input: " + stage)
            if len(np.unique(row["dates"])) != count or (np.diff(row["dates"]).astype(int) <= 0).any():
                raise ValueError("Cohort dates are not unique and ordered: " + stage)
            result[stage], mapping[stage] = row, keys
        result["target_scale"] = float(result["train"]["y"].mean())
        if not math.isfinite(result["target_scale"]) or result["target_scale"] <= 0:
            raise ValueError("Training-only target scale must be finite and positive")
        if "training_target_scale" in source and not math.isclose(
                float(source["training_target_scale"]), result["target_scale"], rel_tol=1e-13, abs_tol=0.):
            raise ValueError("Audited training target scale does not match its cohort")
        for stage in ("train", "cal", "test"):
            expected = (result[stage]["y"] / result["target_scale"]).astype("float32")
            if not np.allclose(expected, result[stage]["y_scaled"], rtol=1e-6, atol=1e-7):
                raise ValueError("Target scaling mismatch: " + stage)
    for left, right in (("train", "cal"), ("train", "test"), ("cal", "test")):
        if set(result[left]["dates"]) & set(result[right]["dates"]):
            raise ValueError("Overlapping objective identifiers: " + left + "/" + right)
    return result, mapping


def source_bindings():
    """Check pinned source bytes without importing models or running forwards."""
    from e2_standalone_code import AUTHOR_MODELS, CONSTRUCTOR_SHA256, DESIGN, METADATA_SHA256
    result = {}
    for relative, expected in {**CONSTRUCTOR_SHA256, **METADATA_SHA256}.items():
        path = ROOT / relative
        if sha(path) != expected:
            raise ValueError("Pinned original source changed: " + relative)
        result[str(path)] = expected
    roots = json.loads(SOURCE_ROOTS.read_text())
    rows = [row for row in json.loads((ROOT / DESIGN).read_text())["author_sources"]
            if row["model"] in AUTHOR_MODELS]
    if len(rows) != 76 or set(roots) != set(AUTHOR_MODELS):
        raise ValueError("Expected all six author subsets and 76 pinned files")
    for row in rows:
        path = Path(roots[row["model"]]) / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Pinned author bytes differ: " + str(path))
        result[str(path)] = row["sha256"]
    for path in (ARCHITECTURES, REGISTER, ROOT / "scripts/e2_staged_training.py",
                 ROOT / "scripts/e2_published_pilot_training.py"):
        result[str(path)] = sha(path)
    return result


def prepare(args):
    audit = json.loads(args.audit.read_text())
    if audit.get("data_ready_for_bounded_controls") is not True:
        raise ValueError("Auditor has not admitted the real cohort for bounded controls")
    actual = sha(args.arrays)
    if audit.get("arrays_npz_sha256") != actual or args.expected_sha256 != actual:
        raise ValueError("Array hash does not match the independent audited receipt")
    cohorts, mapping = load_cohorts(args.arrays, audit.get("array_keys"))
    model_rows = json.loads(ARCHITECTURES.read_text())
    if {r["model"] for r in model_rows} != set(MODELS):
        raise ValueError("Frozen recipes do not match the current thirteen networks")
    import csv
    with REGISTER.open(newline="") as stream:
        register = {r["model"]: r for r in csv.DictReader(stream)}
    recipes = {}
    for row in model_rows:
        name, trial = row["model"], row["trial"]
        epochs = {phase: int(register[name]["fixed_" + phase + "_epochs"])
                  for phase in (("ae1", "ae2", "supervised") if name == SAE else ("supervised",))}
        recipes[name] = {"trial": trial, "fixed_epochs": epochs,
                         "parameters": row["architecture"]["actual_parameter_count"]}
    specification = {
        "version": VERSION, "prepared_at_utc": stamp(), "status": "prepared_before_new_control_fits",
        "scientific_source_commit": ENGINE, "original_primary_modified": False,
        "retrospective": True, "new_blind_test_claimed": False,
        "review": "Root admitted this post-benchmark diagnostic design before control fits; results are not automatic proof or exclusion of leakage.",
        "arrays_path": str(args.arrays.resolve()), "arrays_sha256": actual,
        "audit_path": str(args.audit.resolve()), "audit_sha256": sha(args.audit), "array_keys": mapping,
        "script_sha256": sha(__file__), "source_roots_path": str(SOURCE_ROOTS),
        "source_roots_sha256": sha(SOURCE_ROOTS),
        "sources_sha256": source_bindings(),
        "cohort_array_sha256": {stage: {key: array_hash(value) for key, value in cohorts[stage].items()}
                                for stage in ("train", "cal", "test")},
        "adversarial": {"features": "35 relative-window features only; no target, date or future covariate",
            "folds": 5, "purge_calendar_days": 7, "positive_class": "evaluation-period membership",
            "classifier": "balanced logistic regression, L2 penalty0.01, intercept unpenalized, L-BFGS-B maxiter200, gtol1e-7",
            "fold_scaler": "mean/std fitted only on each classifier-training fold, reused identically under label permutation",
            "permutation_seed": 2026111503, "permutations": 999, "permutation_block_days": 7,
            "permutation_scope": "equal-length contiguous blocks shuffled across both domain cohorts; short blocks grouped by length",
            "p": "(1+count(AUC_perm>=AUC_observed))/(999+1); approximate block exchangeability, not exact",
            "bootstrap_seed": 2026111504, "bootstrap_draws": 10000, "bootstrap_block_days": 14,
            "CI_scope": "fixed OOF predictions, calendar-block sensitivity, not full training uncertainty",
            "alpha": .05, "family_size": 1, "interpretation": "distribution discrimination, not proof of leakage"},
        "negative": {"backend": "new dedicated CPU diagnostic; not production CUDA or original numerical reproduction",
            "models": MODELS, "seed": SEED, "label_permutation_seed": 2026111501,
            "label_block_days": 7, "label_permutation_scope": "equal-length contiguous blocks without crossing missing dates; edge blocks retained or shuffled only with same-length blocks",
            "recipes": recipes, "HPO": False, "validation_used_for_fitting": False,
            "pretraining": "SAE auxiliary phases use original training X only; supervised labels are permuted",
            "optimizer": "same registered AdamW groups,betas(.9,.999),eps1e-8,clip1,FP32,batch_indices,cosine/warmup horizon200epochs",
            "reference": "constant true training mean; permutation preserves marginal labels and target scale",
            "gain": "RMSE_constant-RMSE_control", "gain_null_boundary": DELTA,
            "association": "Spearman(prediction,y)>0 on fixed608day evaluation",
            "bootstrap_seed": 2026111505, "bootstrap_draws": 10000, "bootstrap_block_days": 14,
            "test": "IUT:max(one-sided centered-bootstrap gain>delta p, centered-bootstrap rho>0 p), Holm13 alpha.05",
            "nonestimable": "retain status and p=1 for family accounting; never call a nonestimable control passed",
            "time_pilot": "first fixed model FCN-1D is control1/13, reused without refitting; runtime admission never uses its score",
            "interpretation": "a flag requires investigation, not automatic invalidity; no flag does not certify absence of leakage"},
        "classical_controls": {"status": "not_executed", "supervised_label_permutation": "not applicable to endogenous predictors",
            "needed_alternative": "prefix invariance under future-only perturbations of input histories, with frozen ARIMA parameters and no refit",
            "classical_models": ["Persistence", "SeasonalNaive7", "ARIMA"]}}
    write(args.spec, specification)
    print(json.dumps({"event": "controls_specification_prepared", "path": str(args.spec),
                      "sha256": sha(args.spec), "fits": 0}), flush=True)


def admitted(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    if spec["script_sha256"] != sha(__file__):
        raise ValueError("Control implementation changed after pre-fit specification")
    if sha(spec["arrays_path"]) != spec["arrays_sha256"] or sha(spec["audit_path"]) != spec["audit_sha256"]:
        raise ValueError("Audited input or receipt changed")
    if sha(spec["source_roots_path"]) != spec["source_roots_sha256"]:
        raise ValueError("Author-source roots changed")
    for name, expected in spec["sources_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen training source changed: " + name)
    return spec, load_cohorts(spec["arrays_path"], spec["array_keys"])[0]


def permute_blocks(values, dates, block, seed):
    import numpy as np
    values, dates = np.asarray(values), np.asarray(dates).astype("datetime64[D]")
    if len(values) != len(dates) or len(dates) < 1 or block < 1 or (np.diff(dates).astype(int) <= 0).any():
        raise ValueError("Block permutation needs ordered distinct dates, matched values and a positive block")
    cuts = np.flatnonzero(np.diff(dates).astype(int) != 1) + 1
    runs = np.split(np.arange(len(dates)), cuts)
    chunks = [run[start:start+block] for run in runs for start in range(0, len(run), block)]
    groups = {}
    for chunk in chunks:
        groups.setdefault(len(chunk), []).append(chunk)
    rng, source = np.random.default_rng(seed), np.arange(len(values))
    for same_length in groups.values():
        order = rng.permutation(len(same_length))
        for destination, index in zip(same_length, order):
            source[destination] = same_length[index]
    return values[source].copy(), source


def auc(labels, prediction):
    import numpy as np
    from scipy.stats import rankdata
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    if labels.shape != prediction.shape or not np.isfinite(prediction).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("AUC requires matched finite scores and binary domain labels")
    positive = labels == 1; p, n = int(positive.sum()), int((labels == 0).sum())
    if not p or not n:
        return float("nan")
    ranks = rankdata(prediction, method="average")
    return float((ranks[positive].sum() - p*(p+1)/2) / (p*n))


def purged_folds(x, dates):
    import numpy as np
    result = []
    days = dates.astype("datetime64[D]").astype("int64")
    if len(x) != len(days) or len(days) < 5 or (np.diff(days) <= 0).any():
        raise ValueError("Five purged folds require matched, ordered, distinct calendar observations")
    for fold, validation in enumerate(np.array_split(np.arange(len(x)), 5)):
        keep = np.ones(len(x), dtype=bool)
        for day in days[validation]:
            keep &= np.abs(days-day) > 7
        training = np.flatnonzero(keep)
        if not len(training):
            raise ValueError("No classifier-training rows remain after calendar purge")
        mean, scale = x[training].mean(axis=0), x[training].std(axis=0)
        scale = np.where(scale == 0, 1., scale)
        result.append({"fold": fold, "train": training, "validation": validation,
                       "x_train": (x[training]-mean)/scale, "x_validation": (x[validation]-mean)/scale,
                       "mean": mean, "std": scale})
    return result


def logistic_predict(x, labels, prediction_x):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    counts = np.bincount(labels.astype(int), minlength=2)
    if (counts == 0).any():
        raise ValueError("Classifier-training fold lacks a domain class")
    weights = len(labels) / (2 * counts[labels.astype(int)])
    a = np.column_stack((np.ones(len(x)), x)); b = np.column_stack((np.ones(len(prediction_x)), prediction_x))
    penalty = np.r_[0., np.repeat(.01, x.shape[1])]
    def objective(beta):
        eta = a @ beta
        value = np.mean(weights*(np.logaddexp(0., eta)-labels*eta)) + .5*np.dot(penalty*beta, beta)
        gradient = a.T @ (weights*(expit(eta)-labels))/len(labels)+penalty*beta
        return value, gradient
    fit = minimize(objective, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200, "gtol": 1e-7, "ftol": 1e-12})
    if not fit.success:
        raise RuntimeError("Domain classifier did not converge: " + str(fit.message))
    return expit(b @ fit.x), {"iterations": int(fit.nit), "objective": float(fit.fun),
                              "class_counts": counts.tolist(), "balanced": True,
                              "class_weights": (len(labels)/(2*counts)).tolist()}


def domain_oof(folds, labels):
    import numpy as np
    predictions = np.full(len(labels), np.nan); receipts = []
    for fold in folds:
        prediction, receipt = logistic_predict(fold["x_train"], labels[fold["train"]], fold["x_validation"])
        predictions[fold["validation"]] = prediction
        receipts.append({**receipt, "fold": fold["fold"], "training_count": len(fold["train"]),
                         "validation_count": len(fold["validation"]),
                         "validation_class_counts": np.bincount(labels[fold["validation"]].astype(int), minlength=2).tolist()})
    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF predictions are incomplete")
    return predictions, receipts


def calendar_indices(dates, draws, block, seed):
    import numpy as np
    days = dates.astype("datetime64[D]").astype(int)
    if len(days) < 1 or draws < 1 or block < 1 or (np.diff(days) <= 0).any():
        raise ValueError("Calendar resampling requires ordered distinct dates and positive sizes")
    full = np.full(int(days[-1]-days[0]+1), -1, dtype=int); full[days-days[0]] = np.arange(len(days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(full), size=(draws, math.ceil(len(full)/block)))
    sampled = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(full)
    return full[sampled.reshape(draws, -1)[:, :len(full)]]


def bootstrap_auc(labels, predictions, dates, config):
    import numpy as np
    from scipy.stats import rankdata
    indices = calendar_indices(dates, config["bootstrap_draws"], config["bootstrap_block_days"], config["bootstrap_seed"])
    values = []
    for start in range(0, len(indices), 250):
        block = indices[start:start+250]; valid = block >= 0; safe = np.maximum(block, 0)
        lab = labels[safe]; scores = np.where(valid, predictions[safe], np.nan)
        ranks = rankdata(scores, axis=1, method="average", nan_policy="omit")
        positive = valid & (lab == 1); negative = valid & (lab == 0)
        p, n = positive.sum(axis=1), negative.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = (np.nansum(np.where(positive, ranks, 0.), axis=1)-p*(p+1)/2)/(p*n)
        values.extend(float(x) for x in result if np.isfinite(x))
    return {"CI95": np.quantile(values, [.025, .975]).tolist(), "valid_draws": len(values),
            "requested_draws": len(indices), "scope": "fixed OOF predictions; temporal sensitivity only"}


def adversarial(args):
    import numpy as np
    require_budget(args.phase); spec, data = admitted(args.spec); config = spec["adversarial"]
    write(PUBLIC / "adversarial_request.json", {"spec_sha256": sha(args.spec), "started_at_utc": stamp(), "phase": args.phase})
    dates = np.r_[data["train"]["dates"], data["test"]["dates"]]
    x = np.concatenate((data["train"]["relative"], data["test"]["relative"])).reshape(-1, 35)
    labels = np.r_[np.zeros(655, dtype=int), np.ones(608, dtype=int)]
    folds = purged_folds(x, dates)
    fold_manifest = [{"fold": fold["fold"], "training_indices": fold["train"].tolist(),
                      "validation_indices": fold["validation"].tolist(),
                      "feature_mean_training_only": fold["mean"].tolist(),
                      "feature_std_training_only": fold["std"].tolist(),
                      "minimum_train_validation_calendar_separation_days": int(np.min(np.abs(
                          dates[fold["train"]].astype(int)[:, None]-dates[fold["validation"]].astype(int)[None, :]))) }
                     for fold in folds]
    predictions, receipts = domain_oof(folds, labels); observed = auc(labels, predictions)
    null = []; null_receipts = []
    for index in range(config["permutations"]):
        shuffled, permutation = permute_blocks(labels, dates, 7, config["permutation_seed"]+index)
        values, details = domain_oof(folds, shuffled); value = auc(shuffled, values)
        if not math.isfinite(value):
            raise RuntimeError("Nonestimable domain permutation; result not silently discarded")
        null.append(value); null_receipts.append({"index": index, "AUC": value,
                                                  "permutation_sha256": array_hash(permutation),
                                                  "folds": details})
        if (index+1) % 50 == 0:
            print(json.dumps({"event": "adversarial_permutations", "completed": index+1, "planned": config["permutations"]}), flush=True)
    p = (1+sum(value >= observed for value in null))/(1+len(null))
    ci = bootstrap_auc(labels, predictions, dates, config)
    np.savez_compressed(WORK / "adversarial_predictions.npz", dates=dates.astype(str), labels=labels,
                        predictions=predictions, permutation_AUC=np.asarray(null))
    result = {"version": VERSION, "status": "completed", "spec_sha256": sha(args.spec),
              "AUC_OOF": observed, "p_block_permutation_with_refit": p, "permutations_completed": len(null),
              "permutation_resolution": 1/(1+len(null)), "reject_alpha05": p <= .05,
              "fixed_OOF_temporal_CI": ci, "folds": receipts,
              "feature_count": 35, "train_domain_count": 655, "test_domain_count": 608,
              "folds_use_future_rows_for_distribution_diagnostic_not_causal_forecast": True,
              "labels_are_domains_not_predictive_targets": True,
              "prediction_file": str(WORK / "adversarial_predictions.npz"),
              "prediction_sha256": sha(WORK / "adversarial_predictions.npz"),
              "interpretation": "A rejection supports discrimination between time-period covariates under approximate block exchangeability; does not itself prove leakage.",
              "all_null_fits_refitted": True, "primary_modified": False}
    write(PUBLIC / "adversarial_permutations.json", null_receipts)
    write(PUBLIC / "adversarial_fold_manifest.json", fold_manifest)
    write(PUBLIC / "adversarial_result.json", result)
    print(json.dumps({"event": "adversarial_completed", "AUC": observed, "p": p, "reject": p <= .05}), flush=True)


def phase_cpu(model, x, y, config, epochs, target_scale):
    """The frozen optimization arithmetic, explicitly executed as a new CPU diagnostic."""
    import torch
    from torch import nn
    from torch.nn import functional as functional
    import e2_staged_training as historical
    model.to("cpu"); x, y = x.to("cpu"), y.to("cpu")
    historical._seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    batches = len(historical.batch_indices(len(x), config.batch_size, generator=torch.Generator().manual_seed(config.seed)))
    total = config.max_epochs*batches
    optimizer = torch.optim.AdamW(historical.optimizer_groups(model, config.weight_decay),
                                 lr=config.learning_rate, betas=(.9, .999), eps=1e-8)
    initial = historical.state_sha256(model); logs, steps = [], 0
    for epoch in range(1, epochs+1):
        model.train(); sizes = []
        for indices in historical.batch_indices(len(x), config.batch_size, generator=generator):
            rate = config.learning_rate*historical.learning_rate_multiplier(steps, total, config.warmup_fraction)
            for group in optimizer.param_groups: group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            prediction, truth = model(x[indices]), y[indices]
            if prediction.shape != truth.shape or not bool(torch.isfinite(prediction).all()):
                raise ValueError("CPU diagnostic prediction geometry/nonfinite failure")
            loss = functional.mse_loss(prediction, truth)
            if not bool(torch.isfinite(loss)): raise ValueError("Nonfinite CPU diagnostic objective")
            loss.backward(); trainable = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in trainable):
                raise ValueError("Missing/nonfinite CPU diagnostic gradient")
            norm = nn.utils.clip_grad_norm_(trainable, config.gradient_clip, error_if_nonfinite=False)
            if not bool(torch.isfinite(norm)): raise ValueError("Nonfinite gradient norm")
            optimizer.step(); steps += 1; sizes.append(len(indices))
            if any(not bool(torch.isfinite(p).all()) for p in trainable): raise ValueError("Nonfinite updated weights")
        mse = historical._mse(model, x, y, config.batch_size)
        logs.append({"epoch": epoch, "train_mse_scaled": mse, "train_rmse_original": math.sqrt(mse)*target_scale,
                     "examples_seen": sum(sizes), "optimizer_steps": steps})
    model.eval()
    return {"initial_state_sha256": initial, "final_state_sha256": historical.state_sha256(model),
            "optimizer_steps": steps, "fixed_epochs": epochs, "scheduler_total_steps": total,
            "logs": logs, "backend": "new_CPU_diagnostic_not_production_CUDA"}


def negative(args):
    import numpy as np
    import torch
    import e2_staged_training as historical
    from e2_reduced_fit import _builder
    require_budget(args.phase); spec, data = admitted(args.spec)
    model_name = args.model; recipe = spec["negative"]["recipes"][model_name]
    job = WORK / "negative" / (str(MODELS.index(model_name)+1).zfill(2))
    write(job / "request.json", {"spec_sha256": sha(args.spec), "model": model_name,
                                "seed": SEED, "started_at_utc": stamp(), "phase": args.phase})
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    y_permuted, permutation = permute_blocks(data["train"]["y_scaled"], data["train"]["dates"], 7,
                                              spec["negative"]["label_permutation_seed"])
    x = torch.tensor(data["train"]["X"], dtype=torch.float32)
    y = torch.tensor(y_permuted, dtype=torch.float32)
    trial = recipe["trial"]
    config = historical.TrainConfig(learning_rate=trial["learning_rate"], weight_decay=trial["weight_decay"],
        batch_size=trial["batch_size"], max_epochs=200, patience=20, seed=SEED,
        warmup_fraction=trial.get("warmup_fraction", 0))
    phases = {}; started = time.monotonic()
    if model_name == SAE:
        from e2_recipe_models import SAEPretraining
        from e2_published_sae import _annotate, _verify_source_and_dropout
        _verify_source_and_dropout(trial["dropout"]); historical._seed(SEED)
        manager = SAEPretraining(x, array_hash(data["train"]["X"]), accept_auxiliary_scale=True)
        x1, y1 = manager.phase1_data()
        phases["ae1"] = phase_cpu(manager.ae1, x1, y1, config, recipe["fixed_epochs"]["ae1"], 1.)
        x2, y2 = manager.prepare_phase2(); encoder_before = historical.state_sha256(manager.ae1.encoder)
        if not torch.equal(y1, y2): raise ValueError("SAE auxiliary targets changed")
        phases["ae2"] = phase_cpu(manager.ae2, x2, y2, config, recipe["fixed_epochs"]["ae2"], 1.)
        if historical.state_sha256(manager.ae1.encoder) != encoder_before: raise ValueError("SAE encoder1 changed in phase2")
        historical._seed(SEED); model = _annotate(manager.transfer_predictor(), dropout=trial["dropout"], seed=SEED)
    else:
        model = _builder(model_name, trial, SEED, json.loads(SOURCE_ROOTS.read_text()))
    if sum(parameter.numel() for parameter in model.parameters()) != recipe["parameters"]:
        raise ValueError("CPU diagnostic constructor differs from the frozen parameter count")
    phases["supervised"] = phase_cpu(model, x, y, config, recipe["fixed_epochs"]["supervised"], data["target_scale"])
    checkpoint = job / "diagnostic_model.pt"
    torch.save({"state_dict": historical.clone_state(model), "model": model_name, "seed": SEED,
                "spec_sha256": sha(args.spec), "backend": "CPU_diagnostic", "label_permutation_sha256": array_hash(permutation)}, checkpoint)
    prediction = []
    with torch.inference_mode():
        for row in data["test"]["X"]:
            value = model(torch.tensor(row[None], dtype=torch.float32)).reshape(-1)
            if len(value) != 1 or not bool(torch.isfinite(value).all()): raise ValueError("Invalid diagnostic test prediction")
            prediction.append(float(value.item())*data["target_scale"])
    np.savez_compressed(job / "predictions.npz", dates=data["test"]["dates"].astype(str),
                        prediction=np.asarray(prediction), truth=data["test"]["y"], permutation=permutation)
    result = {"version": VERSION, "status": "completed", "model": model_name, "seed": SEED,
        "new_diagnostic_backend": "CPU", "original_reproduction_claimed": False, "fixed_recipe": recipe,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "spec_sha256": sha(args.spec), "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "prediction_path": str(job / "predictions.npz"), "prediction_sha256": sha(job / "predictions.npz"),
        "permutation_sha256": array_hash(permutation), "permutation_fixed_rows": int(np.sum(permutation == np.arange(655))),
        "phase_records": phases, "elapsed_seconds": time.monotonic()-started, "primary_modified": False}
    write(PUBLIC / ("negative_" + str(MODELS.index(model_name)+1).zfill(2) + ".json"), result)
    print(json.dumps({"event": "negative_control_completed", "model": model_name,
                      "seconds": result["elapsed_seconds"], "checkpoint_sha256": result["checkpoint_sha256"]}), flush=True)


def holm(values):
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Holm requires a nonempty collection of valid p-values")
    order = sorted(range(len(values)), key=lambda i: values[i]); result = [1.]*len(values); running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index]); result[index] = min(1., running)
    return result


def centered_bootstrap_p(draws, estimate, null_boundary):
    """One-sided centered-bootstrap test, H0: parameter <= null_boundary."""
    import numpy as np
    draws = np.asarray(draws, dtype=float)
    if not len(draws) or not np.isfinite(draws).all() or not math.isfinite(estimate):
        raise ValueError("A centered-bootstrap test requires complete finite estimates")
    return (1+int(np.sum(null_boundary+draws-estimate >= estimate)))/(len(draws)+1)


def negative_summary(args):
    import numpy as np
    from scipy.stats import rankdata
    require_budget(args.phase); spec, data = admitted(args.spec); config = spec["negative"]
    truth, dates = data["test"]["y"], data["test"]["dates"]
    indices = calendar_indices(dates, config["bootstrap_draws"], 14, config["bootstrap_seed"])
    if (indices < 0).any(): raise ValueError("Expected contiguous current evaluation calendar")
    constant = np.full(608, data["target_scale"]); constant_error = (constant-truth)**2
    rows = []
    for number, name in enumerate(MODELS, 1):
        path = PUBLIC / ("negative_"+str(number).zfill(2)+".json")
        if not path.exists():
            rows.append({"model": name, "status": "not_executed", "p_IUT": 1.}); continue
        record = json.loads(path.read_text())
        if record.get("spec_sha256") != sha(args.spec) or record.get("model") != name or record.get("seed") != SEED:
            raise ValueError("Negative-control receipt does not match the admitted specification")
        if sha(record["checkpoint_path"]) != record["checkpoint_sha256"]:
            raise ValueError("Negative-control checkpoint changed")
        if sha(record["prediction_path"]) != record["prediction_sha256"]: raise ValueError("Prediction file changed")
        with np.load(record["prediction_path"], allow_pickle=False) as payload:
            prediction = payload["prediction"].copy()
            if (not np.array_equal(payload["dates"].astype("datetime64[D]"), dates)
                    or not np.array_equal(payload["truth"], truth)
                    or array_hash(payload["permutation"]) != record["permutation_sha256"]):
                raise ValueError("Negative-control payload does not match its cohort or permutation receipt")
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError("Negative-control predictions are incomplete or nonfinite")
        gain = math.sqrt(float(constant_error.mean()))-math.sqrt(float(((prediction-truth)**2).mean()))
        ranks_p, ranks_y = rankdata(prediction), rankdata(truth)
        rho = float(np.corrcoef(ranks_p, ranks_y)[0, 1]) if np.std(ranks_p) and np.std(ranks_y) else None
        gs, rs = [], []
        for start in range(0, len(indices), 250):
            ix = indices[start:start+250]; g = np.sqrt(constant_error[ix].mean(axis=1))-np.sqrt(((prediction[ix]-truth[ix])**2).mean(axis=1))
            gs.extend(g.tolist())
            if rho is not None:
                a, b = rankdata(prediction[ix], axis=1), rankdata(truth[ix], axis=1)
                a -= a.mean(axis=1, keepdims=True); b -= b.mean(axis=1, keepdims=True)
                denominator = np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
                rs.extend((np.divide((a*b).sum(axis=1), denominator, out=np.full(len(ix), np.nan), where=denominator>0)).tolist())
        gs = np.asarray(gs); rs = np.asarray(rs); finite = rs[np.isfinite(rs)]
        p_gain = centered_bootstrap_p(gs, gain, DELTA)
        p_rho = None if rho is None or len(finite) != len(indices) else centered_bootstrap_p(finite, rho, 0.)
        rows.append({"model": name, "status": "computed" if p_rho is not None else "association_nonestimable",
            "gain_RMSE_vs_training_mean": gain, "gain_CI95": np.quantile(gs,[.025,.975]).tolist(),
            "Spearman_rho": rho, "rho_CI95": None if not len(finite) else np.quantile(finite,[.025,.975]).tolist(),
            "p_gain_above_delta": p_gain, "p_rho_above_zero": p_rho,
            "p_IUT": 1. if p_rho is None else max(p_gain,p_rho), "valid_rho_draws": len(finite)})
    adjusted = holm([row["p_IUT"] for row in rows])
    for row,p in zip(rows,adjusted): row.update(p_Holm13=p, flag=row["status"]=="computed" and p<=.05)
    write(PUBLIC / "negative_summary.json", {"version": VERSION, "spec_sha256": sha(args.spec), "family_size": 13,
        "rows": rows, "bootstrap_draws": len(indices), "block_days": 14,
        "interpretation": "A flag is unexpected performance under one block-label diagnostic, requiring investigation; no flag does not certify absence of leakage.",
        "primary_modified": False})
    print(json.dumps({"event": "negative_summary", "flags": [row["model"] for row in rows if row["flag"]],
                      "completed": sum(row["status"] != "not_executed" for row in rows), "planned": 13}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--arrays", type=Path, required=True)
    preparation.add_argument("--audit", type=Path, required=True)
    preparation.add_argument("--expected-sha256", required=True)
    for mode in ("prepare", "adversarial", "negative", "negative-summary"):
        child = preparation if mode == "prepare" else sub.add_parser(mode)
        child.add_argument("--spec", type=Path, default=PUBLIC / "specification.json")
        if mode != "prepare": child.add_argument("--phase", required=True)
        if mode == "negative": child.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args(); PUBLIC.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True)
    {"prepare": prepare, "adversarial": adversarial, "negative": negative, "negative-summary": negative_summary}[args.mode](args)


if __name__ == "__main__":
    main()
