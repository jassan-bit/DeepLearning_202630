"""Read-only posthoc diagnostics of the original, frozen E2 predictors.

No optimizer, model selection, acquisition, dependency installation, or mutation
of the scientific output is implemented. Every predictor must reproduce its
saved batch-one predictions before its perturbations are admitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import sys
import time
import uuid

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
VERSION = "e2-frozen-posthoc-inference-v1"
LEDGER_SHA256 = "27aa2336518432d1d873098a3b2e0e5dc039dc501e15caa17599fce7d1cc6c30"
REMAINING_SECONDS = 18169.090972095262
TOP_NEURAL = ("LSTM-autoencoder con predictor", "GRU32")
FEATURES = ("open", "high", "low", "close", "volume")
SEVERITIES = {"noise": (.1, .3, .5), "outlier": (.01, .05, .1),
              "missing": (.05, .15, .3), "structural_break": (.25, .5, 1.)}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def heartbeat(**fields):
    print(json.dumps({"event": "e2_posthoc_progress", **fields}, ensure_ascii=False), flush=True)


def immutable_json(path, value):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Posthoc output must not use symlinks")
    raw = canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Existing posthoc evidence differs; preserve for reconciliation")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def projection_check(actual, expected, *, atol=1e-10):
    import numpy as np
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all()
            or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0, atol=atol)):
        raise ValueError("Frozen predictor does not reproduce original saved batch-one predictions")
    return {"baseline_projection_passed": True, "absolute_tolerance": atol,
            "max_absolute_error": float(np.max(np.abs(a-b))), "n_dates": len(a)}


def fit_reference(training_x):
    """Reference has no target or evaluation argument: training inputs only."""
    import numpy as np
    x = np.asarray(training_x, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (7, 5) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite training-only D7 windows are required")
    median = np.median(x, axis=0)
    scale = np.std(x, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 0.)
    return {"median": median.tolist(), "scale": scale.tolist(), "n_training_windows": len(x),
            "training_inputs_sha256": sha(x.astype('<f8').tobytes()),
            "training_label_end": "2024-10-31", "scope": "standardized_D7_model_inputs"}


def perturb(x, reference, kind, severity, *, seed=271901, break_index=365):
    """Stress predictor inputs; never change truth or fit a transformation."""
    import numpy as np
    original = np.asarray(x, dtype=float)
    if (kind not in SEVERITIES or severity not in SEVERITIES[kind]
            or original.ndim != 3 or original.shape[1:] != (7, 5)
            or not np.isfinite(original).all()):
        raise ValueError("Unknown registered perturbation or invalid D7 input")
    rng = np.random.default_rng(seed)
    center, scale = (np.asarray(reference[k], dtype=float) for k in ("median", "scale"))
    if center.shape != (7, 5) or scale.shape != (7, 5) or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("Invalid frozen training reference")
    if kind == "noise":
        result = original + severity * scale * rng.standard_normal(original.shape)
    elif kind == "outlier":
        mask = rng.random(original.shape) < severity
        sign = np.where(rng.random(original.shape) < .5, -1., 1.)
        result = original + mask * sign * 5. * scale
    elif kind == "missing":
        result = np.where(rng.random(original.shape) < severity, center, original)
    else:
        if type(break_index) is not int or not 0 <= break_index <= len(original):
            raise ValueError("Fixed structural-break calendar index required")
        result = original.copy()
        result[break_index:] += severity * scale
    return result.astype(np.float32)


def block_indices(n, draws=2000, block=14, seed=271902):
    import numpy as np
    if type(n) is not int or n < 2 or type(draws) is not int or draws < 2000 or type(block) is not int or not 1 <= block <= n:
        raise ValueError("At least 2000 circular-block bootstrap draws are required")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, (draws, math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]


def paired_rmse_change(truth, baseline, perturbed, indices):
    """Resample dates jointly across fixed seeds; seeds are not markets."""
    import numpy as np
    y, a, b = np.asarray(truth), np.atleast_2d(baseline), np.atleast_2d(perturbed)
    if a.shape != b.shape or a.shape[1:] != y.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired finite predictions on one unchanged target calendar required")
    la, lb = ((v-y)**2 for v in (a, b))
    base, changed = np.sqrt(la.mean(axis=1)).mean(), np.sqrt(lb.mean(axis=1)).mean()
    samples = []
    for start in range(0, len(indices), 64):
        batch = indices[start:start+64]
        samples.extend((np.sqrt(lb[:, batch].mean(axis=2)).mean(axis=0)
                        - np.sqrt(la[:, batch].mean(axis=2)).mean(axis=0)).tolist())
    return {"baseline_mean_seed_RMSE": float(base), "perturbed_mean_seed_RMSE": float(changed),
            "RMSE_degradation": float(changed-base),
            "relative_RMSE_degradation": None if base == 0 else float((changed-base)/base),
            "paired_date_block_CI95": np.quantile(samples, [.025, .975]).tolist(),
            "CI_scope": "conditional_on_fixed_seeds_and_observed_historical_calendar"}


def occlusion(predict, x, reference):
    """Signed cell effects, magnitude ranks, deletion and insertion curves."""
    import numpy as np
    x, center = np.asarray(x).copy(), np.asarray(reference)
    if x.ndim < 2 or center.shape != x.shape[1:] or not np.isfinite(x).all():
        raise ValueError("Occlusion reference must match predictor input dimensions")
    base = np.asarray(predict(x), dtype=float)
    flat = x.reshape(len(x), -1)
    ref = np.broadcast_to(center, x.shape).reshape(len(x), -1)
    effects = []
    for cell in range(flat.shape[1]):
        altered = flat.copy()
        altered[:, cell] = ref[:, cell]
        effects.append(base - np.asarray(predict(altered.reshape(x.shape)), dtype=float))
    effects = np.asarray(effects).T
    ranks = np.argsort(-np.abs(effects), axis=1, kind="stable")
    fractions = np.linspace(0, 1, 8)
    deletion, insertion, random_deletion, random_insertion = [], [], [], []
    rng = np.random.default_rng(271903)
    random_ranks = np.asarray([rng.permutation(flat.shape[1]) for _ in range(len(flat))])
    for fraction in fractions:
        count = int(round(fraction * flat.shape[1]))
        d, ins = flat.copy(), ref.copy()
        for i in range(len(flat)):
            cells = ranks[i, :count]
            d[i, cells], ins[i, cells] = ref[i, cells], flat[i, cells]
        deletion.append(np.asarray(predict(d.reshape(x.shape)), dtype=float).tolist())
        insertion.append(np.asarray(predict(ins.reshape(x.shape)), dtype=float).tolist())
        rd, rins = flat.copy(), ref.copy()
        for i in range(len(flat)):
            cells = random_ranks[i, :count]
            rd[i, cells], rins[i, cells] = ref[i, cells], flat[i, cells]
        random_deletion.append(np.asarray(predict(rd.reshape(x.shape)), dtype=float).tolist())
        random_insertion.append(np.asarray(predict(rins.reshape(x.shape)), dtype=float).tolist())
    magnitude = np.abs(effects).mean(axis=0)
    return {"signed_effects": effects.tolist(), "mean_absolute_effect": magnitude.tolist(),
            "fractions": fractions.tolist(), "deletion_predictions": deletion,
            "insertion_predictions": insertion, "baseline_predictions": base.tolist(),
            "random_deletion_predictions": random_deletion, "random_insertion_predictions": random_insertion,
            "random_order_seed": 271903,
            "method": "training_reference_cell_occlusion",
            "ranked_by": "absolute_individual_prediction_change",
            "fidelity_scope": "frozen_predictor_response_not_causal_market_effect"}


def stability(explanations):
    """Descriptive rank correlation across optimization seeds, tied ranks kept."""
    import numpy as np
    from scipy.stats import spearmanr
    values = [np.asarray(x["mean_absolute_effect"]) for x in explanations]
    if len(values) < 2:
        return {"seed_pairs": 0, "mean_spearman": None, "scope": "deterministic_R1_no_seed_stability_estimand"}
    pairs = []
    for i in range(len(values)):
        for j in range(i):
            if np.ptp(values[i]) == 0 or np.ptp(values[j]) == 0:
                pairs.append(None)
            else:
                pairs.append(float(spearmanr(values[i], values[j]).statistic))
    valid = [v for v in pairs if v is not None and math.isfinite(v)]
    return {"seed_pairs": len(pairs), "defined_pairs": len(valid),
            "mean_spearman": None if not valid else float(np.mean(valid)),
            "min_spearman": None if not valid else min(valid),
            "scope": "descriptive_same_24_dates_training_reference_and_frozen_recipes"}


def profile_predictor(model, x, *, warmup=32, repeats=100):
    import numpy as np
    import torch
    from e2_cuda_training import _execution_policy
    device = torch.device("cuda:0")
    model.to(device).eval()
    sample = torch.as_tensor(x[:1], dtype=torch.float32, device=device)
    elapsed = []
    with _execution_policy(device, True), torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        for _ in range(repeats):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            model(sample)
            end.record()
            torch.cuda.synchronize()
            elapsed.append(float(begin.elapsed_time(end)))
        allocated_peak, reserved_peak = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "state_dict_bytes": sum(t.numel()*t.element_size() for t in model.state_dict().values()),
            "batch_size": 1, "input_shape": [1, 7, 5], "warmup": warmup, "repeats": repeats,
            "latency_gpu_median_ms": float(np.median(elapsed)),
            "latency_gpu_p95_ms": float(np.quantile(elapsed, .95)),
            "throughput_serial_forecasts_per_second": float(1000/np.mean(elapsed)),
            "latency_scope": "CUDA_event_model_forward_resident_input_excludes_transfer_preprocessing",
            "peak_allocated_bytes": allocated_peak, "peak_reserved_bytes": reserved_peak,
            "incremental_allocated_peak_bytes": max(0, allocated_peak-allocated_before),
            "memory_scope": "isolated_forward_highwater_after_warmup_allocator_cache_retained",
            "FLOPs": None, "energy_joules": None}


def original_context(source_root, output_root):
    """Reuse the independently validated cached export, without publishing it."""
    import numpy as np
    import pandas as pd
    from export_e2_f2a_cached_data import build_bundle, _original_module, read_pinned, PIN_SHA
    source_root, output_root = Path(source_root), Path(output_root)
    bundle = build_bundle(source_root, output_root)
    loader = _original_module(source_root, "e2_reduced_final_data")
    final = _original_module(source_root, "e2_reduced_final")
    development, _ = loader.load_development_bridge(source_root)
    cache = output_root / "final/private-known-E1-data/known_E1_daily.csv"
    evaluation = pd.read_csv(io.BytesIO(read_pinned(cache, PIN_SHA["cache"])), index_col="market_day_utc")
    evaluation.index = pd.DatetimeIndex(pd.to_datetime(evaluation.index, utc=True))
    frame = pd.DataFrame(development.features, index=development.calendar, columns=loader.FEATURES)
    panel = loader._panel_from_frames(frame, evaluation)
    dates = pd.DatetimeIndex(pd.to_datetime(bundle["dates"], utc=True))
    freeze = json.loads(read_pinned(output_root / "final/final_freeze.json", PIN_SHA["freeze"]))
    # Verify all checkpoint/scaler/calibrator bytes before loading pickle/torch.
    loader.validate_final_freeze(freeze, output_root / "final")
    train = final.final_training_split(development)
    if any(d.strftime('%Y-%m-%d') > '2024-10-31' for d in train.scaler.train_dates):
        raise ValueError("A post-training date entered the reference")
    return {"bundle": bundle, "panel": panel, "dates": dates, "freeze": freeze,
            "training": train, "reference": fit_reference(train.x.numpy()), "final_module": final,
            "raw_windows": panel.raw_windows(dates), "y": np.asarray(bundle["y_true"])}


def load_neural(context, source_root, output_root, row):
    import torch
    from export_e2_f2a_cached_data import _inside, read_pinned
    checkpoint = _inside(Path(output_root)/"final", row["checkpoint"])
    raw = read_pinned(checkpoint, row["checkpoint_sha256"])
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    scaler = context["training"].scaler
    if saved["scaler"] != scaler.record():
        raise ValueError("Checkpoint scaler differs from the original training-only scaler")
    fit = json.loads((checkpoint.parent / "job_result.json").read_bytes())
    request = json.loads((checkpoint.parent / "request.json").read_bytes())
    if (saved["request_sha256"] != sha(canonical(request)) or fit["request_sha256"] != saved["request_sha256"]
            or request["name"] != row["model"] or request["seed"] != row["seed"]
            or request["sources"].get("source_commit") != ENGINE_COMMIT):
        raise ValueError("Frozen model request/source coupling changed")
    # The original worker spec carries pinned upstream source roots. Original
    # builders verify the author-source hashes before constructing each model.
    worker = json.loads((checkpoint.parent.parent / "worker_spec.json").read_bytes())
    trial = worker["selection"]["winner_trial"]
    if trial != request["trial"]:
        raise ValueError("Selected optimizer recipe changed")
    model = context["final_module"]._trial_builder(row["model"], trial, row["seed"], worker.get("source_roots"))
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    model.to("cuda:0")
    return model, scaler


def causal_arima_baseline(context, output_root, row, sample_indices, deadline):
    """Frozen coefficients; keep states at t-8 for local lag attribution."""
    import numpy as np
    import pandas as pd
    from export_e2_f2a_cached_data import _inside, read_pinned
    frozen = read_pinned(_inside(Path(output_root)/"final", row["checkpoint"]), row["checkpoint_sha256"])
    model = pickle.loads(frozen)
    parameters = list(model.receipt["parameters"])
    history = context["final_module"]._history(context["panel"])
    dates = context["dates"].tz_localize(None)
    sample_dates = dates[sample_indices]
    snapshots, windows = {}, []
    wanted = {day - pd.Timedelta(days=8): i for i, day in enumerate(sample_dates)}
    values = {}
    while model.current < dates[-1]:
        if time.monotonic() > deadline:
            raise TimeoutError("Posthoc wall cap reached")
        if model.current in wanted:
            snapshots[wanted[model.current]] = pickle.dumps(model)
        day = model.current + pd.Timedelta(days=1)
        if day >= model.origin:
            value = float(model.forecast(day).value)
            if day >= dates[0]:
                values[day] = value
        model.observe(day, history.get(day, float('nan')))
        if day in dates and day.day == 1:
            heartbeat(phase="ARIMA_baseline", date=day.strftime('%Y-%m-%d'))
    if parameters != list(model.receipt["parameters"]):
        raise ValueError("ARIMA coefficients changed")
    for day in sample_dates:
        windows.append([float(history.loc[day-pd.Timedelta(days=lag)]) for lag in range(7, 0, -1)])
    if len(snapshots) != len(sample_dates):
        raise ValueError("Missing causal ARIMA state before attribution window")
    center = float(history.loc[:'2024-10-31'].dropna().median())

    def predict(lags):
        result = []
        for i, sequence in enumerate(lags):
            candidate = pickle.loads(snapshots[i])
            for value in sequence:
                day = candidate.current + pd.Timedelta(days=1)
                if day >= candidate.origin:
                    candidate.forecast(day)
                candidate.observe(day, float(value))
            result.append(float(candidate.forecast(candidate.current+pd.Timedelta(days=1)).value))
        return np.asarray(result)

    return np.asarray([values[d] for d in dates]), predict, np.asarray(windows), np.full(7, center)


def backup_result(destination, receipt, client):
    """Backup failures preserve completed compute; retries never rerun models."""
    for attempt in range(3):
        try:
            client._branch_head()
            client._checked = True
            return client.persist_result(destination, receipt)
        except Exception:
            if attempt == 2:
                raise
            heartbeat(phase="backup_retry", attempt=attempt+1, compute_repeated=False)


def completed_record(folder, run_id):
    receipt_path = folder / "result.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("run_id") != run_id or receipt.get("status") != "completed":
        raise ValueError("Existing predictor result has incompatible identity/status")
    for relative, checksum in receipt["artifacts_sha256"].items():
        path = folder / relative
        if path.is_symlink() or sha(path.read_bytes()) != checksum:
            raise ValueError("Completed posthoc artifact changed")
    return receipt


def _configuration_commit(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("An immutable 40-hex published configuration commit is required")
    return value


def _run_impl(source_root, output_root, destination, client, *, wall_seconds=5400, deadline=None,
              configuration_commit=None):
    configuration_commit = _configuration_commit(configuration_commit)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if type(wall_seconds) is not int or not 60 <= wall_seconds <= 5400 or wall_seconds > REMAINING_SECONDS:
        raise ValueError("Posthoc cap must fit the original unextended remaining balance")
    destination, source_root, output_root = Path(destination), Path(source_root), Path(output_root)
    if any(destination.resolve().is_relative_to(p.resolve()) for p in (source_root, output_root)):
        raise ValueError("Posthoc writes must stay outside original sources and scientific output")
    if not torch.cuda.is_available() or str(torch.__version__) != "2.8.0+cu126" or sys.version_info[:2] != (3, 12):
        raise ValueError("Use the preserved original Python3.12 Torch2.8cu126 CUDA runtime")
    if "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("The registered A100 diagnostic scope is required")
    started = time.monotonic()
    deadline = started+wall_seconds if deadline is None else deadline
    heartbeat(phase="validate_sources_cache_frozen_artifacts")
    context = original_context(source_root, output_root)
    config = {"version": VERSION, "engine_commit": ENGINE_COMMIT,
              "configuration_commit": configuration_commit,
              "diagnostic_files_sha256": {name: sha(Path(__file__).with_name(name).read_bytes())
                  for name in ("e2_frozen_posthoc_gpu.py", "export_e2_f2a_cached_data.py", "e2_posthoc_fit_reproduction.py")},
              "freeze_sha256": context["bundle"]["provenance"]["pinned_source_sha256"]["freeze"],
              "severities": SEVERITIES, "perturbation_seed": 271901,
              "synthetic_break_date": "2026-01-01", "synthetic_break_index": 365,
              "stress_scope": "independent_per_forecast_standardized_measurement_inputs_not_coherent_raw_OHLCV_path",
              "reference": context["reference"], "batch_size": 1, "wall_cap_seconds": wall_seconds,
              "original_ledger_sha256": LEDGER_SHA256, "original_remaining_seconds": REMAINING_SECONDS,
              "budget_extended": False, "training_performed": False,
              "original_score_modified": False, "target_unchanged": True,
              "bootstrap_draws": 2000, "block_lengths": [7, 14, 28],
              "explanation_samples": 24, "sampling": "equally_spaced_calendar_indices_no_outcome_selection"}
    run_id = "posthoc-"+sha(canonical(config))[:24]
    immutable_json(destination / "registration.json", config)
    complete = completed_record(destination, run_id)
    if complete:
        heartbeat(phase="all_compute_completed_backup_only", run_id=run_id)
        return backup_result(destination, complete, client)
    from e2_published_pilot_training import predict_pilot
    x = context["training"].scaler.transform(context["raw_windows"]).astype(np.float32)
    # The frozen ARIMA is already filtered through December30. January8 is
    # the first shared explanation date with all seven preceding observations
    # available after that frozen state. Outcomes do not select these dates.
    sample_indices = np.linspace(7, len(x)-1, 24, dtype=int)
    originals = {(p["model"], p["seed"]): p for p in context["bundle"]["predictors"]}
    results = []
    for row in context["freeze"]["models"]:
        name, seed = row["model"], row["seed"]
        if seed == "deterministic" and name != "ARIMA":
            continue
        key = hashlib.sha256((name+str(seed)).encode()).hexdigest()[:24]
        folder = destination / "predictors" / key
        previous = completed_record(folder, run_id)
        if previous:
            backup_result(folder, previous, client)
            results.append(json.loads((folder / "diagnostics.json").read_bytes()))
            heartbeat(phase="verified_completed_predictor_reused", model=name, seed=seed)
            continue
        if (folder / "started.json").exists():
            raise ValueError("Interrupted predictor preserved; reconcile its charge before any reexecution")
        if time.monotonic() > deadline-30:
            raise TimeoutError("Posthoc cap reached before another predictor")
        immutable_json(folder / "started.json", {"run_id": run_id, "model": name, "seed": seed,
                        "reservation_wall_seconds_bound": wall_seconds, "original_ledger_sha256": LEDGER_SHA256})
        member_started = time.monotonic()
        heartbeat(phase="baseline_projection", model=name, seed=seed)
        member = {"model": name, "seed": seed, "checkpoint_sha256": row["checkpoint_sha256"],
                  "run_id": run_id, "posthoc": True, "training_performed": False,
                  "frozen_calibrator_q": originals[name, seed]["calibrator_q"],
                  "perturbations": [], "explanation": None, "profile": None}
        if name == "ARIMA":
            baseline, predict, lags, center = causal_arima_baseline(context, output_root, row, sample_indices, deadline)
            member.update(projection_check(baseline, originals[name, seed]["prediction_original"]))
            projection_check(predict(lags), baseline[sample_indices])
            heartbeat(phase="attribution", model=name, seed=seed, scope="seven_target_history_lags_older_state_fixed")
            member["explanation"] = occlusion(predict, lags, center)
            member["explanation"].update(input_scope="sigma7_target_history_lags_tminus7_to_tminus1_older_filter_state_fixed",
                feature_scope="univariate_target_history_no_OHLCV_feature_attribution",
                optimization_seed_stability=None, deterministic_R=1)
            member["baseline_predictions"] = baseline.tolist()
            member["profile_scope"] = "ARIMA_CPU_filtering_not_comparable_with_neural_D7_GPU_forward"
        else:
            model, scaler = load_neural(context, source_root, output_root, row)

            def predict(array):
                pieces = []
                array = np.asarray(array)
                for start in range(0, len(array), 64):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Posthoc wall cap reached")
                    tensor = torch.as_tensor(array[start:start+64], dtype=torch.float32)
                    pieces.append(predict_pilot(model, tensor, batch_size=1, device="cuda:0").numpy())
                return np.concatenate(pieces)*scaler.target_scale

            baseline = predict(x)
            member.update(projection_check(baseline, originals[name, seed]["prediction_original"]))
            member["baseline_predictions"] = baseline.tolist()
            member["profile"] = profile_predictor(model, x)
            for kind, levels in SEVERITIES.items():
                for severity in levels:
                    heartbeat(phase="perturbation", model=name, seed=seed, perturbation=kind, severity=severity)
                    changed = predict(perturb(x, context["reference"], kind, severity))
                    member["perturbations"].append({"kind": kind, "severity": severity,
                                                    "predictions": changed.tolist()})
            if name in TOP_NEURAL:
                heartbeat(phase="attribution", model=name, seed=seed)
                member["explanation"] = occlusion(predict, x[sample_indices], context["reference"]["median"])
                magnitude = np.asarray(member["explanation"]["mean_absolute_effect"]).reshape(7, 5)
                member["explanation"].update(feature_mean_absolute_effect=dict(zip(FEATURES, magnitude.sum(axis=0).tolist())),
                    lag_mean_absolute_effect={str(7-i): float(value) for i, value in enumerate(magnitude.sum(axis=1))},
                    input_scope="35_standardized_D7_OHLCV_coordinates_training_median_replacement")
            del model
            torch.cuda.empty_cache()
        member["elapsed_wall_seconds"] = time.monotonic()-member_started
        immutable_json(folder / "diagnostics.json", member)
        receipt = {"status": "completed", "version": VERSION, "run_id": run_id,
                   "attempt_id": run_id+"-"+key, "training_performed": False,
                   "artifacts_sha256": {"diagnostics.json": sha((folder / "diagnostics.json").read_bytes()),
                       "started.json": sha((folder / "started.json").read_bytes())}}
        immutable_json(folder / "result.json", receipt)
        backup_result(folder, receipt, client)
        results.append(member)
        heartbeat(phase="predictor_complete_backed_up", model=name, seed=seed,
                  elapsed_wall_seconds=member["elapsed_wall_seconds"])
    summary = summarize(results, context["y"], sample_indices, deadline=deadline)
    summary.update(version=VERSION, run_id=run_id, models_neural=13, neural_predictors=78,
        original_ledger_sha256=LEDGER_SHA256, original_remaining_seconds=REMAINING_SECONDS,
        wall_cap_seconds=wall_seconds, elapsed_current_invocation_seconds=time.monotonic()-started,
        completed_predictor_wall_seconds=sum(r["elapsed_wall_seconds"] for r in results),
        runtime={"torch": str(torch.__version__), "python": sys.version.split()[0],
                 "device_name": torch.cuda.get_device_name(0), "FP32": True, "AMP": False, "TF32": False},
        original_metrics_replaced=False, scientific_training=False, retuning=False,
        inference_timing_scope="new_posthoc_measurement_same_frozen_architectures_batch1",
        limitations=["Input stress tests do not identify realistic market interventions or distribution-shift performance",
                    "ARIMA has a different univariate causal-history input; its attribution is reported separately",
                    "No new scientific component ablation fits: these require six-seed budgeted training",
                    "Block confidence intervals are conditional descriptions on the already exposed 608-date calendar"])
    immutable_json(destination / "summary.json", summary)
    write_summary_csv(destination / "robustness.csv", summary["robustness"])
    receipt = {"status": "completed", "version": VERSION, "attempt_id": run_id, "run_id": run_id,
               "training_performed": False, "artifacts_sha256": {name: sha((destination/name).read_bytes())
                   for name in ("registration.json", "summary.json", "robustness.csv")}}
    immutable_json(destination / "result.json", receipt)
    result = backup_result(destination, receipt, client)
    heartbeat(phase="completed", run_id=run_id, **{k: result[k] for k in ("status", "remote_commit", "manifest_path")})
    return result


def run(source_root, output_root, destination, client, *, wall_seconds=5400, budget=None,
        configuration_commit=None):
    """Reserve and charge one invocation against the unchanged original80h cap."""
    configuration_commit = _configuration_commit(configuration_commit)
    if budget is None or getattr(budget, "persistence", None) is None:
        raise ValueError("Use el lanzador unificado: se exige el presupuesto compartido con persistencia remota")
    if type(wall_seconds) is not int or not 60 <= wall_seconds <= 5400:
        raise ValueError("Registered diagnostics cap must be 60..5400 seconds")
    job_id = "frozen-posthoc-"+uuid.uuid4().hex[:24]
    started = time.monotonic()
    budget.reserve(job_id, wall_seconds, metadata={"scope": VERSION, "training_performed": False,
                                                "configuration_commit": configuration_commit})
    status = "failed"
    try:
        result = _run_impl(source_root, output_root, destination, client,
                           wall_seconds=wall_seconds, deadline=started+wall_seconds,
                           configuration_commit=configuration_commit)
        status = "completed"
        return result
    except (KeyboardInterrupt, SystemExit):
        status = "interrupted"
        raise
    finally:
        budget.charge(job_id, time.monotonic()-started, status=status,
                      evidence={"scope": VERSION, "destination": str(destination),
                                "original_ledger_sha256": LEDGER_SHA256,
                                "compute_retries_from_completed_receipts_only": True})


def summarize(results, truth, sample_indices, *, deadline=None):
    import numpy as np
    groups = {}
    for row in results:
        groups.setdefault(row["model"], []).append(row)
    robust, explanations, profiles = [], [], []
    for name, rows in groups.items():
        baseline = [r["baseline_predictions"] for r in rows]
        for kind, levels in SEVERITIES.items():
            if not rows[0]["perturbations"]:
                break
            for severity in levels:
                changed = [next(p["predictions"] for p in r["perturbations"] if (p["kind"], p["severity"]) == (kind, severity)) for r in rows]
                for block in (7, 14, 28):
                    if deadline is not None and time.monotonic() > deadline:
                        raise TimeoutError("Posthoc cap reached during conditional bootstrap")
                    heartbeat(phase="block_bootstrap", model=name, perturbation=kind, severity=severity, block=block)
                    coverage = np.mean([np.mean(np.abs(np.asarray(p)-truth) <= r["frozen_calibrator_q"])
                                        for p, r in zip(changed, rows)])
                    robust.append({"model": name, "R": len(rows), "kind": kind, "severity": severity,
                        "block_length": block, "draws": 2000,
                        "coverage90_frozen_q": float(coverage),
                        "mean_interval_width90_frozen_q": float(np.mean([2*r["frozen_calibrator_q"] for r in rows])),
                        "recalibration_performed": False,
                        **paired_rmse_change(truth, baseline, changed, block_indices(len(truth), block=block))})
        items = [r["explanation"] for r in rows if r["explanation"] is not None]
        if items:
            fidelity = []
            sample_truth = np.asarray(truth)[sample_indices]
            for item in items:
                base = np.asarray(item["baseline_predictions"])
                deletion, insertion = np.asarray(item["deletion_predictions"]), np.asarray(item["insertion_predictions"])
                rd, ri = np.asarray(item["random_deletion_predictions"]), np.asarray(item["random_insertion_predictions"])
                fidelity.append({"fractions": item["fractions"],
                    "deletion_mean_absolute_prediction_change": np.abs(deletion-base).mean(axis=1).tolist(),
                    "insertion_mean_absolute_distance_to_original_prediction": np.abs(insertion-base).mean(axis=1).tolist(),
                    "deletion_RMSE": np.sqrt(((deletion-sample_truth)**2).mean(axis=1)).tolist(),
                    "insertion_RMSE": np.sqrt(((insertion-sample_truth)**2).mean(axis=1)).tolist()})
                fidelity[-1].update(
                    random_deletion_mean_absolute_prediction_change=np.abs(rd-base).mean(axis=1).tolist(),
                    random_insertion_mean_absolute_distance_to_original_prediction=np.abs(ri-base).mean(axis=1).tolist(),
                    deletion_ranked_minus_random_prediction_change=(np.abs(deletion-base).mean(axis=1)-np.abs(rd-base).mean(axis=1)).tolist(),
                    insertion_random_minus_ranked_prediction_distance=(np.abs(ri-base).mean(axis=1)-np.abs(insertion-base).mean(axis=1)).tolist(),
                    random_deletion_RMSE=np.sqrt(((rd-sample_truth)**2).mean(axis=1)).tolist(),
                    random_insertion_RMSE=np.sqrt(((ri-sample_truth)**2).mean(axis=1)).tolist())
            explanations.append({"model": name, "R": len(items), "sample_calendar_indices": sample_indices.tolist(),
                                 "stability": stability(items), "fidelity_per_seed": fidelity})
        for r in rows:
            if r["profile"]:
                profiles.append({"model": name, "seed": r["seed"], **r["profile"]})
    auc = []
    for name in groups:
        for kind, levels in SEVERITIES.items():
            selected = [r for r in robust if r['model'] == name and r['kind'] == kind and r['block_length'] == 14]
            if selected and all(r['relative_RMSE_degradation'] is not None for r in selected):
                values = [0.] + [next(r['relative_RMSE_degradation'] for r in selected if r['severity'] == level) for level in levels]
                axis = [0., *levels]
                area = sum((values[i]+values[i+1])*.5*(axis[i+1]-axis[i]) for i in range(len(levels)))
                auc.append({'model': name, 'kind': kind,
                            'normalized_degradation_area': float(area/levels[-1]),
                            'lower_is_better': True,
                            'scope': 'trapezoidal_relative_RMSE_degradation_area_divided_by_max_severity'})
    return {"robustness": robust, "normalized_degradation_areas": auc,
            "interpretability": explanations, "profiles": profiles,
            "predictor_records": len(results), "all_baseline_projections_passed": all(r["baseline_projection_passed"] for r in results)}


def write_summary_csv(path, rows):
    stream = io.StringIO(newline="")
    fields = ["model", "R", "kind", "severity", "block_length", "draws", "baseline_mean_seed_RMSE",
              "perturbed_mean_seed_RMSE", "RMSE_degradation", "relative_RMSE_degradation", "CI95_lower", "CI95_upper",
              "coverage90_frozen_q", "mean_interval_width90_frozen_q"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        record = {key: row[key] for key in fields if key in row}
        record.update(zip(("CI95_lower", "CI95_upper"), row["paired_date_block_CI95"]))
        writer.writerow(record)
    path = Path(path)
    raw = stream.getvalue().encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Existing robustness CSV changed")
    if not path.exists():
        with path.open("xb") as output:
            output.write(raw)


def main(argv=None):
    raise ValueError("Use el lanzador unificado de cierre E2; la ejecución independiente está deshabilitada para conservar un único presupuesto remoto")


if __name__ == "__main__":
    main()
