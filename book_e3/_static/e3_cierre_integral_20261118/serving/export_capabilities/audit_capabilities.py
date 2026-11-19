"""Record actual export capabilities, with no fitting or remote assertions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


NEURAL = (
    ("FCN-1D", "scripts/e2_recipe_models.py:92", "Conv1d, population BatchNorm, ReLU, mean, Linear, Softplus", "standard_ops_candidate"),
    ("ResNet-1D", "scripts/e2_recipe_models.py:103", "Conv1d residual blocks, population BatchNorm, ReLU, mean, Linear, Softplus", "standard_ops_candidate"),
    ("Inception-1D", "scripts/e2_recipe_models.py:132", "padded Conv1d, MaxPool, concatenation, residual addition, BatchNorm, Softplus", "standard_ops_candidate"),
    ("LSTM-autoencoder con predictor", "scripts/e2_recipe_models.py:188;scripts/e2_published_sae.py:81", "two frozen LSTM encoders, repeat, Linear, Softplus; reconstruction decoders belong to pretraining only", "standard_ops_candidate"),
    ("TimesNet", "scripts/e2_published_periodic.py:111;scripts/e2_published_periodic.py:129;scripts/e2_published_periodic.py:165", "rfft, complex magnitude, non-DC topk, detach/cpu/numpy period selection and input-dependent branch reshape; one-window inference", "special_export_analysis_needed"),
    ("TSMixer", "scripts/e2_final_models_mlp.py:143;scripts/e2_final_models_mlp.py:182", "six fixed mixer blocks, eval-only population norms, Linear, ReLU, moments, Softplus", "exported_and_checked"),
    ("TiDE", "scripts/e2_published_tide.py:57;scripts/e2_published_tide.py:107;scripts/e2_published_tide.py:166", "source-default R0 dense residual forecast body, channel embeddings, fixed series ids, Linear and Softplus", "standard_ops_candidate"),
    ("TimeMixer", "scripts/e2_published_periodic.py:165;scripts/e2_published_periodic.py:228", "pinned author multi-scale forecasting body, avg downsampling, moving-average decomposition, Linear, Softplus", "author_graph_inspection_and_export_needed"),
    ("PatchTST", "scripts/e2_published_patchtst.py:40;scripts/e2_published_patchtst.py:96", "seven-day RevIN, end replication padding, unfold P3/S2, residual attention encoder, native flatten head, scalar readout", "author_graph_export_needed"),
    ("iTransformer", "scripts/e2_training_models.py:104;scripts/e2_final_models_attention.py:285", "pinned inverted-token normalization and two-layer 256-wide attention encoder; scalar task head", "author_graph_export_needed"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def run(args):
    target = Path(__file__).resolve().parent
    prior = json.loads(args.predictions.read_bytes())
    assert sha(args.predictions) == "df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655"
    recovered = [{"path_under_inspected_weight_root": str(p.relative_to(args.weights)),
                  "bytes": p.stat().st_size, "sha256": sha(p)}
                 for p in sorted(args.weights.rglob("selected_model.pt"))]
    recovered_by_sha = {r["sha256"]: r for r in recovered}
    report = json.loads((target / "tsmixer/roundtrip.json").read_bytes())
    assert report["within_tolerance"] and report["new_training_runs"] == report["new_calibration_fits"] == 0
    source_refs = set()
    rows = []
    selection_files = {json.loads(p.read_bytes())["model_id"]: p for p in args.selections.glob("*.json")}
    for model, evidence, operators, export_class in NEURAL:
        ref = next(p for p in prior["predictors"] if p["model"] == model and str(p["seed"]) == "271801")
        selection_path = selection_files[model]
        selection = json.loads(selection_path.read_bytes())
        assert selection["model_id"] == model and selection["source_current_recipe_verified"]
        checkpoint = ref["checkpoint_sha256"]
        actual = recovered_by_sha.get(checkpoint)
        tsmixer = model == "TSMixer"
        assert bool(actual) == tsmixer, "Reassess matrix after additional verified weights are recovered"
        rows.append({
            "model": model, "recipe_trial_id": selection["winner_parent_trial_id"],
            "serving_input_contract": "one standardized float32 [1,7,5] OHLCV window; original target restored by frozen scale",
            "seed": "271801", "expected_checkpoint_sha256": checkpoint,
            "checkpoint_payload_verified_locally": bool(actual),
            "local_payload": actual["path_under_inspected_weight_root"] if actual else "not recovered in inspected root",
            "export_state": "ONNX_CPU_parity_passed_not_in_live_registry" if tsmixer else "not_attempted_without_verified_frozen_checkpoint",
            "operator_assessment": export_class, "actual_operator_or_state_requirements": operators,
            "precomputed_under_current_interface_justified_by_contract": False,
            "final_resource_exclusion_justified": False,
            "reason": "Verified original payload exported to ONNX and compared to all608 frozen predictions and53 calibration predictions" if tsmixer else "Historical scores may remain visible while export is pending; lack of a local payload is not evidence of a computational or mathematical impossibility",
            "remaining_action": "Adopt versioned artifact through verified runtime registry; test actual serving flow" if tsmixer else "Recover seed271801 payload matching SHA256, pinned author sources where required, reconstruct exact recipe, export or record actual blocker, and compare608 frozen predictions before enabling live service",
            "resource_measurement": f"one-thread CPU p95 {report['cpu_p95_ms']:.6f}ms; ONNX {report['onnx_bytes']} bytes; runtime peak RAM not measured" if tsmixer else "no serving RAM/latency measurement; no resource exclusion claimed",
            "export_attempted_this_audit": tsmixer, "source_evidence": evidence,
            "selection_receipt_sha256": sha(selection_path),
        })
        source_refs.update(v.split(":")[0] for v in evidence.split(";"))
    classical = [
        ("Persistence", "fixed-lag1", "8 consecutive valid closes t-8..t-1, or the externally supplied frozen sigma7(t-1) label", "sigma7(t-1) uses7 returns, which need8 closes; seven admitted closes contain only6 returns"),
        ("SeasonalNaive", "period7", "sigma7(t-7): closes t-14..t-7; at least14 closes when the upload ends at t-1", "The frozen selected period is7; its source label is outside what seven recent OHLCV bars can reconstruct"),
        ("ARIMA", "order[2,0,1]", "frozen ARIMA parameters and dated filtered state through t-1, or at least90 valid training sigma7 labels plus sequential observation history", "The fixed-parameter causal filter requires dated target-label history/state; replacing it with a one-window neural tensor changes the predictor"),
    ]
    master_path = args.repo / "book_e3/_static/e3_bloques_ab_20261115/deployment_master.csv"
    master = {r["model"]: r for r in csv.DictReader(master_path.open())}
    for model, recipe, contract, reason in classical:
        frozen = next(p for p in prior["predictors"] if p["model"] == model)
        rows.append({
            "model": model, "recipe_trial_id": recipe, "serving_input_contract": contract,
            "seed": "deterministic", "expected_checkpoint_sha256": frozen.get("checkpoint_sha256", ""),
            "checkpoint_payload_verified_locally": False, "local_payload": "not audited as a production state payload",
            "export_state": "historical_precomputed_input_contract_justified", "operator_assessment": "requires_extended_history_or_state",
            "actual_operator_or_state_requirements": "sigma7 rolling7 log returns ddof0 with8 quality flags; exact dated lag" if model != "ARIMA" else "statsmodels0.14.5 frozen fit, forecast-before-observe and sequential extend with unchanged parameters",
            "precomputed_under_current_interface_justified_by_contract": True,
            "final_resource_exclusion_justified": False, "reason": reason,
            "remaining_action": "Keep disclosed historical results for seven-bar UI; an optional live implementation needs a separate admitted history/state contract and parity audit",
            "resource_measurement": "no resource exclusion claimed; mismatch concerns available information/state, not alleged model size",
            "export_attempted_this_audit": False,
            "source_evidence": "scripts/e2_temporal_controls.py:62;scripts/e2_temporal_controls.py:75" + (";scripts/e2_temporal_controls.py:143;scripts/e2_temporal_controls.py:228" if model == "ARIMA" else ""),
            "selection_receipt_sha256": master[model]["selection_source_sha256"],
        })
    assert len(rows) == 13
    path = target / "capability_matrix.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    source_refs.update(("scripts/e2_temporal_controls.py", "scripts/e2_published_registry.py", "scripts/e2_training_models.py"))
    sources = [{"path": rel, "sha256": sha(args.repo / rel)} for rel in sorted(source_refs)]
    # A constructive counterexample: identical seven admitted closes do not
    # determine either classical lag prediction. This invokes no model fit.
    sys.path.insert(0, str(args.repo / "scripts"))
    import numpy as np
    import pandas as pd
    from e2_temporal_controls import sigma7, seasonal_naive
    proofs = []
    origin = pd.Timestamp("2026-11-18", tz="UTC")
    for model, length, period in (("Persistence", 8, 1), ("SeasonalNaive", 14, 7)):
        idx = pd.date_range(origin-pd.Timedelta(days=length), periods=length, freq="D")
        a = np.full(length, 100.0)
        b = a.copy(); b[0] = 200.0
        assert np.array_equal(a[-7:], b[-7:])
        quality = pd.Series(1, index=idx)
        pa = seasonal_naive(sigma7(pd.Series(a, index=idx), quality), origin, period).value
        pb = seasonal_naive(sigma7(pd.Series(b, index=idx), quality), origin, period).value
        assert pa != pb
        proofs.append({"model": model, "identical_admitted_last7_closes": True,
                       "earlier_close_A": 100.0, "earlier_close_B": 200.0,
                       "historical_close_count": length, "forecast_A": pa, "forecast_B": pb,
                       "demonstrates_missing_information": True})
    write(target / "input_contract_counterexamples.json", {"examples": proofs,
        "synthetic_structural_check_only": True, "new_model_fits": 0,
        "meaning": "The seven-bar UI cannot determine the original exact-lag predictor; an extended history contract can enable it."})
    write(target / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "13 systems absent from original four-model ONNX live registry; one CPU session policy",
        "baseline_live_registry": "artifacts/e3_dashboard_v1/manifest.json",
        "baseline_live_registry_sha256": sha(args.repo / "artifacts/e3_dashboard_v1/manifest.json"),
        "source_main_revision_consulted_by_root": "1bae73b5373c11466b226fb3efab706237b5e68b",
        "source_files_actual_bytes": sources,
        "frozen_predictions_sha256": sha(args.predictions), "deployment_master_sha256": sha(master_path),
        "inspected_weight_root": str(args.weights), "inspected_payloads": recovered,
        "weight_search_scope_limitation": "Inventory covers this declared local root, not all remote releases or all possible workspace locations. Missing recovery does not prove missing original weights.",
        "summary": {"systems_audited": 13, "new_verified_ONNX_exports": 1,
                    "historical_only_justified_by_current_information_contract": 3,
                    "neural_exports_still_pending": 9, "additional_models_enabled_live": 0,
                    "new_training_runs": 0, "new_calibration_fits": 0,
                    "blanket_resource_exclusion_for_remaining_neural_systems_supported": False},
        "matrix_sha256": sha(path),
        "limits": ["No attempt was fabricated for checkpoints not recovered.",
                   "TimesNet FFT and numpy period dispatch are actual source-level hazards; no universal ONNX impossibility is claimed.",
                   "Remaining neural systems require measured export/parity or a documented actual deployment constraint.",
                   "New TSMixer artifact is an audit output until explicitly adopted by the runtime."]})
    print(json.dumps({"models": 13, "new_verified_exports": 1, "contract_justified_historical": 3, "neural_pending": 9}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "weights", "predictions", "selections"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())
