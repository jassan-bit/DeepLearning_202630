"""Export four fixed, previously trained predictors to ONNX CPU serving.

This step does not train, select, calibrate or change a recipe. It verifies all
608 original test windows against CPU memory and the frozen saved predictions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from e3_dashboard.inference import transform_window

NPZ_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
ADMISSION_SHA = "7de4ccec3e415484253c3867b4451040a1e9a2fd49e4b999599cbe785f9a9d32"
CHECKPOINT_SHA = "4d440b71db042b916bc4daa95031bd222fd1c3c7a94974cff9dca7b43d9df5c9"
E2_PREDICTIONS_SHA = "df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655"
E3_PREDICTIONS_SHA = "b82ba0a8a0371ea5643d9ff666a3ac5a1a4e91018b95a640bd88b1be50bb5133"
TOLERANCE = 1e-6
NAMES = {"MLP16": "mlp16", "LSTM32": "lstm32", "GRU32": "gru32",
         "CNN-LSTM-atencion-aditiva E3": "cnn_lstm_attention_e3"}


class E3OnnxHead(torch.nn.Module):
    """Same real softplus formula using ORT kernels that support float64.

    ORT 1.23.2 has no double Softplus kernel. This representation is checked
    against the unchanged scientific module, including its softplus threshold.
    """
    def __init__(self, scientific_model):
        super().__init__()
        self.scientific = scientific_model

    def forward(self, windows, variance_floor):
        score = self.scientific.raw_score(windows)
        positive = torch.clamp_min(score, 0) + torch.log(1 + torch.exp(-torch.abs(score)))
        return torch.sqrt(variance_floor) + self.scientific.y_scale * (1e-4 + positive)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise RuntimeError("Frozen input mismatch: " + str(path))


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def options():
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return opt


def run(args):
    torch.set_num_threads(1)
    checked(args.npz, NPZ_SHA)
    checked(args.admission, ADMISSION_SHA)
    checked(args.e3_checkpoint, CHECKPOINT_SHA)
    checked(args.prediction_data / "combined_predictors.json", E2_PREDICTIONS_SHA)
    checked(args.prediction_data / "e3_evaluation_predictions.csv", E3_PREDICTIONS_SHA)
    with np.load(args.npz, allow_pickle=False) as saved:
        train = np.asarray(saved["training_raw"], dtype=np.float64)
        train_dates = [str(x) for x in saved["training_dates"]]
        raw = np.asarray(saved["evaluation_raw"], dtype=np.float64)
        y = np.asarray(saved["evaluation_y_original"], dtype=np.float64)
        dates = [str(x) for x in saved["evaluation_dates"]]
    assert raw.shape == (608, 7, 5) and y.shape == (608,)
    prior = json.loads((args.prediction_data / "combined_predictors.json").read_bytes())
    assert prior["dates"] == dates and np.array_equal(prior["y_true"], y)
    e3rows = list(csv.DictReader((args.prediction_data / "e3_evaluation_predictions.csv").open()))
    e3rows = [r for r in e3rows if int(r["seed"]) == 271801]
    assert [r["date"] for r in e3rows] == dates
    assert np.array_equal([float(r["target_sigma7"]) for r in e3rows], y)
    admission = json.loads(args.admission.read_bytes())
    reports, manifest = [], []
    args.output.mkdir(parents=True, exist_ok=True)
    for name, slug in NAMES.items():
        folder = args.output / slug
        folder.mkdir(parents=True, exist_ok=True)
        e3 = name.endswith(" E3")
        if not e3:
            old = ROOT / "artifacts/e3_handoff" / name
            inherited = json.loads((old / "metadata.json").read_bytes())
            source = old / "model.torchscript.pt"
            checked(source, inherited["inference_artifact_sha256"])
            model = torch.jit.load(str(source), map_location="cpu").eval()
            pre = inherited["preprocessing"]
            preprocessing = {"mean": pre["channel_mean"], "std": pre["channel_std"],
                             "fit_count": 655, "fit_end": "2024-10-31",
                             "target_scale": pre["target_scale"]}
            q = inherited["calibration"]["q"]
            predictor = next(p for p in prior["predictors"] if p["model"] == name and p["seed"] == 271801)
            frozen = np.asarray(predictor["prediction_original"], dtype=np.float64)
            assert inherited["source_checkpoint_sha256"] == predictor["checkpoint_sha256"]
            train_fit = train
            output_scale = pre["target_scale"]
            parameters = {"MLP16": 593, "LSTM32": 5025, "GRU32": 3777}[name]
            scientific_source = {"type": "inherited_E2_TorchScript", "sha256": sha(source),
                                 "original_checkpoint_sha256": inherited["source_checkpoint_sha256"]}
        else:
            from src.e3_innovation_model import InnovationModel
            cp = torch.load(args.e3_checkpoint, map_location="cpu", weights_only=True)
            assert cp["seed"] == 271801 and cp["complete"] and cp["best_epoch"] == 38
            for path in ("src/original_model.py", "src/e3_development.py", "src/e3_innovation_model.py"):
                checked(ROOT / path, cp["source_hashes"][path])
            scaler = admission["scaler"]
            model = InnovationModel("additive_positive_head", u_scale=scaler["u_scale"],
                                    y_scale=scaler["y_scale"], seed=271801).cpu().eval()
            model.load_state_dict(cp["best_model_state"], strict=True)
            preprocessing = {"mean": scaler["mean"], "std": scaler["std"],
                             "fit_count": 532, "fit_end": "2024-06-30",
                             "target_scale": scaler["y_scale"]}
            q = None
            frozen = np.asarray([float(r["prediction_sigma7"]) for r in e3rows])
            train_fit = train[[i for i, day in enumerate(train_dates) if day in set(admission["fit_dates"])]]
            assert len(train_fit) == 532
            output_scale, parameters = 1.0, 1210
            scientific_source = {"type": "E3_frozen_best_model_state", "commit": "510c3bddfdc9926322e3210ac61ee1f5eeffad2e",
                                 "checkpoint_sha256": CHECKPOINT_SHA, "best_epoch": 38,
                                 "source_hashes": {p: cp["source_hashes"][p] for p in
                                                   ("src/original_model.py", "src/e3_development.py", "src/e3_innovation_model.py")}}
        temporary_meta = {"preprocessing": preprocessing}
        transformed = [transform_window(window, temporary_meta) for window in raw]
        fit_x = np.concatenate([transform_window(window, temporary_meta)[0] for window in train_fit])
        x0, a0 = transformed[0]
        onnx_path = folder / "model.onnx"
        if e3:
            torch.onnx.export(E3OnnxHead(model).eval(), (torch.from_numpy(x0), torch.from_numpy(a0)), str(onnx_path),
                              input_names=["windows", "variance_floor"], output_names=["prediction"],
                              opset_version=17, dynamo=False)
        else:
            torch.onnx.export(model, (torch.from_numpy(x0),), str(onnx_path),
                              input_names=["windows"], output_names=["normalized_prediction"],
                              opset_version=17, dynamo=False)
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        session = ort.InferenceSession(str(onnx_path), sess_options=options(), providers=["CPUExecutionProvider"])
        memory, exported = [], []
        with torch.no_grad():
            for x, a in transformed:
                if e3:
                    memory.append(float(model(torch.from_numpy(x), torch.from_numpy(a))[0]))
                else:
                    memory.append(float(model(torch.from_numpy(x))[0]) * output_scale)
                feeds = {"windows": x, **({"variance_floor": a} if e3 else {})}
                exported.append(float(session.run(None, feeds)[0].reshape(-1)[0]) * output_scale)
        memory, exported = np.asarray(memory), np.asarray(exported)
        max_error = float(np.max(np.abs(exported - memory)))
        saved_error = float(np.max(np.abs(exported - frozen)))
        assert max_error <= TOLERANCE and saved_error <= TOLERANCE
        rmse = lambda p: float(np.sqrt(np.mean((p-y) ** 2)))
        coverage = None if q is None else float(100*np.mean((y >= np.maximum(0, exported-q)) & (y <= exported+q)))
        timings = []
        for k in range(110):
            started = perf_counter()
            x, a = transform_window(raw[0], temporary_meta)
            session.run(None, {"windows": x, **({"variance_floor": a} if e3 else {})})
            if k >= 10:
                timings.append((perf_counter() - started) * 1000)
        report = {"model": name, "seed": 271801, "evaluation_dates": 608,
                  "date_start": dates[0], "date_end": dates[-1],
                  "declared_absolute_tolerance_original_sigma7": TOLERANCE,
                  "maximum_absolute_error_vs_CPU_memory": max_error,
                  "maximum_absolute_error_vs_frozen_saved_predictions": saved_error,
                  "RMSE_CPU_memory": rmse(memory), "RMSE_ONNX": rmse(exported),
                  "RMSE_ONNX_minus_memory": rmse(exported)-rmse(memory),
                  "RMSE_frozen_saved_predictor": rmse(frozen), "within_tolerance": True,
                  "cpu_median_ms": float(np.median(timings)), "cpu_p95_ms": float(np.quantile(timings,.95)),
                  "latency_scope": "preprocessing + warm ORT inference, one CPU thread, batch one, 10 warmups/100 calls; excludes explanation",
                  "onnx_bytes": onnx_path.stat().st_size, "trainable_parameters": parameters,
                  "FLOPs": None, "runtime_peak_memory": None,
                  "no_refit_no_recalibration": True}
        meta = {"schema_version": 1, "model": name, "seed": 271801,
                "representative_seed_rule": "smallest fixed seed ID, not selected by test performance",
                "format": "ONNX", "opset": 17, "model_sha256": sha(onnx_path),
                "softplus_export_representation": "max(z,0)+log(1+exp(-abs(z))) for float64 ORT kernels; checked against unchanged PyTorch module" if e3 else "inherited TorchScript graph",
                "input_shape": [1,7,5], "input_dtype": "float32", "uses_observable_floor": e3,
                "floor_input_dtype": "float64" if e3 else None, "output_scale": output_scale,
                "preprocessing": preprocessing, "calibration_q": q,
                "interval_nominal_coverage": None if q is None else .9,
                "historical_608_coverage_percent_for_this_seed": coverage,
                "output_unit": "original population sigma7, not annualized", "horizon_days": 1,
                "raw_schema": ["date","open","high","low","close","volume"],
                "max_csv_bytes": 102400,
                "ood": {"method": "per-lag/channel observed range in standardized fit windows",
                        "minimum_standardized": fit_x.min(axis=0).astype(float).tolist(),
                        "maximum_standardized": fit_x.max(axis=0).astype(float).tolist(),
                        "fit_only": True, "not_a_calibrated_reliability_guarantee": True},
                "explanation_reference_raw": train_fit[0].tolist(),
                "explanation_reference_target_date": train_dates[0],
                "scientific_source": scientific_source,
                "data_source": {"npz_sha256": NPZ_SHA, "immutable_commit": "af3f1223ad5899ed63016d2af2d09bae8470ab00"},
                "roundtrip": report, "libraries": {"python": platform.python_version(), "torch_export_only": str(torch.__version__),
                    "onnx_export_only": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__}}
        json_write(folder / "metadata.json", meta)
        json_write(folder / "roundtrip.json", report)
        manifest.append({"id": name, "slug": slug, "metadata_sha256": sha(folder / "metadata.json")})
        reports.append(report)
        print(json.dumps({"model":name,"max_error":max_error,"saved_error":saved_error,"bytes":report["onnx_bytes"]}))
    json_write(args.output / "manifest.json", {"schema_version":1,"cache_resident_sessions":1,"models":manifest})
    json_write(args.report, {
        "source_commit": "b3b7f5210e8a81fe3e0f35493221bcb0f6f8c4e4",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "reports": reports,
        "new_training_runs":0,"new_calibration_fits":0,"GPU_used":False,
        "limitations":["Results concern one fixed seed per model, not the ten-seed mean.",
                       "Memory of this export process includes Torch; serving RAM requires a separate runtime measurement.",
                       "Occlusion contrasts are exact sensitivities; causal validity and seed-stable interpretability remain unverified."]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz",type=Path,required=True)
    p.add_argument("--admission",type=Path,required=True)
    p.add_argument("--e3-checkpoint",type=Path,required=True)
    p.add_argument("--prediction-data",type=Path,required=True)
    p.add_argument("--output",type=Path,default=ROOT/"work/e3_dashboard_reexport")
    p.add_argument("--report",type=Path,default=ROOT/"work/e3_dashboard_reexport/export_verification.json")
    run(p.parse_args())
