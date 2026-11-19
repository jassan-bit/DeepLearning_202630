"""Verify three disclosed real windows with serving dependencies only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent / "tsmixer"
fixture = json.loads((ROOT / "cpu_examples.json").read_bytes())
assert hashlib.sha256((ROOT / "metadata.json").read_bytes()).hexdigest() == fixture["metadata_sha256"]
meta = json.loads((ROOT / "metadata.json").read_bytes())
assert hashlib.sha256((ROOT / "model.onnx").read_bytes()).hexdigest() == meta["model_sha256"] == fixture["model_sha256"]
opts = ort.SessionOptions()
opts.intra_op_num_threads = opts.inter_op_num_threads = 1
session = ort.InferenceSession(str(ROOT / "model.onnx"), sess_options=opts, providers=["CPUExecutionProvider"])
reports = []
for case in fixture["cases"]:
    raw = np.asarray(case["raw_ohlcv"], dtype=np.float64)
    assert raw.shape == (7, 5) and np.isfinite(raw).all()
    relative = raw.copy()
    relative[:, :4] = np.log(raw[:, :4]) - np.log(raw[-1, 3])
    relative[:, 4] = np.log1p(raw[:, 4])
    x = ((relative - meta["preprocessing"]["mean"]) / meta["preprocessing"]["std"]).astype(np.float32)[None]
    result = float(session.run(None, {"windows": x})[0].reshape(-1)[0]) * meta["output_scale"]
    error = abs(result - case["expected_frozen_prediction_sigma7"])
    assert error <= fixture["absolute_tolerance_sigma7"]
    reports.append({"target_date": case["target_date"], "prediction_sigma7": result,
                    "interval_90_nominal": [max(0, result-meta["calibration_q"]), result+meta["calibration_q"]],
                    "absolute_error_vs_frozen": error})
print(json.dumps({"status": "passed", "provider": session.get_providers(), "torch_required": False, "cases": reports}))
