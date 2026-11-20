"""Minimal guarded offline CPU inference for the new frozen ONNX profile.

This uses the original fixed scaler/target scale/calibration q. It does not
update the live dashboard catalog, fit anything, or promise future coverage.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class FrozenExport:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_bytes())
        path = directory / "model.onnx"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.metadata["model_sha256"]:
            raise ValueError("ONNX hash does not match the frozen offline profile")
        if self.metadata["input_shape"] != [1, 7, 5] or self.metadata["uses_observable_floor"]:
            raise ValueError("Unsupported input contract")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def preprocess(self, window):
        raw = np.asarray(window, dtype=np.float64)
        if raw.shape != (7, 5) or not np.isfinite(raw).all() or not np.all(raw[:, :4] > 0) or not np.all(raw[:, 4] >= 0):
            raise ValueError("Expected seven finite OHLCV rows, positive prices and nonnegative volume")
        if not np.all(raw[:, 1] >= np.maximum(raw[:, 0], raw[:, 3])) or not np.all(raw[:, 2] <= np.minimum(raw[:, 0], raw[:, 3])):
            raise ValueError("Inconsistent OHLC ordering")
        relative = raw.copy()
        relative[:, :4] = np.log(raw[:, :4]) - np.log(raw[-1, 3])
        relative[:, 4] = np.log1p(raw[:, 4])
        scaler = self.metadata["preprocessing"]
        values = np.asarray((relative - scaler["mean"]) / scaler["std"], dtype=np.float32)[None]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite normalized input")
        return values

    def predict(self, window):
        values = self.preprocess(window)
        normalized = self.session.run([self.metadata["output_name"]], {self.metadata["input_name"]: values})[0]
        point = float(np.asarray(normalized).reshape(-1)[0]) * self.metadata["output_scale"]
        if not np.isfinite(point) or point < 0:
            raise ValueError("Invalid sigma7 prediction")
        q = self.metadata["calibration_q"]
        return {"prediction_sigma7": point, "interval_lower_sigma7": max(0.0, point - q), "interval_upper_sigma7": point + q,
                "nominal_coverage": self.metadata["interval_nominal_coverage"], "interval_limitations": self.metadata["interval_limitations"],
                "model_sha256": self.metadata["model_sha256"], "profile": self.metadata["profile"]}
