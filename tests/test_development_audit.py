"""Independent algebraic properties and training-only admission; no market fits."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import e3_audit_development as audit


def test_six_return_floor_matches_seven_return_variance_identity():
    rng = np.random.default_rng(812)
    known = rng.normal(0.0, 0.02, size=(32, 6))
    unknown = rng.normal(0.0, 0.08, size=32)
    closes = 100.0 * np.exp(np.column_stack([np.zeros(32), known.cumsum(axis=1)]))
    floor = audit.variance_floor(closes)
    complete = np.column_stack([known, unknown])
    direct_variance = complete.var(axis=1, ddof=0)
    reconstructed = floor ** 2 + (6.0 / 49.0) * (unknown - known.mean(axis=1)) ** 2
    np.testing.assert_allclose(reconstructed, direct_variance, rtol=2e-13, atol=2e-17)
    assert np.all(floor ** 2 <= direct_variance + 2e-17)
    # Equality occurs for the seventh return equal to the known-return mean.
    minimizing = np.column_stack([known, known.mean(axis=1)])
    np.testing.assert_allclose(floor ** 2, minimizing.var(axis=1), rtol=2e-13, atol=2e-17)
    np.testing.assert_allclose(audit.variance_floor(closes * 37.0), floor, rtol=2e-13, atol=2e-15)


def test_projection_preserves_or_reduces_each_absolute_and_squared_error():
    floor = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    targets = floor + np.array([0.0, 0.001, 0.0, 0.02, 0.01, 0.005])
    predictions = np.array([-0.01, 0.0, 0.03, 0.01, 0.04, 0.08])
    projected = audit.project_to_floor(predictions, floor)
    assert np.all(np.abs(projected - targets) <= np.abs(predictions - targets) + 1e-16)
    assert np.all((projected - targets) ** 2 <= (predictions - targets) ** 2 + 1e-16)
    np.testing.assert_array_equal(projected[predictions >= floor], predictions[predictions >= floor])
    np.testing.assert_array_equal(projected[predictions < floor], floor[predictions < floor])
    np.testing.assert_array_equal(audit.project_to_floor(projected, floor), projected)


def test_admission_rejects_hash_date_drift_and_never_reads_calibration_or_test(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.npz"
    raw = np.ones((4, 7, 5), dtype=np.float64) * 100
    dates = np.array(["2024-06-29", "2024-06-30", "2024-07-08", "2024-07-09"])
    np.savez(path, training_raw=raw, training_y_original=np.array([.01, .02, .03, .04]),
             training_dates=dates, calibration_raw=np.array([object()], dtype=object),
             evaluation_y_original=np.array([object()], dtype=object))
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    real_load, accesses = np.load, []

    class TrainingOnlyArchive:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.source.close()

        def __getitem__(self, key):
            accesses.append(key)
            assert key in audit.TRAINING_KEYS, "Held-out member accessed"
            return self.source[key]

    monkeypatch.setattr(audit.np, "load", lambda *args, **kwargs: TrainingOnlyArchive(real_load(*args, **kwargs)))
    expected_dates = ["2024-07-08", "2024-07-09"]
    admitted = audit.load_validation_data(path, expected_hash, expected_dates=expected_dates)
    assert admitted["dates"] == expected_dates and set(accesses) == set(audit.TRAINING_KEYS)
    read_count = len(accesses)
    with pytest.raises(ValueError, match="SHA-256"):
        audit.load_validation_data(path, "0" * 64, expected_dates=expected_dates)
    assert len(accesses) == read_count  # Bad bytes are rejected before NPZ parsing.
    with pytest.raises(ValueError, match="dates mismatch"):
        audit.load_validation_data(path, expected_hash, expected_dates=["2024-07-08", "2024-07-10"])

    original_inputs = ROOT / audit.OUTPUT / "inputs"
    copied_inputs = tmp_path / "inputs"
    shutil.copytree(original_inputs, copied_inputs)
    audit.load_result_inputs(copied_inputs)
    prediction = copied_inputs / "full_validation_predictions.csv"
    prediction.write_bytes(prediction.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="size/hash mismatch"):
        audit.load_result_inputs(copied_inputs)


def test_comparison_allows_log_ulps_but_keeps_counts_hashes_and_dates_strict():
    generated = {"n": 116, "npz_sha256": "a" * 64,
                 "models": {"full": {"original_rmse": 0.006, "below_floor_count": 32}}}
    saved = json.loads(json.dumps(generated))
    saved["models"]["full"]["original_rmse"] = float(np.nextafter(0.006, np.inf))
    audit.compare_saved("metrics.json", json.dumps(saved).encode(), json.dumps(generated).encode())
    for field, value in (("n", 115), ("npz_sha256", "b" * 64)):
        invalid = {**saved, field: value}
        with pytest.raises(ValueError):
            audit.compare_saved("metrics.json", json.dumps(invalid).encode(), json.dumps(generated).encode())
    saved["models"]["full"]["below_floor_count"] = 31
    with pytest.raises(ValueError):
        audit.compare_saved("metrics.json", json.dumps(saved).encode(), json.dumps(generated).encode())
    changed_metric = json.loads(json.dumps(generated))
    changed_metric["models"]["full"]["original_rmse"] = 0.0061
    with pytest.raises(ValueError, match="numeric value differs"):
        audit.compare_saved("metrics.json", json.dumps(changed_metric).encode(), json.dumps(generated).encode())
    expected_csv = b"model,date,absolute_error\nfull,2024-07-08,0.006\n"
    altered_csv = b"model,date,absolute_error\nfull,2024-07-09,0.006\n"
    with pytest.raises(ValueError, match="identity differs"):
        audit.compare_saved("detail.csv", altered_csv, expected_csv)
