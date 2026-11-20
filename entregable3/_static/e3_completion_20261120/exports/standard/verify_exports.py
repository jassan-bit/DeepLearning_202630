"""Verify hashes, byte-identical frozen metadata and external input guards.

Uses ORT CPU only. The full 608+53 roundtrip was executed by export_standard.py;
this is the lightweight packaging/guard check and does not repeat that analysis.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from predict_export import FrozenExport


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.payload / "input_manifest.json").read_bytes())
    if sha(args.npz) != manifest["cohort_npz"]["sha256"]:
        raise ValueError("Private original cohort hash mismatch")
    with np.load(args.npz, allow_pickle=False) as z:
        raw = np.asarray(z["evaluation_raw"][:1], dtype=np.float64)[0]
    cases = {"wrong_shape": raw[:6].copy(), "nonfinite": raw.copy(),
             "zero_price": raw.copy(), "negative_volume": raw.copy(), "inconsistent_OHLC": raw.copy()}
    cases["nonfinite"][0, 0] = np.nan
    cases["zero_price"][0, 0] = 0
    cases["negative_volume"][0, 4] = -1
    cases["inconsistent_OHLC"][0, 1] = .5 * min(raw[0, 0], raw[0, 3])
    rows = []
    for spec in manifest["models"]:
        directory = args.payload / spec["slug"]
        metadata = json.loads((directory / "metadata.json").read_bytes())
        roundtrip = json.loads((directory / "roundtrip.json").read_bytes())
        if sha(directory / "model.onnx") != metadata["model_sha256"] or not roundtrip["within_tolerance"]:
            raise ValueError("Export integrity check failed")
        for name, expected in spec["original_files_sha256"].items():
            if sha(directory / name) != expected:
                raise ValueError("Original frozen file bytes changed: " + spec["model"] + "/" + name)
        model = FrozenExport(directory)
        point = model.predict(raw)
        rejects = {}
        for name, invalid in cases.items():
            try:
                model.predict(invalid)
            except ValueError:
                rejects[name] = True
            else:
                raise ValueError("External guard failed: " + name)
        rows.append({"model": spec["model"], "seed": 271801,
                     "original_scaler_calibrator_and_selection_bytes_verified": True,
                     "ONNX_hash_verified": True, "all_five_invalid_inputs_rejected_before_inference": rejects,
                     "valid_original_window_inference_passed": True, "valid_window_prediction_sigma7": point["prediction_sigma7"],
                     "roundtrip_reference": spec["slug"] + "/roundtrip.json", "new_training_runs": 0})
    result = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "models": rows,
              "provider": "CPUExecutionProvider", "threads_intra_inter": [1, 1], "new_fits": 0,
              "scope": "Artifact integrity + external guards + one real-window inference each; full 608+53 parity is separately recorded in roundtrip.json."}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "models": len(rows), "invalid_cases_each": len(cases), "new_fits": 0}))


if __name__ == "__main__":
    main()
