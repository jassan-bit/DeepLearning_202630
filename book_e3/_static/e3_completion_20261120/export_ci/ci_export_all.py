"""Run the two strict CPU recreation wrappers and verify all nine artifacts.

This is a release gate, not a training campaign. Upload only --output after this
command succeeds. A changed ONNX SHA fails even if numerical parity is close.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED = {"FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
            "TimesNet", "TiDE", "TimeMixer", "PatchTST", "iTransformer"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args):
    work, output = args.workdir.resolve(), args.output.resolve()
    require(work != output and work not in output.parents and output not in work.parents,
            "Private downloads and public outputs must have separate roots")
    require(not output.exists() or not any(output.iterdir()), "Use a new/empty release output directory")
    require(not sys.flags.optimize, "Python optimization must be off for original guards")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONOPTIMIZE="0")
    groups = (("standard", args.standard_wrapper.resolve(), "standard_export_manifest.json", 4),
              ("temporal", args.temporal_wrapper.resolve(), "temporal_export_manifest.json", 5))
    records = []
    wrappers = {}
    for label, wrapper, manifest_name, expected_count in groups:
        require(wrapper.is_file(), "Missing strict wrapper: " + str(wrapper))
        wrappers[label] = {"script_sha256": sha(wrapper), "manifest": label + "/" + manifest_name}
        subprocess.run([sys.executable, str(wrapper), "--workdir", str(work / label), "--output", str(output / label)], env=env, check=True)
        group_path = output / label / manifest_name
        group = json.loads(group_path.read_bytes())
        require(len(group["models"]) == expected_count and group["new_training_runs"] == 0 and group["new_calibration_fits"] == 0,
                "Wrapper group does not satisfy exact model count / zero-fitting gate")
        require(not group["private_downloads_included"], "Private downloads were included")
        wrappers[label]["manifest_sha256"] = sha(group_path)
        for row in group["models"]:
            directory = output / label / row["directory"]
            require(row["cases"] == 661 and row["absolute_tolerance_sigma7"] == 1e-6 and row["new_training_runs"] == 0 and row["new_calibration_fits"] == 0,
                    "Real 608+53 parity / fixed tolerance / zero-fitting record failed")
            require(sha(directory / "model.onnx") == row["onnx_sha256"] and (directory / "model.onnx").stat().st_size == row["onnx_bytes"], "ONNX artifact differs from strict wrapper manifest")
            require(sha(directory / "metadata.json") == row["metadata_sha256"] and sha(directory / "roundtrip.json") == row["roundtrip_sha256"], "Post-validation metadata or parity report changed")
            records.append({**row, "directory": label + "/" + row["directory"]})
    require(len(records) == 9 and {r["model"] for r in records} == EXPECTED, "Exactly the nine missing neural exports are required")
    for path in output.rglob("*"):
        if not path.is_file():
            continue
        require(path.suffix.lower() not in {".pt", ".pth", ".npz", ".npy", ".zip"}, "Private checkpoint/data/archive in public outputs: " + path.name)
        require("inputs_private" not in path.parts and "author_sources" not in path.parts, "Private input/source directory in public outputs")
    result = {"schema_version": 1, "status": "nine_accepted_ONNX_hashes_and_real_parity_passed",
        "created_utc": datetime.now(timezone.utc).isoformat(), "models": records, "wrappers": wrappers,
        "new_training_runs": 0, "new_calibration_fits": 0, "new_forecast_dates": 0,
        "public_outputs_include_private_downloads": False,
        "scope": "Recreation of existing checkpoint exports and historical CPU inference parity only. No training, variant promotion, independent human reproduction or R1/R3 accreditation.",
        "AI_assistance_declared": True}
    (output / "nine_export_release_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "models": 9, "new_training_runs": 0, "release_manifest": str(output / "nine_export_release_manifest.json")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--standard-wrapper", type=Path, required=True)
    p.add_argument("--temporal-wrapper", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
