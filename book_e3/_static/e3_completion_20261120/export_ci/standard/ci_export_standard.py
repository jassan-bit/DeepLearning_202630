"""Recreate four ONNX artifacts on CPU from immutable public inputs; never fit.

Only --output may be uploaded. --workdir contains existing raw windows, source
modules, checkpoints and logs and must remain private/temporary. SHA mismatch
fails the job; this script never promotes a numerically similar new binary.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
NAMES = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fetch(url, path, expected, size=None):
    if path.exists():
        require(sha(path) == expected, "Cached input hash mismatch: " + str(path))
        require(size is None or path.stat().st_size == size, "Cached input size mismatch")
        return {"url": url, "sha256": expected, "bytes": path.stat().st_size, "status": "verified_cache"}
    req = urllib.request.Request(url, headers={"User-Agent": "E3-frozen-CPU-export/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read(20_000_000)
    require(hashlib.sha256(data).hexdigest() == expected, "Downloaded input SHA mismatch: " + url)
    require(size is None or len(data) == size, "Downloaded input size mismatch: " + url)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download-part")
    temporary.write_bytes(data)
    temporary.replace(path)
    print(json.dumps({"event": "public_input_verified", "file": path.name, "bytes": len(data), "sha256": expected}), flush=True)
    return {"url": url, "sha256": expected, "bytes": len(data), "status": "downloaded_verified"}


def run(args):
    work, output = args.workdir.resolve(), args.output.resolve()
    require(work != output and work not in output.parents and output not in work.parents,
            "Keep private workdir and public output in separate directory trees")
    require(not sys.flags.optimize, "Run without Python optimization: all original guards must remain active")
    manifest_path = HERE / "standard_source_input_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    require(platform.python_version() == manifest["required_python"], "Python version differs from fixed CPU export profile")
    versions = {name: version(name) for name in manifest["required_packages"]}
    require(versions == manifest["required_packages"], "Installed packages differ from fixed CPU export profile: " + json.dumps(versions))
    export_script = HERE / "export_standard.py"
    require(sha(export_script) == manifest["export_script_sha256"], "Export-only CLI source changed")
    work.mkdir(parents=True, exist_ok=True)
    source, selection, freeze, checkpoints = [work / name for name in ("recipe", "selection", "freeze", "checkpoints")]
    base = "https://raw.githubusercontent.com/" + manifest["repository"] + "/" + manifest["recipe_source_commit"] + "/"
    receipts = []
    for relative, expected in manifest["source_code_hashes"].items():
        receipts.append(fetch(base + relative, source / relative, expected))
    extensions = {"cohort": ".npz", "predictions": ".json", "metadata_zip": ".zip"}
    for name, row in manifest["auxiliary_inputs"].items():
        receipts.append(fetch(row["url"], work / (name + extensions[name]), row["sha256"], row["bytes"]))
    for row in manifest["checkpoints"]:
        receipts.append(fetch(row["selection_url"], selection / (row["selection"] + ".json"), row["selection_sha256"]))
        receipts.append(fetch(row["url"], checkpoints / row["path"], row["sha256"], row["bytes"]))
    with zipfile.ZipFile(work / "metadata_zip.zip") as archive:
        for row in manifest["checkpoints"]:
            for kind in ("scaler", "calibrator"):
                relative = row[kind + "_relative_path"]
                data = archive.read(relative)
                require(hashlib.sha256(data).hexdigest() == row[kind + "_sha256"], "Original frozen " + kind + " changed")
                target = freeze / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    write(checkpoints / "checkpoint_assets.json", [{k: row[k] for k in ("model", "asset_id", "bytes", "name", "path", "release_tag", "sha256", "storage", "url")} for row in manifest["checkpoints"]])
    write(work / "download_receipts.json", receipts)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONOPTIMIZE="0")
    if args.prepare_only:
        # This verifies only source import/constructor availability. It does not
        # invoke export, infer model outputs, optimize, backpropagate or calibrate.
        probe = """import sys,json,torch
from pathlib import Path
repo,sel=sys.argv[1:];sys.path.insert(0,str(Path(repo)/'scripts'))
from e2_reduced_fit import _builder
torch.set_num_threads(1)
for p in sorted(Path(sel).glob('*.json')):
 d=json.loads(p.read_text());m=_builder(d['model_id'],d['winner_trial'],271801,{}).eval()
 assert all(v.device.type=='cpu' for v in m.parameters())
 print(d['model_id']+': source constructor available; no fitting')
"""
        subprocess.run([sys.executable, "-c", probe, str(source), str(selection)], check=True, env=env)
        write(work / "prepare_only_receipt.json", {"status": "public_inputs_and_source_constructors_verified", "versions": versions, "new_training_runs": 0, "new_model_forward_calls": 0, "ONNX_reexport_executed": False})
        return
    require(not output.exists() or not any(output.iterdir()), "Public output must be new/empty; preserve earlier artifacts")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in NAMES:
        if args.models and name not in args.models:
            continue
        row = next(r for r in manifest["checkpoints"] if r["model"] == name)
        command = [sys.executable, str(export_script), "--model", name,
            "--repo", str(source), "--checkpoints", str(checkpoints),
            "--npz", str(work / "cohort.npz"), "--predictions", str(work / "predictions.json"),
            "--protocol", str(source / "book_e2/_static/search_protocol_20261109/protocol.json"),
            "--selections", str(selection), "--freeze", str(freeze), "--output", str(output)]
        with (work / (row["slug"] + ".log")).open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        require(result.returncode == 0, name + ": export/parity failed; private log retained in " + row["slug"] + ".log")
        directory = output / row["slug"]
        require(sha(directory / "model.onnx") == row["expected_onnx_sha256"], name + ": ONNX bytes differ from accepted artifact; do not publish. Use verified 128-KiB parts fallback if required.")
        require((directory / "model.onnx").stat().st_size == row["expected_onnx_bytes"], "ONNX byte count changed")
        report = json.loads((directory / "roundtrip.json").read_bytes())
        require(report["within_tolerance"] and report["all_real_windows_verified"] == 661 and report["new_training_runs"] == 0 and report["new_calibration_fits"] == 0, "Real frozen parity or zero-fit contract failed")
        records.append({"model": name, "directory": row["slug"], "checkpoint_sha256": row["sha256"],
            "onnx_sha256": row["expected_onnx_sha256"], "onnx_bytes": row["expected_onnx_bytes"],
            "roundtrip_sha256": sha(directory / "roundtrip.json"), "metadata_sha256": sha(directory / "metadata.json"),
            "cases": 661, "absolute_tolerance_sigma7": 1e-6, "new_training_runs": 0, "new_calibration_fits": 0})
        print(name + ": accepted ONNX SHA and 608+53 real parity verified; zero fitting", flush=True)
    write(output / "standard_export_manifest.json", {"schema_version": 1,
        "scope": "CPU artifact recreation and real frozen inference parity; no fitting, model promotion or R1/R3 accreditation",
        "created_utc": datetime.now(timezone.utc).isoformat(), "models": records,
        "recipe_source_commit": manifest["recipe_source_commit"], "source_input_manifest_sha256": sha(manifest_path),
        "export_script_sha256": sha(export_script), "python": platform.python_version(), "packages": versions,
        "new_training_runs": 0, "new_calibration_fits": 0, "private_downloads_included": False,
        "AI_assistance_declared": True})
    print(json.dumps({"status": "passed", "models": len(records), "output": str(output), "new_training_runs": 0}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--models", nargs="*", choices=NAMES)
    p.add_argument("--prepare-only", action="store_true")
    run(p.parse_args())
