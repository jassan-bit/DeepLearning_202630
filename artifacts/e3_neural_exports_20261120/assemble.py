"""Assemble and verify the exact accepted ONNX fallback; no export or fitting.

Parts are ASCII base64, at most 128 KiB each. --output must not exist. The
manifest pins encoded parts, decoded ZIP, every member and all nine ONNX files.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name and bool(path.parts), "Unsafe relative path")
    return path


def run(args):
    manifest_path = args.manifest.resolve()
    source = manifest_path.parent
    output = args.output.resolve()
    require(not output.exists(), "Output must be new: preserve existing artifacts")
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest["schema_version"] == 1 and manifest["part_encoding"] == "base64" and manifest["new_training_runs"] == 0, "Unsupported manifest")
    require(len(manifest["onnx_models"]) == 9, "Exactly nine accepted ONNX artifacts required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e3-accepted-onnx-", dir=output.parent) as temp:
        temporary = Path(temp)
        zip_path = temporary / "accepted.zip"
        with zip_path.open("wb") as archive:
            for expected_index, row in enumerate(manifest["parts"]):
                require(row["index"] == expected_index, "Part order changed")
                relative = safe_name(row["file"])
                path = source.joinpath(*relative.parts)
                require(path.is_file() and not path.is_symlink(), "Missing or symbolic part")
                encoded = path.read_bytes()
                require(len(encoded) == row["encoded_bytes"] <= 128 * 1024 and digest(encoded) == row["encoded_sha256"], "Encoded part hash/size failed")
                decoded = base64.b64decode(encoded, validate=True)
                require(len(decoded) == row["decoded_bytes"] and digest(decoded) == row["decoded_sha256"], "Decoded part hash/size failed")
                archive.write(decoded)
        require(zip_path.stat().st_size == manifest["zip"]["bytes"] and digest(zip_path.read_bytes()) == manifest["zip"]["sha256"], "ZIP hash/size failed")
        staged = temporary / "verified"
        staged.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            infos = archive.infolist()
            require(len(infos) == len(manifest["members"]) and len({i.filename for i in infos}) == len(infos), "ZIP member count/duplicate failure")
            require({i.filename for i in infos} == set(manifest["members"]), "ZIP member inventory differs from manifest")
            for info in infos:
                relative = safe_name(info.filename)
                require(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16) and info.flag_bits & 1 == 0, "Directory/link/encrypted ZIP member rejected")
                row = manifest["members"][info.filename]
                require(info.file_size == row["bytes"] and info.file_size <= 100_000_000, "ZIP member size failed")
                data = archive.read(info)
                require(len(data) == row["bytes"] and digest(data) == row["sha256"], "ZIP member hash failed")
                target = staged.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for row in manifest["onnx_models"]:
            relative = safe_name(row["file"])
            data = staged.joinpath(*relative.parts).read_bytes()
            require(len(data) == row["bytes"] and digest(data) == row["sha256"], "Accepted ONNX identity failed")
            report = json.loads(staged.joinpath(*relative.parts).with_name("roundtrip.json").read_bytes())
            require(report["within_tolerance"] and report["new_training_runs"] == 0 and report["new_calibration_fits"] == 0, "Original parity/zero-fitting receipt failed")
        receipt = {"status": "nine_exact_accepted_artifacts_assembled", "manifest_sha256": digest(manifest_path.read_bytes()), "zip_sha256": manifest["zip"]["sha256"], "onnx_models": manifest["onnx_models"], "all_member_hashes_verified": True, "new_training_runs": 0, "new_calibration_fits": 0, "new_inference_calls": 0, "scope": "Byte-preserving transport of existing accepted artifacts and receipts. No model promotion, export, training or claim about the cause of cross-runner binary differences."}
        (staged / "fallback_assembly_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        shutil.move(str(staged), str(output))
    print(json.dumps({"status": receipt["status"], "models": 9, "output": str(output), "new_training_runs": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().with_name("manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
