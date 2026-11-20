"""Retrieve only immutable existing checkpoints and saved predictions.

No GitHub token is needed. Destination is private and excluded from the
publishable payload; no data acquisition, training or recalibration occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def retrieve(url, path, expected, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "E3-existing-checkpoint-export/1.0"}), timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != expected or (size is not None and len(data) != size):
        raise RuntimeError("Immutable download hash/size mismatch: " + path.name)
    if not path.exists():
        path.write_bytes(data)
    print(json.dumps({"file": str(path), "sha256": expected, "bytes": len(data), "verified": True}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    for row in manifest["checkpoints"]:
        retrieve(row["url"], args.destination / row["path"], row["sha256"], row["bytes"])
    pred = manifest["saved_predictions"]
    retrieve(pred["url"], args.destination / "combined_predictors_133.json", pred["sha256"], pred["bytes"])
    (args.destination / "checkpoint_assets.json").write_text(json.dumps(manifest["checkpoints"], indent=2) + "\n")
    # Stage the byte-identical original scaler/calibrator and selection receipts
    # already included with each export. No new fit/calibration is needed.
    for row in manifest["models"]:
        source = args.manifest.parent / row["slug"]
        target = args.destination / "freeze/final-fit" / row["key"] / "271801"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("scaler.json", "calibrator.json"):
            data = (source / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["original_files_sha256"][name]:
                raise RuntimeError("Frozen original input hash mismatch: " + row["model"] + "/" + name)
            (target / name).write_bytes(data)
        selections = args.destination / "selection_receipts"
        selections.mkdir(parents=True, exist_ok=True)
        data = (source / "selection_receipt.json").read_bytes()
        if hashlib.sha256(data).hexdigest() != row["original_files_sha256"]["selection_receipt.json"]:
            raise RuntimeError("Frozen selection receipt hash mismatch")
        (selections / (row["selection"] + ".json")).write_bytes(data)


if __name__ == "__main__":
    main()
