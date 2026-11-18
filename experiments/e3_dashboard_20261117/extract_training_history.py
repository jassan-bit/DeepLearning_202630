"""Extract the already completed seed-271801 history; never fit or infer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CHECKPOINT_SHA = "4d440b71db042b916bc4daa95031bd222fd1c3c7a94974cff9dca7b43d9df5c9"
SOURCE_COMMIT = "510c3bddfdc9926322e3210ac61ee1f5eeffad2e"


def extract(checkpoint_path, destination):
    checkpoint_path, destination = Path(checkpoint_path), Path(destination)
    raw = checkpoint_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CHECKPOINT_SHA:
        raise ValueError("Frozen checkpoint hash differs")
    import torch  # Scientific extraction only; never imported by the dashboard.
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert cp["seed"] == 271801 and cp["complete"] and cp["best_epoch"] == 38
    assert len(cp["history"]) == 40
    history = []
    for expected_epoch, row in enumerate(cp["history"], 1):
        assert row["epoch"] == expected_epoch
        train, validation = float(row["training_loss"]), float(row["validation_RMSE"])
        assert math.isfinite(train) and train >= 0 and math.isfinite(validation) and validation >= 0
        history.append({"epoch": expected_epoch, "training_loss_normalized": train,
                        "validation_RMSE_original": validation})
    assert min(history, key=lambda row: row["validation_RMSE_original"])["epoch"] == cp["best_epoch"]
    document = {"schema_version": 1, "model": "CNN-LSTM-atencion-aditiva E3", "seed": 271801,
                "best_epoch": cp["best_epoch"], "history": history,
                "source": {"commit": SOURCE_COMMIT, "checkpoint_sha256": CHECKPOINT_SHA,
                           "path": "runs/estabilidad_aditiva_v1/seeds/271801/checkpoint.pt"},
                "units": {"training_loss_normalized": "mean normalized MSE accumulated during optimizer updates, not full-fit RMSE",
                          "validation_RMSE_original": "RMSE on 116 development validation dates in original sigma7"},
                "scope": "Historical per-epoch trace of one fixed seed; not a sample-size learning curve or a train-validation gap test",
                "new_training_runs": 0, "new_model_inferences": 0}
    encoded = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    manifest = {"schema_version": 1, "history_sha256": hashlib.sha256(encoded).hexdigest(),
                "history_bytes": len(encoded), "source_checkpoint_sha256": CHECKPOINT_SHA,
                "source_commit": SOURCE_COMMIT, "epochs": 40,
                "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in (("history.json", encoded), ("manifest.json", (json.dumps(manifest,indent=2)+"\n").encode())):
        path = destination / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Refusing to overwrite different historical data: " + name)
        path.write_bytes(payload)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.checkpoint, args.output)))
