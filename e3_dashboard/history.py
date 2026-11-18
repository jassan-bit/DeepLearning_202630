"""Read the hash-checked historical epoch trace without training dependencies."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

DEFAULT_HISTORY_DIR = Path(__file__).resolve().parents[1] / "data/e3_dashboard_training_v1"


def load_training_history(folder=None):
    folder = Path(folder or DEFAULT_HISTORY_DIR)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    raw = (folder / "history.json").read_bytes()
    if (manifest.get("schema_version") != 1 or len(raw) != manifest.get("history_bytes") or
            hashlib.sha256(raw).hexdigest() != manifest.get("history_sha256")):
        raise ValueError("La curva histórica no coincide con su huella de integridad.")
    document = json.loads(raw)
    rows = document.get("history", [])
    if (document.get("model") != "CNN-LSTM-atencion-aditiva E3" or document.get("seed") != 271801 or
            document.get("best_epoch") != 38 or len(rows) != 40 or
            [row.get("epoch") for row in rows] != list(range(1,41)) or
            document.get("source", {}).get("checkpoint_sha256") != manifest.get("source_checkpoint_sha256")):
        raise ValueError("El historial no corresponde al predictor y las épocas documentadas.")
    for row in rows:
        for key in ("training_loss_normalized", "validation_RMSE_original"):
            value = row.get(key)
            if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
                raise ValueError("La curva histórica contiene un valor inválido.")
    if min(rows, key=lambda row: row["validation_RMSE_original"])["epoch"] != 38:
        raise ValueError("La mejor época histórica no coincide con el registro.")
    return document
