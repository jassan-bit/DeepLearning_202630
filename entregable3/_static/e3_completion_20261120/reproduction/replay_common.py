"""Frozen V3 preprocessing and reference access; AI assistance declared.

This module imports no training/model code. It opens only the two named raw
development members, never calibration/test arrays or new outcomes.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ABS_TOL_SIGMA7 = 1e-6
NPZ_SHA256 = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
CHECKPOINT_SHA256 = "67dc01ed9c3ce11a140acc7eddc72e9ad097b10aa9e8a536ef14bcdbc568c844"
SEED = 20261118
PROFILE = "e3-v3-atom-crps-experimental-20261120"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    header = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def preprocess(raw, scaler):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (7, 5) or not np.isfinite(raw).all():
        raise ValueError("Finite batch x 7 x 5 OHLCV required")
    if not (raw[..., :4] > 0).all() or not (raw[..., 4] >= 0).all():
        raise ValueError("Positive prices and nonnegative volume required")
    if not ((raw[...,1] >= np.maximum(raw[...,0], raw[...,3])).all()
            and (raw[...,2] <= np.minimum(raw[...,0], raw[...,3])).all()
            and (raw[...,1] >= raw[...,2]).all()):
        raise ValueError("Invalid OHLC geometry")
    returns = np.diff(np.log(raw[...,3]), axis=1)
    mean6 = returns.mean(axis=1)
    A = ((returns-mean6[:,None])**2).sum(axis=1)/7.0
    relative = np.empty_like(raw)
    relative[...,:4] = np.log(raw[...,:4])-np.log(raw[:,-1,3])[:,None,None]
    relative[...,4] = np.log1p(raw[...,4])
    X = ((relative-np.asarray(scaler['mean']))/np.asarray(scaler['std'])).astype(np.float32)
    if not np.isfinite(X).all() or not np.isfinite(A).all() or not (A >= 0).all():
        raise ValueError("Invalid preprocessed inputs")
    return np.ascontiguousarray(X), np.ascontiguousarray(A, dtype=np.float64)


def references(root):
    with (Path(root)/'reference/assessment_predictions.csv').open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['name']=='atom_crps' and int(r['seed'])==SEED]
    if len(rows)!=55 or len({r['date'] for r in rows})!=55:
        raise ValueError("Expected exactly55 frozen assessment rows for the fixed minimum seed")
    return rows


def development_inputs(root, npz):
    root=Path(root); npz=Path(npz)
    if sha(npz)!=NPZ_SHA256: raise ValueError("Original cohort SHA256 differs")
    admission=json.loads((root/'reference/admission.json').read_text())
    rows=references(root); dates=[r['date'] for r in rows]
    with np.load(npz, allow_pickle=False) as src:
        raw=np.asarray(src['training_raw'], dtype=np.float64).copy()
        ids=np.asarray(src['training_dates']).astype(str).copy()
    if len(set(ids.tolist()))!=len(ids): raise ValueError("Duplicate development date")
    lookup={d:i for i,d in enumerate(ids)}
    if any(d not in lookup for d in dates):raise ValueError("Assessment date absent from raw development")
    selected=raw[[lookup[d] for d in dates]]
    X,A=preprocess(selected,admission['scaler'])
    return selected,X,A,rows,admission


def session_options(ort):
    options=ort.SessionOptions()
    options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options
