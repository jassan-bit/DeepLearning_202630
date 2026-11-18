"""Reproduce serving parity against all frozen test origins without Torch.

Uses the bundled fixed representative forecasts and a pinned input archive;
does not fit, select or calibrate a model. The output is an explicit new report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from e3_dashboard import inference
from e3_dashboard.data import load_bundle

NPZ_SHA = "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683"
NPZ_URL = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/af3f1223ad5899ed63016d2af2d09bae8470ab00/book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz"


def run(path, download=False):
    if not path.is_file():
        if not download:
            raise ValueError("The pinned cohort is missing; use --download or supply --npz.")
        with urlopen(NPZ_URL, timeout=60) as response:
            raw = response.read(32 * 1024 * 1024 + 1)
        if len(raw) > 32*1024*1024 or hashlib.sha256(raw).hexdigest() != NPZ_SHA:
            raise ValueError("Pinned cohort download failed verification")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    if hashlib.sha256(path.read_bytes()).hexdigest() != NPZ_SHA:
        raise ValueError("Pinned cohort bytes changed")
    bundle = load_bundle()
    with np.load(path, allow_pickle=False) as arrays:
        raw = np.asarray(arrays["evaluation_raw"], dtype=np.float64)
        dates = [str(x) for x in arrays["evaluation_dates"]]
        y = np.asarray(arrays["evaluation_y_original"], dtype=np.float64)
    assert dates == bundle["predictions"]["dates"]
    assert np.array_equal(y, bundle["predictions"]["targets"])
    records = []
    for item in inference.available_models():
        model_id = item["id"]
        meta, folder = inference._metadata(model_id)
        with inference._LOCK:
            session = inference._session(model_id, meta, folder)
            outputs = np.asarray([inference._run(window, meta, session) for window in raw])
        frozen = np.asarray(bundle["predictions"]["models"][model_id]["predictions"])
        maximum = float(np.max(np.abs(outputs-frozen)))
        assert maximum <= 1e-6 and np.isfinite(outputs).all() and (outputs >= 0).all()
        rmse = float(np.sqrt(np.mean((outputs-y)**2)))
        records.append({"model":model_id,"seed":271801,"dates":608,
                        "maximum_absolute_error_vs_frozen":maximum,
                        "absolute_tolerance_original_sigma7":1e-6,"RMSE_served":rmse,
                        "RMSE_frozen":float(np.sqrt(np.mean((frozen-y)**2))),
                        "model_sha256":meta["model_sha256"]})
    assert len(records) == 4
    assert "torch" not in sys.modules
    assert not any(x.startswith("src.e3_") for x in sys.modules)
    inference.clear_cache()
    return {"checked_at_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
            "models":records,"npz_sha256":NPZ_SHA,"Torch_imported":False,
            "new_training_runs":0,"new_calibration_fits":0,"GPU_used":False,
            "scope":"Serving parity for fixed seed 271801 on the known 608-date cohort, not a new superiority test"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz",type=Path,required=True)
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    record=run(args.npz,args.download)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes((json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps({"models":4,"dates_each":608,"Torch_imported":False,"parity":"verified"}))
