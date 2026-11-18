"""Build a small dashboard bundle solely from pinned saved data and predictions."""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
NPZ_SOURCE = {"commit": "af3f1223ad5899ed63016d2af2d09bae8470ab00",
              "path": "book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz",
              "sha256": "186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683", "bytes": 460090}
BOOK_SOURCES = {
    "metrics": ("book_e3/_static/e3_revision_bloque_d_20261117/metrics_extended.csv", "2a3bd6e5921a977ea5aa91ad9bced23968bad235cbe7925680d9f76091f1632c"),
    "uncertainty": ("book_e3/_static/e3_revision_bloque_d_20261117/metrics_uncertainty.csv", "3f9a7a02757b372c70295d3b5f78c35a729cda3fcf452f0bd64da194fcc872ea"),
    "resources": ("book_e3/_static/e3_bloques_ab_20261115/deployment_master.csv", "d850dd6487a898fa4f7793a1f4f72acb7be1e9a0950051b8e69eea2be5370ee7"),
    "mcs": ("book_e3/_static/e3_estadistica_posterior_20261117/mcs_table.csv", "e3f5f1e9896f224fbc6f51d4356654ee64a93c403ae49483fab00d29e5c54d89"),
    "contrasts": ("book_e3/_static/e3_estadistica_posterior_20261117/primary_contrasts.csv", "218ec87857902bab24757295375949e947e36088397439d68b334897e9cdb64c"),
}
PARTITIONS = ("training", "calibration", "evaluation")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def read_book_sources():
    tables, records = {}, {}
    for name, (relative, expected) in BOOK_SOURCES.items():
        raw = (ROOT / relative).read_bytes()
        require(sha(raw) == expected, "Book source differs: " + relative)
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode())))
        records[relative] = {"sha256": expected, "bytes": len(raw), "commit": "b3b7f5210e8a81fe3e0f35493221bcb0f6f8c4e4"}
    return tables, records


def frozen_validator():
    path = ROOT / "experiments/e3_estadistica_posterior_20261117/inputs.py"
    spec = importlib.util.spec_from_file_location("frozen_dashboard_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric(value):
    return None if value in ("", None) else float(value)


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def build(cohort_path, input_dir):
    lib = frozen_validator()
    documents = lib.read_documents(input_dir)
    panel = lib.validate_documents(documents)
    cohort_raw = Path(cohort_path).read_bytes()
    require(len(cohort_raw) == NPZ_SOURCE["bytes"] and sha(cohort_raw) == NPZ_SOURCE["sha256"], "Audited cohort hash differs")
    tables, source_records = read_book_sources()
    resources = {r["model"]: r for r in tables["resources"]}
    uncertainty = {r["model"]: r for r in tables["uncertainty"]}
    catalogue = []
    metric_fields = ("RMSE_mean", "RMSE_sample_SD", "MAE_mean", "sMAPE_percent", "R2_mean", "coverage90_percent", "mean_interval_width90")
    for row in tables["metrics"]:
        model = row["model"]
        family = "E3 · CNN-LSTM con atención" if model == lib.E3_MODEL else (
            "Top 10 · E2" if model in lib.NEURAL[:10] else "Baseline neuronal" if model in lib.NEURAL else "Baseline clásico")
        current = {"model": model, "family": family, "R": int(row["R"]), "calendar_n": int(row["calendar_n"]),
                   "interval_status": row["interval_status"], "live_status": "precomputed"}
        current.update({field: numeric(row[field]) for field in metric_fields})
        current.update(RMSE_seed_ci95_lower=numeric(uncertainty[model]["RMSE_seed_bootstrap_CI95_low"]),
                       RMSE_seed_ci95_upper=numeric(uncertainty[model]["RMSE_seed_bootstrap_CI95_high"]))
        resource = resources.get(model, {})
        for field in ("parameters", "core_flops_min", "core_flops_max", "cpu_latency_median_ms", "cpu_latency_p95_ms", "cpu_peak_memory_bytes"):
            current[field] = 1210 if field == "parameters" and model == lib.E3_MODEL else numeric(resource.get(field))
        current["resource_scope"] = ("E2 historical handoff: eager/core FLOPs and three TorchScript CPU profiles; conditions differ, not final common-cost comparison"
                                     if resource else "E3 parameter count verified; comparable CPU costs not supplied by this data bundle")
        catalogue.append(current)
    require({r["model"] for r in catalogue} == set(panel["model_order"]), "Catalogue differs from frozen family")
    predictions = {"dates": panel["dates"], "targets": panel["targets"].tolist(), "known_floor": panel["known_floor"].tolist(), "models": {}}
    by_coordinate = {(r["model"], r["seed"]): r for r in documents["combined"]["predictors"]}
    for model in panel["model_order"]:
        seed = 271801 if model not in lib.CLASSICAL else "deterministic"
        idx = list(panel["seeds"][model]).index(seed)
        representative = panel["predictions"][model][idx].tolist()
        q = None if model == lib.E3_MODEL else by_coordinate[model, seed]["calibrator_q"]
        predictions["models"][model] = {"seed": seed, "predictions": representative, "q": q,
            "representative_policy": "lowest_registered_seed_not_selected_by_test",
            "RMSE_representative": lib.metrics(panel["targets"], representative)["RMSE"],
            "mean_metrics_scope": "catalogue averages individual predictors; this curve is one fixed representative"}
    observations, periods, examples = [], {}, []
    with np.load(io.BytesIO(cohort_raw), allow_pickle=False) as arrays:
        for partition in PARTITIONS:
            x, y, dates = arrays[partition + "_raw"], arrays[partition + "_y_original"], arrays[partition + "_dates"]
            expected_n = {"training": 655, "calibration": 53, "evaluation": 608}[partition]
            require(x.shape == (expected_n, 7, 5) and len(y) == len(dates) == expected_n, "Wrong cohort shape")
            require(np.isfinite(x).all() and np.isfinite(y).all() and (y >= 0).all(), "Invalid cohort numbers")
            require((x[:, :, :4] > 0).all() and (x[:, :, 4] >= 0).all(), "Invalid price/volume")
            require((x[:, :, 2] <= x[:, :, [0, 1, 3]].min(axis=2)).all() and
                    (x[:, :, 1] >= x[:, :, [0, 2, 3]].max(axis=2)).all(), "Invalid OHLC coherence")
            periods[partition] = {"n": expected_n, "start": str(dates[0]), "end": str(dates[-1])}
            for window, target, day in zip(x, y, dates, strict=True):
                target_day = date.fromisoformat(str(day))
                returns = np.diff(np.log(window[:, 3]))
                floor = float(np.sqrt(np.square(returns - returns.mean()).sum() / 7.0))
                require(float(target) + 1e-12 >= floor, "Target violates known floor")
                row = {"target_date": target_day.isoformat(), "feature_date": (target_day - timedelta(days=1)).isoformat(),
                       "partition": partition, "target": float(target), "known_return6": float(returns[-1]), "known_floor": floor}
                row.update({"last_" + key: float(value) for key, value in zip(("open", "high", "low", "close", "volume"), window[-1], strict=True)})
                observations.append(row)
            if partition == "evaluation":
                require(list(map(str, dates)) == predictions["dates"] and y.tolist() == predictions["targets"], "Evaluation cohort and predictions differ")
                for index in (0, 303, 607):
                    target_day = date.fromisoformat(str(dates[index]))
                    rows = [{"date": (target_day - timedelta(days=7-i)).isoformat(), **{key: float(value) for key, value in zip(("open", "high", "low", "close", "volume"), bar, strict=True)}} for i, bar in enumerate(x[index])]
                    raw = csv_bytes(rows, ["date", "open", "high", "low", "close", "volume"])
                    examples.append({"id": "evaluation-" + str(index), "label": "Ventana histórica · objetivo " + target_day.isoformat(),
                        "target_date": target_day.isoformat(), "csv_text": raw.decode(), "csv_sha256": sha(raw),
                        "source_npz_sha256": NPZ_SOURCE["sha256"], "selection_policy": "fixed_chronological_positions_0_303_607_not_performance"})
    require(len({r["target_date"] for r in observations}) == len(observations) == 1316, "Duplicate/missing target dates")
    retained = [r["Modelo"] for r in tables["mcs"] if r["MCS90%"] == "Permanece"]
    e3 = next(r for r in catalogue if r["model"] == lib.E3_MODEL)
    metadata = {
        "schema_version": 1, "version": "e3-dashboard-data-v1", "cut_date_local": "2026-11-17", "timezone": "America/Bogota",
        "book_url": "https://jassan-bit.github.io/DeepLearning_202630/entregable3/intro.html",
        "counts": {"observations": 1316, "training": 655, "calibration": 53, "evaluation": 608,
                   "systems": 17, "saved_predictors": 143, "representative_curves": 17}, "periods": periods,
        "data_card": {"source": "Binance Spot BTCUSDT; audited saved OHLCV cohort", "target": "Population sigma7 of seven daily Close log returns; ddof=0; unannualized",
                      "window": "7 observed daily OHLCV bars through t-1", "horizon_days": 1, "time_basis": "UTC",
                      "variables": ["open", "high", "low", "close", "volume"], "personal_data": "none in these public market measurements",
                      "license_status": "Source terms and final redistribution statement require review; no new data license is asserted",
                      "feature_dates": "Each row is a target-date observation. Last OHLCV values belong to t-1, not the target day.",
                      "known_return6": "last of the six observed log returns in the seven input closes",
                      "cohort_scope": "E2 training/calibration/evaluation partitions; E3 fitted on the first 532 training examples and validated on 116",
                      "known_calendar": True, "additional_datasets": "not selected or evaluated"},
        "quality": {"nonfinite_values": 0, "duplicate_target_dates": 0, "invalid_ohlc_windows": 0,
                    "raw_window_count": 1316, "continuous_unique_bar_series_claimed": False,
                    "audit_controls": {"PASS": 12, "PARTIAL": 3, "NA": 1},
                    "scope": "Saved audited windows and dates; source hash and schema checks do not close every leakage control"},
        "model_card": {"model": lib.E3_MODEL, "architecture": "Causal CNN -> unidirectional LSTM -> temporal content attention -> positive additive output",
                       "parameters": 1210, "training_n": 532, "validation_n": 116, "calibration_n": 0,
                       "seeds": list(lib.SEEDS), "RMSE_mean": e3["RMSE_mean"], "MAE_mean": e3["MAE_mean"], "sMAPE_percent": e3["sMAPE_percent"], "R2_mean": e3["R2_mean"],
                       "use": "Academic exploration of historical one-day volatility forecasts",
                       "limitations": ["Known historical calendar and unequal training policies", "No E3 calibrated predictive interval", "Novelty, academic defense and complete R3 remain unaccredited", "Additional datasets, robust ablations and causal explanation remain incomplete"],
                       "AI_assistance": "Declared in code and report; not a human academic endorsement"},
        "mcs": tables["mcs"], "mcs_retained": retained, "contrasts": tables["contrasts"],
        "statistical_scope": "Retrospective 48-contrast Holm family; auxiliary MCS90 on another loss; not R3 certification",
        "curve_scope": "Representative seed 271801 fixed by ID; catalogue averages ten individual predictors per neural model",
    }
    observation_fields = ["target_date", "feature_date", "partition", "target", "last_open", "last_high", "last_low", "last_close", "last_volume", "known_return6", "known_floor"]
    files = {"catalogue.json": json_bytes(catalogue), "observations.csv": csv_bytes(observations, observation_fields),
             "predictions.json": json_bytes(predictions), "examples.json": json_bytes(examples), "metadata.json": json_bytes(metadata)}
    manifest = {"schema_version": 1, "version": "e3-dashboard-data-v1", "source_commit": "b3b7f5210e8a81fe3e0f35493221bcb0f6f8c4e4",
                "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()},
                "sources": {"cohort_npz": NPZ_SOURCE, "saved_predictions": lib.SOURCES, "book_tables": source_records},
                "builder_sha256": sha(Path(__file__).read_bytes()), "new_model_fits": 0, "new_model_inferences": 0, "new_hypothesis_tests": 0}
    files["manifest.json"] = json_bytes(manifest)
    report = {"version": 1, "files": manifest["files"], "counts": metadata["counts"], "periods": periods,
              "representative_seed": 271801, "representative_not_selected_by_test": True, "input_validation": panel["provenance"],
              "cohort_npz_sha256": NPZ_SOURCE["sha256"], "bundle_size_bytes": sum(map(len, files.values())),
              "book_metrics_exactly_reused": True, "new_model_fits": 0, "new_model_inferences": 0, "new_hypothesis_tests": 0}
    return files, report


def acquire(cohort_path, input_dir):
    frozen_validator().download_inputs(input_dir)
    cohort_path = Path(cohort_path)
    if not cohort_path.exists():
        url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + NPZ_SOURCE["commit"] + "/" + NPZ_SOURCE["path"]
        with urlopen(Request(url, headers={"User-Agent": "E3DashboardBundle/1.0"}), timeout=60) as response:
            raw = response.read(NPZ_SOURCE["bytes"] + 1)
        require(len(raw) == NPZ_SOURCE["bytes"] and sha(raw) == NPZ_SOURCE["sha256"], "Downloaded cohort differs")
        cohort_path.parent.mkdir(parents=True, exist_ok=True)
        cohort_path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=ROOT / "work/e3_dashboard_inputs/current_cohort_arrays.npz")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "work/e3_statistical_inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "data/e3_dashboard_v1")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.download:
        acquire(args.cohort, args.input_dir)
    files, report = build(args.cohort, args.input_dir)
    if args.check:
        for name, raw in files.items():
            require((args.output / name).read_bytes() == raw, "Dashboard bundle differs: " + name)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, raw in files.items():
            path = args.output / name
            require(not path.exists() or path.read_bytes() == raw, "Refusing to overwrite unequal dashboard bundle: " + name)
            path.write_bytes(raw)
        (Path(__file__).parent / "data_verification.json").write_bytes(json_bytes(report))
    print(json.dumps({"status": "verified" if args.check else "written", "systems": 17, "observations": 1316,
                      "bundle_bytes": sum(map(len, files.values())), "new_model_fits": 0}))


if __name__ == "__main__":
    main()
