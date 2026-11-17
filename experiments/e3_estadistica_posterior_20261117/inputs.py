"""Validate frozen prediction panels for the posterior E3 statistical audit.

No model, optimizer or hypothesis test is imported or run. The historical
``original_analysis`` field is preserved source evidence, never a new result.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from urllib.request import Request, urlopen

import numpy as np

E2_COMMIT = "5e9023ea821bfa9fb0b92340d518413f6ab89161"
E3_COMMIT = "510c3bddfdc9926322e3210ac61ee1f5eeffad2e"
OLD_COMMIT = "2121b048d3b379c2c28dfe366cba35a1c08a3617"
E2_PREFIX = "runs/e2_extension_10_semillas_v1/"
E3_PREFIX = "runs/estabilidad_aditiva_v1/"
E3_MODEL = "CNN-LSTM-atencion-aditiva E3"
E3_LABEL = "CNN-LSTM con atención de contenido y salida aditiva positiva (E3)"
NEURAL = ("FCN-1D", "ResNet-1D", "Inception-1D", "LSTM-autoencoder con predictor",
          "TimesNet", "TSMixer", "TiDE", "TimeMixer", "PatchTST", "iTransformer",
          "MLP16", "LSTM32", "GRU32")
CLASSICAL = ("Persistence", "SeasonalNaive", "ARIMA")
RIVALS = NEURAL + CLASSICAL
MODEL_ORDER = (E3_MODEL,) + RIVALS
SEEDS = tuple(range(271801, 271811))
OLD_SEEDS = SEEDS[:6]
NEW_SEEDS = SEEDS[6:]
HISTORICAL_SEED = 20261116
DATES = tuple((date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608))
E2_PLAN_FINGERPRINT = "5437fb3019a6b56e72b84928156b017b152b59bf1d04e33cfd82319a280c3845"
E3_PLAN_FINGERPRINT = "0b2dbfea9f14f1506f09fd42b8e74fc299a91438b2a6c01fec6e4181fa7afa8c"


def source(commit, path, sha256, size):
    return {"commit": commit, "path": path, "sha256": sha256, "bytes": size}


SOURCES = {
    "combined_predictors.json": source(E2_COMMIT, E2_PREFIX + "combined_predictors.json", "df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655", 3393801),
    "e2_per_seed_metrics.csv": source(E2_COMMIT, E2_PREFIX + "per_seed_metrics.csv", "d53c970cc37d000a0d05d4666adcaa04384e317ccac33b6fb3717f750c650e4e", 12138),
    "e2_comparison.csv": source(E2_COMMIT, E2_PREFIX + "comparison.csv", "1b33b37f1971e1668904828dae07e7522813553d255164ff228052d8201b699a", 1450),
    "e2_summary.json": source(E2_COMMIT, E2_PREFIX + "summary.json", "d033a3cb9a537e555826a1e46d80ba1abf47fafb06dde4d195679262049da55e", 17619),
    "e2_snapshot.json": source(E2_COMMIT, E2_PREFIX + "snapshot.json", "ded84adce601ba908f3634a47171d04e3d0491b65532199cb9ba532e92fc667d", 169594),
    "e2_plan.json": source(E2_COMMIT, E2_PREFIX + "plan.json", "45fb6a14137b03642737521731c4b56a1312980bfbeca0319182a9ffd600cc07", 24226),
    "e3_evaluation_predictions.csv": source(E3_COMMIT, E3_PREFIX + "evaluation_predictions.csv", "b82ba0a8a0371ea5643d9ff666a3ac5a1a4e91018b95a640bd88b1be50bb5133", 486813),
    "e3_per_seed_metrics.csv": source(E3_COMMIT, E3_PREFIX + "per_seed_metrics.csv", "2fd1f3c6ec918d0c4371a3e3cf25864c271fd5250d459c3ef99d96ab7d31b2ff", 1115),
    "e3_summary.json": source(E3_COMMIT, E3_PREFIX + "summary.json", "7c90c800adcca2218ea94553081cbcf744f3b9d9ff657ece17402df40be8f176", 9498),
    "e3_plan.json": source(E3_COMMIT, E3_PREFIX + "plan.json", "57de7c087113163b18fb384ffc73fed5dbf6045b37b42eae43c6261b1f76b148", 6416),
    "old_e2_predictions.json": source(OLD_COMMIT, "runs/f2a-65ae74b547a6073b2c286dea-posthoc/artifacts/e2_f2a_input.json", "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187", 1371654),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def verify_source(path, record):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            "A regular source file is required: " + str(path))
    require(path.stat().st_size == record["bytes"] and file_sha(path) == record["sha256"],
            "Pinned source SHA256 or byte count differs: " + path.name)


def download_inputs(folder):
    """Explicit immutable public-HTTPS acquisition; existing unequal files stop."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, record in SOURCES.items():
        destination = folder / name
        if not destination.exists():
            url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + record["commit"] + "/" + record["path"]
            with urlopen(Request(url, headers={"User-Agent": "E3FrozenPredictionAudit/1.0"}), timeout=60) as response:
                raw = response.read(10 * 1024**2 + 1)
            require(len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
                    "Downloaded immutable source differs: " + name)
            with destination.open("xb") as stream:
                stream.write(raw)
        verify_source(destination, record)
    return folder


def read_documents(folder):
    folder = Path(folder)
    for name, record in SOURCES.items():
        verify_source(folder / name, record)
    json_names = {"combined": "combined_predictors.json", "old": "old_e2_predictions.json",
        "summary": "e2_summary.json", "e2_plan": "e2_plan.json", "e3_plan": "e3_plan.json",
        "e3_summary": "e3_summary.json", "snapshot": "e2_snapshot.json"}
    result = {key: read_json(folder / name) for key, name in json_names.items()}
    csv_names = {"per_seed_metrics": "e2_per_seed_metrics.csv", "comparison": "e2_comparison.csv",
        "e3_metrics": "e3_per_seed_metrics.csv", "e3_predictions": "e3_evaluation_predictions.csv"}
    for key, name in csv_names.items():
        with (folder / name).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "Duplicate CSV columns")
            result[key] = list(reader)
    return result


def numeric(values, n, label, *, nonnegative=True):
    require(isinstance(values, (list, tuple)) and len(values) == n, label + ": incorrect support")
    require(all(type(v) in (float, int) and math.isfinite(v) and (v >= 0 or not nonnegative) for v in values),
            label + ": finite numeric values required")
    return np.asarray(values, dtype=np.float64)


def coordinate(row):
    seed = row["seed"]
    if seed != "deterministic":
        require(type(seed) is int and seed in SEEDS, "Unexpected or historical prediction seed")
    return row["model"], seed


def metrics(targets, prediction):
    # Match the original descriptive aggregator; this performs no inference.
    errors = [float(p) - float(t) for p, t in zip(prediction, targets, strict=True)]
    return {"RMSE": math.sqrt(statistics.fmean(e * e for e in errors)),
        "MAE": statistics.fmean(abs(e) for e in errors), "bias": statistics.fmean(errors)}


def close_metric(left, right):
    # Saved decimal outputs can differ at the final floating-point rounding bit.
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def validate_documents(documents):
    """Pure semantic validation, separated for tests from immutable byte checks."""
    combined, old = documents["combined"], documents["old"]
    summary, snapshot = documents["summary"], documents["snapshot"]
    e2_plan, e3_plan, e3_summary = (documents[k] for k in ("e2_plan", "e3_plan", "e3_summary"))
    require(digest(e2_plan) == E2_PLAN_FINGERPRINT == snapshot["plan_fingerprint"], "E2 extension plan fingerprint differs")
    require(digest(e3_plan) == E3_PLAN_FINGERPRINT == e3_summary["plan_sha256"], "E3 source plan fingerprint differs")
    for filename, snapshot_name in (("combined_predictors.json", "combined_predictors.json"),
            ("e2_per_seed_metrics.csv", "per_seed_metrics.csv"), ("e2_comparison.csv", "comparison.csv"),
            ("e2_summary.json", "summary.json"), ("e2_plan.json", "plan.json")):
        record = snapshot["files"][snapshot_name]
        require(record == {k: SOURCES[filename][k] for k in ("bytes", "sha256")}, "Final snapshot source binding differs")
    require(combined["dates"] == list(DATES) and old["dates"] == list(DATES), "The exact 608 daily dates are required")
    require(combined["mask"] == [True] * 608 and all(type(v) is bool for v in combined["mask"]), "All 608 common-mask positions must be true")
    targets = numeric(combined["y_true"], 608, "targets")
    require(old["y_true"] == combined["y_true"] and old["mask"] == combined["mask"], "Original targets or mask changed")
    old_expected = {(m, s) for m in NEURAL for s in OLD_SEEDS} | {(m, "deterministic") for m in CLASSICAL}
    require(len(old["predictors"]) == 81 and {coordinate(r) for r in old["predictors"]} == old_expected, "The original 81 predictor coordinates differ")
    require(combined["predictors"][:81] == old["predictors"] and
            all(combined[k] == value for k, value in old.items() if k != "predictors"), "Original 81 predictions or source fields were changed")
    require(summary["old_source_sha256"] == SOURCES["old_e2_predictions.json"]["sha256"] and
            summary["old_predictor_sha256"] == [digest(r) for r in old["predictors"]], "Original predictor provenance hashes differ")
    expected = {(m, s) for m in NEURAL for s in SEEDS} | {(m, "deterministic") for m in CLASSICAL}
    rows = combined["predictors"]
    require(len(rows) == 133 and {coordinate(r) for r in rows} == expected, "Exactly 130 neural and three deterministic trajectories are required")
    require(summary["complete"] is True and summary["status"] == "completed" and summary["E2_predictors"] == 133 and
            summary["new_fits"] == 52 and summary["old_predictions_preserved"] is True and summary["E3_seeds"] == 10 and
            summary["E2_neural_seeds_per_model"] == {m: 10 for m in NEURAL} and summary["E2_deterministic_R"] == 1,
            "The source extension is incomplete or has different counts")
    provenance = combined["extension_provenance"]
    require(provenance == summary["source_provenance"] and provenance["plan_sha256"] == E2_PLAN_FINGERPRINT,
            "Extension provenance differs")
    for name in ("old_e2_predictions.json", "e3_evaluation_predictions.csv", "e3_per_seed_metrics.csv", "e3_summary.json"):
        require(provenance["source_sha256"][name] == SOURCES[name]["sha256"], "Inherited source binding differs")
    predictions, seed_ids, by_coordinate = {}, {}, {}
    for row in rows:
        identity = coordinate(row)
        require(identity not in by_coordinate, "Duplicate E2 predictor coordinate")
        by_coordinate[identity] = row
        numeric(row["prediction_original"], 608, "E2 predictions")
        if identity[1] in NEW_SEEDS:
            require(row["dates"] == list(DATES) and row["target_original"] == combined["y_true"] and
                    row["n_dates"] == 608 and row["batch_size"] == 1 and row["inverse_scaling_count"] == 1 and
                    row["plan_fingerprint"] == E2_PLAN_FINGERPRINT and row["status"] == "completed",
                    "A new E2 prediction has mismatched dates, targets or predictor bindings")
            binding = provenance["new_evaluation_bindings"][row["job_id"]]
            require(binding["fit_receipt_sha256"] == row["fit_receipt_sha256"] and
                    snapshot["files"]["jobs/" + row["job_id"] + "/evaluation.json"]["sha256"] == binding["evaluation_sha256"],
                    "A new evaluation is not bound to the final snapshot")
    for model in RIVALS:
        seeds = ("deterministic",) if model in CLASSICAL else SEEDS
        predictions[model] = np.stack([by_coordinate[model, seed]["prediction_original"] for seed in seeds]).astype(np.float64)
        seed_ids[model] = seeds

    require(e3_plan["seeds_selected_before_this_replication"] == list(SEEDS) and
            e3_summary["prescribed_seeds"] == list(SEEDS) and
            e3_summary["complete_training_seeds"] == 10 and e3_summary["comparison_complete"] is True and
            e3_summary["evaluation_n"] == 608 and e3_summary["rows_dropped"] == 0 and
            e3_summary["historical_seed_excluded"] == HISTORICAL_SEED and
            e3_summary["calibration_members_opened"] is False,
            "E3 must retain exactly its ten prescribed seeds and 608 dates")
    require(set(e3_summary["seeds"]) == {str(s) for s in SEEDS} and
            all(e3_summary["seeds"][str(s)]["complete"] is True for s in SEEDS), "E3 seed completion metadata differs")
    grouped = {seed: [] for seed in SEEDS}
    require(len(documents["e3_predictions"]) == 6080, "E3 prediction CSV must contain 6080 rows")
    for row in documents["e3_predictions"]:
        seed = int(row["seed"])
        require(seed in grouped and seed != HISTORICAL_SEED, "Unexpected or historical E3 seed")
        grouped[seed].append(row)
    e3_values, floors = [], None
    for seed in SEEDS:
        group = grouped[seed]
        require([r["date"] for r in group] == list(DATES), "E3 date ordering, duplicates or support differs")
        require([float(r["target_sigma7"]) for r in group] == combined["y_true"], "E3 targets must equal E2 targets exactly")
        values = numeric([float(r["prediction_sigma7"]) for r in group], 608, "E3 predictions")
        current_floor = numeric([float(r["known_floor"]) for r in group], 608, "E3 known floor")
        require(floors is None or np.array_equal(floors, current_floor), "Known floor differs between E3 seeds")
        require(bool(np.all(values >= current_floor)), "E3 output is below its declared known floor")
        floors = current_floor
        e3_values.append(values)
    predictions[E3_MODEL] = np.stack(e3_values)
    seed_ids[E3_MODEL] = SEEDS
    e2_data, e3_data = e2_plan["data"], e3_plan["data"]
    require(e2_data["training_examples"] == 655 and e2_data["calibration_examples"] == 53 and
            e2_data["evaluation_examples"] == 608 and e2_data["training_end"] == "2024-10-31" and
            e3_data["fit_count"] == 532 and e3_data["validation_count"] == 116 and e3_data["purged_count"] == 7 and
            e3_data["fit_end"] == "2024-06-30" and e3_data["validation_start"] == "2024-07-08" and
            e3_data["validation_end"] == "2024-10-31" and e3_data["calibration_members_allowed"] is False,
            "Training/validation/calibration protocols differ from their frozen declarations")
    require(e3_plan["architecture"] == {"family": "causal_CNN_LSTM_temporal_content_attention",
            "trainable_parameters": 1210, "variant": "additive_positive_head"}, "E3 model identity differs")
    computed = {(model, seed): metrics(targets, predictions[model][i])
        for model in MODEL_ORDER for i, seed in enumerate(seed_ids[model])}
    metric_errors = []
    for key, model_fixed in (("per_seed_metrics", None), ("e3_metrics", E3_MODEL)):
        expected_metrics = set(computed) if model_fixed is None else {(E3_MODEL, seed) for seed in SEEDS}
        seen = set()
        for row in documents[key]:
            seed = row["seed"] if row["seed"] == "deterministic" else int(row["seed"])
            identity = (model_fixed or row["model"], seed)
            require(identity in expected_metrics and identity not in seen, "Unexpected or duplicate per-seed metric")
            seen.add(identity)
            for name in ("RMSE", "MAE", "bias"):
                require(close_metric(row[name], computed[identity][name]), "Saved per-seed metric disagrees with predictions")
                metric_errors.append(abs(float(row[name]) - computed[identity][name]))
        require(seen == expected_metrics, "Incomplete per-seed metric table")
    comparison = documents["comparison"]
    require(len(comparison) == 17 and {r["model"] for r in comparison} == set(MODEL_ORDER), "Comparison model family differs")
    for row in comparison:
        model = row["model"]
        values = [computed[model, seed]["RMSE"] for seed in seed_ids[model]]
        require(int(row["R"]) == len(values) and close_metric(row["RMSE_mean"], statistics.fmean(values)), "Comparison mean or seed count differs")
        require((row["RMSE_sample_SD"] == "" if len(values) == 1 else close_metric(row["RMSE_sample_SD"], statistics.stdev(values))), "Comparison sample seed SD differs")
        require(int(row["train_n"]) == (532 if model == E3_MODEL else 655) and
                int(row["evaluation_n"]) == 608 and int(row["calibration_n"]) == (0 if model == E3_MODEL else 53) and
                row["validation_n"] == ("116" if model == E3_MODEL else "") and
                row["posterior"] == "True" and row["equal_training_budget_claimed"] == "False", "Comparison cohort declarations differ")
    for name, expected_value in (("RMSE_mean", statistics.fmean(computed[E3_MODEL, s]["RMSE"] for s in SEEDS)),
            ("RMSE_sample_SD", statistics.stdev(computed[E3_MODEL, s]["RMSE"] for s in SEEDS))):
        require(close_metric(e3_summary[name], expected_value), "E3 summary metric differs")
    for values in predictions.values():
        values.setflags(write=False)
    targets.setflags(write=False)
    floors.setflags(write=False)
    mask = np.ones(608, dtype=bool)
    mask.setflags(write=False)
    provenance_report = {"sources": SOURCES, "E2_commit": E2_COMMIT, "E3_commit": E3_COMMIT,
        "E2_extension_plan_fingerprint": E2_PLAN_FINGERPRINT, "E3_source_plan_fingerprint": E3_PLAN_FINGERPRINT,
        "old_81_preserved": True, "old_81_source_sha256": SOURCES["old_e2_predictions.json"]["sha256"],
        "historical_seed_excluded": HISTORICAL_SEED, "rows_dropped": 0, "targets_exactly_equal": True,
        "all_608_mask_true": True, "E2_predictors": 133, "E3_predictors": 10, "rival_family_size": 16,
        "metric_comparison_rtol": 1e-12, "metric_comparison_atol": 1e-15,
        "maximum_saved_metric_absolute_difference": max(metric_errors),
        "legacy_original_analysis_ignored": True, "legacy_E3_summary_E2_seed_count_ignored": True,
        "model_order_rationale": "E3 candidate, ten frozen E2 candidates, three E1 neural references, three deterministic references",
        "E3_model_label": E3_LABEL, "E3_architecture": e3_plan["architecture"],
        "E2_data_protocol": e2_data, "E3_data_protocol": e3_data,
        "known_calendar_posterior": True, "equal_training_budget_claimed": False,
        "input_validation_only": True, "hypothesis_tests_performed": False}
    return {"model_order": list(MODEL_ORDER), "rivals": list(RIVALS),
        "predictions": {m: predictions[m] for m in MODEL_ORDER}, "targets": targets,
        "y_true": targets, "dates": list(DATES), "seeds": {m: seed_ids[m] for m in MODEL_ORDER},
        "mask": mask, "known_floor": floors, "provenance": provenance_report,
        "train_counts": {m: 532 if m == E3_MODEL else 655 for m in MODEL_ORDER}}


def load_inputs(folder):
    """Return seventeen immutable R-by-608 arrays after byte and semantic gates."""
    return validate_documents(read_documents(folder))
