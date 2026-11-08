#!/usr/bin/env python3
"""Promote authentic saved D7 scores to the active 608-day E1 benchmark.

This metadata compiler performs no fitting, inference, or raw-data access. Its
editorial promotion follows inspection of the historical results; it does not
register a prospective experiment or alter either original study.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "e1_primary_608_20261107"
STAGE = ROOT / "results" / STUDY_ID
MIRROR = ROOT / "book/_static" / STUDY_ID
MODELS = ("LSTM", "MLP", "GRU")
METRICS = ("mape", "mae", "rmse", "mse", "r2", "mase")
SEEDS = tuple(range(4101, 4111))
DATES = {"cv": {"start": "2024-01-01", "end": "2024-12-31", "n_dates": 366},
         "test": {"start": "2025-01-01", "end": "2026-08-31", "n_dates": 608}}
RECIPES = {model: {"model": model, "history_days": 7, "width": width,
                  "learning_rate": rate, "dropout": .1, "parameters": parameters,
                  "variant": "relative_window"}
           for model, width, rate, parameters in
           (("LSTM", 32, .003, 5025), ("MLP", 16, .001, 593), ("GRU", 32, .003, 3777))}
SCOPE = ("Editorial promotion of existing historical fixed-recipe scores after results were seen; "
         "exploratory and conditional on known market dates and training seeds, without new fits "
         "or forecasts, equal cumulative HPO effort, or fresh temporal confirmation.")
INPUTS = {
    "scores": ("results/e1_historical_core_20261107/run_metrics.csv", "7f9cc6ace6042cdf62f6ae900ab4ee7967b63be41fd75b0309adcadcf747b834"),
    "historical_summary": ("results/e1_historical_core_20261107/summary.json", "f31e0e4714f183a997a76eb589dcd9c779bda2fe4f07741fc090528782da4232"),
    "cohort": ("results/e1_historical_core_20261107/cohort_checks.json", "7ee9ce0ff33a9487e593f7027d8c9be85b0eca3d11a1fe0671be8a2291b93792"),
    "historical_sources": ("results/e1_historical_core_20261107/source_manifest.json", "b81087cca52288c4cecadc7e118e3d4481fb91f8c0653ff56d652f62ae10bd9b"),
    "persistence_cv": ("results/gru_lstm_tcn_20261105/metric_summary_cv.csv", "55f6489138db5dbbbf113bdb15080f0a2baa1ba3d8708e034f34cb285c5762f0"),
    "persistence_test": ("results/gru_lstm_tcn_20261105/metric_summary_test.csv", "2c479dc568b9684aae1fb2ee88c5746572f061eda7020e8eb116a3954d74b876"),
    "execution": ("results/filipe_multihistory_20261104/test/execution_summary.json", "850b4ef78efd8ac360aa6bd41078351f0bacfcd7cae6d1e80a94517f9a0b7049"),
    "events": ("results/filipe_multihistory_20261104/final_test_events.jsonl", "96a17f1ffebff44b96a6ebc9846f3838400ab5387f5369fe834f4aed6e2ba5e4"),
    "models": ("results/filipe_multihistory_20261104/final_models_manifest.json", "c6a1397f7f4f7f93aef36215beb12e421f535667d1ceae9b88b8c97d8a3baf95"),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def csv_bytes(fields, rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def parse_neural_runs(raw):
    rows = []
    for source in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        if int(source["history_days"]) != 7:
            continue
        phase, model, seed = source["phase"], source["model"], int(source["seed"])
        if phase not in DATES or model not in MODELS or seed not in SEEDS:
            raise ValueError("Incompatible primary model/phase/seed")
        if int(source["n_dates"]) != DATES[phase]["n_dates"]:
            raise ValueError("Primary historical date count changed")
        if int(source["parameters"]) != RECIPES[model]["parameters"]:
            raise ValueError("Primary historical recipe changed")
        row = {"phase": phase, "history_days": 7, "model": model, "seed": seed,
               "n_dates": DATES[phase]["n_dates"], "parameters": int(source["parameters"])}
        row.update({metric: float(source[metric]) for metric in METRICS})
        if not all(math.isfinite(row[metric]) for metric in METRICS):
            raise ValueError("Nonfinite primary score")
        if not math.isclose(row["mse"], row["rmse"] ** 2, rel_tol=1e-10, abs_tol=1e-16):
            raise ValueError("Saved primary MSE and RMSE disagree")
        rows.append(row)
    keys = [(r["phase"], r["model"], r["seed"]) for r in rows]
    expected = {(phase, model, seed) for phase in DATES for model in MODELS for seed in SEEDS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Duplicate or missing primary seed score")
    return sorted(rows, key=lambda r: (list(DATES).index(r["phase"]), MODELS.index(r["model"]), r["seed"]))


def parse_persistence(raw, phase):
    selected = [r for r in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
                if r["model"] == "Persistence" and int(r["history_days"]) == 7]
    if len(selected) != 6 or {r["metric"] for r in selected} != set(METRICS):
        raise ValueError("Missing or duplicated deterministic persistence metric")
    row = {"phase": phase, "history_days": 7, "model": "Persistence", "seed": "",
           "n_dates": DATES[phase]["n_dates"], "parameters": 0}
    for source in selected:
        if (int(source["n_dates"]) != row["n_dates"] or source["stochastic_training"] != "False"
                or int(source["n_total"]) != 1 or int(source["n_defined"]) != 1
                or any(source[k] for k in ("sample_sd", "ci95_lower", "ci95_upper"))):
            raise ValueError("Persistence must be a deterministic common-calendar reference")
        value = float(source["mean"])
        if not math.isfinite(value):
            raise ValueError("Nonfinite persistence metric")
        row[source["metric"]] = value
    if not math.isclose(row["mse"], row["rmse"] ** 2, rel_tol=1e-10, abs_tol=1e-16):
        raise ValueError("Persistence MSE and RMSE disagree")
    return row


def summarize(rows):
    output = []
    critical = float(stats.t.ppf(.975, len(SEEDS) - 1))
    for phase in DATES:
        for model in (*MODELS, "Persistence"):
            group = [r for r in rows if r["phase"] == phase and r["model"] == model]
            stochastic = model != "Persistence"
            if len(group) != (len(SEEDS) if stochastic else 1):
                raise ValueError("Wrong primary descriptive denominator")
            for metric in METRICS:
                values = [r[metric] for r in group]
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if stochastic else ""
                margin = critical * sd / math.sqrt(len(values)) if stochastic else 0
                output.append({"phase": phase, "history_days": 7, "model": model, "metric": metric,
                    "n_total": len(values), "n_defined": len(values), "n_dates": DATES[phase]["n_dates"],
                    "n_seeds": len(SEEDS) if stochastic else 0,
                    "n_dates_per_seed": DATES[phase]["n_dates"], "mean": mean, "sample_sd": sd,
                    "ci95_lower": mean - margin if stochastic else "",
                    "ci95_upper": mean + margin if stochastic else "",
                    "ci95_low": mean - margin if stochastic else "",
                    "ci95_high": mean + margin if stochastic else "", "stochastic_training": stochastic,
                    "ci_method": ("Marginal Student t across ten training seeds, df9, conditional on known dates"
                                  if stochastic else "Deterministic reference; no seed SD or confidence interval"),
                    "scope": SCOPE})
    return output


def execution_provenance(execution, events, models):
    if (execution["model_runs"] != 120 or execution["common_prediction_dates"] != 608
            or execution["test_period_model_refits"] != 0
            or execution["test_used_for_configuration_selection"] is not False):
        raise ValueError("Original execution no longer proves the historical frozen test")
    scored = [e for e in events if e.get("event") == "test_run_scored_once"
              and "_MLP_s" in e.get("run_id", "")]
    expected = {f"D{d}_MLP_s{s}" for d in (7, 14, 21, 28) for s in SEEDS}
    if (len(scored) != 40 or {e["run_id"] for e in scored} != expected
            or any(e["predictions"] != 608 or e["checkpoint_forward_calls"] != 1 for e in scored)):
        raise ValueError("Original MLP execution evidence is incomplete")
    d7 = sorted((e for e in scored if e["run_id"].startswith("D7_")), key=lambda e: e["run_id"])
    frozen = next(e for e in events if e["event"] == "all_final_models_frozen_before_test")
    first_access = next(e for e in events if e["event"] == "first_reserved_test_access_authorized")
    if not models["completed_at_utc"] <= frozen["at_utc"] < first_access["at_utc"] < min(e["at_utc"] for e in scored):
        raise ValueError("Original model freeze does not precede test access and scoring")
    if (execution["final_models_manifest_sha256"] != INPUTS["models"][1]
            or frozen["manifest_sha256"] != INPUTS["models"][1]):
        raise ValueError("Original execution and model manifest identity differ")
    jobs = [j for j in models["jobs"] if j["history_days"] == 7 and j["model"] == "MLP"]
    if (len(jobs) != 10 or {j["seed"] for j in jobs} != set(SEEDS)
            or any(j["train_samples"] != 709 or j["embargo_days"] != 7
                   or j["train_last_day_utc"] != "2024-12-24" or j["parameters"] != 593 for j in jobs)):
        raise ValueError("Original MLP final training cohort or recipe changed")
    return {"promotion": "Editorial activation after inspecting existing results; no new experiment execution",
        "original_summary_completed_at_utc": execution["completed_at_utc"],
        "all_final_models_frozen_at_utc": frozen["at_utc"],
        "first_reserved_test_access_at_utc": first_access["at_utc"],
        "original_summary_source": INPUTS["execution"][0], "original_event_log_source": INPUTS["events"][0],
        "original_final_models_manifest_source": INPUTS["models"][0],
        "original_mlp": {"scored_runs": 40, "D7_scored_runs": 10,
            "D7_run_ids": [e["run_id"] for e in d7], "dates_per_run": 608, "all_scored_once": True,
            "first_D7_score_at_utc": min(e["at_utc"] for e in d7),
            "last_D7_score_at_utc": max(e["at_utc"] for e in d7),
            "D7_epochs_by_seed": {str(j["seed"]): j["epochs"] for j in jobs},
            "training_samples": 709, "training_start": "2023-01-08", "training_end": "2024-12-24",
            "test_refits": 0}, "new_model_fits": 0, "new_forecasts": 0,
        "daily_prediction_vectors_read": 0, "weights_or_scalers_read": 0}


def compile_outputs(root=ROOT):
    loaded, sources = {}, {}
    for name, (path, expected) in INPUTS.items():
        raw = (root / path).read_bytes()
        if digest(raw) != expected:
            raise ValueError("Historical source hash changed: " + path)
        loaded[name] = raw
        sources[path] = {"sha256": expected, "bytes": len(raw)}
    # Verify the original authentic sources, not only the later reconciliation.
    for path, identity in json.loads(loaded["historical_sources"])["sources"].items():
        raw = (root / path).read_bytes()
        if digest(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
            raise ValueError("Original historical source hash changed: " + path)
        sources[path] = identity
    checks = json.loads(loaded["cohort"])
    if (checks["lstm_exact_metric_equalities_old_vs_trio"] != 480
            or checks["old_and_trio_data_contracts_equal"] is not True
            or checks["test_calendar_manifest_matches_608_consecutive_days"] is not True
            or checks["same_nominal_seeds"] != list(SEEDS)):
        raise ValueError("Historical compatibility evidence changed")
    rows = parse_neural_runs(loaded["scores"])
    rows += [parse_persistence(loaded["persistence_" + phase], phase) for phase in DATES]
    summaries = summarize(rows)
    provenance = execution_provenance(json.loads(loaded["execution"]),
        [json.loads(line) for line in loaded["events"].splitlines()], json.loads(loaded["models"]))
    cv_neural = [r for r in summaries if r["phase"] == "cv" and r["metric"] == "rmse" and r["model"] in MODELS]
    if min(cv_neural, key=lambda r: r["mean"])["model"] != "GRU":
        raise ValueError("Saved CV no longer supports the original GRU neural baseline")
    summary = {"study_id": STUDY_ID, "effective_date_colombia": "2026-11-07", "active_primary": True,
        "models": list(MODELS), "core_models": list(MODELS), "reference_models": ["Persistence"],
        "baseline_model": "GRU", "baseline": "GRU", "baseline_recipe": RECIPES["GRU"], "selected_recipes": RECIPES,
        "baseline_selection": "Lowest saved mean CV2024 RMSE among the three neural D7 recipes; persistence reported separately",
        "history_days": 7, "target_n_days": 7, "horizon_days": 1,
        "R_per_neural": 10, "R": 10, "n_dates": 608, "dates_per_trajectory": 608,
        "phases": DATES, "neural_score_rows": 60, "deterministic_reference_rows": 2,
        "primary_scope": "historical_known_fixed_recipes", "scope": SCOPE,
        "editorial_promotion_after_results_seen": True, "historical_results_already_observed": True,
        "known_reserve_exploratory": True, "fresh_confirmatory_holdout": False,
        "prospective_registration_claimed": False, "HPO_not_equal": True,
        "HPO_effort": {"MLP": "32 V1 search trajectories; chosen width16/lr.001 inherited without new relative-window HPO",
                       "LSTM": "32 V1 search trajectories; chosen width32/lr.003 inherited without new relative-window HPO",
                       "GRU": "No original GRU HPO; inherits recurrent width32/lr.003"},
        "training_policy": {"optimizer": "AdamW", "weight_decay": .0001, "dropout": .1,
            "batch_size": 32, "max_epochs": 100, "patience": 12, "min_delta": .00001,
            "gradient_clip_norm": 1., "inner_validation_days": 90, "monthly_cv_blocks": 12,
            "embargo_days": 7, "final_epochs": "Per-model/per-seed December2024 inner-validation best_epoch",
            "final_training_samples": 709, "final_training_start": "2023-01-08", "final_training_end": "2024-12-24",
            "test_period_refits": 0, "test_reporting_months": 20},
        "seeds": list(SEEDS), "lstm_exact_control_metric_equalities": 480,
        "new_model_fits": 0, "new_forecasts": 0, "daily_prediction_vectors_read": 0,
        "raw_market_archives_read": 0, "weights_or_scalers_read": 0, "source_files_modified": False,
        "source_summary_sha256": INPUTS["historical_summary"][1], "source_score_rows_sha256": INPUTS["scores"][1],
        "source_studies": ["filipe_multihistory_20261104", "gru_lstm_tcn_20261105"],
        "ci_scope": "Marginal t95 across ten initializations, df9, conditional on known dates; no temporal uncertainty or simultaneous coverage",
        "persistence_scope": "One deterministic sigma7-lag1 reference per phase, no seed SD or CI; requires eight past closes",
        "limitations": ["The 608-day test was already known when GRU was evaluated; promotion is editorial after results",
            "Architecture widths, parameter counts and learning rates differ; cumulative historical HPO effort was unequal",
            "Original MLP daily test forecasts, weights and scalers are unavailable here; no temporal or residual tests are reconstructed",
            "Ten seeds were a fixed historical screening budget, not a prospectively powered design"],
        "preserved_separate_studies": ["e1_historical_core_20261107", "e1_core_mlp_extension_20261107"],
        "rmse_summary": [r for r in summaries if r["metric"] == "rmse"]}
    readme = """# Comparación principal E1: recetas originales, D7 y test histórico de 608 días

Este paquete activa editorialmente LSTM32, MLP16 y GRU32 como comparación principal
de E1 a partir de resultados auténticos ya ejecutados. No entrena redes ni genera
pronósticos. La promoción se decide después de conocer los resultados: no es un
registro prospectivo ni una nueva confirmación temporal.

Las tres redes usan las mismas cinco covariables OHLCV, `relative_window`, D7,
objetivo sigma7/h1, semillas 4101–4110 y reglas de optimización/selección de épocas.
CV son doce bloques mensuales expansivos y 366 fechas de enero–diciembre de 2024.
El test conocido contiene 608 fechas, 01/01/2025–31/08/2026. Sus veinte meses son
intervalos de reporte con pesos y escalas congelados, sin reajustes en test.
El ajuste final comparte 709 etiquetas de 08/01/2023–24/12/2024, con purga D7.

GRU32 (lr .003, dropout .1, 3.777 parámetros) es el baseline neuronal por su menor
RMSE media CV. LSTM32 usa lr .003 y 5.025 parámetros; MLP16 usa lr .001 y 593.
La persistencia determinista se informa como referencia aparte: no tiene semillas,
DE ni IC de inicialización y necesita ocho cierres previos. Las recetas tienen
distinta capacidad y esfuerzo histórico de HPO: MLP/LSTM heredaron selecciones de
32 trayectorias V1 por familia; GRU heredó la receta recurrente sin búsqueda propia.
No se afirma que las arquitecturas estén optimizadas con igual presupuesto.

`run_metrics.csv` guarda 60 puntuaciones neuronales D7 (dos fases × tres redes ×
diez semillas) y dos referencias deterministas, sin vectores diarios. Las seis
métricas de `metric_summary.csv` tienen media, DE muestral e IC t95 marginal df9
para redes; estos intervalos dependen del calendario conocido, no incorporan
incertidumbre temporal ni cobertura simultánea. MAPE conserva unidades porcentuales.

`execution_provenance.json` documenta que la MLP histórica ya produjo las 608
predicciones por cada una de diez semillas D7 el 04/11/2026, después de congelar
los modelos y antes de este cambio editorial. No se atribuyen a esta promoción
nuevos ajustes ni pronósticos. Los 480 controles métricos LSTM coinciden entre
los estudios originales. Se verifican hashes de las fuentes y sus fuentes previas.

Faltan aquí los vectores diarios, pesos y escaladores originales de MLP; las
puntuaciones agregadas no permiten reconstruir DM, residuos o bootstrap temporal.
El calendario de septiembre–octubre de 2026 pertenece al estudio complementario
de 60 días y 465 semillas; sus recetas/resultados permanecen separados e intactos.
También permanece intacto el suplemento histórico de cuatro D.

Reproducción: `python scripts/build_e1_primary_608.py`; verificación sin escribir:
`python scripts/build_e1_primary_608.py --check`. Ningún comando entrena o infiere.
"""
    outputs = {"README.md": readme.encode(), "summary.json": json_bytes(summary),
        "run_metrics.csv": csv_bytes(list(rows[0]), rows),
        "metric_summary.csv": csv_bytes(list(summaries[0]), summaries),
        "execution_provenance.json": json_bytes(provenance),
        "source_manifest.json": json_bytes({"sources": sources, "source_files_modified": False,
            "builder": "scripts/build_e1_primary_608.py", "builder_sha256": digest(Path(__file__).read_bytes())})}
    outputs["manifest.json"] = json_bytes({"publication_schema": "e1_public_metadata_allowlist_v1",
        "approved_public_files": sorted(outputs),
        "files": {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in sorted(outputs.items())}})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = compile_outputs()
    for target in (STAGE, MIRROR):
        if args.check:
            if not target.is_dir() or {p.name for p in target.iterdir()} != set(outputs):
                raise ValueError("Primary 608 output file set differs: " + str(target))
            for name, raw in outputs.items():
                if (target / name).read_bytes() != raw:
                    raise ValueError("Primary 608 output differs: " + str(target / name))
        else:
            target.mkdir(parents=True, exist_ok=True)
            for name, raw in outputs.items():
                (target / name).write_bytes(raw)
    print(json.dumps({"status": "checked" if args.check else "generated", "study_id": STUDY_ID,
        "neural_runs": 60, "reference_rows": 2, "metric_rows": 48,
        "new_model_fits": 0, "new_forecasts": 0, "daily_vectors_read": 0}))


if __name__ == "__main__":
    main()
