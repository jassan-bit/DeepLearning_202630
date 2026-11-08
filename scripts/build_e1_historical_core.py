#!/usr/bin/env python3
"""Reconcile the historical LSTM/MLP/GRU four-history comparison from saved run metrics.

No model, forecast vector, raw market file or holdout payload is opened. The
comparison is explicitly post-hoc and never substitutes for the new core study.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "results/e1_historical_core_20261107"
MIRROR = ROOT / "book/_static/e1_historical_core_20261107"
MODELS = ("LSTM", "MLP", "GRU")
METRICS = ("rmse", "mae", "mse", "r2", "mape", "mase")
HISTORIES = (7, 14, 21, 28)
SEEDS = tuple(range(4101, 4111))
PARAMETERS = {"LSTM": 5025, "GRU": 3777}
DATES = {"cv": ("2024-01-01", "2024-12-31", 366),
         "test": ("2025-01-01", "2026-08-31", 608)}
INPUTS = {
    "old_cv": ("results/filipe_multihistory_20261104/analysis_cv/run_metrics.csv",
               "bbae5e52b9900aab80ae6d886e71559505d5dcf4b0625f212c04c9ef8aa91c43"),
    "old_test": ("results/filipe_multihistory_20261104/test/runs.csv",
                 "13560298f4a1146047d56a381910a7b43de4337a4ef801a5979285d5e5839be9"),
    "trio_cv": ("results/gru_lstm_tcn_20261105/run_metrics_cv.csv",
                "f6877c9f816811274828b49d067f2a6e7f2f18f25ed1d35ae946402bf088b14e"),
    "trio_test": ("results/gru_lstm_tcn_20261105/run_metrics_test.csv",
                  "ba5a7bcba6b82d2c92ed005d80fa5bafa3fa7c3e8f6b326e8628ebe090cddfd2"),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode()


def csv_bytes(fields, rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def parameters(model, history):
    return 16 * (5 * history + 2) + 1 if model == "MLP" else PARAMETERS[model]


def parse_runs(raw, phase, models):
    rows = []
    for source in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        history = int(float(source["history_days"]))
        if history not in HISTORIES or source["model"] not in models:
            continue
        seed = int(float(source["seed"]))
        count = int(float(source["n_dates"]))
        if seed not in SEEDS or count != DATES[phase][2]:
            raise ValueError("Incompatible historical seed/date cohort")
        if "variant" in source and source["variant"] != "relative_window":
            raise ValueError("Historical representation is not relative_window")
        row = {"phase": phase, "history_days": history, "model": source["model"],
               "seed": seed, "n_dates": count, "parameters": parameters(source["model"], history)}
        if "parameters" in source and int(source["parameters"]) != row["parameters"]:
            raise ValueError("Historical parameter count mismatch")
        for metric in METRICS:
            row[metric] = float(source[metric])
            if not math.isfinite(row[metric]):
                raise ValueError("Nonfinite saved metric")
        if not math.isclose(row["mse"], row["rmse"] ** 2, rel_tol=1e-10, abs_tol=1e-16):
            raise ValueError("Saved MSE/RMSE disagree")
        for metric in ("mape", "mase"):
            if float(source[metric + "_coverage"]) != 1.0:
                raise ValueError("Incomplete relative-metric coverage")
        rows.append(row)
    keys = [(row["history_days"], row["model"], row["seed"]) for row in rows]
    expected = {(history, model, seed) for history in HISTORIES for model in models for seed in SEEDS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Duplicate or missing model/seed in historical cohort")
    return rows


def verify_lstm_control(old, trio):
    a = {(row["history_days"], row["seed"]): row for row in old if row["model"] == "LSTM"}
    b = {(row["history_days"], row["seed"]): row for row in trio if row["model"] == "LSTM"}
    expected = {(history, seed) for history in HISTORIES for seed in SEEDS}
    if set(a) != expected or set(b) != expected:
        raise ValueError("Missing shared LSTM control")
    for unit in expected:
        if any(a[unit][key] != b[unit][key] for key in (*METRICS, "n_dates")):
            raise ValueError("Historical LSTM control does not match exactly")
    return len(HISTORIES) * len(SEEDS) * len(METRICS)


def summarize(rows):
    output = []
    critical = float(stats.t.ppf(0.975, len(SEEDS) - 1))
    for phase in DATES:
        for history in HISTORIES:
            for model in MODELS:
                selected = [r for r in rows if r["phase"] == phase and r["model"] == model and r["history_days"] == history]
                if len(selected) != len(SEEDS):
                    raise ValueError("Wrong descriptive denominator")
                for metric in METRICS:
                    values = [r[metric] for r in selected]
                    mean = statistics.mean(values)
                    sd = statistics.stdev(values)
                    margin = critical * sd / math.sqrt(len(values))
                    output.append({"phase": phase, "history_days": history, "model": model,
                                   "metric": metric, "n_seeds": len(values),
                                   "n_dates_per_seed": DATES[phase][2], "mean": mean,
                                   "sample_sd": sd, "ci95_low": mean - margin,
                                   "ci95_high": mean + margin,
                                   "ci_scope": "marginal_t_between_seeds_conditional_on_known_dates"})
    return output


def holm(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (len(values) - rank) * values[index])
        result[index] = min(1.0, running)
    return result


def pairwise_rmse(rows):
    output = []
    pairs = (("LSTM", "MLP"), ("LSTM", "GRU"), ("MLP", "GRU"))
    for phase in DATES:
        selected = {(r["model"], r["seed"]): r["rmse"] for r in rows if r["phase"] == phase and r["history_days"] == 7}
        group = []
        for a, b in pairs:
            differences = [selected[(a, s)] - selected[(b, s)] for s in SEEDS]
            mean, sd = statistics.mean(differences), statistics.stdev(differences)
            if sd == 0:
                raise ValueError("Degenerate paired differences: no t reference reported")
            result = stats.ttest_rel([selected[(a, s)] for s in SEEDS],
                                    [selected[(b, s)] for s in SEEDS])
            margin = float(stats.t.ppf(.975, 9)) * sd / math.sqrt(10)
            group.append({"phase": phase, "history_days": 7, "contrast": a + " minus " + b,
                          "n_seed_pairs": 10, "mean_rmse_difference": mean,
                          "sample_sd_difference": sd, "ci95_low_marginal": mean - margin,
                          "ci95_high_marginal": mean + margin, "t_statistic": float(result.statistic),
                          "degrees_of_freedom": 9, "p_two_sided": float(result.pvalue),
                          "shapiro_difference_p": float(stats.shapiro(differences).pvalue),
                          "cohens_dz": mean / sd,
                          "analysis_scope": "posthoc_exploratory_same_known_dates"})
        for row, adjusted in zip(group, holm([x["p_two_sided"] for x in group])):
            row["p_holm3_within_phase"] = adjusted
            output.append(row)
    return output


def compile_outputs(root=ROOT):
    loaded, source_manifest = {}, {}
    for name, (path, expected) in INPUTS.items():
        raw = (root / path).read_bytes()
        if digest(raw) != expected:
            raise ValueError("Historical source hash changed: " + path)
        loaded[name] = raw
        source_manifest[path] = {"sha256": expected, "bytes": len(raw)}
    rows, exact_controls = [], 0
    for phase in DATES:
        old = parse_runs(loaded["old_" + phase], phase, ("LSTM", "MLP"))
        trio = parse_runs(loaded["trio_" + phase], phase, ("LSTM", "GRU"))
        exact_controls += verify_lstm_control(old, trio)
        rows.extend([r for r in old if r["model"] == "MLP"] + trio)
    rows.sort(key=lambda r: (list(DATES).index(r["phase"]), r["history_days"], MODELS.index(r["model"]), r["seed"]))
    summaries = summarize(rows)
    pairs = pairwise_rmse(rows)
    cohort_path = "results/filipe_multihistory_20261104/test/cohort_manifest.json"
    cohort_raw = (root / cohort_path).read_bytes()
    if digest(cohort_raw) != "44c1d8a439b9a1e10023374ad427e70f823c73514234e5404bdf714c745acd3e":
        raise ValueError("Historical test cohort manifest hash changed")
    cohort = json.loads(cohort_raw)
    days = cohort["common_outcome_days_utc"]
    expected_days = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
    if days != expected_days or cohort["n_common_outcomes"] != 608:
        raise ValueError("Historical test calendar mismatch")
    source_manifest[cohort_path] = {"sha256": digest(cohort_raw), "bytes": len(cohort_raw)}
    protocol_paths = ["experiments/filipe_multihistory_20261104/protocol.json",
                      "experiments/gru_lstm_tcn_20261105/original_registration/cv_protocol.json"]
    protocols = [json.loads((root / p).read_text()) for p in protocol_paths]
    policy_fields = ("library", "function", "call_by_D", "monthly_blocks", "permitted_train_outcomes",
                     "inner_validation_calendar_days", "inner_purge_days", "strict_determinism")
    if (protocols[0]["data_contract"] != protocols[1]["data_contract"]
            or protocols[0]["variant"] != protocols[1]["variant"]
            or any(protocols[0]["cv_policy"][k] != protocols[1]["cv_policy"][k] for k in policy_fields)):
        raise ValueError("Historical data/preprocessing/CV contracts differ")
    for p in protocol_paths:
        raw = (root / p).read_bytes()
        source_manifest[p] = {"sha256": digest(raw), "bytes": len(raw)}
    cv_expected_days = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(366)]
    fold_checks = {}
    for history in HISTORIES:
        path = ("results/dl_r2_cv_20261104/folds.json" if history == 7 else
                f"results/filipe_multihistory_20261104/D{history}/folds.json")
        raw = (root / path).read_bytes()
        folds = json.loads(raw)["folds"]
        dates = [day[:10] for fold in folds for day in fold["daily_prediction_origins_utc"]]
        if len(folds) != 12 or dates != cv_expected_days:
            raise ValueError("Historical CV fold/calendar mismatch")
        source_manifest[path] = {"sha256": digest(raw), "bytes": len(raw)}
        fold_checks[str(history)] = {"monthly_folds": 12, "daily_origins": 366,
                                      "start": dates[0], "end": dates[-1]}
    checks = {"same_nominal_seeds": list(SEEDS), "histories": list(HISTORIES),
              "test_calendar_manifest_matches_608_consecutive_days": True,
              "lstm_exact_metric_equalities_old_vs_trio": exact_controls,
              "lstm_controls": "4 histories × 10 seeds × 6 metrics × 2 phases, exact equality",
              "old_and_trio_data_contracts_equal": True, "old_and_trio_cv_policy_fields_equal": list(policy_fields),
              "cv_fold_calendars": fold_checks,
              "date_scope": "Calendar and saved run counts; daily predictions are not re-read",
              "representation": "relative_window; existing historical source/protocol provenance",
              "all_mape_mase_coverage": 1.0}
    rmse = [row for row in summaries if row["metric"] == "rmse"]
    best_history = {model: min((r for r in rmse if r["phase"] == "cv" and r["model"] == model),
                              key=lambda r: (r["mean"], r["history_days"]))["history_days"] for model in MODELS}
    summary = {"schema_version": "1.0", "analysis_date_colombia": "2026-11-07",
               "purpose": "Post-hoc historical four-history core comparison from authentic saved metrics",
               "core_models": list(MODELS), "historical_source_models_used": ["LSTM", "MLP", "GRU"],
               "TCN_role": "Separate preserved complementary historical study",
               "historical_parameters_by_D": {str(d): {m: parameters(m, d) for m in MODELS} for d in HISTORIES},
               "histories": list(HISTORIES), "runs": len(rows),
               "best_history_by_saved_cv_mean_rmse": best_history,
               "descriptive_cv_minimum_in_core": {"model": min((r for r in rmse if r["phase"] == "cv"), key=lambda r: (r["mean"], r["history_days"]))["model"], "selection_scope": "Derived after historical tests were known; not the new study baseline freeze"},
               "seeds_per_model_phase": 10, "phases": {k: {"start": a, "end": b, "n_dates": n}
                                                        for k, (a, b, n) in DATES.items()},
               "rmse_summary": rmse, "lstm_exact_control_metric_equalities": exact_controls,
               "new_model_fits": 0, "new_forecasts": 0, "daily_prediction_vectors_read": 0,
               "raw_market_archives_read": 0, "source_results_modified": False,
               "historical_core_selection_is_posthoc": True,
               "new_core_final_study_results_substituted": False,
               "ci_scope": "Marginal t95 intervals across initialization seeds on the same known dates",
               "paired_tests_scope": "D7 only: three RMSE contrasts per phase, separate exploratory Holm3 families; labels pair nominal seeds",
               "temporal_DM_or_residual_tests_recomputed": False,
               "limitation": "Original MLP test daily vectors and final weights are unavailable here; aggregate seed metrics do not reconstruct them.",
               "software": {"scipy": scipy.__version__}}
    readme = """# Núcleo histórico de cuatro entradas: LSTM–MLP–GRU

Este suplemento concilia métricas por semilla conservadas de dos estudios ya
realizados. Usa MLP/LSTM de `filipe_multihistory_20261104` y GRU/LSTM de
`gru_lstm_tcn_20261105`; los resultados LSTM coinciden exactamente en las seis métricas
en las cuatro entradas, CV y test (480 igualdades). No entrena ni abre observaciones, predicciones
diarias, pesos o escaladores. Las fuentes permanecen intactas y se verifican sus
hashes antes de calcular los agregados.

Las tres redes comparten D=7/14/21/28, cinco entradas, representación `relative_window`,
objetivo sigma7/h1 y semillas nominales 4101–4110. CV comprende las 366 fechas de
2024; test, las 608 fechas del 01/01/2025 al 31/08/2026. Las recetas históricas
tienen 5.025 parámetros LSTM y 3.777 GRU en cada D. La MLP histórica usa 16
unidades y tiene 593/1.153/1.713/2.273 parámetros al variar D=7/14/21/28.
No son las recetas del estudio
nuevo del núcleo ni se convierten en resultados de 465 semillas o del periodo de
septiembre–octubre de 2026. Los resultados de TCN se conservan por separado como
estudio complementario.

`run_metrics.csv` contiene 240 puntuaciones de corrida (2 fases × 4 entradas × 3 redes × 10
semillas), no vectores de pronóstico. `metric_summary.csv` resume las seis métricas
con media, DE muestral e IC t marginal del 95 %, df=9. Son intervalos entre
inicializaciones condicionados a un calendario conocido, sin cobertura simultánea
ni incertidumbre temporal incorporada. MAPE conserva sus unidades porcentuales.

El mínimo de RMSE media CV se deriva de las cuatro entradas guardadas por familia;
las tres favorecen D7 en esta evidencia histórica. No se elige D mirando test ni
se atribuye esta síntesis posterior al cierre del estudio nuevo.

`paired_rmse.csv` informa únicamente para D7 tres contrastes t bilaterales por fase, diferencia
orientada como A−B, IC marginal, d_z, Shapiro de diferencias y Holm dentro de cada
familia de tres. Es un análisis posterior y exploratorio de etiquetas-semilla;
normalidad e independencia condicional son supuestos, no hechos demostrados por
el emparejamiento. No se suman las fases como confirmaciones independientes ni se
elige una prueba por su p-value. La comparación final nueva tiene su propio
protocolo, piloto, dimensionamiento y resultados, fuera de este suplemento.

Sin los vectores diarios originales MLP de test no se recalculan DM, residuos,
BDS o bootstrap temporal del núcleo desde observaciones. Las métricas guardadas
permiten esta comparación por semilla, pero no reconstruyen esos artefactos.

Reproducción: `python scripts/build_e1_historical_core.py` genera este directorio
y el espejo del libro; `--check` verifica las salidas guardadas. Use el entorno
científico con SciPy fijado del proyecto. No hay ejecución de modelos ni cambios
en las fuentes históricas.
"""
    outputs = {"README.md": readme.encode(), "summary.json": json_bytes(summary),
               "run_metrics.csv": csv_bytes(list(rows[0]), rows),
               "metric_summary.csv": csv_bytes(list(summaries[0]), summaries),
               "paired_rmse.csv": csv_bytes(list(pairs[0]), pairs),
               "cohort_checks.json": json_bytes(checks),
               "source_manifest.json": json_bytes({"sources": source_manifest,
                   "source_files_modified": False, "builder": "scripts/build_e1_historical_core.py",
                   "builder_sha256": digest(Path(__file__).read_bytes())})}
    outputs["manifest.json"] = json_bytes({"publication_schema": "e1_public_metadata_allowlist_v1",
        "approved_public_files": sorted(outputs),
        "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in sorted(outputs.items())}})
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = compile_outputs()
    for target in (STAGE, MIRROR):
        if args.check:
            if not target.is_dir() or {p.name for p in target.iterdir()} != set(outputs):
                raise ValueError("Historical-core output file set differs")
            for name, raw in outputs.items():
                if (target / name).read_bytes() != raw:
                    raise ValueError("Historical-core output differs: " + str(target / name))
        else:
            target.mkdir(parents=True, exist_ok=True)
            for name, raw in outputs.items():
                (target / name).write_bytes(raw)
    print(json.dumps({"status": "checked" if args.check else "generated", "runs": 240,
                      "core_models": list(MODELS), "new_fits": 0, "daily_vectors_read": 0}))


if __name__ == "__main__":
    main()
