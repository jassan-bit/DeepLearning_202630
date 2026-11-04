"""Publish aggregate summaries of the four-history Filipe experiment only.

This exporter never trains, chooses a model from test scores, or opens raw market
archives. The original D7 experiment remains an immutable, separately scoped
record. Prediction vectors, daily market derivatives, checkpoints and archives
remain local. Run --phase complete after the registered final evaluation exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/filipe_multihistory_20261104"
OLD = ROOT / "results/dl_r2_cv_20261104"
DESTINATION = ROOT / "book/_static/filipe_multihistory_20261104"
EXPERIMENT = ROOT / "experiments/filipe_multihistory_20261104"
HISTORIES = (7, 14, 21, 28)
MODELS = ("MLP", "LSTM", "CNN1D")
SEEDS = tuple(range(4101, 4111))
METRICS = ("rmse", "mae", "mse", "mape", "r2", "mase")
ANALYSIS_FILES = (
    "run_metrics.csv", "fold_metrics.csv", "metrics_summary.csv", "fold_metrics_summary.csv",
    "between_fold_spread_descriptive.csv", "bds_by_fold.csv", "bds_by_fold_summary.csv",
    "bds_mean_across_folds_descriptive.csv", "bds_pvalues_descriptive_summary.csv",
    "tables.md", "metrics_by_input_history.png", "cv_selection.json", "summary.json",
)
ROOT_SUMMARIES = ("execution_summary.json", "runtime.json", "code_manifest_before_fitting.json",
                  "runs.csv", "fold_metrics.csv", "run_artifacts_manifest.json")
TEST_SUMMARIES = ("raw_access_started.json", "raw_test_audit.json", "cohort_manifest.json",
                  "eligibility.csv", "execution_summary.json", "runs.csv", "fold_metrics.csv")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def copy_file(source: Path, destination: Path, records: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    try:
        relative = source.relative_to(ROOT).as_posix()
    except ValueError:
        relative = source.as_posix()
    records[relative] = digest(source)


def number(value, digits=7) -> str:
    if value is None or not np.isfinite(float(value)):
        return "No definido"
    return f"{float(value):.{digits}f}".replace(".", ",")


def probability_number(value) -> str:
    if value is None or not np.isfinite(float(value)):
        return "No definido"
    value = float(value)
    if 0 < abs(value) < 1e-5:
        return f"{value:.6e}".replace(".", ",")
    return number(value, 6)


def metric_cell(row: pd.Series, metric: str) -> str:
    mean = row.get("mean", np.nan)
    sd = row.get("sample_sd", row.get("std", row.get("sd", np.nan)))
    digits = 2 if metric == "mape" else (9 if metric == "mse" else (5 if metric in ("r2", "mase") else 7))
    text = number(mean, digits)
    return text + (" ± " + number(sd, digits) if pd.notna(sd) else "; referencia determinista")


def table(headers, rows) -> str:
    return ("| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers)
            + "|\n" + "".join("| " + " | ".join(str(value) for value in row) + " |\n" for row in rows))


def find_summary(directory: Path) -> Path:
    for name in ("metrics_summary.csv", "annual_summary.csv", "annual_metrics_summary.csv"):
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError("No long-form metric summary in " + str(directory))


def validate_analysis(directory: Path, phase: str) -> dict:
    report = json.loads((directory / "summary.json").read_text())
    if report.get("phase") != phase or not report.get("bds"):
        raise ValueError("Completed per-fold BDS analysis required for " + phase)
    for name, expected in report.get("source_sha256", {}).items():
        if digest(ROOT / name) != expected:
            raise ValueError("Analysis input changed: " + name)
    for name, expected in report.get("output_sha256", {}).items():
        if digest(directory / name) != expected:
            raise ValueError("Analysis output changed: " + name)
    return report


def make_metric_tables(directory: Path, phase: str, destination: Path) -> None:
    summary = pd.read_csv(find_summary(directory))
    if not {"history_days", "model", "metric", "mean"}.issubset(summary.columns):
        raise ValueError("Metric summary needs history_days/model/metric/mean columns")
    # The phase directory contains summaries for the full evaluation period;
    # monthly/annual slices belong in their own downloadable CSV files.
    if "period" in summary and summary.period.nunique() > 1:
        candidate = summary.loc[summary.period.astype(str).isin(["all", "full", "annual", "overall"])]
        if not candidate.empty:
            summary = candidate
    if summary.duplicated(["history_days", "model", "metric"]).any():
        raise ValueError("Ambiguous full-period metric summary")
    lookup = summary.set_index(["history_days", "model", "metric"])
    headers = ["Entrada D (días)", "Modelo", "RMSE", "MAE", "MSE", "MAPE (%)", "R²", "MASE"]
    rows = []
    for history in HISTORIES:
        names = [*MODELS]
        if "Persistencia" in set(summary.model):
            names.append("Persistencia")
        for model in names:
            cells = [str(history), model]
            for metric in METRICS:
                key = (history, model, metric)
                cells.append(metric_cell(lookup.loc[key], metric) if key in lookup.index else "No definido")
            rows.append(cells)
    prefix = ("Validación forward chaining de 2024" if phase == "cv" else "Prueba reservada: enero de 2025–agosto de 2026")
    note = (prefix + ". Media ± DE muestral entre diez semillas; las fechas y el objetivo n=7/h=1 se mantienen. "
            "Persistencia es una referencia determinista, sin réplicas de entrenamiento. "
            "Los intervalos, cobertura y recuentos definidos se conservan en el CSV.\n\n")
    public_analysis = "analysis_cv" if phase == "cv" else "test/analysis"
    link = f"\n[Descargar resumen completo](../_static/filipe_multihistory_20261104/{public_analysis}/{find_summary(directory).name}).\n"
    (destination / f"{phase}_models.md").write_text(note + table(headers, rows) + link)
    for model in MODELS:
        selected = [row for row in rows if row[1] == model]
        (destination / f"{phase}_{model}.md").write_text(note + table(headers, selected) + link)


def copy_analysis_summaries(source: Path, destination: Path, records: dict) -> None:
    for name in ANALYSIS_FILES:
        path = source / name
        if path.exists():
            if name == "summary.json":
                report = json.loads(path.read_text())
                report.update({"publication_scope": "aggregate summaries only; referenced prediction vectors retained locally",
                               "individual_predictions_retained_local": True,
                               "source_summary_sha256": digest(path), "derived_summary": True})
                json_write(destination / name, report)
                records[path.relative_to(ROOT).as_posix()] = digest(path)
            else:
                copy_file(path, destination / name, records)


def assert_summaries_only(directory: Path) -> None:
    """Fail closed if a prohibited artifact or per-date market column appears."""
    forbidden_columns = {"y_true", "y_pred", "persistence_pred", "residual", "residuals",
                         "open", "high", "low", "close", "volume", "sigma_7"}
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        if path.name.endswith((".tar.gz", ".zip", ".pt", ".pth", ".npz", ".npy")):
            raise ValueError("Only summary publication is supported: " + str(relative))
        if "predictions" in path.name or path.name.startswith("daily_") or relative.parts[0] == "final_fit":
            raise ValueError("Individual predictions/market derivatives remain local: " + str(relative))
        if path.suffix == ".csv":
            columns = set(pd.read_csv(path, nrows=0).columns)
            if columns & forbidden_columns:
                raise ValueError("Per-date market/prediction columns cannot be exported: " + str(relative))


def update_current_metadata() -> None:
    """Refresh mutable editorial metadata; never rewrite signed old experiments."""
    freeze = json.loads((EXPERIMENT / "freeze.json").read_text())
    cv = pd.read_csv(SOURCE / "analysis_cv/metrics_summary.csv")
    test = pd.read_csv(SOURCE / "test/analysis/metrics_summary.csv")
    execution = json.loads((SOURCE / "execution_summary.json").read_text())
    test_execution = json.loads((SOURCE / "test/execution_summary.json").read_text())
    final = json.loads((SOURCE / "final_models_manifest.json").read_text())
    current = {
        "method": "timeseries-cv monthly forward chaining, then closed development refits and reserved test",
        "path": "results/filipe_multihistory_20261104", "models": list(MODELS),
        "input_histories_days": list(HISTORIES), "target_volatility_returns": 7, "forecast_horizon_days": 1,
        "primary_preprocessing": "relative_window", "folds": 12, "training_seeds_per_history_model": 10,
        "primary_cv_trajectories": 120, "primary_cv_fold_model_segments": 1440,
        "primary_cv_effective_fits": 2880, "primary_cv_prediction_rows": 43920,
        "new_D14_D21_D28_execution": execution, "D7_relative_source_reused": "results/dl_r2_cv_20261104",
        "final_development_fits": int(final["completed_project_fits"]), "test_execution": test_execution,
        "final_test_evaluated": True, "test_selection_allowed": False,
        "configuration_freeze_sha256": digest(EXPERIMENT / "freeze.json"),
        "selection": freeze.get("selection", {}), "applied_literature_counts_unchanged": True,
        "publication_scope": "aggregate summaries and non-market metadata only",
        "individual_predictions_retained_local": True,
        "market_derivatives_retained_local": True,
        "weights_and_archives_retained_local": True,
    }
    public = ROOT / "book/_static/literature_e1"
    for name in ("guide_aligned_proposal_20261104.json", "selection_evidence.json"):
        path = public / name
        value = json.loads(path.read_text())
        prior = value.get("current_project_execution", {})
        if prior.get("path") == "results/dl_r2_cv_20261104":
            value["initial_D7_project_execution"] = prior
        value["current_project_execution"] = current
        value["status"] = "documentary_evidence_with_four_history_forward_cv_and_closed_test_execution"
        if name.startswith("guide_aligned"):
            value["final_test_protocol_frozen"] = True
            value["final_test_protocol_scope"] = (
                "Configuration, epochs, train-only preprocessing, masks and non-updating test policy closed before raw test access; "
                "selection uses2024 only; monthly2025-2026 groups are reporting blocks, not refit folds.")
            for model in value["models"]:
                family = "CNN1D" if model["family"].startswith("CNN1D") else model["family"]
                selected = freeze.get("selection", {}).get("selected_by_model", {}).get(family, {})
                history = int(selected.get("history_days", 7))
                parameters = 80 * history + 33 if family == "MLP" else 5025 if family == "LSTM" else 1057
                by_history = []
                for days in HISTORIES:
                    row = {"history_days": days}
                    for metric in METRICS:
                        for phase, frame in (("cv", cv), ("test", test)):
                            match = frame.loc[frame.history_days.eq(days) & frame.model.eq(family) & frame.metric.eq(metric)]
                            if len(match) != 1:
                                raise ValueError("Ambiguous current model metadata metric")
                            row[f"{phase}_{metric}_mean"] = float(match.iloc[0]["mean"])
                            row[f"{phase}_{metric}_sd"] = float(match.iloc[0]["sample_sd"])
                    by_history.append(row)
                configuration = model.get("current_development_configuration", {}).copy()
                for key in list(configuration):
                    if key.startswith("cv_"):
                        configuration.pop(key)
                configuration.update({"history_days": history, "parameters": parameters, "n_seeds": 10,
                                      "monthly_cv_folds": 12, "selection_moment": "before reserved test",
                                      "primary_preprocessing": "relative_window"})
                configuration.update({key: val for key, val in selected.items() if key.startswith("cv_")})
                model["current_development_configuration"] = configuration
                model["evaluated_results_by_input_history"] = by_history
                model["architecture_configuration"]["parameters"] = parameters
                model["architecture_configuration"]["parameters_by_history"] = {
                    str(days): 80 * days + 33 if family == "MLP" else parameters for days in HISTORIES}
                model["configuration_scope"] = "Project family with input history selected by2024 CV before test; not parameters extracted from a paper"
        json_write(path, value)
    path = ROOT / "book/_static/e1/decisiones.json"
    value = json.loads(path.read_text())
    if value.get("scope") == "CV exploration on development2024 only":
        value["initial_D7_execution"] = value.get("execution")
        value["initial_D7_contrasts"] = value.get("current_only_contrasts")
    value.update({"scope": "Four input histories, exposed2024 development selection, then configuration-closed reserved test",
                  "configuration": "../filipe_multihistory_20261104/freeze.json", "execution": current,
                  "current_only_contrasts": "Per-month/seed BDS families for CV andtest; p-value averages descriptive. InitialD7 Holm12/DM6/LB-BDS15 remain separately scoped.",
                  "reserved_test_evaluated": True, "test_used_for_selection": False,
                  "test_quality_audit": "../filipe_multihistory_20261104/test/raw_test_audit.json",
                  "test_sample_manifest": "../filipe_multihistory_20261104/test/cohort_manifest.json"})
    value["pending"] = [item for item in value.get("pending", []) if item not in {"final development fit", "reserved test evaluation"}]
    value.pop("test_files_opened_in_training_analysis", None)
    json_write(path, value)


def make_bds_table(analysis: Path, phase: str, destination: Path) -> None:
    files = sorted(analysis.glob("*bds*.csv"))
    text = (
        "BDS se calcula sobre los residuos de horizonte h=1 de cada bloque y semilla, con dimensiones 2 y 3 y radio de 1,5 DE muestral. "
        "Los bloques mensuales son cortos y los p-valores asintóticos se interpretan como diagnósticos exploratorios. "
        "El promedio y la DE de p-valores son descriptivos; no constituyen un contraste combinado ni una probabilidad de independencia.\n\n"
    )
    compact = analysis / "bds_pvalues_descriptive_summary.csv"
    if compact.exists():
        frame = pd.read_csv(compact)
        rows = []
        for _, row in frame.sort_values(["history_days", "model", "dimension"]).iterrows():
            rows.append([int(row.history_days), row.model, int(row.dimension),
                         f"{int(row.n_defined)}/{int(row.n_planned)}",
                         probability_number(row.mean_p_value_descriptive), probability_number(row.sd_p_value_descriptive),
                         int(row.n_Holm_below_005)])
        text += table(["D", "Modelo", "Dimensión", "p definidos / previstos", "Media p (descriptiva)",
                       "DE p (descriptiva)", "Rechazos Holm, α=0,05"], rows) + "\n"
    public_analysis = "analysis_cv" if phase == "cv" else "test/analysis"
    for path in files:
        text += f"- [{path.name}](../_static/filipe_multihistory_20261104/{public_analysis}/{path.name}).\n"
    if not files:
        raise FileNotFoundError("No per-fold BDS output for " + phase)
    (destination / f"bds_{phase}.md").write_text(text)


def export(phase: str) -> dict:
    required = [EXPERIMENT / "protocol.json", SOURCE / "execution_summary.json", SOURCE / "analysis_cv/summary.json"]
    if phase == "complete":
        required += [EXPERIMENT / "freeze.json", SOURCE / "test/analysis/summary.json",
                     SOURCE / "final_models_manifest.json", SOURCE / "test/execution_summary.json",
                     SOURCE / "test/cohort_manifest.json", SOURCE / "test/raw_test_audit.json"]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing recorded results: " + ", ".join(missing))
    inputs = {}
    cv_report = validate_analysis(SOURCE / "analysis_cv", "cv")
    if cv_report.get("neural_forecasts") != 43920 or cv_report.get("outcome_days") != 366:
        raise ValueError("The recorded primary CV summary must preserve all four histories")
    test_report = None
    test_prediction_count = test_date_count = 0
    if phase == "complete":
        test_report = validate_analysis(SOURCE / "test/analysis", "test")
        cohort = json.loads((SOURCE / "test/cohort_manifest.json").read_text())
        expected_dates = sorted(cohort["common_outcome_days_utc"])
        test_prediction_count = int(test_report["neural_forecasts"])
        test_date_count = int(test_report["outcome_days"])
        if len(set(expected_dates)) != cohort["n_common_outcomes"] or test_date_count != cohort["n_common_outcomes"]:
            raise ValueError("The test summary differs from its calendar-only common cohort")
        if test_prediction_count != 120 * test_date_count:
            raise ValueError("The test summary must describe 120 predeclared trajectories")
        freeze_sha = digest(EXPERIMENT / "freeze.json")
        final = json.loads((SOURCE / "final_models_manifest.json").read_text())
        audit = json.loads((SOURCE / "test/raw_test_audit.json").read_text())
        execution = json.loads((SOURCE / "test/execution_summary.json").read_text())
        if final["freeze_sha256"] != freeze_sha or audit["freeze_sha256"] != freeze_sha:
            raise ValueError("Test/final models do not belong to the recorded freeze")
        if audit["final_models_manifest_sha256"] != digest(SOURCE / "final_models_manifest.json"):
            raise ValueError("Final model manifest changed after raw test access")
        if pd.Timestamp(final["completed_at_utc"]) > pd.Timestamp(audit["first_raw_access_at_utc"]):
            raise ValueError("Final models must be frozen before raw test access")
        if execution.get("test_period_model_refits") != 0 or execution.get("test_used_for_configuration_selection") is not False:
            raise ValueError("Test may not refit or select the configuration")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".filipe_multihistory_export_", dir=DESTINATION.parent) as temporary:
        build = Path(temporary)
        for name in ROOT_SUMMARIES:
            path = SOURCE / name
            if path.exists():
                if name == "run_artifacts_manifest.json":
                    manifest = json.loads(path.read_text())
                    manifest["publication_scope"] = "Hashes and file identities only; complete artifacts and archives retained locally"
                    manifest["archives_retained_local"] = True
                    manifest["source_manifest_sha256"] = digest(path)
                    manifest["derived_summary"] = True
                    json_write(build / name, manifest)
                    inputs[path.relative_to(ROOT).as_posix()] = digest(path)
                else:
                    copy_file(path, build / name, inputs)
        copy_analysis_summaries(SOURCE / "analysis_cv", build / "analysis_cv", inputs)
        for history in (14, 21, 28):
            # Twelve scalar/date/hash records, not full arrays of market inputs.
            path = SOURCE / f"D{history}/folds.csv"
            if path.exists():
                copy_file(path, build / f"D{history}/folds.csv", inputs)
        if phase == "complete":
            copy_analysis_summaries(SOURCE / "test/analysis", build / "test/analysis", inputs)
            for name in TEST_SUMMARIES:
                path = SOURCE / "test" / name
                if path.exists():
                    copy_file(path, build / "test" / name, inputs)
            copy_file(SOURCE / "final_fit_summary.csv", build / "final_fit_summary.csv", inputs)
            model_manifest = json.loads((SOURCE / "final_models_manifest.json").read_text())
            public_manifest = {key: value for key, value in model_manifest.items() if key != "jobs"}
            public_manifest.update({"publication_scope": "SHA-256 identities and execution metadata only; checkpoints, scales and per-run configuration files retained locally",
                                    "complete_manifest_sha256": digest(SOURCE / "final_models_manifest.json"),
                                    "source_manifest_sha256": digest(SOURCE / "final_models_manifest.json"),
                                    "derived_summary": True,
                                    "complete_manifest_retained_local": True})
            json_write(build / "final_models_manifest.json", public_manifest)
            inputs["results/filipe_multihistory_20261104/final_models_manifest.json"] = digest(SOURCE / "final_models_manifest.json")
        copy_file(EXPERIMENT / "protocol.json", build / "protocol.json", inputs)
        if phase == "complete":
            copy_file(EXPERIMENT / "freeze.json", build / "freeze.json", inputs)
        # No recursive copying: publication uses an explicit summary allowlist.
        for name in ("prepare_filipe_multihistory.py", "train_filipe_multihistory.py", "analyze_filipe_multihistory.py",
                     "load_filipe_reserved_test.py", "final_fit_filipe_multihistory.py", Path(__file__).name):
            path = ROOT / "scripts" / name
            if path.exists():
                copy_file(path, build / "code" / name, inputs)
        tables = build / "tables"
        tables.mkdir()
        make_metric_tables(SOURCE / "analysis_cv", "cv", tables)
        make_bds_table(SOURCE / "analysis_cv", "cv", tables)
        if phase == "complete":
            make_metric_tables(SOURCE / "test/analysis", "test", tables)
            make_bds_table(SOURCE / "test/analysis", "test", tables)
        parameters = [[str(history), str(80 * history + 33), "5.025", "1.057"] for history in HISTORIES]
        (tables / "parameters.md").write_text(table(["Entrada D", "Parámetros MLP", "Parámetros LSTM", "Parámetros CNN1D"], parameters))
        if phase == "complete":
            freeze = json.loads((EXPERIMENT / "freeze.json").read_text())
            selection = freeze.get("selection", {})
            selection_rows = []
            for model in MODELS:
                selected = selection.get("selected_by_model", {}).get(model, {})
                selection_rows.append([model, selected.get("history_days", "No registrado"),
                                       number(selected.get("cv_rmse_mean")), "Elegida con CV 2024"])
            selected_text = table(["Familia", "D seleccionada", "RMSE media de CV", "Momento de selección"], selection_rows)
            baseline = selection.get("baseline", {})
            selected_text += (f"\nBaseline neuronal cerrado antes de test: **{baseline.get('model', 'No registrado')}, "
                              f"D={baseline.get('history_days', 'No registrado')}**. Las doce combinaciones se evalúan y reportan; "
                              "el orden observado en test no reabre esta elección.\n\n"
                              "[Consultar congelación y hashes](../_static/filipe_multihistory_20261104/freeze.json).\n")
            (tables / "selected_configuration.md").write_text(selected_text)
        old_summary = json.loads((OLD / "execution_summary.json").read_text())
        new_summary = json.loads((SOURCE / "execution_summary.json").read_text())
        execution_rows = [
            ["CV principal, cuatro historias", "120 trayectorias; 1.440 segmentos; 2.880 ajustes; 43.920 predicciones", "Incluye treinta trayectorias relativas D7 reutilizadas"],
            ["Ampliación D14/21/28", "90 trayectorias; 1.080 segmentos; 2.160 ajustes; 32.940 predicciones", "Nuevos ajustes de desarrollo, sin volver a ajustar D7"],
        ]
        if phase == "complete":
            execution_rows += [
                ["Ajuste final", "120 modelos y escaladores firmados antes de abrir test", "Desarrollo permitido2023–2024 y embargo D"],
                ["Test", f"{test_prediction_count} predicciones; {test_date_count} fechas comunes", "Veinte meses de reporte, cero reajustes y una llamada de predicción por checkpoint"],
            ]
        (tables / "execution.md").write_text(table(["Fase", "Ejecución", "Alcance"], execution_rows))
        report = {
            "exported_at_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
            "method": "CV forward chaining de Filipe / timeseries-cv, doce bloques mensuales de 2024",
            "histories_days": list(HISTORIES), "volatility_returns": 7, "forecast_horizon_days": 1,
            "primary_preprocessing": "relative_window", "models": list(MODELS), "seeds": list(SEEDS),
            "cv_primary_trajectories": 120, "cv_primary_prediction_rows": int(cv_report["neural_forecasts"]),
            "cv_analysis": cv_report,
            "old_D7_source": "results/dl_r2_cv_20261104; relative_window only, unchanged",
            "old_D7_source_execution": old_summary,
            "new_D14_D21_D28_execution": new_summary,
            "reserved_test_evaluated": phase == "complete",
            "test_prediction_rows": test_prediction_count,
            "publication_scope": "aggregate summaries only",
            "individual_predictions_retained_local": True,
            "market_derivatives_retained_local": True,
            "weights_and_archives_retained_local": True,
            "referenced_private_inputs_retained_local": True,
            "new_fits_in_export": 0, "raw_market_files_opened_in_export": 0,
            "scope": "Desarrollo expuesto para selección; test final separado tras congelación, sin reseleccionar por sus métricas.",
            "bds_scope": "Por bloque y semilla; aproximación asintótica con muestras mensuales cortas; p medio solo descriptivo.",
            "source_sha256": inputs,
        }
        json_write(build / "method.json", report)
        (build / "README.md").write_text(
            "# CV de Filipe con cuatro historias: publicación de resúmenes\n\n"
            "D=7/14/21/28 días de OHLCV; objetivo sigma7 poblacional y horizonte de un día. "
            "D7 reutiliza la representación relativa de su experimento firmado; las demás historias se ajustan en una ampliación independiente. "
            "Los datos originales, predicciones y artefactos registrados no son modificados por este exportador. "
            "El control de precios absolutos pertenece exclusivamente al ciclo inicial D7.\n\n"
            "La evaluación de test, cuando phase=complete, usa pesos y escaladores fijados antes de abrir su bruto; "
            "sus meses son bloques de reporte, no reajustes ni nuevos folds de selección. "
            "Los resultados no acreditan un piloto prospectivo ni cierran el respaldo bibliográfico de MLP/CNN1D.\n\n"
            "## Resultados y auditoría\n\n"
            "`analysis_cv/` conserva métricas y BDS mensuales de validación; `test/analysis/` conserva la evaluación final. "
            "`freeze.json`, `final_models_manifest.json`, `test/raw_test_audit.json` y `test/cohort_manifest.json` "
            "identifican selección previa, firmas, primera lectura y elegibilidad. Los meses de test no implican nuevos entrenamientos.\n\n"
            "## Alcance de la publicación\n\n"
            "Esta carpeta pública contiene exclusivamente resúmenes agregados por corrida o mes, pruebas BDS, "
            "reglas, fechas, calidad, firmas SHA-256 y código. "
            "Las predicciones individuales, vectores de objetivos o residuos, derivados diarios de mercado, "
            "escaladores, checkpoints y archivos TAR completos permanecen en el espacio local del experimento; "
            "no se ofrecen como descargas públicas. `final_models_manifest.json` publica únicamente identidades "
            "y hashes; `run_artifacts_manifest.json` identifica los materiales conservados localmente. "
            "`public_allowlist_manifest.json` enumera y verifica los archivos permitidos en esta vista.\n\n"
            "## Código y entornos\n\n"
            "La carpeta `code/` conserva los nuevos scripts y el repositorio incluye los helpers firmados y locks de dependencias. "
            "`prepare_filipe_multihistory.py` usa el entorno NumPy1.26/timeseries-cv0.1.5; los entrenamientos usan el entorno PyTorch registrado. "
            "`analyze_filipe_multihistory.py --phase cv` y `--phase test` reutilizan predicciones guardadas, sin entrenar. "
            "`export_filipe_multihistory_book.py --phase complete` valida firmas y exporta solo resúmenes, sin abrir brutos o publicar vectores y pesos. "
            "La secuencia de un experimento nuevo es preparar índices, registrar y entrenar CV, cerrar configuraciones, "
            "ajustar y firmar modelos finales, abrir el test autorizado una vez y analizar las predicciones. "
            "No se deben modificar o sobrescribir los registros firmados de este experimento para probar otra decisión después de observar test.\n")
        assert_summaries_only(build)
        outputs = {path.relative_to(build).as_posix(): digest(path) for path in sorted(build.rglob("*")) if path.is_file()}
        json_write(build / "public_allowlist_manifest.json", {
            "publication_scope": "aggregate summaries and non-market metadata only",
            "policy": "Explicit file allowlists; no prediction/residual/market vectors, daily market derivatives, checkpoints or archives",
            "files_sha256": outputs,
            "individual_predictions_retained_local": True,
            "market_derivatives_retained_local": True,
            "weights_and_archives_retained_local": True,
            "csv_headers_checked": True,
            "csv_forbidden_columns": ["y_true", "y_pred", "persistence_pred", "residual", "residuals", "open", "high", "low", "close", "volume", "sigma_7"],
        })
        outputs["public_allowlist_manifest.json"] = digest(build / "public_allowlist_manifest.json")
        json_write(build / "export_manifest.json", {"outputs": outputs, "source_sha256": inputs,
                   "exporter_sha256": digest(Path(__file__))})
        if any(digest(ROOT / name) != expected for name, expected in inputs.items()):
            raise RuntimeError("Recorded input changed while exporting")
        if DESTINATION.exists():
            shutil.rmtree(DESTINATION)
        shutil.copytree(build, DESTINATION)
    if phase == "complete":
        update_current_metadata()
    return {"export": DESTINATION.relative_to(ROOT).as_posix(), "phase": phase,
            "publication_scope": "summaries_only", "cv_prediction_count_reported": int(cv_report["neural_forecasts"]),
            "test_prediction_count_reported": test_prediction_count, "prediction_vectors_exported": 0,
            "checkpoints_exported": 0, "archives_exported": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cv", "complete"), default="cv")
    args = parser.parse_args()
    print(json.dumps(export(args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()
