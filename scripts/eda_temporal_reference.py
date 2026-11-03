"""A static regularized linear reference for temporal-regression EDA only.

Four fixed-alpha Ridge references use allowed 2023 rows and predict 2024 solely
to examine residual dependence. No deep-learning model, pilot, benchmark or test
archive is run. Input calendars, missing values and the agreed embargo remain
unchanged. Predictions are not clipped and no history or alpha is selected.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e1-temporal-reference")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache-e1-temporal-reference")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln, log_ndtr, logsumexp
from scipy.stats import chi2, norm
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf


ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "results/eda_development/daily_market_development.csv"
TARGETS = ROOT / "results/eda_development/daily_volatility_development.csv"
FOLDS = ROOT / "results/cv_forward_chaining/folds.csv"
OUT = ROOT / "results/eda_temporal_reference"
ASSETS = ROOT / "book/_static/eda_temporal_reference"
FEATURES = ("open", "high", "low", "close", "volume")
HISTORIES = (7, 14, 21, 28)
ALPHA = 1.0
LB_LAGS = (7, 14, 28)
ACF_LAGS = 28
CALENDAR = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
FIRST_VALIDATION = pd.Timestamp("2024-01-01", tz="UTC")
CREDIT = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_development(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["market_day_utc"]).set_index("market_day_utc")
    if not frame.index.equals(CALENDAR):
        raise ValueError(f"{path.name}: only the complete 2023--2024 UTC calendar is admitted")
    return frame


def make_windows(values: np.ndarray, history: int) -> np.ndarray:
    """Oldest day first; within each day, the five OHLCV variables in FEATURES."""
    windows = np.full((len(values), history * len(FEATURES)), np.nan)
    for outcome in range(history, len(values)):
        windows[outcome] = values[outcome-history:outcome].reshape(-1)
    return windows


def chi_square_logsf(q: float, degrees: int) -> float:
    """Stable chi-square reference tail for the integer degrees used by Ljung--Box.

    The even case is a finite gamma sum; the odd case starts at the Gaussian
    erfc identity and applies the upper-gamma recurrence. Retaining log tails
    avoids printing a floating-point underflow as an exact zero probability.
    """
    if q == 0:
        return 0.0
    if q < 0 or degrees < 1:
        raise ValueError("Nonnegative Q and positive integer degrees required")
    x = q / 2.0
    if degrees % 2 == 0:
        k = np.arange(degrees // 2, dtype=float)
        return float(-x + logsumexp(k * np.log(x) - gammaln(k + 1.0)))
    result = float(np.log(2.0) + log_ndtr(-np.sqrt(q)))
    a = .5
    for _ in range((degrees - 1) // 2):
        result = float(np.logaddexp(result, a * np.log(x) - x - gammaln(a + 1.0)))
        a += 1.0
    return min(result, 0.0)


def log_holm(log_pvalues: np.ndarray) -> np.ndarray:
    log_pvalues = np.asarray(log_pvalues, dtype=float)
    order = np.argsort(log_pvalues, kind="stable")
    ordered = np.minimum(0.0, np.maximum.accumulate(
        np.log(len(order) - np.arange(len(order))) + log_pvalues[order]))
    adjusted = np.empty_like(ordered)
    adjusted[order] = ordered
    return adjusted


def probability_display(log_value: float) -> str:
    if log_value < -300.0 * np.log(10.0):
        return "<1e-300"
    return f"{np.exp(log_value):.6g}"


def ridge_reference(x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray):
    mean = x_train.mean(axis=0)
    observed_std = x_train.std(axis=0, ddof=0)
    scale = np.where(observed_std == 0.0, 1.0, observed_std)
    standardized = (x_train - mean) / scale
    validation_standardized = (x_validation - mean) / scale
    z_mean = standardized.mean(axis=0)
    y_mean = float(y_train.mean())
    centered = standardized - z_mean
    centered_y = y_train - y_mean
    gram = centered.T @ centered + ALPHA * np.eye(centered.shape[1])
    rhs = centered.T @ centered_y
    coefficient = cho_solve(cho_factor(gram, lower=True, check_finite=True), rhs)
    intercept = float(y_mean - z_mean @ coefficient)
    prediction = validation_standardized @ coefficient + intercept

    # An independent augmented least-squares formulation checks the objective
    # and the unpenalized intercept; it is not an additional model or search.
    design = np.column_stack((np.ones(len(y_train)), standardized))
    penalty = np.column_stack((np.zeros(len(coefficient)), np.sqrt(ALPHA) * np.eye(len(coefficient))))
    solution = np.linalg.lstsq(np.vstack((design, penalty)),
                               np.concatenate((y_train, np.zeros(len(coefficient)))), rcond=None)[0]
    check_prediction = np.column_stack((np.ones(len(prediction)), validation_standardized)) @ solution
    assert np.allclose(check_prediction, prediction, rtol=1e-10, atol=1e-11)
    raw_coefficient = coefficient / scale
    raw_intercept = float(intercept - mean @ raw_coefficient)
    assert np.allclose(x_validation @ raw_coefficient + raw_intercept,
                       prediction, rtol=1e-10, atol=1e-11)
    assert np.max(np.abs(standardized.mean(axis=0))) < 1e-10
    assert np.max(np.abs((gram @ coefficient) - rhs)) < 1e-10
    return {"mean": mean, "scale": scale, "std": observed_std,
            "standardized_mean": z_mean, "coefficient": coefficient,
            "intercept_scaled": intercept, "coefficient_raw": raw_coefficient,
            "intercept_raw": raw_intercept, "prediction": prediction,
            "augmented_solution_prediction_max_abs_difference": float(np.max(np.abs(check_prediction-prediction)))}


def write_csv(frame: pd.DataFrame, filename: str):
    frame.to_csv(OUT / filename, index=False, lineterminator="\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    sources = (MARKET, TARGETS, FOLDS)
    before = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    market = read_development(MARKET)
    targets = read_development(TARGETS)
    manifest = pd.read_csv(FOLDS)
    values = market.loc[:, FEATURES].to_numpy(dtype=float)
    labels = targets["sigma_7"].to_numpy(dtype=float)
    first = int(CALENDAR.get_loc(FIRST_VALIDATION))
    validation_indices = np.flatnonzero(CALENDAR.year == 2024)
    assert len(validation_indices) == 366
    assert market.loc[pd.Timestamp("2023-03-24", tz="UTC"), list(FEATURES)].isna().all()

    # Validate stable reference tails and Holm against independent library cases.
    for degrees in LB_LAGS:
        for q in (.1, 1.0, 10.0, 50.0, 100.0, 500.0):
            assert np.isclose(chi_square_logsf(q, degrees), chi2.logsf(q, degrees),
                              rtol=1e-12, atol=1e-11)
    fixture = np.array([.01, .2, .03, .4])
    assert np.allclose(np.exp(log_holm(np.log(fixture))),
                       multipletests(fixture, method="holm")[1])

    predictions, training_rows, coefficients, summaries, acf_rows, lb_rows = [], [], [], [], [], []
    residual_sequences = {}
    for history in HISTORIES:
        windows = make_windows(values, history)
        cutoff = first - history - 1
        eligible = np.isfinite(windows).all(axis=1) & np.isfinite(labels)
        train_indices = np.flatnonzero(eligible & (np.arange(len(CALENDAR)) <= cutoff)
                                      & (CALENDAR.year == 2023))
        assert len(train_indices) == {7: 343, 14: 322, 21: 301, 28: 280}[history]
        assert train_indices.max() == cutoff
        assert eligible[validation_indices].all()
        index_hash = hashlib.sha256(train_indices.astype("<i8").tobytes()).hexdigest()
        first_manifest = manifest[(manifest.input_days == history) & (manifest.volatility_days == 7)
                                  & (manifest.validation_outcome_day_utc == "2024-01-01")]
        assert len(first_manifest) == 1
        recorded = first_manifest.iloc[0]
        assert int(recorded.train_samples) == len(train_indices)
        assert recorded.train_outcome_indices_sha256 == index_hash
        assert recorded.train_last_outcome_day_utc == CALENDAR[cutoff].strftime("%Y-%m-%d")
        # The fixed 2023 model is at least as embargoed as every later origin.
        assert np.all(train_indices.max() <= validation_indices-history-1)
        fit = ridge_reference(windows[train_indices], labels[train_indices], windows[validation_indices])
        errors = labels[validation_indices] - fit["prediction"]
        assert np.isfinite(errors).all() and len(errors) == 366
        residual_sequences[history] = errors
        for index in train_indices:
            training_rows.append({"history_days": history, "outcome_index": int(index),
                                  "outcome_day_utc": CALENDAR[index].strftime("%Y-%m-%d"),
                                  "first_input_day_utc": CALENDAR[index-history].strftime("%Y-%m-%d"),
                                  "last_input_day_utc": CALENDAR[index-1].strftime("%Y-%m-%d")})
        for position, index in enumerate(validation_indices):
            predictions.append({"history_days": history,
                "outcome_day_utc": CALENDAR[index].strftime("%Y-%m-%d"),
                "prediction_origin_utc": CALENDAR[index].isoformat(),
                "target_available_at_utc": (CALENDAR[index]+pd.Timedelta(days=1)).isoformat(),
                "input_first_day_utc": CALENDAR[index-history].strftime("%Y-%m-%d"),
                "input_last_day_utc": CALENDAR[index-1].strftime("%Y-%m-%d"),
                "fixed_training_last_outcome_day_utc": CALENDAR[cutoff].strftime("%Y-%m-%d"),
                "target_sigma_7": float(labels[index]),
                "reference_prediction_unclipped": float(fit["prediction"][position]),
                "residual_target_minus_prediction": float(errors[position]),
                "negative_reference_prediction": bool(fit["prediction"][position] < 0)})
        for column in range(history * len(FEATURES)):
            day_position, feature_position = divmod(column, len(FEATURES))
            coefficients.append({"history_days": history, "column_position": column,
                "calendar_lag_days": history-day_position, "feature": FEATURES[feature_position],
                "training_mean": float(fit["mean"][column]),
                "training_std_ddof0": float(fit["std"][column]),
                "training_scale_used": float(fit["scale"][column]),
                "coefficient_standardized": float(fit["coefficient"][column]),
                "coefficient_raw": float(fit["coefficient_raw"][column])})
        rho = acf(errors, nlags=ACF_LAGS, fft=False, adjusted=False, missing="raise")
        centered = errors - errors.mean()
        for lag in (1, 7, 28):
            independent_rho = float(centered[:-lag] @ centered[lag:] / (centered @ centered))
            assert np.isclose(rho[lag], independent_rho, rtol=1e-12, atol=1e-13)
        band = float(norm.ppf(.975) / np.sqrt(len(errors)))
        for lag in range(1, ACF_LAGS+1):
            acf_rows.append({"history_days": history, "validation_residuals": len(errors),
                             "lag_calendar_days": lag, "acf": float(rho[lag]),
                             "approximate_white_noise_band_lower": -band,
                             "approximate_white_noise_band_upper": band})
        lb = acorr_ljungbox(errors, lags=list(LB_LAGS), model_df=0, return_df=True)
        for lag in LB_LAGS:
            q = float(lb.loc[lag, "lb_stat"])
            q_check = len(errors)*(len(errors)+2)*sum(rho[k]**2/(len(errors)-k) for k in range(1, lag+1))
            assert np.isclose(q, q_check, rtol=1e-12, atol=1e-10)
            log_p = chi_square_logsf(q, lag)
            lb_rows.append({"history_days": history, "validation_residuals": len(errors),
                "lag_calendar_days": lag, "model_df": 0, "chi_square_reference_df": lag,
                "ljung_box_Q": q, "asymptotic_p_value": float(np.exp(log_p)),
                "log_asymptotic_p_value": log_p, "asymptotic_p_display": probability_display(log_p)})
        summaries.append({"history_days": history, "input_features": history*len(FEATURES),
            "alpha_fixed_for_diagnostic": ALPHA, "training_samples": len(train_indices),
            "training_first_outcome_day_utc": CALENDAR[train_indices.min()].strftime("%Y-%m-%d"),
            "training_last_outcome_day_utc": CALENDAR[cutoff].strftime("%Y-%m-%d"),
            "embargo_days_at_first_validation": history,
            "training_outcome_indices_sha256": index_hash, "cv_manifest_first_origin_matched": True,
            "fixed_model_embargo_verified_for_all_validation_origins": True,
            "validation_residuals": len(errors), "negative_prediction_count": int((fit["prediction"] < 0).sum()),
            "prediction_min": float(fit["prediction"].min()), "prediction_max": float(fit["prediction"].max()),
            "residual_mean": float(errors.mean()), "residual_std_ddof1": float(errors.std(ddof=1)),
            "residual_acf_lag1": float(rho[1]), "residual_acf_lag7": float(rho[7]),
            "intercept_standardized_space": fit["intercept_scaled"], "intercept_raw_space": fit["intercept_raw"],
            "zero_variance_training_features": int((fit["std"] == 0).sum()),
            "augmented_lstsq_prediction_max_abs_difference": fit["augmented_solution_prediction_max_abs_difference"]})

    adjusted_logs = log_holm(np.array([row["log_asymptotic_p_value"] for row in lb_rows]))
    for row, adjusted_log in zip(lb_rows, adjusted_logs):
        row.update(log_p_holm_12=float(adjusted_log), p_holm_12=float(np.exp(adjusted_log)),
                   p_holm_display=probability_display(float(adjusted_log)),
                   below_alpha_under_asymptotic_reference=bool(adjusted_log < np.log(.05)))
    assert len(predictions) == 1464 and len(lb_rows) == 12 and len(acf_rows) == 112
    assert len(training_rows) == 1246 and len(coefficients) == 350
    write_csv(pd.DataFrame(predictions), "validation_predictions_residuals.csv")
    write_csv(pd.DataFrame(training_rows), "training_outcomes.csv")
    write_csv(pd.DataFrame(coefficients), "model_parameters.csv")
    write_csv(pd.DataFrame(summaries), "reference_configurations.csv")
    write_csv(pd.DataFrame(acf_rows), "residual_acf.csv")
    write_csv(pd.DataFrame(lb_rows), "ljung_box_holm.csv")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), constrained_layout=False)
    for row, history in enumerate(HISTORIES):
        errors = residual_sequences[history]
        ax = axes[row, 0]
        ax.plot(CALENDAR[validation_indices], errors*100, color="#265b96", linewidth=1)
        ax.axhline(0, color="#777777", linewidth=.7)
        ax.set_title(f"D={history}: residuos de 2024 · ajuste fijo de 2023")
        ax.set_ylabel("Residuo ×100 (p.p.)")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax = axes[row, 1]
        rho = [item["acf"] for item in acf_rows if item["history_days"] == history]
        band = norm.ppf(.975)/np.sqrt(366)
        ax.axhspan(-band, band, color="#dce7f1", label="Referencia puntual aprox. 95 %")
        ax.vlines(np.arange(1, 29), 0, rho, color="#265b96", linewidth=1)
        ax.scatter(np.arange(1, 29), rho, color="#265b96", s=10)
        ax.axhline(0, color="#777777", linewidth=.7)
        ax.set(title=f"D={history}: ACF residual · bandas no simultáneas", ylabel="ACF",
               xlim=(0, 29), ylim=(-1, 1), xticks=[1, 7, 14, 21, 28])
        if row == 0: ax.legend(loc="lower right", fontsize=7)
    axes[-1, 0].set_xlabel("Día de resultado UTC; etiqueta observada al terminarlo")
    axes[-1, 1].set_xlabel("Rezago en días de calendario")
    fig.suptitle("Referencia Ridge para EDA · α=1 fijo · sin reajuste en 2024 ni datos de test", fontsize=12)
    fig.tight_layout(rect=(0, .045, 1, .97))
    fig.text(.01, .012, CREDIT, fontsize=8, color="#526171")
    figure = "referencia_lineal_residuos_acf.png"
    fig.savefig(ASSETS / figure, dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert all(digest(path) == before[str(path.relative_to(ROOT))] for path in sources)
    summary = {"scope": "Authorized EDA block-C regularized linear residual diagnostic only; not a DL benchmark.",
        "period": {"source_start": "2023-01-01", "source_end_inclusive": "2024-12-31",
                   "training_year": 2023, "out_of_training_residual_year": 2024, "calendar_days": 731},
        "source_files": [{"path": str(path.relative_to(ROOT)), "sha256": before[str(path.relative_to(ROOT))]} for path in sources],
        "generator_script_sha256": digest(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__},
        "reference": {"family": "Ridge with free intercept", "alpha": ALPHA,
            "objective": "sum((y-intercept-Z beta)^2) + alpha sum(beta^2); no 1/N factor; intercept not penalized",
            "implementation": "SciPy Cholesky; training-only NumPy mean and population standard deviation; sklearn not installed or used",
            "input_order": "oldest day first; within each day open, high, low, close, volume",
            "feature_transform": "raw OHLCV flattened then standardized; no target transform",
            "constant_feature_scale": 1.0, "target": "sigma_7 on outcome day, unscaled daily log-return units",
            "negative_predictions": "retained and counted; no clipping or postprocessing",
            "training_schedule": "one fixed 2023 fit per D; no refit using 2024 labels",
            "selection": "all four agreed D reported; alpha fixed without tuning; no D ranking or winner selection",
            "randomness": "deterministic closed-form solve; no training random seed required",
            "fitting_policy_scope": "diagnostic only; does not fix the retraining policy, alpha or implementation of the later DL benchmark"},
        "embargo": {"omitted_outcome_days": "[t-D,t-1] inclusive", "last_training_outcome_at_first_origin": "2024-01-01-D-1 days",
                    "all_later_origins_checked": True, "first_origin_cv_hash_and_count_matched": True},
        "configurations": summaries,
        "diagnostics": {"acf_lags": list(range(1, 29)), "acf_estimator": "demeaned biased autocovariance ratio, statsmodels acf adjusted=False",
            "acf_band": "zero-centred pointwise approximate 95% white-noise reference ±1.96/sqrt(366); not simultaneous or robust to arbitrary dependence",
            "ljung_box_lags": list(LB_LAGS), "model_df": 0,
            "model_df_reason": "out-of-training residuals of a model fixed on 2023; no ARMA order estimated on these validation errors",
            "pvalue_reference": "asymptotic chi-square diagnostic; no exact finite-sample or arbitrary-nonstationarity claim",
            "multiplicity": "Holm over all 12 tests across four D and three lags; does not repair misspecification of the reference distribution",
            "stable_tail_reporting": "log chi-square tails and log-Holm retained; <1e-300 is an asymptotic-reference inequality, never an exact probability zero"},
        "meaningful_checks": ["calendar and source hashes preserved", "training indices reconstructed and matched to first-origin CV hash",
            "all training labels and covariates precede the validation origin and respect embargo",
            "scalers fitted only to the allowed 2023 samples", "Ridge predictions reproduced by independent augmented least squares",
            "raw coefficients reproduce standardized predictions", "ACF and Ljung-Box Q checked independently",
            "stable chi-square tails and Holm verified against library fixtures"],
        "reference_models_fitted": 4, "deep_learning_models_trained": 0, "pilot_runs": 0,
        "benchmark_comparisons": 0, "test_files_opened": 0, "raw_archives_opened": 0,
        "validation_predictions": 1464, "missing_daily_market_days_preserved": ["2023-03-24"],
        "source_files_unchanged": True,
        "limitations": ["The target is a rolling seven-return standard deviation; adjacent outcomes share six returns.",
            "Residual autocorrelation can reflect overlapping targets, a limited linear reference, misspecification or regime changes.",
            "Residual dependence alone does not establish that adding target lags is necessary or sufficient, nor automatically change the current type-4 input contract.",
            "ACF bands and Ljung-Box references are exploratory approximations; Holm controls multiplicity only under the relevant reference validity.",
            "These four static diagnostic models are not the three DL baselines, not a performance benchmark and not a frozen E2 protocol.",
            "2024 is development validation; no test period is opened or evaluated."], "figures": [figure]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    readme = """# Referencia lineal regularizada: diagnóstico del bloque C

Se ajustan cuatro referencias Ridge con α=1, una por D=7/14/21/28. Cada una usa
solo muestras permitidas de 2023 y omite los D días de resultado anteriores al
primer origen de validación (01/01/2024). Los índices coinciden con el manifiesto
CV vigente. Estandarización train-only (media y desviación poblacional), intercepto
libre y objetivo sin transformar. La solución SciPy Cholesky se verifica con un
sistema aumentado de mínimos cuadrados independiente; no se instala sklearn.

Los coeficientes permanecen fijos todo 2024. Se generan 366 residuos fuera del
ajuste por D, sin recortar predicciones negativas ni seleccionar D o α. Esto no
define la política de reajuste de E2. No se calcula un ranking de desempeño, no se
entrenan redes y no se abren ZIP ni datos de test.

- `reference_configurations.csv`: cortes, muestras, hashes de índices, interceptos
  y descripciones de residuos/predicciones.
- `training_outcomes.csv`: exactamente los índices y fechas usados para ajustar.
- `model_parameters.csv`: orden de 5D covariables, media/escala de 2023 y pesos.
- `validation_predictions_residuals.csv`: etiquetas, predicciones sin recorte y
  residuos de 2024, con instante de disponibilidad y soporte de entrada.
- `residual_acf.csv`: rezagos 1–28, con referencia puntual aproximada de ruido
  blanco ±1,96/√366; no son bandas simultáneas o robustas a cualquier dependencia.
- `ljung_box_holm.csv`: Q a 7/14/28 días, `model_df=0`, referencia χ² asintótica y
  Holm para los doce contrastes. Las colas se guardan también en logaritmos;
  un underflow numérico no se presenta como probabilidad cero exacta.

El objetivo móvil comparte seis retornos entre días consecutivos. La dependencia
residual puede reflejar ese solapamiento, límites de una relación lineal o cambios
de régimen; no decide por sí sola cambiar a series de tiempo ni incorporar rezagos
del objetivo. La clasificación vigente sigue el contrato OHLCV sin esos rezagos.
Los contrastes son diagnósticos exploratorios, no inferencia confirmatoria del
benchmark. Holm no corrige la calibración de la referencia asintótica.

Reproducir desde el repositorio:

```bash
/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_temporal_reference.py
```

Fuentes y hashes, software, controles, cuatro ajustes de referencia, cero redes,
cero piloto y cero archivos de test se conservan en `summary.json`. Fuente de datos:
Binance Vision/BTCUSDT; procesamiento Jassan Arteta; derivados y figura sujetos a
CC BY-NC-SA 4.0 y condiciones aplicables del proveedor.
"""
    (OUT / "README.md").write_text(readme)
    for path in OUT.iterdir():
        if path.is_file():
            shutil.copy2(path, ASSETS / path.name)
    print(json.dumps({"reference_models_fitted": 4, "deep_learning_models_trained": 0,
        "validation_predictions": 1464, "test_files_opened": 0,
        "configurations": [{key: item[key] for key in ("history_days", "training_samples", "negative_prediction_count", "residual_acf_lag1", "residual_acf_lag7")} for item in summaries]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
