"""Development-only E2 EDA extensions; no raw ZIP, test observation or DL fitting.

Reuses E1's archived timeseries-cv daily indices grouped into monthly blocks.
Ridge(alpha=1) is a diagnostic reference, not a selected E2 benchmark model.
All transformation/scaling fits use admitted past training. Repeated market
patterns are measured, never silently removed or called independent samples.
"""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e2-eda-decisions")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache-e2-eda-decisions")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["svg.hashsalt"] = "e2-eda-decisions-20261105"
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, special, stats
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e2_eda_decisions_20261105"
STATIC = ROOT / "book_e2/_static/eda_decisions"
MARKET = ROOT / "results/eda_development/daily_market_development.csv"
TARGET = ROOT / "results/eda_development/daily_volatility_development.csv"
FOLDS = ROOT / "results/dl_r2_cv_20261104/folds.json"
DESIGN = ROOT / "results/gru_lstm_tcn_planned_20261105/design.json"
FEATURES = ["open", "high", "low", "close", "volume"]
D = 7
NEAR_THRESHOLD = 0.01
ALPHA_RIDGE = 1.0
LB_LAGS = [7, 14, 28]
CREDIT = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"
INHERITED = [
    "book/_static/eda_general_quality/target_profiles.csv",
    "book/_static/eda_general_quality/quality_by_year.csv",
    "book/_static/eda_general_quality/label_reconstruction_summary.csv",
    "book/_static/eda_general_quality/window_eligibility.csv",
    "book/_static/eda_general_quality/objetivo_7_percentiles_colas.png",
    "book/_static/eda_general_quality/mascaras_calendario_desarrollo.png",
    "book/_static/eda_general_shift/marginal_distribution_differences.csv",
    "book/_static/eda_general_shift/conditional_block_permutation.csv",
    "book/_static/eda_general_shift/marginales_causales_2023_2024.png",
    "book/_static/eda_temporal_specific/lag_peak_summary.csv",
    "book/_static/eda_temporal_specific/lag_correlations_common_train.csv",
    "book/_static/eda_temporal_specific/rolling_relationship_summary.csv",
    "book/_static/eda_temporal_specific/availability_contract.csv",
    "book/_static/eda_temporal_specific/correlaciones_rezagos_cohorte_comun.png",
    "book/_static/eda_doc/relacion_movil_covariables_objetivo.png",
    "book/_static/eda_doc/rolling_60d_correlations_development.csv",
    "book/_static/eda_doc/rolling_60d_pair_counts.csv",
]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def csv(frame, name):
    for folder in (OUT, STATIC):
        frame.to_csv(folder / name, index=False, lineterminator="\n")


def write_json(value, name):
    payload = dumps(value)
    for folder in (OUT, STATIC):
        (folder / name).write_text(payload)


def figure(fig, name):
    fig.tight_layout(rect=(0, .045, 1, 1))
    fig.text(.012, .01, CREDIT, fontsize=7, color="#425568")
    for extension in ("png", "svg"):
        target = STATIC / f"{name}.{extension}"
        fig.savefig(target, dpi=150, bbox_inches="tight", metadata={"Date": None} if extension == "svg" else None)
        shutil.copyfile(target, OUT / target.name)
    plt.close(fig)


def read_daily(path):
    frame = pd.read_csv(path, parse_dates=["market_day_utc"]).set_index("market_day_utc")
    expected = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
    assert frame.index.equals(expected), "Only unchanged, regular 2023--2024 UTC development is admitted"
    return frame


def windows_and_relative(market, target):
    raw = market[FEATURES].to_numpy(dtype=float)
    windows = np.full((len(raw), D, 5), np.nan)
    for outcome in range(D, len(raw)):
        windows[outcome] = raw[outcome-D:outcome]
    good = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(target)
    relative = np.full_like(windows, np.nan)
    admitted = windows[good]
    assert (admitted[..., :4] > 0).all() and (admitted[..., 4] >= 0).all()
    lp = np.log(admitted[..., :4])
    relative[good, :, :4] = lp - lp[:, -1:, 3:4]
    relative[good, :, 4] = np.log1p(admitted[..., 4])
    return windows, relative, good


def objective_transformations(labels, calendar, initial_train):
    fit = labels[initial_train]
    assert len(fit) == 343 and (fit > 0).all()
    lam_box = float(stats.boxcox_normmax(fit, method="mle"))
    lam_yj = float(stats.yeojohnson_normmax(fit))
    rows = []
    transformed = {}
    for name, lam in [("identity", None), ("log", 0.0), ("box_cox", lam_box), ("yeo_johnson", lam_yj)]:
        all_values = np.full(len(labels), np.nan)
        valid = np.isfinite(labels)
        values = labels[valid]
        if name == "identity":
            result, inverse = values.copy(), values.copy()
        elif name == "log":
            result = np.log(values)
            inverse = np.exp(result)
        elif name == "box_cox":
            result = special.boxcox(values, lam)
            inverse = special.inv_boxcox(result, lam)
        else:
            result = stats.yeojohnson(values, lmbda=lam)
            # All targets are positive, so the positive Yeo--Johnson branch applies.
            inverse = np.expm1(np.log1p(lam * result) / lam) if abs(lam) > 1e-12 else np.expm1(result)
        assert np.isfinite(result).all()
        inverse_max = float(np.max(np.abs(inverse-values)))
        assert inverse_max < 1e-12
        all_values[valid] = result
        transformed[name] = all_values
        for year in (2023, 2024):
            part = all_values[(calendar.year == year) & np.isfinite(all_values)]
            rows.append({"transformation": name, "lambda": lam, "fit_n": len(fit),
                         "fit_first_outcome_utc": str(calendar[initial_train[0]].date()),
                         "fit_last_outcome_utc": str(calendar[initial_train[-1]].date()),
                         "profile_year": year, "profile_n": len(part),
                         "mean": float(np.mean(part)), "std_ddof0": float(np.std(part)),
                         "skew_unbiased": float(stats.skew(part, bias=False)),
                         "excess_kurtosis_unbiased": float(stats.kurtosis(part, fisher=True, bias=False)),
                         "inverse_max_absolute_error": inverse_max,
                         "applied_to_benchmark": False})
    frame = pd.DataFrame(rows)
    csv(frame, "objective_transformations.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = {2023: "#195f94", 2024: "#c96525"}
    for ax, (name, values) in zip(axes.ravel(), transformed.items()):
        selected = frame[frame.transformation == name].iloc[0]
        fit_values = values[initial_train]
        center, scale = float(fit_values.mean()), float(fit_values.std())
        all_z = (values[np.isfinite(values)]-center)/scale
        edges = np.linspace(float(all_z.min()), float(all_z.max()), 29)
        for year in (2023, 2024):
            z = (values[(calendar.year == year) & np.isfinite(values)]-center)/scale
            ax.hist(z, bins=edges, density=True, histtype="step", linewidth=1.6,
                    color=colors[year], label=str(year))
        suffix = "" if name in ["identity", "log"] else f" (λ={selected['lambda']:.3f})"
        ax.set_title(name.replace("_", " ") + suffix)
        ax.set_xlabel("Objetivo transformado, escala de entrenamiento admitido")
        ax.set_ylabel("Densidad")
        ax.legend()
    fig.suptitle("Transformaciones diagnósticas: parámetros ajustados hasta 24/12/2023")
    figure(fig, "objective_transformations")
    return {"fit_n": len(fit), "fit_last_outcome_utc": str(calendar[initial_train[-1]].date()),
            "lambda_box_cox": lam_box, "lambda_yeo_johnson": lam_yj,
            "decision": "No transformation adopted; original MSE/RMSE and inherited target retained",
            "moment_profiles": rows}


def nearest_windows(relative, windows, eligible, calendar, initial_train):
    pooled = relative[initial_train].reshape(-1, 5)
    mean, sd = pooled.mean(axis=0), pooled.std(axis=0)
    scale = np.where(sd == 0, 1.0, sd)
    z = ((relative-mean)/scale).reshape(len(relative), -1)
    a = z[initial_train]
    valid_2024 = np.flatnonzero(eligible & (calendar.year == 2024))
    b = z[valid_2024]
    within = cdist(a, a) / np.sqrt(D*5)
    np.fill_diagonal(within, np.inf)
    cross = cdist(b, a) / np.sqrt(D*5)
    within_validation = cdist(b, b) / np.sqrt(D*5)
    np.fill_diagonal(within_validation, np.inf)
    raw_a = windows[initial_train].reshape(len(initial_train), -1)
    raw_b = windows[valid_2024].reshape(len(valid_2024), -1)
    pairs = []
    summaries = []
    sensitivities = []
    for scope, distances, queries, reference_indices, exact_reference, exact_queries in [
        ("within_initial_training", within, initial_train, initial_train, raw_a, raw_a),
        ("validation_2024_to_initial_training", cross, valid_2024, initial_train, raw_a, raw_b),
        ("within_validation_2024", within_validation, valid_2024, valid_2024, raw_b, raw_b),
    ]:
        nearest = np.argmin(distances, axis=1)
        minima = distances[np.arange(len(queries)), nearest]
        assert np.isfinite(minima).all()
        raw_matches = (exact_queries[:, None, :] == exact_reference[None, :, :]).all(axis=2)
        if scope.startswith("within_"):
            np.fill_diagonal(raw_matches, False)
        relative_matches = distances <= 1e-14
        qs = np.quantile(minima, [.01, .05, .5, .95, .99])
        summaries.append({"scope": scope, "query_windows": len(queries),
                          "reference_windows": len(reference_indices), "dimension": D*5,
                          "scaler_fit_last_outcome_utc": str(calendar[initial_train[-1]].date()),
                          "threshold_rms_standardized_distance": NEAR_THRESHOLD,
                          "near_query_count": int((minima <= NEAR_THRESHOLD).sum()),
                          "exact_raw_duplicate_query_count": int(raw_matches.any(axis=1).sum()),
                          "exact_relative_duplicate_query_count": int(relative_matches.any(axis=1).sum()),
                          "relative_zero_distance_numeric_tolerance": 1e-14,
                          "minimum_rms_distance": float(minima.min()),
                          "q01_rms_distance": float(qs[0]), "q05_rms_distance": float(qs[1]),
                          "median_rms_distance": float(qs[2]), "q95_rms_distance": float(qs[3]),
                          "q99_rms_distance": float(qs[4]),
                          "near_definition": "Nearest other window RMS Euclidean distance after five-channel training scaling <=0.01",
                          "interpretation": "Pattern-similarity diagnostic; not proof of copy or independence"})
        for threshold in (.01, .05, .1, .25, .5):
            sensitivities.append({"scope": scope, "threshold_rms_standardized_distance": threshold,
                                  "query_count": len(queries), "near_query_count": int((minima <= threshold).sum()),
                                  "interpretation": "Sensitivity only; no threshold selected for data deletion"})
        for i, query in enumerate(queries):
            reference = int(reference_indices[nearest[i]])
            pairs.append({"scope": scope, "query_outcome_day_utc": str(calendar[query].date()),
                          "nearest_reference_outcome_day_utc": str(calendar[reference].date()),
                          "rms_standardized_distance": float(minima[i]),
                          "below_fixed_threshold": bool(minima[i] <= NEAR_THRESHOLD),
                          "exact_raw_duplicate": bool(raw_matches[i].any()),
                          "calendar_day_separation": int(abs(query-reference)),
                          "input_calendar_days_shared_with_nearest": len(set(range(query-D, query)) & set(range(reference-D, reference)))})
    csv(pd.DataFrame(summaries), "nearest_window_summary.csv")
    csv(pd.DataFrame(pairs), "nearest_window_distances.csv")
    csv(pd.DataFrame(sensitivities), "nearest_window_sensitivity.csv")
    write_json({"fit_windows": len(initial_train), "fit_window_positions": len(initial_train)*D,
                "x_mean": mean.tolist(), "x_scale": scale.tolist(),
                "fit_outcomes_start": str(calendar[initial_train[0]].date()),
                "fit_outcomes_end": str(calendar[initial_train[-1]].date()),
                "aggregation": "Five channel means/scales over all N*D admitted training-window positions",
                "distance": "Euclidean(flattened standardized relative windows)/sqrt(35)",
                "threshold": NEAR_THRESHOLD, "threshold_tuned": False}, "nearest_window_scaler.json")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for scope, group in pd.DataFrame(pairs).groupby("scope", sort=False):
        values = np.sort(group.rms_standardized_distance.to_numpy())
        display = {"within_initial_training": "Entre ventanas train",
                   "validation_2024_to_initial_training": "2024 → train inicial",
                   "within_validation_2024": "Entre ventanas 2024"}
        ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=display[scope])
    ax.axvline(NEAR_THRESHOLD, linestyle="--", color="#9b3131", label="Umbral fijo 0,01")
    ax.set(xlabel="Distancia RMS al vecino más próximo (35 valores estandarizados)", ylabel="Fracción acumulada",
           title="Patrones de ventanas D7: comparación diagnóstica, sin eliminar ejemplos")
    ax.legend()
    figure(fig, "nearest_window_distances")
    return {"threshold": NEAR_THRESHOLD, "threshold_tuned": False, "scopes": summaries,
            "decision": "No window removed; similarity is not copy, causal-history reuse is not independent replication"}


def index_hash(indices):
    return sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def support(indices, left, include_outcome):
    return set((np.asarray(indices)[:, None] - np.arange(left, -1 if include_outcome else 0, -1)).ravel().tolist())


def audit_folds(folds, eligible, calendar):
    rows, indices_seen = [], set()
    for fold in folds:
        origin = int(fold["fit_origin_index"])
        train = np.asarray(fold["train_indices"], dtype=int)
        val = np.asarray(fold["test_indices"], dtype=int)
        inner_train = np.asarray(fold["inner_train_indices"], dtype=int)
        inner_val = np.asarray(fold["inner_validation_indices"], dtype=int)
        assert fold["fold_id"] <= "2024-10"
        assert np.array_equal(train, np.flatnonzero(eligible & (np.arange(len(calendar)) <= origin-D-1)))
        assert eligible[train].all() and eligible[val].all()
        assert train.max() <= origin-D-1 and val.min() == origin
        assert index_hash(train) == fold["train"]["indices_sha256"]
        assert index_hash(val) == fold["test"]["indices_sha256"]
        assert not set(train) & set(val)
        assert not indices_seen & set(val)
        prior_val_now_train = len(indices_seen & set(train))
        indices_seen.update(val)
        train_support = support(train, D, True)
        val_support = support(val, D, True)
        inner_train_support = support(inner_train, D, True)
        inner_val_support = support(inner_val, D, True)
        assert not train_support & val_support
        assert not inner_train_support & inner_val_support
        inputs = np.asarray(fold["test_input_indices"])
        expected_inputs = val[:, None] - np.arange(D, 0, -1)
        assert np.array_equal(inputs, expected_inputs)
        assert (inputs < val[:, None]).all()
        assert fold["availability_checks"]["all_train_labels_available_before_fit"]
        rows.append({"fold_id": fold["fold_id"], "train_n": len(train), "validation_n": len(val),
                     "first_validation_day_utc": str(calendar[val[0]].date()),
                     "last_train_outcome_day_utc": str(calendar[train[-1]].date()),
                     "calendar_days_between_last_train_and_first_validation": int(val[0]-train[-1]),
                     "excluded_outcome_gap_days": int(val[0]-train[-1]-1),
                     "same_fold_train_validation_outcome_intersection": 0,
                     "same_fold_raw_close_support_intersection": len(train_support & val_support),
                     "inner_raw_close_support_intersection": len(inner_train_support & inner_val_support),
                     "validation_input_availability_violations": int((inputs >= val[:, None]).sum()),
                     "prior_validation_dates_now_legal_training": prior_val_now_train,
                     "train_indices_sha256": index_hash(train),
                     "validation_indices_sha256": index_hash(val),
                     "status": "passed; reuse of older validation in later folds is causal but dependent"})
    assert len(indices_seen) == 305
    csv(pd.DataFrame(rows), "fold_leakage_audit.csv")
    return {"monthly_folds": len(folds), "unique_validation_outcomes": len(indices_seen),
            "same_fold_support_intersection_total": sum(row["same_fold_raw_close_support_intersection"] for row in rows),
            "availability_violations": sum(row["validation_input_availability_violations"] for row in rows),
            "gap_days": D, "daily_horizon": 1,
            "library": "Archived timeseries-cv 0.1.5 split_train_val_forwardChaining indices; monthly grouping is project adaptation",
            "raw_support": "Union of completed closes [u-7,u] needed for each label plus preceding X days",
            "limitations": "Calendar checks alone do not certify semantic/provider leakage; folds share past training and rolling labels are dependent",
            "folds": rows}


def acf_values(values, maxlag):
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    centered = values-values.mean()
    denominator = float(centered @ centered)
    assert denominator > 0
    return np.asarray([1.0]+[float(centered[k:] @ centered[:-k]/denominator) for k in range(1, maxlag+1)])


def chi_square_logsf(q, degrees):
    """Stable integer-df chi-square tail, including values below float probability range."""
    if q == 0:
        return 0.0
    x = q/2.0
    if degrees % 2 == 0:
        k = np.arange(degrees//2, dtype=float)
        return float(-x + special.logsumexp(k*np.log(x)-special.gammaln(k+1)))
    result = float(np.log(2.0)+special.log_ndtr(-np.sqrt(q)))
    a = .5
    for _ in range((degrees-1)//2):
        result = float(np.logaddexp(result, a*np.log(x)-x-special.gammaln(a+1)))
        a += 1
    return min(0.0, result)


def log_holm(logp):
    logp = np.asarray(logp)
    order = np.argsort(logp, kind="stable")
    corrected = np.minimum(0.0, np.maximum.accumulate(np.log(len(logp)-np.arange(len(logp)))+logp[order]))
    result = np.empty_like(corrected)
    result[order] = corrected
    return result


def ridge_fit_predict(x_train, y_train, x_val):
    # Same relative-window channel scaling as the inherited preprocessing;
    # the diagnostic Ridge remains unconstrained and has an unpenalized intercept.
    pooled = x_train.reshape(-1, 5)
    mean, sd = pooled.mean(axis=0), pooled.std(axis=0)
    scale = np.where(sd == 0, 1.0, sd)
    a = ((x_train-mean)/scale).reshape(len(x_train), D*5)
    b = ((x_val-mean)/scale).reshape(len(x_val), D*5)
    center = a.mean(axis=0)
    y_scale = float(y_train.mean())
    y = y_train/y_scale
    y_mean = float(y.mean())
    ac = a-center
    matrix = ac.T@ac+ALPHA_RIDGE*np.eye(a.shape[1])
    rhs = ac.T@(y-y_mean)
    coefficient = linalg.cho_solve(linalg.cho_factor(matrix), rhs)
    prediction = ((b-center)@coefficient+y_mean)*y_scale
    # Independent augmented least-squares solution checks Ridge and intercept.
    augmented_x = np.vstack([np.column_stack([np.ones(len(a)), a]),
                             np.column_stack([np.zeros(a.shape[1]), np.sqrt(ALPHA_RIDGE)*np.eye(a.shape[1])])])
    augmented_y = np.r_[y, np.zeros(a.shape[1])]
    alt = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    alt_prediction = np.column_stack([np.ones(len(b)), b])@alt*y_scale
    assert np.allclose(prediction, alt_prediction, rtol=1e-9, atol=1e-11)
    return prediction, {"x_mean": mean.tolist(), "x_scale": scale.tolist(), "y_scale": y_scale,
                        "coefficient": coefficient.tolist(), "center": center.tolist(),
                        "intercept_scaled_centered_x": y_mean,
                        "independent_lstsq_max_abs_error": float(np.max(np.abs(alt_prediction-prediction)))}


def forward_reference(relative, labels, calendar, folds):
    persistence = np.r_[np.nan, labels[:-1]]
    rows, models = [], []
    for fold in folds:
        train = np.asarray(fold["train_indices"])
        val = np.asarray(fold["test_indices"])
        pred, metadata = ridge_fit_predict(relative[train], labels[train], relative[val])
        assert np.isfinite(persistence[val]).all()
        metadata.update({"fold_id": fold["fold_id"], "alpha": ALPHA_RIDGE,
                         "train_n": len(train), "validation_n": len(val),
                         "train_indices_sha256": index_hash(train),
                         "last_train_outcome_utc": str(calendar[train[-1]].date()),
                         "selection": "Fixed diagnostic; no HPO, no target lags and no benchmark selection"})
        models.append(metadata)
        for pos, index in enumerate(val):
            rows.append({"fold_id": fold["fold_id"], "outcome_day_utc": str(calendar[index].date()),
                         "y_true": float(labels[index]), "ridge_prediction": float(pred[pos]),
                         "persistence_prediction": float(persistence[index]),
                         "ridge_residual": float(labels[index]-pred[pos]),
                         "persistence_residual": float(labels[index]-persistence[index])})
    frame = pd.DataFrame(rows)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.outcome_day_utc, utc=True))
    assert dates.is_unique and (np.diff(dates.asi8) == pd.Timedelta(days=1).value).all()
    assert str(dates[0].date()) == "2024-01-01" and str(dates[-1].date()) == "2024-10-31"
    csv(frame, "forward_reference_residuals.csv")
    write_json({"models": models, "scope": "Diagnostic Ridge alpha1, monthly refits on archived forward indices"},
               "forward_reference_configurations.json")
    metric_rows, acf_rows, lb_rows = [], [], []
    groups = [(name, group) for name, group in frame.groupby("fold_id", sort=True)] + [("pooled_Jan_Oct2024", frame)]
    for scope, group in groups:
        for model in ("ridge", "persistence"):
            residual = group[f"{model}_residual"].to_numpy()
            n = len(residual)
            maxlag = min(28, n-1)
            acf = acf_values(residual, maxlag)
            metric_rows.append({"scope": scope, "model": model, "n": n,
                                "rmse": float(np.sqrt(np.mean(residual**2))),
                                "mae": float(np.mean(np.abs(residual))),
                                "residual_mean": float(residual.mean()),
                                "acf_lag1": float(acf[1]), "acf_lag7": float(acf[7]),
                                "acf_lag14": float(acf[14]),
                                "negative_prediction_count": int((group[f"{model}_prediction"] < 0).sum()),
                                "scope_note": "Out-of-fold development diagnostic; fixed-alpha reference, not E2 benchmark"})
            for lag, value in enumerate(acf):
                acf_rows.append({"scope": scope, "model": model, "n": n, "lag_days": lag, "acf_biased": float(value)})
            for lag in LB_LAGS:
                if lag >= n:
                    continue
                q = float(n*(n+2)*np.sum(acf[1:lag+1]**2/(n-np.arange(1, lag+1))))
                logp = chi_square_logsf(q, lag)
                scipy_logp = float(stats.chi2.logsf(q, lag))
                if np.isfinite(scipy_logp):
                    assert np.isclose(logp, scipy_logp, rtol=1e-10, atol=1e-10)
                lb_rows.append({"scope": scope, "model": model, "n": n, "lag_days": lag,
                                "ljung_box_Q": q, "reference_degrees_of_freedom": lag,
                                "estimated_arma_parameters_subtracted": 0,
                                "log_p_value": logp, "log10_p_value": logp/np.log(10.0),
                                "asymptotic_reference_warning": ("Short monthly sample relative to lag" if n < 5*lag else "Pooled residuals include changes of monthly fit"),
                                "inference_scope": "Exploratory diagnostic; overlap, nonstationarity and small folds limit chi-square calibration"})
    lb = pd.DataFrame(lb_rows)
    lb["log_p_holm"] = log_holm(lb.log_p_value.to_numpy())
    lb["log10_p_holm"] = lb.log_p_holm/np.log(10.0)
    lb["p_value_display"] = [f"{np.exp(v):.9g}" if v > -700 else f"10^({v/np.log(10):.6f})" for v in lb.log_p_value]
    lb["p_holm_display"] = [f"{np.exp(v):.9g}" if v > -700 else f"10^({v/np.log(10):.6f})" for v in lb.log_p_holm]
    lb["holm_family_size"] = len(lb)
    metrics = pd.DataFrame(metric_rows)
    csv(metrics, "forward_reference_metrics.csv")
    csv(pd.DataFrame(acf_rows), "forward_reference_acf.csv")
    csv(lb, "forward_reference_ljung_box.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(dates, frame.ridge_residual, linewidth=.9, label="Ridge D7 mensual (forward chaining)")
    axes[0].plot(dates, frame.persistence_residual, linewidth=.8, alpha=.8, label="Persistencia causal")
    axes[0].axhline(0, color="#555555", linewidth=.7)
    axes[0].set(ylabel="Residuo en volatilidad diaria", title="Residuos fuera de muestra en desarrollo: enero–octubre de 2024")
    axes[0].legend()
    for model in ("ridge", "persistence"):
        part = pd.DataFrame(acf_rows)
        part = part[(part.scope == "pooled_Jan_Oct2024") & (part.model == model)]
        axes[1].plot(part.lag_days, part.acf_biased, marker="o", markersize=3, label=model)
    axes[1].axhline(0, color="#555555", linewidth=.7)
    axes[1].set(xlabel="Rezago de calendario (días)", ylabel="ACF descriptiva",
                title="ACF con calendario intacto; no se presume independencia de folds")
    axes[1].legend()
    figure(fig, "forward_residual_diagnostics")
    pooled = metrics[metrics.scope == "pooled_Jan_Oct2024"].to_dict("records")
    return {"model": "Ridge D7 alpha1; diagnostic only", "monthly_fits": len(folds),
            "deep_learning_fits": 0, "validation_outcomes": len(frame),
            "pooled": pooled, "ljung_box_holm_family_size": len(lb),
            "acf_definition": "Centered biased sample autocorrelation on intact daily calendar",
            "ljung_box_reference": "Chi-square with degrees equal to lag; no fitted ARMA parameters. Approximate exploratory diagnostic, not guaranteed calibrated with overlapping labels or refit changes.",
            "decision": "Assess residual temporal structure; no target-lag addition or problem-route change automatically adopted"}


def main():
    for directory in (OUT, STATIC):
        directory.mkdir(parents=True, exist_ok=True)
    inputs = [MARKET, TARGET, FOLDS, DESIGN] + [ROOT/p for p in INHERITED]
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in inputs}
    market = read_daily(MARKET)
    target = read_daily(TARGET)
    assert market.index.equals(target.index)
    calendar = market.index
    labels = target.sigma_7.to_numpy(dtype=float)
    windows, relative, eligible = windows_and_relative(market, labels)
    definition = json.loads(DESIGN.read_text())
    assert definition["history_days"] == D and definition["target_returns_days"] == 7
    assert definition["forecast_days"] == 1 and not definition["data_contract"]["target_lags_added"]
    archived_folds = json.loads(FOLDS.read_text())
    folds = [f for f in archived_folds["folds"] if f["fold_id"] <= "2024-10"]
    assert len(folds) == 10
    initial_train = np.asarray(folds[0]["train_indices"], dtype=int)
    assert len(initial_train) == 343 and str(calendar[initial_train[-1]].date()) == "2023-12-24"
    # Reconstruct labels directly without pandas rolling/std, keeping every calendar gap.
    recalculated = np.full(len(labels), np.nan)
    close = market.close.to_numpy()
    for t in range(7, len(close)):
        support_close = close[t-7:t+1]
        if np.isfinite(support_close).all():
            returns = np.log(support_close[1:])-np.log(support_close[:-1])
            recalculated[t] = np.sqrt(np.mean((returns-returns.mean())**2))
    assert np.array_equal(np.isfinite(labels), np.isfinite(recalculated))
    max_label_error = float(np.nanmax(np.abs(labels-recalculated)))
    assert max_label_error < 2e-14
    summary = {"scope": "E2 EDA extension using unchanged 2023--2024 development and inherited aggregate artifacts only",
               "data_period": {"start": "2023-01-01", "end_inclusive": "2024-12-31", "calendar_days": len(calendar)},
               "test_observations_read": False, "raw_archives_opened": False,
               "new_deep_learning_fits": 0, "target_lags_added": False,
               "primary_metric_or_margin_changed": False,
               "label_reconstruction_max_absolute_error": max_label_error,
               "label_reconstruction_interpretation": "Numerical consistency only, not economic label truth/noise or irreducible error",
               "objective_transformations": objective_transformations(labels, calendar, initial_train),
               "nearest_windows": nearest_windows(relative, windows, eligible, calendar, initial_train),
               "fold_audit": audit_folds(folds, eligible, calendar),
               "forward_reference": forward_reference(relative, labels, calendar, folds),
               "confirmatory_status": "Exploratory development diagnostics; no new preregistered hypothesis test or E2 benchmark result"}
    for source in INHERITED:
        path = ROOT/source
        category = "eda_covariates_documental" if path.parent.name == "eda_doc" else path.parent.name
        dest = STATIC/"heredado"/category/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        assert digest(path) == digest(dest)
    assert all(digest(ROOT/p) == value for p, value in before.items())
    write_json(summary, "summary.json")
    readme = """# Ampliación EDA E2 orientada a decisiones

Ejecución desde la raíz: `python scripts/eda_e2_decisions.py`.
Requiere Python y NumPy, pandas, SciPy y Matplotlib; las versiones observadas se
registran en provenance.json. Los dos CSV de desarrollo 2023–2024 y los índices
archivados de E1 se conservan íntegros. No descarga datos, abre ZIPs ni lee
observaciones del test. Los archivos heredados se copian byte por byte desde E1.
Para restaurar las dependencias mínimas observadas en un entorno aislado:
`python -m pip install -r results/e2_eda_decisions_20261105/requirements.txt`.

- Transformaciones: lambda ajustada solo con los 343 outcomes iniciales permitidos
  hasta el 24/12/2023. Perfiles de todos los datos válidos de cada año son
  descriptivos; no se adopta una transformación del objetivo o pérdida nueva.
- Ventanas: D7, log OHLC relativo al último cierre observable y log1p volumen;
  escalas por cinco canales ajustadas en los 343 train iniciales (N×D posiciones).
  Distancia RMS euclídea en 35 valores, umbral fijo 0,01. Similitud no prueba
  copia ni autoriza borrar patrones; controles por fecha/composición son aparte.
  El cero numérico de distancia relativa usa tolerancia 1e-14; las coincidencias
  de ventanas OHLCV brutas se comprueban por igualdad de todos sus valores.
  También se publica sensibilidad a umbrales 0,01/0,05/0,10/0,25/0,50 sin
  seleccionar un corte para eliminar ejemplos. Se separan train/train,
  validación/train y validación/validación con el mismo escalado de train.
- Folds: los diez bloques mensuales Jan–Oct2024 proceden de índices diarios de
  timeseries-cv agrupados en E1. El embargo es u<=origen−8. Fechas ya validadas
  pueden entrar al train de meses posteriores, pero los folds no son réplicas
  independientes. El CSV histórico llama test_indices a validación de desarrollo.
- Ridge alpha1: una referencia diagnóstica ajustada por mes sin HPO, con escalas
  solo del train admitido e intercepto no penalizado. Se contrasta numéricamente
  con otra solución por mínimos cuadrados aumentados. No es uno de los diez
  modelos E2 seleccionados y no altera las recetas GRU/LSTM/TCN heredadas.
- ACF: centrada y normalizada por suma de cuadrados, con calendario íntegro.
  Ljung–Box: referencia chi-cuadrado df=rezago (sin parámetros ARMA estimados),
  con Holm sobre los 66 diagnósticos publicados. P-valores aproximados y
  exploratorios: muestras mensuales cortas, solapamiento y cambios de ajuste
  limitan calibración. Se conservan log-p para evitar ceros por underflow.

Los CSV de residuos de desarrollo son derivados, no cotizaciones brutas.
Datos: Binance Vision / BTCUSDT. Procesamiento: Jassan Arteta.
Datos y derivados sujetos a CC BY-NC-SA 4.0 y términos de la fuente conservados
en E1; no se atribuye esta licencia automáticamente al código del repositorio.
"""
    for directory in (OUT, STATIC):
        (directory/"README.md").write_text(readme)
        (directory/"requirements.txt").write_text(
            f"numpy=={np.__version__}\npandas=={pd.__version__}\nscipy=={scipy.__version__}\nmatplotlib=={matplotlib.__version__}\n")
    output_files = {}
    for path in sorted(STATIC.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            output_files[path.relative_to(STATIC).as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    write_json({"schema_version": 1, "administrative_date_colombia": "2026-11-05", "calendar_timezone": "UTC",
                "generator_script": "scripts/eda_e2_decisions.py", "generator_sha256": digest(__file__),
                "e1_source_commit": "94a4b0324e93361e8928d708fdfeb50110e3686f",
                "input_sha256": before, "input_files_unchanged_after_execution": True,
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                             "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
                "random_sampling": False, "split_seed": "Not applicable: archived deterministic chronological indices",
                "data_access_scope": "Development 2023--2024 plus public metadata/artifacts only; no raw ZIP or test observation",
                "new_deep_learning_fits": 0, "diagnostic_ridge_fits": 10,
                "outputs_under_book_e2_static": output_files}, "provenance.json")
    print(dumps({"transform_lambdas": {k:summary["objective_transformations"][k] for k in ["lambda_box_cox","lambda_yeo_johnson"]},
                 "near_windows": summary["nearest_windows"]["scopes"], "fold_audit": {k:v for k,v in summary["fold_audit"].items() if k != "folds"},
                 "forward_diagnostic_pooled": summary["forward_reference"]["pooled"]}))


if __name__ == "__main__":
    main()
