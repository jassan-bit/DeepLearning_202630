"""Descriptive block-C EDA on existing development data; no fitted model.

Calendar-aware covariate availability, common-cohort lag correlations and
summaries of previously computed rolling relationships. No raw ZIP or test
file is read, no values are imputed and no predictor window is selected.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e1-temporal-specific")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "results/eda_development/daily_market_development.csv"
TARGET = ROOT / "results/eda_development/daily_volatility_development.csv"
LEGACY = ROOT / "results/eda_covariates_documental"
ROLLING = LEGACY / "rolling_60d_correlations_development.csv"
COUNTS = LEGACY / "rolling_60d_pair_counts.csv"
MARGINAL = LEGACY / "lagged_correlations_train.csv"
OUT = ROOT / "results/eda_temporal_specific"
STATIC = ROOT / "book/_static/eda_temporal_specific"
FEATURES = ("open", "high", "low", "close", "volume")
DISPLAY = ("log(O)", "log(H)", "log(L)", "log(C)", "log(1+V)")
CALENDAR = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def daily(path, date_column):
    frame = pd.read_csv(path, parse_dates=[date_column]).set_index(date_column)
    if not frame.index.equals(CALENDAR):
        raise ValueError(f"Unexpected development calendar: {path.name}")
    return frame


def csv(frame, name, **kwargs):
    frame.to_csv(OUT / name, index=False, lineterminator="\n", **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    inputs = (MARKET, TARGET, ROLLING, COUNTS, MARGINAL)
    original_hashes = {path: digest(path) for path in inputs}
    market = daily(MARKET, "market_day_utc")
    target = daily(TARGET, "market_day_utc")["sigma_7"]
    rolling = daily(ROLLING, "outcome_day_utc")
    counts = daily(COUNTS, "outcome_day_utc")
    marginal = pd.read_csv(MARGINAL)
    missing_day = pd.Timestamp("2023-03-24", tz="UTC")
    assert market.loc[missing_day, list(FEATURES)].isna().all()
    assert len(CALENDAR) == 731
    assert (market.loc[:, list(FEATURES[:4])].dropna() > 0).all().all()
    assert (market["volume"].dropna() >= 0).all()

    transformed = np.log(market.loc[:, list(FEATURES[:4])])
    transformed["volume"] = np.log1p(market["volume"])
    lagged = {lag: transformed.shift(lag) for lag in range(1, 29)}
    all_history = np.logical_and.reduce([
        lagged[lag].notna().all(axis=1).to_numpy() for lag in range(1, 29)
    ])
    valid_common = all_history & target.notna().to_numpy()
    common_train = valid_common & (CALENDAR.year == 2023)
    assert common_train.sum() == 308
    assert valid_common[CALENDAR.year == 2024].sum() == 366

    # Independent calendar membership check; it does not reuse shifted arrays.
    manual_common = []
    for outcome in CALENDAR:
        source_days = [outcome - pd.Timedelta(days=lag) for lag in range(1, 29)]
        has_history = all(day in market.index for day in source_days)
        complete = (has_history and market.loc[source_days, list(FEATURES)].notna().all().all())
        manual_common.append(bool(complete and pd.notna(target.loc[outcome])))
    assert np.array_equal(np.asarray(manual_common), valid_common)
    csv(pd.DataFrame({"outcome_day_utc": CALENDAR, "year": CALENDAR.year,
                      "full_28lag_covariates_and_target": valid_common,
                      "in_common_2023_cohort": common_train}), "common_cohort_calendar.csv")

    structures = []
    for feature, display in zip(FEATURES, DISPLAY):
        structures.append({
            "feature": feature, "diagnostic_transform": display,
            "data_type": "continuous numeric", "units": "BTC" if feature == "volume" else "USDT",
            "frequency": "one completed daily UTC bar", "calendar_days": 731,
            "observed_days": int(market[feature].notna().sum()),
            "missing_days": int(market[feature].isna().sum()),
            "completed_bar_availability": "start UTC of source day + 1 day",
            "admitted_input_lags_calendar_days": "1 through 28",
            "same_outcome_day_completed_bar_admitted": False,
            "predictor": True,
        })
    csv(pd.DataFrame(structures), "covariate_structure.csv")

    availability = []
    for lag in range(1, 29):
        for year in (2023, 2024):
            outcomes = CALENDAR[CALENDAR.year == year]
            source_days = outcomes - pd.Timedelta(days=lag)
            inside = source_days.isin(CALENDAR)
            complete = market.reindex(source_days).loc[:, list(FEATURES)].notna().all(axis=1).to_numpy()
            # A completed bar of d-k is available from d-k+1, at or before d.
            availability_times = source_days + pd.Timedelta(days=1)
            assert (availability_times <= outcomes).all()
            availability.append({
                "year": year, "lag_calendar_days": lag, "outcome_calendar_days": len(outcomes),
                "covariate_source_inside_development_days": int(inside.sum()),
                "outside_acquired_development_history_days": int((~inside).sum()),
                "inside_but_missing_covariate_days": int((inside & ~complete).sum()),
                "complete_covariate_days": int(complete.sum()),
                "availability_violations": 0,
                "scope": "covariate lag only; target/full-window eligibility is a separate mask",
            })
    csv(pd.DataFrame(availability), "availability_by_lag.csv")
    csv(pd.DataFrame([
        {"quantity": "completed OHLCV of d-k, k>=1", "available_at": "start UTC of d-k+1",
         "admissible_at_start_of_d": True, "contract_role": "covariate"},
        {"quantity": "completed OHLCV of d", "available_at": "start UTC of d+1",
         "admissible_at_start_of_d": False, "contract_role": "not a current input"},
        {"quantity": "sigma_7(d) including return d", "available_at": "start UTC of d+1",
         "admissible_at_start_of_d": False, "contract_role": "target only"},
        {"quantity": "rolling correlation labelled d", "available_at": "retrospective after outcome d",
         "admissible_at_start_of_d": False, "contract_role": "diagnostic only; conservative availability"},
    ]), "availability_contract.csv")

    checks = []
    for day, lag, feature in [("2023-01-29", 28, "close"), ("2023-03-25", 1, "close"),
                              ("2024-01-01", 1, "open"), ("2024-12-31", 28, "volume")]:
        outcome = pd.Timestamp(day, tz="UTC")
        source = outcome - pd.Timedelta(days=lag)
        raw = float(market.loc[source, feature])
        expected = (math.log1p(raw) if feature == "volume" else math.log(raw)) if math.isfinite(raw) else np.nan
        actual = float(lagged[lag].loc[outcome, feature])
        assert ((math.isnan(expected) and math.isnan(actual)) or abs(actual-expected) <= 1e-12)
        checks.append({"outcome_day_utc": day, "feature": feature, "lag_calendar_days": lag,
                       "source_day_utc": source.strftime("%Y-%m-%d"),
                       "available_at_utc": (source + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
                       "source_value_missing": math.isnan(raw), "alignment_matches": True})
    csv(pd.DataFrame(checks), "causal_alignment_checks.csv")

    correlations = []
    for lag in range(1, 29):
        for feature in FEATURES:
            x = lagged[lag].loc[common_train, feature]
            y = target.loc[common_train]
            correlations.append({"feature": feature, "lag_calendar_days": lag,
                                 "common_pairs_2023": len(x),
                                 "pearson": x.corr(y, method="pearson"),
                                 "spearman": x.corr(y, method="spearman")})
    correlations = pd.DataFrame(correlations)
    assert len(correlations) == 140 and correlations.common_pairs_2023.eq(308).all()
    assert correlations[["pearson", "spearman"]].notna().all().all()
    csv(correlations, "lag_correlations_common_train.csv")
    comparisons = correlations.merge(marginal.rename(columns={"pairs_2023": "legacy_marginal_pairs_2023",
        "pearson": "legacy_marginal_pearson", "spearman": "legacy_marginal_spearman"}),
        on=["feature", "lag_calendar_days"], validate="one_to_one")
    csv(comparisons, "cohort_comparison.csv")
    peaks = []
    for feature in FEATURES:
        group = correlations[correlations.feature.eq(feature)].sort_values("lag_calendar_days")
        for method in ("pearson", "spearman"):
            maximum = group[method].abs().max()
            ties = group.loc[np.isclose(group[method].abs(), maximum, rtol=0, atol=1e-12)]
            chosen = ties.iloc[0]
            peaks.append({"feature": feature, "coefficient": method,
                          "max_absolute_coefficient": maximum,
                          "signed_coefficient_at_reported_lag": chosen[method],
                          "reported_lag_calendar_days": int(chosen.lag_calendar_days),
                          "ties_within_1e_12": ";".join(str(int(v)) for v in ties.lag_calendar_days),
                          "common_pairs_2023": 308, "role": "descriptive peak; not a selected predictor window"})
    csv(pd.DataFrame(peaks), "lag_peak_summary.csv")

    relationship_summaries = []
    rolling_checks = []
    for year in (2023, 2024):
        for feature in FEATURES:
            series = rolling.loc[str(year), feature]
            finite = series.dropna()
            assert (counts.loc[finite.index, feature] >= 50).all()
            assert counts.loc[str(year), feature].le(60).all()
            sign_changes = ((series * series.shift(1)) < 0).sum()
            relationship_summaries.append({
                "year": year, "feature": feature, "calendar_days": len(series),
                "defined_windows": len(finite), "undefined_windows": int(series.isna().sum()),
                "coverage_fraction": len(finite)/len(series), "min": finite.min(),
                "q05": finite.quantile(.05), "median": finite.median(),
                "q95": finite.quantile(.95), "max": finite.max(),
                "positive_fraction_defined": float((finite > 0).mean()),
                "adjacent_defined_sign_changes": int(sign_changes),
                "window_calendar_days": 60, "minimum_valid_pairs": 50,
                "interpretation": "overlapping retrospective diagnostics; not independent regime tests",
            })
    for day in ("2023-03-31", "2023-12-31", "2024-01-01", "2024-12-31"):
        end = pd.Timestamp(day, tz="UTC")
        start = end - pd.Timedelta(days=59)
        for feature in FEATURES:
            pair = pd.concat([lagged[1][feature].rename("x"), target.rename("y")], axis=1).loc[start:end].dropna()
            expected = float(np.corrcoef(pair.x.to_numpy(), pair.y.to_numpy())[0, 1]) if len(pair) >= 50 else np.nan
            recorded = float(rolling.loc[end, feature])
            assert ((np.isnan(expected) and np.isnan(recorded)) or abs(expected-recorded) <= 1e-8)
            assert len(pair) == int(counts.loc[end, feature])
            rolling_checks.append({"outcome_day_utc": day, "feature": feature,
                "first_calendar_day": start.strftime("%Y-%m-%d"), "pairs": len(pair),
                "absolute_difference": abs(expected-recorded) if np.isfinite(expected) else None,
                "matches_source": True})
    csv(pd.DataFrame(relationship_summaries), "rolling_relationship_summary.csv")
    csv(pd.DataFrame(rolling_checks), "rolling_reconstruction_checks.csv")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
    for axis, method in zip(axes, ("pearson", "spearman")):
        matrix = correlations.pivot(index="feature", columns="lag_calendar_days", values=method).loc[list(FEATURES)]
        graph = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        axis.set(yticks=np.arange(5), yticklabels=DISPLAY,
                 title=f"{method.title()} · misma cohorte de 308 días de entrenamiento de 2023")
        fig.colorbar(graph, ax=axis, fraction=.018, pad=.02, label="Correlación descriptiva")
    axes[-1].set(xticks=[0, 6, 13, 20, 27], xticklabels=[1, 7, 14, 21, 28],
                 xlabel="Rezago de la covariable en días de calendario; objetivo σ7 del día de resultado")
    fig.suptitle("Bloque C: comparación de rezagos con cohorte común, sin selección de ventana", fontsize=12)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.text(.01, .02, "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0", fontsize=8, color="#526171")
    figure = "correlaciones_rezagos_cohorte_comun.png"
    fig.savefig(STATIC / figure, dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert all(digest(path) == value for path, value in original_hashes.items())
    summary = {
        "scope": "block_C_descriptive_causal_EDA_only_no_fitted_model",
        "period": {"start": "2023-01-01", "end_inclusive": "2024-12-31", "calendar_days": 731},
        "source_files": [{"path": str(path.relative_to(ROOT)), "sha256": value} for path, value in original_hashes.items()],
        "generator_script_sha256": digest(Path(__file__)),
        "frequency": "completed daily UTC bars; source acquisition was one minute",
        "covariates": list(FEATURES), "target": "sigma_7 outcome day; not an input lag",
        "diagnostic_transforms": "fixed log(OHLC), log1p(volume); no learned scaler",
        "common_cohort": {"year": 2023, "N": 308, "condition": "all 28 lags of all five covariates and sigma7 valid",
            "first_outcome_day": str(CALENDAR[common_train][0].date()),
            "last_outcome_day": str(CALENDAR[common_train][-1].date()),
            "same_dates_used_for_all_correlations": True,
            "full_window_valid_days_by_year": {"2023": int(common_train.sum()), "2024": int(valid_common[CALENDAR.year == 2024].sum())}},
        "lag_days": list(range(1, 29)), "correlation_rows": 140,
        "legacy_marginal_pair_count_range": [int(marginal.pairs_2023.min()), int(marginal.pairs_2023.max())],
        "peak_selection": "largest absolute descriptive coefficient; ties use first lag; no window/model/hyperparameter selected",
        "rolling": {"window_calendar_days": 60, "minimum_pairs": 50,
            "source": str(ROLLING.relative_to(ROOT)), "new_regime_hypothesis_test": False,
            "retrospective_not_current_predictor": True},
        "independent_checks": {"calendar_common_mask_matches_direct_date_lookup": True,
            "explicit_alignment_snapshots": checks, "rolling_snapshot_comparisons": len(rolling_checks),
            "max_rolling_snapshot_absolute_difference": max(row["absolute_difference"] or 0 for row in rolling_checks)},
        "missing_daily_market_days": ["2023-03-24"], "calendar_preserved": True,
        "source_files_unchanged": True, "models_fitted": 0, "hypothesis_tests_run": 0,
        "raw_archives_opened": 0, "test_files_opened": 0, "randomness": "none; deterministic descriptive calculations",
        "figures": [figure],
        "limitations": [
            "Correlations of price levels can be spurious and are not evidence of predictive skill.",
            "Comparing 28 lags creates descriptive selection; no significance or optimal lag is claimed.",
            "The common cohort has 308 dates, unlike the older marginal analyses with 328--350 pairs.",
            "Rolling windows and volatility targets overlap; sign changes are not independent regime changes.",
            "Target sigma7 and rolling relationships labelled d are not admissible current covariates at the start of d.",
            "Residual diagnostics require a separately authorised reference fit; none is fitted by this script.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (OUT / "README.md").write_text("""# EDA específico descriptivo — regresión temporal (bloque C)

Se usan únicamente CSV de desarrollo 2023–2024 existentes. La cuadrícula UTC conserva 731 días y el 24/03/2023 ausente; no se imputan datos ni se abren ZIP o archivos de test.

`covariate_structure.csv`, `availability_contract.csv` y `availability_by_lag.csv` describen las cinco covariables y su disponibilidad. La barra completa de d−k es observable al comienzo de d−k+1, por lo que k≥1 es causal al comienzo de d. El objetivo que incluye el retorno d y la correlación retrospectiva etiquetada d no se incorporan como entradas de ese origen.

`lag_correlations_common_train.csv` contiene 140 coeficientes Pearson/Spearman con los mismos 308 días de 2023. Se exige que los 28 rezagos de todas las covariables y sigma7 sean válidos. `cohort_comparison.csv` conserva el contraste con los análisis marginales previos, que tenían 328–350 pares. `lag_peak_summary.csv` registra máximos absolutos descriptivos; no selecciona un rezago óptimo, una ventana D o un modelo. No se calculan p-values ni se interpreta causalidad o capacidad de pronóstico.

`rolling_relationship_summary.csv` resume por año las correlaciones de 60 días, con mínimo 50 pares, calculadas previamente. Son ventanas retrospectivas solapadas; los cambios de signo no son rupturas de régimen demostradas. Se verificaron veinte reconstrucciones por fecha y covariable sin reutilizar el cálculo móvil de pandas.

Reproducir desde el repositorio: `/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_temporal_specific.py`. `summary.json` conserva hashes, convenciones y verificaciones independientes. No se ajusta ninguna referencia lineal, red u otro predictor; el diagnóstico de residuos no se presenta como ejecutado por estas correlaciones.

Datos: Binance Vision / Binance Spot BTCUSDT. Procesamiento: Jassan Arteta. Derivados y figura: CC BY-NC-SA 4.0 y condiciones aplicables del proveedor. No modifica los derechos de los trabajos citados ni de figuras ajenas.
""")
    for path in OUT.iterdir():
        if path.is_file():
            (STATIC / path.name).write_bytes(path.read_bytes())
    print(json.dumps({"calendar_days": 731, "common_train_pairs": 308, "correlation_rows": 140,
        "rolling_snapshot_checks": len(rolling_checks), "models_fitted": 0, "test_files_opened": 0}))


if __name__ == "__main__":
    main()
