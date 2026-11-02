"""Descriptive temporal-covariate EDA on existing 2023--2024 development CSVs.

No model is fitted, no hypothesis test is performed, and no test-market file is
opened.  Calendar positions and the existing strict-exclusion policy are kept.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e1-covariates-documental")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "results/eda_development/daily_market_development.csv"
TARGETS = ROOT / "results/eda_development/daily_volatility_development.csv"
OUT = ROOT / "results/eda_covariates_documental"
ASSETS = ROOT / "book/_static/eda_doc"
FEATURES = ("open", "high", "low", "close", "volume")
LABELS = ("log(O)", "log(H)", "log(L)", "log(C)", "log(1+V)")


def read_development(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["market_day_utc"])
    frame = frame.set_index("market_day_utc")
    if frame.index.tz is None:
        raise ValueError("Expected explicitly UTC-indexed development data")
    expected = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
    if not frame.index.equals(expected):
        raise ValueError("Only the existing full 2023--2024 daily calendar is admitted")
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    market = read_development(MARKET)
    targets = read_development(TARGETS)
    if not market.index.equals(targets.index):
        raise ValueError("Market and target calendars differ")
    x = np.log(market.loc[:, FEATURES[:4]])
    x["volume"] = np.log1p(market["volume"])
    y = targets["sigma_7"]
    rows = []
    for lag in range(1, 29):
        lagged = x.shift(lag)
        for feature in FEATURES:
            pair = pd.concat([lagged[feature].rename("covariate"), y.rename("target")], axis=1).loc["2023"].dropna()
            rows.append({"feature": feature, "lag_calendar_days": lag, "pairs_2023": len(pair),
                         "pearson": pair.covariate.corr(pair.target, method="pearson"),
                         "spearman": pair.covariate.corr(pair.target, method="spearman")})
    correlations = pd.DataFrame(rows)
    correlations.to_csv(OUT / "lagged_correlations_train.csv", index=False)

    lag_one = x.shift(1)
    rolling = pd.DataFrame(index=x.index)
    counts = pd.DataFrame(index=x.index)
    for feature in FEATURES:
        rolling[feature] = lag_one[feature].rolling("60D", min_periods=50).corr(y)
        paired = lag_one[feature].notna() & y.notna()
        counts[feature] = paired.astype(int).rolling("60D", min_periods=1).sum().astype(int)
    rolling.index.name = counts.index.name = "outcome_day_utc"
    rolling.to_csv(OUT / "rolling_60d_correlations_development.csv")
    counts.to_csv(OUT / "rolling_60d_pair_counts.csv")

    profiles = []
    for year in (2023, 2024):
        for feature in FEATURES:
            pair = pd.concat([lag_one[feature].rename("covariate"), y.rename("target")], axis=1).loc[str(year)].dropna()
            profiles.append({"year": year, "feature": feature, "lag_calendar_days": 1,
                             "pairs": len(pair), "pearson": pair.covariate.corr(pair.target),
                             "spearman": pair.covariate.corr(pair.target, method="spearman")})
    pd.DataFrame(profiles).to_csv(OUT / "relationship_profiles_2023_2024.csv", index=False)

    # Independently resolve calendar offsets rather than reusing shifted arrays.
    checks = []
    for day_text, lag, feature in [("2023-02-10", 7, "close"), ("2023-04-30", 28, "volume"),
                                   ("2024-01-01", 1, "open")]:
        day = pd.Timestamp(day_text, tz="UTC")
        source = day - pd.Timedelta(days=lag)
        raw = float(market.loc[source, feature])
        expected = np.log1p(raw) if feature == "volume" else np.log(raw)
        observed = float(x.shift(lag).loc[day, feature])
        if not np.isclose(expected, observed, rtol=0, atol=1e-12):
            raise AssertionError("Calendar-lag alignment check failed")
        checks.append({"outcome_day_utc": day_text, "lag_days": lag, "feature": feature,
                       "source_day_utc": source.strftime("%Y-%m-%d"), "passed": True})
    # Missing daily positions remain present and cannot become neighboring rows.
    missing_day = pd.Timestamp("2023-03-24", tz="UTC")
    assert market.loc[missing_day, list(FEATURES)].isna().all()
    assert x.shift(1).loc[missing_day + pd.Timedelta(days=1)].isna().all()
    assert len(correlations) == 28 * 5
    assert correlations[["pearson", "spearman"]].notna().all().all()

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for axis, method in zip(axes, ["pearson", "spearman"]):
        values = correlations.pivot(index="feature", columns="lag_calendar_days", values=method).loc[list(FEATURES)]
        plot = axis.imshow(values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
        axis.set(yticks=np.arange(5), yticklabels=LABELS,
                 title=f"{method.title()}: covariable del día d−rezago frente a σ7 del día d · entrenamiento 2023")
        fig.colorbar(plot, ax=axis, fraction=.018, pad=.02, label="Correlación descriptiva")
    axes[-1].set(xticks=[0, 6, 13, 20, 27], xticklabels=[1, 7, 14, 21, 28], xlabel="Rezago en días de calendario")
    fig.tight_layout(rect=(0, .06, 1, 1))
    credit = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"
    fig.text(.01, .013, credit, fontsize=8, color="#526171")
    fig.savefig(ASSETS / "covariables_rezagos_descriptivos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = ("#174c74", "#b86a21", "#7b548f", "#2b8772")
    for feature, label, color in zip(FEATURES[:4], LABELS[:4], colors):
        axes[0].plot(rolling.index, rolling[feature], label=label, color=color, linewidth=1)
    axes[1].plot(rolling.index, rolling["volume"], color="#2b8772", label=LABELS[-1], linewidth=1)
    for axis in axes:
        axis.axhline(0, color="#9ca8b3", linewidth=.7)
        axis.axvline(pd.Timestamp("2024-01-01", tz="UTC"), color="#445566", linestyle="--", label="Inicio validación 2024")
        axis.set(ylim=(-1, 1), ylabel="Pearson móvil 60 días")
        axis.legend(loc="lower left", fontsize=8, ncol=3)
    axes[0].set_title("Relación covariable d−1 / σ7(d): 60 días de calendario, mínimo 50 pares · desarrollo")
    axes[1].set_xlabel("Día de resultado UTC; correlación diagnóstica conocida después de observar ese objetivo")
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.text(.01, .013, credit, fontsize=8, color="#526171")
    fig.savefig(ASSETS / "relacion_movil_covariables_objetivo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {"scope": "descriptive_development_EDA_only_no_fitted_model_or_experiment",
               "period": {"start": "2023-01-01", "end_inclusive": "2024-12-31", "calendar_days": len(market)},
               "source_files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (MARKET, TARGETS)],
               "input_transform_for_diagnostics_only": "log(OHLC), log1p(volume); no fitted scaler",
               "target": "existing sigma_7 daily-return population standard deviation",
               "lagged_correlation_training_year": 2023, "lag_days": list(range(1, 29)),
               "correlation_rows": len(correlations),
               "pair_counts_by_lag_range": [int(correlations.pairs_2023.min()), int(correlations.pairs_2023.max())],
               "rolling": {"window_calendar_days": 60, "minimum_pairs": 50, "centered": False,
                           "availability": "diagnostic after outcome is observed; not an unshifted predictor"},
               "missing_daily_market_days": [t.strftime("%Y-%m-%d") for t in market.index[market.loc[:, FEATURES].isna().all(axis=1)]],
               "original_raw_quality": {"missing_minutes": 80, "noneligible_minute_close_time_rows": 1,
                                        "policy": "existing strict exclusion; no imputation or recovery performed"},
               "calendar_alignment_checks": checks, "missing_calendar_position_preserved": True,
               "models_fitted": 0, "hypothesis_tests_run": 0, "test_files_opened": 0,
               "limitations": ["Correlations of price levels may be spurious and do not establish causality or forecast value.",
                               "Overlapping rolling windows and volatility targets make plotted values dependent.",
                               "Differences between years are descriptive, not evidence of a significant regime change.",
                               "No variable, history, model, block length or hyperparameter is selected using these plots.",
                               "Reference-model residual diagnostics remain deferred to E2 under the user's scope."]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for name in ["lagged_correlations_train.csv", "rolling_60d_correlations_development.csv",
                 "rolling_60d_pair_counts.csv", "relationship_profiles_2023_2024.csv", "summary.json"]:
        (ASSETS / name).write_bytes((OUT / name).read_bytes())
    (OUT / "README.md").write_text("""# EDA descriptivo de regresión temporal\n\nSe reutilizan únicamente los CSV de desarrollo 2023–2024 ya existentes. No se ajusta ningún modelo, no se ejecutan contrastes de hipótesis y no se abren archivos de prueba. Se conserva la exclusión estricta del día incompleto y la cuadrícula diaria.\n\n`lagged_correlations_train.csv` contiene Pearson/Spearman para cinco covariables con rezagos causales de 1 a 28 días frente a sigma7, solo en 2023. `rolling_60d_correlations_development.csv` y sus conteos describen ventanas móviles de 60 días con al menos 50 pares de desarrollo. `relationship_profiles_2023_2024.csv` no constituye un contraste formal de deriva. Los niveles de precios pueden producir correlaciones espurias.\n\nReproducir desde el repositorio: `/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_covariates_documental.py`. Los hashes, controles de alineación y límites se conservan en `summary.json`. Los diagnósticos de residuos de una referencia requieren ajuste de modelo y permanecen para E2 según el alcance solicitado.\n""")
    print(json.dumps({"calendar_days": len(market), "correlation_rows": len(correlations),
                      "pair_counts": summary["pair_counts_by_lag_range"], "models_fitted": 0,
                      "test_files_opened": 0, "alignment_checks": len(checks)}))


if __name__ == "__main__":
    main()
