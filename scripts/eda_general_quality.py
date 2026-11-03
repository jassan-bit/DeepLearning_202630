"""Objective profiles and quality audit from existing development CSVs only.

The script reads no ZIP and no test period, fits no model and imputes nothing.
Independent label reconstruction checks numerical consistency, not economic truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e1-general-quality")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/eda_development"
OUT = ROOT / "results/eda_general_quality"
ASSETS = ROOT / "book/_static/eda_general_quality"
FEATURES = ["open", "high", "low", "close", "volume"]
WINDOWS = (7, 14, 21, 28)
SOURCE_NAMES = ("daily_market_development.csv", "daily_volatility_development.csv",
                "summary.json", "coverage_monthly.csv")
CALENDAR = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
CREDIT = "Datos: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_calendar_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / name, parse_dates=["market_day_utc"])
    frame = frame.set_index("market_day_utc")
    if not frame.index.equals(CALENDAR):
        raise ValueError(f"{name}: expected unique regular UTC calendar 2023--2024")
    return frame


def write_csv(frame: pd.DataFrame, name: str, *, index=False):
    for directory in (OUT, ASSETS):
        frame.to_csv(directory / name, index=index, lineterminator="\n")


def write_json(data, name: str):
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for directory in (OUT, ASSETS):
        (directory / name).write_text(payload)


def save_figure(fig, name: str):
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .012, CREDIT, fontsize=8, color="#526171")
    fig.savefig(ASSETS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    for directory in (OUT, ASSETS):
        directory.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: digest(SOURCE / name) for name in SOURCE_NAMES}
    market = read_calendar_csv(SOURCE_NAMES[0])
    labels = read_calendar_csv(SOURCE_NAMES[1])
    previous = json.loads((SOURCE / "summary.json").read_text())
    assert market.index.equals(labels.index)
    assert market.loc[pd.Timestamp("2023-03-24", tz="UTC"), FEATURES].isna().all()
    assert len(market) == 731
    finite_ohlcv = pd.Series(np.isfinite(market[FEATURES]).all(axis=1), index=CALENDAR)
    daily_returns = np.log(market["close"]).diff()

    # Population moments and fixed training-reference IQR flags: no observations removed.
    profiles = []
    fences = {}
    for window in WINDOWS:
        training = labels[f"sigma_{window}"].loc["2023"].dropna().to_numpy()
        q25, q75 = np.quantile(training, [.25, .75], method="linear")
        low, high = float(q25 - 1.5*(q75-q25)), float(q75 + 1.5*(q75-q25))
        fences[window] = (low, high)
        for year in (2023, 2024):
            all_values = labels[f"sigma_{window}"].loc[str(year)]
            values = all_values.dropna().to_numpy()
            qs = np.quantile(values, [.01, .05, .25, .5, .75, .95, .99], method="linear")
            row = {"year": year, "volatility_days": window,
                   "role": "principal" if window == 7 else "sensitivity",
                   "calendar_days": len(all_values), "valid_labels": len(values),
                   "missing_labels": int(all_values.isna().sum()),
                   "zero_labels": int((values == 0).sum()),
                   "positive_labels": int((values > 0).sum()),
                   "mean": float(np.mean(values)), "std_ddof0": float(np.std(values, ddof=0)),
                   "min": float(np.min(values)), "max": float(np.max(values)),
                   "q01": float(qs[0]), "q05": float(qs[1]), "q25": float(qs[2]),
                   "median": float(qs[3]), "q75": float(qs[4]), "q95": float(qs[5]),
                   "q99": float(qs[6]), "skew_unbiased": float(skew(values, bias=False)),
                   "excess_kurtosis_unbiased": float(kurtosis(values, fisher=True, bias=False)),
                   "iqr_reference_year": 2023, "iqr_factor": 1.5,
                   "iqr_lower_fence_train": low, "iqr_upper_fence_train": high,
                   "below_training_fence": int((values < low).sum()),
                   "above_training_fence": int((values > high).sum()),
                   "flagged_fraction": float(((values < low) | (values > high)).mean())}
            profiles.append(row)
    write_csv(pd.DataFrame(profiles), "target_profiles.csv")

    # These are calendar-day checks on existing derivatives; not a new raw-data audit.
    duplicate_values = market.loc[finite_ohlcv, FEATURES].duplicated(keep=False)
    quality_rows = []
    for year in (2023, 2024):
        part = market.loc[str(year), FEATURES]
        complete = part.loc[finite_ohlcv.loc[str(year)]]
        bad_range = ((complete["high"] < complete[["open", "close"]].max(axis=1)) |
                     (complete["low"] > complete[["open", "close"]].min(axis=1)) |
                     (complete["high"] < complete["low"]))
        quality_rows.append({"year": year, "calendar_days": len(part),
                             "complete_ohlcv_days": len(complete),
                             "incomplete_ohlcv_days": len(part)-len(complete),
                             "complete_ohlcv_fraction": len(complete)/len(part),
                             "duplicate_calendar_keys": int(part.index.duplicated().sum()),
                             "rows_with_repeated_complete_ohlcv_values": int(duplicate_values.loc[str(year)].sum()),
                             "nonpositive_ohlc_rows": int((complete[FEATURES[:4]] <= 0).any(axis=1).sum()),
                             "incoherent_ohlc_rows": int(bad_range.sum()),
                             "negative_volume_rows": int((complete["volume"] < 0).sum()),
                             "valid_daily_returns": int(daily_returns.loc[str(year)].notna().sum()),
                             "missing_daily_returns": int(daily_returns.loc[str(year)].isna().sum())})
    write_csv(pd.DataFrame(quality_rows), "quality_by_year.csv")

    # Rebuild all 2,924 window-day positions using direct calendar lookups and fsum.
    # This does not reuse pandas rolling/std or remove calendar gaps.
    closes = {day: float(value) for day, value in market["close"].items()}
    audit_rows = []
    for window in WINDOWS:
        for day in CALENDAR:
            needed_days = [day-pd.Timedelta(days=i) for i in range(window, -1, -1)]
            before_calendar = any(d < CALENDAR[0] for d in needed_days)
            complete = all(d in closes and math.isfinite(closes[d]) and closes[d] > 0 for d in needed_days)
            observed = float(labels.at[day, f"sigma_{window}"])
            reconstructed = math.nan
            if complete:
                returns = [math.log(closes[b]/closes[a]) for a, b in zip(needed_days, needed_days[1:])]
                mean = math.fsum(returns)/window
                reconstructed = math.sqrt(math.fsum((value-mean)**2 for value in returns)/window)
            observed_valid = math.isfinite(observed)
            assert observed_valid == complete, (day, window, "eligibility mismatch")
            error = abs(observed-reconstructed) if complete else math.nan
            if complete:
                assert error <= 1e-12, (day, window, error)
            audit_rows.append({"outcome_day_utc": day.strftime("%Y-%m-%d"),
                               "year": day.year, "volatility_days": window,
                               "required_first_close_day_utc": needed_days[0].strftime("%Y-%m-%d"),
                               "valid_expected_from_calendar": complete, "valid_in_existing_csv": observed_valid,
                               "ineligibility_reason": "" if complete else ("initial_history_unavailable" if before_calendar else "excluded_close_in_required_calendar_support"),
                               "existing_label": observed, "independent_label": reconstructed,
                               "absolute_numerical_difference": error})
    audit = pd.DataFrame(audit_rows)
    write_csv(audit, "label_reconstruction.csv")
    reconstruction = []
    for window in WINDOWS:
        part = audit.loc[audit.volatility_days.eq(window)]
        valid = part.loc[part.valid_expected_from_calendar]
        reconstruction.append({"volatility_days": window, "calendar_positions": len(part),
                               "valid_positions": len(valid), "missing_positions": len(part)-len(valid),
                               "eligibility_mismatches": int((part.valid_expected_from_calendar != part.valid_in_existing_csv).sum()),
                               "max_absolute_numerical_difference": float(valid.absolute_numerical_difference.max()),
                               "mean_absolute_numerical_difference": float(valid.absolute_numerical_difference.mean()),
                               "first_valid_outcome_day_utc": valid.outcome_day_utc.iloc[0],
                               "first_valid_after_gap_day_utc": valid.loc[valid.outcome_day_utc.gt("2023-03-24"), "outcome_day_utc"].iloc[0],
                               "comparison_tolerance": 1e-12})
    write_csv(pd.DataFrame(reconstruction), "label_reconstruction_summary.csv")

    masks = pd.DataFrame(index=CALENDAR)
    masks["complete_ohlcv"] = finite_ohlcv
    masks["valid_daily_return"] = daily_returns.notna()
    for window in WINDOWS:
        masks[f"valid_sigma_{window}"] = labels[f"sigma_{window}"].notna()
    input_masks = {}
    eligibility = []
    for history in WINDOWS:
        # d itself is not an input; d-D,...,d-1 must all be observed and complete.
        input_masks[history] = finite_ohlcv.shift(1, fill_value=False).rolling(history, min_periods=history).sum().eq(history)
        masks[f"valid_X{history}_and_sigma_7"] = input_masks[history] & masks.valid_sigma_7
        for window in WINDOWS:
            combined = input_masks[history] & masks[f"valid_sigma_{window}"]
            for year in (2023, 2024):
                selected = combined.loc[str(year)]
                eligibility.append({"year": year, "history_days": history, "volatility_days": window,
                                    "calendar_days": len(selected), "eligible_outcome_days": int(selected.sum()),
                                    "ineligible_outcome_days": int((~selected).sum()),
                                    "scope": "calendar input-and-label eligibility only; not training-fold counts or test eligibility"})
    assert masks.loc[pd.Timestamp("2023-03-25", tz="UTC"), "valid_X7_and_sigma_7"] == False
    masks.index.name = "outcome_day_utc"
    write_csv(masks.reset_index(), "calendar_masks.csv")
    write_csv(pd.DataFrame(eligibility), "window_eligibility.csv")

    missingness = {
        "previous_raw_audit_only_no_raw_archive_reopened": True,
        "missing_minutes": previous["missing_minutes"],
        "missing_intervals": previous["missing_intervals"],
        "noneligible_close_time_rows": previous["unexpected_close_time_rows"],
        "noneligible_close_time_positions": previous["close_time_anomalies"],
        "excluded_daily_market_days": [d.strftime("%Y-%m-%d") for d in CALENDAR[~finite_ohlcv]],
        "daily_return_days_affected_by_excluded_close": [d.strftime("%Y-%m-%d") for d in CALENDAR[daily_returns.isna()] if d != CALENDAR[0]],
        "mechanism_classification": "not_identified",
        "mechanism_evidence": "A single concentrated temporal episode is observed. The observation does not identify MCAR, MAR or MNAR, and no provider-cause document or independent price reference was available in these inputs.",
        "policy": "strict exclusion; no imputation; preserve calendar and invalidate required windows",
        "numerical_label_audit_is_not": "an estimate of provider price measurement error, economic proxy noise, or absence of all data leakage",
    }
    write_json(missingness, "missingness_evidence.json")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {2023: "#265b96", 2024: "#b25b20"}
    for year in (2023, 2024):
        values = np.sort(labels.sigma_7.loc[str(year)].dropna().to_numpy())*100
        p = np.linspace(0, 1, len(values))
        axes[0].plot(p, values, color=colors[year], label=f"{year}, N={len(values)}")
        unique_values, frequencies = np.unique(values, return_counts=True)
        preceding_count = np.r_[0, np.cumsum(frequencies)[:-1]]
        survival = (len(values)-preceding_count)/len(values)
        axes[1].step(unique_values, survival, where="post", color=colors[year], label=str(year))
    axes[0].set(title="Cuantiles empíricos del objetivo principal", xlabel="Probabilidad acumulada", ylabel="σ7 de retornos diarios (%)")
    axes[1].set(title="Cola empírica del objetivo; escala vertical log", xlabel="σ7 de retornos diarios (%)", ylabel="Fracción observada ≥ valor", yscale="log")
    axes[1].axvline(fences[7][1]*100, color="#555555", linestyle="--", linewidth=1, label="Límite IQR de 2023")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.15)
    save_figure(fig, "objetivo_7_percentiles_colas.png")

    display_columns = list(masks.columns)
    display_names = ["OHLCV diario", "Retorno diario", "σ7", "σ14", "σ21", "σ28",
                     "X7 y σ7", "X14 y σ7", "X21 y σ7", "X28 y σ7"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True)
    cmap = ListedColormap(["#dce2e8", "#265b96"])
    cmap.set_bad("white")
    for axis, year in zip(axes, (2023, 2024)):
        values = masks.loc[str(year), display_columns].to_numpy(dtype=float).T
        if values.shape[1] == 365:
            values = np.column_stack([values, np.full(len(display_columns), np.nan)])
        axis.imshow(np.ma.masked_invalid(values), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1, extent=(.5, 366.5, len(display_columns)-.5, -.5))
        axis.set(yticks=np.arange(len(display_columns)), yticklabels=display_names,
                 title=f"{year}: posiciones de calendario conservadas")
    axes[-1].set(xlabel="Día del año UTC", xticks=[1, 60, 120, 180, 240, 300, 366])
    fig.legend(handles=[Patch(facecolor="#265b96", label="Disponible / elegible"),
                        Patch(facecolor="#dce2e8", label="Ausente o historia insuficiente")],
               loc="lower center", ncol=2, bbox_to_anchor=(.5, .025), frameon=False)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.text(.01, .008, CREDIT, fontsize=8, color="#526171")
    fig.savefig(ASSETS / "mascaras_calendario_desarrollo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    assert source_hashes == {name: digest(SOURCE / name) for name in SOURCE_NAMES}
    summary = {
        "scope": "general EDA on existing development derivatives only; no model, imputation, test data or new raw audit",
        "period": {"start": "2023-01-01", "end_inclusive": "2024-12-31", "calendar_days": 731},
        "source_hashes": source_hashes, "generator_script_sha256": digest(Path(__file__)),
        "target_profiles": profiles, "daily_quality": quality_rows,
        "reconstruction": reconstruction,
        "total_label_calendar_positions_checked": len(audit),
        "valid_labels_reconstructed": int(audit.valid_expected_from_calendar.sum()),
        "eligibility_mismatches": 0,
        "max_absolute_label_numerical_difference": float(audit.absolute_numerical_difference.max()),
        "duplicate_complete_ohlcv_value_rows": int(duplicate_values.sum()),
        "calendar_preserved": True, "source_files_unchanged": True,
        "models_fitted": 0, "raw_archives_opened": 0, "test_files_opened": 0,
        "randomness": "none; deterministic descriptive and reconstruction calculations",
        "class_balance_metrics": "not applicable: continuous regression target, no artificial classes created",
        "iqr_rule": "1.5 IQR fences calculated separately per n using 2023 labels only, then applied unchanged to each year; flags are retained, not deleted",
        "quantiles": "numpy linear interpolation", "std": "population ddof=0", "moments": "scipy skew bias=False; excess kurtosis fisher=True,bias=False",
        "label_noise_external_estimate": None,
        "limitations": [
            "Independent reconstruction checks calculation and calendar support; it does not validate the provider prices against an independent source.",
            "Neither repeated values nor IQR flags are automatic grounds for deleting financial observations.",
            "The missingness mechanism remains unidentified; temporal concentration alone does not establish MCAR, MAR or MNAR.",
            "Price/proxy measurement noise is not estimated by the near-zero numerical reconstruction difference.",
            "Window-eligibility counts precede fold-specific embargo/training rules and are not independent samples.",
            "Skewness, kurtosis and empirical survival describe observed tails; no parametric tail law or causal market explanation is established.",
        ],
        "figures": ["objetivo_7_percentiles_colas.png", "mascaras_calendario_desarrollo.png"],
    }
    write_json(summary, "summary.json")
    readme = """# Complemento del EDA general: objetivo, calidad y máscaras

Se reutilizan únicamente cuatro archivos de `results/eda_development/`, con sus hashes:
OHLCV diario, volatilidades, resumen del EDA previo y cobertura mensual. No se abren ZIP,
no se accede a 2025–2026, no se ajusta ningún modelo y no se imputa el 24/03/2023.

Reproducción: `/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_general_quality.py`.

- `target_profiles.csv`: ocho perfiles (dos años por cuatro definiciones de volatilidad),
  percentiles, momentos, ceros y banderas IQR. Los umbrales se calculan solo con 2023 y
  se aplican sin reajustar a 2024; las banderas no eliminan observaciones.
- `quality_by_year.csv`: completitud de días, clave temporal, valores OHLCV repetidos,
  coherencia OHLC y disponibilidad de retornos, sobre derivados existentes.
- `label_reconstruction.csv` y `label_reconstruction_summary.csv`: reconstrucción directa
  por fechas con `math.log`, `math.fsum` y divisor poblacional, sin reutilizar rolling/std
  de pandas. Se comprueban todas las posiciones, incluidos calentamiento y huecos.
  La diferencia numérica no estima ruido de precios ni del proxy económico.
- `calendar_masks.csv`: posiciones diarias preservadas para OHLCV, retornos, etiquetas
  y ventanas OHLCV de las cuatro historias frente al objetivo principal.
- `window_eligibility.csv`: combinaciones año/D/n antes de reglas específicas de folds,
  embargo o entrenamiento; no son tamaños de muestra independientes.
- `missingness_evidence.json`: conserva el episodio de 80 minutos ausentes y un cierre
  no elegible del resumen previo; el mecanismo MCAR/MAR/MNAR permanece no identificado.
- `summary.json`: métodos, límites, hashes y controles; generación determinista.

Las figuras muestran cuantiles/cola empírica de sigma7 y máscaras de calendario 2023/24.
Fuente: Binance Vision / Binance Spot BTCUSDT. Procesamiento: Jassan Arteta.
Derivados y figuras: CC BY-NC-SA 4.0 y condiciones aplicables del proveedor.
"""
    for directory in (OUT, ASSETS):
        (directory / "README.md").write_text(readme)
    print(json.dumps({"objective_profiles": len(profiles), "label_positions_checked": len(audit),
                      "valid_labels_reconstructed": summary["valid_labels_reconstructed"],
                      "max_absolute_numerical_difference": summary["max_absolute_label_numerical_difference"],
                      "models_fitted": 0, "test_files_opened": 0}))


if __name__ == "__main__":
    main()
