"""Development-only marginal distribution-shift diagnostics for the general EDA.

Compare OHLCV(d-1) and sigma_7(d) in 2023 and 2024. No model is fitted and
no raw archive or test-market file is opened. Permutation reference values are
conditional exploratory diagnostics, not calibrated confirmatory claims.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e1-eda-general-shift")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache-e1-eda-general-shift")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "results/eda_development/daily_market_development.csv"
TARGETS = ROOT / "results/eda_development/daily_volatility_development.csv"
OUT = ROOT / "results/eda_general_shift"
ASSETS = ROOT / "book/_static/eda_general_shift"
FEATURES = ("open", "high", "low", "close", "volume", "sigma_7")
LABELS = ("Apertura d−1", "Máximo d−1", "Mínimo d−1", "Cierre d−1", "Volumen d−1", "Volatilidad σ7(d)")
UNITS = ("USDT", "USDT", "USDT", "USDT", "BTC", "Retorno logarítmico diario")
BLOCK_DAYS = (14, 28)
REPLICATES = 2000
SEED = 42
ALPHA = .05


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("KS distance requires two nonempty finite samples")
    grid = np.unique(np.concatenate((a, b)))
    difference = np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(difference).max())


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    adjusted_ordered = np.minimum(1., np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    adjusted = np.empty_like(pvalues)
    adjusted[order] = adjusted_ordered
    return adjusted


def read_development(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["market_day_utc"]).set_index("market_day_utc")
    expected = pd.date_range("2023-01-01", "2025-01-01", inclusive="left", tz="UTC")
    if not frame.index.equals(expected):
        raise ValueError("Only the existing complete UTC daily calendar for 2023--2024 is admitted")
    return frame


def block_reference(frame: pd.DataFrame, days: int, observed: np.ndarray, rng: np.random.Generator):
    """Shuffle fully observed complete blocks; condition on missing blocks/tails.

    Holding incomplete blocks and short year-end tails fixed avoids exchanging
    unequal block sizes or the deterministic warm-up/gap masks between years.
    Their observed values still contribute to every full-sample KS statistic.
    """
    values = frame.loc[:, FEATURES].to_numpy()
    eligible = []
    original_groups = []
    fixed = {2023: [], 2024: []}
    block_rows = []
    for year in (2023, 2024):
        positions = np.flatnonzero(frame.index.year == year)
        for offset in range(0, len(positions), days):
            idx = positions[offset:offset + days]
            valid = np.isfinite(values[idx]).all(axis=1)
            full = len(idx) == days
            permutable = bool(full and valid.all())
            if permutable:
                eligible.append(values[idx])
                original_groups.append(year)
            else:
                fixed[year].append(values[idx][valid])
            block_rows.append({"block_days": days, "source_year": year,
                               "block_start_utc": frame.index[idx[0]].strftime("%Y-%m-%d"),
                               "block_end_utc": frame.index[idx[-1]].strftime("%Y-%m-%d"),
                               "calendar_days": len(idx), "complete_case_days": int(valid.sum()),
                               "full_calendar_block": full, "permutable": permutable,
                               "reason_if_fixed": "" if permutable else ("short_year_end_tail" if not full else "contains_existing_masked_days")})
    eligible = np.stack(eligible)
    groups = np.asarray(original_groups)
    fixed = {year: np.concatenate(fixed[year], axis=0) for year in (2023, 2024)}
    n_train_blocks = int((groups == 2023).sum())
    n_val_blocks = int((groups == 2024).sum())
    fixed_train_n, fixed_val_n = len(fixed[2023]), len(fixed[2024])
    n_train = n_train_blocks * days + fixed_train_n
    n_val = n_val_blocks * days + fixed_val_n
    assert (n_train, n_val) == (350, 366)
    assert eligible.shape[1:] == (days, len(FEATURES))

    # Global ordered support permits computing all six KS statistics from block
    # cumulative counts, without recompressing calendar dates or fitting anything.
    max_grid = n_train + n_val
    block_cdf_counts = np.zeros((len(eligible), len(FEATURES), max_grid), dtype=np.int32)
    fixed_train_counts = np.zeros((len(FEATURES), max_grid), dtype=np.int32)
    total_counts = np.zeros_like(fixed_train_counts)
    grid_lengths = []
    for column in range(len(FEATURES)):
        all_valid = frame.iloc[:, column].dropna().to_numpy()
        grid = np.unique(all_valid)
        grid_lengths.append(len(grid))
        for k, block in enumerate(eligible):
            counts = np.searchsorted(np.sort(block[:, column]), grid, side="right")
            block_cdf_counts[k, column, :len(grid)] = counts
            block_cdf_counts[k, column, len(grid):] = days
        counts = np.searchsorted(np.sort(fixed[2023][:, column]), grid, side="right")
        fixed_train_counts[column, :len(grid)] = counts
        fixed_train_counts[column, len(grid):] = fixed_train_n
        total_counts[column, :len(grid)] = np.searchsorted(np.sort(all_valid), grid, side="right")
        total_counts[column, len(grid):] = len(all_valid)

    def distances(train_blocks):
        a_counts = block_cdf_counts[train_blocks].sum(axis=0) + fixed_train_counts
        b_counts = total_counts - a_counts
        assert np.all(a_counts[:, -1] == n_train)
        assert np.all(b_counts[:, -1] == n_val)
        return np.max(np.abs(a_counts / n_train - b_counts / n_val), axis=1)

    identity = distances(np.flatnonzero(groups == 2023))
    assert np.allclose(identity, observed, rtol=0, atol=1e-15)
    permutation = np.empty((REPLICATES, len(FEATURES)))
    for replicate in range(REPLICATES):
        selected = rng.permutation(len(eligible))[:n_train_blocks]
        permutation[replicate] = distances(selected)
    exceedances = (permutation >= (observed - 1e-15)).sum(axis=0)
    pvalue = (1 + exceedances) / (REPLICATES + 1)
    metadata = {"block_days": days, "permutable_blocks_total": len(eligible),
                "permutable_blocks_original_train": n_train_blocks,
                "permutable_blocks_original_validation": n_val_blocks,
                "fixed_valid_days_train": fixed_train_n, "fixed_valid_days_validation": fixed_val_n,
                "fixed_valid_fraction_train": fixed_train_n / n_train,
                "fixed_valid_fraction_validation": fixed_val_n / n_val,
                "sample_size_train_every_permutation": n_train,
                "sample_size_validation_every_permutation": n_val,
                "short_tails_calendar_days": {"2023": 1, "2024": 2},
                "complete_but_masked_blocks_held_fixed": sum(r["full_calendar_block"] and not r["permutable"] for r in block_rows),
                "grid_lengths": dict(zip(FEATURES, grid_lengths)),
                "original_assignment_recovers_observed_KS": True,
                "pvalue_mc_formula": "(1 + count(D_permutation >= D_observed)) / (2000 + 1)",
                "floating_comparison_tolerance": 1e-15}
    return pvalue, exceedances, permutation, metadata, block_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in (MARKET, TARGETS)}
    market = read_development(MARKET)
    targets = read_development(TARGETS)
    profile = market.loc[:, FEATURES[:5]].shift(1).join(targets[["sigma_7"]])
    complete = profile.notna().all(axis=1)
    profile = profile.where(complete)
    profile.index.name = "outcome_day_utc"
    assert len(profile) == 731
    assert complete.loc["2023"].sum() == 350 and complete.loc["2024"].sum() == 366
    assert market.loc[pd.Timestamp("2023-03-24", tz="UTC"), list(FEATURES[:5])].isna().all()
    assert profile.loc[pd.Timestamp("2023-03-25", tz="UTC")].isna().all()
    audited = profile.copy()
    audited.insert(0, "complete_case", complete)
    audited.insert(0, "covariate_day_utc", (profile.index - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    audited.to_csv(OUT / "causal_daily_profile.csv", lineterminator="\n")

    # Independent implementation checks include ties, unequal lengths and extremes.
    check_pairs = [([0, 0, 1], [0, 1, 1, 2]), ([1, 2], [1, 2]),
                   ([0, 0, 0], [1, 1]), ([-2, .5, 8, 9], [-3, -1, 2])]
    for a, b in check_pairs:
        assert abs(ks_distance(a, b) - ks_2samp(a, b).statistic) < 1e-15
    assert np.allclose(holm_adjust(np.array([.01, .04, .03])), [.03, .06, .06])

    descriptive = []
    differences = []
    observed = []
    for feature, label, units in zip(FEATURES, LABELS, UNITS):
        samples = {year: profile.loc[str(year), feature].dropna().to_numpy() for year in (2023, 2024)}
        for year, sample in samples.items():
            q = np.quantile(sample, [.01, .05, .25, .5, .75, .95, .99])
            descriptive.append({"feature": feature, "label": label, "units": units, "year": year,
                                "valid_days": len(sample), "mean": float(sample.mean()),
                                "sample_sd": float(sample.std(ddof=1)), "min": float(sample.min()),
                                **dict(zip(["q01", "q05", "q25", "median", "q75", "q95", "q99"], q)),
                                "max": float(sample.max())})
        a, b = samples[2023], samples[2024]
        distance = ks_distance(a, b)
        assert abs(distance - ks_2samp(a, b).statistic) < 1e-15
        observed.append(distance)
        scale = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
        iqr = float(np.quantile(a, .75) - np.quantile(a, .25))
        differences.append({"feature": feature, "label": label, "units": units,
                            "n_train": len(a), "n_validation": len(b),
                            "mean_train": float(a.mean()), "mean_validation": float(b.mean()),
                            "mean_difference_validation_minus_train": float(b.mean() - a.mean()),
                            "median_train": float(np.median(a)), "median_validation": float(np.median(b)),
                            "median_difference_validation_minus_train": float(np.median(b) - np.median(a)),
                            "median_ratio_validation_over_train": float(np.median(b) / np.median(a)),
                            "standardized_mean_difference": float((b.mean() - a.mean()) / scale),
                            "median_difference_over_training_IQR": float((np.median(b) - np.median(a)) / iqr),
                            "KS_distance": distance})
    observed = np.asarray(observed)
    pd.DataFrame(descriptive).to_csv(OUT / "descriptive_by_period.csv", index=False, lineterminator="\n")
    pd.DataFrame(differences).to_csv(OUT / "marginal_distribution_differences.csv", index=False, lineterminator="\n")

    rng = np.random.default_rng(SEED)
    contrast_rows = []
    block_metadata = []
    all_block_rows = []
    for days in BLOCK_DAYS:
        pvalue, extremes, null_stats, metadata, blocks = block_reference(profile, days, observed, rng)
        block_metadata.append(metadata)
        all_block_rows.extend(blocks)
        for column, feature in enumerate(FEATURES):
            contrast_rows.append({"feature": feature, "block_days": days,
                                  "n_train": 350, "n_validation": 366,
                                  "KS_observed": observed[column], "permutations": REPLICATES,
                                  "exceedances": int(extremes[column]), "conditional_p_mc": float(pvalue[column]),
                                  "reference_KS_q50": float(np.quantile(null_stats[:, column], .5)),
                                  "reference_KS_q95": float(np.quantile(null_stats[:, column], .95)),
                                  "reference_KS_q99": float(np.quantile(null_stats[:, column], .99)),
                                  "reference_status": "exploratory_conditional_not_confirmatory"})
    adjusted = holm_adjust(np.asarray([r["conditional_p_mc"] for r in contrast_rows]))
    for row, value in zip(contrast_rows, adjusted):
        row["conditional_p_holm_12"] = float(value)
        row["below_alpha_under_conditional_reference"] = bool(value < ALPHA)
    pd.DataFrame(contrast_rows).to_csv(OUT / "conditional_block_permutation.csv", index=False, lineterminator="\n")
    pd.DataFrame(all_block_rows).to_csv(OUT / "calendar_blocks.csv", index=False, lineterminator="\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for column, (axis, feature, label, units) in enumerate(zip(axes.flat, FEATURES, LABELS, UNITS)):
        for year, color in [(2023, "#265b96"), (2024, "#d17a24")]:
            sample = np.sort(profile.loc[str(year), feature].dropna().to_numpy())
            axis.step(sample, np.arange(1, len(sample) + 1) / len(sample), where="post",
                      label=f"{year} (n={len(sample)})", color=color, linewidth=1.5)
        axis.set(title=label, xlabel=units, ylabel="Distribución acumulada", ylim=(0, 1.04))
        axis.grid(alpha=.15)
        if feature in FEATURES[:4]:
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value / 1000:.0f} mil"))
        axis.text(.97, .08, f"Distancia KS = {observed[column]:.3f}", transform=axis.transAxes,
                  ha="right", fontsize=9, bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle("Desarrollo 2023–2024: marginales de OHLCV(d−1) y σ7(d)\nCasos completos comunes; diagnóstico descriptivo, sin datos de test", fontsize=13)
    fig.tight_layout(rect=(0, .04, 1, .94))
    fig.text(.01, .012, "Fuente: Binance Vision / BTCUSDT | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0", fontsize=8, color="#526171")
    fig.savefig(ASSETS / "marginales_causales_2023_2024.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    limitations = [
        "Las seis comparaciones son marginales de OHLCV(d−1) y sigma7(d), no una prueba de igualdad conjunta ni de todas las ventanas D.",
        "La referencia permutada presupone intercambiabilidad aproximada de bloques plenamente observados, condicionada a colas y bloques con máscaras fijados. Ese supuesto no está demostrado.",
        "Los niveles de precios presentan tendencias/regímenes y puede persistir dependencia entre bloques. Los valores p son diagnósticos condicionales exploratorios; no se presentan como pruebas exactas ni confirmatorias de drift.",
        "La corrección de Holm controla la familia de 12 comparaciones solo si las calibraciones marginales son adecuadas; no corrige supuestos de intercambiabilidad o serialidad.",
        "Mantener fijos bloques con huecos y colas conserva las máscaras y todos los casos observados, pero limita la variación de referencia a los bloques plenamente observados.",
        "La hipótesis condicional no es igualdad incondicional de todas las distribuciones de los años: las diferencias en las partes fijadas se condicionan, no se aleatorizan. Con 14 días quedan fijos 28/350 valores de entrenamiento (8 %); con 28 días, 70/350 (20 %). En validación quedan fijos 2/366 en ambos casos. La sensibilidad cambia también el conjunto permutable, no solo la longitud de bloque.",
        "Los largos 14/28, 2000 permutaciones y semilla42 se fijan antes de esta ejecución exploratoria; no son un prerregistro previo al EDA del estudio ni la longitud definitiva de inferencia en E2.",
        "No se seleccionan variables, transformaciones, ventanas ni modelos con estos p-values. No se igualan artificialmente las distribuciones entre periodos.",
        "El test 2025–2026 no se consulta: su reserva impide usarlo para diseñar el preprocesamiento."
    ]
    summary = {"scope": "development_general_EDA_marginal_shift_only_no_fitted_model",
               "source_files": [{"path": path, "sha256": value} for path, value in source_hashes.items()],
               "script_sha256": digest(Path(__file__)),
               "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
               "calendar": {"start": "2023-01-01", "end_inclusive": "2024-12-31", "days": 731, "missing_daily_market_day": "2023-03-24"},
               "alignment": "OHLCV from calendar day d-1; sigma7 labelled by outcome day d; no target lag is a model input",
               "variables": list(FEATURES), "scale": "untransformed original units; no scaler fitted",
               "common_complete_case_days": {"2023": 350, "2024": 366},
               "masked_calendar_days": {"2023": 15, "2024": 0},
               "descriptive_standardized_mean_difference": "(mean_validation-mean_train)/sqrt((sample_variance_train+sample_variance_validation)/2)",
               "seed": SEED, "rng": "numpy.default_rng / PCG64; common assignments for the six variables per replicate",
               "permutations_per_block_length": REPLICATES, "block_days": list(BLOCK_DAYS),
               "block_permutation": block_metadata,
               "multiplicity": {"method": "Holm", "family_size": 12, "alpha": ALPHA, "family": "six marginal KS comparisons times two exploratory block lengths", "minimum_unadjusted_mc_value": 1 / (REPLICATES + 1)},
               "contrasts": contrast_rows,
               "meaningful_checks": {"KS_vs_SciPy_fixtures_and_all_six_observed_samples": True, "identity_assignment_recovers_all_observed_KS": True, "sample_counts_constant_across_permutations": True, "calendar_and_missing_day_preserved": True},
               "models_fitted": 0, "hypothesis_reference": "exploratory conditional block randomization, not a model benchmark", "test_files_opened": 0,
               "limitations": limitations}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    readme = """# Desplazamiento marginal del EDA general: desarrollo 2023–2024

Se reutilizan exclusivamente los dos CSV diarios de desarrollo ya existentes. Se
comparan las cinco covariables OHLCV del día anterior y el objetivo sigma7 del día
de resultado, sobre casos completos comunes: 350 días en 2023 y 366 en 2024. La
cuadrícula de 731 días y el hueco del 24/03/2023 permanecen presentes. No se ajusta
ningún modelo, no se abren ZIP ni archivos de test, no se cambia el preprocesamiento.

## Productos

- `causal_daily_profile.csv`: cuadrícula completa con fechas de covariables,
  resultado y máscara de caso completo, sin comprimir los días ausentes.
- `descriptive_by_period.csv`: distribución y cuantiles por variable y periodo.
- `marginal_distribution_differences.csv`: cambios de medias/medianas, razones,
  efectos estandarizados descriptivos y distancia KS. El signo es 2024 menos 2023.
- `calendar_blocks.csv`: bloques, máscaras, días válidos y decisión de permutación.
- `conditional_block_permutation.csv`: 12 diagnósticos KS por bloques y Holm.
- `summary.json`: métodos, hashes, resultados, comprobaciones y limitaciones.
- Figura: `book/_static/eda_general_shift/marginales_causales_2023_2024.png`.

## Referencia exploratoria de permutación

Los largos 14 y 28 días, 2.000 permutaciones y semilla 42 se fijan antes de esta
ejecución exploratoria. Se permutan solo bloques calendarios completos y plenamente
observados, manteniendo el número original de bloques elegibles en cada periodo.
Las colas de un día de 2023 y dos de 2024 y los bloques con máscaras quedan fijos
en su periodo original. Sus observaciones válidas siguen incluidas en cada KS.
Así, cada comparación conserva 350/366 observaciones y no mueve ni rellena máscaras.
Quedan fijos 28/350 valores de entrenamiento con bloques de 14 días (8 %) y
70/350 con bloques de 28 días (20 %); en validación, 2/366 en ambos casos. Por
ello la sensibilidad cambia también el conjunto permutable. La hipótesis
condicional no es igualdad incondicional de todas las distribuciones anuales:
las diferencias en las partes fijadas se condicionan y no se aleatorizan.

La referencia presupone intercambiabilidad aproximada de los bloques elegibles
condicionada a las partes fijas. No se ha acreditado ese supuesto ni independencia
entre bloques, en particular para niveles de precios con tendencias. Por ello los
valores p son referencias condicionales de sensibilidad, no pruebas exactas ni
confirmatorias. Holm sobre las doce comparaciones no repara una calibración
marginal inadecuada. Las comparaciones son marginales; no cubren toda distribución
conjunta ni todos los rezagos de las cuatro historias de entrada.

Los bloques del diagnóstico no fijan la unidad inferencial definitiva del benchmark.
Un p pequeño no selecciona variables, modelo, objetivo ni ventana; un p grande no
demuestra igualdad. Las diferencias descriptivas sí permiten reconocer cambios de
escala y justificar validación cronológica, cobertura de regímenes y transformaciones
aprendidas solo con entrenamiento permitido, sin consultar test.

Reproducir desde el repositorio:

```bash
/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_general_shift.py
```

Datos: Binance Vision / Binance Spot BTCUSDT. Procesamiento: Jassan Arteta.
Derivados y figura bajo CC BY-NC-SA 4.0 y condiciones aplicables de Binance Vision.
"""
    (OUT / "README.md").write_text(readme)
    for name in ("causal_daily_profile.csv", "descriptive_by_period.csv", "marginal_distribution_differences.csv", "conditional_block_permutation.csv", "calendar_blocks.csv", "summary.json", "README.md"):
        shutil.copyfile(OUT / name, ASSETS / name)
    assert source_hashes == {str(path.relative_to(ROOT)): digest(path) for path in (MARKET, TARGETS)}
    print(json.dumps({"complete_cases": summary["common_complete_case_days"], "KS": dict(zip(FEATURES, observed.tolist())),
                      "permutable_blocks": [{"days": r["block_days"], "train": r["permutable_blocks_original_train"], "validation": r["permutable_blocks_original_validation"]} for r in block_metadata],
                      "contrasts": [{k: row[k] for k in ("feature", "block_days", "conditional_p_mc", "conditional_p_holm_12")} for row in contrast_rows]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
