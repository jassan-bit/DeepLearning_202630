"""Post-hoc descriptive E2 profiles of the already evaluated E1 calendar.

No model is trained or rescored. The original E1 cache is read-only; outputs
contain aggregate distributions, not individual test prices or predictions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-e2-known-shift")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/cache-e2-known-shift")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import kurtosis, skew, wasserstein_distance

from dl_data import FEATURES, make_windows
from dl_data_v2 import relative_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e2_known_shift_20261105"
ASSETS = ROOT / "book_e2/_static/eda_known_shift"
EXPECTED_KNOWN_HASH = "93ec9490137eee80581f56f987067001d346a421235962da5b2e9645970adebf"
DEVELOPMENT = ROOT / "results/eda_development/daily_market_development.csv"
RECORDED_TARGET = ROOT / "results/eda_development/daily_volatility_development.csv"
E1_AUDIT = ROOT / "results/gru_lstm_tcn_planned_20261105/data_quality_audit.json"
MANIFEST = ROOT / "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json"
PERIODS = {
    "train_2023": ("2023-01-01", "2023-12-31"),
    "validation_2024_Jan_Oct": ("2024-01-01", "2024-10-31"),
    "pilot_2024_Nov_Dec30": ("2024-11-01", "2024-12-30"),
    "known_reserve_2026_Sep_Oct30": ("2026-09-01", "2026-10-30"),
}
CREDIT = "Binance Vision / BTCUSDT · Jassan Arteta · CC BY-NC-SA 4.0"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(frame, filename):
    for directory in (OUT, ASSETS):
        frame.to_csv(directory / filename, index=False, lineterminator="\n")


def save_json(value, filename):
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    for directory in (OUT, ASSETS):
        (directory / filename).write_text(payload)


def read_market(path, expected_calendar):
    # Built-in float conversion preserves the binary values written by E1.
    frame = pd.read_csv(path, converters={x: lambda v: float(v) if v else np.nan for x in FEATURES})
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.pop("market_day_utc"), utc=True))
    if not frame.index.equals(expected_calendar):
        raise ValueError("Unexpected or compressed market calendar")
    return frame.loc[:, list(FEATURES)]


def known_market(args):
    if args.known_market_csv:
        path = Path(args.known_market_csv).resolve()
        if sha(path) != EXPECTED_KNOWN_HASH:
            raise ValueError("Existing E1 derivative does not match its published audit")
        return path, "existing_already_evaluated_e1_cache", []
    if not args.rebuild_cache:
        raise ValueError("--raw-root requires --rebuild-cache pointing to a NEW private directory")
    destination = Path(args.rebuild_cache).resolve()
    if destination.exists():
        raise ValueError("The reconstruction cache must be new; originals are never overwritten")
    if destination.is_relative_to(ROOT):
        raise ValueError("Keep the reconstructed individual-price cache outside the repository")
    from planned_study_data import _aggregate_payload, _safe_path, reserve_metadata
    reservation = reserve_metadata(ROOT)
    frames, checks = [], []
    # This is explicitly a reanalysis of a known E1 test, not a first-access gate.
    for record in [reservation["context_archive"], *reservation["archives"]]:
        payload = _safe_path(args.raw_root, record["relative_path"]).read_bytes()
        daily, quality = _aggregate_payload(payload, record)
        frames.append(daily)
        checks.append(quality)
    destination.mkdir(parents=True)
    path = destination / "daily_market_reserved.csv"
    pd.concat(frames).to_csv(path, mode="x")
    (destination / "known_test_reanalysis.json").write_text(json.dumps({
        "status": "Already evaluated E1 historical reserve; post-hoc E2 descriptive reanalysis",
        "source_manifest_sha256": sha(MANIFEST), "archives_verified": checks,
        "models_trained": 0, "predictions_rescored": 0, "daily_csv_sha256": sha(path),
    }, indent=2) + "\n")
    return path, "known_test_reconstructed_from_pinned_raw_checksums", checks


def diagnostic_frame(market):
    raw = market.to_numpy(dtype=np.float64)
    returns = np.log(market["close"]).diff()
    target = returns.rolling(7, min_periods=7).std(ddof=0)
    windows = make_windows(raw, 7)
    valid = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(target.to_numpy())
    frame = pd.DataFrame(index=market.index)
    frame["sigma_7"] = target
    for j, feature in enumerate(FEATURES):
        frame[f"raw_lag1_{feature}"] = market[feature].shift(1)
    relative = np.full_like(windows, np.nan)
    relative[valid] = relative_features(windows[valid])
    for step in range(7):
        lag = 7 - step
        for j, feature in enumerate(FEATURES):
            frame[f"relative_lag{lag}_{feature}"] = relative[:, step, j]
    # Every quantity is evaluated on the same D7 eligibility cohort per period.
    frame.loc[~valid] = np.nan
    return frame


def ks_distance(a, b):
    grid = np.unique(np.concatenate((a, b)))
    return float(np.max(np.abs(np.searchsorted(np.sort(a), grid, side="right") / len(a)
                               - np.searchsorted(np.sort(b), grid, side="right") / len(b))))


def representation(feature):
    return "target" if feature == "sigma_7" else ("raw_lag1" if feature.startswith("raw_") else "relative_D7")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--known-market-csv", type=Path)
    source.add_argument("--raw-root", type=Path)
    parser.add_argument("--rebuild-cache", type=Path)
    args = parser.parse_args()
    known_path, source_mode, raw_checks = known_market(args)
    if json.loads(E1_AUDIT.read_text())["daily_market_sha256"] != EXPECTED_KNOWN_HASH:
        raise ValueError("The published E1 audit identity changed")
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in (DEVELOPMENT, RECORDED_TARGET, E1_AUDIT, MANIFEST)}
    source_hashes["already_evaluated_E1_daily_market_reserved.csv"] = sha(known_path)
    dev = read_market(DEVELOPMENT, pd.date_range("2023-01-01", "2024-12-31", tz="UTC"))
    known = read_market(known_path, pd.date_range("2026-08-01", "2026-10-30", tz="UTC"))
    frame = pd.concat((diagnostic_frame(dev), diagnostic_frame(known)))
    recorded = pd.read_csv(RECORDED_TARGET, index_col="market_day_utc")
    recorded.index = pd.DatetimeIndex(pd.to_datetime(recorded.index, utc=True))
    assert np.allclose(frame.loc[dev.index, "sigma_7"], recorded["sigma_7"], equal_nan=True, atol=1e-13, rtol=1e-9)
    profiles, distances = [], []
    # Initial admitted D7 training outcomes end Dec24; these fixed reference
    # scales are descriptive only, never model scalers learned from later data.
    reference = frame.loc["2023-01-01":"2023-12-24"].dropna()
    assert len(reference) == 343
    for period, (start, finish) in PERIODS.items():
        part = frame.loc[start:finish]
        for feature in frame.columns:
            values = part[feature].dropna().to_numpy()
            train = reference[feature].to_numpy()
            train_sd = float(np.std(train, ddof=0))
            qs = np.quantile(values, [.05, .25, .5, .75, .95])
            constant = bool(np.ptp(values) == 0)
            profiles.append({"period": period, "feature": feature, "representation": representation(feature),
                "calendar_days": len(part), "observed_eligible_days": len(values), "coverage": len(values)/len(part),
                "mean": float(np.mean(values)), "sd_ddof0": float(np.std(values, ddof=0)),
                "q05": float(qs[0]), "q25": float(qs[1]), "median": float(qs[2]),
                "q75": float(qs[3]), "q95": float(qs[4]), "min": float(np.min(values)), "max": float(np.max(values)),
                "skew_unbiased": None if constant else float(skew(values, bias=False)),
                "excess_kurtosis_unbiased": None if constant else float(kurtosis(values, fisher=True, bias=False))})
            distances.append({"period": period, "feature": feature, "representation": representation(feature),
                "reference": "Admitted initial D7 training outcomes 2023-01-01 through 2023-12-24",
                "reference_n": len(train), "n": len(values), "ks_distance": ks_distance(train, values),
                "wasserstein": float(wasserstein_distance(train, values)),
                "wasserstein_over_train_sd": float(wasserstein_distance(train, values)/train_sd) if train_sd > 0 else None,
                "reference_constant": train_sd == 0,
                "inferential_p_value": None, "scope": "Post-hoc descriptive known-calendar comparison; no retuning"})
    profile = pd.DataFrame(profiles)
    distance = pd.DataFrame(distances)
    for directory in (OUT, ASSETS):
        directory.mkdir(parents=True, exist_ok=True)
    save_csv(profile, "aggregated_profile.csv")
    save_csv(distance, "distribution_distances.csv")
    summary = {"schema_version": 1, "analysis": "Known E1 calendar post-hoc distribution profiles for E2 EDA",
        "status": "Exploratory and descriptive; the 2026 reserve was already evaluated in E1 before this analysis",
        "periods": PERIODS, "normalization_reference": "343 eligible initial D7 training outcomes through 2023-12-24",
        "normalization_is_model_preprocessing": False, "features": 41,
        "reference_constant_features": distance.loc[distance.reference_constant, "feature"].unique().tolist(),
        "target_profiles": profile.loc[profile.feature == "sigma_7"].replace({np.nan: None}).to_dict("records"),
        "target_distances": distance.loc[distance.feature == "sigma_7"].replace({np.nan: None}).to_dict("records"),
        "models_trained": 0, "predictions_rescored": 0, "test_used_to_fit_transforms": False,
        "raw_archives_opened_in_this_execution": len(raw_checks), "input_source_mode": source_mode,
        "no_sealed_new_test_claim": True, "unchanged_contract": "D7, sigma7, h1, metric, margin and E1 partitions retained",
        "limits": ["Single series and one provider", "Different regime sample sizes", "Overlapping windows and targets",
                   "Distances describe marginal distributions, not causal drift or a calibrated significance test",
                   "A lower marginal shift after representation does not establish better forecasts"]}
    save_json(summary, "summary.json")
    labels = ["2023", "Validación ene–oct 2024", "Piloto nov–dic 2024", "Reserva conocida sep–oct 2026"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (period, (start, finish)), label in zip(PERIODS.items(), labels):
        values = np.sort(frame.loc[start:finish, "sigma_7"].dropna().to_numpy())
        axes[0].step(100*values, np.arange(1, len(values)+1)/len(values), where="post", label=f"{label} (n={len(values)})")
    axes[0].set(xlabel="σ₇ diaria (%)", ylabel="Distribución acumulada empírica", title="Objetivo; periodos heredados")
    axes[0].legend(fontsize=8)
    cols = ["raw_lag1_close", "raw_lag1_volume", "relative_lag7_close", "relative_lag1_volume", "sigma_7"]
    matrix = distance.pivot(index="feature", columns="period", values="wasserstein_over_train_sd").reindex(index=cols, columns=list(PERIODS))
    image = axes[1].imshow(matrix.to_numpy(), aspect="auto", cmap="Blues")
    axes[1].set_yticks(range(len(cols)), ["Cierre d−1 (nivel)", "Volumen d−1 (nivel)", "Cierre d−7 (relativo)", "log(1+V d−1)", "σ₇"])
    axes[1].set_xticks(range(4), ["2023", "Val. 2024", "Piloto", "Reserva\nconocida"], fontsize=8)
    axes[1].set_title("Wasserstein / DE inicial de entrenamiento")
    for y in range(len(cols)):
        for x in range(4):axes[1].text(x, y, f"{matrix.iloc[y,x]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[1], fraction=.035)
    fig.suptitle("Reanálisis descriptivo posterior al test de E1; no selección ni prueba confirmatoria", fontsize=11)
    fig.tight_layout(rect=(0, .065, 1, .94));fig.text(.01, .015, CREDIT, fontsize=8)
    for suffix in ("png", "svg"):
        name = f"known_distribution_shift.{suffix}"
        fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
        (ASSETS / name).write_bytes((OUT / name).read_bytes())
    plt.close(fig)
    readme = """# Distribuciones de periodos conocidos de E1 · EDA E2

Análisis descriptivo posterior a los resultados de E1. No entrena ni puntúa
modelos, no cambia fechas, inputs, objetivo, métricas o margen. Se publican solo
agregados y figuras. No incluye vectores de precios ni predicciones individuales.

Desde la raíz del repositorio, Python 3.12 con numpy, pandas, scipy y matplotlib:

```bash
python scripts/eda_e2_known_shift.py --known-market-csv /ruta/daily_market_reserved.csv
```

La opción anterior exige el SHA-256 del derivado publicado en el audit de E1.
Para reconstruir desde la adquisición original sin entrenar ningún modelo:

```bash
python scripts/eda_e2_known_shift.py --raw-root /ruta/adquisicion/oficial --rebuild-cache /tmp/e2_known_market_nuevo
```

El directorio privado de caché debe ser nuevo y estar fuera del repositorio.
El script verifica los 32 ZIP ya conocidos (agosto como contexto, septiembre
mensual y treinta diarios de octubre) contra el manifiesto fijado de E1 y usa
la rutina de agregación existente. No llama al ejecutor del benchmark ni a su
control de primer acceso. No modifica los originales. Diferencias de versión
en la serialización del CSV pueden cambiar el hash derivado reconstruido; se
registran el hash real y los hashes originales de cada ZIP.

La referencia de distancias es el entrenamiento inicial D7 admisible hasta
24/12/2023 (343 ejemplos). Las tablas de periodos presentan su calendario entero,
incluido 2023 (350 ejemplos); esto NO ajusta escaladores de modelos en validación.
KS es distancia, sin p-valor iid; Wasserstein se divide por DE poblacional de
la referencia. El componente de último cierre relativo es constante cero y no
se divide por su desviación cero. No hay afirmación de nuevos datos sellados,
significación estadística, utilidad financiera o mejor desempeño de un modelo.

Fuente: Binance Vision / BTCUSDT. Procesamiento: Jassan Arteta. CC BY-NC-SA 4.0
y condiciones adicionales de la fuente documentadas en la Data Card de E2.
"""
    for directory in (OUT, ASSETS):(directory / "README.md").write_text(readme)
    output_hashes = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "provenance.json"}
    provenance = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "administrative_timezone": "America/Bogota", "source_commit_e1": "94a4b0324e93361e8928d708fdfeb50110e3686f",
        "input_hashes": source_hashes, "script_sha256": sha(Path(__file__)), "output_hashes": output_hashes,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "source_access": "Existing already evaluated private E1 daily derivative was read without modification",
        "no_new_confirmatory_test": True, "individual_prices_redistributed": False,
        "models_trained": 0, "predictions_rescored": 0}
    if raw_checks:provenance["source_access"] = "Known historical E1 observations reconstructed from pinned ZIPs in a new private directory"
    save_json(provenance, "provenance.json")
    for path in (DEVELOPMENT, RECORDED_TARGET, E1_AUDIT, MANIFEST):
        assert source_hashes[str(path.relative_to(ROOT))] == sha(path)
    assert source_hashes["already_evaluated_E1_daily_market_reserved.csv"] == sha(known_path)
    print(json.dumps({"target_profiles": summary["target_profiles"], "target_distances": summary["target_distances"],
                      "known_source_hash": sha(known_path), "raw_archives_opened": len(raw_checks)}, indent=2))


if __name__ == "__main__":
    main()
