"""Reproduce EDA on verified Binance BTCUSDT development data, 2023–2024 only.

Outputs JSON/CSV and PNG. Never reads 2025–2026 archives or changes raw data.
Hourly diagnostics are descriptive aggregations, not a redefinition of the target.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import warnings
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-bitcoin-eda")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import signal, stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def longest_complete(series):
    valid = series.notna()
    groups = (~valid).cumsum()
    sizes = series[valid].groupby(groups[valid]).size()
    return series[valid & groups.eq(sizes.idxmax())]


def ranges(mask):
    selected = mask[mask]
    if selected.empty:
        return []
    groups = selected.index.to_series().diff().ne(pd.Timedelta(minutes=1)).cumsum()
    return [{"start_open_utc": str(g.index[0]), "end_open_utc": str(g.index[-1]),
             "minutes": len(g)} for _, g in selected.groupby(groups.to_numpy())]


def describe(series):
    s = series.dropna()
    return {"count": len(s), "mean": s.mean(), "std_ddof0": s.std(ddof=0),
            "min": s.min(), "q01": s.quantile(.01), "median": s.median(),
            "q99": s.quantile(.99), "max": s.max(),
            "skew_unbiased": s.skew(), "excess_kurtosis_unbiased": s.kurt(),
            "zero_fraction": s.eq(0).mean()}


def moving_block_mean(sample, block, rng, repeats=2000):
    n = len(sample)
    starts = rng.integers(0, n - block + 1, size=(repeats, (n + block - 1)//block))
    positions = (starts[..., None] + np.arange(block)).reshape(repeats, -1)[:, :n]
    return sample[positions].mean(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/workspace/binance-spot-data"))
    parser.add_argument("--volatility-days", type=int, nargs="+", default=[7,14,21,28])
    args = parser.parse_args()
    if any(n < 2 for n in args.volatility_days):
        parser.error("Volatility requires at least two daily returns")
    out = ROOT / "results/eda_development"
    images = ROOT / "book/_static/eda"
    out.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/manifests/binance_spot_BTCUSDT_1m_20230101_20261030.json"
    manifest = json.loads(manifest_path.read_text())
    frames, sources = [], []
    for record in manifest["archives"]:
        # Reject test files before resolving/opening their local paths.
        label = record["archive"].removeprefix("BTCUSDT-1m-").removesuffix(".zip")
        if record["kind"] != "monthly" or not ("2023-01" <= label <= "2024-12"):
            continue
        path = args.raw_root / record["relative_path"]
        with path.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == record["sha256"], path
        with zipfile.ZipFile(path) as archive:
            assert len(archive.namelist()) == 1
            with archive.open(archive.namelist()[0]) as f:
                frame = pd.read_csv(f, names=COLUMNS, header=None)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.open_time, unit="ms", utc=True))
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        assert frame.index.min().strftime("%Y-%m") == label
        assert frame.index.max().strftime("%Y-%m") == label
        frames.append(frame)
        sources.append({"archive": record["archive"], "sha256": record["sha256"]})
    assert len(frames) == 24, "Expected exactly 24 monthly development archives"
    raw = pd.concat(frames).sort_index()
    assert raw.index.max() < pd.Timestamp("2025-01-01", tz="UTC")
    assert raw.index.is_unique
    grid = pd.date_range("2023-01-01", "2025-01-01", freq="min", inclusive="left", tz="UTC")
    assert raw.index.isin(grid).all()
    assert (raw.open_time % 60000).eq(0).all()
    missing = pd.Series(~grid.isin(raw.index), index=grid)
    numeric_nonfinite = {c: int((~np.isfinite(raw[c])).sum()) for c in COLUMNS}
    positive = raw[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ohlc_ok = (raw.high.ge(raw[["open", "close", "low"]].max(axis=1)) &
               raw.low.le(raw[["open", "close", "high"]].min(axis=1)))
    close_time_ok = raw.close_time.eq(raw.open_time + 59999)
    all_finite = np.isfinite(raw[COLUMNS]).all(axis=1)
    nonnegative = raw[["volume", "quote_volume", "trades", "taker_base", "taker_quote"]].ge(0).all(axis=1)
    usable = positive & ohlc_ok & close_time_ok & all_finite & nonnegative
    close = raw.close.where(usable).reindex(grid)
    returns = np.log(close).diff()  # Full grid: never bridge missing/invalid rows.
    train = close.loc["2023"]
    train_r = returns.loc["2023"]
    val_r = returns.loc["2024"]
    hourly = close.resample("h", label="right", closed="left").agg(["last", "count"])
    hourly_close = hourly["last"].where(hourly["count"].eq(60))
    # The final candle at 2023-12-31 23:59 becomes observable at 2024-01-01 00:00.
    hourly_train = hourly_close.loc[:"2024-01-01 00:00:00+00:00"]
    hourly_r = np.log(hourly_train).diff()
    segment = longest_complete(hourly_r)
    price_segment = longest_complete(np.log(hourly_train))
    assert segment.index.max() <= pd.Timestamp("2024-01-01", tz="UTC")
    # Model frequency: one complete UTC day per observation, from original minute bars.
    daily_market = raw.resample("D").agg(open=("open","first"), high=("high","max"),
        low=("low","min"), close=("close","last"), volume=("volume","sum"),
        trades=("trades","sum"))
    day_count = usable.astype(int).resample("D").sum()
    daily_market = daily_market.reindex(pd.date_range("2023-01-01","2025-01-01",
                                       inclusive="left",freq="D",tz="UTC"))
    daily_market.loc[day_count.reindex(daily_market.index).ne(1440), :] = np.nan
    daily_market.index.name="market_day_utc"
    daily_market.to_csv(out/"daily_market_development.csv")
    daily_return = np.log(daily_market.close).diff()
    daily_train_return = daily_return.loc["2023"]
    daily_segment = longest_complete(daily_train_return)
    daily_price_segment = longest_complete(np.log(daily_market.close.loc["2023"]))
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources, "training": "2023 UTC", "validation": "2024 UTC",
        "test_archives_opened": 0, "volatility_target_executed": False,
        "grid_minutes": len(grid), "raw_rows": len(raw),
        "missing_minutes": int(missing.sum()), "missing_intervals": ranges(missing),
        "nonfinite_by_column": numeric_nonfinite,
        "nonpositive_ohlc_rows": int((~positive).sum()),
        "incoherent_ohlc_rows": int((~ohlc_ok).sum()),
        "negative_volume_or_trades_rows": int((~nonnegative).sum()),
        "unexpected_close_time_rows": int((~close_time_ok).sum()),
        "close_time_anomalies": [str(t) for t in raw.index[~close_time_ok]],
        "invalid_rows_excluded_from_analysis": int((~usable).sum()),
        "valid_close_train": int(train.notna().sum()),
        "valid_close_validation": int(close.loc["2024"].notna().sum()),
        "stats": {"train_close": describe(train), "train_return": describe(train_r),
                  "validation_return": describe(val_r),
                  "train_daily_close":describe(daily_market.close.loc["2023"]),
                  "train_daily_return":describe(daily_train_return),
                  "validation_daily_return":describe(daily_return.loc["2024"])},
        "model_frequency":"daily UTC",
        "input_days":[7,14,21,28],
        "input_raw_minutes":[10080,20160,30240,40320],
        "forecast_days":1,
        "daily_rows": {"train_complete":int(daily_market.close.loc["2023"].notna().sum()),
                       "validation_complete":int(daily_market.close.loc["2024"].notna().sum())},
        "hourly_test_segment": {"start": str(segment.index.min()),
                                "end": str(segment.index.max()), "n": len(segment)},
        "figures": [],
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    def save(fig, name):
        fig.tight_layout(rect=(0, .035, 1, 1))
        fig.text(.01, .005, "Datos: Binance Vision | Procesamiento: Jassan Arteta | CC BY-NC-SA 4.0", fontsize=7, color="#555555")
        fig.savefig(images / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        report["figures"].append(name)

    # Coverage and the full set of raw gaps. Counts do not imply a missingness mechanism.
    monthly = pd.DataFrame({"valid_close": close.notna().resample("MS").sum(),
                            "expected": close.resample("MS").size(),
                            "missing": missing.resample("MS").sum()})
    monthly.to_csv(out/"coverage_monthly.csv", index_label="month_utc")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    labels = monthly.index.strftime("%Y-%m")
    axes[0].bar(labels, 100*monthly.valid_close/monthly.expected, color="#265b96")
    axes[0].set(ylabel="Cierres elegibles (%)", ylim=(99.7, 100.02),
                title="Cobertura mensual de desarrollo; eje vertical ampliado")
    axes[0].tick_params(axis="x", rotation=60)
    for year, color in [(2023, "#265b96"), (2024, "#b25b20")]:
        series = close.loc[str(year)].isna().resample("D").sum()
        axes[1].plot(series.index, series, color=color, label=str(year))
    axes[1].set(ylabel="Minutos no elegibles por día", xlabel="Fecha UTC de apertura")
    axes[1].legend()
    save(fig, "01_calidad_cobertura.png")

    daily = close.resample("D", label="right", closed="left").agg(["last", "count"])
    daily_close = daily["last"].where(daily["count"].eq(1440))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(daily_close.index, daily_close, color="#265b96", linewidth=1)
    axes[0].set(ylabel="BTCUSDT (USDT)", title="Último cierre de días completos; fecha de disponibilidad UTC")
    # Hourly extrema preserve extreme minute returns without plotting a million points.
    q = returns.resample("h").agg(["min", "max"])
    axes[1].fill_between(q.index, q["min"]*100, q["max"]*100, color="#265b96", alpha=.8)
    axes[1].set(ylabel="Retorno de 1 min (%)", xlabel="Fecha UTC",
                title="Envolvente horaria de retornos de un minuto; no es un retorno horario")
    for ax in axes:
        ax.axvline(pd.Timestamp("2024-01-01", tz="UTC"), color="#b25b20", ls="--", label="Inicio validación")
        ax.legend()
    save(fig, "02_precios_retornos.png")

    sample = train_r.dropna()
    bounds = sample.quantile([.001, .999]).to_numpy()
    report["outlier_rule"] = {"lower_train_q001": bounds[0], "upper_train_q999": bounds[1],
                              "training_flags": int(((sample < bounds[0]) | (sample > bounds[1])).sum()),
                              "validation_flags": int(((val_r < bounds[0]) | (val_r > bounds[1])).sum())}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(sample*100, bins=180, log=True, color="#265b96")
    axes[0].set(xlabel="Retorno de un minuto (%)", ylabel="Conteo (log)", title="Train 2023: distribución completa")
    axes[1].hist(sample*100, bins=np.linspace(bounds[0]*100, bounds[1]*100, 100), color="#265b96")
    axes[1].set(xlabel="Retorno (%)", title="Zoom q0,1 %–q99,9 %; colas no eliminadas")
    ordered = np.sort(sample.to_numpy())
    axes[2].plot(ordered*100, np.arange(1, len(ordered)+1)/len(ordered), color="#265b96")
    axes[2].set(xlabel="Retorno (%)", ylabel="Probabilidad acumulada", title="ECDF completa, entrenamiento")
    save(fig, "03_distribucion_retornos.png")

    # Daily abs minute returns are a historical diagnostic, not future-volatility labels.
    daily_abs = returns.abs().resample("D").mean()
    daily_counts = returns.resample("D").count()
    daily_abs = daily_abs.where(daily_counts.eq(1440))
    train_days = daily_abs.loc["2023"].to_numpy()
    val_days = daily_abs.loc["2024"].to_numpy()
    report["drift_block_bootstrap"] = []
    for block in (7, 14):
        rng = np.random.default_rng(42)
        # Retain calendar positions including NaN; averages ignore incomplete days.
        def block_average(values):
            n = len(values)
            starts = rng.integers(0, n-block+1, size=(2000, (n+block-1)//block))
            idx = (starts[..., None]+np.arange(block)).reshape(2000, -1)[:, :n]
            return np.nanmean(values[idx], axis=1)
        delta = block_average(val_days) - block_average(train_days)
        report["drift_block_bootstrap"].append({"block_days": block, "replicates": 2000,
            "seed": 42, "difference_val_minus_train": np.nanmean(val_days)-np.nanmean(train_days),
            "ci95_percentile": np.quantile(delta, [.025,.975]).tolist()})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for year, series, color in [(2023, train_r, "#265b96"), (2024, val_r, "#b25b20")]:
        v = np.sort(series.abs().dropna().to_numpy())
        axes[0].plot(v*100, np.arange(1,len(v)+1)/len(v), color=color, label=str(year))
    axes[0].set(xlabel="|Retorno de un minuto| (%)", ylabel="ECDF", xlim=(0,.3),
                title="Desarrollo: zoom descriptivo; colas conservadas")
    axes[0].legend()
    axes[1].boxplot([daily_abs.loc["2023"].dropna()*100, daily_abs.loc["2024"].dropna()*100],
                    tick_labels=["Entrenamiento 2023", "Validación 2024"], showfliers=True)
    axes[1].set(ylabel="Media diaria de |r de 1 min| (%)", title="Días completos; no es el objetivo futuro")
    save(fig, "04_desplazamiento_desarrollo.png")

    # Stationarity: longest continuous train segment; fixed lags declared before testing.
    tests = []
    for name, series in [("log_close_hourly", price_segment), ("return_hourly", segment),
                         ("abs_return_hourly", segment.abs())]:
        values = series.to_numpy()
        adf = adfuller(values, maxlag=24, regression="c", autolag=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            k = kpss(values, regression="c", nlags="auto")
        tests.append({"series": name, "n": len(values), "start_utc": str(series.index.min()),
            "end_utc": str(series.index.max()), "adf_statistic": adf[0], "adf_p": adf[1],
            "adf_lags": adf[2], "adf_nobs": adf[3], "adf_regression": "c",
            "kpss_statistic": k[0], "kpss_p_table": k[1], "kpss_lags": k[2],
            "kpss_regression": "c", "kpss_warnings": [str(w.message) for w in caught]})
    report["stationarity"] = tests
    pd.DataFrame(tests).to_csv(out/"stationarity.csv", index=False)
    residual = segment - segment.mean()
    lm, pval, fvalue, fp = het_arch(residual, nlags=24, ddof=1)
    # Lagged polynomial mean diagnostic with HAC errors; no omnibus nonlinearity claim.
    import statsmodels.api as sm
    scaled = residual / residual.std(ddof=0)
    design = pd.DataFrame({"lag": scaled.shift(1), "lag2": scaled.shift(1)**2,
                           "lag3": scaled.shift(1)**3, "y": scaled}).dropna()
    model = sm.OLS(design.y, sm.add_constant(design[["lag","lag2","lag3"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags":24})
    nonlinear = model.wald_test("lag2 = 0, lag3 = 0", scalar=True)
    report["dependence_tests"] = {
        "arch_lm": float(lm), "arch_p": float(pval), "arch_lags": 24,
        "arch_mean_model": "constant mean hourly return, ddof=1",
        "polynomial_hac_chi2": float(nonlinear.statistic),
        "polynomial_hac_p": float(nonlinear.pvalue), "hac_lags":24,
        "polynomial_model": "standardized r_t on constant, r_(t-1), r_(t-1)^2, r_(t-1)^3",
        "n_polynomial": len(design), "multiplicity": "exploratory; no correction"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_acf(segment, lags=72, alpha=.05, ax=axes[0,0], zero=False)
    axes[0,0].set(title="ACF: retornos horarios train", xlabel="Rezago (horas)", ylim=(-.12,.12))
    plot_pacf(segment, lags=72, alpha=.05, method="ywm", ax=axes[0,1], zero=False)
    axes[0,1].set(title="PACF: retornos horarios train", xlabel="Rezago (horas)", ylim=(-.12,.12))
    plot_acf(segment**2, lags=72, alpha=.05, ax=axes[1,0], zero=False)
    axes[1,0].set(title="ACF: retornos horarios al cuadrado", xlabel="Rezago (horas)", ylim=(-.03,.15))
    freq, power = signal.periodogram(segment, fs=1, detrend="constant")
    axes[1,1].semilogy(freq[1:]*24, power[1:], color="#265b96")
    axes[1,1].axvline(1, color="#b25b20", ls="--", label="1 ciclo/día")
    axes[1,1].set(xlabel="Ciclos por día", ylabel="Densidad espectral", title="Periodograma de retornos horarios")
    axes[1,1].legend()
    save(fig, "05_dependencia_temporal.png")

    # Predeclared visualization window: last 60 train days, complete hourly log closes.
    stl_input = hourly_train.iloc[-1440:]
    assert stl_input.notna().all() and len(stl_input)==1440
    decomposition = STL(np.log(stl_input), period=24, robust=True).fit()
    report["stl"] = {"period_hours":24, "n":len(stl_input), "robust":True,
                     "start_utc":str(stl_input.index.min()), "end_utc":str(stl_input.index.max()),
                     "role":"descriptive only, not causal features or target selection"}
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, label, values in zip(axes, ["Log-cierre", "Tendencia", "Estacional (24 h)", "Residuo"],
                                 [decomposition.observed, decomposition.trend,
                                  decomposition.seasonal, decomposition.resid]):
        ax.plot(stl_input.index, values, color="#265b96", linewidth=.9)
        ax.set_ylabel(label)
    axes[0].set_title("STL descriptivo: últimos 60 días de entrenamiento; periodo diario exploratorio")
    axes[-1].set_xlabel("Fecha de disponibilidad UTC")
    save(fig, "06_descomposicion_stl.png")

    abs_r = segment.abs()
    cusum = (abs_r-abs_r.mean()).cumsum()/(abs_r.std(ddof=0)*np.sqrt(len(abs_r)))
    boundary = cusum.abs().iloc[168:-168].idxmax()
    report["regime_diagnostic"] = {"candidate_utc": str(boundary),
        "method": "max absolute standardized CUSUM of |hourly train return|; exclude 168h edges",
        "significance_test": False, "value":float(cusum.loc[boundary])}
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(abs_r.index, abs_r.rolling(168,min_periods=168).mean()*100, color="#265b96")
    axes[0].set(ylabel="Media móvil |r| (%)", title="Variabilidad histórica: ventana causal de 168 horas")
    axes[1].plot(cusum.index, cusum, color="#265b96")
    axes[1].axvline(boundary, color="#b25b20", ls="--", label="Candidato exploratorio; sin significación")
    axes[1].set(ylabel="CUSUM estandarizado", xlabel="Fecha UTC")
    axes[1].legend()
    save(fig, "07_regimenes.png")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    stages=[("Entrenamiento inicial", "2023-01-01", "2024-01-01", "#265b96"),
            ("Validación forward chaining", "2024-01-01", "2025-01-01", "#b25b20"),
            ("Test previsto: fuera del EDA", "2025-01-01", "2026-09-01", "#888888"),
            ("Reserva sin asignar", "2026-09-01", "2026-10-31", "#cccccc")]
    for row, (label,start,end,color) in enumerate(stages):
        lo=mdates.date2num(pd.Timestamp(start)); hi=mdates.date2num(pd.Timestamp(end))
        ax.barh(row,hi-lo,left=lo,color=color)
    ax.set_yticks(range(4),[x[0] for x in stages]);ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.YearLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set(xlabel="Calendario UTC; cortes exactos en el texto", title="Esquema temporal propuesto: sin barajar ni inspeccionar test")
    save(fig, "08_particion_temporal.png")

    # Additional original-frequency diagnostics. Window fixed to the last 30 train days.
    minute_close = train.iloc[-30*1440:]
    minute_returns = train_r.iloc[-30*1440:]
    assert minute_close.notna().all() and minute_returns.notna().all()
    minute_tests = []
    for name, values in [("log_close_1m", np.log(minute_close).to_numpy()),
                         ("return_1m", minute_returns.to_numpy()),
                         ("abs_return_1m", minute_returns.abs().to_numpy())]:
        adf = adfuller(values, maxlag=60, regression="c", autolag=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            k = kpss(values, regression="c", nlags="auto")
        minute_tests.append({"series": name, "n":len(values),
            "start_open_utc":str(minute_close.index.min()),
            "end_open_utc":str(minute_close.index.max()),
            "adf_statistic":adf[0],"adf_p":adf[1],"adf_lags":adf[2],
            "adf_nobs":adf[3],"adf_regression":"c",
            "kpss_statistic":k[0],"kpss_p_table":k[1],"kpss_lags":k[2],
            "kpss_regression":"c","kpss_warnings":[str(w.message) for w in caught]})
    report["stationarity_1m"] = minute_tests
    pd.DataFrame(minute_tests).to_csv(out/"stationarity_1m.csv", index=False)
    minute_segment = longest_complete(train_r)
    coeff, ci = acf(minute_segment**2, nlags=28*1440, alpha=.05, fft=True)
    report["minute_dependence"] = {"acf_squared_n":len(minute_segment),
        "acf_squared_start_open_utc":str(minute_segment.index.min()),
        "acf_squared_end_open_utc":str(minute_segment.index.max()),
        "pacf_window_minutes":len(minute_returns),"bands":"approximate 95 percent"}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    plot_acf(minute_returns,lags=120,alpha=.05,ax=axes[0,0],zero=False)
    plot_pacf(minute_returns,lags=120,alpha=.05,method="ywm",ax=axes[0,1],zero=False)
    axes[0,0].set(title="ACF: retornos 1 min, últimos 30 días train",xlabel="Rezago (minutos)",ylim=(-.12,.12))
    axes[0,1].set(title="PACF: retornos 1 min, últimos 30 días train",xlabel="Rezago (minutos)",ylim=(-.12,.12))
    selection=np.arange(1,len(coeff),60)
    axes[1,0].plot(selection/1440,coeff[selection],color="#265b96",lw=.8)
    axes[1,0].fill_between(selection/1440,ci[selection,0]-coeff[selection],
                            ci[selection,1]-coeff[selection],color="#265b96",alpha=.15)
    axes[1,0].set(title="ACF de r² de 1 min: tramo continuo de train",xlabel="Rezago (días; curva cada 60 min)",ylabel="Correlación")
    f,power=signal.periodogram(minute_returns,fs=1,detrend="constant")
    axes[1,1].semilogy(f[1:]*1440,power[1:],color="#265b96",lw=.6)
    axes[1,1].set(title="Periodograma de retornos de 1 min",xlabel="Ciclos/día",ylabel="Densidad espectral")
    save(fig,"09_dependencia_un_minuto.png")

    daily_tests=[]
    for name,series in [("log_close_daily",daily_price_segment),("return_daily",daily_segment),
                        ("abs_return_daily",daily_segment.abs())]:
        v=series.to_numpy()
        adf=adfuller(v,maxlag=7,regression="c",autolag=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            k=kpss(v,regression="c",nlags="auto")
        daily_tests.append({"series":name,"n":len(v),"start_day_utc":str(series.index.min()),
            "end_day_utc":str(series.index.max()),"adf_statistic":adf[0],"adf_p":adf[1],
            "adf_lags":adf[2],"adf_nobs":adf[3],"adf_regression":"c",
            "kpss_statistic":k[0],"kpss_p_table":k[1],"kpss_lags":k[2],"kpss_regression":"c",
            "kpss_warnings":[str(w.message) for w in caught]})
    report["stationarity_daily"]=daily_tests
    pd.DataFrame(daily_tests).to_csv(out/"stationarity_daily.csv",index=False)
    dr=daily_segment-daily_segment.mean()
    lm,pval,_,_=het_arch(dr,nlags=7,ddof=1)
    scaled=dr/dr.std(ddof=0)
    design=pd.DataFrame({"lag":scaled.shift(1),"lag2":scaled.shift(1)**2,
                         "lag3":scaled.shift(1)**3,"y":scaled}).dropna()
    model=sm.OLS(design.y,sm.add_constant(design[["lag","lag2","lag3"]])).fit(
        cov_type="HAC",cov_kwds={"maxlags":7})
    w=model.wald_test("lag2 = 0, lag3 = 0",scalar=True)
    report["daily_dependence_tests"]={"n":len(dr),"arch_lm":float(lm),"arch_p":float(pval),
        "arch_lags":7,"mean_model":"constant", "polynomial_hac_chi2":float(w.statistic),
        "polynomial_hac_p":float(w.pvalue),"hac_lags":7,"multiplicity":"exploratory, no correction"}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    plot_acf(daily_segment,lags=28,alpha=.05,ax=axes[0,0],zero=False)
    plot_pacf(daily_segment,lags=28,alpha=.05,method="ywm",ax=axes[0,1],zero=False)
    plot_acf(daily_segment**2,lags=28,alpha=.05,ax=axes[1,0],zero=False)
    axes[0,0].set(title="ACF de retornos diarios: entrenamiento",xlabel="Rezago (días)")
    axes[0,1].set(title="PACF de retornos diarios: entrenamiento",xlabel="Rezago (días)")
    axes[1,0].set(title="ACF de retornos diarios al cuadrado",xlabel="Rezago (días)")
    f,power=signal.periodogram(daily_segment,fs=1,detrend="constant")
    axes[1,1].semilogy(f[1:],power[1:],color="#265b96")
    axes[1,1].axvline(1/7,color="#b25b20",ls="--",label="Referencia: 1 ciclo/semana")
    axes[1,1].set(title="Periodograma de retornos diarios",xlabel="Ciclos por día",ylabel="Densidad espectral")
    axes[1,1].legend()
    save(fig,"10_dependencia_diaria.png")

    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    axes[0].plot(daily_market.index,daily_market.close,color="#265b96")
    axes[0].set(ylabel="Cierre BTCUSDT (USDT)",title="Observación de modelado: OHLCV diario UTC")
    axes[1].plot(daily_return.index,daily_return*100,color="#265b96",lw=.7)
    axes[1].set(ylabel="Retorno diario (%)")
    axes[2].plot(daily_market.index,daily_market.volume,color="#265b96",lw=.7)
    axes[2].set(ylabel="Volumen diario (BTC)",xlabel="Día de mercado UTC; disponible al finalizar el día")
    for ax in axes: ax.axvline(pd.Timestamp("2024-01-01",tz="UTC"),color="#b25b20",ls="--")
    save(fig,"11_modelado_diario.png")

    daily_stl=STL(daily_price_segment,period=7,robust=True).fit()
    report["stl_daily"]={"period_days":7,"n":len(daily_price_segment),
        "start_day_utc":str(daily_price_segment.index.min()),"end_day_utc":str(daily_price_segment.index.max()),
        "role":"exploratory weekly decomposition; no significant seasonality asserted"}
    fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for ax,label,values in zip(axes,["Log-cierre diario","Tendencia","Estacional (7 días)","Residuo"],
        [daily_stl.observed,daily_stl.trend,daily_stl.seasonal,daily_stl.resid]):
        ax.plot(daily_price_segment.index,values,color="#265b96",lw=.9);ax.set_ylabel(label)
    axes[0].set_title("STL diario descriptivo: tramo continuo de entrenamiento, periodo semanal")
    axes[-1].set_xlabel("Día de mercado UTC")
    save(fig,"12_descomposicion_diaria.png")

    report["daily_volatility_candidates"]={}
    candidate_sigmas={}
    for n in args.volatility_days:
        sigma=daily_return.rolling(n,min_periods=n).std(ddof=0)
        # Value for market day d is available after its close. To forecast it, only prior days are inputs.
        report["volatility_target_executed"]=True
        report["daily_volatility_candidates"][str(n)]={"window_days":n,"ddof":0,"forecast_days":1,
            "definition":"population standard deviation of last n daily log returns, ending on outcome market day",
            "stats":{str(y):describe(sigma.loc[str(y)]) for y in [2023,2024]},
            "scale":"raw log-return units, no annualization"}
        candidate_sigmas[f"sigma_{n}"]=sigma
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for y,color in [(2023,"#265b96"),(2024,"#b25b20")]:
            values=sigma.loc[str(y)].dropna()
            axes[0,0].plot(values.index,values*100,color=color,label=str(y))
            axes[0,1].hist(values*100,bins=40,histtype="step",density=True,color=color,label=str(y))
            ordered=np.sort(values)
            axes[1,0].plot(ordered*100,np.arange(1,len(ordered)+1)/len(ordered),color=color,label=str(y))
        axes[0,0].set(title=f"Volatilidad histórica diaria, n={n}",ylabel="Desviación estándar (%)")
        axes[0,1].set(title="Distribución del objetivo",xlabel="Desviación estándar (%)",ylabel="Densidad")
        axes[1,0].set(title="ECDF",xlabel="Desviación estándar (%)",ylabel="Probabilidad acumulada")
        axes[1,1].boxplot([sigma.loc["2023"].dropna()*100,sigma.loc["2024"].dropna()*100],tick_labels=["2023","2024"])
        axes[1,1].set(title="Comparación descriptiva de desarrollo",ylabel="Desviación estándar (%)")
        for ax in axes.flat:
            if ax is not axes[1,1]:ax.legend()
        save(fig,f"13_volatilidad_{n}_dias.png")

    sigmas=pd.DataFrame(candidate_sigmas)
    sigmas.to_csv(out/"daily_volatility_development.csv")
    # Predictors are previous-day OHLCV; target is volatility on the next outcome day.
    aligned=daily_market[["open","high","low","close","volume"]].shift(1).join(sigmas)
    train_aligned=aligned.loc["2023"].dropna()
    report["correlation_alignment"]={"predictors":"OHLCV of previous UTC day",
        "targets":"volatility ending on outcome day, w=7/14/21/28",
        "train_complete_common_days":len(train_aligned),"method":"descriptive, not causal"}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,method in zip(axes,["pearson","spearman"]):
        matrix=train_aligned.corr(method=method).loc[["open","high","low","close","volume"],list(sigmas.columns)]
        matrix.to_csv(out/f"correlation_{method}.csv")
        im=ax.imshow(matrix,vmin=-1,vmax=1,cmap="RdBu_r",aspect="auto")
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=35)
        ax.set_yticks(range(5),["Apertura t−1","Máximo t−1","Mínimo t−1","Cierre t−1","Volumen t−1"])
        ax.set_title(f"{method.title()}: train; siguiente día de resultado")
        for i in range(5):
            for j in range(len(matrix.columns)):
                ax.text(j,i,f"{matrix.iloc[i,j]:.2f}",ha="center",va="center",fontsize=10)
        fig.colorbar(im,ax=ax,fraction=.04)
    save(fig,"14_correlaciones.png")
    columns=["open","high","low","close","volume"]
    tr=daily_market.loc["2023",columns].dropna()
    va=daily_market.loc["2024",columns].dropna()
    mean=tr.mean();std=tr.std(ddof=0)
    _,singular,vt=np.linalg.svd(((tr-mean)/std).to_numpy(),full_matrices=False)
    variance=singular**2/(singular**2).sum()
    pca_train=((tr-mean)/std).to_numpy()@vt.T
    pca_val=((va-mean)/std).to_numpy()@vt.T
    report["pca"]={"variables":columns,"fit_period":"2023 only",
        "train_rows":len(tr),"validation_rows":len(va),"explained_ratio":variance.tolist(),
        "means":mean.to_dict(),"std_ddof0":std.to_dict(),"components":vt.tolist(),
        "role":"exploratory only, no PCA transform selected for the model"}
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(np.arange(1,6),variance*100,color="#265b96")
    axes[0].set(xlabel="Componente",ylabel="Varianza explicada (%)",title="PCA de OHLCV; ajuste exclusivamente en 2023")
    axes[1].scatter(pca_train[:,0],pca_train[:,1],s=10,alpha=.5,label="Train 2023")
    axes[1].scatter(pca_val[:,0],pca_val[:,1],s=10,alpha=.5,label="Validación 2024")
    axes[1].set(xlabel="PC1",ylabel="PC2",title="Proyección con parámetros congelados de train")
    axes[1].legend()
    save(fig,"15_pca.png")

    dabs=daily_segment.abs()
    dcusum=(dabs-dabs.mean()).cumsum()/(dabs.std(ddof=0)*np.sqrt(len(dabs)))
    candidate=dcusum.abs().iloc[28:-28].idxmax()
    lo,hi=daily_train_return.quantile([.01,.99])
    report["daily_regime"]={"candidate_day":str(candidate),"cusum":float(dcusum.loc[candidate]),
        "method":"max absolute CUSUM of absolute daily return, 28-day boundary exclusion",
        "significance_test":False}
    report["daily_outliers"]={"train_q01":float(lo),"train_q99":float(hi),
        "train_flags":int(((daily_train_return<lo)|(daily_train_return>hi)).sum()),
        "validation_flags":int(((daily_return.loc["2024"]<lo)|(daily_return.loc["2024"]>hi)).sum()),
        "action":"flag only, no clipping or deletion"}
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(daily_train_return.index,daily_train_return*100,color="#265b96",lw=.8)
    flagged=daily_train_return[(daily_train_return<lo)|(daily_train_return>hi)]
    axes[0].scatter(flagged.index,flagged*100,color="#b25b20",label="Fuera de q1 %–q99 % de train")
    axes[0].set(ylabel="Retorno diario (%)",title="Extremos marcados, conservados en los datos")
    axes[0].legend()
    axes[1].plot(dcusum.index,dcusum,color="#265b96")
    axes[1].axvline(candidate,color="#b25b20",ls="--",label="Candidato exploratorio, no ruptura confirmada")
    axes[1].set(ylabel="CUSUM de |r diario|",xlabel="Día UTC del entrenamiento")
    axes[1].legend()
    save(fig,"16_anomalias_regimen_diario.png")
    vol_tests=[]
    for name in sigmas.columns:
        values=longest_complete(sigmas.loc["2023",name])
        a=adfuller(values,maxlag=7,regression="c",autolag=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            k=kpss(values,regression="c",nlags="auto")
        vol_tests.append({"series":name,"n":len(values),"adf_p":a[1],"adf_lags":a[2],
            "kpss_p_table":k[1],"kpss_lags":k[2],
            "start_day_utc":str(values.index.min()),"end_day_utc":str(values.index.max()),
            "kpss_warnings":[str(w.message) for w in caught]})
    report["stationarity_volatility"]=vol_tests
    pd.DataFrame(vol_tests).to_csv(out/"stationarity_volatility.csv",index=False)

    report=finite_json(report)
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(report["stats"]).T.to_csv(out/"descriptive_statistics.csv",index_label="series")
    print(json.dumps({k:report[k] for k in ["raw_rows","missing_minutes","unexpected_close_time_rows",
        "valid_close_train","valid_close_validation","test_archives_opened","volatility_target_executed","figures"]},indent=2))


if __name__=="__main__":
    main()
