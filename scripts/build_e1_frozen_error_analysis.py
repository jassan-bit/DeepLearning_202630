"""Add descriptive error diagnostics to a closed E1 run without fitting or scoring.

Inputs are the already closed forecast panel and its completion receipts. The only
market-derived table opened separately is the registered *development* sigma7
table, used to define common descriptive ranges. No raw archive, checkpoint or
scaler is opened. All original public and scientific-source hashes are checked
before and after. Public output contains segment aggregates, never daily values.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STUDY = "e1_608_remediation_20261107"
SUPPLEMENT = "e1_frozen_error_analysis_20261108"
MODELS = ("GRU", "LSTM", "MLP")
ORDER = (*MODELS, "Persistence")
METRICS = ("rmse", "mae", "mse", "r2", "mape", "mase")
ERROR_METRICS = (*METRICS, "residual_bias", "residual_sd", "absolute_error_p90")
DEVELOPMENT = "results/eda_development/daily_volatility_development.csv"
COLORS = {"GRU": "#2266aa", "LSTM": "#d97706", "MLP": "#008a67", "Persistence": "#7c3ea2"}
LABELS = {"GRU": "GRU32", "LSTM": "LSTM32", "MLP": "MLP16", "Persistence": "Persistencia"}
PUBLIC_FILES = frozenset({
    "README.md", "summary.json", "source_manifest.json", "validation.json", "thresholds.json",
    "segment_run_metrics.csv", "segment_summary.csv", "monthly_forecast_summary.csv",
    "target_range_forecast_summary.csv", "residual_structure.csv", "variance_decomposition.json",
    "observed_forecast_monthly.png", "observed_forecast_by_target_range.png",
    "errors_by_regime_and_year.png", "residual_bias_dispersion.png",
    "observed_forecast_density.csv", "observed_forecast_density.png",
})
SCOPE = ("Post-hoc descriptive analysis of existing fixed predictions on the already observed608-date "
         "historical cohort; no refit, recipe selection, replacement seeds, new forecasts or new confirmatory tests")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def checked_file(root, relative, expected):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular file inside the declared boundary: " + relative)
    if sha(path) != expected:
        raise ValueError("A closed input changed: " + relative)
    return path


def verify_public_stage(directory):
    directory = Path(directory)
    manifest = read(directory / "manifest.json")
    if (manifest["publication_schema"] != "e1_public_metadata_allowlist_v1"
            or set(manifest["approved_public_files"]) != set(manifest["files"])):
        raise ValueError("Unexpected original public manifest")
    result = {}
    for name, row in manifest["files"].items():
        path = checked_file(directory, name, row["sha256"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Original public metadata length changed")
        result[name] = row["sha256"]
    result["manifest.json"] = sha(directory / "manifest.json")
    return result


def validate_panel(frame, seeds, dates, execution_id, source_signature):
    frame = frame.copy()
    required = {"model", "seed", "outcome_day", "y_true", "y_pred", "persistence_pred", "mase_scale",
                "study_id", "execution_id", "source_signature"}
    if not required <= set(frame) or set(frame.model) != set(MODELS):
        raise ValueError("Require the exact closed three-family panel")
    for name, expected in (("study_id", STUDY), ("execution_id", execution_id), ("source_signature", source_signature)):
        if not frame[name].eq(expected).all():
            raise ValueError("Mixed or missing execution identity")
    if (not np.isfinite(frame.seed).all() or not frame.seed.mod(1).eq(0).all()
            or not np.isfinite(frame[["y_true", "y_pred", "persistence_pred", "mase_scale"]]).all().all()
            or frame[["y_true", "y_pred", "persistence_pred"]].lt(0).any().any()
            or frame.mase_scale.le(0).any() or frame.mase_scale.nunique() != 1):
        raise ValueError("Invalid forecast numbers, seed identifiers or frozen MASE scale")
    frame["seed"] = frame.seed.astype(int)
    frame["outcome_day"] = pd.to_datetime(frame.outcome_day, utc=True).dt.strftime("%Y-%m-%d")
    if frame.duplicated(["model", "seed", "outcome_day"]).any():
        raise ValueError("Duplicated fixed prediction")
    if set(frame[["model", "seed"]].itertuples(index=False, name=None)) != {(m, int(s)) for m in MODELS for s in seeds}:
        raise ValueError("Incomplete family/seed cohort")
    reference = None
    for _, part in frame.groupby(["model", "seed"], sort=True):
        part = part.sort_values("outcome_day")
        if part.outcome_day.tolist() != list(dates):
            raise ValueError("Every model/seed requires the same complete ordered calendar")
        values = part[["outcome_day", "y_true", "persistence_pred"]].reset_index(drop=True)
        if reference is None:
            reference = values
        elif not values.equals(reference):
            raise ValueError("Observed outcomes or persistence differ across matched runs")
    persistence = frame.loc[frame.model.eq(MODELS[0]) & frame.seed.eq(seeds[0])].copy()
    persistence["model"], persistence["seed"], persistence["y_pred"] = "Persistence", 0, persistence.persistence_pred
    return pd.concat([frame, persistence], ignore_index=True).sort_values(["model", "seed", "outcome_day"])


def metric_values(observed, forecast, mase_scale):
    observed, forecast = np.asarray(observed, float), np.asarray(forecast, float)
    if len(observed) != len(forecast) or not len(observed) or not np.isfinite(observed + forecast).all() or mase_scale <= 0:
        raise ValueError("Finite matched segment outcomes and a positive training scale required")
    residual = forecast - observed
    absolute = np.abs(residual)
    mse = float(np.mean(residual**2))
    sst = float(np.sum((observed - observed.mean())**2))
    positive = observed > 0
    return {"rmse": math.sqrt(mse), "mae": float(absolute.mean()), "mse": mse,
        "r2": 1 - float(np.sum(residual**2)) / sst if sst > 0 else np.nan,
        "mape": float(100*np.mean(absolute[positive]/observed[positive])) if positive.any() else np.nan,
        "mase": float(absolute.mean()/mase_scale), "mape_defined_dates": int(positive.sum()),
        "residual_bias": float(residual.mean()),
        "residual_sd": float(residual.std(ddof=1)) if len(residual) > 1 else np.nan,
        "absolute_error_p90": float(np.quantile(absolute, .9)), "n_dates": len(observed)}


def thresholds_from_development(frame):
    if not {"market_day_utc", "sigma_7"} <= set(frame):
        raise ValueError("Registered development sigma7 table required")
    dates = pd.to_datetime(frame.market_day_utc, utc=True)
    if dates.isna().any() or dates.duplicated().any() or not dates.dt.year.isin([2023, 2024]).all():
        raise ValueError("Range boundaries must use only2023–2024 development")
    values = pd.to_numeric(frame.sigma_7, errors="raise").dropna().to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Development target must contain finite valid nonnegative values")
    probabilities = [.1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9]
    quantiles = {str(q): float(np.quantile(values, q)) for q in probabilities}
    if not quantiles["0.25"] < quantiles["0.75"]:
        raise ValueError("Three volatility ranges require distinct development quartiles")
    deciles = [quantiles[str(x/10)] for x in range(1, 10)]
    if not np.all(np.diff(deciles) > 0):
        raise ValueError("Development deciles must be strictly increasing")
    return {"source_period": "2023-01-01/2024-12-31", "valid_development_labels": len(values),
        "quantile_method": "NumPy linear interpolation", "quantiles": quantiles,
        "regime_rule": "low: y<=Q25; middle: Q25<y<=Q75; high: y>Q75",
        "decile_rule": "Ten common observed-target intervals split by developmentD10..D90; lower-open upper-closed, unbounded tails",
        "boundaries_derived_from_test": False, "uses_observed_target_for_descriptive_stratification": True,
        "not_available_as_a_forecast_input": True, "minimum_dates_for_public_bin_means": 5,
        "development_target_max":float(values.max()),
        "density_grid_upper":float(max(.1,2*values.max())),
        "density_grid_bins_per_axis":30,
        "density_grid_rule":"Thirty equally spaced bins from0 to max(0.1,twice development target maximum); beyond-range values recorded in boundary cells with separate overflow counts"}


def add_segments(frame, thresholds):
    frame = frame.copy()
    dates = pd.to_datetime(frame.outcome_day, utc=True)
    frame["calendar_month"] = dates.dt.strftime("%Y-%m")
    frame["calendar_year"] = dates.dt.strftime("%Y")
    quantiles = thresholds["quantiles"]
    frame["target_regime"] = np.where(frame.y_true <= quantiles["0.25"], "low",
        np.where(frame.y_true <= quantiles["0.75"], "middle", "high"))
    cuts = np.array([quantiles[str(i/10)] for i in range(1,10)])
    frame["target_decile"] = [f"D{index+1:02d}" for index in np.searchsorted(cuts, frame.y_true, side="left")]
    return frame


def describe_seed_values(values, stochastic):
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]
    n = len(valid)
    mean = float(valid.mean()) if n else np.nan
    sd = float(valid.std(ddof=1)) if stochastic and n > 1 else np.nan
    half = float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if stochastic and n > 1 else np.nan
    return {"n_runs": len(values), "n_defined": n, "mean": mean, "sample_sd": sd,
        "ci95_lower": mean-half, "ci95_upper": mean+half}


def segment_metrics(frame):
    rows = []
    for (model,seed), run in frame.groupby(["model", "seed"], sort=True):
        for dimension in ("overall", "calendar_month", "calendar_year", "target_regime"):
            groups = [("all", run)] if dimension == "overall" else list(run.groupby(dimension, sort=True))
            for name, part in groups:
                rows.append({"model": model, "seed": int(seed), "segment_dimension": dimension,
                    "segment": str(name), **metric_values(part.y_true, part.y_pred, float(run.mase_scale.iloc[0]))})
    runs = pd.DataFrame(rows)
    summaries = []
    for (model,dimension,segment), part in runs.groupby(["model", "segment_dimension", "segment"], sort=True):
        if part.n_dates.nunique() != 1:
            raise ValueError("Segment support differs across training seeds")
        for metric in ERROR_METRICS:
            summaries.append({"model":model, "segment_dimension":dimension, "segment":segment,
                "metric":metric, "n_dates_per_run":int(part.n_dates.iloc[0]),
                **describe_seed_values(part[metric], model != "Persistence"),
                "ci_scope":"Marginal training-seed interval conditional on fixed segment dates; not temporal or simultaneous",
                "posthoc_descriptive": True})
    return runs, pd.DataFrame(summaries)


def forecast_aggregates(frame, dimension, minimum_dates=5):
    rows = []
    for (model,segment), part in frame.groupby(["model",dimension], sort=True):
        counts = part.groupby("seed").size()
        if counts.nunique() != 1:
            raise ValueError("Each aggregate must retain identical support across seeds")
        n_dates = int(counts.iloc[0])
        per_run = part.groupby("seed").agg(mean_forecast=("y_pred","mean"),mean_observed=("y_true","mean"))
        if not np.allclose(per_run.mean_observed,per_run.mean_observed.iloc[0],rtol=0,atol=1e-15):
            raise ValueError("Bin observed target differs between seed cohorts")
        suppressed = n_dates < minimum_dates
        description = describe_seed_values(per_run.mean_forecast, model != "Persistence")
        rows.append({"model":model, "segment_dimension":dimension, "segment":str(segment),
            "n_dates":n_dates, "n_runs":len(per_run),
            "observed_segment_mean":np.nan if suppressed else float(per_run.mean_observed.iloc[0]),
            "forecast_segment_mean":np.nan if suppressed else description["mean"],
            "forecast_mean_seed_sd":np.nan if suppressed else description["sample_sd"],
            "forecast_mean_seed_ci95_lower":np.nan if suppressed else description["ci95_lower"],
            "forecast_mean_seed_ci95_upper":np.nan if suppressed else description["ci95_upper"],
            "status":"suppressed_fewer_than5_dates" if suppressed else "aggregate_defined",
            "scope":"Means of monthly/range aggregates; depiction only, not metrics of an ensemble or independent calendar intervals"})
    return pd.DataFrame(rows)


def residual_structure(frame):
    rows=[]
    for (model,seed),part in frame.groupby(["model","seed"],sort=True):
        residual=part.y_pred.to_numpy()-part.y_true.to_numpy()
        absolute=np.abs(residual)
        rho=float(stats.spearmanr(part.y_true,absolute).statistic)
        rows.append({"model":model,"seed":int(seed),"n_dates":len(part),
            "absolute_error_vs_observed_target_spearman":rho,
            "p_value_reported":False,"scope":"Descriptive monotonic association, overlapping labels and dependent dates; not a significance test"})
    return pd.DataFrame(rows)


def variance_decomposition(runs):
    part=runs.loc[runs.segment_dimension.eq("overall") & runs.model.isin(MODELS)]
    result={"scope":"Balanced descriptive sum-of-squares across model means and training seeds; not ANOVA inference or independent markets", "metrics":{}}
    for metric in METRICS:
        panel=part.pivot(index="seed",columns="model",values=metric).loc[:,list(MODELS)]
        if panel.isna().any().any():
            raise ValueError("Variance decomposition needs all balanced defined model/seed scores")
        values=panel.to_numpy(float)
        means=values.mean(axis=0);grand=float(values.mean())
        between=float(len(values)*np.sum((means-grand)**2));within=float(np.sum((values-means)**2));total=float(np.sum((values-grand)**2))
        if not np.isclose(between+within,total,rtol=1e-12,atol=1e-15):
            raise ValueError("Descriptive variance decomposition fails its arithmetic identity")
        result["metrics"][metric]={"model_count":3,"seeds_per_model":len(values),"within_model_seed_sum_squares":within,
            "between_model_means_sum_squares":between,"total_sum_squares":total,
            "within_fraction":within/total if total>0 else None,"between_fraction":between/total if total>0 else None}
    return result


def forecast_density(frame,thresholds):
    """Fixed common 30x30 counts, without publishing individual forecast pairs."""
    upper=float(thresholds["density_grid_upper"]);count=int(thresholds["density_grid_bins_per_axis"])
    edges=np.linspace(0,upper,count+1);rows=[];support={}
    for model in ORDER:
        part=frame.loc[frame.model.eq(model)]
        x=part.y_true.to_numpy(float);y=part.y_pred.to_numpy(float)
        outside=int(((x>upper)|(y>upper)).sum())
        matrix,_,_=np.histogram2d(np.clip(x,0,np.nextafter(upper,0)),np.clip(y,0,np.nextafter(upper,0)),bins=(edges,edges))
        if int(matrix.sum())!=len(part):raise ValueError("The fixed common density grid dropped a frozen prediction")
        support[model]={"pairs":len(part),"dates":int(part.outcome_day.nunique()),"seeds":int(part.seed.nunique()),"pairs_above_common_grid":outside}
        for i in range(count):
            for j in range(count):
                rows.append({"model":model,"observed_bin":i+1,"forecast_bin":j+1,
                    "observed_lower":float(edges[i]),"observed_upper":float(edges[i+1]),
                    "forecast_lower":float(edges[j]),"forecast_upper":float(edges[j+1]),
                    "count":int(matrix[i,j]),"percent_of_model_pairs":float(100*matrix[i,j]/len(part)),
                    "n_model_pairs":len(part),"unique_calendar_dates":int(part.outcome_day.nunique()),
                    "last_bin_includes_upper_overflow":i==count-1 or j==count-1,
                    "scope":"Counts on fixed development-based common intervals; repeated outcomes across seeds are not independent dates"})
    return pd.DataFrame(rows),support


def verify_metrics(runs, public_runs):
    overall=runs.loc[runs.segment_dimension.eq("overall")]
    if set(overall[["model","seed"]].itertuples(index=False,name=None)) != set(public_runs[["model","seed"]].itertuples(index=False,name=None)):
        raise ValueError("The six-metric verification cohort differs from the original")
    joined=overall.merge(public_runs,on=["model","seed"],suffixes=("_recomputed","_original"),validate="one_to_one")
    differences={}
    for metric in METRICS:
        a,b=joined[metric+"_recomputed"].to_numpy(),joined[metric+"_original"].to_numpy()
        if not np.allclose(a,b,rtol=1e-11,atol=1e-14,equal_nan=True):
            raise ValueError("An original final metric does not reconcile: "+metric)
        differences[metric]=float(np.nanmax(np.abs(a-b)))
    return {"status":"PASS","matched_runs":len(joined),"metrics_per_run":6,"max_absolute_difference":differences,
        "comparison_rtol":1e-11,"comparison_atol":1e-14,"changed_original_metric_files":0}


def plots(monthly,ranges,summary,density,thresholds,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.22})
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True,layout="constrained")
    for ax,model in zip(axes.flat,ORDER):
        part=monthly.loc[monthly.model.eq(model)].sort_values("segment")
        x=np.arange(len(part));ax.plot(x,part.observed_segment_mean,color="#222222",label="Observado, media mensual",linewidth=1.8)
        ax.plot(x,part.forecast_segment_mean,color=COLORS[model],label="Pronosticado, media mensual entre semillas",linewidth=1.7)
        if model!="Persistence":ax.fill_between(x,part.forecast_mean_seed_ci95_lower.to_numpy(float),part.forecast_mean_seed_ci95_upper.to_numpy(float),color=COLORS[model],alpha=.18,label="IC95% entre semillas")
        ax.set_title(LABELS[model]);ax.set_xticks(x[::3],part.segment.iloc[::3],rotation=35,ha="right");ax.set_ylabel("σ7 media mensual");ax.legend(fontsize=7)
    fig.suptitle("Observado y pronosticado: veinte medias mensuales; diagnóstico posterior con pesos fijos")
    fig.savefig(output/"observed_forecast_monthly.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6),layout="constrained")
    for model in ORDER:
        part=ranges.loc[ranges.model.eq(model)&ranges.status.eq("aggregate_defined")].sort_values("segment")
        x=part.observed_segment_mean.to_numpy(float);y=part.forecast_segment_mean.to_numpy(float)
        ax.plot(x,y,"o-",label=LABELS[model],color=COLORS[model],markersize=4)
        if model!="Persistence":ax.fill_between(x,part.forecast_mean_seed_ci95_lower.to_numpy(float),part.forecast_mean_seed_ci95_upper.to_numpy(float),color=COLORS[model],alpha=.12)
    finite=ranges.observed_segment_mean.dropna();lo,hi=float(finite.min()),float(finite.max());ax.plot([lo,hi],[lo,hi],"--",color="black",label="Identidad")
    ax.set_xlabel("Media observada por rango común");ax.set_ylabel("Media pronosticada por rango y semillas");ax.set_title("Comparación agregada por deciles del objetivo de desarrollo\nRangos definidos por el observado; no es una prueba formal de calibración");ax.legend()
    fig.savefig(output/"observed_forecast_by_target_range.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout="constrained")
    for col,(dimension,names,labels) in enumerate((("target_regime",["low","middle","high"],["Baja","Media","Alta"]),("calendar_year",["2025","2026"],["2025","2026 ene–ago"]))):
        for row,metric in enumerate(("rmse","mae","residual_bias")):
            ax=axes[row,col];x=np.arange(len(names));width=.18
            for j,model in enumerate(ORDER):
                part=summary.loc[summary.model.eq(model)&summary.segment_dimension.eq(dimension)&summary.metric.eq(metric)].set_index("segment").reindex(names)
                y=part["mean"].to_numpy(float);sd=part.sample_sd.fillna(0).to_numpy(float)
                ax.bar(x+(j-1.5)*width,y,width,label=LABELS[model],color=COLORS[model],alpha=.85)
                ax.errorbar(x+(j-1.5)*width,y,yerr=sd,fmt="none",ecolor="#333333",linewidth=.7,capsize=2)
            ax.set_xticks(x,labels);ax.set_ylabel({"rmse":"RMSE media individual ± DE semillas","mae":"MAE media individual ± DE semillas","residual_bias":"Sesgo medio ± DE entre semillas"}[metric]);ax.axhline(0,color="#555555",linewidth=.6)
            if row==0:ax.legend(fontsize=8)
    fig.suptitle("Errores por rangos de volatilidad y año; mismos días y pesos fijos por modelo")
    fig.savefig(output/"errors_by_regime_and_year.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    for ax,metric,title in zip(axes,("residual_bias","residual_sd"),("Sesgo residual (pronosticado−observado)","Dispersión residual dentro de cada rango")):
        for model in ORDER:
            part=summary.loc[summary.model.eq(model)&summary.segment_dimension.eq("target_regime")&summary.metric.eq(metric)].set_index("segment").reindex(["low","middle","high"])
            ax.plot(np.arange(3),part["mean"],"o-",color=COLORS[model],label=LABELS[model]);ax.set_xticks(range(3),["Baja","Media","Alta"])
        ax.set_title(title);ax.set_ylabel("Escala original de σ7");ax.axhline(0,color="#333333",linewidth=.7);ax.legend(fontsize=8)
    fig.suptitle("Estructura del error por umbrales Q25/Q75 de desarrollo; descripción posterior")
    fig.savefig(output/"residual_bias_dispersion.png",dpi=160);plt.close(fig)
    from matplotlib.colors import LogNorm
    upper=float(thresholds["density_grid_upper"]);n=int(thresholds["density_grid_bins_per_axis"])
    positive=density.loc[density.percent_of_model_pairs.gt(0),"percent_of_model_pairs"]
    norm=LogNorm(vmin=max(float(positive.min()),.0001),vmax=float(positive.max()))
    fig,axes=plt.subplots(2,2,figsize=(11,9),sharex=True,sharey=True,layout="constrained")
    for ax,model in zip(axes.flat,ORDER):
        part=density.loc[density.model.eq(model)].pivot(index="observed_bin",columns="forecast_bin",values="percent_of_model_pairs").sort_index().sort_index(axis=1)
        values=np.ma.masked_where(part.to_numpy().T<=0,part.to_numpy().T)
        drawn=ax.imshow(values,origin="lower",extent=(0,upper,0,upper),aspect="equal",cmap="viridis",norm=norm)
        ax.plot([0,upper],[0,upper],"--",color="#dd4c4c",linewidth=1,label="Identidad")
        ax.set_title(LABELS[model]);ax.set_xlabel("Observado σ7, intervalos comunes");ax.set_ylabel("Pronosticado σ7, intervalos comunes");ax.legend(fontsize=8)
    fig.colorbar(drawn,ax=axes,label="Porcentaje de pares de cada modelo por celda (escala log)",shrink=.8)
    fig.suptitle(f"Observado frente a pronosticado: densidad agregada en rejilla común {n}×{n}\n608 fechas repetidas entre 309 semillas por red; persistencia es determinista")
    fig.savefig(output/"observed_forecast_density.png",dpi=170);plt.close(fig)


def build(private_dir, output_dir, mirror_dir=None):
    private=Path(private_dir).resolve();output=Path(output_dir).resolve();mirror=Path(mirror_dir).resolve() if mirror_dir else None
    if private.is_relative_to(ROOT) or output==private or output.is_relative_to(private):raise ValueError("Private run and public output must remain separate")
    for destination in (output,mirror):
        if destination and destination.exists() and any(destination.iterdir()):raise ValueError("Use fresh empty output destinations")
    public=ROOT/"results"/STUDY;static=ROOT/"book/_static"/STUDY
    public_before=verify_public_stage(public);static_before=verify_public_stage(static)
    if public_before!=static_before:raise ValueError("Original result/static mirrors differ")
    record=read(private/"registration.json");score=read(private/"score/complete.json");analysis=read(private/"analysis/manifest.json");plan=read(private/"plan.json")
    if record["study_id"]!=STUDY or score["registration_signature"]!=record["registration_signature"] or analysis["registration_signature"]!=record["registration_signature"]:raise ValueError("Mixed closed execution receipts")
    if analysis["score_complete_sha256"]!=sha(private/"score/complete.json"):raise ValueError("Analysis is not bound to the closed score")
    if score["trajectories"]!=927 or score["dates"]!=608 or score["prediction_rows"]!=563616 or score["scoring_passes_per_checkpoint"]!=1 or plan["seeds"]!=list(range(18101,18410)):raise ValueError("Require the original completeR309/608run")
    prediction=checked_file(private,"score/predictions.csv",score["artifact_sha256"]["score/predictions.csv"])
    source_before={name:sha(checked_file(ROOT,name,expected)) for name,expected in record["sources_sha256"].items()}
    original_paths={str(public/name):digest for name,digest in public_before.items()}|{str(static/name):digest for name,digest in static_before.items()}
    original_paths|={str(ROOT/name):digest for name,digest in source_before.items()}
    original_paths|={str(private/name):sha(private/name) for name in ("registration.json","score/complete.json","analysis/manifest.json","score/predictions.csv")}
    frame=validate_panel(pd.read_csv(prediction),plan["seeds"],pd.date_range("2025-01-01","2026-08-31").strftime("%Y-%m-%d").tolist(),record["registration_signature"],record["source_signature"])
    development=checked_file(ROOT,DEVELOPMENT,record["sources_sha256"][DEVELOPMENT])
    thresholds=thresholds_from_development(pd.read_csv(development));frame=add_segments(frame,thresholds)
    runs,summaries=segment_metrics(frame);validation=verify_metrics(runs,pd.read_csv(public/"run_metrics.csv"))
    monthly=forecast_aggregates(frame,"calendar_month");ranges=forecast_aggregates(frame,"target_decile");residuals=residual_structure(frame);variance=variance_decomposition(runs);density,density_support=forecast_density(frame,thresholds)
    counts=frame.loc[frame.model.eq("GRU")&frame.seed.eq(18101)].target_regime.value_counts().to_dict()
    report={"supplement_id":SUPPLEMENT,"parent_study_id":STUDY,"source_execution_id":record["registration_signature"],"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "scope":SCOPE,"posthoc_after_original_analysis":True,"original_metrics_unchanged":True,"model_fits":0,"new_checkpoint_forecasts":0,"raw_market_archive_reads":0,"weights_or_scalers_loaded":0,
        "models":list(ORDER),"neural_runs_per_model":309,"neural_runs":927,"deterministic_reference_runs":1,"common_dates":608,"neural_prediction_rows_read":563616,
        "calendar_months":20,"calendar_year_segments":2,"target_regime_counts":{k:int(v) for k,v in counts.items()},
        "density_grid_shape":[30,30],"density_rows":len(density),"density_support":density_support,
        "segment_run_metric_rows":len(runs),"segment_summary_rows":len(summaries),"monthly_aggregate_rows":len(monthly),"target_range_aggregate_rows":len(ranges),"residual_structure_rows":len(residuals),
        "public_daily_observations_or_forecasts":False,"test_driven_model_selection":False,"new_confirmatory_hypothesis_tests":False,
        "residual_sign":"forecast minus observed; positive means overprediction",
        "segment_metric_estimand":"Average of each individual model/seed metric on matched segment dates, never an ensemble metric",
        "illustration_estimand":"Mean forecast per month/range and across seeds; illustrative aggregation, not an ensemble benchmark score",
        "range_assignment_caveat":"Uses observed outcome retrospectively to diagnose errors; boundaries from development; range membership unavailable at forecast time",
        "seed_interval_caveat":"Marginal seed intervals condition on fixed dates; neither independent market intervals nor temporal/family-wise coverage",
        "temporal_generalization":"Known historical calendar retained; no new unseen validation or temporal power claim"}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e1-frozen-errors-",dir=output.parent) as temporary:
        stage=Path(temporary)
        for name,data in {"segment_run_metrics.csv":runs,"segment_summary.csv":summaries,"monthly_forecast_summary.csv":monthly,"target_range_forecast_summary.csv":ranges,"residual_structure.csv":residuals,"observed_forecast_density.csv":density}.items():data.to_csv(stage/name,index=False)
        plots(monthly,ranges,summaries,density,thresholds,stage)
        after={name:sha(Path(name)) for name in original_paths}
        if after!=original_paths:raise ValueError("An original scientific/result input changed during the diagnostic")
        validation.update({"original_public_files_verified_per_mirror":len(public_before),"registered_scientific_sources_verified":len(source_before),"original_files_checked_before_and_after":len(original_paths),"all_original_hashes_unchanged":True})
        sources={"parent_scientific_source_commit":record["source_commit"],"execution_id":record["registration_signature"],"source_signature":record["source_signature"],
            "private_predictions_sha256":sha(prediction),"private_score_complete_sha256":sha(private/"score/complete.json"),"private_analysis_manifest_sha256":sha(private/"analysis/manifest.json"),
            "original_public_files_sha256":public_before,"registered_scientific_sources_sha256":source_before,"development_threshold_source":DEVELOPMENT,"development_threshold_source_sha256":sha(development),"diagnostic_script_sha256":sha(Path(__file__))}
        for name,data in {"summary.json":report,"validation.json":validation,"thresholds.json":thresholds,"source_manifest.json":sources,"variance_decomposition.json":variance}.items():(stage/name).write_bytes(encode(data))
        (stage/"README.md").write_text("""# Análisis posterior de errores con predicciones congeladas

Las tres redes y persistencia conservan sus predicciones y resultados finales. Este suplemento no entrena, no carga pesos ni escaladores, no vuelve a pronosticar y no cambia recetas o semillas. Lee exclusivamente predicciones cerradas y sus registros, además de sigma7 de desarrollo para fijar rangos comunes.

Se publican veinte medias mensuales observado/pronosticado y comparaciones agregadas por deciles del objetivo de desarrollo, junto a errores por mes, año y rangos Q25/Q75. Todos los rangos usan exactamente los mismos días para las redes y persistencia. Los errores se calculan primero por corrida y después se resumen entre semillas; las líneas de pronóstico medio ilustran agregados, no un nuevo ensemble evaluado.

La figura de densidad observado/pronosticado usa una rejilla común 30×30, fijada mediante desarrollo. Contiene todos los pares de las corridas, normalizados por familia. Los extremos fuera del rango se acumulan en la última celda y se contabilizan aparte. Repetir el mismo observado entre 309 semillas no genera fechas independientes. El CSV sólo contiene intervalos, conteos y porcentajes agregados.

El sesgo residual es pronosticado menos observado. La asociación entre error absoluto y objetivo es descriptiva y no tiene p-valor. La descomposición de sumas de cuadrados distingue variación entre medias de modelos y variación dentro de modelos; no constituye ANOVA inferencial.

Los rangos emplean el observado para analizar errores retrospectivamente: esa pertenencia no estaba disponible como entrada. Los límites provienen exclusivamente de desarrollo 2023–2024. Los IC son marginales entre inicializaciones y condicionados a los mismos días; no son intervalos temporales o simultáneos. No hay nuevas pruebas confirmatorias, selección a partir del test ni generalización independiente.

Los archivos públicos no contienen filas diarias de observaciones o pronósticos, datos de mercado, pesos o valores de escaladores. Las medias por bin se suprimen si contienen menos de cinco días. Los hashes de todas las fuentes registradas y los dos espejos originales se verifican antes y después. Las seis métricas completas se recalculan sólo para verificar que coinciden con las originales, sin sobrescribirlas.
""",encoding="utf-8")
        if {p.name for p in stage.iterdir()}!=PUBLIC_FILES:raise ValueError("Unexpected public diagnostic payload")
        forbidden={"outcome_day","y_true","y_pred","persistence_pred","mase_scale","state_dict","weights","scalers"}
        for path in stage.glob("*.csv"):
            if set(pd.read_csv(path,nrows=0))&forbidden:raise ValueError("Daily/private columns cannot be published")
        mapping={name:{"sha256":sha(stage/name),"bytes":(stage/name).stat().st_size} for name in sorted(PUBLIC_FILES)}
        (stage/"manifest.json").write_bytes(encode({"publication_schema":"e1_public_metadata_allowlist_v1","approved_public_files":sorted(PUBLIC_FILES),"files":mapping,"private_payloads_exported":False,"parent_study_id":STUDY,"posthoc_descriptive":True}))
        shutil.copytree(stage,output,dirs_exist_ok=True)
    if mirror:shutil.copytree(output,mirror,dirs_exist_ok=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--mirror-dir",type=Path)
    args=parser.parse_args();result=build(args.private_dir,args.output_dir,args.mirror_dir)
    print(json.dumps({k:result[k] for k in ("supplement_id","original_metrics_unchanged","common_dates","segment_run_metric_rows","target_regime_counts")},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
