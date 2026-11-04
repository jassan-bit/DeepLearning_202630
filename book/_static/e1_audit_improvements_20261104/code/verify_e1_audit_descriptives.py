"""Independent audit of the saved descriptive supplement, without model execution.

No project analyzer or its mathematical helpers are imported. Saved forecast
vectors and development/test daily derivatives are read locally; raw ZIP files,
model checkpoints and training or forward routines are never opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e1_audit_improvements_20261104/descriptive"
PROTOCOL = ROOT / "experiments/e1_audit_improvements_20261104/protocol.json"
REPORT = ROOT / "references/audits/revision_e1_audit_descriptives_20261104.json"
MODELS = ("MLP", "LSTM", "CNN1D")
METHODS = (*MODELS, "Persistencia")
SEEDS = tuple(range(4101, 4111))
HISTORIES = (7, 14, 21, 28)
METRICS = ("rmse", "mae", "mse", "mape", "r2", "mase")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def independent_metrics(frame):
    """Scalar high-precision sums, separately written from the analyzer."""
    y, p = frame.y_true.to_numpy(float), frame.y_pred.to_numpy(float)
    e = y-p
    n = len(e)
    if not n or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite nonempty saved forecasts required")
    mean_y = math.fsum(y)/n
    average = math.fsum(e)/n
    squared = math.fsum(float(v)*float(v) for v in e)/n
    total = math.fsum((float(v)-mean_y)**2 for v in y)
    positive = y > 0
    scales = frame.mase_train_scale.to_numpy(float)
    valid = np.isfinite(scales) & (scales > 0)
    return {"n_dates": n, "rmse": math.sqrt(squared), "mse": squared,
            "mae": math.fsum(abs(float(v)) for v in e)/n,
            "r2": 1-n*squared/total if total > 0 else np.nan,
            "mape": 100*math.fsum(abs(float(a/b)) for a,b in zip(e[positive],y[positive]))/int(positive.sum()) if positive.any() else np.nan,
            "mape_coverage": int(positive.sum())/n,
            "mase": math.fsum(abs(float(a))/float(s) for a,s in zip(e[valid],scales[valid]))/int(valid.sum()) if valid.any() else np.nan,
            "mase_coverage": int(valid.sum())/n,
            "mean_error": average, "error_variance": math.fsum((float(v)-average)**2 for v in e)/n,
            "target_mean": mean_y, "mean_prediction": math.fsum(p)/n}


def linear_quantile(values, probabilities):
    ordered = sorted(float(v) for v in values)
    result = []
    for probability in probabilities:
        where = (len(ordered)-1)*probability
        lower, upper = math.floor(where), math.ceil(where)
        result.append(ordered[lower]+(where-lower)*(ordered[upper]-ordered[lower]))
    return np.asarray(result)


def empirical_wasserstein(first, second):
    a, b = np.sort(first), np.sort(second)
    knots = np.unique(np.concatenate([a,b]))
    if len(knots) < 2:
        return 0.
    cdf_a = np.searchsorted(a,knots[:-1],side="right")/len(a)
    cdf_b = np.searchsorted(b,knots[:-1],side="right")/len(b)
    return float(np.diff(knots)@np.abs(cdf_a-cdf_b))


class Audit:
    def __init__(self):
        self.counters = {}
        self.maximum_absolute_difference = 0.
        self.checks = []

    def count(self, key, number=1):
        self.counters[key] = self.counters.get(key,0)+number

    def require(self, condition, text):
        if not bool(condition):
            raise AssertionError(text)

    def near(self, actual, expected, label):
        a, b = np.asarray(actual,dtype=float), np.asarray(expected,dtype=float)
        if a.shape != b.shape:
            raise AssertionError(label+": unequal numeric shapes")
        finite = np.isfinite(a)&np.isfinite(b)
        if finite.any():
            self.maximum_absolute_difference = max(self.maximum_absolute_difference,float(np.max(np.abs(a[finite]-b[finite]))))
        self.require(np.allclose(a,b,rtol=1e-9,atol=1e-12,equal_nan=True),label)
        self.count("numeric_values_checked",a.size)

    def hash(self,path,expected):
        self.require(digest(path)==expected,"Changed signed file: "+str(path))
        self.count("file_hashes_checked")

    def verify_registration(self):
        summary = load_json(OUT/"summary.json")
        manifest = load_json(OUT/"code_manifest_before_analysis.json")
        self.hash(PROTOCOL,summary["protocol_sha256"])
        self.hash(PROTOCOL,manifest["protocol_sha256"])
        analyzer = ROOT/"scripts/analyze_e1_audit_descriptives.py"
        self.hash(analyzer,summary["analysis_code_sha256"])
        self.hash(analyzer,manifest["analysis_code_sha256"])
        for name,expected in summary["source_sha256"].items():
            self.hash(ROOT/name,expected)
        for name,expected in summary["output_sha256"].items():
            self.hash(OUT/name,expected)
        self.require(pd.Timestamp(manifest["registered_before_descriptive_calculations_at_utc"]) < pd.Timestamp(summary["completed_at_utc"]),
                     "Registration must precede the new descriptive calculations")
        for key in ("new_model_fits","new_checkpoint_forward_calls","raw_archives_opened"):
            self.require(summary[key]==0,"Descriptive analysis must not execute models or raw reads")
        self.checks.append("Source, protocol, analyzer and aggregate-output hashes match; registration precedes supplemental calculation with zero raw reads, fits or forward calls.")

    def load_forecasts(self):
        old = pd.read_csv(ROOT/"results/dl_r2_cv_20261104/predictions.csv")
        old = old.loc[old.variant.eq("relative_window") & old.history_days.eq(7)]
        new = pd.read_csv(ROOT/"results/filipe_multihistory_20261104/predictions.csv")
        test = pd.read_csv(ROOT/"results/filipe_multihistory_20261104/test/predictions.csv")
        result = {}
        for phase,data,first,last in (("cv",pd.concat([old,new],ignore_index=True),"2024-01-01","2024-12-31"),
                                       ("test",test,"2025-01-01","2026-08-31")):
            days = pd.date_range(first,last).strftime("%Y-%m-%d").tolist()
            self.require(len(data)==120*len(days),"Wrong primary forecast count")
            self.require(not data.duplicated(["history_days","model","seed","outcome_day"]).any(),"Duplicate saved forecast")
            reference = None
            for (history,model,seed),group in data.groupby(["history_days","model","seed"]):
                group = group.sort_values("outcome_day")
                self.require(history in HISTORIES and model in MODELS and seed in SEEDS,"Unexpected forecast configuration")
                self.require(group.outcome_day.tolist()==days,"Unequal evaluation dates")
                target = group[["y_true","persistence_pred"]].to_numpy(float)
                self.require(np.isfinite(target).all(),"Nonfinite target/reference")
                if reference is None: reference = target
                else:self.near(target,reference,"Unequal targets/reference across configurations")
                self.count("neural_trajectories_checked")
            extra=[]
            for history in HISTORIES:
                context=data.loc[data.history_days.eq(history)&data.model.eq("MLP")&data.seed.eq(4101)].sort_values("outcome_day").copy()
                context["y_pred"]=context.persistence_pred
                context["model"]="Persistencia";context["seed"]=0;context["stochastic_training"]=False
                extra.append(context)
                self.count("deterministic_reference_paths")
            data=data.copy();data["stochastic_training"]=True
            data=pd.concat([data,*extra],ignore_index=True)
            data["phase"]=phase
            self.require(len(data)==124*len(days),"Persistence was duplicated as pseudoseeds")
            result[phase]=data
        self.checks.append("All four histories share their phase's dates, labels and persistence. Each contextual reference appears once per history, with seed0 and no stochastic-training repetitions.")
        return result

    def check_metric_rows(self,rows,expected,key_fields):
        self.require(len(rows)==len(expected),"Metric row count mismatch")
        self.require(not rows.duplicated(key_fields).any(),"Duplicate metric keys")
        indexed=rows.set_index(key_fields)
        for record in expected:
            key=tuple(record[k] for k in key_fields)
            self.require(key in indexed.index,"Missing metric identity: "+str(key))
            actual=indexed.loc[key]
            self.require(bool(actual.stochastic_training)==record["stochastic_training"],"Wrong reference-seed type")
            for name,value in record.items():
                if name in key_fields or name in ("stochastic_training","status"):continue
                if isinstance(value,str):
                    self.require(str(actual.get(name))==value,"Wrong text identity: "+name)
                else:
                    self.near(actual.get(name,np.nan),value,"Metric "+name+" "+str(key))
            if "status" in record:self.require(actual.status==record["status"],"Wrong small-cell suppression status")
            self.count("aggregate_metric_rows_checked")

    def summary_rows(self,records,keys):
        result=[]
        for identity,group in records.groupby(keys,sort=True):
            identity=identity if isinstance(identity,tuple) else (identity,)
            stochastic=bool(group.stochastic_training.iloc[0]);counts=group.n_dates.to_numpy(int)
            for metric in METRICS:
                values=group[metric].to_numpy(float);values=values[np.isfinite(values)]
                n=len(values);mean=math.fsum(values)/n if n else np.nan
                sd=math.sqrt(math.fsum((float(v)-mean)**2 for v in values)/(n-1)) if stochastic and n>1 else np.nan
                half=float(stats.t.ppf(.975,n-1))*sd/math.sqrt(n) if np.isfinite(sd) else np.nan
                result.append({**dict(zip(keys,identity)),"metric":metric,"mean":mean,"sample_sd":sd,
                               "ci95_lower":mean-half,"ci95_upper":mean+half,"n_defined":n,
                               "n_seed_records":len(group),"n_dates":int(counts[0]) if len(set(counts))==1 else np.nan,
                               "n_dates_min":int(counts.min()),"n_dates_max":int(counts.max()),
                               "stochastic_training":stochastic})
        return result

    def verify_years_and_months(self,frames):
        test=frames["test"].copy()
        test["period"]=test.outcome_day.str[:4].map({"2025":"2025","2026":"2026_Jan_Aug"})
        records=[]
        keys=["history_days","model","seed","period"]
        for identity,group in test.groupby(keys,sort=True):
            result={**dict(zip(keys,identity)),"stochastic_training":bool(group.stochastic_training.iloc[0]),**independent_metrics(group)}
            self.require(result["n_dates"]== (365 if result["period"]=="2025" else 243),"Wrong annual date coverage")
            records.append(result)
        self.require(len(records)==248,"Expected248 annual neural/reference records")
        self.check_metric_rows(pd.read_csv(OUT/"annual_run_metrics.csv"),records,keys)
        annual=pd.DataFrame(records)
        summary=self.summary_rows(annual,["history_days","model","period"])
        self.require(len(summary)==192,"Expected192 annual summaries")
        self.check_metric_rows(pd.read_csv(OUT/"annual_metrics_summary.csv"),summary,["history_days","model","period","metric"])
        monthly=[]
        keys=["phase","history_days","model","seed","fold_id"]
        for phase,frame in frames.items():
            for identity,group in frame.groupby(keys,sort=True):
                monthly.append({**dict(zip(keys,identity)),"stochastic_training":bool(group.stochastic_training.iloc[0]),**independent_metrics(group)})
        monthly=pd.DataFrame(monthly).rename(columns={"fold_id":"month"})
        self.checks.append("All248 annual records and192 summaries independently reproduce2025(365dates)/Jan–Aug2026(243dates), including period-specific R2 and single-path deterministic references.")
        return monthly

    def verify_variance(self,monthly):
        table=pd.read_csv(OUT/"variance_components.csv")
        indexed=table.set_index(["phase","history_days","panel","model","factor"])
        count=0
        neural=monthly.loc[monthly.model.isin(MODELS)]
        for (phase,history),group in neural.groupby(["phase","history_days"]):
            panels=[("three_neural_models","all_three",group,["model","seed","month"])]
            panels += [("within_model",model,part,["seed","month"]) for model,part in group.groupby("model")]
            for name,model,panel,factors in panels:
                self.require(not panel.duplicated(factors).any(),"Duplicate factorial cells")
                sizes=[panel[f].nunique() for f in factors]
                self.require(len(panel)==math.prod(sizes),"Unbalanced factorial panel")
                x=panel.rmse.to_numpy(float);n=len(x);mean=math.fsum(x)/n
                total=math.fsum((float(v)-mean)**2 for v in x)
                components=[]
                for factor in factors:
                    grouped=panel.groupby(factor).rmse
                    cellsize=n/panel[factor].nunique()
                    ss=cellsize*math.fsum((float(values.mean())-mean)**2 for _,values in grouped)
                    components.append((factor,ss))
                remainder=max(0.,total-math.fsum(v for _,v in components))
                components.append(("interactions_residual",remainder))
                actual_shares=[]
                for factor,ss in components:
                    key=(phase,history,name,model,factor);self.require(key in indexed.index,"Missing variance component")
                    row=indexed.loc[key]
                    self.near([row.total_variance,row.component_variance,row.share,row.n_cells],
                              [total/n,ss/n,ss/total if total>0 else np.nan,n],"Factorial variance "+str(key))
                    actual_shares.append(row.share);count+=1
                self.near(sum(actual_shares),1.,"Variance shares must sum to one")
                self.count("balanced_factorial_panels_checked")
        self.require(count==len(table)==104,"Wrong variance-component grid")
        self.count("variance_components_checked",count)
        self.checks.append("104 variance components reproduce from balanced seed×month×model sums of squares; shares sum to1 within every panel, without inferential ANOVA or iid-fold assumptions.")

    def verify_thresholds(self):
        market=pd.read_csv(ROOT/"results/eda_development/daily_market_development.csv")
        target=pd.read_csv(ROOT/"results/eda_development/daily_volatility_development.csv")
        self.require(market.market_day_utc.tolist()==target.market_day_utc.tolist(),"Development calendars differ")
        x=market[["open","high","low","close","volume"]].to_numpy(float)
        y=target.sigma_7.to_numpy(float);days=pd.to_datetime(target.market_day_utc,utc=True)
        values=[]
        for index in range(28,len(y)):
            if (days.iloc[index]<=pd.Timestamp("2024-12-03",tz="UTC") and np.isfinite(y[index])
                    and np.isfinite(x[index-28:index]).all()):values.append(y[index])
        self.require(len(values)==646,"Threshold cohort must be646 complete common D28 training outcomes")
        q=linear_quantile(values,[.25,.5,.75])
        threshold=load_json(OUT/"development_bin_thresholds.json")
        self.near([threshold["q25"],threshold["q50"],threshold["q75"]],q,"Development-only manual quantiles")
        self.require(threshold["training_nonmissing_targets"]==646 and threshold["common_training_last_outcome_day"]=="2024-12-03","Wrong threshold support")
        self.checks.append("Segmentation quartiles independently match646 eligible development labels through2024-Dec03 using28 prior finite OHLCVdays; no test value selects the thresholds. CV segmentation remains retrospective diagnosis.")
        return q

    def verify_segments_and_bins(self,frames,quantiles):
        records=[]
        for phase,frame in frames.items():
            frame=frame.copy()
            frame["severity"]=np.asarray(["low","middle","high"])[np.searchsorted(quantiles[[0,2]],frame.y_true,side="right")]
            keys=["history_days","model","seed","severity"]
            for identity,part in frame.groupby(keys,sort=True):
                record={"phase":phase,**dict(zip(keys,identity)),"stochastic_training":bool(part.stochastic_training.iloc[0]),**independent_metrics(part)}
                suppressed=len(part)<10
                record["status"]="suppressed_fewer_than_10_dates" if suppressed else "defined"
                if suppressed:
                    for field in (*METRICS,"mean_error","error_variance","target_mean","mean_prediction"):record[field]=np.nan
                records.append(record)
        self.require(len(records)==744,"Wrong three-severity grid")
        keys=["phase","history_days","model","seed","severity"]
        self.check_metric_rows(pd.read_csv(OUT/"severity_run_metrics.csv"),records,keys)
        summary=self.summary_rows(pd.DataFrame(records),["phase","history_days","model","severity"])
        self.require(len(summary)==576,"Wrong severity-summary grid")
        self.check_metric_rows(pd.read_csv(OUT/"severity_metrics_summary.csv"),summary,["phase","history_days","model","severity","metric"])
        bin_records=[]
        for (history,model,seed),group in frames["test"].groupby(["history_days","model","seed"],sort=True):
            labels=np.searchsorted(quantiles,group.y_pred,side="right")+1
            for category in range(1,5):
                part=group.loc[labels==category];valid=len(part)>=10
                result={"history_days":history,"model":model,"seed":seed,"predicted_bin":category,
                        "n_dates":len(part),"stochastic_training":bool(group.stochastic_training.iloc[0]),
                        "status":"defined" if valid else "suppressed_fewer_than_10_dates"}
                if valid:result.update(independent_metrics(part))
                else:result.update({key:np.nan for key in (*METRICS,"mean_error","error_variance","target_mean","mean_prediction")})
                bin_records.append(result)
        self.require(len(bin_records)==496,"Wrong residual-bin grid")
        table=pd.read_csv(OUT/"residual_bins_by_seed.csv")
        self.check_metric_rows(table,bin_records,["history_days","model","seed","predicted_bin"])
        suppressed=table.n_dates.lt(10)
        guarded=[*METRICS,"mean_error","error_variance","target_mean","mean_prediction"]
        self.require(table.loc[suppressed,guarded].isna().all().all(),"Small cells must suppress sensitive error moments")
        self.count("suppressed_residual_bin_rows",int(suppressed.sum()))
        summaries=pd.read_csv(OUT/"residual_bins_summary.csv").set_index(["history_days","model","predicted_bin"])
        count=0
        for identity,group in pd.DataFrame(bin_records).groupby(["history_days","model","predicted_bin"],sort=True):
            row=summaries.loc[identity];valid=group.loc[group.n_dates.ge(10)]
            self.near([row.n_seed_records,row.n_defined_seed_records,row.n_dates_min,row.n_dates_max],
                      [len(group),len(valid),group.n_dates.min(),group.n_dates.max()],"Residual-bin summary support")
            for field in ["mean_error","error_variance","rmse","mae","mean_prediction"]:
                values=valid[field].to_numpy(float);n=len(values)
                mean=math.fsum(values)/n if n else np.nan
                sd=math.sqrt(math.fsum((float(v)-mean)**2 for v in values)/(n-1)) if n>1 else np.nan
                self.near([row[field],row[field+"_sample_sd"]],[mean,sd],"Residual-bin mean/dispersion")
            count+=1
        self.require(count==len(summaries)==64,"Wrong residual-bin summary count")
        self.count("residual_bin_summaries_checked",count)
        self.checks.append("744 target-severity rows,576 summaries,496 prediction-bin rows and64 binned summaries reproduce from development-frozen cutpoints. All cells with n<10 suppress metrics; no formal heteroscedasticity test is fabricated.")

    def verify_monthly_and_cases(self,monthly):
        rows=[]
        for identity,group in monthly.groupby(["phase","history_days","model","month"],sort=True):
            self.require(group.n_dates.nunique()==1,"Unequal monthly calendars")
            means={"target_mean":group.target_mean.iloc[0],"prediction_mean":group.mean_prediction.mean(),
                   "prediction_mean_sample_sd_across_seeds":group.mean_prediction.std(ddof=1) if len(group)>1 else np.nan,
                   "rmse_mean":group.rmse.mean(),"mae_mean":group.mae.mean(),
                   "bias_mean":group.mean_error.mean(),"error_variance_mean":group.error_variance.mean()}
            rows.append({**dict(zip(["phase","history_days","model","month"],identity)),
                         "stochastic_training":bool(group.stochastic_training.iloc[0]),
                         "n_dates":int(group.n_dates.iloc[0]),"n_seed_records":len(group),**means})
        table=pd.read_csv(OUT/"monthly_prediction_aggregates.csv")
        self.check_metric_rows(table,rows,["phase","history_days","model","month"])
        self.require(len(table)==512 and table.n_dates.min()>=28,"Public monthly points must summarize28+ dates")
        candidates=pd.DataFrame(rows);test=candidates.loc[candidates.phase.eq("test")]
        expected=[]
        for identity,group in test.groupby(["history_days","model"],sort=True):
            hardest=group.sort_values(["rmse_mean","month"],ascending=[False,True]).head(3)
            for rank,(_,row) in enumerate(hardest.iterrows(),1):expected.append({**row.to_dict(),"difficulty_rank":rank})
        self.check_metric_rows(pd.read_csv(OUT/"difficult_months.csv"),expected,["history_days","model","month"])
        self.require(len(expected)==48,"Expected top3 reporting months per history/method")
        self.checks.append("512 monthly aggregate points and48 retrospectively difficult-month records match the saved trajectories; each public point summarizes28–31 dates, not individual forecasts.")

    def verify_distributions(self):
        development=pd.read_csv(ROOT/"results/eda_development/daily_market_development.csv")
        evaluation=pd.read_csv(ROOT/"results/filipe_multihistory_20261104/test/daily_market_test.csv")
        market=pd.concat([development,evaluation],ignore_index=True)
        dates=pd.DatetimeIndex(pd.to_datetime(market.market_day_utc,utc=True))
        self.require(dates.equals(pd.date_range("2023-01-01","2026-08-31",tz="UTC")),"Daily derivatives changed their admitted calendar")
        dev_y=pd.read_csv(ROOT/"results/eda_development/daily_volatility_development.csv")
        test_y=pd.read_csv(ROOT/"results/filipe_multihistory_20261104/test/daily_volatility_test.csv")
        labels=np.concatenate([dev_y.sigma_7.to_numpy(float),test_y.sigma_7.to_numpy(float)])
        close=market.close.to_numpy(float);returns=np.full(len(close),np.nan)
        for t in range(1,len(close)):
            if np.isfinite(close[t]) and np.isfinite(close[t-1]):returns[t]=math.log(close[t])-math.log(close[t-1])
        sigma=np.full(len(close),np.nan)
        for t in range(7,len(close)):
            window=returns[t-6:t+1]
            if np.isfinite(window).all():
                mean=math.fsum(window)/7
                sigma[t]=math.sqrt(math.fsum((float(v)-mean)**2 for v in window)/7)
        self.near(labels,sigma,"Independent complete-calendar sigma7 reconstruction")
        self.count("volatility_days_reconstructed",len(labels))
        market=market.set_index(dates)
        market["daily_log_return"]=returns;market["log1p_volume"]=np.log1p(market.volume)
        market["sigma7"]=labels
        features=["open","high","low","close","volume","daily_log_return","log1p_volume","sigma7"]
        reference=market.loc[:"2024-12-03",features]
        table=pd.read_csv(OUT/"distribution_by_period.csv").set_index(["period","feature"])
        count=0
        for label,year in [("2023",2023),("2024",2024),("2025",2025),("2026_Jan_Aug",2026)]:
            period=market.loc[market.index.year==year]
            for feature in features:
                values=period[feature].dropna().to_numpy(float)
                training=reference[feature].dropna().to_numpy(float)
                mean=math.fsum(values)/len(values);trainmean=math.fsum(training)/len(training)
                sd=math.sqrt(math.fsum((float(v)-mean)**2 for v in values)/len(values))
                trainsd=math.sqrt(math.fsum((float(v)-trainmean)**2 for v in training)/len(training))
                quantiles=linear_quantile(values,[.1,.25,.5,.75,.9])
                expected=[len(period),len(values),len(values)/len(period),mean,sd,*quantiles,
                          (mean-trainmean)/trainsd if trainsd>0 else np.nan,
                          empirical_wasserstein(values,training)/trainsd if trainsd>0 else np.nan]
                row=table.loc[(label,feature)]
                fields=["n_calendar_days","n_observed","coverage","mean","sd_ddof0","q10","q25","median","q75","q90","standardized_mean_shift_vs_training","wasserstein_over_training_sd"]
                self.near(row[fields].to_numpy(float),expected,"Period distribution "+str((label,feature)))
                count+=1
        self.require(count==len(table)==32,"Expected32 distribution aggregates")
        self.count("distribution_aggregates_checked",count)
        self.checks.append("All32 four-period covariate/target distributions reproduce, including independent empirical-CDF Wasserstein distances and1339-day population sigma7 reconstruction without gap bridging.")

    def verify_public_scope(self):
        for path in OUT.glob("*.csv"):
            table=pd.read_csv(path)
            self.require(not {"y_true","y_pred","persistence_pred","open","high","low","close","volume","sigma_7","residual"}.intersection(table.columns),
                         "A public descriptive table must not contain raw forecast/market vector columns")
        self.require(not any(p.suffix in {".pt",".pth",".npz",".npy",".zip"} or p.name.endswith(".tar.gz") for p in OUT.iterdir()),
                     "No weights or full archives belong in descriptive public outputs")
        self.checks.append("Descriptive outputs contain only period/bin/factor aggregates; no individual prediction, daily market, residual vectors, checkpoints or full archives are exported.")

    def run(self):
        self.verify_registration()
        frames=self.load_forecasts()
        monthly=self.verify_years_and_months(frames)
        self.verify_variance(monthly)
        thresholds=self.verify_thresholds()
        self.verify_segments_and_bins(frames,thresholds)
        self.verify_monthly_and_cases(monthly)
        self.verify_distributions()
        self.verify_public_scope()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=REPORT)
    args=parser.parse_args()
    audit=Audit();failure=None
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),
            "independence":"No project analysis/training helpers imported; only stored forecasts and daily derivatives read.",
            "raw_archives_opened":0,"new_model_fits":0,"new_checkpoint_forward_calls":0}
    try:
        audit.run();report["status"]="PASS"
    except Exception as exc:
        failure=exc;report["status"]="FAIL";report["error"]=f"{type(exc).__name__}: {exc}"
    report.update({"counters":audit.counters,"checks":audit.checks,
                   "maximum_numeric_absolute_difference":audit.maximum_absolute_difference,
                   "verifier_sha256":digest(__file__),"protocol_sha256":digest(PROTOCOL)})
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    text=["# Auditoría independiente del complemento descriptivo", "",f"Estado: **{report['status']}**.","",report["independence"],""]
    text += ["- "+item for item in audit.checks]
    text += ["","| Control | Cantidad |","|---|---:|"]+[f"| {key} | {value} |" for key,value in audit.counters.items()]
    text += ["",f"Máxima diferencia numérica absoluta: {audit.maximum_absolute_difference:.12g}.",
             "","La auditoría no ejecuta modelos ni lecturas de ZIP. Los segmentos se definieron con desarrollo, pero su diagnóstico es posterior a inspeccionar test. Las varianzas son identidades descriptivas, no pruebas ANOVA de mercados independientes.",""]
    if failure:text += ["Fallo: "+report["error"],""]
    args.report.with_suffix(".md").write_text("\n".join(text))
    print(json.dumps({"status":report["status"],"counters":audit.counters,"error":report.get("error"),"report":str(args.report)}))
    if failure:raise failure


if __name__=="__main__":main()
