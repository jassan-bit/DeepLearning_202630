#!/usr/bin/env python3
"""E2 benchmark analysis proposal, templates and tested numerical primitives.

Default invocation uses only the standard library and publishes no numerical
benchmark results. Optional input validation accepts explicitly declared
synthetic fixtures in a private output directory. Real analysis remains blocked
until the experimental/statistical protocol is frozen and its evidence exists.
NumPy/SciPy are imported only by numerical/fixture functions, never by default.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = "experiments/e2_benchmark_analysis_20261106/proposal.json"
BENCHMARK_PROPOSAL = "experiments/e2_benchmark_design_20261105/proposal.json"
OUTPUT = "book_e2/_static/benchmark_analysis"
DELTA = 0.00025772842664981106
SOURCE_COMMIT = "94a4b0324e93361e8928d708fdfeb50110e3686f"
PRIMARY_BASELINE = "GRU/D7-c03"
HASH = re.compile(r"[0-9a-f]{64}\Z")
PREDICTION_FIELDS = ["model", "seed", "origin_utc", "horizon", "y_true", "y_pred", "eligible", "run_id", "scope"]
RUN_FIELDS = ["run_id", "model", "seed", "phase", "dataset_sha256", "split_sha256", "config_sha256", "checkpoint_sha256", "scope"]
COST_FIELDS = ["model", "seed", "measurement_block", "latency_ms_per_window", "training_seconds", "peak_memory_bytes", "parameters", "FLOPs_per_window", "device", "precision", "threads", "batch_size", "input_shape", "timing_protocol_sha256", "scope"]


def numeric():
    import numpy as np
    from scipy import stats
    return np, stats


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if hasattr(value,"tolist"): return clean(value.tolist())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def array(values, name, ndim=None):
    np,_=numeric();result=np.asarray(values,dtype=np.float64)
    if not result.size or not np.isfinite(result).all() or (ndim is not None and result.ndim!=ndim):
        raise ValueError(name+" must be finite, nonempty and have the declared dimensions")
    return result


def utc_day(value):
    text=str(value)
    if len(text)==10: text += "T00:00:00+00:00"
    moment=datetime.fromisoformat(text.replace("Z","+00:00"))
    if moment.tzinfo is None or moment.utcoffset()!=timedelta(0):
        raise ValueError("Origins must have an explicit UTC calendar convention")
    if moment.time().replace(tzinfo=None)!=datetime.min.time():
        raise ValueError("D7 origins must be UTC midnight dates")
    return moment.astimezone(timezone.utc)


def daily_calendar(origins, *, calendar_timezone):
    if calendar_timezone!="UTC":raise ValueError("The inherited calendar timezone is UTC")
    dates=[utc_day(x) for x in origins]
    if len(dates)<2 or any(b-a!=timedelta(days=1) for a,b in zip(dates,dates[1:])):
        raise ValueError("An explicit complete, ordered daily calendar is required; do not compress gaps")
    return dates


def bool_field(value):
    if value in (True,"true","True",1,"1"):return True
    if value in (False,"false","False",0,"0"):return False
    raise ValueError("Eligibility must be an explicit boolean")


def validate_dense_panel(rows, models, seeds, origins, *, calendar_timezone="UTC", deterministic_models=()):
    """Require every model×seed×calendar cell, including ineligible placeholders.

    Masked dates remain on the calendar and never become an adjacent-day pair.
    The same seed keys pair training randomizations; they are not market units.
    """
    np,_=numeric();dates=daily_calendar(origins,calendar_timezone=calendar_timezone)
    models=list(models);seeds=[str(x) for x in seeds]
    deterministic_models=set(deterministic_models)
    if not deterministic_models.issubset(set(models)):raise ValueError("Unknown deterministic reference")
    if len(set(models))!=len(models) or len(set(seeds))!=len(seeds) or not models or not seeds:
        raise ValueError("Model and paired seed registries must be nonempty and unique")
    model_map={m:i for i,m in enumerate(models)};seed_map={s:i for i,s in enumerate(seeds)}
    date_map={d:i for i,d in enumerate(dates)}
    pred=np.full((len(models),len(seeds),len(dates)),np.nan)
    truth=np.full(len(dates),np.nan);eligible=np.zeros(len(dates),dtype=bool)
    seen=set();eligibility_by_date={};truth_by_date={}
    for row in rows:
        m=row["model"];s=str(row["seed"]);d=utc_day(row["origin_utc"])
        if m not in model_map or d not in date_map or (s!="deterministic" if m in deterministic_models else s not in seed_map):
            raise ValueError("Panel cell lies outside its registered model/seed/calendar")
        if str(row.get("horizon"))!="1":raise ValueError("The inherited benchmark horizon is one day")
        key=(m,s,d)
        if key in seen:raise ValueError("Duplicate model/seed/origin cell")
        seen.add(key);ok=bool_field(row["eligible"])
        if d in eligibility_by_date and eligibility_by_date[d]!=ok:
            raise ValueError("Every model and seed must share the same eligible cohort")
        eligibility_by_date[d]=ok;t=date_map[d];eligible[t]=ok
        if ok:
            y=float(row["y_true"]);p=float(row["y_pred"])
            if not math.isfinite(y) or not math.isfinite(p) or y<0 or p<0:
                raise ValueError("Eligible D7 volatility targets/predictions must be finite and nonnegative")
            if d in truth_by_date and truth_by_date[d]!=y:
                raise ValueError("Models/seeds have different ground truth for the same origin")
            truth_by_date[d]=y;truth[t]=y
            if m in deterministic_models:pred[model_map[m],:,t]=p
            else:pred[model_map[m],seed_map[s],t]=p
        elif row.get("y_pred") not in (None,""):
            raise ValueError("An ineligible placeholder cannot invent a numerical prediction")
    expected=((len(models)-len(deterministic_models))*len(seeds)+len(deterministic_models))*len(dates)
    if len(seen)!=expected:
        raise ValueError("The paired panel is incomplete; missing cells cannot be silently dropped")
    if not eligible.any():raise ValueError("The panel has no common eligible origins")
    return {"models":models,"seeds":seeds,"origins":[d.isoformat() for d in dates],
            "calendar_timezone":"UTC","eligible":eligible,"targets":truth,"predictions":pred,
            "n_calendar_origins":len(dates),"n_eligible_origins":int(eligible.sum()),
            "negative_prediction_count":int(np.sum(pred[np.isfinite(pred)]<0)),
            "deterministic_references_stored_once":sorted(deterministic_models),
            "independent_randomized_fits_for_deterministic_references":0,
            "independent_market_series":1,
            "seed_scope":"paired training randomizations, conditional on the same calendar"}


def regression_metrics(target, prediction, *, mase_scale=None):
    np,_=numeric();y=array(target,"targets",1);p=array(prediction,"predictions",1)
    if y.shape!=p.shape:raise ValueError("Target/prediction lengths differ")
    err=p-y;mse=float(np.mean(err**2));den=np.abs(y)+np.abs(p);positive=y>0;smask=den>0
    variation=float(np.sum((y-y.mean())**2))
    return {"n":len(y),"RMSE":math.sqrt(mse),"MAE":float(np.abs(err).mean()),"MSE":mse,
            "R2":None if variation==0 else 1-float(np.sum(err**2))/variation,
            "MAPE_percent":float(np.mean(np.abs(err[positive])/y[positive])*100) if positive.any() else None,
            "MAPE_defined":int(positive.sum()),"MAPE_coverage":float(positive.mean()),
            "SMAPE_percent":float(np.mean(2*np.abs(err[smask])/den[smask])*100) if smask.any() else None,
            "SMAPE_defined":int(smask.sum()),"SMAPE_coverage":float(smask.mean()),
            "SMAPE_both_zero_convention":"undefined, coverage reported; no epsilon or silently assigned zero",
            "MASE":float(np.abs(err).mean()/mase_scale) if mase_scale is not None and math.isfinite(mase_scale) and mase_scale>0 else None,
            "MASE_scale_source":"external train-only consecutive-day denominator; never fitted from evaluation targets"}


def training_mase_scale(training_target, origins, eligible, *, calendar_timezone="UTC"):
    np,_=numeric();daily_calendar(origins,calendar_timezone=calendar_timezone)
    y=np.asarray(training_target,dtype=float);mask=np.asarray(eligible,dtype=bool)
    if y.ndim!=1 or mask.shape!=y.shape or len(y)!=len(origins) or not np.isfinite(y[mask]).all():
        raise ValueError("A training target and explicit complete eligibility calendar are required")
    pairs=mask[1:]&mask[:-1]
    return {"scale":float(np.abs(np.diff(y)[pairs]).mean()) if pairs.any() else None,
            "consecutive_training_pairs":int(pairs.sum()),"source":"training-only lag1; gaps not compressed"}


def rmse_seed_mean(panel, model, indices=None):
    np,_=numeric();m=panel["models"].index(model)
    idx=np.arange(len(panel["eligible"])) if indices is None else np.asarray(indices,dtype=int)
    idx=idx[panel["eligible"][idx]]
    if not len(idx):return None
    y=panel["targets"][idx];p=panel["predictions"][m][:,idx]
    return float(np.sqrt(np.mean((p-y[None,:])**2,axis=1)).mean())


def mean_paired_seed_ranks(panel):
    """Rank models inside each paired seed first, then average those ranks.

    This differs from ordering the mean RMSE. Neither descriptive ranking turns
    algorithmic seeds into independent market series for a Friedman/CD test.
    """
    np,stats=numeric();mask=panel['eligible'];y=panel['targets'][mask]
    errors=np.sqrt(np.mean((panel['predictions'][:,:,mask]-y[None,None,:])**2,axis=2))
    ranks=stats.rankdata(errors,axis=0,method='average')
    return {'models':list(panel['models']),'paired_seed_ids':list(panel['seeds']),
            'RMSE_by_model_seed':errors.tolist(),'ranks_by_model_seed':ranks.tolist(),
            'mean_rank_by_model':ranks.mean(axis=1).tolist(),
            'status':'descriptive_conditional_on_shared_calendar',
            'not_rank_of_aggregate_RMSE':True,'does_not_supply_independent_datasets_for_Friedman':True}


def adjust_pvalues(pvalues, *, method, family_size):
    """Keep missing hypotheses in the declared family; missing p remains null."""
    np,_=numeric();p=[1. if x is None else float(x) for x in pvalues]
    if family_size<len(p) or family_size<1 or any(not math.isfinite(x) or not 0<=x<=1 for x in p):
        raise ValueError("Declared family size and valid p-values are required")
    p=np.asarray(p+[1.]*(family_size-len(p)));order=np.argsort(p,kind="stable");ranked=p[order]
    if method=="Holm":adj=np.minimum(1,np.maximum.accumulate(ranked*np.arange(family_size,0,-1)))
    elif method=="Bonferroni":adj=np.minimum(1,ranked*family_size)
    elif method=="BH_FDR":adj=np.minimum(1,np.minimum.accumulate((ranked*family_size/np.arange(1,family_size+1))[::-1])[::-1])
    else:raise ValueError("Choose Holm, Bonferroni or BH_FDR before results")
    restored=np.empty(family_size);restored[order]=adj
    return [None if original is None else float(restored[i]) for i,original in enumerate(pvalues)]


def intersection_union_p(p_error,p_cost):
    """RQ3 needs both component tests; missing inference does not become p=0."""
    if p_error is None or p_cost is None:
        return {"status":"not_estimable","p_IUT":None,"reason":"Both registered component p-values are required"}
    values=[float(p_error),float(p_cost)]
    if any(not math.isfinite(p) or not 0<=p<=1 for p in values):raise ValueError("Invalid component p-value")
    return {"status":"prototype_combination","p_IUT":max(values),"method":"intersection-union max of component p-values"}


def block_indices(n, block, rng):
    np,_=numeric()
    if type(block) is not int or not 1<=block<=n:raise ValueError("Invalid block length")
    starts=rng.integers(0,n,size=math.ceil(n/block))
    return ((starts[:,None]+np.arange(block))%n).ravel()[:n]


def paired_temporal_gain(panel, reference, candidate, *, block_length, repetitions=49999,
                         seed=20261105, null_margin=DELTA, family_size=30, alpha=.05):
    """Recompute mean-seed RMSE on paired resampled dense-calendar origins.

    Seed trajectories remain fixed; this estimates temporal variability rather
    than pretending repeated initialization creates independent markets.
    """
    np,_=numeric()
    if repetitions<99 or not 0<alpha<1 or family_size<1:raise ValueError("Invalid registered inference settings")
    n=len(panel["eligible"]);rng=np.random.default_rng(seed)
    observed=rmse_seed_mean(panel,reference)-rmse_seed_mean(panel,candidate)
    boot=[]
    for _ in range(repetitions):
        idx=block_indices(n,block_length,rng);a=rmse_seed_mean(panel,reference,idx);b=rmse_seed_mean(panel,candidate,idx)
        if a is not None and b is not None:boot.append(a-b)
    if len(boot)<.95*repetitions:
        return {"status":"not_estimable","effect":None,"ci95":None,"p_one_sided":None,
                "reason":"Insufficient resamples retain eligible origins","finite_draws":len(boot)}
    boot=np.asarray(boot);p=(1+np.count_nonzero(boot-observed>=observed-null_margin))/(len(boot)+1)
    lower=float(2*observed-np.quantile(boot,1-alpha/family_size))
    return {"status":"prototype_estimate","effect":observed,
            "effect_direction":"reference RMSE minus candidate RMSE; positive is improvement",
            "ci95_nominal":np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),
            "simultaneous_lower_bound_Bonferroni_approximate":lower,
            "declared_family_size":family_size,"p_one_sided_unadjusted":float(p),
            "null_margin":null_margin,"bootstrap_draws":len(boot),"block_length_days":block_length,
            "approximate_calendar_blocks":n/block_length,"minimum_MC_p":1/(len(boot)+1),
            "MC_tail_draws_expected_at_Bonferroni_alpha":len(boot)*alpha/family_size,
            "assumptions":"Approximate circular moving-block coverage under suitable weak temporal dependence; not exact under distribution shift",
            "seed_variability_included_as_new_market_samples":False,
            "formal_scientific_decision_requires_closed_protocol_and_family_adjustment":True}


def paired_seed_diagnostics(differences, *, unit_scope, repetitions=4999, seed=20261105):
    """Conditional training-randomness diagnostics, never a temporal-iid test."""
    np,stats=numeric()
    if unit_scope not in ("independent_units","training_seeds_conditional_fixed_calendar"):
        raise ValueError("Paired t/Wilcoxon need independent pairs; dates/folds cannot silently replace them")
    d=array(differences,"paired differences",1)
    if len(d)<2:raise ValueError("At least two pairs are required")
    sd=float(d.std(ddof=1));mean=float(d.mean());dz=mean/sd if sd>0 else None
    shapiro=stats.shapiro(d) if len(d)>=3 and sd>0 else None
    tt=stats.ttest_1samp(d,0.) if sd>0 else None
    if np.all(d==0):wilcox=None
    else:wilcox=stats.wilcoxon(d,zero_method="wilcox",alternative="two-sided",method="auto")
    rng=np.random.default_rng(seed);effects=[]
    for _ in range(repetitions):
        sample=d[rng.integers(0,len(d),size=len(d))];s=float(sample.std(ddof=1))
        if s>0:effects.append(float(sample.mean()/s))
    ci=np.quantile(effects,[.025,.975]).tolist() if len(effects)>=.95*repetitions else None
    nonzero=int(np.count_nonzero(d))
    return {"unit_scope":unit_scope,"n_pairs":len(d),"mean_difference":mean,
            "paired_Cohen_dz":dz,"paired_Cohen_dz_ci95":ci,
            "effect_CI_method":"pair bootstrap over independent units or conditional training seeds, not dates",
            "Shapiro_W":None if shapiro is None else float(shapiro.statistic),
            "Shapiro_p":None if shapiro is None else float(shapiro.pvalue),
            "paired_t":None if tt is None else float(tt.statistic),"paired_t_p":None if tt is None else float(tt.pvalue),
            "Wilcoxon_W":None if wilcox is None else float(wilcox.statistic),
            "Wilcoxon_p":None if wilcox is None else float(wilcox.pvalue),
            "Wilcoxon_min_two_sided_p_no_ties_nonzero_exact_case":2/(2**nonzero) if nonzero else None,
            "Wilcoxon_exact_resolution_bound_requires_no_ties":True,
            "test_not_selected_by_observed_Shapiro_p":True,
            "does_not_estimate_market_independence":True}


def levene_independent_groups(groups, *, independent_units_asserted):
    _,stats=numeric()
    if not independent_units_asserted:raise ValueError("Levene needs independent-unit design; paired model groups are not automatically independent")
    values=[array(x,"independent group",1) for x in groups]
    if len(values)<2 or any(len(x)<2 for x in values):raise ValueError("At least two independent groups with replication are required")
    result=stats.levene(*values,center="median")
    return {"method":"Levene/Brown-Forsythe, median centered","statistic":float(result.statistic),"p":float(result.pvalue),
            "does_not_test_temporal_independence":True,"not_a_required_equal_marginal_variance_assumption_for_paired_t":True}


def independent_unit_global_tests(scores, *, unit_kind, independent_units_asserted, assume_sphericity=False):
    """Generic future multi-dataset utility; invalid for the single BTC series."""
    np,stats=numeric();x=array(scores,"unit-by-model scores",2);n,k=x.shape
    if (not independent_units_asserted or unit_kind not in ("independent_datasets","independent_series","independent_subjects") or n<2 or k<3):
        raise ValueError("Friedman/RM-ANOVA need independent units; seeds/months/overlapping folds are not markets")
    friedman=stats.friedmanchisquare(*x.T)
    result={"independent_units":n,"models":k,"unit_kind":unit_kind,
            "Friedman_Q":float(friedman.statistic),"Friedman_p":float(friedman.pvalue),
            "posthoc_status":"Requires a predeclared comparison family; no automatic Nemenyi/CD diagram",
            "RM_ANOVA_status":"not_computed_without_sphericity_assumption"}
    if assume_sphericity:
        grand=x.mean();ss_total=float(((x-grand)**2).sum())
        ss_models=float(n*((x.mean(axis=0)-grand)**2).sum());ss_units=float(k*((x.mean(axis=1)-grand)**2).sum())
        residual=max(0.,ss_total-ss_models-ss_units);df1=k-1;df2=(n-1)*(k-1)
        f=(ss_models/df1)/(residual/df2) if residual>0 else None
        result.update({"RM_ANOVA_status":"sphericity_explicitly_assumed_not_verified",
                       "RM_ANOVA_F":f,"RM_ANOVA_p":None if f is None else float(stats.f.sf(f,df1,df2)),
                       "RM_ANOVA_df":[df1,df2]})
    return result


def dm_hln(loss_reference, loss_candidate, origins, *, horizon, hac_bandwidth, calendar_timezone="UTC"):
    """DM on a declared loss differential; not a test of mean-seed RMSE gain."""
    np,stats=numeric();daily_calendar(origins,calendar_timezone=calendar_timezone)
    a=array(loss_reference,"reference losses",1);b=array(loss_candidate,"candidate losses",1)
    if a.shape!=b.shape or len(a)!=len(origins):raise ValueError("Losses and dense paired calendar must match")
    n=len(a)
    if type(horizon) is not int or not 1<=horizon<n or type(hac_bandwidth) is not int or not 1<=hac_bandwidth<n:
        raise ValueError("Declare valid horizon and HAC bandwidth before results")
    d=a-b;center=d-d.mean();lrv=float(center@center/n)
    for lag in range(1,hac_bandwidth+1):lrv+=2*(1-lag/(hac_bandwidth+1))*float(center[lag:]@center[:-lag]/n)
    hln_square=(n+1-2*horizon+horizon*(horizon-1)/n)/n
    if lrv<=0 or hln_square<=0:
        return {"status":"not_estimable","statistic":None,"p_two_sided":None,"mean_loss_difference":float(d.mean()),"reason":"Nonpositive long-run variance or HLN factor"}
    statistic=float(d.mean()/math.sqrt(lrv/n)*math.sqrt(hln_square))
    return {"status":"prototype_diagnostic","statistic":statistic,"p_two_sided":float(2*stats.t.sf(abs(statistic),n-1)),
            "mean_loss_difference":float(d.mean()),"HAC_long_run_variance":lrv,"HAC_bandwidth":hac_bandwidth,
            "HAC_kernel":"Bartlett/Newey-West","HLN_factor":math.sqrt(hln_square),"horizon":horizon,"df":n-1,
            "loss_function_must_be_declared":True,
            "not_the_primary_nonlinear_RMSE_margin_test":True,
            "h1_does_not_remove_sigma7_label_overlap_dependence":True}


def ljung_box_diagnostics(residuals, origins, *, lags, model_df, calendar_timezone="UTC"):
    np,stats=numeric();daily_calendar(origins,calendar_timezone=calendar_timezone)
    e=array(residuals,"residuals",1)
    if len(e)!=len(origins) or type(model_df) is not int or model_df<0:raise ValueError("Residual calendar and explicit degrees of freedom are required")
    rows=[]
    for kind,series in (("residuals",e),("squared_residuals",e**2)):
        centered=series-series.mean();den=float(centered@centered);n=len(series)
        for lag in lags:
            if type(lag) is not int or not 1<=lag<n:raise ValueError("Every lag must be inside the sample")
            df=lag-model_df
            if den==0 or df<=0:statistic=p=None;status="not_estimable"
            else:
                acf=[float(centered[k:]@centered[:-k]/den) for k in range(1,lag+1)]
                statistic=float(n*(n+2)*sum(r*r/(n-k) for k,r in enumerate(acf,1)));p=float(stats.chi2.sf(statistic,df));status="prototype_diagnostic"
            rows.append({"series":kind,"lag":lag,"model_df":model_df,"df":df,"Q":statistic,"p_unadjusted":p,"status":status})
    return rows


def bds_optional(residuals, origins, *, max_dimension=3, calendar_timezone="UTC"):
    daily_calendar(origins,calendar_timezone=calendar_timezone);e=array(residuals,"residuals",1)
    if len(e)!=len(origins):raise ValueError("BDS requires a complete residual calendar")
    try:from statsmodels.tsa.stattools import bds
    except ImportError:
        return {"status":"not_available","statistic":None,"p":None,"reason":"Optional statsmodels is absent from the minimal NumPy/SciPy runtime; no installation is silently performed"}
    statistic,pvalue=bds(e,max_dim=max_dimension)
    return {"status":"optional_prototype_diagnostic","embedding_dimensions":list(range(2,max_dimension+1)),
            "statistic":clean(statistic),"p":clean(pvalue),
            "null":"iid; rejection does not uniquely diagnose nonlinear structure"}


def descriptive_ranks_and_pareto(rows, *, error_key="RMSE", cost_key="latency_ms_per_window"):
    np,stats=numeric();names=[r["model"] for r in rows]
    if len(names)!=len(set(names)):raise ValueError("Descriptive summaries require unique model rows")
    valid_error=[r for r in rows if r.get(error_key) is not None and math.isfinite(float(r[error_key]))]
    complete=[r for r in valid_error if r.get(cost_key) is not None and math.isfinite(float(r[cost_key]))]
    if any(float(r[error_key])<0 for r in valid_error) or any(float(r[cost_key])<0 for r in complete):raise ValueError("RMSE and cost cannot be negative")
    ranks=stats.rankdata([float(r[error_key]) for r in valid_error],method="average") if valid_error else []
    result=[]
    for row in rows:
        rank=float(ranks[valid_error.index(row)]) if row in valid_error else None
        if row not in complete:result.append({"model":row["model"],"rank":rank,"rank_of_aggregate_metric":rank,"rank_scope":"aggregate metric order, not mean paired-seed rank","Pareto_nondominated":None,"status":"not_estimable_Pareto_missing_cost_or_metric"});continue
        dominated=any(float(other[error_key])<=float(row[error_key]) and float(other[cost_key])<=float(row[cost_key]) and
                      (float(other[error_key])<float(row[error_key]) or float(other[cost_key])<float(row[cost_key])) for other in complete)
        result.append({"model":row["model"],"rank":rank,"rank_of_aggregate_metric":rank,"rank_scope":"aggregate metric order, not mean paired-seed rank","Pareto_nondominated":not dominated,
                       "status":"descriptive_point_estimate_only","not_a_population_superiority_or_CD_result":True})
    return result


def csv_bytes(rows, fields):
    stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode()


def model_registry(root=ROOT):
    source=json.loads((Path(root)/BENCHMARK_PROPOSAL).read_text());models=source["models"]
    if len(models)!=15 or len({m["id"] for m in models})!=15:raise ValueError("Expected the minimum fifteen-model reference catalogue")
    return models


def write_templates(root=ROOT, output_dir=None):
    root=Path(root);out=Path(output_dir) if output_dir is not None else root/OUTPUT;out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/PROPOSAL).read_text());models=model_registry(root)
    reference=json.loads((root/"book_e2/_static/e1_reference.json").read_text())
    if (plan["status"]!="proposed_not_frozen" or plan["new_Bitcoin_results"]!=0 or
            plan["primary_metric"]!="mean_seed_RMSE_original_scale" or
            plan["absolute_margin"]!=DELTA or reference["source_commit"]!=SOURCE_COMMIT or
            reference["practical_margin"]["absolute_delta"]!=DELTA):
        raise ValueError("Analysis proposal status or inherited scientific contract differs")
    # Validate the complete pinned scientific snapshot and its six public E1
    # receipts; existence of a matching delta alone is not enough.
    from audit_e2_benchmark_readiness import validate as validate_preflight
    benchmark=json.loads((root/BENCHMARK_PROPOSAL).read_text())
    preflight,_,_=validate_preflight(benchmark,root)
    guide=json.loads((root/"book_e2/_static/guia_e2_recibida.json").read_text())
    requirements=[{"id":r["id"],"requirement":r["requirement"],"status":"pending_benchmark_execution_and_protocol_closure"}
                  for r in guide.get("section_6_requirements",[])]
    if not requirements:raise ValueError("Received section6 requirement registry is missing")
    fields=["model","role","status","eligible_origins","runs","RMSE_mean","RMSE_SD","RMSE_CI95_low","RMSE_CI95_high","MAE_mean","MSE_mean","R2_mean","MAPE_percent","MAPE_coverage","SMAPE_percent","SMAPE_coverage","MASE_mean","training_seconds","latency_ms_per_window","peak_memory_bytes","parameters","FLOPs_per_window","scope"]
    table=[{**{k:"" for k in fields},"model":m["id"],"role":m["role"],"status":"pending_not_trained_in_D7_benchmark","scope":"D7 benchmark template; not external ETTh1 replica results"} for m in models]
    schemas={"prediction_panel":{"fields":PREDICTION_FIELDS,"dense_key":["model","seed","origin_utc","horizon"],"scale":"original unannualized daily volatility","calendar":"explicit assigned daily UTC origins including missing-eligibility placeholders","horizon":1},
             "runs":{"fields":RUN_FIELDS,"hash_semantics":"64hex SHA-256 receipts identify input bytes, not scientific correctness or actual fairness"},
             "costs":{"fields":COST_FIELDS,"measurement_unit":"interleaved timing block; not a new predictive-market sample","required_comparability":["device","precision","threads","batch_size","input_shape","timing_protocol_sha256"]},
             "calendar":{"fields":["timezone","frequency","origins","seeds","scope"],"timezone":"UTC","frequency":"D"},
             "metric_coverage":"MAPE only positive truth; SMAPE excludes both-zero denominator; no arbitrary epsilon; MASE denominator external train-only"}
    report={"schema_version":1,"status":"proposed_not_frozen","analysis_kind":"templates_and_synthetic-tested_numerical_primitives_not_benchmark_results",
            "benchmark_models":15,"Top10_candidates":10,"independent_market_series":1,
            "new_Bitcoin_results":0,"benchmark_executed":False,"D7_models_fitted_by_this_tool":0,"real_predictions_loaded":False,
            "numerical_rankings_published":False,"actual_test_score_table_available":False,
            "primary_metric":"mean per-seed RMSE in original units","absolute_margin":DELTA,
            "inherited_scientific_contract_validated":preflight["proposal_valid"],
            "benchmark_ready_to_train":preflight["ready_to_train"],
            "benchmark_preflight_blockers":len(preflight["blockers"]),
            "pinned_E1_source_hashes_verified":preflight["pinned_E1_source_hashes_verified"],
            "protected_scientific_snapshot_sha256":preflight["inherited_scientific_contract_sha256"],
            "received_section6_requirements":len(requirements),"proposed_family_size":30,
            "all_15_pair_count_context_only_not_adopted_family":105,
            "primary_family_status":"draft ten Top10 candidates versus GRU across RQ1/RQ2/RQ3; not frozen",
            "implemented_prototype_functions":["dense paired calendar validation","RMSE/MAE/MSE/R2/MAPE/SMAPE/train-only MASE","paired dense-calendar bootstrap margin effects","conditional seed effects/t/Wilcoxon/Shapiro","Levene for genuinely independent groups","DM-HLN with explicit HAC","Ljung-Box residuals and squares","optional BDS if statsmodels present","mean within-seed descriptive ranks, aggregate-metric ranks and Pareto kept distinct","guarded independent-unit Friedman/RM-ANOVA"],
            "not_completed":["D7 benchmark fits/predictions/costs","formal leakage/overfit/robustness/calibration execution","power and sample size","closed inference families and variance decomposition","RQ2 production regime bootstrap and joint latency/error interval generation for RQ3","simultaneous production scientific decisions; numerical primitives are fixture-tested only","MCS/Nemenyi/Bayesian ROPE implementation if justified","BDS runtime unless optional dependency is present"],
            "single_series_inference_limits":{"Friedman_CD_diagram":"not produced from seeds or overlapping months; requires independent units and justified posthoc design",
                                              "Nadeau_Bengio":"not applied automatically to expanding temporal folds; its random-resampling variance formula is not justified by this calendar",
                                              "non_rejection":"does not establish equivalence or absence of residual structure"},
            "external_ETTh1_section5_not_Bitcoin_results":True}
    readme="""# Análisis del benchmark E2: propuesta y plantillas

Estado propuesto, no congelado. Los quince renglones son modelos pendientes;
las celdas numéricas vacías no significan error cero. Las corridas ETTh1 del
capítulo5 no se convierten en resultados del benchmark Bitcoin D7.

Desde la raíz: `python scripts/analyze_e2_benchmark.py`. Este modo sólo usa la
biblioteca estándar y no abre predicciones ni datos de mercado. Las funciones
numéricas usan NumPy/SciPy y se prueban con fixtures sintéticos identificados:
`python tests/test_e2_benchmark_analysis.py`.

El CLI con --panel/--runs/--costs/--calendar sólo valida insumos declarados
`--scope synthetic_fixture` y exige una carpeta privada --output-dir fuera del
libro. No publica rankings de los fixtures ni acepta un análisis real mientras
el protocolo esté sin congelar. Los hashes identifican archivos y no prueban
por sí solos entrenamiento justo, ausencia de fuga o validez estadística.

Las primitivas permiten evaluar métricas, efectos y diagnósticos futuros; no
son evidencia de su ejecución sobre Bitcoin. La primaria promedia las RMSE
por semilla; el bootstrap usa el calendario completo emparejado y conserva las
trayectorias, sin convertir semillas o ventanas solapadas en mercados nuevos.
DM contrasta la pérdida declarada, no la misma hipótesis no lineal de RMSE y δ.
Las familias, supuestos, potencia, coste y código de producción deben cerrarse
antes del scoring. La reserva heredada sigue conocida y exploratoria.
"""
    contents={"report.json":json_bytes(report),"table_template.csv":csv_bytes(table,fields),
              "requirements.csv":csv_bytes(requirements,["id","requirement","status"]),
              "predictions_template.csv":csv_bytes([],PREDICTION_FIELDS),"runs_template.csv":csv_bytes([],RUN_FIELDS),
              "costs_template.csv":csv_bytes([],COST_FIELDS),"schemas.json":json_bytes(schemas),
              "proposal.json":json_bytes(plan),"README.md":readme.encode()}
    inputs=[PROPOSAL,BENCHMARK_PROPOSAL,"book_e2/_static/e1_reference.json","book_e2/_static/dataset_card.json","book_e2/_static/guia_e2_recibida.json","scripts/audit_e2_benchmark_readiness.py"]
    contents["provenance.json"]=json_bytes({"generator":"scripts/analyze_e2_benchmark.py","generator_sha256":digest(__file__),
                                         "input_hashes":{p:digest(root/p) for p in inputs},
                                         "output_hashes":{k:sha256(v).hexdigest() for k,v in contents.items()},
                                         "market_files_opened":0,"real_benchmark_predictions_loaded":0,"model_fits":0})
    for name,payload in contents.items():(out/name).write_bytes(payload)
    return report


def validate_fixture_inputs(args):
    if args.scope!="synthetic_fixture":raise ValueError("Real analysis is blocked: the proposal is not frozen; only synthetic_fixture validation is available")
    if not args.output_dir:raise ValueError("Supply an explicit private fixture output directory")
    out=Path(args.output_dir).resolve()
    if out.is_relative_to(ROOT.resolve()):
        raise ValueError("Synthetic fixture outputs must remain outside the repository and all published books")
    def rows(path):
        with Path(path).open(newline="") as stream:return list(csv.DictReader(stream))
    runs=rows(args.runs);costs=rows(args.costs);panel_rows=rows(args.panel)
    calendar=json.loads(Path(args.calendar).read_text())
    if calendar.get("scope")!="synthetic_fixture" or calendar.get("frequency")!="D":raise ValueError("Fixture calendar scope/frequency differ")
    expected_models=[m["id"] for m in model_registry()]
    for group in (runs,costs,panel_rows):
        if not group or any(r.get("scope")!="synthetic_fixture" for r in group):raise ValueError("Every input row must declare synthetic_fixture scope")
    by_id={}
    seeds=[str(x) for x in calendar["seeds"]]
    expected_pairs={(m,s) for m in expected_models for s in (["deterministic"] if m=="Persistence" else seeds)}
    run_pairs=set();dataset_split=set()
    for r in runs:
        if r["run_id"] in by_id:raise ValueError("Duplicate run identity")
        pair=(r["model"],r["seed"])
        if pair not in expected_pairs or pair in run_pairs:raise ValueError("Run model/seed set is unregistered or duplicated")
        run_pairs.add(pair)
        if r["phase"] not in ("development","validation","synthetic_fixture"):raise ValueError("This fixture prototype does not accept a new test or confirmatory phase")
        if any(not HASH.fullmatch(r.get(k,"")) for k in ("dataset_sha256","split_sha256","config_sha256")):
            raise ValueError("Run provenance must identify each source by SHA-256")
        if r["model"]=="Persistence":
            if r.get("checkpoint_sha256") not in (None,""):raise ValueError("Deterministic persistence has no trained checkpoint")
        elif not HASH.fullmatch(r.get("checkpoint_sha256","")):raise ValueError("Learned run checkpoint hash is required")
        dataset_split.add((r["dataset_sha256"],r["split_sha256"]))
        by_id[r["run_id"]]=r
    if run_pairs!=expected_pairs or len(dataset_split)!=1:raise ValueError("Complete registered runs must share dataset and split identities")
    for r in panel_rows:
        if r["run_id"] not in by_id or (r["model"],r["seed"])!=(by_id[r["run_id"]]["model"],by_id[r["run_id"]]["seed"]):
            raise ValueError("Panel run/model/seed identities differ")
    panel=validate_dense_panel(panel_rows,expected_models,seeds,calendar["origins"],calendar_timezone=calendar["timezone"],deterministic_models=["Persistence"])
    contexts={tuple(r.get(k) for k in ("device","precision","threads","batch_size","input_shape","timing_protocol_sha256")) for r in costs}
    if len(contexts)!=1 or any(x in (None,"") for x in next(iter(contexts))):raise ValueError("Cost contexts are incomplete or unequal")
    cost_keys=set();blocks=set()
    for r in costs:
        key=(r["model"],r["seed"],r["measurement_block"])
        if key[:2] not in expected_pairs or key in cost_keys:raise ValueError("Cost model/seed/block identity is unregistered or duplicated")
        cost_keys.add(key);blocks.add(r["measurement_block"])
        if not HASH.fullmatch(r["timing_protocol_sha256"]):raise ValueError("Timing protocol hash is invalid")
        if not math.isfinite(float(r["latency_ms_per_window"])) or float(r["latency_ms_per_window"])<0:raise ValueError("Invalid latency measurement")
    if cost_keys!={(m,s,b) for m,s in expected_pairs for b in blocks}:raise ValueError("Every registered run needs the same complete timing-block set")
    out.mkdir(parents=True,exist_ok=True)
    report={"scope":"synthetic_fixture","panel_valid":True,"models":len(panel["models"]),"paired_seeds":len(panel["seeds"]),
            "calendar_origins":panel["n_calendar_origins"],"eligible_origins":panel["n_eligible_origins"],
            "metric_values_published":False,"rankings_published":False,"real_Bitcoin_results":0,
            "input_hashes":{k:digest(getattr(args,k)) for k in ("panel","runs","costs","calendar")},
            "scope_note":"Schema and pairing validated only; source hashes do not establish scientific execution, fairness or leakage absence"}
    (out/"fixture_validation.json").write_bytes(json_bytes(report));return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir")
    for name in ("panel","runs","costs","calendar"):p.add_argument("--"+name)
    p.add_argument("--scope",default="proposal")
    args=p.parse_args(argv)
    try:
        inputs=[args.panel,args.runs,args.costs,args.calendar]
        if any(inputs):
            if not all(inputs):raise ValueError("Fixture validation requires panel, runs, costs and calendar together")
            report=validate_fixture_inputs(args)
        else:
            if args.scope!="proposal":raise ValueError("Without inputs use proposal scope")
            report=write_templates(output_dir=args.output_dir)
        print(json.dumps({k:report[k] for k in ("status","benchmark_models","new_Bitcoin_results","scope","panel_valid") if k in report},ensure_ascii=False));return 0
    except (ValueError,KeyError,OSError,TypeError) as error:print(str(error),file=sys.stderr);return 1


if __name__=="__main__":raise SystemExit(main())
