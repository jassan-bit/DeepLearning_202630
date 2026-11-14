"""Retrospective causal interval calibration on saved E2 point forecasts.

No scientific model is fitted or evaluated here. Original forecasts and the
frozen 53-residual calibrators are verified and never overwritten. Online
feedback wrappers issue an interval before observing that day's outcome.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from e2_reduced_protocol import SYSTEMS, NEURAL

VERSION = "e2-conditional-calibration-v1"
INPUT_SHA = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
DEV_SHA = "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a"
SEEDS = tuple(range(271801, 271807))
POLICIES = ("original_frozen", "rolling53_fixed_alpha", "ACI53_gamma005")
PLAN_COMMIT = "8d9cc8c3397242a07378fb1a678196dab36744da"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def quantile_radius(scores, alpha):
    """Finite-sample split-score rank, with honest full/empty extensions.

    This uses ceil((n+1)*(1-alpha)), preserving the original rank49/53.
    ACI is the feedback rule, not a claim of replicating the paper's entire
    forecasting experiment or its empirical n-based quantile convention.
    """
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Finite nonnegative prior conformity scores required")
    if not math.isfinite(alpha):
        raise ValueError("Finite adaptive alpha required")
    if alpha <= 0:
        return math.inf, None
    if alpha >= 1:
        return -math.inf, None
    rank = int(math.ceil((len(x)+1)*(1-alpha)))
    if rank > len(x):
        return math.inf, rank
    return float(np.sort(x)[rank-1]), rank


def interval_metrics(prediction, target, radius):
    """Full sets and empty sets cannot become spuriously narrow finite CIs."""
    if not all(math.isfinite(v) for v in (prediction, target)) or target < 0:
        raise ValueError("Nonnegative finite target and finite prediction required")
    if radius == math.inf:
        return {"state": "full", "lower": 0., "upper": math.inf, "hit": True,
                "width": math.inf, "interval_score": math.inf,
                "pinball05": .05*target, "pinball95": math.inf}
    if radius == -math.inf:
        return {"state": "empty", "lower": None, "upper": None, "hit": False,
                "width": math.nan, "interval_score": math.nan,
                "pinball05": math.nan, "pinball95": math.nan}
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("Finite residual radius must be nonnegative")
    lower, upper = max(0., prediction-radius), prediction+radius
    if upper < lower:
        raise ValueError("Frozen forecast cannot form a nonnegative interval")
    hit = lower <= target <= upper
    width = upper-lower
    score = width+20*max(lower-target, 0)+20*max(target-upper, 0)
    u05, u95 = target-lower, target-upper
    return {"state": "finite", "lower": lower, "upper": upper, "hit": hit,
            "width": width, "interval_score": score,
            "pinball05": max(.05*u05, -.95*u05),
            "pinball95": max(.95*u95, -.05*u95)}


def simulate(predictions, targets, dates, calibration_predictions, calibration_targets,
             calibration_dates, policy, frozen_q, *, target_alpha=.1, gamma=.005, window=53):
    """The new target is read only after the interval is committed to trace.

    Predetermined point forecasts may depend on their observed causal inputs.
    This program never re-estimates those point forecasts. ACI feedback is
    allowed to use previous test residuals, never the current/future residual.
    """
    if policy not in POLICIES:
        raise ValueError("Unknown separately declared interval policy")
    if len(predictions) != len(targets) or len(dates) != len(targets):
        raise ValueError("Dated trajectories must align")
    history = list(zip(calibration_dates, np.abs(np.asarray(calibration_targets)-np.asarray(calibration_predictions))))
    if len(history) != window or len(history) != len(calibration_predictions):
        raise ValueError("Exactly the original 53 dated calibration scores required")
    if not all(history[i][0] < history[i+1][0] for i in range(len(history)-1)):
        raise ValueError("Calibration dates must increase")
    alpha = target_alpha
    trace = []
    for day, prediction, target in zip(dates, predictions, targets):
        if history[-1][0] >= day:
            raise ValueError("A current/future label entered calibration")
        used_alpha = alpha if policy == "ACI53_gamma005" else target_alpha
        if policy == "original_frozen":
            radius, rank = frozen_q, 49
        else:
            radius, rank = quantile_radius([s for _,s in history[-window:]], used_alpha)
        # At this point the interval has been selected without using target.
        selected_lower = None if radius == -math.inf else max(0., prediction-radius)
        selected_upper = None if radius == -math.inf else prediction+radius
        outcome = interval_metrics(float(prediction), float(target), radius)
        if selected_lower != outcome["lower"] or selected_upper != outcome["upper"]:
            raise AssertionError("Interval selected after observing target")
        next_alpha = alpha+gamma*(target_alpha-int(not outcome["hit"])) if policy == "ACI53_gamma005" else target_alpha
        trace.append({"date": day, "prediction": float(prediction), "target": float(target),
                      "history_latest_date": history[-1][0], "history_size": min(len(history), window),
                      "alpha_used": used_alpha, "alpha_next": next_alpha, "radius": radius,
                      "rank": rank, **outcome})
        if policy != "original_frozen":
            history.append((day, abs(float(target)-float(prediction))))
            history = history[-window:]
        alpha = next_alpha
    if policy == "ACI53_gamma005" and trace:
        errors = sum(not r["hit"] for r in trace)
        expected = target_alpha+gamma*(len(trace)*target_alpha-errors)
        if not math.isclose(alpha, expected, abs_tol=5e-13, rel_tol=0):
            raise AssertionError("ACI drift identity broken")
        if not all(-gamma-1e-12 <= r["alpha_used"] <= 1+gamma+1e-12 for r in trace):
            raise AssertionError("ACI alpha extension bound broken")
    return trace


def full_calendar_lag(dates, targets, development):
    lookup = {pd.Timestamp(d).date().isoformat():float(y) for d,y in
              zip(development.market_day_utc, development.sigma_7) if pd.notna(y)}
    # Test targets are used only for the next day's observable subgroup.
    lookup.update(zip(dates, map(float, targets)))
    values = []
    for d in dates:
        previous = (date.fromisoformat(d)-timedelta(days=1)).isoformat()
        values.append(lookup.get(previous, math.nan))
    return np.asarray(values)


def three_ranges(values, cuts):
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    return (valid&(x <= cuts[0]), valid&(x > cuts[0])&(x <= cuts[1]), valid&(x > cuts[1]))


def hac_variance(values, lags=14):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) <= lags+1 or not np.isfinite(x).all():
        raise ValueError("Sufficient complete calendar sequence required")
    z = x-x.mean();n = len(x)
    variance = float(z@z/n)
    for lag in range(1, lags+1):
        variance += 2*(1-lag/(lags+1))*float(z[lag:]@z[:-lag])/n
    return max(variance, 0.)


def coverage_tests(numerator, denominator, *, minimum_dates=20, nominal=.9, lags=14):
    num, den = np.asarray(numerator,dtype=float), np.asarray(denominator,dtype=float)
    support = int((den > 0).sum())
    coverage = float(num.sum()/den.sum()) if den.sum() else None
    result = {"coverage":coverage,"calendar_days_with_membership":support,
              "z_HAC14":None,"p_equal90":None,"p_minimum90":None,
              "estimable":False,"test_status":"insufficient_support"}
    if support < minimum_dates or len(num) <= lags+1:
        return result
    # Delta-method ratio influence: estimate conditional coverage first, then
    # retain the FULL calendar with zero membership on unselected dates.
    influence = num-coverage*den
    variance = hac_variance(influence, lags)
    if variance <= np.finfo(float).eps*max(float(np.mean(influence*influence)),np.finfo(float).tiny):
        result["test_status"]="degenerate_variance_no_probability_claim"
        return result
    z = float((coverage-nominal)*den.mean()/math.sqrt(variance/len(influence)))
    result.update(z_HAC14=z,p_equal90=float(2*stats.norm.sf(abs(z))),
                  p_minimum90=float(stats.norm.cdf(z)),estimable=True,
                  test_status="asymptotic_calendar_HAC_requires_weak_dependence")
    return result


def holm_rows(rows):
    order = sorted(range(len(rows)), key=lambda i:rows[i]["p_raw_for_family"])
    previous = 0.
    for rank, idx in enumerate(order):
        previous = max(previous,min(1.,(len(rows)-rank)*rows[idx]["p_raw_for_family"]))
        rows[idx]["p_Holm1344"] = previous if rows[idx]["estimable"] else None
        rows[idx]["reject_FWER05"] = previous < .05 if rows[idx]["estimable"] else None
    return rows


def block_bootstrap_ratios(numerators, denominators, *, draws=10000, block_days=14,
                           seed=2026111411, chunk=96):
    """Fixed realized traces, joint circular calendar blocks; no policy rerun.

    Conditional coverage uses a ratio. Bootstrap draws with no group members
    are counted and reported, never silently replaced by zero or called a
    nominal unconditional interval. All-zero groups remain not estimable.
    """
    num,den=np.asarray(numerators,dtype=float),np.asarray(denominators,dtype=float)
    if num.ndim != 2 or num.shape != den.shape or draws < 10000:
        raise ValueError("At least10000 aligned calendar bootstrap draws required")
    if not np.isfinite(num).all() or not np.isfinite(den).all() or np.any(den<0):
        raise ValueError("Finite nonnegative membership counts required")
    n,c=num.shape
    values=np.concatenate((num,den),axis=1)
    block_sum=np.zeros_like(values)
    for offset in range(block_days):block_sum += np.roll(values,-offset,axis=0)
    full,tail=divmod(n,block_days)
    tail_sum=np.zeros_like(values)
    for offset in range(tail):tail_sum += np.roll(values,-offset,axis=0)
    result=np.full((draws,c),np.nan);rng=np.random.default_rng(seed)
    for start in range(0,draws,chunk):
        count=min(chunk,draws-start)
        starts=rng.integers(0,n,size=(count,full+int(bool(tail))))
        sums=block_sum[starts[:,:full]].sum(axis=1) if full else np.zeros((count,2*c))
        if tail:sums += tail_sum[starts[:,-1]]
        a,b=sums[:,:c],sums[:,c:]
        np.divide(a,b,out=result[start:start+count],where=b>0)
    rows=[]
    for col in range(c):
        valid=result[:,col][np.isfinite(result[:,col])];missing=draws-len(valid)
        ci=np.quantile(valid,[.025,.975]).tolist() if len(valid) else [None,None]
        rows.append({"CI95_low":ci[0],"CI95_high":ci[1],"undefined_bootstrap_draws":missing,
                     "CI_status":"marginal_temporal_percentile" if missing==0 else
                        "conditional_on_nonempty_resampled_group" if len(valid) else "not_estimable",
                     "draws":draws,"block_days":block_days,"seed":seed})
    return rows


def average_metric(values):
    x=np.asarray(values,dtype=float)
    if np.isnan(x).any():return None,"undefined_empty_prediction_set"
    if np.isinf(x).any():return None,"infinite_full_prediction_set"
    return float(x.mean()),"finite"


def validate_input(payload):
    dates=payload["dates"];y=np.asarray(payload["y_true"],dtype=float)
    expected=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(608)]
    if dates != expected or y.shape != (608,) or not np.isfinite(y).all() or np.any(y<0):
        raise ValueError("The exact complete nonnegative608-day calendar is required")
    if payload["mask"] != [True]*608 or len(payload["predictors"]) != 81:
        raise ValueError("Frozen support/predictor inventory changed")
    seen=set();calibration_targets=None
    for row in payload["predictors"]:
        key=(row["model"],row["seed"])
        if key in seen:raise ValueError("Duplicate predictor")
        seen.add(key)
        expected_seeds=SEEDS if row["model"] in NEURAL else ("deterministic",)
        if row["seed"] not in expected_seeds:raise ValueError("Seed mismatch")
        cp,cy=np.asarray(row["calibration_predictions"]),np.asarray(row["calibration_targets"])
        cd=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
        if row["calibration_dates"] != cd or cp.shape != (53,) or cy.shape != (53,):
            raise ValueError("Original calibration support changed")
        if calibration_targets is None:calibration_targets=cy
        elif not np.array_equal(cy,calibration_targets):raise ValueError("Calibration targets not shared")
        q,rank=quantile_radius(np.abs(cp-cy),.1)
        if rank != 49 or not math.isclose(q,row["calibrator_q"],abs_tol=1e-15,rel_tol=0):
            raise ValueError("Original49th score calibration differs")
        if len(row["prediction_original"]) != 608 or not np.isfinite(row["prediction_original"]).all():
            raise ValueError("Incomplete frozen point forecasts")
    for model in SYSTEMS:
        expected=SEEDS if model in NEURAL else ("deterministic",)
        if {(m,s) for m,s in seen if m==model} != {(model,s) for s in expected}:
            raise ValueError("Incomplete model inventory")
    return dates,y


def validate_plan(plan):
    if plan["version"] != VERSION or plan["input_sha256"] != INPUT_SHA or plan["development_sha256"] != DEV_SHA:
        raise ValueError("Matching registered plan required")
    if plan["new_policies"][1]["gamma"] != .005 or plan["new_policies"][1]["alpha_clipped"]:
        raise ValueError("ACI policy changed")
    if plan["inference"]["draws"] != 10000 or plan["inference"]["block_days"] != 14:
        raise ValueError("Registered inference settings changed")
    if [v["name"] for v in plan["new_policies"]] != list(POLICIES[1:]):
        raise ValueError("Registered policies changed")


def run(input_path, development_path, plan_path, output):
    started=time.monotonic();output=Path(output)
    if output.exists():raise FileExistsError("Use a new supplemental output directory")
    if digest(input_path)!=INPUT_SHA or digest(development_path)!=DEV_SHA:
        raise ValueError("Frozen public input hash changed")
    plan=json.loads(Path(plan_path).read_text());validate_plan(plan)
    payload=json.loads(Path(input_path).read_text());dates,y=validate_input(payload)
    development=pd.read_csv(development_path)
    finite=development.loc[np.isfinite(development.sigma_7),"sigma_7"].to_numpy()
    lagcuts=np.quantile(finite,[.25,.75]);lagged=full_calendar_lag(dates,y,development)
    laggroups=three_ranges(lagged,lagcuts)
    quarters=np.asarray([f"{d[:4]}Q{(int(d[5:7])-1)//3+1}" for d in dates])
    quarter_names=list(dict.fromkeys(quarters));outputs={};trace_rows=[];predictor_rows=[];threshold_rows=[];calibration_diagnostics=[]
    for row in payload["predictors"]:
        model,seed=row["model"],row["seed"]
        cuts=np.quantile(row["calibration_predictions"],[.25,.75])
        residuals=np.abs(np.asarray(row["calibration_targets"])-np.asarray(row["calibration_predictions"]))
        lag_correlations={}
        for lag in (1,7):
            a,b=residuals[lag:],residuals[:-lag]
            lag_correlations["calibration_abs_residual_Pearson_lag"+str(lag)]=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
        calibration_diagnostics.append({"model":model,"seed":seed,"calibration_n":53,
            "frozen_rank":49,"verified_q":float(np.sort(residuals)[48]),
            "seven_day_nonoverlapping_complete_blocks":53//7,
            "corrected_90pct_block_rank_with7blocks":int(math.ceil((53//7+1)*.9)),
            "finite_90pct_distribution_free_block_quantile_available_with7blocks":False,**lag_correlations})
        threshold_rows.append({"model":model,"seed":seed,"prediction_Q25_calibration53":float(cuts[0]),
            "prediction_Q75_calibration53":float(cuts[1]),"lagged_sigma7_Q25_development":float(lagcuts[0]),
            "lagged_sigma7_Q75_development":float(lagcuts[1])})
        groups={"global":np.ones(608,dtype=bool)}
        groups.update({"prediction_"+label:g for label,g in zip(("low","middle","high"),three_ranges(row["prediction_original"],cuts))})
        groups.update({"lagged_volatility_"+label:g for label,g in zip(("low","middle","high"),laggroups)})
        groups.update({"quarter_"+q:quarters==q for q in quarter_names})
        for policy in POLICIES:
            trace=simulate(row["prediction_original"],y,dates,row["calibration_predictions"],
                row["calibration_targets"],row["calibration_dates"],policy,row["calibrator_q"])
            outputs[(model,seed,policy)]={"groups":groups,"trace":trace}
            metrics={}
            for m in ("width","interval_score","pinball05","pinball95"):
                value,status=average_metric([t[m] for t in trace]);metrics["mean_"+m]=value;metrics[m+"_status"]=status
            predictor_rows.append({"model":model,"seed":seed,"policy":policy,
                "coverage":float(np.mean([r["hit"] for r in trace])),"n_dates":608,
                "full_sets":sum(r["state"]=="full" for r in trace),"empty_sets":sum(r["state"]=="empty" for r in trace),
                "alpha_min":min(r["alpha_used"] for r in trace),"alpha_max":max(r["alpha_used"] for r in trace),
                "final_alpha":trace[-1]["alpha_next"],"all_history_dates_strictly_before_origin":all(r["history_latest_date"]<r["date"] for r in trace),**metrics})
            for t in trace:trace_rows.append({"model":model,"seed":seed,"policy":policy,**t})
        print(json.dumps({"event":"saved_predictor_interval_audit_completed","model":model,"seed":seed}),flush=True)
    output.mkdir(parents=True)
    write_csv(output/"predictor_metrics.csv",predictor_rows)
    write_csv(output/"group_thresholds.csv",threshold_rows)
    # The immutable input already supplies y and point predictions. Keep the
    # minimal auditable causal policy trace rather than repeating those arrays.
    fields=["model","seed","policy","date","history_latest_date","history_size",
            "alpha_used","alpha_next","radius","rank","state","hit"]
    raw=io.StringIO(newline="");writer=csv.DictWriter(raw,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(trace_rows)
    (output/"daily_interval_trace.csv.gz").write_bytes(gzip.compress(raw.getvalue().encode(),mtime=0))
    cells=[];family=[];group_seed_rows=[]
    group_names=["global",*("prediction_"+s for s in ("low","middle","high")),
                 *("lagged_volatility_"+s for s in ("low","middle","high")),*("quarter_"+q for q in quarter_names)]
    for model in SYSTEMS:
        seeds=SEEDS if model in NEURAL else ("deterministic",)
        for policy in POLICIES:
            selected=[outputs[(model,seed,policy)] for seed in seeds]
            for group in group_names:
                member=np.asarray([r["groups"][group] for r in selected])
                hits=np.asarray([[t["hit"] for t in r["trace"]] for r in selected])
                num=np.mean(member*hits,axis=0);den=np.mean(member,axis=0)
                temporal=quarters==group.removeprefix("quarter_") if group.startswith("quarter_") else np.ones(608,dtype=bool)
                tests=coverage_tests(num[temporal],den[temporal])
                key={"model":model,"policy":policy,"group":group}
                metrics={}
                for metric in ("width","interval_score","pinball05","pinball95"):
                    values=np.asarray([[t[metric] for t in r["trace"]] for r in selected],dtype=float)[member]
                    value,status=average_metric(values) if len(values) else (None,"empty_group")
                    metrics["mean_"+metric]=value;metrics[metric+"_status"]=status
                cell={**key,"R":len(seeds),"individual_seed_date_memberships":int(member.sum()),
                    "full_sets":sum(t["state"]=="full" for r in selected for t,g in zip(r["trace"],r["groups"][group]) if g),
                    "empty_sets":sum(t["state"]=="empty" for r in selected for t,g in zip(r["trace"],r["groups"][group]) if g),
                    **tests,**metrics,"_numerator":num,"_denominator":den,"_temporal":temporal}
                cells.append(cell)
                for null,column in (("coverage_equals90","p_equal90"),("coverage_at_least90","p_minimum90")):
                    family.append({**key,"null":null,"p_raw":tests[column],
                        "p_raw_for_family":tests[column] if tests["estimable"] else 1.,"estimable":tests["estimable"]})
                for seed, r, g, h in zip(seeds,selected,member,hits):
                    group_seed_rows.append({**key,"seed":seed,"n_dates":int(g.sum()),
                        "covered":int((g*h).sum()),"coverage":float(h[g].mean()) if g.sum() else None})
    if len(cells)!=672 or len(family)!=1344:raise AssertionError("Registered hypothesis family changed")
    holm_rows(family);write_csv(output/"coverage_HAC_Holm1344.csv",family)
    ci_rows=[]
    for block in (7,14,28):
        for local_id,quarter in enumerate([None,*quarter_names]):
            selected=[c for c in cells if (not c["group"].startswith("quarter_")) == (quarter is None)
                      and (quarter is None or c["group"]=="quarter_"+quarter)]
            mask=np.ones(608,dtype=bool) if quarter is None else quarters==quarter
            num=np.stack([c["_numerator"][mask] for c in selected],axis=1)
            den=np.stack([c["_denominator"][mask] for c in selected],axis=1)
            results=block_bootstrap_ratios(num,den,draws=10000,block_days=block,
                seed=2026111411+block*100+local_id)
            for cell,result in zip(selected,results):
                row={k:cell[k] for k in ("model","policy","group","coverage","calendar_days_with_membership")}
                ci_rows.append({**row,**result,"resampling_scope":"within_contiguous_quarter" if quarter else "full608calendar_joint_methods_and_seeds_fixed"})
        print(json.dumps({"event":"conditional_coverage_block_IC_completed","block_days":block}),flush=True)
    write_csv(output/"conditional_coverage_CI.csv",ci_rows)
    write_csv(output/"conditional_metrics.csv",[{k:v for k,v in c.items() if not k.startswith("_")} for c in cells])
    write_csv(output/"predictor_group_coverage.csv",group_seed_rows)
    write_csv(output/"calibration_support_diagnostics.csv",calibration_diagnostics)
    # Realized y regimes are visibly separate: they are NOT observable at the
    # prediction origin and do not define the primary coverage-test family.
    realized_rows=[]
    for model in SYSTEMS:
        seeds=SEEDS if model in NEURAL else ("deterministic",)
        for policy in POLICIES:
            selected=[outputs[(model,seed,policy)] for seed in seeds]
            for label,g in zip(("low","middle","high"),three_ranges(y,lagcuts)):
                h=np.asarray([[t["hit"] for t in r["trace"]] for r in selected])
                realized_rows.append({"model":model,"policy":policy,"observed_target_regime":label,
                    "n_calendar_dates":int(g.sum()),"individual_seed_date_cases":int(g.sum()*len(seeds)),
                    "coverage":float(h[:,g].mean()) if g.sum() else None,
                    "observable_at_prediction_origin":False,"included_in_primary_coverage_test_family":False})
    write_csv(output/"realized_target_regimes_descriptive.csv",realized_rows)
    models=[]
    for cell in cells:
        if cell["group"]=="global":models.append({k:v for k,v in cell.items() if not k.startswith("_")})
    write_csv(output/"model_metrics.csv",models)
    dump(output/"analysis_plan.json",plan)
    summary={"version":VERSION,"analysis_status":plan["analysis_status"],"plan_registration_commit":PLAN_COMMIT,
        "systems":16,"frozen_predictors":81,"policies":list(POLICIES),"new_point_model_fits":0,
        "new_point_model_inferences":0,"original_A2_replaced":False,"original_calibrators_replaced":False,
        "new_interval_policies_selected_from_results":False,"new_blind_test_claimed":False,
        "original49of53_quantiles_verified":81,"causal_trace_rows":len(trace_rows),
        "all_histories_strictly_before_origin":all(r["all_history_dates_strictly_before_origin"] for r in predictor_rows),
        "model_group_policy_cells":len(cells),"coverage_hypotheses":len(family),
        "estimable_coverage_hypotheses":sum(r["estimable"] for r in family),
        "coverage_Holm1344_rejections":sum(r["reject_FWER05"] is True for r in family),
        "ACI_proposition41_bound_T608":(.9+.005)/(608*.005),
        "conditional_coverage_guaranteed":False,"calendar_bootstrap_reexecutes_feedback_policy":False,
        "runtime":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,
                   "device":"CPU_saved_arithmetic_only","elapsed_seconds":time.monotonic()-started},
        "limitations":["posthoc known test; no revived preregistration or delta_g",
            "ACI aggregate frequency control is not finite-horizon conditional coverage",
            "discrete finite-corrected quantile adaptation, not replication of full ACI paper experiment",
            "bootstrap holds realized adaptive paths fixed, not sampling distribution of rerun online ACI",
            "HAC approximations require weak dependence and sufficient subgroup support",
            "nonfinite sets and undefined resampled groups explicitly retained"]}
    dump(output/"summary.json",summary)
    dump(output/"source_evidence.json",{
        "primary_source":{"authors":"Isaac Gibbs and Emmanuel J. Candes","year":2021,
            "title":"Adaptive Conformal Inference Under Distribution Shift","version":"arxiv2106.00170v3",
            "url":"https://arxiv.org/html/2106.00170v3",
            "locators":{"section2_equation2":"alpha feedback after observing miscoverage",
                "section2_1":"gamma0.005 chosen by authors; used here unchanged without test search",
                "section4_1_Lemma4_1_Proposition4_1":"extended full/empty sets; long-run aggregate error frequency, finite bound"}},
        "implementation_adaptations":["fixed previously trained E2 point forecasts replace GARCH in source experiment",
            "53 prior available scores instead of1250 source-example dates",
            "finite-sample corrected ceil((n+1)*(1-alpha)) empirical rank preserves original49/53 instead of source n convention",
            "nonnegative target restricts full set to[0,infinity)",
            "exact source experiment and conditional-coverage guarantees are not claimed"],
        "first_origin_lagged_target":{"date":"2024-12-31","source":"verified development CSV; actual observed sigma7, not an invented forecast/residual"},
        "missing_score":{"date":"2024-12-31","handling":"not available; not inserted or forward filled"},
        "probability_scope":"one already exposed market chronology; no new independent markets; seeds fixed in calendar inference"})
    manifest={"version":VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "input_sha256":INPUT_SHA,"development_sha256":DEV_SHA,"plan_registration_commit":PLAN_COMMIT,
        "plan_sha256":digest(plan_path),"script_sha256":digest(__file__),
        "files":{p.name:{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(output.iterdir()) if p.is_file()}}
    dump(output/"manifest.json",manifest)
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True,type=Path)
    p.add_argument("--development",required=True,type=Path);p.add_argument("--plan",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    print(json.dumps(run(a.input,a.development,a.plan,a.output),ensure_ascii=False,indent=2))
