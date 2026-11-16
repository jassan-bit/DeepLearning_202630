"""Frozen reduced-E2 metrics and conditional inference; never fits or loads data.

The inherited 608-date E1 history is already known. The procedures below do not
claim renewed blinding, independent markets or finite-sample exact coverage.
Seed-t/sign-flip diagnostics condition on that same observed calendar; the
primary moving-block/seed bootstrap additionally varies dependent dated blocks.
"""
from __future__ import annotations

from datetime import date, timedelta
import itertools
import math

import numpy as np
from scipy import optimize, stats

from e2_reduced_protocol import DELTA, NEURAL, SYSTEMS, TOP10, digest

VERSION = "e2-reduced-final-analysis-v1"
SEEDS = tuple(range(271801, 271807))
PILOT_SEEDS = tuple(range(271701, 271706))
ANALYSIS_SEED = 271901


def analysis_registration():
    value = {
        "version": VERSION, "systems": list(SYSTEMS), "neural_seed_ids": list(SEEDS),
        "primary_metric": "mean_individual_seed_RMSE_original_units",
        "delta": DELTA, "family_size": 20, "family_wise_alpha": .05,
        "primary": {"test": "centered_paired_seed_moving_block_bootstrap",
                    "draws": 4999, "block_days": 14, "seed": ANALYSIS_SEED,
                    "null": "mean individual-seed RMSE gain <= delta", "multiplicity": "Holm20"},
        "diagnostics": [
            {"test": "paired_seed_student_t", "conditional_on": "fixed observed calendar",
             "assumptions": "independent seed trajectories and approximately normal paired seed gains; R6 is small", "multiplicity": "Holm20 separately"},
            {"test": "exact_seed_sign_flip", "conditional_on": "fixed observed calendar",
             "assumptions": "symmetry of paired gains about practical null; enumeration does not establish that symmetry", "multiplicity": "Holm20 separately"}],
        "block_sensitivity_days": [7, 28], "confidence": .95,
        "interval_nominal_coverage": .90, "calibration_days": 53,
        "pilot": {"models": ["iTransformer", "GRU32"], "seed_ids": list(PILOT_SEEDS),
            "operative_R": 6, "power_target": .80, "planned_gain": 2*DELTA,
            "null_gain": DELTA, "conservative_test_alpha": .05/20,
            "method": "noncentral paired-t planning conditional on 53-day development chronology; no automatic R increase"},
        "scope": "known inherited BTCUSDT chronology; optimization seeds are not independent markets",
        "retune_from_final_scores_allowed": False,
        "exact_distribution_free_interval_coverage_claimed": False,
    }
    return {**value, "sha256": digest(value)}


def _finite(values, *, dimensions=None):
    array = np.asarray(values, dtype=np.float64)
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError("Unexpected prediction/target dimensions")
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Finite nonempty aligned observations are required")
    return array


def residual_calibrator(predictions, targets, *, dates, predictor_sha256):
    predictions, targets = _finite(predictions, dimensions=1), _finite(targets, dimensions=1)
    expected = [(date(2024, 11, 8) + timedelta(days=i)).isoformat() for i in range(53)]
    if list(dates) != expected or predictions.shape != targets.shape or len(targets) != 53:
        raise ValueError("Calibration must retain all 53 registered development dates")
    if not isinstance(predictor_sha256, str) or len(predictor_sha256) != 64:
        raise ValueError("A frozen predictor checksum is required")
    residuals = np.abs(targets-predictions)
    rank = int(math.ceil((len(residuals)+1)*.9))
    if rank > len(residuals):
        raise ValueError("Insufficient residuals for the registered quantile")
    value = {"version": VERSION, "nominal_coverage": .9,
        "q": float(np.sort(residuals)[rank-1]), "rank": rank, "n": len(residuals),
        "dates": expected, "predictor_sha256": predictor_sha256,
        "method": "absolute residual order statistic; lower endpoint clipped to0",
        "same_weights_and_scaler_through_evaluation": True,
        "exact_exchangeable_coverage_claimed": False,
        "calibration_predictions_sha256": digest(predictions.tolist()),
        "calibration_targets_sha256": digest(targets.tolist())}
    return {**value, "sha256": digest(value)}


def prediction_intervals(predictions, calibrator):
    predictions = _finite(predictions)
    if calibrator.get("sha256") != digest({k:v for k,v in calibrator.items() if k != "sha256"}):
        raise ValueError("Calibrator checksum mismatch")
    q = float(calibrator["q"])
    if not math.isfinite(q) or q < 0:
        raise ValueError("Invalid frozen residual quantile")
    lower, upper = np.maximum(0., predictions-q), predictions+q
    if np.any(upper < lower):
        raise ValueError("The frozen predictor produced an invalid nonnegative interval")
    return lower, upper


def metrics(targets, predictions):
    y, p = _finite(targets, dimensions=1), _finite(predictions, dimensions=2)
    if p.shape[1] != len(y):
        raise ValueError("Predictions must align to the identical observed dates")
    errors = p-y[None, :]
    rmse = np.sqrt(np.mean(errors**2, axis=1))
    mae = np.mean(np.abs(errors), axis=1)
    smape = np.mean(2*np.abs(errors)/np.maximum(np.abs(p)+np.abs(y)[None,:], 1e-15), axis=1)
    denominator = float(np.sum((y-y.mean())**2))
    r2 = [None if denominator == 0 else float(1-np.sum(e**2)/denominator) for e in errors]
    sd = float(np.std(rmse, ddof=1)) if len(rmse)>1 else None
    ci = None
    if sd is not None:
        half = float(stats.t.ppf(.975,len(rmse)-1)*sd/math.sqrt(len(rmse)))
        ci = [float(np.mean(rmse)-half),float(np.mean(rmse)+half)]
    return {"RMSE":float(np.mean(rmse)), "RMSE_seed_SD":sd,
        "RMSE_per_seed":[float(x) for x in rmse], "RMSE_seed_t_CI95":ci,
        "MAE":float(np.mean(mae)), "sMAPE":float(np.mean(smape)),
        "R2_mean":None if denominator==0 else float(np.mean(r2)),
        "R":len(p), "n_common_dates":len(y),
        "primary_is_ensemble_RMSE":False,
        "seed_CI_scope":"optimization variation conditional on the observed chronology"}


def _seed_gains(y, model, baseline):
    rm = np.sqrt(np.mean((model-y[None,:])**2, axis=1))
    rb = np.sqrt(np.mean((baseline-y[None,:])**2, axis=1))
    if len(rb)==1:
        rb=np.repeat(rb,len(rm))
    if len(rb)!=len(rm):
        raise ValueError("Paired neural contrasts require the same seed trajectories")
    return rb-rm


def _blocks(rng, n, size):
    starts=rng.integers(0,n,size=int(math.ceil(n/size)))
    return np.concatenate([(s+np.arange(size))%n for s in starts])[:n]


def bootstrap_contrast(y, model, baseline, *, delta=DELTA, block_days=14, draws=4999, seed=ANALYSIS_SEED, mask=None):
    y,model,baseline=(np.asarray(v,dtype=np.float64) for v in (y,model,baseline))
    if y.ndim!=1 or model.ndim!=2 or baseline.ndim!=2:
        raise ValueError('Aligned scalar dated trajectories required')
    mask=np.ones(len(y),dtype=bool) if mask is None else np.asarray(mask)
    if mask.shape!=(len(y),) or mask.dtype!=bool or not mask.any():
        raise ValueError('An explicit calendar mask is required')
    _finite(y[mask]);_finite(model[:,mask]);_finite(baseline[:,mask])
    if model.shape[1]!=len(y) or baseline.shape[1]!=len(y) or model.shape[0]<5:
        raise ValueError("Aligned targets and at least five independent seed trajectories required")
    if type(draws) is not int or draws<99 or type(block_days) is not int or block_days<1:
        raise ValueError("Explicit bootstrap count/block size required")
    observed=float(np.mean(_seed_gains(y[mask],model[:,mask],baseline[:,mask])))
    rng=np.random.default_rng(seed); values=np.empty(draws)
    for i in range(draws):
        idx=_blocks(rng,len(y),block_days)
        idx=idx[mask[idx]]  # keep actual calendar gaps; never define blocks on retained-row offsets
        if not len(idx):
            raise ValueError('A bootstrap draw has no common support; do not silently redraw dates')
        seeds=rng.integers(0,len(model),size=len(model))
        b=baseline if len(baseline)==1 else baseline[seeds]
        values[i]=np.mean(_seed_gains(y[idx],model[seeds][:,idx],b[:,idx]))
    p=float((1+np.count_nonzero(values-observed>=observed-delta))/(draws+1))
    return {"test":"centered_paired_seed_moving_block_bootstrap","gain":observed,
        "delta":delta,"p_one_sided":p,"CI95":np.quantile(values,[.025,.975]).tolist(),
        "block_days":block_days,"draws":draws,"seed":seed,
        "exact_finite_sample_test_claimed":False,"resampling_unit":"true calendar positions before common-mask filtering",
        "assumptions":"local stationarity within the known chronology; block length adequate for dependence; exchangeable independent seed trajectories"}


def _seed_diagnostics(gains, delta=DELTA):
    gains=_finite(gains,dimensions=1); centered=gains-delta
    n=len(centered); sd=float(np.std(centered,ddof=1)); mean=float(np.mean(centered))
    if n<5:
        raise ValueError("At least five paired seed gains required")
    statistic=(mean/(sd/math.sqrt(n))) if sd>0 else (math.inf if mean>0 else -math.inf if mean<0 else 0.)
    p_t=float(stats.t.sf(statistic,n-1))
    # Enumerate2^R centered signs. Symmetry is an explicit diagnostic assumption.
    simulated=np.array([np.mean(centered*np.asarray(signs)) for signs in itertools.product((-1,1),repeat=n)])
    p_sign=float(np.count_nonzero(simulated>=mean)/len(simulated))
    return [{"test":"paired_seed_student_t","gain":float(gains.mean()),"delta":delta,
             "statistic":None if not math.isfinite(statistic) else statistic,"df":n-1,
             "p_one_sided":p_t,"scope":"optimization seeds conditional on this fixed calendar"},
            {"test":"exact_seed_sign_flip","gain":float(gains.mean()),"delta":delta,
             "p_one_sided":p_sign,"enumerated":len(simulated),
             "scope":"conditional seed diagnostic; symmetry assumption is not established by enumeration"}]


def holm(rows, *, alpha=.05):
    pvalues=np.asarray([float(r["p_one_sided"]) for r in rows])
    if not np.isfinite(pvalues).all() or np.any((pvalues<0)|(pvalues>1)):
        raise ValueError("Finite p values in[0,1] required")
    order=np.argsort(pvalues,kind="stable");adjusted=np.empty(len(rows));running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(rows)-rank)*pvalues[index]);adjusted[index]=min(1.,running)
    return [{**row,"p_Holm":float(adjusted[i]),"reject_at_FWER05":bool(adjusted[i]<=alpha)} for i,row in enumerate(rows)]


def pilot_power_mde(targets, predictions, *, dates):
    y=_finite(targets,dimensions=1)
    expected=[(date(2024,11,8)+timedelta(days=i)).isoformat() for i in range(53)]
    if set(predictions)!={"iTransformer","GRU32"} or list(dates)!=expected or len(y)!=53:
        raise ValueError("The two registered representatives on53dates are required")
    a,b=(_finite(predictions[k],dimensions=2) for k in ("iTransformer","GRU32"))
    if a.shape!=(5,53) or b.shape!=(5,53):
        raise ValueError("Exactly five registered pilot seeds per representative required")
    gain=_seed_gains(y,a,b); sd=float(np.std(gain,ddof=1)); se=sd/math.sqrt(6)
    critical=float(stats.t.ppf(1-.05/20,5))
    def power(excess):
        if se==0:return 1. if excess>0 else .05/20
        return float(stats.nct.sf(critical,5,excess/se))
    mde=None if sd==0 else float(optimize.brentq(lambda excess:power(excess)-.8,0,max(DELTA,sd)*100))
    planned_power=None if sd==0 else power(DELTA)
    value={"version":VERSION,"status":"development_pilot_completed_power_reported",
        "models":["iTransformer","GRU32"],"seed_ids":list(PILOT_SEEDS),"R_operative":6,
        "n_calendar_dates":53,"dates":list(dates),"pilot_paired_seed_gains":gain.tolist(),
        "pilot_gain_SD":sd,"power_at_planned_gain_2delta":planned_power,
        "power_target":.8,"power80_achieved_in_conditional_planning":False if planned_power is None else planned_power>=.8,
        "MDE_excess_above_delta_at_80pct":mde,"minimum_gain_at80pct":None if mde is None else DELTA+mde,
        "zero_estimated_seed_variance_is_not_proof_of_zero_population_variance":True,
        "test_alpha":.05/20,"training_seed_df_at_R6":5,
        "method":"noncentral paired-t planning using five-seed development SD; conditional on same53dates",
        "limitations":["SD estimated from only five paired pilot seeds",
            "No independent market replication or temporal power guarantee",
            "At most two model representatives measured; power does not establish capability for every Top10 contrast",
            "Resource-bound R6 is retained even when target is unmet; no equivalence claim"],
        "automatic_R_increase":False,"calibration_labels_used_for_HPO":False}
    return {**value,"sha256":digest(value)}


def analyze_benchmark(targets, predictions, *, dates, mask, calibrators, draws=4999):
    registration=analysis_registration()
    expected=[(date(2025,1,1)+timedelta(days=i)).isoformat() for i in range(608)]
    if list(dates)!=expected or set(predictions)!=set(SYSTEMS) or set(calibrators)!=set(SYSTEMS):
        raise ValueError("The complete registered16-system inherited608-date panel is required")
    y=np.asarray(targets,dtype=np.float64);mask=np.asarray(mask)
    if y.shape!=(608,) or mask.shape!=(608,) or mask.dtype!=bool or mask.sum()<30 or not np.isfinite(y[mask]).all():
        raise ValueError("The frozen common support mask must have finite observed targets")
    arrays={};table=[]
    for name in SYSTEMS:
        p=np.asarray(predictions[name],dtype=np.float64);r=6 if name in NEURAL else 1
        if p.shape!=(r,608) or not np.isfinite(p[:,mask]).all() or len(calibrators[name])!=r:
            raise ValueError("All registered seed trajectories and calibrators must align: "+name)
        arrays[name]=p[:,mask];row={"model":name,**metrics(y[mask],arrays[name])}
        cover=[];width=[]
        for seed in range(r):
            lo,hi=prediction_intervals(arrays[name][seed],calibrators[name][seed])
            cover.append(float(np.mean((y[mask]>=lo)&(y[mask]<=hi))));width.append(float(np.mean(hi-lo)))
        row.update(coverage90=float(np.mean(cover)),mean_interval_width90=float(np.mean(width)))
        table.append(row)
    primary=[];seed_t=[];sign=[];sensitivity=[]
    for index,(name,baseline) in enumerate(itertools.product(TOP10,("GRU32","Persistence"))):
        identity={"model":name,"baseline":baseline}
        primary.append({**identity,**bootstrap_contrast(y,np.asarray(predictions[name]),np.asarray(predictions[baseline]),mask=mask,draws=draws,seed=ANALYSIS_SEED+index)})
        diagnostics=_seed_diagnostics(_seed_gains(y[mask],arrays[name],arrays[baseline]))
        seed_t.append({**identity,**diagnostics[0]});sign.append({**identity,**diagnostics[1]})
        for size in (7,28):
            sensitivity.append({**identity,**bootstrap_contrast(y,np.asarray(predictions[name]),np.asarray(predictions[baseline]),mask=mask,draws=draws,block_days=size,seed=ANALYSIS_SEED+index)})
    return {"version":VERSION,"analysis_registration_sha256":registration["sha256"],
        "table_A2":table,"primary_contrasts":holm(primary),
        "diagnostic_seed_t":holm(seed_t),"diagnostic_seed_sign_flip":holm(sign),
        "block_sensitivity_unadjusted_diagnostic":sensitivity,
        "calendar_dates":608,"n_common_dates":int(mask.sum()),
        "common_mask_sha256":digest(mask.tolist()),"retrospective_known_calendar":True,
        "test_battery_count_per_contrast":3,"no_retuning_after_scoring":True}
