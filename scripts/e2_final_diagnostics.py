"""Frozen-predictor robustness, gradient-free explanations and pre-fit ablations.

Numerical helpers only: no data acquisition, model fitting or test discovery.
Ablation hooks define a new architecture before a NEW matched training run;
applying them to a checkpoint is never reported as a retrained ablation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import copy
import json
import math

import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn


TOP10 = ("TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "Informer",
         "Autoformer", "iTransformer", "Crossformer", "S-Mamba")
FEATURE_NAMES=('Open','High','Low','Close','Volume')
FEATURE_GROUPS=tuple(group for lag in range(7) for group in (
    {'id':f'lag{7-lag}_OHLC','lag_days':7-lag,'kind':'OHLC_prices','channels':list(FEATURE_NAMES[:4]),'indices':list(range(lag*5,lag*5+4))},
    {'id':f'lag{7-lag}_Volume','lag_days':7-lag,'kind':'Volume','channels':['Volume'],'indices':[lag*5+4]}))


def _groups(groups=None):
    result=copy.deepcopy(list(FEATURE_GROUPS if groups is None else groups))
    indices=[i for g in result for i in g['indices']]
    if sorted(indices)!=list(range(35)) or len({g['id'] for g in result})!=len(result):
        raise ValueError('Feature groups must partition all35 observed D7coordinates exactly once')
    return result


def _group_values(attributions,n,groups):
    a=np.asarray(attributions,float)
    if a.shape==(n,len(groups)):return a
    if a.shape==(n,7,5):
        flat=a.reshape(n,35)
        return np.stack([flat[:,g['indices']].sum(1) for g in groups],axis=1)
    raise ValueError('Attributions must align with14 groups or35 IG coordinates')


SEVERITIES = {"noise": (.01, .03, .05), "missing": (.05, .10, .20), "outliers": (.10, .25, .50)}
DESIGN = {
    "id": "E2-diagnostics-completion-20261106", "status": "definition_pending_registration_and_real_execution",
    "robustness": {"severities": SEVERITIES, "outlier_day_probability": .05,
        "random_seed": 202611067, "coupling": "same physical calendar day, perturbation and seed => same shocks in all windows/models",
        "price_noise": "one shared log shock for O/H/L/C on each observed day",
        "volume_noise": "independent log shock; nonnegative volume preserved",
        "missing": "whole-day OHLCV mask; impute raw five-channel training-only means BEFORE relative-window anchoring",
        "target_policy": "original unperturbed sigma7; input measurement-noise scenario, not counterfactual market labels",
        "normalization": "unchanged training-only preprocessing after perturbation; no refit or added channel",
        "primary_metric": "mean of per-seed RMSE, not ensemble RMSE",
        "AUC": "trapezoid over severity/max_severity; score/clean and (score-clean)/clean; no clipping improvements",
        "distinct_endogenous_history": "all three endogenous estimators require actual causal target-history callbacks; stress observed post-fit history with frozen coefficients, never architectural immunity or an N/A shortcut"},
    "explanations": {"primary": "signed leave-one-registered-group-out occlusion (14groups)",
        "baseline": "TRAIN-only35cellmeans in standardized space, preserving constantrelativeCloseanchor; same baseline for primaryocclusion and secondaryIG",
        "production_requires_scaler_and_training_cutoff_receipt": True,
        "feature_groups": copy.deepcopy(FEATURE_GROUPS),
        "IG": "secondary only, trapezoidal64steps (32sensitivity); retain completeness residual and reject incomplete explanation claims",
        "fidelity": "deletion/insertion14group paths versus10 fixed random rankings; normalized by original-minus-baseline prediction",
        "stability": "absolute14group-attribution Spearman, exact first5 registered training seeds, all10 pairs",
        "causal_interpretation": False, "known_limit": "correlated-feature interventions may leave observed data manifold; pooled-zero replacement can disturb deterministic relative-Close anchor cells",
        "required_registered_training_baseline": "fit_explanation_baseline learns35cellmeans only in training; preserve constantlagcells; freeze receipt before evaluation"},
    "ablation": {"selection": "best Top10 by registered development ranking, chosen BEFORE final/test scoring",
        "baselines_may_win_overall": True, "components_each": 3,
        "matched_retraining": "same frozen recipe, registered seeds/R, data, early-stop/refit policy and inference; fresh initialization for every lesion",
        "inference_lesion_is_not_retrained_ablation": True,
        "parameter_count_scope": "masked or bypassed components remain in state_dict; report frozen/effective parameters, not fictitious physical removal",
        "fit_outcome_max": "2024-10-31", "evaluation_calendar": ["2024-11-08", "2024-12-30"],
        "evaluation_unique_days": 53, "matched_R": "same selected final R and exact seed IDs for control and each of3lesions",
        "inference": "separate exploratory family3, Holm alpha.05, B9999/block7/sens14,28; two-sided practical null abs(delta_RMSE)<=inherited_delta; not included silently in main30/30/199"},
    "uncertainty": {"minimum_bootstrap_replicates": 2000, "principal_block_days": 7,
        "sensitivity_block_days": [14, 28], "CI": "nominal percentile95, paired dense daily calendar blocks",
        "minimum_estimable_fraction": .95, "new_hypothesis_decisions": 0},
    "real_data_fits_performed_by_definition": 0,
}


def _dates(origins):
    days = [date.fromisoformat(str(x)[:10]) for x in origins]
    if not days or any(b-a != timedelta(days=1) for a,b in zip(days,days[1:])):
        raise ValueError("Retain the complete strictly daily calendar; do not compact missing dates")
    return days


def _mask(n, eligible=None):
    if eligible is None:
        a = np.ones(n, bool)
    else:
        raw = np.asarray(eligible)
        if raw.dtype.kind != 'b' and (raw.dtype.kind not in 'iu' or not np.isin(raw,[0,1]).all()):
            raise ValueError("Eligibility must contain explicit booleans or0/1, not truthy strings")
        a = raw.astype(bool)
    if a.shape != (n,) or not a.any():
        raise ValueError("At least one eligible daily origin and a dense common mask are required")
    return a


def _raw(windows, eligible=None):
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 5 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Raw windows must have shape [N,history,5OHLCV]")
    selected = x[_mask(len(x), eligible)]
    if (not np.isfinite(selected).all() or (selected[..., :4] <= 0).any()
            or (selected[..., 4] < 0).any()
            or (selected[..., 1] < np.maximum(selected[..., 0], selected[..., 3])).any()
            or (selected[..., 2] > np.minimum(selected[..., 0], selected[..., 3])).any()):
        raise ValueError("Eligible OHLCV windows must be finite, ordered and physically valid")
    return x


def _digest(values):
    return sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _day_rng(seed, kind, day):
    key = f"{int(seed)}|{kind}|{day.isoformat()}".encode()
    return np.random.default_rng(int.from_bytes(sha256(key).digest()[:8], "little"))


def fit_perturbation_reference(training_windows, *, training_outcome_dates, maximum_training_outcome):
    """Fit only raw imputation means; observed values never come from evaluation."""
    x = _raw(training_windows)
    days = [date.fromisoformat(str(v)[:10]) for v in training_outcome_dates]
    cutoff = date.fromisoformat(str(maximum_training_outcome)[:10])
    if len(days) != len(x) or len(set(days)) != len(days) or days != sorted(days) or max(days) > cutoff:
        raise ValueError("Training-only imputation dates must be unique, ordered and within the declared cutoff")
    means = x.reshape(-1, 5).mean(axis=0)
    return {"source": "training_windows_only", "training_outcome_max": max(days).isoformat(),
        "declared_training_outcome_max": cutoff.isoformat(), "training_windows": len(x),
        "training_input_positions": int(x.shape[0]*x.shape[1]), "raw_OHLCV_mean": means.tolist(),
        "training_window_sha256": _digest(x), "pooled_positions_not_independent_sample_count": True}


def perturb_raw_windows(windows, origins, *, kind, severity, reference,
                        eligible=None, seed=202611067):
    """Coupled calendar-consistent corruption, with no hidden close anchor."""
    days = _dates(origins); mask = _mask(len(days), eligible); x = _raw(windows, mask)
    if len(x) != len(days) or kind not in SEVERITIES or not math.isfinite(float(severity)) or severity < 0:
        raise ValueError("Unknown perturbation, negative severity or mismatched calendar")
    if kind == "missing" and severity > 1:
        raise ValueError("Missingness is a probability")
    if (reference.get("source") != "training_windows_only"
            or date.fromisoformat(reference["training_outcome_max"]) > days[0]-timedelta(days=8)):
        raise ValueError("Frozen imputation reference must respect the evaluation-origin embargo")
    mean = np.asarray(reference["raw_OHLCV_mean"], float)
    _raw(mean[None, None, :])
    out = x.copy()
    cache = {}; observed_values = {}
    for row, origin in enumerate(days):
        if not mask[row]:
            continue
        for lag in range(x.shape[1]):
            bar = origin - timedelta(days=x.shape[1]-lag)
            if bar not in cache:
                rng = _day_rng(seed, kind, bar)
                cache[bar] = (rng.normal(), rng.normal(), rng.random(), rng.random(),
                              -1 if rng.random() < .5 else 1, -1 if rng.random() < .5 else 1,
                              rng.random())
                observed_values[bar] = x[row, lag].copy()
            elif not np.array_equal(observed_values[bar], x[row, lag]):
                raise ValueError("Repeated physical input day has inconsistent raw OHLCV values")
            price, volume, missing, outlier, sign_p, sign_v, outlier_volume = cache[bar]
            if kind == "noise":
                out[row, lag, :4] *= np.exp(float(severity)*price)
                out[row, lag, 4] *= np.exp(float(severity)*volume)
            elif kind == "missing":
                if missing < severity:
                    out[row, lag] = mean
            else:
                if outlier < DESIGN["robustness"]["outlier_day_probability"]:
                    out[row, lag, :4] *= np.exp(float(severity)*sign_p)
                if outlier_volume < DESIGN["robustness"]["outlier_day_probability"]:
                    out[row, lag, 4] *= np.exp(float(severity)*sign_v)
    _raw(out, mask)
    return out, {"kind":kind, "severity":float(severity), "seed":int(seed),
        "distinct_corruption_days":len(cache), "calendar_consistent":True,
        "raw_reference_sha256":reference["training_window_sha256"],
        "corrupted_window_sha256":_digest(out), "targets_changed":False,
        "scalers_refitted":False, "extra_input_channels":0}


def _predict(predictor, values):
    result = np.asarray(predictor(values), dtype=np.float64)
    if result.ndim == 1:
        result = result[None]
    if result.ndim != 2 or result.shape[1] != len(values) or not result.shape[0]:
        raise ValueError("Predictor must return [N] or paired [registered_seeds,N]")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Predictive volatility must be finite and nonnegative")
    return result


def _block_draws(n, block, repetitions, seed):
    if (int(repetitions) != repetitions or repetitions < 2000 or int(block) != block or not 1 <= int(block) <= n):
        raise ValueError("Use at least2000 daily block resamples and a valid block length")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(int(repetitions), math.ceil(n/int(block))))
    return ((starts[:, :, None]+np.arange(int(block))) % n).reshape(int(repetitions), -1)[:, :n]


def _interval(values, repetitions):
    a = np.asarray(values, float); good = a[np.isfinite(a)]
    return {"ci95_nominal": np.quantile(good,[.025,.975]).tolist() if len(good)>=.95*repetitions else None,
        "estimable_draws":int(len(good)), "requested_draws":int(repetitions),
        "status":"estimable" if len(good)>=.95*repetitions else "not_estimable"}


def _score(error, mask):
    value = float(np.sqrt((error[:, mask]**2).mean(axis=1)).mean())
    if not math.isfinite(value):raise ValueError("Nonfinite diagnostic score; retain the failed attempt, never replace it by zero")
    return value


def _boot_score(error, mask, draws):
    values = np.full(len(draws), np.nan)
    for b, idx in enumerate(draws):
        keep = mask[idx]
        if keep.any():
            values[b] = _score(error[:, idx], keep)
    return values


def _panel(windows, truth, origins, eligible):
    days = _dates(origins); keep = _mask(len(days), eligible); x = _raw(windows, keep)
    y = np.asarray(truth, float)
    if len(x) != len(days) or y.shape != (len(days),) or not np.isfinite(y[keep]).all() or (y[keep]<0).any():
        raise ValueError("Truth, raw windows and eligible daily calendar differ")
    if np.isfinite(y[~keep]).any():
        raise ValueError("Ineligible truth rows must remain missing, not fabricated values")
    return x,y,days,keep


def evaluate_robustness(predict_raw, windows, truth, origins, *, reference, eligible=None,
        input_contract="OHLCV_D7", repetitions=2000, block_length=7,
        seed=202611067, include_private_predictions=False):
    """Evaluate fixed predictors; preprocess/scaling belongs to predict_raw.

    For stateful endogenous models the caller must supply an appropriate causal
    raw-history/state intervention. Otherwise return an explicit N/A entry.
    """
    if not input_contract.startswith("OHLCV_"):
        return {"status":"not_applicable_distinct_information_set", "input_contract":input_contract,
            "reason":"No validated causal raw-to-endogenous-history/state corruption callback supplied",
            "architectural_immunity_claimed":False, "new_hypothesis_decisions":0}
    x,y,days,mask = _panel(windows,truth,origins,eligible)
    # Validate the train-only reference BEFORE obtaining any prediction.
    perturb_raw_windows(x,origins,kind='noise',severity=0,reference=reference,eligible=mask,seed=seed)
    draws = _block_draws(len(days),block_length,repetitions,seed+1)
    clean = np.full((1,len(days)),np.nan)
    predicted = _predict(predict_raw,x[mask]); clean = np.full((len(predicted),len(days)),np.nan); clean[:,mask]=predicted
    clean_error=clean-y[None]; baseline=_score(clean_error,mask); base_boot=_boot_score(clean_error,mask,draws)
    reports=[];auc=[];private={"clean":clean};provenance=[]
    for kind,severities in SEVERITIES.items():
        curve=[baseline];boot_curve=[base_boot]
        for severity in severities:
            perturbed,receipt=perturb_raw_windows(x,origins,kind=kind,severity=severity,reference=reference,eligible=mask,seed=seed)
            pred=_predict(predict_raw,perturbed[mask])
            if pred.shape[0]!=clean.shape[0]:raise ValueError("Seed members changed between perturbations")
            dense=np.full_like(clean,np.nan);dense[:,mask]=pred;error=dense-y[None]
            point=_score(error,mask);sampled=_boot_score(error,mask,draws)
            with np.errstate(divide='ignore',invalid='ignore'):
                relative=(sampled-base_boot)/base_boot
            reports.append({"kind":kind,"severity":severity,"RMSE_mean_seed":point,
                "clean_RMSE_mean_seed":baseline,"RMSE_difference":point-baseline,
                "relative_degradation":(point-baseline)/baseline if baseline>0 else None,
                "RMSE_interval":_interval(sampled,repetitions),
                "paired_difference_interval":_interval(sampled-base_boot,repetitions),
                "relative_degradation_interval":_interval(relative,repetitions)})
            curve.append(point);boot_curve.append(sampled);provenance.append(receipt)
            if include_private_predictions:private[f'{kind}_{severity}']=dense
        coordinate=np.asarray((0,*severities),float)/max(severities)
        if baseline>0:
            normalized=float(np.trapezoid(np.asarray(curve)/baseline,coordinate))
            degradation=normalized-1.
        else:normalized=degradation=None
        with np.errstate(divide='ignore',invalid='ignore'):
            sample_auc=np.trapezoid(np.stack(boot_curve,axis=1)/base_boot[:,None],coordinate,axis=1)
        auc.append({"kind":kind,"normalized_score_AUC":normalized,"relative_degradation_AUC":degradation,
            "relative_degradation_AUC_interval":_interval(sample_auc-1,repetitions),
            "direction":"lower relative degradation area is better; negative observed improvement is retained"})
    result={"status":"evaluated_frozen_predictions","design_id":DESIGN["id"],
        "metric":"mean per-seed RMSE in original volatility units", "seed_members":clean.shape[0],
        "calendar_origins":len(days),"eligible_unique_origins":int(mask.sum()),
        "input_contract":input_contract,"history_days":x.shape[1],"rows":reports,"AUC":auc,
        "bootstrap":{"replicates":repetitions,"block_days":block_length,"paired_full_calendar":True},
        "provenance":{"raw_window_sha256":_digest(x),"truth_sha256":_digest(y),
            "clean_prediction_sha256":_digest(clean),"perturbations":provenance},
        "train_imputation_reference":reference,"new_model_fits":0,"new_hypothesis_decisions":0,
        "uncertainty_scope":"nominal temporal intervals conditional on these frozen predictors; no exact coverage claim"}
    if include_private_predictions:result['private_predictions']=private
    return result


def torch_normalized_predictor(model, *, y_scale=1., batch_size=64):
    if not math.isfinite(float(y_scale)) or y_scale<=0 or batch_size<1:
        raise ValueError("A positive frozen y_scale and batch size are required")
    device=next(model.parameters(),torch.empty(0)).device
    def predict(values):
        x=np.asarray(values,np.float32)
        if x.ndim!=3 or x.shape[1:]!=(7,5) or not np.isfinite(x).all():raise ValueError("Finite standardized D7 inputs required")
        old=model.training;model.eval();out=[]
        try:
            with torch.no_grad():
                for start in range(0,len(x),batch_size):
                    p=model(torch.from_numpy(x[start:start+batch_size].copy()).to(device))
                    out.append(p.detach().cpu().numpy().astype(float)*float(y_scale))
        finally:model.train(old)
        return np.concatenate(out) if out else np.empty(0)
    return predict


def torch_raw_predictor(model, preprocessing, *, batch_size=64):
    from dl_data_v2 import relative_features
    mean=np.asarray(preprocessing['x_mean'],float);scale=np.asarray(preprocessing['x_scale'],float)
    if mean.shape!=(5,) or scale.shape!=(5,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError("Invalid frozen train-only channel preprocessing")
    predict=torch_normalized_predictor(model,y_scale=float(preprocessing['y_scale']),batch_size=batch_size)
    def raw(values):return predict((relative_features(_raw(values))-mean)/scale)
    return raw


def fit_explanation_baseline(training_standardized_windows,*,training_outcome_dates,maximum_training_outcome,scaler_sha256):
    x=np.asarray(training_standardized_windows,float)
    days=[date.fromisoformat(str(d)[:10]) for d in training_outcome_dates]
    maximum=date.fromisoformat(str(maximum_training_outcome)[:10])
    if (x.ndim!=3 or x.shape[1:]!=(7,5) or len(x)<1 or not np.isfinite(x).all()
            or len(days)!=len(x) or days!=sorted(set(days)) or max(days)>maximum):
        raise ValueError('A baseline requires distinct training-only standardized D7windows within the declared cutoff')
    if not isinstance(scaler_sha256,str) or len(scaler_sha256)!=64 or any(c not in '0123456789abcdef' for c in scaler_sha256):
        raise ValueError('The explanation baseline must identify its frozen scaler hash')
    constant=np.all(x==x[:1],axis=0);baseline=x.mean(axis=0);baseline[constant]=x[0][constant]
    return {'kind':'training_cellwise_standardized_feature_mean',
        'baseline':baseline.tolist(),'baseline_sha256':_digest(baseline),
        'constant_training_cells':constant.tolist(),'scaler_sha256':scaler_sha256,
        'training_windows':len(x),'training_outcome_max':max(days).isoformat(),
        'training_window_sha256':_digest(x),'test_or_evaluation_data_used':False,
        'why_not_pooled_zero':'preserves deterministic lag-specific features such as the relative Close anchor'}


def validate_explanation_baseline(receipt,*,expected_scaler_sha256,maximum_training_outcome):
    if (receipt.get('kind')!='training_cellwise_standardized_feature_mean'
            or receipt.get('scaler_sha256')!=expected_scaler_sha256
            or receipt.get('test_or_evaluation_data_used') is not False
            or date.fromisoformat(receipt['training_outcome_max'])>date.fromisoformat(str(maximum_training_outcome)[:10])):
        raise ValueError('Explanation baseline is not bound to this training cutoff and frozen scaler')
    baseline=np.asarray(receipt['baseline'],float)
    if baseline.shape!=(7,5) or not np.isfinite(baseline).all() or _digest(baseline)!=receipt.get('baseline_sha256'):
        raise ValueError('Explanation baseline values or hash changed')
    return baseline


def occlusion_attributions(predict_normalized,standardized_windows,*,baseline=None,groups=None):
    """Primary signed group occlusion:7whole OHLC price bars and7volume values."""
    x=np.asarray(standardized_windows,float);definition=_groups(groups)
    if x.ndim!=3 or x.shape[1:]!=(7,5) or not len(x) or not np.isfinite(x).all():raise ValueError('Finite standardized D7 windows required')
    base=np.zeros_like(x) if baseline is None else np.broadcast_to(np.asarray(baseline,float),x.shape).copy()
    if not np.isfinite(base).all():raise ValueError('Finite frozen training-reference baseline required')
    original=_predict(predict_normalized,x)
    if original.shape[0]!=1:raise ValueError('Explain each registered seed separately, not an unidentified ensemble')
    attribution=np.empty((len(x),len(definition)))
    for j,group in enumerate(definition):
        masked=x.copy().reshape(len(x),35);masked[:,group['indices']]=base.reshape(len(x),35)[:,group['indices']]
        attribution[:,j]=original[0]-_predict(predict_normalized,masked.reshape(x.shape))[0]
    return {'method':'signed_leave_one_registered_group_out_occlusion','primary':True,
        'private_attributions':attribution,'groups':definition,'group_count':len(definition),
        'baseline':'explicit frozen TRAIN-only35cellmeans required in production; zero is a synthetic test reference only',
        'private_original_predictions':original[0],'new_model_fits':0,
        'causal_effect_claimed':False,'additive_completeness_claimed':False,
        'individual_OHLC_field_importance_identifiable':False,
        'limitation':'Grouped replacements preserve within-bar grouping but may still leave the observed joint data manifold'}


def integrated_gradients(model, standardized_windows, *, baseline=None, steps=64, y_scale=1.,
        relative_tolerance=.05, absolute_tolerance=1e-5):
    """Secondary IG with explicit completeness; stop-gradient failures remain N/E."""
    if steps not in (32,64) or y_scale<=0:raise ValueError("Use the registered32/64-step IG and positive y_scale")
    x=torch.as_tensor(standardized_windows,dtype=torch.float32,
        device=next(model.parameters(),torch.empty(0)).device)
    if x.ndim!=3 or x.shape[1:]!=(7,5) or not torch.isfinite(x).all():raise ValueError("Finite D7 input required")
    base=torch.zeros_like(x) if baseline is None else torch.broadcast_to(torch.as_tensor(baseline,dtype=x.dtype,device=x.device),x.shape).clone()
    if not torch.isfinite(base).all():raise ValueError("Finite frozen baseline required")
    old=model.training;model.eval();integral=torch.zeros_like(x)
    try:
        for index in range(steps+1):
            z=(base+(x-base)*(index/steps)).detach().requires_grad_(True)
            prediction=model(z)
            gradient=(torch.autograd.grad(prediction.sum(),z,retain_graph=False,
                create_graph=False,allow_unused=True)[0] if prediction.requires_grad else None)
            if gradient is None:gradient=torch.zeros_like(z)
            integral+=gradient.detach()*(.5 if index in (0,steps) else 1.)
        attribution=(x-base)*integral/steps*float(y_scale)
        with torch.no_grad():delta=(model(x)-model(base))*float(y_scale)
    finally:model.train(old)
    total=attribution.sum(dim=(1,2));residual=delta-total
    tolerance=float(y_scale)*absolute_tolerance+relative_tolerance*delta.abs()
    valid=torch.isfinite(attribution).all(dim=(1,2)) & torch.isfinite(residual) & (residual.abs()<=tolerance)
    return {"method":"integrated_gradients_trapezoid", "primary":False,"steps":steps,
        "private_attributions":attribution.detach().cpu().numpy(),
        "private_prediction_difference":delta.detach().cpu().numpy(),
        "private_completeness_residual":residual.detach().cpu().numpy(),
        "private_completeness_passed":valid.detach().cpu().numpy(),
        "completeness_pass_fraction":float(valid.float().mean()),
        "status":"complete_for_all_rows" if bool(valid.all()) else "incomplete_not_accepted_as_primary_explanation",
        "primary_alternative":"gradient-free occlusion remains prespecified primary for every model",
        "model_weights_updated":False,"causal_effect_claimed":False}


def _attribution_rank(values):
    return np.argsort(-np.abs(values.reshape(len(values),-1)),axis=1,kind='stable')


def deletion_insertion_fidelity(predict_normalized,standardized_windows,attributions,*,baseline=None,groups=None,seed=202611068,random_rankings=10):
    x=np.asarray(standardized_windows,float);definition=_groups(groups);g=len(definition)
    if x.ndim!=3 or x.shape[1:]!=(7,5) or not len(x) or not np.isfinite(x).all():raise ValueError('Finite standardized D7 windows required')
    a=_group_values(attributions,len(x),definition)
    if not np.isfinite(a).all() or random_rankings<1:raise ValueError('Finite aligned group attributions and fixed random controls required')
    base=np.zeros_like(x) if baseline is None else np.broadcast_to(np.asarray(baseline,float),x.shape).copy()
    if not np.isfinite(base).all():raise ValueError('Finite frozen explanation baseline required')
    def one(values):
        result=_predict(predict_normalized,values)
        if result.shape[0]!=1:raise ValueError('Fidelity explains one registered seed at a time')
        return result[0]
    original=one(x);reference=one(base);difference=original-reference
    valid=np.abs(difference)>1e-7*np.maximum(1.,np.maximum(np.abs(original),np.abs(reference)))
    membership=np.zeros((g,35),bool)
    for index,group in enumerate(definition):membership[index,group['indices']]=True
    def curve(rank):
        deletion=np.empty((len(x),g+1,35));insertion=np.empty_like(deletion)
        flat=x.reshape(len(x),35);bf=base.reshape(len(x),35)
        for step in range(g+1):
            group_mask=np.zeros((len(x),g),bool)
            group_mask[np.arange(len(x))[:,None],rank[:,:step]]=True
            feature_mask=(group_mask.astype(int)@membership.astype(int))>0
            deletion[:,step]=np.where(feature_mask,bf,flat);insertion[:,step]=np.where(feature_mask,flat,bf)
        d=one(deletion.reshape(-1,7,5)).reshape(len(x),g+1)
        i=one(insertion.reshape(-1,7,5)).reshape(len(x),g+1)
        with np.errstate(divide='ignore',invalid='ignore'):
            d=(d-reference[:,None])/difference[:,None];i=(i-reference[:,None])/difference[:,None]
        return d,i
    deletion,insertion=curve(_attribution_rank(a));rng=np.random.default_rng(seed);random_d=[];random_i=[]
    for _ in range(random_rankings):
        ranks=np.stack([rng.permutation(g) for _ in x]);d,i=curve(ranks);random_d.append(d);random_i.append(i)
    random_deletion=np.mean(random_d,axis=0);random_insertion=np.mean(random_i,axis=0)
    def auc(values):
        result=np.trapezoid(values,np.arange(g+1)/g,axis=1);result[~valid]=np.nan;return result
    d_auc,i_auc,rd_auc,ri_auc=map(auc,(deletion,insertion,random_deletion,random_insertion))
    return {'method':'group_deletion_insertion_vs_fixed_random_rankings','random_rankings':random_rankings,'seed':seed,
        'groups':definition,'group_count':g,'AUC_axis':'fraction of registered groups, not number of individual coordinates',
        'normalization':'(prediction_at_step-prediction_baseline)/(prediction_original-prediction_baseline)',
        'private_deletion_AUC':d_auc,'private_insertion_AUC':i_auc,'private_random_deletion_AUC':rd_auc,
        'private_random_insertion_AUC':ri_auc,'private_deletion_advantage':rd_auc-d_auc,
        'private_insertion_advantage':i_auc-ri_auc,'private_estimable_rows':valid,
        'not_estimable_near_zero_prediction_difference':int((~valid).sum()),'AUC_not_clipped_to_unit_interval':True,
        'gradient_or_causal_validity_claimed':False,'new_hypothesis_decisions':0}


def summarize_daily_values(values,origins,*,eligible=None,repetitions=2000,block_length=7,seed=202611069):
    days=_dates(origins);a=np.asarray(values,float)
    if a.shape!=(len(days),):raise ValueError("Retain one value or missing placeholder per daily origin")
    mask=_mask(len(days),eligible)&np.isfinite(a)
    if not mask.any():return {"status":"not_estimable","mean":None,"ci95_nominal":None,"eligible_unique_origins":0}
    draws=_block_draws(len(days),block_length,repetitions,seed);boot=[]
    for idx in draws:
        selected=mask[idx]
        boot.append(float(a[idx][selected].mean()) if selected.any() else np.nan)
    return {"mean":float(a[mask].mean()),"eligible_unique_origins":int(mask.sum()),
        **_interval(boot,repetitions),"calendar_not_compacted":True,"block_days":block_length,
        "new_hypothesis_decisions":0}


def explanation_stability(seed_attributions,registered_seeds,origins,*,groups=None,eligible=None,repetitions=2000,block_length=7,seed=202611069):
    selected=list(registered_seeds)[:5];definition=_groups(groups)
    if len(selected)!=5 or len(set(selected))!=5:raise ValueError('Five prespecified distinct training seeds are required')
    days=_dates(origins);mask=_mask(len(days),eligible);arrays=[]
    for key in selected:
        value=seed_attributions.get(key,seed_attributions.get(str(key)))
        a=_group_values(value,len(days),definition)
        if not np.isfinite(a[mask]).all():raise ValueError('Each prespecified seed needs an aligned finite group attribution map')
        arrays.append(a)
    records=[];daily=[]
    for i in range(5):
        for j in range(i+1,5):
            rho=np.full(len(days),np.nan)
            for k in np.flatnonzero(mask):
                u,v=rankdata(np.abs(arrays[i][k])),rankdata(np.abs(arrays[j][k]))
                if u.std()>0 and v.std()>0:rho[k]=np.corrcoef(u,v)[0,1]
            records.append({'seed_pair':[selected[i],selected[j]],**summarize_daily_values(rho,origins,eligible=mask,repetitions=repetitions,block_length=block_length,seed=seed)})
            daily.append(rho)
    matrix=np.stack(daily);count=np.isfinite(matrix).sum(axis=0)
    average=np.divide(np.nansum(matrix,axis=0),count,out=np.full(len(days),np.nan),where=count>0)
    return {'prespecified_first_five_seeds':selected,'pair_count':10,'ranking':'absolute group attribution, average ties',
        'groups':definition,'pairs':records,'overall':summarize_daily_values(average,origins,eligible=mask,repetitions=repetitions,block_length=block_length,seed=seed),
        'private_daily_pair_Spearman':matrix,'constant_attribution_rank_is_not_estimable':True,'new_hypothesis_decisions':0}


def explain_clipped_linear(coefficients,intercept,standardized_windows,*,baseline=None):
    x=np.asarray(standardized_windows,float)
    if x.ndim!=3 or x.shape[1:]!=(7,5):raise ValueError("Ridge explanation requires exactly the same35 ordered features")
    b=np.zeros_like(x) if baseline is None else np.broadcast_to(np.asarray(baseline,float),x.shape)
    coef=np.asarray(coefficients,float).reshape(-1)
    if coef.shape!=(35,) or not np.isfinite(x).all() or not np.isfinite(b).all() or not np.isfinite(coef).all() or not np.isfinite(intercept):raise ValueError("Finite frozen linear parameters and inputs required")
    raw=(x.reshape(len(x),35)*coef).sum(1)+intercept;raw_base=(b.reshape(len(x),35)*coef).sum(1)+intercept
    difference=raw-raw_base;clipped=np.maximum(raw,0)-np.maximum(raw_base,0)
    factor=np.divide(clipped,difference,out=(raw>0).astype(float),where=np.abs(difference)>1e-14)
    attribution=(x-b)*coef.reshape(1,7,5)*factor[:,None,None]
    return {"method":"analytic_integrated_contribution_for_clipped_linear_regression",
        "private_attributions":attribution,"private_prediction_difference":clipped,
        "completeness_max_absolute_residual":float(np.max(np.abs(clipped-attribution.sum((1,2))))),
        "causal_effect_claimed":False}


def explain_naive_lag(observed_past_labels,*,lag,training_mean):
    x=np.asarray(observed_past_labels,float)
    if x.ndim!=2 or lag not in (1,7) or x.shape[1]<lag or not np.isfinite(x).all() or (x<0).any() or not np.isfinite(training_mean) or training_mean<0:
        raise ValueError("Supply only finite observed pre-origin target history and lag1/7")
    attribution=np.zeros_like(x);attribution[:,-lag]=x[:,-lag]-training_mean
    return {"method":"analytic_selected_observed_target_lag","lag":lag,
        "private_attributions":attribution,"private_prediction":x[:,-lag],
        "baseline_training_mean":float(training_mean),"information_set":"endogenous pre-origin labels, distinct from D7OHLCV"}


def causal_permutation_importance(predict_history,observed_history,truth,origins,*,groups,
        eligible=None,repetitions=2000,block_length=7,seed=202611070):
    """Permute only positions already observed at EACH origin, never future rows.

    Stateful ARIMA callers must clone/replay their frozen state causally. This
    callback receives interventions, not permission to refit model parameters.
    """
    x=np.asarray(observed_history,float);days=_dates(origins);mask=_mask(len(days),eligible);y=np.asarray(truth,float)
    if x.ndim!=3 or len(x)!=len(days) or y.shape!=(len(days),) or not np.isfinite(x[mask]).all() or not np.isfinite(y[mask]).all():raise ValueError("Aligned observed-only history and daily truth required")
    clean=_predict(predict_history,x[mask]);dense=np.full((len(clean),len(days)),np.nan);dense[:,mask]=clean
    base_error=dense-y;draws=_block_draws(len(days),block_length,repetitions,seed)
    base=_score(base_error,mask);base_boot=_boot_score(base_error,mask,draws);records=[]
    for name,channels in groups.items():
        channels=list(channels)
        if not channels or any(c<0 or c>=x.shape[2] for c in channels):raise ValueError("Valid explicit history-feature groups required")
        transformed=x.copy()
        for index in np.flatnonzero(mask):
            rng=_day_rng(seed,str(name),days[index]);order=rng.permutation(x.shape[1])
            transformed[index][:,channels]=x[index][order][:,channels]
        pred=_predict(predict_history,transformed[mask]);densep=np.full_like(dense,np.nan);densep[:,mask]=pred
        errors=densep-y;score=_score(errors,mask);boot=_boot_score(errors,mask,draws)
        records.append({"group":name,"RMSE_change":score-base,"paired_difference_interval":_interval(boot-base_boot,repetitions)})
    return {"method":"within_observed_history_permutation","groups":records,"future_history_rows_used":False,
        "stateful_callback_requires_frozen_causal_replay":True,"new_model_fits":0,"new_hypothesis_decisions":0}


def _family(model):
    metadata=getattr(model,'architecture_metadata',{})
    name=metadata.get('model',getattr(model,'family',None))
    if name not in TOP10:raise ValueError("Three-component registry applies to the preselected best Top10 model only")
    return name


def ablation_definitions(model):
    """Resolve exactly three prespecified single-component lesions to REAL paths."""
    name=_family(model);modules=dict(model.named_modules());readout='readout' if 'readout' in modules else 'head'
    if not isinstance(modules.get(readout),nn.Linear) or modules[readout].in_features<11:raise ValueError("Known scalar readout with final10 moment inputs required")
    def action(path,mode,freeze=None,freeze_parameters=None):
        if path not in modules:raise ValueError("Registered ablation module missing: "+path)
        return {'path':path,'mode':mode,'freeze_paths':list([path] if freeze is None else freeze),'freeze_parameters':list(freeze_parameters or [])}
    def paths(suffix):return [p for p in modules if p.endswith(suffix)]
    result={'without_moments':{'component':'amplitude moments in scalar task readout','actions':[action(readout,'mask_last10',freeze=[])]}}
    if name=='TSMixer':
        result['without_temporal_mixing']={'component':'all residual temporal-mixing branches','actions':[action(p,'zero',freeze=[p,p.rsplit('.',1)[0]+'.temporal_norm']) for p in paths('.temporal')]}
        result['without_feature_mixing']={'component':'all residual channel-mixing branches','actions':[action(p,'zero',freeze=[p.rsplit('.',1)[0]+'.feature_norm',p.rsplit('.',1)[0]+'.feature_1',p]) for p in paths('.feature_2')]}
    elif name=='TiDE':
        result['without_covariate_encoder']={'component':'shared past/fixed-zero-future covariate encoder','actions':[action('feature_encoder','zero')]}
        result['without_linear_skip']={'component':'global learned linear history skip','actions':[action('residual_projection','zero')]}
    elif name=='TimeMixer':
        result['without_seasonal_mixing']={'component':'all multiscale seasonal branches','actions':[action(p,'zero') for p in paths('.mixing_multi_scale_season')]}
        result['without_trend_mixing']={'component':'all multiscale trend branches','actions':[action(p,'zero') for p in paths('.mixing_multi_scale_trend')]}
    elif name=='TimesNet':
        result['without_periodic_mixing']={'component':'periodic2D residual branches, leaving their identity residual','actions':[action('backbone.model.'+str(i),'identity') for i in range(len(model.backbone.model))]}
        result['without_learned_temporal_extension']={'component':'learned7to8 latent forecast projection, replaced by deterministic last-token replication','actions':[action('backbone.predict_linear','extend_last')]}
    elif name=='PatchTST':
        result['without_patch_attention']={'component':'all residual patch-attention updates; FF/residuals retained','actions':[action(p,'zero_first') for p in paths('.self_attn')]}
        result['without_RevIN']={'component':'observed-window RevIN normalization/affine before unchanged source patching','actions':[action('backbone.revin_layer','identity')]}
    elif name=='Informer':
        chosen=[p for p in modules if p.startswith('backbone.encoder.encoders.') and p.endswith('.attention')]
        result['without_encoder_attention']={'component':'ProbSparse encoder attention updates in both stacks, residualFF retained','actions':[action(p,'zero_first') for p in chosen]}
        conv=[p for p in modules if p.startswith('backbone.encoder.encoders.') and '.conv_layers.' in p and p.rsplit('.',1)[-1].isdigit()]
        result['without_learned_distillation']={'component':'learned convolutional distillation replaced by fixed average pooling with identical length','actions':[action(p,'pool_half') for p in conv]}
    elif name=='Autoformer':
        chosen=[p for p in modules if p.startswith('backbone.encoder.attn_layers.') and p.endswith('.attention')]
        result['without_encoder_autocorrelation']={'component':'encoder autocorrelation updates; decomposition and residualFF retained','actions':[action(p,'zero_first') for p in chosen]}
        result['without_learned_decoder_trend']={'component':'learned progressive decoder trend residual, preserving observed trend and seasonal decoder','actions':[action('backbone.decoder.layers.'+str(i)+'.projection','zero') for i in range(len(model.backbone.decoder.layers))]}
    elif name=='iTransformer':
        result['without_inverted_attention']={'component':'inverted variable-attention updates, residualFF retained','actions':[action(p,'zero_first') for p in modules if p.startswith('backbone.encoder.attn_layers.') and p.endswith('.attention')]}
        result['without_final_normalization']={'component':'final encoder LayerNorm','actions':[action('backbone.encoder.norm','identity')]}
    elif name=='Crossformer':
        actions=[]
        for path in paths('.dim_receiver'):
            parent=path.rsplit('.',1)[0]
            actions.append(action(path,'zero',freeze=[path,parent+'.dim_sender'],freeze_parameters=[parent+'.router']))
        result['without_cross_variable_routing']={'component':'TSA cross-variable router communications, temporal attention/FF retained','actions':actions}
        result['without_multiscale_output_sum']={'component':'fine/intermediate additive decoder predictions removed; finalcoarse prediction and decoder cascade retained','actions':[action('backbone.decoder.decode_layers.'+str(i)+'.linear_pred','zero') for i in range(len(model.backbone.decoder.decode_layers)-1)]}
    else:
        result['without_reverse_SSM']={'component':'reverse-variable selective-state branch in each bidirectional layer','actions':[action('layers.'+str(i)+'.attention_r','zero') for i in range(len(model.layers))]}
        result['without_FF_refinement']={'component':'post-SSM residual feed-forward refinement, both selective scans retained','actions':[action('layers.'+str(i)+'.conv2','zero',freeze=['layers.'+str(i)+'.conv1','layers.'+str(i)+'.conv2']) for i in range(len(model.layers))]}
    if len(result)!=3 or any(not value['actions'] for value in result.values()):raise ValueError("Exactly three nonempty real component definitions are required")
    for key,record in result.items():
        record.update({'id':key,'model':name,'unitary_lesion':True,'retraining_required':True,'inference_deletion_is_not_this_experiment':True})
    fixed=globals().get('ABLATION_BLUEPRINTS')
    if fixed is not None and result!=fixed[name]:raise ValueError("Actual component routes differ from the registered final-backbone blueprint")
    return result


def _zero_tree(output):
    if isinstance(output,torch.Tensor):return torch.zeros_like(output)
    if isinstance(output,tuple):return tuple(_zero_tree(x) for x in output)
    if isinstance(output,list):return [_zero_tree(x) for x in output]
    if output is None:return None
    raise ValueError("Unexpected output in registered branch lesion")


@dataclass
class AblationHandle:
    model: nn.Module
    handles: list
    parameter_status: dict
    original_metadata: dict
    receipt: dict

    def remove(self):
        if getattr(self.model,'_e2_fit_started',False):raise ValueError("Do not remove an ablation after its new training began")
        for handle in self.handles:handle.remove()
        for name,p in self.model.named_parameters():p.requires_grad_(self.parameter_status[name])
        self.model.architecture_metadata=self.original_metadata
        delattr(self.model,'_e2_ablation_definition')
        self.handles=[]


def apply_ablation(model,ablation_id,*,application_stage='pre_fit_definition_only'):
    if (application_stage!='pre_fit_definition_only' or not model.training
            or getattr(model,'_e2_fit_started',False) or hasattr(model,'_e2_ablation_definition')):
        raise ValueError("Apply exactly one lesion to a fresh training-mode model BEFORE any optimizer fit")
    definitions=ablation_definitions(model)
    if ablation_id not in definitions:raise ValueError("Unknown predeclared component lesion")
    definition=copy.deepcopy(definitions[ablation_id]);modules=dict(model.named_modules());parameters=dict(model.named_parameters())
    status={n:p.requires_grad for n,p in parameters.items()};metadata=copy.deepcopy(model.architecture_metadata);handles=[]
    # Resolve the entire lesion transaction before installing even its first hook.
    for action in definition['actions']:
        if any(path not in modules for path in action['freeze_paths']) or any(path not in parameters for path in action['freeze_parameters']):
            raise ValueError("A registered ablation freeze path or parameter is absent")
    for action in definition['actions']:
        path,mode=action['path'],action['mode'];module=modules[path]
        if mode=='mask_last10':
            def mask_input(layer,args):
                x=args[0].clone();x[...,-10:]=0
                return (x,*args[1:])
            handles.append(module.register_forward_pre_hook(mask_input))
        else:
            def hook(layer,args,output,mode=mode):
                if mode=='zero':return _zero_tree(output)
                if mode=='zero_first':return (_zero_tree(output[0]),*output[1:]) if isinstance(output,tuple) else _zero_tree(output)
                if mode=='identity':return args[0]
                if mode=='extend_last':return torch.cat((args[0],args[0][...,-1:]),dim=-1)
                if mode=='pool_half':return torch.nn.functional.avg_pool1d(args[0].transpose(1,2),3,stride=2,padding=1,count_include_pad=False).transpose(1,2)
                raise ValueError("Unknown registered ablation operator")
            handles.append(module.register_forward_hook(hook))
        for freeze_path in action['freeze_paths']:
            if freeze_path not in modules:raise ValueError("Missing module to freeze in lesion: "+freeze_path)
            for p in modules[freeze_path].parameters():p.requires_grad_(False)
        for parameter in action['freeze_parameters']:
            if parameter not in parameters:raise ValueError("Missing registered parameter to freeze")
            parameters[parameter].requires_grad_(False)
    model._e2_ablation_definition=definition
    model.architecture_metadata={**metadata,'ablation_definition':definition}
    receipt={'definition':definition,'application_stage':application_stage,
        'model_fits_performed':0,'retrained_ablation_result_claimed':False,
        'new_training_receipt_required_before_scoring':True,
        'operator_must_be_reinstalled_when_loading_ablated_checkpoint':True,
        'remaining_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_stored_parameters':sum(p.numel() for p in model.parameters())}
    return AblationHandle(model,handles,status,metadata,receipt)


def public_diagnostic_summary(value):
    if isinstance(value,dict):
        hidden={'raw_OHLCV_mean','x_mean','x_scale','y_scale','linear_coefficients','linear_intercept'}
        return {k:public_diagnostic_summary(v) for k,v in value.items()
            if not k.startswith('private_') and k not in hidden
            and not (k=='baseline' and not isinstance(v,str))}
    if isinstance(value,(list,tuple)):return [public_diagnostic_summary(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.integer,np.floating,np.bool_)):return value.item()
    return value


def summarize_fidelity(fidelity,origins,*,eligible=None,repetitions=2000,block_length=7,seed=202611069):
    fields=['deletion_AUC','insertion_AUC','random_deletion_AUC','random_insertion_AUC',
            'deletion_advantage','insertion_advantage']
    result={}
    for field in fields:
        key='private_'+field
        if key not in fidelity:raise ValueError('Complete matched deletion/insertion curves are required')
        result[field]=summarize_daily_values(fidelity[key],origins,eligible=eligible,
            repetitions=repetitions,block_length=block_length,seed=seed)
    return {'metric_summaries':result,'normalization':fidelity['normalization'],
        'random_rankings':fidelity['random_rankings'],'seed':fidelity['seed'],
        'direction':'positive paired advantage means stronger response than fixed random ranks; not a causal effect',
        'uncertainty':'nominal temporal95percentile, no added formal hypothesis decisions'}


def attribution_summary(attributions,origins,*,groups=None,eligible=None,repetitions=2000,block_length=7,seed=202611069):
    days=_dates(origins);mask=_mask(len(days),eligible);definition=_groups(groups)
    a=_group_values(attributions,len(days),definition)
    if not np.isfinite(a[mask]).all():raise ValueError('Aligned finite group attribution maps required')
    magnitude=np.abs(a);denominator=magnitude.sum(axis=1)
    normalized=np.divide(magnitude,denominator[:,None],out=np.full_like(magnitude,np.nan),where=denominator[:,None]>0)
    records=[]
    for j,group in enumerate(definition):
        records.append({**group,**summarize_daily_values(normalized[:,j],origins,eligible=mask,repetitions=repetitions,block_length=block_length,seed=seed)})
    by_channel=[{'kind':kind,**summarize_daily_values(normalized[:,[i for i,g in enumerate(definition) if g['kind']==kind]].sum(1),origins,eligible=mask,repetitions=repetitions,block_length=block_length,seed=seed)} for kind in ('OHLC_prices','Volume')]
    by_lag=[{'lag_days':lag,**summarize_daily_values(normalized[:,[i for i,g in enumerate(definition) if g['lag_days']==lag]].sum(1),origins,eligible=mask,repetitions=repetitions,block_length=block_length,seed=seed)} for lag in range(7,0,-1)]
    return {'groups':records,'channels':by_channel,'lags':by_lag,
        'normalization':'absolute group attribution share per eligible origin; allzero maps are not estimable',
        'individual_OHLC_field_importance_identifiable':False,'new_hypothesis_decisions':0,'causal_effect_claimed':False,
        'confidence_intervals':'nominal descriptive, not simultaneous across14groups'}


# Actual named-module bindings checked on all ten full final backbones.
ABLATION_BLUEPRINTS = {'TSMixer': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                 'actions': [{'path': 'readout',
                                              'mode': 'mask_last10',
                                              'freeze_paths': [],
                                              'freeze_parameters': []}],
                                 'id': 'without_moments',
                                 'model': 'TSMixer',
                                 'unitary_lesion': True,
                                 'retraining_required': True,
                                 'inference_deletion_is_not_this_experiment': True},
             'without_temporal_mixing': {'component': 'all residual temporal-mixing branches',
                                         'actions': [{'path': 'blocks.0.temporal',
                                                      'mode': 'zero',
                                                      'freeze_paths': ['blocks.0.temporal',
                                                                       'blocks.0.temporal_norm'],
                                                      'freeze_parameters': []},
                                                     {'path': 'blocks.1.temporal',
                                                      'mode': 'zero',
                                                      'freeze_paths': ['blocks.1.temporal',
                                                                       'blocks.1.temporal_norm'],
                                                      'freeze_parameters': []},
                                                     {'path': 'blocks.2.temporal',
                                                      'mode': 'zero',
                                                      'freeze_paths': ['blocks.2.temporal',
                                                                       'blocks.2.temporal_norm'],
                                                      'freeze_parameters': []},
                                                     {'path': 'blocks.3.temporal',
                                                      'mode': 'zero',
                                                      'freeze_paths': ['blocks.3.temporal',
                                                                       'blocks.3.temporal_norm'],
                                                      'freeze_parameters': []},
                                                     {'path': 'blocks.4.temporal',
                                                      'mode': 'zero',
                                                      'freeze_paths': ['blocks.4.temporal',
                                                                       'blocks.4.temporal_norm'],
                                                      'freeze_parameters': []},
                                                     {'path': 'blocks.5.temporal',
                                                      'mode': 'zero',
                                                      'freeze_paths': ['blocks.5.temporal',
                                                                       'blocks.5.temporal_norm'],
                                                      'freeze_parameters': []}],
                                         'id': 'without_temporal_mixing',
                                         'model': 'TSMixer',
                                         'unitary_lesion': True,
                                         'retraining_required': True,
                                         'inference_deletion_is_not_this_experiment': True},
             'without_feature_mixing': {'component': 'all residual channel-mixing branches',
                                        'actions': [{'path': 'blocks.0.feature_2',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['blocks.0.feature_norm',
                                                                      'blocks.0.feature_1',
                                                                      'blocks.0.feature_2'],
                                                     'freeze_parameters': []},
                                                    {'path': 'blocks.1.feature_2',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['blocks.1.feature_norm',
                                                                      'blocks.1.feature_1',
                                                                      'blocks.1.feature_2'],
                                                     'freeze_parameters': []},
                                                    {'path': 'blocks.2.feature_2',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['blocks.2.feature_norm',
                                                                      'blocks.2.feature_1',
                                                                      'blocks.2.feature_2'],
                                                     'freeze_parameters': []},
                                                    {'path': 'blocks.3.feature_2',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['blocks.3.feature_norm',
                                                                      'blocks.3.feature_1',
                                                                      'blocks.3.feature_2'],
                                                     'freeze_parameters': []},
                                                    {'path': 'blocks.4.feature_2',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['blocks.4.feature_norm',
                                                                      'blocks.4.feature_1',
                                                                      'blocks.4.feature_2'],
                                                     'freeze_parameters': []},
                                                    {'path': 'blocks.5.feature_2',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['blocks.5.feature_norm',
                                                                      'blocks.5.feature_1',
                                                                      'blocks.5.feature_2'],
                                                     'freeze_parameters': []}],
                                        'id': 'without_feature_mixing',
                                        'model': 'TSMixer',
                                        'unitary_lesion': True,
                                        'retraining_required': True,
                                        'inference_deletion_is_not_this_experiment': True}},
 'TiDE': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                              'actions': [{'path': 'readout',
                                           'mode': 'mask_last10',
                                           'freeze_paths': [],
                                           'freeze_parameters': []}],
                              'id': 'without_moments',
                              'model': 'TiDE',
                              'unitary_lesion': True,
                              'retraining_required': True,
                              'inference_deletion_is_not_this_experiment': True},
          'without_covariate_encoder': {'component': 'shared past/fixed-zero-future covariate encoder',
                                        'actions': [{'path': 'feature_encoder',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['feature_encoder'],
                                                     'freeze_parameters': []}],
                                        'id': 'without_covariate_encoder',
                                        'model': 'TiDE',
                                        'unitary_lesion': True,
                                        'retraining_required': True,
                                        'inference_deletion_is_not_this_experiment': True},
          'without_linear_skip': {'component': 'global learned linear history skip',
                                  'actions': [{'path': 'residual_projection',
                                               'mode': 'zero',
                                               'freeze_paths': ['residual_projection'],
                                               'freeze_parameters': []}],
                                  'id': 'without_linear_skip',
                                  'model': 'TiDE',
                                  'unitary_lesion': True,
                                  'retraining_required': True,
                                  'inference_deletion_is_not_this_experiment': True}},
 'TimeMixer': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                   'actions': [{'path': 'readout',
                                                'mode': 'mask_last10',
                                                'freeze_paths': [],
                                                'freeze_parameters': []}],
                                   'id': 'without_moments',
                                   'model': 'TimeMixer',
                                   'unitary_lesion': True,
                                   'retraining_required': True,
                                   'inference_deletion_is_not_this_experiment': True},
               'without_seasonal_mixing': {'component': 'all multiscale seasonal branches',
                                           'actions': [{'path': 'backbone.pdm_blocks.0.mixing_multi_scale_season',
                                                        'mode': 'zero',
                                                        'freeze_paths': ['backbone.pdm_blocks.0.mixing_multi_scale_season'],
                                                        'freeze_parameters': []},
                                                       {'path': 'backbone.pdm_blocks.1.mixing_multi_scale_season',
                                                        'mode': 'zero',
                                                        'freeze_paths': ['backbone.pdm_blocks.1.mixing_multi_scale_season'],
                                                        'freeze_parameters': []}],
                                           'id': 'without_seasonal_mixing',
                                           'model': 'TimeMixer',
                                           'unitary_lesion': True,
                                           'retraining_required': True,
                                           'inference_deletion_is_not_this_experiment': True},
               'without_trend_mixing': {'component': 'all multiscale trend branches',
                                        'actions': [{'path': 'backbone.pdm_blocks.0.mixing_multi_scale_trend',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['backbone.pdm_blocks.0.mixing_multi_scale_trend'],
                                                     'freeze_parameters': []},
                                                    {'path': 'backbone.pdm_blocks.1.mixing_multi_scale_trend',
                                                     'mode': 'zero',
                                                     'freeze_paths': ['backbone.pdm_blocks.1.mixing_multi_scale_trend'],
                                                     'freeze_parameters': []}],
                                        'id': 'without_trend_mixing',
                                        'model': 'TimeMixer',
                                        'unitary_lesion': True,
                                        'retraining_required': True,
                                        'inference_deletion_is_not_this_experiment': True}},
 'TimesNet': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                  'actions': [{'path': 'readout',
                                               'mode': 'mask_last10',
                                               'freeze_paths': [],
                                               'freeze_parameters': []}],
                                  'id': 'without_moments',
                                  'model': 'TimesNet',
                                  'unitary_lesion': True,
                                  'retraining_required': True,
                                  'inference_deletion_is_not_this_experiment': True},
              'without_periodic_mixing': {'component': 'periodic2D residual branches, leaving their identity '
                                                       'residual',
                                          'actions': [{'path': 'backbone.model.0',
                                                       'mode': 'identity',
                                                       'freeze_paths': ['backbone.model.0'],
                                                       'freeze_parameters': []},
                                                      {'path': 'backbone.model.1',
                                                       'mode': 'identity',
                                                       'freeze_paths': ['backbone.model.1'],
                                                       'freeze_parameters': []}],
                                          'id': 'without_periodic_mixing',
                                          'model': 'TimesNet',
                                          'unitary_lesion': True,
                                          'retraining_required': True,
                                          'inference_deletion_is_not_this_experiment': True},
              'without_learned_temporal_extension': {'component': 'learned7to8 latent forecast projection, '
                                                                  'replaced by deterministic last-token '
                                                                  'replication',
                                                     'actions': [{'path': 'backbone.predict_linear',
                                                                  'mode': 'extend_last',
                                                                  'freeze_paths': ['backbone.predict_linear'],
                                                                  'freeze_parameters': []}],
                                                     'id': 'without_learned_temporal_extension',
                                                     'model': 'TimesNet',
                                                     'unitary_lesion': True,
                                                     'retraining_required': True,
                                                     'inference_deletion_is_not_this_experiment': True}},
 'PatchTST': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                  'actions': [{'path': 'readout',
                                               'mode': 'mask_last10',
                                               'freeze_paths': [],
                                               'freeze_parameters': []}],
                                  'id': 'without_moments',
                                  'model': 'PatchTST',
                                  'unitary_lesion': True,
                                  'retraining_required': True,
                                  'inference_deletion_is_not_this_experiment': True},
              'without_patch_attention': {'component': 'all residual patch-attention updates; FF/residuals '
                                                       'retained',
                                          'actions': [{'path': 'backbone.backbone.encoder.layers.0.self_attn',
                                                       'mode': 'zero_first',
                                                       'freeze_paths': ['backbone.backbone.encoder.layers.0.self_attn'],
                                                       'freeze_parameters': []},
                                                      {'path': 'backbone.backbone.encoder.layers.1.self_attn',
                                                       'mode': 'zero_first',
                                                       'freeze_paths': ['backbone.backbone.encoder.layers.1.self_attn'],
                                                       'freeze_parameters': []},
                                                      {'path': 'backbone.backbone.encoder.layers.2.self_attn',
                                                       'mode': 'zero_first',
                                                       'freeze_paths': ['backbone.backbone.encoder.layers.2.self_attn'],
                                                       'freeze_parameters': []}],
                                          'id': 'without_patch_attention',
                                          'model': 'PatchTST',
                                          'unitary_lesion': True,
                                          'retraining_required': True,
                                          'inference_deletion_is_not_this_experiment': True},
              'without_RevIN': {'component': 'observed-window RevIN normalization/affine before unchanged source '
                                             'patching',
                                'actions': [{'path': 'backbone.revin_layer',
                                             'mode': 'identity',
                                             'freeze_paths': ['backbone.revin_layer'],
                                             'freeze_parameters': []}],
                                'id': 'without_RevIN',
                                'model': 'PatchTST',
                                'unitary_lesion': True,
                                'retraining_required': True,
                                'inference_deletion_is_not_this_experiment': True}},
 'Informer': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                  'actions': [{'path': 'head',
                                               'mode': 'mask_last10',
                                               'freeze_paths': [],
                                               'freeze_parameters': []}],
                                  'id': 'without_moments',
                                  'model': 'Informer',
                                  'unitary_lesion': True,
                                  'retraining_required': True,
                                  'inference_deletion_is_not_this_experiment': True},
              'without_encoder_attention': {'component': 'ProbSparse encoder attention updates in both stacks, '
                                                         'residualFF retained',
                                            'actions': [{'path': 'backbone.encoder.encoders.0.attn_layers.0.attention',
                                                         'mode': 'zero_first',
                                                         'freeze_paths': ['backbone.encoder.encoders.0.attn_layers.0.attention'],
                                                         'freeze_parameters': []},
                                                        {'path': 'backbone.encoder.encoders.0.attn_layers.1.attention',
                                                         'mode': 'zero_first',
                                                         'freeze_paths': ['backbone.encoder.encoders.0.attn_layers.1.attention'],
                                                         'freeze_parameters': []},
                                                        {'path': 'backbone.encoder.encoders.0.attn_layers.2.attention',
                                                         'mode': 'zero_first',
                                                         'freeze_paths': ['backbone.encoder.encoders.0.attn_layers.2.attention'],
                                                         'freeze_parameters': []},
                                                        {'path': 'backbone.encoder.encoders.1.attn_layers.0.attention',
                                                         'mode': 'zero_first',
                                                         'freeze_paths': ['backbone.encoder.encoders.1.attn_layers.0.attention'],
                                                         'freeze_parameters': []}],
                                            'id': 'without_encoder_attention',
                                            'model': 'Informer',
                                            'unitary_lesion': True,
                                            'retraining_required': True,
                                            'inference_deletion_is_not_this_experiment': True},
              'without_learned_distillation': {'component': 'learned convolutional distillation replaced by fixed '
                                                            'average pooling with identical length',
                                               'actions': [{'path': 'backbone.encoder.encoders.0.conv_layers.0',
                                                            'mode': 'pool_half',
                                                            'freeze_paths': ['backbone.encoder.encoders.0.conv_layers.0'],
                                                            'freeze_parameters': []},
                                                           {'path': 'backbone.encoder.encoders.0.conv_layers.1',
                                                            'mode': 'pool_half',
                                                            'freeze_paths': ['backbone.encoder.encoders.0.conv_layers.1'],
                                                            'freeze_parameters': []}],
                                               'id': 'without_learned_distillation',
                                               'model': 'Informer',
                                               'unitary_lesion': True,
                                               'retraining_required': True,
                                               'inference_deletion_is_not_this_experiment': True}},
 'Autoformer': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                    'actions': [{'path': 'head',
                                                 'mode': 'mask_last10',
                                                 'freeze_paths': [],
                                                 'freeze_parameters': []}],
                                    'id': 'without_moments',
                                    'model': 'Autoformer',
                                    'unitary_lesion': True,
                                    'retraining_required': True,
                                    'inference_deletion_is_not_this_experiment': True},
                'without_encoder_autocorrelation': {'component': 'encoder autocorrelation updates; decomposition '
                                                                 'and residualFF retained',
                                                    'actions': [{'path': 'backbone.encoder.attn_layers.0.attention',
                                                                 'mode': 'zero_first',
                                                                 'freeze_paths': ['backbone.encoder.attn_layers.0.attention'],
                                                                 'freeze_parameters': []},
                                                                {'path': 'backbone.encoder.attn_layers.1.attention',
                                                                 'mode': 'zero_first',
                                                                 'freeze_paths': ['backbone.encoder.attn_layers.1.attention'],
                                                                 'freeze_parameters': []}],
                                                    'id': 'without_encoder_autocorrelation',
                                                    'model': 'Autoformer',
                                                    'unitary_lesion': True,
                                                    'retraining_required': True,
                                                    'inference_deletion_is_not_this_experiment': True},
                'without_learned_decoder_trend': {'component': 'learned progressive decoder trend residual, '
                                                               'preserving observed trend and seasonal decoder',
                                                  'actions': [{'path': 'backbone.decoder.layers.0.projection',
                                                               'mode': 'zero',
                                                               'freeze_paths': ['backbone.decoder.layers.0.projection'],
                                                               'freeze_parameters': []}],
                                                  'id': 'without_learned_decoder_trend',
                                                  'model': 'Autoformer',
                                                  'unitary_lesion': True,
                                                  'retraining_required': True,
                                                  'inference_deletion_is_not_this_experiment': True}},
 'iTransformer': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                      'actions': [{'path': 'head',
                                                   'mode': 'mask_last10',
                                                   'freeze_paths': [],
                                                   'freeze_parameters': []}],
                                      'id': 'without_moments',
                                      'model': 'iTransformer',
                                      'unitary_lesion': True,
                                      'retraining_required': True,
                                      'inference_deletion_is_not_this_experiment': True},
                  'without_inverted_attention': {'component': 'inverted variable-attention updates, residualFF '
                                                              'retained',
                                                 'actions': [{'path': 'backbone.encoder.attn_layers.0.attention',
                                                              'mode': 'zero_first',
                                                              'freeze_paths': ['backbone.encoder.attn_layers.0.attention'],
                                                              'freeze_parameters': []},
                                                             {'path': 'backbone.encoder.attn_layers.1.attention',
                                                              'mode': 'zero_first',
                                                              'freeze_paths': ['backbone.encoder.attn_layers.1.attention'],
                                                              'freeze_parameters': []}],
                                                 'id': 'without_inverted_attention',
                                                 'model': 'iTransformer',
                                                 'unitary_lesion': True,
                                                 'retraining_required': True,
                                                 'inference_deletion_is_not_this_experiment': True},
                  'without_final_normalization': {'component': 'final encoder LayerNorm',
                                                  'actions': [{'path': 'backbone.encoder.norm',
                                                               'mode': 'identity',
                                                               'freeze_paths': ['backbone.encoder.norm'],
                                                               'freeze_parameters': []}],
                                                  'id': 'without_final_normalization',
                                                  'model': 'iTransformer',
                                                  'unitary_lesion': True,
                                                  'retraining_required': True,
                                                  'inference_deletion_is_not_this_experiment': True}},
 'Crossformer': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                     'actions': [{'path': 'head',
                                                  'mode': 'mask_last10',
                                                  'freeze_paths': [],
                                                  'freeze_parameters': []}],
                                     'id': 'without_moments',
                                     'model': 'Crossformer',
                                     'unitary_lesion': True,
                                     'retraining_required': True,
                                     'inference_deletion_is_not_this_experiment': True},
                 'without_cross_variable_routing': {'component': 'TSA cross-variable router communications, '
                                                                 'temporal attention/FF retained',
                                                    'actions': [{'path': 'backbone.encoder.encode_blocks.0.encode_layers.0.dim_receiver',
                                                                 'mode': 'zero',
                                                                 'freeze_paths': ['backbone.encoder.encode_blocks.0.encode_layers.0.dim_receiver',
                                                                                  'backbone.encoder.encode_blocks.0.encode_layers.0.dim_sender'],
                                                                 'freeze_parameters': ['backbone.encoder.encode_blocks.0.encode_layers.0.router']},
                                                                {'path': 'backbone.encoder.encode_blocks.1.encode_layers.0.dim_receiver',
                                                                 'mode': 'zero',
                                                                 'freeze_paths': ['backbone.encoder.encode_blocks.1.encode_layers.0.dim_receiver',
                                                                                  'backbone.encoder.encode_blocks.1.encode_layers.0.dim_sender'],
                                                                 'freeze_parameters': ['backbone.encoder.encode_blocks.1.encode_layers.0.router']},
                                                                {'path': 'backbone.encoder.encode_blocks.2.encode_layers.0.dim_receiver',
                                                                 'mode': 'zero',
                                                                 'freeze_paths': ['backbone.encoder.encode_blocks.2.encode_layers.0.dim_receiver',
                                                                                  'backbone.encoder.encode_blocks.2.encode_layers.0.dim_sender'],
                                                                 'freeze_parameters': ['backbone.encoder.encode_blocks.2.encode_layers.0.router']},
                                                                {'path': 'backbone.decoder.decode_layers.0.self_attention.dim_receiver',
                                                                 'mode': 'zero',
                                                                 'freeze_paths': ['backbone.decoder.decode_layers.0.self_attention.dim_receiver',
                                                                                  'backbone.decoder.decode_layers.0.self_attention.dim_sender'],
                                                                 'freeze_parameters': ['backbone.decoder.decode_layers.0.self_attention.router']},
                                                                {'path': 'backbone.decoder.decode_layers.1.self_attention.dim_receiver',
                                                                 'mode': 'zero',
                                                                 'freeze_paths': ['backbone.decoder.decode_layers.1.self_attention.dim_receiver',
                                                                                  'backbone.decoder.decode_layers.1.self_attention.dim_sender'],
                                                                 'freeze_parameters': ['backbone.decoder.decode_layers.1.self_attention.router']},
                                                                {'path': 'backbone.decoder.decode_layers.2.self_attention.dim_receiver',
                                                                 'mode': 'zero',
                                                                 'freeze_paths': ['backbone.decoder.decode_layers.2.self_attention.dim_receiver',
                                                                                  'backbone.decoder.decode_layers.2.self_attention.dim_sender'],
                                                                 'freeze_parameters': ['backbone.decoder.decode_layers.2.self_attention.router']},
                                                                {'path': 'backbone.decoder.decode_layers.3.self_attention.dim_receiver',
                                                                 'mode': 'zero',
                                                                 'freeze_paths': ['backbone.decoder.decode_layers.3.self_attention.dim_receiver',
                                                                                  'backbone.decoder.decode_layers.3.self_attention.dim_sender'],
                                                                 'freeze_parameters': ['backbone.decoder.decode_layers.3.self_attention.router']}],
                                                    'id': 'without_cross_variable_routing',
                                                    'model': 'Crossformer',
                                                    'unitary_lesion': True,
                                                    'retraining_required': True,
                                                    'inference_deletion_is_not_this_experiment': True},
                 'without_multiscale_output_sum': {'component': 'fine/intermediate additive decoder predictions '
                                                                'removed; finalcoarse prediction and decoder '
                                                                'cascade retained',
                                                   'actions': [{'path': 'backbone.decoder.decode_layers.0.linear_pred',
                                                                'mode': 'zero',
                                                                'freeze_paths': ['backbone.decoder.decode_layers.0.linear_pred'],
                                                                'freeze_parameters': []},
                                                               {'path': 'backbone.decoder.decode_layers.1.linear_pred',
                                                                'mode': 'zero',
                                                                'freeze_paths': ['backbone.decoder.decode_layers.1.linear_pred'],
                                                                'freeze_parameters': []},
                                                               {'path': 'backbone.decoder.decode_layers.2.linear_pred',
                                                                'mode': 'zero',
                                                                'freeze_paths': ['backbone.decoder.decode_layers.2.linear_pred'],
                                                                'freeze_parameters': []}],
                                                   'id': 'without_multiscale_output_sum',
                                                   'model': 'Crossformer',
                                                   'unitary_lesion': True,
                                                   'retraining_required': True,
                                                   'inference_deletion_is_not_this_experiment': True}},
 'S-Mamba': {'without_moments': {'component': 'amplitude moments in scalar task readout',
                                 'actions': [{'path': 'head',
                                              'mode': 'mask_last10',
                                              'freeze_paths': [],
                                              'freeze_parameters': []}],
                                 'id': 'without_moments',
                                 'model': 'S-Mamba',
                                 'unitary_lesion': True,
                                 'retraining_required': True,
                                 'inference_deletion_is_not_this_experiment': True},
             'without_reverse_SSM': {'component': 'reverse-variable selective-state branch in each bidirectional '
                                                  'layer',
                                     'actions': [{'path': 'layers.0.attention_r',
                                                  'mode': 'zero',
                                                  'freeze_paths': ['layers.0.attention_r'],
                                                  'freeze_parameters': []},
                                                 {'path': 'layers.1.attention_r',
                                                  'mode': 'zero',
                                                  'freeze_paths': ['layers.1.attention_r'],
                                                  'freeze_parameters': []}],
                                     'id': 'without_reverse_SSM',
                                     'model': 'S-Mamba',
                                     'unitary_lesion': True,
                                     'retraining_required': True,
                                     'inference_deletion_is_not_this_experiment': True},
             'without_FF_refinement': {'component': 'post-SSM residual feed-forward refinement, both selective '
                                                    'scans retained',
                                       'actions': [{'path': 'layers.0.conv2',
                                                    'mode': 'zero',
                                                    'freeze_paths': ['layers.0.conv1', 'layers.0.conv2'],
                                                    'freeze_parameters': []},
                                                   {'path': 'layers.1.conv2',
                                                    'mode': 'zero',
                                                    'freeze_paths': ['layers.1.conv1', 'layers.1.conv2'],
                                                    'freeze_parameters': []}],
                                       'id': 'without_FF_refinement',
                                       'model': 'S-Mamba',
                                       'unitary_lesion': True,
                                       'retraining_required': True,
                                       'inference_deletion_is_not_this_experiment': True}}}

def get_ablation_definitions(name):
    if name not in ABLATION_BLUEPRINTS:raise ValueError("No registered Top10 component definitions for "+str(name))
    return copy.deepcopy(ABLATION_BLUEPRINTS[name])


def _json_ready(value):
    from pathlib import Path
    from datetime import datetime
    if isinstance(value,dict):return {str(k):_json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json_ready(v) for v in value]
    if isinstance(value,np.ndarray):return _json_ready(value.tolist())
    if isinstance(value,(np.integer,np.floating,np.bool_)):return _json_ready(value.item())
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,(date,datetime,Path)):return str(value)
    if isinstance(value,(str,int,bool)) or value is None:return value
    raise TypeError('Only numerical metadata, not fitted objects or callbacks, can be serialized')


def preprocessing_fingerprint(preprocessing):
    """Exclude explanation receipts to avoid a self-referential scaler hash."""
    core={k:v for k,v in preprocessing.items() if not k.startswith('explanation_') and k!='scaler_sha256'}
    for field in ('x_mean','x_scale','y_scale'):
        if field not in core:raise ValueError('Missing frozen preprocessing field: '+field)
    return sha256(json.dumps(_json_ready(core),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _history_series(history_source):
    if not isinstance(history_source,dict) or 'dates' not in history_source or 'values' not in history_source:
        raise ValueError('A dense observed target-history series is required')
    days=_dates(history_source['dates']);values=np.asarray(history_source['values'],float)
    if values.shape!=(len(days),) or np.isinf(values).any() or (values[np.isfinite(values)]<0).any():
        raise ValueError('Target history must be nonnegative original-scale sigma with missing placeholders')
    return days,values


def _history_at(history_source,origin):
    days,values=_history_series(history_source);day=date.fromisoformat(str(origin)[:10])
    indices=np.flatnonzero(np.asarray([d<day for d in days]))
    if not len(indices) or days[indices[-1]]!=day-timedelta(days=1):
        raise ValueError('Observed target history must reach exactly the previous calendar day')
    return {'dates':[days[i].isoformat() for i in indices],'values':values[indices].copy()}


def _history_predict(callback,history,origin):
    if any(date.fromisoformat(str(d)[:10])>=date.fromisoformat(str(origin)[:10]) for d in history['dates']):
        raise ValueError('A history callback can never receive current or future target labels')
    point=float(np.asarray(callback(history,str(origin))).reshape(()))
    if not math.isfinite(point) or point<0:raise ValueError('Endogenous predictor returned an invalid volatility')
    return point


def perturb_target_history(history,*,kind,severity,fit_outcome_max,training_sigma_mean,seed=202611067):
    days,values=_history_series(history);cutoff=date.fromisoformat(str(fit_outcome_max)[:10])
    if kind not in SEVERITIES or severity<0 or not math.isfinite(float(training_sigma_mean)) or training_sigma_mean<0:
        raise ValueError('Unknown history fault or invalid frozen training mean')
    if kind=='missing' and severity>1:raise ValueError('History missingness is a probability')
    output=values.copy()
    for i,day in enumerate(days):
        if day<=cutoff or not math.isfinite(values[i]):continue
        rng=_day_rng(seed,'endogenous_'+kind,day)
        if kind=='noise':output[i]*=np.exp(severity*rng.normal())
        elif kind=='missing':
            if rng.random()<severity:output[i]=training_sigma_mean
        elif rng.random()<.05:output[i]*=np.exp(severity*(-1 if rng.random()<.5 else 1))
    if np.isinf(output).any():raise ValueError('Nonfinite target-history measurement stress')
    return {'dates':list(history['dates']),'values':output}, {
        'kind':kind,'severity':severity,'seed':seed,'input_information':'observed endogenous target history, not OHLCV',
        'parameters_refitted':False,'prefit_history_changed':False,'current_or_future_target_used':False,
        'history_sha256':_digest(output),'original_missing_positions_preserved':True}


def evaluate_endogenous_robustness(entry,history_source,truth,origins,*,eligible=None,
        repetitions=2000,block_length=7,seed=202611067):
    days=_dates(origins);mask=_mask(len(days),eligible);y=np.asarray(truth,float)
    if y.shape!=(len(days),) or not np.isfinite(y[mask]).all() or (y[mask]<0).any() or np.isfinite(y[~mask]).any():
        raise ValueError('Dense common outcome calendar required for endogenous robustness')
    cutoff=date.fromisoformat(str(entry['fit_outcome_max'])[:10]);mean=float(entry['training_sigma_mean'])
    if cutoff>days[0]-timedelta(days=8):raise ValueError('Endogenous fit cutoff violates the evaluation embargo')
    callback=entry['predict_from_history'];histories=[_history_at(history_source,d.isoformat()) for d in days]
    for h in histories:
        hd=[date.fromisoformat(d) for d in h['dates']]
        if min(hd)>cutoff+timedelta(days=1):raise ValueError('Frozen-state replay cannot silently omit post-fit calendar observations')
    clean=np.full((1,len(days)),np.nan)
    for i in np.flatnonzero(mask):clean[0,i]=_history_predict(callback,histories[i],origins[i])
    draws=_block_draws(len(days),block_length,repetitions,seed+1)
    base_error=clean-y;baseline=_score(base_error,mask);base_boot=_boot_score(base_error,mask,draws)
    rows=[];aucs=[];receipts=[]
    for kind,levels in SEVERITIES.items():
        curve=[baseline];bootcurve=[base_boot]
        for severity in levels:
            pred=np.full_like(clean,np.nan)
            for i in np.flatnonzero(mask):
                changed,receipt=perturb_target_history(histories[i],kind=kind,severity=severity,
                    fit_outcome_max=entry['fit_outcome_max'],training_sigma_mean=mean,seed=seed)
                pred[0,i]=_history_predict(callback,changed,origins[i]);receipts.append(receipt)
            error=pred-y;point=_score(error,mask);boot=_boot_score(error,mask,draws)
            with np.errstate(divide='ignore',invalid='ignore'):relative=(boot-base_boot)/base_boot
            rows.append({'kind':kind,'severity':severity,'RMSE_mean_seed':point,'clean_RMSE_mean_seed':baseline,
                'RMSE_difference':point-baseline,'relative_degradation':(point-baseline)/baseline if baseline>0 else None,
                'RMSE_interval':_interval(boot,repetitions),'paired_difference_interval':_interval(boot-base_boot,repetitions),
                'relative_degradation_interval':_interval(relative,repetitions)})
            curve.append(point);bootcurve.append(boot)
        coordinate=np.asarray((0,*levels),float)/max(levels)
        normalized=float(np.trapezoid(np.asarray(curve)/baseline,coordinate)) if baseline>0 else None
        with np.errstate(divide='ignore',invalid='ignore'):sample=np.trapezoid(np.stack(bootcurve,axis=1)/base_boot[:,None],coordinate,axis=1)
        aucs.append({'kind':kind,'normalized_score_AUC':normalized,'relative_degradation_AUC':normalized-1 if normalized is not None else None,
            'relative_degradation_AUC_interval':_interval(sample-1,repetitions)})
    return {'status':'evaluated_frozen_predictions','input_contract':'endogenous_observed_target_history',
        'input_faults_are_not_the_same_OHLCV_faults':True,'architectural_immunity_claimed':False,
        'calendar_origins':len(days),'eligible_unique_origins':int(mask.sum()),'seed_members':1,
        'deterministic_trajectory_replicated_as_independent_seeds':False,'rows':rows,'AUC':aucs,
        'bootstrap':{'replicates':repetitions,'block_days':block_length,'paired_full_calendar':True},
        'coefficient_parameters_refitted':False,'new_model_fits':0,'new_hypothesis_decisions':0,
        'provenance':{'clean_prediction_sha256':_digest(clean),'truth_sha256':_digest(y),
            'history_fault_receipt_sha256':sha256(json.dumps(_json_ready(receipts),sort_keys=True).encode()).hexdigest()}}


def _history_groups(history,origin,fit_outcome_max,lag):
    days=[date.fromisoformat(d) for d in history['dates']];day=date.fromisoformat(str(origin)[:10])
    cutoff=date.fromisoformat(str(fit_outcome_max)[:10])
    if lag in (1,7):
        return [{'id':f'lag{k}_sigma','indices':[days.index(day-timedelta(days=k))]} for k in range(int(lag),0,-1)]
    recent=[{'id':f'lag{k}_sigma','indices':[days.index(day-timedelta(days=k))]} for k in range(7,0,-1)]
    old=[i for i,d in enumerate(days) if cutoff<d<day-timedelta(days=7)]
    return [{'id':'older_observed_postfit_history','indices':old},*recent]


def explain_endogenous_history(entry,history_source,origins,*,eligible=None,repetitions=2000,
        block_length=7,seed=202611068,random_rankings=10):
    """Actual frozen-state group occlusion and random-controlled fidelity."""
    days=_dates(origins);mask=_mask(len(days),eligible);mean=float(entry['training_sigma_mean'])
    if not math.isfinite(mean) or mean<0:raise ValueError('Frozen training sigma mean required')
    callback=entry['predict_from_history'];histories=[_history_at(history_source,d.isoformat()) for d in days]
    lag=entry.get('lag');groupmaps=[_history_groups(h,d.isoformat(),entry['fit_outcome_max'],lag) for h,d in zip(histories,days)]
    labels=[{'id':g['id'],'input_information':'observed target history'} for g in groupmaps[0]];g=len(labels)
    attribution=np.full((len(days),g),np.nan);deletion_auc=np.full(len(days),np.nan);insertion_auc=deletion_auc.copy()
    random_d_auc=deletion_auc.copy();random_i_auc=deletion_auc.copy();estimable=np.zeros(len(days),bool)
    for row in np.flatnonzero(mask):
        history=histories[row];maps=groupmaps[row];original_values=np.asarray(history['values'],float)
        replacement=original_values.copy()
        for group in maps:
            idx=[i for i in group['indices'] if math.isfinite(original_values[i])]
            replacement[idx]=mean
        original=_history_predict(callback,history,origins[row])
        baseline_history={'dates':history['dates'],'values':replacement};base=_history_predict(callback,baseline_history,origins[row])
        for j,group in enumerate(maps):
            deleted=original_values.copy();idx=group['indices'];deleted[idx]=replacement[idx]
            attribution[row,j]=original-_history_predict(callback,{'dates':history['dates'],'values':deleted},origins[row])
        if lag in (1,7):
            selected=date.fromisoformat(str(origins[row])[:10])-timedelta(days=int(lag))
            actual=original_values[history['dates'].index(selected.isoformat())]
            if not np.isfinite(actual) or abs(actual-original)>1e-10:
                raise ValueError('Declared naive lag does not match the frozen callback forecast')
        delta=original-base
        if abs(delta)<=1e-7*max(1.,abs(original),abs(base)):continue
        estimable[row]=True
        def curves(rank):
            d=[];i=[]
            for step in range(g+1):
                chosen=[index for group in rank[:step] for index in maps[int(group)]['indices']]
                dv=original_values.copy();iv=replacement.copy();dv[chosen]=replacement[chosen];iv[chosen]=original_values[chosen]
                d.append((_history_predict(callback,{'dates':history['dates'],'values':dv},origins[row])-base)/delta)
                i.append((_history_predict(callback,{'dates':history['dates'],'values':iv},origins[row])-base)/delta)
            return np.trapezoid(d,np.arange(g+1)/g),np.trapezoid(i,np.arange(g+1)/g)
        rank=np.argsort(-np.abs(attribution[row]),kind='stable');deletion_auc[row],insertion_auc[row]=curves(rank)
        rng=_day_rng(seed,'history_fidelity',days[row]);random=[curves(rng.permutation(g)) for _ in range(random_rankings)]
        random_d_auc[row]=np.mean([v[0] for v in random]);random_i_auc[row]=np.mean([v[1] for v in random])
    fidelity={'method':'observed_history_group_deletion_insertion','groups':labels,'group_count':g,
        'random_rankings':random_rankings,'seed':seed,'normalization':'(prediction_step-prediction_baseline)/(prediction_original-prediction_baseline)',
        'private_deletion_AUC':deletion_auc,'private_insertion_AUC':insertion_auc,
        'private_random_deletion_AUC':random_d_auc,'private_random_insertion_AUC':random_i_auc,
        'private_deletion_advantage':random_d_auc-deletion_auc,'private_insertion_advantage':insertion_auc-random_i_auc,
        'private_estimable_rows':estimable,'not_estimable_near_zero_prediction_difference':int((mask&~estimable).sum())}
    magnitude=np.abs(attribution);total=magnitude.sum(1)
    shares=np.divide(magnitude,total[:,None],out=np.full_like(magnitude,np.nan),where=total[:,None]>0)
    importance=[{**label,**summarize_daily_values(shares[:,j],origins,eligible=mask,repetitions=repetitions,block_length=block_length,seed=seed)} for j,label in enumerate(labels)]
    return {'method':'analytic_selected_lag_and_frozen_history_occlusion' if lag in (1,7) else 'frozen_ARIMA_history_group_occlusion',
        'private_attributions':attribution,'importance':importance,'groups':labels,
        'fidelity':fidelity,'fidelity_summary':summarize_fidelity(fidelity,origins,eligible=mask,repetitions=repetitions,block_length=block_length,seed=seed),
        'seed_stability':{'status':'not_applicable_deterministic_estimator','independent_training_seed_count':0},
        'past_history_only':True,'coefficient_parameters_refitted':False,'causal_effect_claimed':False,
        'one_lag_fidelity_random_control_may_be_degenerate':g==1,'new_model_fits':0,'new_hypothesis_decisions':0}


def _standardize_raw(raw,preprocessing):
    from dl_data_v2 import relative_features
    x=(relative_features(raw)-np.asarray(preprocessing['x_mean'],float))/np.asarray(preprocessing['x_scale'],float)
    if not np.isfinite(x).all():raise ValueError('Invalid frozen standardized diagnostic features')
    return x.astype(np.float32)


def evaluate_final_diagnostics(entries,raw_windows,truth,origins,*,best3_development,reference,
        history_source,private_output,closure_receipt,public_output=None,eligible=None,
        seed_ids=(670101,670102,670103,670104,670105),repetitions=2000,block_length=7):
    """Execute all20 robustness and preselected-top3 explanation diagnostics.

    Entries expose frozen callbacks; the engine remains responsible for actual
    checkpoint closure, identity and causal state replay. No callback is allowed
    to fit parameters, choose models or change preprocessing in this phase.
    """
    from datetime import datetime,timezone
    from pathlib import Path
    import time
    root=Path(__file__).resolve().parents[1];private=Path(private_output).resolve()
    if private.is_relative_to(root) or (private.exists() and any(private.iterdir())):
        raise ValueError('Use a fresh private diagnostics directory outside the repository')
    if (len(entries)!=20 or len(set(best3_development))!=3 or not set(best3_development).issubset(entries)
            or closure_receipt.get('all_model_weights_closed') is not True
            or set(closure_receipt.get('catalogue',[]))!=set(entries)):
        raise ValueError('The complete closed20-model catalogue and three development-selected models are required')
    if tuple(seed_ids)!=(670101,670102,670103,670104,670105):raise ValueError('Use exactly the five diagnostic seeds fixed before P1')
    days=_dates(origins);mask=_mask(len(days),eligible);raw=_raw(raw_windows,mask);y=np.asarray(truth,float)
    if len(raw)!=len(days) or y.shape!=(len(days),):raise ValueError('Common raw outcome calendar required')
    out=None
    if public_output is not None:
        out=Path(public_output).resolve()
        if out.is_relative_to(root) and not out.is_relative_to(root/'book_e2/_static'):
            raise ValueError('Public diagnostic aggregates may only be written under the designated book static tree')
        if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh public aggregate directory')
    private.mkdir(parents=True)
    (private/'operator_reference.json').write_text(json.dumps(_json_ready({'raw_imputation':reference,'design':DESIGN}),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    start=time.perf_counter();robustness={};explanations={};private_cases={}
    for name,entry in entries.items():
        if entry['input_kind']=='endogenous':
            result=evaluate_endogenous_robustness(entry,history_source,y,origins,eligible=mask,repetitions=repetitions,block_length=block_length)
        elif entry['input_kind']=='OHLCV':
            declared=list(entry['seed_ids'])
            expected=5 if declared==list(seed_ids) else 1 if declared==['deterministic'] else None
            if expected is None:raise ValueError('Raw model does not expose the fixed5or1 diagnostic member registry')
            result=evaluate_robustness(entry['predict_raw'],raw,y,origins,
                reference=entry.get('training_reference',reference),eligible=mask,repetitions=repetitions,block_length=block_length)
            if result['seed_members']!=expected:raise ValueError('Frozen prediction member count differs from the registry')
        else:raise ValueError('Each catalogue entry needs an explicit OHLCV or endogenous information contract')
        robustness[name]=result
        (private/'progress.json').write_text(json.dumps({'robustness_completed':list(robustness),'explanations_completed':list(explanations)},indent=2)+'\n')
    for name in best3_development:
        entry=entries[name]
        if entry['input_kind']=='endogenous':
            value=explain_endogenous_history(entry,history_source,origins,eligible=mask,repetitions=repetitions,block_length=block_length)
            private_cases[name]=value;explanations[name]=public_diagnostic_summary(value)
            continue
        members=entry.get('members')
        if members is not None:
            if len(members)!=5 or [m['seed'] for m in members]!=list(seed_ids):raise ValueError('Neural explanations use all five prespecified members')
        else:
            if list(entry['seed_ids'])!=['deterministic']:raise ValueError('Missing neural members, not permission to fabricate an ensemble')
            members=[entry]
        maps={};records=[];case_records=[]
        for member in members:
            pre=member['preprocessing'];fingerprint=preprocessing_fingerprint(pre)
            if member.get('scaler_sha256',fingerprint)!=fingerprint:raise ValueError('Frozen scaler content fingerprint mismatch')
            baseline_receipt=member.get('explanation_baseline_receipt',pre.get('explanation_baseline_receipt'))
            if baseline_receipt is None:raise ValueError('No training-only35cell baseline receipt for this frozen predictor')
            cutoff=pre.get('last_fit_outcome_day_utc',entry.get('fit_outcome_max'))
            if cutoff is None or date.fromisoformat(str(cutoff)[:10])>days[0]-timedelta(days=8):raise ValueError('Explanation fit cutoff violates the evaluation embargo')
            baseline=validate_explanation_baseline(baseline_receipt,expected_scaler_sha256=fingerprint,maximum_training_outcome=cutoff)
            x=_standardize_raw(raw[mask],pre)
            if isinstance(member.get('model'),nn.Module):predict=torch_normalized_predictor(member['model'],y_scale=float(pre['y_scale']))
            else:predict=member['predict_normalized']
            occlusion=occlusion_attributions(predict,x,baseline=baseline)
            dense=np.full((len(days),len(FEATURE_GROUPS)),np.nan);dense[mask]=occlusion['private_attributions']
            fidelity=deletion_insertion_fidelity(predict,x,occlusion['private_attributions'],baseline=baseline)
            # Restore ineligible positions before every temporal confidence interval.
            for key,value in list(fidelity.items()):
                if key.startswith('private_') and isinstance(value,np.ndarray):
                    fill=np.zeros(len(days),bool) if value.dtype.kind=='b' else np.full(len(days),np.nan)
                    fill[mask]=value;fidelity[key]=fill
            member_id=member.get('seed','deterministic');maps[member_id]=dense
            record={'member':member_id,'method':occlusion['method'],'group_count':14,
                'importance':attribution_summary(dense,origins,eligible=mask,repetitions=repetitions,block_length=block_length),
                'fidelity':summarize_fidelity(fidelity,origins,eligible=mask,repetitions=repetitions,block_length=block_length),
                'baseline_receipt_sha256':sha256(json.dumps(_json_ready(baseline_receipt),sort_keys=True).encode()).hexdigest()}
            private_record={'member':member_id,'occlusion':occlusion,'dense_group_attributions':dense,'fidelity':fidelity}
            if isinstance(member.get('model'),nn.Module):
                ig=integrated_gradients(member['model'],x,baseline=baseline,steps=64,y_scale=float(pre['y_scale']))
                record['IG_secondary']={k:v for k,v in ig.items() if not k.startswith('private_')}
                private_record['IG_secondary']=ig
            if 'linear_coefficients' in member:
                analytic=explain_clipped_linear(member['linear_coefficients'],member['linear_intercept'],x,baseline=baseline)
                record['analytic_linear_completeness_max_residual']=analytic['completeness_max_absolute_residual']
                private_record['analytic_linear']=analytic
            records.append(record);case_records.append(private_record)
        if tuple(entry['seed_ids'])==tuple(seed_ids):
            stability=explanation_stability(maps,seed_ids,origins,eligible=mask,repetitions=repetitions,block_length=block_length)
            stability_public=public_diagnostic_summary(stability)
        else:stability_public={'status':'not_applicable_deterministic_estimator','independent_training_seed_count':0}
        explanations[name]={'primary_method':'TRAIN-only-baseline14group_occlusion','members':records,
            'seed_stability':stability_public,'information_contract':'OHLCV_D7','causal_effect_claimed':False}
        private_cases[name]=case_records
        (private/'progress.json').write_text(json.dumps({'robustness_completed':list(robustness),'explanations_completed':list(explanations)},indent=2)+'\n')
    if len(robustness)!=20 or set(explanations)!=set(best3_development):raise ValueError('Incomplete final diagnostics cannot be marked executed')
    result={'scope':'frozen final-model diagnostics; first5 neural training seeds and one deterministic trajectory',
        'diagnostic_seed_ids':list(seed_ids),'main_R_not_replaced_by_this_subset':True,
        'models_evaluated':20,'robustness':robustness,'explanation_models_selected_in_development':list(best3_development),
        'explanations':explanations,'calendar_origins':len(days),'eligible_unique_origins':int(mask.sum()),
        'real_model_fits_performed':0,'new_hypothesis_decisions':0,
        'all_scientific_validation_claimed':False,'elapsed_diagnostics_seconds':time.perf_counter()-start,
        'uncertainty':'nominal temporal block intervals conditional on fixed diagnostic members; not new independent markets',
        'information_contracts_compared_as_if_identical':False}
    private_data=_json_ready(private_cases);payload=json.dumps(private_data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    (private/'diagnostics_private.json').write_text(payload)
    public=public_diagnostic_summary(result)
    (private/'summary.json').write_text(json.dumps(_json_ready(public),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    provenance={'created_at_utc':datetime.now(timezone.utc).isoformat(),'generator':'scripts/e2_final_diagnostics.py',
        'generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'truth_sha256':_digest(y),
        'raw_windows_sha256':_digest(raw),'closure_receipt_sha256':sha256(json.dumps(_json_ready(closure_receipt),sort_keys=True).encode()).hexdigest(),
        'private_case_artifact_sha256':sha256(payload.encode()).hexdigest(),'individual_cases_exported_publicly':False,
        'model_fits':0,'diagnostic_seed_ids':list(seed_ids),'all20robustness_and3explanations_executed':True}
    (private/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/'summary.json').write_bytes((private/'summary.json').read_bytes())
        (out/'provenance.json').write_bytes((private/'provenance.json').read_bytes())
    return public
