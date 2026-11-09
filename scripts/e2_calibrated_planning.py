"""Independent-null calibration for conditional empirical-pilot planning.

No model training, Torch import, or market-file loader is supplied. The fitted
empirical DGP retains the pilot's paired calendar blocks and seed residuals;
its Monte Carlo guarantees concern that conditional DGP, not future markets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math

import numpy as np
from scipy.stats import beta, binom, chi2

VERSION = "e2-calibrated-conditional-planning-v1"


def _probability(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(name+" must be a finite probability")
    return float(value)


def _integer(value, name, lower=1):
    if type(value) is not int or value < lower:
        raise ValueError(name+" must be an integer >= "+str(lower))
    return value


@dataclass(frozen=True)
class CalibrationConfig:
    calibration_draws: int = 200000
    validation_draws: int = 200000
    alternative_draws: int = 200000
    alpha: float = .05
    power_target: float = .8
    R_values: tuple[int, ...] = (6, 10, 20, 30)
    family_mc_error: float = .05
    calibration_pool: int = 272401
    validation_pool: int = 272402
    alternative_pool: int = 272403
    block_length: int = 7
    batch_size: int = 2048

    def __post_init__(self):
        for key in ("calibration_draws", "validation_draws", "alternative_draws"):
            _integer(getattr(self,key),key)
        for key in ("alpha", "power_target", "family_mc_error"):
            _probability(getattr(self,key),key)
        if (not isinstance(self.R_values,(tuple,list)) or not self.R_values
                or len(set(self.R_values)) != len(self.R_values)
                or any(type(R) is not int or not 6 <= R <= 30 for R in self.R_values)):
            raise ValueError("Unique registered R scenarios within6..30 required")
        object.__setattr__(self,"R_values",tuple(sorted(self.R_values)))
        pools=(self.calibration_pool,self.validation_pool,self.alternative_pool)
        for pool in pools:_integer(pool,"RNG pool",0)
        if len(set(pools)) != 3:
            raise ValueError("Calibration, validation and alternative RNG pools must be disjoint")
        if self.block_length != 7:
            raise ValueError("The preregistered DGP uses circular seven-day blocks")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 8192:
            raise ValueError("Use streaming batches of1..8192 draws")


def clopper_pearson_upper(hits, count, gamma):
    _integer(count,"count")
    if type(hits) is not int or not 0 <= hits <= count:
        raise ValueError("Hit count must lie in0..count")
    _probability(gamma,"one-sided MC error")
    if hits == count:return 1.0
    if hits == 0:return float(-math.expm1(math.log(gamma)/count))
    return float(beta.ppf(1-gamma,hits+1,count-hits))


def clopper_pearson_lower(hits, count, gamma):
    _integer(count,"count")
    if type(hits) is not int or not 0 <= hits <= count:
        raise ValueError("Hit count must lie in0..count")
    _probability(gamma,"one-sided MC error")
    if hits == 0:return 0.0
    return float(beta.ppf(gamma,hits,count-hits+1))


def calibrate_abs_threshold(null_noise, tail_target, gamma):
    """One upper |noise| order statistic with a binomial tolerance bound.

    k is1-based from the largest sample. It is the largest integer for which
    P[Bin(N,tail_target) <= k-1] <= gamma. Reject strictly above the threshold;
    ties/atoms therefore cannot enlarge the empirical rejection set.
    """
    values=np.asarray(null_noise,dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A complete finite one-dimensional null sample is required")
    _probability(tail_target,"absolute-tail target")
    _probability(gamma,"calibration tolerance error")
    count=len(values)
    k=int(binom.ppf(gamma,count,tail_target))
    while k >= 1 and float(binom.cdf(k-1,count,tail_target)) > gamma:k-=1
    while k < count and float(binom.cdf(k,count,tail_target)) <= gamma:k+=1
    if k < 1:
        return {"status":"insufficient_calibration_tail_draws","critical_abs":None,
                "draws":count,"tail_target":tail_target,"gamma":gamma,
                "reason":"Even the maximum sample cannot certify the requested tail tolerance"}
    index=count-k
    critical=float(np.partition(np.abs(values),index)[index])
    return {"status":"calibrated_on_independent_null","critical_abs":critical,
            "draws":count,"tail_target":tail_target,"gamma":gamma,
            "order_rank_1based":index+1,"numpy_order_index":index,
            "tail_order_k":k,"allowed_exceedances_in_calibration":k-1,
            "binomial_failure_probability_bound":float(binom.cdf(k-1,count,tail_target)),
            "rejection_rule":"abs(noise) > critical_abs; strict inequality"}


def validate_null(null_noise, critical_abs, nominal_alpha, gamma):
    """Validate once on a new stream; never alter the critical value here."""
    values=np.asarray(null_noise,dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Complete independent validation-null sample required")
    if type(critical_abs) not in (float,int) or not math.isfinite(critical_abs) or critical_abs < 0:
        raise ValueError("Frozen finite nonnegative critical value required")
    _probability(nominal_alpha,"nominal comparison alpha")
    hits=int(np.count_nonzero(np.abs(values)>critical_abs))
    upper=clopper_pearson_upper(hits,len(values),gamma)
    return {"draws":len(values),"hits":hits,"rejection_frequency":hits/len(values),
            "CP_upper":upper,"gamma":gamma,"nominal_alpha":nominal_alpha,
            "validated":bool(upper <= nominal_alpha),
            "critical_value_retuned":False}


def _calendar(dates, count):
    if dates is None or len(dates) != count:
        raise ValueError("Explicit consecutive daily development dates are required")
    parsed=[]
    for value in dates:
        text=value.isoformat() if hasattr(value,"isoformat") else str(value)
        stamp=datetime.fromisoformat(text.replace("Z","+00:00"))
        stamp=stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
        if stamp.hour or stamp.minute or stamp.second or stamp.microsecond:
            raise ValueError("Dates must be UTC midnights")
        if stamp >= datetime(2025,1,1,tzinfo=timezone.utc):
            raise ValueError("Final-period dates are outside this development analysis")
        parsed.append(stamp)
    if any(right-left != timedelta(days=1) for left,right in zip(parsed,parsed[1:])):
        raise ValueError("Calendar cannot compress gaps or duplicate dates")
    return [value.date().isoformat() for value in parsed]


def _array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype="<f8").tobytes()).hexdigest()


def _noise_chunks(pair, R, count, pool, pair_index, batch_size):
    """Identical conditional DGP for all three independent role streams."""
    # Separate calendar/seed generators also make chunk size irrelevant to
    # which variates belong to each role. No global NumPy RNG is mutated.
    calendar_rng=np.random.default_rng(np.random.SeedSequence([pool,R,pair_index,0]))
    seed_rng=np.random.default_rng(np.random.SeedSequence([pool,R,pair_index,1]))
    squared=pair["squared_errors"]
    seeds,days=squared.shape[1:]
    blocks=math.ceil(days/7)
    for start in range(0,count,batch_size):
        size=min(batch_size,count-start)
        beginnings=calendar_rng.integers(0,days,size=(size,blocks))
        indices=((beginnings[:,:,None]+np.arange(7)[None,None,:])%days).reshape(size,-1)[:,:days]
        scores=np.sqrt(squared[:,:,indices].mean(axis=-1))
        difference=(scores[0]-scores[1]).T
        mean=difference.mean(axis=1)
        residual=difference-mean[:,None]
        chosen=seed_rng.integers(0,seeds,size=(size,R))
        seed_mean=np.take_along_axis(residual,chosen,axis=1).mean(axis=1)*pair["seed_residual_inflation"]
        # Fixed original-pilot centering is shared by calibration/validation/
        # alternatives. Never center a validation sample on its own mean.
        noise=mean-pair["original_mean_difference"]+seed_mean
        if not np.isfinite(noise).all():raise ValueError("Conditional DGP produced nonfinite noise")
        yield noise


def _sample_noise(pair,R,count,pool,pair_index,batch_size):
    result=np.empty(count,dtype=np.float64)
    position=0
    for chunk in _noise_chunks(pair,R,count,pool,pair_index,batch_size):
        result[position:position+len(chunk)]=chunk
        position+=len(chunk)
    return result


def _effect_result(noise, delta, critical, gamma, validated):
    result={}
    for sign,label in ((1,"plus_delta"),(-1,"minus_delta")):
        hits=int(np.count_nonzero(np.abs(noise+sign*delta)>critical))
        lower=clopper_pearson_lower(hits,len(noise),gamma)
        result[label]={"effect_shift":sign*delta,"draws":len(noise),"hits":hits,
                       "conditional_rejection_frequency":hits/len(noise),
                       "CP_lower":lower,"gamma":gamma,
                       "interpretation_admitted_by_null_validation":bool(validated)}
    return result


def calibrated_power_analysis(predictions, y, delta, m=153, *, dates, model_names=None,
                              seed_ids=None, config=None):
    """Calibrate and independently validate conditional-DGP planning cells.

    The only power target certified by Monte Carlo is the two prespecified
    shifts +delta/-delta. MDE is a descriptive one-sided empirical-quantile
    diagnostic, without a post-selection CP guarantee.
    """
    config=CalibrationConfig() if config is None else config
    if not isinstance(config,CalibrationConfig):raise TypeError("Explicit CalibrationConfig required")
    if type(delta) not in (float,int) or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Positive finite inherited delta required")
    _integer(m,"multiplicity family size")
    prediction=np.asarray(predictions,dtype=np.float64)
    truth=np.asarray(y,dtype=np.float64)
    if (prediction.ndim != 3 or prediction.shape[0] not in (2,3)
            or prediction.shape[1:] != (5,53) or truth.shape != (53,)):
        raise ValueError("This registered empirical DGP requires2or3 models x5 seeds x53 dates")
    if (not np.isfinite(prediction).all() or not np.isfinite(truth).all()
            or (prediction < 0).any() or (truth < 0).any() or truth.mean() <= 0):
        raise ValueError("Complete nonnegative finite original-unit predictions/targets required")
    calendar=_calendar(dates,53)
    names=list(model_names) if model_names is not None else ["model_"+str(i) for i in range(len(prediction))]
    if (len(names) != len(prediction) or any(not isinstance(name,str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Unique model labels required")
    if seed_ids is not None and (len(seed_ids) != 5 or len(set(seed_ids)) != 5
                                 or any(type(value) is not int for value in seed_ids)):
        raise ValueError("Five distinct seed IDs required")
    pairs=list(itertools_combinations(len(prediction)))
    cells=len(config.R_values)*len(pairs)
    gamma_cal=config.family_mc_error/cells
    gamma_val=config.family_mc_error/cells
    gamma_power=config.family_mc_error/(2*cells)
    alpha_pair=config.alpha/m
    tail_target=alpha_pair/2  # TOTAL absolute tail, not a separate target per sign.
    # Conservative accounting for major NumPy arrays/temporary batches, not an
    # assertion about the host process's pre-existing imported-runtime RSS.
    batch_bytes=config.batch_size*(2*5*53*8*2+53*8*2+2*30*8+5*8*4)
    estimated_arrays=3*max(config.calibration_draws,config.validation_draws,
                           config.alternative_draws)*8+batch_bytes
    if estimated_arrays >= 1024**3:
        raise ValueError("Requested array budget exceeds the1GiB streaming design")
    errors=prediction-truth[None,None,:]
    with np.errstate(over="ignore",invalid="ignore"):
        squared=errors*errors
        scores=np.sqrt(squared.mean(axis=-1))
    if not np.isfinite(scores).all():raise ValueError("Pilot squared errors overflow")
    rows=[]
    factor=math.sqrt(4/float(chi2.ppf(.05,4)))
    for pair_index,(first,second) in enumerate(pairs):
        differences=scores[first]-scores[second]
        with np.errstate(over="ignore",invalid="ignore"):
            sd=float(differences.std(ddof=1))
        upper_sd=sd*factor
        unresolved=float(np.ptp(differences)) <= 16*np.finfo(float).eps*max(float(np.abs(differences).max()),delta,1e-12)
        pair={"models":[names[first],names[second]],"difference_definition":"RMSE(first)-RMSE(second)",
              "paired_seed_differences":differences.tolist(),"sigma_d":sd if math.isfinite(sd) else None,
              "original_mean_difference":float(differences.mean()),
              "sigma_d_upper95_normal_assumption":upper_sd if math.isfinite(upper_sd) else None,
              "seed_residual_inflation":factor*math.sqrt(5/4),
              "status":"not_estimable" if unresolved or not math.isfinite(upper_sd) or sd<=0 else "empirical_DGP_defined",
              "cells":[]}
        if pair["status"] == "not_estimable":
            pair["reason"]="Seed variability is zero/nonfinite/unresolvable; do not infer perfect power"
            rows.append(pair)
            continue
        simulation={**pair,"squared_errors":squared[[first,second]].copy()}
        for R in config.R_values:
            calibration=_sample_noise(simulation,R,config.calibration_draws,config.calibration_pool,
                                      pair_index,config.batch_size)
            calibration_hash=_array_sha(calibration)
            tolerance=calibrate_abs_threshold(calibration,tail_target,gamma_cal)
            calibration_spread=float(np.ptp(calibration))
            calibration_scale=max(float(np.max(np.abs(calibration))),delta,1e-12)
            dgp_unresolved=calibration_spread <= 16*np.finfo(float).eps*calibration_scale
            cell={"R":R,"calibration":tolerance,"calibration_noise_sha256":calibration_hash,
                  "calibration_noise_spread":calibration_spread,
                  "null_validated":False,"supports_conditional_target":False,
                  "streams":{role:{"pool":pool,"calendar_entropy":[pool,R,pair_index,0],
                                    "seed_entropy":[pool,R,pair_index,1]}
                             for role,pool in (("calibration",config.calibration_pool),
                                               ("validation",config.validation_pool),
                                               ("alternative",config.alternative_pool))}}
            del calibration
            if dgp_unresolved:
                cell.update(status="conditional_DGP_zero_or_unresolvable_variance",
                            reason="Do not infer perfect power from a degenerate conditional DGP",
                            validation=None,alternatives=None,descriptive_MDE=None,
                            draws_executed={"calibration":config.calibration_draws,
                                            "validation":0,"alternative":0})
                pair["cells"].append(cell)
                continue
            if tolerance["critical_abs"] is None:
                cell.update(status="calibration_tail_not_estimable",validation=None,alternatives=None,
                            descriptive_MDE=None,draws_executed={"calibration":config.calibration_draws,
                                                               "validation":0,"alternative":0})
                pair["cells"].append(cell)
                continue
            critical=tolerance["critical_abs"]
            validation=_sample_noise(simulation,R,config.validation_draws,config.validation_pool,
                                     pair_index,config.batch_size)
            valid=validate_null(validation,critical,alpha_pair,gamma_val)
            cell["validation_noise_sha256"]=_array_sha(validation)
            del validation
            alternative=_sample_noise(simulation,R,config.alternative_draws,config.alternative_pool,
                                      pair_index,config.batch_size)
            effects=_effect_result(alternative,delta,critical,gamma_power,valid["validated"])
            q20,q80=np.quantile(alternative,(.2,.8),method="inverted_cdf")
            mde={"plus_effect":max(0.,float(critical-q20)),
                 "minus_effect_magnitude":max(0.,float(critical+q80)),
                 "interpretation":"Descriptive empirical80% one-sided subset effect; no CP guarantee after selecting this effect",
                 "null_validation_required_for_interpretation":True,
                 "scientific_or_distribution_free_MDE":False}
            lower=min(value["CP_lower"] for value in effects.values())
            cell.update(status="null_validated" if valid["validated"] else "independent_null_validation_failed",
                        validation=valid,alternatives=effects,null_validated=valid["validated"],
                        minimum_direction_CP_lower=lower,
                        supports_conditional_target=bool(valid["validated"] and lower>=config.power_target),
                        alternative_noise_sha256=_array_sha(alternative),descriptive_MDE=mde,
                        draws_executed={"calibration":config.calibration_draws,
                                        "validation":config.validation_draws,"alternative":config.alternative_draws})
            del alternative
            pair["cells"].append(cell)
        rows.append(pair)
    supported=[]
    for R in config.R_values:
        if all(pair["status"]=="empirical_DGP_defined" and next(
            cell for cell in pair["cells"] if cell["R"]==R)["supports_conditional_target"] for pair in rows):
            supported.append(R)
    recommendation=min(supported) if supported else None
    registered=(config.calibration_draws==config.validation_draws==config.alternative_draws==200000
                and config.R_values==(6,10,20,30) and config.alpha==.05
                and config.power_target==.8 and config.family_mc_error==.05
                and (config.calibration_pool,config.validation_pool,config.alternative_pool)
                    ==(272401,272402,272403) and m==153)
    return {"version":VERSION,"status":"conditional_empirical_DGP_planning_only",
            "registered_design_match":registered,
            "config":asdict(config),"models":names,"seed_ids":seed_ids,"calendar":calendar,
            "prediction_sha256":_array_sha(prediction),"truth_sha256":_array_sha(truth),
            "delta":delta,"multiplicity_family_size":m,"nominal_alpha_per_comparison":alpha_pair,
            "calibration_total_absolute_tail_target":tail_target,
            "gamma_calibration_per_cell":gamma_cal,"gamma_validation_per_cell":gamma_val,
            "gamma_power_per_cell_direction":gamma_power,"observed_pair_count":len(pairs),
            "cells_per_MC_family":cells,"estimated_numpy_working_set_bytes":estimated_arrays,
            "calibration_MC_family_confidence":1-config.family_mc_error,
            "validation_MC_family_confidence":1-config.family_mc_error,
            "power_MC_family_confidence":1-config.family_mc_error,
            "joint_three_family_confidence_claimed":False,
            "pairs":rows,"conditionally_supported_R":recommendation,
            "resource_envelope_R":recommendation if recommendation is not None else 30,
            "resource_envelope_basis":"conditional_DGP_MC_target_supported" if recommendation is not None else "R30_cap_without_conditional80percent_support",
            "final_R":None,"achieved_final_power":None,"models_fit_in_this_analysis":0,
            "limits":[
                "Monte Carlo calibration/validation concern only this empirical conditional DGP, not future markets.",
                "The observed5-seed,53-date reference recipes are not final selected models or new independent markets.",
                "Normal chi-square SD inflation is a sensitivity assumption, not a simultaneous real-population guarantee.",
                "Calendar-block variation remains a common floor and is not divided by sqrt(R).",
                "A single absolute critical value targets total tail alpha_pair/2, deliberately below nominal alpha_pair.",
                "Critical values are frozen from calibration and never retuned on independent validation failure.",
                "Power CP bounds apply only to the two fixed shifts plus/minus delta, not a searched MDE.",
                "MDE values are descriptive empirical-quantile diagnostics without selected-effect confidence guarantees.",
                "Each Monte Carlo family has its own1-family_mc_error confidence accounting; no joint confidence claim is made.",
                "A shift delta is a location-effect planning alternative, not proof of a real improvement beyond practical delta.",
                "Unobserved model-pair variances, full-study power and the original final-R gate remain unresolved."
            ]}


def itertools_combinations(count):
    # Keep canonical pair indices independent of data values or pilot winners.
    return [(first,second) for first in range(count) for second in range(first+1,count)]
