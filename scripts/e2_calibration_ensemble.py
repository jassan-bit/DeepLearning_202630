"""One registered calibration-only convex ensemble; never refits the 16 models.

The statistical fit of the 16 combination weights is explicit. All calibration
observations predate the fixed test. The entire exercise is post hoc and cannot
restore an unseen test, establish Bayes risk, or overwrite the original A2.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import optimize
from e2_reduced_protocol import SYSTEMS, NEURAL
from e2_reduced_final_analysis import holm

VERSION = "e2-calibration-fitted-diverse-ensemble-v1"
INPUT_SHA = "65ae74b547a6073b2c286dea8039a2e7a1b1f6063b1d9ac0eb760c9ce727d187"
DEV_SHA = "20b96374a659eb87dadfa68aa3d6b6b5fb6bea5b10eebe6f2c186ad69861844a"
SEEDS = tuple(range(271801, 271807))
SAE = "LSTM-autoencoder con predictor"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")

def csv_save(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

def panel(payload):
    by={}
    for row in payload["predictors"]:
        key=(row["model"], row["seed"])
        if key in by: raise ValueError("Duplicate frozen predictor")
        by[key]=row
    cp, tp, calibration_targets, calibration_dates=[], [], None, None
    for model in SYSTEMS:
        seeds=SEEDS if model in NEURAL else ("deterministic",)*6
        calibration, test=[], []
        for seed in seeds:
            row=by[(model,seed)]
            if calibration_targets is None:
                calibration_targets=np.asarray(row["calibration_targets"],float)
                calibration_dates=list(row["calibration_dates"])
            if not np.array_equal(calibration_targets,row["calibration_targets"]) or calibration_dates!=row["calibration_dates"]:
                raise ValueError("Calibration targets/dates must align for every predictor")
            calibration.append(row["calibration_predictions"]);test.append(row["prediction_original"])
        cp.append(calibration);tp.append(test)
    cp,tp=np.asarray(cp,float),np.asarray(tp,float)
    y=np.asarray(payload["y_true"],float)
    if cp.shape!=(16,6,53) or tp.shape!=(16,6,608) or y.shape!=(608,) or len(by)!=81:
        raise ValueError("Original complete panel required")
    if not all(np.isfinite(a).all() for a in (cp,tp,y,calibration_targets)):
        raise ValueError("Nonfinite frozen panel")
    if not all(payload["mask"]) or len(payload["dates"])!=608:
        raise ValueError("Complete common calendar required")
    if max(calibration_dates)>=min(payload["dates"]):
        raise ValueError("Calibration must precede all evaluated forecasts")
    return cp, calibration_targets, tp, y, calibration_dates

def fit_weights(calibration, targets, scale, *, ridge=1e-4):
    """Convex simplex fit: only the explicitly supplied calibration panel."""
    calibration=np.asarray(calibration,float);targets=np.asarray(targets,float)
    if calibration.ndim!=3 or targets.shape!=(calibration.shape[2],) or not np.isfinite(calibration).all() or not np.isfinite(targets).all():
        raise ValueError("Finite model x seed x calibration-date panel required")
    if not np.isfinite(scale) or scale<=0 or ridge<=0:
        raise ValueError("Positive fixed development scale and ridge required")
    m,r,n=calibration.shape
    x=np.moveaxis(calibration,0,-1).reshape(r*n,m)/scale
    y=np.tile(targets,r)/scale;uniform=np.full(m,1/m)
    def objective(w):
        e=x@w-y
        return float(e@e/len(e)+ridge*np.sum((w-uniform)**2))
    def gradient(w):
        return 2*x.T@(x@w-y)/len(y)+2*ridge*(w-uniform)
    result=optimize.minimize(objective,uniform,jac=gradient,method="SLSQP",
        bounds=[(0.,1.)]*m,
        constraints=[{"type":"eq","fun":lambda w:float(w.sum()-1),"jac":lambda w:np.ones(m)}],
        options={"ftol":1e-12,"maxiter":1000})
    if not result.success or abs(result.x.sum()-1)>1e-8 or result.x.min()<-1e-8:
        raise ValueError("The single registered optimization did not converge: "+result.message)
    w=np.maximum(result.x,0);w=w/w.sum()
    info={"solver":"SLSQP","success":True,"message":str(result.message),"iterations":int(result.nit),
        "calibration_scaled_objective":objective(w),"uniform_scaled_objective":objective(uniform),
        "constraint_sum":float(w.sum()),"minimum_weight":float(w.min()),
        "numerical_feasibility_adjustment_L1":float(np.abs(w-result.x).sum()),
        "ridge":ridge,"development_target_mean_scale":float(scale),"optimizations":1,
        "test_targets_used_for_weights":False,"new_statistical_weight_fit":True}
    return w,info

def combine(weights, forecasts):
    return np.einsum("m,mst->st",weights,np.asarray(forecasts,float))

def risk(predictions, targets):
    return float(np.sqrt(np.mean((predictions-targets[None,:])**2,axis=1)).mean())

def gamma_values(best, baseline, reference):
    denominator=baseline-reference
    raw=None if denominator==0 else float((best-reference)/denominator)
    valid=bool(reference<best<baseline)
    return {"raw_algebraic_gamma":raw,"interpretable_empirical_reference_order":valid,
        "gamma_remaining_fraction":raw if valid else None,
        "empirical_reference_is_Bayes_or_optimal_ceiling":False}

def resample_risks(forecasts, targets, *, draws=10000, block=14, seed=272201):
    """Joint calendar and paired seed bootstrap of three fixed risk estimands."""
    if type(draws)is not int or draws<10000 or type(block)is not int or block<1:
        raise ValueError("At least10000 draws and a positive calendar block required")
    p=np.asarray(forecasts,float);y=np.asarray(targets,float)
    if p.ndim!=3 or p.shape[0]!=3 or p.shape[2]!=len(y):raise ValueError("Three aligned systems required")
    rng=np.random.default_rng(seed);out=np.empty((draws,3));squared=(p-y[None,None,:])**2
    n=len(y);r=p.shape[1]
    for i in range(draws):
        tids=((rng.integers(0,n,size=int(np.ceil(n/block)))[:,None]+np.arange(block))%n).reshape(-1)[:n]
        sids=rng.integers(0,r,size=r)
        out[i]=np.sqrt(squared[:,sids,:][:,:,tids].mean(axis=2)).mean(axis=1)
    return out

def analyze(input_path, development, plan_path, output, registration):
    started=time.perf_counter();output=Path(output)
    if output.exists():raise FileExistsError("Use a new documentary output directory")
    if sha(input_path)!=INPUT_SHA or sha(development)!=DEV_SHA:raise ValueError("Frozen input hash changed")
    plan=json.loads(Path(plan_path).read_text())
    if plan["version"]!=VERSION or plan["weight_fit"]["ridge"]!=1e-4 or plan["evaluation"]["primary_bootstrap"]["draws"]!=10000:
        raise ValueError("Registered settings differ")
    payload=json.loads(Path(input_path).read_text());cp,cy,tp,y,caldates=panel(payload)
    with Path(development).open() as f:
        values=np.asarray([float(row["sigma_7"]) if row["sigma_7"].strip() else np.nan for row in csv.DictReader(f)])
    scale=float(values[np.isfinite(values)].mean())
    weights,fit=fit_weights(cp,cy,scale,ridge=plan["weight_fit"]["ridge"])
    # No test target was passed to the optimization. Evaluation begins here.
    ep=combine(weights,tp);calep=combine(weights,cp)
    sae=tp[list(SYSTEMS).index(SAE)];base=tp[list(SYSTEMS).index("Persistence")]
    forecasts=np.stack([ep,sae,base]);observed=np.asarray([risk(p,y) for p in forecasts])
    intervals=[];contrasts=[];bootstrap_summaries=[]
    for block in (14,7,28):
        boot=resample_risks(forecasts,y,block=block,seed=272201 if block==14 else 272201+block)
        if block==14:
            for j,label in enumerate((SAE,"Persistence"),1):
                gain=float(observed[j]-observed[0]);draw_gains=boot[:,j]-boot[:,0]
                p=float((1+np.count_nonzero(draw_gains-gain>=gain))/(len(boot)+1))
                contrasts.append({"model":"calibration_simplex_ensemble","baseline":label,
                    "gain_RMSE":gain,"CI95_low":float(np.quantile(draw_gains,.025)),
                    "CI95_high":float(np.quantile(draw_gains,.975)),"p_one_sided":p,
                    "null":"gain<=0","draws":len(boot),"block_days":block})
        denom=boot[:,2]-boot[:,0];defined=np.abs(denom)>1e-12
        rawgamma=(boot[defined,1]-boot[defined,0])/denom[defined]
        bootstrap_summaries.append({"block_days":block,"draws":len(boot),
            "risk_CI95":{"ensemble":np.quantile(boot[:,0],[.025,.975]).tolist(),
                "SAE":np.quantile(boot[:,1],[.025,.975]).tolist(),"Persistence":np.quantile(boot[:,2],[.025,.975]).tolist()},
            "algebraic_gamma_defined_replicates":int(defined.sum()),
            "algebraic_gamma_CI95":np.quantile(rawgamma,[.025,.975]).tolist() if len(rawgamma) else None,
            "reference_order_valid_replicates":int(((boot[:,0]<boot[:,1])&(boot[:,1]<boot[:,2])).sum()),
            "gamma_interval_scope":"algebraic_only_if_reference_order_or_denominator_change; never silently restrict to favorable replicates"})
    for i,day in enumerate(payload["dates"]):
        for s,seed in enumerate(SEEDS):intervals.append({"date":day,"seed":seed,"ensemble_prediction":float(ep[s,i]),"target":float(y[i])})
    summary={"version":VERSION,"registered_commit":registration,"posthoc":True,
        "original_A2_changed":False,"original_calibrators_changed":False,"new_neural_fits":0,
        "new_model_inference":0,"new_statistical_weight_fits":1,
        "weights_fit_only_on_calibration53":True,"test_used_to_choose_method_or_weights":False,
        "calibration_dates":caldates,"calibration_ensemble_mean_RMSE":risk(calep,cy),
        "test_mean_individual_seed_RMSE":{"ensemble":float(observed[0]),"SAE":float(observed[1]),"Persistence":float(observed[2])},
        "gamma":gamma_values(observed[1],observed[2],observed[0]),
        "same_known_test_as_E1":True,"empirical_reference_scope":"one explicitly fitted and separately registered attainable ensemble, not Bayes risk or an optimum over all ensembles",
        "fit":fit,"bootstrap":bootstrap_summaries,
        "CPU_analysis_wall_seconds":time.perf_counter()-started}
    output.mkdir(parents=True)
    csv_save(output/"weights.csv",[{"model":m,"weight":float(w)}for m,w in zip(SYSTEMS,weights)])
    csv_save(output/"ensemble_saved_forecasts.csv",intervals)
    csv_save(output/"contrasts.csv",holm(contrasts));save(output/"summary.json",summary)
    (output/"analysis_plan.json").write_bytes(Path(plan_path).read_bytes())
    save(output/"manifest.json",{"version":VERSION,"input_sha256":INPUT_SHA,"development_sha256":DEV_SHA,
        "script_sha256":sha(__file__),"plan_sha256":sha(plan_path),"registered_commit":registration,
        "observed_at_utc":datetime.now(timezone.utc).isoformat(),
        "files":{p.name:sha(p)for p in output.iterdir()if p.is_file()}})
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("input","development","plan","output","registration"):p.add_argument("--"+name,required=True)
    args=p.parse_args();print(json.dumps(analyze(args.input,args.development,args.plan,args.output,args.registration),ensure_ascii=False,indent=2))
