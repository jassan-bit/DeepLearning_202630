"""Reproduce the six saved negative-control analyses without fitting models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        path = PACKAGE / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("Manifest check failed: " + str(relative))
    return manifest


def calendar_indices(dates, draws, block, seed):
    import numpy as np
    days = dates.astype("datetime64[D]").astype(int)
    if len(days) < 1 or draws < 1 or block < 1 or (np.diff(days) <= 0).any():
        raise ValueError("Calendar resampling requires ordered distinct dates and positive sizes")
    full = np.full(int(days[-1]-days[0]+1), -1, dtype=int)
    full[days-days[0]] = np.arange(len(days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(full), size=(draws, math.ceil(len(full)/block)))
    sampled = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(full)
    return full[sampled.reshape(draws, -1)[:, :len(full)]]


def centered_bootstrap_p(draws, estimate, null_boundary):
    import numpy as np
    draws = np.asarray(draws, dtype=float)
    if not len(draws) or not np.isfinite(draws).all() or not math.isfinite(estimate):
        raise ValueError("A centered-bootstrap test requires complete finite estimates")
    return (1+int(np.sum(null_boundary+draws-estimate >= estimate)))/(len(draws)+1)


def holm(values):
    if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Holm requires a nonempty collection of valid p-values")
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, running = [1.]*len(values), 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index])
        result[index] = min(1., running)
    return result


def reanalyze():
    import numpy as np
    from scipy.stats import rankdata
    plan = json.loads((PACKAGE / "analysis_plan.json").read_text(encoding="utf-8"))
    config, delta = plan["bootstrap"], plan["equivalence"]["margin"]
    scalar = json.loads((PACKAGE / "training_reference.json").read_text(encoding="utf-8"))
    target_scale = scalar["target_scale"]
    reference = json.loads((PACKAGE / "reference_summary.json").read_text(encoding="utf-8"))
    payloads, truth, dates = {}, None, None
    for model in plan["models"]:
        for seed in plan["label_permutation_seeds"]:
            key = model.replace("-", "_") + "_" + str(seed)
            with np.load(PACKAGE / "predictions" / (key + ".npz"), allow_pickle=False) as saved:
                payload = {k: saved[k].copy() for k in saved.files}
            candidate_dates = payload["dates"].astype("datetime64[D]")
            if truth is None:
                truth, dates = payload["truth"], candidate_dates
            if not np.array_equal(payload["truth"], truth) or not np.array_equal(candidate_dates, dates):
                raise ValueError("Saved forecasts do not share the same truth/calendar")
            if payload["prediction"].shape != (608,) or not np.isfinite(payload["prediction"]).all() or payload["permutation"].shape != (655,):
                raise ValueError("Invalid prediction/permutation geometry")
            if not np.array_equal(np.sort(payload["permutation"]), np.arange(655)):
                raise ValueError("Invalid training permutation")
            payloads[(model, seed)] = payload
    if truth.shape != (608,) or not np.isfinite(truth).all():
        raise ValueError("Invalid truth vector")
    indices = calendar_indices(dates, config["draws"], config["calendar_block_days"], config["seed"])
    if (indices < 0).any():
        raise ValueError("Complete contiguous evaluation calendar required")
    reference_error = (target_scale-truth)**2
    rows, models = [], []
    for model in plan["models"]:
        gains, gain_draws = [], []
        for seed in plan["label_permutation_seeds"]:
            saved = payloads[(model, seed)]
            prediction = saved["prediction"]
            errors = (prediction-truth)**2
            gain = float(np.sqrt(reference_error.mean())-np.sqrt(errors.mean()))
            bootstrap_gains = np.sqrt(reference_error[indices].mean(1))-np.sqrt(errors[indices].mean(1))
            ranks_p, ranks_y = rankdata(prediction), rankdata(truth)
            rho = float(np.corrcoef(ranks_p, ranks_y)[0, 1]) if ranks_p.std() and ranks_y.std() else None
            rs = []
            for start in range(0, len(indices), 250):
                ix = indices[start:start+250]
                a, b = rankdata(prediction[ix], axis=1), rankdata(truth[ix], axis=1)
                a -= a.mean(1, keepdims=True)
                b -= b.mean(1, keepdims=True)
                den = np.sqrt((a*a).sum(1)*(b*b).sum(1))
                rs.extend(np.divide((a*b).sum(1), den, out=np.full(len(ix), np.nan), where=den>0).tolist())
            finite = np.asarray(rs)[np.isfinite(rs)]
            rows.append({"model":model,"permutation_seed":seed,"gain_RMSE_vs_training_mean":gain,
                         "gain_CI95_conditional":np.quantile(bootstrap_gains,[.025,.975]).tolist(),
                         "Spearman_rho":rho,"rho_CI95_conditional":None if not len(finite) else np.quantile(finite,[.025,.975]).tolist(),
                         "valid_rho_draws":len(finite),"fixed_rows":int(np.sum(saved["permutation"]==np.arange(655)))})
            gains.append(gain)
            gain_draws.append(bootstrap_gains)
        mean_gain, draws = float(np.mean(gains)), np.mean(gain_draws,axis=0)
        p_lower = centered_bootstrap_p(draws, mean_gain, -delta)
        p_upper = (1+int(np.sum(delta+draws-mean_gain <= mean_gain)))/(len(draws)+1)
        models.append({"model":model,"registered_new_permutations":len(gains),"mean_gain_RMSE_fixed_three":mean_gain,
                       "range_gain_over_fixed_three":[float(min(gains)),float(max(gains))],
                       "CI90_calendar_conditional":np.quantile(draws,[.05,.95]).tolist(),
                       "p_TOST_lower":p_lower,"p_TOST_upper":p_upper,"p_TOST":max(p_lower,p_upper)})
    for row, adjusted in zip(models,holm([r["p_TOST"] for r in models])):
        row["p_TOST_Holm2"] = adjusted
        row["equivalence_fixed_mean_RMSE_endpoint"] = adjusted <= .05
    result = {"version":plan["version"],"plan_sha256":sha(PACKAGE/"analysis_plan.json"),"completed_fits":len(rows),"planned_fits":6,
              "rows":rows,"models":models,"bootstrap":config,"margin":delta,"original13_model_audit_accepted":False,
              "interpretation":reference["interpretation"],"original_primary_modified":False}
    return result, reference


def compare(actual, expected):
    numeric, maximum = 0, 0.
    def visit(a,b,where):
        nonlocal numeric,maximum
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b): raise ValueError("Different fields: "+where)
            for k in a: visit(a[k],b[k],where+"/"+k)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b): raise ValueError("Different list lengths: "+where)
            for i,(x,y) in enumerate(zip(a,b)): visit(x,y,where+"/"+str(i))
        elif isinstance(a,bool) or isinstance(b,bool) or a is None or b is None or isinstance(a,str) or isinstance(b,str):
            if a!=b: raise ValueError("Different value: "+where)
        elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
            numeric+=1
            difference=abs(a-b)
            maximum=max(maximum,difference)
            if not math.isfinite(float(a)) or not math.isfinite(float(b)) or difference>1e-12:
                raise ValueError("Numeric discrepancy beyond1e-12: "+where)
        else:
            raise TypeError("Unsupported comparison type: "+where)
    visit(actual,expected,"")
    return {"numeric_values_compared":numeric,"max_absolute_difference":maximum,"tolerance":1e-12,"all_fields_match":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true",help="Check included file hashes without importing NumPy/SciPy")
    parser.add_argument("--output",type=Path,default=Path.cwd()/"resultado_auditoria")
    args=parser.parse_args()
    manifest=verify()
    if args.verify_only:
        print(json.dumps({"manifest_files_verified":len(manifest["files"]),"new_fits":0}))
        return
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory: "+str(args.output))
    result,reference=reanalyze()
    agreement=compare(result,reference)
    args.output.mkdir(parents=True)
    for name,value in (("summary_recomputed.json",result),("verification.json",dict(agreement,manifest_files_verified=len(manifest["files"]),new_model_fits=0,new_model_forwards=0))):
        (args.output/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(agreement))


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status":"failed","type":type(error).__name__,"error":str(error)},ensure_ascii=False),file=sys.stderr)
        raise
