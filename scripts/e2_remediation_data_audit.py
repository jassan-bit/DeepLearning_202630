"""Audit development information boundaries before the registered E2 pilot.

Only the two established 2023--2024 daily development derivatives are loaded.
This writes aggregate checks and hashes, never prices or individual predictions.
It covers data/support checks for P0/OOF, not all of guide section 10's future
model-level leakage and permutation tests.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dl_data import load_development
from dl_data_v2 import split_for_origin, fit_relative_preprocessing

ROOT=Path(__file__).resolve().parents[1]

def audit(root=ROOT):
    root=Path(root);data=load_development(root)
    calendar=data["calendar"];features=data["features"];target=data["targets"]
    eligible=data["eligible_by_history"][7]
    if len(calendar)!=731 or not calendar.is_unique or set(calendar.year)!={2023,2024}:
        raise ValueError("Only the original, uncompressed development calendar is allowed")
    expected=pd.Series(np.log(features[:,3]),index=calendar).diff().rolling(7,min_periods=7).std(ddof=0).to_numpy()
    mismatch=np.abs(expected[np.isfinite(expected)]-target[np.isfinite(expected)])
    cohorts=[];all_oof=[]
    for month in range(1,12):
        origin=pd.Timestamp(f"2024-{month:02d}-01",tz="UTC")
        end=(origin+pd.offsets.MonthEnd(0)) if month<11 else pd.Timestamp("2024-12-30",tz="UTC")
        split=split_for_origin(data,7,origin,90)
        train=split["train_indices"];it=split["inner_train_indices"];iv=split["inner_validation_indices"]
        evaluation=np.flatnonzero(eligible&(calendar>=origin)&(calendar<=end))
        if calendar[train].max()>origin-pd.Timedelta(days=8):
            raise ValueError("Outer training outcome crossed its embargo")
        if len(np.intersect1d(train,evaluation)) or len(np.intersect1d(it,iv)):
            raise ValueError("Training and evaluation/stop target identities intersect")
        if calendar[it].max()>calendar[iv].min()-pd.Timedelta(days=8):
            raise ValueError("Internal chronological embargo failed")
        if not np.isfinite(data["windows_by_history"][7][evaluation]).all():
            raise ValueError("Evaluation contains incomplete D7 information")
        def hashes(indices):
            return {sha256(np.asarray(data["windows_by_history"][7][i],dtype="<f8").tobytes()).hexdigest()
                    for i in indices}
        collisions=len(hashes(train)&hashes(evaluation))
        if collisions:raise ValueError("An exact D7 content duplicate crosses train/evaluation")
        pre=fit_relative_preprocessing(data,7,train)
        if pd.Timestamp(pre["last_fit_outcome_day_utc"],tz="UTC")>origin-pd.Timedelta(days=8):
            raise ValueError("Preprocessing fit crossed the allowed target cutoff")
        cohorts.append({"origin":origin.strftime("%Y-%m-%d"),"evaluation_end":end.strftime("%Y-%m-%d"),
                        "training_outcome_max":calendar[train].max().strftime("%Y-%m-%d"),
                        "training_samples":len(train),"inner_training_samples":len(it),
                        "inner_validation_samples":len(iv),"evaluation_unique_origins":len(evaluation),
                        "target_ID_intersections":0,"exact_D7_cross_partition_duplicates":collisions,
                        "train_index_sha256":sha256(np.asarray(train,dtype="<i8").tobytes()).hexdigest(),
                        "evaluation_index_sha256":sha256(np.asarray(evaluation,dtype="<i8").tobytes()).hexdigest(),
                        "scaler_fit_outcome_max":pre["last_fit_outcome_day_utc"]})
        if month<11:all_oof.extend(evaluation.tolist())
    if len(all_oof)!=305 or len(set(all_oof))!=305 or cohorts[-1]["evaluation_unique_origins"]!=60:
        raise ValueError("Registered 305 OOF and 60 pilot calendars differ")
    return {"scope":"P0/OOF development data and support audit; not a full section10 pipeline certificate",
            "at_utc":datetime.now(timezone.utc).isoformat(),"status":"passed",
            "market_test_data_2025_or_later_read":False,"model_fits":0,"model_predictions":0,
            "calendar_days":731,"incomplete_OHLCV_days":["2023-03-24"],"source_hashes":data["source_hashes"],
            "target_reconstruction_max_absolute_difference":float(mismatch.max()),
            "OOF_unique_origins":305,"pilot_unique_origins":60,"folds":cohorts,
            "checks_not_claimed":["new model-level label-permutation sanity","all future adapters",
                                  "near-duplicate performance sensitivity","independent-market generalization"]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--project-root",type=Path,default=ROOT)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=audit(a.project_root);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":r["status"],"folds":len(r["folds"]),"OOF_origins":305,"pilot_origins":60,"model_fits":0}))

if __name__=="__main__":main()
