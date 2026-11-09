"""Bounded, self-generated integration fixture for eighteen E2 systems.

This CLI accepts no market data paths. It is not the18-trial/1000-architecture
scientific campaign: every reduced domain and epoch cap is declared in output.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

import e2_search_plan as planner
import e2_training_models as registry
from e2_nas_factory import MODEL_IDS
from e2_temporal_training import DatedPanel, prepare_b, refit_b
from e2_training_bindings import (CheckpointStore, NeuralTrainingAdapter,
                                  run_neural_selection, selection_record,run_selected_refits)
from e2_staged_training import state_sha256
from e2_classical_training import select_classical, refit_classical
from e2_real_executor import DatedReplay, DatedJournal, run_frozen_block

SOURCE_BASE="f0c15f9df39ebf7c1073a607dab82cbee54fcb18"


def artificial_panel():
    rng=np.random.default_rng(291109)
    n=250
    dates=pd.date_range("2001-01-01",periods=n,freq="D",tz="UTC")
    volatility=.008+.002*np.sin(np.arange(n)/19)
    returns=rng.normal(size=n)*volatility
    close=100*np.exp(np.cumsum(returns))
    opened=np.r_[close[0],close[:-1]]
    high=np.maximum(opened,close)*(1+.001+rng.random(n)*.002)
    low=np.minimum(opened,close)*(1-.001-rng.random(n)*.002)
    volume=np.exp(rng.normal(7,.3,n))
    features=np.column_stack([opened,high,low,close,volume])
    quality=np.ones(n,dtype=bool);quality[50]=False
    log_returns=pd.Series(np.log(close),index=dates).diff()
    targets=log_returns.rolling(7,min_periods=7).std(ddof=0).to_numpy()
    return DatedPanel(dates,features,targets,quality)


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    hashes={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if not filename:
            continue
        path=Path(filename).resolve()
        if path.suffix==".py" and path.is_relative_to(root/"scripts") and path.is_file():
            hashes[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["scripts/run_e2_training_synthetic.py"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def _write(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def run(output, *, names=None, source_roots=None, threads=4):
    output=Path(output)
    if output.exists():
        raise FileExistsError("Synthetic integration output must be a new directory")
    output.mkdir(parents=True)
    torch.set_num_threads(threads)
    panel=artificial_panel()
    prepared=prepare_b(panel,panel.calendar[220],panel.calendar[222])
    protocol=registry.protocol()
    names=tuple(names or registry.NEURAL_NAMES)
    if len(set(names))!=len(names) or any(n not in registry.NEURAL_NAMES for n in names):
        raise ValueError("Unique declared neural names required")
    store=CheckpointStore(output/"checkpoints")
    neural,adapters,chosen,stage_records={},[],[],{}
    started=time.perf_counter()
    try:
        for name in names:
            adapter=NeuralTrainingAdapter(name,store,source_roots=source_roots,
                full_epochs=2,short_epochs=1,patience=20,wall_limit_seconds=120,
                short_wall_limit_seconds=120)
            trials=registry.registered_trials(name)[:2]
            architectures=[]
            if name in MODEL_IDS:
                reference=registry.reference_architecture(name)
                smaller=next(a for a in planner.enumerate_architectures(protocol,name)
                             if a["widths"]==[16,32,64] and a["dropout_pattern"]==[0,1,0])
                architectures=[reference,smaller]
            selection,stages=run_neural_selection(prepared,name,adapter,
                hpo_trials=trials,architectures=architectures,
                hpo_seeds=protocol["common_hpo"]["optimization_seeds"],
                phase1_seeds=[protocol["nas"]["phase1_seed"]],
                phase2_seeds=protocol["nas"]["phase2_seeds"],k=2,
                delta=protocol["contract"]["delta"],scope="synthetic_fixture")
            # One artificial replay replicate. This is not the R6..30 protocol.
            refits=run_selected_refits(prepared,selection,adapter,
                [protocol["budget"]["final_seeds"][0]],scope="synthetic_fixture")
            if any(r.status!="completed" for r in refits):
                raise RuntimeError("A fresh refit failed; do not replace it")
            chosen.extend(r.artifact.frozen_system() for r in refits)
            adapters.append(adapter)
            neural[name]={"selection":selection_record(selection),
                          "refits":[{"run_id":r.run_id,"phase_epochs":r.phase_epochs,
                                     "resource_use":r.resource_use,
                                     "fresh_initialization":r.fresh_initialization} for r in refits]}
            stage_records[name]={key:selection_record(value) for key,value in stages.items()}
            _write(output/"neural_progress.json",{"neural":neural,"stages":stage_records})
            print(json.dumps({"neural_system":name,"status":"selection_and_fresh_refit_completed"}),flush=True)
        inner_journal=DatedJournal(output/"classical_inner_predictions")
        try:
            classical=select_classical(panel,prepared,source_commit=SOURCE_BASE,
                prediction_sink=lambda event:inner_journal.append("inner_forecast_before_observe",**event))
        finally:
            inner_journal.close()
        classical_refits=refit_classical(panel,prepared,classical,source_commit=SOURCE_BASE)
        chosen.extend(classical_refits.systems)
        _write(output/"classical.json",{"selection":classical.record(),"refits":classical_refits.receipts})
        signatures_before={s.name:s.fingerprint() for s in chosen if s.fingerprint}
        replay=DatedReplay(panel.calendar,panel.features,panel.targets,quality=panel.quality)
        journal=DatedJournal(output/"outer_replay")
        try:
            block=run_frozen_block(prepared.outer_plan,replay,chosen,journal)
        finally:
            journal.close()
        signatures_after={s.name:s.fingerprint() for s in chosen if s.fingerprint}
        if signatures_before!=signatures_after:
            raise AssertionError("Frozen weights/scales changed during outer replay")
        runs=[r for adapter in adapters for r in adapter.run_records]
        completed=[r for r in runs if r["status"]=="completed"]
        checkpoints=[r["resource_use"]["checkpoint"] for r in completed]
        expected_files=set(record["file"] for record in checkpoints)
        actual_files=set(path.name for path in (output/"checkpoints").glob("*.pt"))
        if expected_files!=actual_files:
            raise AssertionError("Checkpoint inventory differs from fit receipts")
        for record in checkpoints:
            path=output/"checkpoints"/record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record["sha256"]:
                raise AssertionError("Checkpoint changed after durable creation")
            restored=torch.load(path,weights_only=True)
            expected=restored["phase_records"]["supervised"]["selected_state_sha256"]
            if state_sha256(restored["state_dict"])!=expected:
                raise AssertionError("Safe-loaded checkpoint does not match its selected predictor")
        report={"version":"e2-staged-synthetic-integration-v1","status":"complete",
            "synthetic_only":True,"market_data_read":False,"full_protocol_search_executed":False,
            "source_base_commit":SOURCE_BASE,"scope":{"hpo_trials_per_neural":2,
            "nas_architectures":2,"nas_finalists":2,"nas_phase2_seeds":5,
            "full_epochs":2,"short_epochs":1,"refit_replay_seeds":1,
            "temporal_origins":1,"classical_domains":"18Ridge/18ARIMA/3seasonal/1persistence"},
            "logical_system_count":len(names)+4,"neural_system_count":len(names),
            "fitted_neural_runs":len(completed),"failed_neural_runs":len(runs)-len(completed),
            "optimization_stages":sum(r["resource_use"]["optimization_stages"] for r in completed),
            "optimizer_steps":sum(r["resource_use"]["optimizer_steps"] for r in completed),
            "checkpoint_count":len(checkpoints),"checkpoint_bytes":sum(c["bytes"] for c in checkpoints),
            "checkpoints_safely_reloaded":len(checkpoints),
            "neural":neural,"stage_selections":stage_records,"training_runs":runs,
            "classical":{"selection":classical.record(),"refits":classical_refits.receipts},
            "prepared":{"signature":prepared.signature,"inner_train":prepared.inner_train.record(),
                        "inner_validation":prepared.inner_validation.record(),
                        "refit_train":prepared.refit_train.record(),"outer_plan":prepared.outer_plan.record()},
            "outer_replay":block,"source_sha256":source_hashes(),
            "elapsed_seconds":time.perf_counter()-started,
            "checkpoints_are_inference_artifacts_not_optimizer_resume":True,
            "limits":["Artificial fixture is software integration, not predictive evidence.",
                      "Reduced synthetic domains do not satisfy full18/1000 scientific search.",
                      "One replay seed does not establish statistical power or the final R.",
                      "No final-market data access or support certification is performed."]}
        _write(output/"summary.json",report)
        print(json.dumps({"status":"complete","systems":len(chosen),"optimizer_steps":report["optimizer_steps"]}),flush=True)
        return report
    except Exception as error:
        _write(output/"failure.json",{"status":"failed","synthetic_only":True,
            "market_data_read":False,"error_type":type(error).__name__,"message":str(error),
            "completed_neural":list(neural),"neural":neural,
            "source_sha256":source_hashes(),"elapsed_seconds":time.perf_counter()-started})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-root-itransformer",type=Path,required=True)
    parser.add_argument("--threads",type=int,choices=(1,2,4),default=4)
    args=parser.parse_args()
    run(args.output,source_roots={"iTransformer":str(args.source_root_itransformer)},threads=args.threads)


if __name__=="__main__":
    main()

