"""Registered operational CPU reconnaissance on admitted development only.

Resources phase: fourteen fixed/reference fits and the largest TimesNet case,
seed271601, up to200 epochs per stage, patience20,150s internal per stage and
180s external per worker. These smaller operational process caps do not replace
the7200s scientific-stage budget. No HPO/NAS search or final-data access.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import tempfile
import os
import time
from typing import Mapping

from profile_e2_training import run_isolated_profile, hardware_snapshot

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"book_e2/_static/search_protocol_20261109/protocol.json"
CANONICAL_REGISTRATION = ROOT/"experiments/e2_development_pilot_20261109/registration.json"
VERSION = "e2-development-resources-runner-v1"
REGISTRATION_SHA256 = "cc10ddba82f0c59cfcd8caf70e713481ec0701e71860862ed13ead0df0bf4797"
ORIGIN = "2023-10-01"
ADMITTED_END = "2023-09-23"
SEED = 271601
INNER_SECONDS = 150
OUTER_SECONDS = 180
MEMORY_BYTES = 12*1024**3
NEURAL_NAMES = ("FCN-1D","ResNet-1D","Inception-1D","LSTM-autoencoder con predictor",
                "TimesNet","TSMixer","TiDE","TimeMixer","PatchTST","iTransformer",
                "CNN-1D + LSTM + atención temporal","MLP16","LSTM32","GRU32")
NAS_NAMES = {"LSTM-autoencoder con predictor","TimesNet","TiDE","TimeMixer",
             "PatchTST","CNN-1D + LSTM + atención temporal"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if value is None or type(value) in (str,bool,int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite pilot metadata cannot be serialized")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k,str) for k in value):
            raise TypeError("Pilot JSON keys must be strings")
        return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_jsonable(v) for v in value]
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,datetime):
        return value.isoformat()
    if type(value).__module__.startswith("numpy") and hasattr(value,"item"):
        return _jsonable(value.item())
    raise TypeError("Non-JSON pilot metadata: "+type(value).__name__)


def _canonical(value):
    return json.dumps(_jsonable(value),sort_keys=True,separators=(",",":"),
                      ensure_ascii=False,allow_nan=False).encode()


def write_json_new(path,value):
    path=Path(path)
    if path.exists():
        raise FileExistsError("Refusing to overwrite pilot record: "+str(path))
    fd,temporary=tempfile.mkstemp(prefix=path.name+".tmp-",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as handle:
            json.dump(_jsonable(value),handle,ensure_ascii=False,indent=2,allow_nan=False)
            handle.write("\n");handle.flush();os.fsync(handle.fileno())
        os.link(temporary,path)  # atomic exclusive publication, never replace
    finally:
        Path(temporary).unlink(missing_ok=True)


class DurableEvents:
    def __init__(self,path):
        self.path=Path(path)
        self.handle=self.path.open("x",encoding="utf-8")
        self.previous="0"*64
        self.sequence=0
        self.started=time.monotonic()

    def append(self,kind,**fields):
        row={"sequence":self.sequence,"kind":kind,"utc":datetime.now(timezone.utc).isoformat(),
             "elapsed_seconds":time.monotonic()-self.started,"previous_sha256":self.previous,
             **fields}
        row=_jsonable(row)
        row["sha256"]=hashlib.sha256(_canonical(row)).hexdigest()
        self.handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        self.handle.flush();os.fsync(self.handle.fileno())
        self.sequence+=1;self.previous=row["sha256"]
        return row

    def close(self):
        self.handle.close()


def read_events(path):
    path=Path(path)
    if not path.exists():
        return {"events":[],"incomplete_tail":False,"sha256":None}
    raw=path.read_bytes();lines=raw.splitlines(keepends=True);events=[];previous="0"*64
    incomplete=False
    for index,line in enumerate(lines):
        if not line.endswith(b"\n") and index==len(lines)-1:
            incomplete=True;break
        row=json.loads(line)
        claimed=row.pop("sha256")
        if (row.get("sequence")!=len(events) or row.get("previous_sha256")!=previous
                or hashlib.sha256(_canonical(row)).hexdigest()!=claimed):
            raise ValueError("Durable pilot event chain changed")
        row["sha256"]=claimed;events.append(row);previous=claimed
    return {"events":events,"incomplete_tail":incomplete,
            "sha256":hashlib.sha256(raw).hexdigest(),"last_event_sha256":previous}


def _default_registration():
    return CANONICAL_REGISTRATION if CANONICAL_REGISTRATION.is_file() else ROOT/"registration.json"


def registration(path):
    path=Path(path).resolve()
    if _sha(path)!=REGISTRATION_SHA256:
        raise ValueError("Registration bytes differ from the pre-fit GitHub commit")
    value=json.loads(path.read_text())
    expected={"origin":ORIGIN,"admitted_end":ADMITTED_END,"seed":SEED,
              "max_epochs_per_stage":200,"patience_epochs":20,
              "internal_stage_wall_seconds":INNER_SECONDS,
              "external_worker_wall_seconds":OUTER_SECONDS,
              "sampled_worker_group_rss_limit_bytes":MEMORY_BYTES}
    if value.get("version")!="e2-development-pilot-registration-v1":
        raise ValueError("Wrong development-pilot registration version")
    if value.get("status")!="registered_before_model_fits":
        raise ValueError("Resources require the published pre-fit registration")
    if any(value.get("resources",{}).get(k)!=v for k,v in expected.items()):
        raise ValueError("Resources phase differs from the pre-fit registration")
    if value.get("source_protocol_sha256")!=_sha(PROTOCOL):
        raise ValueError("Section27 protocol hash differs from pre-fit registration")
    return value


def source_hashes():
    paths={Path(__file__).resolve(),ROOT/"scripts/e2_nas_common.py",
           ROOT/"scripts/e2_nas_factory.py",ROOT/"scripts/e2_nas_attention.py",
           ROOT/"scripts/e2_nas_autoencoder.py",ROOT/"scripts/e2_nas_spectral.py"}
    scripts=(ROOT/"scripts").resolve()
    for module in list(sys.modules.values()):
        filename=getattr(module,"__file__",None)
        if filename:
            p=Path(filename).resolve()
            if p.suffix==".py" and p.is_relative_to(scripts):
                paths.add(p)
    return {str(p.relative_to(ROOT)):_sha(p) for p in sorted(paths) if p.is_file()}


def _failure_code(error):
    receipt=getattr(error,"receipt",None)
    if isinstance(receipt,dict) and isinstance(receipt.get("error_code"),str):
        return receipt["error_code"]
    code=getattr(error,"error_code",None)
    if isinstance(code,str):
        return code
    if isinstance(error,TimeoutError):return "timeout"
    if isinstance(error,MemoryError):return "memory"
    if isinstance(error,(ValueError,TypeError)):return "input_validation"
    if isinstance(error,OSError):return "io"
    return "runtime"


def _partial_steps(receipt):
    if not isinstance(receipt,dict):
        return None
    own=receipt.get("optimizer_steps")
    if type(own) is not int or own<0:
        return None
    done=receipt.get("completed_phases",{})
    total=own
    if isinstance(done,dict):
        for phase in done.values():
            steps=phase.get("optimizer_steps") if isinstance(phase,dict) else None
            if type(steps) is not int or steps<0:return None
            total+=steps
    return total


def run_worker(spec,output):
    output=Path(output)
    events=DurableEvents(output/"events.jsonl")
    started=time.perf_counter();data_receipt=None;before=None
    events.append("worker_started",version=VERSION,phase="resources",model=spec["model"],
                  case=spec["case"],seed=SEED,origin=ORIGIN,admitted_end=ADMITTED_END,
                  internal_stage_wall_seconds=INNER_SECONDS,
                  external_worker_wall_seconds=OUTER_SECONDS,
                  checkpoint_policy="No state_dict files persisted; receipts and selected-state hashes only")
    try:
        import torch
        import e2_search_plan as planner
        from e2_development_data import load_development
        from e2_development_models import fit_pilot
        from e2_temporal_training import prepare_b
        if spec.get("phase")!="resources" or spec.get("seed")!=SEED:
            raise ValueError("Worker requires the registered resources phase and seed")
        if spec["model"] not in NEURAL_NAMES or spec["case"] not in ("reference","largest_timesnet"):
            raise ValueError("Unknown resource worker/model")
        if spec["case"]=="largest_timesnet" and spec["model"]!="TimesNet":
            raise ValueError("Largest resource case is TimesNet only")
        if spec["threads"]!=4:
            raise ValueError("Registered resource workers use four intraop threads")
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        reg=registration(spec["registration"])
        events.append("registration_verified",registration_sha256=_sha(spec["registration"]),
                      registered_at_utc=reg["registered_at_utc"],protocol_sha256=_sha(PROTOCOL))
        panel,data_receipt=load_development(spec["data_root"],ADMITTED_END)
        events.append("data_admitted",data_receipt=data_receipt)
        prepared=prepare_b(panel,ORIGIN,ORIGIN);prepared.validate()
        if (len(prepared.inner_train.dates),len(prepared.inner_validation.dates),
            len(prepared.refit_train.dates))!=(161,83,251):
            raise ValueError("Admitted resource cohort counts changed")
        if any(prepared.outer_plan.eligible):
            raise ValueError("Resources must not have an eligible outer target")
        prepared_record={"signature":prepared.signature,
                         "inner_train":prepared.inner_train.record(),
                         "inner_validation":prepared.inner_validation.record(),
                         "refit_train_prepared_not_fitted":prepared.refit_train.record()}
        events.append("prepared",prepared=prepared_record)
        protocol=json.loads(PROTOCOL.read_text())
        patterns=planner.noise_pilot_architectures(protocol)
        if spec["model"] in NAS_NAMES:
            pattern=(next(p for p in patterns if p["widths"]==[256]*3 and p["dropout_pattern"]==[1]*3)
                     if spec["case"]=="largest_timesnet"
                     else next(p for p in patterns if p["is_reference"]))
        else:
            pattern=None
        before=source_hashes()
        events.append("fit_started",model=spec["model"],pattern=pattern,seed=SEED,
                      epoch_cap=200,patience_epochs=20,internal_stage_wall_seconds=150,
                      source_sha256=before)
        result=fit_pilot(spec["model"],pattern,SEED,prepared,
                         source_roots=spec.get("source_roots"),epoch_cap=200,
                         wallcap=150,memory_limit_bytes=MEMORY_BYTES)
        phase_receipts={name:stage.to_dict() for name,stage in result.phases.items()}
        steps=sum(stage.optimizer_steps for stage in result.phases.values())
        state_bytes={name:sum(t.numel()*t.element_size() for t in stage.state_dict.values())
                     for name,stage in result.phases.items()}
        after=source_hashes()
        if any(after.get(path)!=sha for path,sha in before.items()):
            raise RuntimeError("A loaded source file changed during the resource fit")
        events.append("fit_finished",optimizer_steps=steps,phase_receipts=phase_receipts,
                      selected_state_bytes=state_bytes,source_sha256=after)
        worker={"version":VERSION,"status":"success","phase":"resources",
                "model":spec["model"],"case":spec["case"],"seed":SEED,
                "development_market_data_read":True,"admitted_end":ADMITTED_END,
                "final_data_read":False,"hpo_executed":False,"nas_search_executed":False,
                "fixed_reference_not_hpo_winner":True,"source_sha256_before":before,
                "source_sha256_after":after,"sources_unchanged_during_fit":True,
                "registration_sha256":_sha(spec["registration"]),
                "data_receipt":data_receipt,"prepared":prepared_record,
                "fit_receipt":result.receipt,"phase_receipts":phase_receipts,
                "optimizer_steps":steps,"optimizer_steps_censored":False,
                "selected_state_bytes":state_bytes,
                "selected_states_changed":{name:stage.initial_state_sha256!=stage.selected_state_sha256
                                           for name,stage in result.phases.items()},
                "prediction_count":len(result.predictions_original),
                "predictions_original_sha256":hashlib.sha256(_canonical(result.predictions_original)).hexdigest(),
                "validation_rmse_original":result.rmse_original,
                "elapsed_worker_seconds":time.perf_counter()-started,
                "worker_os_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                "runtime":{"torch":str(torch.__version__),"device":"cpu","threads":4},
                "limits":["Resource reference fit only; not model/architecture selection.",
                          "Refit tensors were prepared for contract verification but no refit or outer scoring occurred.",
                          "Elapsed includes data checks, construction, fit, logging and metadata.",
                          "No full scientific campaign cost or final k/R/power closure."]}
        write_json_new(output/"worker_profile.json",worker)
        events.append("worker_finished",status="success",optimizer_steps=steps)
        return 0
    except Exception as error:
        receipt=getattr(error,"receipt",None)
        steps=_partial_steps(receipt)
        code=_failure_code(error)
        events.append("fit_failed" if before else "worker_failed",error_code=code,
                      error_type=type(error).__name__,message=str(error),
                      optimizer_steps_known=steps,training_failure_receipt=receipt)
        worker={"version":VERSION,"status":"failed","phase":"resources",
                "model":spec.get("model"),"case":spec.get("case"),"seed":SEED,
                "development_market_data_read":data_receipt is not None,
                "final_data_read":False,"admitted_end":ADMITTED_END,
                "error_code":code,"error_type":type(error).__name__,"message":str(error),
                "optimizer_steps":steps,"optimizer_steps_censored":steps is None,
                "training_failure_receipt":receipt,"data_receipt":data_receipt,
                "source_sha256_before":before,"source_sha256_after":source_hashes(),
                "elapsed_worker_seconds":time.perf_counter()-started,
                "worker_os_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        write_json_new(output/"worker_profile.json",worker)
        return 1
    finally:
        events.close()


def resources(data_root,source_root,output,*,models=None,timeout=180.,threads=4,
              registration_path=None):
    output=Path(output)
    if output.exists():raise FileExistsError("Pilot output directory must be new")
    if type(timeout) not in (int,float) or timeout!=OUTER_SECONDS or threads!=4:
        raise ValueError("Registered resources require external180s and four threads")
    names=list(NEURAL_NAMES) if models is None else list(models)
    if not names or len(set(names))!=len(names) or any(n not in NEURAL_NAMES for n in names):
        raise ValueError("Choose distinct registered neural reference model names")
    data_root=Path(data_root).resolve()
    if not data_root.is_dir():raise ValueError("Development source root must exist")
    if "iTransformer" in names and source_root is None:
        raise ValueError("Explicit pinned iTransformer source root required")
    registration_path=Path(registration_path or _default_registration()).resolve()
    reg=registration(registration_path)
    output.mkdir(parents=True)
    events=DurableEvents(output/"events.jsonl")
    jobs=[{"model":name,"case":"reference"} for name in names]
    if "TimesNet" in names:jobs.append({"model":"TimesNet","case":"largest_timesnet"})
    write_json_new(output/"plan.json",{
        "version":VERSION,"phase":"resources","models":names,"jobs":jobs,"seed":SEED,
        "registration":reg,"registration_sha256":_sha(registration_path),
        "protocol_sha256":_sha(PROTOCOL),"source_script_sha256":_sha(__file__),
        "internal_stage_wall_seconds":INNER_SECONDS,"external_worker_wall_seconds":OUTER_SECONDS,
        "rss_watchdog_limit_bytes":MEMORY_BYTES,"rss_watchdog_is_hard_limit":False,
        "scope":"Registered operational resource reconnaissance; not7200s scientific stage budget",
        "hardware":hardware_snapshot()})
    events.append("phase_planned",phase="resources",job_count=len(jobs),
                  full_registered_model_set=names==list(NEURAL_NAMES),
                  registration_sha256=_sha(registration_path))
    rows=[]
    try:
        for index,item in enumerate(jobs):
            worker_dir=output/f"worker-{index:02d}"
            spec={**item,"phase":"resources","seed":SEED,"threads":4,
                  "data_root":str(data_root),"registration":str(registration_path),
                  "source_roots":{"iTransformer":str(Path(source_root).resolve())} if source_root else {}}
            spec_path=output/f"spec-{index:02d}.json";write_json_new(spec_path,spec)
            events.append("worker_launched",index=index,**item,seed=SEED)
            result=run_isolated_profile(
                [sys.executable,str(Path(__file__).resolve()),"--worker-spec",str(spec_path.resolve()),
                 "--worker-output",str(worker_dir.resolve())],
                worker_dir,timeout_seconds=OUTER_SECONDS,memory_limit_bytes=MEMORY_BYTES,
                poll_interval=.05,cwd=ROOT)
            journal=read_events(worker_dir/"events.jsonl")
            worker=result.get("worker_profile")
            if result["status"]=="success" and (not isinstance(worker,dict) or worker.get("status")!="success"):
                status="invalid_worker_result"
            else:
                status=result["status"]
            censored=status in ("timeout","memory_limit_observed")
            if censored:
                error_code="external_wall_timeout" if status=="timeout" else "external_memory_watchdog"
            else:
                error_code=worker.get("error_code") if isinstance(worker,dict) else None
            steps=worker.get("optimizer_steps") if isinstance(worker,dict) else None
            if censored:steps=None
            row={**item,"index":index,"status":status,"error_code":error_code,
                 "returncode":result["returncode"],"optimizer_steps":steps,
                 "optimizer_steps_censored":censored or steps is None,
                 "elapsed_wall_seconds":result["elapsed_wall_seconds"],
                 "sampled_peak_process_group_rss_bytes":result["sampled_peak_process_group_rss_bytes"],
                 "worker_profile":worker,"durable_events":journal}
            rows.append(row)
            write_json_new(output/f"outcome-{index:02d}.json",row)
            events.append("worker_outcome",index=index,model=item["model"],case=item["case"],
                          status=status,error_code=error_code,optimizer_steps=steps,
                          optimizer_steps_censored=row["optimizer_steps_censored"],
                          elapsed_wall_seconds=row["elapsed_wall_seconds"],
                          sampled_peak_process_group_rss_bytes=row["sampled_peak_process_group_rss_bytes"])
            print(json.dumps({"index":index,**item,"status":status,"optimizer_steps":steps},
                             ensure_ascii=False),flush=True)
        complete=all(row["status"]=="success" for row in rows)
        summary={"version":VERSION,"phase":"resources",
                 "status":"complete_requested_fits" if complete else "completed_with_failures_or_censoring",
                 "all_requested_jobs_attempted":len(rows)==len(jobs),
                 "full_registered_job_set":names==list(NEURAL_NAMES),
                 "all_fits_completed":complete,"requested_jobs":jobs,"results":rows,
                 "known_optimizer_steps_sum":sum(r["optimizer_steps"] for r in rows if type(r["optimizer_steps"]) is int),
                 "unknown_optimizer_step_workers":sum(r["optimizer_steps"] is None for r in rows),
                 "elapsed_workers_sum_seconds":sum(r["elapsed_wall_seconds"] for r in rows),
                 "development_market_data_read":any(isinstance(r["worker_profile"],dict)
                    and r["worker_profile"].get("development_market_data_read") for r in rows),
                 "final_data_read":False,"admitted_end":ADMITTED_END,"origin":ORIGIN,
                 "seed":SEED,"internal_stage_wall_seconds":150,"external_worker_wall_seconds":180,
                 "hardware":hardware_snapshot(),"registration_sha256":_sha(registration_path),
                 "source_script_sha256":_sha(__file__),
                 "k_final":None,"R_final":None,"power_achieved":None,
                 "full_hpo_or_nas_executed":False,
                 "limits":["Reference resource pilot only; scores are not used for recipe/HPO selection.",
                           "External180s applies to the whole worker, including all three SAE stages and construction.",
                           "Internal150s is checked by each training stage at safe boundaries.",
                           "Unknown steps on external censoring remain null, never zero.",
                           "Sampled RSS watchdog is not a kernel-enforced memory limit.",
                           "No scientific full-noise pilot, final-data evaluation or k/R/power closure."]}
        events.append("phase_finished",status=summary["status"],attempted_jobs=len(rows),
                      known_optimizer_steps_sum=summary["known_optimizer_steps_sum"],
                      unknown_optimizer_step_workers=summary["unknown_optimizer_step_workers"])
        summary["phase_events"]=read_events(output/"events.jsonl")
        write_json_new(output/"summary.json",summary)
        return summary
    finally:
        events.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--source-root-itransformer",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--phase",choices=("resources",),default="resources")
    parser.add_argument("--models",nargs="+",choices=NEURAL_NAMES)
    parser.add_argument("--timeout",type=float,default=180.)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--registration",type=Path)
    parser.add_argument("--worker-spec",type=Path,help=argparse.SUPPRESS)
    parser.add_argument("--worker-output",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker_spec is not None:
        if args.worker_output is None:parser.error("Worker directory required")
        return run_worker(json.loads(args.worker_spec.read_text()),args.worker_output)
    if args.worker_output is not None:parser.error("Worker directory is internal")
    if args.data_root is None or args.output is None:parser.error("--data-root and --output NEWDIR required")
    summary=resources(args.data_root,args.source_root_itransformer,args.output,models=args.models,
                      timeout=args.timeout,threads=args.threads,registration_path=args.registration)
    return 0 if summary["all_fits_completed"] else 1


if __name__=="__main__":
    raise SystemExit(main())

