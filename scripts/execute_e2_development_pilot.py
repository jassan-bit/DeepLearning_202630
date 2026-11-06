"""Run the registered P0/OOF development workflow; never load test-period data.

The default is a metadata-only preflight. The explicit 'all' workflow registers,
fits the fixed 220 trajectories, scores only after closure, then analyzes saved
development predictions. This is not the optimized final 20-model benchmark.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def jobs(plan):
    result=[["--phase","pilot","--model",model,"--seed",str(seed)]
            for model in plan["models"] for seed in plan["seeds"]]
    if plan["OOF"]["enabled"]:
        result += [["--phase","oof","--model",model,"--seed",str(seed),"--origin",origin]
                   for origin in plan["calendar"]["OOF_origins"]
                   for model in plan["models"] for seed in plan["seeds"]]
    identities=[tuple(x) for x in result]
    if len(identities)!=len(set(identities)):
        raise ValueError("Registered jobs are not unique")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["preflight","register","fit","score","analyze","all"],default="preflight")
    p.add_argument("--design",type=Path,default=ROOT/"experiments/e2_remediation_20261106/pilot_design.json")
    p.add_argument("--source-root",required=True);p.add_argument("--control-source-root",required=True)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--workers",type=int,default=2)
    a=p.parse_args()
    if a.workers not in (1,2):raise ValueError("Use one or two processes, each with one numerical thread")
    output=a.output_dir.resolve()
    if output.is_relative_to(ROOT.resolve()):raise ValueError("Experimental outputs must remain outside the repository")
    output.mkdir(parents=True,exist_ok=True);logs=output/"logs";logs.mkdir(exist_ok=True)
    plan=json.loads(a.design.read_text())
    env=os.environ.copy()
    env.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
    base=[sys.executable,str(ROOT/"scripts/run_e2_remediation_pilot.py"),
          "--design",str(a.design.resolve()),"--source-root",a.source_root,
          "--control-source-root",a.control_source_root,"--output-dir",str(output)]
    def call(command,log):
        with (logs/log).open("ab") as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if a.stage in ("preflight","all"):
        call(base+["--phase","validate"],"preflight.log")
        if not (output/"data_support_audit.json").exists():
            call([sys.executable,str(ROOT/"scripts/e2_remediation_data_audit.py"),
                  "--output",str(output/"data_support_audit.json")],"data_audit.log")
        else:
            saved=json.loads((output/"data_support_audit.json").read_text())
            if saved["source_hashes"]!=plan["development_sources"] or saved["status"]!="passed":
                raise ValueError("The saved data preflight belongs to different development bytes")
    if a.stage in ("register","all"):
        prior=output/"protocol_registration/registration.json"
        if prior.exists():
            from e2_protocol_plan import check_registration
            check_registration(json.loads(prior.read_text()),root=ROOT)
        else:
            call([sys.executable,str(ROOT/"scripts/e2_protocol_plan.py"),"register",
                  "--output-dir",str(output/"protocol_registration")],"protocol_registration.log")
        call(base+["--phase","register"],"registration.log")
    if a.stage in ("fit","all"):
        schedule=jobs(plan)
        def run(index,job):
            call(base+job,f"fit_{index:03d}.log")
            return {"index":index,"arguments":job,"status":"closed",
                    "at_utc":datetime.now(timezone.utc).isoformat()}
        completed=[]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending={pool.submit(run,index,job):(index,job) for index,job in enumerate(schedule)}
            try:
                for future in as_completed(pending):
                    completed.append(future.result())
                    progress={"completed":len(completed),"planned":len(schedule),
                              "new_test_data_read":False,"jobs":completed}
                    (output/"progress.json").write_text(json.dumps(progress,indent=2)+"\n")
                    print(json.dumps({k:progress[k] for k in ["completed","planned","new_test_data_read"]}),flush=True)
            except Exception:
                for future in pending:future.cancel()
                raise
    if a.stage in ("score","all"):
        call(base+["--phase","score"],"score_pilot.log")
        if plan["OOF"]["enabled"]:call(base+["--phase","score-oof"],"score_oof.log")
    if a.stage in ("analyze","all"):
        call([sys.executable,str(ROOT/"scripts/analyze_e2_remediation_pilot.py"),
              "--input-root",str(output),"--output-dir",str(output/"statistical_analysis")],"analysis.log")
    print(json.dumps({"stage":a.stage,"finished":True,"scope":plan["scope"]}))

if __name__=="__main__":main()
