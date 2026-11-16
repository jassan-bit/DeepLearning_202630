"""Run the registered thirteen real controls, reusing only verified completions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "book_e2/_static/e2_scientific_closure_20261116/controls"
WORK = ROOT / "work/e2-scientific-controls"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_existing(path, name, spec_sha):
    row = json.loads(path.read_text())
    if row.get("status") != "completed" or row.get("model") != name or row.get("seed") != 271801 or row.get("spec_sha256") != spec_sha:
        raise ValueError("Previous control is not a matching completed result: "+name)
    for kind in ("checkpoint", "prediction"):
        artifact = Path(row[kind+"_path"]).resolve()
        if not artifact.is_relative_to(WORK.resolve()) or not artifact.is_file() or artifact.is_symlink() or sha(artifact) != row[kind+"_sha256"]:
            raise ValueError("Previous control artifact differs: "+name+"/"+kind)
    return row


def main():
    import e2_corrections_controls as original
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    spec_sha = sha(args.spec)
    registered = json.loads(args.spec.read_text())
    if registered["negative"]["models"] != original.MODELS:
        raise ValueError("The fixed thirteen-model family differs")
    plan_path = PUBLIC/"negative_family_execution_plan.json"
    if args.register_only:
        if plan_path.exists(): raise ValueError("Preserve the previously registered execution plan")
        completed = []
        for i, name in enumerate(original.MODELS, 1):
            path = PUBLIC/("negative_"+str(i).zfill(2)+".json")
            if path.exists():
                validate_existing(path, name, spec_sha)
                completed.append({"model":name,"result_sha256":sha(path)})
        plan={"registered_at_utc":datetime.now(timezone.utc).isoformat(),
              "spec_path":args.spec.resolve().relative_to(ROOT).as_posix(),"spec_sha256":spec_sha,
              "controller_sha256":sha(__file__),"adapter_sha256":sha(ROOT/"scripts/e2_scientific_controls.py"),
              "models_in_fixed_order":original.MODELS,"completed_controls_to_reuse":completed,
              "scope":"Execute every remaining registered fixed-seed control; no score-based exclusion, HPO or recipe changes",
              "partial_attempt_policy":"A previous request without a verified completion blocks implicit retries",
              "new_blind_test_or_original_primary_change":False}
        with plan_path.open("x",encoding="utf-8") as stream:
            json.dump(plan,stream,ensure_ascii=False,indent=2);stream.write("\n")
        print(json.dumps({"registered_plan":str(plan_path),"sha256":sha(plan_path),"fits":0}))
        return 0
    plan=json.loads(plan_path.read_text())
    if plan["controller_sha256"]!=sha(__file__) or plan["adapter_sha256"]!=sha(ROOT/"scripts/e2_scientific_controls.py") or plan["spec_sha256"]!=spec_sha:
        raise ValueError("Registered execution identity changed")
    original.require_budget(args.phase)
    for i,name in enumerate(original.MODELS,1):
        path=PUBLIC/("negative_"+str(i).zfill(2)+".json")
        if path.exists():
            validate_existing(path,name,spec_sha)
            print(json.dumps({"model":name,"status":"verified_completed_control_reused"}),flush=True)
            continue
        if (WORK/"negative"/str(i).zfill(2)/"request.json").exists():
            raise ValueError("Unresolved previous control attempt: "+name)
        command=[sys.executable,str(ROOT/"scripts/e2_scientific_controls.py"),"negative","--model",name,
                 "--spec",str(args.spec.resolve()),"--phase",args.phase]
        print(json.dumps({"model":name,"status":"starting_real_fixed_recipe_fit"}),flush=True)
        started=time.monotonic();process=subprocess.run(command,check=False)
        if process.returncode: return process.returncode
        validate_existing(path,name,spec_sha)
        print(json.dumps({"model":name,"status":"verified_completed_control","wall_seconds":time.monotonic()-started}),flush=True)
    print(json.dumps({"completed_controls":13,"HPO":False,"new_primary_model_selected":False}))
    return 0


if __name__=="__main__":raise SystemExit(main())
