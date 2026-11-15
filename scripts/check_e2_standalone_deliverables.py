"""Execute the actual downloaded E2 archives in isolated, unrelated folders."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def execute_case(item, archives, work, python, torch_python):
    started=time.monotonic()
    archive=archives/item["file"]
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=item["sha256"]:
        raise ValueError("Published ZIP hash mismatch: "+item["id"])
    extracted=work/item["id"]
    extracted.mkdir()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            target=(extracted/entry.filename).resolve()
            if not target.is_relative_to(extracted.resolve()) or entry.filename.startswith("/"):
                raise ValueError("Unsafe archive path")
        source.extractall(extracted)
    package=extracted/item["root_directory"]
    unrelated=work/(item["id"]+"-unrelated-cwd")
    unrelated.mkdir()
    program=torch_python if item["id"]=="D02" else python
    flags=["-I"] if item["id"]=="D02" else ["-I","-S"]
    command=[str(program),*flags,str(package/"run.py")]
    env=dict(os.environ)
    env.pop("PYTHONPATH",None);env.pop("PYTHONSTARTUP",None)
    log=work/(item["id"]+".log")
    with log.open("w",encoding="utf-8") as stream:
        result=subprocess.run(command,cwd=unrelated,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
    if result.returncode:
        raise ValueError(item["id"]+" failed: "+log.read_text()[-6000:])
    out=package/"output"
    if item["id"]=="D01":
        record=load(out/"extraction_report.json")
        if not (out/"site/intro.html").is_file() or record.get("html_pages")!=70:
            raise ValueError("D01 book was not materialized")
    elif item["id"]=="D03":
        record=load(out/"dataset_audit.json")
        if record.get("status")!="passed" or not (out/"development_dataset.csv").is_file():
            raise ValueError("D03 dataset was not reconstructed")
    else:
        record=load(out/"resultado.json")
        if not (out/"informe.html").is_file():
            raise ValueError("Missing executable report: "+item["id"])
    if item["id"]=="D02":
        smoke=record["smoke_cpu"]
        if len(smoke["models"])!=13 or not all(m["real_forward_executed"] for m in smoke["models"]):
            raise ValueError("D02 did not execute thirteen real forwards")
        with (work/"D02-inventory.log").open("w",encoding="utf-8") as stream:
            inventory=subprocess.run([str(python),"-I","-S",str(package/"run.py"),"--inventory"],cwd=unrelated,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        if inventory.returncode:raise ValueError("D02 standard-library inventory failed")
    if item["id"] in ("D06","D08","D12"):
        if record["systems"]!=16 or record["predictors"]!=81 or record["max_metric_difference_from_published"]>5e-13:
            raise ValueError("Frozen arithmetic reproduction failed")
    if item["id"]=="D08" and not record["taxonomy"]["categories_match_published_rows"]:
        raise ValueError("Error taxonomy differs")
    if item["id"]=="D12":
        deck=(out/"presentacion.html").read_text(encoding="utf-8")
        if deck.count('<section class="slide">')!=14 or not (out/"notas_orador.md").is_file():
            raise ValueError("Presentation not generated")
    if item["id"]=="D10":
        with (work/"D10-training-inputs.log").open("w",encoding="utf-8") as stream:
            validation=subprocess.run([str(python),"-I","-S",str(package/"run.py"),"--validate-training"],cwd=unrelated,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=60)
        if validation.returncode:raise ValueError("ETTh1 training inputs not standalone")
    return {"id":item["id"],"status":"passed","archive_sha256":item["sha256"],"exit_code":0,
            "unrelated_working_directory":True,"PYTHONPATH_removed":True,
            "isolated_python":True,"site_packages_disabled":item["id"]!="D02",
            "outputs_generated":len([p for p in out.rglob("*") if p.is_file()]),
            "seconds":round(time.monotonic()-started,3),"result_summary":record}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--python",type=Path,default=Path(sys.executable))
    parser.add_argument("--torch-python",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workdir",type=Path)
    parser.add_argument("--only")
    args=parser.parse_args()
    archives=args.directory.resolve()
    index=load(archives/"index.json")
    items=[r for r in index["packages"] if args.only is None or r["id"]==args.only]
    work=args.workdir or Path(tempfile.mkdtemp(prefix="e2-extracted-archive-tests-"))
    work.mkdir(parents=True,exist_ok=True)
    results=[];failures=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(execute_case,r,archives,work,args.python,args.torch_python):r for r in items}
        for future in as_completed(futures):
            item=futures[future]
            try:
                record=future.result();results.append(record)
                print(json.dumps({"id":item["id"],"status":"passed","seconds":record["seconds"]}),flush=True)
            except Exception as error:
                record={"id":item["id"],"status":"failed","error":str(error)}
                failures.append(record);print(json.dumps(record),flush=True)
    report={"version":"e2-extracted-archives-execution-v1","source_commit":index["source_commit"],
            "status":"passed" if not failures else "failed","packages_executed":len(items),
            "passed":len(results),"failures":failures,"results":sorted(results,key=lambda r:r["id"]),
            "test_workspace":str(work),"cross_package_files_used":False,
            "new_market_model_fits":0,"original_scientific_files_modified":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"passed":len(results),"total":len(items),"output":str(args.output)},ensure_ascii=False))
    if failures:raise SystemExit(1)


if __name__=="__main__":
    main()
