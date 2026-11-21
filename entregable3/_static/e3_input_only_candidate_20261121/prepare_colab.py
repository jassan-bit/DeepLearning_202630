"""Pinned preparation of the input-only dossier, never a training launcher.

The standard-library audit needs no GPU, credentials or Drive mount. An optional
isolated CPU environment runs only generated-fixture software checks. This file
has no command that opens a reserved test or trains a forecasting model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import urllib.request
import venv

REPOSITORY = "jassan-bit/DeepLearning_202630"
PREFIX = "experiments/e3_input_only_candidate_20261121"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def download(url, *, limit):
    request=urllib.request.Request(url,headers={"User-Agent":"e3-input-only-dossier-1"})
    with urllib.request.urlopen(request,timeout=90) as response:
        body=response.read(limit+1)
    require(len(body)<=limit,"Download exceeds pinned size limit")
    return body


def atomic_write(path, body):
    path=Path(path)
    require(not any(p.is_symlink() for p in (path,*path.parents)),"Symlink path refused")
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_bytes()==body,"Existing cached file changed; preserve it and use a fresh cache")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        stream.write(body);stream.flush();os.fsync(stream.fileno())
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def acquire(source_commit, manifest_sha256, cache):
    require(re.fullmatch(r"[0-9a-f]{40}",source_commit or ""),"Immutable source commit required")
    require(re.fullmatch(r"[0-9a-f]{64}",manifest_sha256 or ""),"Pinned manifest SHA256 required")
    base=f"https://raw.githubusercontent.com/{REPOSITORY}/{source_commit}/{PREFIX}"
    body=download(base+"/SOURCE_MANIFEST.json",limit=256*1024)
    require(sha(body)==manifest_sha256,"Manifest hash mismatch")
    manifest=json.loads(body)
    require(manifest.get("algorithm")=="sha256" and isinstance(manifest.get("files"),dict),"Invalid source manifest")
    root=Path(cache).absolute()/f"source-{source_commit}"
    for name,expected in manifest["files"].items():
        member=PurePosixPath(name)
        require(member.as_posix()==name and not member.is_absolute() and ".." not in member.parts and not name.startswith("."),"Unsafe manifest member")
        require(re.fullmatch(r"[0-9a-f]{64}",expected or ""),"Invalid file digest")
        size=manifest.get("sizes",{}).get(name)
        require(type(size) is int and 0<=size<=8*1024*1024,"Pinned byte size required")
        target=root/name
        require(not any(p.is_symlink() for p in (target,*target.parents)),"Symlink path refused")
        raw=target.read_bytes() if target.exists() else download(base+"/"+name,limit=size)
        require(len(raw)==size and sha(raw)==expected,"Source file hash mismatch: "+name)
        atomic_write(target,raw)
    atomic_write(root/"SOURCE_MANIFEST.json",body)
    return root


def cpu_environment():
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="")
    for key in ("GITHUB_TOKEN","GH_TOKEN","E3_V5_BACKUP_KEY"):
        env.pop(key,None)
    return env


def software_tests(root, cache, *, existing_python=None):
    require((3,11)<=sys.version_info[:2]<=(3,13),"Python 3.11 to 3.13 required for the pinned CPU lock")
    lock=root/"requirements-cpu.lock.txt"
    if existing_python:
        python=Path(existing_python)
    else:
        runtime=Path(cache).absolute()/("venv-cpu-"+sha(lock.read_bytes())[:16]+f"-py{sys.version_info.major}{sys.version_info.minor}")
        require(not any(p.is_symlink() for p in (runtime,*runtime.parents)),"Symlink runtime refused")
        python=runtime/"bin/python"
        if not python.exists():
            venv.EnvBuilder(with_pip=False,system_site_packages=False,symlinks=True).create(runtime)
        subprocess.run([sys.executable,"-m","pip","--python",str(python),"install","--disable-pip-version-check","-r",str(lock)],check=True,env=cpu_environment())
    check=("import importlib.metadata as m,pathlib,sys; "
           "pins=[x.strip().split('==') for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if '==' in x and not x.startswith('--')]; "
           "bad=[(n,v,m.version(n)) for n,v in pins if m.version(n)!=v]; assert not bad,bad")
    subprocess.run([str(python),"-c",check,str(lock)],check=True,env=cpu_environment())
    subprocess.run([str(python),"-m","unittest","discover","-s","tests","-v"],cwd=root,check=True,env=cpu_environment())
    return str(python)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--manifest-sha256",required=True)
    parser.add_argument("--cache",type=Path,default=Path("/content/e3-input-only-dossier-1"))
    parser.add_argument("--software-tests",action="store_true")
    parser.add_argument("--existing-python",help="Use an existing environment only if every package matches the lock")
    args=parser.parse_args(argv)
    root=acquire(args.source_commit,args.manifest_sha256,args.cache)
    output=Path(args.cache).absolute()/"audit-output"
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(root/"analysis/audit_saved_results.py"),"--zip",str(root/"inputs/public_results.zip"),"--output",str(output)],cwd=root,check=True)
    admission=subprocess.run([sys.executable,"-m","runtime.preflight","--root",str(root),"--output","audit/preflight.json"],cwd=root,capture_output=True,text=True)
    require(admission.returncode in (0,2),"Preparation audit failed; inspect its diagnostic")
    report_file=root/"audit/preflight.json"
    require(report_file.is_file(),"Admission audit did not produce its report: "+admission.stdout[-500:])
    report=json.loads(report_file.read_text())
    require(report.get("gate_counts",{}).get("integrity")==0,"Source integrity check failed; inspect "+str(report_file))
    print(json.dumps({"audit":"admission","admission_ready":report["admission_ready"],
                      "gate_counts":report["gate_counts"],"full_report":str(report_file),
                      "confirmation_ready":False},ensure_ascii=False))
    tested_python=software_tests(root,args.cache,existing_python=args.existing_python) if args.software_tests else None
    print(json.dumps({"status":"dossier_prepared_no_training", "source_commit":args.source_commit,
                      "source_root":str(root),"audit_output":str(output),"preflight_output":str(root/"audit/preflight.json"),"software_tests_requested":args.software_tests,
                      "test_python":tested_python,"new_forecasting_model_fits":0,"reserved_test_opened":False,
                      "admission_exit_code":admission.returncode,"confirmation_execution_in_this_launcher":False,"R3_accredited":False},ensure_ascii=False))


if __name__=="__main__":main()
