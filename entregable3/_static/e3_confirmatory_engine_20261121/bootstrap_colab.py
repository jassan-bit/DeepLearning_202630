"""Acquire pinned engine sources and install an isolated CPU environment.

Setup checks software with generated fixtures. It never acquires market bars,
trains on market data, opens a reserved test or receives storage credentials.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import venv

REPOSITORY="jassan-bit/DeepLearning_202630"
PREFIX="experiments/e3_confirmatory_engine_20261121"


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(body):return hashlib.sha256(body).hexdigest()


def cpu_env():
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTHONNOUSERSITE="1")
    for key in ("PYTHONPATH","GITHUB_TOKEN","GH_TOKEN","E3_V5_BACKUP_KEY","E3_R3_BACKUP_KEY",
                "E3_R3_GITHUB_TOKEN","E3_R3_BACKUP_SECRET","ALPHAVANTAGE_API_KEY","NASDAQ_API_KEY","NASDAQ_DATA_LINK_API_KEY"):
        env.pop(key,None)
    return env


def download(url,max_bytes):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"E3-temporal-engine-1"}),timeout=90) as response:
                data=response.read(max_bytes+1)
            require(len(data)<=max_bytes,"Pinned download exceeds byte limit")
            return data
        except urllib.error.HTTPError as error:
            if attempt==2 or error.code not in (404,429,500,502,503,504):raise
            delay=error.headers.get("Retry-After")
            time.sleep(min(30,float(delay)) if delay and delay.isdigit() else 2*(attempt+1))


def atomic(path,raw):
    path=Path(path)
    require(not any(p.is_symlink() for p in (path,*path.parents)),"Symlink path refused")
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        require(path.read_bytes()==raw,"Cached source changed; preserve this cache and use a fresh path")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        tmp=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def acquire(source_commit,manifest_sha256,cache_root):
    require(re.fullmatch(r"[a-f0-9]{40}",source_commit or ""),"Immutable source commit required")
    require(re.fullmatch(r"[a-f0-9]{64}",manifest_sha256 or ""),"Pinned source manifest hash required")
    root=Path(cache_root).absolute()/("source-"+source_commit)
    base=f"https://raw.githubusercontent.com/{REPOSITORY}/{source_commit}/{PREFIX}/"
    raw=download(base+"SOURCE_MANIFEST.json",512*1024)
    require(sha(raw)==manifest_sha256,"Source manifest SHA256 mismatch")
    manifest=json.loads(raw)
    require(manifest.get("algorithm")=="sha256" and isinstance(manifest.get("files"),dict),"Invalid source manifest")
    for relative,expected in manifest["files"].items():
        p=PurePosixPath(relative)
        require(not p.is_absolute() and p.as_posix()==relative and ".." not in p.parts and not relative.startswith("."),"Unsafe source path")
        size=manifest.get("sizes",{}).get(relative)
        require(type(size) is int and 0<=size<=64*1024*1024,"Pinned source byte size required")
        target=root/relative
        require(not any(x.is_symlink() for x in (target,*target.parents)),"Symlink cached source refused")
        body=target.read_bytes() if target.exists() else download(base+urllib.parse.quote(relative,safe="/"),size)
        require(len(body)==size and sha(body)==expected,"Source size/hash mismatch: "+relative)
        atomic(target,body)
    atomic(root/"SOURCE_MANIFEST.json",raw)
    anchor={"source_commit":source_commit,"manifest_sha256":manifest_sha256,"scope":"immutable published engine; not retrospective preregistration"}
    atomic(root/"PUBLIC_SOURCE_ANCHOR.json",(json.dumps(anchor,sort_keys=True)+"\n").encode())
    return root


def install(root,cache_root,*,existing_python=None):
    require((3,11)<=sys.version_info[:2]<=(3,13),"Python3.11–3.13 required by this CPU lock")
    lock=root/"requirements-cpu.lock.txt"
    if existing_python:
        python=Path(existing_python)
    else:
        directory=Path(cache_root).absolute()/("venv-cpu-"+sha(lock.read_bytes())[:16]+f"-py{sys.version_info.major}{sys.version_info.minor}")
        require(not any(x.is_symlink() for x in (directory,*directory.parents)),"Symlink runtime refused")
        python=directory/"bin/python"
        if not python.exists():venv.EnvBuilder(with_pip=False,system_site_packages=False,symlinks=True).create(directory)
        subprocess.run([sys.executable,"-m","pip","--python",str(python),"install","--disable-pip-version-check","-r",str(lock)],env=cpu_env(),check=True)
    check=("import importlib.metadata as m,pathlib,sys; pins=[x.strip().split('==') for x in pathlib.Path(sys.argv[1]).read_text().splitlines() if '==' in x and not x.startswith('--')]; "
           "bad=[(n,v,m.version(n)) for n,v in pins if m.version(n)!=v]; assert not bad,bad")
    subprocess.run([str(python),"-c",check,str(lock)],env=cpu_env(),check=True)
    subprocess.run([sys.executable,"-m","pip","--python",str(python),"check"],env=cpu_env(),check=True)
    return python


def setup(source_commit,manifest_sha256,cache_root,*,existing_python=None,run_tests=True):
    root=acquire(source_commit,manifest_sha256,cache_root)
    python=install(root,cache_root,existing_python=existing_python)
    if run_tests:
        # Tests include minimal fitting on generated fixtures, never market bars.
        subprocess.run([str(python),"-m","pytest","-q","tests"],cwd=root,env=cpu_env(),check=True)
    subprocess.run([str(python),"-m","engine.cli","--source-root",str(root),"check"],cwd=root,env=cpu_env(),check=True)
    settings={"python":str(python),"source_root":str(root),"source_commit":source_commit,
              "cache_root":str(Path(cache_root).absolute()),"software_tests_run":run_tests,
              "native_compiler_available":bool(shutil.which("c++") or shutil.which("g++") or shutil.which("cc")),
              "new_market_fits":0,"reserved_test_opened":False}
    print(json.dumps({"status":"engine_installed_no_market_training",**settings},ensure_ascii=False))
    return settings


def command(settings,args,*,credentials=None,allow_blocked=False):
    """Stream real stage progress, redacting before display and retention.

    Credentials go only to child stdin. Returned stdout contains the same
    sanitized lines already displayed, including the final structured result.
    """
    require(isinstance(args,list) and all(isinstance(x,str) for x in args),"CLI arguments must be explicit strings")
    argv=[settings["python"],"-u","-m","engine.cli","--source-root",settings["source_root"],*args]
    replacements=set()
    for value in (credentials or {}).values():
        if isinstance(value,str) and value:
            replacements.update((value,urllib.parse.quote(value,safe=""),urllib.parse.quote_plus(value,safe="")))
            # Private terminal entry is one line, but sanitize each component
            # too if a caller explicitly supplied a multiline backup secret.
            replacements.update(part for part in value.splitlines() if part)
    replacements=sorted(replacements,key=len,reverse=True)
    lines=[]
    process=subprocess.Popen(argv,cwd=settings["source_root"],env=cpu_env(),text=True,
                             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=1)
    def close_private_input():
        try:process.stdin.close()
        except BrokenPipeError:return False
        return True
    try:
        try:
            if credentials is not None:
                process.stdin.write(json.dumps(credentials))
        except BrokenPipeError:
            # Preserve the child's structured early error instead of replacing
            # it with a private-input transport traceback.
            close_private_input()
        finally:
            if not process.stdin.closed:close_private_input()
        for raw_line in process.stdout:
            clean=raw_line
            for secret in replacements:clean=clean.replace(secret,"[REDACTED]")
            lines.append(clean)
            print(clean,end="",flush=True)
        result=subprocess.CompletedProcess(argv,process.wait(),stdout="".join(lines),stderr=None)
    except BaseException:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
        raise
    finally:
        if not process.stdin.closed:close_private_input()
        process.stdout.close()
    if result.returncode not in ((0,2) if allow_blocked else (0,)):
        raise RuntimeError("Engine stage stopped; inspect its structured output. Local evidence retained.")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit",required=True);p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--cache",type=Path,default=Path("/content/e3-temporal-engine-1"))
    p.add_argument("--existing-python");p.add_argument("--skip-tests",action="store_true")
    a=p.parse_args(argv)
    setup(a.source_commit,a.manifest_sha256,a.cache,existing_python=a.existing_python,run_tests=not a.skip_tests)


if __name__=="__main__":main()
