"""Optional Colab launcher. Calling execute with enabled=False performs no I/O.

No setup/run/assess/backup is called. Secrets are read only by the explicitly
invoked private recovery cell and passed via stdin, never argv, environment or
output. Original snapshots and output CSVs stay local and private.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import venv

SOURCE_COMMIT='c4ca86f1431afeaa958ff0b544b453ab17a4d7b8'
MANIFEST_SHA='4706f20a00508e492a54e25873b4885cead3fa7660d43bdb18c5aeda4fce2509'
BOOTSTRAP_SHA='3a6bce99e149e3a1e6cfbb07b8e23fb6482d9bb876597d4f8d5df198cde07941'
BASE='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'
SOURCE_BASE=BASE+SOURCE_COMMIT+'/experiments/e3_seoi_v5_pilot_20261121/'
DATA_URL=BASE+'8f65eac34aaf49293924ae8328c235cedbdd2866/book_e2/_static/e2_scientific_closure_20261116/data_audit/current_cohort_arrays.npz'
RESULT_URL=BASE+'05c1a637f13356dd1df6b8c5aa8f4676dda61c12/runs/e3_seoi_v5/run-01/public_results.zip'
DATA_SHA='186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'
RESULT_SHA='beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c'


def download_verified(url,path,expected,size=None,limit=1048576):
    path=Path(path)
    if path.is_symlink():raise RuntimeError('Symlink refused')
    if path.exists():raw=path.read_bytes()
    else:
        with urllib.request.urlopen(url,timeout=60) as stream:raw=stream.read(limit+1)
    if len(raw)>limit or (size is not None and len(raw)!=size) or hashlib.sha256(raw).hexdigest()!=expected:
        raise RuntimeError('Pinned public download differs; preserve cache and stop')
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.verified-',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as stream:stream.write(raw)
            os.chmod(tmp,0o600);os.replace(tmp,path)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    return path


def cpu_env():
    result={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PYTHONHOME'} and
        not any(word in k.upper() for word in ('TOKEN','SECRET','BACKUP_KEY','PASSWORD','CREDENTIAL'))}
    result.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',PYTHONHASHSEED='0',MPLBACKEND='Agg')
    return result


def compatible(python):
    probe='import numpy,torch,cryptography;assert numpy.__version__=="2.3.5";assert torch.__version__=="2.8.0+cpu"'
    try:return subprocess.run([str(python),'-c',probe],env=cpu_env(),capture_output=True,timeout=60).returncode==0
    except (OSError,subprocess.TimeoutExpired):return False


def select_python(preferred,source,work,install_runtime):
    choices=[preferred] if preferred else []
    choices.extend([work/'runtime'/ 'bin/python',sys.executable])
    for choice in choices:
        if choice and compatible(choice):return Path(choice)
    if not install_runtime:
        raise RuntimeError('No compatible CPU runtime. Set INSTALL_RUNTIME=True only in this optional notebook, or set PYTHON to an existing Torch2.8.0+cpu / NumPy2.3.5 interpreter. No secrets have been requested.')
    directory=work/'runtime'
    if directory.exists():raise RuntimeError('Preserve incompatible runtime and choose a fresh WORK directory')
    venv.EnvBuilder(with_pip=True).create(directory)
    python=directory/'bin/python'
    # Only the verified original lock, in this isolated optional environment.
    subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','-r',str(source/'requirements-cpu.lock.txt')],
        env=cpu_env(),check=True)
    if not compatible(python):raise RuntimeError('Optional CPU runtime verification failed')
    return python


def private_secret(name,prompt):
    try:
        from google.colab import userdata
        value=userdata.get(name)
        if value:return value
    except Exception:pass
    import getpass
    return getpass.getpass(prompt)


def execute(*,enabled=False,install_runtime=False,python=None,work='/content/e3-v5-readonly-recovery',
            helper=None,helper_sha256=None):
    if not enabled:
        return {'status':'DISABLED_NO_DOWNLOAD_NO_INSTALL_NO_SECRET_NO_INFERENCE','training_steps':0,'remote_writes':0}
    helper=Path(helper).resolve()
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=helper_sha256:
        raise RuntimeError('Recovery helper differs from notebook source pin')
    work=Path(os.path.abspath(work))
    if any(p.is_symlink() for p in (work,*work.parents)):raise RuntimeError('Symlink workspace refused')
    work.mkdir(parents=True,exist_ok=True);os.chmod(work,0o700)
    if work.is_relative_to(helper.parent):
        raise RuntimeError('Private work must be outside the public code packet')
    old_umask=os.umask(0o077)
    try:
        output=work/'fit_selection_PRIVATE'
        if output.exists():
            process=subprocess.run([sys.executable,str(helper),'--output-dir',str(output),'--verify-only'],
                input='{}',text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=cpu_env(),timeout=60)
            try:return json.loads(process.stdout.strip().splitlines()[-1])
            except Exception:return {'status':'STOPPED','diagnostic':'Existing output verification failed; private files preserved'}
        bootstrap=download_verified(SOURCE_BASE+'bootstrap_colab.py',work/'cache/bootstrap_colab.py',BOOTSTRAP_SHA)
        spec=importlib.util.spec_from_file_location('original_v5_bootstrap_readonly_acquisition',bootstrap)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        # acquire_source only: never setup(), runner.run() or GithubStore.backup().
        source=module.acquire_source(SOURCE_COMMIT,MANIFEST_SHA,work/'cache')
        interpreter=select_python(python,source,work,install_runtime)
        data=download_verified(DATA_URL,work/'cache/cohort.npz',DATA_SHA,460090)
        public=download_verified(RESULT_URL,work/'cache/public_results.zip',RESULT_SHA,124067)
        case=work/'original_private_case'
        args=[str(interpreter),str(helper),'--output-dir',str(output)]
        credentials={}
        if output.exists():args+=['--verify-only']
        else:
            args+=['--source-root',str(source),'--data-file',str(data),'--public-zip',str(public),'--case-dir',str(case)]
            if not case.exists():
                credentials['token']=private_secret('GITHUB_TOKEN','Token GitHub de lectura (oculto; no se publicará): ')
                credentials['backup_secret']=private_secret('E3_V5_BACKUP_KEY','Clave ORIGINAL de respaldo V5 (oculta; no crear una nueva): ')
        try:
            process=subprocess.run(args,input=json.dumps(credentials),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                env=cpu_env(),timeout=900)
        finally:credentials.clear()
        # The child sanitizes exceptions. Do not display stderr, environment or credentials.
        try:result=json.loads(process.stdout.strip().splitlines()[-1])
        except Exception:result={'status':'STOPPED','diagnostic':'Private subprocess stopped. Original files preserved; no raw stderr displayed.'}
        if process.returncode:result['exit_code']=process.returncode
        return result
    finally:os.umask(old_umask)
