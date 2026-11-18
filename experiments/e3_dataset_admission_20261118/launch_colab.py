"""Colab entry point for the next CPU-only admission audit and GitHub snapshot."""
from pathlib import Path
from datetime import datetime, timezone
from getpass import getpass
import importlib.util
import json
import os
import subprocess
import sys
import zipfile

SOURCE = globals().get('E3_ADMISSION_COMMIT','')
if len(SOURCE) != 40 or any(c not in '0123456789abcdef' for c in SOURCE):
    raise ValueError('Set E3_ADMISSION_COMMIT to the published full commit')
BASE = Path('/content') if Path('/content').exists() else Path.cwd()

def token():
    value = os.environ.get('GITHUB_TOKEN','').strip()
    if not value:
        try:
            from google.colab import userdata
            value = (userdata.get('GITHUB_TOKEN') or '').strip()
        except Exception:
            value = ''
    if not value:
        value = globals().get('_E3_GITHUB_TOKEN','').strip()
    if not value:
        value = getpass('GITHUB_TOKEN (Contents: Read and write): ').strip()
    if not value:
        raise RuntimeError('GITHUB_TOKEN is required to persist this audit')
    return value

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result

_E3_GITHUB_TOKEN=token()
context=globals().get('_E3_DATASET_AUDIT_CONTEXT')
if not (context and context['source']==SOURCE and (context['output']/'dataset_audit.json').is_file()
        and context['archive'].is_file()):
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    checkout=BASE/('e3-dataset-source-'+run_id)
    repo='https://github.com/jassan-bit/DeepLearning_202630.git'
    subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout',repo,str(checkout)],check=True)
    def git(*args):
        return subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()
    git('sparse-checkout','init','--cone')
    git('fetch','--depth','1','origin',SOURCE)
    git('checkout','--detach',SOURCE)
    git('sparse-checkout','set','experiments/e3_dataset_admission_20261118','experiments/e3_remediacion_20261118/evaluation','experiments/e3_d01_admission_20261118')
    assert git('rev-parse','HEAD')==SOURCE
    api=module('e3_dataset_git_'+run_id,checkout/'experiments/e3_remediacion_20261118/evaluation/colab_github.py')
    api.validate_credentials(_E3_GITHUB_TOKEN)
    audit=module('e3_dataset_audit_'+run_id,checkout/'experiments/e3_dataset_admission_20261118/audit.py')
    output=BASE/('e3-dataset-admission-'+run_id)
    audit.run(output,SOURCE)
    archive=BASE/('e3-dataset-admission-'+run_id+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.log'}:
                package.write(path,path.relative_to(output))
    context={'source':SOURCE,'run_id':run_id,'api':api,'output':output,'archive':archive}
    _E3_DATASET_AUDIT_CONTEXT=context
else:
    print('Reutilizando la auditoría local; sólo se reintenta el guardado.')
try:
    DATASET_PUBLICATION=context['api'].save_dataset_audit(context['output'],context['archive'],
        _E3_GITHUB_TOKEN,SOURCE,context['run_id'])
except Exception as error:
    if str(error) in {'github_auth_or_contents_write_permission_required','repository_push_access_required'}:
        _E3_GITHUB_TOKEN=''
    print('Conserva la sesión: tras corregir el acceso, reejecutar esta celda reintenta sólo el guardado.')
    raise
print('Auditoría guardada en GitHub:',DATASET_PUBLICATION['run_url'])
current=json.loads((context['output']/'dataset_audit.json').read_text())
print('D01: 3270 filas revisadas. Datasets admitidos:',str(current['datasets_admitted'])+'/3. Sin entrenamiento nuevo.')
