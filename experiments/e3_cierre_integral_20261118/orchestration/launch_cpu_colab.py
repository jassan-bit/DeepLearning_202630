"""Colab entry point: pinned checkout, CPU evidence and exact new-branch storage."""
from datetime import datetime, timezone
from getpass import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


RELATIVE = 'experiments/e3_cierre_integral_20261118'
SPARSE = [RELATIVE, 'experiments/e3_remediacion_20261118/evaluation',
          'experiments/e3_dataset_admission_20261118/inputs',
          'experiments/e3_d01_admission_20261118',
          'experiments/e3_bloques_fg_20261117', 'artifacts/e3_dashboard_v1',
          'artifacts/e3_serving_calibration_v2', 'artifacts/e3_serving_extensions_v2',
          'data/e3_dashboard_v1','assets']


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def main(namespace):
    context=namespace.get('_E3_CLOSURE_CPU_CONTEXT')
    if context:
        source=context['source']
    else:
        source=namespace.get('E3_CLOSURE_SOURCE_COMMIT','')
        if re.fullmatch(r'[0-9a-f]{40}',source or '') is None:
            raise ValueError('Fija E3_CLOSURE_SOURCE_COMMIT a un SHA completo antes del launcher.')
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        base=Path('/content') if Path('/content').is_dir() else Path.cwd()
        context={'source':source,'run_id':run_id,'checkout':str(base/('e3-closure-source-'+run_id)),
                 'output':str(base/('e3-closure-cpu-'+run_id))}
        namespace['_E3_CLOSURE_CPU_CONTEXT']=context
    checkout=Path(context['checkout'])
    if not checkout.is_dir():
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout',
                        'https://github.com/jassan-bit/DeepLearning_202630.git',str(checkout)],check=True)
        subprocess.run(['git','-C',str(checkout),'sparse-checkout','init','--cone'],check=True)
        subprocess.run(['git','-C',str(checkout),'sparse-checkout','set',*SPARSE],check=True)
        subprocess.run(['git','-C',str(checkout),'fetch','--depth','1','origin',source],check=True)
        subprocess.run(['git','-C',str(checkout),'checkout','--detach',source],check=True)
    head=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if head!=source:
        raise RuntimeError('El checkout parcial se conserva; no coincide con el SHA fijado.')
    print('Código fijado:',source,flush=True)
    token=(os.environ.get('GITHUB_TOKEN') or namespace.get('_E3_GITHUB_TOKEN')
           or namespace.get('_E3_R1_GITHUB_TOKEN') or '').strip()
    if not token:
        try:
            from google.colab import userdata
            token=(userdata.get('GITHUB_TOKEN') or '').strip()
        except Exception:
            token=''
    if not token:
        token=getpass('GITHUB_TOKEN (Contents: Read and write para este repositorio): ').strip()
    api=load(checkout/'experiments/e3_remediacion_20261118/evaluation/colab_github.py','e3_closure_github_api')
    api.validate_credentials(token)
    namespace['_E3_GITHUB_TOKEN']=token
    print('Acceso comprobado. La escritura requiere Contents: Read and write.',flush=True)
    runner=load(checkout/RELATIVE/'orchestration/cpu_stage.py','e3_closure_cpu_stage')
    receipt=runner.run(checkout,context['output'],source)
    storage=load(checkout/RELATIVE/'orchestration/github_store.py','e3_closure_github_store')
    saved=storage.save(context['output'],token,source,context['run_id'],api,profile='cpu')
    print('Evidencia guardada en GitHub:',saved['run_url'],flush=True)
    print('Commit:',saved['commit_sha'],flush=True)
    print('Datasets admitidos:',str(receipt['datasets_admitted'])+'/3',flush=True)
    print('Aceptación pública Render:',receipt['public_deployment_smoke_passed'],flush=True)
    print('R1 y R3 continúan sin acreditar por esta etapa de CPU.',flush=True)
    namespace['_E3_CLOSURE_CPU_SAVED']=saved
    return saved


if __name__=='__main__':
    main(globals())
