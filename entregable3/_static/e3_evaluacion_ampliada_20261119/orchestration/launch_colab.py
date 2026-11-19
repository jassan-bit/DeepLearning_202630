"""One Colab entry point for actual fitting, analysis and durable continuation."""
from getpass import getpass
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

SCOPE='experiments/e3_evaluacion_ampliada_20261119'
SPARSE=['src','scripts',SCOPE,'deployment/science',
        'experiments/e3_cierre_integral_20261118/campaign',
        'experiments/e3_cierre_integral_20261118/data',
        'experiments/e3_cierre_integral_20261118/orchestration',
        'experiments/e3_remediacion_20261118/evaluation',
        'experiments/e3_dataset_admission_20261118/inputs',
        'experiments/e3_d01_admission_20261118',
        'experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/verification',
        'experiments/e3_estadistica_posterior_20261117']


def main(namespace):
    source=namespace.get('E3_EXTENDED_SOURCE_COMMIT','')
    if re.fullmatch('[0-9a-f]{40}',source or '') is None:
        raise ValueError('Fija E3_EXTENDED_SOURCE_COMMIT a un SHA completo.')
    token=(os.environ.get('GITHUB_TOKEN') or namespace.get('_E3_GITHUB_TOKEN') or '').strip()
    if not token:
        try:
            from google.colab import userdata
            token=(userdata.get('GITHUB_TOKEN') or '').strip()
        except Exception:
            token=''
    if not token:
        token=getpass('GITHUB_TOKEN (Contents: Read and write para este repositorio): ').strip()
    namespace['_E3_GITHUB_TOKEN']=token
    base=Path('/content') if Path('/content').is_dir() else Path.cwd()
    directory=base/('e3-extended-source-'+source)
    env=dict(os.environ)
    for key in list(env):
        if re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL)(?:_|$)',key.upper()):
            env.pop(key,None)
    env.update(GIT_TERMINAL_PROMPT='0',PYTHONNOUSERSITE='1')
    if not directory.exists():
        subprocess.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout',
                        'https://github.com/jassan-bit/DeepLearning_202630.git',str(directory)],env=env,check=True)
        subprocess.run(['git','sparse-checkout','init','--cone'],cwd=directory,env=env,check=True)
        subprocess.run(['git','sparse-checkout','set',*SPARSE],cwd=directory,env=env,check=True)
        subprocess.run(['git','fetch','--depth','1','origin',source],cwd=directory,env=env,check=True)
        subprocess.run(['git','checkout','--detach',source],cwd=directory,env=env,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=directory,env=env,text=True).strip()
    if actual!=source:
        raise RuntimeError('Checkout parcial conservado; el SHA no coincide.')
    subprocess.run(['git','diff','--exit-code','HEAD'],cwd=directory,env=env,check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    path=directory/SCOPE/'orchestration/controller.py'
    spec=importlib.util.spec_from_file_location('e3_extended_controller',path)
    controller=importlib.util.module_from_spec(spec);sys.modules[spec.name]=controller
    spec.loader.exec_module(controller)
    start_new=namespace.get('E3_EXTENDED_START_NEW',False)
    if type(start_new) is not bool:
        raise ValueError('E3_EXTENDED_START_NEW debe ser booleano.')
    namespace['E3_EXTENDED_START_NEW']=False
    result=controller.run(source,token,base=base,start_new=start_new,
                          master_run_id=namespace.get('E3_EXTENDED_MASTER_RUN'),
                          admission_reuse_url=namespace.get('E3_EXTENDED_ADMISSION_REUSE_URL'))
    namespace['_E3_EXTENDED_AUDIT_RESULT']=result
    return result


if __name__=='__main__':
    main(globals())
