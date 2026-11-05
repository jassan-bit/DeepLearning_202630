"""Unpublished, fixed-recipe LSTM/GRU/TCN comparison on legacy Filipe CV.

All writes remain in the sibling local experiment. The historical loader,
preprocessing, fold validation, optimization, inference and artifact writer are
reused without changing their source. No reserved-test source is admitted.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path('/workspace/DeepLearning_202630').resolve()
EXPERIMENT = Path(__file__).resolve().parents[1]
OUT = EXPERIMENT / 'results'
PROTOCOL = EXPERIMENT / 'protocol.json'
ALLOWED_RESULT_READS = {
    (ROOT / name).resolve() for name in (
        'results/eda_development/daily_market_development.csv',
        'results/eda_development/daily_volatility_development.csv',
        'results/dl_r2_cv_20261104/folds.json',
        'results/dl_r2_cv_20261104/runs.csv',
        'results/dl_r2_cv_20261104/predictions.csv',
        'results/dl_r2_cv_20261104/fold_metrics.csv',
        'results/filipe_multihistory_20261104/runs.csv',
        'results/filipe_multihistory_20261104/predictions.csv',
        'results/filipe_multihistory_20261104/fold_metrics.csv',
        *(f'results/filipe_multihistory_20261104/D{d}/folds.json' for d in (14,21,28)),
    )
}
def project_guard(event, args):
    if event == 'open' and args and isinstance(args[0], (str,bytes,os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT):
            mode = args[1] if len(args)>1 else None
            flags = args[2] if len(args)>2 and isinstance(args[2],int) else 0
            if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                raise PermissionError(f'Local comparison forbids repository writes: {path}')
            if path.is_relative_to(ROOT/'data') or (path.is_relative_to(ROOT/'results') and path not in ALLOWED_RESULT_READS):
                raise PermissionError(f'Only explicit development inputs may be opened: {path}')
    elif event in ('os.remove','os.rmdir','os.mkdir','os.rename','os.link','os.symlink'):
        for arg in args[:2] if event in ('os.rename','os.link','os.symlink') else args[:1]:
            if isinstance(arg,(str,bytes,os.PathLike)):
                path=Path(os.fsdecode(arg)).resolve()
                if path.is_relative_to(ROOT):
                    raise PermissionError(f'Local comparison forbids repository mutation: {path}')
sys.addaudithook(project_guard)
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(EXPERIMENT/'implementation'))
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,multiprocessing as mp,time
import train_filipe_multihistory as historical
import train_dl_benchmark as fitting
import local_models

def configure():
    historical.ROOT=ROOT
    historical.OUT=OUT
    historical.PROTOCOL=PROTOCOL
    historical.NEW_HISTORIES=(7,14,21,28)
    historical.architecture_spec=local_models.architecture_spec
    local_models.install_into(fitting)

def initialize_worker(plan,manifests,context):
    configure()
    historical.init_worker(plan,manifests,context)

def check_freeze():
    freeze=json.loads((EXPERIMENT/'freeze_before_training.json').read_text())
    for path,sha in freeze['frozen_sha256'].items():
        if fitting.digest(path)!=sha:
            raise RuntimeError('Frozen input/code changed: '+path)
    if fitting.digest(PROTOCOL)!=freeze['protocol_sha256']:
        raise RuntimeError('Local protocol changed after freeze')
    return freeze

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['validate','run'],default='validate')
    parser.add_argument('--limit-trajectories',type=int)
    args=parser.parse_args()
    configure()
    freeze=check_freeze()
    plan,manifests,context=historical.register()
    if args.phase=='validate':
        print(json.dumps({'status':'validated','models':list(plan['models']),
            'histories':plan['histories'],'planned_trajectories':120,'planned_fits':2880,
            'project_writes_allowed':False,'reserved_test_access_allowed':False,
            'code_signature':context['code_signature']}),flush=True)
        return
    jobs=[{'variant':'relative_window','model':model,'seed':seed,'history_days':history,
           'width':config['width'],'learning_rate':config['learning_rate']}
          for history in plan['histories'] for seed in plan['seeds']
          for model,config in plan['models'].items()]
    if args.limit_trajectories is not None:
        if not 1<=args.limit_trajectories<=len(jobs):raise ValueError('Invalid trajectory limit')
        jobs=jobs[:args.limit_trajectories]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=plan['workers'],mp_context=mp.get_context('spawn'),
            initializer=initialize_worker,initargs=(plan,manifests,context)) as executor:
        futures={executor.submit(historical.run_trajectory,job):job for job in jobs}
        for count,future in enumerate(as_completed(futures),1):
            record=future.result()
            message={'at_utc':datetime.now(timezone.utc).isoformat(),
                'completed_in_invocation':count,'submitted':len(jobs),
                'history_days':record['history_days'],'model':record['model'],'seed':record['seed'],
                'rmse':record['rmse'],'r2':record['r2'],'training_seconds':record['training_seconds'],
                'invocation_wall_seconds':time.perf_counter()-started}
            with (EXPERIMENT/'progress.jsonl').open('a') as f:
                f.write(json.dumps(message)+'\n')
            print(json.dumps(message),flush=True)
    complete=list((OUT/'runs').glob('*/summary.json'))
    if len(complete)==120:
        historical.aggregate(plan,context)
    elif args.limit_trajectories is None:
        raise RuntimeError(f'Incomplete planned comparison: {len(complete)}/120')
    else:
        print(json.dumps({'status':'partial technical run; no aggregate selection',
            'completed_trajectories':len(complete),'planned':120}),flush=True)
    check_freeze()

if __name__=='__main__':main()
