"""Registered fresh-seed D7 study conditional on known 2024, never a new market test.

Registration and validation do not construct models. Variance extension and final
confirmation are separate signed phases; the original experiments remain intact.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import multiprocessing as mp
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import run_filipe_pilot as pilot

core, fitting = pilot.core, pilot.fitting
ROOT = Path(__file__).resolve().parents[1]
STUDY = 'e1_filipe_confirmatory_d7_20261105'
OUT = ROOT / 'results' / STUDY
PROTOCOL = ROOT / 'experiments' / STUDY / 'protocol.json'
CONTEXT = OUT / 'code_manifest_before_fitting.json'
LOCK = OUT / 'locked_design.json'
EXTENSION_SEEDS = list(range(6301, 6341, 2))
FINAL_START = 50000001
FINAL_STEP = 2
ANALYSIS = {'bootstrap_replicates':4999,'bootstrap_seed':20261105,
    'primary_test':'Two-sided paired Student t on annual per-seed RMSE; Holm correction for exactly three neural pairs',
    'confidence_intervals':'95% marginal and simultaneous Bonferroni 1-alpha/3 family intervals',
    'effect_size':'Paired Cohen d_z with 4999 paired-seed bootstrap replicates as sensitivity',
    'normality':'Shapiro-Wilk and QQ diagnostics only; never switches the primary test',
    'interpretation':'Rejecting zero is distinct from exceeding practical delta; compare family intervals with +/-delta',
    'advanced_models_trained':False}
FOLD_SOURCE = 'results/dl_r2_cv_20261104/folds.json'
DECISION = 'references/literature_e1/selection_20261105/decision.json'
CODE = ['scripts/run_filipe_confirmatory_d7.py', 'scripts/analyze_filipe_confirmatory_d7.py'] + pilot.CODE
PILOT_RECORDS = ['results/e1_filipe_pilot_20261105/'+name for name in
                 ['protocol.json','execution_summary.json','runs.csv']]
OLD_RECORDS = ['experiments/filipe_multihistory_20261104/protocol.json',
               'experiments/filipe_multihistory_20261104/freeze.json',
               'results/filipe_multihistory_20261104/test/execution_summary.json',
               'results/filipe_multihistory_20261104/test/raw_test_audit.json',
               'results/filipe_multihistory_20261104/test/analysis/metrics_summary.csv']
stamp, digest, read, write = pilot.stamp, pilot.digest, pilot.read, pilot.write
_PHASE_OUT = _PHASE = _PHASE_CONTEXT = None


def moment(value):
    value = datetime.fromisoformat(value.replace('Z','+00:00'))
    if value.tzinfo is None:
        raise ValueError('Chronology requires timezone-aware UTC timestamps')
    return value


def previous_seeds():
    seeds = set()
    for name in ['results/dl_r2_cv_20261104/runs.csv',
                 'results/filipe_multihistory_20261104/runs.csv',
                 'results/e1_filipe_pilot_20261105/runs.csv']:
        seeds.update(pd.read_csv(ROOT / name, usecols=['seed']).seed.astype(int))
    return seeds


def seed_namespace(seeds):
    # Odd base seeds and even +17 batch seeds occupy distinct role states.
    return {int(s)+month*100000+offset for s in seeds for month in range(12)
            for offset in (0, 17, 10000000, 10000017)}


def validate_plan(plan):
    old = read(ROOT / OLD_RECORDS[0])
    pairs = [{'history_days': 7, 'model_a': a, 'model_b': b}
             for a, b in combinations(('MLP', 'LSTM', 'CNN1D'), 2)]
    dim = plan['dimensioning']
    if (plan['study_id'] != STUDY or plan['histories'] != [7]
            or plan['variance_extension_seeds'] != EXTENSION_SEEDS or plan['original_pilot_seeds'] != list(range(6201,6206))
            or list(plan['models']) != ['MLP', 'LSTM', 'CNN1D']
            or plan['models'] != old['models'] or plan['training'] != old['training']
            or plan['workers'] != 4 or plan['variant'] != 'relative_window'
            or plan['target_return_days'] != 7 or plan['forecast_horizon_days'] != 1
            or plan['primary_pairs'] != pairs or plan['test_files_opened_allowed'] != 0
            or plan['raw_archives_opened_allowed'] != 0
            or plan['definitive_seed_start'] != FINAL_START or plan['definitive_seed_step'] != FINAL_STEP
            or dim['primary_family_size'] != 3 or dim['alpha_global'] != .05
            or dim['alpha_each'] != .05/3 or dim['power_target'] != .8
            or dim['execution_floor'] != 30 or dim['variance_extension_n'] != 20
            or dim['sd_upper_confidence'] != .95 or dim['maximum_planning_R'] != 10000
            or dim['sd_source'] != '20 new extension seeds only'
            or plan['analysis'] != ANALYSIS or plan['maximum_definitive_seeds'] != 300
            or not np.isclose(dim['delta_absolute'], .05*dim['reference_persistence_rmse'], rtol=1e-15)):
        raise ValueError('Registered D7 recipe, family, sample size or data boundary changed')
    prior = seed_namespace(previous_seeds())
    if seed_namespace(EXTENSION_SEEDS).intersection(prior):
        raise ValueError('Extension seed namespace must be fresh across initialization/refit/minibatch roles')
    final = [FINAL_START+FINAL_STEP*i for i in range(10000)]
    for values in [EXTENSION_SEEDS,final]:
        if len(seed_namespace(values)) != len(values)*12*4:
            raise ValueError('Each initialization/refit/minibatch seed must be unique within a phase')
    if seed_namespace(final).intersection(prior | seed_namespace(EXTENSION_SEEDS)):
        raise ValueError('Definitive seed namespace must exclude every historical and pilot namespace')
    required = set(pilot.SOURCE_PATHS) | {FOLD_SOURCE, DECISION} | set(OLD_RECORDS+PILOT_RECORDS)
    if set(plan['source_sha256']) != required or set(plan['code_sha256']) != set(CODE):
        raise ValueError('Missing or additional registered source guards')


def validate_inputs(plan):
    validate_plan(plan)
    for name, expected in {**plan['source_sha256'], **plan['code_sha256']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Registered source changed: '+name)
    data = pilot.load_development(ROOT)
    manifest = read(ROOT / FOLD_SOURCE)
    if (manifest['history_days'] != 7 or manifest['target_returns_days'] != 7
            or manifest['forecast_days'] != 1 or manifest['monthly_fold_count'] != 12
            or manifest['library']['version'] != '0.1.5'):
        raise ValueError('Wrong frozen D7 fold manifest')
    outcomes = [i for fold in manifest['folds'] for i in core.validate_fold(data, 7, fold).tolist()]
    expected = np.flatnonzero(data['calendar'].year == 2024).tolist()
    if outcomes != expected or len(set(outcomes)) != 366:
        raise ValueError('Exactly one outcome per chronological 2024 day is required')
    indices = np.asarray(expected)
    anchor = float(np.sqrt(np.mean((data['targets'][indices]-data['persistence'][indices])**2)))
    if not np.isclose(anchor, plan['dimensioning']['reference_persistence_rmse'], rtol=1e-12):
        raise ValueError('Persistence anchor differs from the registered known cohort')
    return {7: manifest}


def make_plan():
    old = read(ROOT / OLD_RECORDS[0]); freeze = read(ROOT / OLD_RECORDS[1])
    anchor = freeze['selection']['persistence_cv_rmse']
    sources = list(pilot.SOURCE_PATHS)+[FOLD_SOURCE, DECISION]+OLD_RECORDS+PILOT_RECORDS
    return {'study_id': STUDY, 'registered_at_utc': stamp(), 'histories': [7],
        'variance_extension_seeds': EXTENSION_SEEDS, 'original_pilot_seeds':list(range(6201,6206)),
        'definitive_seed_start': FINAL_START,
        'definitive_seed_step': FINAL_STEP,
        'models': old['models'], 'training': old['training'], 'workers': 4,
        'variant': 'relative_window', 'target_return_days': 7, 'forecast_horizon_days': 1,
        'primary_metric': 'Annual pooled RMSE per independent training-seed trajectory, original sigma7 units',
        'primary_pairs': [{'history_days': 7, 'model_a': a, 'model_b': b}
                          for a,b in combinations(old['models'],2)],
        'scope': 'Prospective RNG confirmation conditional on known development2024; not independent temporal validation',
        'monthly_folds': 12, 'embargo_days': 7, 'HPO': 'None; three original recipes fixed before new runs',
        'seed_rule': 'base+100000*monthordinal; refit adds10000000; minibatch adds17; paired across families',
        'seed_role_safety': 'Odd base seeds in steps2; +17 batch states even; all month/stage/role states distinct and fresh across studies',
        'test_files_opened_allowed': 0, 'raw_archives_opened_allowed': 0,
        'prior_exposure': '2024 and original test already known; old6201..6205 contextual and pooled25 sensitivity only',
        'advanced_models_trained': False, 'unassigned_2026_09_10': 'Not assigned or opened',
        'dimensioning': {'variance_extension_n':20, 'sd_source':'20 new extension seeds only',
            'primary_family_size':3, 'alpha_global':.05, 'alpha_each':.05/3,
            'correction':'Bonferroni planning; two-sided paired t with Holm3 final reporting',
            'power_target':.8, 'power_scope':'Marginal design power under normal independent seed differences',
            'delta_absolute':.05*anchor, 'reference_persistence_rmse':anchor,
            'hypothesis':'Detect an absolute delta difference against zero, not superiority above a delta boundary',
            'sd_upper_confidence':.95, 'upper_sd_scope':'One-sided marginal chi-square bound, not simultaneous95%',
            'execution_floor':30, 'maximum_planning_R':10000,
            'algorithm':'Common maximum exact two-sided noncentral-t R using each n20 upperSD95; no silent cap',
            'pooled25_sensitivity':'Prior5 plus new20 reported separately; never determines definitive R',
            'independence':'Independent training randomizations conditional on one known market trajectory'},
        'analysis': dict(ANALYSIS),
        'maximum_definitive_seeds':300,
        'phases': {'variance_extension':{'trajectories':60,'monthly_segments':720,'project_fits':1440},
                   'definitive':'R new seeds per model, excluding all variance/pilot trajectories'},
        'publication': 'Versioned protocol/code/literature decision pushed before extension; separate R lock pushed before definitive',
        'publication_scope':'Aggregate metadata public; predictions, weights, scalers and fold copies private',
        'source_sha256':{name:digest(ROOT/name) for name in sources},
        'code_sha256':{name:digest(ROOT/name) for name in CODE}}


def register():
    if PROTOCOL.exists() or OUT.exists():
        raise FileExistsError('Preserve prior study; registration requires unused paths')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit implementation and literature decision before registration')
    tracked = set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    if not (set(CODE)|{DECISION}).issubset(tracked):
        raise ValueError('Registered implementation and literature decision must be versioned')
    plan = make_plan(); validate_inputs(plan)
    runtime = fitting.runtime(); runtime['timing_scope'] = 'Four concurrent CPU trajectories; not serial hardware benchmark'
    write(PROTOCOL,plan)
    signature = hashlib.sha256(json.dumps({'study':STUDY,'protocol':digest(PROTOCOL),
        'code':plan['code_sha256'],'versions':runtime['versions']},sort_keys=True).encode()).hexdigest()
    context = {'registered_at_utc':stamp(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'protocol_sha256':digest(PROTOCOL),'code_sha256':plan['code_sha256'],
        'code_signature':signature,'runtime':runtime,'test_files_opened':0}
    write(CONTEXT,context); write(OUT/'protocol.json',plan); write(OUT/'runtime.json',runtime)
    print(json.dumps({'registered':True,'protocol_sha256':digest(PROTOCOL),'fits_executed':0}),flush=True)


def validated_registration():
    plan,context = read(PROTOCOL),read(CONTEXT)
    if digest(PROTOCOL) != context['protocol_sha256'] or read(OUT/'protocol.json') != plan:
        raise ValueError('Registered protocol changed')
    if fitting.runtime()['versions'] != context['runtime']['versions']:
        raise ValueError('Registered runtime versions changed')
    expected_signature = hashlib.sha256(json.dumps({'study':STUDY,'protocol':digest(PROTOCOL),
        'code':plan['code_sha256'],'versions':context['runtime']['versions']},sort_keys=True).encode()).hexdigest()
    if context['code_sha256'] != plan['code_sha256'] or context['code_signature'] != expected_signature:
        raise ValueError('Registered execution context changed')
    manifests = validate_inputs(plan)
    return plan,manifests,context


def validate_lock(plan,context):
    lock = read(LOCK); n = lock.get('common_R')
    if (type(n) is not int or not plan['dimensioning']['execution_floor'] <= n <= 10000
            or lock.get('registered_protocol_sha256') != context['protocol_sha256']
            or lock.get('analysis_code_sha256') != plan['code_sha256']['scripts/analyze_filipe_confirmatory_d7.py']):
        raise ValueError('Definitive R lock is invalid or belongs to changed code/protocol')
    for key,name in [('variance_extension_runs_sha256','runs.csv'),('variance_extension_summary_sha256','execution_summary.json')]:
        if lock.get(key) != digest(OUT/'variance_extension'/name):
            raise ValueError('R lock inputs changed: '+name)
    if moment(lock['locked_at_utc']) <= moment(read(OUT/'variance_extension/execution_summary.json')['completed_at_utc']):
        raise ValueError('R lock must follow completed variance extension')
    expected_inputs = {str(PROTOCOL.relative_to(ROOT)), PILOT_RECORDS[-1],
        str((OUT/'variance_extension/runs.csv').relative_to(ROOT)),
        str((OUT/'variance_extension/execution_summary.json').relative_to(ROOT))}
    if set(lock.get('input_sha256',{})) != expected_inputs:
        raise ValueError('Missing or additional R lock input guards')
    for name,expected in lock['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('R lock input changed: '+name)
    from analyze_filipe_confirmatory_d7 import verify_dimension_lock
    verify_dimension_lock(pd.read_csv(ROOT/PILOT_RECORDS[-1]),
                          pd.read_csv(OUT/'variance_extension/runs.csv'),plan,lock)
    return lock


def publication_receipt(phase,context):
    path = OUT/('preregistration_git_receipt.json' if phase == 'variance_extension' else 'locked_design_publication_receipt.json')
    receipt = read(path)
    if (receipt.get('registered_protocol_sha256') != context['protocol_sha256']
            or receipt.get('publication_completed_before_runs') is not True):
        raise ValueError('Missing publication-before-runs evidence')
    commit = receipt['published_commit']
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','merge-base','--is-ancestor',context['source_commit'],commit],cwd=ROOT,check=True,capture_output=True)
    for file in [PROTOCOL]+[ROOT/name for name in CODE+[DECISION]]+([LOCK] if phase == 'definitive' else []):
        committed = subprocess.check_output(['git','show',commit+':'+str(file.relative_to(ROOT))],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest() != digest(file):
            raise ValueError('Published revision differs from registered protocol/design')
    if phase == 'definitive' and receipt.get('locked_design_sha256') != digest(LOCK):
        raise ValueError('Published lock hash changed')
    if moment(receipt['published_at_utc']) >= moment(stamp()):
        raise ValueError('Publication receipt must precede launch')
    lower = read(LOCK)['locked_at_utc'] if phase == 'definitive' else context['registered_at_utc']
    if moment(receipt['published_at_utc']) < moment(lower):
        raise ValueError('Publication receipt predates registration or locked design')
    return receipt


def jobs_for(plan,seeds):
    return [{'variant':'relative_window','model':model,'seed':seed,'history_days':7,
             'width':cfg['width'],'learning_rate':cfg['learning_rate']}
            for model,cfg in plan['models'].items() for seed in seeds]


def worker_init(plan,manifests,context,phase):
    global _PHASE_OUT,_PHASE,_PHASE_CONTEXT
    _PHASE_OUT,_PHASE,_PHASE_CONTEXT = OUT/phase,phase,context
    core.OUT = _PHASE_OUT  # Only in new spawned workers, never the parent/old files.
    core.init_worker(plan,manifests,context)


def execute(job):
    receipt = _PHASE_OUT/'trajectory_start_receipts'/f"D7_{job['model']}_s{job['seed']}.json"
    if receipt.exists():
        previous = read(receipt)
        if previous['job'] != job or previous['code_signature'] != _PHASE_CONTEXT['code_signature']:
            raise ValueError('Changed trajectory-start receipt')
    else:
        receipt.parent.mkdir(parents=True,exist_ok=True)
        with receipt.open('x') as stream:
            json.dump({'started_at_utc':stamp(),'job':job,'phase':_PHASE,
                       'code_signature':_PHASE_CONTEXT['code_signature'],
                       'protocol_sha256':_PHASE_CONTEXT['protocol_sha256']},stream,indent=2)
    return core.run_trajectory(job)


def aggregate(plan,context,phase,seeds,publication):
    target=OUT/phase; jobs=jobs_for(plan,seeds); records=[]; forecasts=[]; folds=[]; starts=[]
    expected_dirs={f"relative_window_{j['model']}_D7_s{j['seed']}_{context['code_signature'][:10]}" for j in jobs}
    expected_receipts={f"D7_{j['model']}_s{j['seed']}.json" for j in jobs}
    if {p.name for p in (target/'trajectory_start_receipts').glob('*.json')} != expected_receipts:
        raise ValueError('Missing or unexpected trajectory-start receipts')
    if {p.name for p in (target/'runs').iterdir() if p.is_dir()} != expected_dirs:
        raise ValueError('Missing or unexpected trajectory directories')
    for job in jobs:
        name=f"relative_window_{job['model']}_D7_s{job['seed']}_{context['code_signature'][:10]}"; directory=target/'runs'/name
        record=core.verify_completion(directory,context['code_signature'],'summary.json')
        if record is None or any(record[k] != v for k,v in job.items()) or record['n_dates'] != 366 or record['folds_fitted'] != 12 or record['model_fits'] != 24 or record['run_id'] != name:
            raise ValueError('Incomplete or mismatched signed trajectory')
        receipt=read(target/'trajectory_start_receipts'/f"D7_{job['model']}_s{job['seed']}.json")
        if receipt['job'] != job or receipt['code_signature'] != context['code_signature'] or receipt['protocol_sha256'] != context['protocol_sha256'] or receipt['phase'] != phase:
            raise ValueError('Changed trajectory receipt')
        completions=list(directory.glob('*/completion.json'))
        if len(completions) != 12:
            raise ValueError('Exactly twelve signed monthly completions required')
        for monthly in completions:
            core.verify_completion(monthly.parent,context['code_signature'],'completion.json')
        earliest_fit=min(moment(read(p)['completed_at_utc']) for p in completions)
        if not moment(context['registered_at_utc']) < moment(receipt['started_at_utc']) < earliest_fit or moment(publication['published_at_utc']) >= moment(receipt['started_at_utc']):
            raise ValueError('Publication/registration must precede every new trajectory and fit')
        prediction=pd.read_csv(directory/'predictions.csv')
        monthly_metrics=pd.read_csv(directory/'fold_metrics.csv')
        if len(monthly_metrics) != 12 or monthly_metrics.fold_id.nunique() != 12:
            raise ValueError('Exactly twelve monthly metric records required')
        if prediction.outcome_day.tolist() != pd.date_range('2024-01-01','2024-12-31').strftime('%Y-%m-%d').tolist() or prediction.duplicated(['outcome_day']).any():
            raise ValueError('Trajectory prediction dates differ from frozen 366-day cohort')
        records.append(record);forecasts.append(prediction);folds.append(monthly_metrics);starts.append(receipt['started_at_utc'])
    frame=pd.DataFrame(records).sort_values(['model','seed']); merged=pd.concat(forecasts,ignore_index=True); fold=pd.concat(folds,ignore_index=True)
    for name,data in [('runs.csv',frame),('predictions.csv',merged),('fold_metrics.csv',fold)]:
        payload=data.to_csv(index=False,lineterminator='\n').encode(); path=target/name
        if path.exists() and path.read_bytes() != payload: raise ValueError('Refuse changed completed aggregate: '+name)
        if not path.exists(): path.write_bytes(payload)
    summary={'study_id':STUDY,'phase':phase,'completed_at_utc':stamp(),'first_new_trajectory_started_at_utc':min(starts),
        'registered_protocol_sha256':context['protocol_sha256'],'protocol_sha256':context['protocol_sha256'],
        'code_signature':context['code_signature'],'trajectories':len(jobs),'seeds_per_model':len(seeds),
        'seeds':seeds,'monthly_segments':len(jobs)*12,'project_fits':len(jobs)*24,
        'unique_validation_predictions':len(merged),'inference_forward_batches':len(jobs)*12*25,
        'dates_per_trajectory':366,'workers':4,'test_files_opened':0,'raw_archives_opened':0,
        'publication_precedes_every_trajectory':True,'all_fits_are_new':True,
        'locked_design_sha256':digest(LOCK) if phase=='definitive' else None,
        'output_sha256':{name:digest(target/name) for name in ['runs.csv','predictions.csv','fold_metrics.csv']}}
    path=target/'execution_summary.json'
    if path.exists():
        previous=read(path)
        if {k:v for k,v in previous.items() if k!='completed_at_utc'} != {k:v for k,v in summary.items() if k!='completed_at_utc'}:
            raise ValueError('Refuse changed completed execution summary')
        return previous
    write(path,summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['register','validate','variance_extension','definitive'],default='validate')
    parser.add_argument('--max-definitive-R',type=int,default=300,help='Explicit compute authorization; never truncates locked R')
    args=parser.parse_args()
    if args.phase=='register': register();return
    plan,manifests,context=validated_registration()
    if args.phase=='validate':
        print(json.dumps({'status':'validated','folds':12,'validation_dates':366,'test_files_opened':0,'fits_executed':0}));return
    phase=args.phase; lock=validate_lock(plan,context) if phase=='definitive' else None
    if lock and lock['common_R'] > args.max_definitive_R:
        raise ValueError('Locked R exceeds explicit compute budget; no runs launched and no cap applied')
    seeds=[FINAL_START+FINAL_STEP*i for i in range(lock['common_R'])] if lock else EXTENSION_SEEDS
    publication=publication_receipt(phase,context); target=OUT/phase; target.mkdir(parents=True,exist_ok=True)
    folds=target/'D7/folds.json';folds.parent.mkdir(parents=True,exist_ok=True)
    if folds.exists() and digest(folds)!=plan['source_sha256'][FOLD_SOURCE]: raise ValueError('Copied D7 folds changed')
    if not folds.exists(): folds.write_bytes((ROOT/FOLD_SOURCE).read_bytes())
    marker=target/'run_started.json';marker_value={'protocol_sha256':context['protocol_sha256'],'phase':phase,'locked_design_sha256':digest(LOCK) if lock else None}
    if marker.exists():
        if any(read(marker).get(k)!=v for k,v in marker_value.items()): raise ValueError('Changed phase start marker')
    else: write(marker,{**marker_value,'started_at_utc':stamp()})
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(plan,manifests,context,phase)) as pool:
        futures=[pool.submit(execute,job) for job in jobs_for(plan,seeds)]
        for count,future in enumerate(as_completed(futures),1):
            result=future.result(); print(json.dumps({'phase':phase,'completed':count,'total':len(futures),'model':result['model'],'seed':result['seed'],'rmse':result['rmse']}),flush=True)
    print(json.dumps(aggregate(plan,context,phase,seeds,publication)),flush=True)


if __name__=='__main__': main()
