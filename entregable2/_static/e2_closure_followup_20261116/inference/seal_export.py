"""Register and execute metadata-only recovery plus a portable numerical QA.

This recovery retains all original numerical outputs. The optional QA repeats
the registered analysis from a local NPZ using the standalone replay script.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
def rec(path):return {'path':str(path),'sha256':sha(path),'bytes':Path(path).stat().st_size}


def prepare(repo,base):
    science=base/'recovery_v3/analysis_plan.json'
    output=base/'recovery_v3/result_v1'
    files={p.name:rec(p) for p in sorted(output.iterdir()) if p.is_file()}
    plan={'version':'inference-export-recovery-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
          'script_sha256':sha(__file__),'scientific_plan':rec(science),
          'registered_analysis_script':rec(base/'recovery_v3/registered_analysis_script.py'),
          'future_runnable_script':rec(repo/'scripts/e2_inference_closure_v2.py'),
          'portable_replay':rec(base/'replay_inference.py'),'existing_numerical_files':files,
          'failed_logs':[rec(p) for p in sorted((base/'run_logs').glob('attempt_v*.log'))],
          'recovery':'Normalize manifest paths to absolute before repository-relative serialization; preserve prior scientific outputs and their hashes.',
          'new_scientific_calculation':False,'QA':'Full5000-draw replay from localNPZ in fresh temporary directory, isolated cwd and noPYTHONPATH; compare both outputCSVs byteforbyte.',
          'original_benchmark_modified':False,'new_fits':0,'new_model_forwards':0}
    write(base/'export_recovery_plan.json',plan)
    print(json.dumps({'export_plan_sha256':sha(base/'export_recovery_plan.json'),'script_sha256':sha(__file__)}))


def run(repo,base,expected):
    planpath=base/'export_recovery_plan.json';require(sha(planpath)==expected,'Wrong recoveryplan')
    plan=json.loads(planpath.read_text());require(plan['script_sha256']==sha(__file__),'Recoverycodechanged')
    for name in ('scientific_plan','registered_analysis_script','future_runnable_script','portable_replay'):
        require(sha(plan[name]['path'])==plan[name]['sha256'],'Recoveryprogram/source changed')
    for record in plan['existing_numerical_files'].values():require(sha(record['path'])==record['sha256'],'Existingnumericaloutputchanged')
    scientific=json.loads(Path(plan['scientific_plan']['path']).read_text())
    require(scientific['script_sha256']==plan['registered_analysis_script']['sha256'],'Archivedscientificcode mismatch')
    original=Path(plan['registered_analysis_script']['path']).read_text()
    fixed=Path(plan['future_runnable_script']['path']).read_text()
    require(fixed==original.replace('path.relative_to(ROOT).as_posix()','path.resolve().relative_to(ROOT).as_posix()'),
            'Recoverymustonlyfixmanifestpathserialization')
    for record in scientific['sources'].values():require(sha(repo/record['path'])==record['sha256'],'Scientificinputchanged')
    models=scientific['models'];seeds=scientific['test_seed_ids'];folds=scientific['fold_origins']
    targets={r['market_day_utc'][:10]:float(r['sigma_7']) for r in csv.DictReader(
        (repo/scientific['sources']['development_targets']['path']).open()) if r['sigma_7']}
    oof_prediction=np.empty((13,3,83));oof_y=np.empty((3,83));oof_dates=[];oof_sha=np.empty((13,3),dtype='U64')
    for mi,model in enumerate(models):
        for fi,fold in enumerate(folds):
            source=scientific['selected_fold_records'][model+':'+fold]
            row=json.loads((repo/source['path']).read_text());dates=[d[:10] for d in row['prediction_dates']]
            oof_prediction[mi,fi]=row['predictions_original'];y=np.array([targets[d] for d in dates])
            if mi==0:oof_dates.append(dates);oof_y[fi]=y
            else:require(dates==oof_dates[fi] and np.array_equal(oof_y[fi],y),'OOFtruth/calendar discrepancy')
            oof_sha[mi,fi]=source['sha256']
    payload=json.loads((repo/scientific['sources']['original_test']['path']).read_text())
    indexed={(r['model'],r['seed']):r for r in payload['predictors']}
    bundle=base/'inference_inputs.npz'
    require(not bundle.exists(),'Inputsareappend-only')
    np.savez_compressed(bundle,models=np.array(models),test_seed_ids=np.array(seeds),
        OOF_fold_origins=np.array(folds),OOF_dates=np.array(oof_dates),OOF_y_original=oof_y,
        OOF_prediction_original=oof_prediction,OOF_source_result_sha256=oof_sha,
        OOF_seed_ids=np.array([[scientific['selected_fold_records'][m+':'+f]['hpo_seed'] for f in folds] for m in models]),
        test_dates=np.array(payload['dates']),test_mask=np.array(payload['mask']),test_y_original=np.array(payload['y_true'],float),
        test_prediction_original=np.array([[indexed[m,s]['prediction_original'] for s in seeds] for m in models],float),
        test_checkpoint_sha256=np.array([[indexed[m,s]['checkpoint_sha256'] for s in seeds] for m in models]))
    write(base/'inference_inputs_manifest.json',{'version':'inference-local-inputs-v1','sha256':sha(bundle),'bytes':bundle.stat().st_size,
        'scientific_plan_sha256':plan['scientific_plan']['sha256'],'models':models,'OOF_shape':[13,3,83],'test_shape':[13,6,608],
        'contains':'Saved forecasts, targets, UTCdays, masks, model/seedidentifiers and originalsourcehashes; no checkpointsorrawminuteCSV.',
        'scope':'Conditional known chronology, selectedOOFfoldpredictors and originalA100testtrajectories; no training or selection replay.'})
    sourceoutput=base/'recovery_v3/result_v1'
    write(sourceoutput/'manifest.json',{'version':'inference-export-recovered-v1',
        'scientific_plan_sha256':plan['scientific_plan']['sha256'],'export_recovery_plan_sha256':expected,
        'files':[{'path':str(p.relative_to(repo)),'sha256':sha(p),'bytes':p.stat().st_size}
                 for p in sorted(sourceoutput.iterdir()) if p.is_file()]})
    with tempfile.TemporaryDirectory(prefix='e2-inference-replay-') as temporary:
        temporary=Path(temporary);output=temporary/'fresh_output'
        environment=os.environ.copy();environment.pop('PYTHONPATH',None)
        command=[sys.executable,str(base/'replay_inference.py'),'--inputs',str(bundle),'--output',str(output)]
        completed=subprocess.run(command,cwd=temporary,env=environment,capture_output=True,text=True,check=True)
        report={'full_bootstrap_replay':True,'outside_repository':True,'PYTHONPATH_removed':True,'numpy':np.__version__,
                'comparisons':[],'new_fits':0,'new_model_forwards':0}
        for filename in ('conditional_OOF_test_comparisons_39.csv','finite_loss_variance_components_15.csv'):
            expectedhash,actualhash=sha(sourceoutput/filename),sha(output/filename)
            require(expectedhash==actualhash,'ReplayCSV differs: '+filename)
            report['comparisons'].append({'file':filename,'source_sha256':expectedhash,'replay_sha256':actualhash,
                                         'byte_identical':True,'maximum_numeric_difference':0.0})
        report['replay_receipt']=json.loads((output/'replay_receipt.json').read_text())
        write(base/'portable_replay_QA.json',report)
        with (base/'run_logs/portable_replay.log').open('x') as stream:stream.write(completed.stdout);stream.write(completed.stderr)
    for record in plan['existing_numerical_files'].values():require(sha(record['path'])==record['sha256'],'Numericaloutputchangedafterrecovery')
    write(base/'metadata_recovery_receipt.json',{'version':'inference-export-recovery-v1','completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'export_recovery_plan_sha256':expected,'numerical_outputs_preserved':True,'original_numerical_analysis_reexecuted':False,
        'portable_numerical_QA_reexecuted':True,'new_fits':0,'new_model_forwards':0,
        'cause':'Existingresultsv3werecomplete; manifestonlyfailedbecauseaCLIrelativepathwasserializedasrepositoryrelativewithoutresolve.',
        'separate_prior_admission_failures':'UTCtimestampversusdatekeys, then historicaldate-onlyformat; all three attemptlogs and budgetcharges retained.'})
    print(json.dumps({'status':'export_and_full_portable_QA_completed','inputs_sha256':sha(bundle),'portable_QA':report}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','run'])
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--plan-sha')
    args=p.parse_args();root=args.repo.resolve();base=Path(__file__).resolve().parent
    if args.operation=='prepare':prepare(root,base)
    else:run(root,base,args.plan_sha)
