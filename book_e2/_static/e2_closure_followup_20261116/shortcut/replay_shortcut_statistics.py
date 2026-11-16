"""Replay the saved-input diagnostic using local NPZ only; needs NumPy, no Torch."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import platform
import numpy as np


def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=Path(__file__).resolve().parent/'study_v1')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();base=args.study
    plan=json.loads((base/'analysis_plan.json').read_text())
    report=json.loads((base/'inference/inference_report.json').read_text())
    receipt=json.loads((base/'statistics/execution_receipt.json').read_text())
    if file_sha(base/'analysis_plan.json')!=report['plan_sha256'] or report['plan_sha256']!=receipt['plan_sha256']:raise ValueError('Plan identity differs')
    npz=base/'inference/predictions.npz';csv_path=base/'statistics/input_dependency_by_model.csv'
    if file_sha(npz)!=report['files']['predictions.npz']['sha256']:raise ValueError('Saved predictions changed')
    if file_sha(csv_path)!=receipt['files']['input_dependency_by_model.csv']['sha256']:raise ValueError('Reference statistics changed')
    with np.load(npz,allow_pickle=False) as f:
        pred=f['predictions'].copy();y=f['y_true'].copy();models=f['models'].tolist()
    if pred.shape!=(3,13,6,608):raise ValueError('Wrong prediction geometry')
    n=len(y);spec=plan['statistics'];draws=spec['draws'];block=spec['block_days']
    rng=np.random.default_rng(spec['rng_seed']);loss=(pred-y)**2
    rmse=np.sqrt(loss.mean(axis=3)).mean(axis=2)
    response=np.abs(pred[1:]-pred[0]).mean(axis=2);risks=[];changes=[]
    for start in range(0,draws,256):
        count=min(256,draws-start)
        starts=rng.integers(0,n-block+1,size=(count,math.ceil(n/block)))
        indices=(starts[:,:,None]+np.arange(block)).reshape(count,-1)[:,:n]
        weights=np.zeros((count,n),dtype=np.float64)
        np.add.at(weights,(np.arange(count)[:,None],indices),1)
        q=np.sqrt(np.einsum('bt,cmst->bcms',weights,loss,optimize=True)/n).mean(axis=3)
        risks.append(q-q[:,0,None,:]);changes.append(np.einsum('bt,cmt->bcm',weights,response,optimize=True)/n)
    risk=np.concatenate(risks);change=np.concatenate(changes)
    actual=[]
    for c,name in enumerate(['volume_training_donor','OHLC_training_donor'],1):
        for m,model in enumerate(models):
            a=np.quantile(risk[:,c,m],[.025,.975]);b=np.quantile(change[:,c-1,m],[.025,.975])
            actual.append({'model':model,'condition':name,'original_CPU_RMSE_mean6':float(rmse[0,m]),'intervention_CPU_RMSE_mean6':float(rmse[c,m]),
                           'RMSE_change':float(rmse[c,m]-rmse[0,m]),'RMSE_change_CI95_low':float(a[0]),'RMSE_change_CI95_high':float(a[1]),
                           'mean_absolute_prediction_change':float(response[c-1,m].mean()),'prediction_change_CI95_low':float(b[0]),'prediction_change_CI95_high':float(b[1])})
    with csv_path.open(newline='',encoding='utf-8') as f:expected=list(csv.DictReader(f))
    if len(expected)!=26:raise ValueError('Wrong reference row count')
    maxdiff=0.;comparisons=0
    for a,b in zip(actual,expected):
        if a['model']!=b['model'] or a['condition']!=b['condition']:raise ValueError('Reference row identity differs')
        for key in a:
            if key in ['model','condition']:continue
            delta=abs(a[key]-float(b[key]));maxdiff=max(maxdiff,delta);comparisons+=1
    if maxdiff>1e-12:raise ValueError('Saved-statistics arithmetic differs')
    result={'status':'passed_saved_predictions_reanalysis','models':13,'conditions_with_contrasts':2,'dates':608,'fixed_seed_trajectories':78,
            'rows_compared':26,'numeric_comparisons':comparisons,'max_absolute_difference':maxdiff,'new_fits':0,'new_model_forwards':0,
            'python':platform.python_version(),'numpy':np.__version__,'script_sha256':file_sha(__file__),
            'scope':'Arithmetic replay of saved intervention predictions; not reproduction of model training or new input forwards.'}
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
