"""Publish aggregate receipts and learning curves for the external paper case.

Reads only local run metadata/curves and hashes; never exports learned weights,
individual forecasts or source CSV observations. Run after the registered fits.
"""
from __future__ import annotations
import argparse,csv,json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--diagnostics-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=ROOT/'book_e2/_static/paper_guide_evidence');a=p.parse_args();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
 registration=load(a.run_dir/'registration.json');design_sha=registration['immutable_inputs']['design_sha256'];runs=[];curves=[]
 for phase,models in [('replication',['TimeMixer']),('pilot',['TimeMixer','iTransformer','PatchTST'])]:
  for model in models:
   for seed in range(2019,2024):
    d=a.run_dir/phase/model/f'seed_{seed}';r=load(d/'completed.json');setup=load(d/'setup.json')
    if r['design_sha256']!=design_sha:raise ValueError('Different design')
    for n,k in [('checkpoint.pth','checkpoint_sha256'),('setup.json','setup_sha256'),('scaler.json','scaler_sha256')]:
     if digest(d/n)!=r[k]:raise ValueError('Changed frozen artifact')
    keep={k:r[k] for k in ['phase','model','seed','completed_at_utc','design_sha256','checkpoint_sha256','setup_sha256','scaler_sha256','best_epoch','epochs_completed','optimizer_steps','training_elapsed_seconds','test_predictions_computed','Bitcoin_model_fits']};keep.update({'parameters':setup['parameters'],'training_windows':setup['n_train_windows'],'training_batches':setup['n_train_batches']});runs.append(keep)
    if phase=='replication':
     keep['learning_curves_sha256_at_publication']=digest(d/'curves.json');curves.append(load(d/'curves.json'))
 score_start=load(a.run_dir/'replication_scores/scoring_started.json');score=load(a.run_dir/'replication_scores/completed.json')
 closed_before=all(datetime.fromisoformat(r['completed_at_utc'])<datetime.fromisoformat(score_start['at_utc']) for r in runs)
 if not closed_before or len(runs)!=20 or len(score['rows'])!=5:raise ValueError('Incomplete chronology')
 diags=[]
 for model in ['TimeMixer','iTransformer','PatchTST']:
  for seed in range(2019,2024):
   d=a.diagnostics_dir/model/f'seed_{seed}';r=load(d/'receipt.json')
   if r['design_sha256']!=design_sha or digest(d/'diagnostics.npz')!=r['diagnostics_sha256']:raise ValueError('Changed diagnostic')
   diags.append({'model':model,'seed':seed,'receipt_sha256':digest(d/'receipt.json'),'diagnostics_sha256':r['diagnostics_sha256']})
 receipt={'schema_version':1,'scope':'External ETTh1 reproduction and a small learned-model pilot; no Bitcoin training','public_design_commit':registration['source_commit'],'registered_at_utc':registration['registered_at_utc'],'design_sha256':design_sha,'primary_replication_fits':5,'pilot_fits':15,'pilot_models':3,'training_seeds':[2019,2020,2021,2022,2023],'all_twenty_checkpoints_closed_before_first_test_scoring':closed_before,'test_scoring_started_at_utc':score_start['at_utc'],'test_scoring_completed_at_utc':score['completed_at_utc'],'test_origins_per_seed':2688,'new_Bitcoin_fits':0,'new_Bitcoin_results':0,'dataset_fits_before_registration':0,'synthetic_profiling_excluded_from_dataset_fit_count':True,'fit_receipts':runs,'pilot_diagnostic_receipts':diags,'learning_curves_hash_scope':'Hashes recorded at aggregate publication; not asserted to be prefit commitments','generator':'scripts/build_e2_paper_execution_evidence.py','generator_sha256':digest(__file__)}
 (out/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
 rows=[]
 for epoch in range(1,11):
  epoch_rows=[next(r for r in c if r['epoch']==epoch) for c in curves]
  tr=np.array([r['train_loss'] for r in epoch_rows]);va=np.array([r['validation_loss'] for r in epoch_rows]);rows.append({'epoch':epoch,'seeds':5,'train_loss_mean':float(tr.mean()),'train_loss_sample_sd':float(tr.std(ddof=1)),'validation_loss_mean':float(va.mean()),'validation_loss_sample_sd':float(va.std(ddof=1))})
 with(out/'training_curve_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import os
 os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-e2-paper-curves');os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-e2-paper-curves')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,3.6));e=np.arange(1,11)
 for prefix,label in [('train','Train'),('validation','Validación')]:
  means=np.array([r[prefix+'_loss_mean'] for r in rows]);sd=np.array([r[prefix+'_loss_sample_sd'] for r in rows]);line=ax.plot(e,means,label=label)[0];ax.fill_between(e,means-sd,means+sd,alpha=.15,color=line.get_color())
 ax.set(xlabel='Época',ylabel='MSE estandarizada',title='TimeMixer: curvas de cinco réplicas, media ± DE');ax.legend();fig.tight_layout();fig.savefig(out/'replication_learning_curves.png',dpi=160);plt.close(fig)
 print(json.dumps({'verified_dataset_fits':20,'verified_pilot_diagnostics':15,'Bitcoin_fits':0,'chronology_passed':True}))
if __name__=='__main__':main()
