"""Validate synthetic-only implementation evidence and mirror a public allowlist."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/e2_current_synthetic_20261108'
DEST=ROOT/'book_e2/_static/current_implementation_20261108'
FILES=['synthetic_receipt.json','recipes.json','summary.json','source_design.json','README.md']
NAMES=['FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor','TimesNet','TSMixer','TiDE','TimeMixer','PatchTST','iTransformer','CNN-1D + LSTM + atención temporal']
def dump(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate_receipt(s):
 assert s['synthetic_only'] is True and s['status']=='synthetic_mechanism_smoke_only'
 for key in ['market_data_read','heldout_market_data_read','registered_experiment_run','full_current_benchmark_executed','generalization_or_superiority_demonstrated']:assert s[key] is False,key
 assert s['requested_models']==NAMES and s['executed_model_count']==11 and s['skipped_model_count']==0 and s['skipped_models']==[]
 assert [r['model'] for r in s['models']]==NAMES
 for r in s['models']:
  assert 1<=r['supervised_steps']<=3 and r['all_losses_and_predictions_finite'] is True
  assert 'source_roots' not in r['recipe']
  assert r['pretrain_steps']==(s['config']['pretrain_steps'] if r['model']==NAMES[3] else 0)
 for n,h in s['source_sha256'].items():assert sha(ROOT/'scripts'/n)==h,n
 assert s['fold']['training_labels_at_or_before_origin_minus_8'] is True
 assert s['fold']['validation_used_for_optimization_or_selection'] is False
 return s

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
 if a.receipt:
  assert not a.check
  s=validate_receipt(json.loads(a.receipt.read_text()))
  for r in s['models']:r.pop('synthetic_predictions_original_units',None)
  (SOURCE/'synthetic_receipt.json').write_bytes(dump(s))
 s=validate_receipt(json.loads((SOURCE/'synthetic_receipt.json').read_text()))
 recipes={'status':'provisional_synthetic_recipes_not_final_experimental_design','recipes':[{'model':r['model'],'parameters':r['parameter_count'],'recipe':r['recipe'],'architecture':r['architecture']} for r in s['models']]}
 summary={'status':'implemented_and_synthetic_checked_not_market_ready','models_executed':11,'models_skipped':0,'supervised_steps_per_model':s['config']['supervised_steps'],'ae_pretrain_steps':s['config']['pretrain_steps'],'synthetic_days':s['config']['days'],'training_windows':s['config']['train_count'],'validation_windows':s['config']['validation_count'],'market_data_read':False,'market_training_executed':False,'preregistered':False,'superiority_demonstrated':False,'actual_model_parameters':{r['model']:r['parameter_count'] for r in s['models']},'source_sha256':s['source_sha256']}
 for n,x in [('recipes.json',recipes),('summary.json',summary)]:
  if a.check:assert (SOURCE/n).read_bytes()==dump(x),n
  else:(SOURCE/n).write_bytes(dump(x))
 manifest={'schema':'current_synthetic_public_allowlist_v1','files':{n:{'sha256':sha(SOURCE/n),'bytes':(SOURCE/n).stat().st_size} for n in FILES}}
 if a.check:assert (SOURCE/'manifest.json').read_bytes()==dump(manifest)
 else:(SOURCE/'manifest.json').write_bytes(dump(manifest))
 DEST.mkdir(parents=True,exist_ok=True)
 for n in FILES+['manifest.json']:
  if a.check:assert (DEST/n).read_bytes()==(SOURCE/n).read_bytes(),n
  else:(DEST/n).write_bytes((SOURCE/n).read_bytes())
 assert {q.name for q in DEST.iterdir()}==set(FILES+['manifest.json'])
 print('PASS:11 synthetic models,0 skips,source hashes verified,no market runs or final recipe claim.')
if __name__=='__main__':main()
