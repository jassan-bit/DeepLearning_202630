"""Validate and publish a bounded allowlist; never imports models or reads market."""
import argparse,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/e2_common_synthetic_20261109'
DEST=ROOT/'book_e2/_static/common_executor_20261109'
FILES=('synthetic_receipt.json.gz','events.jsonl.gz','availability.json','summary.json','README.md')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(x):return (json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def canonical(x):return json.dumps(x,sort_keys=True,allow_nan=False).encode()
def require(x,message):
 if not x:raise ValueError(message)
def validate():
 r=json.loads(gzip.decompress((SOURCE/FILES[0]).read_bytes()))
 events=[json.loads(x) for x in gzip.decompress((SOURCE/FILES[1]).read_bytes()).decode().splitlines()]
 require(r['synthetic_only'] and r['system_count']==18 and r['catalogue_count']==11,'Synthetic eighteen-system scope')
 for k in ('market_data_read','market_training_executed','HPO_executed','NAS_executed','pilot_executed','final_benchmark_executed','historical_E1_metrics_reused'):require(r[k] is False,k)
 require(r['total_fit_stages']==36 and r['neural_optimizer_steps']==32,'Fit accounting')
 require(r['verified_author_file_count']==70,'Source subset')
 require(r['selection']['search_executed'] is False,'No search permitted')
 for name,h in r['source_sha256'].items():
  p=ROOT/name if name.startswith('experiments/') else ROOT/'scripts'/name
  require(sha(p.read_bytes())==h,'Source hash differs: '+name)
 previous='0'*64; pending={}; observations=0; stages=set()
 for i,event in enumerate(events):
  e=dict(event);h=e.pop('event_sha256')
  require(e['sequence']==i and e['previous_sha256']==previous,'Journal chain order')
  require(sha(canonical(e))==h,'Journal hash');previous=h
  if e['kind']=='predictions':
   key=(e['stage'],e['index']);require(key not in pending and set(e['systems'])==set(r['systems']),'Complete unique forecasts')
   pending[key]=h
  elif e['kind']=='observation':
   key=(e['stage'],e['index']);require(pending.pop(key,None)==e['prediction_event_sha256'],'Observation without prior predictions');observations+=1
  elif e['kind']=='scores':
   require(not any(k[0]==e['stage'] for k in pending),'Unobserved predictions before scoring');stages.add(e['stage'])
 require(not pending and observations==91 and stages=={'inner','refit'},'Complete stages')
 require(events[-1]['kind']=='run_completed' and previous==r['journal_final_sha256'],'Completion receipt binding')
 rr=dict(r);rr.pop('journal_final_sha256')
 require(sha(canonical(rr))==events[-1]['receipt_payload_sha256'],'Receipt payload binding')
 require(events[-1]['previous_sha256']==r['journal_before_completion_sha256'],'Before completion binding')
 for stage,count in [('inner',83),('refit',8)]:
  s=r['stages'][stage];require(len(s['plan']['evaluate'])==count,'Evaluation length')
  require(s['plan']['train'][-1]==s['plan']['origin']-8,'Embargo')
  require(all(m['n']==count for m in s['metrics_original_units'].values()),'Same scoring support')
 a=json.loads((SOURCE/'availability.json').read_text())
 for k in ('model_fits','optimizer_steps','scaler_fits','forecasts','performance_metrics','HPO_trials','NAS_trials','raw_archives_opened','market_values_exported'):require(a[k]==0,k)
 for name,h in a['source_and_contract_sha256'].items():require(sha((ROOT/name).read_bytes())==h,'Availability source: '+name)
 require(a['final_support']['common_count'] is None and a['final_support']['status']=='not_certified','Final not inspected')
 for name,n in [('CV',305),('historical_E1_pilot',60),('proposed_E2_pilot',53),('development',666)]:require(a['development_support'][name]['common_count']==n,'Availability count '+name)
 for obj in (r,events,a):
  text=json.dumps(obj)
  for private in ('/workspace/','/tmp/','/home/','source_roots'):require(private not in text,'Private path/config')
 return r,a,len(events)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
 r,a,n=validate();summary=json.loads((SOURCE/'summary.json').read_text())
 require(summary['journal_events']==n and summary['functional_tests']==40,'Summary counts')
 for name,h in summary['tests_sha256'].items():require(sha((ROOT/name).read_bytes())==h,'Test hash')
 manifest={'allowlist':list(FILES),'files':{name:{'sha256':sha((SOURCE/name).read_bytes()),'bytes':(SOURCE/name).stat().st_size} for name in FILES}}
 if args.check:require((SOURCE/'manifest.json').read_bytes()==enc(manifest),'Manifest mismatch')
 else:(SOURCE/'manifest.json').write_bytes(enc(manifest))
 DEST.mkdir(parents=True,exist_ok=True)
 for name in (*FILES,'manifest.json'):
  if args.check:require((SOURCE/name).read_bytes()==(DEST/name).read_bytes(),'Mirror mismatch')
  else:(DEST/name).write_bytes((SOURCE/name).read_bytes())
 require({p.name for p in DEST.iterdir()}==set((*FILES,'manifest.json')),'Unexpected public artifact')
 print('PASS: 18 synthetic systems, 36 fit stages, 91 chronological forecast days, real development support only')
if __name__=='__main__':main()
