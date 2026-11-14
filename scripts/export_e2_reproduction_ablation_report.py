"""Export recorded final E2 replication/ablation evidence; no fitting or scoring."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,csv,hashlib,json,math,statistics
COMMIT='d604ede6bf0978e14a1b1d7bafaae6250fb8ed07'
STATE='a4dc792a8f77992ca61d52959ce26b4fbe99c83f286475d64e1a902c8ac23c03'
SUMMARY='runs/posthoc-clean-summary-7e0d7830f75f832c'
SUMMARY_SHA='c430ac4ff165370382a98921fb31e7af6dc34c04dfbba64a81b992f95b8ff340'
ORIGINAL_CHARGED=269830.90902790474
SEEDS=set(range(271801,271807))
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def checked(row):
 if sha(canonical({k:v for k,v in row.items()if k!='sha256'}))!=row['sha256']:raise ValueError('Journal row hash differs')
def write_json(path,value):path.write_bytes(canonical(value)+b'\n')
def write_csv(path,rows):
 with path.open('w',newline='',encoding='utf-8')as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
def export(source,destination):
 source,destination=Path(source),Path(destination)
 if destination.exists():raise FileExistsError('Use a new documentary output directory')
 state=read(source/'control/e2-guide-closure-posthoc-v1/head.json')
 assert state['state_id']==STATE and sha(canonical({k:v for k,v in state.items()if k!='state_id'}))==STATE
 summary_raw=(source/SUMMARY/'artifacts/summary.json').read_bytes();assert sha(summary_raw)==SUMMARY_SHA
 summary=json.loads(summary_raw);manifest=read(source/SUMMARY/'manifest.json');receipt=read(source/SUMMARY/'receipt.json')
 assert manifest['receipt_sha256']==sha(canonical(receipt))
 assert receipt['artifacts_sha256']=={'summary.json':SUMMARY_SHA}
 assert manifest['artifacts'][0]['sha256']==SUMMARY_SHA
 assert summary['status']=='completed'and summary['reproduction_passed']and not summary['reproduction_failures']and not summary['unstarted_jobs']
 ledger_path=source/'metadata/budget/posthoc_budget.jsonl';ledger_raw=ledger_path.read_bytes()
 assert sha(ledger_raw)==state['files']['budget/posthoc_budget.jsonl']['sha256']
 journal=[json.loads(line)for line in ledger_raw.splitlines()];active={};charges=0.;previous='0'*64;grants=[]
 completed_charges={}
 for i,row in enumerate(journal):
  checked(row);assert row['sequence']==i and row['previous_sha256']==previous;previous=row['sha256']
  if row['kind']=='reserved':assert not active;active[row['job_id']]=row['seconds']
  elif row['kind']=='charged':
   assert row['job_id']in active;active.pop(row['job_id']);charges+=row['elapsed_seconds']
   if row['status']=='completed'and row.get('evidence',{}).get('result_sha256'):completed_charges[row['job_id']]=row
  elif row['kind']=='overhead':
   assert not active;charges+=row['elapsed_seconds']
   if row.get('description')=='authorized_budget_extension_80_to_84':grants.append(row)
  elif row['kind']!='interrupted':raise ValueError('Unknown journal event')
 assert not active and len(grants)==1 and grants[0]['evidence']['budget_amendment']['added_seconds']==14400
 repro=[];ablations=[];sources=[];environments=[]
 for path in sorted((source/'metadata/reproduction/jobs').glob('*/result.json')):
  relative=path.relative_to(source/'metadata').as_posix();raw=path.read_bytes();result=json.loads(raw)
  assert state['files'][relative]=={'bytes':len(raw),'sha256':sha(raw)}
  assert result['status']=='completed'and result['original_freeze_unchanged']and not result['new_winner_selection_performed']
  job=result['attempt_id'];assert completed_charges[job]['evidence']['result_sha256']==sha(raw)
  spec_path=path.parent/'worker_spec.json';spec=read(spec_path);assert result['worker_spec_sha256']==sha(canonical(spec))
  assert state['files'][spec_path.relative_to(source/'metadata').as_posix()]['sha256']==sha(spec_path.read_bytes())
  binding=state['result_backups'][str(path.parent.relative_to(source/'metadata'))]
  manifest_raw=(source/binding['manifest_path']).read_bytes();assert sha(manifest_raw)==binding['manifest_sha256']
  fit_manifest=json.loads(manifest_raw);assert fit_manifest['receipt_sha256']==sha(canonical(result))
  descriptors={r['path']:r for r in fit_manifest['artifacts']}
  assert all(descriptors[p]['sha256']==h for p,h in result['artifacts_sha256'].items())
  metadata=result['metadata'];assert metadata['fresh_fit']
  item={'model':result['model'],'seed':result['seed'],'condition':result['condition'],'attempt_id':job,
        'rmse_original':result['rmse_original']}
  if result['kind']=='reproduction':
   comparison=metadata['state_comparison']
   item.update(reproduction_passed=metadata['clean_reproduction_passed'],
      max_abs_calibration53=metadata['calibration53_maximum_absolute_difference'],
      max_abs_test608=metadata['test608_maximum_absolute_difference'],
      state_tensor_bitwise_equal=comparison.get('state_tensor_bitwise_equal',''),
      maximum_state_tensor_absolute_difference=comparison.get('maximum_state_tensor_absolute_difference',''),
      classical_parameters_exactly_equal=comparison.get('classical_parameters_exactly_equal',''))
   assert item['reproduction_passed']and item['max_abs_calibration53']==0 and item['max_abs_test608']==0
   repro.append(item);environments.append(result['new_clean_environment'])
  else:
   assert result['kind']=='ablation'and result['model']=='LSTM-autoencoder con predictor'
   architecture=result['fit_result']['architecture'];item.update(parameter_count=architecture['actual_parameter_count'],
       HPO_repeated=False,same_frozen_remaining_phase_epochs=architecture['same_frozen_remaining_phase_epochs'])
   ablations.append(item)
  sources.append({'attempt_id':job,'kind':result['kind'],'model':result['model'],'seed':result['seed'],'condition':result['condition'],
   'result_control_path':relative,'result_bytes':len(raw),'result_sha256':sha(raw),'worker_spec_sha256':sha(spec_path.read_bytes()),
   'canonical_worker_spec_sha256':result['worker_spec_sha256'],'manifest_path':binding['manifest_path'],
   'manifest_sha256':binding['manifest_sha256'],'charged_event_sha256':completed_charges[job]['sha256'],
   'recorded_artifact_count':len(result['artifacts_sha256']),
   'source_receipt_url':'https://github.com/jassan-bit/DeepLearning_202630/blob/'+COMMIT+'/'+binding['manifest_path'].replace('manifest.json','receipt.json')})
 assert len(repro)==81 and len(ablations)==18
 grouped=defaultdict(list)
 for r in repro:grouped[r['model']].append(r)
 assert len(grouped)==16
 models=[]
 for name,rows in sorted(grouped.items()):
  if len(rows)==6:assert {r['seed']for r in rows}==SEEDS
  else:assert len(rows)==1 and rows[0]['seed']=='deterministic'
  models.append({'model':name,'reproductions':len(rows),'all_passed':all(r['reproduction_passed']for r in rows),
     'max_abs_calibration53':max(r['max_abs_calibration53']for r in rows),'max_abs_test608':max(r['max_abs_test608']for r in rows),
     'state_verification':'recorded_tensor_bitwise_equality'if len(rows)==6 else'recorded_classical_parameter_equality',
     'RMSE_mean':statistics.mean(r['rmse_original']for r in rows)})
 full=grouped['LSTM-autoencoder con predictor'];full_mean=statistics.mean(r['rmse_original']for r in full);comparisons=[]
 for row in summary['ablation_comparisons']:
  rows=[r for r in ablations if r['condition']==row['condition']];assert len(rows)==6 and{r['seed']for r in rows}==SEEDS
  mean=statistics.mean(r['rmse_original']for r in rows);delta=mean-full_mean
  assert math.isclose(delta,row['delta_mean_seed_RMSE_ablated_minus_full'],rel_tol=0,abs_tol=1e-15)
  ci=row['CI95_percentile_unadjusted']
  comparisons.append({'condition':row['condition'],'R':6,'rmse_full_mean':full_mean,'rmse_ablated_mean':mean,
      'delta_RMSE_ablated_minus_full':delta,'relative_RMSE_increase_percent':100*delta/full_mean,
      'CI95_low_unadjusted':ci[0],'CI95_high_unadjusted':ci[1],'p_Holm3':row['p_Holm3'],
      'reject_Holm05':row['reject_Holm05'],'bootstrap_draws':row['draws'],'block_days':row['block_days'],
      'method':row['method'],'post_hoc':True})
 assert len(comparisons)==3
 for environment in environments:assert environment==environments[0]
 budget={'approved_total_hours':84,'original_approved_total_hours':80,'original_charged_seconds':ORIGINAL_CHARGED,
    'posthoc_charged_seconds':charges,'total_charged_seconds':ORIGINAL_CHARGED+charges,
    'total_charged_hours':(ORIGINAL_CHARGED+charges)/3600,'remaining_seconds':84*3600-ORIGINAL_CHARGED-charges,
    'unresolved_reservations':active,'journal_records':len(journal),'journal_sha256':sha(ledger_raw),
    'last_record_sha256':previous,'closed_at_utc':journal[-1]['utc'],
    'closed_at_bogota':datetime.fromisoformat(journal[-1]['utc']).astimezone(ZoneInfo('America/Bogota')).isoformat(),
    'budget_reset':False,'scope':'cumulative_charged_wall_budget_not_active_GPU_hours_or_money',
    'amendment':grants[0]['evidence']['budget_amendment']}
 destination.mkdir(parents=True)
 write_csv(destination/'reproduction.csv',repro);write_csv(destination/'reproduction_by_model.csv',models)
 write_csv(destination/'ablation_fits.csv',ablations);write_csv(destination/'ablation_comparisons.csv',comparisons)
 write_json(destination/'budget_final.json',budget)
 for name in('summary.json','manifest.json','receipt.json'):
  source_path=source/SUMMARY/('artifacts/summary.json'if name=='summary.json'else name)
  (destination/('producer_'+name)).write_bytes(source_path.read_bytes())
 base={k:v for k,v in environments[0].items()if not k.startswith('synthetic_acceptance_')}
 write_json(destination/'historical_clean_runtime.json',base)
 status={'version':'e2-final99-documentary-export-v1','status':'completed','planned':99,'completed':99,
    'reproduction_fits':81,'ablation_fits':18,'all_recorded_reproduction_checks_passed':True,
    'max_recorded_calibration53_absolute_difference':0.,'max_recorded_test608_absolute_difference':0.,
    'neural_tensor_comparisons':78,'classical_parameter_comparisons':3,'budget':budget,
    'original_A2_replaced':False,'new_winner_selected':False,'new_blind_test_claimed':False,
    'current_export_performed_training':False,'current_export_performed_inference':False,'current_export_performed_bootstrap':False}
 write_json(destination/'status.json',status)
 write_json(destination/'source_evidence.json',{'results_commit':COMMIT,'control_state_id':STATE,
    'source_summary_sha256':SUMMARY_SHA,'source_manifest_sha256':sha((source/SUMMARY/'manifest.json').read_bytes()),
    'exporter_sha256':sha(Path(__file__).read_bytes()),'source_original_ledger_sha256':summary['original_snapshot']['ledger'],
    'source_original_freeze_sha256':summary['original_snapshot']['freeze'],'final_budget':budget,
    'summary_budget_scope':'producer summary predates final publication charges; budget_final.json is authoritative for this cut',
    'verified_jobs':sources,'checks_performed':['state identity','control result bytes and SHA256','worker-spec canonical binding',
      '99 GitHub manifests and artifact hash inventories','99 result-to-completed-charge bindings','351-row journal chain',
      'producer-summary receipt and artifact binding','recorded 81 numerical reproduction checks','recorded ablation effect arithmetic'],
    'not_performed':['weight downloads or tensor deserialization','new model fitting','new inference','new bootstrap'],
    'state_equality_scope':'Numerical equality was checked by the producing A100 workers; this export audits their bound receipts and accounting.'})
 write_json(destination/'manifest.json',{'version':'e2-final99-documentary-manifest-v1','results_commit':COMMIT,
    'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}for p in sorted(destination.iterdir())if p.is_file()}})
 return status
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
 a=p.parse_args();print(json.dumps(export(a.source,a.destination),ensure_ascii=False,indent=2))
