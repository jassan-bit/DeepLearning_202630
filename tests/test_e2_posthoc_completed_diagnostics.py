import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_completed_diagnostics as m

class CompletedDiagnosticsTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  base=Path(self.temp.name);self.output=base/'posthoc';self.original=base/'original';self.extension=base/'extension'
  self.rows=[{'model':'Network'+str(i),'seed':seed,'checkpoint_sha256':m.sha(f'{i}-{seed}'.encode())}
             for i in range(13)for seed in range(271801,271807)]
  self.rows.append({'model':'ARIMA','seed':'deterministic','checkpoint_sha256':'a'*64})
  freeze={'models':self.rows+[{'model':'Persistence','seed':'deterministic'},{'model':'SeasonalNaive','seed':'deterministic'}]}
  raw=self.write(self.original/'final/final_freeze.json',freeze);self.freeze_sha=m.sha(raw)
  self.sources={}
  for name in m.SOURCES:
   p=self.extension/'scripts'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(name.encode());self.sources[name]=m.sha(p.read_bytes())
  config={'version':m.VERSION,'engine_commit':m.ENGINE,'configuration_commit':m.CONFIGURATION,
          'freeze_sha256':self.freeze_sha,'diagnostic_files_sha256':self.sources}
  self.run_id='posthoc-'+m.sha(m.canonical(config))[:24]
  self.write(self.output/'robustness/registration.json',config)
  summary={'version':m.VERSION,'run_id':self.run_id,'predictor_records':79,'models_neural':13,'neural_predictors':78,
           'all_baseline_projections_passed':True,'profiles':[{}]*78,'robustness':[{}]*468,
           'interpretability':[{'model':name}for name in ['LSTM-autoencoder con predictor','GRU32','ARIMA']]}
  self.write(self.output/'robustness/summary.json',summary)
  (self.output/'robustness/robustness.csv').write_bytes(b'model,effect\nfixture,0\n')
  self.folders=[]
  for row in self.rows:
   key=hashlib.sha256((row['model']+str(row['seed'])).encode()).hexdigest()[:24]
   folder=self.output/'robustness/predictors'/key;self.folders.append(folder)
   self.write(folder/'started.json',{'run_id':self.run_id,'model':row['model'],'seed':row['seed']})
   self.write(folder/'diagnostics.json',{**row,'run_id':self.run_id,'training_performed':False,
       'baseline_projection_passed':True,'n_dates':608,'baseline_predictions':[.1]*608})
   self.receipt(folder,('started.json','diagnostics.json'),self.run_id+'-'+key)
  self.receipt(self.output/'robustness',('registration.json','summary.json','robustness.csv'),self.run_id)
  self.root_sha=m.sha((self.output/'robustness/result.json').read_bytes())
  self.state={'version':'e2-guide-closure-posthoc-state-v1','scientific_commit':m.ENGINE,'output_path':str(self.output),
              'files':{},'result_backups':{}}
  for folder in [self.output/'robustness',*self.folders]:
   rel=folder.relative_to(self.output).as_posix();raw=(folder/'result.json').read_bytes()
   self.state['files'][rel+'/result.json']={'bytes':len(raw),'sha256':m.sha(raw)};self.state['result_backups'][rel]={}
  self.restate()
 def write(self,path,value):
  path.parent.mkdir(parents=True,exist_ok=True);raw=m.canonical(value)+b'\n';path.write_bytes(raw);return raw
 def receipt(self,folder,claims,attempt):
  return self.write(folder/'result.json',{'version':m.VERSION,'run_id':self.run_id,'attempt_id':attempt,
      'status':'completed','training_performed':False,'artifacts_sha256':{name:m.sha((folder/name).read_bytes())for name in claims}})
 def restate(self):
  self.state['state_id']=m.sha(m.canonical({k:v for k,v in self.state.items()if k!='state_id'}))
 def verify(self):
  with patch.multiple(m,RUN_ID=self.run_id,ROOT_RESULT_SHA=self.root_sha,FREEZE_SHA=self.freeze_sha,SOURCES=self.sources):
   return m.verify_completed(self.output,self.extension,self.state,original_output=self.original)
 def rebind_root(self):
  raw=self.receipt(self.output/'robustness',('registration.json','summary.json','robustness.csv'),self.run_id)
  self.root_sha=m.sha(raw);self.state['files']['robustness/result.json']={'bytes':len(raw),'sha256':self.root_sha};self.restate()
 def test_all_79_completed_are_verified_without_changing_any_files(self):
  before={str(p):m.sha(p.read_bytes())for p in Path(self.temp.name).rglob('*')if p.is_file()}
  result=self.verify();self.assertEqual(result['completed_predictors'],79);self.assertEqual(result['robustness_rows'],468)
  self.assertFalse(result['model_inference_performed']);self.assertFalse(result['ledger_modified'])
  self.assertEqual(before,{str(p):m.sha(p.read_bytes())for p in Path(self.temp.name).rglob('*')if p.is_file()})
 def test_missing_predictor_result_cannot_skip_diagnostics(self):
  (self.folders[-1]/'result.json').unlink()
  with self.assertRaisesRegex(ValueError,'Missing completed'):self.verify()
 def test_corrupted_predictor_payload_is_rejected(self):
  (self.folders[0]/'diagnostics.json').write_bytes(b'{}')
  with self.assertRaisesRegex(ValueError,'checksum differs'):self.verify()
 def test_remote_receipt_binding_is_required(self):
  key=self.folders[0].relative_to(self.output).as_posix()+'/result.json'
  self.state['files'][key]['sha256']='0'*64;self.restate()
  with self.assertRaisesRegex(ValueError,'Remote control does not bind'):self.verify()
 def test_completion_without_durable_backup_binding_is_rejected(self):
  self.state['result_backups'].pop(self.folders[0].relative_to(self.output).as_posix());self.restate()
  with self.assertRaisesRegex(ValueError,'backup is absent'):self.verify()
 def test_changed_original_diagnostic_source_is_rejected(self):
  next((self.extension/'scripts').iterdir()).write_bytes(b'altered')
  with self.assertRaisesRegex(ValueError,'source changed'):self.verify()
 def test_changed_freeze_is_rejected(self):
  (self.original/'final/final_freeze.json').write_bytes(b'{}')
  with self.assertRaisesRegex(ValueError,'catalogue changed'):self.verify()
 def test_incomplete_aggregate_is_rejected_even_when_new_checksums_match(self):
  path=self.output/'robustness/summary.json';v=json.loads(path.read_bytes());v['predictor_records']=78
  self.write(path,v);self.rebind_root()
  with self.assertRaisesRegex(ValueError,'incomplete diagnostic population'):self.verify()
 def test_extra_remote_predictor_cannot_be_silently_ignored(self):
  self.state['files']['robustness/predictors/extra/result.json']={'bytes':0,'sha256':'0'*64};self.restate()
  with self.assertRaisesRegex(ValueError,'catalogue differs'):self.verify()
 def test_symlink_payload_is_rejected(self):
  original=self.folders[0]/'diagnostics.json';saved=original.with_name('saved.json');original.rename(saved);original.symlink_to(saved)
  with self.assertRaisesRegex(ValueError,'regular'):self.verify()
 def test_unchecked_control_identity_is_rejected_before_reading_artifacts(self):
  self.state['files'].clear()
  with self.assertRaisesRegex(ValueError,'control identity'):self.verify()
if __name__=='__main__':unittest.main()
