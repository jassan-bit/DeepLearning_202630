"""Acquisition and immutable-score fixtures; no market payload or model fit."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import e2_score_source_recovery as c
import e2_reduced_final as final
import e2_reduced_final_data as data
from e2_reduced_supervision import supervise_command


class SourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='e2-score-source-fixture-')
        self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.output=self.root/'campaign/final';self.old=self.output/'score-worker'
        self.old.mkdir(parents=True)
        self.budget=SimpleNamespace(root=self.output.parent/'shared_budget',stage_wall_seconds=1800,
            reserve=Mock(),completed_receipt=Mock())
        self.budget.root.mkdir();self.ledger=self.budget.root/'pilot_budget.jsonl'
        self.source={'source_commit':c.ENGINE_COMMIT,'files_sha256':{'fixture.py':'a'*64}}
        self.runtime={'device_name':'synthetic-A100'};self.sources={'TimesNet':'fixture-author-source'}
        self.args=(self.root/'engine',self.root/'data',self.output,self.budget)
        self.kwargs={'source_identity':self.source,'runtime_identity':self.runtime,
            'source_roots':self.sources,'raw_root':self.output.parent/'private_raw_E1','allow_download':True}
        files={};rows=[]
        schedule=[(m,s)for m in data.NEURAL for s in data.SEEDS]+[(m,'deterministic')for m in data.CLASSICAL]
        for index,(name,seed)in enumerate(schedule):
            row={'model':name,'seed':seed}
            for field in ['checkpoint','scaler','calibrator']:
                relative=f'frozen/{index}/{field}.fixture'
                path=self.output/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(f'{index}:{field}'.encode());checksum=hashlib.sha256(path.read_bytes()).hexdigest()
                files[relative]=checksum;row[field]=relative;row[field+'_sha256']=checksum
            rows.append(row)
        self.freeze={'protocol_id':data.PROTOCOL_ID,'protocol_sha256':data.PROTOCOL_SHA256,
            'systems':list(data.SYSTEMS),'final_seed_ids':list(data.SEEDS),'models':rows,
            'frozen_files_sha256':files,'pilot_power_MDE_closed':True,'analysis_frozen':True,
            'analysis_registration_sha256':'b'*64}
        freeze_raw=c.canonical(self.freeze)+b'\n';self.freeze_path=self.output/'final_freeze.json'
        self.freeze_path.write_bytes(freeze_raw)
        freeze_file=hashlib.sha256(freeze_raw).hexdigest();freeze_digest=c.digest(self.freeze)
        base=final._base_spec(self.args[1],self.source,self.runtime,self.sources,'cuda:0')
        spec={**base,'kind':'final-score','model':'all16','project_root':str(self.args[0].resolve()),
            'final_output':str(self.output),'final_freeze_sha256':freeze_file,'daily_csv':None,
            'raw_root':str(self.kwargs['raw_root']),'allow_download':True,
            'output':str(self.old),'fit_wall_seconds':1800}
        spec_sha=c.digest(spec);spec['worker_spec_sha256']=spec_sha
        job=c.digest({'version':final.VERSION,'spec_sha256':spec_sha})
        failure={'status':'failed','error_type':'HTTPError','reason':'HTTP Error 404: Not Found','automatic_retry':False}
        marker={'freeze_sha256':freeze_digest,'analysis_registration_sha256':'b'*64,'retune_allowed':False}
        payloads={'worker_spec.json':c.canonical(spec)+b'\n','worker_failure.json':c.canonical(failure)+b'\n',
            'scientific_outputs/scoring_started.json':c.canonical(marker)+b'\n',
            'worker.log':b'synthetic HTTP404\n','arima_fix_admission.json':b'{}\n',
            'execution_extension_admission.json':b'{}\n'}
        for name,raw in payloads.items():
            p=self.old/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(raw)
        reserve={'sequence':0,'previous_sha256':'0'*64,'kind':'reserved','attempt_id':job,
            'reservation_seconds':1860,'metadata':{'worker_spec_sha256':spec_sha}}
        reserve['sha256']=c.digest(reserve)
        charge={'sequence':1,'previous_sha256':reserve['sha256'],'kind':'charged','attempt_id':job,
            'elapsed_seconds':c.FAILED_SECONDS,'reservation_seconds':1860,'status':'failed',
            'reservation_overrun':False,'resource_use':{'worker_spec_sha256':spec_sha,
                'failure':'worker_exit_2','optimizer_steps_unknown':True,'automatic_retry':False}}
        charge['sha256']=c.digest(charge)
        self.ledger.write_bytes(c.canonical(reserve)+b'\n'+c.canonical(charge)+b'\n')
        self.payloads=payloads
        for name,value in [('FAILED_SPEC',spec_sha),('FAILED_JOB',job),('FAILED_CHARGE',charge['sha256']),
                ('FREEZE_FILE',freeze_file),('FREEZE_DIGEST',freeze_digest),
                ('OLD_FILES',{p:hashlib.sha256(raw).hexdigest()for p,raw in payloads.items()})]:
            patched=patch.object(c,name,value);patched.start();self.addCleanup(patched.stop)
        self.registration=self.root/'registration';self.registration.mkdir()
        subprocess.run(['git','init','-q',str(self.registration)],check=True)
        for relative,raw in [(c.SOURCE_PATH,(ROOT/c.SOURCE_PATH).read_bytes()),
                (c.POLICY_PATH,c.canonical(c.policy_template())+b'\n')]:
            p=self.registration/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        hashes={p:hashlib.sha256((self.registration/p).read_bytes()).hexdigest()for p in(c.SOURCE_PATH,c.POLICY_PATH)}
        record={'protocol_id':'e2-reduced-delivery-v1','engine_commit':c.ENGINE_COMMIT,
            'prior_arima_commit':c.PRIOR_ARIMA_COMMIT,'funding_continuation_commit':c.FUNDING_COMMIT,
            'files_sha256':hashes,'policy_sha256':c.policy_template()['sha256'],
            'policy_file_sha256':hashes[c.POLICY_PATH]}
        (self.registration/'registration.json').write_bytes(c.canonical(record)+b'\n')
        p=self.registration/c.REGISTRATION_PATH;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(c.canonical(record)+b'\n')
        subprocess.run(['git','-C',str(self.registration),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.registration),'-c','user.name=Fixture',
            '-c','user.email=fixture@example.invalid','commit','-qm','Synthetic scoring acquisition fixture'],check=True)
        commit=subprocess.check_output(['git','-C',str(self.registration),'rev-parse','HEAD'],text=True).strip()
        self.run=c.make_run_final_scoring(final,registration_commit=commit,
            registration_root=self.registration,engine_root=self.registration)

    def test_acquisition_continuation_isolated_and_old_attempt_never_changed(self):
        before={p:p.read_bytes()for p in(self.ledger,self.freeze_path,*[self.old/n for n in self.payloads])}
        supervisor=Mock();persist=Mock()
        with patch.object(final,'_dispatch',return_value={'status':'completed'})as dispatch:
            result=self.run(*self.args,**self.kwargs,persist_result=persist,supervisor=supervisor)
        spec,directory,budget=dispatch.call_args.args
        self.assertEqual(result['status'],'completed');self.assertIs(budget,self.budget)
        self.assertEqual(directory,self.output/c.RECOVERY_DIRECTORY)
        self.assertEqual(spec['daily_csv'],str(self.output/c.ABSENT_DAILY))
        self.assertEqual(spec['raw_root'],str(self.kwargs['raw_root']))
        self.assertTrue(spec['allow_download']);self.assertEqual(spec['final_freeze_sha256'],c.FREEZE_FILE)
        self.assertIs(dispatch.call_args.kwargs['supervisor'],supervisor)
        self.assertFalse((self.output/c.ABSENT_DAILY).exists())
        self.assertFalse((self.old/'result.json').exists());self.budget.reserve.assert_not_called()
        for p,raw in before.items():self.assertEqual(p.read_bytes(),raw)
        receipt=persist.call_args.args[1]
        self.assertTrue(receipt['optimizer_steps_unknown']);self.assertIsNone(receipt['optimizer_steps'])
        self.assertFalse(receipt['scientific_score_repeated'])

    def test_all81_frozen_artifacts_verified_before_dispatch(self):
        checksum_path=self.output/next(iter(self.freeze['frozen_files_sha256']))
        checksum_path.write_bytes(b'changed-frozen-weight')
        with patch.object(final,'_dispatch')as dispatch:
            with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs)
            dispatch.assert_not_called()

    def test_any_old_partial_scientific_artifact_blocks_continuation(self):
        for name in ['scientific_outputs/predictions_x.json','result.json','scientific_outputs/data_admission.json']:
            p=self.old/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'{}')
            with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs)
            p.unlink()

    def test_any_local_first_access_cache_or_partial_decode_blocks(self):
        private=self.output/'private-known-E1-data';private.mkdir()
        for name in ['current_E2_first_known_E1_access.json','known_E1_data_receipt.json','known_E1_daily.csv']:
            p=private/name;p.write_bytes(b'synthetic')
            with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs)
            p.unlink()
        private.rmdir()
        sentinel=self.output/c.ABSENT_DAILY;sentinel.parent.mkdir();sentinel.write_bytes(b'not-absent')
        with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs)

    def test_source_identity_or_consent_change_is_rejected(self):
        for changes in [{'allow_download':False},{'raw_root':self.root/'other'},
                {'daily_csv':self.root/'other.csv'},{'source_roots':{}},
                {'runtime_identity':{'device_name':'changed'}}]:
            with self.assertRaises(ValueError):self.run(*self.args,**dict(self.kwargs,**changes))

    def test_failed_budget_charge_and_metadata_cannot_be_rewritten(self):
        before=self.ledger.read_bytes();rows=[json.loads(r)for r in before.splitlines()]
        rows[-1]['resource_use']['optimizer_steps_unknown']=False
        rows[-1]['sha256']=c.digest({k:v for k,v in rows[-1].items()if k!='sha256'})
        self.ledger.write_bytes(b''.join(c.canonical(r)+b'\n'for r in rows))
        with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs)
        self.ledger.write_bytes(before)
        (self.old/'worker_failure.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs)

    def test_partial_new_worker_is_preserved_and_not_automatically_retried(self):
        recovery=self.output/c.RECOVERY_DIRECTORY;recovery.mkdir();p=recovery/'worker_failure.json'
        p.write_bytes(b'new-partial')
        with patch.object(final,'_dispatch')as dispatch:
            with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs)
            dispatch.assert_not_called()
        self.assertEqual(p.read_bytes(),b'new-partial')

    def test_completed_recovery_reuses_verified_cost_and_artifacts_without_rereading_private_data(self):
        with patch.object(final,'_dispatch')as dispatch:self.run(*self.args,**self.kwargs)
        spec,directory,budget=dispatch.call_args.args
        directory.mkdir();spec={**spec,'output':str(directory),'fit_wall_seconds':1800}
        spec_sha=c.digest(spec);spec['worker_spec_sha256']=spec_sha
        final.write_json(directory/'worker_spec.json',spec)
        artifact=directory/'synthetic_predictions.json';artifact.write_bytes(b'fixture-result')
        result={'status':'completed','worker_spec_sha256':spec_sha,
            'artifacts_sha256':{artifact.name:hashlib.sha256(artifact.read_bytes()).hexdigest()}}
        final.write_json(directory/'result.json',result)
        private=self.output/'private-known-E1-data';private.mkdir()
        (private/'known_E1_daily.csv').write_bytes(b'private data must not be opened again')
        self.budget.completed_receipt.return_value={'elapsed_seconds':12.5}
        returned=self.run(*self.args,**self.kwargs,supervisor=supervise_command)
        self.assertEqual(returned,result);self.budget.reserve.assert_not_called()
        expected_use=self.budget.completed_receipt.call_args.kwargs['expected_resource_use']
        self.assertEqual(expected_use['worker_spec_sha256'],spec_sha)
        artifact.write_bytes(b'changed-result')
        with self.assertRaises(ValueError):self.run(*self.args,**self.kwargs,supervisor=supervise_command)


if __name__=='__main__':unittest.main()
