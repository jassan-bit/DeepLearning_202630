"""Exercise the actual backup/state APIs with an in-memory Git transport."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import e2_score_receipt_compat as c
from e2_reduced_github_backup import GitHubBackup,BackupPause,canonical,digest_file
from e2_reduced_github_state import GitHubState

DISPOSITION = {'allow_download': True, 'final_data_admission_found_before_continuation': False,
    'final_freeze_sha256': 'c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4',
    'neural_refits_requested': False, 'optimizer_steps': None, 'optimizer_steps_unknown': True,
    'original_attempt_preserved': True,
    'original_charge_sha256': '9d9cd441d09b635b7049b3ac118a28ba27ea3d04b3b57adf35380eb38dcd643a',
    'original_charged_wall_seconds': 18.68105469799957,
    'original_failed_files_sha256': {
        'arima_fix_admission.json': '153fccf65a54a9b2d08e66a7329a535ad2f81f8223031b039600ad5e4944813d',
        'execution_extension_admission.json': 'b752d0206243af49d831143d3d2e7ee01e64c1a58331f82a323f8325c4186f1b',
        'scientific_outputs/scoring_started.json': 'd8619e5d62679c846c74d851768d9bea3d8c4e58f20297ee980577d5ae2714e9',
        'worker.log': '6d144ea0abe44bb059d8a8cc4fcc5371e5413b3692c86a9f573140b11e929c48',
        'worker_failure.json': '3f11f5ac3c0dbd2c002ce59c3ddd2222951d3cdcf44f32949a5dad9036e16e5c',
        'worker_spec.json': '7b7a6b1a3d85b0c5c5312481549fe1b41088440b29e9e1054b31e59e1063a717'},
    'original_failed_job_id': 'ff53264094dc1e03fe8586b6aaabeb7b91b951e4ed54b7d9b6e9c020a7c2435a',
    'original_worker_spec_sha256': 'c1226c9eb398b44825fdc927cdf8a8edd904629749992fcdbf5696dbb883cc33',
    'policy_proof': {'policy_sha256': '67fde32352a381b48c75aaad28984f356f998806f5cb61c7c5dc19d6bbfe8935',
        'registration_commit': '24344bd6db20f0546b26167eda4c3076a38c0f4b',
        'registration_file_sha256': '2aa351d17af50022b5002082e36c19ab46339df341e86b6d9f8c31330f0828cf',
        'registration_path': 'experiments/e2_score_source_recovery_20261113/registration.json',
        'source_sha256': '793612f4ca2216b472bd1602b5da13530e1e2bea0b9ec85694de0e3c2179b3ed'},
    'raw_root_unchanged': True, 'scientific_score_repeated': False,
    'status': 'pre_payload_failure_reconciled', 'version': 'e2-score-source-recovery-v1'}


class MemoryBackup(GitHubBackup):
    """Keep actual validation/manifest generation, replacing remote Git only."""
    def __init__(self):
        super().__init__('fixture-secret-that-does-not-occur-in-artifacts')
        self._checked=True;self.remote={};self.commits=0
    def _request(self,*args,**kwargs):
        raise AssertionError('No real network call is allowed')
    def _commit_files(self,files,manifest_path=None,manifest=None):
        exists=manifest_path in self.remote if manifest_path else False
        for path,payload in files.items():
            if path in self.remote and self.remote[path]!=payload:
                raise AssertionError('An immutable remote artifact would change')
        self.remote.update(files);self.commits+=1
        return 'a'*40,exists


class ReceiptCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(prefix='e2-receipt-fixture-');self.addCleanup(temp.cleanup)
        self.output=Path(temp.name)/'campaign';self.directory=self.output/'final/scoring_source_recovery'
        self.directory.mkdir(parents=True)
        self.artifact=self.directory/c.DISPOSITION_FILENAME
        self.raw=canonical(DISPOSITION)+b'\n';self.artifact.write_bytes(self.raw)
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(),c.DISPOSITION_SHA256)
        self.receipt={**deepcopy(DISPOSITION),'attempt_id':c.RECEIPT_ATTEMPT,
            'artifacts_sha256':{c.DISPOSITION_FILENAME:c.DISPOSITION_SHA256}}
        self.commit='b'*40;self.client=MemoryBackup()

    def normalized(self):
        return c.compatible_persist_result(self.client.persist_result,registration_commit=self.commit)

    def test_real_backup_reproduces_the_original_nonterminal_status_failure(self):
        with self.assertRaisesRegex(BackupPause,'terminal_result_receipt_required'):
            self.client.persist_result(self.directory,self.receipt)
        self.assertEqual(self.client.remote,{})
        self.assertEqual(self.artifact.read_bytes(),self.raw)

    def test_actual_backup_accepts_only_transport_normalization_without_false_scientific_completion(self):
        original=deepcopy(self.receipt)
        result=self.normalized()(self.directory,self.receipt)
        self.assertEqual(result['status'],'backed_up')
        manifest=json.loads(self.client.remote[result['manifest_path']])
        self.assertEqual(manifest['fit_status'],'failed')
        prefix='runs/'+c.RECEIPT_ATTEMPT+'/'
        stored=json.loads(self.client.remote[prefix+'receipt.json'])
        self.assertFalse(stored['scientific_result']);self.assertFalse(stored['benchmark_complete'])
        self.assertEqual(stored['kind'],'pre_payload_failure_disposition')
        self.assertEqual(stored['reconciliation_status'],c.DISPOSITION_STATUS)
        self.assertEqual(stored['backup_compat_registration_commit'],self.commit)
        self.assertEqual(self.client.remote[prefix+'artifacts/'+c.DISPOSITION_FILENAME],self.raw)
        self.assertEqual(self.artifact.read_bytes(),self.raw);self.assertEqual(self.receipt,original)
        again=self.normalized()(self.directory,self.receipt)
        self.assertTrue(again['existing_identical_result']);self.assertEqual(self.artifact.read_bytes(),self.raw)

    def test_real_state_persists_metadata_without_pending_cost_or_benchmark_claim(self):
        state=GitHubState(self.output,self.client);state._restored=True
        state.snapshot=Mock(return_value={'state_id':'c'*64})
        callback=c.compatible_persist_result(state.persist_result,registration_commit=self.commit)
        backed=callback(self.directory,self.receipt)
        self.assertEqual(backed['status'],'backed_up');self.assertEqual(state._pending,{})
        self.assertIn('final/scoring_source_recovery',state._backups)
        manifest=json.loads(self.client.remote[backed['manifest_path']])
        self.assertEqual(manifest['fit_status'],'failed')
        self.assertEqual(self.artifact.read_bytes(),self.raw)

    def test_completed_score_receipt_passes_by_identity_and_closes_actual_state_cost_binding(self):
        directory=self.output/'final/score-worker-source-recovery-v1';directory.mkdir()
        predictions=directory/'synthetic_predictions.json';predictions.write_bytes(b'{"synthetic_fixture":true}\n')
        result={'status':'completed','attempt_id':'synthetic-score-result','worker_spec_sha256':'d'*64,
            'artifacts_sha256':{predictions.name:digest_file(predictions)}}
        result_path=directory/'result.json';result_path.write_bytes(canonical(result)+b'\n')
        receipt={'result':result,'result_sha256':digest_file(result_path),'charged_wall_seconds':3.0}
        state=GitHubState(self.output,self.client);state._restored=True
        state._pending={'synthetic-score-result':{'worker_spec_sha256':'d'*64,'result_sha256':digest_file(result_path)}}
        state.snapshot=Mock(return_value={'state_id':'e'*64})
        forwarded=[]
        def persist(path,got):
            forwarded.append(got)
            return state.persist_result(path,got)
        backed=c.compatible_persist_result(persist,registration_commit=self.commit)(directory,receipt)
        self.assertIs(forwarded[0],receipt);self.assertEqual(state._pending,{})
        manifest=json.loads(self.client.remote[backed['manifest_path']])
        self.assertEqual(manifest['fit_status'],'completed')
        stored=json.loads(self.client.remote['runs/synthetic-score-result/receipt.json'])
        self.assertEqual(stored,receipt)
        self.assertNotIn('backup_compat_registration_commit',stored)

    def test_changed_payload_receipt_or_path_is_rejected_before_any_remote_write(self):
        changed=deepcopy(self.receipt);changed['original_charge_sha256']='0'*64
        with self.assertRaises(ValueError):self.normalized()(self.directory,changed)
        self.artifact.write_bytes(self.raw+b' ')
        with self.assertRaises(ValueError):self.normalized()(self.directory,self.receipt)
        self.artifact.write_bytes(self.raw)
        wrong=self.directory.parent/'wrong';wrong.mkdir();(wrong/c.DISPOSITION_FILENAME).write_bytes(self.raw)
        with self.assertRaises(ValueError):self.normalized()(wrong,self.receipt)
        self.assertEqual(self.client.remote,{})

    def test_other_receipts_pass_through_unchanged_and_invalid_registration_is_rejected(self):
        callback=Mock();wrapped=c.compatible_persist_result(callback,registration_commit=self.commit)
        for receipt in [None,{'status':'failed'},{'status':'completed'},{'result':{'status':'completed'}}]:
            wrapped(self.directory,receipt);self.assertIs(callback.call_args.args[1],receipt)
        for invalid in ['',None,'main','x'*40]:
            with self.assertRaises(ValueError):c.compatible_persist_result(callback,registration_commit=invalid)
        with self.assertRaises(TypeError):c.compatible_persist_result(None,registration_commit=self.commit)


if __name__=='__main__':unittest.main()
