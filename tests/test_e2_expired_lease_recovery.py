"""Pure CPU/GitHub-memory recovery tests; never train or contact a service."""
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_expired_lease_recovery as R


class MemoryGitHub:
    max_conflicts=3
    _token='synthetic-token-only'
    def __init__(self,remote,archive_path,archive):
        self.head='a'*40
        self.files={R.GitHubState.__module__:b'unused',
            'control/github-primary-v1/head.json':R.canonical(remote)+b'\n',archive_path:archive}
        self.blobs={};self.trees={};self.commits={};self.writes=[]
        self.fail_patch_once=False;self.late_after_reads=None;self.head_reads=0
    def _branch_head(self):
        self.head_reads+=1
        if self.late_after_reads==self.head_reads:
            self.files['runs/'+R.KNOWN_ATTEMPT+'/manifest.json']=b'{"late":true}\n'
        return self.head
    def _content(self,path,head):
        value=self.files.get(path)
        if value is None:return None
        return {'content':base64.b64encode(value).decode(),'encoding':'base64',
            'sha':R.git_blob_sha(value)}
    def _request(self,method,path,payload=None,**unused):
        if method=='GET' and path.startswith('/git/commits/'):
            return {'tree':{'sha':'b'*40}}
        if method=='GET' and path.startswith('/git/blobs/'):
            return {'content':base64.b64encode(self.blobs[path.rsplit('/',1)[1]]).decode()}
        self.writes.append((method,path))
        if (method,path)==('POST','/git/blobs'):
            raw=base64.b64decode(payload['content']);sha=R.git_blob_sha(raw);self.blobs[sha]=raw
            return {'sha':sha}
        if (method,path)==('POST','/git/trees'):
            sha=hashlib.sha1(R.canonical(payload)).hexdigest();self.trees[sha]=payload
            return {'sha':sha}
        if (method,path)==('POST','/git/commits'):
            sha=hashlib.sha1(R.canonical(payload)).hexdigest();self.commits[sha]=payload
            return {'sha':sha}
        if method=='PATCH' and path.startswith('/git/refs/heads/'):
            if self.fail_patch_once:
                self.fail_patch_once=False
                raise R.BackupPause('synthetic_lost_transport_response')
            assert payload['force'] is False
            commit=self.commits[payload['sha']]
            assert commit['parents']==[self.head]
            for entry in self.trees[commit['tree']]['tree']:
                self.files[entry['path']]=self.blobs[entry['sha']]
            self.head=payload['sha'];return {'object':{'sha':self.head}}
        raise AssertionError((method,path))


class ExpiredRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'output';self.proc=self.root/'proc';self.proc.mkdir()
        self.now=datetime(2026,11,13,3,0,tzinfo=timezone.utc)
        self.job={'version':'e2-reduced-selection-executor-v1','stage':'hpo',
            'model_id':'TSMixer','fold_origin':'2024-07-01','configuration_id':'C02',
            'seed':271201,'parent_trial_id':'TSM:hpo-007','configuration_sha256':'5'*64,'trial':{}}
        self.job['logical_job_id']=R.digest_bytes(R.canonical(self.job))
        self.spec={'version':'e2-reduced-selection-executor-v1','job':self.job,
            'output':str(self.output/'jobs'/self.job['logical_job_id']),
            'source_sha256':{'scripts/frozen.py':'f'*64},'fit_wall_seconds':1800}
        spec_sha=R.digest_bytes(R.canonical(self.spec))
        self.rows=[]
        for i in range(234):
            self.add_row(kind='overhead',description='synthetic_old_cost',elapsed_seconds=1.0)
        reservation=self.add_row(kind='reserved',attempt_id=self.job['logical_job_id'],
            reservation_seconds=1860.0,effective_total_hours=72,
            funding_policy_sha256=R.FUNDING_POLICY,funding_registration_commit=R.FUNDING,
            metadata={'resource_registration_commit':R.RESOURCE,
                'resource_policy_sha256':R.RESOURCE_POLICY,'worker_spec_sha256':spec_sha})
        self.metadata={R.LEDGER:b''.join(R.canonical(r)+b'\n'for r in self.rows),
            'jobs/'+self.job['logical_job_id']+'/worker_spec.json':R.canonical(self.spec)+b'\n'}
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,raw in self.metadata.items():z.writestr(name,raw)
        archive=stream.getvalue();sha=R.digest_bytes(archive);archive_path='control/github-primary-v1/deltas/'+sha+'.zip'
        body={'version':'e2-github-primary-state-v1','scientific_commit':'1e8745497b484518f766b6da5c19be8362d9add7',
            'output_path':str(self.output),'previous_state_id':'0'*64,
            'reason':'lease_reserved_before_worker',
            'files':{n:{'sha256':R.digest_bytes(b),'bytes':len(b)}for n,b in self.metadata.items()},
            'metadata_deltas':[{'path':archive_path,'sha256':sha,'bytes':len(archive),'git_blob_sha':R.git_blob_sha(archive)}],
            'result_backups':{'jobs/old-'+str(i):{'immutable':str(i)}for i in range(97)},
            'persistent_destination':'GitHub','Google_Drive_used':False}
        self.remote={**body,'state_id':R.digest_bytes(R.canonical(body))}
        self.client=MemoryGitHub(self.remote,archive_path,archive)
        for key,value in [('KNOWN_ATTEMPT',self.job['logical_job_id']),('KNOWN_SPEC',spec_sha),
                ('KNOWN_TIP',reservation['sha256']),('KNOWN_STATE',self.remote['state_id'])]:
            p=patch.object(R,key,value);p.start();self.addCleanup(p.stop)
        self.policy=R.policy_template();self.commit='c'*40
        self.proof={'registration_commit':self.commit,'registration_path':R.REGISTRATION,
            'registration_file_sha256':'8'*64,'source_sha256':R.digest_file(R.__file__),
            'policy_sha256':self.policy['sha256'],'registered_at_utc':'2026-11-13T02:40:00+00:00'}
        p=patch.object(R,'verify_registration',return_value=self.proof);p.start();self.addCleanup(p.stop)
    def add_row(self,**fields):
        row={'version':'e2-amended-pilot-budget-v1','sequence':len(self.rows),
            'reconciliation_sha256':'9'*64,'previous_sha256':self.rows[-1]['sha256']if self.rows else'0'*64,
            'utc':'2026-11-13T02:01:04.368851+00:00',**fields}
        row['sha256']=R.digest_bytes(R.canonical(row));self.rows.append(row);return row
    def run_recovery(self,**kwargs):
        return R.reconcile_expired_lease(self.output,self.client,recovery_policy=self.policy,
            registration_commit=self.commit,registration_root='unused',proc_root=self.proc,
            now=kwargs.get('now',self.now))
    def test_conservative_closure_preserves_backups_and_unknown_claims(self):
        result=self.run_recovery();self.assertTrue(result['reconciled'])
        ledger=(self.output/R.LEDGER).read_bytes();self.assertTrue(ledger.startswith(self.metadata[R.LEDGER]))
        rows,active=R._rows(ledger);self.assertFalse(active);self.assertEqual(len(rows),236)
        charge=rows[-1];self.assertEqual(charge['elapsed_seconds'],1860)
        self.assertEqual(charge['resource_use']['outcome'],'unknown')
        self.assertIsNone(charge['resource_use']['optimizer_steps'])
        self.assertFalse(charge['resource_use']['elapsed_seconds_measured'])
        remote=json.loads(self.client.files['control/github-primary-v1/head.json'])
        self.assertEqual(remote['result_backups'],self.remote['result_backups'])
        self.assertNotIn('runs/'+R.KNOWN_ATTEMPT+'/manifest.json',self.client.files)
        directory=self.output/'jobs'/R.KNOWN_ATTEMPT
        row,receipt=R.verify_recovery_disposition(self.job,directory,SimpleNamespace(_events=rows),
            self.spec['source_sha256'],list(range(83)))
        self.assertEqual(row['status'],'failed');self.assertIsNone(row['rmse_original'])
        self.assertTrue(row['resource_use']['partial_optimizer_update_may_have_occurred'])
        self.assertFalse((directory/'result.json').exists())
        self.assertFalse((directory/'failure_result.json').exists())
    def test_idempotent_success_does_not_append_another_charge(self):
        self.run_recovery();before=(self.output/R.LEDGER).read_bytes();writes=len(self.client.writes)
        result=self.run_recovery(now=self.now+timedelta(minutes=5))
        self.assertFalse(result['reconciled']);self.assertEqual(before,(self.output/R.LEDGER).read_bytes())
        self.assertEqual(writes,len(self.client.writes))
    def test_transport_failure_local_transaction_replays_without_double_charge(self):
        self.client.fail_patch_once=True
        with self.assertRaisesRegex(R.BackupPause,'synthetic_lost'):
            self.run_recovery()
        before=(self.output/R.LEDGER).read_bytes()
        self.assertEqual(len(R._rows(before)[0]),236)
        result=self.run_recovery(now=self.now+timedelta(minutes=5))
        self.assertTrue(result['reconciled']);self.assertEqual(before,(self.output/R.LEDGER).read_bytes())
    def test_unexpired_lease_active_controller_or_existing_result_blocks(self):
        with self.assertRaisesRegex(R.BackupPause,'not_yet_expired'):
            self.run_recovery(now=datetime(2026,11,13,2,10,tzinfo=timezone.utc))
        self.assertFalse(self.client.writes)
        pid=self.proc/'123';pid.mkdir();(pid/'cmdline').write_bytes(b'python\0/path/e2_resource_worker.py\0')
        with self.assertRaisesRegex(R.BackupPause,'existing_controller'):
            self.run_recovery()
        (pid/'cmdline').unlink()
        self.client.files['runs/'+R.KNOWN_ATTEMPT+'/manifest.json']=b'{"completed":true}'
        with self.assertRaisesRegex(R.BackupPause,'terminal_artifacts_exist'):
            self.run_recovery()
        self.assertFalse(self.client.writes)
    def test_late_terminal_artifact_is_rechecked_inside_cas(self):
        self.client.late_after_reads=3
        with self.assertRaisesRegex(R.BackupPause,'late_terminal_artifacts'):
            self.run_recovery()
        current=json.loads(self.client.files['control/github-primary-v1/head.json'])
        self.assertEqual(current['state_id'],self.remote['state_id'])
        self.assertFalse(self.client.writes)
    def test_changed_immutable_spec_or_policy_or_disposition_blocks(self):
        policy=dict(self.policy);policy['reservation_seconds']=1
        policy['sha256']=R.digest_bytes(R.canonical({k:v for k,v in policy.items()if k!='sha256'}))
        with self.assertRaisesRegex(R.BackupPause,'only_the_declared'):
            R.verify_policy(policy)
        self.run_recovery();directory=self.output/'jobs'/R.KNOWN_ATTEMPT
        saved=json.loads((directory/'recovery_disposition.json').read_bytes());saved['optimizer_steps']=0
        saved['sha256']=R.digest_bytes(R.canonical({k:v for k,v in saved.items()if k!='sha256'}))
        (directory/'recovery_disposition.json').write_bytes(R.canonical(saved)+b'\n')
        with self.assertRaisesRegex(R.BackupPause,'identity_or_claim_changed'):
            R.verify_recovery_disposition(self.job,directory,SimpleNamespace(_events=R._rows((self.output/R.LEDGER).read_bytes())[0]),
                self.spec['source_sha256'],list(range(83)))


if __name__=='__main__':unittest.main()
