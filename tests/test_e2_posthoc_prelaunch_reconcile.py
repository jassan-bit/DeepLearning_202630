"""Incident-specific synthetic cache reconciliation checks; never train."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_prelaunch_reconcile as m

def event_hash(value):return m.sha(m.canonical({k:v for k,v in value.items() if k!='sha256'}))
def checked(value):
    if value.get('sha256')!=event_hash(value):raise ValueError('Invalid event SHA')

class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.output=self.root/'posthoc';self.original=self.root/'original';self.archive=self.root/'outside-archive'
        (self.output/'budget').mkdir(parents=True);(self.original/'shared_budget').mkdir(parents=True)
        (self.original/'shared_budget/pilot_budget.jsonl').write_bytes(b'original immutable ledger\n')
        self.old=SimpleNamespace(ZERO='0'*64,VERSION='fixture-v1',checked=checked,validate_original_anchor=Mock(return_value={}))
        rows=[]
        def add(kind,**extra):
            row={'version':'fixture-v1','sequence':len(rows),'previous_sha256':rows[-1]['sha256'] if rows else self.old.ZERO,
                 'kind':kind,**extra};row['sha256']=event_hash(row);rows.append(row)
        add('reserved',job_id='completed',seconds=10)
        add('charged',job_id='completed',elapsed_seconds=5)
        add('overhead',elapsed_seconds=1)
        self.remote=m.canonical(rows[0])+b'\n'+m.canonical(rows[1])+b'\n'+m.canonical(rows[2])+b'\n'
        add('reserved',job_id=m.JOB_ID,seconds=1860.0,metadata={'kind':'reproduction','model':'iTransformer','seed':271801})
        self.local=self.remote+m.canonical(rows[-1])+b'\n';self.tail=rows[-1]
        self.ledger=self.output/'budget/posthoc_budget.jsonl';self.ledger.write_bytes(self.local)
        self.job=self.output/'reproduction/jobs'/m.JOB_ID;self.job.mkdir(parents=True)
        self.spec={'job_id':m.JOB_ID,'kind':'reproduction','model':'iTransformer','seed':271801,'condition':'full',
                   'configuration_commit':m.CONFIGURATION,'controller_source_sha256':m.CONTROLLER_SHA,'original_output':str(self.original)}
        self.spec_raw=m.canonical(self.spec)+b'\n';(self.job/'worker_spec.json').write_bytes(self.spec_raw)
        completed=self.output/'reproduction/jobs/already-complete/result.json';completed.parent.mkdir();completed.write_bytes(b'untouched')
        self.state={'state_id':m.STATE_ID,'files':{'budget/posthoc_budget.jsonl':{'sha256':m.sha(self.remote)},
                    'reproduction/jobs/one/result.json':{},'reproduction/jobs/two/result.json':{}}}
        self.check=Mock(side_effect=lambda:json.loads(json.dumps(self.state)))
        def backed(folder,receipt):return {'status':'backed_up','remote_commit':'a'*40,'manifest_path':'runs/proof/manifest.json','attempt_id':receipt['attempt_id']}
        self.backup=Mock(side_effect=backed);self.idle=Mock(return_value=True)
    def patches(self):return patch.multiple(m,REMOTE_SHA=m.sha(self.remote),LOCAL_SHA=m.sha(self.local),TAIL_SHA=self.tail['sha256'],REMOTE_ROWS=3,COMPLETED_FITS=2)
    def run_recovery(self):
        with self.patches():
            return m.reconcile_local_only(self.old,self.output,self.original,self.archive,self.remote,self.state,
                                          self.backup,self.check,self.idle,clock=Mock(side_effect=[100.,103.]))
    def assert_intact(self):
        self.assertEqual(self.ledger.read_bytes(),self.local);self.assertEqual((self.job/'worker_spec.json').read_bytes(),self.spec_raw)
        self.assertEqual((self.output/'reproduction/jobs/already-complete/result.json').read_bytes(),b'untouched')
    def test_preserves_durable_full_evidence_before_cache_prefix_restore(self):
        def backup(folder,receipt):
            self.assert_intact()
            self.assertEqual((folder/'local_journal_253.jsonl').read_bytes(),self.local)
            self.assertEqual((folder/'remote_journal_252.jsonl').read_bytes(),self.remote)
            self.assertEqual((folder/'unused_worker_spec.json').read_bytes(),self.spec_raw)
            self.assertTrue(all(m.sha((folder/name).read_bytes())==sha for name,sha in receipt['artifacts_sha256'].items()))
            return {'status':'backed_up','remote_commit':'a'*40,'manifest_path':'runs/proof/manifest.json','attempt_id':receipt['attempt_id']}
        self.backup.side_effect=backup;result=self.run_recovery();folder=Path(result['archive_directory'])
        self.assertEqual(self.ledger.read_bytes(),self.remote);self.assertFalse(self.job.exists())
        self.assertEqual((folder/'unlaunched_job/worker_spec.json').read_bytes(),self.spec_raw)
        self.assertEqual((self.output/'reproduction/jobs/already-complete/result.json').read_bytes(),b'untouched')
        self.assertEqual(result['prior_failed_transport_conservative_seconds'],300.)
        self.assertEqual(result['recovery_wall_seconds'],3.);self.assertFalse(result['budget_reset'])
        self.assertEqual(self.check.call_count,2);self.assertEqual(self.idle.call_count,2)
    def test_backup_failure_keeps_local_admission_and_all_results_unchanged(self):
        self.backup.side_effect=RuntimeError('Network unavailable')
        with self.assertRaisesRegex(RuntimeError,'Network'):self.run_recovery()
        self.assert_intact()
    def test_unverified_backup_receipt_never_releases_local_admission(self):
        self.backup.side_effect=lambda *args:{'status':'pending'}
        with self.assertRaisesRegex(RuntimeError,'durable GitHub'):self.run_recovery()
        self.assert_intact()
    def test_remote_change_during_archival_is_not_adopted(self):
        changed=dict(self.state,state_id='new-unknown-state');self.check.side_effect=[self.state,changed]
        with self.assertRaisesRegex(RuntimeError,'advanced'):self.run_recovery()
        self.assert_intact()
    def test_any_worker_log_result_or_phase_prohibits_reconciliation(self):
        for extra in ('worker.log','result.json','phases'):
            p=self.job/extra
            if extra=='phases':p.mkdir()
            else:p.write_text('execution evidence')
            with self.assertRaisesRegex(ValueError,'execution evidence'):self.run_recovery()
            if p.is_dir():p.rmdir()
            else:p.unlink()
        self.backup.assert_not_called();self.assert_intact()
    def test_active_controller_blocks_before_backup_or_mutation(self):
        self.idle.return_value=False
        with self.assertRaisesRegex(RuntimeError,'activity'):self.run_recovery()
        self.backup.assert_not_called();self.assert_intact()
    def test_worker_appearing_after_backup_blocks_mutation(self):
        self.idle.side_effect=[True,False]
        with self.assertRaisesRegex(RuntimeError,'activity appeared'):self.run_recovery()
        self.assert_intact()
    def test_changed_local_prefix_is_never_truncated(self):
        self.ledger.write_bytes(self.local.replace(b'completed',b'other-job'))
        with self.assertRaisesRegex(ValueError,'journal differs'):self.run_recovery()
        self.backup.assert_not_called();self.assertTrue(self.job.exists())
    def test_wrong_unused_spec_is_never_launched_or_archived_as_eligible(self):
        wrong=dict(self.spec,seed=271802);(self.job/'worker_spec.json').write_bytes(m.canonical(wrong)+b'\n')
        with self.assertRaisesRegex(ValueError,'specification'):self.run_recovery()
        self.backup.assert_not_called();self.assertEqual(self.ledger.read_bytes(),self.local)
    def test_remote_active_reservation_is_not_credited(self):
        # Rebind the synthetic hashes solely to reach the remote-active guard.
        rows=[json.loads(x) for x in self.remote.splitlines()];rows[-1].update(kind='reserved',job_id='unknown',seconds=10)
        rows[-1]['sha256']=event_hash(rows[-1]);self.remote=b''.join(m.canonical(x)+b'\n' for x in rows)
        self.tail['previous_sha256']=rows[-1]['sha256'];self.tail['sha256']=event_hash(self.tail)
        self.local=self.remote+m.canonical(self.tail)+b'\n';self.ledger.write_bytes(self.local)
        self.state['files']['budget/posthoc_budget.jsonl']['sha256']=m.sha(self.remote)
        with self.assertRaisesRegex(ValueError,'remote unresolved'):self.run_recovery()
        self.backup.assert_not_called();self.assert_intact()
    def test_failed_atomic_restore_rolls_unused_directory_back_without_deleting_proof(self):
        with patch.object(m,'_atomic_restore',side_effect=OSError('Disk full')):
            with self.assertRaisesRegex(OSError,'Disk full'):self.run_recovery()
        self.assert_intact();self.assertTrue(list(self.archive.rglob('local_journal_253.jsonl')))
    def test_archive_cannot_be_inside_scientific_campaign(self):
        self.archive=self.output/'bad-archive'
        with self.assertRaisesRegex(ValueError,'outside'):self.run_recovery()
        self.backup.assert_not_called();self.assert_intact()

if __name__=='__main__':unittest.main()
