"""Real ledger/Git fixtures for prospective resource bounds; no model execution."""
from copy import deepcopy
import hashlib,json,os
from pathlib import Path
import subprocess,sys,tempfile,unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,os.environ.get('E2_ENGINE_SCRIPTS','/workspace/work/e2architecture/scripts'))
sys.path.insert(0,str(Path('/workspace/e2-budget72/scripts')))
sys.path.insert(0,str(ROOT/'scripts'))
import e2_reduced_budget as base
import e2_registered_budget_extension as funding
import e2_registered_resource_extension as resource


class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value
    def advance(self,n):self.value+=n


def hashed(v):return v|{'sha256':base.digest(v)}
def rec():return hashed({'version':base.historical.VERSION,'status':'reconciled_before_new_pilot_reservations',
    'authorized_combined_hours':64,'authorized_combined_seconds':230400,'historical_charged_seconds':222927.98951942776,
    'remaining_seconds_before_pilots':7472.010480572237,'budget_reset':False})
def migration():return hashed({'version':base.MIGRATION_VERSION,'authorized_total_hours':64,'budget_reset':False,
    'budget_extension_authorized':False,'historical_reconciliation_sha256':rec()['sha256'],'scope':'new_reduced_jobs_only',
    'stage_wall_seconds':600,'worker_overhead_seconds':60,'max_optimization_phases':3,'max_epochs':200,'patience_epochs':20})
def policy(commit):return hashed({'version':resource.VERSION,'engine_commit':resource.ENGINE_COMMIT,
    'funding_registration_commit':commit,'initial_phase_wall_seconds':600,'stage_wall_seconds':1800,
    'worker_overhead_seconds':60,'max_optimization_phases':3,'max_epochs':200,'patience_epochs':20,
    'authorized_total_hours':72,'budget_reset':False,'new_opportunities_only':True,'scientific_retry_allowed':False,
    'models_seeds_data_unchanged':True,'prior_failed_attempts_retried':False})


def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL).decode().strip()
def commit(root):
    git(root,'add','.')
    git(root,'-c','user.name=Synthetic','-c','user.email=synthetic@example.invalid','commit','-qm','Synthetic fixture')
    return git(root,'rev-parse','HEAD')


class ResourceBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.repo=self.root/'git';self.repo.mkdir();git(self.repo,'init','-q')
        approval=funding.policy_draft();approval.pop('sha256');approval.update(status='authorized_by_user',
            authorization={'source':'explicit_user_confirmation','approved_total_hours':72,'evidence_sha256':hashlib.sha256(b'synthetic-only').hexdigest()})
        approval=hashed(approval)
        p=self.repo/funding.REGISTRATION_PATH;p.parent.mkdir(parents=True)
        p.write_text(json.dumps({'protocol_id':'e2-reduced-delivery-v1','engine_commit':resource.ENGINE_COMMIT,'funding_policy':approval}))
        self.h=commit(self.repo)
        self.funded=funding.make_funded_budget_class(approval,funding_registration_commit=self.h,funding_git_root=self.repo)
        self.patch=patch.object(resource,'FUNDING_COMMIT',self.h);self.patch.start();self.addCleanup(self.patch.stop)
        self.policy=policy(self.h)
        source=(ROOT/'scripts/e2_registered_resource_extension.py').read_bytes()
        q=self.repo/resource.SOURCE_PATH;q.parent.mkdir(parents=True);q.write_bytes(source)
        p=self.repo/resource.REGISTRATION_PATH;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'protocol_id':'e2-reduced-delivery-v1','engine_commit':resource.ENGINE_COMMIT,
            'resource_policy':self.policy,'files_sha256':{resource.SOURCE_PATH:hashlib.sha256(source).hexdigest()}}))
        self.p=commit(self.repo)
        self.cls=resource.make_resource_budget_class(self.funded,self.policy,registration_commit=self.p,
            registration_root=self.repo,extension_root=self.repo)
        self.clock=Clock();self.path=self.root/'ledger'
        with self.make(base.ReducedBudgetLedger)as old:
            old.reserve('oldnormal',660);self.clock.advance(5);old.charge('oldnormal',5,status='completed')
            old.reserve('oldfailed',1860,metadata={'logical_job_id':'prior-failed'});self.clock.advance(7);old.charge('oldfailed',7,status='failed')
        with self.make(self.funded):pass

    def make(self,cls):return cls(self.path,rec(),migration=migration(),migration_registration_commit='a'*40,clock=self.clock)

    def test_new_bounds_preserve_funding_migration_and_all_prior_receipts(self):
        prefix=(self.path/'pilot_budget.jsonl').read_bytes();oldmigration=(self.path/'reduced_migration.json').read_bytes()
        oldfunding=(self.path/'funding_extension.json').read_bytes()
        with self.make(self.cls)as current:
            self.assertEqual(current.stage_wall_seconds,1800)
            self.assertEqual(current.reservation_for(),1860);self.assertEqual(current.reservation_for(3),5460)
            self.assertEqual(current.worker_timeout_for(3),5455)
            self.assertAlmostEqual(current.remaining_seconds,7472.010480572237+28800-12)
            current.reserve('new',1860,metadata={'phases':1});self.clock.advance(11)
            current.charge('new',11,status='completed')
        self.assertTrue((self.path/'pilot_budget.jsonl').read_bytes().startswith(prefix))
        self.assertEqual((self.path/'reduced_migration.json').read_bytes(),oldmigration)
        self.assertEqual((self.path/'funding_extension.json').read_bytes(),oldfunding)
        rows=[json.loads(x)for x in(self.path/'pilot_budget.jsonl').read_bytes().splitlines()]
        reserve=next(x for x in rows if x.get('attempt_id')=='new'and x['kind']=='reserved')
        self.assertEqual(reserve['metadata']['resource_registration_commit'],self.p)
        self.assertEqual(reserve['metadata']['resource_policy_sha256'],self.policy['sha256'])
        self.assertEqual(reserve['metadata']['funding_policy_sha256']if'funding_policy_sha256'in reserve['metadata']else reserve['funding_policy_sha256'],json.loads(oldfunding)['funding_policy_sha256'])

    def test_reopen_has_no_extra_campaign_credit_and_failed_job_stays_failed(self):
        with self.make(self.cls)as current:
            current.reserve('one',1860);self.clock.advance(25);current.charge('one',25,status='completed')
        anchor=(self.path/resource.ANCHOR_NAME).read_bytes()
        with self.make(self.cls)as again:
            self.assertAlmostEqual(again.remaining_seconds,7472.010480572237+28800-37)
            with self.assertRaises(base.BudgetReconciliationRequired):
                again.reserve('different',1860,metadata={'logical_job_id':'prior-failed'})
            with self.assertRaisesRegex(ValueError,'prospective'):
                again.reserve('old-bound',660)
        self.assertEqual((self.path/resource.ANCHOR_NAME).read_bytes(),anchor)

    def test_unknown_open_worker_cannot_be_released_or_retried(self):
        old=self.make(self.funded);old.reserve('unknown',660);self.clock.advance(2);old.close()
        before=(self.path/'pilot_budget.jsonl').read_bytes()
        with self.assertRaises(base.BudgetReconciliationRequired):self.make(self.cls)
        self.assertEqual((self.path/'pilot_budget.jsonl').read_bytes(),before)
        self.assertFalse((self.path/resource.ANCHOR_NAME).exists())

    def test_modified_or_unregistered_resource_policy_is_rejected(self):
        changed={k:v for k,v in self.policy.items()if k!='sha256'};changed['stage_wall_seconds']=600
        with self.assertRaisesRegex(ValueError,'prospective_resource'):
            resource.make_resource_budget_class(self.funded,hashed(changed),registration_commit=self.p,registration_root=self.repo)
        with self.assertRaisesRegex(ValueError,'missing'):
            resource.make_resource_budget_class(self.funded,self.policy,registration_commit='f'*40,registration_root=self.repo)

    def test_changed_anchor_or_new_reservation_authority_blocks(self):
        with self.make(self.cls)as current:
            current.reserve('one',1860);self.clock.advance(1);current.charge('one',1,status='completed')
        path=self.path/'pilot_budget.jsonl';rows=[json.loads(x)for x in path.read_bytes().splitlines()]
        for row in rows:
            if row.get('kind')=='reserved'and row.get('attempt_id')=='one':row['metadata'].pop('resource_policy_sha256')
        previous='0'*64
        for row in rows:
            row.pop('sha256');row['previous_sha256']=previous;row['sha256']=base.digest(row);previous=row['sha256']
        path.write_bytes(b''.join(base.canonical(x)+b'\n'for x in rows))
        with self.assertRaisesRegex(ValueError,'resource_reservation_missing'):
            self.make(self.cls)

    def test_phase_count_cannot_hide_a_shorter_reservation(self):
        with self.make(self.cls)as current:
            with self.assertRaisesRegex(ValueError,'phase_count'):
                current.reserve('wrong',1860,metadata={'phases':3})
            self.assertFalse(current._active_started)
            self.assertTrue((self.path/resource.ANCHOR_NAME).is_file())


if __name__=='__main__':unittest.main()
