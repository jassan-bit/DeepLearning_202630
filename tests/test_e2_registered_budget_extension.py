"""Synthetic budget/Git-proof checks. No market data or model execution."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,unittest

ENGINE_SCRIPTS=Path(os.environ.get('E2_ENGINE_SCRIPTS','/workspace/work/e2architecture/scripts'))
sys.path.insert(0,str(ENGINE_SCRIPTS))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import e2_reduced_budget as base
import e2_registered_budget_extension as funding


class Clock:
    def __init__(self):self.value=0.
    def __call__(self):return self.value
    def advance(self,n):self.value+=n


def hashed(v):return v|{'sha256':base.digest(v)}

def reconciliation():
    return hashed({'version':base.historical.VERSION,'status':'reconciled_before_new_pilot_reservations',
        'authorized_combined_hours':64,'authorized_combined_seconds':230400,
        'historical_charged_seconds':222927.98951942776,
        'remaining_seconds_before_pilots':7472.010480572237,'budget_reset':False})

def migration():
    return hashed({'version':base.MIGRATION_VERSION,'authorized_total_hours':64,'budget_reset':False,
        'budget_extension_authorized':False,'historical_reconciliation_sha256':reconciliation()['sha256'],
        'scope':'new_reduced_jobs_only','stage_wall_seconds':600,'worker_overhead_seconds':60,
        'max_optimization_phases':3,'max_epochs':200,'patience_epochs':20})

def approved():
    v=funding.policy_draft();v.pop('sha256');v['status']='authorized_by_user'
    v['authorization']={'source':'explicit_user_confirmation','approved_total_hours':72,
        'evidence_sha256':hashlib.sha256(b'synthetic confirmation, no real user approval').hexdigest()}
    return hashed(v)


def register(root,p):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','init','-q',str(root)],check=True)
    path=root/funding.REGISTRATION_PATH;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'protocol_id':'e2-reduced-delivery-v1','engine_commit':funding.ENGINE_COMMIT,'funding_policy':p}))
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Synthetic','-c','user.email=test@example.invalid','commit','-qm','Synthetic funding fixture'],check=True)
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()


def make_base(path,clock):
    return base.ReducedBudgetLedger(path,reconciliation(),migration=migration(),
        migration_registration_commit='a'*40,clock=clock)


def make_funded(cls,path,clock):
    return cls(path,reconciliation(),migration=migration(),migration_registration_commit='a'*40,clock=clock)


class FundingTests(unittest.TestCase):
    def test_proposed_or_unregistered_policy_cannot_grant_credit(self):
        with tempfile.TemporaryDirectory() as d:
            c=register(Path(d)/'git',funding.policy_draft())
            with self.assertRaisesRegex(base.BudgetError,'authorization'):
                funding.make_funded_budget_class(funding.policy_draft(),funding_registration_commit=c,funding_git_root=Path(d)/'git')
            with self.assertRaisesRegex(base.BudgetError,'missing_or_invalid'):
                funding.make_funded_budget_class(approved(),funding_registration_commit='f'*40,funding_git_root=Path(d)/'git')

    def test_missing_base_campaign_does_not_create_a_new_funded_allowance(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);p=approved();commit=register(root/'git',p)
            cls=funding.make_funded_budget_class(p,funding_registration_commit=commit,funding_git_root=root/'git')
            before=base.historical.PilotBudgetLedger
            with self.assertRaisesRegex(base.BudgetReconciliationRequired,'existing_reduced_migration'):
                make_funded(cls,root/'empty',Clock())
            self.assertFalse((root/'empty'/funding.ANCHOR_NAME).exists())
            self.assertIs(base.historical.PilotBudgetLedger,before)

    def test_git_binding_must_contain_the_exact_approved_policy(self):
        with tempfile.TemporaryDirectory() as d:
            p=approved();c=register(d,p);changed=dict(p);changed['authorization']=dict(p['authorization'],evidence_sha256='a'*64)
            changed=hashed({k:v for k,v in changed.items()if k!='sha256'})
            with self.assertRaisesRegex(base.BudgetError,'prior_git_funding'):
                funding.make_funded_budget_class(changed,funding_registration_commit=c,funding_git_root=d)

    def test_original_prefix_and_migration_bytes_preserved_and_credit_once(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'ledger';clock=Clock()
            with make_base(path,clock)as old:
                old.reserve('old',660);clock.advance(10);old.charge('old',10,status='completed')
            prefix=(path/'pilot_budget.jsonl').read_bytes();migration_bytes=(path/'reduced_migration.json').read_bytes()
            p=approved();commit=register(root/'git',p)
            cls=funding.make_funded_budget_class(p,funding_registration_commit=commit,funding_git_root=root/'git')
            with make_funded(cls,path,clock)as new:
                self.assertAlmostEqual(new.remaining_seconds,7472.010480572237-10+28800)
                self.assertEqual(new.reservation_for(),660);self.assertEqual(new.reservation_for(3),1860)
                new.reserve('new',660);clock.advance(11);new.charge('new',11,status='completed')
            self.assertTrue((path/'pilot_budget.jsonl').read_bytes().startswith(prefix))
            self.assertEqual((path/'reduced_migration.json').read_bytes(),migration_bytes)
            anchor=(path/funding.ANCHOR_NAME).read_bytes()
            with make_funded(cls,path,clock)as again:
                self.assertAlmostEqual(again.remaining_seconds,7472.010480572237-21+28800)
                self.assertEqual(again.report()['effective_authorized_total_hours'],72)
                with self.assertRaises(base.BudgetReconciliationRequired):again.reserve('different',660,metadata={'logical_job_id':'new'})
            self.assertEqual((path/funding.ANCHOR_NAME).read_bytes(),anchor)

    def test_spending_past64_depletes72_before_clamping(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'ledger';clock=Clock()
            # Known overhead overrun remains a charge, even when the old cap is crossed.
            with make_base(path,clock)as old:
                clock.advance(7500)
                with self.assertRaises(base.BudgetPause):old.charge_overhead(0)
            p=approved();commit=register(root/'git',p)
            cls=funding.make_funded_budget_class(p,funding_registration_commit=commit,funding_git_root=root/'git')
            with make_funded(cls,path,clock)as new:
                self.assertAlmostEqual(new.remaining_seconds,7472.010480572237+28800-7500)
                new.reserve('funded',660);clock.advance(30);new.charge('funded',30,status='completed')
            with make_funded(cls,path,clock)as again:
                self.assertAlmostEqual(again.remaining_seconds,7472.010480572237+28800-7530)

    def test_active_unknown_lease_cannot_be_cleared_by_extension(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'ledger';clock=Clock();old=make_base(path,clock)
            old.reserve('unknown',660);clock.advance(5);old.close()
            raw=(path/'pilot_budget.jsonl').read_bytes();p=approved();commit=register(root/'git',p)
            cls=funding.make_funded_budget_class(p,funding_registration_commit=commit,funding_git_root=root/'git')
            with self.assertRaises(base.BudgetReconciliationRequired):make_funded(cls,path,clock)
            self.assertEqual((path/'pilot_budget.jsonl').read_bytes(),raw)
            self.assertFalse((path/funding.ANCHOR_NAME).exists())

    def test_changed_funding_or_original_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'ledger';clock=Clock()
            with make_base(path,clock):pass
            p=approved();commit=register(root/'git',p)
            cls=funding.make_funded_budget_class(p,funding_registration_commit=commit,funding_git_root=root/'git')
            with make_funded(cls,path,clock):pass
            a=json.loads((path/funding.ANCHOR_NAME).read_text());a['effective_total_hours']=80
            a=hashed({k:v for k,v in a.items()if k!='sha256'});(path/funding.ANCHOR_NAME).write_text(json.dumps(a))
            with self.assertRaisesRegex(base.BudgetReconciliationRequired,'anchor_changed'):make_funded(cls,path,clock)

    def test_funded_reservation_rows_cannot_remove_their_authority(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'ledger';clock=Clock()
            with make_base(path,clock):pass
            p=approved();commit=register(root/'git',p)
            cls=funding.make_funded_budget_class(p,funding_registration_commit=commit,funding_git_root=root/'git')
            with make_funded(cls,path,clock)as ledger:
                ledger.reserve('one',660);clock.advance(1);ledger.charge('one',1,status='completed')
            records=[json.loads(x)for x in(path/'pilot_budget.jsonl').read_bytes().splitlines()]
            last=records[-1];last.pop('funding_policy_sha256');last.pop('sha256');last['sha256']=base.digest(last)
            records[-1]=last;(path/'pilot_budget.jsonl').write_bytes(b''.join(base.canonical(r)+b'\n'for r in records))
            with self.assertRaisesRegex(base.BudgetReconciliationRequired,'authority'):make_funded(cls,path,clock)


if __name__=='__main__':unittest.main()
