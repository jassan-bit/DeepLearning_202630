import copy,hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_budget84 as adapter

def load_original():
    name='budget84_original_'+str(next(load_original.ids))
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/e2_posthoc_fit_reproduction.py')
    old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old);return old
load_original.ids=iter(range(100000))

def authorization():
    return {'amendment_id':adapter.AMENDMENT_ID,'previous_total_hours':80,'total_hours':84,
            'added_seconds':14400,'user_statement':'si amplialo','persistent_destination':'GitHub',
            'receipt_manifest_path':'runs/e2-budget84-approval-v1/manifest.json','receipt_sha256':'a'*64,
            'approval_sha256':'b'*64,'scope':'Existing cumulative campaign; no scientific changes'}

def anchor(old):
    return {'approved_total_seconds':old.TOTAL_SECONDS,'original_ledger_sha256':old.LEDGER_SHA256,
            'original_charged_seconds':old.ORIGINAL_CHARGED_SECONDS,'initial_remaining_seconds':old.REMAINING_SECONDS,
            'budget_reset':False}

class Budget84Tests(unittest.TestCase):
    def setUp(self):
        self.old=load_original();self.auth=authorization();self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'budget';adapter.install(self.old,self.auth)
    def ledger(self):return self.old.PosthocLedger(self.root,anchor(self.old))
    def attach(self,budget):
        budget.set_persistence(SimpleNamespace(snapshot=lambda reason:{'state_id':reason}))
    def test_extension_preserves_costs_prefix_anchor_and_allows_actual_clone_admission(self):
        with self.ledger()as budget:
            budget.charge_overhead(self.old.REMAINING_SECONDS-1703.1306821202925)
            before=budget.path.read_bytes();anchor_bytes=(self.root/'original_anchor.json').read_bytes();charged=budget.charges
            self.assertEqual(budget.report()['approved_total_hours'],80)
            with self.assertRaisesRegex(ValueError,'unaffordable'):budget.reserve('not-admitted',1860)
            self.attach(budget);result=adapter.apply_extension(budget,self.auth)
            self.assertEqual(result['status'],'budget_extension_recorded');self.assertEqual(budget.charges,charged)
            self.assertAlmostEqual(budget.remaining_seconds,1703.1306821202925+14400)
            self.assertTrue(budget.path.read_bytes().startswith(before));self.assertEqual((self.root/'original_anchor.json').read_bytes(),anchor_bytes)
            budget.reserve('first-admitted',1860);budget.charge('first-admitted',40,status='completed')
            self.assertAlmostEqual(budget.remaining_seconds,1703.1306821202925+14400-40)
            self.assertEqual(budget.report()['approved_total_hours'],84)
    def test_replay_preserves_grant_and_costs_after_original_allowance_is_exceeded(self):
        with self.ledger()as budget:
            self.attach(budget);adapter.apply_extension(budget,self.auth)
            budget.charge_overhead(self.old.REMAINING_SECONDS+100)
            budget.reserve('after-original-exhausted',1860);budget.charge('after-original-exhausted',30)
            expected=budget.remaining_seconds;before=budget.path.read_bytes()
        with self.ledger()as budget:
            self.assertAlmostEqual(budget.remaining_seconds,expected);self.assertEqual(budget.path.read_bytes(),before)
            self.assertEqual(budget.report()['approved_total_hours'],84)
    def test_reapplication_does_not_grant_twice_or_append_another_event(self):
        with self.ledger()as budget:
            self.attach(budget);adapter.apply_extension(budget,self.auth);count=len(budget.events);before=budget.remaining_seconds
            result=adapter.apply_extension(budget,self.auth)
            self.assertEqual(result['status'],'budget_extension_already_recorded');self.assertEqual(len(budget.events),count)
            self.assertEqual(budget.remaining_seconds,before)
    def test_duplicate_amendment_is_rejected_before_any_append(self):
        with self.ledger()as budget:
            self.attach(budget);adapter.apply_extension(budget,self.auth);before=budget.path.read_bytes()
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                budget._append('overhead',elapsed_seconds=0,description=adapter.DESCRIPTION,evidence={'budget_amendment':self.auth})
            self.assertEqual(budget.path.read_bytes(),before)
    def test_unknown_reservation_is_never_forgiven_and_blocks_reopen(self):
        budget=self.ledger();self.attach(budget);budget.reserve('unknown',1860)
        with self.assertRaisesRegex(ValueError,'unresolved'):adapter.apply_extension(budget,self.auth)
        budget.charge('unknown',1,status='interrupted');budget.close()
        with self.assertRaisesRegex(RuntimeError,'Unresolved'):self.ledger()
    def test_no_persistence_means_no_new_grant(self):
        with self.ledger()as budget:
            with self.assertRaisesRegex(RuntimeError,'persistence'):adapter.apply_extension(budget,self.auth)
            self.assertEqual(budget.report()['approved_total_hours'],80);self.assertEqual(budget.events,[])
    def test_conflicting_authorization_installation_fails(self):
        other=dict(self.auth,receipt_sha256='c'*64)
        with self.assertRaisesRegex(ValueError,'Another'):adapter.install(self.old,other)
        self.assertFalse(adapter.install(self.old,self.auth)['installed'])
    def test_forged_amendment_replay_cannot_increase_allowance(self):
        with self.ledger()as budget:
            self.attach(budget);adapter.apply_extension(budget,self.auth)
        rows=[json.loads(x)for x in(self.root/'posthoc_budget.jsonl').read_bytes().splitlines()]
        rows[-1]['evidence']['budget_amendment']['added_seconds']=999999
        rows[-1]['sha256']=self.old.digest({k:v for k,v in rows[-1].items()if k!='sha256'})
        (self.root/'posthoc_budget.jsonl').write_bytes(b''.join(self.old.canonical(r)+b'\n'for r in rows))
        with self.assertRaisesRegex(ValueError,'differs'):self.ledger()
    def test_different_extension_description_or_nonzero_cost_cannot_grant(self):
        with self.ledger()as budget:
            for description,amount in [('authorized_budget_extension_80_to_88',0),(adapter.DESCRIPTION,1)]:
                with self.assertRaisesRegex(ValueError,'differs'):
                    budget._append('overhead',elapsed_seconds=amount,description=description,evidence={'budget_amendment':self.auth})
            self.assertFalse(budget.events)
    def test_ordinary_zero_overhead_does_not_activate_credit(self):
        with self.ledger()as budget:
            budget.charge_overhead(0)
            self.assertEqual(budget.remaining_seconds,self.old.REMAINING_SECONDS)
            self.assertEqual(budget.report()['approved_total_hours'],80)
    def test_bad_authorization_is_rejected_before_patch(self):
        for bad in [dict(self.auth,total_hours=88),dict(self.auth,receipt_manifest_path='../manifest.json'),dict(self.auth,receipt_sha256='bad')]:
            with self.assertRaises(ValueError):adapter.install(load_original(),bad)
    def test_caller_mutation_cannot_change_installed_authority(self):
        changed=copy.deepcopy(self.auth);changed['receipt_sha256']='c'*64
        with self.ledger()as budget:
            self.attach(budget)
            with self.assertRaisesRegex(ValueError,'exact authorization'):adapter.apply_extension(budget,changed)
            self.assertEqual(budget.report()['approved_total_hours'],80)
    def test_upload_head_is_provenance_not_an_extra_credit_identity(self):
        changed=dict(self.auth,receipt_remote_commit='c'*40)
        self.assertFalse(adapter.install(self.old,changed)['installed'])
        with self.ledger()as budget:
            self.attach(budget);adapter.apply_extension(budget,changed)
            before=budget.remaining_seconds
            other=dict(changed,receipt_remote_commit='d'*40)
            self.assertEqual(adapter.apply_extension(budget,other)['status'],'budget_extension_already_recorded')
            self.assertEqual(budget.remaining_seconds,before)

    def test_mutated_in_memory_amendment_cannot_be_claimed_idempotent(self):
        with self.ledger()as budget:
            self.attach(budget);adapter.apply_extension(budget,self.auth)
            budget.events[-1]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'checksum'):adapter.apply_extension(budget,self.auth)

    def test_original_scientific_source_bytes_remain_unchanged(self):
        path=Path(self.old.__file__);before=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.ledger()as budget:
            self.attach(budget);adapter.apply_extension(budget,self.auth)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
        self.assertEqual(self.old.TOTAL_SECONDS,80*3600)
if __name__=='__main__':unittest.main()
