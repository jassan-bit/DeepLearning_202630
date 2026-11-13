import io,json,signal,sys,tempfile,unittest,zipfile
from datetime import datetime,timezone,timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import pause_e2_posthoc_inference as m

def rows_fixture(root,interrupted=False):
    rows=[]
    def append(kind,**kw):
        r={'version':'e2-posthoc-clean-fit-reproduction-v1','kind':kind,'sequence':len(rows),
           'previous_sha256':rows[-1]['sha256'] if rows else m.ZERO,'anchor_sha256':'a'*64,
           'utc':(datetime.now(timezone.utc)-timedelta(seconds=20)).isoformat(),**kw}
        r['sha256']=m.sha(m.canonical(r));rows.append(r)
    append('reserved',job_id='frozen-posthoc-test',seconds=5400.,metadata={'training_performed':False})
    if interrupted:append('interrupted',job_id='frozen-posthoc-test',elapsed_seconds_lower_bound=10.,full_reservation_retained=True)
    p=Path(root)/'budget/posthoc_budget.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(b''.join(m.canonical(r)+b'\n' for r in rows));return p,rows

class PauseTests(unittest.TestCase):
    def test_chain_and_exact_inference_reservation(self):
        with tempfile.TemporaryDirectory() as t:
            p,rows=rows_fixture(t,True);raw,actual,r=m.ledger_rows(p)
            self.assertEqual(actual,rows);self.assertEqual(r['job_id'],'frozen-posthoc-test')
            p.write_bytes(raw.replace(b'5400.0',b'5401.0'))
            with self.assertRaises(ValueError):m.ledger_rows(p)
    def test_no_optimizer_specs_guard(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'reproduction/jobs/fit/worker_spec.json';p.parent.mkdir(parents=True);p.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'optimizer'):m.no_optimizer(t,records=[])
    def test_upper_charge_preserves_lowerbound_and_stop_overhead(self):
        r={'utc':'2026-11-13T12:00:00+00:00','seconds':5400.}
        proof={'interrupted_record':{'elapsed_seconds_lower_bound':100.},
            'controller_exited_at_utc':'2026-11-13T12:01:42+00:00','signal_to_exit_monotonic_seconds':3.}
        self.assertEqual(m.upper_charge(proof,r),108.)
        r['seconds']=107.
        with self.assertRaisesRegex(ValueError,'exceeds'):m.upper_charge(proof,r)
    def test_archive_moves_only_partial_inference_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);partial=root/'robustness/predictors/partial';partial.mkdir(parents=True)
            (partial/'started.json').write_bytes(b'original-started-bytes')
            done=root/'robustness/predictors/complete';done.mkdir();(done/'result.json').write_text('{}')
            archive=root/'transport_amendments/test';result=m.archive_partials(root,archive)
            self.assertEqual((archive/'partial/started.json').read_bytes(),b'original-started-bytes')
            self.assertFalse(partial.exists());self.assertTrue((done/'result.json').exists());self.assertEqual(len(result),1)
    def test_archive_rejects_weights_in_partial_inference(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'robustness/predictors/partial';p.mkdir(parents=True);(p/'model.pt').write_bytes(b'weights')
            with self.assertRaisesRegex(ValueError,'Weights'):m.archive_partials(t,Path(t)/'archive')
            self.assertTrue((p/'model.pt').exists())
    def test_controlled_pause_signals_only_exact_controller_and_records_exit(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path,rs=rows_fixture(root)
            ext=root/('guide-closure-extension-'+m.PIN)
            args=['python',str(ext/'scripts/e2_posthoc_fit_reproduction.py'),'--orchestrate',
                '--configuration-commit',m.PIN,'--destination',str(root),
                '--extension-root',str(ext),'--original-output','/original']
            proc={'pid':12345,'argv':args,'ppid':1,'start_ticks':'991'};state={'alive':True};signals=[]
            def kill(pid,sig):
                signals.append((pid,sig));state['alive']=False
                r={'version':'e2-posthoc-clean-fit-reproduction-v1','kind':'interrupted','sequence':1,
                    'previous_sha256':rs[-1]['sha256'],'anchor_sha256':'a'*64,'utc':m.utc(),
                    'job_id':'frozen-posthoc-test','elapsed_seconds_lower_bound':10.,'full_reservation_retained':True}
                r['sha256']=m.sha(m.canonical(r))
                with path.open('ab') as f:f.write(m.canonical(r)+b'\n')
            with patch.object(m,'processes',side_effect=lambda:[proc] if state['alive'] else []), \
                 patch.object(m,'process',side_effect=lambda pid:proc if state['alive'] else None), \
                 patch.object(m,'no_optimizer'), \
                 patch.object(m,'complete_predictors',side_effect=[{}, {'robustness/predictors/new/result.json':'b'*64}]), \
                 patch.object(m,'remote_completed',return_value=True),patch.object(m.os,'kill',side_effect=kill):
                proof=m.pause(root,pid=12345,timeout=1.)
            self.assertEqual(signals,[(12345,signal.SIGINT)])
            self.assertEqual(proof['ledger_paused_sha256'],m.file_sha(path));self.assertTrue(proof['no_optimizer_specs'])
    def test_exact_prefix_restore_accepts_only_proven_interrupted_suffix(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path,rows=rows_fixture(root,True);local=path.read_bytes();prefix=m.canonical(rows[0])+b'\n'
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w') as z:z.writestr('budget/posthoc_budget.jsonl',prefix)
            archive=stream.getvalue();descriptor={'path':'delta.zip','bytes':len(archive),'sha256':m.sha(archive)}
            current={'output_path':str(root),'files':{'budget/posthoc_budget.jsonl':{'bytes':len(prefix),'sha256':m.sha(prefix)}},
                     'metadata_deltas':[descriptor],'result_backups':{}}
            state=SimpleNamespace(_restored=False,client=SimpleNamespace(_branch_head=lambda:'a'*40),
                _current=lambda head:current,_scan=lambda:({'budget/posthoc_budget.jsonl':{'bytes':len(local),'sha256':m.sha(local)}},{}),
                _content_bytes=lambda path,head:archive)
            class Remote:
                def restore(self):raise RuntimeError('local_control_diverges_preserved_without_overwrite')
            remote=Remote();remote.state=state
            m.restore_controlled_prefix(remote,root,{'interrupted_record':rows[-1]})
            self.assertTrue(state._restored);self.assertIs(state._state,current)
            with self.assertRaisesRegex(ValueError,'proven controlled'):
                m.restore_controlled_prefix(remote,root,{'interrupted_record':{}})
    def test_replayed_real_ledger_closes_only_inference_and_preserves_history(self):
        import e2_posthoc_fit_reproduction as old
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);original=root/'original/pilot_budget.jsonl';original.parent.mkdir()
            original.write_bytes(b'original-closed-journal\n');original_hash=m.file_sha(original)
            anchor={'approved_total_seconds':old.TOTAL_SECONDS,'original_ledger_sha256':original_hash,
                    'original_charged_seconds':old.ORIGINAL_CHARGED_SECONDS,'initial_remaining_seconds':old.REMAINING_SECONDS,
                    'budget_reset':False,'original_ledger_path':str(original)}
            with patch.object(old,'LEDGER_SHA256',original_hash):
                with old.PosthocLedger(root/'budget',anchor) as ledger:
                    ledger.charge_overhead(7.)
                    ledger.reserve('frozen-posthoc-test',5400.,metadata={'training_performed':False})
                    ledger.charge('frozen-posthoc-test',10.,status='interrupted')
                prior=(root/'budget/posthoc_budget.jsonl').read_bytes()
                ledger=m.recovery_ledger(old,root/'budget',anchor)
                try:
                    ledger.charge('frozen-posthoc-test',25.,status='failed',evidence={'observed_PID_exited':True})
                    self.assertEqual(ledger.charges,32.)
                    self.assertEqual(ledger.active,{})
                    self.assertEqual(ledger.remaining_seconds,old.REMAINING_SECONDS-32.)
                    self.assertTrue((root/'budget/posthoc_budget.jsonl').read_bytes().startswith(prior))
                finally:ledger.close()
if __name__=='__main__':unittest.main()
