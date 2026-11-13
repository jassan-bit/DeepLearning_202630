import importlib.util,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import e2_posthoc_fresh_resume as m

class FreshResumeTests(unittest.TestCase):
    def test_SIGTERM_becomes_keyboard_interrupt_for_original_worker_cleanup(self):
        import signal
        previous=object()
        with patch.object(signal,'signal',return_value=previous) as installed:
            self.assertIs(m.install_shutdown_handler(),previous)
        installed.assert_called_once()
        signum,handler=installed.call_args.args
        self.assertEqual(signum,signal.SIGTERM)
        with self.assertRaisesRegex(KeyboardInterrupt,'preserve unresolved reservations'):
            handler(signal.SIGTERM,None)
    def test_original_restore_omits_HPO_classical_CV_and_pilot_weights(self):
        names=['final','final/final-fit/a/1','final/classical-final/a','final/score-worker-source-recovery-v1',
               'final/classical-cv/a','final/pilot/a','selection/jobs/a']
        state={'result_backups':dict.fromkeys(names,{})}
        self.assertEqual(set(m.original_backups(state)),set(names[:4]))
        files=dict.fromkeys(['shared_budget/pilot_budget.jsonl','final/final_freeze.json',*names[1:]],{})
        self.assertEqual(m.original_paths({'files':files}),{'shared_budget/pilot_budget.jsonl','final/final_freeze.json',*names[1:4]})
    def test_attach_requires_all_metadata_exact_and_keeps_all_backups(self):
        state={'state_id':'a','files':{'budget/a':{'sha256':'x'}},'metadata_deltas':[1], 'result_backups':{'completed':2}}
        raw=SimpleNamespace(_scan=lambda:(dict(state['files']),{}));remote=SimpleNamespace(state=raw)
        m.attach(remote,state);self.assertTrue(raw._restored);self.assertEqual(raw._backups,state['result_backups'])
        raw._scan=lambda:({}, {})
        with self.assertRaisesRegex(ValueError,'metadata'):m.attach(remote,state)
    def test_runtime_annotations_preserve_historical_file_and_same_worker_metadata(self):
        base={'clean_python':'same/path','package_profile':{'packages':{'torch':'2.8'}}}
        full={**base,'synthetic_acceptance_path':'same/acceptance','synthetic_acceptance_sha256':'a'*64,'synthetic_acceptance_scope':'synthetic'}
        old=SimpleNamespace(verify_clean_runtime=Mock(return_value=True));expected=m.sha(m.canonical(base)+b'\n')
        original=m.install_runtime_metadata(old,full,expected,'a'*64)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'clean_runtime.json';p.write_bytes(m.canonical(base)+b'\n');before=p.read_bytes()
            value=json.loads(before);old.verify_clean_runtime(value)
            self.assertEqual(value,full);self.assertEqual(p.read_bytes(),before);original.assert_called_once()
        altered={**base,'synthetic_acceptance_scope':'changed'}
        with self.assertRaisesRegex(ValueError,'metadata differs'):old.verify_clean_runtime(altered)
    def test_runtime_drift_cannot_be_enriched(self):
        base={'clean_python':'original'};full={**base,'synthetic_acceptance_sha256':'a'*64}
        old=SimpleNamespace(verify_clean_runtime=lambda v:True)
        m.install_runtime_metadata(old,full,m.sha(m.canonical(base)+b'\n'),'a'*64)
        with self.assertRaisesRegex(ValueError,'Historical'):old.verify_clean_runtime({'clean_python':'different'})
    def test_artifact_restore_uses_only_selected_groups_and_propagates_failure(self):
        restore=SimpleNamespace(restore_results=Mock(return_value={'manifests':1,'files_restored':2,'existing_identical_files':3}))
        totals=m.restore_artifacts('/root',object(),{'a':1,'b':2},'h',restore,float('inf'))
        self.assertEqual(totals,{'manifests':2,'files_restored':4,'existing_identical_files':6})
        restore.restore_results.side_effect=RuntimeError('preserved download failure')
        with self.assertRaisesRegex(RuntimeError,'preserved'):m.restore_artifacts('/root',object(),{'a':1},'h',restore,float('inf'))
    def test_preparation_timeout_closes_entire_inherited_process_group(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);pid_path=root/'child.pid'
            command=[sys.executable,'-c',
                "import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                "Path(sys.argv[1]).write_text(str(child.pid));time.sleep(30)",str(pid_path)]
            began=time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                m.bounded_preparation(command,deadline=began+.6,cwd=root,log=root/'prepare.log',env=os.environ.copy(),token='dummy-token')
            self.assertLess(time.monotonic()-began,5)
            self.assertTrue(pid_path.exists())
            status=Path('/proc')/pid_path.read_text()/'stat'
            # A reparented zombie has exited and cannot execute further work.
            self.assertTrue(not status.exists() or status.read_text().split(') ')[1].startswith('Z'))
    def test_preparation_subcommands_do_not_escape_global_process_group(self):
        with patch.object(m.subprocess,'run') as run:
            m.preparation_run(['synthetic-restore'],timeout=1,env={})
        self.assertIs(run.call_args.kwargs['start_new_session'],False)
        self.assertIs(run.call_args.kwargs['check'],True)

    def test_dry_run_has_no_file_network_GPU_or_ledger_activity(self):
        with patch.object(m,'validate_bundle',side_effect=AssertionError('No I/O')):
            value=m.run_fresh_session('/missing',{},execute=False)
        self.assertFalse(value['budget_reset']);self.assertEqual(value['authorized_total_hours'],80)
        spec=importlib.util.spec_from_file_location('fresh_colab',ROOT/'notebooks/resume_e2_posthoc_fresh_colab.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with patch.object(module,'fetch',side_effect=AssertionError('No network')):
            self.assertEqual(module.resume_e2_posthoc_fresh_colab('0'*40,execute=False)['status'],'dry_run')
if __name__=='__main__':unittest.main()
