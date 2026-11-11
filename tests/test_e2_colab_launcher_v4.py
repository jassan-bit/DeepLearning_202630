"""Inert preview, repair-only invocation and single-controller launch checks."""
from contextlib import redirect_stdout
from io import StringIO
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('r4_launcher',ROOT/'scripts/colab_recovery_v4_launcher.py')
launcher=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(launcher)

class LauncherTests(unittest.TestCase):
    def test_preview_does_not_use_network_gpu_or_credentials(self):
        with patch.object(launcher,'_run',side_effect=AssertionError('process started')), \
             patch.object(launcher,'_github_environment',side_effect=AssertionError('credential read')), \
             redirect_stdout(StringIO()):
            value=launcher.run_launcher({})
        self.assertEqual(value['status'],'preview_only')
        self.assertFalse(value['network_used'])
        self.assertEqual(value['registered_effective_campaign_hours'],48)

    def test_command_repairs_only_and_cannot_extend_budget(self):
        command=launcher._command('python',Path('controller'),Path('engine'),'a'*40,
            Path('auth.json'),'b'*64,Path('campaign'),execute=True,
            report=Path('report.json'),data=Path('data'))
        self.assertIn('--repair-only',command)
        self.assertIn('--retry-lost-once',command)
        self.assertNotIn('--extend-campaign-hours',command)
        self.assertEqual(command[command.index('--campaign-hours')+1],'24')

    def test_old_r3_and_amended_process_handles_block_before_preparation(self):
        for prefix in ('E2_RECOVERY_V3','E2_GUIDE'):
            active=Mock();active.poll.return_value=None
            namespace={'RUN_SEARCH':True,'RECOVER_PENDING_ONCE':True,'RECOVERY_COMMIT':'b'*40,
                'EXPECTED_RECOVERY_AUTHORIZATION_SHA256':'c'*64,prefix+'_ACTIVE_PROCESS':active}
            with self.subTest(prefix=prefix),self.assertRaisesRegex(launcher.LauncherBlocked,'runner_already_active'):
                launcher.run_launcher(namespace)

    def test_original_r3_retry_flag_does_not_authorize_r4(self):
        with redirect_stdout(StringIO()):
            value=launcher.run_launcher({'RUN_SEARCH':True,'RETRY_LOST_ONCE':True})
        self.assertEqual(value['status'],'preview_only')

    def test_repair_only_stops_before_normal_selection(self):
        from types import SimpleNamespace
        import run_e2_cuda_recovery_v4 as controller
        session=Mock()
        session.pending_record_paths=()
        session.completed_job_ids=('recovered',)
        engine=SimpleNamespace(ledger=SimpleNamespace(SearchSession=Mock(return_value=session)),
            frozen=SimpleNamespace(run_selection=Mock(),write_json=Mock(),
                report=lambda plan,session,status,*args,**kwargs:{'status':status}))
        result=controller.continue_selection(SimpleNamespace(),engine,{},Path('cache'),Mock(),
            'unused_panel',200,clock=lambda:100,repair_only=True,accounted_through=100)
        self.assertEqual(result,(0,'recovery_completed',1))
        engine.frozen.run_selection.assert_not_called()
        session.close.assert_called_once()

    def test_published_authorization_binds_local_implementation(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import run_e2_cuda_recovery_v4 as controller
        path=ROOT/'experiments/e2_recovery_20261111_r4/authorization.json'
        raw,value,sha=controller.load_authorization(path,None,execute=True)
        self.assertEqual(value['previous_completed_jobs'],1801)
        self.assertEqual(value['target']['sequence'],3632)
        self.assertEqual(value['limits']['campaign_seconds'],172800)
        self.assertFalse(value['limits']['budget_extension_authorized'])

if __name__=='__main__':unittest.main()
