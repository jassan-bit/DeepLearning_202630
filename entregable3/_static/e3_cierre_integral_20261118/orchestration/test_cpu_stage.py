"""Offline orchestration checks; no HTTP, browser installation or model fit."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('cpu_stage_under_test',Path(__file__).with_name('cpu_stage.py'))
stage=importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
SOURCE='a'*40


class CPUStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.out=self.root/'evidence'

    def tearDown(self):
        self.temporary.cleanup()

    def collect(self,output,source):
        self.assertEqual(source,SOURCE)
        output.mkdir(parents=True)
        (output/'admission.json').write_text('{"datasets_admitted":3}')
        return {'datasets_admitted':3}

    def setup(self,workspace,log):
        return Path('/isolated/python'),{}, {'packages':{'playwright':'1.56.0'}}

    def execute(self,cmd,cwd,log,env):
        target=Path(cmd[cmd.index('--output')+1])
        target.write_text(json.dumps({'public_deployment_smoke_passed':True}))

    def test_completed_stage_is_reused_without_collect_or_browser_rerun(self):
        first=stage.run(self.root,self.out,SOURCE,collector=self.collect,
                        browser_setup=self.setup,execute=self.execute)
        self.assertEqual(first['datasets_admitted'],3)
        self.assertTrue(first['public_deployment_smoke_passed'])
        self.assertEqual(first['new_real_data_fits'],0)
        raw=(self.out/'stage_receipt.json').read_bytes()
        def forbidden(*args):
            self.fail('A retry must preserve the completed stage')
        second=stage.run(self.root,self.out,SOURCE,collector=forbidden,browser_setup=forbidden,execute=forbidden)
        self.assertEqual(second,first)
        self.assertEqual((self.out/'stage_receipt.json').read_bytes(),raw)

    def test_health_or_process_success_does_not_become_public_acceptance(self):
        def weak(cmd,cwd,log,env):
            Path(cmd[cmd.index('--output')+1]).write_text('{"status":"ok","HTTP":200}')
        result=stage.run(self.root,self.out,SOURCE,collector=self.collect,browser_setup=self.setup,execute=weak)
        self.assertFalse(result['public_deployment_smoke_passed'])
        self.assertFalse(result['R1_accredited'])
        self.assertFalse(result['R3_accredited'])

    def test_browser_install_failure_preserves_data_and_failed_receipt(self):
        def blocked(*args):
            raise RuntimeError('synthetic_package_error')
        result=stage.run(self.root,self.out,SOURCE,collector=self.collect,browser_setup=blocked)
        self.assertEqual(result['datasets_admitted'],3)
        self.assertFalse(result['public_deployment_smoke_passed'])
        detail=json.loads((self.out/'render/public_smoke.json').read_bytes())
        self.assertEqual(detail['status'],'blocked_before_browser_acceptance')

    def test_data_failure_does_not_suppress_actual_browser_result(self):
        def blocked(*args):
            raise ValueError('synthetic_input_error')
        result=stage.run(self.root,self.out,SOURCE,collector=blocked,browser_setup=self.setup,execute=self.execute)
        self.assertEqual(result['datasets_admitted'],0)
        self.assertTrue(result['public_deployment_smoke_passed'])
        self.assertIn('synthetic_input_error',result['data_error'])

    def test_different_source_and_symlink_are_refused(self):
        stage.run(self.root,self.out,SOURCE,collector=self.collect,browser_setup=self.setup,execute=self.execute)
        with self.assertRaisesRegex(ValueError,'source_mismatch'):
            stage.run(self.root,self.out,'b'*40)
        link=self.root/'link'
        link.symlink_to(self.out,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlink'):
            stage.run(self.root,link,SOURCE)

    def test_child_environment_excludes_credentials_and_kernel_overrides(self):
        with patch.dict(os.environ,{'GITHUB_TOKEN':'test-secret','ANOTHER_PASSWORD':'test-secret',
                                   'PYTHONPATH':'bad','PIP_INDEX_URL':'bad','NORMAL_SETTING':'keep'}):
            env=stage.environment()
        self.assertNotIn('GITHUB_TOKEN',env)
        self.assertNotIn('ANOTHER_PASSWORD',env)
        self.assertNotIn('PYTHONPATH',env)
        self.assertNotIn('PIP_INDEX_URL',env)
        self.assertEqual(env['NORMAL_SETTING'],'keep')

    def test_browser_install_targets_a_pipless_venv_instead_of_the_kernel(self):
        commands=[]
        def command(args,*unused):
            commands.append([str(x) for x in args])
        with patch.object(stage,'streamed',side_effect=command), \
             patch.object(stage.subprocess,'run',side_effect=[SimpleNamespace(returncode=1),SimpleNamespace(returncode=0)]), \
             patch.object(stage.subprocess,'check_output',return_value='{"chromium":"synthetic-browser-version"}'):
            python,env,receipt=stage.prepare_browser(self.root/'isolated',self.root/'install.log')
        self.assertIn('--without-pip',commands[0])
        self.assertEqual(commands[1][1:5],['-m','pip','--python',str(python)])
        self.assertNotEqual(str(python),stage.sys.executable)
        self.assertEqual(env['PLAYWRIGHT_BROWSERS_PATH'],str(self.root/'isolated/browsers'))
        self.assertFalse(receipt['Torch_changed'])


if __name__=='__main__':
    unittest.main()
