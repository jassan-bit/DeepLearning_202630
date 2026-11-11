"""Stdlib-only checks for diagnostic isolation, errors, and cleanup."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent/'scripts/diagnose_e2_optimized_selection.py'
if not SCRIPT.is_file():
    SCRIPT = HERE/'diagnose_e2_optimized_selection.py'
spec = importlib.util.spec_from_file_location('e2_diagnostic_test_subject', SCRIPT)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class DiagnosticTests(unittest.TestCase):
    @contextmanager
    def fake_runtime(self, *, profile_error=False, validation_error=False):
        calls = []
        restores = []
        mirrors = []
        sessions = []
        legacy_plan = {'source_sha256': {'scripts/frozen.py': 'a'*64},
                       'initial_controls': {'campaign_budget_seconds': 86400},
                       'coverage': {'selected_origins': 11}, 'models': ['TimesNet']}
        hardware = {'device_name': 'Tesla T4'}
        profile = {'fast_cases': [[16,16]], 'synthetic_only': True}
        registration = {'legacy_min_commit': 'b'*40,
                        'implementation_sha256': {'scripts/optimized.py': 'c'*64}}
        opt = types.ModuleType('run_e2_optimized_selection_v2')
        opt.REPOSITORY = 'owner/repo'
        opt.BRANCH = 'campaigns/e2-cuda-crop-11-v2'
        opt.OLD_BRANCH = 'campaigns/e2-cuda-selection-v1'
        opt.VERSION = 'e2-cuda-crop-plan-v2'
        opt.ORIGINS = ['2024-01-01']
        opt.LEGACY_PLAN = digest(legacy_plan)
        opt.digest = digest
        opt.read_registration = lambda: (registration, 'd'*64)
        runner = types.SimpleNamespace(cuda_hardware=lambda: deepcopy(hardware))
        opt.load_engine = lambda root: runner
        opt.install_public_schema = lambda module: calls.append('install_schema')
        opt.require_no_previous_optimization = lambda mirror: calls.append('previous_checked')
        opt.validate_profile = lambda value: calls.append('profile_validated')

        public = types.ModuleType('e2_github_campaign')
        def validate(value):
            calls.append('plan_validated')
            if validation_error:
                raise ValueError('public_plan_validation_failed')
        public._validate_plan = validate
        crop = types.ModuleType('e2_timesnet_exact_crop')
        def benchmark(**kwargs):
            calls.append('synthetic_profile')
            self.assertEqual(kwargs, {'device':'cuda:0','repeats':5})
            if profile_error:
                raise RuntimeError('synthetic_cuda_profile_numeric_disagreement')
            return deepcopy(profile)
        crop.benchmark = benchmark

        class FakeMirror:
            def __init__(self, repository, branch, prefix):
                self.branch, self.closed = branch, False
                self._head = 'b'*40
                self.git_transport = types.SimpleNamespace(_run=lambda *a,**k: types.SimpleNamespace(returncode=0))
                mirrors.append(self)
            def restore(self, path):
                path = Path(path)
                restores.append((self.branch, path))
                if self.branch == opt.OLD_BRANCH:
                    path.mkdir(parents=True)
                    (path/'campaign.json').write_text(json.dumps({'plan':legacy_plan}))
            def close(self):
                self.closed = True
            def push(self, *args, **kwargs):
                raise AssertionError('REMOTE WRITE ATTEMPTED')
        transport = types.ModuleType('e2_github_git_transport')
        transport.GitCampaignMirror = FakeMirror
        class FakeSession:
            def __init__(self, directory, plan, **kwargs):
                self.directory = Path(directory)
                self.closed = False
                self.campaign_budget_seconds = 172800
                self.charged_wall_seconds = 160000
                self.completed_job_ids = set(range(1802))
                sessions.append(self)
            def close(self):
                self.closed = True
            def _start_session(self):
                raise AssertionError('SESSION START ATTEMPTED')
            def run_job(self, *args, **kwargs):
                raise AssertionError('TRAINING ATTEMPTED')
        session_module = types.ModuleType('e2_search_session')
        session_module.SearchSession = FakeSession
        modules = {m.__name__:m for m in (opt,public,crop,transport,session_module)}
        with tempfile.TemporaryDirectory() as root:
            args = types.SimpleNamespace(controller_root=Path(root),engine_root=Path(root)/'engine',controller_commit='e'*40)
            result = subprocess.CompletedProcess([],0,'e'*40+'\n','')
            with patch.dict(sys.modules,modules), patch.dict(diag.os.environ,{'GITHUB_TOKEN':'test-secret'}), \
                    patch.object(diag.subprocess,'run',return_value=result), patch.object(sys,'path',list(sys.path)):
                yield args,calls,restores,mirrors,sessions

    def test_success_has_no_market_execution_or_writes_and_cleans_all_temporaries(self):
        with self.fake_runtime() as (args,calls,restores,mirrors,sessions):
            output=io.StringIO()
            with redirect_stdout(output):
                diag.diagnose(args)
            state=json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(state['diagnostic'],'passed_pre_seal_checks')
            self.assertFalse(state['market_training_started'])
            self.assertFalse(state['campaign_session_started'])
            self.assertFalse(state['remote_writes'])
            self.assertEqual(state['remaining_seconds'],12800)
            self.assertEqual(state['previous_completed_jobs'],1802)
            self.assertEqual(len(mirrors),3)
            self.assertTrue(all(m.closed for m in mirrors))
            self.assertTrue(all(s.closed for s in sessions))
            self.assertTrue(all(not p.exists() for _,p in restores))

    def test_profile_failure_has_precise_traceback_and_never_restores_legacy(self):
        with self.fake_runtime(profile_error=True) as (args,calls,restores,mirrors,sessions):
            argv=['diagnostic','--controller-root',str(args.controller_root),'--engine-root',str(args.engine_root),
                  '--controller-commit',args.controller_commit]
            output=io.StringIO()
            with patch.object(sys,'argv',argv), redirect_stdout(output):
                code=diag.main()
            self.assertEqual(code,2)
            self.assertIn('RuntimeError: synthetic_cuda_profile_numeric_disagreement',output.getvalue())
            self.assertNotIn('test-secret',output.getvalue())
            self.assertEqual(len(mirrors),2)
            self.assertTrue(all(m.closed for m in mirrors))
            self.assertTrue(all('selection-v1' not in branch for branch,_ in restores))
            self.assertEqual(sessions,[])
            self.assertTrue(all(not p.exists() for _,p in restores))

    def test_final_plan_validation_failure_closes_replayed_legacy_and_mirrors(self):
        with self.fake_runtime(validation_error=True) as (args,calls,restores,mirrors,sessions):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'public_plan_validation_failed'):
                diag.diagnose(args)
            self.assertEqual(len(sessions),1)
            self.assertTrue(sessions[0].closed)
            self.assertTrue(all(m.closed for m in mirrors))
            self.assertTrue(all(not p.exists() for _,p in restores))

    def test_diagnostic_source_has_no_execution_or_publication_call(self):
        tree=ast.parse((SCRIPT).read_text())
        forbidden={'push','_start_session','run_job','run_selection','isolated_trial','prepare_panel','execute'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else None
                self.assertNotIn(name,forbidden)


if __name__=='__main__':
    unittest.main()
