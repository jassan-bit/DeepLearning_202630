"""CPU-only launcher tests: no data download or model fitting."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('v3_launcher_tests_subject', Path(__file__).with_name('launch_colab.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class FakeStore:
    def __init__(self, fail=False):
        self.observed = []
        self.fail = fail

    def publish(self, output):
        self.observed.append(sorted(path.name for path in output.iterdir()))
        if self.fail:
            raise RuntimeError('simulated durability failure')
        return {'git_commit': 'a' * 40, 'run_url': 'local-test-only'}


class LauncherTests(unittest.TestCase):
    def test_child_environment_drops_credentials_and_unsafe_python_paths(self):
        secret = 'secret-fixture-never-published'
        with patch.dict(os.environ, {'GITHUB_TOKEN': secret, 'PAT': secret,
                                    'NESTED_SECRET_KEY': secret, 'UNUSUAL': secret,
                                    'PYTHONPATH': '/untrusted', 'GIT_ASKPASS': '/bad'}):
            env = launcher.environment({'DANGEROUS_TOKEN': secret, 'SAFE': 'yes'}, secret)
        self.assertNotIn(secret, env.values())
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('GIT_ASKPASS', env)
        self.assertEqual(env['SAFE'], 'yes')

    def test_existing_remote_run_blocks_implicit_fresh_campaign(self):
        run_id = '20261120T120000123456Z'
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, '_git', return_value=(
                'a' * 40 + '\trefs/heads/e3-v3-development-' + run_id + '\n').encode()):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                launcher.choose_run(temp, 'b' * 40, None)
            self.assertFalse((Path(temp) / 'e3-v3-development-session.json').exists())

    def test_local_session_reuses_original_identity_and_rejects_other_source(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(launcher, '_git', return_value=b''):
            run_id, restoring = launcher.choose_run(temp, 'b' * 40, None)
            self.assertFalse(restoring)
            self.assertEqual(launcher.choose_run(temp, 'b' * 40, None), (run_id, False))
            with self.assertRaisesRegex(ValueError, 'original source'):
                launcher.choose_run(temp, 'c' * 40, None)

    def test_hardware_profile_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'hardware_profile.json'
            launcher.atomic(path, {'GPU': 'A100-80GB'}, immutable=True)
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                launcher.atomic(path, {'GPU': 'A100-40GB'}, immutable=True)
            self.assertEqual(json.loads(path.read_text()), {'GPU': 'A100-80GB'})

    def test_real_sparse_checkout_has_quadrature_and_sidecar_without_model_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            remote, work = root / 'remote.git', root / 'work'
            git_env = launcher.environment()
            git_env.update(GIT_AUTHOR_NAME='Synthetic QA', GIT_COMMITTER_NAME='Synthetic QA',
                           GIT_AUTHOR_EMAIL='qa@example.invalid', GIT_COMMITTER_EMAIL='qa@example.invalid')
            def git(*args, cwd=None):
                return subprocess.check_output(['git', *map(str, args)], cwd=cwd, env=git_env,
                                               stderr=subprocess.DEVNULL).decode().strip()
            git('init', '--bare', '--quiet', remote)
            git('--git-dir', remote, 'config', 'uploadpack.allowFilter', 'true')
            git('init', '--quiet', work)
            names = ['src/original_model.py', launcher.SCOPE + '/pilot_engine.py',
                     *launcher.FROZEN_FILES, 'old_large_assets/model.pt', 'README.md']
            for name in names:
                path = work / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('synthetic software fixture only\n')
            git('add', '.', cwd=work)
            git('commit', '--quiet', '-m', 'Synthetic source with required quadrature', cwd=work)
            source_commit = git('rev-parse', 'HEAD', cwd=work)
            git('remote', 'add', 'origin', remote.as_uri(), cwd=work)
            git('push', '--quiet', 'origin', 'HEAD:refs/heads/main', cwd=work)
            with patch.object(launcher, 'REPOSITORY', remote.as_uri()):
                source = launcher.checkout(source_commit, root)
            for name in names[:-2]:
                self.assertTrue((source / name).is_file(), name)
            self.assertFalse((source / 'old_large_assets/model.pt').exists())
            self.assertFalse((source / 'README.md').exists())
            self.assertEqual(git('rev-parse', 'HEAD', cwd=source), source_commit)

    def worker(self, output, durability):
        return '''import json,pathlib,time
o=pathlib.Path(%r);d=pathlib.Path(%r)
(o/'before_fit.json').write_text('{}')
print(json.dumps({'event':'durability_barrier','barrier_id':'%s','stage':'before_fit'}),flush=True)
started=time.monotonic()
while not (d/'%s.json').exists():
 if time.monotonic()-started>10: raise RuntimeError('ACK timeout')
 time.sleep(.01)
ack=json.loads((d/'%s.json').read_text())
assert ack['remote_verified'] is True
(o/'simulated_fit_done.json').write_text('{}')
print(json.dumps({'event':'durability_barrier','barrier_id':'%s','stage':'after_job'}),flush=True)
while not (d/'%s.json').exists(): time.sleep(.01)
''' % (str(output), str(durability), 'a' * 64, 'a' * 64, 'a' * 64, 'b' * 64, 'b' * 64)

    def test_before_fit_and_after_job_wait_for_verified_remote_save(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            output, durability = source / 'output', source / 'acks'
            output.mkdir(); durability.mkdir()
            store = FakeStore()
            launcher.run_worker([sys.executable, '-u', '-c', self.worker(output, durability)],
                                source=source, env=launcher.environment(), output=output,
                                store=store, durability_dir=durability, token='')
            self.assertNotIn('simulated_fit_done.json', store.observed[0])
            self.assertIn('before_fit.json', store.observed[0])
            self.assertIn('simulated_fit_done.json', store.observed[1])
            self.assertEqual(len(store.observed), 3)

    def test_storage_failure_stops_worker_before_simulated_fit(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            output, durability = source / 'output', source / 'acks'
            output.mkdir(); durability.mkdir()
            with self.assertRaisesRegex(RuntimeError, 'durability failure'):
                launcher.run_worker([sys.executable, '-u', '-c', self.worker(output, durability)],
                                    source=source, env=launcher.environment(), output=output,
                                    store=FakeStore(fail=True), durability_dir=durability, token='')
            self.assertFalse((output / 'simulated_fit_done.json').exists())
            self.assertFalse(any(durability.iterdir()))


if __name__ == '__main__':
    unittest.main()
