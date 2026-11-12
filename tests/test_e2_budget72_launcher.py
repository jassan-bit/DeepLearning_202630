"""Draft continuation tests; no GPU, network, funding or market execution."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load('funding_launcher_tested', 'notebooks/run_e2_reduced_github72_colab.py')
driver = load('funding_driver_tested', 'scripts/run_e2_reduced_github72_delivery.py')


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, count): return b'changed remote source'


class FundingLauncherTests(unittest.TestCase):
    def test_changed_base_source_cannot_be_executed(self):
        with patch.object(launcher.urllib.request, 'urlopen', return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, 'bytes changed'):
                launcher._base_launcher()

    def test_continuation_routes_same_base_engine_and_local_target_without_signal(self):
        calls = []
        base = {'CONTROLLERS': set(), '_stream': lambda command, *a: calls.append(command) or 0}
        def run(commit, **kwargs):
            calls.append((commit, kwargs))
            return base['_stream'](['python', Path('/content/extension/scripts/run_e2_reduced_github_delivery.py'), '--execute'])
        base['run_reduced_github_colab'] = run
        namespace = {'GPU_PYTHON': '/existing/isolated/python'}
        with patch.object(launcher, '_base_launcher', return_value=base):
            self.assertEqual(launcher.run_reduced_github72_colab('a' * 40, namespace=namespace, execute=True), 0)
        self.assertEqual(calls[0], ('a' * 40, {'namespace': namespace, 'execute': True}))
        self.assertEqual(Path(calls[1][1]).name, 'run_e2_reduced_github72_delivery.py')
        self.assertEqual(base['EXTENSION_REGISTRATION'], launcher.EXTENSION_REGISTRATION)
        self.assertIn('scripts/e2_registered_budget_extension.py', base['EXTENSION_FILES'])
        self.assertIn('run_e2_reduced_github72_delivery.py', base['CONTROLLERS'])
        self.assertNotIn('drive', str(calls).lower())
        code = (ROOT / 'notebooks/run_e2_reduced_github72_colab.py').read_text()
        self.assertNotIn('os.kill', code)
        self.assertNotIn('/content/drive', code)

    def test_extension_tamper_is_rejected_without_replacing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = ['scripts/run_e2_reduced_github72_delivery.py', 'scripts/e2_registered_budget_extension.py',
                     'scripts/e2_reduced_github_state.py', 'scripts/e2_reduced_github_restore.py',
                     'scripts/reconcile_e2_github_completed.py']
            hashes = {}
            for name in paths:
                p = root / name; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(name.encode()); hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            record = {'protocol_id': 'e2-reduced-delivery-v1', 'engine_commit': driver.ENGINE_COMMIT,
                      'files_sha256': hashes, 'funding_policy': {'status': 'draft'}}
            (root / 'registration.json').write_text(json.dumps(record))
            self.assertEqual(driver._extension_root(root, 'a' * 40), (root, record))
            changed = root / paths[1]; changed.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source differs'):
                driver._extension_root(root, 'a' * 40)
            self.assertEqual(changed.read_bytes(), b'changed')

    def test_funding_then_remote_restore_precede_ledger_and_science(self):
        calls = []
        args = SimpleNamespace(execute=True, stage='all', output=Path('/temporary'),
            data_root='dev', source_roots='roots', runtime_profile='runtime', budget_evidence='budget',
            migration='migration', cache_audit='audit', reconciliation_root='recon',
            migration_registration_commit=driver.ENGINE_COMMIT, inherited_ledger=[])
        scientific = SimpleNamespace(parser=lambda: SimpleNamespace(parse_args=lambda unused: args),
            read_json=lambda p: {'file': p}, execute_delivery=lambda *a: calls.append('execute'))
        class Budget:
            _closed = False
            def report(self): return {'reserved_seconds': 0}
            def charge_overhead(self, *a, **kw): calls.append('idle_flushed')
        class Backup:
            def __init__(self): calls.append('client')
            def preflight(self): calls.append('preflight')
        class State:
            def __init__(self, *a): pass
            def restore(self): calls.append('restore')
            @contextmanager
            def budget_factory(self, *a, budget_class=None, **kw):
                calls.append('ledger'); value = budget_class('/budget', {})
                yield value
                calls.append('closed')
            def snapshot(self, *a, **kw): calls.append('snapshot')
        def factory(policy, **kw):
            calls.append('funding_validated')
            self.assertEqual(policy, {'status': 'approved_fixture'})
            self.assertEqual(kw, {'funding_registration_commit': 'a' * 40, 'funding_git_root': Path('/engine')})
            return lambda *a, **kw: Budget()
        modules = {'run_e2_reduced_delivery': scientific,
            'e2_reduced_protocol': SimpleNamespace(verify_reduced_registration=lambda *a, **kw: None),
            'e2_reduced_github_backup': SimpleNamespace(GitHubBackup=Backup),
            'e2_reduced_github_state': SimpleNamespace(GitHubState=State),
            'e2_registered_budget_extension': SimpleNamespace(make_funded_budget_class=factory),
            'reconcile_e2_github_completed': SimpleNamespace(
                reconcile_completed_local=lambda *args: calls.append('reconcile'))}
        argv = ['--engine-root', '/engine', '--extension-root', '/extension', '--extension-commit', 'a' * 40]
        record = {'funding_policy': {'status': 'approved_fixture'}}
        with patch.dict(sys.modules, modules), patch.object(driver, '_engine_root', return_value=Path('/engine')), \
                patch.object(driver, '_extension_root', return_value=(Path('/extension'), record)), \
                patch.object(driver.subprocess, 'run') as fetch:
            self.assertEqual(driver.main(argv), 0)
            self.assertEqual(fetch.call_args.args[0], ['git', '-C', '/engine', 'fetch', '--filter=blob:none',
                '--depth', '1', 'origin', 'a' * 40])
            self.assertLess(calls.index('funding_validated'), calls.index('preflight'))
            self.assertLess(calls.index('preflight'), calls.index('reconcile'))
            self.assertLess(calls.index('reconcile'), calls.index('restore'))
            self.assertLess(calls.index('restore'), calls.index('ledger'))
            self.assertLess(calls.index('ledger'), calls.index('execute'))
            calls.clear()
            def broken_completed_backup(*args):
                calls.append('reconcile')
                raise ValueError('completed local weights cannot be verified')
            modules['reconcile_e2_github_completed'] = SimpleNamespace(
                reconcile_completed_local=broken_completed_backup)
            with patch.dict(sys.modules, modules):
                self.assertEqual(driver.main(argv), 3)
                self.assertEqual(calls, ['funding_validated', 'client', 'preflight', 'reconcile'])
            calls.clear()
            modules['e2_registered_budget_extension'] = SimpleNamespace(
                make_funded_budget_class=lambda *a, **kw: (_ for _ in ()).throw(ValueError('draft not authorized')))
            with patch.dict(sys.modules, modules):
                self.assertEqual(driver.main(argv), 3)
                self.assertEqual(calls, [])

    def test_exact_restored_result_skips_only_duplicate_backup_and_snapshot(self):
        calls = []
        class Base:
            def persist_result(self, *args):
                calls.append("regular_backup")
                return {"status": "checked_by_base"}
        cls = driver.funded_state_class(Base, lambda *a, **kw: None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); job = root / "jobs/one"; job.mkdir(parents=True)
            receipt = {"status": "completed", "worker_spec_sha256": "a" * 64}
            raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            (job / "result.json").write_bytes(raw)
            previous = {"status": "backed_up", "manifest_sha256": "b" * 64}
            state = cls(); state.output = root; state._restored = True; state._pending = {}
            state._backups = {"jobs/one": previous}
            state._files = {"jobs/one/result.json": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
            self.assertEqual(state.persist_result(job, receipt), previous)
            self.assertEqual(state.persist_result(job, {"result": receipt}), previous)
            self.assertEqual(calls, [])
            # A valid JSON result with changed bytes cannot use the fast path.
            (job / "result.json").write_bytes(raw + b" ")
            self.assertEqual(state.persist_result(job, receipt), {"status": "checked_by_base"})
            self.assertEqual(calls, ["regular_backup"])
            calls.clear(); (job / "result.json").write_bytes(raw)
            changed = dict(receipt, status="failed")
            state.persist_result(job, changed)
            self.assertEqual(calls, ["regular_backup"])
            calls.clear(); state._pending = {"live": {}}
            state.persist_result(job, receipt)
            self.assertEqual(calls, ["regular_backup"])
            calls.clear(); state._pending = {}; state._restored = False
            state.persist_result(job, receipt)
            self.assertEqual(calls, ["regular_backup"])

    def test_idle_cost_is_flushed_but_reserved_or_closed_budget_is_not_double_charged(self):
        calls = []
        class Base:
            def snapshot(self, reason, **kw): calls.append('snapshot'); return reason
        class Budget:
            _closed = False
            reserved = 0
            def report(self): return {'reserved_seconds': self.reserved}
            def charge_overhead(self, *a, **kw): calls.append('idle_flushed')
        value = driver.funded_state_class(Base, lambda *a, **kw: None)()
        live = Budget(); value._live_funding_budget = live
        self.assertEqual(value.snapshot('idle'), 'idle')
        self.assertEqual(calls, ['idle_flushed', 'snapshot'])
        calls.clear(); live.reserved = 660
        value.snapshot('worker_active'); self.assertEqual(calls, ['snapshot'])
        calls.clear(); live.reserved = 0; live._closed = True
        value.snapshot('closed'); self.assertEqual(calls, ['snapshot'])


if __name__ == '__main__': unittest.main()
