"""Lifecycle tests: all scientific workers are synthetic; transport is native Git."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import bootstrap as b


CORE_PATH = Path('/workspace/work/e3_github_storage/checkout/scripts/e3_github_storage.py')


def plan_fixture():
    return {'jobs': [{'id': f'm{m}-{seed}', 'model': f'M{m}', 'seed': seed}
            for m in range(13) for seed in b.SEEDS], 'inputs': {'a': {'sha256': 'a' * 64}}}


def git(*args, cwd=None):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True).stdout.decode().strip()


class MemoryRemote:
    """Immutable snapshots and an injected lost response after an actual write."""
    def __init__(self, payloads):
        self.current = None
        self.count = 0
        self.fail_after_write = False
        self.publish(b.make_snapshot(payloads), payloads, None)

    def read(self):
        return self.current

    def publish(self, snapshot, payloads, expected):
        b.validate_snapshot(snapshot, payloads)
        if self.current and self.current[1:] == (snapshot, payloads):
            return self.current[0]
        if (self.current[0] if self.current else None) != expected:
            raise RuntimeError('conflict')
        self.count += 1
        head = f'{self.count:040x}'
        self.current = (head, snapshot, dict(payloads))
        if self.fail_after_write:
            self.fail_after_write = False
            raise ConnectionError('lost publication ACK')
        return head


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name)
        self.plan = plan_fixture()
        self.files = b.initial_files(self.plan, {'bootstrap.py': 'b'*64}, run_id='fixed-run')
        self.remote = MemoryRemote(self.files)
        self.calls = []
        self.on_execute = None

    def tearDown(self):
        self.temp.cleanup()

    def command(self, phase, job_id, claim_path):
        return phase, job_id, claim_path

    def executor(self, command, env, cap, log):
        phase, jid, claim_path = command
        claim = b.read_json(claim_path)
        manifest = b.read_json(self.workspace / 'run/run_manifest.json')
        rows = b.events_from((self.workspace / 'run/ledger.jsonl').read_bytes())
        self.assertEqual(claim['ledger_event_sha256'], rows[-1]['sha256'])
        self.assertEqual(claim['ack_commit'], self.remote.read()[0])
        self.assertEqual(claim['version'], 'e2-seed-extension-v1')
        self.assertEqual(self.remote.read()[2]['ledger.jsonl'], (self.workspace / 'run/ledger.jsonl').read_bytes())
        self.calls.append((phase, jid))
        directory = self.workspace / 'run'
        base = {'run_id': manifest['run_id'], 'plan_fingerprint': manifest['plan_fingerprint'],
            'status': 'completed', 'phase': phase, 'job_id': jid}
        if phase == 'fit':
            artifacts = []
            for name in ('checkpoint.pt', 'scaler.json', 'calibrator.json'):
                path = f'jobs/{jid}/{name}'
                data = (jid + name).encode()
                b.atomic(directory / path, data)
                artifacts.append({'path': path, 'sha256': b.sha(data), 'bytes': len(data)})
            value = {**base, 'checkpoint': 'checkpoint.pt', 'scaler': 'scaler.json',
                'calibrator': 'calibrator.json', 'artifacts': artifacts}
            b.write_json(directory / f'jobs/{jid}/fit_receipt.json', value)
        elif phase == 'evaluate':
            gate = b.read_json(directory / 'all_fits_remote_ack.json')
            self.assertEqual(len(gate['fits']), 52)
            self.assertEqual(sum(p == 'fit' for p, _ in self.calls), 52)
            self.assertEqual(sum('fit' in v for v in b.read_json(directory / 'state.json')['jobs'].values()), 52)
            value = {**base, 'fit_receipt_sha256': b.sha((directory / f'jobs/{jid}/fit_receipt.json').read_bytes()), 'n_dates': 608}
            b.write_json(directory / f'jobs/{jid}/evaluation.json', value)
        else:
            data = b'{}\n'
            b.atomic(directory / 'summary.json', data)
            value = {**base, 'artifacts': [{'path': 'summary.json', 'sha256': b.sha(data), 'bytes': len(data)}]}
            b.write_json(directory / 'aggregate_receipt.json', value)
        if self.on_execute:
            self.on_execute()
        return 'success', 1.0

    def run_coordinator(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return b.run_coordinator(self.workspace, self.plan, self.remote, ('python', {}), self.command,
                executor=self.executor, **kwargs)

    def test_complete_52_fits_gate_52_evaluations_and_aggregate(self):
        self.run_coordinator()
        self.assertEqual(len(self.calls), 105)
        state = b.read_json(self.workspace / 'run/state.json')
        ledger = b.read_json(self.workspace / 'run/ledger.json')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(ledger['charged_seconds'], 105.0)
        self.assertEqual(ledger['historic_e3_charged_seconds'], 948.439625042)
        self.assertEqual(ledger['limit_seconds'], 7200.0)
        self.assertEqual(len(list((self.workspace / 'run/remote_acks').glob('*.json'))), 52)
        self.run_coordinator()
        self.assertEqual(len(self.calls), 105)

    def test_lost_finish_ack_never_repeats_completed_fit(self):
        # Network fails after successful child and remote write, before local ACK.
        self.on_execute = lambda: setattr(self.remote, 'fail_after_write', True)
        with self.assertRaises(ConnectionError):
            self.run_coordinator(max_jobs=1)
        self.assertEqual(len(self.calls), 1)
        self.on_execute = None
        self.run_coordinator(max_jobs=1)
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(self.calls[0][1], self.calls[1][1])
        self.assertEqual(b.read_json(self.workspace / 'run/ledger.json')['charged_seconds'], 2.0)

    def test_failed_upload_before_write_republishes_finish_before_next_child(self):
        original = self.remote.publish
        flag = [False]
        def fail(snapshot, payloads, expected):
            if flag[0]:
                flag[0] = False
                raise ConnectionError('offline before write')
            return original(snapshot, payloads, expected)
        self.remote.publish = fail
        self.on_execute = lambda: flag.__setitem__(0, True)
        with self.assertRaises(ConnectionError):
            self.run_coordinator(max_jobs=1)
        self.on_execute = None
        self.run_coordinator(max_jobs=1)
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(self.calls[0][1], self.calls[1][1])

    def test_lost_active_reservation_charges_cap_and_halts(self):
        directory = self.workspace / 'run'
        b.restore(directory, self.files)
        b.write_json(self.workspace / 'owner.json', {'owner_id': 'owner'})
        head = self.remote.read()[0]
        b.append_event(directory, {'event': 'reserve', 'job_id': self.plan['jobs'][0]['id'], 'phase': 'fit',
            'reservation_id': 'r', 'cap_seconds': 1800.0, 'owner_id': 'owner',
            'lease_until_utc': b.utc().isoformat()})
        b.publish(self.remote, directory, self.workspace / 'ack.json', head)
        with self.assertRaisesRegex(RuntimeError, 'Unreconciled'):
            self.run_coordinator()
        self.assertEqual(self.calls, [])
        self.assertEqual(b.read_json(directory / 'ledger.json')['charged_seconds'], 1800.0)
        self.assertEqual(b.read_json(directory / 'state.json')['status'], 'unreconciled')
        self.run_coordinator()
        self.assertEqual(self.calls, [])

    def test_foreign_live_lease_stops_without_spending(self):
        directory = self.workspace / 'run'
        b.restore(directory, self.files)
        b.append_event(directory, {'event': 'reserve', 'job_id': self.plan['jobs'][0]['id'], 'phase': 'fit',
            'reservation_id': 'r', 'cap_seconds': 1800.0, 'owner_id': 'foreign',
            'lease_until_utc': (b.utc() + b.timedelta(seconds=1900)).isoformat()})
        b.publish(self.remote, directory, self.workspace / 'ack.json', self.remote.read()[0])
        with self.assertRaisesRegex(RuntimeError, 'Another VM'):
            self.run_coordinator()
        self.assertEqual(b.read_json(directory / 'ledger.json')['charged_seconds'], 0.0)
        self.assertEqual(self.calls, [])

    def test_torn_projection_recovered_from_successful_finish(self):
        self.run_coordinator(max_jobs=1)
        directory = self.workspace / 'run'
        b.write_json(directory / 'ledger.json', {'torn': True})
        b.write_json(directory / 'state.json', {'torn': True})
        self.run_coordinator(max_jobs=1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(b.read_json(directory / 'ledger.json')['charged_seconds'], 2.0)

    def test_budget_and_completed_state_cannot_reset(self):
        self.run_coordinator(max_jobs=1)
        payloads = dict(self.remote.read()[2])
        ledger = json.loads(payloads['ledger.json']); ledger['charged_seconds'] = 0
        payloads['ledger.json'] = b.canonical(ledger) + b'\n'
        with self.assertRaisesRegex(RuntimeError, 'ledger projection'):
            b.validate_snapshot(b.make_snapshot(payloads), payloads)
        payloads = dict(self.remote.read()[2])
        state = json.loads(payloads['state.json']); state['jobs'] = {}
        payloads['state.json'] = b.canonical(state) + b'\n'
        with self.assertRaisesRegex(RuntimeError, 'append-only'):
            b.validate_snapshot(b.make_snapshot(payloads), payloads)

    def test_worker_environment_strips_credentials(self):
        env = b.filtered_environment({'GITHUB_TOKEN': 'secret', 'GH_TOKEN': 's', 'API_KEY': 's',
            'GIT_ASKPASS': '/s', 'PATH': '/bin'})
        self.assertEqual(env, {'PATH': '/bin', 'PYTHONDONTWRITEBYTECODE': '1'})


class ActualPlanContractTests(unittest.TestCase):
    def test_current_plan_and_current_worker_claim_contract(self):
        import worker
        plan = b.read_json(b.HERE / "plan.json")
        self.assertEqual(b.sha(b.canonical(plan)), "5437fb3019a6b56e72b84928156b017b152b59bf1d04e33cfd82319a280c3845")
        hashes = {name: b.sha((b.HERE / name).read_bytes()) for name in ("bootstrap.py", "worker.py", "prepare.py", "aggregate.py")}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            b.restore(directory, b.initial_files(plan, hashes, run_id="actual-plan-synthetic-admission"))
            reservation = b.append_event(directory, {"event": "reserve", "job_id": plan["jobs"][0]["id"], "phase": "fit",
                "reservation_id": "actual-plan-reservation", "cap_seconds": 1800.0, "owner_id": "synthetic-owner",
                "lease_until_utc": (b.utc() + b.timedelta(seconds=1860)).isoformat()})
            manifest = b.read_json(directory / "run_manifest.json")
            claim = {"version": "e2-seed-extension-v1", "run_id": manifest["run_id"], "plan_fingerprint": manifest["plan_fingerprint"],
                "job_id": reservation["job_id"], "phase": "fit", "reservation_id": reservation["reservation_id"],
                "cap_seconds": 1800.0, "ledger_event_sha256": reservation["sha256"], "ack_commit": "a"*40}
            b.write_json(directory / "claim.json", claim)
            admitted_manifest, admitted_claim, job = worker.claim_context(directory, directory / "claim.json", reservation["job_id"], "fit")
            self.assertEqual((admitted_manifest, admitted_claim, job), (manifest, claim, plan["jobs"][0]))
            (directory / "claim.json").unlink()
            payloads = b.collect(directory)
            b.validate_snapshot(b.make_snapshot(payloads), payloads)

    def test_hard_timeout_returns_and_charges_real_child_wall(self):
        import sys
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "child.log"
            outcome, elapsed = b.execute_worker([sys.executable, "-B", "-c", "import time; print('starting', flush=True); time.sleep(10)"],
                dict(os.environ), 0.1, log)
            self.assertEqual(outcome, "timeout")
            self.assertGreaterEqual(elapsed, 0.09)
            self.assertLess(elapsed, 3.0)
            self.assertIn("starting", log.read_text())


@unittest.skipUnless(CORE_PATH.is_file(), 'pinned original Git core checkout unavailable')
class NativeGitTests(unittest.TestCase):
    def test_real_local_git_sparse_tree_conflict_and_close(self):
        self.assertEqual(b.sha(CORE_PATH.read_bytes()), b.CORE_SHA)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            bare, work = folder / 'remote.git', folder / 'work'
            git('init', '--bare', str(bare)); git('init', str(work))
            git('config', 'user.name', 'Test', cwd=work); git('config', 'user.email', 'test@example.test', cwd=work)
            (work / 'old.txt').write_text('preserved parent')
            git('add', '.', cwd=work); git('commit', '-m', 'parent', cwd=work)
            parent = git('rev-parse', 'HEAD', cwd=work)
            git('remote', 'add', 'origin', str(bare), cwd=work)
            git('push', 'origin', 'HEAD:main', cwd=work)
            core = b.load_private('test_core', CORE_PATH)
            with patch.object(b, 'INITIAL_PARENT', parent):
                client = b.NativeGit(core, 'test-secret-never-published', test_remote=bare)
                try:
                    files = b.initial_files(plan_fixture(), {'bootstrap.py': 'a'*64})
                    snap = b.make_snapshot(files)
                    head = client.publish(snap, files, None)
                    self.assertEqual(client.read(), (head, snap, files))
                    names = git('--git-dir=' + str(bare), 'ls-tree', '-r', '--name-only', head).splitlines()
                    self.assertTrue(all(n.startswith(b.PREFIX + '/') for n in names))
                    self.assertEqual(git('--git-dir=' + str(bare), 'rev-parse', head + '^'), parent)
                    self.assertEqual(git('--git-dir=' + str(bare), 'rev-parse', 'main'), parent)
                    self.assertEqual(client.publish(snap, files, None), head) # idempotent
                    changed = dict(files)
                    changed['runtime.json'] = b'{}\n'
                    with self.assertRaisesRegex(RuntimeError, 'writer conflict'):
                        client.publish(b.make_snapshot(changed), changed, None)
                finally:
                    client.close()
                self.assertEqual(client.token, '')
                self.assertTrue(client.transport._closed)


if __name__ == '__main__':
    unittest.main()
