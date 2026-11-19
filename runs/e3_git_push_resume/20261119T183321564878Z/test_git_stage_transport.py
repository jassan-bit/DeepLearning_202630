"""Offline real-Git contracts, including missing ancestral model blobs and races."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import git_stage_transport as m

PART = '20261119T170007583476Z'
TOKEN = 'not-a-real-token-DO-NOT-LOG-791af3'


def git(folder, *args, data=None, check=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_') and k != 'E3_GIT_PUSH_TOKEN'}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_LAZY_FETCH='1')
    value = subprocess.run(['git', '-C', str(folder), *args], input=data, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, env=env, check=check)
    return value.stdout.decode().strip() if check else value


class Fixture:
    def __init__(self, folder):
        self.folder, self.work = Path(folder), Path(folder) / 'work'
        self.work.mkdir()
        git(self.work, 'init', '-q')
        git(self.work, 'config', 'user.name', 'Offline verification')
        git(self.work, 'config', 'user.email', 'offline@example.invalid')
        old = self.work / 'runs/e3_extended_audit/20261119T165424079731Z'
        old.mkdir(parents=True)
        # A real missing old binary blob; it must remain absent after both pushes.
        self.old_raw = bytes(range(256)) * 1024
        (old / 'checkpoint.pt').write_bytes(self.old_raw)
        (self.work / 'runs/other_run.json').write_text('{"untouched": true}\n')
        (self.work / 'execute.sh').write_text('#!/bin/sh\nexit 0\n')
        (self.work / 'execute.sh').chmod(0o755)
        (self.work / 'symbolic').symlink_to('execute.sh')
        (self.work / 'a.c').write_text('preserved\n')
        git(self.work, 'add', '.')
        git(self.work, 'update-index', '--add', '--cacheinfo', '160000,' + 'a'*40 + ',vendor')
        git(self.work, 'commit', '-qm', 'Original immutable scientific state')
        self.parent = git(self.work, 'rev-parse', 'HEAD')
        self.base = git(self.work, 'rev-parse', 'HEAD^{tree}')
        self.old_sha = hashlib.sha1(b'blob ' + str(len(self.old_raw)).encode() + b'\0' + self.old_raw).hexdigest()
        self.remote = self.folder / 'remote.git'
        git(self.folder, 'clone', '--bare', '-q', str(self.work), str(self.remote))
        git(self.remote, 'config', 'uploadpack.allowFilter', 'true')
        self.url = self.remote.as_uri()
        self.files = {'stage_receipt.json': b'{"frozen":true}\n',
                      'payload/jobs/D03/checkpoint.pt': b'\0new-weights\xff' * 8192,
                      'payload/area/z.json': b'[]\n', 'payload/area.c': b'correct-order\n'}
        self.store = self.open_store()

    def open_store(self):
        return m.GitStageStore(self.folder/'recovery', self.parent, self.url, TOKEN)

    def payload(self, store=None):
        store = store or self.store
        return {'base_tree': self.base, 'tree': [
            {'path': 'runs/e3_extended_audit/'+PART+'/'+path, 'mode': '100644', 'type': 'blob',
             'sha': store.blob(raw)} for path, raw in self.files.items()]}

    def expected(self):
        for path, raw in self.files.items():
            target = self.work/'runs/e3_extended_audit'/PART/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        git(self.work, 'add', 'runs/e3_extended_audit/'+PART)
        return git(self.work, 'write-tree')

    def absent_old_blob(self):
        return not self.store._has(self.old_sha, 'blob')


class GitStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def test_exact_root_missing_old_blobs_two_pushes_and_unchanged_original(self):
        f = self.f
        self.assertTrue(f.absent_old_blob())
        payload = f.payload()
        expected = f.expected()
        root = f.store.stage_tree(payload)['sha']
        self.assertEqual(root, expected)
        before = {v['path']: v for v in f.store.read_tree(f.base)['tree']}
        after = {v['path']: v for v in f.store.read_tree(root)['tree']}
        for name in ('execute.sh', 'symbolic', 'vendor', 'a.c'):
            self.assertEqual(before[name], after[name])
        staged = f.store.publish(root, PART)
        audit = f.store.create_commit(root, f.parent, 'E3 extended audit frozen part '+PART, PART)['sha']
        ref = 'refs/heads/e3-extended-audit-'+PART
        f.store.publish_ref(ref, audit)
        self.assertEqual(git(f.remote, 'rev-parse', ref+'^{tree}'), expected)
        self.assertEqual(git(f.remote, 'rev-parse', ref+'^'), f.parent)
        self.assertEqual(git(f.remote, 'rev-parse', staged['ref']+'^{tree}'), expected)
        self.assertEqual(git(f.remote, 'rev-parse', 'HEAD'), f.parent)
        self.assertEqual(git(f.remote, 'show', f.parent+':runs/other_run.json'), '{"untouched": true}')
        self.assertTrue(f.absent_old_blob(), 'Packing/pushing must not lazy-fetch old model weights')
        self.assertEqual(len(list((f.folder/'recovery').glob('git-*.json'))), 2)
        self.assertEqual(git(f.remote, 'fsck', '--full', '--no-reflogs'), '')

    def test_frozen_parent_only_reachable_via_advanced_remote_head(self):
        f = self.f
        (f.work/'later.txt').write_text('Later remote state must remain intact\n')
        git(f.work, 'add', 'later.txt')
        git(f.work, 'commit', '-qm', 'Remote advanced after frozen parent')
        branch = git(f.work, 'symbolic-ref', 'HEAD')
        git(f.work, 'push', '-q', f.url, 'HEAD:'+branch)
        remote_head = git(f.remote, 'rev-parse', 'HEAD')
        fresh = m.GitStageStore(f.folder/'advanced-recovery', f.parent, f.url, TOKEN)
        self.assertFalse(fresh._has(remote_head, 'commit'))
        self.assertFalse(fresh._has(f.old_sha, 'blob'))
        ordinary_env = fresh._env
        def legacy_env():
            env = ordinary_env()
            env.pop('GIT_NO_LAZY_FETCH')
            return env
        # The promisor override also prevents lazy fetches on older Git versions.
        with patch.object(fresh, '_env', side_effect=legacy_env):
            root = fresh.stage_tree(f.payload(fresh))['sha']
            result = fresh.publish(root, PART)
        self.assertEqual(git(f.remote, 'rev-parse', result['ref']+'^{tree}'), root)
        self.assertEqual(git(f.remote, 'rev-parse', 'HEAD'), remote_head)
        self.assertFalse(fresh._has(f.old_sha, 'blob'))

    def test_restart_reuses_journal_sha_and_existing_exact_staging(self):
        f = self.f
        root = f.store.stage_tree(f.payload())['sha']
        staged = f.store.publish(root, PART)
        first = f.store.create_commit(root, f.parent, 'Exact original audit message\n', PART)['sha']
        saved = (f.folder/'recovery'/('git-audit-'+PART+'.json')).read_bytes()
        journal = json.loads(saved)
        recorded = datetime.strptime(journal['date'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        self.assertLess(abs((datetime.now(timezone.utc)-recorded).total_seconds()), 60)
        again = f.open_store()
        root2 = again.stage_tree(f.payload(again))['sha']
        with patch.object(again, '_git', wraps=again._git) as calls:
            reused = again.publish(root2, PART)
            self.assertTrue(reused['reused'])
            self.assertEqual(reused['sha'], staged['sha'])
            self.assertFalse(any('push' in v.args for v in calls.call_args_list))
        self.assertEqual(again.create_commit(root2, f.parent, 'Exact original audit message\n', PART)['sha'], first)
        self.assertEqual((f.folder/'recovery'/('git-audit-'+PART+'.json')).read_bytes(), saved)
        ref = 'refs/heads/e3-extended-audit-'+PART
        again.publish_ref(ref, first)
        again.publish_ref(ref, first)
        self.assertTrue(f.absent_old_blob())
        with self.assertRaises(ValueError):
            again.create_commit(root2, f.parent, 'Changed message', PART)
        self.assertEqual((f.folder/'recovery'/('git-audit-'+PART+'.json')).read_bytes(), saved)

    def test_existing_staging_reuses_exact_tree_parent_with_different_commit_message(self):
        f = self.f
        root = f.store.stage_tree(f.payload())['sha']
        other = f.store.create_commit(root, f.parent, 'Earlier transport staging message', PART, role='staging')['sha']
        f.store.publish_ref('refs/heads/e3-git-stage-'+PART, other)
        result = f.store.publish(root, PART)
        self.assertTrue(result['reused'])
        self.assertEqual(result['sha'], other)
        self.assertEqual(json.loads((f.folder/'recovery'/('git-staging-'+PART+'.json')).read_text())['message'],
                         'Earlier transport staging message')

    def test_existing_branch_conflict_and_main_ref_are_never_updated(self):
        f = self.f
        root = f.store.stage_tree(f.payload())['sha']
        ref = 'refs/heads/e3-git-stage-'+PART
        git(f.remote, 'update-ref', ref, f.parent)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            f.store.publish(root, PART)
        self.assertEqual(git(f.remote, 'rev-parse', ref), f.parent)
        sha = f.store.create_commit(root, f.parent, 'audit', PART)['sha']
        with self.assertRaises(ValueError):
            f.store.publish_ref('refs/heads/main', sha)
        audit_ref = 'refs/heads/e3-extended-audit-'+PART
        git(f.remote, 'update-ref', audit_ref, f.parent)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            f.store.publish_ref(audit_ref, sha)
        self.assertEqual(git(f.remote, 'rev-parse', audit_ref), f.parent)

    def test_concurrent_create_lease_rejects_even_fast_forwardable_branch(self):
        f = self.f
        root = f.store.stage_tree(f.payload())['sha']
        commit = f.store.create_commit(root, f.parent, 'audit race', PART)['sha']
        ref = 'refs/heads/e3-extended-audit-'+PART
        real = f.store._git
        def race(*args, **kwargs):
            if 'push' in args:
                git(f.remote, 'update-ref', ref, f.parent)
            return real(*args, **kwargs)
        with patch.object(f.store, '_git', side_effect=race):
            with self.assertRaises(RuntimeError):
                f.store.publish_ref(ref, commit)
        self.assertEqual(git(f.remote, 'rev-parse', ref), f.parent)
        self.assertTrue(f.absent_old_blob())

    def test_lost_push_reply_is_reconciled_without_rewriting_remote(self):
        f = self.f
        root = f.store.stage_tree(f.payload())['sha']
        sha = f.store.create_commit(root, f.parent, 'audit lost response', PART)['sha']
        ref = 'refs/heads/e3-extended-audit-'+PART
        real = f.store._git
        def lost(*args, **kwargs):
            result = real(*args, **kwargs)
            if 'push' in args:
                raise RuntimeError('Simulated lost successful response')
            return result
        with patch.object(f.store, '_git', side_effect=lost):
            self.assertEqual(f.store.publish_ref(ref, sha)['object']['sha'], sha)
        self.assertEqual(git(f.remote, 'rev-parse', ref), sha)

    def test_auth_is_ephemeral_env_only_and_local_commands_cannot_lazy_fetch(self):
        f = self.f
        original_run = subprocess.run
        calls = []
        def observe(cmd, **kwargs):
            env = kwargs['env']
            self.assertFalse(any(k in env for k in ('GH_TOKEN', 'GITHUB_TOKEN', 'GITHUB_PAT',
                                                       'CUSTOM_CREDENTIAL', 'AWS_SECRET_ACCESS_KEY',
                                                       'SERVICE_TOKEN', 'PRIVATE_PAT')))
            self.assertNotIn(TOKEN, '\n'.join(cmd))
            self.assertEqual(env['GIT_NO_LAZY_FETCH'], '1')
            if 'E3_GIT_PUSH_TOKEN' in env:
                self.assertEqual(env['E3_GIT_PUSH_TOKEN'], TOKEN)
                self.assertTrue(any(v in cmd for v in ('fetch', 'push', 'ls-remote')))
                script = Path(env['GIT_ASKPASS'])
                self.assertEqual(stat.S_IMODE(script.stat().st_mode), 0o700)
                self.assertNotIn(TOKEN, script.read_text())
            else:
                self.assertNotIn('GIT_ASKPASS', env)
            calls.append((tuple(cmd), 'E3_GIT_PUSH_TOKEN' in env))
            return original_run(cmd, **kwargs)
        with patch.dict(os.environ, {'E3_GIT_PUSH_TOKEN': TOKEN, 'GH_TOKEN': TOKEN, 'GITHUB_TOKEN': TOKEN,
                                     'GITHUB_PAT': TOKEN, 'CUSTOM_CREDENTIAL': 'other-credential',
                                     'AWS_SECRET_ACCESS_KEY': 'other-secret', 'SERVICE_TOKEN': 'other-token',
                                     'PRIVATE_PAT': 'other-pat', 'GIT_CONFIG_COUNT': '1',
                                     'GIT_CONFIG_KEY_0': 'user.name', 'GIT_CONFIG_VALUE_0': 'Injected'}):
            with patch.object(m.subprocess, 'run', side_effect=observe):
                store = m.GitStageStore(f.folder/'auth-recovery', f.parent, f.url, TOKEN)
                root = store.stage_tree(f.payload(store))['sha']
                store.publish(root, PART)
        self.assertTrue(any('fetch' in cmd and auth for cmd, auth in calls))
        self.assertTrue(any('push' in cmd and auth for cmd, auth in calls))
        self.assertFalse(list((f.folder/'auth-recovery').glob('git-askpass-*')))
        for path in (f.folder/'auth-recovery').rglob('*'):
            if path.is_file():
                self.assertNotIn(TOKEN.encode(), path.read_bytes(), str(path))
        self.assertNotIn(TOKEN, (f.folder/'auth-recovery/git-stage-cache.git/config').read_text())

    def test_bounded_receipt_blob_fetches_only_known_exact_object_and_verifies_content(self):
        f = self.f
        sha = git(f.work, 'rev-parse', f.parent+':runs/other_run.json')
        self.assertFalse(f.store._has(sha, 'blob'))
        with patch.object(f.store, '_git', wraps=f.store._git) as calls:
            self.assertEqual(f.store.read_blob(sha), b'{"untouched": true}\n')
            fetches = [v.args for v in calls.call_args_list if 'fetch' in v.args]
            self.assertEqual(len(fetches), 1)
            self.assertIn('--no-filter', fetches[0])
            self.assertIn('--no-write-fetch-head', fetches[0])
            self.assertEqual(fetches[0][-2:], ('origin', sha))
        self.assertTrue(f.absent_old_blob())
        with patch.object(f.store, '_git', wraps=f.store._git) as calls:
            self.assertEqual(f.store.read_blob(sha), b'{"untouched": true}\n')
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                f.store.read_blob(sha, max_bytes=1)
            with self.assertRaisesRegex(ValueError, 'not a Git blob'):
                f.store.read_blob(f.parent)
            with self.assertRaisesRegex(ValueError, 'verified local commit tree'):
                f.store.read_blob('f'*40)
            self.assertFalse(any('fetch' in v.args for v in calls.call_args_list))
        real = f.store._git
        def corrupt(*args, **kwargs):
            return b'corrupt' if args == ('cat-file', 'blob', sha) else real(*args, **kwargs)
        with patch.object(f.store, '_git', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'Git SHA differs'):
                f.store.read_blob(sha)

    def test_symlink_base_cache_and_journal_are_rejected_before_mutation(self):
        f = self.f
        alias = f.folder/'base-link'
        alias.symlink_to(f.folder/'recovery', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            m.GitStageStore(alias, f.parent, f.url, TOKEN)
        bad = f.folder/'linked-cache-base'
        bad.mkdir()
        (bad/'git-stage-cache.git').symlink_to(f.store.cache_path, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            m.GitStageStore(bad, f.parent, f.url, TOKEN)
        root = f.store.stage_tree(f.payload())['sha']
        target = f.folder/'external-journal.json'
        target.write_text('{"preserved":true}\n')
        journal = f.folder/'recovery'/('git-audit-'+PART+'.json')
        journal.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            f.store.create_commit(root, f.parent, 'audit', PART)
        self.assertEqual(target.read_text(), '{"preserved":true}\n')

    def test_wrong_base_missing_blob_and_payload_mutation_fail_before_tree_changes(self):
        f = self.f
        payload = f.payload()
        wrong = deepcopy(payload)
        wrong['base_tree'] = 'f'*40
        with self.assertRaises(ValueError):
            f.store.stage_tree(wrong)
        wrong = deepcopy(payload)
        wrong['tree'][0]['sha'] = 'f'*40
        with self.assertRaisesRegex(ValueError, 'Every frozen'):
            f.store.stage_tree(wrong)
        wrong = deepcopy(payload)
        wrong['tree'][0]['path'] = 'arbitrary/output.bin'
        with self.assertRaises(ValueError):
            f.store.stage_tree(wrong)
        self.assertFalse(f.store._roots)
        self.assertFalse(f.store._trees)
        root = f.store.stage_tree(payload)['sha']
        with self.assertRaises(ValueError):
            f.store.create_commit(root, f.parent, 'mismatched part', '20261119T165424079731Z')
        with self.assertRaises(ValueError):
            f.store.publish(root, '20261119T165424079731Z')
        with self.assertRaises(ValueError):
            m.GitStageStore(f.folder/'bad', f.parent, 'https://user:'+TOKEN+'@github.com/a/b.git', TOKEN)


if __name__ == '__main__':
    unittest.main(verbosity=2)
