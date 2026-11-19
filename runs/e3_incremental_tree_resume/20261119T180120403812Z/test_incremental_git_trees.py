"""Offline Merkle and incremental Git tree contracts; no network or model fits."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('incremental_git_trees', HERE/'incremental_git_trees.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PART = '20261119T170007583476Z'
OLD = '20261119T165424079731Z'


def row(name, sha, mode='100644'):
    return {'path': name, 'mode': mode, 'type': m.MODES[mode], 'sha': sha}


class Git:
    def __init__(self):
        self.db = {}
        self.posts = []
        self.calls = []
        self.only_sha = False
        self.fail_after_create = None
        self.bad_post_sha = False

    def put(self, rows):
        sha = m.tree_sha(rows)
        self.db[sha] = deepcopy(rows)
        return sha

    def request(self, method, path, token, payload=None, *, allow_missing=False):
        self.calls.append((method, path))
        if method == 'GET':
            assert path.startswith(m.ENDPOINT+'/')
            sha = path.rsplit('/', 1)[1]
            if sha not in self.db:
                if allow_missing:
                    return None
                raise AssertionError('Missing base tree')
            return {'sha': sha, 'truncated': False, 'tree': deepcopy(self.db[sha])}
        assert method == 'POST' and path == m.ENDPOINT and set(payload) == {'tree'}
        self.posts.append(deepcopy(payload))
        sha = self.put(payload['tree'])
        if self.fail_after_create == len(self.posts):
            self.fail_after_create = None
            raise OSError('Simulated response lost after immutable object creation')
        if self.bad_post_sha:
            return {'sha': 'f'*40}
        if self.only_sha:
            return {'sha': sha}
        return {'sha': sha, 'truncated': False, 'tree': deepcopy(self.db[sha])}


def fixture():
    git = Git()
    leaf = git.put([row('known.json', '1'*40)])
    audit = git.put([row(OLD, leaf, '040000')])
    runs = git.put([row('e3_extended_audit', audit, '040000'), row('other_run.json', '2'*40)])
    base = git.put([row('runs', runs, '040000'), row('execute.sh', '3'*40, '100755'),
                    row('symbolic', '4'*40, '120000'), row('vendor', '5'*40, '160000'),
                    row('a.c', '6'*40)])
    paths = ['stage_receipt.json', 'storage_manifest.json', 'payload/jobs/D03/model/supervised/checkpoint.pt',
             'payload/jobs/D03/model/supervised/curve.json', 'payload/jobs/D03/model/result.json',
             'payload/área/z.json', 'payload/área.c']
    entries = [row('runs/e3_extended_audit/'+PART+'/'+name, hashlib.sha1(name.encode()).hexdigest())
               for name in paths]
    return git, {'base_tree': base, 'tree': entries}


def git_mktree(folder, rows):
    entries = []
    for entry in rows:
        entries.append((entry['mode']+' '+entry['type']+' '+entry['sha']+'\t'+entry['path']).encode()+b'\0')
    return subprocess.check_output(['git', '-C', str(folder), 'mktree', '--missing', '-z'],
                                   input=b''.join(entries)).decode().strip()


def monolithic_reference(git, base, flat, folder):
    """Independent recursive application, hashing each result with real git."""
    def change(sha, additions):
        current = {r['path']: deepcopy(r) for r in git.db.get(sha, [])}
        nested = {}
        for parts, record in additions:
            if len(parts) == 1:
                current[parts[0]] = row(parts[0], record['sha'], record['mode'])
            else:
                nested.setdefault(parts[0], []).append((parts[1:], record))
        for name, more in nested.items():
            child = current.get(name, {}).get('sha')
            current[name] = row(name, change(child, more), '040000')
        entries = list(current.values())
        result = git_mktree(folder, entries)
        git.db[result] = entries
        return result
    return change(base, [(r['path'].split('/'), r) for r in flat])


class IncrementalTests(unittest.TestCase):
    def test_exact_root_matches_real_git_monolithic_and_preserves_siblings(self):
        git, payload = fixture()
        original = deepcopy(git.db)
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(['git', 'init', '-q', folder], check=True)
            expected = monolithic_reference(git, payload['base_tree'], payload['tree'], folder)
        # Remove precreated reference trees: the incremental path must build them.
        git.db = original
        result = m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertEqual(result['sha'], expected)
        root = {r['path']: r for r in git.db[result['sha']]}
        before = {r['path']: r for r in original[payload['base_tree']]}
        for name in set(before)-{'runs'}:
            self.assertEqual(root[name], before[name])
        self.assertTrue(git.posts)
        self.assertTrue(all(set(p) == {'tree'} and all('/' not in r['path'] for r in p['tree']) for p in git.posts))
        self.assertEqual({k: git.db[k] for k in original}, original)

    def test_git_sort_modes_unicode_and_directory_prefix(self):
        rows = [row('a', '1'*40, '040000'), row('a.c', '2'*40), row('á', '3'*40),
                row('a0', '4'*40, '100755'), row('link', '5'*40, '120000'), row('git', '6'*40, '160000')]
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(['git', 'init', '-q', folder], check=True)
            self.assertEqual(m.tree_sha(rows), git_mktree(folder, rows))
        self.assertEqual(m.tree_sha([]), '4b825dc642cb6eb9a060e54bf8d69288fbee4904')

    def test_complete_and_partial_retries_reuse_content_objects(self):
        git, payload = fixture()
        git.fail_after_create = 3
        with self.assertRaises(OSError):
            m.build_incremental_tree(git.request, 'offline-token', payload)
        reports = []
        result = m.build_incremental_tree(git.request, 'offline-token', payload, reports.append)
        self.assertTrue(any(r['reused'] for r in reports))
        count = len(git.posts)
        self.assertEqual(m.build_incremental_tree(git.request, 'offline-token', payload), result)
        self.assertEqual(len(git.posts), count)

    def test_missing_post_content_is_checked_by_complete_get(self):
        git, payload = fixture()
        git.only_sha = True
        result = m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertIn(result['sha'], git.db)

    def test_missing_ancestors_are_created_without_losing_base_files(self):
        git = Git()
        base = git.put([row('keep', '1'*40)])
        payload = {'base_tree': base, 'tree': [row('runs/e3_extended_audit/'+PART+'/x.json', '2'*40)]}
        result = m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertIn(row('keep', '1'*40), git.db[result['sha']])

    def test_all_input_failures_happen_before_mutations(self):
        for edit in ('duplicate', 'parentfile', 'parentdir', 'wrongprefix', 'multiparts', 'unsafe', 'mode', 'sha'):
            with self.subTest(edit=edit):
                git, payload = fixture()
                prefix = 'runs/e3_extended_audit/'+PART+'/'
                if edit == 'duplicate': payload['tree'].append(deepcopy(payload['tree'][0]))
                if edit == 'parentfile': payload['tree'].append(row(prefix+'payload', 'a'*40))
                if edit == 'parentdir': payload['tree'].insert(0, row(prefix+'payload', 'a'*40))
                if edit == 'wrongprefix': payload['tree'][0]['path'] = 'other/'+PART+'/x'
                if edit == 'multiparts': payload['tree'][0]['path'] = prefix.replace(PART, OLD)+'x'
                if edit == 'unsafe': payload['tree'][0]['path'] = prefix+'../x'
                if edit == 'mode': payload['tree'][0]['mode'] = '120000'
                if edit == 'sha': payload['tree'][0]['sha'] = 'invalid'
                with self.assertRaises(ValueError):
                    m.build_incremental_tree(git.request, 'offline-token', payload)
                self.assertFalse(git.posts)

    def test_existing_part_and_non_directory_ancestors_rejected_before_posts(self):
        git, payload = fixture()
        part = git.put([row('existing', '1'*40)])
        audit = git.put([row(PART, part, '040000')])
        runs = git.put([row('e3_extended_audit', audit, '040000')])
        payload['base_tree'] = git.put([row('runs', runs, '040000')])
        with self.assertRaises(ValueError): m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertFalse(git.posts)
        payload['base_tree'] = git.put([row('runs', '2'*40)])
        with self.assertRaises(ValueError): m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertFalse(git.posts)

    def test_bad_complete_base_and_bad_created_sha_fail_closed(self):
        git, payload = fixture()
        git.db[payload['base_tree']][0]['sha'] = 'a'*40
        with self.assertRaises(ValueError): m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertFalse(git.posts)
        git, payload = fixture()
        git.bad_post_sha = True
        with self.assertRaises(ValueError): m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertEqual(len(git.posts), 1)

    def test_truncated_existing_or_base_tree_is_rejected(self):
        git, payload = fixture()
        def bad(*args, **kwargs):
            result = git.request(*args, **kwargs)
            if isinstance(result, dict): result['truncated'] = True
            return result
        with self.assertRaises(ValueError): m.build_incremental_tree(bad, 'offline-token', payload)
        self.assertFalse(git.posts)

    def test_input_payload_remains_byte_identical(self):
        git, payload = fixture()
        before = json.dumps(payload, sort_keys=True)
        m.build_incremental_tree(git.request, 'offline-token', payload)
        self.assertEqual(json.dumps(payload, sort_keys=True), before)

    def test_459_entry_pending_scale_has_identical_real_git_root(self):
        git, payload = fixture()
        prefix = 'runs/e3_extended_audit/'+PART+'/payload/'
        payload['tree'] = [row(prefix+'jobs/D03/model_'+str(n//7)+'/file_'+str(n%7)+'.json',
                               hashlib.sha1(str(n).encode()).hexdigest()) for n in range(459)]
        before = deepcopy(git.db)
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(['git', 'init', '-q', folder], check=True)
            expected = monolithic_reference(git, payload['base_tree'], payload['tree'], folder)
        git.db = before
        self.assertEqual(m.build_incremental_tree(git.request, 'offline-token', payload)['sha'], expected)
        self.assertLess(max(len(p['tree']) for p in git.posts), 100)

    def test_truncated_reusable_new_tree_is_not_accepted(self):
        git, payload = fixture()
        m.build_incremental_tree(git.request, 'offline-token', payload)
        saved_count = len(git.posts)
        original_objects = set(fixture()[0].db)
        def bad(method, path, *args, **kwargs):
            result = git.request(method, path, *args, **kwargs)
            if method == 'GET' and path.rsplit('/', 1)[1] not in original_objects:
                if isinstance(result, dict): result['truncated'] = True
            return result
        with self.assertRaises(ValueError): m.build_incremental_tree(bad, 'offline-token', payload)
        self.assertEqual(len(git.posts), saved_count)


if __name__ == '__main__':
    unittest.main()
