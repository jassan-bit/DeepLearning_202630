import importlib.util
import json
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import e2_posthoc_control_consistency as m


class Pause(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ControlConsistencyTests(unittest.TestCase):
    def modules(self):
        class Backup:
            def __init__(self, token='private-secret'):
                self._token = token
                self._opener = SimpleNamespace(open=Mock(return_value='opened'))
                self._request = Mock(return_value={'object': {'sha': 'head'}})
            def __repr__(self):
                return 'Backup(token=<redacted>)'
        class Remote:
            def __init__(self, state):
                self.state = state
        return SimpleNamespace(GitHubBackup=Backup, BRANCH='results/e2', BackupPause=Pause), SimpleNamespace(PosthocRemoteState=Remote)

    def state(self, sequence, *, actual=None):
        client = SimpleNamespace(_branch_head=Mock(side_effect=['fresh-head-1', 'fresh-head-2']))
        state = SimpleNamespace(_state={'state_id': 'base', 'files': {'ledger': 'closed'}},
            client=client, _publish=Mock(side_effect=sequence), _current=Mock(side_effect=actual))
        return state

    def test_ref_reads_have_a_unique_cache_busting_query_and_remain_GET(self):
        backup, old = self.modules();m.install(backup, old);client = backup.GitHubBackup()
        self.assertEqual(client._branch_head(), 'head');self.assertEqual(client._branch_head(), 'head')
        calls = client._request.call_args_list
        self.assertTrue(all(c.args[0] == 'GET' for c in calls))
        self.assertTrue(all(c.args[1].startswith('/git/ref/heads/results/e2?e2_refresh=') for c in calls))
        self.assertNotEqual(calls[0].args[1], calls[1].args[1])

    def test_get_headers_preserve_auth_and_delegate_without_mutating_writes(self):
        delegate = SimpleNamespace(open=Mock(return_value='response'))
        wrapper = m._FreshReads(delegate)
        get = Request('https://api.github.com/repos/a/b/git/ref/heads/c', headers={'Authorization': 'Bearer secret'}, method='GET')
        self.assertEqual(wrapper.open(get, timeout=7), 'response')
        self.assertIn('no-cache', get.get_header('Cache-control'))
        self.assertEqual(get.get_header('Pragma'), 'no-cache')
        self.assertEqual(get.get_header('Authorization'), 'Bearer secret')
        delegate.open.assert_called_once_with(get, timeout=7)
        post = Request('https://api.github.com/repos/a/b/git/blobs', data=b'{}', method='POST')
        wrapper.open(post)
        self.assertIsNone(post.get_header('Cache-control'))
        self.assertNotIn('secret', repr(wrapper))

    def test_real_backup_request_retains_auth_redaction_and_fresh_query(self):
        spec=importlib.util.spec_from_file_location('_consistency_private_backup',ROOT/'scripts/e2_reduced_github_backup.py')
        backup=importlib.util.module_from_spec(spec);spec.loader.exec_module(backup)
        _,old=self.modules();m.install(backup,old)
        client=backup.GitHubBackup('private-test-secret')
        raw=io.BytesIO(b'{"object":{"sha":"fresh-real-head"}}')
        delegate=SimpleNamespace(open=Mock(return_value=raw))
        client._opener=delegate
        self.assertEqual(client._branch_head(),'fresh-real-head')
        request=delegate.open.call_args.args[0]
        self.assertEqual(request.get_method(),'GET')
        self.assertIn('?e2_refresh=',request.full_url)
        self.assertEqual(request.get_header('Authorization'),'Bearer private-test-secret')
        self.assertIn('no-cache',request.get_header('Cache-control'))
        self.assertNotIn('private-test-secret',repr(client))

    def test_install_is_idempotent_and_preserves_client_redaction(self):
        backup, old = self.modules();m.install(backup, old)
        first_init, first_head, first_remote = backup.GitHubBackup.__init__, backup.GitHubBackup._branch_head, old.PosthocRemoteState.__init__
        record = m.install(backup, old)
        self.assertIs(first_init, backup.GitHubBackup.__init__)
        self.assertIs(first_head, backup.GitHubBackup._branch_head)
        self.assertIs(first_remote, old.PosthocRemoteState.__init__)
        self.assertNotIn('private-secret', repr(backup.GitHubBackup()))
        self.assertFalse(record['fit_retry_performed']);self.assertFalse(record['budget_modified'])

    def test_unknown_foreign_state_stops_without_republication_or_adoption(self):
        backup, old = self.modules()
        foreign = {'state_id': 'foreign', 'files': {'ledger': 'other'}}
        state = self.state([Pause(m.CAS_CODE)], actual=[foreign]);original = state._publish
        m.bind_state(state, backup)
        with self.assertRaises(Pause):state._publish({'state_id': 'new'}, ('delta', b'bytes'))
        self.assertEqual(original.call_count, 1);self.assertEqual(state._state['state_id'], 'base')
        self.assertEqual(state.client._branch_head.call_count, 1)

    def test_ambiguous_success_is_accepted_only_for_identical_complete_state(self):
        backup, old = self.modules()
        proposed = {'state_id': 'new', 'files': {'ledger': 'reserved'}, 'metadata_deltas': [1]}
        state = self.state([Pause(m.CAS_CODE)], actual=[json.loads(json.dumps(proposed))]);original = state._publish
        m.bind_state(state, backup)
        self.assertEqual(state._publish(proposed, ('delta', b'bytes')), 'fresh-head-1')
        self.assertEqual(state._state, proposed);self.assertEqual(original.call_count, 1)
        self.assertIs(state._state, proposed)

    def test_equal_identifier_with_changed_state_bytes_is_not_idempotent_success(self):
        backup, old = self.modules()
        proposed = {'state_id': 'new', 'files': {'ledger': 'reserved'}}
        state = self.state([Pause(m.CAS_CODE)], actual=[{'state_id': 'new', 'files': {'ledger': 'changed'}}])
        m.bind_state(state, backup)
        with self.assertRaises(Pause):state._publish(proposed, None)
        self.assertEqual(state._state['state_id'], 'base')

    def test_exact_expected_base_allows_one_same_publication_retry(self):
        backup, old = self.modules()
        state = self.state([Pause(m.CAS_CODE), 'new-commit'], actual=[{'files': {'ledger': 'closed'}, 'state_id': 'base'}])
        original = state._publish;proposed = {'state_id': 'new'};delta = ('delta', b'bytes')
        m.bind_state(state, backup)
        self.assertEqual(state._publish(proposed, delta), 'new-commit')
        self.assertEqual(original.call_count, 2)
        self.assertTrue(all(c.args == (proposed, delta) for c in original.call_args_list))
        self.assertTrue(all(c.args[0] is proposed and c.args[1] is delta for c in original.call_args_list))

    def test_second_mismatch_is_read_once_more_without_a_third_publish(self):
        backup, old = self.modules()
        base = {'state_id': 'base', 'files': {'ledger': 'closed'}}
        state = self.state([Pause(m.CAS_CODE), Pause(m.CAS_CODE)], actual=[base, base]);original=state._publish
        m.bind_state(state, backup)
        with self.assertRaises(Pause):state._publish({'state_id': 'new'}, None)
        self.assertEqual(original.call_count, 2);self.assertEqual(state.client._branch_head.call_count, 2)

    def test_unrelated_transport_errors_are_never_retried(self):
        backup, old = self.modules()
        state = self.state([Pause('github_transport_or_response_failed')], actual=[]);original = state._publish
        m.bind_state(state, backup)
        with self.assertRaisesRegex(Pause, 'transport'):state._publish({'state_id': 'new'}, None)
        self.assertEqual(original.call_count, 1);state.client._branch_head.assert_not_called()

    def test_new_remote_instance_gets_only_its_own_publish_wrapper(self):
        backup, old = self.modules();m.install(backup, old)
        client=backup.GitHubBackup()
        state=SimpleNamespace(client=client, _state={'state_id': 'base'}, _publish=Mock(return_value='published'))
        remote=old.PosthocRemoteState(state)
        before=state._publish;m.bind_state(state, backup)
        self.assertIs(state._publish, before)
        self.assertIs(remote.state, state)
        self.assertEqual(state._publish({'state_id': 'new'}, None), 'published')
        self.assertTrue(getattr(state, m.MARKER))


if __name__ == '__main__':
    unittest.main()
