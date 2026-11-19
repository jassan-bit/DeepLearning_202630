"""Git-backed metadata/writes for the unchanged audit publisher.

Objects are read from the isolated Git cache only after authenticated Git
publication or fetch. All content identity/manifest checks remain active.
"""
import base64
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlsplit
from git_stage_transport import GitStageStore
from incremental_git_trees import validate_payload, tree_sha

SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
MASTER = '20261119T031104515520Z'
PROTOCOL = '5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a'
REPO = 'jassan-bit/DeepLearning_202630'
ROOT = '/repos/' + REPO
SHA = re.compile(r'[0-9a-f]{40}')
PART = re.compile(r'[0-9]{8}T[0-9]{12}Z')


def require(condition, message):
    if not condition:
        raise ValueError(message)


class GitPackContext:
    def __init__(self, base, token, identity=None, repo_url=None, emit=None):
        self.base, self.token, self.identity = Path(base), token, identity
        self.repo_url = repo_url or 'https://github.com/' + REPO + '.git'
        self.emit = emit or (lambda value: print(json.dumps(value, ensure_ascii=False), flush=True))
        self.stores, self.ready_trees, self.ready_commits, self.commit_owners = {}, {}, {}, {}
        self.ready_blobs = {}
        self.advertised, self.tree_owners = set(), {}
        self.anchor = 'c5141ade210734bd4eaac8279a32d8d04c41269d'

    def head(self):
        path = self.base / ('e3-extended-continuation-' + MASTER + '.json')
        if not path.exists():
            return self.anchor
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink state refused')
        state = json.loads(path.read_bytes())
        require(state.get('kind') == 'e3_audit_continuation_state'
                and state.get('repository') == REPO and state.get('source_commit') == SOURCE
                and state.get('master_run_id') == MASTER and state.get('protocol_sha256') == PROTOCOL
                and isinstance(state.get('head_commit'), str) and SHA.fullmatch(state['head_commit']),
                'Git transport cannot select another campaign or parent')
        return state['head_commit']

    def store(self, parent=None):
        parent = parent or self.head()
        if parent not in self.stores:
            require(self.identity is not None, 'Git author identity was not authenticated')
            self.stores[parent] = GitStageStore(self.base / ('e3-git-publisher-' + MASTER),
                parent, self.repo_url, self.token, identity=self.identity)
        return self.stores[parent]

    def blob(self, raw):
        require(self.token.encode() not in raw, 'Credential bytes cannot enter a Git publication')
        return self.store().blob(raw)

    def mark(self, store, commit):
        self.ready_commits[commit] = store
        self.ready_trees[store.read_commit(commit)['tree']['sha']] = store
        # Tree walking reads metadata only, not historical checkpoint bytes.
        raw = store._git('ls-tree', '-r', '-t', '-z', store.read_commit(commit)['tree']['sha'])
        for item in raw.split(b'\0'):
            if not item:
                continue
            header = item.split(b'\t', 1)[0].decode('ascii').split(' ')
            if header[1] == 'tree':
                self.ready_trees[header[2]] = store
            elif header[1] == 'blob':
                self.ready_blobs[header[2]] = store

    def tree(self, payload):
        require(isinstance(payload, dict) and set(payload) == {'base_tree', 'tree'}, 'Unexpected audit tree request')
        part, _ = validate_payload(payload)
        store = self.store()
        result = store.stage_tree(payload)
        self.emit({'event': 'git_pack_publication_started', 'part': part, 'tree': result['sha']})
        staged = store.publish(result['sha'], part)
        self.mark(store, staged['sha'])
        self.emit({'event': 'git_pack_staging_verified', 'part': part,
                   'commit': staged['sha'], 'ref': staged['ref']})
        return result

    def commit(self, payload):
        require(isinstance(payload, dict) and set(payload) == {'message', 'tree', 'parents'}
                and isinstance(payload['parents'], list) and len(payload['parents']) == 1,
                'Unexpected audit commit request')
        message = payload['message']
        part = message.removeprefix('Guardar delta auditoría E3 ') if isinstance(message, str) else ''
        require(isinstance(message, str) and message == 'Guardar delta auditoría E3 ' + part
                and PART.fullmatch(part), 'Only the original audit commit message is accepted')
        store = self.store(payload['parents'][0])
        result = store.create_commit(payload['tree'], payload['parents'][0], message, part)
        self.commit_owners[result['sha']] = store
        return result

    def reference(self, payload):
        require(isinstance(payload, dict) and set(payload) == {'ref', 'sha'}
                and isinstance(payload['ref'], str)
                and re.fullmatch(r'refs/heads/e3-extended-audit-[0-9]{8}T[0-9]{12}Z', payload['ref'])
                and payload['sha'] in self.commit_owners, 'Only the registered audit reference may be published')
        store = self.commit_owners[payload['sha']]
        result = store.publish_ref(payload['ref'], payload['sha'])
        self.mark(store, payload['sha'])
        self.emit({'event': 'git_pack_audit_ref_verified', 'ref': payload['ref'], 'commit': payload['sha']})
        return result

    def refs(self, prefix):
        store = self.store()
        raw = store._git('ls-remote', '--refs', 'origin', prefix + '*', network=True).decode()
        rows = []
        for line in raw.splitlines():
            sha, ref = line.split('\t')
            require(SHA.fullmatch(sha) and ref.startswith(prefix), 'Unexpected remote Git reference')
            rows.append({'ref': ref, 'object': {'type': 'commit', 'sha': sha}})
            self.advertised.add(sha)
        return sorted(rows, key=lambda row: row['ref'])

    def ref(self, ref):
        store = self.store()
        sha = store._remote_ref(ref)
        if sha is not None:
            self.advertised.add(sha)
            self.fetch_commit(sha)
        return None if sha is None else {'ref': ref, 'object': {'type': 'commit', 'sha': sha}}

    def fetch_commit(self, sha):
        if sha not in self.ready_commits:
            store = self.store()
            if not store._has(sha, 'commit'):
                store._fetch(sha)
            self.mark(store, sha)
        return self.ready_commits[sha]

    def read_tree(self, store, sha, recursive):
        def walk(tree, prefix=''):
            result = []
            for item in store.read_tree(tree)['tree']:
                row = dict(item, path=prefix + item['path'])
                if recursive and row['type'] == 'blob' and store._has(row['sha'], 'blob'):
                    row['size'] = int(store._git('cat-file', '-s', row['sha']).decode().strip())
                result.append(row)
                if recursive and row['type'] == 'tree':
                    result.extend(walk(row['sha'], row['path'] + '/'))
            return result
        return {'sha': sha, 'truncated': False, 'tree': walk(sha)}

    def contents(self, store, commit, path):
        require(path and not path.startswith('/') and '\\' not in path and '..' not in Path(path).parts,
                'Unsafe Git content path')
        listing = store._git('ls-tree', '-z', commit, '--', path).split(b'\0')
        found = [value for value in listing if value]
        require(len(found) == 1, 'Exact committed metadata file is required')
        header, actual_path = found[0].split(b'\t', 1)
        mode, kind, sha = header.decode('ascii').split(' ')
        require(actual_path.decode('utf-8') == path and mode == '100644' and kind == 'blob',
                'Committed metadata has another path or type')
        raw = store.read_blob(sha)
        return {'type': 'file', 'path': path, 'encoding': 'base64',
                'content': base64.b64encode(raw).decode(), 'sha': sha, 'size': len(raw)}


class GitPackAdapter:
    def __init__(self, api, context):
        self.api, self.original, self.context = api, api._request, context

    def normalize_tree(self, response, recursive):
        require(isinstance(response, dict) and response.get('truncated') is False
                and isinstance(response.get('tree'), list), 'A complete Git tree response is required')
        rows = []
        for row in response['tree']:
            clean = {key: row[key] for key in ('path', 'mode', 'type', 'sha')}
            if recursive and 'size' in row:
                clean['size'] = row['size']
            rows.append(clean)
        if not recursive:
            require(tree_sha(rows) == response.get('sha'), 'Tree response does not match its Git SHA')
        return {'sha': response['sha'], 'truncated': False, 'tree': rows}

    def request(self, method, path, token, payload=None, **kwargs):
        self.api._inputs(method, path, token, payload)
        if method == 'POST' and path == ROOT + '/git/blobs':
            require(isinstance(payload, dict) and set(payload) == {'encoding', 'content'}
                    and payload['encoding'] == 'base64', 'Only original base64 blob writes are accepted')
            raw = base64.b64decode(payload['content'], validate=True)
            return {'sha': self.context.blob(raw)}
        if method == 'POST' and path == ROOT + '/git/trees' and isinstance(payload, dict):
            entries = payload.get('tree', [])
            if isinstance(entries, list) and any(isinstance(row, dict)
                    and str(row.get('path', '')).startswith('runs/e3_extended_audit/') for row in entries):
                return self.context.tree(payload)
        if method == 'POST' and path == ROOT + '/git/commits' and isinstance(payload, dict):
            if str(payload.get('message', '')).startswith('Guardar delta auditoría E3 '):
                return self.context.commit(payload)
        if method == 'POST' and path == ROOT + '/git/refs' and isinstance(payload, dict):
            if str(payload.get('ref', '')).startswith('refs/heads/e3-extended-audit-'):
                return self.context.reference(payload)
        prefix = ROOT + '/git/matching-refs/heads/e3-extended-audit-'
        if method == 'GET' and path == prefix:
            return self.context.refs('refs/heads/e3-extended-audit-')
        if method == 'GET' and path.startswith(ROOT + '/git/ref/heads/e3-extended-audit-'):
            ref = 'refs/heads/' + path.removeprefix(ROOT + '/git/ref/heads/')
            result = self.context.ref(ref)
            if result is None and not kwargs.get('allow_missing', False):
                raise self.api.StorageError('Required audit reference is not published: ' + ref)
            return result
        parsed = urlsplit(path)
        if method == 'GET' and parsed.path.startswith(ROOT + '/git/trees/'):
            sha = parsed.path.rsplit('/', 1)[1]
            recursive = 'recursive' in parse_qs(parsed.query)
            if sha in self.context.ready_trees:
                value = self.context.read_tree(self.context.ready_trees[sha], sha, recursive)
            else:
                try:
                    value = self.original(method, path, token, payload, **kwargs)
                except Exception as error:
                    if (type(error).__name__ != 'StorageError' or 'status=404' not in str(error)
                            or sha not in self.context.tree_owners):
                        raise
                    self.context.fetch_commit(self.context.tree_owners[sha])
                    require(sha in self.context.ready_trees, 'Fetched commit does not contain the required tree')
                    value = self.context.read_tree(self.context.ready_trees[sha], sha, recursive)
                if value is None:
                    return None
            owner = self.context.tree_owners.get(sha)
            if owner is not None:
                for row in value['tree']:
                    if row.get('type') == 'tree':
                        self.context.tree_owners[row['sha']] = owner
            return self.normalize_tree(value, recursive)
        if method == 'GET' and path.startswith(ROOT + '/git/commits/'):
            sha = path.rsplit('/', 1)[1]
            if sha in self.context.ready_commits:
                return self.context.ready_commits[sha].read_commit(sha)
            try:
                value = self.original(method, path, token, payload, **kwargs)
            except Exception as error:
                if (type(error).__name__ != 'StorageError' or 'status=404' not in str(error)
                        or sha not in self.context.advertised):
                    raise
                value = self.context.fetch_commit(sha).read_commit(sha)
            require(value.get('sha') == sha and isinstance(value.get('tree'), dict), 'Commit response differs')
            self.context.tree_owners[value['tree']['sha']] = sha
            return value
        if method == 'GET' and parsed.path.startswith(ROOT + '/contents/'):
            query = parse_qs(parsed.query)
            commits = query.get('ref', [])
            if len(commits) == 1 and commits[0] in self.context.ready_commits:
                relative = unquote(parsed.path.removeprefix(ROOT + '/contents/'))
                return self.context.contents(self.context.ready_commits[commits[0]], commits[0], relative)
        return self.original(method, path, token, payload, **kwargs)
