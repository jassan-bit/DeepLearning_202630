"""Git PACK transport for immutable audit objects; isolated from scientific files.

Only an absent staging/audit branch may be created. The empty expected-value
lease is deliberately stricter than an ordinary push: even a concurrent
fast-forwardable branch creation is rejected, never overwritten.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
from urllib.parse import urlsplit

import incremental_git_trees as trees

SHA = re.compile(r'[0-9a-f]{40}')
PART = re.compile(r'[0-9]{8}T[0-9]{12}Z')
REF = re.compile(r'refs/heads/(e3-git-stage|e3-extended-audit)-([0-9]{8}T[0-9]{12}Z)')
CREDENTIAL_ENV = re.compile(r'TOKEN|(?:^|_)PAT(?:_|$)|SECRET|CREDENTIAL', re.IGNORECASE)


class GitStageStore:
    def __init__(self, base, source_master, repo_url, token, identity=None):
        trees.require(isinstance(source_master, str) and SHA.fullmatch(source_master), 'Invalid parent commit')
        url = urlsplit(repo_url)
        trees.require(url.scheme in ('https', 'file') and not url.username and not url.password
                      and not url.query and not url.fragment and
                      ((url.scheme == 'https' and url.hostname) or
                       (url.scheme == 'file' and url.netloc in ('', 'localhost') and url.path.startswith('/'))),
                      'A credential-free HTTPS or local file Git URL is required')
        self.base = Path(base)
        trees.require(not self.base.is_symlink(), 'Git transport base must not be a symlink')
        self.base.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cache_path = self.base / 'git-stage-cache.git'
        trees.require(not self.cache_path.is_symlink(), 'Git cache must not be a symlink')
        self.parent_sha, self.repo_url, self._token = source_master, repo_url, token
        self.identity = identity or {'name': 'E3 audit staging', 'email': 'e3-audit-staging@users.noreply.github.com'}
        trees.require(set(self.identity) == {'name', 'email'} and all(
            isinstance(v, str) and v and not any(c in v for c in '\0\n\r<>') for v in self.identity.values()),
            'Invalid Git author identity')
        self._trees, self._roots, self._commits = {}, {}, {}
        self._blob_shas, self._commit_trees, self._scanned_blob_roots = set(), set(), set()
        if not self.cache_path.exists():
            self._git('init', '--bare', '--quiet', str(self.cache_path), outside=True)
            self._git('config', 'remote.origin.url', repo_url)
            self._git('config', 'remote.origin.promisor', 'true')
            self._git('config', 'remote.origin.partialclonefilter', 'blob:none')
        trees.require(self._git('config', '--get', 'remote.origin.url').decode().strip() == repo_url,
                      'Git cache remote differs from the requested repository')
        if not self._has(source_master, 'commit'):
            self._fetch(source_master)
        self.parent_tree = self.read_commit(source_master)['tree']['sha']

    def _env(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('GIT_') and not CREDENTIAL_ENV.search(k) and k not in ('E3_GIT_PUSH_TOKEN', 'SSH_ASKPASS',
                   'GH_TOKEN', 'GITHUB_TOKEN', 'GH_ENTERPRISE_TOKEN', 'GITHUB_ENTERPRISE_TOKEN', 'GITHUB_PAT')
               and (not self._token or v != self._token)}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_TERMINAL_PROMPT='0', GIT_NO_LAZY_FETCH='1', LC_ALL='C')
        return env

    def _git(self, *args, data=None, network=False, outside=False, extra_env=None, check=True):
        trees.require(not self.base.is_symlink() and not self.cache_path.is_symlink(),
                      'Git transport base/cache must not be symlinks')
        env = self._env()
        env.update(extra_env or {})
        cmd = ['git'] + ([] if outside else ['--git-dir=' + str(self.cache_path)])
        # Never persist credentials or consult a user/global credential helper.
        cmd += ['-c', 'credential.helper=']
        if not network or 'push' in args:
            # Disable implicit object fetches even on Git versions predating GIT_NO_LAZY_FETCH.
            cmd += ['-c', 'remote.origin.promisor=false']
        def run():
            result = subprocess.run(cmd + list(args), input=data, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env, check=False)
            if check and result.returncode:
                detail = result.stderr.decode('utf-8', 'replace')[-4000:]
                if self._token:
                    detail = detail.replace(self._token, '[REDACTED]')
                raise RuntimeError('Git transport command failed: ' + detail.strip())
            return result.stdout if check else result
        if not network:
            return run()
        with tempfile.TemporaryDirectory(prefix='git-askpass-', dir=self.base) as folder:
            script = Path(folder) / 'askpass'
            script.write_text('#!/bin/sh\ncase "$1" in\n*Username*) printf "%s\\n" x-access-token ;;\n*) printf "%s\\n" "$E3_GIT_PUSH_TOKEN" ;;\nesac\n')
            script.chmod(0o700)
            env.update(GIT_ASKPASS=str(script), E3_GIT_PUSH_TOKEN=self._token or '')
            return run()

    def _has(self, sha, kind):
        result = self._git('cat-file', '-t', sha, check=False)
        return result.returncode == 0 and result.stdout.decode().strip() == kind

    def _fetch(self, sha):
        self._git('fetch', '--quiet', '--no-tags', '--depth=1', '--filter=blob:none', 'origin', sha, network=True)
        trees.require(self._has(sha, 'commit'), 'Fetched parent is not the exact required commit')

    def blob(self, raw):
        trees.require(isinstance(raw, bytes), 'Git blob input must be exact bytes')
        expected = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        actual = self._git('hash-object', '-w', '--stdin', data=raw).decode().strip()
        trees.require(actual == expected, 'Stored blob SHA differs from the frozen content')
        return actual

    def read_blob(self, sha, max_bytes=48 * 1024 * 1024):
        """Read bounded receipt metadata, fetching only this exact object if absent."""
        trees.require(isinstance(sha, str) and SHA.fullmatch(sha) and
                      isinstance(max_bytes, int) and not isinstance(max_bytes, bool) and
                      0 <= max_bytes <= 48 * 1024 * 1024, 'Invalid bounded blob request')
        kind = self._git('cat-file', '-t', sha, check=False)
        if kind.returncode:
            for root in self._commit_trees - self._scanned_blob_roots:
                self.read_tree(root)
                listing = self._git('ls-tree', '-r', '-z', root)
                for item in listing.split(b'\0'):
                    if item:
                        header = item.split(b'\t', 1)[0].decode('ascii').split(' ')
                        if header[1] == 'blob':
                            self._blob_shas.add(header[2])
                self._scanned_blob_roots.add(root)
                if sha in self._blob_shas:
                    break
            trees.require(sha in self._blob_shas, 'Missing blob must belong to a verified local commit tree')
            # An exact blob has no children: this retrieves no commit history or sibling weights.
            self._git('fetch', '--quiet', '--no-tags', '--no-filter', '--no-write-fetch-head',
                      'origin', sha, network=True)
        trees.require(self._has(sha, 'blob'), 'Requested metadata object is not a Git blob')
        size = int(self._git('cat-file', '-s', sha).decode().strip())
        trees.require(0 <= size <= max_bytes, 'Metadata blob exceeds the permitted byte limit')
        raw = self._git('cat-file', 'blob', sha)
        expected = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        trees.require(len(raw) == size and expected == sha, 'Metadata blob size or Git SHA differs')
        return raw

    def read_tree(self, sha):
        trees.require(isinstance(sha, str) and SHA.fullmatch(sha), 'Invalid tree SHA')
        raw = self._git('ls-tree', '-z', sha)
        rows = []
        for value in raw.split(b'\0'):
            if value:
                header, path = value.split(b'\t', 1)
                mode, kind, obj = header.decode('ascii').split(' ')
                rows.append({'path': path.decode('utf-8'), 'mode': mode, 'type': kind, 'sha': obj})
        trees.require(trees.tree_sha(rows) == sha, 'Local Git tree differs from its SHA')
        self._blob_shas.update(r['sha'] for r in rows if r['type'] == 'blob')
        return {'sha': sha, 'truncated': False, 'tree': rows}

    def _request(self, method, path, token=None, payload=None, *, allow_missing=False):
        if method == 'GET' and path.startswith(trees.ENDPOINT + '/'):
            sha = path.rsplit('/', 1)[1]
            # Optional reuse probes consult only trees created in this process.
            return self._trees.get(sha) if allow_missing else self.read_tree(sha)
        trees.require(method == 'POST' and path == trees.ENDPOINT and set(payload) == {'tree'},
                      'Unexpected local Git tree request')
        rows = [trees.normalized(r) for r in payload['tree']]
        raw = b''.join((r['mode']+' '+r['type']+' '+r['sha']+'\t'+r['path']).encode()+b'\0' for r in rows)
        sha = self._git('mktree', '--missing', '-z', data=raw).decode().strip()
        trees.require(sha == trees.tree_sha(rows), 'Git mktree differs from the intended tree')
        self._trees[sha] = self.read_tree(sha)
        return self._trees[sha]

    def stage_tree(self, payload):
        part, _ = trees.validate_payload(payload)
        trees.require(payload['base_tree'] == self.parent_tree, 'Audit base differs from the frozen parent')
        trees.require(all(self._has(r['sha'], 'blob') for r in payload['tree']),
                      'Every frozen audit blob must be stored locally before staging')
        result = trees.build_incremental_tree(self._request, None, payload)
        self._roots[result['sha']] = part
        return result

    def read_commit(self, sha):
        trees.require(isinstance(sha, str) and SHA.fullmatch(sha), 'Invalid commit SHA')
        raw = self._git('cat-file', 'commit', sha)
        trees.require(hashlib.sha1(b'commit ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == sha,
                      'Local commit content differs from its Git SHA')
        raw = raw.split(b'\n\n', 1)[0]
        tree = [v[5:].decode() for v in raw.splitlines() if v.startswith(b'tree ')]
        parents = [v[7:].decode() for v in raw.splitlines() if v.startswith(b'parent ')]
        trees.require(len(tree) == 1 and SHA.fullmatch(tree[0]) and all(SHA.fullmatch(p) for p in parents),
                      'Malformed local commit')
        self._commit_trees.add(tree[0])
        return {'sha': sha, 'tree': {'sha': tree[0]}, 'parents': [{'sha': p} for p in parents]}

    def create_commit(self, tree_sha, parent_sha, message, part, role='audit'):
        trees.require(self._roots.get(tree_sha) == part and parent_sha == self.parent_sha and PART.fullmatch(part)
                      and role in ('staging', 'audit') and isinstance(message, str) and message and '\0' not in message,
                      'Commit must bind the registered audit tree and original parent')
        journal_path = self.base / ('git-' + role + '-' + part + '.json')
        trees.require(not journal_path.is_symlink(), 'Immutable commit journal must not be a symlink')
        metadata = {'schema': 1, 'part': part, 'role': role, 'tree': tree_sha, 'parents': [parent_sha],
                    'message': message, 'identity': dict(self.identity)}
        if journal_path.exists():
            recorded = json.loads(journal_path.read_text())
            trees.require(all(recorded.get(k) == v for k, v in metadata.items())
                          and set(recorded) == set(metadata) | {'nonce', 'date'}
                          and re.fullmatch('[0-9a-f]{32}', recorded.get('nonce', ''))
                          and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ', recorded.get('date', '')),
                          'Existing immutable commit journal differs')
        else:
            recorded = dict(metadata, nonce=secrets.token_hex(16),
                            date=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
            # Write once before commit creation; retries retain the current-time identity/date/message.
            fd = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'w') as stream:
                json.dump(recorded, stream, sort_keys=True, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
        author = {'GIT_AUTHOR_NAME': recorded['identity']['name'], 'GIT_COMMITTER_NAME': recorded['identity']['name'],
                  'GIT_AUTHOR_EMAIL': recorded['identity']['email'], 'GIT_COMMITTER_EMAIL': recorded['identity']['email'],
                  'GIT_AUTHOR_DATE': recorded['date'], 'GIT_COMMITTER_DATE': recorded['date']}
        sha = self._git('commit-tree', tree_sha, '-p', parent_sha, data=message.encode(), extra_env=author).decode().strip()
        trees.require(self.read_commit(sha) == {'sha': sha, 'tree': {'sha': tree_sha},
                                              'parents': [{'sha': parent_sha}]}, 'Local commit content differs')
        self._commits[sha] = {'part': part, 'role': role, 'tree': tree_sha}
        return {'sha': sha}

    def _remote_ref(self, ref):
        raw = self._git('ls-remote', '--refs', 'origin', ref, network=True).decode().splitlines()
        trees.require(len(raw) <= 1, 'Remote returned multiple staging refs')
        if not raw:
            return None
        fields = raw[0].split('\t')
        trees.require(len(fields) == 2 and fields[1] == ref and SHA.fullmatch(fields[0]), 'Malformed remote ref')
        return fields[0]

    def publish_ref(self, ref, commit_sha):
        match = REF.fullmatch(ref) if isinstance(ref, str) else None
        record = self._commits.get(commit_sha)
        trees.require(match and record and record['part'] == match[2]
                      and record['role'] == ('staging' if match[1] == 'e3-git-stage' else 'audit'),
                      'Only the registered immutable audit/staging commit may be published')
        found = self._remote_ref(ref)
        if found is not None:
            trees.require(found == commit_sha, 'Conflicting remote audit/staging branch; never overwrite')
        else:
            try:
                self._git('-c', 'push.negotiate=true', '-c', 'pack.window=0', '-c', 'pack.depth=0', '-c', 'core.compression=1',
                          '-c', 'pack.threads=1', 'push', '--atomic', '--porcelain', '--no-thin',
                          '--force-with-lease=' + ref + ':', 'origin', commit_sha + ':' + ref, network=True)
            except RuntimeError:
                if self._remote_ref(ref) != commit_sha:
                    raise
            trees.require(self._remote_ref(ref) == commit_sha, 'Remote ref verification failed')
        return {'ref': ref, 'object': {'type': 'commit', 'sha': commit_sha}}

    def publish(self, tree_sha, part):
        trees.require(self._roots.get(tree_sha) == part and PART.fullmatch(part), 'Unregistered staging tree')
        ref = 'refs/heads/e3-git-stage-' + part
        existing = self._remote_ref(ref)
        if existing:
            if not self._has(existing, 'commit'):
                self._fetch(existing)
            value = self.read_commit(existing)
            trees.require(value['tree']['sha'] == tree_sha and value['parents'] == [{'sha': self.parent_sha}],
                          'Conflicting remote staging branch; never overwrite')
            return {'sha': existing, 'tree': tree_sha, 'ref': ref, 'reused': True}
        commit = self.create_commit(tree_sha, self.parent_sha, 'E3 Git object staging for audit part ' + part,
                                    part, role='staging')['sha']
        self.publish_ref(ref, commit)
        return {'sha': commit, 'tree': tree_sha, 'ref': ref, 'reused': False}
