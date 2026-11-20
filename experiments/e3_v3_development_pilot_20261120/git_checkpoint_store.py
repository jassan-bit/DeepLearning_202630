"""Isolated Git PACK checkpoints for the V3 development pilot.

Only one new, explicitly named e3-v3-development branch may move. Source trees,
older audit refs and other branches are retained. The token is supplied only in
the environment of a network Git child via a temporary askpass program.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
import zipfile

SHA = re.compile(r'[0-9a-f]{40}')
RUN = re.compile(r'[0-9]{8}T[0-9]{12}Z')
BRANCH = re.compile(r'e3-v3-development-[0-9]{8}T[0-9]{12}Z')
SECRET = re.compile(r'TOKEN|SECRET|PASSWORD|CREDENTIAL|(?:^|_)PAT(?:_|$)', re.I)
ALLOWED_SUFFIXES = {'.json', '.csv', '.log', '.txt', '.pt'}
MAX_FILE = 16 * 1024 * 1024
MAX_SNAPSHOT = 64 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(path, *, directory=False):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink refused')
    if path.exists():
        require(path.is_dir() if directory else path.is_file(), 'Unexpected filesystem object')
    return path


def atomic(path, raw):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-v3-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sanitized_environment(token=''):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GIT_') and not SECRET.search(k)
           and k != 'SSH_ASKPASS' and (not token or v != token)}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_TERMINAL_PROMPT='0', GIT_NO_LAZY_FETCH='1', LC_ALL='C',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    return env


class GitCheckpointStore:
    def __init__(self, cache, source_commit, run_id, token, *,
                 repo_url='https://github.com/jassan-bit/DeepLearning_202630.git',
                 allow_local=False):
        require(isinstance(source_commit, str) and SHA.fullmatch(source_commit), 'Exact source SHA required')
        require(isinstance(run_id, str) and RUN.fullmatch(run_id), 'Exact pilot run identifier required')
        require(isinstance(token, str) and '\n' not in token and '\r' not in token, 'Invalid credential shape')
        parsed = urlsplit(repo_url)
        valid_remote = (parsed.scheme == 'https' and parsed.netloc == 'github.com'
                        and parsed.path == '/jassan-bit/DeepLearning_202630.git')
        valid_local = allow_local and parsed.scheme == 'file' and parsed.netloc in {'', 'localhost'} and parsed.path.startswith('/')
        require((valid_remote or valid_local) and not parsed.username and not parsed.password
                and not parsed.query and not parsed.fragment, 'Unexpected credential-free repository URL')
        self.cache = safe(cache, directory=True)
        self.cache.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.gitdir = safe(self.cache / 'publisher.git', directory=True)
        self.state_path = self.cache / 'publisher_state.json'
        self.pending_path = self.cache / 'pending_publication.json'
        self.source, self.run_id, self._token = source_commit, run_id, token
        self.repo_url = repo_url
        self.branch = 'e3-v3-development-' + run_id
        self.ref = 'refs/heads/' + self.branch
        require(BRANCH.fullmatch(self.branch), 'Branch is outside pilot namespace')
        self.prefix = 'runs/e3_v3_development_pilot/' + run_id
        if not self.gitdir.exists():
            self._git('init', '--bare', '--quiet', str(self.gitdir), outside=True)
            self._git('config', 'remote.origin.url', repo_url)
            self._git('config', 'remote.origin.promisor', 'true')
            self._git('config', 'remote.origin.partialclonefilter', 'blob:none')
        require(self._git('config', '--get', 'remote.origin.url').decode().strip() == repo_url,
                'Existing Git cache has another remote')
        self._fetch(source_commit)
        if self.state_path.exists():
            self.state = json.loads(safe(self.state_path).read_bytes())
            self._compatible(self.state)
        else:
            self.state = {'source_commit': self.source, 'run_id': self.run_id,
                          'head': self.source, 'sequence': 0, 'inventory': {}}

    def _compatible(self, value):
        require(value.get('source_commit') == self.source and value.get('run_id') == self.run_id
                and isinstance(value.get('head'), str) and SHA.fullmatch(value['head'])
                and type(value.get('sequence')) is int and value['sequence'] >= 0
                and isinstance(value.get('inventory'), dict), 'Publisher state belongs to another pilot')

    def _git(self, *args, data=None, network=False, outside=False, check=True, extra_env=None):
        safe(self.cache, directory=True)
        env = sanitized_environment(self._token)
        env.update(extra_env or {})
        command = ['git'] + ([] if outside else ['--git-dir=' + str(self.gitdir)])
        command += ['-c', 'credential.helper=']
        if not network or 'push' in args:
            command += ['-c', 'remote.origin.promisor=false']

        def run():
            process = subprocess.run(command + list(args), input=data, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, timeout=180, check=False)
            if check and process.returncode:
                details = process.stderr.decode('utf-8', 'replace')[-2500:]
                if self._token:
                    details = details.replace(self._token, '[REDACTED]')
                raise RuntimeError('Git pilot transport failed: ' + details.strip())
            return process.stdout if check else process

        if not network:
            return run()
        with tempfile.TemporaryDirectory(prefix='askpass-', dir=self.cache) as directory:
            program = Path(directory) / 'askpass'
            program.write_text('#!/bin/sh\ncase "$1" in\n*Username*) printf "%s\\n" x-access-token ;;\n*) printf "%s\\n" "$E3_V3_GIT_TOKEN" ;;\nesac\n')
            program.chmod(0o700)
            env.update(GIT_ASKPASS=str(program), E3_V3_GIT_TOKEN=self._token)
            return run()

    def _fetch(self, commit):
        require(SHA.fullmatch(commit), 'Unsafe fetch identity')
        present = self._git('cat-file', '-t', commit, check=False)
        if present.returncode == 0:
            require(present.stdout.strip() == b'commit', 'Fetched identity is not a commit')
            return
        last = None
        for attempt in range(3):
            try:
                self._git('fetch', '--quiet', '--no-tags', '--depth=1', '--filter=blob:none',
                          'origin', commit, network=True)
                require(self._git('cat-file', '-t', commit).strip() == b'commit', 'Fetched commit differs')
                return
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                last = error
                if attempt < 2:
                    time.sleep(2 + attempt)
        raise RuntimeError('Exact pilot commit fetch failed after three attempts') from last

    def remote_head(self):
        last = None
        for attempt in range(3):
            try:
                result = self._git('ls-remote', '--exit-code', 'origin', self.ref,
                                   network=True, check=False)
                if result.returncode == 2:
                    return None
                if result.returncode:
                    raise RuntimeError('Unable to read pilot branch')
                rows = result.stdout.decode().splitlines()
                require(len(rows) == 1, 'Ambiguous pilot branch')
                sha, ref = rows[0].split('\t')
                require(SHA.fullmatch(sha) and ref == self.ref, 'Remote pilot identity differs')
                return sha
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                last = error
                if attempt < 2:
                    time.sleep(2 + attempt)
        raise RuntimeError('Pilot branch read failed after three attempts') from last

    def _read_path(self, commit, name, max_bytes=MAX_FILE):
        require(SHA.fullmatch(commit), 'Invalid commit')
        path = self.prefix + '/' + name
        # The named blob belongs to a fetched, verified commit. Explicitly fetch
        # it if the metadata-only clone has not materialized this object yet.
        obj = self._git('rev-parse', commit + ':' + path).decode().strip()
        present = self._git('cat-file', '-t', obj, check=False)
        if present.returncode:
            last = None
            for attempt in range(3):
                try:
                    self._git('fetch', '--quiet', '--no-tags', '--no-filter', '--no-write-fetch-head',
                              'origin', obj, network=True)
                    break
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    last = error
                    if attempt == 2:
                        raise RuntimeError('Named snapshot blob fetch failed after three attempts') from last
                    time.sleep(2 + attempt)
        require(self._git('cat-file', '-t', obj).strip() == b'blob', 'Saved path is not a blob')
        size = int(self._git('cat-file', '-s', obj).strip())
        require(0 <= size <= max_bytes, 'Saved metadata exceeds limit')
        raw = self._git('cat-file', 'blob', obj)
        require(len(raw) == size, 'Saved blob size differs')
        require(hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + raw).hexdigest() == obj,
                'Saved Git blob identity differs')
        return raw

    def _no_credentials(self, raw):
        if not self._token:
            return
        secret = self._token.encode()
        require(secret not in raw, 'Credential found in evidence')
        if zipfile.is_zipfile(io.BytesIO(raw)):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                size = 0
                for entry in archive.infolist():
                    size += entry.file_size
                    require(entry.file_size <= 32 * 1024 * 1024 and size <= 64 * 1024 * 1024,
                            'Compressed checkpoint exceeds safe inspection limit')
                    require(secret not in archive.read(entry), 'Credential found in compressed evidence')

    def _inventory(self, output):
        output = safe(output, directory=True)
        require(output.is_dir(), 'Pilot output is missing')
        files = {}
        total = 0
        for path in sorted(output.rglob('*')):
            if path.is_symlink():
                raise ValueError('Pilot evidence symlink refused')
            if not path.is_file() or path.name.startswith('.partial-') or path.name.endswith('.tmp'):
                continue
            require(path.suffix in ALLOWED_SUFFIXES, 'Unexpected pilot evidence type: ' + path.suffix)
            name = path.relative_to(output).as_posix()
            require('\n' not in name and '\r' not in name and '\\' not in name, 'Unsafe evidence path')
            raw = path.read_bytes()
            require(len(raw) <= MAX_FILE, 'One pilot evidence file exceeds limit')
            self._no_credentials(raw)
            # Checkpoint ZIP bytes are not rewritten or reserialized here.
            files[name] = raw
            total += len(raw)
        require(files and total <= MAX_SNAPSHOT and len(files) <= 1500, 'Pilot snapshot exceeds limit')
        missing = set(self.state['inventory']) - set(files)
        require(not missing, 'Previously saved pilot evidence is missing: ' + ', '.join(sorted(missing)[:4]))
        return files

    def _reconcile_pending(self):
        if not self.pending_path.exists():
            return
        pending = json.loads(safe(self.pending_path).read_bytes())
        self._compatible(pending['next_state'])
        require(pending['parent'] == self.state['head'], 'Pending publication has another parent')
        actual = self.remote_head()
        if actual == pending['commit']:
            self.state = pending['next_state']
            atomic(self.state_path, canonical(self.state))
            self.pending_path.unlink()
            return
        expected = None if self.state['sequence'] == 0 else self.state['head']
        require(actual == expected, 'Pilot publication conflict; remote branch was preserved')
        self._push_pending(pending)

    def _push_pending(self, pending):
        expected = '' if self.state['sequence'] == 0 else self.state['head']
        require(pending['parent'] == self.state['head'], 'Pending parent differs')
        # A push is not blindly retried after an uncertain outcome. Read back
        # the exact expected branch: an acknowledged/lost-ack publication is
        # accepted only if it points to this immutable candidate commit.
        error = None
        try:
            self._git('-c', 'pack.window=0', '-c', 'pack.depth=0', '-c', 'pack.threads=1',
                      '-c', 'core.compression=1', '-c', 'push.negotiate=true',
                      'push', '--quiet', '--atomic', '--no-thin',
                      '--force-with-lease=' + self.ref + ':' + expected, 'origin',
                      pending['commit'] + ':' + self.ref, network=True)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            error = exc
        actual = self.remote_head()
        require(actual == pending['commit'], 'Pilot save is unconfirmed; local evidence retained')
        self.state = pending['next_state']
        atomic(self.state_path, canonical(self.state))
        self.pending_path.unlink(missing_ok=True)
        return error

    def publish(self, output):
        with (self.cache / 'publication.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._reconcile_pending()
            expected = None if self.state['sequence'] == 0 else self.state['head']
            require(self.remote_head() == expected, 'Pilot branch conflict; no remote ref changed')
            files = self._inventory(output)
            inventory = {name: {'bytes': len(raw), 'sha256': digest(raw)} for name, raw in files.items()}
            if inventory == self.state['inventory']:
                return self.receipt()
            sequence = self.state['sequence'] + 1
            manifest = {'kind': 'e3_v3_development_git_snapshot', 'source_commit': self.source,
                        'run_id': self.run_id, 'sequence': sequence,
                        'parent': self.state['head'], 'files': inventory,
                        'R1_accredited': False, 'R3_accredited': False}
            self._git('read-tree', self.state['head'])
            index_rows = []
            for name, raw in {**{'payload/' + n: b for n, b in files.items()},
                              'snapshot_manifest.json': canonical(manifest)}.items():
                obj = self._git('hash-object', '-w', '--stdin', data=raw).decode().strip()
                full = self.prefix + '/' + name
                index_rows.append(('100644 ' + obj + '\t' + full).encode() + b'\0')
            self._git('update-index', '-z', '--index-info', data=b''.join(index_rows))
            # Existing base-tree blobs are intentionally not materialized. New
            # payload blobs were written above; inherited source blob identities
            # remain those of the exact source commit on the same remote.
            tree = self._git('write-tree', '--missing-ok').decode().strip()
            changed = self._git('diff-tree', '--no-commit-id', '--name-only', '-r',
                                self.state['head'], tree).decode().splitlines()
            require(changed and all(name.startswith(self.prefix + '/') for name in changed),
                    'A snapshot attempted to alter paths outside this pilot')
            date = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            env = {'GIT_AUTHOR_NAME': 'E3 V3 development pilot',
                   'GIT_COMMITTER_NAME': 'E3 V3 development pilot',
                   'GIT_AUTHOR_EMAIL': 'e3-v3-development@users.noreply.github.com',
                   'GIT_COMMITTER_EMAIL': 'e3-v3-development@users.noreply.github.com',
                   'GIT_AUTHOR_DATE': date, 'GIT_COMMITTER_DATE': date}
            commit = self._git('commit-tree', tree, '-p', self.state['head'],
                               data=('Guardar piloto V3 ' + self.run_id + ' parte ' + str(sequence) + '\n').encode(),
                               extra_env=env).decode().strip()
            next_state = {**self.state, 'head': commit, 'sequence': sequence, 'inventory': inventory}
            pending = {'commit': commit, 'parent': self.state['head'], 'tree': tree,
                       'created_at_utc': date, 'next_state': next_state}
            atomic(self.pending_path, canonical(pending))
            self._git('update-ref', 'refs/v3-pending/' + self.run_id, commit)
            self._push_pending(pending)
            return self.receipt()

    def receipt(self):
        return {'git_commit': self.state['head'], 'branch': self.branch,
                'sequence': self.state['sequence'], 'source_commit': self.source,
                'master_run_id': self.run_id,
                'run_url': 'https://github.com/jassan-bit/DeepLearning_202630/tree/' +
                           self.branch + '/' + self.prefix,
                'R1_accredited': False, 'R3_accredited': False}

    def restore(self, output):
        """Restore only the declared branch and exact manifested payload bytes."""
        require(not self.pending_path.exists(), 'Reconcile pending local save before restore')
        head = self.remote_head()
        require(head is not None, 'The specified pilot branch does not exist')
        self._fetch(head)
        changed = self._git('diff-tree', '--no-commit-id', '--name-only', '-r',
                            self.source, head).decode().splitlines()
        require(changed and all(name.startswith(self.prefix + '/') for name in changed),
                'Remote pilot changed source or paths outside its namespace')
        manifest = json.loads(self._read_path(head, 'snapshot_manifest.json'))
        require(manifest.get('kind') == 'e3_v3_development_git_snapshot'
                and manifest.get('source_commit') == self.source
                and manifest.get('run_id') == self.run_id
                and type(manifest.get('sequence')) is int and manifest['sequence'] > 0,
                'Remote snapshot identity differs')
        files = manifest.get('files')
        require(isinstance(files, dict) and 0 < len(files) <= 1500, 'Invalid remote snapshot inventory')
        require(all(isinstance(record, dict) and type(record.get('bytes')) is int
                    and 0 <= record['bytes'] <= MAX_FILE for record in files.values()),
                'Remote snapshot has an invalid size declaration')
        require(sum(record['bytes'] for record in files.values()) <= MAX_SNAPSHOT,
                'Remote snapshot exceeds limit')
        output = safe(output, directory=True)
        if output.exists():
            require(not any(output.iterdir()), 'Restore requires an empty output; existing local evidence retained')
        output.mkdir(parents=True, exist_ok=True)
        total = 0
        for name, record in files.items():
            path = Path(name)
            require(path.as_posix() == name and not path.is_absolute() and '..' not in path.parts
                    and '\\' not in name and path.suffix in ALLOWED_SUFFIXES,
                    'Remote evidence path is unsafe')
            require(type(record.get('bytes')) is int and 0 <= record['bytes'] <= MAX_FILE,
                    'Remote evidence size is invalid')
            raw = self._read_path(head, 'payload/' + name, record['bytes'])
            require(len(raw) == record['bytes'] and digest(raw) == record['sha256'],
                    'Remote checkpoint bytes do not match the manifest')
            atomic(output / path, raw)
            total += len(raw)
        require(total <= MAX_SNAPSHOT, 'Remote snapshot exceeds limit')
        self.state = {'source_commit': self.source, 'run_id': self.run_id,
                      'head': head, 'sequence': manifest['sequence'], 'inventory': files}
        atomic(self.state_path, canonical(self.state))
        return self.receipt()
