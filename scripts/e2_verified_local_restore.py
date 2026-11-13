"""Reuse a complete local mirror only after checking GitHub's sealed artifacts.

The remote control pointer remains authoritative. Every metadata file and every
artifact, including tensors, is checked before the original state is admitted.
Missing metadata or artifacts falls back to the unchanged full restore. No
ledger, historical artifact, or scientific result is rewritten here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

from e2_reduced_github_backup import BackupPause, _NoRedirect
from e2_reduced_github_restore import _bytes, _target, _spec, _existing, HASH
from e2_reduced_github_state import _no_active_lease


def _manifest(raw, backup):
    if hashlib.sha256(raw).hexdigest() != backup.get('manifest_sha256'):
        raise BackupPause('GitHub_manifest_sha256_mismatch')
    value = json.loads(raw)
    rows = value.get('artifacts')
    if not isinstance(rows, list):
        raise BackupPause('invalid_GitHub_artifact_manifest')
    paths = [row.get('path') for row in rows]
    if len(set(paths)) != len(paths):
        raise BackupPause('duplicate_GitHub_artifact_path')
    return value


def _store_manifest(path, raw):
    """An interrupted disposable-cache write must not leave a partial entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.manifest-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise BackupPause('symlinked_verified_manifest_cache_entry')
        if path.exists():
            if path.read_bytes() != raw:
                raise BackupPause('verified_manifest_cache_changed')
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verified_local_restore(base_class):
    """Add a read-only complete-mirror path to the pinned GitHubState class."""
    class VerifiedLocalState(base_class):
        def restore(self):
            head = self.client._branch_head()
            state = self._current(head)
            local, _ = self._scan()
            if state is None:
                return super().restore()
            if state.get('output_path') != str(self.output):
                raise BackupPause('control_output_path_changed')
            expected = state.get('files')
            if not isinstance(expected, dict):
                raise BackupPause('invalid_control_file_index')
            for path, record in local.items():
                if record != expected.get(path):
                    raise BackupPause('local_control_diverges_preserved_without_overwrite')
            if local != expected:
                return super().restore()
            _no_active_lease(self.output)
            backups = state.get('result_backups')
            if not isinstance(backups, dict):
                raise BackupPause('invalid_GitHub_restore_index')
            # Only excluded, disposable assets are cached; control metadata,
            # original receipts and tensor bytes are never written by this path.
            cache = _target(self.output, 'assets/verified_manifest_cache', directory=True)
            readers = threading.local()
            def load(item):
                relative, backup = item
                directory = _target(self.output, relative, directory=True)
                identity = backup.get('manifest_sha256')
                if not isinstance(identity, str) or not HASH.fullmatch(identity):
                    raise BackupPause('invalid_GitHub_manifest_identity')
                cached = _target(cache, identity + '.json')
                if cached.exists():
                    if not cached.is_file():
                        raise BackupPause('invalid_verified_manifest_cache_entry')
                    raw = cached.read_bytes()
                else:
                    if not hasattr(readers, 'client'):
                        readers.client = copy(self.client)
                        # Dedicated read-only HTTP handlers per thread. Request
                        # headers/token and bounded retry policy are retained.
                        if hasattr(self.client, '_opener'):
                            readers.client._opener = urllib.request.build_opener(_NoRedirect())
                    raw, _ = _bytes(readers.client, backup['manifest_path'], head)
                value = _manifest(raw, backup)
                count, missing = 0, False
                for row in value['artifacts']:
                    target = _target(directory, row.get('path'))
                    sha, size = _spec(row)
                    if not _existing(target, sha, size):
                        missing = True
                    count += 1
                if not cached.exists():
                    _store_manifest(cached, raw)
                return count, missing
            print('Verificando archivos locales contra los manifiestos de GitHub; se reutilizan las descargas completas.', flush=True)
            with ThreadPoolExecutor(max_workers=4) as pool:
                verified = list(pool.map(load, sorted(backups.items())))
            if any(missing for _, missing in verified):
                print('Faltan artefactos locales; se conserva la restauración completa verificada.', flush=True)
                return super().restore()
            # A second observation rejects a concurrent controller, while
            # artifact-only branch additions with the same control are harmless.
            latest = self._current(self.client._branch_head())
            if latest is None or latest.get('state_id') != state.get('state_id'):
                raise BackupPause('another_session_advanced_control_state')
            after, _ = self._scan()
            if after != expected:
                raise BackupPause('local_control_changed_during_verification')
            _no_active_lease(self.output)
            self._state = deepcopy(state)
            self._files = deepcopy(expected)
            self._archives = deepcopy(state['metadata_deltas'])
            self._backups = deepcopy(backups)
            self._restored = True
            report = {'status': 'github_complete_local_mirror_verified', 'restored': True,
                'state_id': state['state_id'], 'manifests': len(verified),
                'files_restored': 0, 'existing_identical_files': sum(n for n, _ in verified),
                'metadata_deltas_replayed': 0, 'training_performed': False,
                'budget_reset': False}
            print(json.dumps(report, ensure_ascii=False), flush=True)
            return report
    return VerifiedLocalState
