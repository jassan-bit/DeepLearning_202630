"""Restore already backed-up E2 artifacts; never train or reset a ledger.

The caller first restores the authoritative control snapshot. This module
materializes the artifact bytes named by its immutable GitHub manifests.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile

from e2_reduced_github_backup import BackupPause, _public_url

HASH = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def _target(root, relative, *, directory=False):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise BackupPause('invalid_restore_artifact_path')
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or str(name) != relative:
        raise BackupPause('invalid_restore_artifact_path')
    if relative == '.' and not directory:
        raise BackupPause('invalid_restore_artifact_path')
    root = Path(root).absolute()
    path = root.joinpath(*name.parts)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise BackupPause('symlinked_restore_artifact_path')
    return path


def _bytes(client, path, head):
    record = client._content(path, head)
    if record is None:
        raise BackupPause('backed_up_metadata_missing')
    encoded = record.get('content')
    if not encoded or record.get('encoding') == 'none':
        encoded = client._request('GET', '/git/blobs/' + record['sha'])['content']
    try:
        return base64.b64decode(encoded, validate=False), record
    except (ValueError, TypeError):
        raise BackupPause('invalid_backed_up_metadata_encoding') from None


def _spec(row):
    expected, size = row.get('sha256'), row.get('bytes')
    if not isinstance(expected, str) or not HASH.fullmatch(expected) or type(size) is not int or size < 0:
        raise BackupPause('invalid_backed_up_artifact_identity')
    return expected, size


def _existing(path, expected, size):
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != size or _sha(path) != expected:
        raise BackupPause('local_artifact_differs_from_GitHub_backup')
    return True


def _write_chunks(path, chunks, expected, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.e2-github-restore-', dir=path.parent)
    h, count = hashlib.sha256(), 0
    try:
        with os.fdopen(fd, 'wb') as target:
            for chunk in chunks:
                count += len(chunk)
                if count > size:
                    raise BackupPause('backed_up_artifact_size_mismatch')
                h.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        if count != size or h.hexdigest() != expected:
            raise BackupPause('backed_up_artifact_sha256_mismatch')
        if path.is_symlink():
            raise BackupPause('symlinked_restore_artifact_path')
        if path.exists():
            _existing(path, expected, size)
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _download(client, row, path):
    expected, size = _spec(row)
    _public_url(row.get('url', ''))
    try:
        with client._public_opener.open(row['url'], timeout=client.timeout) as response:
            if hasattr(response, 'geturl'):
                _public_url(response.geturl())
            _write_chunks(path, iter(lambda: response.read(1024**2), b''), expected, size)
    except BackupPause:
        raise
    except (OSError, TimeoutError, ValueError):
        raise BackupPause('GitHub_artifact_download_failed') from None


def restore_results(root, client, backups, head):
    """Restore exact artifacts for ``{relative_directory: backup_receipt}``.

    Existing byte-identical files are reused. Divergent local files are retained
    and cause a pause. No token is attached to public release downloads.
    """
    root = Path(root).absolute()
    if not isinstance(backups, dict) or not isinstance(head, str) or not re.fullmatch(r'[0-9a-f]{40}', head):
        raise BackupPause('invalid_GitHub_restore_index')
    restored = reused = manifests = 0
    for relative_directory, backup in sorted(backups.items()):
        directory = _target(root, relative_directory, directory=True)
        expected_manifest = backup.get('manifest_sha256')
        if not isinstance(expected_manifest, str) or not HASH.fullmatch(expected_manifest):
            raise BackupPause('invalid_GitHub_manifest_identity')
        raw, _ = _bytes(client, backup['manifest_path'], head)
        if hashlib.sha256(raw).hexdigest() != expected_manifest:
            raise BackupPause('GitHub_manifest_sha256_mismatch')
        manifest = json.loads(raw)
        rows = manifest.get('artifacts')
        if not isinstance(rows, list):
            raise BackupPause('invalid_GitHub_artifact_manifest')
        paths = [row.get('path') for row in rows]
        if len(set(paths)) != len(paths):
            raise BackupPause('duplicate_GitHub_artifact_path')
        archived = []
        for row in rows:
            path = _target(directory, row.get('path'))
            expected, size = _spec(row)
            if _existing(path, expected, size):
                reused += 1
                continue
            storage = row.get('storage')
            if storage == 'git_blob':
                data, record = _bytes(client, row['remote_path'], head)
                if row.get('git_blob_sha') and record.get('sha') != row['git_blob_sha']:
                    raise BackupPause('GitHub_artifact_blob_identity_mismatch')
                _write_chunks(path, (data,), expected, size)
                restored += 1
            elif storage == 'release_asset':
                _download(client, row, path)
                restored += 1
            elif storage == 'release_archive_member':
                archived.append((row, path))
            else:
                raise BackupPause('unsupported_GitHub_artifact_storage')
        if archived:
            transport = manifest.get('archive_transport')
            if not isinstance(transport, dict) or transport.get('kind') != 'deterministic_zip':
                raise BackupPause('missing_GitHub_metadata_archive_transport')
            with tempfile.TemporaryDirectory(prefix='e2-github-metadata-') as temporary:
                archive_path = Path(temporary) / 'metadata.zip'
                _download(client, transport, archive_path)
                try:
                    with zipfile.ZipFile(archive_path) as archive:
                        entries = archive.infolist()
                        names = [entry.filename for entry in entries]
                        allowed = {row['archive_member'] for row in rows if row.get('storage') == 'release_archive_member'}
                        if len(set(names)) != len(names) or set(names) != allowed:
                            raise BackupPause('GitHub_metadata_archive_members_changed')
                        for entry in entries:
                            _target(directory, entry.filename)
                            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1:
                                raise BackupPause('unsafe_GitHub_metadata_archive_member')
                        for row, path in archived:
                            expected, size = _spec(row)
                            member = row.get('archive_member')
                            if row.get('archive_sha256') != transport.get('sha256'):
                                raise BackupPause('GitHub_archive_member_binding_changed')
                            entry = archive.getinfo(member)
                            if entry.file_size != size:
                                raise BackupPause('GitHub_archive_member_size_mismatch')
                            with archive.open(entry) as source:
                                _write_chunks(path, iter(lambda: source.read(1024**2), b''), expected, size)
                            restored += 1
                except (zipfile.BadZipFile, KeyError):
                    raise BackupPause('invalid_GitHub_metadata_archive') from None
        manifests += 1
    return {'status': 'restored_from_GitHub', 'manifests': manifests,
            'files_restored': restored, 'existing_identical_files': reused,
            'training_performed': False, 'budget_reset': False}
