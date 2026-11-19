"""Repair a restored persistence view, then resume the unchanged 80 GB audit.

AI assistance declared. Copies only already-recorded, hash-verified files into
missing view paths. Never edits the scientific checkout, output files, state,
protocol, checkpoints or phase deadlines; never starts a new campaign.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import urllib.request

SOURCE = "5a07d6157a9796ca54416dc5dea1e1527e731597"
MASTER = "20261119T031104515520Z"
PROTOCOL = "5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a"
OLD_WRAPPER_SHA256 = "0b187e0654bc6931760d99c6772a87b41ce99d5851f6a78c6f0ad42d1f51fee7"
ALLOWED = {'.json', '.csv', '.txt', '.md', '.log', '.zip', '.CHECKSUM', '.png', '.xml', '.pt', '.npz', '.svg'}
MAX_FILE = 48 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _regular(path):
    path = Path(path).absolute()
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "Recovery refuses symlinks")
    return path


def _safe(name):
    if not isinstance(name, str) or not 0 < len(name) <= 512:
        return False
    p = PurePosixPath(name)
    return (str(p) == name and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and not any(ord(c) < 32 for c in name)
            and p.suffix in ALLOWED and not p.name.endswith('.tmp')
            and not p.name.startswith('.partial-'))


def _metadata(meta):
    return (isinstance(meta, dict) and set(meta) == {'bytes', 'sha256', 'git_blob_sha'}
            and type(meta['bytes']) is int and 0 <= meta['bytes'] <= MAX_FILE
            and isinstance(meta['sha256'], str) and re.fullmatch('[0-9a-f]{64}', meta['sha256'])
            and isinstance(meta['git_blob_sha'], str) and re.fullmatch('[0-9a-f]{40}', meta['git_blob_sha']))


def _verified_bytes(path, meta, name):
    _regular(path)
    _require(path.is_file(), "Recorded file also missing from restored output: " + name)
    _require(path.stat().st_size == meta['bytes'], "Restored file size differs: " + name)
    raw = path.read_bytes()
    actual = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
              'git_blob_sha': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}
    _require(actual == meta, "Restored file hashes differ: " + name)
    return raw


def seed_view(base, *, required=False):
    base = _regular(base)
    output = _regular(base / ('e3-extended-audit-' + MASTER))
    view = _regular(base / ('e3-extended-persist-view-' + MASTER))
    statepath = _regular(base / ('e3-extended-continuation-' + MASTER + '.json'))
    protocolpath = _regular(output / 'protocol.json')
    if not (statepath.is_file() and protocolpath.is_file()):
        _require(not required, "The original campaign must finish restoring before view recovery")
        return {'status': 'awaiting_original_restore', 'copied_files': 0}
    state_raw = statepath.read_bytes()
    state = json.loads(state_raw)
    _require(isinstance(state, dict) and state.get('kind') == 'e3_audit_continuation_state'
             and state.get('schema_version') == 1
             and state.get('repository') == 'jassan-bit/DeepLearning_202630'
             and state.get('source_commit') == SOURCE and state.get('master_run_id') == MASTER
             and state.get('protocol_sha256') == PROTOCOL, "Another campaign or protocol was found")
    _require(hashlib.sha256(protocolpath.read_bytes()).hexdigest() == PROTOCOL,
             "The original frozen protocol bytes differ")
    if state.get('pending') is not None:
        _require(not required, "Pending GitHub snapshot must be reconciled by the original controller first")
        return {'status': 'awaiting_pending_reconciliation', 'copied_files': 0}
    inventory = state.get('files')
    _require(isinstance(inventory, dict) and inventory, "Saved file inventory is absent")
    missing = []
    for name, meta in sorted(inventory.items()):
        _require(_safe(name) and _metadata(meta), "Unsafe path or metadata in the saved inventory")
        destination = _regular(view / name)
        if destination.exists():
            _require(destination.is_file(), "Persistence view has a conflicting path: " + name)
            continue
        source = _regular(output / name)
        _verified_bytes(source, meta, name)
        missing.append((name, source, destination, meta))
    # Every missing file is validated before any publication. A second check
    # binds the bytes copied, without loading Torch checkpoints as objects.
    copied = []
    for name, source, destination, meta in missing:
        raw = _verified_bytes(source, meta, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _regular(destination)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.partial-view-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                _verified_bytes(destination, meta, name)
        finally:
            temporary.unlink(missing_ok=True)
        copied.append(name)
    _require(statepath.read_bytes() == state_raw, "Continuation state changed during recovery")
    for name in inventory:
        _require(_regular(view / name).is_file(), "A saved path is still absent from the persistence view")
    report = {'status': 'previously_saved_inventory_present', 'copied_files': len(copied),
              'copied_bytes': sum(inventory[n]['bytes'] for n in copied), 'paths': copied,
              'state_unchanged': True, 'source_commit': SOURCE, 'master_run_id': MASTER,
              'new_real_data_fits': 0, 'new_campaign_started': False}
    print('Vista de guardado recuperada:', json.dumps(report, ensure_ascii=False), flush=True)
    return report


def _expected_view_error(error, base):
    if (type(error).__name__ != 'StorageError'
            or str(error) != 'continuation_previously_saved_file_removed'):
        return False
    expected = (base / ('e3-extended-source-' + SOURCE)
                / 'experiments/e3_evaluacion_ampliada_20261119/orchestration/continuation.py').resolve()
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        values = frame.f_locals
        location = values.get('output')
        if (frame.f_code.co_name == 'save_changed'
                and Path(frame.f_code.co_filename).resolve() == expected
                and values.get('source') == SOURCE and values.get('master_run_id') == MASTER
                and values.get('protocol_sha256') == PROTOCOL
                and isinstance(location, (str, os.PathLike))
                and Path(location).absolute()
                == (base / ('e3-extended-persist-view-' + MASTER)).absolute()):
            return True
        trace = trace.tb_next
    return False


def main(namespace, *, base=None):
    base = Path('/content') if base is None else Path(base)
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    print('GPU asignada:', gpu, flush=True)
    _require(gpu == 'NVIDIA A100-SXM4-80GB', 'Resume requires the original A100 80 GB cohort')
    namespace['_E3_VIEW_RECOVERY_REPORT'] = seed_view(base)
    url = ('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'
           '0307a6878d0d1854e6b5838a078001612fe7ad03/'
           'runs/e3_tree_read_resume/20261119T061446338127Z/resume_cell.py')
    with urllib.request.urlopen(url, timeout=60) as response:
        wrapper = response.read()
    _require(hashlib.sha256(wrapper).hexdigest() == OLD_WRAPPER_SHA256,
             'Immutable original resume wrapper hash differs')
    for attempt in range(2):
        try:
            exec(compile(wrapper, 'e3_original_80GB_resume', 'exec'), namespace)
            return
        except Exception as error:
            if attempt or not _expected_view_error(error, base):
                raise
            report = seed_view(base, required=True)
            _require(report['copied_files'] > 0, 'No verified missing paths were recovered; preserve this error')
            namespace['_E3_VIEW_RECOVERY_REPORT'] = report
            print('Se reanuda el mismo master con la vista completa; no se crea otra campaña.', flush=True)


if __name__ == '__main__':
    main(globals())
