"""Consolidate the checked source capsule and nine already approved ONNX exports.

Stdlib, public GET only. Never import model code, deserialize checkpoints, infer,
train, publish, or assign a new global license. The complete historical snapshot
is not downloaded again. Prepared with declared AI assistance.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile
import zlib

SOURCE = '8d03a73d529eb3f9b2e89992b4488cfeb88a99e6'
BASE = {'bytes': 58823392, 'sha256': 'd7a97766af592629efafcc64c386c9aeae39b4ab107b909217d9629b9d8a6d35',
        'members': 18564, 'expanded_bytes': 281641001}
NINE = {'url': 'https://github.com/jassan-bit/DeepLearning_202630/releases/download/e3-nine-onnx-4ef050d047b1/e3-nine-onnx-20261120.zip',
        'bytes': 12987835, 'sha256': '0fb857c407250949bfc8c96fd7b9525bab624b650b5c94dd4d1d1d8c60f62c66',
        'manifest_bytes': 11779, 'manifest_sha256': 'e8eb71d5356b20b2950544e52a7a63fcfc22cfa323cadda98e9365db120f90bf',
        'members': 66}
MODELS = {
 'standard/fcn_1d/model.onnx': (1092005, '80df90b1cb38fbde7f47c8d4ca43f04a5df0c6936a740eda1e5ec7a87c8df1d1'),
 'standard/inception_1d/model.onnx': (2070967, 'c8ca0dcdc942fe0ef59c2d79d8ffae688bc37c28e72de789f9dd4f438b5b3428'),
 'standard/lstm_autoencoder/model.onnx': (41397, '0ed6c1348d5bc1ea3312aaf00d59f8651cb4949212eeb38c42a732c8f33480d3'),
 'standard/resnet_1d/model.onnx': (2156012, '01e2283793b019d90f77489bb8cbfb82845638772532c0f00302e98a708a050b'),
 'temporal/itransformer/model.onnx': (3201493, '8a0f3829388ef20d489581ebbbeba64f9192abf05e674de09b4273b90553afcd'),
 'temporal/patchtst/model.onnx': (107745, 'f36ecb0397bd63cd86062e527e8769ae922ece64fe5660d0eaa57f3e65046861'),
 'temporal/tide/model.onnx': (664848, '8c93d7789414f9db51fd0ce25595bbcdb5948c5379e22b2bc00b5e7d518217e4'),
 'temporal/timemixer/model.onnx': (172380, '1cbab4bdf3040a981936220d2899e848f2319a4671dff61f371f25f7d1d8f18c'),
 'temporal/timesnet/model.onnx': (4805411, 'ac110c3cc1c0d6830f03919dce1723e97a0f2ce94bea8aa03161dacd5c1382bf')}
BASE_MODELS = {
 '881dd637837265829aa96900d7a0c864dcdab113a57a91485feb20fd2a443abc',
 '10cedceea5da544268ecda882d9ee3d9f4abc153448a599a3a9947b7cf74d255',
 'ee942ca947998a3c5c76fa21eabe401ce60989558faf82608e8107a830227033',
 '1409d0cf84aac43720b4c6978f9a1bb3b5f6c9832518acab0cba279ba8a88bd2',
 '6928a973a6639513bc94cde15996c0b06bb5615e3c0b1a77d16d47e9c7342f3b',
 'c74530282e04ac06e84932f6a65b3fd5a040829b71396d4324252972b84c5d50'}
ALL_MODELS = BASE_MODELS | {value[1] for value in MODELS.values()}
ARCHIVE_NAME = 'e3-source-evidence-8d03a73-v0.3.1-REVIEW.tar.gz'
PUBLIC_HOSTS = {'github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'}
FORBIDDEN = {'.pt', '.pth', '.npz', '.npy', '.whl', '.pkl', '.pickle', '.safetensors'}
ADDON_EXTENSIONS = {'.py', '.md', '.rst', '.json', '.csv', '.txt', '.yml', '.yaml', '.lock', '.sha256', '.sh', '.cff', '.ini', '.toml'}
C_REFERENCE = {'path': 'references/assessment_predictions.csv', 'bytes': 71967,
               'sha256': '0399c7c80f46ac2fdfccaf218a2866fe110f7ec5051d8484e488299c679ad193'}


def require(ok, message):
    if not ok: raise ValueError(message)


def safe(name):
    require(isinstance(name, str) and bool(name) and '\\' not in name, 'Unsafe member name')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and all(x not in ('', '.', '..') for x in p.parts), 'Unsafe or noncanonical member')
    return p


def sha(raw): return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def json_bytes(value): return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def write(path, raw):
    require(not path.is_symlink(), 'Symlink output forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), 'Output member already exists: ' + path.name)
    path.write_bytes(raw)


def checked(raw, expected, name):
    require(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'], 'Pinned bytes mismatch: ' + name)


def manifest_records(manifest):
    files = manifest.get('files')
    require(isinstance(files, (dict, list)), 'Unsupported member manifest')
    if isinstance(files, dict): records = [dict(value, path=name) for name, value in files.items()]
    else: records = files
    out = {}
    for item in records:
        name = item.get('path'); safe(name)
        require(name not in out and type(item.get('bytes')) is int and item['bytes'] >= 0
                and isinstance(item.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid or duplicate manifest member')
        out[name] = {'bytes': item['bytes'], 'sha256': item['sha256']}
    return out


def read_base(archive, payload, expected=None, expected_models=None):
    expected = BASE if expected is None else expected
    expected_models = BASE_MODELS if expected_models is None else expected_models
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size == expected['bytes'], 'Wrong base archive size/type')
    require(file_sha(archive) == expected['sha256'], 'Wrong base archive SHA256')
    observed = {}; total = 0; manifest_raw = None; onnx = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for item in tar:
            safe(item.name)
            require(item.isfile() and item.name not in observed and 0 <= item.size <= 200_000_000, 'Unsafe/duplicate base member')
            total += item.size; require(total <= 400_000_000, 'Base expansion exceeds bound')
            dest = payload / item.name; dest.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256(); size = 0
            with dest.open('xb') as output:
                stream = tar.extractfile(item)
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk: break
                    output.write(chunk); h.update(chunk); size += len(chunk)
            require(size == item.size, 'Truncated base member')
            observed[item.name] = {'bytes': size, 'sha256': h.hexdigest()}
            if item.name == 'CAPSULE_MANIFEST.json': manifest_raw = dest.read_bytes()
            if PurePosixPath(item.name).suffix.lower() == '.onnx': onnx.add(h.hexdigest())
    require(len(observed) == expected['members'] and total == expected['expanded_bytes'], 'Base inventory differs from executed receipt')
    require(manifest_raw is not None, 'Base capsule manifest missing')
    manifest = json.loads(manifest_raw); records = manifest_records(manifest)
    require(set(observed) == set(records) | {'CAPSULE_MANIFEST.json'}, 'Base manifest member inventory differs')
    for name, wanted in records.items(): require(observed[name] == wanted, 'Base member hash differs: ' + name)
    require(onnx == expected_models, 'Base ONNX identities differ')
    require(manifest.get('source_commit') == SOURCE and manifest.get('capsule_version') == '0.3.0-REVIEW', 'Wrong base source/version')
    require(manifest.get('Zenodo_publication_performed') is False and manifest.get('global_MIT_or_CC_license_assigned') is False, 'Base archival/license scope changed')
    return {'members': len(observed), 'expanded_bytes': total, 'ONNX_unique_hashes': sorted(onnx)}, manifest_raw


def allowed_url(url):
    p = urlsplit(url)
    require(p.scheme == 'https' and p.hostname in PUBLIC_HOSTS and not p.username and not p.password and p.port in (None, 443), 'Unapproved public release URL')
    return url


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, message, headers, url):
        allowed_url(url)
        return super().redirect_request(req, fp, code, message, headers, url)


def get_nine(path):
    if path is not None:
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == NINE['bytes'], 'Wrong local nine-export ZIP')
        raw = path.read_bytes()
    else:
        request = Request(allowed_url(NINE['url']), headers={'User-Agent': 'E3-archive-consolidation/0.3.1'})
        for attempt in range(4):
            try:
                with build_opener(Redirects()).open(request, timeout=90) as response:
                    require(response.status == 200, 'Nine-export public response not HTTP200')
                    raw = response.read(NINE['bytes'] + 1)
                break
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3: raise
            except (URLError, TimeoutError):
                if attempt == 3: raise
            time.sleep(min(15, 3 * (attempt + 1)))
    checked(raw, NINE, 'approved nine-export ZIP')
    return raw


def add_nine(raw, payload, pinned=None, expected_models=None):
    pinned = NINE if pinned is None else pinned
    expected_models = MODELS if expected_models is None else expected_models
    checked(raw, pinned, 'approved nine-export ZIP')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist(); names = [item.filename for item in infos]
        require(len(names) == pinned['members'] and len(names) == len(set(names)), 'Nine-export ZIP inventory differs')
        require(sum(item.file_size for item in infos) < 50_000_000 and archive.testzip() is None, 'Nine-export ZIP CRC/expansion differs')
        for item in infos:
            safe(item.filename)
            require(not item.is_dir() and (item.external_attr >> 16) & 0o170000 != 0o120000 and item.file_size <= 20_000_000, 'Unsafe ZIP member')
            require(PurePosixPath(item.filename).suffix.lower() not in FORBIDDEN, 'Unreviewed model/input payload in nine-export ZIP')
        manifest_raw = archive.read('exports_manifest.json')
        checked(manifest_raw, {'bytes': pinned['manifest_bytes'], 'sha256': pinned['manifest_sha256']}, 'nine-export manifest')
        manifest = json.loads(manifest_raw); records = manifest_records(manifest)
        require(set(names) == set(records) | {'exports_manifest.json'}, 'Nine-export manifest inventory mismatch')
        declared = manifest.get('models', {})
        require(set(declared) == set(expected_models), 'Nine-export model names mismatch')
        seen_models = {}
        for name in names:
            value = archive.read(name)
            if name != 'exports_manifest.json': checked(value, records[name], name)
            if PurePosixPath(name).suffix.lower() == '.onnx':
                require(name in expected_models, 'Unexpected ONNX graph')
                wanted_size, wanted_sha = expected_models[name]
                checked(value, {'bytes': wanted_size, 'sha256': wanted_sha}, name)
                require(declared[name]['bytes'] == wanted_size and declared[name]['sha256'] == wanted_sha, 'Model manifest identity mismatch')
                seen_models[name] = {'bytes': wanted_size, 'sha256': wanted_sha}
            write(payload / 'approved_external_nine' / name, value)
    require(set(seen_models) == set(expected_models), 'Not all nine ONNX graphs included')
    return {'source_commit_observed': manifest.get('source_commit'), 'members_preserved': len(names),
            'models': seen_models, 'all_member_bytes_preserved': True, 'weights_scalers_calibrators_changed': False,
            'manifest_sha256': sha(manifest_raw)}


def add_C(root, manifest_path, manifest_sha, payload):
    require(root is not None and manifest_path is not None and manifest_sha is not None, 'C addon requires root, frozen manifest, and SHA256 together')
    require(not manifest_path.is_symlink() and not root.is_symlink(), 'C addon symlink root/manifest forbidden')
    root = root.resolve(); manifest_path = manifest_path.resolve()
    require(manifest_path.is_relative_to(root) and manifest_path.is_file() and not manifest_path.is_symlink(), 'Invalid C addon manifest')
    require(re.fullmatch('[0-9a-f]{64}', manifest_sha) and file_sha(manifest_path) == manifest_sha, 'C addon manifest SHA256 mismatch')
    raw_manifest = manifest_path.read_bytes(); manifest = json.loads(raw_manifest); records = manifest_records(manifest)
    require(0 < len(records) <= 100 and sum(x['bytes'] for x in records.values()) <= 10_000_000, 'C addon exceeds source-only bound')
    included = []
    for name, wanted in records.items():
        path = root / safe(name)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'Unsafe or missing C addon source')
        require(PurePosixPath(name).suffix.lower() in ADDON_EXTENSIONS and wanted['bytes'] <= 2_000_000, 'Non-source C addon member')
        value = path.read_bytes(); checked(value, wanted, name)
        if PurePosixPath(name).suffix.lower() == '.csv':
            require(name == C_REFERENCE['path'], 'Only the fixed historical C reference CSV is allowed')
            checked(value, C_REFERENCE, name)
            rows = list(csv.DictReader(io.StringIO(value.decode('utf-8'))))
            require(len(rows) == 935 and set(rows[0]) == {'name', 'seed', 'date', 'target', 'prediction'}, 'C reference support/schema differs')
        write(payload / 'tools/reproduction_C_preparation' / name, value); included.append(name)
    manifest_name = manifest_path.relative_to(root).as_posix()
    if manifest_name not in records: write(payload / 'tools/reproduction_C_preparation' / manifest_name, raw_manifest)
    return {'included': True, 'manifest_sha256': manifest_sha, 'members': len(included) + (manifest_name not in records),
            'source_members': included, 'training_executed': False, 'human_execution_claimed': False,
            'reference_outputs_scope': 'frozen_C_reference_outputs_not_new_predictions;935rows=825neural+110analytic from already public development pilot'}


def build(payload, target, provenance):
    records = []
    for path in sorted(payload.rglob('*')):
        if not path.is_file(): continue
        require(not path.is_symlink(), 'Capsule symlink forbidden')
        name = path.relative_to(payload).as_posix(); safe(name)
        require(name != 'CAPSULE_MANIFEST.json', 'Do not carry forward an old root manifest')
        records.append({'path': name, 'bytes': path.stat().st_size, 'sha256': file_sha(path)})
    manifest = {'schema_version': 1, 'capsule_version': '0.3.1-REVIEW', 'source_commit': SOURCE,
                'scope': 'scoped source and evidence archive with explicit exclusions; not integral scientific master',
                'provenance': provenance, 'files': records, 'ONNX_unique_graphs': 15,
                'global_MIT_or_CC_license_assigned': False, 'Zenodo_publication_performed': False,
                'new_model_fits': 0, 'new_model_predictions': 0, 'human_reproduction_executed': False, 'R3_accredited': False}
    manifest_raw = json_bytes(manifest); write(payload / 'CAPSULE_MANIFEST.json', manifest_raw)
    with target.open('xb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0, compresslevel=6) as gz:
        with tarfile.open(fileobj=gz, mode='w|', format=tarfile.PAX_FORMAT) as tar:
            for path in sorted(payload.rglob('*')):
                if not path.is_file(): continue
                info = tarfile.TarInfo(path.relative_to(payload).as_posix()); info.size = path.stat().st_size
                info.mode = 0o644; info.mtime = info.uid = info.gid = 0; info.uname = info.gname = ''
                with path.open('rb') as member: tar.addfile(info, member)
    return manifest, manifest_raw


def verify_final(archive, manifest, expected_models=None):
    expected_models = ALL_MODELS if expected_models is None else expected_models
    expected = manifest_records(manifest); seen = set(); total = 0; models = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for item in tar:
            safe(item.name)
            require(item.isfile() and item.name not in seen and item.size <= 200_000_000, 'Unsafe final member')
            seen.add(item.name); total += item.size; require(total <= 400_000_000, 'Final expanded archive exceeds bound')
            h = hashlib.sha256(); n = 0; capture = [] if item.name == 'CAPSULE_MANIFEST.json' else None
            member = tar.extractfile(item)
            while True:
                chunk = member.read(1024 * 1024)
                if not chunk: break
                h.update(chunk); n += len(chunk)
                if capture is not None: capture.append(chunk)
            require(n == item.size, 'Truncated final member')
            if capture is not None: require(json.loads(b''.join(capture)) == manifest, 'Final inner manifest mismatch')
            else: require(expected.get(item.name) == {'bytes': n, 'sha256': h.hexdigest()}, 'Final member differs: ' + item.name)
            if PurePosixPath(item.name).suffix.lower() == '.onnx': models.add(h.hexdigest())
    require(seen == set(expected) | {'CAPSULE_MANIFEST.json'}, 'Missing final members')
    require(models == expected_models, 'Consolidated archive ONNX identities differ')
    return {'members_verified': len(seen), 'extracted_bytes_hashed': total, 'all_member_hashes_verified': True,
            'unique_ONNX_graphs_verified': len(models), 'historical_ONNX_graphs': 14 if expected_models == ALL_MODELS else None,
            'experimental_V3_graphs': 1 if expected_models == ALL_MODELS else None,
            'model_code_or_graph_executed': False}


def run(args):
    require(not args.output.exists() or not any(args.output.iterdir()), 'Choose a new empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.launcher_commit is not None: require(re.fullmatch('[0-9a-f]{40}', args.launcher_commit), 'Invalid launcher commit')
    work = Path(tempfile.mkdtemp(prefix='consolidate-', dir=args.output)); payload = work / 'payload'; payload.mkdir(); completed = False
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        base_summary, base_manifest_raw = read_base(args.base_archive, payload)
        old_readme = (payload / 'README.md').read_bytes()
        (payload / 'README.md').unlink(); (payload / 'CAPSULE_MANIFEST.json').unlink()
        write(payload / 'provenance/base_capsule_manifest.json', base_manifest_raw)
        write(payload / 'provenance/base_capsule_README.md', old_readme)
        base_report_path = args.base_archive.parent / 'execution_receipt.json'
        require(base_report_path.is_file(), 'Executed base hydration receipt missing')
        base_report = json.loads(base_report_path.read_bytes())
        require(base_report['status'] == 'ephemeral_full_snapshot_verified_and_scoped_capsule_built'
                and base_report['capsule']['sha256'] == BASE['sha256'] and base_report['source_commit'] == SOURCE
                and base_report['launcher_commit'] == '84ee7beb8dde08216ccaffac8f087bb7c67f15c8'
                and base_report['engine_sha256'] == '39b23bea859e9c4c53efa5ca51f97c467c1e62846931bbb943260d2e635dd349'
                and base_report['historical_snapshot']['all_8841_indexed_payload_paths_materialized_and_verified'] is True
                and base_report['historical_snapshot']['all_193_receipt_storage_pairs_verified'] is True
                and base_report['historical_snapshot']['unique_objects_verified'] == 9217,
                'Executed base hydration report differs')
        write(payload / 'provenance/base_hydration_execution_receipt.json', base_report_path.read_bytes())
        nine_raw = get_nine(args.nine_zip); nine_summary = add_nine(nine_raw, payload)
        addon = {'included': False, 'training_executed': False, 'human_execution_claimed': False}
        if any(x is not None for x in (args.C_addon_root, args.C_addon_manifest, args.C_addon_manifest_sha256)):
            addon = add_C(args.C_addon_root, args.C_addon_manifest, args.C_addon_manifest_sha256, payload)
        provenance = {'base_archive': BASE, 'base_launcher_commit': base_report['launcher_commit'],
                      'consolidation_launcher_commit': args.launcher_commit,
                      'nine_export_ZIP': {key: NINE[key] for key in ('url', 'bytes', 'sha256', 'manifest_sha256')},
                      'C_addon': addon, 'existing_documentary_DOI': '10.5281/zenodo.24356217',
                      'existing_documentary_ZIP_unchanged': True, 'full_master_download_repeated': False,
                      'full_master_deposited': False, 'new_model_fits': 0, 'new_model_predictions': 0,
                      'global_license_assigned': False, 'R3_accredited': False}
        write(payload / 'CONSOLIDATION_PROVENANCE.json', json_bytes(provenance))
        write(payload / 'tools/archive_consolidation/consolidate_archive.py', Path(__file__).read_bytes())
        write(payload / 'README.md', (
            '# Source and evidence archive0.3.1-REVIEW\n\n'
            'This revision preserves the reviewed source/evidence capsule from8d03a73 and adds the nine already approved public ONNX exports, unchanged metadata/calibrators/notices, and the declared C preparation addon if present. There are14 historical unique ONNX graphs plus one separately experimental V3 graph. V3 is not enabled in Render and has no new calibrator.\n\n'
            'All underlying source members remain unchanged, including original/sanitized notebook hashes and exclusion inventories. The previous root manifest and README remain under provenance/. The4.27GB scientific master was previously hydrated only in an ephemeral integrity job and is not deposited in this archive. No complete master hydration is repeated here.\n\n'
            'Original component terms and notices remain. No blanket MIT/CC BY license is invented. Raw/unreviewed inputs, unreviewed checkpoints and full master prediction panels retain their exclusions. The documentary0.2.0 DOI and exact ZIP remain unchanged. This review archive is not a new Zenodo deposit, an executed human reproduction, or confirmation of superiority. R3 remains unaccredited. No model fitting, inference, export or calibration was performed.\n').encode())
        target = args.output / ARCHIVE_NAME
        manifest, manifest_raw = build(payload, target, provenance)
        require(target.stat().st_size <= 190_000_000, 'Final compressed archive exceeds190MB bound')
        verification = verify_final(target, manifest)
        (args.output / 'capsule_manifest.json').write_bytes(manifest_raw)
        receipt = {'status': 'scoped_archive_consolidated_with15_exact_approved_ONNX',
                   'started_at_utc': started, 'finished_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'source_commit': SOURCE, 'launcher_commit': args.launcher_commit, 'version': '0.3.1-REVIEW',
                   'archive': {'file': target.name, 'bytes': target.stat().st_size, 'sha256': file_sha(target)},
                   'base_verified': base_summary, 'approved_nine': nine_summary, 'C_addon': addon,
                   'verification': verification, 'full_historical_master_hydration_repeated': False,
                   'full_historical_master_deposited': False, 'existing_documentary_DOI_changed': False,
                   'new_model_fits': 0, 'new_model_predictions': 0, 'new_calibrations': 0,
                   'model_promotion_performed': False, 'V3_serving_enabled': False,
                   'human_independent_reproduction_claimed': False, 'R3_accredited': False,
                   'Zenodo_writes': 0, 'GitHub_writes': 0, 'global_license_assigned': False,
                   'runtime': {'python': platform.python_version(), 'zlib_compile': zlib.ZLIB_VERSION,
                               'zlib_runtime': zlib.ZLIB_RUNTIME_VERSION, 'gzip': 'level6/mtime0/blank filename'},
                   'script_sha256': file_sha(Path(__file__))}
        (args.output / 'consolidation_receipt.json').write_bytes(json_bytes(receipt))
        (args.output / 'RELEASE_NOTES.md').write_text(
            'AI-assisted review archive0.3.1 from immutable source8d03a73, verified previous capsule and nine previously approved frozen ONNX exports.\n\n'
            'Contains14 historical unique graphs and one separately experimental V3 graph, source/evidence inventories and explicit exclusions. No new training, inference, export, calibration, model promotion or R3 accreditation. No raw full scientific master or new Zenodo deposit is claimed. Original terms remain; no global license is assigned. See consolidation_receipt.json and CAPSULE_MANIFEST.json for actual bytes and scope.\n\n'
            'Archive SHA256: '+receipt['archive']['sha256']+'\n')
        require(sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()) <= 200_000_000, 'Final public output exceeds200MB bound')
        completed = True
        print('E3_ARCHIVE_CONSOLIDATION_RECEIPT_BEGIN'); print(json.dumps(receipt, ensure_ascii=False, indent=2)); print('E3_ARCHIVE_CONSOLIDATION_RECEIPT_END', flush=True)
        return receipt
    finally:
        shutil.rmtree(work)
        if not completed:
            for candidate in args.output.glob('*.tar.gz'): candidate.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-archive', type=Path, required=True)
    parser.add_argument('--nine-zip', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--C-addon-root', type=Path)
    parser.add_argument('--C-addon-manifest', type=Path)
    parser.add_argument('--C-addon-manifest-sha256')
    parser.add_argument('--launcher-commit', default=os.environ.get('GITHUB_SHA'))
    run(parser.parse_args())
