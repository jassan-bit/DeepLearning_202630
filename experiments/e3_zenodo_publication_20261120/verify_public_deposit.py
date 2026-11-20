"""Read-only, bounded verification of the owner-published documentary deposit.

Uses only public Zenodo/DOI GET requests. No account, secrets, model execution,
uploads, or license decisions. Prepared with declared AI assistance.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

EXPECTED = {
    'record_id': 24356217,
    'doi': '10.5281/zenodo.24356217',
    'file': 'e3-entregable3-documentary-60a9d988-v0.2.0.zip',
    'bytes': 916861,
    'sha256': 'aa59b73311624f2a92b4497f98bfe182e3c2c320c0e0b092fefee5686bce8349',
    'md5': '61bc587e0e6ea29ab4e3356e2e8b16ef',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def allowed(url):
    p = urlsplit(url)
    require(p.scheme == 'https' and p.hostname in {'zenodo.org', 'doi.org'}
            and not p.username and not p.password and p.port in (None, 443),
            'Unexpected public origin')
    return url


class Redirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        allowed(url)
        return super().redirect_request(request, fp, code, message, headers, url)


def fetch(url, cap):
    request = Request(allowed(url), headers={'User-Agent': 'E3-documentary-public-verifier/0.2.0'})
    for attempt in range(3):
        try:
            with build_opener(Redirects()).open(request, timeout=30) as response:
                raw = response.read(cap + 1)
                require(len(raw) <= cap, 'Public response exceeds declared bound')
                return raw, response.status, response.url
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(5 * (attempt + 1))


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def main():
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    report = {'started_at_utc': now(), 'expected': EXPECTED,
              'status': 'not_verified', 'Zenodo_writes': 0, 'DOI_writes': 0,
              'new_model_fits': 0, 'new_model_predictions': 0,
              'R3_accredited': False, 'full_training_archive': False,
              'human_independent_training_reproduction_verified': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        raw, http, url = fetch('https://zenodo.org/api/records/24356217', 2_000_000)
        record = json.loads(raw)
        (output / 'public_record.json').write_bytes(raw)
        metadata = record.get('metadata', {})
        observed_doi = record.get('doi') or metadata.get('doi') or record.get('pids', {}).get('doi', {}).get('identifier')
        require(http == 200 and str(record.get('id')) == str(EXPECTED['record_id'])
                and observed_doi == EXPECTED['doi'], 'Public record identity mismatch')
        report['public_API'] = {'http': http, 'effective_url': url, 'record_id': record['id'],
                                'doi': observed_doi, 'response_sha256': hashlib.sha256(raw).hexdigest(),
                                'state': record.get('state'), 'is_published': record.get('is_published')}
        report['metadata_observed'] = {key: metadata.get(key) for key in
                                       ['title', 'publication_date', 'version', 'license', 'rights', 'creators']}
        files = record.get('files', [])
        if isinstance(files, dict):
            files = list(files.get('entries', files).values())
        require(isinstance(files, list), 'Unsupported file inventory')
        matches = [entry for entry in files if entry.get('key', entry.get('filename')) == EXPECTED['file']]
        require(len(matches) == 1, 'Expected final archive missing or ambiguous')
        entry = matches[0]
        require(entry.get('size') == EXPECTED['bytes'], 'Final archive size mismatch')
        links = entry.get('links', {})
        download_url = links.get('content') or links.get('download') or links.get('self')
        raw, http, url = fetch(download_url, EXPECTED['bytes'])
        require(http == 200 and len(raw) == EXPECTED['bytes'], 'Final archive incomplete')
        require(hashlib.sha256(raw).hexdigest() == EXPECTED['sha256'], 'Final archive SHA256 mismatch')
        require(hashlib.md5(raw).hexdigest() == EXPECTED['md5'], 'Final archive MD5 mismatch')
        require(entry.get('checksum') in (None, 'md5:' + EXPECTED['md5'], 'sha256:' + EXPECTED['sha256']),
                'Zenodo reported checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            require(len(names) == 66 and len(names) == len(set(names)), 'Final member inventory mismatch')
            require(sum(item.file_size for item in archive.infolist()) < 20_000_000, 'Unbounded archive')
            require(archive.testzip() is None, 'Archive CRC failed')
            manifest = json.loads(archive.read('MANIFEST.json'))
            require(set(names) == set(manifest['files']) | {'MANIFEST.json'}, 'Manifest inventory mismatch')
            for name, wanted in manifest['files'].items():
                path = PurePosixPath(name)
                require(not path.is_absolute() and '..' not in path.parts and '\\' not in name, 'Unsafe member')
                payload = archive.read(name)
                require(len(payload) == wanted['bytes'] and hashlib.sha256(payload).hexdigest() == wanted['sha256'],
                        'Member differs from manifest')
        (output / EXPECTED['file']).write_bytes(raw)
        report['archive'] = {**EXPECTED, 'http': http, 'effective_url': url,
                              'Zenodo_checksum': entry.get('checksum'), 'CRC_verified': True,
                              'member_hashes_verified': 65, 'sha256_verified': True}
        report['additional_public_files'] = sorted(entry.get('key', entry.get('filename', '')) for entry in files
                                                    if entry.get('key', entry.get('filename')) != EXPECTED['file'])
        wanted_metadata = {'title': 'DeepLearning_202630 — Entregable 3: suplemento documental y herramientas de verificación',
                           'publication_date': '2026-11-20', 'version': '0.2.0'}
        report['metadata_changes_needed'] = {key: {'observed': metadata.get(key), 'expected': value}
                                               for key, value in wanted_metadata.items() if metadata.get(key) != value}
        _, http, url = fetch('https://doi.org/' + EXPECTED['doi'], 2_000_000)
        p = urlsplit(url)
        require(http == 200 and p.hostname == 'zenodo.org' and p.path.rstrip('/') in
                {'/records/24356217', '/record/24356217'}, 'DOI target not verified')
        report['DOI_resolution'] = {'http': http, 'effective_url': url, 'expected_record_verified': True}
        report['status'] = 'public_DOI_and_exact_final_archive_verified'
    except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)[:1000]}
    report['finished_at_utc'] = now()
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('E3_ZENODO_VERIFICATION_JSON_BEGIN')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('E3_ZENODO_VERIFICATION_JSON_END')
    if (output / 'public_record.json').exists():
        print('E3_ZENODO_PUBLIC_RECORD_JSON_BEGIN')
        print((output / 'public_record.json').read_text())
        print('E3_ZENODO_PUBLIC_RECORD_JSON_END')
    return 0 if report['status'] == 'public_DOI_and_exact_final_archive_verified' else 2


if __name__ == '__main__':
    raise SystemExit(main())
