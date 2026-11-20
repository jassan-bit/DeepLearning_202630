"""Read-only public DOI/record verification and exact candidate ZIP download.

No credentials, uploads, publication, model execution, or license decisions.
Prepared with AI assistance. Rights are not established by this verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def public_url(url, hosts):
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or parsed.hostname not in hosts or parsed.username
            or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('unapproved public URL')
    return url


class PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl, {'doi.org', 'zenodo.org'})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def get_public(url, max_bytes):
    public_url(url, {'doi.org', 'zenodo.org'})
    request = Request(url, headers={'User-Agent': 'E3-Zenodo-read-only-verifier/1'})
    with build_opener(PublicRedirects()).open(request, timeout=30) as response:
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError('response exceeds declared download budget')
        return body, response.url, response.status


def files_by_name(record):
    files = record.get('files', [])
    if isinstance(files, dict):
        files = list(files.values())
    if not isinstance(files, list):
        raise ValueError('unsupported public file inventory')
    found = {}
    for entry in files:
        name = entry.get('key', entry.get('filename'))
        if not isinstance(name, str):
            raise ValueError('public file has no name')
        if name in found:
            raise ValueError('duplicate public file name')
        found[name] = entry
    return found


def verify_bytes(raw, expected, reported_checksum):
    observed_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected['bytes'] or observed_sha != expected['sha256']:
        raise ValueError('deposited file differs from reviewed candidate')
    md5_verified = False
    if reported_checksum:
        algorithm, _, value = reported_checksum.partition(':')
        if algorithm == 'md5':
            if hashlib.md5(raw).hexdigest() != value:
                raise ValueError('Zenodo MD5 differs from downloaded bytes')
            md5_verified = True
        elif algorithm == 'sha256' and observed_sha != value:
            raise ValueError('Zenodo SHA-256 differs from downloaded bytes')
    return {'bytes': len(raw), 'sha256': observed_sha, 'candidate_SHA256_verified': True,
            'reported_MD5_verified': md5_verified}


def verify_deposit(manifest, max_total_bytes, get=get_public):
    record_id = manifest['record_id']
    if not isinstance(record_id, int) or record_id <= 0:
        raise ValueError('invalid record ID')
    doi = manifest['reserved_doi']
    if doi != '10.5281/zenodo.' + str(record_id):
        raise ValueError('DOI and record identity disagree')
    expected = manifest['files']
    if not isinstance(expected, dict) or not expected:
        raise ValueError('expected file inventory must not be empty')
    if sum(item['bytes'] for item in expected.values()) > max_total_bytes:
        raise ValueError('expected downloads exceed explicit total budget')
    api_url = 'https://zenodo.org/api/records/' + str(record_id)
    body, effective_api_url, status = get(api_url, 2_000_000)
    record = json.loads(body)
    observed_doi = record.get('doi', record.get('metadata', {}).get('doi'))
    if status != 200 or str(record.get('id')) != str(record_id) or observed_doi != doi:
        raise ValueError('public record identity not verified')
    if not record.get('metadata', {}).get('title'):
        raise ValueError('public record title missing')
    _, doi_url, doi_status = get('https://doi.org/' + doi, 2_000_000)
    parsed_doi = urlsplit(doi_url)
    if doi_status != 200 or parsed_doi.hostname != 'zenodo.org' or parsed_doi.path.rstrip('/') not in {
            '/record/' + str(record_id), '/records/' + str(record_id)}:
        raise ValueError('DOI did not resolve to expected public record')
    public_files = files_by_name(record)
    result = {}
    for name, wanted in expected.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or len(path.parts) != 1 or '\\' in name:
            raise ValueError('unsafe expected filename')
        if not isinstance(wanted['bytes'], int) or wanted['bytes'] < 0:
            raise ValueError('invalid expected byte size')
        if not re.fullmatch('[0-9a-f]{64}', wanted['sha256']):
            raise ValueError('invalid expected SHA-256')
        entry = public_files.get(name)
        if not entry or entry.get('size') != wanted['bytes']:
            raise ValueError('expected candidate missing or size differs: ' + name)
        links = entry.get('links', {})
        url = links.get('self', links.get('download'))
        public_url(url or '', {'zenodo.org'})
        raw, actual_url, download_status = get(url, wanted['bytes'])
        if download_status != 200:
            raise ValueError('file download did not return HTTP 200')
        result[name] = {**verify_bytes(raw, wanted, entry.get('checksum')),
                        'effective_url': actual_url}
    return {'status': 'public_DOI_and_expected_candidate_bytes_verified',
            'record_id': record_id, 'doi': doi, 'effective_API_url': effective_api_url,
            'DOI_effective_url': doi_url, 'title': record['metadata']['title'],
            'files': result,
            'additional_public_files_not_verified': sorted(set(public_files) - set(expected)),
            'rights_verified': False, 'archive_integral_completeness_verified': False,
            'human_independent_training_reproduction_verified': False,
            'Zenodo_writes': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deposit-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-total-bytes', type=int, default=20_000_000)
    args = parser.parse_args()
    report = {'started_at_utc': utc_now(), 'status': 'not_verified',
              'Zenodo_writes': 0, 'publication_attempted': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        manifest = json.loads(args.deposit_manifest.read_bytes())
        report['deposit_manifest_sha256'] = hashlib.sha256(args.deposit_manifest.read_bytes()).hexdigest()
        report.update(verify_deposit(manifest, args.max_total_bytes))
    except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError) as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)[:1000]}
    report['finished_at_utc'] = utc_now()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['status'] == 'public_DOI_and_expected_candidate_bytes_verified' else 2)


if __name__ == '__main__':
    main()
