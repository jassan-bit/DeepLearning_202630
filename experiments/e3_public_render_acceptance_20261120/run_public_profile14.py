"""Run the unchanged pinned client against the public Render origin.

AI assistance declared. This is engineering acceptance on saved/adversarial
fixtures, not confirmation, training, calibration or human reproduction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

SOURCE = 'feb83dbc08197bde7076857bd8ff475de197efd2'
ORIGIN = 'https://deeplearning-e3-dashboard.onrender.com'
CLIENT = 'experiments/e3_product_followup_20261120/smoke_profile14.py'
CLIENT5 = 'experiments/e3_cierre_integral_20261118/acceptance/smoke_render_v2.py'
LEGACY5 = 'experiments/e3_bloques_fg_20261117/smoke_render.py'
REFERENCE = 'experiments/e3_product_followup_20261120/acceptance_reference.json'
PROFILE = 'e3_dashboard_v3/profile.json'
INPUTS = {
    'MODEL_CARD.md': (13417, '99bd3f7f60fcb3e1ae1a60cbcec8883a3f602868ad73ae11660e3b34db3ca402'),
    'DATA_STATEMENT.md': (9983, '181eff31c2d328ab8ebd97077aadbee470e06b81bec43119460defe8940828ef'),
    'assets/vendor/flatly-5.3.6.min.css': (233177, '9fc7027d8b071e6a77c882b03a037bbed1d6d83922bda9a4ca370e182d0503bb'),
    CLIENT5: (27517, '54ba00ebf49308fe525f7f4b27b4befa5f98912f2abdc61fc2d818eb314583ad'),
    LEGACY5: (19828, '4dbbc83603d28224cfef4d1f8159eab55fa8dbef7c23081b5feebccf5e039190'),
    CLIENT: (16489, '9d383864f97ad77e0371dbe17cfae974282f986ae0fbce0dacaf584fb74ba7d5'),
    REFERENCE: (6777, '99f0ea4bb4a973dcc724fded76546f33e91ee111dfbe5f30e1ff86dd580b6a9b'),
    PROFILE: (7984, '33eb501dbe7b9bdf525ddfd5c1a4c919191cb1a021adaa8856b382fea1f4947e'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def verify_source(root):
    observed = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    require(observed == SOURCE, 'immutable_source_checkout_mismatch')
    rows = {}
    for relative, (size, expected) in INPUTS.items():
        path = root / relative
        require(not path.is_symlink(), 'reference_symlink_rejected:' + relative)
        raw = path.read_bytes()
        require(len(raw) == size and digest(raw) == expected, 'reference_hash_mismatch:' + relative)
        rows[relative] = {'bytes': len(raw), 'sha256': digest(raw)}
    reference = json.loads((root / REFERENCE).read_bytes())
    require(reference['profile_sha256'] == INPUTS[PROFILE][1], 'reference_profile_binding_mismatch')
    require(len(reference['models']) == 14, 'reference_model_count_mismatch')
    return {'source_commit': observed, 'files': rows,
            'profile_sha256': reference['profile_sha256'], 'models': len(reference['models'])}


def health(output, timeout):
    row = {'started_at_utc': now(), 'origin_HTTP': None, 'origin_response_received': False}
    try:
        with build_opener(NoRedirect()).open(Request(ORIGIN + '/healthz', headers={
                'User-Agent': 'E3-public-profile14-gate/1', 'Accept': 'application/json'}), timeout=timeout) as r:
            raw = r.read(65537)
            row.update(origin_HTTP=r.status, origin_response_received=True)
    except HTTPError as error:
        raw = error.read(65537)
        blocked = any(k.lower() == 'x-mitmproxy-blocked-reason' for k in error.headers)
        row.update(origin_HTTP=None if blocked else error.code,
                   origin_response_received=not blocked,
                   status='proxy_block_before_origin' if blocked else 'origin_HTTP_error')
    except (URLError, OSError, TimeoutError) as error:
        row.update(status='transport_failed_before_origin', error=str(error)[:1000], finished_at_utc=now())
        return row
    row.update(bytes=len(raw), sha256=digest(raw), finished_at_utc=now())
    if len(raw) > 65536:
        row['status'] = 'health_body_limit_exceeded'
        return row
    output.mkdir(parents=True, exist_ok=True)
    (output / 'latest_health.body').write_bytes(raw)
    try:
        row['body'] = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        row['body_preview'] = raw[:500].decode('utf-8', 'replace')
    return row


def public_gate(output, seconds):
    deadline = time.monotonic() + seconds
    attempts = []
    while time.monotonic() < deadline:
        row = health(output, max(.1, min(15, deadline - time.monotonic())))
        attempts.append(row)
        body = row.get('body', {})
        if row['origin_HTTP'] == 200 and isinstance(body, dict) and body.get('status') == 'ok':
            if body.get('source_commit') == SOURCE:
                try:
                    require(body.get('source_commit_source') in ('RENDER_GIT_COMMIT', 'APP_SOURCE_COMMIT'),
                            'deployment_identity_source_missing')
                    require(body.get('profile_predictors') == 14 and body.get('original_UI_predictors') == 5,
                            'expected_commit_is_served_without_profile14')
                    require(body.get('V3_enabled') is False, 'unexpected_V3_enablement')
                    require(body.get('profile_sha256') == INPUTS[PROFILE][1], 'served_profile_hash_mismatch')
                except ValueError as error:
                    return {'status': 'expected_source_wrong_profile', 'attempts': attempts,
                            'health': body, 'reason': str(error)}
                return {'status': 'expected_source_and_profile_ready', 'attempts': attempts, 'health': body}
        if row.get('status') in ('proxy_block_before_origin', 'transport_failed_before_origin'):
            if row.get('status') == 'proxy_block_before_origin' or 'Tunnel connection failed' in row.get('error', ''):
                break
        time.sleep(min(3, max(0, deadline - time.monotonic())))
    return {'status': 'source_or_profile_not_ready', 'attempts': attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warm-seconds', type=float, default=180)
    args = parser.parse_args()
    require(0 < args.warm_seconds <= 180, 'readiness_budget_invalid')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 2, 'launcher_revision': 'v2_complete_five_client_local_asset_fixture',
              'started_at_utc': now(), 'status': 'running',
              'scope': 'public_Render_acceptance_of_pinned_historical14_profile',
              'source_commit': SOURCE, 'origin': ORIGIN, 'python': platform.python_version(),
              'launcher_sha256': digest(Path(__file__).read_bytes()),
              'AI_assistance_declared': True, 'public_Render_acceptance_passed': False,
              'new_training_runs': 0, 'new_calibrations': 0, 'ONNX_reexported': False,
              'new_scientific_forecast_dates': 0, 'GPU_used': False,
              'Render_mutations': 0, 'GitHub_mutations': 0, 'Zenodo_mutations': 0,
              'client_executed': False, 'errors': [],
              'limitations': ['Engineering acceptance is not R1/R2/R3 or independent human training reproduction.',
                              'The SHA is observed from the service; not cryptographic attestation of every remote byte.',
                              'This does not measure Render RAM, energy, uptime or concurrency capacity.']}
    exit_code = 1
    try:
        root = args.reference_root.resolve()
        report['immutable_reference'] = verify_source(root)
        gate = public_gate(output / 'readiness', args.warm_seconds)
        report['public_readiness'] = gate
        require(gate['status'] == 'expected_source_and_profile_ready', 'public_gate_not_passed_no_predictions_issued')
        # Git identity and accepted file hashes bind the references. No unsigned
        # GitHub REST request is needed, so shared runner API quota is irrelevant.
        command5 = [sys.executable, str(root / CLIENT5), '--url', ORIGIN,
                    '--deployment-commit', SOURCE, '--output', str(output / 'five_public_client.json'),
                    '--screenshots-dir', str(output / 'captures_five'), '--wake-seconds', '180',
                    '--navigation-seconds', '30', '--prediction-seconds', '30', '--tolerance', '0.000001']
        report['client_executed'] = True
        result5 = subprocess.run(command5, capture_output=True, text=True, timeout=300)
        (output / 'five_client.stdout.txt').write_text(result5.stdout)
        (output / 'five_client.stderr.txt').write_text(result5.stderr)
        report['five_client_exit_code'] = result5.returncode
        five_path = output / 'five_public_client.json'
        if five_path.exists():
            five_raw = five_path.read_bytes(); five = json.loads(five_raw)
            report['five_report_sha256'] = digest(five_raw)
            report['five_summary'] = {key: five.get(key) for key in (
                'status', 'scope', 'operator_declared_deployment_commit', 'render_public_smoke_passed',
                'expected_live_models', 'deployment_identity', 'errors', 'finished_at_utc')}
            report['five_counts'] = {'models': len(five.get('models', [])), 'pages': len(five.get('pages', []))}
        else:
            five = {}; report['five_summary'] = {'status': 'missing_receipt'}
        command = [sys.executable, str(root / CLIENT), '--url', ORIGIN, '--expected-sha', SOURCE,
                   '--reference', str(root / REFERENCE), '--output', str(output / 'profile14_public_client.json'),
                   '--screenshots-dir', str(output / 'captures')]
        report['client_executed'] = True
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        (output / 'client.stdout.txt').write_text(result.stdout)
        (output / 'client.stderr.txt').write_text(result.stderr)
        report['client_exit_code'] = result.returncode
        client_path = output / 'profile14_public_client.json'
        require(client_path.exists(), 'client_did_not_leave_receipt')
        raw = client_path.read_bytes(); client = json.loads(raw)
        report['client_report_sha256'] = digest(raw)
        report['client_summary'] = {key: client.get(key) for key in (
            'status', 'scope', 'declared_expected_commit', 'public_Render_acceptance_passed',
            'deployment_identity_matched', 'health', 'errors', 'finished_at_utc')}
        report['client_counts'] = {key: len(client.get(key, [])) for key in (
            'API_predictions', 'browser_predictions', 'interval_date_guards', 'OOD_cases',
            'invalid_requests', 'navigation', 'screenshots', 'browser_errors', 'console_errors')}
        require(client.get('scope') == 'public_acceptance', 'unexpected_client_scope')
        require(result.returncode == 0 and client.get('status') == 'passed'
                and client.get('public_Render_acceptance_passed') is True, 'public_client_acceptance_failed')
        require(report['client_counts']['API_predictions'] == 14
                and report['client_counts']['browser_predictions'] == 14, 'incomplete_model_acceptance')
        require(result5.returncode == 0 and five.get('status') == 'passed'
                and five.get('scope') == 'public_Render_browser_smoke'
                and five.get('render_public_smoke_passed') is True, 'original_five_public_client_failed')
        require(report['five_counts']['models'] == 5 and report['five_counts']['pages'] == 3,
                'incomplete_original_five_acceptance')
        report.update(status='passed', public_Render_acceptance_passed=True)
        exit_code = 0
    except Exception as error:
        report.update(status='failed_or_blocked')
        report['errors'].append({'type': type(error).__name__, 'message': str(error)[:1800]})
    report['finished_at_utc'] = now()
    dump(output / 'public_profile14_report.json', report)
    files = {}
    for p in sorted(output.rglob('*')):
        if p.is_file() and p.name != 'files_manifest.json':
            raw = p.read_bytes(); files[str(p.relative_to(output))] = {'bytes': len(raw), 'sha256': digest(raw)}
    dump(output / 'files_manifest.json', {'files': files})
    print('E3_PUBLIC_PROFILE14_RESULT=' + json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True))
    client_path = output / 'profile14_public_client.json'
    if client_path.exists():
        print('E3_PUBLIC_PROFILE14_CLIENT=' + json.dumps(json.loads(client_path.read_bytes()), ensure_ascii=False,
                                                       allow_nan=False, sort_keys=True))
    five_path = output / 'five_public_client.json'
    if five_path.exists():
        print('E3_PUBLIC_FIVE_CLIENT=' + json.dumps(json.loads(five_path.read_bytes()), ensure_ascii=False,
                                                  allow_nan=False, sort_keys=True))
    print('E3_PUBLIC_PROFILE14_FILES=' + json.dumps(files, sort_keys=True))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
