"""Validate prospective data-admission METADATA without reading market values.

No network, arrays, prices, candles, model imports, acquisition, training or
inference. A complete declaration can only be ready for external admission
review; it never authenticates a human or enables a confirmation campaign.
"""
from __future__ import annotations
import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

BASE = 'f5c960eefedb5fc49b313dd795b1ec84e784bd78'
KNOWN_EXPOSED = {
    '186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683',
    '722632d5a134a2b2c83d2e62cd929e6d56f2bfd27bc1777b8fd8beb0987ee3a1',
    '0b02d3f47d020dde37e5a18f5371e9608ecce0907e7296760a7281524bc6b510',
    '523beddb7d5252b7855ee562ee99e8bcd95ec932c0937ac21709d17d82f7eead',
}
VALUE_KEYS = {'open', 'high', 'low', 'close', 'volume', 'prices', 'candles',
              'returns', 'targets', 'predictions', 'ohlcv', 'bars'}
GUIDE_AXES = {'size', 'domain', 'noise', 'imbalance', 'dimensionality'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'Duplicate JSON key: ' + key)
        out[key] = value
    return out


def load(path):
    raw = Path(path).read_bytes()
    require(len(raw) <= 2_000_000, 'Metadata exceeds bound')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_pairs)
    reject_values(value)
    return value, hashlib.sha256(raw).hexdigest()


def reject_values(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key.lower() not in VALUE_KEYS, 'Market-value field forbidden: ' + key)
            reject_values(child)
    elif isinstance(value, list):
        for child in value:
            reject_values(child)
    elif isinstance(value, str) and value.startswith(('http://', 'https://')):
        parsed = urlsplit(value)
        require(parsed.scheme == 'https' and not parsed.username and not parsed.password,
                'Credentialed/non-HTTPS metadata URL forbidden')
        for key, values in parse_qs(parsed.query, keep_blank_values=True).items():
            if key.lower() in {'apikey', 'api_key', 'token', 'access_token'}:
                require(all(x in ('{API_KEY}', '${API_KEY}', 'YOUR_API_KEY') for x in values),
                        'Only a placeholder API key may appear in a public metadata URL')


def instant(value):
    require(isinstance(value, str) and value.endswith('Z'), 'UTC timestamp ending Z required')
    result = datetime.fromisoformat(value[:-1] + '+00:00')
    require(result.utcoffset().total_seconds() == 0, 'UTC required')
    return result


def day(value):
    require(isinstance(value, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', value), 'ISO date required')
    return date.fromisoformat(value)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate(registry, declaration):
    reject_values(registry); reject_values(declaration)
    require(registry.get('schema_version') == 1 and registry.get('source_commit') == BASE,
            'Unexpected registry schema/source')
    require(registry.get('kind') == 'concrete_unadmitted_prospective_panel_proposals', 'Wrong scope')
    require(registry.get('current_V3_decision') == 'NO_GO_CONFIRMATION_CURRENT_RECIPE', 'V3 decision changed')
    require(registry.get('original_fixed_calendar_guard_unchanged') is True, 'Historical guard changed')
    require(registry.get('R3_accredited') is False, 'R3 cannot be accredited by metadata')
    require(registry.get('acquisition_or_model_execution_enabled') is False, 'This registry cannot execute')
    panels = registry.get('panels')
    require(isinstance(panels, list) and len(panels) == 4, 'Principal plus three additions required')
    require([p.get('id') for p in panels] == ['F00', 'F01', 'F02', 'F03'], 'Panel identities/order changed')
    require(len({p.get('instrument') for p in panels}) == 4, 'Duplicate instruments')
    require(registry.get('status') == 'DRAFT_NOT_PREREGISTERED_NOT_ADMITTED', 'Proposal promoted to admission')
    blocked = []
    def gate(ok, key, detail):
        if not ok:
            blocked.append({'gate': key, 'detail': detail})
    operator = declaration.get('operator', {})
    gate(bool(operator.get('name')) and bool(operator.get('role')), 'author_identity',
         'Named real author/custodian and relationship declaration are absent')
    gate(declaration.get('authentic_external_declaration') is True, 'external_declaration',
         'Software cannot supply or authenticate author exposure/rights declarations')
    declared_at = declaration.get('declared_at_utc')
    if declared_at is not None:
        instant(declared_at)
    gate(declared_at is not None, 'declaration_timestamp', 'Real UTC declaration timestamp is absent')
    access = declaration.get('panel_access', {})
    panel_reports = []
    file_hash_owners = {}
    for panel in panels:
        pid = panel['id']; observations = access.get(pid, {})
        source = urlsplit(panel.get('source_URL', ''))
        require(source.scheme == 'https' and source.hostname ==
                ('data.binance.vision' if pid == 'F00' else 'www.alphavantage.co'),
                'Provider change requires a new reviewed proposal version')
        if pid != 'F00':
            query = parse_qs(source.query)
            require(query.get('function') == ['TIME_SERIES_DAILY']
                    and query.get('symbol') == [panel.get('instrument')]
                    and query.get('outputsize') == ['full'], 'Raw-daily symbol source template changed')
        require(panel.get('prior_panel_relabelled') is False and panel.get('admitted_unexposed') is False,
                'A future proposal must not relabel known data or claim admission')
        require(panel.get('raw_price_policy') == 'RAW_as_traded_unadjusted', 'Target raw-price policy changed')
        require(panel.get('missing_session_policy') == 'no_forward_fill_no_synthetic_weekend_bars',
                'Session filling would change the task')
        require(panel.get('target', {}).get('returns_count') == 7
                and panel['target'].get('ddof') == 0
                and panel['target'].get('annualized') is False
                and panel['target'].get('required_close_support') == 8, 'Target identity changed')
        window = panel['draft_windows']
        plan_start = day(window['planning_raw_start']); plan_end = day(window['planning_last_target'])
        conf_start = day(window['confirmation_raw_start']); conf_end = day(window['confirmation_last_target'])
        require(plan_start < plan_end < conf_start < conf_end, 'Overlapping/reversed raw support windows')
        require(day(window['confirmation_first_target']) > conf_start, 'No separate confirmation warmup')
        gate(panel.get('licensing', {}).get('specific_use_reviewed_by_owner') is True,
             pid + ':rights', 'Read source terms; the selected-documentary license does not cover provider data')
        gate(panel.get('provider_symbol_metadata_verified') is True,
             pid + ':provider_symbol', 'Capability documentation does not verify this symbol-specific endpoint')
        gate(panel.get('public_repository_and_external_reproducer_access_reviewed') is True,
             pid + ':public_access', 'Guide public-source requirement and independently reproducible access need review')
        gate(panel.get('origin_availability_contract_verified') is True,
             pid + ':availability', 'Date labels alone do not prove bar availability at the prediction origin')
        clock = panel['clock']
        if clock != 'UTC_calendar_daily':
            review = panel.get('task_equivalence_review', {})
            gate(review.get('accepted') is True
                 and review.get('choice') == 'native_observation_ahead_separate_elapsed_time'
                 and bool(review.get('reviewer')) and bool(review.get('rationale')),
                 pid + ':task_clock', 'Next exchange session is not a fixed 24-hour horizon; review the future protocol')
            gate(panel.get('future_venue_calendar', {}).get('pinned_and_reviewed') is True,
                 pid + ':calendar', 'Actual venue/session/holiday and early-close calendar must be pinned')
        if pid != 'F00':
            axes = panel.get('guide_difference_axes', [])
            require(all(x.get('axis') in GUIDE_AXES for x in axes),
                    'Source, ticker or clock alone is not a second listed guide axis')
            verified = {x['axis'] for x in axes if x.get('established') is True}
            gate(len(verified) >= 2, pid + ':two_axes',
                 'At least two listed axes versus the principal must be established before outcome access')
        gate(observations.get('outcomes_seen_or_used_for_selection') is False
             and observations.get('declaration_before_outcome_access') is True
             and valid_hash(observations.get('custodian_access_log_sha256')),
             pid + ':exposure', 'A blank declaration cannot certify author or selector non-exposure')
        freeze = observations.get('real_confirmation_freeze_utc')
        frozen = False
        if freeze is not None:
            frozen = instant(freeze).date() < conf_start
        gate(frozen, pid + ':prospective_freeze', 'Real design/data freeze must precede all confirmation raw support')
        metadata = panel.get('future_file_metadata')
        good_files = isinstance(metadata, list) and bool(metadata)
        if good_files:
            hashes = []
            for item in metadata:
                require(isinstance(item, dict), 'File metadata object required')
                h = item.get('sha256'); hashes.append(h)
                require(h not in KNOWN_EXPOSED, 'Known D00-D03 bytes cannot be relabelled unexposed')
                if valid_hash(h):
                    require(h not in file_hash_owners or file_hash_owners[h] == pid,
                            'Identical file bytes cannot be presented as distinct instruments')
                    file_hash_owners[h] = pid
                good_files = good_files and valid_hash(h) and type(item.get('bytes')) is int and item['bytes'] > 0
                good_files = good_files and bool(item.get('source_version')) and bool(item.get('source_URL'))
            require(len(hashes) == len(set(hashes)), 'Duplicate file hashes in panel metadata')
        gate(good_files, pid + ':materialization', 'Future files, bytes, versions and hashes do not exist in this proposal')
        gate(panel.get('seven_endpoint_boundary_purge_checked') is True,
             pid + ':support_purge', 'Check eight-close supports and purge at least seven target endpoints at every split')
        count = panel.get('confirmation_endpoint_count_verified')
        gate(type(count) is int and count > 0, pid + ':endpoint_count', 'Scheduled count is not actual complete endpoint count')
        gate(panel.get('power_dimensioning_accepts_R_and_T') is True,
             pid + ':dimensioning', 'Draft dates and expected sizes are not final sufficient R/T')
        panel_reports.append({'id': pid, 'instrument': panel['instrument'], 'clock': clock,
                              'admitted_unexposed': False, 'price_or_array_read': False})
    return {'status': 'BLOCKED_UNADMITTED_FUTURE_PROPOSALS' if blocked else 'DATA_METADATA_READY_FOR_EXTERNAL_ADMISSION_REVIEW',
            'blocked_gates': blocked, 'blocked_gate_count': len(blocked), 'concrete_instrument_proposals': 4,
            'panels': panel_reports, 'datasets_admitted': 0, 'human_identity_authenticated': False,
            'rights_clearance_inferred': False, 'confirmation_enabled': False, 'R3_accredited': False,
            'market_dataset_or_quote_downloads': 0, 'market_values_read': 0, 'new_model_fits': 0,
            'new_model_predictions': 0, 'network_requests': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registry, registry_hash = load(args.registry); declaration, declaration_hash = load(args.declaration)
    result = validate(registry, declaration)
    result.update(checked_at_utc=datetime.now(timezone.utc).isoformat(), registry_sha256=registry_hash,
                  declaration_sha256=declaration_hash)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    require(not args.output.exists(), 'Do not overwrite a prior receipt')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result['blocked_gates'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
