"""Bounded network retries for the twenty already-registered final archives.

The frozen decoder retains aggregation, first-access, and cohort ownership.
Only a transient archive transport can be retried; no fit, prediction, decoding
function, or scientific worker is retried by this adapter.
"""
from __future__ import annotations

import hashlib
from http.client import IncompleteRead
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit

RESERVE_PATH = 'data/manifests/test_raw_reserve_20250101_20260831.json'
RESERVE_SHA256 = 'd2eb4e84fff85ee957e6b02890cead3cec37fd593ec2a8db487dbe137d2d3a10'
MAX_ATTEMPTS = 3
TRANSIENT_HTTP_CODES = frozenset((429, 500, 502, 503, 504))


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()


def _regular_path(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if (not path.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('The registered archive must remain inside its ordinary private cache')
    return path


def load_registered_archives(engine_root):
    """Read only the exact existing acquisition metadata, never market data."""
    root = Path(engine_root).absolute()
    path = _regular_path(root / RESERVE_PATH, root)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RESERVE_SHA256:
        raise ValueError('The original twenty-archive reservation changed')
    value = json.loads(raw)
    months = [f'2025-{month:02d}' for month in range(1, 13)]
    months += [f'2026-{month:02d}' for month in range(1, 9)]
    records = value.get('archives', [])
    if (len(records) != 20 or value.get('calendar_days') != 608
            or [r.get('archive') for r in records] != ['BTCUSDT-1m-' + month + '.zip' for month in months]):
        raise ValueError('Only the twenty inherited January2025--August2026 archives are allowed')
    for record in records:
        name = record['archive']
        expected = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/' + name
        parsed = urlsplit(record.get('url', ''))
        checksum = record.get('sha256', '')
        if (record.get('url') != expected or parsed.hostname != 'data.binance.vision'
                or parsed.scheme != 'https' or parsed.username is not None
                or parsed.password is not None or parsed.port is not None
                or record.get('relative_path') != 'BTCUSDT/1m/monthly/' + name
                or type(record.get('bytes')) is not int or record['bytes'] <= 0
                or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum)):
            raise ValueError('A reserved archive source, size, digest or path differs')
    return tuple(records)


def _transient(error):
    # HTTPError subclasses URLError, so test it first: 404 must never retry.
    if isinstance(error, HTTPError):
        return error.code in TRANSIENT_HTTP_CODES
    return isinstance(error, (URLError, TimeoutError, ConnectionResetError, ConnectionAbortedError, IncompleteRead))


def bounded_archive_transport(original_download, engine_root, raw_root, *,
                              sleep=time.sleep, emit=print):
    """Wrap only the frozen `_download_archive` call with an exact allowlist."""
    records = load_registered_archives(engine_root)
    allowed = {row['archive']: (index + 1, _canonical(row)) for index, row in enumerate(records)}
    raw_root = Path(raw_root).absolute()
    if any(p.is_symlink() for p in (raw_root, *raw_root.parents)):
        raise ValueError('The private archive cache must not follow a symlink')

    def event(name, record, index, **extra):
        emit(json.dumps({'event': name, 'archive': record['archive'],
            'archive_number': index, 'archive_count': len(records), **extra}, sort_keys=True), flush=True)

    def download(record, target, opener):
        bound = allowed.get(record.get('archive')) if isinstance(record, dict) else None
        if bound is None or _canonical(record) != bound[1]:
            raise ValueError('An unregistered final market archive was requested')
        index = bound[0]
        target = _regular_path(target, raw_root)
        if target != raw_root / record['relative_path']:
            raise ValueError('The final archive cache path differs from its reservation')
        cached = target.exists()
        if cached and (not target.is_file() or target.stat().st_size != record['bytes']):
            raise ValueError('A cached final archive has the wrong type or byte count')
        event('final_archive_transport_started', record, index, cached=cached)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                raw = original_download(record, target, opener)
                if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
                    raise ValueError('A cached or downloaded final archive differs from its pinned bytes')
            except (HTTPError, URLError, TimeoutError, ConnectionResetError, ConnectionAbortedError, IncompleteRead) as error:
                if cached or not _transient(error) or attempt == MAX_ATTEMPTS:
                    raise
                event('final_archive_transport_retry', record, index,
                    completed_attempt=attempt, maximum_attempts=MAX_ATTEMPTS,
                    error_type=type(error).__name__, http_status=getattr(error, 'code', None),
                    scientific_retry=False)
                sleep(float(attempt))
            else:
                event('final_archive_transport_completed', record, index,
                    cached=cached, transport_attempts=attempt, bytes=len(raw),
                    sha256=record['sha256'], scientific_retry=False)
                return raw
        raise AssertionError('Unreachable bounded transport state')
    return download
