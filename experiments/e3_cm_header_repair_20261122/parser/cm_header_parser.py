"""Narrow COIN-M archive-header compatibility layer over immutable S2.

No files are written. Only the exact documented legacy generic CSV header is
adapted in memory, then the original S2 COIN-M numeric parser runs unchanged.
Source archive bytes, checksums, calendars, target and base-volume column remain
the original ones. AI assistance declared.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
from pathlib import Path
import zipfile

VERSION = "coinm-exact-generic-header-compat-1"
SOURCE_COMMIT = "db74afd0763596f71bcbafb4b9150a44a2e83339"
SOURCE_DATA_SHA256 = "1e64a500a62b3ec803471f87a2a81161a81d0e73d2c4cce7fd17d73b2b044e60"

# Exact complete header reproduced with a COIN-M archive URL in the upstream
# first-hand report binance/binance-public-data#283. The README's CM schema,
# not this generic field label, fixes column7 as BASE asset volume.
LEGACY_CM_HEADER = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
)
LEGACY_HEADER_EVIDENCE = "https://github.com/binance/binance-public-data/issues/283"
PHYSICAL_SCHEMA_EVIDENCE = (
    "https://github.com/binance/binance-public-data/blob/"
    "d571690eb66760763b0bd6e27e8cd35603c7e244/README.md"
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _adapt_exact_header(raw):
    """Return a temporary ZIP only for the exact first-row legacy CM schema.

    All payload bytes after the first physical CSV line are retained. This
    temporary representation is never persisted or substituted for the raw
    manifest/checksum source. Other schemas are delegated to S2 unchanged.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [info for info in archive.infolist() if info.filename.endswith('.csv')]
        if len(members) != 1:
            return None
        member = members[0]
        original_csv = archive.read(member)
        text = original_csv.decode('utf-8-sig')
        first = next(csv.reader(io.StringIO(text)), None)
        if first is None or tuple(first) != LEGACY_CM_HEADER:
            return None
        # The exact allowed field names contain no quoted newline; therefore
        # only the physical first line can encode that one logical CSV row.
        end = text.find('\n')
        if end < 0:
            first_line, rest, newline = text, '', ''
        else:
            first_line, rest = text[:end], text[end+1:]
            newline = '\r\n' if first_line.endswith('\r') else '\n'
        fixed = list(LEGACY_CM_HEADER)
        fixed[7] = 'base_volume'
        adapted_csv = (','.join(fixed) + newline + rest).encode('utf-8')
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as adapted:
            for info in archive.infolist():
                body = adapted_csv if info.filename == member.filename else archive.read(info)
                adapted.writestr(copy.copy(info), body)
        return output.getvalue()


def bind_parser(original_data):
    """Return a compatible parser bound to the exact, unchanged S2 data file.

    The caller explicitly installs this function for its subprocess and records
    this add-on's hash in the operational envelope. This function does not edit
    S2, change its adapter/pending metadata, or bypass source/rights validation.
    """
    source = Path(original_data.__file__).resolve()
    if _sha(source.read_bytes()) != SOURCE_DATA_SHA256:
        raise RuntimeError('COIN-M header repair requires the exact immutable S2 engine/data.py')
    original = original_data.parse_binance_zip
    if getattr(original, '__cm_header_repair_version__', None) == VERSION:
        return original
    if Path(original.__code__.co_filename).resolve() != source:
        raise RuntimeError('S2 parser was already replaced; preserve that execution and investigate')

    events = []
    recorded = set()

    def parse_binance_zip(raw, *, acquired_utc=None, market_type='spot'):
        if market_type != 'cm':
            return original(raw, acquired_utc=acquired_utc, market_type=market_type)
        try:
            adapted = _adapt_exact_header(raw)
        except (zipfile.BadZipFile, UnicodeError, csv.Error):
            # Preserve S2's refusal/exception for malformed input. In particular
            # no unknown first row is skipped merely because it is text.
            return original(raw, acquired_utc=acquired_utc, market_type=market_type)
        if adapted is None:
            return original(raw, acquired_utc=acquired_utc, market_type=market_type)
        result = original(adapted, acquired_utc=acquired_utc, market_type='cm')
        digest = _sha(raw)
        if digest not in recorded:
            events.append({
                'repair_version': VERSION,
                'source_archive_sha256': digest,
                'observed_header': list(LEGACY_CM_HEADER),
                'interpretation_basis': 'Frozen official COIN-M physical schema, not generic CSV field name',
                'base_volume_column_zero_based': 7,
                'volume_units': 'base_asset',
                'raw_source_bytes_rewritten': False,
                'source_data_sha256': SOURCE_DATA_SHA256,
            })
            recorded.add(digest)
        return result

    parse_binance_zip.__cm_header_repair_version__ = VERSION
    parse_binance_zip.__cm_header_repair_events__ = events
    parse_binance_zip.__cm_header_original_parser__ = original
    return parse_binance_zip
