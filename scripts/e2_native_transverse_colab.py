"""Retrieve native transverse-search receipts without exporting article abstracts.

These three searches supplement the documented application-domain corpus. Native
totals and the first ten metadata records are separate counts; neither establishes
inclusion or a completed systematic review. Network failures remain unknown.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

QUERIES = (
    ('vision_residual', 'all:"residual network" AND all:vision AND (all:transfer OR all:adaptation)'),
    ('vision_patches', 'all:attention AND all:"image patches" AND (all:transfer OR all:adaptation)'),
    ('industrial_recurrent', 'all:"LSTM autoencoder" AND all:industrial AND (all:transfer OR all:adaptation)'),
)
NS = {'a': 'http://www.w3.org/2005/Atom', 'o': 'http://a9.com/-/spec/opensearch/1.1/'}


def parse_native_response(raw):
    if len(raw) > 2 * 1024**2:
        raise ValueError('Native response exceeds the metadata limit')
    root = ET.fromstring(raw)
    text = root.findtext('o:totalResults', namespaces=NS)
    if text is None or not text.strip().isdigit():
        raise ValueError('Native totalResults is absent or invalid')
    records = []
    for entry in root.findall('a:entry', NS):
        identifier = entry.findtext('a:id', namespaces=NS)
        title = entry.findtext('a:title', namespaces=NS)
        if not identifier or not title or not identifier.startswith(('http://arxiv.org/abs/', 'https://arxiv.org/abs/')):
            raise ValueError('Entry is not a native article record')
        records.append({'identifier': identifier, 'title': ' '.join(title.split()),
                        'published': entry.findtext('a:published', namespaces=NS),
                        'authors': [n.text for n in entry.findall('a:author/a:name', NS)]})
    total = int(text)
    if total < len(records) or len(records) > 10:
        raise ValueError('Retrieved/native counts are inconsistent')
    return {'native_total': total, 'retrieved_records': len(records), 'records': records,
            'native_response_sha256': hashlib.sha256(raw).hexdigest()}


def retrieve(destination, *, opener=urlopen, clock=None, sleeper=time.sleep):
    """Attempt each query once; do not block GPU work on unavailable sources."""
    path = Path(destination)
    if path.exists():
        old = json.loads(path.read_bytes())
        if old.get('version') != 'e2-native-transverse-search-colab-v1':
            raise ValueError('Existing search receipt has a different identity')
        return old
    rows = []
    for index, (query_id, query) in enumerate(QUERIES):
        if index:
            sleeper(3.)
        url = 'https://export.arxiv.org/api/query?' + urlencode(
            {'search_query': query, 'start': 0, 'max_results': 10})
        at = clock() if clock else datetime.now(timezone.utc).isoformat()
        row = {'query_id': query_id, 'query': query, 'url': url, 'observed_at_utc': at}
        try:
            with opener(Request(url, headers={'User-Agent': 'E2-academic-review/1.0'}), timeout=25) as response:
                raw = response.read(2 * 1024**2 + 1)
            row.update(parse_native_response(raw), status='native_metadata_retrieved')
        except Exception as error:
            row.update(status='not_retrieved', native_total=None, retrieved_records=None,
                       error_type=type(error).__name__)
        rows.append(row)
    result = {'version': 'e2-native-transverse-search-colab-v1', 'queries': rows,
              'scope': 'native identification totals plus first ten records, no eligibility decisions',
              'abstracts_exported': False, 'raw_responses_exported': False,
              'null_policy': 'Network or parsing failure is unknown, never zero.'}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2).encode() + b'\n')
    return result
