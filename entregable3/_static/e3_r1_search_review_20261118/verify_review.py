"""Offline byte, provenance and scope checks; no academic accreditation."""
from collections import Counter
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile


def verify(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        p = PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts
        raw = (root / name).read_bytes()
        assert len(raw) == expected['bytes'], name
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], name
    status = json.loads((root / 'status.json').read_text())
    assert status['R1_accredited'] is False
    assert status['human_external_review_completed'] is False
    assert status['human_reviewer_name'] is None and status['human_concept'] is None
    assert status['candidate']['trainable_parameters'] == 1210
    assert status['real_campaign_allowed'] is False
    for k in ('new_real_data_fits','new_market_predictions','new_statistical_contrasts'):
        assert type(status[k]) is int and status[k] == 0
    meta = json.loads((root / 'colab_receipts_archive.json').read_text())
    raw = (root / 'colab_receipts.zip').read_bytes()
    assert len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256']
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == 27
        saved = {name: archive.read(name) for name in names}
    original_manifest = json.loads(saved['storage_manifest.json'])
    assert set(saved) == set(original_manifest['files']) | {'storage_manifest.json'}
    for name, expected in original_manifest['files'].items():
        raw = saved[name]
        assert len(raw) == expected['bytes'], name
        assert hashlib.sha256(raw).hexdigest() == expected['sha256'], name
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert blob == expected['git_blob_sha'], name
    receipt = json.loads(saved['search_receipt.json'])
    assert receipt['source_commit'] == status['source_commit']
    assert original_manifest['source_commit'] == receipt['source_commit']
    assert receipt['R1_accredited'] is False and receipt['screening_complete'] is False
    valid = []
    expected_rows = []
    for attempt in receipt['attempts']:
        if 'response_file' in attempt:
            raw = saved[attempt['response_file']]
            assert len(raw) == attempt['response_bytes']
            assert hashlib.sha256(raw).hexdigest() == attempt['response_sha256']
        if attempt['status'] != 'response_parsed_not_screened':
            assert attempt['native_result_count'] is None
            continue  # DBLP HTML bodies are intentionally not valid JSON.
        provider = attempt['provider']
        if provider == 'Crossref':
            native = json.loads(raw)['message']; count = native['total-results']
            titles = [r['title'][0] for r in native['items']]
        elif provider == 'OpenAlex':
            native = json.loads(raw); count = native['meta']['count']
            titles = [r.get('title') or '' for r in native['results']]
        else:
            assert provider == 'arXiv'
            feed = ET.fromstring(raw)
            count = int(feed.findtext('{http://a9.com/-/spec/opensearch/1.1/}totalResults'))
            ns = '{http://www.w3.org/2005/Atom}'
            titles = [' '.join((e.findtext(ns+'title') or '').split()) for e in feed.findall(ns+'entry')]
        assert count == attempt['native_result_count']
        assert titles == [r['title'] for r in attempt['records']]
        assert len(titles) == attempt['returned_records_count'] == 10
        valid.append((provider, attempt['query_id']))
        expected_rows.extend((provider, attempt['query_id'], title) for title in titles)
    assert len(receipt['attempts']) == 30 and len(valid) == 18
    assert Counter(p for p,q in valid) == Counter({'Crossref':6,'arXiv':6,'OpenAlex':6})
    assert len(expected_rows) == 180
    original_csv = list(csv.DictReader(io.StringIO(saved['screening.csv'].decode('utf-8'))))
    assert len(original_csv) == 180
    assert all(r['title_decision'] == 'not_screened' for r in original_csv)
    with (root / 'screening/title_screening.csv').open(newline='') as stream:
        screened = list(csv.DictReader(stream))
    assert len(screened) == 180
    assert Counter((r['provider'],r['q'],r['title']) for r in screened) == Counter(expected_rows)
    assert all(r['reason'] and r['human_review'] == 'False' for r in screened)
    assert all(r['abstract_reviewed_in_this_screen'] == 'False' and r['full_text_reviewed_in_this_screen'] == 'False' for r in screened)
    summary = json.loads((root / 'screening/title_screening_summary.json').read_text())
    assert dict(Counter(r['decision'] for r in screened)) == summary['decision_counts']
    assert summary['title_screening_complete_for_saved_first_pages'] is True
    assert summary['screening_complete'] is False and summary['R1_accredited'] is False
    verdict = json.loads((root / 'ai_review/verdict.json').read_text())
    assert verdict['status'] == 'AIREVIEWONLY' and verdict['reviewer_type'] == 'AI'
    assert verdict['human_external_review_completed'] is False and verdict['R1_accredited'] is False
    assert verdict['reviewer_independence']['independent_human'] is False
    assert verdict['academic_grade'] is None
    return {'status':'passed_integrity_and_scope_only','original_files':27,
            'valid_native_responses':18,'effective_bases':['Crossref','arXiv','OpenAlex'],
            'saved_title_occurrences_reviewed':180,'screening_complete':False,
            'AI_review':'AIREVIEWONLY','human_external_review_completed':False,
            'R1_accredited':False,'new_real_data_fits':0,'new_market_predictions':0}


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
