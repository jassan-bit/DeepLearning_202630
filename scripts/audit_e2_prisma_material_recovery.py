"""Audit restored literature provenance without exports, network or training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'references/literature_e2/prisma_material_recovery_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify(bundle=DEFAULT, root=ROOT):
    bundle, root = Path(bundle), Path(root)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        path = bundle / name
        assert path.stat().st_size == record['bytes'], name
        assert sha(path) == record['sha256'], name
    for name, digest in manifest['repository_inputs_sha256'].items():
        assert sha(root / name) == digest, name
    coverage = rows(bundle / 'group_coverage.csv')
    groups = {row['group_id'] for row in coverage}
    assert len(coverage) == len(groups) == 9559
    summary = json.loads((bundle / 'coverage_summary.json').read_text())
    available = sum(row['abstract_available'] == 'True' for row in coverage)
    assert available == summary['groups_with_abstract'] == 8396
    assert len(groups) - available == summary['groups_without_abstract'] == 1163
    assert summary['nonempty_abstract_groups_before_placeholder_check'] == 8416
    assert summary['placeholder_only_groups_not_abstracts'] == 20
    assert summary['original_query_universe_extended'] is False
    links = rows(bundle / 'identifier_links.csv')
    assert len(links) == summary['linked_occurrences'] == 10698
    assert all(row['group_id'] in groups for row in links)
    assert {row['match_rule'] for row in links} <= {
        'exact_identifier', 'exact_normalized_title_year_no_doi_conflict'}
    receipt = json.loads((bundle / 'input_receipt.json').read_text())
    sources = {row['source_id']: row for row in receipt['inputs']}
    expected = {
        'Scopus_global': (7500, '387f673a2146902ce12888ab3d662c095abc33d2cf5b54a168e9e2a12099f7de'),
        'WoS0': (302, '901c724c248348a284dd49c829c0153b199b8e8d8e29e369a191ac6496554281'),
        'WoS1': (500, '6e3b7ec2d5b8b94b2b85aa314678714c061d449213d41c0805059ff859ec4bdb'),
        'WoS2': (500, 'c5d75a485971aecb9bf941e57663732f05fd1fe8c5fb497fd899d0c87511b4f1'),
        'WoS7': (340, '0515f0a00d05e0aead83f56744d267ef148ccada00351f56773856b17a362d8f'),
    }
    for name, (count, digest) in expected.items():
        assert sources[name]['records'] == count
        assert sources[name]['sha256'] == digest
        assert sources[name]['exact_historical_identity']
    assert sum(row['records'] for row in sources.values()) == summary['input_occurrences']
    assert receipt['full_abstracts_published'] is False
    assert receipt['private_local_paths_published'] is False
    assert receipt['original_records_added_to_received_universe'] == 0
    protocol = json.loads((bundle / 'screening_protocol.json').read_text())
    universe = protocol['universe']
    assert universe['received_occurrences'] - universe['within_round_duplicates'] - universe['cross_round_links'] == 9559
    assert universe['scopus_shown_but_not_received'] == 374
    # Metadata and original short adjudications may be public; source bodies stay private.
    for path in bundle.rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.csv', '.md', '.txt'}:
            content = path.read_text()
            assert '/workspace/' not in content and '/content/' not in content, path
            if path.suffix == '.csv':
                assert 'abstract' not in (rows(path)[0] if rows(path) else {}), path
    mirror = root / 'book_e2/_static/e2_prisma_material_recovery_v1'
    if mirror.exists():
        for name in manifest['files']:
            assert (mirror / name).read_bytes() == (bundle / name).read_bytes(), name
        assert (mirror / 'manifest.json').read_bytes() == (bundle / 'manifest.json').read_bytes()
    return {'status': 'verified', 'fixed_received_groups': 9559,
            'restored_abstract_groups': 8396, 'placeholder_groups_not_abstracts': 20,
            'initial_original_files_verified': 4,
            'focused_WoS_original_verified': True, 'new_market_model_fits': 0,
            'abstracts_or_private_paths_published': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT)
    parser.add_argument('--root', type=Path, default=ROOT)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.bundle, arguments.root), ensure_ascii=False, indent=2))
