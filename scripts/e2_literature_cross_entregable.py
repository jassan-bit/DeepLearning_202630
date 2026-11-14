"""Link already recorded E1 primary-reading receipts to received E2 groups.

This documentary index never turns a metadata hit into a full-text reading or
imports E1's stricter eligibility decisions as new E2 decisions. Exact DOI
identity is required. No paper, raw abstract, or scientific experiment is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = Path('references/literature_e1/native_620_screening_20261108/primary_assessments.json')
GROUPS = Path('references/literature_e2/final_audit_20261113/reconciled_groups.csv')
VERSION = 'e2-cross-entregable-primary-evidence-index-v1'


def normalize_doi(value):
    value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', (value or '').strip().casefold())
    return value if value.startswith('10.') else ''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_links(reviews, groups):
    """Index receipt provenance only; exact matches cannot close global review."""
    by_doi = {}
    for group in groups:
        for value in json.loads(group['DOIs']):
            doi = normalize_doi(value)
            if doi:
                by_doi.setdefault(doi, []).append(group)
    links = []
    for review in reviews:
        doi = normalize_doi(review.get('doi'))
        if not doi:
            continue
        for group in by_doi.get(doi, []):
            primary = review.get('primary_sections_assessed') is True
            links.append({
                'E2_group_id': group['group_id'], 'E2_title': group['representative_title'],
                'E2_group_has_received_record': any(c.startswith(('native_', 'arxiv_')) for c in json.loads(group['corpora'])),
                'E1_group_id': review.get('group_id'), 'DOI': doi,
                'E1_title': review.get('title'), 'prior_assessed_at_utc': review.get('assessed_at_utc'),
                'prior_document_access': review.get('document_access'),
                'prior_primary_sections_assessed': primary,
                'prior_E1_disposition_not_automatically_imported': review.get('disposition'),
                'prior_target_quantity': review.get('target_quantity'),
                'prior_source_locators': review.get('source_locators', []),
                'prior_source_urls': review.get('source_urls', []),
                'primary_reading_repeated': False,
                'new_E2_global_eligibility_assigned': False,
                'evidence_role': 'prior_primary_reading_receipt_link' if primary else 'prior_metadata_or_unrecovered_receipt_link',
            })
    return links


def generate(root, destination):
    root, destination = Path(root), Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new documentary directory')
    reviews = json.loads((root/REVIEWS).read_text())['reviews']
    with (root/GROUPS).open(newline='', encoding='utf-8-sig') as stream:
        groups = list(csv.DictReader(stream))
    links = make_links(reviews, groups)
    primary = [r for r in links if r['prior_primary_sections_assessed']]
    summary = {
        'version': VERSION,
        'E1_receipt_records_examined': len(reviews),
        'E1_receipts_with_primary_sections_assessed': sum(r.get('primary_sections_assessed') is True for r in reviews),
        'E2_received_groups': sum(any(c.startswith(('native_', 'arxiv_')) for c in json.loads(g['corpora'])) for g in groups),
        'E2_groups_including_separate_directed_additions': len(groups),
        'exact_DOI_receipt_links': len(links),
        'E2_groups_with_any_exact_prior_receipt': len({r['E2_group_id'] for r in links}),
        'exact_prior_primary_reading_links': len(primary),
        'E2_groups_with_prior_primary_reading_receipt': len({r['E2_group_id'] for r in primary}),
        'prior_primary_access_categories': dict(Counter(r['prior_document_access'] for r in primary)),
        'prior_primary_E1_dispositions_not_new_E2_decisions': dict(Counter(r['prior_E1_disposition_not_automatically_imported'] for r in primary)),
        'new_paper_readings': 0, 'new_bibliographic_queries': 0,
        'E1_eligibility_imported_automatically': False,
        'global_E2_screening_complete': False,
        'global_E2_eligible_or_included_count': None,
        'scope': 'Exact-DOI provenance index of prior E1 receipts, not an E2 full-text eligibility adjudication or a claim of new access.'
    }
    destination.mkdir(parents=True)
    with (destination/'cross_entregable_primary_links.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(links[0])); writer.writeheader()
        for row in links:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k,v in row.items()})
    (destination/'cross_entregable_coverage.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    manifest = {'version': VERSION, 'inputs': {str(p): digest(root/p) for p in [REVIEWS,GROUPS]},
                'script_sha256': digest(__file__),
                'files': {p.name: digest(p) for p in destination.iterdir() if p.is_file()}}
    (destination/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    print(json.dumps(generate(a.root, a.output), ensure_ascii=False, indent=2))
