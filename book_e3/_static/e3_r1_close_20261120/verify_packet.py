"""Verify this bounded review packet without model code, market or network."""
from pathlib import Path
import csv
import hashlib
import io
import json


def check(root=None):
    root = Path(__file__).resolve().parent if root is None else Path(root).resolve()
    manifest = json.loads((root / 'packet_manifest.json').read_bytes())
    for name, record in manifest['files'].items():
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or
                any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()):
            raise ValueError('Unsafe or missing packet file: ' + name)
        raw = path.read_bytes()
        if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError('Packet file changed: ' + name)
    screening = json.loads((root / 'search_review/abstract_screening_summary.json').read_bytes())
    rows = list(csv.DictReader(io.StringIO((root / 'search_review/abstract_screening_all_50.csv').read_bytes().decode())))
    if len(rows) != 50 or len({r['canonical_record_id'] for r in rows}) != 50:
        raise ValueError('Abstract disposition identities differ')
    if (screening['complete_abstracts_observed'] != 41 or
            screening['indexed_primary_abstract_only'] != 3 or
            screening['publisher_listing_only_not_abstract'] != 1 or
            screening['records_without_observed_abstract_content'] != 6 or
            screening['all_fifty_abstracts_read'] is not False or
            screening['R1_accredited'] is not False):
        raise ValueError('Screening scope differs')
    actual_complete = sum(row['complete_abstract_observed'] == 'True' for row in rows)
    if actual_complete != 41:
        raise ValueError('Complete-abstract count is not supported by records')
    sources = json.loads((root / 'primary_review/sources.json').read_bytes())
    records = sources['source_records']
    if len(records) != 10 or sum(r['access_scope'] == 'full_text_accessible_focal_read' for r in records) != 8:
        raise ValueError('Closest-work reading scope differs')
    citations = json.loads((root / 'primary_review/citation_followup.json').read_bytes())
    if citations['seed_count'] != 3 or citations['verified_edges'] != 6:
        raise ValueError('Citation follow-up scope differs')
    analytic = json.loads((root / 'adaptation_design/analytic_verification.json').read_bytes())
    source = (root / 'adaptation_design/analytic_checks.py').read_bytes()
    if (analytic['status'] != 'passed' or analytic['curvature_and_plugin_risk_cases'] != 100 or
            analytic['new_real_data_fits'] != 0 or analytic['new_model_inference'] != 0 or
            analytic['source_sha256'] != hashlib.sha256(source).hexdigest()):
        raise ValueError('Analytical receipt differs')
    identity = json.loads((root / 'model_identity.json').read_bytes())
    if (identity['current_reference']['parameters'] != 1210 or
            identity['candidate']['parameters'] != 1262 or identity['candidate']['inheritance_of_additive_796_jobs'] is not False or
            identity['R1_accredited'] is not False or identity['R3_accredited'] is not False):
        raise ValueError('Model or accreditation identities differ')
    return {'status': 'verified_bounded_review_packet', 'files_verified': len(manifest['files']),
            'closest_works': 10, 'full_text_focal': 8, 'abstract_dispositions': 50,
            'complete_abstracts': 41, 'partial_abstracts': 3,
            'citation_seeds': 3, 'verified_citation_edges': 6,
            'analytic_finite_law_cases': 100, 'new_model_fits': 0,
            'new_model_inference': 0, 'new_statistical_resamples': 0,
            'R1_accredited': False, 'R3_accredited': False}


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, sort_keys=True))
