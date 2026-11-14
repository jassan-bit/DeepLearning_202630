#!/usr/bin/env python3
"""Read-only reconciliation of Top10 identities against received export derivatives.

Writes only this output directory. Does not query a service or mutate the repo.
"""
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import unicodedata

OUT = pathlib.Path(__file__).resolve().parent


def repository_root():
    """Use an explicit repository root or locate the published script's ancestor."""
    configured = os.environ.get('E2_REPOSITORY_ROOT')
    if configured:
        root = pathlib.Path(configured).expanduser().resolve()
        if not (root / 'book_e2').is_dir():
            raise RuntimeError('E2_REPOSITORY_ROOT must point to a repository containing book_e2')
        return root
    for candidate in pathlib.Path(__file__).resolve().parents:
        if (candidate / 'book_e2').is_dir():
            return candidate
    raise RuntimeError('Repository not found among script ancestors; set E2_REPOSITORY_ROOT')


ROOT = repository_root()
BASE = ROOT / 'references/literature_e2'


def norm(value):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode().lower())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


coverage = list(csv.DictReader((BASE/'citation_access_20261114/citation_coverage.csv').open()))
components = list(csv.DictReader((BASE/'final_audit_20261113/current10_score_component_audit.csv').open()))
editorial = {
    '10.48550/arXiv.1611.06455': '10.1109/IJCNN.2017.7966039',
    '10.48550/arXiv.1909.04939': '10.1007/s10618-020-00710-y',
    '10.1038/s41598-019-55320-6': '10.1038/s41598-019-55320-6',
}
paths = {
    'Scopus_received_export_derivative': BASE/'scopus_20261105/records.csv',
    'Scopus_WoS_combined_derivative': BASE/'scopus_20261105/combined_records.csv',
    'WoS_received_export_derivative': BASE/'wos_20261105/records.csv',
}
corpora = {name:list(csv.DictReader(path.open())) for name,path in paths.items()}
rows = []
for s in coverage:
    ids = {s['study_id'].lower(), editorial.get(s['study_id'], '').lower()} - {''}
    title = norm(s['title'])
    for corpus, records in corpora.items():
        hits = []
        for record in records:
            idhit = record.get('DOI','').strip().lower() in ids
            titlehit = norm(record.get('title','')) == title
            if idhit or titlehit:
                hits.append((record, 'DOI' if idhit else 'normalized exact title'))
        if not hits:
            hits = [({}, '')]
        for record, rule in hits:
            rows.append({
                'study_id':s['study_id'], 'models':s['models'], 'title':s['title'],
                'editorial_doi_alias':editorial.get(s['study_id'], ''),
                'corpus':corpus, 'corpus_records':len(records),
                'match_rule':rule, 'matched':bool(record),
                'record_id':record.get('record_id',''),
                'matched_title':record.get('title',''), 'matched_DOI':record.get('DOI',''),
                'Scopus_Cited_by':record.get('Scopus_Cited_by',''),
                'WoS_TC':record.get('WoS_TC',record.get('TC','')),
                'source_occurrences':record.get('source_occurrences',''),
                'source_path':str(paths[corpus].relative_to(ROOT)),
                'not_found_is_zero_citations':False,
                'snapshot':'received_export_labeled_20261105_source_filename_and_WoS_DA_20261106_no_live_refresh',
            })
write_csv('local_export_identity_search.csv', rows)

current=[]
for s in components:
    current.append({
        'model':s['model'], 'study_id':s['study_id'],
        'editorial_doi_alias':editorial.get(s['study_id'],''),
        'C_preserved':s['C'], 'citation_count_preserved':s['citation_count'],
        'citation_snapshot_preserved':s['citation_snapshot'],
        'new_uniform_C':'',
        'C_status':'known_from_received_Scopus_export' if s['C'] else 'unresolved_in_received_export_and_no_new_uniform_provider_snapshot',
        'old_C_replaced':False, 'P_original_preserved':s['P'],
        'selection_membership_changed':False,
    })
write_csv('current10_C_status_v2.csv', current)

manifest={
    'generated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
    'scope':'read-only metadata reconciliation; new output artifacts only',
    'input_paths_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [*paths.values(),BASE/'citation_access_20261114/citation_coverage.csv',BASE/'final_audit_20261113/current10_score_component_audit.csv',BASE/'scopus_20261105/source_manifest.json',BASE/'wos_20261105/source_manifest.json']},
    'corpus_sizes':{name:len(records) for name,records in corpora.items()},
    'ten_models_distinct_studies':len(coverage),
    'known_C_models':sum(bool(s['C']) for s in components),
    'known_C_studies':len({s['study_id'] for s in components if s['C']}),
    'unresolved_C_models':[s['model'] for s in components if not s['C']],
    'unresolved_C_studies':sorted({s['study_id'] for s in components if not s['C']}),
    'missing_imputed_zero':False,
    'new_uniform_provider_snapshot_obtained':False,
    'counts_mixed_across_providers_or_snapshots':False,
    'OpenAlex_previous_failures_retried':False,
    'normalized_C_formula_preserved':'clip(Scopus count/(2026-first_publication_year+1)/487,0,1)',
    'normalizer_scope':'fixed reference from prior received cohort; not global maximum and not re-estimated from new publisher observations',
    'local_search_limits':'Searches received public export derivatives. Raw native source exports are not present in this repository; absence is not database-wide absence.',
}
(OUT/'citation_reconciliation_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:manifest[k] for k in ['corpus_sizes','known_C_models','known_C_studies','unresolved_C_models','new_uniform_provider_snapshot_obtained']},ensure_ascii=False,indent=2))
