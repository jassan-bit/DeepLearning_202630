#!/usr/bin/env python3
"""Read source CSVs; write a reproducible identifier/title comparison here only.

No eligibility decisions, network access, source changes, or automatic merges.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = Path('/workspace/attachments/6e81a805-1d7a-4ed4-af4a-6077bd0ad04e/export_b93b2bae-2696-49cc-ae91-a4e4a8593990_2026-11-14T140618.516834071.csv')
CATALOG = Path('/workspace/e2-guide-closure-20261113/references/literature_e2/final_audit_20261113')
NODES = CATALOG / 'source_nodes.csv'
GROUPS = CATALOG / 'reconciled_groups.csv'
EXPECTED_SHA256 = 'ef9650c38364f5f0a3b75e592719fda2689a0f83544bce16386ddceb7e70eb4b'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def normalize_doi(value):
    # Deliberately preserve suffix punctuation rather than guessing corrections.
    return re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)', '', value.strip(), flags=re.I).strip().casefold()


def normalize_title(value):
    # Exact equality after Unicode, case, and whitespace normalization only.
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def js(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(filename, rows, fieldnames):
    with (OUT / filename).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_csv', type=Path, help='Original Scopus CSV (read only)')
    parser.add_argument('--catalog-dir', type=Path, default=CATALOG, help='Directory containing source_nodes.csv and reconciled_groups.csv')
    parser.add_argument('--output-dir', type=Path, default=OUT, help='Directory for comparison results; defaults to script directory')
    parser.add_argument('--expected-sha256', default=EXPECTED_SHA256)
    parser.add_argument('--expected-rows', type=int, default=1104)
    args = parser.parse_args()
    source = args.raw_csv.resolve()
    nodes_path = args.catalog_dir.resolve() / 'source_nodes.csv'
    groups_path = args.catalog_dir.resolve() / 'reconciled_groups.csv'
    OUT = args.output_dir.resolve()
    assert sha256(source) == args.expected_sha256, 'Unexpected export SHA-256'
    incoming, nodes, groups = map(read_csv, (source, nodes_path, groups_path))
    assert len(incoming) == args.expected_rows, 'Unexpected export row count'
    OUT.mkdir(parents=True, exist_ok=True)
    by_node = {row['node_id']: row for row in nodes}
    by_group = {row['group_id']: row for row in groups}
    assert len(by_node) == len(nodes)
    assert len(by_group) == len(groups)

    doi_nodes, doi_groups = collections.defaultdict(set), collections.defaultdict(set)
    title_nodes, title_groups = collections.defaultdict(set), collections.defaultdict(set)
    group_dois = {}
    for row in nodes:
        doi, title = normalize_doi(row['doi']), normalize_title(row['title'])
        if doi:
            doi_nodes[doi].add(row['node_id'])
        if title:
            title_nodes[title].add(row['node_id'])
    for row in groups:
        dois = sorted({normalize_doi(value) for value in json.loads(row['DOIs']) if normalize_doi(value)})
        group_dois[row['group_id']] = dois
        for doi in dois:
            doi_groups[doi].add(row['group_id'])
        title = normalize_title(row['representative_title'])
        if title:
            title_groups[title].add(row['group_id'])

    ledger, candidates, ambiguities = [], [], []
    for index, row in enumerate(incoming, 1):
        doi, title = normalize_doi(row['DOI']), normalize_title(row['Title'])
        matching_nodes = sorted(doi_nodes.get(doi, []))
        matching_groups = sorted(doi_groups.get(doi, []))
        title_node_ids = sorted(title_nodes.get(title, []))
        representative_group_ids = sorted(title_groups.get(title, []))
        all_title_group_ids = sorted(set(representative_group_ids) | {
            by_node[node_id]['unified_group_id'] for node_id in title_node_ids
            if by_node[node_id]['unified_group_id']
        })
        doi_match = bool(matching_nodes or matching_groups)
        title_candidate = bool(title_node_ids or representative_group_ids)
        if doi_match:
            status = 'doi_match'
        elif doi:
            status = 'doi_unseen_with_title_candidate' if title_candidate else 'doi_unseen_no_title_candidate'
        else:
            status = 'no_doi_with_title_candidate' if title_candidate else 'no_doi_no_title_candidate'
        record = {
            'record_number': index,
            'csv_physical_line': index + 1,
            'eid': row['EID'],
            'doi_raw': row['DOI'],
            'doi_normalized': doi,
            'title': row['Title'],
            'title_normalized': title,
            'year': row['Year'],
            'source_title': row['Source title'],
            'document_type': row['Document Type'],
            'comparison_status': status,
            'doi_match_node_count': len(matching_nodes),
            'doi_match_node_ids': js(matching_nodes),
            'doi_match_group_count': len(matching_groups),
            'doi_match_group_ids': js(matching_groups),
            'title_exact_node_count': len(title_node_ids),
            'title_exact_node_ids': js(title_node_ids),
            'title_exact_representative_group_ids': js(representative_group_ids),
            'title_candidate_all_group_count': len(all_title_group_ids),
            'title_candidate_all_group_ids': js(all_title_group_ids),
        }
        ledger.append(record)

        # Candidate evidence is only emitted when identifier matching failed.
        if not doi_match and title_candidate:
            if len(all_title_group_ids) > 1:
                ambiguities.append(record)
            for entity_type, ids in [('source_node', title_node_ids), ('reconciled_group', representative_group_ids)]:
                for entity_id in ids:
                    if entity_type == 'source_node':
                        prior = by_node[entity_id]
                        prior_title = prior['title']
                        prior_dois = [normalize_doi(prior['doi'])] if normalize_doi(prior['doi']) else []
                        prior_group = prior['unified_group_id']
                        prior_year = prior['year']
                    else:
                        prior = by_group[entity_id]
                        prior_title = prior['representative_title']
                        prior_dois = group_dois[entity_id]
                        prior_group = entity_id
                        prior_year = ''
                    relation = 'incoming_doi_missing' if not doi else ('prior_doi_missing' if not prior_dois else 'different_doi')
                    candidates.append({
                        'record_number': index, 'eid': row['EID'], 'incoming_doi': doi,
                        'incoming_title': row['Title'], 'incoming_year': row['Year'],
                        'comparison_status': status, 'match_method': 'exact_NFKC_casefold_whitespace',
                        'prior_entity_type': entity_type, 'prior_entity_id': entity_id,
                        'prior_title': prior_title, 'prior_year': prior_year,
                        'prior_dois': js(prior_dois), 'prior_group_id': prior_group,
                        'doi_relation': relation,
                        'all_title_candidate_group_ids': js(all_title_group_ids),
                        'needs_manual_identity_review': 'true',
                    })

    fields = list(ledger[0])
    # CSV row is a logical record; embedded source newlines may affect physical lines.
    for row in ledger:
        row.pop('csv_physical_line')
    fields.remove('csv_physical_line')
    matches = [row for row in ledger if row['comparison_status'] == 'doi_match']
    unseen = [row for row in ledger if row['doi_normalized'] and row['comparison_status'] != 'doi_match']
    missing = [row for row in ledger if not row['doi_normalized']]
    title_records = [row for row in ledger if row['comparison_status'].endswith('_with_title_candidate')]
    for filename, rows in [
        ('all_records.csv', ledger), ('doi_matches.csv', matches),
        ('new_doi_records.csv', unseen), ('no_doi_records.csv', missing),
        ('title_candidate_records.csv', title_records), ('title_ambiguities.csv', ambiguities),
    ]:
        write_csv(filename, rows, fields)
    write_csv('title_candidates.csv', candidates, [
        'record_number', 'eid', 'incoming_doi', 'incoming_title', 'incoming_year',
        'comparison_status', 'match_method', 'prior_entity_type', 'prior_entity_id',
        'prior_title', 'prior_year', 'prior_dois', 'prior_group_id', 'doi_relation',
        'all_title_candidate_group_ids', 'needs_manual_identity_review',
    ])

    doi_counts = collections.Counter(normalize_doi(row['DOI']) for row in incoming if normalize_doi(row['DOI']))
    eid_counts = collections.Counter(row['EID'].strip() for row in incoming if row['EID'].strip())
    title_counts = collections.Counter(normalize_title(row['Title']) for row in incoming if normalize_title(row['Title']))
    summary = {
        'input_files': [{'path': str(path), 'sha256': sha256(path), 'rows': count} for path, count in [
            (source, len(incoming)), (nodes_path, len(nodes)), (groups_path, len(groups)),
        ]],
        'scope': 'Identifier and conservative title comparison only; no eligibility decisions or automatic merges.',
        'record_number_definition': '1-based logical data record, excluding CSV header; not physical line number.',
        'doi_normalization': 'Trim; remove leading http(s)://[dx.]doi.org/ or doi:; trim; Unicode casefold. No suffix punctuation correction.',
        'title_normalization': 'Unicode NFKC, Unicode casefold, collapse whitespace; preserve punctuation, words and word order. Exact equality only.',
        'incoming_rows': len(incoming),
        'incoming_doi_present_rows': sum(doi_counts.values()),
        'incoming_unique_dois': len(doi_counts),
        'incoming_doi_duplicate_groups': {key: value for key, value in sorted(doi_counts.items()) if value > 1},
        'incoming_missing_doi_rows': len(missing),
        'incoming_eid_present_rows': sum(eid_counts.values()),
        'incoming_unique_eids': len(eid_counts),
        'incoming_eid_duplicate_groups': {key: value for key, value in sorted(eid_counts.items()) if value > 1},
        'incoming_normalized_title_duplicate_groups': {key: value for key, value in sorted(title_counts.items()) if value > 1},
        'catalog_unique_dois_in_nodes': len(doi_nodes),
        'catalog_unique_dois_in_groups': len(doi_groups),
        'catalog_dois_only_in_nodes': sorted(set(doi_nodes) - set(doi_groups)),
        'catalog_dois_only_in_groups': sorted(set(doi_groups) - set(doi_nodes)),
        'doi_match_rows': len(matches),
        'new_doi_rows_relative_to_catalog': len(unseen),
        'comparison_status_counts': dict(sorted(collections.Counter(row['comparison_status'] for row in ledger).items())),
        'title_candidate_rows_without_doi_match': len(title_records),
        'title_candidate_evidence_rows': len(candidates),
        'title_candidate_rows_with_multiple_prior_groups': len(ambiguities),
        'title_candidate_rows_with_incoming_doi': sum(bool(row['doi_normalized']) for row in title_records),
        'title_candidate_rows_without_incoming_doi': sum(not row['doi_normalized'] for row in title_records),
        'new_doi_title_candidates_with_different_prior_doi_records': sorted({row['record_number'] for row in candidates if row['doi_relation'] == 'different_doi'}),
        'limits': [
            'New DOI means absent from these two local catalog CSVs, not a new study, new publication, or eligible study.',
            'Exact normalized title agreement provides a candidate only; no records were merged.',
            'Different editions, conference containers, preprints, or publications may share a title; differing DOI or multiple groups require manual identity review.',
            'Conservative title matching can miss differences in punctuation, wording, subtitles, transliteration, accents or word order; no fuzzy or semantic matching used.',
            'No DOI registry checks, full-text review, abstract screening, network requests or source metadata corrections performed.',
            'No dedicated EID comparison was used because these catalog files lack a dedicated EID column; incoming EID is retained as provenance.',
            'Source eligibility and screening columns are not interpreted or copied into the comparison outputs.',
        ],
    }
    assert len(matches) + len(unseen) + len(missing) == len(incoming)
    assert sum(summary['comparison_status_counts'].values()) == len(incoming)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    (OUT / 'METHOD.txt').write_text(
        'Scopus export vs. prior local catalog: identifier/title comparison\n\n'
        f'Reproduce: python {Path(__file__).resolve()} {source} --catalog-dir {args.catalog_dir.resolve()} --output-dir {OUT}\n'
        'Inputs are read only; output writes are confined to the script directory.\n'
        'The script validates the expected input export SHA-256 and 1,104 data records before writing.\n'
        'record_number is 1-based data-record order, excluding the header, not a physical line number.\n\n'
        'DOI: trim, remove a leading DOI URL/doi: label, trim, casefold; no guessed repairs.\n'
        'Titles: NFKC, casefold, whitespace collapse. Punctuation and word order remain significant.\n'
        'DOI matches use both source_nodes.csv DOI and reconciled_groups.csv JSON DOIs.\n'
        'Title candidates use source node titles and reconciled representative titles; group IDs\n'
        'also include group membership of matching nodes. Only exact normalized equality is used.\n'
        'title_candidates.csv contains each title evidence entity for records without a DOI match.\n'
        'title_candidate_records.csv contains those incoming records once each.\n'
        'title_ambiguities.csv flags title candidates spanning multiple prior groups.\n'
        'new_doi_records.csv includes DOI-absent-from-catalog records even when their titles match.\n'
        'no_doi_records.csv preserves every incoming record with an empty DOI.\n'
        'all_records.csv contains the complete partition and provenance. DOI match status takes precedence.\n'
        'No automatic identity merge or eligibility classification is performed.\n'
        'See summary.json for verified counts, input hashes and limitations.\n', encoding='utf-8')
    artifacts = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != 'SHA256SUMS')
    (OUT / 'SHA256SUMS').write_text(''.join(f'{sha256(path)}  {path.name}\n' for path in artifacts), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key not in ('limits', 'input_files', 'incoming_normalized_title_duplicate_groups')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
