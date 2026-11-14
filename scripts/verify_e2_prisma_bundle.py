#!/usr/bin/env python3
"""Independent, read-only consistency checks on a generated public bundle."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

p=Path(sys.argv[1])
manifest=json.loads((p/'manifest.json').read_text())
for a in manifest['artifacts']:
    body=(p/a['name']).read_bytes()
    assert hashlib.sha256(body).hexdigest()==a['sha256'],a['name']
    assert len(body)==a['bytes'],a['name']
rows=list(csv.DictReader((p/'received_groups_9559.csv').open()))
assert len(rows)==9559==len({r['group_id'] for r in rows})
flow=json.loads((p/'received_flow.json').read_text())
assert flow['received_occurrences']-flow['within_round_duplicate_occurrences']-flow['cross_round_link_reductions']==len(rows)
assert dict(Counter(r['final_disposition'] for r in rows))==flow['received_group_final_dispositions']
reports=[json.loads(s) for s in (p/'full_reports_compact.jsonl').read_text().splitlines()]
quality=list(csv.DictReader((p/'quality_12_dimensions.csv').open()))
assert len(quality)==12*len(reports)
assert all(n==12 for n in Counter(r['group_id'] for r in quality).values())
applied=list(csv.DictReader((p/'quality_applied_studies.csv').open()))
expected_applied=flow['unique_studies_included_applied_narrative']
assert len(applied)==12*expected_applied
assert len({r['study_id'] for r in applied})==expected_applied
assert all(n==12 for n in Counter(r['study_id'] for r in applied).values())
for r in applied:
    assert r['no_value_merging_across_report_versions']=='True'
    assert r['source_pointer']
for r in quality:
    if r['value_source_field']=='summary':
        assert 'quality_extraction_matrix_17.json#/studies/' in r['source_pointer']
        assert r['value_brief']
assert sum(r['value_source_field']=='summary' for r in quality)==204
assert all(r['own_reason_plus_values_word_count']==len(r['reason_brief'].split())+sum(len(v['value_brief'].split()) for v in r['quality'].values())<=70 for r in reports)
assert all(not r['primary_report_assessed'] for r in reports if r['final_disposition'] in {'report_not_retrieved','excluded_after_metadata_or_abstract','container_linked_existing_report'})
assert all(r['primary_report_assessed'] for r in reports if r['final_disposition'] in {'included_applied_narrative','excluded_after_primary_sections','context_criteria_unidentified','eligibility_unresolved_after_primary_sections'})
studies=list(csv.DictReader((p/'study_identities.csv').open()))
assert sum(r['included_applied_narrative']=='True' for r in studies)==flow['unique_studies_included_applied_narrative']
index={r['group_id']:r for r in rows}
for r in rows:
    if r['final_reason_ref']:
        assert r['final_reason_ref']=='screening_reason_brief' and not r['final_reason_brief'] and r['screening_reason_brief']
for pair in [('lit-80ef43205192de5e3be0','lit-9645876f486a0e1485b1'),('lit-e8ded7c56180aa56f833','lit-26714704cb0a4329c015')]:
    assert index[pair[0]]['study_id']==index[pair[1]]['study_id']
for g in ['lit-6ac30f85d00b7750ad80','lit-587c6ea8918442a511f0']:
    assert index[g]['final_disposition']=='container_linked_existing_report'
    assert not index[g]['study_id']
synth_id=index['lit-26714704cb0a4329c015']['study_id']
synth=[r for r in applied if r['study_id']==synth_id]
assert len(synth)==12
assert all(r['preferred_report_review_group_id']=='lit-26714704cb0a4329c015' and r['source_version']=='journal_final_2025' for r in synth)
assert 'unique_empirical_or_candidate_study_identities_excluding_containers' not in flow
assert flow['bibliographic_identity_components_excluding_containers']==9555
for g in ['lit-201a17eb40edd3fb9243','lit-c70eca4a44d19f60564c']:
    assert index[g]['final_disposition']=='eligibility_unresolved_after_primary_sections'
    assert index[g]['retrieval_status']=='primary_report_retrieved'
    assert index[g]['operational_pending']=='False'
assert index['lit-f818106993466a7a1c67']['adjudication_applied']=='neural_sde_adjudication'
sentiment=next(r for r in reports if r['group_id']=='lit-4f45507b8ea83fe2704b')
assert sentiment['final_disposition']=='context_primary_source'
assert sentiment['primary_report_assessed'] and sentiment['target_identified'] and sentiment['temporal_order_verified']
assert sentiment['adjudication_applied']=='sentiment_igarch_adjudication'
assert sentiment['empirical_dispersion_forecast_evaluation']=='not_verified_in_examined_sections'
assert sentiment['criterion_boundary_code']=='empirical_dispersion_forecast_evaluation_no_additional_oos_rule'
assert index[sentiment['group_id']]['operational_pending']=='False'
assert sentiment['study_id'] not in {r['study_id'] for r in applied}
hybrid=next(r for r in reports if r['group_id']=='lit-9a7dda468b8cd448360d')
assert hybrid['final_disposition']=='included_applied_narrative' and hybrid['empirical_dispersion_forecast_evaluation'] is True
assert hybrid['study_id'] in {r['study_id'] for r in applied}
history=json.loads((p/'report_adjudication_history.json').read_text())
event=next(r for r in history if r.get('adjudication_source')=='sentiment_igarch_adjudication' and r['group_id']==sentiment['group_id'])
assert event['superseded_decision']=='included_narrative' and event['final_decision']=='context_modeling_forecast_unverified'
original52=next(r for r in manifest['sources'] if r['source_id']=='fulltext_reviews_0')['sha256']
assert original52==event['original_52_reviews_sha256']=='6a7ebf6dc8731104a11d60ecce92a9ea83608238266a07da1aed8a4373aee89b'
ats=list(csv.DictReader((p/'retrieval_attempts.csv').open()))
for r in ats:
    u=urlsplit(r['url'])
    assert not u.query and not u.fragment and not u.username
    assert '@' not in unquote(u.path)
attempt_groups={r['group_id'] for r in ats}
assert all(r['group_id'] in attempt_groups for r in reports if r['final_disposition']=='report_not_retrieved')
for f in p.iterdir():
    if not f.is_file(): continue
    body=f.read_text()
    assert not re.search(r'"abstracts?"\s*:',body),f.name
    assert not re.search(r'/workspace/|/tmp/|file://|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',body),f.name
audit=json.loads((p/'audit.json').read_text())
assert audit['stored_abstract_coverage']=={'base':8396,'additional_arxiv':530,'additional_focal':13,'disjoint_total':8939}
assert audit['screening_groups_with_real_abstract_read']==2197
assert audit['prior_focal_decisions_reused']==30
assert audit['prior_primary_section_assessments_reused']==25
assert audit['prior_title_or_abstract_exclusions_reused']==5
assert not audit['operational_pending'] if manifest['strict_ready'] else True
print(json.dumps({'verified':True,'received_groups':len(rows),'compact_dispositions':len(reports),'quality_rows':len(quality),'applied_quality_rows':len(applied),'unique_included_studies':flow['unique_studies_included_applied_narrative'],'operational_pending':len(audit['operational_pending'])}))
