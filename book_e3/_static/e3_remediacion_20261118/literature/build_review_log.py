"""Combine actual API receipts with explicitly scoped web observations; no training."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
api = json.loads((ROOT / 'native_api_attempts.json').read_text())
alternate_api = json.loads((ROOT / 'alternative_api_attempts.json').read_text())
alternate_probes = json.loads((ROOT / 'alternative_access_probes.json').read_text())
api = api + alternate_api
supp = json.loads((ROOT / 'supplemental_observations.json').read_text())
rows = list(api)
rows.extend(alternate_probes['observations'])
for row in supp['native_ui_observations']:
    # Exact per-request times were not provided by the web tool. These calls
    # occurred between two separately read UTC clock receipts.
    row['observed_at_utc'] = None
    row['observation_date_utc'] = '2026-11-18'
    row['observed_time_window_utc'] = ['2026-11-18T14:16:17Z', '2026-11-18T14:20:24Z']
    row.pop('time_precision', None)
    rows.append(row)
for row in supp['additional_native_access_attempts']:
    row['observation_date_utc'] = '2026-11-18'
    row['query'] = 'CNN LSTM attention cryptocurrency volatility'
    rows.append(row)
for index, query in enumerate(supp['general_web_discovery']['queries_executed'], 1):
    rows.append(dict(query_id=f'D{index}', query=query,
        language='es' if any(word in query for word in ['atención','volatilidad','pronóstico']) else 'en',
        provider='general_web_search', mode='search_engine_discovery_not_native_database',
        observation_date_utc='2026-11-18', status='discovery_results_observed_in_shared_batch',
        url=None,
        portable_query_url='https://www.google.com/search?q=' + __import__('urllib.parse', fromlist=['quote']).quote(query),
        url_note='Tool search provider and its request URL are unspecified. Portable link does not claim Google was queried.',
        native_result_count=None, returned_records_count=None,
        count_interpretation='Search engine outputs were merged within batches; no per-query total is claimed.'))
review = dict(schema_version=1, review_date='2026-11-18',
    base_source_commit=supp['base_source_commit'],
    reviewer_kind=supp['reviewer_kind'],
    scope=supp['scope'],
    api_attempts=len(api), native_api_successes=sum(r['status']=='response_parsed_not_yet_screened' for r in api),
    native_ui_queries_observed=len(supp['native_ui_observations']),
    native_bases_with_observed_results=['arXiv'],
    other_native_bases_attempted=['OpenAlex','Crossref','SemanticScholar','DBLP','OpenReview','IEEE Xplore'],
    native_api_requests_by_base={base:sum(row['provider']==base for row in api)
        for base in ['OpenAlex','Crossref','SemanticScholar','arXiv','DBLP','OpenReview']},
    alternative_native_web_access_probes=len(alternate_probes['observations']),
    excluded_biomedical_scope_probes=len(alternate_probes['excluded_scope_probes']),
    native_three_base_coverage_completed=False,
    inaccessible_results_count_is_unknown=True,
    total_unique_relevant_works_in_matrix=11,
    complete_texts_available_for_focal_review=8,
    human_external_verification='pending',
    search_exhaustiveness='partial; no inference of absence from not-found or blocked access',
    quote_review='Paraphrases; no source passage quoted at length. Shared mathematical formulas and article titles retained for identification.',
    rows=rows)
(ROOT / 'search_log.json').write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n')
columns=sorted(set().union(*(row.keys() for row in rows)))
with (ROOT / 'search_log.csv').open('w', newline='') as stream:
    writer=csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows({k:json.dumps(v, ensure_ascii=False) if isinstance(v,(list,dict)) else v
                     for k,v in row.items()} for row in rows)
# Normalize the supplemental observations too, avoiding guessed request times.
(ROOT / 'supplemental_observations.json').write_text(json.dumps(supp, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({key:review[key] for key in ['api_attempts','native_ui_queries_observed',
    'native_bases_with_observed_results','native_three_base_coverage_completed',
    'complete_texts_available_for_focal_review']}, ensure_ascii=False))
