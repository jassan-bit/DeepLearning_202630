#!/usr/bin/env python3
"""Retrieve a coherent citation snapshot; missing is never converted to zero.

Inputs: current10_extraction.json. Outputs: nine study-level records plus an
access receipt. FCN and ResNet share a source study. No final E2 scores are read.
The endpoint is DOI-specific; arXiv DOI coverage is not assumed. A failed lookup
is not replaced with a journal/preprint count without an explicit identity audit.
"""
import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import pathlib
import urllib.error
import urllib.parse
import urllib.request


def retrieve(study):
    doi = study['study_id']
    url = 'https://api.openalex.org/works/' + urllib.parse.quote('https://doi.org/' + doi, safe='/:')
    out = dict(study, provider='OpenAlex', request_url=url, citation_count=None,
               openalex_id=None, retrieval_status='not_retrieved')
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'E2-literature-audit/1.0 (public metadata; no credentials)'})
        with urllib.request.urlopen(req, timeout=12) as response:
            body = response.read()
            out['http_status'] = response.status
        data = json.loads(body)
        out.update(retrieval_status='retrieved_identity_requires_review',
                   response_sha256=hashlib.sha256(body).hexdigest(),
                   response_title=data.get('title'), response_doi=data.get('doi'),
                   openalex_id=data.get('id'), citation_count=data.get('cited_by_count'),
                   provider_updated_date=data.get('updated_date'))
    except Exception as exc:
        out.update(retrieval_status='failed', error_type=type(exc).__name__, error=str(exc))
    out['observed_at_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    args = p.parse_args()
    body = args.source.read_bytes()
    systems = json.loads(body)
    studies = {}
    for row in systems:
        study = studies.setdefault(row['study_id'], {
            'study_id':row['study_id'], 'title':row['title'],
            'original_year':row['original_year'], 'models':[]})
        study['models'].append(row['model'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(retrieve, studies.values()))
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        'version':'e2-citation-coverage-audit-v1',
        'observed_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'source_sha256':hashlib.sha256(body).hexdigest(),
        'systems':len(systems), 'distinct_studies':len(studies),
        'provider':'OpenAlex',
        'retrieved':sum(r['retrieval_status'].startswith('retrieved') for r in results),
        'identity_review_completed':False,
        'historical_scopus_snapshot_replaced':False,
        'test_A2_read_or_used_for_literature_selection':False,
        'missing_citations_imputed':False,
        'records':results,
    }
    (args.out/'openalex_access_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fields=['study_id','title','models','original_year','provider','citation_count','openalex_id','retrieval_status','request_url','observed_at_utc','error_type','error']
    with (args.out/'citation_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in results:
            w.writerow(dict(row,models=' | '.join(row['models'])))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
