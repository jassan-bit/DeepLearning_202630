"""Try six bilingual queries in two additional native public scientific APIs.

Uses ordinary public requests with no proxy, access-control or routing override.
AI-assisted preparation declared; failures keep counts unknown. No training.
"""
import concurrent.futures
import datetime
import json
from pathlib import Path
import urllib.parse
import urllib.request
from run_native_queries import QUERIES

def attempt(item):
    base, (qid, language, query) = item
    if base == 'DBLP':
        url='https://dblp.org/search/publ/api?' + urllib.parse.urlencode(
            dict(q=query, format='json', h=20))
    else:
        url='https://api2.openreview.net/notes/search?' + urllib.parse.urlencode(
            dict(term=query, limit=20))
    row=dict(query_id=qid, query=query, language=language, provider=base,
        mode='native_api', url=url,
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        native_result_count=None, returned_records_count=None,
        titles_reviewed=[], abstracts_reviewed=[], full_texts_reviewed=[], relevant_work_ids=[])
    try:
        req=urllib.request.Request(url, headers={'User-Agent':'E3-literature-review/1.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            data=json.loads(response.read().decode('utf-8'))
            row['http_status']=response.status
        if base=='DBLP':
            hits=data['result']['hits']
            row['native_result_count']=int(hits['@total'])
            row['returned_records_count']=int(hits['@sent'])
        else:
            notes=data.get('notes')
            if not isinstance(notes,list):
                raise ValueError('Response does not contain public note results')
            row['native_result_count']=data.get('count')
            row['returned_records_count']=len(notes)
        row['status']='response_parsed_not_yet_screened'
        row['count_interpretation']='Native API response; not yet screened'
    except Exception as exc:
        row['status']='blocked_or_unavailable'
        row['error_type']=type(exc).__name__
        row['error_message']=str(exc)
        row['count_interpretation']='unknown; request failed before bibliographic results'
    return row

if __name__=='__main__':
    tasks=[(base, query) for query in QUERIES for base in ['DBLP','OpenReview']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(attempt,tasks))
    out=Path(__file__).with_name('alternative_api_attempts.json')
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(attempts=len(rows),blocked_or_unavailable=sum(
        r['status']=='blocked_or_unavailable' for r in rows),output=str(out)),ensure_ascii=False))
