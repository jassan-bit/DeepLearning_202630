"""Record bounded public-index requests without treating denied access as zero hits.

Run from this folder or pass --output. This makes read-only network requests;
it does not modify a scientific review or declare novelty.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, parse_qs, urlsplit
from urllib.request import Request, urlopen

QUERIES = [
    ('Q01', 'robust attention bounded influence Bitcoin cryptocurrency volatility',
     '("robust attention" OR "bounded influence" OR "atención robusta") AND (Bitcoin OR cryptocurrency OR criptomonedas) AND (volatility OR volatilidad)'),
    ('Q02', 'multiscale attention volatility outlier corruption',
     '(multiscale OR "multi-scale" OR multiescala) AND (attention OR atención) AND (volatility OR volatilidad) AND (outlier OR corruption OR atípicos)'),
    ('Q03', 'Lipschitz attention normalization time series forecasting',
     '("Lipschitz attention" OR "Lipschitz normalization") AND ("time series" OR "series temporales" OR forecasting)'),
    ('Q04', 'CNN LSTM attention Bitcoin cryptocurrency volatility',
     '(CNN AND LSTM AND (attention OR atención)) AND (Bitcoin OR cryptocurrency) AND (volatility OR volatilidad)'),
    ('Q05', 'radial saturation bounded values value clipping attention graph transformer',
     '("radial saturation" OR "bounded values" OR "value clipping" OR "saturación radial") AND (attention OR "graph attention" OR transformer)'),
    ('Q06', 'causal disagreement multiscale discrepancy temporal attention forecasting',
     '("causal disagreement" OR "multiscale discrepancy" OR "discordancia multiescala") AND (attention OR atención OR forecasting)'),
]


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def targets(query):
    yield 'OpenAlex', 'free-text search; Boolean translation not asserted', 'https://api.openalex.org/works?' + urlencode({'search': query, 'per-page': 10})
    yield 'Crossref', 'bibliographic free-text search; not Boolean execution', 'https://api.crossref.org/works?' + urlencode({'query.bibliographic': query, 'rows': 10})
    # The explicit AND translation is saved, independently of planned Boolean strings.
    arxiv_query = ' AND '.join('all:' + token for token in query.split())
    yield 'arXiv API', 'explicit all:token AND sequence', 'https://export.arxiv.org/api/query?' + urlencode({'search_query': arxiv_query, 'start': 0, 'max_results': 10})


def probe(item):
    qid, query, database, syntax, url = item
    parameters = parse_qs(urlsplit(url).query)
    actual_query = parameters['search_query' if database == 'arXiv API' else 'search' if database == 'OpenAlex' else 'query.bibliographic'][0]
    result = {'query_id': qid, 'database': database, 'query_sent': actual_query,
              'query_syntax': syntax, 'url': url, 'requested_at_utc': utcnow(),
              'http_status': None, 'proxy_status': None, 'database_executed': False,
              'total_results': None, 'retrieved_records': None,
              'status': 'unresolved', 'error': None}
    try:
        with urlopen(Request(url, headers={'User-Agent': 'E3-Literature-Audit/1.0'}), timeout=20) as response:
            body = response.read(2_000_000)
            result['http_status'] = response.status
            result['status'] = 'response_received_not_screened'
            if database in ('OpenAlex', 'Crossref'):
                document = json.loads(body)
                if database == 'OpenAlex':
                    result['total_results'] = document['meta']['count']
                    result['retrieved_records'] = len(document['results'])
                else:
                    result['total_results'] = document['message']['total-results']
                    result['retrieved_records'] = len(document['message']['items'])
                result['database_executed'] = True
            else:
                import xml.etree.ElementTree as ET
                element = ET.fromstring(body)
                result['total_results'] = int(element.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text)
                result['retrieved_records'] = len(element.findall('{http://www.w3.org/2005/Atom}entry'))
                result['database_executed'] = True
    except (HTTPError, URLError, OSError, ValueError, KeyError) as error:
        result['error'] = str(error)
        if 'Tunnel connection failed: 403' in str(error):
            result['proxy_status'] = 403
            result['status'] = 'destination_denied_by_environment_proxy'
        else:
            result['status'] = 'request_failed_not_a_search_result'
    result['finished_at_utc'] = utcnow()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(qid, query, database, syntax, url)
            for qid, query, _ in QUERIES for database, syntax, url in targets(query)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        attempts = list(executor.map(probe, jobs))
    (args.output/'search_plan.json').write_text(json.dumps([
        {'id': qid, 'api_free_text': query, 'planned_boolean_bilingual': boolean,
         'purpose': 'mechanism and problem discovery; source-language adaptation required'}
        for qid, query, boolean in QUERIES], ensure_ascii=False, indent=2)+'\n')
    (args.output/'search_attempts.json').write_text(json.dumps(attempts, ensure_ascii=False, indent=2)+'\n')
    with (args.output/'search_attempts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(attempts[0]))
        writer.writeheader(); writer.writerows(attempts)
    print(json.dumps({'requests': len(attempts), 'executed_by_database': sum(x['database_executed'] for x in attempts),
                      'denied_by_proxy': sum(x['proxy_status'] == 403 for x in attempts),
                      'novelty_established': False}))


if __name__ == '__main__':
    main()
