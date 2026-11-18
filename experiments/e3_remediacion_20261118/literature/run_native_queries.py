"""Record authorized bibliographic API attempts; never interpret blocked access as zero.

Prepared with declared AI assistance on 2026-11-18. No models are trained.
The companion search log also records separately observed native arXiv pages
and general web discovery; this script does not conflate those mechanisms.
"""
import concurrent.futures
import csv
import datetime
import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

QUERIES = [
    ("Q1", "en", "cryptocurrency volatility"),
    ("Q2", "en", "CNN LSTM attention cryptocurrency"),
    ("Q3", "es", "volatilidad bitcoin atención"),
    ("Q4", "es", "volatilidad piso positivo"),
    ("Q5", "en", "neural network output constraints"),
    ("Q6", "en", "rolling variance conditional forecast"),
]

def api_url(base, query):
    params = {
        "OpenAlex": ("https://api.openalex.org/works", {"search": query, "per-page": 5}),
        "Crossref": ("https://api.crossref.org/works", {"query": query, "rows": 5}),
        "SemanticScholar": ("https://api.semanticscholar.org/graph/v1/paper/search", {
            "query": query, "limit": 5, "fields": "title,year,url,openAccessPdf"}),
        "arXiv": ("https://export.arxiv.org/api/query", {
            "search_query": "all:" + query, "max_results": 5}),
    }
    endpoint, values = params[base]
    return endpoint + "?" + urllib.parse.urlencode(values)

def attempt(item):
    base, (qid, language, query) = item
    row = dict(query_id=qid, language=language, query=query, provider=base,
               mode="native_api", url=api_url(base, query),
               observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               native_result_count=None, returned_records_count=None,
               titles_reviewed=[], abstracts_reviewed=[], full_texts_reviewed=[],
               relevant_work_ids=[], count_interpretation="unknown unless response parsed")
    try:
        req = urllib.request.Request(row["url"], headers={"User-Agent": "E3-literature-review/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8")
            row["http_status"] = response.status
        if base == "OpenAlex":
            data = json.loads(body)
            row["native_result_count"] = data["meta"]["count"]
            records = data["results"]
        elif base == "Crossref":
            data = json.loads(body)["message"]
            row["native_result_count"] = data["total-results"]
            records = data["items"]
        elif base == "SemanticScholar":
            data = json.loads(body)
            row["native_result_count"] = data["total"]
            records = data["data"]
        else:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(body)
            row["native_result_count"] = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults"))
            records = root.findall("{http://www.w3.org/2005/Atom}entry")
        row["returned_records_count"] = len(records)
        row["status"] = "response_parsed_not_yet_screened"
    except Exception as exc:
        row["status"] = "blocked_or_unavailable"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
        row["count_interpretation"] = "unknown; request failed before bibliographic results"
    return row

if __name__ == "__main__":
    target = Path(__file__).with_name("native_api_attempts.json")
    tasks = [(base, query) for query in QUERIES for base in
             ["OpenAlex", "Crossref", "SemanticScholar", "arXiv"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(attempt, tasks))
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"attempts": len(rows), "blocked_or_unavailable": sum(
        row["status"] == "blocked_or_unavailable" for row in rows),
        "output": str(target)}, ensure_ascii=False))
