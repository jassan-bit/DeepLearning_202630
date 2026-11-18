"""CPU-only native bibliographic collector. AI assistance declared, 2026-11-18.

Successful HTTP requests are not novelty decisions. Blocked counts remain null.
No models, market data, credentials, or third-party PDFs are collected.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

QUERIES = [
    ("Q1", "en", "cryptocurrency volatility"),
    ("Q2", "en", "CNN LSTM attention cryptocurrency"),
    ("Q3", "es", "volatilidad bitcoin atención"),
    ("Q4", "es", "volatilidad piso positivo"),
    ("Q5", "en", "neural network output constraints"),
    ("Q6", "en", "rolling variance conditional forecast"),
]
PROVIDERS = ("Crossref", "DBLP", "arXiv", "OpenAlex", "SemanticScholar")
MAX_RESPONSE = 4 * 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def api_url(provider, query, limit=10):
    endpoints = {
        "Crossref": ("https://api.crossref.org/works", {"query": query, "rows": limit}),
        "DBLP": ("https://dblp.org/search/publ/api", {"q": query, "format": "json", "h": limit}),
        "arXiv": ("https://export.arxiv.org/api/query", {"search_query": "all:" + query, "max_results": limit}),
        "OpenAlex": ("https://api.openalex.org/works", {"search": query, "per-page": limit}),
        "SemanticScholar": ("https://api.semanticscholar.org/graph/v1/paper/search", {
            "query": query, "limit": limit, "fields": "title,year,url,externalIds,openAccessPdf"}),
    }
    endpoint, params = endpoints[provider]
    return endpoint + "?" + urllib.parse.urlencode(params)


def normalized(provider, raw):
    """Return native count and metadata; never infer a missing count as zero."""
    records = []
    if provider == "arXiv":
        root = ET.fromstring(raw)
        count = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults"))
        atom = "{http://www.w3.org/2005/Atom}"
        for item in root.findall(atom + "entry"):
            records.append({"title": " ".join((item.findtext(atom + "title") or "").split()),
                            "url": item.findtext(atom + "id"),
                            "year": (item.findtext(atom + "published") or "")[:4],
                            "doi": item.findtext("{http://arxiv.org/schemas/atom}doi")})
    else:
        data = json.loads(raw)
        if provider == "Crossref":
            data = data["message"]
            count = int(data["total-results"])
            for item in data["items"]:
                date = item.get("published", {}).get("date-parts", [[]])
                title = item.get("title", [])
                records.append({"title": title[0] if title else "", "doi": item.get("DOI"),
                                "url": item.get("URL"), "year": date[0][0] if date[0] else None})
        elif provider == "DBLP":
            hits = data["result"]["hits"]
            count = int(hits["@total"])
            items = hits.get("hit", [])
            if isinstance(items, dict):
                items = [items]
            for hit in items:
                item = hit["info"]
                records.append({"title": item["title"], "url": item.get("url"),
                                "doi": item.get("doi"), "year": item.get("year")})
        elif provider == "OpenAlex":
            count = int(data["meta"]["count"])
            for item in data["results"]:
                records.append({"title": item.get("title") or "", "url": item.get("id"),
                                "doi": item.get("doi"), "year": item.get("publication_year")})
        else:
            count = int(data["total"])
            for item in data["data"]:
                records.append({"title": item.get("title") or "", "url": item.get("url"),
                                "doi": item.get("externalIds", {}).get("DOI"), "year": item.get("year")})
    if count < 0 or count < len(records) or any(not r["title"] for r in records):
        raise ValueError("inconsistent_native_response")
    return count, records


def attempt(provider, query, output, limit=10, opener=urllib.request.urlopen):
    qid, language, text = query
    url = api_url(provider, text, limit)
    row = {"provider": provider, "query_id": qid, "language": language, "query": text,
           "query_scope": "provider_native_search; operators/ranking differ by provider",
           "mode": "native_api", "url": url, "observed_at_utc": utc_now(),
           "native_result_count": None, "returned_records_count": None,
           "records": [], "titles_reviewed": [], "abstracts_reviewed": [],
           "full_texts_reviewed": [], "relevant_work_ids": [], "screening_complete": False}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "E3-R1-LiteratureReview/1.0"})
        with opener(req, timeout=15) as response:
            row["http_status"] = response.status
            raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise ValueError("response_exceeds_4MiB")
        # Raw native metadata, not articles/PDFs, makes the count auditable.
        suffix = ".xml" if provider == "arXiv" else ".json"
        relative = "responses/" + provider + "_" + qid + suffix
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        row.update(response_file=relative, response_bytes=len(raw),
                   response_sha256=hashlib.sha256(raw).hexdigest())
        row["native_result_count"], row["records"] = normalized(provider, raw)
        row["returned_records_count"] = len(row["records"])
        row["status"] = "response_parsed_not_screened"
    except Exception as error:
        if isinstance(error, urllib.error.HTTPError):
            row["http_status"] = error.code
        row.update(status="blocked_or_unavailable", error_type=type(error).__name__,
                   error_message=str(error), native_result_count=None,
                   returned_records_count=None, records=[],
                   count_interpretation="unknown; no validated bibliographic result")
    return row


def run(output, source_commit, *, limit=10):
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("exact_source_commit_required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def provider_run(provider):
        rows = []
        for index, query in enumerate(QUERIES):
            if index:
                time.sleep(3.3)  # At most one in-flight request/provider; arXiv pacing.
            rows.append(attempt(provider, query, output, limit))
        return rows

    with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as pool:
        attempts = [row for batch in pool.map(provider_run, PROVIDERS) for row in batch]
    effective = [p for p in PROVIDERS if all(r["status"] == "response_parsed_not_screened"
                 for r in attempts if r["provider"] == p)]
    receipt = {"kind": "e3_r1_native_bibliographic_collection", "source_commit": source_commit,
               "created_at_utc": utc_now(), "queries": [dict(zip(("id", "language", "query"), q)) for q in QUERIES],
               "providers": list(PROVIDERS), "attempts": attempts,
               "effective_bases_all_six_queries": effective,
               "three_base_response_coverage": len(effective) >= 3,
               "screening_complete": False, "human_external_review_completed": False,
               "R1_accredited": False, "new_real_data_fits": 0,
               "new_market_predictions": 0, "new_statistical_contrasts": 0,
               "scope": "First page only; native counts/ranking differ; no exhaustiveness or novelty certification."}
    (output / "search_receipt.json").write_bytes(canonical(receipt))
    with (output / "screening.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["provider", "query_id", "title", "year", "doi", "url", "title_decision", "reason", "full_text_locator"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in attempts:
            for record in row["records"]:
                writer.writerow(dict(record, provider=row["provider"], query_id=row["query_id"],
                                     title_decision="not_screened", reason="", full_text_locator=""))
    print(json.dumps({"native_attempts": len(attempts), "effective_bases": effective,
                      "three_base_response_coverage": len(effective) >= 3,
                      "screening_complete": False, "R1_accredited": False,
                      "new_real_data_fits": 0, "output": str(output)}, ensure_ascii=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--limit", type=int, default=10, choices=range(1, 21))
    args = parser.parse_args()
    run(args.output, args.source_commit, limit=args.limit)
