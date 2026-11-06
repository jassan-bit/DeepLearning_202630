"""Import user-supplied WoS exports into minimal, auditable bibliographic tables.

This script performs no database query, full-text review or model selection.
Original exports and full abstracts/references remain outside public outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import itertools
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
QUERIES = {
    "Q1": '''TS=((bitcoin OR ethereum OR cryptocurrenc* OR cryptoasset*)
AND (volatility OR "realized volatility" OR "realised volatility")
AND ("deep learning" OR "neural network*" OR transformer*
OR LSTM OR GRU OR CNN OR TCN))''',
    "Q2": '''TS=(("time series" OR forecasting OR "temporal prediction")
AND (TSMixer OR TiDE OR TimeMixer OR PatchTST OR iTransformer
OR Crossformer OR TimesNet OR MICN OR Mamba
OR "structured state space"))''',
}
SCIENTIFIC_PRIVATE_FIELDS = {
    "UT", "DI", "TI", "AB", "AU", "AF", "SO", "PY", "EA", "PD", "DT", "TC",
    "WE", "WC", "DE", "ID", "NR", "CR", "DA", "OA", "VL", "IS", "AR", "BP", "EP",
}
LIST_FIELDS = {"AU", "AF", "CR"}
CRYPTO = re.compile(r"\b(?:bitcoin|ethereum|ether|cryptocurrenc\w*|cryptoasset\w*|crypto-assets?|digital assets?|litecoin|dogecoin|ripple)\b", re.I)
VOLATILITY = re.compile(r"\b(?:volatilit\w*|realized variance|realised variance|conditional variance)\b", re.I)
FINANCE = re.compile(r"\b(?:financ\w*|stock\w*|equity|equities|futures|exchange rate\w*|oil price\w*|trading|asset\w*|market\w*)\b", re.I)
URL = re.compile(r"https?://[^\s<>\]\[\"{}]+", re.I)
MODEL_PATTERNS = {
    "TSMixer": re.compile(r"\bTSMixer\b", re.I),
    "TiDE": re.compile(r"\bTiDE\b|(?i:\btime[- ]series dense encoder\b)"),
    "TimeMixer": re.compile(r"\bTimeMixer(?:\+\+)?\b", re.I),
    "PatchTST": re.compile(r"\bPatchTST\b", re.I),
    "iTransformer": re.compile(r"\biTransformer\b", re.I),
    "Crossformer": re.compile(r"\bCrossformer\b", re.I),
    "TimesNet": re.compile(r"\bTimesNet\b", re.I),
    "MICN": re.compile(r"\bMICN\b|multi[- ]scale isometric convolution", re.I),
    "Mamba": re.compile(r"\b(?:CryptoMamba(?:-SSM)?|Mamba(?:-SSM)?)\b", re.I),
    "S4": re.compile(r"\bS4\b|structured state[- ]space", re.I),
    "TFT": re.compile(r"temporal fusion transformer|\bTFT\b", re.I),
    "N-BEATS": re.compile(r"\bN[- ]?BEATS(?:x)?\b", re.I),
    "N-HiTS": re.compile(r"\bN[- ]?HiTS\b", re.I),
    "BiLSTM": re.compile(r"\bbi[- ]?LSTM\b|bidirectional long short[- ]term memory", re.I),
    "BiGRU": re.compile(r"\bbi[- ]?GRU\b|bidirectional gated recurrent", re.I),
    "CNN-LSTM": re.compile(r"\bCNN[- +/]+LSTM\b|convolutional.*?long short[- ]term memory", re.I),
    "CNN-GRU": re.compile(r"\bCNN[- +/]+GRU\b|convolutional.*?gated recurrent", re.I),
    "Transformer": re.compile(r"\btransformers?\b", re.I),
    "Attention-LSTM": re.compile(r"attention[- ]?(?:based|enhanced)?[- ]?LSTM|\bATT[- ]LSTM\b", re.I),
    "ConvLSTM": re.compile(r"\bConvLSTM\b|convolutional LSTM", re.I),
    "HARNet": re.compile(r"\bHARNet\b", re.I),
    "DeepVol": re.compile(r"\bDeepVol\b", re.I),
    "MLP": re.compile(r"\bMLP\b|multilayer perceptron|multi-layer perceptron", re.I),
    "CNN": re.compile(r"\bCNN\b|convolutional neural network", re.I),
    "GRU": re.compile(r"\bGRU\b|gated recurrent unit", re.I),
    "LSTM": re.compile(r"\bLSTM\b|long short[- ]term memory", re.I),
    "TCN": re.compile(r"\bTCN\b|temporal convolutional network", re.I),
}
TOP10_CANDIDATES = tuple(list(MODEL_PATTERNS)[:10])


def normalize_doi(value: str) -> str:
    result = value.strip().lower()
    result = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", result)
    return result.strip(' <>\"').rstrip(".,;")


def normalize_title(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(re.findall(r"\w+", text))


def parse_wos_text(text: str) -> tuple[list[dict], dict]:
    """Preserve field continuations and reject incomplete/nested records."""
    records, fields, tag = [], None, None
    headers, ef = {}, False
    for line_no, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if line == "EF":
            if fields is not None:
                raise ValueError(f"Unterminated record before EF at line {line_no}")
            ef = True
            continue
        if ef:
            raise ValueError(f"Unexpected data after EF at line {line_no}")
        if line == "ER":
            if fields is None:
                raise ValueError(f"Orphan ER at line {line_no}")
            record = {k: list(v) if k in LIST_FIELDS else " ".join(v)
                      for k, v in fields.items()}
            records.append(record)
            fields, tag = None, None
            continue
        if line.startswith("PT "):
            if fields is not None:
                raise ValueError(f"Nested PT before ER at line {line_no}")
            fields, tag = {"PT": [line[3:]]}, "PT"
            continue
        if fields is None:
            if re.match(r"^(FN|VR) ", line):
                headers[line[:2]] = line[3:]
                continue
            raise ValueError(f"Unexpected non-record content at line {line_no}")
        if line.startswith("   "):
            if tag is None:
                raise ValueError(f"Continuation without field at line {line_no}")
            fields[tag].append(line[3:])
            continue
        match = re.match(r"^([A-Z0-9]{2}) (.*)$", line)
        if not match:
            raise ValueError(f"Invalid field syntax at line {line_no}")
        tag, value = match.groups()
        fields.setdefault(tag, []).append(value)
    if fields is not None:
        raise ValueError("Unterminated record at EOF")
    if not ef:
        raise ValueError("Export has no final EF marker")
    if not records:
        raise ValueError("Export contains no complete records")
    if "Web of Science" not in headers.get("FN", ""):
        raise ValueError("Expected Web of Science export header")
    return records, {"headers": headers, "complete_records": len(records), "final_EF": ef}


def integer(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def publication_metadata(record):
    py = integer(record.get("PY"))
    years = [int(x) for x in re.findall(r"\b(?:19|20)\d{2}\b", record.get("EA", ""))]
    ea = min(years) if years else None
    known = [year for year in (py, ea) if year is not None]
    first = min(known) if known else None
    if py is not None and 2019 <= py <= 2026:
        status = "inside_nominal_2019_2026_window"
    elif ea is not None and 2019 <= ea <= 2026:
        status = "inside_via_early_access_year_needs_date_verification"
    elif first is not None and first < 2019:
        status = "outside_window_foundational_review_needed"
    else:
        status = "publication_window_not_confirmed"
    tc = integer(record.get("TC"))
    denominator = 2026 - first + 1 if first is not None and first <= 2026 else None
    return {"year": py, "early_access_year": ea, "earliest_known_publication_year": first,
            "publication_window_status": status,
            "citations_per_calendar_year": tc / denominator if tc is not None and denominator else None}


def model_mentions(record):
    text = record.get("TI", "") + " " + record.get("AB", "")
    result = [name for name, pattern in MODEL_PATTERNS.items() if pattern.search(text)]
    if "S4" in result and not re.search(r"state[- ]space|structured state", text, re.I):
        result.remove("S4")
    return result


def application_scope(record):
    text = record.get("TI", "") + " " + record.get("AB", "")
    if CRYPTO.search(text) and VOLATILITY.search(text):
        return "crypto_and_volatility_terms_screening_seed"
    if CRYPTO.search(text):
        return "crypto_other_target_or_unspecified_volatility"
    if FINANCE.search(text) and VOLATILITY.search(text):
        return "other_financial_volatility_terms_screening_seed"
    return "general_forecasting_or_other_domain"


def code_urls(record):
    text = record.get("TI", "") + " " + record.get("AB", "")
    return sorted({u.rstrip(".,;)") for u in URL.findall(text)
                   if any(host in u.lower() for host in ("github.com/", "gitlab.com/", "bitbucket.org/", "codeocean.com/"))})


def deduplicate(records):
    parents = list(range(len(records)))
    edges = []
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b, rule, identity):
        aroot, broot = find(a), find(b)
        if aroot != broot:
            parents[broot] = aroot
            edges.append({"record_a": a, "record_b": b, "rule": rule, "identity": identity})
    for field, rule in (("UT", "exact_UT"), ("DI", "normalized_DOI")):
        seen = {}
        for i, r in enumerate(records):
            key = normalize_doi(r.get(field, "")) if field == "DI" else r.get(field, "").strip()
            if key:
                if key in seen:
                    union(seen[key], i, rule, key)
                else:
                    seen[key] = i
    titles = defaultdict(list)
    for i, r in enumerate(records):
        title = normalize_title(r.get("TI", ""))
        if len(title) >= 20:
            titles[(title, r.get("PY", ""))].append(i)
    title_conflicts = []
    for (title, year), indices in titles.items():
        for a, b in itertools.combinations(indices, 2):
            da, db = normalize_doi(records[a].get("DI", "")), normalize_doi(records[b].get("DI", ""))
            if da and db and da != db:
                title_conflicts.append({"record_a": a, "record_b": b, "reason": "same_title_year_different_DOI_not_merged"})
            elif not da or not db:
                union(a, b, "normalized_title_and_year_fallback", title + " | " + year)
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return list(groups.values()), edges, title_conflicts


def clean_cell(value):
    """Keep values as text when a spreadsheet might evaluate a leading formula."""
    if value is None:
        return ""
    text = str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_cell(row.get(k)) for k in fields})


def build_outputs(inputs):
    occurrences, sources = [], []
    for label, path in inputs:
        payload = path.read_bytes()
        parsed, audit = parse_wos_text(payload.decode("utf-8-sig"))
        start = len(occurrences)
        for index, r in enumerate(parsed, 1):
            r.update(source_file=label, record_index=index)
            occurrences.append(r)
        hypotheses = "Q1" if label == "savedrecs.txt" else ("Q2" if label in ("savedrecs (1).txt", "savedrecs (2).txt") else None)
        sources.append({"source_file": label, "bytes": len(payload), "sha256": sha256(payload).hexdigest(),
                        "record_count": len(parsed), "PT_ER_integrity": "passed", "final_EF": audit["final_EF"],
                        "WoS_header": audit["headers"], "export_DA_values": sorted({r.get("DA", "") for r in parsed}),
                        "export_DA_has_time_or_timezone": False,
                        "query_assignment_hypothesis": hypotheses,
                        "query_assignment_confirmed_by_export": False,
                        "record_range_in_native_result_list": None,
                        "native_full_result_count": None,
                        "occurrence_offset_zero_based": start,
                        "PY_counts": dict(sorted(Counter(r.get("PY", "missing") for r in parsed).items())),
                        "missing_abstracts": sum(not r.get("AB", "").strip() for r in parsed),
                        "missing_DOIs": sum(not r.get("DI", "").strip() for r in parsed),
                        "missing_TC": sum(integer(r.get("TC")) is None for r in parsed)})
    groups, edges, title_conflicts = deduplicate(occurrences)
    public, private, evidence = [], [], []
    for indices in groups:
        indices.sort()
        canonical = occurrences[indices[0]]
        aliases = [occurrences[i] for i in indices]
        # Multiple occurrences are retained in provenance; canonical is first input occurrence.
        r = dict(canonical)
        mentions = sorted(set().union(*(model_mentions(x) for x in aliases)))
        scopes = [application_scope(x) for x in aliases]
        scope_order = {"crypto_and_volatility_terms_screening_seed": 0,
                       "crypto_other_target_or_unspecified_volatility": 1,
                       "other_financial_volatility_terms_screening_seed": 2,
                       "general_forecasting_or_other_domain": 3}
        scope = min(scopes, key=scope_order.get)
        identity = r.get("UT") or ("DOI:" + normalize_doi(r.get("DI", ""))) or f"record_{indices[0]+1:04d}"
        pm = publication_metadata(r)
        provenance = [{"source_file": x["source_file"], "record_index": x["record_index"], "UT": x.get("UT", "")} for x in aliases]
        conflicts = {tag: sorted({x.get(tag, "") for x in aliases}) for tag in ("DI", "TI", "PY", "TC")
                     if len({x.get(tag, "") for x in aliases}) > 1}
        row = {"record_id": identity, "UT": r.get("UT", ""), "UT_all": json.dumps(sorted({x.get("UT", "") for x in aliases}), ensure_ascii=False),
               "DOI": normalize_doi(r.get("DI", "")), "title": r.get("TI", ""),
               "volume": r.get("VL", ""), "issue": r.get("IS", ""), "first_page": r.get("BP", ""), "last_page": r.get("EP", ""),
               "year": pm["year"], "early_access": r.get("EA", ""), "early_access_year": pm["early_access_year"],
               "earliest_known_publication_year": pm["earliest_known_publication_year"],
               "publication_window_status": pm["publication_window_status"],
               "document_type": r.get("DT", ""), "source": r.get("SO", ""),
               "WC": r.get("WC", ""), "WE": r.get("WE", ""), "TC": integer(r.get("TC")),
               "citations_per_calendar_year": pm["citations_per_calendar_year"],
               "source_file": r["source_file"], "record_index": r["record_index"],
               "occurrence_count": len(aliases), "source_occurrences": json.dumps(provenance, ensure_ascii=False),
               "abstract_available": any(bool(x.get("AB", "").strip()) for x in aliases),
               "code_URLs_in_title_or_abstract": json.dumps(sorted(set().union(*(code_urls(x) for x in aliases))), ensure_ascii=False),
               "model_mentions_title_abstract": "; ".join(mentions), "application_scope_screening_seed": scope,
               "ambiguous_tide_term": any(bool(re.search(r"\btide\b", x.get("TI", "")+" "+x.get("AB", ""), re.I)) for x in aliases) and "TiDE" not in mentions,
               "metadata_conflicts": json.dumps(conflicts, ensure_ascii=False),
               "full_text_eligibility": "not_assessed_by_this_import", "journal_quartile": "not_verified"}
        public.append(row)
        private_row = {k: v for k, v in r.items() if k in SCIENTIFIC_PRIVATE_FIELDS}
        private_row.update(record_id=identity, source_file=r["source_file"], record_index=r["record_index"], source_occurrences=provenance,
                           model_mentions=mentions, application_scope_screening_seed=scope, metadata_conflicts=conflicts,
                           occurrence_indices_zero_based=indices)
        private.append(private_row)
        for model in mentions:
            evidence.append({"model": model, "record_id": identity, "UT": row["UT"], "DOI": row["DOI"],
                             "title": row["title"], "year": row["year"], "early_access": row["early_access"],
                             "TC": row["TC"], "citations_per_calendar_year": row["citations_per_calendar_year"],
                             "application_scope_screening_seed": scope, "source": row["source"],
                             "code_URLs_in_title_or_abstract": row["code_URLs_in_title_or_abstract"],
                             "model_use_role": "not_verified; title/abstract mention is not proof of an independent comparator",
                             "evidence_locator": "WoS TI and/or AB; CR excluded from matching"})
    overlaps = []
    for a, b in itertools.combinations(sources, 2):
        sa = {r.get("UT") for r in occurrences if r["source_file"] == a["source_file"] and r.get("UT")}
        sb = {r.get("UT") for r in occurrences if r["source_file"] == b["source_file"] and r.get("UT")}
        overlaps.append({"source_a": a["source_file"], "source_b": b["source_file"], "shared_exact_UT": len(sa & sb)})
    summary = {
        "schema_version": 1, "administrative_receipt_date_colombia": "2026-11-05", "administrative_timezone": "America/Bogota",
        "source": "User-supplied native Web of Science plain-text exports; no database query executed by this script",
        "source_files": len(sources), "exported_record_occurrences": len(occurrences),
        "unique_exact_UT": len({r["UT"] for r in occurrences if r.get("UT")}),
        "unique_records_after_identifier_deduplication": len(groups), "duplicate_occurrences_removed": len(occurrences)-len(groups),
        "deduplication_merge_rules": dict(Counter(e["rule"] for e in edges)), "source_overlaps_by_UT": overlaps,
        "metadata_conflict_groups": sum(bool(json.loads(r["metadata_conflicts"])) for r in public),
        "title_year_different_DOI_pairs_not_merged": len(title_conflicts),
        "PY_counts_unique_records": dict(sorted(Counter(str(r["year"]) for r in public).items())),
        "publication_window_status_counts": dict(Counter(r["publication_window_status"] for r in public)),
        "application_scope_counts_preliminary": dict(Counter(r["application_scope_screening_seed"] for r in public)),
        "model_mentions_unique_records": dict(sorted(Counter(e["model"] for e in evidence).items())),
        "top10_candidate_mentions": {name: sum(e["model"] == name for e in evidence) for name in TOP10_CANDIDATES},
        "missing_abstracts_unique": sum(not r["abstract_available"] for r in public),
        "missing_DOIs_unique": sum(not r["DOI"] for r in public), "missing_TC_unique": sum(r["TC"] is None for r in public),
        "records_with_code_url_in_title_or_abstract": sum(json.loads(r["code_URLs_in_title_or_abstract"]) != [] for r in public),
        "citation_normalization": "TC / (2026 - min(PY, EA year when available) + 1); undefined if earliest year >2026 or TC missing; descriptive snapshot, not selection score",
        "export_date_semantics": "DA records 2026-11-06 without time or timezone; not equated automatically to local query execution time",
        "query_assignment_confirmed": False, "Q2_global_result_count": None,
        "screenshot_evidence": {"displayed_results": 302, "basis": "User screenshot before exports", "query_assignment": "not encoded in image/export; Q1 association remains a hypothesis"},
        "raw_exports_redistributed": False, "full_abstracts_or_cited_reference_lists_redistributed": False,
        "screening_status": "Identifier import and preliminary title/abstract keyword seeds only; full-text eligibility and model roles require review",
        "deduplication_scope": "Exact identifiers and title/year fallback yield bibliographic record groups; journal/preprint versions with different identifiers can still represent the same scientific study and require reviewer adjudication",
        "PRISMA_scope": "1302 is imported occurrences, not the unknown combined native-search universe; included studies are not assigned by import",
        "human_independent_reviewers": "not established; no Cohen kappa calculated",
    }
    return public, private, evidence, sources, summary, edges, title_conflicts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--reference-output", type=Path, default=ROOT/"references/literature_e2/wos_20261105")
    parser.add_argument("--book-output", type=Path, default=ROOT/"book_e2/_static/literature_wos")
    parser.add_argument("--private-json", type=Path)
    args = parser.parse_args(argv)
    inputs = []
    for value in args.input:
        if "=" not in value:
            parser.error("--input requires LABEL=PATH")
        label, path = value.split("=", 1)
        if not label or "/" in label or "\\" in label:
            parser.error("Input label must be a public basename, without directories")
        inputs.append((label, Path(path)))
    if len({x[0] for x in inputs}) != len(inputs):
        parser.error("Input labels must be unique")
    public, private, evidence, sources, summary, edges, title_conflicts = build_outputs(inputs)
    edge_rows = []
    flat = []
    for label, path in inputs:
        parsed, _ = parse_wos_text(path.read_text(encoding="utf-8-sig"))
        flat.extend(dict(r, source_file=label, record_index=i) for i, r in enumerate(parsed, 1))
    for e in edges:
        a, b = flat[e["record_a"]], flat[e["record_b"]]
        edge_rows.append({"rule": e["rule"], "identity": e["identity"], "source_a": a["source_file"], "record_index_a": a["record_index"],
                          "UT_a": a.get("UT", ""), "source_b": b["source_file"], "record_index_b": b["record_index"], "UT_b": b.get("UT", "")})
    manifest = {"schema_version": 1, "generator_script": "scripts/import_e2_wos.py",
                "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "sources": sources,
                "declared_queries": QUERIES, "two_queries_not_a_comma_joined_query": True,
                "query_to_file_mapping_confirmed": False, "raw_exports_redistributed": False,
                "public_derivatives": "Minimal bibliographic metadata, identifiers, counts and source positions; full abstracts, CR, email/address fields excluded"}
    summary["PRISMA_scope"] = f"{summary['exported_record_occurrences']} is imported occurrences, not the unknown combined native-search universe; included studies are not assigned by import"
    for out in (args.reference_output, args.book_output):
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out/"records.csv", public, list(public[0]))
        write_csv(out/"model_evidence.csv", evidence, list(evidence[0]) if evidence else ["model", "record_id", "title"])
        write_csv(out/"deduplication.csv", edge_rows, ["rule", "identity", "source_a", "record_index_a", "UT_a", "source_b", "record_index_b", "UT_b"])
        for name, data in (("summary.json", summary), ("title_identifier_conflicts.json", title_conflicts)):
            (out/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        manifest["public_output_files"] = {
            name: {"bytes": (out/name).stat().st_size,
                   "sha256": sha256((out/name).read_bytes()).hexdigest()}
            for name in ("records.csv", "model_evidence.csv", "deduplication.csv", "summary.json", "title_identifier_conflicts.json")
        }
        (out/"source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    if args.private_json:
        protected = args.private_json.resolve()
        if protected.is_relative_to(ROOT.resolve()):
            raise ValueError("Full private scientific fields must be written outside the public repository")
        protected.parent.mkdir(parents=True, exist_ok=True)
        protected.write_text(json.dumps(private, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: summary[k] for k in ("exported_record_occurrences", "unique_exact_UT", "unique_records_after_identifier_deduplication", "duplicate_occurrences_removed", "deduplication_merge_rules", "top10_candidate_mentions")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
