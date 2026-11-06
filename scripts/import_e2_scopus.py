"""Import a supplied Scopus CSV and link it to the existing WoS record groups.

No live database query, eligibility decision, peer-review assertion or sum of
citation counts is performed. Full abstracts and source exports stay private.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

from import_e2_wos import (MODEL_PATTERNS, application_scope, clean_cell, code_urls,
                           integer, normalize_doi, normalize_title, publication_metadata)

ROOT = Path(__file__).resolve().parents[1]
MODELS = dict(MODEL_PATTERNS)
MODELS.update({
    "S-Mamba": re.compile(r"\bS[- ]Mamba\b", re.I),
    "Informer": re.compile(r"\bInformer\b", re.I),
    "Autoformer": re.compile(r"\bAutoformer\b", re.I),
})
REQUIRED_HEADERS = {"Title", "Year", "Source title", "Cited by", "DOI", "Abstract", "Document Type", "Publication Stage", "Source", "EID"}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n"


def mentions(entry):
    text = entry["title"]+" "+entry["abstract"]
    result = [name for name, pattern in MODELS.items() if pattern.search(text)]
    if "S4" in result and not re.search(r"state[- ]space|structured state", text, re.I):
        result.remove("S4")
    return result


def read_scopus(path, public_label=None):
    csv.field_size_limit(5_000_000)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED_HEADERS <= set(reader.fieldnames):
            raise ValueError("Scopus export lacks required columns")
        headers = reader.fieldnames
        entries = []
        for index, row in enumerate(reader, 1):
            if None in row:
                raise ValueError(f"Malformed CSV record {index}")
            if not row["Title"].strip() or not row["EID"].strip():
                raise ValueError(f"Title/EID absent at record {index}")
            if row["Source"].strip().casefold() != "scopus":
                raise ValueError(f"Unexpected provider in Source at record {index}")
            tc = integer(row["Cited by"])
            if row["Cited by"].strip() and tc is None:
                raise ValueError(f"Non-numeric citation count at record {index}")
            entries.append({"database": "Scopus", "source_file": public_label or Path(path).name,
                            "record_index": index, "title": row["Title"].strip(),
                            "abstract": row["Abstract"].strip(), "source": row["Source title"].strip(),
                            "year": integer(row["Year"]), "doi": normalize_doi(row["DOI"]),
                            "ut": "", "eid": row["EID"].strip(), "citations": tc,
                            "stage": row["Publication Stage"].strip(), "early_access": "",
                            "document_type": row["Document Type"].strip(), "raw": row})
    if not entries:
        raise ValueError("Scopus export contains no records")
    return entries, headers


class Groups:
    def __init__(self, entries):
        self.entries = entries
        self.parent = list(range(len(entries)))
        self.dois = [{e["doi"]} if e["doi"] else set() for e in entries]
        self.edges = []
        self.conflicts = []

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def merge(self, a, b, rule, identity, strict_doi=False):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if strict_doi and self.dois[ra] and self.dois[rb] and self.dois[ra].isdisjoint(self.dois[rb]):
            self.conflicts.append({"record_a": a, "record_b": b, "reason": "conflicting_DOIs_not_merged", "identity": identity})
            return False
        self.parent[rb] = ra
        self.dois[ra] |= self.dois[rb]
        self.edges.append({"record_a": a, "record_b": b, "rule": rule, "identity": identity})
        return True

    def collect(self):
        grouped = defaultdict(list)
        for index in range(len(self.entries)):
            grouped[self.find(index)].append(index)
        return list(grouped.values())


def within_scopus(entries):
    groups = Groups(entries)
    for field, rule in (("eid", "same_Scopus_EID"), ("doi", "same_normalized_DOI_inside_Scopus")):
        seen = {}
        for index, entry in enumerate(entries):
            key = entry[field]
            if key:
                if key in seen:
                    groups.merge(seen[key], index, rule, key)
                else:
                    seen[key] = index
    return groups


def wos_entries(records):
    result = []
    for r in records:
        result.append({"database": "WoS", "source_file": r.get("source_file", ""),
                       "record_index": r.get("record_index"), "title": r.get("TI", ""),
                       "abstract": r.get("AB", ""), "source": r.get("SO", ""),
                       "year": integer(r.get("PY")), "doi": normalize_doi(r.get("DI", "")),
                       "ut": r.get("UT", ""), "eid": "", "citations": integer(r.get("TC")),
                       "stage": "", "early_access": r.get("EA", ""),
                       "document_type": r.get("DT", ""), "raw": r})
    return result


def combine(wos, scopus):
    entries = wos+scopus
    groups = Groups(entries)
    seen_doi = {}
    for i, entry in enumerate(entries):
        if entry["doi"]:
            if entry["doi"] in seen_doi:
                groups.merge(seen_doi[entry["doi"]], i, "cross_source_normalized_DOI", entry["doi"])
            else:
                seen_doi[entry["doi"]] = i
    titles = defaultdict(list)
    for i, entry in enumerate(entries):
        key = (normalize_title(entry["title"]), entry["year"], normalize_title(entry["source"]))
        if len(key[0]) >= 20 and key[1] is not None and key[2]:
            titles[key].append(i)
    for key, candidates in titles.items():
        ww = [i for i in candidates if entries[i]["database"] == "WoS"]
        ss = [i for i in candidates if entries[i]["database"] == "Scopus"]
        for a in ww:
            for b in ss:
                if groups.find(a) == groups.find(b):
                    continue
                if entries[a]["doi"] and entries[b]["doi"]:
                    if entries[a]["doi"] != entries[b]["doi"]:
                        groups.conflicts.append({"record_a": a, "record_b": b, "reason": "same_title_year_source_conflicting_DOI_not_merged", "identity": " | ".join(map(str,key))})
                    continue
                groups.merge(a, b, "cross_source_title_year_source_with_missing_DOI", " | ".join(map(str,key)), strict_doi=True)
    return groups


def canonical_group(entries, indices):
    first = entries[min(indices)]
    source_groups = [entries[i] for i in sorted(indices)]
    members = [alias for e in source_groups for alias in e.get("_source_aliases", [e])]
    doi_values = sorted({e["doi"] for e in members if e["doi"]})
    ut_values = sorted({e["ut"] for e in members if e["ut"]} | {
        origin.get("UT", "") for e in members if e["database"] == "WoS"
        for origin in e["raw"].get("source_occurrences", []) if origin.get("UT")
    })
    eid_values = sorted({e["eid"] for e in members if e["eid"]})
    rid = "doi:"+doi_values[0] if doi_values else (ut_values[0] if ut_values else "Scopus:"+eid_values[0])
    wtc = sorted({e["citations"] for e in members if e["database"] == "WoS" and e["citations"] is not None} | {
        integer(value) for e in members if e["database"] == "WoS"
        for value in e["raw"].get("metadata_conflicts", {}).get("TC", []) if integer(value) is not None
    })
    stc = sorted({e["citations"] for e in members if e["database"] == "Scopus" and e["citations"] is not None})
    wos_canonical = next((e for e in members if e["database"] == "WoS"), None)
    scopus_canonical = next((e for e in members if e["database"] == "Scopus"), None)
    ea = wos_canonical["early_access"] if wos_canonical else ""
    year = first["year"]
    pm = publication_metadata({"PY": str(year) if year else "", "EA": ea})
    model_names = sorted(set().union(*(mentions(e) for e in members)))
    scope = application_scope({"TI": first["title"], "AB": " ".join(e["abstract"] for e in members)})
    urls = sorted(set().union(*(code_urls({"TI": e["title"], "AB": e["abstract"]}) for e in members)))
    sources = sorted({e["database"] for e in members})
    wtc_value = wos_canonical["citations"] if wos_canonical else None
    stc_value = scopus_canonical["citations"] if scopus_canonical else None
    age = 2026-pm["earliest_known_publication_year"]+1 if pm["earliest_known_publication_year"] and pm["earliest_known_publication_year"] <= 2026 else None
    origins = []
    for e in members:
        if e["database"] == "WoS" and e["raw"].get("source_occurrences"):
            origins.extend({"database": "WoS", "source_file": origin["source_file"],
                            "record_index": origin["record_index"], "UT": origin.get("UT", ""), "EID": ""}
                           for origin in e["raw"]["source_occurrences"])
        else:
            origins.append({"database": e["database"], "source_file": e["source_file"],
                            "record_index": e["record_index"], "UT": e["ut"], "EID": e["eid"]})
    public = {"record_id": rid, "title": first["title"], "DOI": doi_values[0] if doi_values else "",
              "year": year, "year_values": json.dumps(sorted({e["year"] for e in members if e["year"] is not None})),
              "early_access_from_WoS": ea, "earliest_known_publication_year": pm["earliest_known_publication_year"],
              "publication_window_status": pm["publication_window_status"], "source_title": first["source"],
              "document_type": first["document_type"], "Scopus_publication_stage": scopus_canonical["stage"] if scopus_canonical else "",
              "source_databases": "; ".join(sources), "UTs": json.dumps(ut_values), "EIDs": json.dumps(eid_values),
              "WoS_TC": wtc_value, "Scopus_Cited_by": stc_value, "WoS_TC_values": json.dumps(wtc), "Scopus_Cited_by_values": json.dumps(stc),
              "WoS_TC_per_calendar_year": wtc_value/age if wtc_value is not None and age else None,
              "Scopus_Cited_by_per_calendar_year": stc_value/age if stc_value is not None and age else None,
              "citation_counts_summed": False, "source_record_groups": len(source_groups),
              "native_source_occurrences": len(origins),
              "source_occurrences": json.dumps(origins, ensure_ascii=False), "abstract_available": any(e["abstract"] for e in members),
              "code_URLs_title_abstract": json.dumps(urls, ensure_ascii=False), "model_mentions_title_abstract": "; ".join(model_names),
              "application_scope_screening_seed": scope, "model_use_role": "Not verified; a title/abstract mention is not proof of an independent comparator",
              "full_text_eligibility": "not_assessed_by_import", "peer_review_status": "not_independently_verified",
              "journal_quartile": "not_verified"}
    private = {"record_id": rid, "title": first["title"], "abstract": max((e["abstract"] for e in members),key=len),
               "DOI": public["DOI"], "year": year, "source": first["source"], "UTs": ut_values, "EIDs": eid_values,
               "source_databases": sources, "WoS_TC": wtc_value, "Scopus_Cited_by": stc_value,
               "WoS_TC_values": wtc, "Scopus_Cited_by_values": stc,
               "model_mentions": model_names, "application_scope_screening_seed": scope,
               "TI": first["title"], "AB": max((e["abstract"] for e in members),key=len), "DI": public["DOI"],
               "PY": str(year) if year else "", "SO": first["source"], "TC": str(wtc_value) if wtc_value is not None else "",
               "UT": ut_values[0] if ut_values else "", "EID": eid_values[0] if eid_values else "",
               "source_records": [{"database": e["database"], "source_file": e["source_file"], "record_index": e["record_index"],
                                   "raw_scientific_record": e["raw"]} for e in members]}
    return public, private


def original_matches(models, scopus_entries, wos):
    result, details = [], []
    for model in models:
        title = normalize_title(model["paper_title"])
        doi = normalize_doi(model.get("doi") or "")
        ss = [e for e in scopus_entries if (doi and e["doi"] == doi) or normalize_title(e["title"]) == title]
        ww = [e for e in wos if (doi and e["doi"] == doi) or normalize_title(e["title"]) == title or any(e["doi"] and e["doi"] == x["doi"] for x in ss)]
        # DOI duplication has already been collapsed; one row per distinct Scopus source group.
        if ss:
            for e in ss:
                identity = "exact normalized primary title" if normalize_title(e["title"]) == title else "exact normalized primary DOI"
                metadata = publication_metadata({"PY": str(e["year"]), "EA": ww[0]["early_access"] if ww else ""})
                earliest = metadata["earliest_known_publication_year"]
                age = 2026-earliest+1 if earliest is not None and earliest <= 2026 else None
                result.append({"model": model["name"], "expected_primary_title": model["paper_title"], "registry_DOI": model.get("doi") or "",
                               "found_in_Scopus_export": True, "identity_rule": identity, "matching_Scopus_groups": len(ss),
                               "Title": e["title"], "EID": e["eid"], "DOI_in_Scopus": e["doi"], "Cited_by_Scopus": e["citations"],
                               "Year_in_Scopus": e["year"], "Source_title_in_Scopus": e["source"], "Scopus_record_index": e["record_index"],
                               "WoS_UT": ww[0]["ut"] if ww else "", "WoS_TC": ww[0]["citations"] if ww else None,
                               "early_access_from_matching_WoS": ww[0]["early_access"] if ww else "",
                               "citation_age_year": earliest, "Cited_by_Scopus_per_calendar_year": e["citations"]/age if e["citations"] is not None and age else None,
                               "citation_counts_summed": False,
                               "identity_note": "Primary title/DOI identity, not a proxy applied paper; Google All-MLP TSMixer kept distinct from KDD Lightweight MLP-Mixer"})
        else:
            result.append({"model": model["name"], "expected_primary_title": model["paper_title"], "registry_DOI": model.get("doi") or "",
                           "found_in_Scopus_export": False, "identity_rule": "No exact primary title/DOI match in received 7500 records",
                           "matching_Scopus_groups": 0, "Title": "", "EID": "", "DOI_in_Scopus": "", "Cited_by_Scopus": None,
                           "Year_in_Scopus": None, "Source_title_in_Scopus": "", "Scopus_record_index": None,
                           "WoS_UT": ww[0]["ut"] if ww else "", "WoS_TC": ww[0]["citations"] if ww else None,
                           "early_access_from_matching_WoS": ww[0]["early_access"] if ww else "", "citation_age_year": None,
                           "Cited_by_Scopus_per_calendar_year": None, "citation_counts_summed": False,
                           "identity_note": "Absence in partial export is not zero citations or absence from Scopus"})
        details.append({"model": model["name"], "Scopus_exact_matches": [{k:e[k] for k in ("title","doi","eid","year","source","citations","record_index")} for e in ss],
                        "WoS_exact_matches": [{k:e[k] for k in ("title","doi","ut","year","source","citations","early_access")} for e in ww]})
    return result, details


def write_csv(path, rows, headers=None):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=headers or list(rows[0]),lineterminator="\n")
        writer.writeheader()
        for row in rows:writer.writerow({k:clean_cell(row.get(k)) for k in writer.fieldnames})


def public_evidence(rows):
    evidence=[]
    for row in rows:
        for model in row["model_mentions_title_abstract"].split("; "):
            if model:
                evidence.append({k:row[k] for k in ("record_id","title","DOI","year","source_databases","UTs","EIDs","WoS_TC","Scopus_Cited_by","application_scope_screening_seed","model_use_role")}|{"model":model})
    return evidence


def coverage(count, total):
    return {"screenshot_displayed_result_count":total, "received_export_occurrences":count,
            "difference_from_screenshot_count": total-count if total is not None and total>=count else None,
            "received_fraction_of_screenshot_count":count/total if total and total>=count else None,
            "export_is_complete_native_search_universe":False,
            "record_ranges_and_sort_order_confirmed":False,
            "note":"Difference is unreceived or unexported coverage, not a PRISMA exclusion count"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--wos-private",required=True,type=Path)
    parser.add_argument("--primary-registry",type=Path,default=ROOT/"references/literature_e2/top10_20261105/primary_source_registry.json")
    parser.add_argument("--screen-total",type=int,default=7874)
    parser.add_argument("--private-combined",required=True,type=Path)
    parser.add_argument("--reference-output",type=Path,default=ROOT/"references/literature_e2/scopus_20261105")
    parser.add_argument("--book-output",type=Path,default=ROOT/"book_e2/_static/literature_scopus")
    args=parser.parse_args(argv)
    if args.private_combined.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Full abstracts/source records must remain outside the public repository")
    input_hash=digest(args.input)
    se,headers=read_scopus(args.input)
    sg=within_scopus(se)
    source_groups=sg.collect()
    scopus=[]
    for indices in source_groups:
        representative=dict(se[min(indices)])
        representative["_source_aliases"]=[se[i] for i in indices]
        scopus.append(representative)
    # Alias EIDs, citation values and source positions are retained after DOI deduplication.
    ws=wos_entries(json.loads(args.wos_private.read_text()))
    cg=combine(ws,scopus)
    combined=[canonical_group(cg.entries,indices) for indices in cg.collect()]
    public=[x[0] for x in combined]; private=[x[1] for x in combined]
    source_public=[canonical_group(se,indices)[0] for indices in source_groups]
    registry=json.loads(args.primary_registry.read_text())
    originals,original_details=original_matches(registry["models"],scopus,ws)
    cross_counts=Counter(row["source_databases"] for row in public)
    summary={"schema_version":1,"administrative_date_colombia":"2026-11-05","administrative_timezone":"America/Bogota",
             "source":"User-supplied Scopus CSV; no native query executed by this importer",
             "coverage":coverage(len(se),args.screen_total),"Scopus_export_records":len(se),
             "Scopus_unique_EIDs":len({e['eid'] for e in se}),"Scopus_records_with_DOI":sum(bool(e['doi']) for e in se),
             "Scopus_unique_nonempty_DOIs":len({e['doi'] for e in se if e['doi']}),
             "Scopus_groups_after_EID_DOI_dedup":len(scopus),"Scopus_duplicate_occurrences_removed":len(se)-len(scopus),
             "Scopus_deduplication_rules":dict(Counter(e['rule'] for e in sg.edges)),
             "WoS_input_bibliographic_groups":len(ws),"combined_pre_link_groups":len(ws)+len(scopus),
             "received_native_record_occurrences":len(se)+sum(len(e["raw"].get("source_occurrences", [])) or 1 for e in ws),
             "duplicate_occurrences_removed_all_stages":len(se)+sum(len(e["raw"].get("source_occurrences", [])) or 1 for e in ws)-len(public),
             "combined_bibliographic_groups":len(public),"cross_database_record_groups_linked":sum(e['source_databases']=='Scopus; WoS' for e in public),
             "combined_source_membership_counts":dict(cross_counts),"cross_database_link_rules":dict(Counter(e['rule'] for e in cg.edges)),
             "conflicting_identifier_candidates_not_merged":len(cg.conflicts),
             "Scopus_abstract_available":sum(bool(e['abstract']) for e in se),"Scopus_missing_citation_counts":sum(e['citations'] is None for e in se),
             "Scopus_publication_year_counts":dict(sorted(Counter(str(e['year']) for e in se).items())),
             "Scopus_year_after_2026_needs_date_verification":sum(e['year'] is not None and e['year']>2026 for e in se),
             "Scopus_no_explicit_early_access_dates_in_schema":True,
             "Scopus_2027_records_are_automatically_eligible":False,
             "primary_models_found_in_received_Scopus":sorted({r['model'] for r in originals if r['found_in_Scopus_export']}),
             "primary_models_not_matched_in_received_Scopus":sorted({r['model'] for r in originals if not r['found_in_Scopus_export']}),
             "citation_counts_kept_separate":True,"WoS_TC_plus_Scopus_Cited_by_calculated":False,
             "combined_model_mentions_title_abstract":dict(sorted(Counter(model for row in public for model in row['model_mentions_title_abstract'].split('; ') if model).items())),
             "abstract_keywords_are_screening_seeds_not_model_role_or_target_verification":True,
             "peer_review_and_quartile_not_inferred_from_index_membership":True,
             "raw_exports_and_full_abstracts_redistributed":False,
             "eligibility_assessed":False,
             "universe_scope":"Received partial native exports only; grouped records are not included full-text studies"}
    def edge_rows(group):
        result=[]
        for edge in group.edges:
            a,b=group.entries[edge['record_a']],group.entries[edge['record_b']]
            result.append({"rule":edge['rule'],"identity":edge['identity'],"database_a":a['database'],"UT_a":a['ut'],"EID_a":a['eid'],
                           "database_b":b['database'],"UT_b":b['ut'],"EID_b":b['eid']})
        return result
    manifest={"schema_version":1,"generator_script":"scripts/import_e2_scopus.py","generator_sha256":digest(__file__),
              "source_file":{"filename":args.input.name,"bytes":args.input.stat().st_size,"sha256":input_hash,"CSV_headers":headers,
                             "record_count":len(se),"filename_timestamp_is_not_assigned_a_timezone":True,
                             "query_execution_datetime":None,"native_query_full_string_embedded":False,
                             "intended_query_assignment":"Q2 candidate time-series families; correspondence plausible, not fully encoded in export",
                             "query_field_evidence_from_user":"Article title, Abstract, Keywords", "manual_native_export":True},
              "WoS_private_input_sha256":digest(args.wos_private),"primary_registry_sha256":digest(args.primary_registry),
              "citation_count_semantics":"WoS TC and Scopus Cited by remain independent snapshots; never summed",
              "raw_export_and_full_abstract_redistribution":False,"coverage":summary['coverage']}
    for out in (args.reference_output,args.book_output):
        out.mkdir(parents=True,exist_ok=True)
        write_csv(out/'records.csv',source_public)
        write_csv(out/'combined_records.csv',public)
        write_csv(out/'model_evidence.csv',public_evidence(source_public))
        write_csv(out/'combined_model_evidence.csv',public_evidence(public))
        write_csv(out/'original_model_records.csv',originals)
        write_csv(out/'deduplication.csv',edge_rows(sg),["rule","identity","database_a","UT_a","EID_a","database_b","UT_b","EID_b"])
        write_csv(out/'cross_database_links.csv',edge_rows(cg),["rule","identity","database_a","UT_a","EID_a","database_b","UT_b","EID_b"])
        for name,data in [('summary.json',summary),('original_model_records.json',{'models':originals,'identity_details':original_details}),('identifier_conflicts.json',cg.conflicts)]:
            (out/name).write_text(dumps(data))
        manifest['public_output_files']={p.name:{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='source_manifest.json'}
        (out/'source_manifest.json').write_text(dumps(manifest))
    args.private_combined.parent.mkdir(parents=True,exist_ok=True)
    args.private_combined.write_text(dumps(private))
    assert digest(args.input)==input_hash
    print(dumps({k:summary[k] for k in ['coverage','Scopus_groups_after_EID_DOI_dedup','WoS_input_bibliographic_groups','combined_bibliographic_groups','combined_source_membership_counts','cross_database_link_rules','primary_models_found_in_received_Scopus','primary_models_not_matched_in_received_Scopus']}))


if __name__=='__main__':
    main()
