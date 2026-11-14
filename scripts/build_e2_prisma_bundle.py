#!/usr/bin/env python3
"""Deterministic, public-metadata-only reconciliation of the received E2 corpus.

No network, model calls, article downloads, or repository writes. Private inputs
are snapshotted once per file; only explicit allowlisted fields are exported.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

VERSION = "e2-received-prisma-final-v1"
GUIDE_SHA = "879e2ce52709657a06bc6c009f0cd553423f2eaa0bbdf16dc7f5844dfec9fcc1"
DIMS = ("target", "data", "horizon", "split", "metrics", "models", "repetitions", "hpo", "tests", "code", "leakage", "fairness")
CONTAINERS = {"lit-6ac30f85d00b7750ad80", "lit-587c6ea8918442a511f0"}
PRIOR_SCREENING_ONLY = {
    "10.1016/j.energy.2025.138706": "title_only",
    "10.1016/j.epsr.2025.112196": "title_only",
    "10.1016/j.physa.2024.130185": "title_only",
    "10.3390/en18102434": "title_only",
    "10.1108/cfri-01-2024-0032": "primary_structured_abstract",
}
CRYPTO_ALIAS = ("lit-80ef43205192de5e3be0", "lit-9645876f486a0e1485b1")
HEX = re.compile(r"^[0-9a-f]{64}$")
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
LOCAL = re.compile(r"(?:/(?:workspace|tmp|home|Users|mnt)/|[A-Za-z]:[\\/]|file://)")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def word_count(value):
    return len(str(value or "").split())


def safe_text(value):
    s = " ".join(str(value or "").split())
    if LOCAL.search(s) or EMAIL.search(s):
        return "[omitted: private locator]"
    return s


def compact(value, limit):
    """Bound verbatim *own review notes*, never abstracts; explicitly mark cuts."""
    s = safe_text(value)
    words = s.split()
    if len(words) <= limit:
        return s
    prefix = " ".join(words[:limit])
    # Prefer a complete sentence/clause, without silently expanding its scope.
    cuts = [m.end() for m in re.finditer(r"[.;](?:\s|$)", prefix)]
    if cuts and cuts[-1] >= len(prefix) // 2:
        prefix = prefix[:cuts[-1]].strip()
    return prefix.rstrip(" ,;:.") + "…"


def compact_quality_value(value, limit):
    """Never turn a qualified source statement into an unqualified assertion.

    If a qualifier/negation would disappear in a short excerpt, publish a
    neutral abbreviation marker and the exact source pointer instead.
    """
    s=safe_text(value)
    if word_count(s)<=limit:
        return s
    if limit<2:
        return ""
    if s.startswith("Valor y límites conservados en la ficha pública previa, campo "):
        return "Ficha previa…"
    qualification=re.compile(r"\b(?:no|sin|not|never|unknown|unverified|unidentified|missing|unavailable|unresolved|uncertain|pendiente\w*|parcial\w*|falta\w*|limit\w*|incomplet\w*|inciert\w*|discrepan\w*|riesgo\w*)\b",re.I)
    candidate=compact(s,limit)
    if qualification.search(s) and candidate!=s:
        return "[abreviado; fuente]"
    return candidate


def string_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x)]
    if isinstance(value, str) and value.startswith("["):
        try:
            return string_list(json.loads(value))
        except json.JSONDecodeError:
            pass
    return [str(value)]


def doi(value):
    s = str(value or "").strip().lower()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", s).rstrip(".,")


def arxiv(value):
    s = re.sub(r"^(?:arxiv:|https?://arxiv\.org/(?:abs|pdf)/)", "", str(value or "").strip(), flags=re.I)
    return re.sub(r"v\d+(?:\.pdf)?$|\.pdf$", "", s)


def clean_url(value):
    """Remove all query strings, fragments, userinfo, tokens, and email URLs."""
    try:
        u = urlsplit(str(value or ""))
        if u.scheme not in ("http", "https") or not u.hostname:
            return ""
        if EMAIL.search(u.path) or "@" in u.path or LOCAL.search(u.path):
            return ""
        host = u.hostname.lower()
        port = f":{u.port}" if u.port else ""
        return urlunsplit((u.scheme.lower(), host + port, u.path, "", ""))
    except (ValueError, UnicodeError):
        return ""


def urls(row):
    candidates = string_list(row.get("source_urls")) + string_list(row.get("source_url"))
    return sorted({u for v in candidates if (u := clean_url(v))})


def own_reason(row):
    for key in ("own_reason_es", "adjudicated_reason_es", "decision_reason", "reason_es", "own_reason", "reason", "title_evidence_paraphrase", "candidate_reason"):
        if row.get(key):
            return str(row[key])
    return ""


def evidence(row):
    return safe_text(row.get("evidence_level") or row.get("material_read") or "not_stated")


class Inputs:
    def __init__(self):
        self.sources = []

    def read(self, path, role, kind="jsonl"):
        data = path.read_bytes()
        self.sources.append({"source_id": role, "name": path.name, "sha256": digest(data), "bytes": len(data)})
        if kind == "json":
            return json.loads(data)
        if kind == "csv":
            return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        return [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]


class Union:
    def __init__(self, ids):
        self.p = {g: g for g in ids}

    def find(self, a):
        if self.p[a] != a:
            self.p[a] = self.find(self.p[a])
        return self.p[a]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def unique(rows, label):
    result = {}
    for row in rows:
        g = row["group_id"]
        if g in result:
            raise ValueError(f"Duplicate group in {label}: {g}")
        result[g] = row
    return result


def get_quality(row):
    q = row.get("quality_fields", row.get("fields", row.get("quality", {})))
    return q if isinstance(q, dict) else {}


def access(row):
    raw = str(row.get("report_access") or row.get("access_status") or row.get("status") or "")
    lev = evidence(row).lower()
    explicit = row.get("primary_report_assessed", row.get("assessed_fulltext"))
    if row.get("_prior_screening_only"):
        return "not_sought_after_verified_screening", False
    if row.get("_prior"):
        return "primary_report_reused", True
    if "not_retrieved" in raw or raw in {"access_failed", "unavailable"}:
        return "not_retrieved", False
    if "metadata_only" in raw or "metadata" in lev and "fulltext" not in lev:
        return "metadata_only", False
    assessed = bool(explicit) or any(t in lev for t in ("fulltext_sections_read", "methods_and_results", "methods_results", "primary_report_methods", "primary_methods_results"))
    if row.get("fulltext_eligibility_claimed") is True or row.get("fulltext_read_claimed") is True:
        assessed = True
    # Legacy shard explicitly distinguishes retrieved primary reports and carries
    # examined sections. No decision alone is allowed to imply retrieval.
    if raw == "retrieved" and row.get("examined_sections"):
        assessed = True
    if any(t in raw for t in ("fulltext_retrieved", "retrieved_primary_fulltext", "primary_report_sections_retrieved")):
        assessed = True
    if assessed:
        return "primary_report_retrieved", True
    return ("metadata_only" if raw else "not_stated"), False


def final_from_report(row):
    decision = row.get("decision", "")
    retrieval, primary = access(row)
    if row["group_id"] in CONTAINERS or decision == "volume_linked_existing_report":
        return "container_linked_existing_report", "identity", retrieval, False
    if row.get("_prior"):
        decision = {"incluido": "included_narrative", "excluido": "excluded_fulltext", "contexto_elegibilidad_pendiente": "context_criteria_unidentified"}[decision]
        if row.get("_prior_screening_only"):
            return "excluded_after_metadata_or_abstract", "screening", retrieval, False
    if decision == "included_narrative":
        if not primary:
            raise ValueError(f"Included without primary report evidence: {row['group_id']}")
        return "included_applied_narrative", "primary_report", retrieval, primary
    if decision == "excluded_fulltext":
        if not primary:
            raise ValueError(f"Fulltext exclusion without primary evidence: {row['group_id']}")
        return "excluded_after_primary_sections", "primary_report", retrieval, primary
    if decision in {"excluded_by_metadata_date", "excluded_metadata", "excluded_by_abstract", "exclude_screening"}:
        return "excluded_after_metadata_or_abstract", "metadata_or_abstract", retrieval, primary
    if decision in {"context_construct_unidentified", "context_criteria_unidentified"}:
        if not primary:
            raise ValueError(f"Scientific context disposition without primary evidence: {row['group_id']}")
        return "context_criteria_unidentified", "primary_report", retrieval, primary
    if decision == "context_modeling_forecast_unverified":
        if not primary:
            raise ValueError(f"Forecast-modeling context without primary evidence: {row['group_id']}")
        return "context_primary_source", "primary_report", retrieval, primary
    if decision == "mechanism_context":
        return ("context_primary_source" if primary else "context_candidate"), ("primary_report" if primary else "metadata_or_abstract"), retrieval, primary
    if decision == "eligibility_unknown":
        if primary:
            return "eligibility_unresolved_after_primary_sections", "primary_report", retrieval, primary
        return ("report_not_retrieved" if retrieval == "not_retrieved" else "eligibility_unknown_access_not_established"), "retrieval", retrieval, primary
    raise ValueError(f"Unsupported report decision {decision!r}: {row['group_id']}")


def write_csv(rows, columns):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False, sort_keys=True, separators=(",", ":")) if isinstance(row.get(k), (list, dict)) else row.get(k, "") for k in columns})
    return out.getvalue().encode()


def build(args):
    repo, reviews, output = (Path(x).resolve() for x in (args.repo_root, args.reviews_dir, args.output_dir))
    if output == repo or repo in output.parents or output == reviews or reviews in output.parents:
        raise ValueError("Output must be external to repository and private reviews directory")
    inp = Inputs()
    material_path = Path(args.materials).resolve() if args.materials else reviews.parent / "group_materials.jsonl"
    materials = unique(inp.read(material_path, "fixed_group_materials"), "materials")
    if len(materials) != 9559:
        raise ValueError(f"Expected exactly 9559 materials groups, got {len(materials)}")
    ids = set(materials)
    material_hashes, base_has_abstract, invalid_material_hashes = {}, 0, []
    for g, row in materials.items():
        hashes = set()
        for ab in row.get("abstracts", []):
            # Abstract bodies are read exclusively for hash validation and never
            # copied, summarized, or made reachable by an export dictionary.
            raw = ab.get("abstract", "")
            h = digest(raw.encode())
            if h != ab.get("abstract_sha256"):
                invalid_material_hashes.append(g)
            hashes.add(h)
        material_hashes[g] = hashes
        base_has_abstract += bool(hashes)
    if invalid_material_hashes:
        raise ValueError(f"Material abstract hash mismatch: {sorted(set(invalid_material_hashes))[:8]}")
    title_path = Path(args.arxiv_titles).resolve() if args.arxiv_titles else reviews.parent.parent / "arxiv/title_screening/PRIVATE/title_decisions_974.jsonl"
    stages, initial, current = {}, {}, {}
    for stem in ("crypto_review_0", "crypto_review_1", "crypto_review_2", "other_title_review_0", "other_title_review_1", "other_title_review_2"):
        rows = unique(inp.read(reviews / f"{stem}.jsonl", stem), stem)
        for g, row in rows.items():
            if g in initial:
                raise ValueError(f"Screening partitions overlap: {g}")
            row = dict(row, _source=stem)
            initial[g] = current[g] = row
            stages[g] = [row]
    arxiv_rows = unique(inp.read(title_path, "arxiv_title_screening_974"), "arxiv titles")
    if len(arxiv_rows) != 974:
        raise ValueError("Expected 974 arXiv title reviews")
    for g, original in arxiv_rows.items():
        if g in initial:
            raise ValueError(f"Screening partitions overlap: {g}")
        applied = original.get("applied_screen_decision")
        dec = "seek_abstract" if applied == "retain_for_abstract" else "mechanism_context" if original.get("mechanism_screen_decision") == "retain_original_candidate" else "exclude_screening"
        row = dict(original, decision=dec, evidence_level="title_only", _source="arxiv_title_screening_974")
        initial[g] = current[g] = row
        stages[g] = [row]
    focal = unique(inp.read(reviews / "focal_current_review.jsonl", "focal_current_review"), "focal current")
    if len(focal) != 169:
        raise ValueError("Expected 169 focal reviews")
    for g, row in focal.items():
        if g in initial:
            raise ValueError(f"Screening partitions overlap: {g}")
        row = dict(row, _source="focal_current_review")
        initial[g] = current[g] = row
        stages[g] = [row]
    focal_initial_path = reviews / "focal_missing_title_review.jsonl"
    if focal_initial_path.exists():
        focal_initial = unique(inp.read(focal_initial_path,"focal_initial_title_review"),"focal initial title review")
        if set(focal_initial) != set(focal): raise ValueError("Focal initial/current partition differs")
        for g, row in focal_initial.items():
            initial[g] = dict(row,_source="focal_initial_title_review")
            stages[g].insert(0,initial[g])
    if set(initial) != ids:
        raise ValueError(f"Screening universe mismatch: missing={len(ids-set(initial))}, extra={len(set(initial)-ids)}")
    abrows = unique(inp.read(reviews / "arxiv_abstract_review.jsonl", "arxiv_abstract_review"), "arxiv abstracts")
    if not set(abrows) <= set(arxiv_rows):
        raise ValueError("Recovered arXiv reviews outside title partition")
    for g, row in abrows.items():
        current[g] = dict(row, _source="arxiv_abstract_review")
        stages[g].append(current[g])
    # Additional material is consumed only to validate exact abstract hashes.
    # No abstract body, primary passage, or response is exported.
    arxiv_material_path = title_path.parent / "screening_ready_530.jsonl"
    recovered_abstract_groups = set()
    arxiv_material = unique(inp.read(arxiv_material_path, "arxiv_recovered_530_material"), "arxiv recovered material")
    if len(arxiv_material) != 530 or set(arxiv_material) != set(abrows):
        raise ValueError("530 recovered arXiv materials do not match reviewed IDs")
    for g, material in arxiv_material.items():
        if material_hashes[g]: raise ValueError("Recovered arXiv abstract overlaps base abstract coverage")
        for ab in material.get("abstracts", []):
            h=digest(ab.get("abstract", "").encode())
            if h != ab.get("abstract_sha256"): raise ValueError("Recovered arXiv hash mismatch")
            material_hashes[g].add(h)
        if not material_hashes[g]: raise ValueError("Recovered arXiv abstract missing")
        recovered_abstract_groups.add(g)
    material_doi_groups = collections.defaultdict(set)
    for g, row in materials.items():
        for d in string_list(row.get("DOIs")): material_doi_groups[doi(d)].add(g)
    focal_material = inp.read(reviews / "focal_abstract_recovery/recovered_abstracts.jsonl", "focal_recovered_abstract_material")
    focal_recovered_groups = set()
    for ab in focal_material:
        if not ab.get("complete_abstract_recovered"): continue
        gs=material_doi_groups.get(doi(ab.get("doi")),set())
        if len(gs)!=1: raise ValueError("Focal abstract material identity ambiguous")
        g=next(iter(gs))
        if material_hashes[g]: raise ValueError("Recovered focal abstract overlaps existing coverage")
        h=digest(ab.get("abstract", "").encode())
        if h!=ab.get("abstract_sha256"): raise ValueError("Recovered focal hash mismatch")
        material_hashes[g].add(h)
        focal_recovered_groups.add(g)
    if len(focal_recovered_groups)!=13: raise ValueError("Expected 13 disjoint recovered focal abstracts")
    validation_counts = collections.Counter()
    for g in sorted(ids):
        for row in stages[g]:
            h = row.get("abstract_sha256") or row.get("abstract_sha256_if_read") or ""
            if not h:
                validation_counts["no_abstract_hash_claimed"] += 1
            elif not HEX.fullmatch(h):
                raise ValueError(f"Invalid abstract hash: {g}")
            elif h in material_hashes[g]:
                validation_counts["matches_material_abstract"] += 1
            else:
                raise ValueError(f"Screening abstract hash is not in validated material for {g}")
    meta = repo / "references/literature_e2"
    nodes = inp.read(meta / "final_audit_20261113/source_nodes.csv", "historical_source_nodes", "csv")
    years = collections.defaultdict(set)
    node_group = {}
    for row in nodes:
        g = row.get("unified_group_id")
        if g in ids:
            node_group[row["node_id"]] = g
            if row.get("year"):
                years[g].add(safe_text(row["year"]))
    protocol = inp.read(meta / "prisma_material_recovery_v1/screening_protocol.json", "retrospective_screening_protocol", "json")
    if protocol["universe"]["fixed_received_groups"] != 9559:
        raise ValueError("Protocol universe differs")
    prior_rows = inp.read(meta / "closure_final/eligibility.json", "prior_primary_eligibility_30", "json")
    if len(prior_rows) != 30 or collections.Counter(x["decision"] for x in prior_rows) != {"incluido": 17, "excluido": 11, "contexto_elegibilidad_pendiente": 2}:
        raise ValueError("Prior focal evidence differs from 17/11/2")
    prior_quality = inp.read(meta / "closure_final/quality/quality_extraction_matrix_17.json", "prior_quality_17", "json")
    q_by_doi={}
    for i,study in enumerate(prior_quality["studies"]):
        fields={}
        for dim,field in study.get("fields",{}).items():
            fields[dim]={**field,"source_pointer":f"references/literature_e2/closure_final/quality/quality_extraction_matrix_17.json#/studies/{i}/fields/{dim}"}
        q_by_doi[doi(study.get("doi"))]=fields
    doi_groups, arxiv_groups = collections.defaultdict(set), collections.defaultdict(set)
    for g, row in materials.items():
        for d in string_list(row.get("DOIs")):
            if doi(d): doi_groups[doi(d)].add(g)
        for a in string_list(row.get("arxiv_ids")):
            if arxiv(a): arxiv_groups[arxiv(a)].add(g)
    union, links = Union(ids), []
    def link(a, b, kind, identifier=""):
        if a not in ids or b not in ids:
            raise ValueError(f"Identity link outside received universe: {a}, {b}")
        if a in CONTAINERS or b in CONTAINERS:
            return
        if a != b:
            union.join(a, b)
            links.append({"group_id_a": min(a,b), "group_id_b": max(a,b), "relationship": kind, "identifier": safe_text(identifier)})
    for kind, index in (("same_exact_doi", doi_groups), ("same_exact_arxiv", arxiv_groups)):
        for key, gs in sorted(index.items()):
            ordered = sorted(gs)
            for other in ordered[1:]: link(ordered[0], other, kind, key)
    link(*CRYPTO_ALIAS, "explicit_same_study_alias", "CryptoWinter")
    published_alias = inp.read(meta / "delivery_closure_20261114_v2/eligibility/historical43_source_roles.json", "published_preprint_final_identity_evidence", "json")
    evidence_row = next((r for r in published_alias if doi(r.get("doi")) == "10.1109/access.2025.3584243"),None)
    if not evidence_row or "https://arxiv.org/html/2211.08281" not in evidence_row.get("source_urls",[]):
        raise ValueError("Documentary preprint/final identity evidence absent")
    link("lit-e8ded7c56180aa56f833","lit-26714704cb0a4329c015","documented_preprint_final_version","10.1109/access.2025.3584243 = arXiv:2211.08281")
    reports, report_history, queue_lookup = {}, [], {}
    excluded_inputs = []
    for f in sorted(reviews.glob("fulltext_reviews*.jsonl")):
        if any(x in f.stem for x in ("history", "receipt", "batch")):
            excluded_inputs.append(f.name)
            continue
        for row in inp.read(f, f.stem):
            g = row.get("group_id")
            if g not in ids:
                raise ValueError(f"Report outside received universe in {f.name}: {g}")
            if g in reports:
                raise ValueError(f"Duplicate report rows require explicit adjudication: {g}")
            row = dict(row, _source=f.stem)
            reports[g] = row
            if row.get("queue_index") is not None:
                queue_lookup[(f.stem.removeprefix("fulltext_reviews"), str(row["queue_index"]))] = g
            for a in string_list(row.get("alias_group_ids")):
                link(g, a, "review_explicit_same_study")
            for a in string_list(row.get("aliases")):
                if a.startswith("lit-"):
                    link(g, a, "review_explicit_same_study")
                elif a.startswith("10."):
                    for other in doi_groups.get(doi(a), ()): link(g, other, "review_exact_doi_alias", doi(a))
                elif a.lower().startswith("arxiv:"):
                    for other in arxiv_groups.get(arxiv(a), ()): link(g, other, "review_exact_arxiv_alias", arxiv(a))
    private_report_count = len(reports)
    prior_gids = set()
    for row in prior_rows:
        candidates = doi_groups.get(doi(row.get("doi")), set())
        by_study = {g for node, g in node_group.items() if node.endswith(":" + row["study_id"])}
        if by_study:
            candidates = candidates & by_study if candidates else by_study
        if not candidates:
            raise ValueError(f"Prior primary review not linked: {row['study_id']}")
        if len({union.find(g) for g in candidates}) != 1:
            raise ValueError(f"Ambiguous prior primary review: {row['study_id']}")
        g = sorted(candidates)[0]
        prior_gids.add(g)
        if g in reports:
            report_history.append({"group_id": g, "event": "prior_primary_reused_over_new_proposal", "superseded_decision": reports[g]["decision"], "superseded_source": reports[g]["_source"], "superseded_reason_sha256": digest(own_reason(reports[g]).encode())})
        screening_only = doi(row.get("doi")) in PRIOR_SCREENING_ONLY
        reports[g] = dict(row, group_id=g, _prior=True, _prior_screening_only=screening_only, _source="prior_primary_eligibility_30", evidence_level=PRIOR_SCREENING_ONLY.get(doi(row.get("doi")),"previously_verified_primary_report_sections"), quality_fields=q_by_doi.get(doi(row.get("doi")), {}), examined_sections=row.get("source_locators", []))
    adjudication_attempts=[]
    adj_path = reviews / "fulltext_adjudication_neural_sde.json"
    if adj_path.exists():
        adj = inp.read(adj_path, "neural_sde_adjudication", "json")
        g = adj["group_id"]
        if g not in reports: raise ValueError("SDE adjudication has no original report")
        row = reports[g]
        if row["decision"] != adj["original_decision"]:
            raise ValueError("SDE original decision mismatch")
        report_history.append({"group_id":g,"event":adj["change_type"],"superseded_decision":row["decision"],"superseded_reason_sha256":digest(own_reason(row).encode()),"adjudication_source":"neural_sde_adjudication"})
        reports[g] = dict(row, decision=adj["adjudicated_decision"], own_reason_es=adj["adjudicated_reason_es"], _adjudication="neural_sde_adjudication")
    sentiment_path=reviews / "fulltext_adjudication_sentiment_igarch.json"
    if not sentiment_path.exists():
        raise ValueError("Final v1 requires the separate Sentiment-IGARCH adjudication; original inclusion must not be silently restored")
    if sentiment_path.exists():
        adj=inp.read(sentiment_path,"sentiment_igarch_adjudication","json")
        g=adj["group_id"]
        if g not in reports or reports[g]["decision"]!=adj["original_decision"]:
            raise ValueError("Sentiment-IGARCH adjudication has no matching original decision")
        original_sha=next(s["sha256"] for s in inp.sources if s["source_id"]=="fulltext_reviews_0")
        if original_sha!=adj["audit"]["reviews_sha256_before"] or original_sha!=adj["audit"]["reviews_sha256_after"]:
            raise ValueError("Original 52-review history changed across Sentiment-IGARCH adjudication")
        if adj["adjudicated_decision"]!="context_modeling_forecast_unverified" or not adj["target_identified"] or not adj["temporal_order"]["verified"]:
            raise ValueError("Unexpected Sentiment-IGARCH scientific adjudication")
        original=reports[g]
        report_history.append({"group_id":g,"event":adj["change_type"],"superseded_decision":original["decision"],"final_decision":adj["adjudicated_decision"],"superseded_reason_sha256":digest(own_reason(original).encode()),"original_52_reviews_sha256":original_sha,"adjudication_source":"sentiment_igarch_adjudication","criterion_boundary_code":"empirical_dispersion_forecast_evaluation_no_additional_oos_rule"})
        locators=[p["locator"] for p in adj["primary_evidence"]]
        scoped_quality={dim:{"status":field["status"],"value":field.get("evidence",""),"locators":locators if field.get("evidence") else []} for dim,field in adj["quality_scoped_to_variance_pipeline"].items()}
        reports[g]=dict(original,decision=adj["adjudicated_decision"],own_reason_es=adj["adjudicated_reason_es"],status=adj["status"],evidence_level=adj["evidence_level"],source_url=adj["source_url"],version=adj["version_examined"],examined_sections=locators,quality=scoped_quality,neural_role=adj["neural_role"],forecast_head=adj["forecast_head"],population=adj["population"],target_identified=True,temporal_order_verified=True,empirical_dispersion_forecast_evaluation=adj["empirical_dispersion_forecast_evaluation"],criterion_boundary_code="empirical_dispersion_forecast_evaluation_no_additional_oos_rule",_adjudication="sentiment_igarch_adjudication")
        comparison=adj["comparison_hybrid_check"]
        cg=comparison["group_id"]
        if cg not in reports or reports[cg]["decision"]!=comparison["decision"] or comparison["decision_changed"]:
            raise ValueError("Verified indirect-hybrid comparison inclusion was not preserved")
        reports[cg]=dict(reports[cg],neural_role=comparison["neural_role"],forecast_head=comparison["forecast_head"],empirical_dispersion_forecast_evaluation=comparison["empirical_dispersion_forecast_evaluation"])
        report_history.append({"group_id":cg,"event":"additional_primary_hybrid_check_decision_unchanged","final_decision":comparison["decision"],"adjudication_source":"sentiment_igarch_adjudication"})
        for attempt in adj.get("adjudication_access_attempts",[]):
            ag=g if clean_url(attempt["url"])==clean_url(adj["source_url"]) else cg
            adjudication_attempts.append({"group_id":ag,"url":clean_url(attempt["url"]),"status":safe_text(attempt["status"]),"kind":"additional_primary_adjudication","source":"sentiment_igarch_adjudication"})
    history_path = reviews / "fulltext_reviews_arxiv_decision_history.jsonl"
    if history_path.exists():
        for row in inp.read(history_path, "arxiv_decision_history"):
            if row["group_id"] not in reports or reports[row["group_id"]]["decision"] != row["final_decision"]:
                raise ValueError("arXiv decision amendment disagrees with final report")
            report_history.append({"group_id":row["group_id"],"event":row["event"],"superseded_decision":row.get("provisional_decision"),"final_decision":row["final_decision"],"source":"arxiv_decision_history","superseded_record_sha256":row.get("provisional_record_sha256","")})
    components = collections.defaultdict(list)
    for g in sorted(ids-CONTAINERS): components[union.find(g)].append(g)
    study_ids = {g: "study-" + digest("\n".join(gs).encode())[:20] for gs in components.values() for g in gs}
    peer_reports = collections.defaultdict(list)
    for g in reports:
        if g not in CONTAINERS: peer_reports[union.find(g)].append(g)
    decisions, pending, records, full_public, qualities = {}, [], [], [], []
    allowed_screen = {"exclude_screening", "mechanism_context", "seek_abstract", "seek_fulltext"}
    for g in sorted(ids):
        proposal = current[g]
        if proposal.get("decision") not in allowed_screen:
            raise ValueError(f"Unknown screening decision: {g}, {proposal.get('decision')}")
        report_gid = g if g in reports else None
        if report_gid is None and g not in CONTAINERS and peer_reports[union.find(g)]:
            candidates = sorted(peer_reports[union.find(g)], key=lambda z: (not reports[z].get("_prior",False), -int(access(reports[z])[1]), z))
            report_gid = candidates[0]
        row = reports.get(report_gid, {})
        if row:
            final, stage, retrieval, primary = final_from_report(row)
            reason = compact(own_reason(row), 35)
            final_evidence = evidence(row)
        elif g in CONTAINERS:
            final, stage, retrieval, primary = "container_linked_existing_report", "identity", "not_stated", False
            reason, final_evidence = "Contenedor bibliográfico; no constituye otro estudio empírico.", "bibliographic_identity"
        elif proposal["decision"] == "exclude_screening":
            final, stage, retrieval, primary = "excluded_at_title_or_abstract", "screening", "not_sought_after_screening", False
            reason, final_evidence = compact(own_reason(proposal), 35), evidence(proposal)
        elif proposal["decision"] == "mechanism_context":
            final, stage, retrieval, primary = "context_candidate", "screening", "not_sought_for_applied_core", False
            reason, final_evidence = compact(own_reason(proposal), 35), evidence(proposal)
        else:
            final, stage, retrieval, primary = "pending_report_disposition", "pending", "not_documented", False
            reason, final_evidence = "Falta una disposición documentada de recuperación o revisión del informe.", evidence(proposal)
            pending.append({"group_id":g,"screening_proposal":proposal["decision"],"required":"document_report_retrieval_or_review_disposition"})
        if final == "eligibility_unknown_access_not_established":
            pending.append({"group_id":g,"screening_proposal":proposal["decision"],"required":"clarify_access_status_without_inventing_retrieval"})
        decisions[g] = final
        ph = proposal.get("abstract_sha256") or proposal.get("abstract_sha256_if_read") or proposal.get("new_abstract_sha256") or ""
        rec = {"group_id":g,"study_id":study_ids.get(g,""),"title":safe_text(materials[g]["title"]),"DOIs":sorted({doi(d) for d in string_list(materials[g].get("DOIs"))}),"arxiv_ids":sorted({arxiv(a) for a in string_list(materials[g].get("arxiv_ids"))}),"years":sorted(years[g]),"initial_screening_proposal":initial[g]["decision"],"current_screening_proposal":proposal["decision"],"screening_read_scope":evidence(proposal),"abstract_sha256_if_read":ph,"screening_source":proposal["_source"],"screening_reason_code":safe_text(proposal.get("reason_code") or proposal["decision"]),"screening_reason_brief":compact(own_reason(proposal),35),"final_disposition":final,"final_stage":stage,"final_evidence_level":final_evidence,"retrieval_status":retrieval,"primary_report_assessed":primary,"report_review_group_id":report_gid or "","final_reason_brief":reason,"adjudication_applied":row.get("_adjudication", "prior_primary_reused" if row.get("_prior") else ""),"operational_pending":any(p["group_id"]==g for p in pending),"numerical_pooling_permitted":False}
        rec["final_reason_ref"]="screening_reason_brief" if rec["final_reason_brief"]==rec["screening_reason_brief"] else ""
        if rec["final_reason_ref"]: rec["final_reason_brief"]=""
        records.append(rec)
        if row and report_gid == g:
            fields = get_quality(row)
            # The 70-word limit covers the final own reason plus all 12 values.
            report_reason=reason
            remaining, compact_fields = 70-word_count(report_reason), {}
            priority = ("target","data","horizon","models","split","metrics","repetitions","hpo","tests","code","leakage","fairness")
            caps = {"target":10,"data":8,"horizon":5,"models":8}
            raw_values={dim:(fields.get(dim,{}).get("value") or fields.get(dim,{}).get("summary") or "") if isinstance(fields.get(dim,{}),dict) else "" for dim in priority}
            for idx,dim in enumerate(priority):
                q = fields.get(dim, {})
                if not isinstance(q, dict): q = {"status":"not_indexed_in_this_bundle"}
                val=raw_values[dim]
                reserved=2*sum(bool(raw_values[d]) for d in priority[idx+1:])
                allowance=max(0,min(caps.get(dim,4),remaining-reserved))
                value=compact_quality_value(val,allowance) if val and allowance else ""
                remaining -= word_count(value)
                locators = string_list(q.get("locators")) + string_list(q.get("source_locator"))
                compact_fields[dim] = {"status":safe_text(q.get("status") or "not_indexed_in_this_bundle"),"value_brief":value,"value_compacted":safe_text(val)!=value,"value_source_field":"value" if q.get("value") else "summary" if q.get("summary") else "none","source_pointer":q.get("source_pointer",f"full_reports_compact.jsonl#group_id={g}&dimension={dim}"),"locators":[safe_text(s) for s in locators]}
            fp={"group_id":g,"study_id":study_ids.get(g,""),"title":rec["title"],"DOIs":rec["DOIs"],"arxiv_ids":rec["arxiv_ids"],"final_disposition":final,"retrieval_status":retrieval,"primary_report_assessed":primary,"evidence_level":final_evidence,"reason_brief":report_reason,"reason_compacted":safe_text(own_reason(row))!=report_reason,"private_reason_sha256":digest(own_reason(row).encode()),"source_urls":urls(row),"examined_sections":[safe_text(s) for s in string_list(row.get("examined_sections"))],"source_version":safe_text(row.get("version") or row.get("version_examined") or ""),"review_source":row["_source"],"adjudication_applied":rec["adjudication_applied"],"quality":{dim:compact_fields[dim] for dim in DIMS},"own_reason_plus_values_word_count":word_count(report_reason)+sum(word_count(q["value_brief"]) for q in compact_fields.values()),"numerical_pooling_permitted":False}
            for key in ("neural_role","forecast_head","population","criterion_boundary_code"):
                if isinstance(row.get(key),str): fp[key]=safe_text(row[key])
            for key in ("target_identified","temporal_order_verified","empirical_dispersion_forecast_evaluation"):
                if key in row: fp[key]=row[key]
            if fp["own_reason_plus_values_word_count"]>70: raise ValueError("Public prose budget exceeded")
            full_public.append(fp)
            for dim in DIMS:
                qualities.append({"group_id":g,"study_id":study_ids.get(g,""),"dimension":dim,**compact_fields[dim]})
    # Attempts contain only URL, bounded status/kind codes, and identity; never
    # provider extracts, response bodies, notes, email addresses, or URL queries.
    attempts, attempts_unmapped = list(adjudication_attempts), 0
    attempt_paths = sorted(set(reviews.glob("fulltext_attempts*.jsonl")) | set(reviews.glob("attempts_1_part_*.jsonl")))
    for f in attempt_paths:
        for row in inp.read(f, f.stem):
            g=row.get("group_id")
            if not g and row.get("queue_index") is not None:
                suffix=f.stem.removeprefix("fulltext_attempts").removeprefix("attempts")
                g=queue_lookup.get((suffix,str(row["queue_index"])))
            if g and g not in ids: raise ValueError("Attempt outside received universe")
            if not g: attempts_unmapped+=1
            u=clean_url(row.get("url") or row.get("source_url"))
            if not u: continue
            def code(v):
                return re.sub(r"[^a-zA-Z0-9_.:/-]", "_", str(v or ""))[:120]
            attempts.append({"group_id":g or "","url":u,"status":code(row.get("status") or row.get("http_status") or "not_stated"),"kind":code(row.get("kind") or row.get("request_kind") or row.get("role") or row.get("action") or row.get("route")),"source":f.stem})
    attempts=list({tuple(sorted(x.items())):x for x in attempts}.values())
    attempts.sort(key=lambda x:(x["group_id"],x["url"],x["status"],x["source"]))
    attempted_groups={a["group_id"] for a in attempts if a["group_id"]}
    for r in full_public:
        if r["final_disposition"]=="report_not_retrieved" and r["group_id"] not in attempted_groups:
            pending.append({"group_id":r["group_id"],"required":"document_at_least_one_retrieval_attempt_url_and_status"})
        if r["primary_report_assessed"] and (not r["source_urls"] or not r["examined_sections"]):
            raise ValueError(f"Primary assessment lacks URL or examined-section locator: {r['group_id']}")
    study_rows, conflicts = [], []
    for root, gs in sorted(components.items()):
        statuses={decisions[g] for g in gs}
        core=any(decisions[g]=="included_applied_narrative" for g in gs)
        excluded=any(decisions[g]=="excluded_after_primary_sections" for g in gs)
        if core and excluded: conflicts.append({"study_id":study_ids[gs[0]],"group_ids":gs,"conflict":"included_and_primary_excluded"})
        version_candidates=[g for g in gs if g in reports and access(reports[g])[1]]
        preferred=sorted(version_candidates,key=lambda g:(not reports[g].get("_prior",False),g))[0] if version_candidates else ""
        study_rows.append({"study_id":study_ids[gs[0]],"group_ids":gs,"DOIs":sorted({doi(d) for g in gs for d in string_list(materials[g].get("DOIs"))}),"arxiv_ids":sorted({arxiv(a) for g in gs for a in string_list(materials[g].get("arxiv_ids"))}),"group_dispositions":sorted(statuses),"included_applied_narrative":core and not excluded,"preferred_report_review_group_id":preferred,"no_value_merging_across_report_versions":True,"operational_pending":any(r["group_id"] in gs for r in pending)})
    full_by_group={r["group_id"]:r for r in full_public}
    applied_quality=[]
    for study in study_rows:
        if not study["included_applied_narrative"]: continue
        selected=full_by_group.get(study["preferred_report_review_group_id"])
        if not selected or selected["final_disposition"]!="included_applied_narrative":
            raise ValueError("Applied study lacks included preferred report quality source")
        for dim in DIMS:
            applied_quality.append({"study_id":study["study_id"],"group_ids":study["group_ids"],"preferred_report_review_group_id":selected["group_id"],"title":selected["title"],"DOIs":selected["DOIs"],"arxiv_ids":selected["arxiv_ids"],"dimension":dim,**selected["quality"][dim],"source_urls":selected["source_urls"],"source_version":"journal_final_2025" if selected["group_id"]=="lit-26714704cb0a4329c015" else selected["source_version"] or ("previously_verified_primary_report_version" if selected["adjudication_applied"]=="prior_primary_reused" else "examined_primary_report_version"),"no_value_merging_across_report_versions":True})
    for conflict in conflicts: pending.append({"group_id":conflict["group_ids"][0],"required":"adjudicate_same_study_conflicting_primary_decisions"})
    pending_ids={r["group_id"] for r in pending}
    for r in records: r["operational_pending"]=r["group_id"] in pending_ids
    for r in study_rows: r["operational_pending"]=bool(pending_ids.intersection(r["group_ids"]))
    flow={"scope":"received_corpus_only","historical_search_coverage_complete":False,"received_occurrences":11635,"within_round_duplicate_occurrences":1720,"cross_round_link_reductions":356,"received_groups_screened":9559,"conservation_equation":"11635 - 1720 - 356 = 9559","directed_only_groups_outside_received_flow":6,"scopus_shown_not_received":374,"wos_native_query_totals_and_batch_ranges":"not_fully_verified","received_group_final_dispositions":dict(sorted(collections.Counter(decisions.values()).items())),"final_group_disposition_sum":sum(collections.Counter(decisions.values()).values()),"primary_report_review_rows":len(full_public),"primary_reports_assessed_or_reused":sum(x["primary_report_assessed"] for x in full_public),"report_not_retrieved_rows":sum(x["final_disposition"]=="report_not_retrieved" for x in full_public),"unique_empirical_or_candidate_study_identities_excluding_containers":len(study_rows),"unique_studies_included_applied_narrative":sum(x["included_applied_narrative"] for x in study_rows),"operational_pending_count":len(pending),"operationally_reconciled_at_snapshot":not pending,"scientific_eligibility_unresolved_is_not_an_operational_task":True,"no_exhaustiveness_or_preregistration_claim":True}
    flow["report_or_screening_disposition_rows_including_reuse"]=flow.pop("primary_report_review_rows")
    flow["bibliographic_identity_components_excluding_containers"]=flow.pop("unique_empirical_or_candidate_study_identities_excluding_containers")
    flow["primary_assessment_scope"]="localized/focal primary report sections examined, or previous primary assessment reused; not a claim of complete reading of every PDF"
    audit={"version":VERSION,"coverage_exact_9559":len(records)==9559,"all_screening_groups_exactly_once":set(initial)==ids,"prior_focal_decisions_reused":len(prior_gids),"prior_primary_section_assessments_reused":25,"prior_title_or_abstract_exclusions_reused":5,"prior_decisions":{"included_primary":17,"excluded_primary":6,"excluded_title_or_abstract":5,"context_primary":2},"current_arxiv_abstract_reviews":len(abrows),"base_groups_with_abstract_material":base_has_abstract,"stored_abstract_coverage":{"base":base_has_abstract,"additional_arxiv":len(recovered_abstract_groups),"additional_focal":len(focal_recovered_groups),"disjoint_total":sum(bool(v) for v in material_hashes.values())},"screening_groups_with_real_abstract_read":sum(evidence(current[g]) in {"title_and_abstract_read","prior_abstract_review_reused","abstract_full_public"} for g in ids),"abstract_hash_validation":dict(sorted(validation_counts.items())),"private_report_review_rows":private_report_count,"public_report_or_screening_rows_including_prior_reuse":len(full_public),"max_public_reason_plus_values_words":max(x["own_reason_plus_values_word_count"] for x in full_public),"unmapped_attempt_rows":attempts_unmapped,"identity_conflicts":conflicts,"operational_pending":pending,"excluded_non_review_file_names":excluded_inputs,"ready_for_strict_bundle":not pending,"public_content_policy":"allowlisted metadata, hashes, compact own adjudications; no abstracts, raw papers, private paths, responses, emails or URL query strings"}
    audit["applied_quality_rows"]=len(applied_quality)
    audit["prior_quality_dimensions_with_summary_fallback"]=sum(r["value_source_field"]=="summary" for r in qualities)
    audit["known_prior_quality_dimensions_with_empty_brief"]=sum(r["status"]=="conocido" and not r["value_brief"] for r in qualities if r["value_source_field"]=="summary")
    manifest={"version":VERSION,"original_A2_sha256":GUIDE_SHA,"builder_sha256":digest(Path(__file__).read_bytes()),"sources":sorted(inp.sources,key=lambda x:x["source_id"]),"source_snapshot_policy":"one byte snapshot per file; no filesystem timestamps in deterministic products","semantic_screening":"AI-assisted semantic reviews recorded in source files; no independent human double-review or kappa claim","temporal_status":"retrospective documentary reconciliation; not bibliographic preregistration","historical_coverage_limits":protocol.get("historical_limits",[]),"abstract_hash_validation":"Every claimed screening abstract hash must match a validated base or recovered abstract; no abstract retrieval or copying is performed.","strict_ready":not pending}
    manifest["identity_policy"]="study_id is a deterministic surrogate key for an exact/documented bibliographic identity component; components are not verified empirical studies. Applied inclusion is a separate adjudicated property."
    if pending and not args.allow_partial:
        output.mkdir(parents=True,exist_ok=True)
        (output/"audit_failure.json").write_bytes(encode(audit))
        print(json.dumps({"status":"strict_failed","operational_pending":len(pending),"audit_file":"audit_failure.json"}))
        return 2
    columns=list(records[0])
    files={"received_groups_9559.csv":write_csv(records,columns),"full_reports_compact.jsonl":b"".join((json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n").encode() for x in full_public),"quality_12_dimensions.csv":write_csv(qualities,["group_id","study_id","dimension","status","value_brief","value_compacted","value_source_field","source_pointer","locators"]),"quality_applied_studies.csv":write_csv(applied_quality,list(applied_quality[0]) if applied_quality else ["study_id","dimension"]),"study_identities.csv":write_csv(study_rows,list(study_rows[0])),"identity_links.csv":write_csv(sorted({tuple(sorted(x.items())):x for x in links}.values(),key=lambda x:(x["group_id_a"],x["group_id_b"],x["relationship"])),["group_id_a","group_id_b","relationship","identifier"]),"report_adjudication_history.json":encode(sorted(report_history,key=lambda x:(x["group_id"],x["event"]))),"retrieval_attempts.csv":write_csv(attempts,["group_id","url","status","kind","source"]),"received_flow.json":encode(flow),"audit.json":encode(audit),"operational_pending.csv":write_csv(pending,["group_id","screening_proposal","required"])}
    # Independent boundary validation of serialized products, before writing.
    forbidden=[b'"abstract":',b'"abstracts":',b'/workspace/',b'/tmp/',b'file://']
    for name,data in files.items():
        if any(token in data for token in forbidden) or EMAIL.search(data.decode()):
            raise ValueError(f"Public boundary check failed: {name}")
    files["README.md"]=("# Borrador de reconciliación del corpus recibido\n\n"+f"Estado operativo: {'cerrado al snapshot' if not pending else 'parcial; '+str(len(pending))+' pendientes operativos'}.\n\n"+"11.635 ocurrencias − 1.720 duplicados internos − 356 enlaces = 9.559 grupos cribados. Seis dirigidos quedan aparte; faltan 374 ocurrencias Scopus mostradas y no recibidas; no están confirmados todos los totales/rangos nativos WoS. No se declara cobertura histórica completa ni preregistro bibliográfico.\n\n"+"El CSV conserva propuestas de cribado y dictamen final en columnas distintas. Se reutilizan 30 decisiones focales previas: 25 evaluaciones de secciones primarias localizadas (17 incluidas, 6 excluidas, 2 contextos) y 5 exclusiones por título/resumen. Una evaluación primaria focal no afirma lectura íntegra de cada PDF. Los contenedores y alias no añaden estudios. Un informe no recuperado no es exclusión; la incertidumbre científica tras lectura primaria tampoco supone una tarea técnica nueva. Los candidatos contextuales por título/resumen no son fuentes primarias incluidas.\n\n"+"full_reports_compact.jsonl limita la razón propia y los 12 valores a 70 palabras por fila. Los textos con … están recortados para publicación; las notas completas quedan privadas, con hash de trazabilidad. Los estados y localizadores se preservan. No reportado, no verificado y no indexado son categorías diferentes.\n\n"+"Los conteos de estudios usan identificadores exactos y alias explícitos, sin unión difusa. El flujo conserva grupos históricos; por eso grupos, informes y estudios tienen denominadores distintos. Las versiones mantienen sus extracciones separadas; se prefiere la ficha editorial focal previa para síntesis, sin importar fórmulas del preprint. No se permite pooling numérico.\n").encode()
    files["README.md"]+=(f"\n`study_id` es una clave sustituta determinista para un componente de identidad bibliográfica; 9.555 componentes no equivalen a 9.555 estudios empíricos verificados. La inclusión aplicada se adjudica aparte. `quality_applied_studies.csv` contiene {len(applied_quality)} filas: doce dimensiones para los {flow['unique_studies_included_applied_narrative']} estudios incluidos, usando una sola versión preferida por estudio. `quality_12_dimensions.csv` abarca los {len(full_public)} dictámenes generales, con otro denominador.\n\nLos campos previos usan `summary` como alternativa a `value` y enlazan cada dimensión al índice público de calidad de 17 estudios. Si un recorte pudiera eliminar una negación o limitación, se publica una referencia abreviada neutral a la fuente, sin una afirmación positiva recortada. `final_reason_ref=screening_reason_brief` indica reutilización exacta de esa razón y evita duplicarla.\n").encode()
    if sentiment_path.exists():
        files["README.md"]+=("\nLa adjudicación primaria adicional de Sentiment-IGARCH conserva target, orden temporal y vínculo neural identificados, pero lo reclasifica como contexto: la evidencia examinada verifica estimación/significación y trayectoria de varianza, sin establecer evaluación empírica del forecast de dispersión. La comparación de retornos no sustituye esa evaluación; no se añade OOS como regla. La inclusión de Lehrer/Xie/Yi se conserva por su evaluación predictiva de BTC-RV. El historial de los 52 dictámenes originales permanece íntegro y ambas verificaciones están registradas por separado.\n").encode()
    manifest["artifacts"]=[{"name":n,"sha256":digest(b),"bytes":len(b)} for n,b in sorted(files.items())]
    files["manifest.json"]=encode(manifest)
    output.mkdir(parents=True,exist_ok=True)
    for name,data in sorted(files.items()):
        target=output/name
        temporary=target.with_suffix(target.suffix+".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    if (output/"audit_failure.json").exists(): (output/"audit_failure.json").unlink()
    print(json.dumps({"status":"partial" if pending else "strict_ready","received_groups":len(records),"report_reviews":len(full_public),"included_unique_studies":flow["unique_studies_included_applied_narrative"],"operational_pending":len(pending),"max_public_words":audit["max_public_reason_plus_values_words"]},ensure_ascii=False))
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",required=True)
    p.add_argument("--reviews-dir",required=True,help="Private global_screening directory")
    p.add_argument("--output-dir",required=True,help="External destination; never inside repository/private inputs")
    p.add_argument("--materials",help="Optional corrected group_materials.jsonl; default: parent of reviews-dir")
    p.add_argument("--arxiv-titles",help="Optional 974-title decisions file")
    p.add_argument("--allow-partial",action="store_true",help="Export explicit pending dispositions; strict default requires zero operational pending")
    args=p.parse_args()
    try:
        return build(args)
    except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
