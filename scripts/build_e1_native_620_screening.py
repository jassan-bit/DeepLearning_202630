#!/usr/bin/env python3
"""Compile the dated, agent-assisted screening of received native E1 exports.

This compiler does not classify papers, browse, retrieve documents or train
models. It consumes explicit reviewed decisions. The original search campaign,
earlier documentary cuts and frozen experimental results are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "references/reviews/native_620_screening_20261108"
NATIVE = ROOT / "book/_static/literature_e1/native_execution_20261107"
PREVIOUS = ROOT / "references/literature_e1/homogeneous_expansion_20261107/adjudications.json"
OUTPUT = ROOT / "references/literature_e1/native_620_screening_20261108"
MIRROR = ROOT / "book/_static/literature_e1/native_620_screening_20261108"
STAGE = "native_620_screening_20261108"
PUBLIC_SCHEMA = "e1_public_metadata_allowlist_v1"
SCREEN_DECISIONS = {"exclude_title_abstract", "advance_primary_assessment"}
DISPOSITIONS = {"included_strict", "excluded_primary_target", "excluded_primary_reporting",
                "report_not_retrieved", "excluded_after_metadata_enrichment", "included_context_only"}
DECISION_FIELDS = ("group_id", "screening_decision", "reason_code", "reason", "target_category",
                   "evidence_level", "abstract_present", "reviewer", "evidence_note",
                   "final_disposition", "prior_review_ids")
REVIEW_FIELDS = ("group_id", "doi", "title", "authors", "year", "reviewer", "assessed_at_utc",
                "document_access", "primary_sections_assessed", "target_quantity", "target_crypto",
                "continuous_construct_eligible", "architecture_reference_adequate",
                "reporting_metrics_adequate", "reporting_protocol_adequate", "dataset", "horizon",
                "architecture_details", "metrics_reported", "neural_baseline_role_verified", "comparator_families", "proposed_or_component_families",
                "independence_key", "disposition", "reason", "source_urls", "source_locators",
                "protocol_clarity", "leakage_risk", "repetitions", "statistical_tests", "limitations")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                         if isinstance(row.get(key), (list, dict)) else row.get(key, "") for key in fields})
    return stream.getvalue().encode()


def normalize_doi(value):
    value = str(value or "").strip().lower()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value).rstrip(" .;")


def native_panel(records, occurrences):
    """Preserve the original provisional candidate grouping, not a paper count."""
    all_ids = [row["occurrence_id"] for row in occurrences]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Repeated occurrence identity")
    fresh = [row for row in occurrences if row["origin"] == "new_native_export"]
    groups = {row["group_id"] for row in fresh}
    selected = [row for row in records if row["group_id"] in groups]
    if len(selected) != len(groups) or len({row["group_id"] for row in selected}) != len(selected):
        raise ValueError("Native groups do not map one-to-one to original records")
    return selected, fresh


def validate_decisions(records, decisions, *, expected_count=620):
    expected = {row["group_id"] for row in records}
    actual = [row.get("group_id") for row in decisions]
    if len(expected) != expected_count or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Screening needs every original native group exactly once")
    for row in decisions:
        if row.get("decision") not in SCREEN_DECISIONS:
            raise ValueError("Unknown screening decision")
        for key in ("reason_code", "reason", "target_category", "reviewer", "evidence_note"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("Every screening judgement needs a substantive reason and provenance")
        if type(row.get("abstract_present")) is not bool:
            raise ValueError("Abstract availability must be explicitly observed")
        if row.get("evidence_level") not in {"title_and_export_abstract", "title_only", "existing_primary_review"}:
            raise ValueError("Unknown screening evidence level")
        if row["abstract_present"] and row["evidence_level"] == "title_only":
            raise ValueError("Available abstract must be reviewed, not silently title-screened")
        if not row["abstract_present"] and row["decision"] != "advance_primary_assessment":
            raise ValueError("The registered protocol advances missing abstracts to enrichment/retrieval")
    return {row["group_id"]: row for row in decisions}


def validate_reviews(decisions, reviews):
    advanced = {row["group_id"] for row in decisions if row["decision"] == "advance_primary_assessment"}
    actual = [row.get("group_id") for row in reviews]
    if len(actual) != len(set(actual)) or set(actual) != advanced:
        raise ValueError("Every advanced group needs one explicit terminal retrieval/assessment disposition")
    for row in reviews:
        if row.get("disposition") not in DISPOSITIONS or not row.get("reason"):
            raise ValueError("Unknown or unexplained terminal assessment disposition")
        if not isinstance(row.get("source_urls"), list) or not row["source_urls"]:
            raise ValueError("Every advanced assessment needs retrieval or existing-evidence source URLs")
        if row["disposition"] == "report_not_retrieved":
            if row.get("primary_sections_assessed") or not row.get("retrieval_attempts"):
                raise ValueError("Unretrieved report cannot claim primary reading and needs actual attempts")
        if row["disposition"] in {"excluded_primary_target", "excluded_primary_reporting", "included_context_only"}:
            if row.get("primary_sections_assessed") is not True or not row.get("source_locators"):
                raise ValueError("A primary exclusion/context disposition needs assessed primary sections and locators")
        if row["disposition"] == "excluded_after_metadata_enrichment" and row.get("primary_sections_assessed"):
            raise ValueError("Metadata enrichment must remain distinct from primary-section assessment")
        if row["disposition"] == "included_strict":
            if any(row.get(key) is not True for key in ("primary_sections_assessed", "target_crypto",
                    "continuous_construct_eligible", "architecture_reference_adequate", "neural_baseline_role_verified",
                    "reporting_metrics_adequate", "reporting_protocol_adequate")):
                raise ValueError("Strict inclusion requires primary evidence for construct, architecture, baseline and reporting")
            if not row.get("comparator_families") or not row.get("independence_key") or not row.get("source_locators"):
                raise ValueError("Strict inclusion requires positively verified autonomous families and identity")
            if any(not row.get(key) for key in ("dataset", "horizon", "architecture_details", "metrics_reported")):
                raise ValueError("Strict inclusion needs extracted dataset, horizon, architecture and metrics")
        if not isinstance(row.get("comparator_families", []), list) or len(row.get("comparator_families", [])) != len(set(row.get("comparator_families", []))):
            raise ValueError("Family votes cannot duplicate variants, assets or runs")
    return {row["group_id"]: row for row in reviews}


def experiment_votes(reviews):
    """A paper and its preprint can supply evidence without two study votes."""
    votes = {}
    for row in reviews:
        if row["disposition"] != "included_strict":
            continue
        key = row["independence_key"]
        votes.setdefault(key, set()).update(row["comparator_families"])
    return votes


def cumulative_votes(previous, strict_reviews):
    old_by_id = {row["id"]:row for row in previous}
    previous_strict = [row for row in previous if row["decision"]["guide31_reporting_sensitivity_eligible"]
        and row["decision"]["continuous_crypto_construct_corpus"] and row["decision"]["baseline_role_verified"]]
    votes = {}
    for row in previous_strict:
        votes.setdefault(row["identity"]["dependence_group"],set()).update(row["independentfamilies"])
    previous_keys = set(votes)
    previous_dois = {normalize_doi(row["doi"]):row["identity"]["dependence_group"] for row in previous_strict if row.get("doi")}
    new_keys = set()
    for row in strict_reviews:
        aliases = {old_by_id[alias]["identity"]["dependence_group"] for alias in row.get("prior_review_ids",[]) if alias in old_by_id}
        matches = aliases & previous_keys
        doi_key = previous_dois.get(normalize_doi(row.get("doi")))
        if doi_key:
            matches.add(doi_key)
        if row["independence_key"] in previous_keys:
            matches.add(row["independence_key"])
        if len(matches)>1:
            raise ValueError("One new report ambiguously joins multiple previous experiments")
        key = next(iter(matches)) if matches else row["independence_key"]
        if key not in previous_keys:
            new_keys.add(key)
        votes.setdefault(key,set()).update(row["comparator_families"])
    count=Counter(family for families in votes.values() for family in families)
    return votes,previous_keys,new_keys,dict(sorted(count.items(),key=lambda x:(-x[1],x[0])))


def summarize(records, fresh, decisions, reviews):
    final = Counter(row["disposition"] for row in reviews)
    rejected = sum(row["decision"] == "exclude_title_abstract" for row in decisions)
    advanced = len(reviews)
    full = [r for r in reviews if r.get("primary_sections_assessed")]
    included = [r for r in reviews if r["disposition"] == "included_strict"]
    votes = experiment_votes(included)
    count = Counter(family for families in votes.values() for family in families)
    if rejected + advanced != len(records) or sum(final.values()) != advanced:
        raise ValueError("Flow does not reconcile")
    enriched=final["excluded_after_metadata_enrichment"]
    not_retrieved=final["report_not_retrieved"]
    if len(full)+enriched+not_retrieved!=advanced:
        raise ValueError("Every advanced group must reconcile as enriched exclusion, primary assessment or nonretrieval")
    return {"stage_id": STAGE, "source_campaign_id": "E1-NATIVE-20261106-COLOMBIA",
        "received_native_occurrences": len(fresh), "screened_groups": len(records),
        "reviewed_groups": len(decisions), "screening_complete": True,
        "export_abstracts_available_and_reviewed":sum(r["abstract_present"] for r in decisions),
        "export_abstracts_missing_and_advanced":sum(not r["abstract_present"] for r in decisions),
        "screening_excluded": rejected, "advanced_for_primary_or_metadata_enrichment": advanced,
        "abstract_enrichment_excluded": final["excluded_after_metadata_enrichment"],
        "total_title_abstract_exclusions_including_enrichment":rejected+enriched,
        "reports_sought_or_previous_primary_reused":advanced-enriched,
        "primary_sections_assessed_groups": len(full),
        "primary_assessments_reused": sum(r.get("document_access") == "existing_primary_review" for r in full),
        "primary_assessments_new": sum(r.get("document_access") != "existing_primary_review" for r in full),
        "reports_not_retrieved_at_cut": final["report_not_retrieved"],
        "terminal_assessment_dispositions": dict(sorted(final.items())),
        "strict_native_included_group_reports": len(included),
        "strict_native_included_experiments": len(votes),
        "strict_native_family_occurrences": dict(sorted(count.items(), key=lambda x: (-x[1], x[0]))),
        "strict_2019_2026_family_occurrences": dict(Counter(f for families in experiment_votes([r for r in included
            if str(r.get("year", "")).isdigit() and 2019 <= int(r["year"]) <= 2026]).values() for f in families)),
        "reviewers": sorted({r["reviewer"] for r in decisions}),
        "screening_method": "Agent-assisted review and documented adjudication; not independent human double screening",
        "provisional_identity_groups_preserved": True, "unexported_IEEE_occurrences_outside_screen": 7,
        "all_database_results_exported_or_screened": False,
        "new_native_search_executions": 0, "benchmark_results_changed": False,
        "models_retrained": 0, "new_predictions_generated": 0,
        "ranking_cannot_retroactively_select_frozen_models": True,
        "scope": "Dated post-hoc screening of 620 provisional groups in received exports. Primary reading, missing reports, context and strict inclusions remain separate. This is not a fresh representative sample or a new test of the frozen benchmark."}


def make_flow(summary):
    s = summary
    dispositions = s["terminal_assessment_dispositions"]
    boxes = ["Resultados nativos: 841 ocurrencias; recibidas 834 (7 IEEE fuera del export)",
        f"620 grupos candidatos provisionales; cribados {s['screened_groups']}",
        f"Exclusión título/resumen: {s['screening_excluded']}; avanzados: {s['advanced_for_primary_or_metadata_enrichment']}",
        f"Excluidos tras enriquecer metadatos: {s['abstract_enrichment_excluded']}; informes buscados/reutilizados: {s['reports_sought_or_previous_primary_reused']}",
        f"Secciones primarias evaluadas: {s['primary_sections_assessed_groups']}; informes no recuperados al corte: {s['reports_not_retrieved_at_cut']}",
        f"Evaluaciones primarias nuevas: {s['primary_assessments_new']}; previas reutilizadas: {s['primary_assessments_reused']}",
        f"Exclusiones primarias: objetivo {dispositions.get('excluded_primary_target',0)}, reporte {dispositions.get('excluded_primary_reporting',0)}; contexto {dispositions.get('included_context_only',0)}",
        f"Inclusión estricta: {s['strict_native_included_group_reports']} informes de {s['strict_native_included_experiments']} experimentos independientes"]
    import html
    height = len(boxes)*112
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}" role="img" aria-label="Flujo conciliado del cribado nativo de E1">', f'<rect width="1120" height="{height}" fill="white"/>']
    for i, label in enumerate(boxes):
        y = 20 + i*112
        content += [f'<rect x="20" y="{y}" width="1080" height="72" rx="8" fill="#eaf2fa" stroke="#234d70"/>',
                    f'<text x="560" y="{y+43}" text-anchor="middle" font-size="19" font-family="sans-serif">{html.escape(label)}</text>']
        if i < len(boxes)-1:
            content.append(f'<path d="M560 {y+74} v30 l-6 -8 m6 8 l6 -8" fill="none" stroke="#234d70" stroke-width="2"/>')
    content.append('</svg>')
    return ('\n'.join(content)+'\n').encode()


def build(source=SOURCE, native=NATIVE, previous=PREVIOUS, *, verify_sources=True):
    source, native, previous = Path(source), Path(native), Path(previous)
    protocol = read(source / "protocol.json")
    completion = read(source / "completion.json")
    if verify_sources:
        for name, expected in protocol["sources_sha256"].items():
            if digest((ROOT/name).read_bytes()) != expected:
                raise ValueError("Frozen native/prior documentary source changed: "+name)
        for name, expected in completion["frozen_benchmark_files_sha256"].items():
            if digest((ROOT/name).read_bytes()) != expected:
                raise ValueError("Frozen benchmark result changed: "+name)
        for name, expected in completion.get("additional_source_files_sha256",{}).items():
            if digest((ROOT/name).read_bytes()) != expected:
                raise ValueError("Original identity ledger changed: "+name)
    records, fresh = native_panel(read_csv(native/"records.csv"),read_csv(native/"occurrences.csv"))
    decisions = []
    for index in (1,2,3):
        decisions.extend(read(source/f"decisions_chunk{index}.json"))
    bydecision = validate_decisions(records, decisions)
    reviews = read(source/"primary_reviews.json")["reviews"]
    byreview = validate_reviews(decisions, reviews)
    summary = summarize(records, fresh, decisions, reviews)
    summary.update(closed_at_utc=completion["closed_at_utc"], protocol_registered_at_utc=protocol["registered_at_utc"])
    title_links=[r for r in read_csv(native/"deduplication.csv")
                 if r["method"]=="normalized_title_provisional" and r["decision"]=="candidate_group_link"]
    fresh_ids={r["occurrence_id"] for r in fresh}
    summary.update(original_global_provisional_title_link_decisions_preserved=len(title_links),
        original_provisional_title_links_between_new_occurrences=sum(r["left_occurrence"] in fresh_ids and r["right_occurrence"] in fresh_ids for r in title_links),
        native_groups_with_provisional_title_links=sum(r["identity_status"]=="provisional_title_link" for r in records),
        identity_limit_detail=completion["identity_limit_detail"])
    old = read(previous)["adjudications"]
    old_by_id = {r["id"]:r for r in old}
    old_by_doi = {normalize_doi(r["doi"]):r for r in old if r.get("doi")}
    decision_rows, crosswalk = [], []
    for row in records:
        key = row["group_id"]; d = bydecision[key]
        aliases = sorted({old_by_doi[normalize_doi(doi)]["id"] for doi in row["dois"].split(";") if normalize_doi(doi) in old_by_doi})
        review = byreview.get(key)
        decision_rows.append({**d, "screening_decision":d["decision"],
            "final_disposition":review["disposition"] if review else "excluded_screening",
            "prior_review_ids":";".join(aliases)})
        crosswalk.append({"group_id":key,"dois":row["dois"],"prior_review_ids":";".join(aliases),
            "historical_group_ids":row["historical_group_ids"],"native_disposition":decision_rows[-1]["final_disposition"],
            "earlier_decisions_preserved":True,"benchmark_models_preserved":True})
    strict = [r for r in reviews if r["disposition"] == "included_strict"]
    cumulative,previous_keys,new_keys,cumulative_count = cumulative_votes(old,strict)
    summary.update(previous_strict_experiments=len(previous_keys), new_strict_experiments_not_in_previous=len(new_keys),
        cumulative_strict_experiments=len(cumulative),
        cumulative_strict_family_occurrences=cumulative_count,
        frozen_experimental_core=["GRU32","LSTM32","MLP16"])
    native_counts=Counter(row["group_id"] for row in fresh)
    record_rows=[{**r,"native_occurrence_count":native_counts[r["group_id"]],
        "historical_occurrence_count":int(r["occurrence_count"])-native_counts[r["group_id"]],
        "native_occurrence_ids":";".join(o["occurrence_id"] for o in fresh if o["group_id"]==r["group_id"])} for r in records]
    artifacts = {"protocol.json":encode(protocol),"summary.json":encode(summary),
        "records.csv":csv_bytes(record_rows[0].keys(),record_rows),
        "decisions.csv":csv_bytes(DECISION_FIELDS,decision_rows),
        "primary_assessments.json":encode({"reviews":reviews}),
        "evidence_matrix.csv":csv_bytes(REVIEW_FIELDS,reviews),
        "corpus_crosswalk.csv":csv_bytes(crosswalk[0].keys(),crosswalk),"flow.svg":make_flow(summary)}
    retrieval = [{"group_id":r["group_id"],"doi":r.get("doi", ""),"attempt_index":i+1,**a}
                 for r in reviews for i,a in enumerate(r.get("retrieval_attempts",[]))]
    fields = sorted(set().union(*(r.keys() for r in retrieval))) if retrieval else ["group_id","doi","attempt_index","url","result"]
    artifacts["retrieval.csv"] = csv_bytes(fields,retrieval)
    quality_fields=("group_id","doi","disposition","primary_sections_assessed","protocol_clarity","leakage_risk",
        "repetitions","statistical_tests","reporting_metrics_adequate","reporting_protocol_adequate","limitations")
    artifacts["quality.csv"]=csv_bytes(quality_fields,reviews)
    frequency_rows=[{"stratum":stratum,"family":family,"count":count,"independent_experiments":denominator}
        for stratum,counts,denominator in [("strict_native",summary["strict_native_family_occurrences"],summary["strict_native_included_experiments"]),
            ("cumulative_strict",summary["cumulative_strict_family_occurrences"],summary["cumulative_strict_experiments"])]
        for family,count in counts.items()]
    artifacts["frequencies.csv"]=csv_bytes(("stratum","family","count","independent_experiments"),frequency_rows)
    source_map = {p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in
        [source/"protocol.json",source/"completion.json",source/"primary_reviews.json",*(source/f"decisions_chunk{i}.json" for i in(1,2,3)),source/"base_bibliography.bib",previous]}
    source_manifest = {"stage_id":STAGE,"native_execution_public_files_sha256":{
        name:digest((native/name).read_bytes()) for name in ("records.csv","occurrences.csv")},
        "sources_sha256":source_map,"raw_exports_sha256":protocol["raw_export_sha256"],
        "additional_original_identity_sources_sha256":completion.get("additional_source_files_sha256",{}),
        "original_benchmark_files_verified_sha256":completion["frozen_benchmark_files_sha256"],
        "bibliography_base_original_path":"book/references.bib",
        "bibliography_base_snapshot_sha256":digest((source/"base_bibliography.bib").read_bytes()),
        "bibliography_base_source_commit":completion["bibliography_base_source_commit"],
        "raw_exports_or_articles_republished":False,"benchmark_results_changed":False,
        "review_payload_scope":"Bibliographic metadata and original critical paraphrases only; no full abstracts or PDFs."}
    artifacts["source_manifest.json"] = encode(source_manifest)
    base_bib = (source/"base_bibliography.bib").read_text()
    if digest(base_bib.encode()) != completion["bibliography_base_sha256"]:
        raise ValueError("The historical70-reference bibliography snapshot changed")
    base_dois = {normalize_doi(x) for x in re.findall(r"\bdoi\s*=\s*[\{\"]([^}\"]+)",base_bib,re.I)}
    bib, new_bib, seen = [],[],set()
    for r in reviews:
        doi = normalize_doi(r.get("doi"))
        if not doi or doi in seen or not r.get("primary_sections_assessed"):
            continue
        seen.add(doi); key="native620_"+hashlib.sha256(doi.encode()).hexdigest()[:10]
        title=str(r.get("title", "")).replace("{", "").replace("}", "")
        authors=r.get("authors", "")
        authors=" and ".join(authors) if isinstance(authors,list) else str(authors).replace(";", " and ")
        if not authors.strip():
            raise ValueError("Assessed primary bibliography needs verified authors: "+doi)
        # The evidence ledger verifies work identity, not a complete journal or
        # proceedings citation for every assessed report. Generic entries avoid
        # asserting an unverified publication type or inventing a venue.
        entry=f'@misc{{{key},\n  title = {{{title}}},\n  author = {{{authors}}},\n  year = {{{r.get("year", "")}}},\n  doi = {{{doi}}},\n  url = {{https://doi.org/{doi}}}\n}}\n'
        bib.append(entry)
        if doi not in base_dois:new_bib.append(entry)
    artifacts["references.bib"] = ('\n'.join(bib)+'\n').encode()
    artifacts["bibliography_additions.bib"] = ('\n'.join(new_bib)+'\n').encode()
    summary["new_book_bibliography_entries"] = len(new_bib)
    artifacts["summary.json"] = encode(summary)
    artifacts["validation.json"] = encode({"status":"PASS","exact_native_group_panel":620,
        "native_occurrences":834,"every_group_has_substantive_decision":True,
        "every_advanced_group_has_terminal_disposition":True,"strict_inclusions_require_primary_evidence":True,
        "family_votes_per_independent_experiment":True,"frozen_benchmark_files_unchanged":True,
        "earlier_search_and_review_files_changed":False,"no_search_training_or_prediction_execution":True})
    artifacts["README.md"] = ("# Cribado de la ronda nativa recibida de E1\n\n"
        "Se revisan 620 grupos candidatos provisionales de 834 ocurrencias exportadas de WoS/Scopus/IEEE. "
        "Los siete resultados IEEE no exportados permanecen fuera del universo recibido. No se ejecutan búsquedas nativas nuevas ni se reconstruyen sus fechas o filtros.\n\n"
        "Cada grupo tiene una decisión y razón en decisions.csv. Los resúmenes disponibles se revisan; los candidatos avanzados conservan disposición terminal de lectura primaria, exclusión documentada o informe no recuperado al corte. "
        "Acceso al primario, calidad, contexto e inclusión estricta se separan en evidence_matrix.csv y primary_assessments.json. "
        "Las revisiones previas enlazadas se reutilizan con identidad, sin presentarlas como lecturas nuevas. "
        "records.csv mantiene los campos originales de la unión y añade conteos nativos/históricos separados; el flujo834→620 corresponde únicamente a los exports nuevos.\n\n"
        "Se conserva el ledger global de diez enlaces provisionales por título (cinco históricos y cinco entre ocurrencias nuevas); ocho grupos del panel nativo conservan esa marca. Los 620 grupos no se presentan como 620 artículos únicos verificados.\n\n"
        "La síntesis es posterior al benchmark cerrado. Ninguna revisión bibliográfica reentrena modelos, selecciona semillas ni cambia sus métricas. "
        "El núcleo experimental permanece GRU32–LSTM32–MLP16. Las frecuencias describen la evidencia evaluada y no su prevalencia mundial.\n\n"
        "La revisión y reconciliación se realizan mediante agentes; no se inventa doble cribado humano ni acuerdo interevaluador. "
        "No se redistribuyen abstracts completos, PDFs, precios, objetivos, pronósticos, pesos o escaladores.\n\n"
        "review_inputs/ contiene copias exactas de los juicios y snapshots bibliográficos necesarios para recompilar. "
        "Los alias canónicos del ZIP8.4 conservan las rutas del compilador; no convierten el cribado en una consulta o lectura automática nueva.\n").encode()
    for name in ("protocol.json","completion.json","primary_reviews.json","decisions_chunk1.json","decisions_chunk2.json","decisions_chunk3.json","base_bibliography.bib"):
        artifacts["review_inputs/"+name]=(source/name).read_bytes()
    artifacts["review_inputs/prior_adjudications.json"]=previous.read_bytes()
    return artifacts, summary


def with_manifest(artifacts):
    names = sorted(artifacts)
    manifest = {"publication_schema":PUBLIC_SCHEMA,"approved_public_files":names,
                "files":{n:{"sha256":digest(artifacts[n]),"bytes":len(artifacts[n])} for n in names}}
    return {**artifacts,"manifest.json":encode(manifest)}


def write_artifacts(artifacts, output, mirror=None):
    artifacts=with_manifest(artifacts)
    for directory in [Path(output), *([Path(mirror)] if mirror else [])]:
        directory.mkdir(parents=True,exist_ok=True)
        unexpected={p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}-set(artifacts)
        if unexpected:raise ValueError("Unexpected public neighbouring files: "+str(sorted(unexpected)))
        for name,payload in artifacts.items():
            target=directory/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    artifacts,summary=build()
    if args.check:
        for folder in(OUTPUT,MIRROR):
            expected=with_manifest(artifacts)
            actual={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
            if actual!=set(expected):raise ValueError("Public allowlist differs: "+str(folder))
            for name,payload in expected.items():
                if (folder/name).read_bytes()!=payload:raise ValueError("Public stage differs: "+str(folder/name))
        print(json.dumps({"status":"PASS","screened_groups":summary["screened_groups"],"benchmark_results_changed":False}))
    else:
        write_artifacts(artifacts,OUTPUT,MIRROR)
        print(json.dumps({k:summary[k] for k in("screened_groups","screening_excluded","advanced_for_primary_or_metadata_enrichment","strict_native_included_experiments","reports_not_retrieved_at_cut","cumulative_strict_family_occurrences")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
