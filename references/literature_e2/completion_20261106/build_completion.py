"""Build a bounded literature assessment; never invent missing searches or papers.

Inputs remain immutable. Private native abstracts and cached author files may be
verified explicitly; none of their contents are redistributed in the derivatives.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "references/literature_e2/completion_20261106"
PUBLIC = ROOT / "book_e2/_static/literature_completion"
INPUTS = (
    "book_e2/_static/literature_fulltext/models.json",
    "book_e2/_static/literature_selection/model_selection.csv",
    "book_e2/_static/literature_remediation/candidate_reassessment.csv",
    "book_e2/_static/literature_remediation/editorial_title_abstract_checks.json",
    "book_e2/_static/literature_scopus/combined_records.csv",
    "references/literature_e2/top10_20261105/primary_source_registry.json",
    "scripts/screen_e2_literature.py",
    "book_e2/_static/literature_selection/representative_studies.csv",
    "references/literature_e1/crypto_extension_20261105/evidence_matrix.json",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def csv_bytes(rows):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    return stream.getvalue().encode()


def abstract_join(row):
    values = {row.get("abstract", ""), row.get("AB", "")}
    for source in row.get("source_records", []):
        raw = source.get("raw_scientific_record", {})
        values.update([raw.get("AB", ""), raw.get("Abstract", "")])
    return "\n".join(sorted({value.strip() for value in values if value and value.strip()}))


def find_primary(model, corpus):
    rules = {
        "MICN": ("exact_title", "micn: multi-scale local and global context modeling for long-term series forecasting"),
        "CryptoMamba-SSM": ("DOI", "10.1109/ojcs.2026.3651226"),
    }
    if model not in rules:
        return None
    kind, value = rules[model]
    matches = [r for r in corpus if (r.get("DOI", "").lower() == value if kind == "DOI"
               else " ".join(r["title"].lower().split()) == value)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one verified native primary identity: " + model)
    return matches[0]


def recovery_assessment(directory):
    """Verify scoped recovery receipts without exporting manuscripts or local paths."""
    directory = Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError("Recovery receipts and bodies must remain outside the public repository")
    names = ("recovery_summary.json", "new_E2_targets_recovery_status.json",
             "inherited_E1_recovered_bodies.json", "target_repositories_recovery.json",
             "author_subtrees_checked.json", "local_PDF_body_scan.json")
    sources = {name: read_json(directory / name) for name in names}
    inherited = sources["inherited_E1_recovered_bodies.json"]
    matrix = {r["study_id"]: r for r in read_json(ROOT / INPUTS[8])["studies"]}
    bodies = []
    for study_id, row in inherited.items():
        body = Path(row["primary_private_body_path"])
        text = Path(row["private_text_path"])
        assert not body.resolve().is_relative_to(ROOT) and not text.resolve().is_relative_to(ROOT)
        assert digest(body) == row["sha256"] == matrix[study_id]["primary_source_sha256"]
        assert digest(text) == row["text_sha256"]
        assert body.stat().st_size == row["bytes"]
        assert row["new_E2_applied_support_body"] is False
        bodies.append({"study_id": study_id, "title": row["title"], "DOI": row["DOI"],
                       "source_url": row["url"], "body_sha256": row["sha256"],
                       "body_bytes": row["bytes"], "body_format": row["body_format"],
                       "private_extracted_text_sha256": row["text_sha256"],
                       "inherited_E1_document_type": row["document_type_in_existing_E1_matrix"],
                       "body_and_text_identity_reverified": True,
                       "already_in_inherited_E1_corpus": True, "new_E2_support_body": False,
                       "new_E2_fulltext_eligibility_decision": None,
                       "scope": "Existing E1 evidence preserved; not seven newly recovered E2 support studies"})
    assert {r["study_id"] for r in bodies} == {"S1", "S2", "S5", "S6", "S7", "S8", "S9"}
    targets = []
    for row in sources["new_E2_targets_recovery_status.json"]:
        assert row["status"] == "full_body_not_recovered_in_this_audit"
        assert row["private_full_body_path"] is None and row["full_body_sha256"] is None
        targets.append({"set": row["set"], "DOI": row["DOI"], "title": row["title"],
                        "status": row["status"], "external_input_required": True,
                        "original_body_identity_verified": False,
                        "body_metrics_or_methodological_quality_adjudicated": False,
                        "code_repository_context_only": row.get("code_repository_context_only"),
                        "limit": row["limit"]})
    assert Counter(r["set"] for r in targets) == {"seven_new_E2_applied_supports": 7,
                                                    "eleven_priority_financial_DOIs": 11}
    assert len({r["DOI"].lower() for r in targets}) == 18
    repositories = []
    for row in sources["target_repositories_recovery.json"]:
        recovered = []
        for item in row["recovered_files"]:
            assert digest(item["path"]) == item["sha256"]
            assert Path(item["path"]).stat().st_size == item["bytes"]
            recovered.append({k: item[k] for k in ("source_path", "url", "sha256", "bytes")})
        repositories.append({"repository": row["repo"], "source_url": row["url"],
                             "commit": row["refInfo"]["currentOid"], "files_verified": recovered,
                             "manuscript_body_recovered": False,
                             "limit": "README identity and source code context do not establish original-paper methods or results"})
    scan = sources["local_PDF_body_scan.json"]
    assert scan["PDF_files"] == 40 and scan["unique_PDFs"] == 35
    assert scan["no_target_full_body_verified_by_manual_identity_check"] is True
    for item in scan["sources"]:
        assert digest(item["PDF_path"]) == item["PDF_sha256"]
        assert digest(item["text_path"]) == item["text_sha256"]
        assert item["pdftotext_exit"] == 0
    non_body_hits = [{"target_DOI": r["target_DOI"], "containing_document_sha256": r["PDF_sha256"],
                      "resolved_identity": r["manual_classification"]} for r in scan["target_matches"]]
    summary = {"status": "scoped_recovery_not_global_unavailability_or_PRISMA_closure",
               "inherited_E1_bodies_reverified": len(bodies), "new_E2_target_bodies_recovered": 0,
               "new_E2_applied_support_targets": 7, "priority_financial_targets": 11,
               "private_PDF_files_scanned": 40, "private_unique_PDFs_scanned": 35,
               "author_repositories_with_pinned_identity_READMEs": len(repositories),
               "native_bibliographic_queries_executed": 0,
               "new_complete_manuscript_readings_claimed": 0,
               "global_review_closed": False, "full_bodies_redistributed": False,
               "global_unavailability_inferred": False,
               "non_body_matches_rejected": non_body_hits,
               "private_receipt_sha256": {name: digest(directory / name) for name in names}}
    return {"inherited_E1_body_reverification.json": json_bytes(bodies),
            "applied_fulltext_recovery_status.json": json_bytes(targets),
            "applied_author_repository_context.json": json_bytes(repositories),
            "recovery_scope.json": json_bytes(summary)}


def build(private_records=None, author_source_root=None, paper_text_root=None, recovery_root=None):
    models = read_json(ROOT / INPUTS[0])["models"]
    historic = {r["model"]: r for r in read_csv(ROOT / INPUTS[1])}
    candidates = read_csv(ROOT / INPUTS[2])
    checks = read_json(ROOT / INPUTS[3])["checks"]
    corpus = read_csv(ROOT / INPUTS[4])
    registry = {r["name"]: r for r in read_json(ROOT / INPUTS[5])["models"]}
    assert len(models) == 10 and len(candidates) == 29 and len(checks) == 17 and len(corpus) == 7833
    source_hashes_before = {name: digest(ROOT / name) for name in INPUTS}
    private = None
    if private_records:
        if Path(private_records).resolve().is_relative_to(ROOT):
            raise ValueError("Native abstracts must remain outside the public repository")
        pp = read_json(private_records)
        private = {r["record_id"]: r for r in pp}
        assert len(private) == 7833 and set(private) == {r["record_id"] for r in corpus}

    admissibility, code_receipts = [], []
    for model in models:
        name = model["model"]; source = registry[name]; pair = model["canonical_metric_pair"]
        assert model["fulltext_verified"] and pair["same_row_pair"]
        assert pair["source_sha256"] == model["source_receipt"]["sha256"]
        assert 2019 <= int(model["publication_year"]) <= 2026
        code = source["code_source"]; code_verified = None
        if author_source_root:
            local = Path(author_source_root) / code["repository"].replace("/", "__") / code["path"]
            code_verified = local.is_file() and digest(local) == code["sha256"]
            assert code_verified, "Pinned author file missing or changed: " + name
        code_receipts.append({"model": name, "author_repository": code["repository"],
                              "commit": code["commit"], "relative_file": code["path"],
                              "sha256": code["sha256"], "offline_cache_hash_verified": code_verified,
                              "license_as_recorded": source.get("license"),
                              "dataset_access_or_license_live_verified": False})
        q = model["quality"]
        quality = {k: v.get("status", "NV") for k, v in q.items() if isinstance(v, dict)}
        admissibility.append({"model": name, "cohort": "bounded_architectural_originals_not_complete_systematic_universe",
            "study_identity": source["doi"] or source.get("paper_title"),
            "source_version": model["source_version"], "source_PDF_sha256": model["source_receipt"]["sha256"],
            "physical_PDF_pages": model["source_receipt"]["pdf_pages"],
            "publication_window_2019_2026": True,
            "primary_deep_forecast_mechanism": True,
            "full_text_previously_read_and_extracted": True,
            "reported_metric_pair_located": True,
            "author_code_identity_available": True,
            "author_code_reverified_offline": code_verified,
            "applicability": "mechanism_transfer_from_other_forecasting_domains; not direct BTC sigma7 effectiveness",
            "D7_transfer_reference": "book_e2/sections/03_literatura.md; registered model-specific architecture_decisions.json",
            "methodological_quality_statuses": quality,
            "year_specific_quartile": "NV; preferred quality attribute, not fabricated or treated as0",
            "bounded_admissibility_decision": "admissible_for_mechanism_review_with_reporting_limitations",
            "final_D7_adapter_verified_by_this_review": False,
            "final_benchmark_or_global_review_completed": False,
            "assessment_basis": "Complete-paper extraction already versioned, pinned author identities; no new manuscript reading claimed"})

    abstract_routes = []
    for check in checks:
        rid = check["record_id"]; verified = None
        if private is not None:
            verified = sha256(abstract_join(private[rid]).encode()).hexdigest() == check["private_abstract_sha256"]
            assert verified, "Native abstract identity changed: " + rid
        category = check["target_class"]
        route = {"explicit_volatility_forecast": "retain_for_direct_volatility_fulltext_assessment",
                 "price_return_or_trading_context": "not_direct_volatility_support_at_abstract_level; retain_only_if_transfer_or_context_justified",
                 "ambiguous_return_volatility_definition": "hold_until_mathematical_target_definition_is_verified"}[category]
        abstract_routes.append({"record_id": rid, "title": check["title"], "DOI": check["DOI"],
            "native_abstract_sha256": check["private_abstract_sha256"], "native_abstract_identity_reverified": verified,
            "target_class": category, "bounded_route": route, "curated_target_and_role_note": check["note"],
            "body_available_and_read_in_this_step": False,
            "architecture_depth_and_standalone_role": "NV until original fulltext",
            "dataset_source_split_scaler_and_leakage": "NV until original fulltext",
            "seeds_HPO_metrics_uncertainty": "NV until original fulltext",
            "year_specific_quartile": "NV", "final_global_eligibility": "open", "independent_human_reviews_claimed": 0})

    scoring = []; identity_updates = []
    weights = {"R": .35, "A": .20, "U": .15, "C": .15, "P": .15}
    normalizer = 608.75
    for row in candidates:
        name = row["model_or_family"]; old = historic.get(name); primary = find_primary(name, corpus)
        components = {k: None for k in weights}; age = None; citations = None; rate = None
        if old:
            components.update(R=float(old["relevance_component"]), A=float(old["verified_code_component"]),
                              U=float(old["recency_component"]))
            if old["normalized_citation_component"] not in ("", "unknown", "not_available"):
                try: components["C"] = float(old["normalized_citation_component"])
                except ValueError: pass
        if primary:
            age = int(primary["earliest_known_publication_year"] or primary["year"])
            citations = int(primary["Scopus_Cited_by"]); rate = citations / (2026 - age + 1)
            components.update(U=max(0, min(1, (age - 2019) / 7)), C=max(0, min(1, rate / normalizer)))
            identity_updates.append({"model": name, "native_record_id": primary["record_id"], "title": primary["title"],
                "DOI": primary["DOI"], "Scopus_EIDs": primary["EIDs"], "year": primary["year"],
                "citation_age_year": age, "Scopus_Cited_by": citations, "Scopus_citations_per_calendar_year": rate,
                "evidence": "Exact original title or DOI in received native metadata; not an applied variant or raw mention frequency",
                "fulltext_recovered_in_this_step": False, "code_or_year_quartile_verified_in_this_step": False})
        low = sum(weights[k] * (0 if v is None else v) for k, v in components.items()) * 100
        high = sum(weights[k] * (1 if v is None else v) for k, v in components.items()) * 100
        role = ("additional_E1_reference_not_new_top10_cup" if name in {"GRU", "LSTM", "TCN"}
                else "linear_contextual_reference_not_deep_cup" if name == "DLinear"
                else "current_bounded_working_cohort" if old else "alternative_requires_identity_fulltext_and_contract_verification")
        scoring.append({"model_or_family": name, "family": row["family"], "role": role,
            **{k+"_component": v for k, v in components.items()},
            "unknown_components": [k for k, v in components.items() if v is None],
            "score_lower_algebraic_not_unknown_equals0": low,
            "score_upper_algebraic_not_unknown_equals1": high,
            "family_rule": "at least4families in10; hard diversity constraint, not a mention-frequency bonus",
            "native_primary_identity_addition": primary["record_id"] if primary else None,
            "final_rank_or_global_optimality_claimed": False,
            "primary_body_or_code_still_needed": not bool(old),
            "notes": "Intervals expose missing evidence; bounds0/1 are not observed values and cannot be used as a final rank"})

    dataset_candidates = []
    if private is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("e2_existing_screening", ROOT / INPUTS[6])
        screen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(screen)
        public_index = {r["record_id"]: r for r in corpus}
        source_pattern = re.compile(r"\b(?:data|observations|prices|OHLCV|series|dataset)\s+(?:\w+\s+){0,8}(?:from|via)\s+(?:the\s+)?Binance(?!\s+Coin)|\bBinance(?:'s)?\s+(?:exchange|API|platform|dataset|data\s+(?:portal|service))\b|\b(?:collected|obtained|downloaded)\s+(?:\w+\s+){0,8}(?:from|via)\s+(?:the\s+)?Binance(?!\s+Coin)", re.I)
        for rid, native in private.items():
            text = abstract_join(native)
            if not re.search(r"\bBinance\b", text, re.I):
                continue
            features = screen.detect_features(native["title"], text)
            if not all(features[k] for k in ("crypto", "volatility", "forecast", "deep")):
                continue
            bare = re.sub(r"Binance\s+Coin(?:\s*\(BNB\))?", "", text, flags=re.I)
            if not re.search(r"\bBinance\b", bare, re.I):
                relation = "Binance_Coin_asset_only_not_exchange_source_evidence"
            elif source_pattern.search(text):
                relation = "abstract_explicitly_connects_Binance_to_data_or_exchange; verify_body"
            else:
                relation = "Binance_mention_not_enough_to_identify_data_source"
            p = public_index[rid]
            dataset_candidates.append({"record_id": rid, "title": native["title"], "DOI": p["DOI"],
                "source_title": p.get("source_title", ""), "year": p["year"],
                "native_abstract_sha256": sha256(text.encode()).hexdigest(),
                "source_mention_relation": relation,
                "fulltext_source_and_protocol_verified": False, "year_specific_Q1_verified": False,
                "exact_snapshot_identity_required_by_guide": False,
                "next_step": "Verify actual exchange/source in the body and obtain venue quartile with year; coin-name matches do not establish Binance data use"})

    origin_trace = []
    if paper_text_root:
        sources = {r["model"]: r for r in models}
        origin_definitions = [
            ("attention_segments", "Crossformer", ["partitioning images into patches", "An image is worth 16x16 words", "Swin transformer: Hierarchical"],
             [{"title": "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale", "authors": "Dosovitskiy et al.", "venue": "ICLR", "year": 2021, "DOI": None},
              {"title": "Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows", "authors": "Liu et al.", "venue": "ICCV", "year": 2021, "DOI": None}],
             "Vision/image patches", "Partition an observed context into segments and model relationships between time segments and variables; different axes retain their meaning.",
             "D7 needs valid segment lengths and no future observations; image successes do not certify Bitcoin gains or make temporal and feature axes interchangeable."),
            ("token_feature_mixing", "TSMixer", ["Mlp-mixer: An all-mlp architecture for vision"],
             [{"title": "MLP-Mixer: An All-MLP Architecture for Vision", "authors": "Tolstikhin et al.", "venue": "NeurIPS34, pp24261–24272", "year": 2021, "DOI": None}],
             "Vision/token and channel mixing", "Separate the mixing of observed positions and channels; the D7×5 tensor has two explicit axes.",
             "Seven days and five channels are a small domain; mixing and extra readouts must be assessed by ablation and equal tuning, not assumed efficient or superior."),
            ("input_selective_state", "S-Mamba", ["Mamba: Linear-time sequence modeling with", "selective state spaces. arXiv preprint arXiv:2312.00752"],
             [{"title": "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", "authors": "Gu and Dao", "venue": "arXiv2312.00752, as cited in the supplied S-Mamba preprint", "year": 2023, "identifier": "arXiv:2312.00752", "DOI": None}],
             "Language/sequence modeling", "Input-dependent state selection can be adapted over tokens formed from already observed variable histories.",
             "The retrieved S-Mamba scans variable tokens; long language sequences are not D7. Adding target lags would change the contract and is not licensed by this transfer rationale."),
        ]
        for mechanism, source_name, needles, original_sources, modality, property_text, limitation in origin_definitions:
            path = Path(paper_text_root) / (source_name + "_pages.txt")
            text = path.read_text(); chunks = re.split(r"\[\[PDF_PAGE:(\d+)\]\]", text)
            pages = {}
            for i in range(1, len(chunks), 2):
                pages[int(chunks[i])] = chunks[i+1]
            assert len(pages) == sources[source_name]["source_receipt"]["pdf_pages"]
            locators = {needle: [page for page, value in pages.items() if needle.casefold() in value.casefold()] for needle in needles}
            assert all(locators.values()), "Origin-reference locator missing: " + source_name
            origin_trace.append({"mechanism": mechanism, "origin_modality": modality,
                "identified_primary_sources": original_sources,
                "identification_basis": "Bibliography and/or motivation in a previously read complete forecasting original; not a new native database search",
                "forecast_source": source_name, "forecast_source_version": sources[source_name]["source_version"],
                "forecast_source_PDF_sha256": sources[source_name]["source_receipt"]["sha256"],
                "forecast_source_PDF_page_locators": locators,
                "private_extracted_text_sha256": digest(path),
                "shared_structural_property": property_text, "D7_limitations": limitation,
                "origin_full_manuscripts_read_in_this_step": 0,
                "new_native_transversal_queries": 0, "Bitcoin_improvement_claimed": False,
                "E3_architecture_selected": False})

    assert sum(r["physical_PDF_pages"] for r in admissibility) == 215
    assert Counter(r["target_class"] for r in abstract_routes) == {
        "explicit_volatility_forecast": 10, "price_return_or_trading_context": 6,
        "ambiguous_return_volatility_definition": 1}
    assert len(identity_updates) == 2
    assert all(0 <= r["score_lower_algebraic_not_unknown_equals0"] <= r["score_upper_algebraic_not_unknown_equals1"] <= 100.000001 for r in scoring)
    assert {name: digest(ROOT / name) for name in INPUTS} == source_hashes_before
    summary = {"status": "bounded_completion_assessment_not_global_PRISMA_closure", "date_America_Bogota": "2026-11-06",
        "architectural_originals_bounded_admissibility_assessed": 10, "prior_complete_PDF_pages": 215,
        "original_author_code_cache_hashes_reverified": sum(r["offline_cache_hash_verified"] is True for r in code_receipts),
        "native_abstract_routes_reverified": sum(r["native_abstract_identity_reverified"] is True for r in abstract_routes),
        "abstract_route_counts": dict(Counter(r["target_class"] for r in abstract_routes)),
        "candidate_score_interval_rows": 29, "additional_native_primary_identities_verified": [r["model"] for r in identity_updates],
        "dataset_source_keyword_candidates": len(dataset_candidates),
        "dataset_source_candidate_types": dict(Counter(r["source_mention_relation"] for r in dataset_candidates)),
        "origin_mechanism_families_traced_to_complete_paper_references": len(origin_trace),
        "native_queries_executed": 0, "new_manuscripts_read": 0, "new_models_trained": 0,
        "independent_human_double_reviews_claimed": 0, "global_review_closed": False, "final_top10_rank_certified": False,
        "Scopus_records_still_unreceived": 374, "WoS_query_totals_and_ranges_confirmed": False,
        "native_transverse_queries_still_required": True, "year_specific_quartiles_still_unverified": True,
        "global_reproducibility_code_and_data_denominator_closed": False,
        "operational_cohort_note": "The ten originals are individually admissible as forecasting-mechanism sources under explicit limitations. This does not prove exhaustive selection or certify final D7 adapters."}
    contents = {"architectural_originals_eligibility.csv": csv_bytes(admissibility),
                "author_code_verification.json": json_bytes(code_receipts),
                "financial_abstract_routes_17.csv": csv_bytes(abstract_routes),
                "candidate_score_intervals_29.csv": csv_bytes(scoring),
                "additional_native_primary_identities.json": json_bytes(identity_updates),
                "summary.json": json_bytes(summary)}
    contents["score_protocol.json"] = json_bytes({
        "status": "post_search_bounded_score_intervals_not_final_rank",
        "formula": "100*(0.35R+0.20A+0.15U+0.15C+0.15P)",
        "weights": weights, "citation_rate_normalizer": normalizer,
        "normalizer_scope": "Largest verified original-paper Scopus citations/year in the current received working cohort: PatchTST2435/4=608.75; incomplete coverage disclosed",
        "relevance_R": "Historical current-cohort evidence retained; alternative target/relevance remains unknown until source verification",
        "implementation_A": "Pinned original author-code identity, not an unverified repository mention",
        "recency_U": "clip((earliest verified publication year-2019)/7,0,1)",
        "citation_C": "clip(Scopus citations/(2026-citation_age_year+1)/608.75,0,1); WoS andScopus counts never summed",
        "performance_P": "Unknown when datasets, scale, horizon or target do not support comparable performance; not inferred from paper ranking or model mentions",
        "unknown_value_policy": "Unknowns remain null/empty. Bounds replace unknowns by0/1 only algebraically, never as observed measurements",
        "family_policy": "At least4distinct families within10; E1references and nondeep comparators do not count as new deep models",
        "selection_limit": "Admissibility and a reproducible interval do not identify a unique optimal Top10. Unread alternatives and missing records keep global selection open"})
    supports = read_csv(ROOT / INPUTS[7])
    support_bodies_pending = [r for r in supports if r.get("doi", "").lower() != "10.1016/j.neucom.2024.129178"]
    assert len(support_bodies_pending) == 7
    contents["manual_inputs_required.json"] = json_bytes({
        "scope": "External evidence missing from the available files; no credentials requested",
        "Scopus": {"original_total": 7874, "received": 7500, "unreceived": 374,
            "required": "Original exact query, filters, execution date, sort order and native records with EID/DOI/title/abstract/keywords/year/source/citations",
            "safe_export_rule": "Export missing7501–7874 only if the supplied7500 are confirmed first7500 in the unchanged original ordering; otherwise export the full original7874 and reconcile byEID/DOI",
            "reuploads_already_received_are_identical": True},
        "WoS": {"required": "Map each savedrecs file to Q1 orQ2; provide each original query/date/filters/result total and exported record ranges. 302/500/500 file sizes do not establish Q2 total."},
        "transversal": {"required": "Execute at least3native mechanism+othermodality+transfer/adaptation chains with exact query, date, total and export; local102hits do not replace them",
            "defined_queries": "book_e2/_static/literature_remediation/local_transversal_searches.json"},
        "fulltexts": {"seven_applied_support_bodies_not_yet_extracted": support_bodies_pending,
            "priority_direct_or_ambiguous_volatility_DOIs": [r["DOI"] for r in checks if r["target_class"] != "price_return_or_trading_context"],
            "inherited_E1_bodies_are_not_these_seven_new_E2_supports": True,
            "note": "Seven curated support bodies are not the total retained fulltexts. Price/return-context abstracts do not establish a volatility target. A scoped recovery status, if supplied, distinguishes inherited E1 bodies, new E2 targets and short author-repository READMEs."},
        "venue_quality": {"required": "Verify journal quartile/impact with venue/ISSN and reference year. If only a latest available report exists, disclose its year separately. Conference quality is documented by actual venue/publication, not invented journal Q1."},
        "dataset_prior_use": {"required": "Verify actual Binance exchange/source in an academic body and its year-specific venue evidence; mentioning BinanceCoin does not establish use of Binance data. An identical snapshot/hash is not required by guide§1."}})
    if (REF / "README.md").exists():
        contents["README.md"] = (REF / "README.md").read_bytes()
    if dataset_candidates:
        contents["dataset_source_candidates.csv"] = csv_bytes(dataset_candidates)
    if origin_trace:
        contents["origin_mechanism_trace.json"] = json_bytes(origin_trace)
    if recovery_root:
        contents.update(recovery_assessment(recovery_root))
    contents["closure_checklist.json"] = json_bytes({
        "scope": "Bibliographic closure conditions; completion of implementation or benchmark does not close these items",
        "global_eligible_study_count": None, "global_included_study_count": None,
        "rows": [
            {"item": "received_corpus_deduplication_and_first_pass", "status": "completed_for_received_subset",
             "evidence": "7833groups=4323provisionalexclusions+2436retained+1074manual; 602financialpriority",
             "remaining": "Validate recommendations editorially; received coverage remains partial"},
            {"item": "Scopus_original_coverage", "status": "external_records_missing",
             "evidence": "7500native records received of7874originally displayed; reupload identical",
             "remaining": "Recover374records using unchanged original order or reconcile a complete re-export byEID/DOI"},
            {"item": "WoS_query_history", "status": "external_confirmation_missing",
             "evidence": "302/500/500 native occurrences preserve file identity, not original query totals or ranges",
             "remaining": "Confirm exact file-query mapping, execution date, filters, result totals and ranges"},
            {"item": "architectural_originals", "status": "bounded_admissibility_completed",
             "evidence": "10complete originals/215physicalpages, author-core hashes and source-located metrics",
             "remaining": "Do not infer exhaustive Top10selection, BTC effectiveness or final adapter fidelity from this assessment"},
            {"item": "editorial_eligibility_and_PRISMA", "status": "open",
             "evidence": "First-pass recommendations and focused abstract checks are not final exclusions or fulltext inclusions",
             "remaining": "Adjudicate eligible text bodies, record per-stage exclusion reasons and reconcile study-level denominators; account separately for reports not retrieved"},
            {"item": "applied_support_and_financial_bodies", "status": "bodies_pending",
             "evidence": "7new applied support targets and11direct/ambiguous financial targets; inherited E1bodies are distinct",
             "remaining": "Obtain original bodies, verify actual target/standalone models/splits/scaling/HPO/seeds/metrics; README or citation matches do not suffice"},
            {"item": "transversal_mechanism_search", "status": "native_searches_pending",
             "evidence": "3local queries/102unique groups plus source-located origin references",
             "remaining": "Execute at least3native mechanism+othermodality+transfer/adaptation searches and inspect the relevant complete origin manuscripts"},
            {"item": "selection_scores_and_exclusions", "status": "intervals_only",
             "evidence": "29candidate/family intervals expose unknown components; no unique final ranking established",
             "remaining": "Verify candidate primary identities/bodies and comparable components, then justify constrained diversity and exclusion of highly cited alternatives"},
            {"item": "quality_prior_data_use_and_reproducibility", "status": "year_specific_and_global_denominators_open",
             "evidence": "Venue indexing is not a year-specific quartile; BinanceCoin mention is not exchange-data evidence",
             "remaining": "Verify actual source usage and venue year; compute code/data public proportions only over an adjudicated study denominator"}
        ],
        "unknown_is_not_zero": True, "new_native_queries_claimed": 0,
        "global_PRISMA_closed": False, "top10_optimality_certified": False})
    contents["provenance.json"] = json_bytes({"generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": digest(__file__),
        "input_sha256": source_hashes_before,
        "private_native_corpus_sha256": digest(private_records) if private_records else None,
        "outputs_sha256": {name: sha256(payload).hexdigest() for name, payload in contents.items()},
        "private_abstracts_or_author_source_contents_exported": False,
        "preexisting_extraction_or_registered_configs_changed": False})
    for directory in (REF, PUBLIC):
        directory.mkdir(parents=True, exist_ok=True)
        for name, payload in contents.items():
            (directory / name).write_bytes(payload)
    return summary


def check_public():
    provenance = read_json(REF / "provenance.json")
    assert provenance["generator_sha256"] == digest(__file__)
    assert provenance["input_sha256"] == {name: digest(ROOT / name) for name in INPUTS}
    for name, expected in provenance["outputs_sha256"].items():
        assert digest(REF / name) == expected and (REF / name).read_bytes() == (PUBLIC / name).read_bytes(), name
    summary = read_json(REF / "summary.json")
    assert summary["Scopus_records_still_unreceived"] == 374
    assert not summary["global_review_closed"] and not summary["final_top10_rank_certified"]
    if "recovery_scope.json" in provenance["outputs_sha256"]:
        recovery = read_json(REF / "recovery_scope.json")
        assert recovery["inherited_E1_bodies_reverified"] == 7
        assert recovery["new_E2_target_bodies_recovered"] == 0
        assert not recovery["global_review_closed"] and not recovery["global_unavailability_inferred"]
        assert len(read_json(REF / "applied_fulltext_recovery_status.json")) == 18
    return {"public_derivative_hashes_verified": len(provenance["outputs_sha256"]),
            "private_inputs_required_for_this_check": False,
            "global_review_closed": False, "new_model_fits": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-records", type=Path)
    parser.add_argument("--author-source-root", type=Path)
    parser.add_argument("--paper-text-root", type=Path)
    parser.add_argument("--recovery-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check_public() if args.check else build(args.private_records, args.author_source_root, args.paper_text_root, args.recovery_root), ensure_ascii=False))
