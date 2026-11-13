"""Reconcile a bounded literature corpus without fabricating global eligibility.

This is a documentary synthesis. It never searches, trains, changes old receipts,
or turns missing observations into measurements. Run with --check after building.
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
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "book_e2/_static/e2_literature_closure_20261113"
EVIDENCE = ROOT / "references/literature_e2"
DATE = "2026-11-13"
NATIVE_ATTEMPTS = [
    ("vision_residual", 'all:"residual network" AND all:vision AND (all:transfer OR all:adaptation)', "2026-11-13T18:52:37.878592+00:00"),
    ("vision_patches", 'all:attention AND all:"image patches" AND (all:transfer OR all:adaptation)', "2026-11-13T18:52:37.899633+00:00"),
    ("industrial_recurrent", 'all:"LSTM autoencoder" AND all:industrial AND (all:transfer OR all:adaptation)', "2026-11-13T18:52:37.906126+00:00"),
]
WEIGHTS = {
    "reference": {"C": .15, "U": .15, "P": .25, "A": .25, "F": .20},
    "equal": {"C": .20, "U": .20, "P": .20, "A": .20, "F": .20},
    "implementation": {"C": .10, "U": .10, "P": .20, "A": .40, "F": .20},
    "performance": {"C": .10, "U": .10, "P": .40, "A": .20, "F": .20},
}
SUPPLEMENTS = {
    "FCN-1D": {
        "study_id": "10.48550/arXiv.1611.06455", "year": 2016,
        "title": "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline",
        "authors": "Zhiguang Wang; Weizhong Yan; Tim Oates",
        "version": "1611.06455v4", "url": "https://arxiv.org/html/1611.06455v4",
        "read_level": "complete_primary_main_HTML_20261113_including_tables_and_Wang_appendix_not_PDF_visual_verification",
        "dataset": "UCR44; selected example Adiac", "task": "univariate series classification",
        "metric_1": "test error rate", "value_1": .143,
        "metric_2": "ResNet test error rate, same dataset", "value_2": .174,
        "metric_locator": "Table I, Adiac row; III.A-B",
        "quality": {"split": "reported", "train_only_scaler": "reported", "seed_ids": "NR", "repetitions": "NR", "variance_selected_pair": "NR", "formal_tests": "reported_other_scope", "equal_hpo_budget": "NR"},
        "critical_limit": "Minimum-training-loss checkpoint; proposed baselines exclude HPO/CV, without proving equal effort against imported comparators. Wilcoxon rank-sum and paired PCE tests concern datasets, not seed uncertainty.",
    },
    "ResNet-1D": {
        "study_id": "10.48550/arXiv.1611.06455", "year": 2016,
        "title": "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline",
        "authors": "Zhiguang Wang; Weizhong Yan; Tim Oates",
        "version": "1611.06455v4", "url": "https://arxiv.org/html/1611.06455v4",
        "read_level": "complete_primary_main_HTML_20261113_including_tables_and_Wang_appendix_not_PDF_visual_verification",
        "dataset": "UCR44; selected example Adiac", "task": "univariate series classification",
        "metric_1": "test error rate", "value_1": .174,
        "metric_2": "FCN test error rate, same dataset", "value_2": .143,
        "metric_locator": "Table I, Adiac row; III.A-B",
        "quality": {"split": "reported", "train_only_scaler": "reported", "seed_ids": "NR", "repetitions": "NR", "variance_selected_pair": "NR", "formal_tests": "reported_other_scope", "equal_hpo_budget": "NR"},
        "critical_limit": "One shared source study with FCN; two systems are not two independent studies.",
    },
    "Inception-1D": {
        "study_id": "10.48550/arXiv.1909.04939", "year": 2019,
        "title": "InceptionTime: Finding AlexNet for Time Series Classification",
        "authors": "Hassan Ismail Fawaz et al.", "version": "1909.04939v3",
        "url": "https://arxiv.org/html/1909.04939v3",
        "read_level": "complete_primary_main_HTML_20261113_including_tables_and_captions_not_PDF_visual_verification",
        "dataset": "UCR85", "task": "series classification; reported result is five-network ensemble",
        "metric_1": "wins/ties/losses versus ResNet(5)", "value_1": "54/8/23",
        "metric_2": "published p-value", "value_2": "<0.01",
        "metric_locator": "Sections4-5; Figure10; Section6 defaults; Section3.2 ensemble",
        "quality": {"split": "reported", "train_only_scaler": "NR", "seed_ids": "NR", "repetitions": "partial", "variance_selected_pair": "NR", "formal_tests": "reported_other_scope", "equal_hpo_budget": "NR"},
        "critical_limit": "Median over multiple runs is reported without exact count. Five members per ensemble are not five repeated ensembles. Friedman/Wilcoxon-Holm concern classification ensembles; current single-network regression inherits no ensemble score.",
    },
    "LSTM-autoencoder con predictor": {
        "study_id": "10.1038/s41598-019-55320-6", "year": 2019,
        "title": "Unsupervised Pre-training of a Deep LSTM-based Stacked Autoencoder for Multivariate Time Series Forecasting Problems",
        "authors": "Alaa Sagheer; Mostafa Kotb", "version": "Scientific Reports2019; PMC full main text accessed20261113",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6911101/",
        "read_level": "complete_primary_main_web_text_20261113_no_PDF_receipt_claimed",
        "dataset": "Beijing PM2.5; 30660/4380/8760 reported train/validation/test observations",
        "task": "next-step scalar forecasting with historical target information; lag25",
        "metric_1": "RMSE", "value_1": 24.041, "metric_2": "MAE", "value_2": 12.060,
        "metric_locator": "Table2, two-layer LSTM-SAE; Methodology; Datasets; Parameters selection; Discussion",
        "quality": {"split": "reported", "train_only_scaler": "NV", "seed_ids": "NR", "repetitions": "NR", "variance_selected_pair": "NR", "formal_tests": "NR", "equal_hpo_budget": "NR"},
        "critical_limit": "Two-layer DLSTM RMSE23.750 is lower; SAE is not best for every metric. Hyperopt uses validation RMSE but equal budgets are unreported. Chronological partition is not established by counts.",
    },
}


def read_json(path: Path):
    return json.loads(path.read_text())


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def csv_bytes(rows: list[dict]) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else "" if v is None else v for k, v in row.items()})
    return out.getvalue().encode()


def score_interval(components: dict, weights: dict) -> tuple[float, float]:
    if set(components) != set(weights) or abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError("five declared components and normalized weights required")
    known = 0.0
    unknown = 0.0
    for key, weight in weights.items():
        value = components[key]
        if value is None:
            unknown += weight
        elif not 0 <= value <= 1:
            raise ValueError("observed components must be in [0,1]")
        else:
            known += weight * value
    return round(100 * known, 8), round(100 * (known + unknown), 8)


def rank_bounds(intervals: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Conservative ranks: ties and overlapping intervals remain unresolved."""
    return [(1 + sum(a > hi for j, (a, _) in enumerate(intervals) if i != j),
             len(intervals) - sum(b < lo for j, (_, b) in enumerate(intervals) if i != j))
            for i, (lo, hi) in enumerate(intervals)]


def identity_year(study_id: str | None, fallback: int) -> int:
    """Use first arXiv submission year consistently, else source publication year."""
    match = re.search(r"arXiv\.(\d{2})\d{2}\.", study_id or "", flags=re.I)
    return 2000 + int(match[1]) if match else fallback


def build() -> dict[str, bytes]:
    source_path = EVIDENCE / "fulltext_20261105/models.json"
    historical = read_json(source_path)
    old = {m["model"]: m for m in historical["models"]}
    catalogue_path = ROOT / "experiments/e2_published_architectures_20261112/architecture_policy.json"
    policy = read_json(catalogue_path)
    current = [m for m in policy["models"] if m["role"] == "Top10"]
    tide_path = ROOT / "experiments/e2_tide_published_r0_20261112/protocol_extension.json"
    tide = read_json(tide_path)
    catalogue = read_json(ROOT / "experiments/e2_published_architectures_20261112/catalogue.json")
    families = {m["model_id"]: m["family"] for m in catalogue["systems"] if m["role"] == "Top10"}
    if len(current) != 10 or set(families) != {m["model_id"] for m in current}:
        raise ValueError("current Top10 identity mismatch")
    citation_path = EVIDENCE / "scopus_20261105/original_model_records.csv"
    with citation_path.open() as fh:
        citations = {r["model"]: r for r in csv.DictReader(fh)}
    registry = {r["name"]: r for r in read_json(EVIDENCE / "top10_20261105/primary_source_registry.json")["models"]}
    citation_years = {name: identity_year(registry[name]["doi"], int(old[name]["publication_year"])) for name in citations}
    rates = [int(r["Cited_by_Scopus"]) / (2026 - citation_years[name] + 1) for name, r in citations.items() if r["Cited_by_Scopus"]]
    normalizer = max(rates)
    family_counts = Counter(families.values())
    extraction = []
    score_rows = []
    for body in current:
        name = body["model_id"]
        body_policy_source = str(catalogue_path.relative_to(ROOT))
        if name == "TiDE":
            body = {**body, "recipe_id": tide["new_recipe_id"], "nas_required": False, "fixed_body": True, "body_definition": tide["body_definition"], "adaptations": tide["adaptations"]}
            body_policy_source = str(tide_path.relative_to(ROOT))
        q = old.get(name)
        primary = [p for p in body["primary_evidence"] if p["kind"] == "paper"]
        code = [p for p in body["primary_evidence"] if p["kind"] == "author_code"]
        if not primary or not code:
            raise ValueError(f"missing current primary/code identity for {name}")
        if q:
            pair = q["canonical_metric_pair"]
            study_id = registry[name]["doi"]
            year = identity_year(study_id, int(q["publication_year"]))
            item = {"model": name, "study_id": study_id, "title": q["title"], "authors": q["authors"], "source_version": q["source_version"], "source_url": pair["source_url"], "source_PDF_sha256": q["source_receipt"]["sha256"], "read_level": "prior_complete_PDF_and_appendices_extraction", "dataset": pair["dataset"], "task": pair["task"], "metric_1": "MSE", "value_1": pair["MSE"], "metric_2": "MAE", "value_2": pair["MAE"], "metric_locator": pair["locators"], "quality": q["quality"], "critical_limit": "External forecast result; different task and information set from BTC sigma7."}
        else:
            s = SUPPLEMENTS[name]
            year = s["year"]
            item = {"model": name, "study_id": s["study_id"], "title": s["title"], "authors": s["authors"], "source_version": s["version"], "source_url": s["url"], "source_PDF_sha256": None, "read_level": s["read_level"], "dataset": s["dataset"], "task": s["task"], "metric_1": s["metric_1"], "value_1": s["value_1"], "metric_2": s["metric_2"], "value_2": s["value_2"], "metric_locator": s["metric_locator"], "quality": s["quality"], "critical_limit": s["critical_limit"]}
        item.update(family=families[name], original_year=year, current_recipe=body["recipe_id"], recipe_policy_source=body_policy_source, published_body=body["fixed_body"], NAS_required=body["nas_required"], architecture=body["body_definition"], original_efficiency=q["efficiency"] if q else None, author_code=code, adaptation=body["adaptations"], direct_sigma7_result_verified=False, performance_component=None, benchmark_data_download_verified=None)
        extraction.append(item)
        cit = citations.get(name, {})
        count = int(cit["Cited_by_Scopus"]) if cit.get("Cited_by_Scopus") else None
        age_year = citation_years[name] if count is not None else None
        rate = count / (2026 - age_year + 1) if count is not None else None
        components = {"C": min(rate / normalizer, 1) if rate is not None else None,
                      "U": max(0, min((year - 2019) / 7, 1)), "P": None,
                      "A": 1., "F": 1 / family_counts[families[name]]}
        score_rows.append({"model": name, "family": families[name], "source_study_id": item["study_id"], "citation_snapshot": "received_Scopus_export_20261105_not_live_20261113", "Scopus_citations": count, "citation_age_year": age_year, "citation_rate": rate, **components, "unknown_components": [k for k, v in components.items() if v is None], "selection_status": "authorized_current_catalogue_not_score_selected"})
    sensitivity = []
    for scenario, weights in WEIGHTS.items():
        intervals = [score_interval({k: r[k] for k in weights}, weights) for r in score_rows]
        ranks = rank_bounds(intervals)
        for row, (lo, hi), (best, worst) in zip(score_rows, intervals, ranks):
            sensitivity.append({"scenario": scenario, "model": row["model"], "score_lower_algebraic": lo, "score_upper_algebraic": hi, "best_possible_rank_bound": best, "worst_possible_rank_bound": worst, "unique_rank_identified": best == worst, "unknowns_are_observed_zero": False})
            if scenario == "reference":
                row.update(score_lower_algebraic=lo, score_upper_algebraic=hi, best_possible_rank_bound=best, worst_possible_rank_bound=worst)
    fields = ["split", "scaling", "train_only_scaler", "repetitions", "seed_ids", "variance_selected_pair", "formal_tests", "baseline_effort", "equal_hpo_budget"]
    quality_counts = [{"cohort": "historical_architectural10_complete_PDFs_not_current10", "field": f, "denominator": len(old), "status_counts": dict(Counter(m["quality"][f]["status"] for m in old.values()))} for f in fields]
    historical_rows = [{"study": m["model"], "source_version": m["source_version"], "study_identity": m["source_receipt"]["source_url"], "PDF_sha256": m["source_receipt"]["sha256"], "physical_pages": m["source_receipt"]["pdf_pages"], "cohort_entry": "directed_primary_source_retrieval_not_global_screening", "decision": "included_for_mechanism_synthesis_with_reporting_limits", "current_Top10": m["model"] in families, "quality_pointer": f"references/literature_e2/fulltext_20261105/models.json#/models/{i}/quality", "same_row_metric": m["canonical_metric_pair"], "quality": m["quality"]} for i, m in enumerate(historical["models"])]
    flow = {"cohort": "historical_architectural10_fixed_source_identities", "selection_method": "directed retrieval of named originals; retrospective bounded eligibility assessment; not probability sampling", "source_identities_presented": 10, "distinct_source_identities": 10, "duplicates_within_this_fixed_cohort": 0, "complete_PDFs_received": 10, "physical_pages": sum(m["source_receipt"]["pdf_pages"] for m in old.values()), "assessed_for_mechanism_synthesis": 10, "included_for_mechanism_synthesis": 10, "excluded_from_this_mechanism_cohort": 0, "current_Top10_overlap_systems": sorted(set(old) & set(families)), "current_Top10_systems": len(current), "current_Top10_distinct_primary_studies": len({r["study_id"] for r in extraction}), "current_primary_studies_with_prior_complete_PDF_extraction": len(set(old) & set(families)), "current_additional_complete_main_web_text": 3, "current_additional_targeted_primary_studies": 0, "global_eligible_studies": None, "global_included_studies": None, "global_review_complete": False, "global_unreceived_historical_Scopus_records": 374, "eligibility_rule": "Primary deep forecasting mechanism; complete original PDF available; located metric and original code identity. Reporting limitations do not demonstrate model invalidity. This rule is local and does not retroactively create a global screen."}
    gap_rows = [{"model": r["model"], "study_id": r["study_id"], "source_task": r["task"], "source_dataset": r["dataset"], "source_locator": r["metric_locator"], "matches_BTC_sigma7_D7x5_h1_information_set": False, "reason": "classification_of_complete_series" if r["model"] in {"FCN-1D", "ResNet-1D", "Inception-1D"} else "forecast_of_different_target_dataset_and_context", "matched_all10_equal_budget_BTC_evidence": False, "inference": "No matching result was identified in this inspected source; no claim of global absence."} for r in extraction]
    mechanisms = [
        {"mechanism": "residual_convolution_and_global_pool", "origin": "vision: image segmentation/object recognition", "primary_origin": "Long et al., Fully Convolutional Networks for Semantic Segmentation; He et al., Deep Residual Learning for Image Recognition", "origin_urls": ["https://arxiv.org/abs/1411.4038", "https://arxiv.org/abs/1512.03385"], "transfer_bridge": "Wang1611.06455v4 II.B-C; current FCN/ResNet policy", "shared_property": "local neighborhoods and residual optimization", "D7_limit": "padding and receptive fields dominate a seven-day context; classification CAM and accuracy do not become volatility explanations/results", "origin_fulltext_read_this_round": False},
        {"mechanism": "parallel_receptive_fields", "origin": "vision: Inception image recognition", "primary_origin": "Szegedy et al., Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning", "origin_urls": ["https://arxiv.org/abs/1602.07261"], "transfer_bridge": "InceptionTime1909.04939v3 sections3.1-3.2", "shared_property": "parallel filters capture patterns at different scales", "D7_limit": "40/20/10 kernels exceed seven observations; preserve padding and test contribution; single network differs from ensemble", "origin_fulltext_read_this_round": False},
        {"mechanism": "patch_token_attention", "origin": "vision: image patches", "primary_origin": "Dosovitskiy et al., An Image Is Worth16x16 Words; Liu et al., Swin Transformer", "origin_urls": ["https://arxiv.org/abs/2010.11929", "https://arxiv.org/abs/2103.14030"], "transfer_bridge": "previously extracted PatchTST section3.1; Crossformer origin trace pp3,10-11", "shared_property": "segments of observed inputs become tokens", "D7_limit": "P16/S8 from PatchTST/64 does not directly fit D7; temporal axes are not interchangeable with variables", "origin_fulltext_read_this_round": False},
        {"mechanism": "token_channel_mixing", "origin": "vision: token/channel MLP mixing", "primary_origin": "Tolstikhin et al., MLP-Mixer: An All-MLP Architecture for Vision", "origin_urls": ["https://arxiv.org/abs/2105.01601"], "transfer_bridge": "TSMixer fulltext reference p15; origin_mechanism_trace.json", "shared_property": "mix positions and channels separately", "D7_limit": "short context does not inherit long-series efficiency or accuracy; TSMixer feature-mixing ablation can lose", "origin_fulltext_read_this_round": False},
    ]
    queries = {"source": "arXiv native API; no date filter; start0,max_results10", "queries": [{"query_id": key, "query": query, "url": "https://export.arxiv.org/api/query?" + urlencode(dict(search_query=query, start=0, max_results=10)), "attempted_at_utc": stamp, "native_total": None, "retrieved_records": None, "status": "attempted_not_retrieved", "error_type": "URLError", "error": "<urlopen error Tunnel connection failed: 403 Forbidden>"} for key, query, stamp in NATIVE_ATTEMPTS], "null_policy": "Failure does not mean zero results; unknown counts remain null.", "web_connector_attempt": "All three identical API URLs also failed with not accessible via this tool.", "supplementary_web_search": "The three corresponding site:arxiv.org keyword searches returned a pooled set of10 snippets. Per-query totals were not exposed. These snippets are not native retrieval or eligibility evidence."}
    summary = {"schema": "e2.literature.bounded_closure.v1", "date": DATE, "current_systems": 10, "current_primary_studies": flow["current_Top10_distinct_primary_studies"], "current_families": dict(family_counts), "historical_complete_originals": 10, "historical_complete_physical_pages": flow["physical_pages"], "source_metric_rows_current_systems": len(extraction), "current_shared_source_pairs": [["FCN-1D", "ResNet-1D"]], "complete_main_text_source_refresh_today": ["Wang1611.06455v4", "InceptionTime1909.04939v3", "SAE; PMC6911101"], "targeted_source_refresh_today": [], "new_PDF_binary_receipts_today": 0, "new_training_runs": 0, "double_human_review_accredited": False, "global_review_complete": False, "final_optimal_Top10_certified": False, "native_transversal_results_retrieved_today": False, "external_replication_credit": "Existing TimeMixer ETTh1 five-seed partial compatible replication and three-model pilot remain valid within their original task; not BTC evidence.", "null_policy": "JSON null/empty CSV is unverified; NR only after examined source scope; algebraic interval endpoints are not missing measurements."}
    score_protocol = {"status": "prospective_support_audit_after_authorized_catalogue_not_historical_selection", "weights": WEIGHTS, "formula": "S=100*(wC*C+wU*U+wP*P+wA*A+wF*F)", "C": "Scopus citations/(2026-citation_age_year+1)/normalizer, clipped1; same received original-paper counts only; missing citations stay null", "citation_rate_normalizer": normalizer, "citation_normalizer_scope": "largest observed original-paper rate in the old10 received export; fixed reference, not a world maximum", "age_and_recency_year_policy": "First arXiv submission year encoded in the verified source identity; documented publication year when no arXiv identity is available. Applied consistently to citations and recency. Citation counts are unchanged; older artifacts retain their original age convention.", "U": "clip((original_year-2019)/7,0,1); foundational2016 source valid with recency0, an observed date-derived value", "P": "Comparable BTC sigma7 performance under the same information set, scale, horizon and protocol; null for every current source; no external-metric ranks transferred", "A": "1 when a fixed author-code identity is documented in current architecture policy; documentary availability, not execution success", "F": "1/(number of current systems assigned to the same functional family); explicit within-catalogue rarity, not external merit", "family_constraint": "At least4 functional families in the current10; families come from the authorized catalogue", "unknown_policy": "Algebraic min/max vary each null within[0,1]; do not fill observed fields or rank by interval midpoint", "rank_policy": "Conservative bounds based on disjoint score intervals; overlapping intervals imply unknown order", "selection_warning": "Current membership was authorized20261112. These20261113 weights did not cause that selection, are not preregistered, and do not certify global optimality. Four scenarios expose dependence on weights."}
    alternatives = []
    for name in ["Informer", "Autoformer", "Crossformer", "S-Mamba"]:
        row = citations[name]
        alternatives.append({"model": name, "source_version": old[name]["source_version"], "Scopus_citations": int(row["Cited_by_Scopus"]) if row["Cited_by_Scopus"] else None, "current_status": "not_in_authorized_Top10", "reason": "Current catalogue prioritizes four functional families including a recurrent/autoencoder and three temporal convolution baselines; this is an authorized scope choice, not experimental inferiority or score dominance.", "remaining_obligation": "No global most-cited-excluded list can be certified from incomplete citation coverage; unknown counts are not zero."})
    reading_ledger = {"date": DATE, "method": "Assistant reading of primary full main web text, including tables, captions, results and discussion; not independent human double review.", "source_studies": [{"study_id": s["study_id"], "title": s["title"], "url": s["url"], "version": s["version"], "read_level": s["read_level"], "metric_locator": s["metric_locator"], "binary_PDF_downloaded_this_round": False, "new_PDF_sha256": None} for name, s in SUPPLEMENTS.items() if name != "ResNet-1D"], "access_limit": "Nature URL redirected to an inaccessible identity endpoint; SAE was read from primary PMC full text. No supplied full papers or web full text are redistributed."}
    files = {"summary.json": json_bytes(summary), "bounded_flow.json": json_bytes(flow), "directed_originals10.csv": csv_bytes(historical_rows), "current10_extraction.json": json_bytes(extraction), "current10_extraction.csv": csv_bytes(extraction), "quality_coverage_architectural10.json": json_bytes(quality_counts), "quality_coverage_architectural10.csv": csv_bytes(quality_counts), "support_score_protocol.json": json_bytes(score_protocol), "current10_support_scores.csv": csv_bytes(score_rows), "score_sensitivity.csv": csv_bytes(sensitivity), "gap_falsification_matrix.csv": csv_bytes(gap_rows), "transverse_mechanisms.json": json_bytes(mechanisms), "transverse_mechanisms.csv": csv_bytes(mechanisms), "excluded_originals_context.csv": csv_bytes(alternatives), "native_query_attempts.json": json_bytes(queries)}
    files["additional_source_reading_ledger.json"] = json_bytes(reading_ledger)
    input_paths = [source_path, catalogue_path, tide_path, citation_path, ROOT / "experiments/e2_published_architectures_20261112/catalogue.json", EVIDENCE / "top10_20261105/primary_source_registry.json", Path(__file__)]
    files["manifest.json"] = json_bytes({"schema": "e2.literature.closure.manifest.v1", "inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}, "outputs": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}, "original_papers_redistributed": False})
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--output-dir", type=Path, default=STATIC)
    args = p.parse_args()
    files = build()
    if args.check:
        bad = [name for name, data in files.items() if not (args.output_dir / name).exists() or (args.output_dir / name).read_bytes() != data]
        if bad:
            raise SystemExit("outdated/missing: " + ", ".join(bad))
        print(f"verified {len(files)} bounded literature artifacts")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (args.output_dir / name).write_bytes(data)
        print(f"wrote {len(files)} bounded literature artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
