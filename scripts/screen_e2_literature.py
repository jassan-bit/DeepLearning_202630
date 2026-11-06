"""Screen the received E2 corpus without claiming a new database search.

Standard-library only. Full abstracts are read privately; public outputs contain
identifiers, literal matched *terms*, hashes, rules and provisional decisions.
This is a post-search, rule-assisted first pass, not independent human review or
completed systematic eligibility. Frozen imports and original extractions stay
unchanged. Missing records, unverified quartiles and unread texts remain unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "e2_received_corpus_screening_20261106_v1"
BASE = ROOT / "book_e2/_static"
REF = ROOT / "references/literature_e2/remediation_20261106"
PUBLIC = BASE / "literature_remediation"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def text_hash(value):
    return sha256(value.encode("utf-8")).hexdigest()


def normalized_title(value):
    import unicodedata
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKD", value).casefold()))


def doi(value):
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip().lower()).rstrip(".,;")


def rx(pattern):
    return re.compile(pattern, re.I)


PATTERNS = {
    "crypto": rx(r"\b(?:bitcoin|btc(?:usdt)?|ethereum|ether|ethusdt|cryptocurrenc\w*|crypto[- ]?assets?|cryptoassets?|litecoin|dogecoin|ripple|binance coin|solana|cardano)\b"),
    "finance": rx(r"\b(?:financ\w*|stock\w*|equity|equities|asset\w*|trading|traded|exchange rates?|foreign exchange|forex|S&P\s*\d+|Dow Jones|NASDAQ|FTSE|CAC\s*40|commodit\w*|oil prices?|energy prices?|electricity prices?|market returns?|market volatility)\b"),
    "volatility": rx(r"\b(?:volatilit\w*|realized variance|realised variance|conditional variance)\b"),
    "forecast": rx(r"\b(?:forecast\w*|predict\w*|nowcast\w*)\b"),
    "deep": rx(r"\b(?:deep learning|deep neural|neural networks?|neural forecast\w*|multilayer perceptrons?|multi-layer perceptrons?|MLP|LSTM|GRU|CNN|TCN|transformers?|attention mechanisms?|state[- ]space models?|selective state|structured state)\b"),
    "review": rx(r"\b(?:systematic (?:literature )?review|literature review|scoping review|bibliometric|survey(?:s)? (?:of|on)|a comprehensive review)\b"),
    "vision": rx(r"\b(?:computer vision|images?|visual|vision|object detection|image classification)\b"),
    "language": rx(r"\b(?:language|speech|audio|text|natural language processing|NLP)\b"),
    "transfer": rx(r"\b(?:transfer\w*|adapt\w*)\b"),
    "attention": rx(r"\b(?:vision transformers?|self[- ]attention|ViT|Swin)\b"),
    "mixing": rx(r"\b(?:MLP[- ]Mixer|token[- ]mixing|channel[- ]mixing)\b"),
    "ssm": rx(r"\b(?:selective state[- ]space|selective state|structured state[- ]space|Mamba)\b"),
    "volatility_target_hint": rx(r"\b(?:volatilit\w*|realized variance|realised variance|conditional variance)\s+(?:(?:forecast\w*|predict\w*)\b)|\b(?:forecast\w*|predict\w*)\s+(?:(?:the|future|financial|stock|cryptocurrency|bitcoin|realized|realised|conditional|market|price|daily|intraday|of)\s+){0,5}(?:volatilit\w*|variance)\b"),
}

# TiDE is deliberately case-sensitive: the English word tide is not this model.
MODEL_PATTERNS = {
    "TSMixer": rx(r"\bTSMixer\b"),
    "TiDE": re.compile(r"\bTiDE\b|(?i:time[- ]series dense encoder)"),
    "TimeMixer": rx(r"\bTimeMixer(?:\+\+)?\b"),
    "TimesNet": rx(r"\bTimesNet\b"),
    "PatchTST": rx(r"\bPatchTST\b"),
    "Informer": rx(r"\bInformer\b"),
    "Autoformer": rx(r"\bAutoformer\b"),
    "iTransformer": rx(r"\biTransformer\b"),
    "Crossformer": rx(r"\bCrossformer\b"),
    "S-Mamba": rx(r"\bS[- ]Mamba\b"),
    "CryptoMamba-SSM": rx(r"\bCryptoMamba(?:[- ]SSM)?\b"),
    "MICN": rx(r"\bMICN\b|multi[- ]scale isometric convolution"),
    "N-BEATS": rx(r"\bN[- ]?BEATS\b"),
    "N-BEATSx": rx(r"\bN[- ]?BEATSx\b"),
    "N-HiTS": rx(r"\bN[- ]?HiTS\b"),
    "TFT": rx(r"temporal fusion transformer|\bTFT\b"),
    "DeepAR": rx(r"\bDeepAR\b"),
    "DLinear": rx(r"\bDLinear\b"),
    "S4": rx(r"\bS4\b"),
    "Mamba": rx(r"\bMamba(?:[- ]SSM)?\b"),
    "BiLSTM": rx(r"\bbi[- ]?LSTM\b|bidirectional long short[- ]term memory"),
    "CNN-LSTM": rx(r"\bCNN[- +/]+LSTM\b"),
    "GRU": rx(r"\bGRU\b|gated recurrent unit"),
    "LSTM": rx(r"\bLSTM\b|long short[- ]term memory"),
    "TCN": rx(r"\bTCN\b|temporal convolutional network"),
    "MLP": rx(r"\bMLP\b|multilayer perceptrons?|multi-layer perceptrons?"),
    "DFFNN": rx(r"\bDFFNNs?\b|deep feed[- ]?forward neural networks?"),
    "CNN": rx(r"\bCNN\b|convolutional neural networks?"),
    "Jordan-RNN": rx(r"Jordan (?:neural|recurrent|network)"),
}
FAMILIES = {
    **dict.fromkeys(["TSMixer", "TiDE", "TimeMixer", "N-BEATS", "N-BEATSx", "N-HiTS"], "MLP/mixing"),
    **dict.fromkeys(["PatchTST", "Informer", "Autoformer", "iTransformer", "Crossformer", "TFT"], "Attention"),
    **dict.fromkeys(["S-Mamba", "CryptoMamba-SSM", "S4", "Mamba"], "State-space"),
    **dict.fromkeys(["GRU", "LSTM", "BiLSTM", "DeepAR"], "Recurrent"),
    "TimesNet": "Convolution", "MICN": "Convolution", "TCN": "Convolution",
    "CNN-LSTM": "Hybrid", "DLinear": "Linear contextual reference; not a deep-model cup",
    "MLP": "MLP; depth requires verification", "DFFNN": "MLP/deep feedforward",
    "CNN": "Convolution; dimensionality requires verification", "Jordan-RNN": "Recurrent",
}
TRANSVERSAL = {
    "T1_attention_vision": {"family": "attention", "modality": "vision", "expression": '("vision transformer" OR "self-attention") AND ("computer vision" OR image*) AND (transfer* OR adapt*)'},
    "T2_mixing_vision": {"family": "mixing", "modality": "vision", "expression": '("MLP-Mixer" OR "token mixing" OR "channel mixing") AND (vision OR image*) AND (transfer* OR adapt*)'},
    "T3_ssm_language": {"family": "ssm", "modality": "language", "expression": '("selective state space" OR Mamba) AND (language OR speech OR audio) AND (transfer* OR adapt*)'},
}


def detect_features(title, abstract):
    value = title + "\n" + abstract
    found = {name: bool(pattern.search(value)) for name, pattern in PATTERNS.items()}
    models = [name for name, pattern in MODEL_PATTERNS.items() if pattern.search(value)]
    if "S4" in models and not rx(r"state[- ]space|structured state").search(value):
        models.remove("S4")
    if "Mamba" in models and not (found["deep"] or found["forecast"] or rx(r"sequence model|linear[- ]time|time series").search(value)):
        models.remove("Mamba")
    # Generic recurrent comparators do not alone identify a transferable original.
    found["forecasting_architecture"] = any(m not in {"GRU", "LSTM", "TCN", "BiLSTM", "CNN-LSTM", "MLP", "CNN", "DFFNN", "Jordan-RNN"} for m in models)
    found["deep"] = found["deep"] or bool(set(models) - {"DLinear", "TFT"})
    found["models"] = models
    found["origin_modality"] = found["vision"] or found["language"]
    found["transfer_token"] = found["transfer"]
    found["matched_terms"] = {k: sorted({m.group(0) for m in p.finditer(value)})[:12] for k, p in PATTERNS.items() if p.search(value)}
    return found


def local_transversal_hits(public, abstract):
    f = detect_features(public.get("title", ""), abstract)
    return [qid for qid, q in TRANSVERSAL.items() if f[q["family"]] and f[q["modality"]] and f["transfer"]]


def classify_record(public, abstract, known_original=False, known_support=False):
    f = detect_features(public.get("title", ""), abstract)
    trans = [qid for qid, q in TRANSVERSAL.items() if f[q["family"]] and f[q["modality"]] and f["transfer"]]
    review = f["review"] or "review" in public.get("document_type", "").lower()
    flags = []
    try:
        y = int(public.get("earliest_known_publication_year") or public.get("year") or 0)
    except ValueError:
        y = 0
    if not (2019 <= y <= 2026):
        flags.append("publication_date_or_foundational_exception_requires_review")
    if review:
        flags.append("review_is_context_not_independent_primary_experiment")
    if not abstract.strip():
        flags.append("missing_abstract")
    if known_original:
        decision, reason, route, priority = "retain_for_fulltext", "R01_verified_architectural_original", "original_architecture", 1
    elif known_support:
        decision, reason, route, priority = "retain_for_fulltext", "R02_preexisting_applied_support_text_not_yet_extracted", "applied_support", 1
    elif f["crypto"] and f["volatility"] and f["forecast"] and f["deep"]:
        decision, reason, route, priority = "retain_for_fulltext", "R03_crypto_volatility_deep_forecast_terms", "crypto_volatility", 0 if f["volatility_target_hint"] else 1
    elif f["finance"] and f["volatility"] and f["forecast"] and f["deep"]:
        decision, reason, route, priority = "retain_for_fulltext", "R04_other_financial_volatility_deep_forecast_terms", "financial_volatility", 1
    elif (f["crypto"] or f["finance"]) and (f["forecast"] or f["volatility"]):
        decision, reason, route, priority = "manual_review_pending", "M01_finance_target_or_deep_mechanism_uncertain", "finance_other_target", 2
    elif trans:
        decision, reason, route, priority = "retain_for_fulltext", "R05_local_transversal_conjunction", "transversal_mechanism", 3
    elif f["forecasting_architecture"] and f["forecast"]:
        decision, reason, route, priority = "retain_for_fulltext", "R06_named_transferable_forecasting_mechanism", "transferable_forecasting", 3
    elif f["forecast"] and f["deep"]:
        decision, reason, route, priority = "manual_review_pending", "M02_other_domain_deep_forecast_needs_transfer_rationale", "transferable_forecasting", 4
    elif not abstract.strip():
        decision, reason, route, priority = "manual_review_pending", "M03_missing_abstract", "uncertain", 4
    elif f["forecasting_architecture"] or (f["origin_modality"] and (f["attention"] or f["mixing"] or (f["ssm"] and f["deep"]))):
        decision, reason, route, priority = "manual_review_pending", "M04_origin_mechanism_or_homonym_needs_review", "transversal_mechanism", 4
    elif f["forecast"] and not f["deep"]:
        decision, reason, route, priority = "exclude_title_abstract_provisional", "E01_other_domain_forecast_without_deep_mechanism_signals", "out_of_scope", 9
    else:
        decision, reason, route, priority = "exclude_title_abstract_provisional", "E02_no_financial_forecast_or_transfer_mechanism_signals", "out_of_scope", 9
    # Reviews can inform chaining but cannot be promoted to primary evidence.
    if review and not known_original and decision == "retain_for_fulltext":
        decision, reason = "manual_review_pending", "M05_review_for_context_and_reference_chaining"
    return {"screen_decision": decision, "primary_reason": reason, "route": route,
            "priority": priority, "flags": flags, "features": f, "local_transversal_hits": trans,
            "target_role_verification": "not_verified_from_keywords",
            "final_eligibility": "pending_fulltext_adjudication",
            "human_adjudication": "not_performed", "model_comparator_roles": "not_verified"}


def validate_join(public_rows, private_rows):
    def index(rows, label):
        result = {}
        for row in rows:
            rid = row.get("record_id")
            if not rid or rid in result:
                raise ValueError(f"Missing or duplicate record_id in {label}")
            result[rid] = row
        return result
    pp, qq = index(public_rows, "public"), index(private_rows, "private")
    if pp.keys() != qq.keys():
        raise ValueError("Public/private record-ID sets differ; no silent dropping")
    for rid, row in pp.items():
        q = qq[rid]
        if normalized_title(row.get("title", "")) != normalized_title(q.get("title", "")) or doi(row.get("DOI", "")) != doi(q.get("DOI", "")):
            raise ValueError(f"Public/private identity metadata mismatch: {rid}")
    return qq


def abstracts(private):
    values = {private.get("abstract", ""), private.get("AB", "")}
    for r in private.get("source_records", []):
        raw = r.get("raw_scientific_record", {})
        values.update([raw.get("AB", ""), raw.get("Abstract", "")])
    return sorted({v.strip() for v in values if v and v.strip()})


LEDGER_FIELDS = ["record_id", "title", "DOI", "year", "earliest_known_publication_year", "source_title", "source_databases", "UTs", "EIDs", "WoS_TC", "Scopus_Cited_by", "WoS_native_index", "journal_quartile", "screen_decision", "primary_reason", "route", "priority", "flags", "screening_feature_flags", "model_mentions_screening_only", "local_transversal_hits", "matched_terms", "private_abstract_sha256", "distinct_private_abstracts", "source_occurrences", "known_original_model", "known_applied_support", "fulltext_extraction_status", "target_role_verification", "model_comparator_roles", "final_eligibility", "human_adjudication", "assistant_editorial_check", "editorial_target_class", "editorial_target_and_role_note"]


def public_record(public, assessment, abstract_hash):
    row = {k: public.get(k, "") for k in LEDGER_FIELDS if k in public}
    row.update({k: assessment[k] for k in ["screen_decision", "primary_reason", "route", "priority", "flags", "target_role_verification", "model_comparator_roles", "final_eligibility", "human_adjudication"]})
    row.update(model_mentions_screening_only=assessment["features"]["models"],
               local_transversal_hits=assessment["local_transversal_hits"], matched_terms=assessment["features"]["matched_terms"],
               screening_feature_flags={k: v for k, v in assessment["features"].items() if isinstance(v, bool)},
               private_abstract_sha256=abstract_hash, journal_quartile="unknown_not_verified_by_year")
    return row


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def json_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) if isinstance(value, (dict, list)) else value


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: json_value(r.get(k)) for k in fields})


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def apply_editorial_check(row, check):
    """A dated assistant reading of an abstract is not full-text eligibility."""
    if check["private_abstract_sha256"] != row["private_abstract_sha256"]:
        raise ValueError("Editorial annotation refers to a different abstract version")
    row.update(assistant_editorial_check="native_title_abstract_read_by_assistant_20261106_not_independent_human_review",
               editorial_target_class=check["target_class"], editorial_target_and_role_note=check["note"],
               target_role_verification="abstract_editorially_checked_fulltext_pending")
    if check["target_class"] == "price_return_or_trading_context":
        row.update(screen_decision="manual_review_pending", primary_reason="M06_editorial_abstract_identifies_other_target", route="finance_other_target", priority=2)
    elif check["target_class"] == "ambiguous_return_volatility_definition":
        row.update(screen_decision="manual_review_pending", primary_reason="M07_editorial_abstract_target_definition_ambiguous", priority=1)
    elif check["target_class"] == "explicit_volatility_forecast":
        row["flags"].append("volatility_target_explicit_in_abstract_fulltext_definition_pending")
    else:
        raise ValueError("Unknown editorial target class")
    return row


def candidate_table(ledger, registry, native_originals, prior_scores):
    by_name = {m["name"]: m for m in registry["models"]}
    orig = {r["model"]: r for r in native_originals}
    scores = {r["model"]: r for r in prior_scores}
    result = []
    for name in MODEL_PATTERNS:
        refs = [r for r in ledger if name in r["model_mentions_screening_only"]]
        counts = Counter(r["route"] for r in refs)
        m, o, s = by_name.get(name), orig.get(name, {}), scores.get(name, {})
        result.append({"model_or_family": name, "family": FAMILIES[name],
            "current_top10": bool(m), "independent_comparator_frequency": None,
            "mention_record_groups_not_comparator_frequency": len(refs),
            "crypto_volatility_candidate_groups": sum(all(r["screening_feature_flags"][f] for f in ["crypto", "volatility", "forecast", "deep"]) for r in refs),
            "financial_volatility_candidate_groups": sum(not r["screening_feature_flags"]["crypto"] and all(r["screening_feature_flags"][f] for f in ["finance", "volatility", "forecast", "deep"]) for r in refs),
            "editorially_checked_explicit_volatility_forecast_groups": sum(r.get("editorial_target_class") == "explicit_volatility_forecast" for r in refs),
            "editorially_checked_price_return_context_groups": sum(r.get("editorial_target_class") == "price_return_or_trading_context" for r in refs),
            "editorially_checked_ambiguous_target_groups": sum(r.get("editorial_target_class") == "ambiguous_return_volatility_definition" for r in refs),
            "other_route_counts": dict(counts), "native_primary_identity_status": "matched" if o.get("found_in_Scopus_export") == "True" else "unknown_no_verified_native_primary_match",
            "primary_Scopus_EID": o.get("EID") or None, "primary_WoS_UT": o.get("WoS_UT") or None,
            "primary_year": (m or {}).get("paper_year"), "primary_citation_age_year": o.get("citation_age_year") or None,
            "primary_Scopus_citations": o.get("Cited_by_Scopus") or None,
            "primary_citations_per_calendar_year": o.get("Cited_by_Scopus_per_calendar_year") or None,
            "primary_code_verified": True if m else None,
            "original_fulltext_extracted": True if m else None,
            "year_specific_quartile": None, "comparable_BTC_performance_component": None,
            "family_representativeness_verified": None,
            "historical_preliminary_score_lower": s.get("preliminary_score_lower") or None,
            "historical_preliminary_score_upper": s.get("preliminary_score_upper") or None,
            "reassessment_outcome": "retain_as_provisional_candidate_not_final_rank" if m else "alternative_requires_primary_identity_fulltext_and_contract_review",
            "next_action": "Complete applied fulltexts and comparative selection; current score is not a new rank" if m else "Review priority ledger; verify standalone/hybrid role, original identity, code and D7 compatibility",
            "score_of_alternative": None})
    return result


def protocol():
    return {"protocol_id": PROTOCOL_ID, "status": "post_search_rule_assisted_first_pass_not_preregistered",
        "date": "2026-11-06", "primary_problem": "BTC volatility forecasting; financial volatility prioritized; other-domain mechanisms require transfer justification",
        "publication_window": [2019, 2026], "outside_window_policy": "flag for date verification or justified foundational exception; never automatic exclusion on year",
        "inputs": "Frozen received 7833 bibliographic groups and matching private abstracts; no new remote database query",
        "reviewer_model": "One editorial workflow assisted by deterministic rules; zero independent human adjudications asserted; no fabricated Cohen kappa",
        "decision_semantics": {"retain_for_fulltext": "Recover or adjudicate full text; not included study", "manual_review_pending": "Ambiguous, contextual or origin mechanism; requires human decision", "exclude_title_abstract_provisional": "Machine exclusion recommendation only; human validation pending"},
        "precedence": ["verified architectural originals", "existing applied support", "crypto volatility + forecast + deep signals", "other financial volatility + forecast + deep signals", "other finance target/unclear deep mechanism", "local transversal conjunction", "named other-domain forecasting mechanism", "ambiguous forecast/origin/missing abstract", "provisional exclusion"],
        "rule_source": "scripts/screen_e2_literature.py::classify_record; features and matched terms are recorded per group",
        "target_role_rule": "Keyword conjunction does not prove volatility target, standalone comparator, causality, performance or methodological quality; these remain unverified",
        "fulltext_eligibility_checklist": ["actual forecast target and units", "standalone/hybrid/comparator role", "dataset, period, frequency and horizon", "architecture and layers/dimensions", "split and train-only preprocessing", "seeds and repetitions", "metrics and uncertainty", "baseline HPO fairness", "code and data availability/license", "year-specific venue quality", "D7 transfer assumptions"],
        "complete_review_eligibility_closed": False, "new_models_trained": 0,
        "local_transversal_queries": {qid: {**q, "executed_local_regex": {"family": PATTERNS[q["family"]].pattern, "modality": PATTERNS[q["modality"]].pattern, "transfer": PATTERNS["transfer"].pattern}} for qid, q in TRANSVERSAL.items()},
        "native_transversal_database_execution": False,
        "transversal_local_semantics": "Case-insensitive regex conjunction over received title and distinct private abstracts. Local lexical variants are in code; not equal to native WoS TS indexing or a new global search.",
        "coverage_limits": ["374 Scopus records unreceived", "WoS per-query totals/ranges and file-to-query assignment unconfirmed", "Exact Scopus query/filters not encoded", "Seven additional applied texts not fully extracted", "Year-specific quartiles unknown"],
        "selection_rule": "Preserve historical Top10 score intervals; expose alternatives without treating mentions as comparator frequency or assigning unknown score=0. No final replacement/rank before source verification."}


def run(args):
    private_path = args.private_combined.resolve()
    if private_path.is_relative_to(ROOT.resolve()):
        raise ValueError("Private full abstracts must remain outside the public repository")
    pub_rows = csv_rows(args.combined)
    private_rows = json.loads(private_path.read_text())
    private = validate_join(pub_rows, private_rows)
    if len(pub_rows) != args.expected_groups:
        raise ValueError("Received group count differs from the declared corpus")
    for output in [args.reference_output.resolve(), args.book_output.resolve()]:
        if output.is_relative_to(ROOT.resolve()) and output not in {REF.resolve(), PUBLIC.resolve()}:
            raise ValueError("Refusing to overwrite a frozen or unrelated repository directory")
    registry = json.loads(args.registry.read_text())
    manifests = json.loads(args.fulltext_manifest.read_text())
    original_title = {normalized_title(m["paper_title"]): m["name"] for m in registry["models"]}
    original_doi = {doi(m["doi"]): m["name"] for m in registry["models"] if m.get("doi")}
    support = csv_rows(args.support)
    support_ids = {doi(r["doi"]) for r in support if r.get("doi")}
    annotation_path = getattr(args, "editorial_checks", None)
    annotations = json.loads(annotation_path.read_text())["checks"] if annotation_path and annotation_path.exists() else []
    checks = {}
    for check in annotations:
        rid = check["record_id"]
        if rid in checks or rid not in private:
            raise ValueError("Duplicate or unmatched editorial annotation")
        checks[rid] = check
    ledger, hits = [], []
    for row in pub_rows:
        q = private[row["record_id"]]
        aa = abstracts(q)
        full = "\n".join(aa)
        name = original_title.get(normalized_title(row["title"])) or original_doi.get(doi(row.get("DOI", "")))
        known_support = doi(row.get("DOI", "")) in support_ids
        assessment = classify_record(row, full, bool(name), known_support)
        public = public_record(row, assessment, text_hash(full))
        indices = sorted({r.get("raw_scientific_record", {}).get("WE", "") for r in q.get("source_records", []) if r.get("raw_scientific_record", {}).get("WE")})
        public.update(distinct_private_abstracts=len(aa), WoS_native_index="; ".join(indices), known_original_model=name,
            known_applied_support=known_support,
            fulltext_extraction_status="original_architectural_text_extracted_global_eligibility_pending" if name else "pending_fulltext_recovery_or_extraction")
        if row["record_id"] in checks:
            apply_editorial_check(public, checks[row["record_id"]])
        ledger.append(public)
        for qid in assessment["local_transversal_hits"]:
            hits.append({k: public[k] for k in ["record_id", "title", "DOI", "year", "source_title", "source_databases", "screen_decision", "primary_reason"]} | {"query_id": qid, "matched_terms": public["matched_terms"], "native_database_query": False, "fulltext_eligibility": "pending"})
    ledger.sort(key=lambda r: r["record_id"])
    queue = sorted([r for r in ledger if r["screen_decision"] != "exclude_title_abstract_provisional"], key=lambda r: (r["priority"], -(int(r.get("year") or 0)), r["record_id"]))
    decisions, reasons, routes = (dict(sorted(Counter(r[k] for r in ledger).items())) for k in ["screen_decision", "primary_reason", "route"])
    originals_in_corpus = sorted({r["known_original_model"] for r in ledger if r["known_original_model"]})
    originals_external = sorted(set(original_title.values()) - set(originals_in_corpus))
    coverage = json.loads(args.scopus_summary.read_text())["coverage"]
    native_occurrences = sum(int(r.get("native_source_occurrences") or 1) for r in pub_rows)
    duplicates_removed = native_occurrences - len(ledger)
    summary = {"protocol_id": PROTOCOL_ID, "status": "machine_first_pass_complete_human_adjudication_and_fulltext_eligibility_open",
        "received_native_occurrences": native_occurrences, "duplicate_occurrences_removed": duplicates_removed,
        "received_bibliographic_groups": len(ledger), "groups_rule_screened": len(ledger),
        "screen_decisions": decisions, "primary_reason_counts": reasons, "route_counts": routes,
        "priority_queue_groups": len(queue), "human_adjudicated_title_abstract_groups": 0,
        "human_validated_exclusions": None, "fulltext_global_eligibility_assessed": 0,
        "assistant_title_abstract_editorial_checks": len(checks),
        "editorial_target_classes": dict(sorted(Counter(c["target_class"] for c in checks.values()).items())),
        "included_studies_in_complete_systematic_review": None,
        "global_systematic_review_closed": False, "Scopus_coverage": coverage,
        "fulltext_architectural_originals_preexisting": len(manifests["papers"]),
        "fulltext_original_physical_pages": manifests["physical_page_total"],
        "known_originals_matched_in_received_corpus": originals_in_corpus,
        "known_originals_from_directed_expansion_not_matched_in_native_corpus": originals_external,
        "preexisting_applied_support_records_matched": sum(r["known_applied_support"] for r in ledger),
        "additional_applied_fulltexts_pending": len(support_ids - set(original_doi)),
        "additional_applied_fulltexts_pending_scope": "Seven preexisting curated support texts; not the total of all newly queued candidate texts",
        "editorial_checked_candidate_fulltexts_not_read_in_this_remediation": len(checks),
        "local_transversal_hits": {qid: sum(h["query_id"] == qid for h in hits) for qid in TRANSVERSAL},
        "local_transversal_unique_groups": len({h["record_id"] for h in hits}),
        "native_transversal_queries_executed_by_this_script": 0,
        "independent_comparator_frequencies_verified": False, "year_specific_quartiles_verified": False,
        "new_models_trained": 0, "native_abstracts_redistributed": False,
        "method_warning": "All decisions are rule-assisted recommendations. New ledger is not a claim of completed independent review or an exhaustive database universe."}
    candidates = candidate_table(ledger, registry, csv_rows(args.originals), csv_rows(args.prior_scores))
    trans_report = {"scope": "Local searches in already received records only", "date": "2026-11-06", "native_database_search": False,
        "queries": [{"query_id": qid, **q, "executed_local_regex": {"family": PATTERNS[q["family"]].pattern, "modality": PATTERNS[q["modality"]].pattern, "transfer": PATTERNS["transfer"].pattern}, "hits": summary["local_transversal_hits"][qid], "source_scope_groups": len(ledger), "fulltext_review_status": "pending", "limitations": "Received search universe is incomplete and selected for prior queries; expression is a conceptual search string, executed local regex variants are given explicitly; local zero cannot establish absence in native database"} for qid, q in TRANSVERSAL.items()]}
    prisma = {"scope": "Received exports only; count machine screening separately from human eligibility", "identified_occurrences_received": native_occurrences, "duplicates_removed": duplicates_removed,
        "records_after_deduplication": len(ledger), "records_rule_screened": len(ledger), "machine_screening_recommendations": decisions,
        "records_human_screened": 0, "human_validated_screening_exclusions": None,
        "fulltext_architectural_originals_previously_extracted": len(manifests["papers"]), "of_those_native_corpus_matches": len(originals_in_corpus),
        "directed_expansion_originals_not_in_received_corpus": originals_external,
        "fulltexts_formally_assessed_against_global_review_eligibility": 0, "fulltexts_excluded_after_global_eligibility": None,
        "systematic_review_included_studies": None, "Scopus_unreceived_records_not_exclusions": coverage["difference_from_screenshot_count"], "complete": False}
    reference_inputs = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: digest(p) for p in [args.combined, args.registry, args.fulltext_manifest, args.support, args.originals, args.prior_scores, args.scopus_summary]}
    if annotation_path and annotation_path.exists():
        reference_inputs[str(annotation_path.relative_to(ROOT)) if annotation_path.is_relative_to(ROOT) else annotation_path.name] = digest(annotation_path)
    provenance = {"protocol_id": PROTOCOL_ID, "generator": "scripts/screen_e2_literature.py", "generator_sha256": digest(__file__),
        "private_combined_sha256": digest(private_path), "private_abstracts_publicly_written": False,
        "input_sha256": reference_inputs, "supplied_resubmission_hash_note": "Scopus reupload SHA256 387f673a2146902ce12888ab3d662c095abc33d2cf5b54a168e9e2a12099f7de matches original; no new occurrences added", "frozen_inputs_mutated": False}
    report = make_report(summary, candidates)
    for out in [args.reference_output, args.book_output]:
        out.mkdir(parents=True, exist_ok=True)
        write_csv(out / "screening_ledger.csv", ledger, LEDGER_FIELDS)
        write_csv(out / "priority_queue.csv", queue, LEDGER_FIELDS)
        write_csv(out / "financial_priority_queue.csv", [r for r in queue if r["route"] in {"crypto_volatility", "financial_volatility", "finance_other_target", "applied_support"}], LEDGER_FIELDS)
        write_csv(out / "candidate_reassessment.csv", candidates, list(candidates[0]))
        write_csv(out / "local_transversal_hits.csv", hits, ["query_id", "record_id", "title", "DOI", "year", "source_title", "source_databases", "screen_decision", "primary_reason", "matched_terms", "native_database_query", "fulltext_eligibility"])
        write_csv(out / "reason_counts.csv", [{"reason_code": k, "group_count": v} for k, v in reasons.items()], ["reason_code", "group_count"])
        for name, value in [("protocol.json", protocol()), ("summary.json", summary), ("prisma_status.json", prisma), ("local_transversal_searches.json", trans_report)]:
            json_write(out / name, value)
        if annotations and (out / "editorial_title_abstract_checks.json").resolve() != annotation_path.resolve():
            json_write(out / "editorial_title_abstract_checks.json", {"date": "2026-11-06", "scope": "Assistant reading of selected native titles/abstracts, not full manuscripts or independent human double review", "checks": annotations})
        (out / "README.md").write_text(report)
        provenance["public_output_sha256"] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "provenance.json"}
        json_write(out / "provenance.json", provenance)
    return summary


def make_report(summary, candidates):
    ds = summary["screen_decisions"]
    coverage = summary['Scopus_coverage']
    lines = ["# Cribado del corpus recibido E2 — 2026-11-06", "", "**Primera pasada por reglas completada; adjudicación humana y elegibilidad final abiertas.** No es una nueva consulta WoS/Scopus ni una revisión sistemática global cerrada.", "",
        f"Se procesaron los **{summary['received_bibliographic_groups']:,} grupos** de los archivos recibidos ({summary['received_native_occurrences']:,} ocurrencias, {summary['duplicate_occurrences_removed']:,} duplicaciones). Scopus sigue en {coverage['received_export_occurrences']:,}/{coverage['screenshot_displayed_result_count']:,}; los **{coverage['difference_from_screenshot_count']} no recibidos** no son exclusiones PRISMA. El reenvío CSV tiene el mismo hash del original y no se suma.", "", "## Decisiones por registro", "",
        "`screening_ledger.csv` registra cada grupo, regla, señales literales, huella del abstract privado, procedencia, estado y motivo. `priority_queue.csv` ordena por cercanía temática; ninguna prioridad es un resultado predictivo o una inclusión final.", ""]
    lines += [f"- {k}: **{v:,}** grupos." for k, v in ds.items()]
    lines += ["", "Las exclusiones son recomendaciones automáticas provisionales; los términos de volatilidad pueden ser sólo contexto. Objetivo real, rol de comparador autónomo, temporalidad y calidad se verifican en texto completo. Hay cero adjudicaciones humanas independientes afirmadas y no se calcula κ de Cohen.", "", "## Originales y apoyo aplicado", "",
        f"Se conservan {summary['fulltext_architectural_originals_preexisting']} originales extraídos ({summary['fulltext_original_physical_pages']} páginas). Coinciden en el corpus nativo **{len(summary['known_originals_matched_in_received_corpus'])}**; las obras no localizadas en el corpus nativo ({', '.join(summary['known_originals_from_directed_expansion_not_matched_in_native_corpus']) or 'ninguna'}) corresponden a ampliación dirigida. No se insertan como nuevos registros nativos.", "",
        "Los ocho apoyos representativos se enlazan por DOI; S-Mamba coincide con un original. Los siete cuerpos aplicados adicionales siguen pendientes de extracción; sus abstracts no se convierten en evidencia metodológica completa. Las cifras históricas no se reescriben.", "", "## Búsqueda transversal local", "",
        "Se ejecutaron tres conjunciones de mecanismo + modalidad de origen + transferencia/adaptación sobre título y abstracts privados recibidos. Las expresiones y resultados se guardan en `local_transversal_searches.json` y `local_transversal_hits.csv`. No reemplazan nuevas búsquedas nativas con sus fechas, totales y cobertura global.", ""]
    lines += [f"- {k}: **{v:,}** grupos coincidentes." for k, v in summary["local_transversal_hits"].items()]
    lines += ["", "## Reconsideración del Top 10 y alternativas", "",
        f"`candidate_reassessment.csv` contiene {len(candidates)} modelos/familias, incluidos alternativas y baselines. Las cantidades son **menciones por grupo**, no recurrencia verificada como comparador. Se mantienen separados los originales y sus citas de los estudios de apoyo; no se suman WoS y Scopus. Los intervalos anteriores de los diez propuestos son históricos; no se impone un ranking nuevo a alternativas con código, identidad, cuartil o rendimiento desconocidos.", "",
        f"Además se leyeron editorialmente **{summary['assistant_title_abstract_editorial_checks']} resúmenes nativos** seleccionados: las verificaciones están en `editorial_title_abstract_checks.json`. Se identifican explícitamente trabajos de precio/retorno/trading que mencionan volatilidad como contexto; no se suman como respaldo directo de pronóstico de volatilidad. Esta lectura asistida no equivale a doble revisión humana ni a lectura de sus cuerpos completos.", "",
        "Los campos desconocidos permanecen vacíos/null; no reciben puntuación cero. La pertenencia al índice se conserva cuando consta en WoS, pero no acredita Q1. DLinear figura como contexto lineal y no completa un cupo profundo; MLP/CNN requieren verificar profundidad/dimensionalidad.", "", "## Cierre pendiente", "",
        "Validar decisiones por lectura editorial, recuperar textos retenidos priorizando volatilidad cripto/financiera, confirmar objetivo/roles y calidad, ejecutar búsquedas nativas transversales, completar los 374 Scopus y el historial de consultas, verificar cuartiles por año, y comparar seleccionados/excluidos con componentes verificables. La elegibilidad final y el PRISMA completo quedan abiertos.", "", "## Reproducción", "",
        "```bash", "python scripts/screen_e2_literature.py --private-combined /RUTA_PRIVADA/private_combined_records.json", "python -m unittest discover -s tests -p 'test_e2_literature_screening.py'", "```", "",
        "El input privado debe corresponder exactamente a los 7.833 IDs y metadatos públicos. Un desfase, duplicación o ausencia aborta; no hay descarte silencioso. El output público no contiene abstracts completos, referencias nativas ni registros científicos íntegros."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-combined", required=True, type=Path)
    p.add_argument("--combined", type=Path, default=BASE / "literature_scopus/combined_records.csv")
    p.add_argument("--registry", type=Path, default=REF.parent / "top10_20261105/primary_source_registry.json")
    p.add_argument("--fulltext-manifest", type=Path, default=BASE / "literature_fulltext/fulltext_manifest.json")
    p.add_argument("--support", type=Path, default=BASE / "literature_selection/representative_studies.csv")
    p.add_argument("--originals", type=Path, default=BASE / "literature_scopus/original_model_records.csv")
    p.add_argument("--prior-scores", type=Path, default=BASE / "literature_selection/model_selection.csv")
    p.add_argument("--scopus-summary", type=Path, default=BASE / "literature_scopus/summary.json")
    p.add_argument("--editorial-checks", type=Path, default=REF / "editorial_title_abstract_checks.json")
    p.add_argument("--expected-groups", type=int, default=7833)
    p.add_argument("--reference-output", type=Path, default=REF)
    p.add_argument("--book-output", type=Path, default=PUBLIC)
    args = p.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
