"""Reconcile received literature rounds without manufacturing a global PRISMA.

The output is a retrospective documentary audit, not a new search or selection.
Only public normalized metadata and existing source-reading receipts are read.
No experiment metrics, market data, raw abstracts, or fitted models are inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path("references/literature_e2")
OLD = Path("book_e2/_static/e2_literature_closure_20261113")
OUT = EVIDENCE / "final_audit_20261113"
VERSION = "e2-received-literature-final-audit-v1"
DATE = "2026-11-14"
INPUTS = [
    Path("scripts/e2_literature_final_audit.py"),
    EVIDENCE/"scopus_20261105/summary.json",
    EVIDENCE/"scopus_20261105/combined_records.csv",
    EVIDENCE/"remediation_20261106/screening_ledger.csv",
    EVIDENCE/"native_20261108/native/resumen.json",
    EVIDENCE/"native_20261108/native/cribado_metadatos.csv",
    EVIDENCE/"native_20261108/arxiv/summary.json",
    EVIDENCE/"native_20261108/arxiv/records.csv",
    EVIDENCE/"native_20261108/arxiv/screening_initial.csv",
    EVIDENCE/"title_abstract_review_20261108/summary.json",
    EVIDENCE/"screening_followup_20261108/summary.json",
    EVIDENCE/"screening_followup_20261108/source_reviews.json",
    EVIDENCE/"recovery_second_20261108/summary.json",
    EVIDENCE/"recovery_second_20261108/source_reviews.json",
    EVIDENCE/"native_20261108/focal_source_reviews.json",
    EVIDENCE/"fulltext_20261105/models.json",
    OLD/"current10_extraction.json",
    OLD/"current10_support_scores.csv",
    OLD/"support_score_protocol.json",
    OLD/"gap_falsification_matrix.csv",
    OLD/"excluded_originals_context.csv",
    EVIDENCE/"citation_access_20261114/openalex_access_receipt.json",
    EVIDENCE/"citation_access_20261114/fetch_openalex.py",
    EVIDENCE/"citation_access_20261114/citation_coverage.csv",
    EVIDENCE/"citation_access_20261114/web_access_receipt.json",
    EVIDENCE/"citation_access_20261114/current_primary_adjudication.json",
]


def read_json(root, path):
    return json.loads((root/path).read_text())


def read_csv(root, path):
    with (root/path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode()


def csv_bytes(rows):
    if not rows:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (list, dict)) else v for k,v in row.items()})
    return buffer.getvalue().encode()


def norm_doi(value):
    value = value.strip().casefold()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return value if value.startswith("10.") else ""


def arxiv_id(value):
    text = (value or "").strip()
    if "arxiv" not in text.casefold() and not re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?", text, flags=re.I):
        return ""
    match = re.search(r"(?<![0-9])([0-9]{2}(?:0[1-9]|1[0-2])\.[0-9]{4,5})(?:v[0-9]+)?", text, flags=re.I)
    return match.group(1) if match else ""


def norm_title(value):
    return "".join(c for c in unicodedata.normalize("NFKD", value).casefold() if c.isalnum())


def deduplicate(nodes):
    """Exact DOI/arXiv, then title+year only when a DOI is missing.

    Different editorial DOIs never collapse solely on matching titles. Metadata
    groups containing several DOIs remain visible; a group is not a proven study.
    """
    parents = list(range(len(nodes)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    links, conflicts = [], []
    def union(a,b,rule,key):
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[max(ra,rb)] = min(ra,rb)
            links.append({"left": nodes[a]["node_id"], "right": nodes[b]["node_id"], "rule": rule, "key": key})
    for field,rule in [("doi","exact_normalized_DOI"),("arxiv","exact_arXiv_base_id")]:
        seen = {}
        for i,node in enumerate(nodes):
            value = node[field]
            if not value:
                continue
            if value in seen:
                union(seen[value],i,rule,value)
            else:
                seen[value] = i
    titles = defaultdict(list)
    for i,node in enumerate(nodes):
        if node["title"] and node["year"]:
            titles[(norm_title(node["title"]),node["year"])].append(i)
    for key,indices in titles.items():
        known_dois = {nodes[i]["doi"] for i in indices if nodes[i]["doi"]}
        if len(known_dois) > 1:
            conflicts.append({"rule":"title_year_with_distinct_DOIs_not_merged", "node_ids":[nodes[i]["node_id"] for i in indices],"dois":sorted(known_dois)})
            continue
        for i in indices[1:]:
            if not nodes[i]["doi"] or not nodes[indices[0]]["doi"]:
                union(indices[0],i,"exact_normalized_title_and_year_with_missing_DOI",key[0]+":"+key[1])
    groups = defaultdict(list)
    for i,node in enumerate(nodes):
        groups[find(i)].append(node)
    ordered = sorted(groups.values(), key=lambda members:min(n["node_id"] for n in members))
    return ordered, links, conflicts


def build_nodes(root):
    nodes=[]
    def add(corpus,source_id,title,year,doi,arxiv,screening,reason,source):
        nodes.append(dict(node_id=corpus+":"+source_id, corpus=corpus, source_id=source_id,title=title,
                          year=str(year or ""),doi=norm_doi(doi),arxiv=arxiv_id(arxiv),
                          screening=screening,screening_reason=reason, source=str(source)))
    ledger = {r["record_id"]:r for r in read_csv(root,EVIDENCE/"remediation_20261106/screening_ledger.csv")}
    for r in read_csv(root,EVIDENCE/"scopus_20261105/combined_records.csv"):
        s=ledger[r["record_id"]]
        add("native_20261105",r["record_id"],r["title"],r["year"],r["DOI"],r["DOI"],s["screen_decision"],s["primary_reason"],EVIDENCE/"scopus_20261105/combined_records.csv")
    for r in read_csv(root,EVIDENCE/"native_20261108/native/cribado_metadatos.csv"):
        add("native_20261108",r["study_id"],r["title"],r["year"],r["doi"],r["doi"],r["screening_class"],r["reason"],EVIDENCE/"native_20261108/native/cribado_metadatos.csv")
    screens={r["arxiv_id"]:r for r in read_csv(root,EVIDENCE/"native_20261108/arxiv/screening_initial.csv")}
    for r in read_csv(root,EVIDENCE/"native_20261108/arxiv/records.csv"):
        add("arxiv_20261108",r["arxiv_id"],r["title"],r["published"][:4],r["doi"],r["arxiv_id"],screens[r["arxiv_id"]]["screening_status"],"candidate_classification_not_final_eligibility",EVIDENCE/"native_20261108/arxiv/records.csv")
    current=read_json(root,OLD/"current10_extraction.json")
    by_study={r["study_id"]:r for r in current}
    for study,r in sorted(by_study.items()):
        add("directed_current",study,r["title"],r["original_year"],study,study,"retained_mechanism_source_with_limitations","bounded_current_source_receipt_not_global_eligibility",OLD/"current10_extraction.json")
    for r in read_json(root,EVIDENCE/"fulltext_20261105/models.json")["models"]:
        a=arxiv_id(r["source_version"])
        add("directed_historical",r["model"],r["title"],r["publication_year"],"10.48550/arXiv."+a if a else "",a,"retained_mechanism_source_with_limitations","bounded_historical_source_receipt_not_global_eligibility",EVIDENCE/"fulltext_20261105/models.json")
    return nodes


def source_adjudication(root):
    decisions=[]
    # This is an audit of already stored source readings, not a claim that the
    # present script or a second reviewer has read the underlying manuscripts.
    by_study=defaultdict(list)
    for row in read_json(root,OLD/"current10_extraction.json"):
        by_study[row["study_id"]].append(row)
    for study, rows in sorted(by_study.items()):
        r=rows[0]
        decisions.append(dict(source_id=study,related_models=[x["model"] for x in rows],title=r["title"],
            source_url=r["source_url"],source_version=r["source_version"],evidence_pointer=str(OLD/"current10_extraction.json"),
            evidence_locator=r["metric_locator"],prior_read_level=r["read_level"],
            audit_decision="retain_for_architectural_mechanism_synthesis_with_reporting_limits",
            direct_same_contract_effectiveness=False, target=r["task"]+"; "+r["dataset"],
            reason="Different target/domain or history/horizon from BTC sigma7 D7x5 h1; source performance is not E2 evidence.",
            remaining_limit=r["critical_limit"],foundational_exception=r["original_year"]<2019,
            final_global_eligibility="not_claimed",independent_human_review=False))
    reasons={
      "E2-0012":("retain_BTC_volatility_context_deep_eligibility_uncertain",False,"five_day_forward_return_variance_and_depth_uncertain"),
      "E2-0025":("retain_BTC_volatility_context_with_reporting_limits",False,"squared_daily_return_target_not_sigma7"),
      "E2-0059":("pending_operational_target_reconciliation",None,"target_formula_daily_data_discrepancy"),
      "E2-0253":("retain_BTC_realized_variance_context",False,"intraday_sum_squared_returns_not_sigma7"),
      "E2-0867":("retain_BTC_volatility_context_version_qualification",False,"sqrt_intraday_realized_variance_not_sigma7_editorial_equivalence_pending"),
      "E2-0064":("retain_implied_volatility_context",False,"BitVol_implied_index_not_realized_sigma7"),
      "E2-0285":("pending_target_and_editorial_version_reconciliation",None,"unresolved_window_and_2022_preprint_2025_editorial_equivalence"),
      "E2-0593":("pending_operational_target_reconciliation",None,"model_implied_vs_realized_proxy_discrepancy"),
    }
    for path in [EVIDENCE/"screening_followup_20261108/source_reviews.json",EVIDENCE/"recovery_second_20261108/source_reviews.json"]:
        for r in read_json(root,path)["studies"]:
            decision,direct,reason=reasons[r["study_id"]]
            decisions.append(dict(source_id=r["study_id"],related_models=[],title=r.get("title",""),source_url=r["source_url"],
                source_version=r.get("version",r.get("access_level","")),evidence_pointer=str(path),evidence_locator=r["locator"],
                prior_read_level="prior_bounded_method_empirical_fulltext_receipt_not_replication",
                audit_decision=decision,direct_same_contract_effectiveness=direct,target=r.get("target_and_data",r.get("target","")),
                reason=reason,remaining_limit=r["limits"],foundational_exception=False,
                final_global_eligibility="not_claimed",independent_human_review=False))
    for r in read_json(root,EVIDENCE/"native_20261108/focal_source_reviews.json")["reviews"]:
        decisions.append(dict(source_id="arxiv:"+r["arxiv_id"],related_models=[],title="",source_url=r["url"],source_version=r["arxiv_id"],
            evidence_pointer=str(EVIDENCE/"native_20261108/focal_source_reviews.json"),evidence_locator=r["locator"],prior_read_level=r["read_level"],
            audit_decision="retain_definition_or_transfer_context_with_limitations",direct_same_contract_effectiveness=False,
            target=r["finding"],reason="source_is_BTC_other_volatility_definition_or_nonBTC_transfer_mechanism",
            remaining_limit=r["limitation"],foundational_exception=False,final_global_eligibility="not_claimed",independent_human_review=False))
    return decisions


def score_audit(root):
    rows=read_csv(root,OLD/"current10_support_scores.csv")
    protocol=read_json(root,OLD/"support_score_protocol.json")
    extraction={r["model"]:r for r in read_json(root,OLD/"current10_extraction.json")}
    result=[]
    for r in rows:
        e=extraction[r["model"]]
        components={k:float(r[k]) if r[k] else None for k in ("C","U","P","A","F")}
        weights=protocol["weights"]["reference"]
        known=sum(weights[k]*v for k,v in components.items() if v is not None)
        unknown=sum(weights[k] for k,v in components.items() if v is None)
        result.append(dict(model=r["model"],study_id=r["source_study_id"] or e["study_id"],family=r["family"],
            **components,citation_count=float(r["Scopus_citations"]) if r["Scopus_citations"] else None,
            citation_snapshot=r["citation_snapshot"],performance_comparable=False,
            P_unidentified_reason="No source result matches target/information/horizon/protocol; source metric cannot be ranked against different tasks.",
            P_source_dataset=e["dataset"],P_source_task=e["task"],P_source_locator=e["metric_locator"],
            score_lower=round(100*known,8),score_upper=round(100*(known+unknown),8),
            known_weight=round(1-unknown,8),unknown_weight=round(unknown,8),
            causal_selection_status="catalogue_authorized_20261112_before_support_weights_20261113",
            final_test_A2_used_for_bibliographic_score=False))
    return result


def contextual_performance_audit(root, scores):
    """New posthoc descriptive alternatives; never overwrite the old P.

    Same dataset/horizon is only nominal comparability. The stricter alternative
    additionally stratifies lookback; both still expose split/scale differences.
    A singleton stratum has unidentified P rather than an arbitrary perfect1.
    """
    historical={r["model"]:r for r in read_json(root,EVIDENCE/"fulltext_20261105/models.json")["models"]}
    model_scores={r["model"]:r for r in scores}
    current={r["model"]:r for r in read_json(root,OLD/"current10_extraction.json")}
    rows=[]
    for model in current:
        r=historical.get(model)
        if r is None:
            rows.append(dict(model=model,dataset=current[model]["dataset"],task=current[model]["task"],
                context_steps=None,horizon_steps=None,MSE=None,MAE=None,split_status=None,
                train_only_scaler_status=None,source_locator=current[model]["metric_locator"],
                context_eligibility="different_task_or_dataset_no_numeric_pooling",shared_ETTh1_H96=False,
                controlled_cross_paper_comparability_verified=False))
            continue
        pair=r["canonical_metric_pair"]
        rows.append(dict(model=model,dataset=pair["dataset"],task=pair["task"],context_steps=pair.get("context_steps"),
            horizon_steps=pair.get("horizon_steps"),MSE=float(pair["MSE"]),MAE=float(pair["MAE"]),
            split_status=r["quality"]["split"]["status"],train_only_scaler_status=r["quality"]["train_only_scaler"]["status"],
            source_locator=pair.get("locators",[]),context_eligibility="shared_dataset_and_horizon_different_or_unverified_protocol",
            shared_ETTh1_H96=True,controlled_cross_paper_comparability_verified=False))
    alternatives=[]
    for scenario in ("posthoc_nominal_ETTh1_H96", "posthoc_ETTh1_H96_stratified_by_lookback"):
        strata=defaultdict(list)
        for r in rows:
            if r["shared_ETTh1_H96"]:
                key=str(r["context_steps"]) if scenario.endswith("lookback") else "all_L"
                strata[key].append(r)
        for r in rows:
            key=str(r["context_steps"]) if scenario.endswith("lookback") else "all_L"
            peers=strata.get(key,[]) if r["shared_ETTh1_H96"] else []
            values=[x["MSE"] for x in peers]
            contextual_P=(max(values)-r["MSE"])/(max(values)-min(values)) if len(values)>1 and max(values)>min(values) else None
            old=model_scores[r["model"]]
            comps={k:old[k] for k in ("C","U","P","A","F")}
            comps["P"]=contextual_P
            weights={"C":.15,"U":.15,"P":.25,"A":.25,"F":.20}
            lo=sum(weights[k]*v for k,v in comps.items() if v is not None)
            unknown=sum(weights[k] for k,v in comps.items() if v is None)
            alternatives.append(dict(scenario=scenario,model=r["model"],stratum="ETTh1_H96_L"+key if peers else "not_pooled",
                stratum_n=len(peers),P_contextual=None if contextual_P is None else round(contextual_P,8),
                score_lower=round(100*lo,8),score_upper=round(100*(lo+unknown),8),
                score_complete_under_relaxed_rule=unknown==0,original_P_overwritten=False,
                controlled_comparability_verified=False,final_test_used=False,
                caveat="posthoc descriptive source-MSE normalization only; split/scale/HPO and uncertainty are not harmonized; no final cross-model selection"))
    protocol=dict(date=DATE,status="new_posthoc_sensitivity_not_original_selection_score",
        guide_interpretation="Guide3.2 asks comparable datasets, not exact Bitcoin sigma7; these external results may inform context without becoming BTC effectiveness evidence.",
        P_formula="(max_source_MSE_in_stratum - source_MSE)/(max_source_MSE_in_stratum-min_source_MSE_in_stratum)",
        missing_rule="P stays null for different tasks/datasets, singletons, or constant strata; no cross-task rank or zero imputation",
        scenarios=["six nominal ETTh1,H96 sources across their reported lookbacks", "ETTh1,H96 with separate L96,L512,L720 strata"],
        weights={"C":.15,"U":.15,"P":.25,"A":.25,"F":.20},
        protocol_differences=["lookback96/512/720", "TiDE declares7:1:2 while TSMixer ETT12/4/4months", "PatchTST split not reported in verified PDF", "global metric scaling/fit population unresolved in several papers", "seeds, uncertainty and equal baseline-HPO effort not harmonized"],
        original_C_U_A_F_unchanged=True,original_P_preserved=True,ranking_is_not_a_controlled_benchmark=True,
        present_catalogue_membership_not_caused_by_these_scores=True,final_test_A2_used=False)
    return rows,alternatives,protocol


def prior_review_count(root):
    prior=read_json(root,EVIDENCE/"title_abstract_review_20261108/summary.json")
    later=read_json(root,EVIDENCE/"screening_followup_20261108/summary.json")
    return prior["priority_review"]["reviewed"]+prior["exclusion_audit"]["reviewed"]+later["remaining_exclusions"]["reviewed"]


def build(root=ROOT):
    nodes=build_nodes(root)
    groups,links,conflicts=deduplicate(nodes)
    by_node={}
    reconciled=[]
    memberships=Counter()
    for members in groups:
        group_id="lit-"+hashlib.sha256("\n".join(sorted(n["node_id"] for n in members)).encode()).hexdigest()[:20]
        for node in members:
            by_node[node["node_id"]]=group_id
        corpus=sorted({n["corpus"] for n in members})
        memberships["|".join(corpus)]+=1
        reconciled.append(dict(group_id=group_id,representative_title=members[0]["title"],corpora=corpus,
            member_node_ids=sorted(n["node_id"] for n in members),DOIs=sorted({n["doi"] for n in members if n["doi"]}),
            arxiv_ids=sorted({n["arxiv"] for n in members if n["arxiv"]}),
            source_screening_decisions=[dict(node_id=n["node_id"],decision=n["screening"],reason=n["screening_reason"]) for n in members],
            global_final_eligibility="not_determined",global_inclusion_claimed=False))
    for n in nodes:
        n["unified_group_id"]=by_node[n["node_id"]]
    hist=read_json(root,EVIDENCE/"scopus_20261105/summary.json")
    focused=read_json(root,EVIDENCE/"native_20261108/native/resumen.json")
    arxiv=read_json(root,EVIDENCE/"native_20261108/arxiv/summary.json")
    # Source artifacts disagree in whether cumulative recovery is editorial;
    # prefer the later explicit correction: six editorial, two preprints.
    recovery=read_json(root,EVIDENCE/"recovery_second_20261108/summary.json")
    reviewed_count=prior_review_count(root)
    rounds=[dict(corpus="native_20261105",raw_received=hist["received_native_record_occurrences"],
                 within_round_duplicates_removed=hist["duplicate_occurrences_removed_all_stages"],
                 within_round_groups=hist["combined_bibliographic_groups"],
                 screening_kind="deterministic_rule_recommendation",global_eligibility_closed=False),
            dict(corpus="native_20261108",raw_received=focused["raw_total"],
                 within_round_duplicates_removed=focused["duplicates_removed"],within_round_groups=focused["unique_total"],
                 screening_kind="metadata_rules_plus_396_prior_AI_individual_readings",global_eligibility_closed=False),
            dict(corpus="arxiv_20261108",raw_received=arxiv["records_before_deduplication"],
                 within_round_duplicates_removed=arxiv["duplicate_occurrences_removed"],within_round_groups=arxiv["unique_records"],
                 screening_kind="21_candidate_labels_983_pending",global_eligibility_closed=False)]
    actual_counts=Counter(n["corpus"] for n in nodes)
    for round in rounds:
        if round["raw_received"]-round["within_round_duplicates_removed"] != round["within_round_groups"] or actual_counts[round["corpus"]] != round["within_round_groups"]:
            raise ValueError("Source round summary does not reconcile to actual metadata rows")
    received_nodes=[n for n in nodes if n["corpus"] not in ("directed_current","directed_historical")]
    received_groups={by_node[n["node_id"]] for n in received_nodes}
    raw=sum(r["raw_received"] for r in rounds)
    within=sum(r["within_round_duplicates_removed"] for r in rounds)
    cross=len(received_nodes)-len(received_groups)
    primary_reasons=Counter(n["screening_reason"] for n in nodes if n["corpus"]=="native_20261105")
    decisions=source_adjudication(root)
    for d in decisions:
        key=("native_20261108:"+d["source_id"] if d["source_id"].startswith("E2-") else
             "arxiv_20261108:"+d["source_id"][6:] if d["source_id"].startswith("arxiv:") else
             "directed_current:"+d["source_id"])
        d["unified_group_id"]=by_node.get(key)
        d["retrospective_adjudication_date"]=DATE
        d["new_manuscript_reading_claimed"]=False
    flow=dict(version=VERSION,date=DATE,scope="union_of_three_received_corpora_only_plus_separate_directed_source_receipts",
        native_search_universe_total=None,identified_received_occurrences=raw,within_round_duplicates_removed=within,
        cross_round_exact_identifier_or_title_year_links_removed=cross,
        received_union_bibliographic_groups=len(received_groups),directed_source_nodes=len(nodes)-len(received_nodes),
        additional_directed_groups_not_already_in_received_union=len(groups)-len(received_groups),
        union_including_directed_groups=len(groups),rounds=rounds,
        accounting_identity=f"{raw}-{within}-{cross}={len(received_groups)}",
        missing_Scopus_records_not_PRISMA_exclusions=hist["coverage"]["difference_from_screenshot_count"],
        human_independent_reviewers_asserted=0,Cohen_kappa=None,
        prior_AI_individual_title_abstract_readings_in_focused_native_round=reviewed_count,
        prior_AI_readings_not_attributable_to_every_union_group=True,
        focused_native_exclusion_cohort_individually_reviewed=337,
        focused_native_exclusion_cohort_second_round_secondary_role_criteria_harmonized=False,
        focused_native_original_40_candidate_recovery=dict(sought=recovery["examined_cumulative"],not_sought=recovery["unsearched"],
             complete_editorial=recovery["editorial_fulltext_cumulative"],complete_preprint_editorial_equivalence_pending=2,
             partial_or_unrecovered=recovery["partial_cumulative"],final_eligibility=None),
        bounded_current_primary_source_studies=9,bounded_current_model_systems=10,
        fulltext_prior_source_records_adjudicated_for_bounded_roles=len(decisions),
        final_global_eligible_studies=None,final_global_included_studies=None,final_global_exclusion_counts=None,
        global_systematic_review_complete=False,
        other_native_query_receipts="Three later query totals106/78/3 and23 metadata occurrences are not added: no verified cross-query/cross-round union.",
        caveats=["Bibliographic groups are conservatively linked metadata, not necessarily independent studies.",
          "374 absent Scopus exports are an export-coverage deficit, never a screening exclusion.",
          "Rules and AI individual reading are separately labelled; no independent human reviewer or kappa is manufactured.",
          "396 aggregate AI review receipts are restricted to the focused1078 cohort; per-record later decisions are not all public.",
          "Current mechanism-source admissibility is retrospective and does not certify a completed global review."])
    scores=score_audit(root)
    comparable,contextual,context_protocol=contextual_performance_audit(root,scores)
    excluded=read_csv(root,OLD/"excluded_originals_context.csv")
    for r in excluded:
        r["reason_is_retrospective_authorized_scope_explanation"]=True
        r["experimental_inferiority_claimed"]=False
        r["global_highest_cited_excluded_certified"]=False
    exclusions=[dict(reason=k,count=v,stage="native_20261105_machine_recommendation_not_final_exclusion") for k,v in sorted(primary_reasons.items())]
    coverage=dict(version=VERSION,criteria=dict(
        population="BTC volatility forecasting; other domains retained only with explicit structural transfer rationale",
        target_direct_effectiveness="same sigma7 definition, D7x5 historical OHLCV, horizon1 and comparable protocol required for direct E2-effectiveness pooling",
        mechanisms="Primary named architectural source; located result and enough public code/details for architectural audit; methodological limits retained",
        publication_window="2019–2026 preferred; Wang2016 explicitly retained as a foundational exception for FCN and ResNet",
        report_quality="split/scaling/seeds/repetitions/inference/baseline effort audited, not assumed from venue membership",
        exclusion_rule="Price/return/classification/implied-volatility or other definitions are not direct sigma7 effectiveness, but may be retained for context or mechanisms"),
        C_observed_systems=sum(r["C"] is not None for r in scores),C_unidentified_systems=sum(r["C"] is None for r in scores),
        P_comparable_observed_systems=0,P_unidentified_systems=10,
        known_system_components=sum(sum(r[k] is not None for k in ("C","U","P","A","F")) for r in scores),
        total_system_components=50,full_score_identified_systems=0,
        native_citation_snapshot="same received Scopus snapshot only; no cross-provider or live-date substitution",
        bibliographic_selection_test_leakage=False,score_selection_is_retrospective=True,
        completed_now=["cross-round received-corpus deduplication with row-level provenance", "stage/count/denominator reconciliation", "bounded primary-source role-adjudication receipts", "component-level score missingness and score intervals"],
        not_completed=["missing374 Scopus exports and unspecified WoS global query universe", "full union title/abstract and fulltext final eligibility", "coherent C for four current systems", "P on comparable datasets across all10 models", "historical score-based selection or blinded prospective review"],
        endpoint="Auditable received-corpus account; not a claim of guideline-complete systematic review or a fully identified literature-selected Top10")
    manifest={str(p):dict(sha256=hashlib.sha256((root/p).read_bytes()).hexdigest(),bytes=(root/p).stat().st_size) for p in INPUTS}
    results={"flow_received_corpus.json":json_bytes(flow),"rounds.csv":csv_bytes(rounds),
       "source_nodes.csv":csv_bytes(nodes),"reconciled_groups.csv":csv_bytes(reconciled),
       "cross_round_link_ledger.csv":csv_bytes(links),"title_year_identifier_conflicts.json":json_bytes(conflicts),
       "bounded_source_role_adjudication.json":json_bytes(decisions),"bounded_source_role_adjudication.csv":csv_bytes(decisions),
       "screening_reasons_not_final_exclusions.csv":csv_bytes(exclusions),"current10_score_component_audit.csv":csv_bytes(scores),
       "coverage_and_criteria.json":json_bytes(coverage),"source_manifest.json":json_bytes(manifest),
       "contextual_performance_comparability.csv":csv_bytes(comparable),
       "posthoc_contextual_score_sensitivity.csv":csv_bytes(contextual),
       "posthoc_contextual_score_protocol.json":json_bytes(context_protocol),
       "excluded_named_originals_scope_audit.csv":csv_bytes(excluded)}
    results["README.md"]=report(flow,coverage).encode()
    results["manifest.json"]=json_bytes(dict(version=VERSION,date=DATE,method="deterministic_reconciliation_of_frozen_receipts",
        script="scripts/e2_literature_final_audit.py",files={k:dict(sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for k,v in sorted(results.items())}))
    return results


def report(flow,coverage):
    return f'''# Conciliación final de literatura recibida

Auditoría retrospectiva del {DATE}. Concilia las rondas recibidas y las lecturas
ya documentadas; no declara una nueva búsqueda, una nueva lectura de todos los
artículos ni revisión independiente humana. Los artefactos del 05–13/11 permanecen
intactos. No se usan métricas del test A2 para seleccionar literatura.

## Flujo y denominadores

Los tres corpus contienen **{flow['identified_received_occurrences']:,} ocurrencias recibidas**.
Se conservan {flow['within_round_duplicates_removed']:,} duplicados resueltos dentro de sus
rondas y se detectan {flow['cross_round_exact_identifier_or_title_year_links_removed']:,} enlaces
adicionales entre rondas: **{flow['received_union_bibliographic_groups']:,} grupos bibliográficos**
recibidos. Estos grupos no equivalen a estudios elegibles. Los 19 recibos dirigidos
(9 fuentes actuales y10 históricas, con solapamiento) se contabilizan por separado:
aportan {flow['additional_directed_groups_not_already_in_received_union']} grupos adicionales.
La identidad contable es `{flow['accounting_identity']}`.

La deduplicación enlaza DOI exacto normalizado, identificador arXiv sin versión o
título exacto normalizado y año cuando falta DOI. Dos DOI distintos nunca se
fusionan sólo por el título. El registro de enlaces y los conflictos preservan
cada decisión; no se deduce equivalencia entre versiones editoriales y preprints.

**No hay PRISMA global cerrado.** Faltan374 registros del total Scopus mostrado en
la captura; no son exclusiones. Las fechas/cadenas/totales WoS globales y la
correspondencia de todos los ficheros con cada consulta no están acreditadas.
Los totales nativos posteriores106/78/3 no se suman sin unión verificable.

La ronda del05/11 tiene7833 recomendaciones automáticas, no exclusiones humanas.
Las396 lecturas individuales AI agregadas (59prioridades y337 exclusiones
propuestas) pertenecen al corpus enfocado de1078 registros del08/11. No acreditan
revisión individual del corpus completo. La clasificación de secundarios entre
rondas tampoco estaba armonizada. Se mantienen `null` los elegibles/incluidos y
las exclusiones globales finales; no se informa kappa sin revisores independientes.

De los40 candidatos de recuperación dirigida,22 fueron buscados:6 textos completos
editoriales,2 preprints con equivalencia editorial pendiente y14 parciales/no
recuperados;18 no buscados. Disponibilidad de texto no equivale a inclusión final.

## Decisiones comprobables sobre fuentes

El registro de20 fichas concilia9 fuentes actuales,8 antecedentes aplicados y3
lecturas focales arXiv. Es una adjudicación retrospectiva de su **rol** apoyada en
las lecturas previas y sus localizadores; no transforma los8 antecedentes ni los
3 preprints en estudios globalmente incluidos. FCN yResNet comparten una fuente;
Wang2016 se conserva como excepción fundacional justificada. Las métricas de
clasificación, PM2.5, ETTh1 y otras definiciones de volatilidad se mantienen fuera
de una comparación numérica directa con sigma7. Las discrepancias de objetivo o
versión que no están resueltas siguen pendientes.

## Puntaje actual y límite de identificación

Se auditan los50 componentes C/U/P/A/F de10 sistemas. Hay
**{coverage['known_system_components']}/50 observados**: C en6 sistemas y U/A/F en10;
P no es comparable en ninguno. Los4 C ausentes corresponden a3 fuentes (Wang se
comparte). Se conserva el corte Scopus recibido sin mezclar contadores de otras
fechas/proveedores. El archivo de componentes incluye los límites algebraicos,
las tareas/datasets de las fuentes y el motivo de cada P no identificado; no
imputa cero ni usa el punto medio como clasificación.

El catálogo fue autorizado el12/11; los pesos de apoyo son del13/11. Esta auditoría
**no convierte ese catálogo en una selección bibliográfica previa por puntaje**.
Completar C exige un corte coherente verificable; completar P requiere evidencia
en datasets comparables o una modificación explícita y retrospectiva del criterio,
que no sería equivalente al criterio original ni justificaría retrospectivamente
la selección. La revisión global y el puntaje plenamente observado permanecen
como limitaciones reales de cumplimiento de la guía 3.1–3.2.

La guía 3.2 pide datasets comparables y no exige que P proceda de sigma7. Por eso
se añade una **sensibilidad contextual nueva y retrospectiva**, separada del
puntaje previo: seis fuentes reportan MSE/MAE de ETTh1 a horizonte96. Se muestran
sus lookbacks96/512/720, los splits y la cobertura del escalado. Dos alternativas
normalizan descriptivamente el MSE dentro de ese grupo nominal, o dentro de cada
lookback. No se mezclan clasificación ni PM2.5 con ETTh1. La fila única L720 queda
sin P en el segundo escenario. Estas alternativas observan P en6 o5 sistemas,
pero no solucionan splits/escalas/esfuerzo desiguales: son sensibilidad, no una
comparación controlada ni una selección previa justificada retrospectivamente.
El protocolo y las20 filas se publican sin sustituir P original.

Reproducción sin red: `python scripts/e2_literature_final_audit.py --check`.
Los hashes de todas las entradas normalizadas están en `source_manifest.json`.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/OUT)
    args=parser.parse_args()
    result=build()
    if args.check:
        missing=[name for name,data in result.items() if not (args.output/name).exists() or (args.output/name).read_bytes()!=data]
        if missing:
            raise SystemExit("Changed or missing audit artifacts: "+", ".join(missing))
    else:
        args.output.mkdir(parents=True,exist_ok=True)
        for name,data in result.items():
            (args.output/name).write_bytes(data)
    flow=json.loads(result["flow_received_corpus.json"])
    print(json.dumps({"status":"verified" if args.check else "written", "union_groups":flow["received_union_bibliographic_groups"],
          "global_systematic_review_complete":False,"output":str(args.output)},ensure_ascii=False))


if __name__=="__main__":
    main()
