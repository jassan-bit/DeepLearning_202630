"""Contract tests for screening identity, evidence levels and flow accounting."""
import copy
import csv
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC=importlib.util.spec_from_file_location("native620",Path(__file__).resolve().parents[1]/"scripts/build_e1_native_620_screening.py")
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def decision(group="G1",advanced=False):
    return {"group_id":group,"decision":"advance_primary_assessment" if advanced else "exclude_title_abstract",
        "reason_code":"different_target","reason":"The reported outcome is price, without a volatility outcome.",
        "target_category":"price","evidence_level":"title_and_export_abstract","abstract_present":True,
        "reviewer":"reviewer_a","evidence_note":"S:0001 Abstract states a price-output task."}


def review(group="G1"):
    return {"group_id":group,"doi":"10.1234/example","title":"Example","authors":"Example author","year":2025,
        "reviewer":"reviewer_b","assessed_at_utc":"2026-11-08T02:00:00+00:00","document_access":"fulltext_web_html",
        "primary_sections_assessed":True,"target_quantity":"realized_variance","target_crypto":True,
        "continuous_construct_eligible":True,"architecture_reference_adequate":True,
        "reporting_metrics_adequate":True,"reporting_protocol_adequate":True,
        "dataset":"BTC daily observations","horizon":"one day","architecture_details":"One recurrent layer; width16; Adam.",
        "metrics_reported":"Table2 RMSE0.01 and MAE0.008",
        "neural_baseline_role_verified":True,"comparator_families":["LSTM","RNN_simple"],
        "independence_key":"doi:10.1234/example","disposition":"included_strict","reason":"Primary methods and tables verify both baselines.",
        "source_urls":["https://example.org/paper"],"source_locators":["Methods section; Table2"],
        "retrieval_attempts":[{"url":"https://example.org/paper","result":"full primary article accessed"}]}


class ScreeningContractTests(unittest.TestCase):
    def test_native_panel_excludes_historical_only_groups(self):
        records=[{"group_id":"G1"},{"group_id":"G2"},{"group_id":"H"}]
        occurrences=[{"occurrence_id":"A1","origin":"new_native_export","group_id":"G1"},
            {"occurrence_id":"A2","origin":"new_native_export","group_id":"G1"},
            {"occurrence_id":"A3","origin":"new_native_export","group_id":"G2"},
            {"occurrence_id":"H1","origin":"historical_export","group_id":"H"}]
        selected,fresh=M.native_panel(records,occurrences)
        self.assertEqual({r["group_id"] for r in selected},{"G1","G2"})
        self.assertEqual(len(fresh),3)

    def test_duplicate_occurrence_is_not_a_second_record(self):
        rows=[{"occurrence_id":"A1","origin":"new_native_export","group_id":"G1"}]*2
        with self.assertRaises(ValueError):M.native_panel([{"group_id":"G1"}],rows)

    def test_missing_and_duplicate_screen_decisions_fail(self):
        records=[{"group_id":"G1"},{"group_id":"G2"}]
        for rows in ([decision()],[decision(),decision()]):
            with self.assertRaises(ValueError):M.validate_decisions(records,rows,expected_count=2)

    def test_available_abstract_cannot_be_claimed_title_only(self):
        d=decision();d["evidence_level"]="title_only"
        with self.assertRaises(ValueError):M.validate_decisions([{"group_id":"G1"}],[d],expected_count=1)

    def test_missing_abstract_is_explicitly_allowed(self):
        d=decision(advanced=True);d.update(abstract_present=False,evidence_level="title_only")
        self.assertEqual(set(M.validate_decisions([{"group_id":"G1"}],[d],expected_count=1)),{"G1"})

    def test_missing_abstract_cannot_bypass_registered_retrieval(self):
        d=decision();d.update(abstract_present=False,evidence_level="title_only")
        with self.assertRaises(ValueError):M.validate_decisions([{"group_id":"G1"}],[d],expected_count=1)

    def test_unexplained_exclusion_is_rejected(self):
        d=decision();d["reason"]=" "
        with self.assertRaises(ValueError):M.validate_decisions([{"group_id":"G1"}],[d],expected_count=1)

    def test_every_advanced_candidate_needs_terminal_assessment(self):
        with self.assertRaises(ValueError):M.validate_reviews([decision(advanced=True)],[])

    def test_metadata_only_cannot_be_strict_inclusion(self):
        r=review();r["primary_sections_assessed"]=False
        with self.assertRaises(ValueError):M.validate_reviews([decision(advanced=True)],[r])

    def test_component_without_autonomous_comparator_cannot_be_included(self):
        r=review();r["comparator_families"]=[]
        with self.assertRaises(ValueError):M.validate_reviews([decision(advanced=True)],[r])

    def test_retrieval_failure_is_not_a_quality_exclusion(self):
        r=review();r.update(disposition="report_not_retrieved",primary_sections_assessed=False,comparator_families=[])
        result=M.validate_reviews([decision(advanced=True)],[r])
        self.assertEqual(result["G1"]["disposition"],"report_not_retrieved")
        r["retrieval_attempts"]=[]
        with self.assertRaises(ValueError):M.validate_reviews([decision(advanced=True)],[r])

    def test_two_reports_for_one_experiment_vote_once(self):
        a,b=review(),review("G2")
        result=M.summarize([{},{}],[{},{},{}],[decision(advanced=True),decision("G2",True)],[a,b])
        self.assertEqual(result["strict_native_included_group_reports"],2)
        self.assertEqual(result["strict_native_included_experiments"],1)
        self.assertEqual(result["strict_native_family_occurrences"],{"LSTM":1,"RNN_simple":1})
        self.assertEqual(result["strict_2019_2026_family_occurrences"],{"LSTM":1,"RNN_simple":1})

    def test_different_doi_and_aliases_preserve_previous_experiment_identity(self):
        previous=[{"id":"OLD-ARTICLE","doi":"10.1/article","identity":{"dependence_group":"EXPERIMENT-1"},
            "decision":{"guide31_reporting_sensitivity_eligible":True,"continuous_crypto_construct_corpus":True,
                        "baseline_role_verified":True},"independentfamilies":["LSTM"]}]
        a=review();a.update(doi="10.1/preprint",independence_key="OTHER-KEY",prior_review_ids=["OLD-ARTICLE"])
        b=review("G2");b.update(doi="10.1/version",independence_key="EXPERIMENT-1")
        votes,old,new,counts=M.cumulative_votes(previous,[a,b])
        self.assertEqual(len(votes),1)
        self.assertEqual(new,set())
        self.assertEqual(counts,{"LSTM":1,"RNN_simple":1})

    def test_strict_inclusion_requires_metrics_and_protocol_evidence(self):
        for key in("reporting_metrics_adequate","reporting_protocol_adequate","metrics_reported"):
            r=review();r[key]=False if key!="metrics_reported" else ""
            with self.assertRaises(ValueError):M.validate_reviews([decision(advanced=True)],[r])

    def test_all_verified_families_count_without_forcing_experimental_core(self):
        summary=M.summarize([{},{}],[{},{},{}],[decision(advanced=True),decision("G2")],[review()])
        self.assertEqual(summary["strict_native_family_occurrences"],{"LSTM":1,"RNN_simple":1})
        self.assertEqual(summary["screening_excluded"],1)
        self.assertEqual(summary["advanced_for_primary_or_metadata_enrichment"],1)
        self.assertEqual(summary["new_predictions_generated"],0)
        self.assertFalse(summary["benchmark_results_changed"])


class ClosedPanelEvidenceTests(unittest.TestCase):
    """Validate the actual dated evidence, in addition to synthetic contracts."""

    @classmethod
    def setUpClass(cls):
        cls.artifacts,cls.summary=M.build()
        cls.reviews=json.loads(cls.artifacts["primary_assessments.json"])["reviews"]

    def test_exact_export_panel_and_absence_of_unreviewed_abstracts(self):
        import io
        source=M.read_csv(M.NATIVE/"occurrences.csv")
        expected={r["group_id"] for r in source if r["origin"]=="new_native_export"}
        rows=list(csv.DictReader(io.StringIO(self.artifacts["decisions.csv"].decode())))
        self.assertEqual(len(rows),620)
        self.assertEqual({r["group_id"] for r in rows},expected)
        self.assertEqual(self.summary["received_native_occurrences"],834)
        self.assertEqual(self.summary["export_abstracts_available_and_reviewed"],610)
        self.assertEqual(self.summary["original_global_provisional_title_link_decisions_preserved"],10)
        self.assertEqual(self.summary["original_provisional_title_links_between_new_occurrences"],5)
        self.assertEqual(self.summary["native_groups_with_provisional_title_links"],8)
        missing=[r for r in rows if r["abstract_present"]=="False"]
        self.assertEqual(len(missing),10)
        self.assertTrue(all(r["screening_decision"]=="advance_primary_assessment" for r in missing))

    def test_no_advanced_candidate_is_left_without_terminal_evidence(self):
        s=self.summary
        self.assertEqual(s["advanced_for_primary_or_metadata_enrichment"],111)
        self.assertEqual(s["primary_sections_assessed_groups"],44)
        self.assertEqual(s["primary_assessments_new"],33)
        self.assertEqual(s["primary_assessments_reused"],11)
        self.assertEqual(s["abstract_enrichment_excluded"],9)
        self.assertEqual(s["reports_not_retrieved_at_cut"],58)
        self.assertEqual(s["strict_native_included_experiments"],13)
        self.assertEqual(s["cumulative_strict_experiments"],18)
        for r in self.reviews:
            self.assertNotIn("abstract",r)
            self.assertNotIn("full_text",r)
            self.assertNotIn("pdf_bytes",r)
            if r["disposition"]=="report_not_retrieved":
                self.assertFalse(r["primary_sections_assessed"])
                self.assertTrue(r["retrieval_attempts"])

    def test_hybrid_feature_only_models_receive_same_family_role_rule(self):
        bydoi={r["doi"]:r for r in self.reviews}
        seo=bydoi["10.3390/app10144768"]
        ozgur=bydoi["10.30798/makuiibf.1530528"]
        self.assertEqual(seo["comparator_families"],["HONN"])
        self.assertIn("MLP_GARCH_hybrid",seo["proposed_or_component_families"])
        self.assertEqual(ozgur["disposition"],"included_context_only")
        self.assertFalse(ozgur["neural_baseline_role_verified"])
        self.assertEqual(ozgur["comparator_families"],[])
        self.assertFalse(ozgur["family_role_adjudication"]["quality_exclusion"])

    def test_public_allowlist_includes_exact_rebuild_inputs_and_hashes(self):
        artifacts=M.with_manifest(self.artifacts)
        manifest=json.loads(artifacts["manifest.json"])
        self.assertEqual(manifest["publication_schema"],M.PUBLIC_SCHEMA)
        self.assertEqual(manifest["approved_public_files"],sorted(self.artifacts))
        for name,payload in self.artifacts.items():
            self.assertEqual(manifest["files"][name],{"sha256":M.digest(payload),"bytes":len(payload)})
        for name in ("protocol.json","completion.json","primary_reviews.json","decisions_chunk1.json",
                     "decisions_chunk2.json","decisions_chunk3.json","base_bibliography.bib"):
            self.assertEqual(self.artifacts["review_inputs/"+name],(M.SOURCE/name).read_bytes())
        self.assertEqual(self.artifacts["review_inputs/prior_adjudications.json"],M.PREVIOUS.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            M.write_artifacts(self.artifacts,directory)
            self.assertEqual({p.relative_to(directory).as_posix() for p in Path(directory).rglob("*") if p.is_file()},set(artifacts))
            Path(directory,"unexpected.pdf").write_bytes(b"unapproved neighbour")
            with self.assertRaises(ValueError):M.write_artifacts(self.artifacts,directory)


if __name__=="__main__":unittest.main()
