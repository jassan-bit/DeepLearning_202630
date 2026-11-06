"""Synthetic fixtures only: screening rules must not manufacture eligibility."""
from pathlib import Path
import csv
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from screen_e2_literature import (ROOT, abstracts, apply_editorial_check, candidate_table, classify_record,
    detect_features, local_transversal_hits, protocol, public_record, run,
    validate_join, write_csv)


def paper(title="A forecasting study", **kwargs):
    return {"record_id": "doi:10.1234/test", "title": title, "DOI": "10.1234/test", "year": "2024", "document_type": "Article", **kwargs}


class LiteratureScreeningTests(unittest.TestCase):
    def test_crypto_volatility_forecast_is_priority_not_final_inclusion(self):
        a = classify_record(paper("Bitcoin volatility forecasting"), "We predict realized volatility using LSTM and GRU networks.")
        self.assertEqual(a["route"], "crypto_volatility")
        self.assertEqual(a["screen_decision"], "retain_for_fulltext")
        self.assertEqual(a["priority"], 0)
        self.assertEqual(a["final_eligibility"], "pending_fulltext_adjudication")
        self.assertEqual(a["model_comparator_roles"], "not_verified")

    def test_ethereum_and_other_crypto_not_bitcoin_only(self):
        for coin in ["Ethereum", "Solana", "cryptocurrencies", "Litecoin"]:
            a = classify_record(paper(f"{coin} volatility prediction"), "A deep learning forecasting architecture.")
            self.assertEqual(a["route"], "crypto_volatility")

    def test_crypto_price_is_context_not_verified_volatility(self):
        a = classify_record(paper("Bitcoin price prediction with GRU"), "We forecast future price movements.")
        self.assertEqual(a["route"], "finance_other_target")
        self.assertEqual(a["screen_decision"], "manual_review_pending")
        self.assertEqual(a["target_role_verification"], "not_verified_from_keywords")

    def test_financial_volatility_prioritized(self):
        a = classify_record(paper("Forecasting realized stock volatility"), "A TimesNet model for equities.")
        self.assertEqual(a["route"], "financial_volatility")
        self.assertEqual(a["screen_decision"], "retain_for_fulltext")

    def test_non_deep_financial_forecaster_remains_ambiguous(self):
        a = classify_record(paper("Stock volatility forecasting"), "GARCH and HAR are compared.")
        self.assertEqual(a["screen_decision"], "manual_review_pending")
        self.assertFalse(a["features"]["deep"])

    def test_non_financial_forecaster_can_be_transferable(self):
        a = classify_record(paper("Wind power forecasting using PatchTST"), "We evaluate long term predictions at wind turbines.")
        self.assertEqual(a["route"], "transferable_forecasting")
        self.assertEqual(a["screen_decision"], "retain_for_fulltext")

    def test_homonyms_do_not_become_tide_s4_or_mamba_models(self):
        f = detect_features("Ocean tide forecasting", "Tidal dynamics from harmonic analysis.")
        self.assertNotIn("TiDE", f["models"])
        self.assertIn("TiDE", detect_features("TiDE for wind forecasting", "")['models'])
        f = detect_features("S4 superconducting phase", "The black Mamba snake feeds on mammals.")
        self.assertNotIn("S4", f["models"])
        self.assertNotIn("Mamba", f["models"])
        self.assertNotIn('S4', detect_features('CryptoMamba for volatility forecasting', 'We use structured state space representations.')['models'])

    def test_review_not_promoted_to_primary_experimental_study(self):
        a = classify_record(paper("A systematic review of Bitcoin volatility forecasting", document_type="Review"), "Deep learning models include LSTM.")
        self.assertEqual(a["screen_decision"], "manual_review_pending")
        self.assertIn("review_is_context_not_independent_primary_experiment", a["flags"])

    def test_old_and_future_years_need_date_or_foundational_review(self):
        for y in ["2017", "2027", ""]:
            a = classify_record(paper("Bitcoin volatility forecasting", year=y), "A deep neural network.")
            self.assertEqual(a["screen_decision"], "retain_for_fulltext")
            self.assertIn("publication_date_or_foundational_exception_requires_review", a["flags"])

    def test_known_sources_preserved_without_claiming_fulltext_of_support(self):
        a = classify_record(paper("Architecture"), "", known_original=True)
        self.assertEqual(a["route"], "original_architecture")
        a = classify_record(paper("Applied study"), "", known_support=True)
        self.assertEqual(a["route"], "applied_support")
        self.assertEqual(a["final_eligibility"], "pending_fulltext_adjudication")

    def test_missing_abstract_is_manual_not_silent_exclusion(self):
        a = classify_record(paper("Unknown title"), "")
        self.assertEqual(a["screen_decision"], "manual_review_pending")

    def test_unrelated_text_has_explicit_provisional_exclusion_reason(self):
        a = classify_record(paper("Crystal growth in thin films"), "We describe a chemical synthesis experiment.")
        self.assertEqual(a["screen_decision"], "exclude_title_abstract_provisional")
        self.assertTrue(a["primary_reason"].startswith("E02"))
        self.assertEqual(a["human_adjudication"], "not_performed")

    def test_three_local_queries_require_all_conjuncts(self):
        p = paper("Mechanisms")
        for text, expected in [("Self-attention transfer for image classification", "T1_attention_vision"), ("MLP-Mixer adaptation in vision", "T2_mixing_vision"), ("Mamba state space adaptation to speech", "T3_ssm_language")]:
            self.assertIn(expected, local_transversal_hits(p, text))
        self.assertEqual(local_transversal_hits(p, "Self-attention in images"), [])
        self.assertEqual(local_transversal_hits(p, "Mamba adaptation for wind forecasting"), [])
        self.assertFalse(protocol()["native_transversal_database_execution"])

    def test_join_requires_complete_unique_identity_matched_panel(self):
        a, b = paper(), paper("Second study", record_id="w2", DOI="")
        self.assertEqual(set(validate_join([a, b], [b, a])), {a['record_id'], b['record_id']})
        for private in [[a], [a, a], [a, {**b, 'title': 'Different'}], [a, {**b, 'DOI': '10.bad/changed'}]]:
            with self.assertRaises(ValueError):
                validate_join([a, b], private)

    def test_abstract_union_and_public_fields_do_not_export_full_abstract(self):
        q = {"abstract": "A", "AB": "A", "source_records": [{"raw_scientific_record": {"Abstract": "B"}}, {"raw_scientific_record": {"AB": " A "}}]}
        self.assertEqual(abstracts(q), ["A", "B"])
        secret = "A unique confidential complete abstract with numerical results 12345."
        a = classify_record(paper(), secret)
        row = public_record({**paper(), 'AB': secret, 'abstract': secret, 'raw': {'AB': secret}}, a, 'fixturehash')
        self.assertNotIn(secret, json.dumps(row))
        self.assertNotIn('AB', row)
        self.assertNotIn('raw', row)
        self.assertEqual(row['private_abstract_sha256'], 'fixturehash')

    def test_alternative_unknowns_are_null_not_zero_and_mentions_are_not_frequency(self):
        p = paper("N-HiTS for stock volatility forecasting")
        row = public_record(p, classify_record(p, "A neural model for equities."), 'hash')
        rows = candidate_table([row], {'models': []}, [], [])
        r = next(x for x in rows if x['model_or_family'] == 'N-HiTS')
        self.assertEqual(r['mention_record_groups_not_comparator_frequency'], 1)
        self.assertEqual(r['financial_volatility_candidate_groups'], 1)
        for k in ['independent_comparator_frequency', 'primary_code_verified', 'score_of_alternative', 'year_specific_quartile']:
            self.assertIsNone(r[k])

    def test_output_guard_blocks_frozen_repository_locations(self):
        # Guard is exercised before any source registry reads or mutation.
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            public, private = [paper()], [paper()]
            write_csv(t/'combined.csv', public, list(public[0]))
            (t/'private.json').write_text(json.dumps(private))
            args = SimpleNamespace(private_combined=t/'private.json', combined=t/'combined.csv', expected_groups=1,
                reference_output=ROOT/'references/literature_e2/top10_20261105', book_output=t/'out')
            with self.assertRaisesRegex(ValueError, "frozen"):
                run(args)

    def test_editorial_abstract_check_corrects_context_and_pins_evidence_hash(self):
        p = paper('Bitcoin forecasting with TimesNet')
        a = classify_record(p, 'A deep learning price predictor in a volatility forecasting setting.')
        row = public_record(p, a, 'correcthash')
        check = {'private_abstract_sha256': 'correcthash', 'target_class': 'price_return_or_trading_context', 'note': 'Price is the target; volatility is motivation.'}
        apply_editorial_check(row, check)
        self.assertEqual(row['screen_decision'], 'manual_review_pending')
        self.assertEqual(row['route'], 'finance_other_target')
        self.assertEqual(row['final_eligibility'], 'pending_fulltext_adjudication')
        self.assertIn('not_independent_human_review', row['assistant_editorial_check'])
        with self.assertRaisesRegex(ValueError, 'different abstract version'):
            apply_editorial_check(row, {**check, 'private_abstract_sha256': 'wronghash'})

    def test_end_to_end_fixture_exposes_complete_partition_and_no_abstracts(self):
        with tempfile.TemporaryDirectory() as td:
            t = Path(td)
            secret = 'PRIVATE_TEST_PHRASE_MUST_NOT_BE_PUBLISHED'
            rows = [paper('Bitcoin volatility forecasting with LSTM', native_source_occurrences='2'),
                    paper('Crystal growth', record_id='w2', DOI='', native_source_occurrences='1')]
            priv = [{**r, 'abstract': a} for r, a in zip(rows, [f'A neural network predicts volatility. {secret}', f'Chemical synthesis. {secret}'])]
            write_csv(t/'combined.csv', rows, list(rows[0]))
            (t/'private.json').write_text(json.dumps(priv))
            (t/'registry.json').write_text(json.dumps({'models': []}))
            (t/'manifest.json').write_text(json.dumps({'papers': [], 'physical_page_total': 0}))
            for name, fields in [('support.csv',['doi']), ('originals.csv',['model']), ('scores.csv',['model'])]:
                write_csv(t/name, [], fields)
            (t/'scopus.json').write_text(json.dumps({'coverage': {'received_export_occurrences': 2, 'screenshot_displayed_result_count': 3, 'difference_from_screenshot_count': 1}}))
            args = SimpleNamespace(private_combined=t/'private.json', combined=t/'combined.csv', expected_groups=2,
                registry=t/'registry.json', fulltext_manifest=t/'manifest.json', support=t/'support.csv',
                originals=t/'originals.csv', prior_scores=t/'scores.csv', scopus_summary=t/'scopus.json',
                reference_output=t/'refs', book_output=t/'public', editorial_checks=t/'missing.json')
            summary = run(args)
            self.assertEqual(summary['groups_rule_screened'], 2)
            self.assertEqual(summary['received_native_occurrences'], 3)
            self.assertEqual(summary['duplicate_occurrences_removed'], 1)
            self.assertEqual(sum(summary['screen_decisions'].values()), 2)
            self.assertEqual(summary['screen_decisions']['exclude_title_abstract_provisional'], 1)
            self.assertFalse(summary['global_systematic_review_closed'])
            self.assertIsNone(summary['included_studies_in_complete_systematic_review'])
            self.assertEqual(summary['native_transversal_queries_executed_by_this_script'], 0)
            for f in (t/'public').iterdir():
                self.assertNotIn(secret, f.read_text())


if __name__ == '__main__':
    unittest.main()
