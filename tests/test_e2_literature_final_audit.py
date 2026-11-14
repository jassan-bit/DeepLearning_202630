"""Accounting and identity tests for an incomplete received-corpus audit."""
from importlib.util import module_from_spec, spec_from_file_location
import csv
import io
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location('final_literature_audit', ROOT/'scripts/e2_literature_final_audit.py')
audit = module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def node(ident, doi='', arxiv='', title='Same title', year='2023'):
    return dict(node_id=ident, doi=doi, arxiv=arxiv, title=title, year=year)


class ConservativeIdentityTest(unittest.TestCase):
    def test_journal_doi_year_and_article_number_is_not_arxiv_id(self):
        self.assertEqual(audit.arxiv_id('10.1016/j.ijforecast.2023.101234'), '')
        self.assertEqual(audit.arxiv_id('10.1016/j.ijforecast.2303.06053'), '')
        self.assertEqual(audit.arxiv_id('10.48550/arXiv.2303.06053'), '2303.06053')
        self.assertEqual(audit.arxiv_id('2303.06053v5'), '2303.06053')
        self.assertEqual(audit.arxiv_id('2323.06053'), '')

    def test_different_doi_title_matches_preserve_conflict(self):
        groups, links, conflicts = audit.deduplicate([
            node('A', '10.a/first'), node('B', '10.b/second'), node('C')])
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(conflicts), 1)
        self.assertFalse(links)

    def test_exact_identifier_and_missing_doi_fallback(self):
        groups,links,conflicts=audit.deduplicate([
            node('A', '10.a/first'),node('B','10.a/first'),node('C')])
        self.assertEqual(len(groups),1)
        self.assertEqual(len(links),2)
        self.assertFalse(conflicts)

    def test_different_year_no_title_only_merge(self):
        groups,_,_=audit.deduplicate([node('A'),node('B',year='2024')])
        self.assertEqual(len(groups),2)


class ReceivedCorpusAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts=audit.build(ROOT)
        cls.flow=json.loads(cls.artifacts['flow_received_corpus.json'])
        cls.coverage=json.loads(cls.artifacts['coverage_and_criteria.json'])

    def test_round_and_global_accounting(self):
        f=self.flow
        self.assertEqual(f['identified_received_occurrences'],11635)
        self.assertEqual(f['within_round_duplicates_removed'],1720)
        self.assertEqual(f['received_union_bibliographic_groups'],9559)
        self.assertEqual(f['identified_received_occurrences']-f['within_round_duplicates_removed']-
                         f['cross_round_exact_identifier_or_title_year_links_removed'],f['received_union_bibliographic_groups'])
        for r in f['rounds']:
            self.assertEqual(r['raw_received']-r['within_round_duplicates_removed'],r['within_round_groups'])

    def test_never_invents_global_eligibility_or_coverage(self):
        f=self.flow
        self.assertEqual(f['missing_Scopus_records_not_PRISMA_exclusions'],374)
        self.assertIsNone(f['final_global_eligible_studies'])
        self.assertIsNone(f['final_global_included_studies'])
        self.assertIsNone(f['final_global_exclusion_counts'])
        self.assertFalse(f['global_systematic_review_complete'])
        self.assertIsNone(f['Cohen_kappa'])

    def test_recovers_corrected_editorial_and_preprint_counts(self):
        c=self.flow['focused_native_original_40_candidate_recovery']
        self.assertEqual(c['sought']+c['not_sought'],40)
        self.assertEqual(c['complete_editorial']+c['complete_preprint_editorial_equivalence_pending']+
                         c['partial_or_unrecovered'],c['sought'])
        self.assertEqual((c['complete_editorial'],c['complete_preprint_editorial_equivalence_pending']),(6,2))

    def test_score_unknowns_not_zero_or_final_test_selection(self):
        rows=list(csv.DictReader(io.StringIO(self.artifacts['current10_score_component_audit.csv'].decode())))
        self.assertEqual(len(rows),10)
        self.assertTrue(all(r['P']=='' for r in rows))
        self.assertEqual(sum(bool(r['C']) for r in rows),6)
        self.assertTrue(all(r['final_test_A2_used_for_bibliographic_score']=='False' for r in rows))
        self.assertTrue(all(float(r['score_upper'])>float(r['score_lower']) for r in rows))
        self.assertEqual(self.coverage['known_system_components'],36)

    def test_shared_wang_source_and_prior_reading_limits(self):
        rows=json.loads(self.artifacts['bounded_source_role_adjudication.json'])
        self.assertEqual(len(rows),20)
        architectural=[r for r in rows if r['related_models']]
        self.assertEqual(len(architectural),9)
        wang=next(r for r in architectural if '1611.06455' in r['source_id'])
        self.assertEqual(set(wang['related_models']),{'FCN-1D','ResNet-1D'})
        self.assertTrue(wang['foundational_exception'])
        self.assertTrue(all(r['unified_group_id'] for r in rows))
        self.assertTrue(all(not r['new_manuscript_reading_claimed'] for r in rows))
        self.assertTrue(all(not r['independent_human_review'] for r in rows))

    def test_contextual_P_is_separate_and_preserves_singleton_unknown(self):
        rows=list(csv.DictReader(io.StringIO(self.artifacts['posthoc_contextual_score_sensitivity.csv'].decode())))
        nominal=[r for r in rows if r['scenario']=='posthoc_nominal_ETTh1_H96']
        stratified=[r for r in rows if r['scenario']=='posthoc_ETTh1_H96_stratified_by_lookback']
        self.assertEqual(sum(bool(r['P_contextual']) for r in nominal),6)
        self.assertEqual(sum(bool(r['P_contextual']) for r in stratified),5)
        self.assertEqual(next(r for r in stratified if r['model']=='TiDE')['P_contextual'],'')
        self.assertTrue(all(r['original_P_overwritten']=='False' for r in rows))
        self.assertTrue(all(r['controlled_comparability_verified']=='False' for r in rows))
        self.assertTrue(all(r['final_test_used']=='False' for r in rows))

    def test_input_scope_cannot_contain_benchmark_metrics(self):
        paths=json.loads(self.artifacts['source_manifest.json'])
        self.assertTrue(all('literature' in p for p in paths))
        self.assertFalse(any('benchmark_A2' in p or 'final_evaluation' in p for p in paths))


if __name__=='__main__':
    unittest.main()
