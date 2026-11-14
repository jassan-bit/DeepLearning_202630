import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from e2_literature_cross_entregable import make_links, normalize_doi


class ExactEvidenceIndexTests(unittest.TestCase):
    def test_exact_doi_normalizes_case_prefix_only(self):
        self.assertEqual(normalize_doi('https://doi.org/10.1000/ABC'), '10.1000/abc')
        self.assertEqual(normalize_doi('same title'), '')

    def test_title_similarity_never_substitutes_identity(self):
        group = {'group_id':'g', 'representative_title':'Identical title', 'DOIs':json.dumps(['10.1000/A']), 'corpora':'["native_20261105"]'}
        review = {'doi':'10.1000/B', 'title':'Identical title', 'primary_sections_assessed':True}
        self.assertEqual(make_links([review],[group]), [])

    def test_metadata_receipt_does_not_become_primary_or_global_inclusion(self):
        group = {'group_id':'g', 'representative_title':'A', 'DOIs':json.dumps(['10.1000/A']), 'corpora':'["native_20261105"]'}
        review = {'doi':'10.1000/A', 'primary_sections_assessed':False, 'disposition':'included_strict'}
        row = make_links([review],[group])[0]
        self.assertFalse(row['prior_primary_sections_assessed'])
        self.assertFalse(row['new_E2_global_eligibility_assigned'])
        self.assertFalse(row['primary_reading_repeated'])
        self.assertEqual(row['prior_E1_disposition_not_automatically_imported'], 'included_strict')


if __name__ == '__main__': unittest.main()
