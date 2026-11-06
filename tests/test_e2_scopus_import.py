"""Cross-source identity and source-specific citation safeguards."""
from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_e2_scopus import (canonical_group,combine,coverage,mentions,original_matches,
                             read_scopus,within_scopus)


def entry(db='Scopus',eid='e1',ut='',doi='10.1/a',title='A scientific long title about forecasting',year=2024,source='Journal One',cites=3,abstract='Mamba time series model'):
    return {'database':db,'source_file':'source.csv','record_index':1,'title':title,'abstract':abstract,'source':source,
            'year':year,'doi':doi,'ut':ut,'eid':eid,'citations':cites,'stage':'Final','early_access':'','document_type':'Article','raw':{}}


class ScopusImportTests(unittest.TestCase):
    def test_csv_bom_quoted_multiline_and_required_schema(self):
        headers=['Title','Year','Source title','Cited by','DOI','Abstract','Document Type','Publication Stage','Source','EID']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'export.csv'
            with p.open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,headers);w.writeheader();w.writerow(dict(zip(headers,['A title','2027','Journal','0','HTTPS://DOI.ORG/10.1/ABC','Line1\nLine2','Article','Article in Press','Scopus','eid1'])))
            rows,_=read_scopus(p)
            self.assertEqual(rows[0]['doi'],'10.1/abc');self.assertEqual(rows[0]['abstract'],'Line1\nLine2')
            public,_=canonical_group(rows,[0]);self.assertEqual(public['publication_window_status'],'publication_window_not_confirmed')
            self.assertEqual(public['Scopus_publication_stage'],'Article in Press');self.assertIsNone(public['Scopus_Cited_by_per_calendar_year'])
            p.write_text('Title\nwrong\n')
            with self.assertRaises(ValueError):read_scopus(p)

    def test_internal_eid_and_doi_deduplication(self):
        rows=[entry(),entry(),entry(eid='e2'),entry(eid='e3',doi='10.1/b')]
        g=within_scopus(rows)
        self.assertEqual(sorted(map(len,g.collect())),[1,3])
        self.assertEqual({e['rule'] for e in g.edges},{'same_Scopus_EID','same_normalized_DOI_inside_Scopus'})

    def test_cross_database_counts_and_citations_remain_distinct(self):
        w=entry(db='WoS',eid='',ut='w1',cites=7)
        s=entry(cites=11)
        g=combine([w],[s]);self.assertEqual(len(g.collect()),1)
        public,private=canonical_group(g.entries,g.collect()[0])
        self.assertEqual(public['WoS_TC'],7);self.assertEqual(public['Scopus_Cited_by'],11)
        self.assertFalse(public['citation_counts_summed']);self.assertNotIn('abstract',public);self.assertNotIn('AB',public)
        self.assertEqual(private['AB'],s['abstract'])

    def test_title_fallback_requires_same_year_source_and_missing_doi(self):
        w=entry(db='WoS',ut='w1',eid='',doi='')
        s=entry()
        self.assertEqual(len(combine([w],[s]).collect()),1)
        self.assertEqual(len(combine([w],[entry(source='Different journal')]).collect()),2)
        self.assertEqual(len(combine([entry(db='WoS',ut='w1',eid='',doi='10.1/b')],[s]).collect()),2)

    def test_missing_doi_bridge_does_not_join_conflicting_scopus_dois(self):
        w=entry(db='WoS',ut='w1',eid='',doi='')
        g=combine([w],[entry(doi='10.1/a'),entry(eid='e2',doi='10.1/b')])
        self.assertEqual(len(g.collect()),2);self.assertTrue(g.conflicts)

    def test_duplicate_native_eids_and_citation_values_survive_combined_group(self):
        a=entry(eid='e1',cites=4)
        b=entry(eid='e2',cites=5)
        representative=dict(a);representative['_source_aliases']=[a,b]
        g=combine([],[representative])
        public,private=canonical_group(g.entries,g.collect()[0])
        self.assertEqual(json.loads(public['EIDs']),['e1','e2'])
        self.assertEqual(json.loads(public['Scopus_Cited_by_values']),[4,5])
        self.assertEqual(public['native_source_occurrences'],2)
        self.assertEqual(len(private['source_records']),2)

    def test_partial_coverage_difference_is_not_exclusion(self):
        c=coverage(7500,7874)
        self.assertEqual(c['difference_from_screenshot_count'],374)
        self.assertFalse(c['export_is_complete_native_search_universe'])
        self.assertIn('not a PRISMA exclusion',c['note'])

    def test_primary_identity_distinguishes_tsmixer_studies_and_unknown_from_zero(self):
        m=[{'name':'TSMixer','paper_title':'TSMixer: An All-MLP Architecture for Time Series Forecasting','doi':'10.48550/arxiv.2303.06053'},
           {'name':'Informer','paper_title':'Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting','doi':'10.1609/aaai.v35i12.17325'}]
        rows=[entry(title=m[0]['paper_title'],doi='',cites=243),entry(title='TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting',doi='10.1145/3580305.3599533',cites=446)]
        result,_=original_matches(m,rows,[])
        self.assertEqual(result[0]['Cited_by_Scopus'],243);self.assertEqual(result[0]['matching_Scopus_groups'],1)
        self.assertFalse(result[1]['found_in_Scopus_export']);self.assertIsNone(result[1]['Cited_by_Scopus'])

    def test_strict_tide_and_s_mamba_aliases(self):
        self.assertNotIn('TiDE',mentions(entry(abstract='Ocean tide forecasting')))
        self.assertIn('S-Mamba',mentions(entry(abstract='The S-Mamba model uses selective state space blocks')))
        self.assertIn('Autoformer',mentions(entry(abstract='Autoformer is a reference')))


if __name__=='__main__':unittest.main()
