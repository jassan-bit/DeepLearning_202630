"""No-fit/no-inference guards for the comprehensive saved-result procedure."""
import csv,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import audit_primary as audit
import run_extended_case as runner
ROOT=Path(os.environ.get('E3_REPRODUCTION_SOURCE_ROOT',str(HERE.parent.parent/'source-checkout')))

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.profile=json.loads((HERE/'SOFTWARE_PROFILE.json').read_text())
    def tearDown(self):self.tmp.cleanup()
    @unittest.skipUnless((ROOT/audit.PREFIX/'analysis/per_seed_metrics.csv').is_file(),'Set E3_REPRODUCTION_SOURCE_ROOT to the complete pinned checkout to test actual cached inputs')
    def test_full_fixed_cached_results_reaggregate(self):
        report=audit.regenerate(ROOT,self.root/'out',self.profile['audit_tolerances'])
        self.assertEqual(report['main_clean_rows'],76);self.assertEqual(report['aggregate_rows'],532)
        self.assertTrue(all(x['status']=='pass' for x in report['comparisons'].values()))
        self.assertEqual(report['published_summary']['datasets'],4);self.assertEqual(report['new_model_fits'],0)
    def test_average_tie_rank_matches_original_rule(self):
        self.assertEqual(audit.average_ranks({'a':1,'b':1,'c':2}),{'a':1.5,'b':1.5,'c':3.0})
    def test_null_audit_tolerance_blocks(self):
        p=json.loads((HERE/'HUMAN_REQUEST.template.json').read_text())
        with self.assertRaisesRegex(ValueError,'not null'):audit.validate_tolerances(p['audit_tolerances'])
    def test_missing_audit_tolerance_blocks(self):
        with self.assertRaisesRegex(ValueError,'Declare all'):audit.validate_tolerances({})
    def test_nan_audit_tolerance_blocks(self):
        self.profile['audit_tolerances'][audit.NUMERIC[0]]['absolute']=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):audit.validate_tolerances(self.profile['audit_tolerances'])
    def test_old_seven_row_scope_cannot_pass_as_main19x4(self):
        self.profile['principal_table']='V3-development-seven-recipes-section8.19'
        with self.assertRaisesRegex(ValueError,'Scope must'):runner.validate_request(self.profile,'software-validation')
    def test_software_profile_cannot_claim_human(self):
        self.profile['human_execution_claimed']=True
        with self.assertRaisesRegex(ValueError,'human=false'):runner.validate_request(self.profile,'software-validation')
    def test_base_manifest_tamper_blocks_before_case(self):
        p=self.root/'badkit';p.mkdir();(p/'MANIFEST.json').write_text('{}')
        with patch.object(runner,'BASE_KIT',p):
            with self.assertRaisesRegex(ValueError,'base-v1 manifest'):runner.run(self.root/'absent','software-validation')
    def test_numerical_difference_fails_no_dropping(self):
        fields=['dataset','value'];one=self.root/'one.csv';two=self.root/'two.csv'
        audit.write_csv(one,fields,[{'dataset':'D00','value':1}]);audit.write_csv(two,fields,[{'dataset':'D00','value':2}])
        report=audit.compare_csv(one,two,['dataset'],['value'],{'value':{'absolute':0,'relative':0}})
        self.assertEqual(report['status'],'fail');self.assertEqual(report['max_absolute_discrepancy'],1)
    def test_nonfinite_csv_blocked(self):
        fields=['dataset','value'];one=self.root/'one.csv';two=self.root/'two.csv'
        audit.write_csv(one,fields,[{'dataset':'D00','value':1}]);audit.write_csv(two,fields,[{'dataset':'D00','value':'nan'}])
        with self.assertRaisesRegex(ValueError,'Nonfinite'):audit.compare_csv(one,two,['dataset'],['value'],{'value':{'absolute':0,'relative':0}})
    def test_duplicate_key_blocked(self):
        fields=['dataset','value'];one=self.root/'one.csv';two=self.root/'two.csv'
        audit.write_csv(one,fields,[{'dataset':'D00','value':1}]);audit.write_csv(two,fields,[{'dataset':'D00','value':1}]*2)
        with self.assertRaisesRegex(ValueError,'Duplicate'):audit.compare_csv(one,two,['dataset'],['value'],{'value':{'absolute':0,'relative':0}})
    def test_missing_support_blocked(self):
        fields=['dataset','value'];one=self.root/'one.csv';two=self.root/'two.csv'
        audit.write_csv(one,fields,[{'dataset':'D00','value':1},{'dataset':'D01','value':2}]);audit.write_csv(two,fields,[{'dataset':'D00','value':1}])
        with self.assertRaisesRegex(ValueError,'row support'):audit.compare_csv(one,two,['dataset'],['value'],{'value':{'absolute':0,'relative':0}})
    def test_generated_HTML_scope_83rows_and_no_old_tables(self):
        p=self.root/'page.html';p.write_text('<table><tr></tr></table><section><span id="e3-human-generated-principal"></span><section><table>'+'<tr></tr>'*77+'</table></section><section><table>'+'<tr></tr>'*8+'</table></section></section><table><tr></tr></table>')
        self.assertEqual(runner.verify_generated_html(p)['rendered_additive_rows'],76)
        p.write_text(p.read_text().replace('<tr></tr>','',1+1))
        with self.assertRaisesRegex(ValueError,'must render76'):runner.verify_generated_html(p)

if __name__=='__main__':unittest.main(verbosity=2)
