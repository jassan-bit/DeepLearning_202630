"""Contract/fixture tests, with no model/data acquisition or human attestation."""
import csv, importlib.util, json, math, tempfile, unittest
from pathlib import Path
import prepare_external as prepare
import reproduce_primary as primary
import run_book_case as runner
KIT=Path(__file__).resolve().parent

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.profile=json.loads((KIT/'SOFTWARE_PROFILE.json').read_text())
    def tearDown(self):self.tmp.cleanup()
    def predictions(self,change):
        with (KIT/'references/assessment_predictions.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        rows=change(rows)
        p=self.root/'changed.csv'
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['name','seed','date','target','prediction']);w.writeheader();w.writerows(rows)
        return p
    def test_pinned_saved_points_full_support(self):
        out=self.root/'primary.csv';result=primary.regenerate(KIT/'references/assessment_predictions.csv',out)
        self.assertEqual(result['prediction_rows'],935);self.assertFalse(result['original_two_percent_rule_passed'])
        report=primary.compare(KIT/'references/primary_V3.csv',out,self.profile['tolerances'])
        self.assertEqual(report['status'],'pass');self.assertLessEqual(report['max_absolute_discrepancy'],1e-15)
    def test_modified_prediction_hash_blocks(self):
        p=self.predictions(lambda rows:[dict(rows[0],prediction='0.1')]+rows[1:])
        with self.assertRaisesRegex(ValueError,'SHA256'):primary.regenerate(p,self.root/'out.csv')
    def test_duplicate_point_blocks_even_without_transport_check(self):
        p=self.predictions(lambda rows:rows+[rows[0]])
        with self.assertRaisesRegex(ValueError,'Duplicate'):primary.regenerate(p,self.root/'out.csv',verify_hash=False)
    def test_missing_point_blocks_even_without_transport_check(self):
        p=self.predictions(lambda rows:rows[:-1])
        with self.assertRaisesRegex(ValueError,'Incomplete'):primary.regenerate(p,self.root/'out.csv',verify_hash=False)
    def test_nonfinite_point_blocks(self):
        p=self.predictions(lambda rows:[dict(rows[0],prediction='nan')]+rows[1:])
        with self.assertRaisesRegex(ValueError,'Nonfinite'):primary.regenerate(p,self.root/'out.csv',verify_hash=False)
    def test_changed_shared_target_blocks(self):
        p=self.predictions(lambda rows:[dict(rows[0],target='0.1')]+rows[1:])
        with self.assertRaisesRegex(ValueError,'Targets differ'):primary.regenerate(p,self.root/'out.csv',verify_hash=False)
    def test_unknown_recipe_blocks(self):
        p=self.predictions(lambda rows:[dict(rows[0],name='new_posthoc')]+rows[1:])
        with self.assertRaisesRegex(ValueError,'Unexpected recipe'):primary.regenerate(p,self.root/'out.csv',verify_hash=False)
    def test_null_tolerance_blocks(self):
        request=prepare.request_template()
        with self.assertRaisesRegex(ValueError,'tolerances'):runner.validate_request(request,'human-external')
    def test_missing_external_operator_blocks(self):
        request=prepare.request_template();request['tolerances']=self.profile['tolerances']
        with self.assertRaisesRegex(ValueError,'own identity'):runner.validate_request(request,'human-external')
    def test_automation_cannot_claim_human(self):
        self.profile['human_execution_claimed']=True
        with self.assertRaisesRegex(ValueError,'human=false'):runner.validate_request(self.profile,'software-validation')
    def test_comparison_nonfinite_blocks(self):
        p=self.root/'bad.csv';text=(KIT/'references/primary_V3.csv').read_text();p.write_text(text.replace('0.002073832857860484','nan'))
        with self.assertRaisesRegex(ValueError,'Nonfinite'):primary.compare(KIT/'references/primary_V3.csv',p,self.profile['tolerances'])
    def test_comparison_missing_row_blocks(self):
        p=self.root/'bad.csv';p.write_text('\n'.join((KIT/'references/primary_V3.csv').read_text().splitlines()[:-1])+'\n')
        with self.assertRaisesRegex(ValueError,'row support'):primary.compare(KIT/'references/primary_V3.csv',p,self.profile['tolerances'])
    def test_schema_difference_blocks(self):
        p=self.root/'bad.csv';p.write_text((KIT/'references/primary_V3.csv').read_text().replace('mean_individual_MAE','new_MAE'))
        with self.assertRaisesRegex(ValueError,'schema'):primary.compare(KIT/'references/primary_V3.csv',p,self.profile['tolerances'])
    def test_exact_support_field_difference_fails(self):
        p=self.root/'bad.csv';p.write_text((KIT/'references/primary_V3.csv').read_text().replace('neural,3,55','neural,3,54',1))
        self.assertEqual(primary.compare(KIT/'references/primary_V3.csv',p,self.profile['tolerances'])['status'],'fail')
    def test_existing_output_blocks_before_book_execution(self):
        case=self.root/'case';(case/'output').mkdir(parents=True);(case/'output/old').write_text('fixture')
        # run() first verifies the real kit manifest; no book or subprocess is
        # executed after the new-case/output guard rejects this fixture.
        with self.assertRaisesRegex(ValueError,'Output must be empty'):runner.run(case,'software-validation')
    def test_symlink_case_ancestor_blocks_before_source_copy(self):
        source=self.root/'source';source.mkdir();link=self.root/'link';link.symlink_to(source,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlink case path/ancestor'):prepare.prepare(source,link/'new-case',KIT)
        self.assertFalse((source/'new-case').exists())

if __name__=='__main__':unittest.main(verbosity=2)
