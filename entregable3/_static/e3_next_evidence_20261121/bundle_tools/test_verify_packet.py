"""Corruption guards use only temporary artificial fixtures, never market/model execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('packet_verifier', Path(__file__).with_name('verify_packet.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)

def write(root, path, value):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value), encoding='utf-8')

class CorruptionGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def entry(self, path):
        p = self.root / path
        return {'path':path, 'bytes':p.stat().st_size, 'sha256':v.digest(p)}
    def minimal_packet(self):
        (self.root/'sample.txt').write_text('exact payload\n')
        d = {'schema_version':1, 'files':[self.entry('sample.txt')], 'components':[]}
        write(self.root,'PACKET_MANIFEST.json',d)
        return d
    def verify_minimal(self):
        with patch.object(v,'REQUIRED_COMPONENTS',set()), patch.object(v,'check_semantics',return_value={}):
            return v.verify_packet(self.root)
    def semantic_fixture(self):
        dev = dict(actual_new_optimizer_fits=20, actual_new_assessment_predictions=1210,
                   trajectories=22, recipes=6, preserved_additive_jobs=796, preserved_V3_fits=30,
                   preserved_transfer_rows=20400, new_designs=1, paired_neural_seeds=5,
                   original_V3_NO_GO_preserved=True,no_control_promoted=True,R1_inherited=False,
                   R3_accredited=False,V5_started=False,confirmation_allowed=False,
                   independent_human_execution_claimed=False,private_checkpoints_or_NPZ_payload_published=False,
                   CPU_source_manifest_sha256=v.SOURCE20_SHA,CPU_plan_public_anchor_commit=v.CPU_ANCHOR,
                   percent_candidate_worse=1.4432610846934824,decision='ABANDON_CURRENT_V4_RECIPE')
        write(self.root,'development/CURRENT_DEVELOPMENT_DECISION.json',dev)
        write(self.root,'development/cpu_pilot_results/pilot_result.json',dict(new_real_data_fits=20,
              new_assessment_predictions=1210,new_assessment_trajectories=22,new_confirmation_predictions=0,
              old796_old30_old20400_mutated=False))
        path=self.root/'development/cpu_pilot_results/assessment_predictions.csv'
        with path.open('w') as f:
            f.write('name,seed,date,target,prediction\n')
            for trajectory in range(22):
                name = 'recipe_'+str(min(trajectory//4,5))
                for date in range(55):
                    f.write(f'{name},{trajectory},{date},0.1,0.2\n')
        write(self.root,'statistics/VALIDATION_DECISION.json',dict(controlled_DGP_replicates_completed=1024,
              controlled_bootstrap_draws_completed=2096128,method_configuration_approved_for_confirmation=False,
              R3_accredited=False,MCS_implemented_extension_EPA_failure=dict(events=27,replicates=128,
              rate=27/128,nominal=.1,exploratory_tolerance_screen_passed=False)))
        write(self.root,'data_admission/MANIFEST.json',dict(datasets_admitted=0,R3_accredited=False))
        write(self.root,'data_admission/concrete_panels.DRAFT.json',dict(panels=[dict(admitted_unexposed=False)]*4))
        write(self.root,'container_profile/EXECUTION_RECEIPT.json',dict(status='book_additive19x4_and_V3_software_passed',
              human_execution_claimed=False,human_attestation_still_required=True,R2_or_R3_accredited_by_automation=False,
              source_commit=v.BASE_COMMIT,generated_appendix_HTML_verification=dict(rendered_V3_rows=7,rendered_additive_rows=76)))
        write(self.root,'planning/POST_PILOT_DISPOSITION.json',dict(current_V4_recipe_abandoned=True,
              full_family_planning_for_current_V4_authorized=False,confirmation_protocol_registered=False,
              confirmation_Colab_generated=False))
    def mutate(self,path,field,value):
        d=json.loads((self.root/path).read_text());d[field]=value;write(self.root,path,d)
    def test_intact_inventory(self):
        self.minimal_packet();self.assertEqual(self.verify_minimal()['status'],'PASS')
    def test_changed_payload_rejected(self):
        self.minimal_packet();(self.root/'sample.txt').write_text('exact payloae\n')
        with self.assertRaisesRegex(v.VerificationError,'SHA256 mismatch'):self.verify_minimal()
    def test_removed_payload_rejected(self):
        self.minimal_packet();(self.root/'sample.txt').unlink()
        with self.assertRaisesRegex(v.VerificationError,'missing'):self.verify_minimal()
    def test_unlisted_payload_rejected(self):
        self.minimal_packet();(self.root/'surprise.txt').write_text('extra')
        with self.assertRaisesRegex(v.VerificationError,'inventory mismatch'):self.verify_minimal()
    def test_duplicate_manifest_member_rejected(self):
        d=self.minimal_packet();d['files'].append(copy.deepcopy(d['files'][0]));write(self.root,'PACKET_MANIFEST.json',d)
        with self.assertRaisesRegex(v.VerificationError,'duplicate manifest'):self.verify_minimal()
    def test_traversal_rejected(self):
        with self.assertRaises(v.VerificationError):v.verify_entries(self.root,[dict(path='../outside',bytes=0,sha256='0'*64)])
    def test_symlink_rejected(self):
        self.minimal_packet();(self.root/'linked').symlink_to(self.root/'sample.txt')
        with self.assertRaisesRegex(v.VerificationError,'symlink'):self.verify_minimal()
    def test_private_weight_and_raw_payloads_rejected(self):
        for name in ['private_cases/receipt.json','x.pt','x.npz','a.whl','.git/HEAD','a/__pycache__/x.pyc']:
            with self.subTest(name=name),self.assertRaises(v.VerificationError):v.allowed_path(name)
    def test_intact_scope(self):
        self.semantic_fixture();self.assertEqual(v.check_semantics(self.root)['new_development_predictions'],1210)
    def test_zero_new_fits_relabel_rejected(self):
        self.semantic_fixture();self.mutate('development/CURRENT_DEVELOPMENT_DECISION.json','actual_new_optimizer_fits',0)
        with self.assertRaisesRegex(v.VerificationError,'count/scope'):v.check_semantics(self.root)
    def test_human_claim_relabel_rejected(self):
        self.semantic_fixture();self.mutate('container_profile/EXECUTION_RECEIPT.json','human_execution_claimed',True)
        with self.assertRaisesRegex(v.VerificationError,'human/R2/R3'):v.check_semantics(self.root)
    def test_confirmation_promotion_rejected(self):
        self.semantic_fixture();self.mutate('development/CURRENT_DEVELOPMENT_DECISION.json','confirmation_allowed',True)
        with self.assertRaisesRegex(v.VerificationError,'claim changed'):v.check_semantics(self.root)
    def test_failed_calibration_approval_rejected(self):
        self.semantic_fixture();self.mutate('statistics/VALIDATION_DECISION.json','method_configuration_approved_for_confirmation',True)
        with self.assertRaisesRegex(v.VerificationError,'validation decision'):v.check_semantics(self.root)
    def test_duplicate_prediction_rejected(self):
        self.semantic_fixture();p=self.root/'development/cpu_pilot_results/assessment_predictions.csv'
        with p.open('a') as f:f.write('recipe_0,0,0,0.1,0.2\n')
        with self.assertRaisesRegex(v.VerificationError,'duplicate prediction'):v.check_semantics(self.root)
    def test_new_family_reactivation_rejected(self):
        self.semantic_fixture();self.mutate('planning/POST_PILOT_DISPOSITION.json','full_family_planning_for_current_V4_authorized',True)
        with self.assertRaisesRegex(v.VerificationError,'reactivated'):v.check_semantics(self.root)
    def test_duplicate_JSON_keys_rejected(self):
        (self.root/'bad.json').write_text('{"scope":1,"scope":2}')
        with self.assertRaisesRegex(v.VerificationError,'duplicate JSON key'):v.read_json(self.root,'bad.json')
    def test_cli_rejects_inside_report_without_mutation(self):
        self.minimal_packet()
        before={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report=self.root/'inside-report.json'
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('verify_packet.py')),
            '--packet',str(self.root),'--output',str(report)],capture_output=True,text=True)
        self.assertEqual(result.returncode,1)
        self.assertIn('report must be outside',result.stdout)
        after={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

if __name__=='__main__':unittest.main(verbosity=2)
