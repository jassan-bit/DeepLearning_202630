"""Software-only checks; no market acquisition, fitting or test-partition reads."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('input_audit',HERE/'audit_power_inputs.py')
audit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(audit)
SOURCE=Path('/workspace/e3_engine_published_source_cache/source-c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c')


def arguments(**kwargs):
    value=dict(source_root=str(SOURCE),expected_source_commit=audit.SOURCE_COMMIT,
        expected_manifest_sha256=audit.SOURCE_MANIFEST_SHA256,design=None,predictions=[],receipts=[],
        panel_map=None,verify_ledgers=False,output='unused')
    value.update(kwargs)
    return argparse.Namespace(**value)


class AuditTests(unittest.TestCase):
    def test_strictly_positive_risk_counterexample_reverses_order(self):
        value=audit.endpoint_counterexample()
        self.assertFalse(value['historical_candidate_better'])
        self.assertTrue(value['prospective_candidate_better'])
        self.assertTrue(all(x>0 for x in value['seed_MSE_candidate']+value['seed_MSE_rival']))
        self.assertAlmostEqual(value['historical_mean_seed_RMSE_candidate'],1.1)
        self.assertAlmostEqual(value['historical_mean_seed_RMSE_rival'],1.05)

    def test_exhaustive_composition_is_not_a_new_calibration(self):
        value=audit.deterministic_composition_check()
        self.assertEqual(value['current_two_of_three_checks'],192)
        self.assertEqual(value['alternative_all_three_guard_checks'],240)
        self.assertFalse(value['alternative_is_implemented'])
        self.assertTrue(any(len(w['fixed_null_panel_witness'])==2 for w in value['current_rule_null_witnesses']))
        self.assertEqual(value['nominal_bound_if_valid_marginal_size'],.05)

    def test_archived_confidence_is_not_upgraded_to_joint_or_nominal(self):
        report=audit.read(SOURCE/'qa/statistical_validation/v4_precision/report.json')
        value=audit.method_scope(report)
        self.assertTrue(value['composite_no_pooling'])
        self.assertEqual(value['earlier_full_null_preserved_excluded'],512)
        self.assertAlmostEqual(value['worst_pointwise_upper'],.04786087313588523)
        self.assertAlmostEqual(value['worst_two_panel_union_diagnostic'],.09572174627177046)
        self.assertFalse(value['joint_error_control_certified_by_these_diagnostics'])

    def test_existing_frozen_source_is_verified_and_no_final_values_imputed(self):
        value=audit.audit(arguments())
        self.assertEqual(value['source_files_verified'],288)
        self.assertEqual(value['final_status'],'NOT_DIMENSIONED')
        self.assertFalse(value['planning_inputs_valid'])
        self.assertEqual(value['planning_contract']['expected_successful_jobs'],340)
        self.assertEqual(value['planning_contract']['minimum_physical_CSV_rows'],27200)
        self.assertEqual(value['family']['Holm_positive_slots_per_panel'],120)
        self.assertEqual(value['family']['additional_EPA_Holm_slots'],180)
        for key in ('final_R','final_T','final_B','achieved_power','MDE_achieved'):
            self.assertIsNone(value[key])
        self.assertEqual(value['new_market_acquisitions'],0)
        self.assertEqual(value['new_model_fits'],0)
        self.assertFalse(value['reserved_test_opened'])

    def test_changed_scientific_design_rejected_before_study(self):
        design=audit.read(SOURCE/'configs/power_design.json')
        design['alpha']=.05
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'changed.json';path.write_text(json.dumps(design))
            with self.assertRaisesRegex(ValueError,'Scientific power settings differ'):
                audit.audit(arguments(design=str(path)))

    def test_data_only_design_metadata_amendment_does_not_change_science(self):
        design=audit.read(SOURCE/'configs/power_design.json')
        design['panel_instrument_proposals']={'F00':'BTCUSDT','F01':'BTCUSDT-PERP','F02':'ETHUSDT-PERP','F03':'BTCUSD-PERP'}
        design['version']='metadata-amendment-software-fixture'
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dataonly.json';path.write_text(json.dumps(design))
            result=audit.audit(arguments(design=str(path)))
            self.assertEqual(result['scientific_power_design_sha256'],audit.SCIENTIFIC_DESIGN_SHA256)
            self.assertFalse(result['planning_inputs_valid'])

    def test_manifest_corruption_fails_before_module_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'SOURCE_MANIFEST.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest differs'):
                audit.frozen_modules(tmp)

    def test_test_or_fixture_panel_is_rejected_without_partition_read(self):
        class ForbiddenData:
            def read_admission(self,path):
                return dict(status='software_fixture',partition_paths={'test':'test.npz'})
            def load_partition(self,*args,**kwargs):
                raise AssertionError('no partition read allowed')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'admission.json').write_text('{}')
            path=root/'map.json';path.write_text(json.dumps({p:str(root) for p in ('F00','F01','F02','F03')}))
            value=audit.panel_status(path,ForbiddenData())
            self.assertEqual(len(value['errors']),4)
            self.assertEqual(value['development_verified'],[])
            self.assertFalse(value['test_opened'])

    def test_audit_output_never_overwrites_prior_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audit.json'
            audit.save(path,{'preserve':'original'})
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'retained'):
                audit.save(path,{'changed':True})
            self.assertEqual(path.read_bytes(),before)

    def test_planning_tail_and_final_prefix_are_disjoint_and_jobs_not_pseudoreplicated(self):
        source,power,_,_,_=audit.frozen_modules(SOURCE)
        config=audit.read(source/'configs/power_design.json')
        value=audit.contract(config,power)
        self.assertFalse(set(value['planning_seed_ids']) & set(power.seed_prefix(config,4091)))
        jobs=value['planning_jobs']
        self.assertEqual(sum(j['kind']=='neural' for j in jobs),320)
        self.assertEqual(sum(j['kind']=='arima' for j in jobs),4)
        self.assertEqual(sum(j['kind']=='analytic' for j in jobs),16)
        self.assertAlmostEqual(value['nominal_planning_budget']['nominal_neural_active_hours'],10+40/60)

    def test_final_gate_preserves_development_and_blocks_unresolved_confirmation(self):
        gate=audit.read(HERE/'final_gate.json')
        self.assertTrue(gate['data_admission_allowed'])
        self.assertTrue(gate['development_planning_allowed_with_explicit_activation'])
        self.assertFalse(gate['final_confirmation_enabled'])
        self.assertEqual(gate['unresolved_final_gate_codes'],[
            'PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED','JOINT_DECISION_VALIDATION_NOT_ESTABLISHED'])


if __name__=='__main__':
    unittest.main(verbosity=2)
