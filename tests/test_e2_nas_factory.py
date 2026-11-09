"""Binding and HPO-axis tests for section-27 NAS factories; synthetic tensors only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from e2_nas_common import NASConfig, validate_input, moment_features
from e2_nas_factory import MODEL_IDS, build_nas_model, build_from_plan
import e2_search_plan as planner
from check_e2_nas_factory import evaluate_case, conformance_trial

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/"book_e2/_static/search_protocol_20261109/protocol.json"


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.protocol = json.loads(PROTOCOL.read_text())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_common_contract_rejects_invalid_dimensions_bits_and_nonfinite_values(self):
        for widths, bits in [((16,32),(0,1,0)), ((15,32,64),(0,1,0)),
                             ((True,32,64),(0,1,0)), ((16,32,64),(0,2,0)),
                             ((16,32,64),(False,1,0))]:
            with self.subTest(widths=widths,bits=bits), self.assertRaises(ValueError):
                NASConfig(widths,bits)
        config = NASConfig([16,32,64],[0,1,0])
        self.assertEqual(config.widths,(16,32,64))
        for x in [torch.ones(1,6,5), torch.ones(0,7,5), torch.ones(1,7,5,dtype=torch.int64),
                  torch.full((1,7,5),float("nan")), torch.full((1,7,5),float("inf"))]:
            with self.assertRaises(ValueError):
                validate_input(x)
        x=torch.arange(35,dtype=torch.float32).reshape(1,7,5)
        torch.testing.assert_close(moment_features(x),torch.cat((x.mean(1),x.std(1,unbiased=False)),1))

    def test_only_registered_trials_and_matching_architecture_ids_are_bound(self):
        name="CNN-1D + LSTM + atención temporal"
        row=planner.enumerate_architectures(self.protocol,name)[0]
        trial=planner.generate_hpo(self.protocol,name)[0]
        with torch.device("meta"):
            model=build_from_plan(self.protocol,name,row,trial)
        self.assertEqual(model.architecture_metadata["recipe_id"],MODEL_IDS[name])
        self.assertFalse(model.architecture_metadata["training_axes_applied"])
        modified=deepcopy(trial);modified["learning_rate"]*=1.001
        with self.assertRaises(ValueError):
            build_from_plan(self.protocol,name,row,modified)
        changed=deepcopy(row);changed["architecture_id"]+="changed"
        with self.assertRaises(ValueError):
            build_from_plan(self.protocol,name,changed,trial)
        wrong=deepcopy(trial);wrong["recipe_id"]=MODEL_IDS["TiDE"]
        with self.assertRaises(ValueError):
            build_from_plan(self.protocol,name,row,wrong)

    def test_hpo_row_is_copied_and_cannot_change_constructed_model_metadata(self):
        name="CNN-1D + LSTM + atención temporal"
        row=planner.enumerate_architectures(self.protocol,name)[0]
        trial=planner.generate_hpo(self.protocol,name)[0]
        with torch.device("meta"):
            model=build_from_plan(self.protocol,name,row,trial)
        expected=deepcopy(model.architecture_metadata)
        trial["kernel"]=999
        trial["optimization_seeds"][0]=999
        self.assertEqual(model.architecture_metadata,expected)

    def test_fixed_published_models_and_unregistered_structural_options_are_rejected(self):
        with self.assertRaises(ValueError):
            build_nas_model("FCN-1D",(16,32,64),(0,1,0))
        for options in [{"dropout":.1},{"kernel":7},{"learning_rate":float("nan")}]:
            with self.assertRaises(ValueError):
                build_nas_model("CNN-1D + LSTM + atención temporal",(16,32,64),(0,1,0),hpo=options)

    def test_all_18_hpo_rows_per_family_have_numerical_reference_gradients(self):
        fixture=torch.randn(1,7,5,generator=torch.Generator().manual_seed(281108))
        for name in MODEL_IDS:
            architectures=planner.enumerate_architectures(self.protocol,name)
            reference=next(a for a in architectures if a["is_reference"])
            for trial in planner.generate_hpo(self.protocol,name):
                with self.subTest(model=name,trial=trial["trial_id"]):
                    result=evaluate_case(name,reference,protocol=self.protocol,mode="numeric",
                                         fixture=fixture,trial=trial)
                    self.assertTrue(result["all_parameter_gradients_connected"])
                    self.assertTrue(result["numerical_finiteness_checked"])

    def test_exhaustive_trial_is_an_explicit_registered_nonwinning_configuration(self):
        for name in MODEL_IDS:
            trial=conformance_trial(self.protocol,name)
            self.assertIn(trial,planner.generate_hpo(self.protocol,name))
        self.assertEqual(conformance_trial(self.protocol,"TimesNet")["num_kernels"],2)


if __name__=="__main__":
    unittest.main()

