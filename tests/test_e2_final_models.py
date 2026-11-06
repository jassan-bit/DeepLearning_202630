"""Inherited recipe and optimizer-group contracts; no market data."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2_final_models import ANCHORS, architecture_spec, build_model, optimizer_groups


class CatalogueTests(unittest.TestCase):
    def test_three_anchors_retain_parameter_identity_and_positive_output(self):
        for name, recipe in ANCHORS.items():
            model = build_model(name).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), recipe["parameters"])
            prediction = model(torch.randn(3, 7, 5))
            self.assertEqual(tuple(prediction.shape), (3,))
            self.assertTrue(torch.isfinite(prediction).all() and (prediction >= 0).all())
            self.assertEqual(architecture_spec(model)["role"], "inherited_fixed_anchor")

    def test_anchor_cannot_receive_another_dropout_or_learning_rate(self):
        with self.assertRaises(ValueError):
            build_model("GRU/D7-c03", candidate={"dropout": .2})
        with self.assertRaises(ValueError):
            build_model("LSTM/D7-c02", candidate={"learning_rate": .001})

    def test_retuned_base_is_distinct_and_accepts_candidate_dropout(self):
        model = build_model("GRU-E2", candidate={"parameters": {"dropout": .25}})
        self.assertEqual(architecture_spec(model)["role"], "retuned_E2_base")
        self.assertEqual(model.dropout_probability, .25)
        with self.assertRaises(ValueError):
            build_model("TCN-E2", candidate={"width": 16})

    def test_model_mismatch_and_classical_misrouting_rejected(self):
        with self.assertRaises(ValueError):
            build_model("GRU-E2", candidate={"model": "LSTM-E2", "parameters": {}})
        with self.assertRaises(ValueError):
            build_model("ARIMA")

    def test_source_no_decay_is_explicit_and_each_parameter_occurs_once(self):
        model = nn.Linear(2, 1)
        model.bias._no_weight_decay = True
        groups, receipt = optimizer_groups(model, .02)
        self.assertEqual(receipt["source_excluded_names"], ["bias"])
        self.assertEqual([g["weight_decay"] for g in groups], [.02, 0.])
        ids = [id(p) for group in groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(model.weight), id(model.bias)})

    def test_bias_has_no_automatic_exemption(self):
        groups, receipt = optimizer_groups(nn.Linear(2, 1), .01)
        self.assertEqual(receipt["source_excluded_names"], [])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["params"]), 2)

    def test_lesion_is_reinstalled_when_rebuilding_a_checkpoint_model(self):
        torch.manual_seed(3)
        model = build_model("TSMixer", candidate={"dropout": 0., "ablation_id": "without_moments"})
        saved = model.state_dict()
        restored = build_model("TSMixer", candidate={"dropout": 0., "ablation_id": "without_moments"})
        restored.load_state_dict(saved)
        model.eval(); restored.eval()
        x = torch.randn(3, 7, 5)
        torch.testing.assert_close(model(x), restored(x))
        self.assertEqual(architecture_spec(restored)["ablation_id"], "without_moments")
        self.assertTrue(architecture_spec(restored)["ablation_requires_retraining"])

    def test_inherited_anchor_cannot_be_silently_lesioned(self):
        with self.assertRaises(ValueError):
            build_model("GRU/D7-c03", candidate={"ablation_id": "without_moments"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
