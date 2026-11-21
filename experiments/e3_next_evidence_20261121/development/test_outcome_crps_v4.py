"""CPU software/analytic contracts with fixed synthetic inputs; no market data."""
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import torch

from src.e3_overlap_model_v3 import OverlapDistributionModel
from outcome_crps_v4 import (BASE_SOURCE_SHA256, C, VERSION, OutcomeCRPSDevelopmentModel,
                             outcome_loss_components, verify_base_source)
from synthetic_geometry import finite_targets, raw_head, outputs_from_head, cosine, tie_status

torch.set_num_threads(2)
LAW3 = {"id": "representable", "atoms": [0.2, 0.8, 2.0], "p": [0.2, 0.5, 0.3]}
LAW5 = {"id": "misspecified", "atoms": [0.02, 0.2, 0.75, 1.8, 4.0],
        "p": [0.1, 0.2, 0.3, 0.25, 0.15]}


def case(A=0.1, law=LAW5, atoms=(0.3, 0.8, 1.6), p=(1/3, 1/3, 1/3)):
    U, Y, us, ys = finite_targets(law, A)
    theta = raw_head(atoms, p, us)
    out = outputs_from_head(theta, A, us, len(U))
    return theta, out, Y, U, ys, us


class Contracts(unittest.TestCase):
    def test_source_pin_refuses_changed_source_before_construction(self):
        self.assertEqual(verify_base_source(), BASE_SOURCE_SHA256)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "changed.py"; path.write_text("# changed source\n")
            with self.assertRaises(ValueError): verify_base_source(path)
            link = Path(root) / "link.py"; link.symlink_to(path)
            with self.assertRaises(ValueError): verify_base_source(link)

    def test_identical_parameters_initial_values_and_original_forward(self):
        base = OverlapDistributionModel("atom_crps", u_scale=0.03, y_scale=0.01, seed=20261121)
        new = OutcomeCRPSDevelopmentModel(u_scale=0.03, y_scale=0.01, seed=20261121)
        self.assertEqual(sum(p.numel() for p in new.parameters()), 1262)
        self.assertEqual(set(base.state_dict()), set(new.state_dict()))
        for key, tensor in base.state_dict().items():
            self.assertTrue(torch.equal(tensor, new.state_dict()[key]), key)
        x = torch.linspace(-1, 1, 140, dtype=torch.float32).reshape(4, 7, 5)
        A = torch.tensor([0, 1e-8, 0.5, 1e4], dtype=torch.float64)
        with torch.no_grad():
            a, b = base.forward_components(x, A), new.forward_components(x, A)
        for key in ("prediction", "atoms", "probabilities", "sigma_atoms", "raw_head"):
            self.assertTrue(torch.equal(a[key], b[key]), key)
        self.assertTrue(bool((b["prediction"] >= torch.sqrt(A)).all()))

    def test_training_labels_do_not_enter_or_change_forward(self):
        model = OutcomeCRPSDevelopmentModel(u_scale=0.03, y_scale=0.01, seed=20261121)
        x = torch.zeros((2, 7, 5), dtype=torch.float32)
        A = torch.tensor([0, 0.0001], dtype=torch.float64)
        first = model.forward_components(x, A)
        initial = first["prediction"].detach().clone()
        model.loss_components(first, torch.tensor([0.01, 0.02]), torch.tensor([0.02, 0.03]))
        model.loss_components(first, torch.tensor([0.03, 0.04]), torch.tensor([0.05, 0.06]))
        self.assertTrue(torch.equal(model(x, A).detach(), initial))

    def test_configuration_distinguishes_new_version_and_unaccredited_status(self):
        config = OutcomeCRPSDevelopmentModel(u_scale=1, y_scale=1).configuration()
        self.assertEqual(config["version"], VERSION)
        self.assertTrue(config["scoring_coordinate_and_normalization_both_changed"])
        self.assertFalse(config["R1_inherited"]); self.assertFalse(config["R3_accredited"])
        self.assertTrue(config["original_V3_NO_GO_preserved"])
        with self.assertRaises(TypeError): OutcomeCRPSDevelopmentModel(u_scale=1, y_scale=1, lambda_crps=0.01)
        with self.assertRaises(ValueError): OutcomeCRPSDevelopmentModel("unregistered", u_scale=1, y_scale=1)

    def test_A0_normalized_scores_and_full_head_gradients_are_equal(self):
        theta, out, Y, U, ys, us = case(A=0)
        self.assertAlmostEqual(ys, math.sqrt(C) * us, places=14)
        loss = outcome_loss_components(out, Y, U, y_scale=ys, u_scale=us)
        self.assertAlmostEqual(float(loss["crps_u"].detach()), float(loss["crps_sigma"].detach()), places=13)
        a = torch.autograd.grad(loss["crps_u"], theta, retain_graph=True)[0]
        b = torch.autograd.grad(loss["crps_sigma"], theta)[0]
        self.assertTrue(torch.allclose(a, b, atol=1e-12, rtol=1e-12))

    def test_positive_normalization_changes_norm_without_changing_direction(self):
        theta, out, Y, U, ys, us = case(A=1)
        loss = outcome_loss_components(out, Y, U, y_scale=ys, u_scale=us)
        a = torch.autograd.grad(loss["crps_sigma"], theta, retain_graph=True)[0]
        b = torch.autograd.grad(loss["crps_sigma_a0matched"], theta)[0]
        self.assertTrue(torch.allclose(a, (math.sqrt(C) * us / ys) * b, atol=1e-12, rtol=1e-12))
        self.assertAlmostEqual(cosine(a, b), 1, places=12)

    def test_nonlinear_coordinate_can_change_direction_not_just_norm(self):
        theta, out, Y, U, ys, us = case(A=1)
        loss = outcome_loss_components(out, Y, U, y_scale=ys, u_scale=us)
        a = torch.autograd.grad(loss["crps_u"], theta, retain_graph=True)[0]
        b = torch.autograd.grad(loss["crps_sigma"], theta)[0]
        self.assertLess(cosine(a, b), 1 - 1e-10)

    def test_expected_scores_minimized_by_true_representable_law_in_fixed_comparison(self):
        for A in (0, 0.1, 100):
            _, truth, Y, U, ys, us = case(A, LAW3, LAW3["atoms"], LAW3["p"])
            true_loss = outcome_loss_components(truth, Y, U, y_scale=ys, u_scale=us)
            _, wrong, _, _, _, _ = case(A, LAW3, (0.1, 0.4, 1.5), (1/3, 1/3, 1/3))
            wrong_loss = outcome_loss_components(wrong, Y, U, y_scale=ys, u_scale=us)
            for key in ("crps_u", "crps_sigma", "mse_scaled"):
                self.assertGreaterEqual(float(wrong_loss[key].detach()) + 1e-12, float(true_loss[key].detach()))

    def test_gradcheck_of_sigma_objective_away_from_atom_ties(self):
        theta, _, Y, U, ys, us = case(A=0.1)
        def objective(raw):
            out = outputs_from_head(raw, 0.1, us, len(U))
            return outcome_loss_components(out, Y, U, y_scale=ys, u_scale=us)["loss"]
        self.assertTrue(torch.autograd.gradcheck(objective, (theta,), eps=1e-6, atol=1e-6, rtol=1e-4))

    def test_invalid_targets_scales_laws_and_overflow_are_refused(self):
        _, out, Y, U, ys, us = case()
        for target in (Y[:-1], Y * float("nan"), -Y):
            with self.assertRaises(ValueError): outcome_loss_components(out, target, U, y_scale=ys, u_scale=us)
        for bad in (0, -1, float("nan"), True, torch.tensor(1., requires_grad=True)):
            with self.assertRaises(ValueError): outcome_loss_components(out, Y, U, y_scale=bad, u_scale=us)
        invalid = dict(out); invalid["probabilities"] = out["probabilities"] * 2
        with self.assertRaises(ValueError): outcome_loss_components(invalid, Y, U, y_scale=ys, u_scale=us)
        with self.assertRaises(ValueError): outcome_loss_components(out, Y, U, y_scale=1e-308, u_scale=us)

    def test_MSE_control_is_preserved_point_objective_not_promoted(self):
        _, out, Y, U, ys, us = case()
        result = outcome_loss_components(out, Y, U, y_scale=ys, u_scale=us, variant="mse_only_control")
        self.assertTrue(torch.equal(result["loss"], result["mse_scaled"]))
        config = OutcomeCRPSDevelopmentModel("mse_only_control", u_scale=1, y_scale=1).configuration()
        self.assertFalse(config["market_utility_verified"]); self.assertFalse(config["R1_inherited"])

    def test_zero_gradient_direction_is_undefined_not_a_favorable_observation(self):
        self.assertIsNone(cosine(torch.zeros(5, dtype=torch.float64), torch.ones(5, dtype=torch.float64)))
        self.assertAlmostEqual(cosine(torch.ones(5, dtype=torch.float64), -torch.ones(5, dtype=torch.float64)), -1, places=13)

    def test_exact_and_roundoff_atom_ties_are_flagged_without_removing_targets(self):
        _, out, Y, U, ys, us = case(A=0.1, law=LAW3, atoms=LAW3["atoms"], p=LAW3["p"])
        self.assertTrue(tie_status(out, U, Y, us, ys)["near_tie_any_coordinate"])
        self.assertEqual(len(U), 20)
        shifted = dict(out); shifted["atoms"] = out["atoms"] + 1e-16
        self.assertTrue(tie_status(shifted, U, Y, us, ys)["near_tie_any_coordinate"])

    def test_differentiable_case_is_not_labelled_an_atom_tie(self):
        _, out, Y, U, ys, us = case(A=0.1)
        self.assertFalse(tie_status(out, U, Y, us, ys)["near_tie_any_coordinate"])


if __name__ == "__main__":
    unittest.main()
