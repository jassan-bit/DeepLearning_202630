"""Synthetic-only tests of UNREGISTERED prototypes, never market evaluation."""
from datetime import date, timedelta
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from e2_proposed_models import build_prototype
from e2_proposed_temporal import har_features, validate_training_scope, residual_training_targets, OOFResidual

NAMES = ('FCN1D', 'ResNet1D', 'Inception1D', 'LSTM-AE', 'HAR-residual')


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.x = torch.randn(4, 7, 5)

    def predict(self, model, name, x):
        return model(x, torch.ones(len(x))) if name == 'HAR-residual' else model(x)

    def test_shape_batch_invariance_gradients_and_roundtrip(self):
        for name in NAMES:
            with self.subTest(name=name):
                model = build_prototype(name, 8).eval()
                y = self.predict(model, name, self.x)
                self.assertEqual(y.shape, (4,))
                self.assertTrue(torch.isfinite(y).all() and (y >= 0).all())
                single = self.predict(model, name, self.x[:1])
                torch.testing.assert_close(y[:1], single)
                loss = y.square().mean()
                if name == 'LSTM-AE':
                    reconstruction = model.reconstruct(self.x)
                    self.assertEqual(reconstruction.shape, self.x.shape)
                    loss = loss + (reconstruction - self.x).square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                restored = build_prototype(name, 8).eval()
                restored.load_state_dict(model.state_dict())
                torch.testing.assert_close(y, self.predict(restored, name, self.x))

    def test_reject_wrong_contract_and_nonfinite(self):
        for name in NAMES:
            model = build_prototype(name)
            for x in (torch.zeros(4, 8, 5), torch.zeros(4, 7, 6),
                      torch.full((4, 7, 5), float('nan'))):
                with self.assertRaises(ValueError):
                    self.predict(model, name, x)

    def test_tiny_synthetic_fit_decreases_loss(self):
        # Training only synthetic constants, no model selection or performance claim.
        for name in NAMES:
            with self.subTest(name=name):
                torch.manual_seed(19)
                model = build_prototype(name, 8)
                optimizer = torch.optim.Adam(model.parameters(), lr=.01)
                target = torch.full((4,), 1.5)
                initial = (self.predict(model, name, self.x) - target).square().mean().item()
                for _ in range(60):
                    optimizer.zero_grad()
                    loss = (self.predict(model, name, self.x) - target).square().mean()
                    loss.backward(); optimizer.step()
                final = (self.predict(model, name, self.x) - target).square().mean().item()
                self.assertLess(final, initial * .1)

    def test_negative_hybrid_link_has_zero_gradient_and_is_not_training_ready(self):
        model = build_prototype('HAR-residual', 8)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.correction[-1].bias.fill_(-2.)
        output = model(self.x, torch.ones(len(self.x)))
        self.assertTrue((output == 0).all())
        (output - 1.).square().mean().backward()
        # Documents the limitation, not evidence of an acceptable training rule.
        self.assertTrue((model.correction[-1].bias.grad == 0).all())

    def test_ae_reconstruction_pretrain_decreases_loss(self):
        model = build_prototype('LSTM-AE', 8)
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        initial = (model.reconstruct(self.x) - self.x).square().mean().item()
        for _ in range(120):
            optimizer.zero_grad()
            loss = (model.reconstruct(self.x) - self.x).square().mean()
            loss.backward(); optimizer.step()
        self.assertLess(loss.item(), initial * .6)
        self.assertIsNone(model.head.weight.grad)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.origin = date(2024, 6, 1)
        self.history = {self.origin - timedelta(days=i): float(i) for i in range(1, 23)}

    def test_har_uses_22_calendar_days_and_no_future(self):
        features = har_features(self.history, self.origin)
        self.assertEqual(features, (1., 4., 11.5))
        self.history[self.origin] = float('nan')
        self.history[self.origin + timedelta(days=1)] = 9999
        self.assertEqual(features, har_features(self.history, self.origin))

    def test_gap_not_compressed_and_d7_insufficient(self):
        del self.history[self.origin - timedelta(days=10)]
        with self.assertRaises(ValueError):
            har_features(self.history, self.origin)
        with self.assertRaises(ValueError):
            har_features({d: v for d, v in self.history.items() if v <= 7}, self.origin)

    def test_fold_local_pretraining_rejects_validation(self):
        train = [self.origin - timedelta(days=i) for i in range(1, 20)]
        validate_training_scope(train, train)
        with self.assertRaises(ValueError):
            validate_training_scope(train + [self.origin], train)

    def test_oof_targets_reject_leakage_and_unavailable_labels(self):
        valid = OOFResidual(self.origin, self.origin - timedelta(days=8), self.origin, 2., 3.)
        self.assertEqual(residual_training_targets([valid], [self.origin], self.origin), (1.,))
        for row in (OOFResidual(self.origin, self.origin, self.origin, 2., 3.),
                    OOFResidual(self.origin, valid.fit_label_end, self.origin + timedelta(days=1), 2., 3.)):
            with self.assertRaises(ValueError):
                residual_training_targets([row], [self.origin], self.origin)
        with self.assertRaises(ValueError):
            residual_training_targets([valid], [self.origin - timedelta(days=1)], self.origin)

    def test_oof_generator_matches_list_and_rejects_leakage(self):
        valid = OOFResidual(self.origin, self.origin - timedelta(days=8), self.origin, 2., 3.)
        self.assertEqual(residual_training_targets(iter([valid]), [self.origin], self.origin), (1.,))
        invalid = OOFResidual(self.origin, self.origin, self.origin, 2., 3.)
        with self.assertRaises(ValueError):
            residual_training_targets((r for r in [invalid]), [self.origin], self.origin)
        with self.assertRaises(ValueError):
            residual_training_targets(iter([]), [self.origin], self.origin)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
