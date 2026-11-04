"""Synthetic model-contract checks; no market or held-out dataset is opened."""
from __future__ import annotations

import json
import unittest

import torch

from scripts.dl_models import architecture_spec, build_model, count_parameters


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_common_input_and_nonnegative_scalar_output(self) -> None:
        for family in ("MLP", "LSTM", "CNN1D"):
            for history in (7, 14, 21, 28):
                for batch in (1, 4):
                    with self.subTest(family=family, history=history, batch=batch):
                        model = build_model(family, history, 16).eval()
                        result = model(torch.randn(batch, history, 5))
                        self.assertEqual(result.shape, (batch,))
                        self.assertTrue(torch.isfinite(result).all())
                        self.assertTrue((result >= 0).all())

    def test_parameter_counts_match_architecture_formulas(self) -> None:
        for history, width, features in ((7, 16, 5), (28, 32, 5), (4, 3, 2)):
            expected = {
                "MLP": history * features * width + 2 * width + 1,
                "LSTM": 4 * width * (features + width + 2) + width + 1,
                "CNN1D": 3 * width * (features + width) + 3 * width + 1,
            }
            for family, total in expected.items():
                with self.subTest(family=family, history=history, width=width):
                    model = build_model(family, history, width, input_features=features)
                    self.assertEqual(count_parameters(model),
                                     {"trainable": total, "non_trainable": 0, "total": total})
                    metadata = architecture_spec(model)
                    self.assertEqual(metadata["parameters"]["total"], total)
                    json.dumps(metadata)

    def test_frozen_parameters_are_not_counted_as_trainable(self) -> None:
        model = build_model("MLP", 7, 16)
        parameter = next(model.parameters())
        frozen = parameter.numel()
        total = count_parameters(model)["total"]
        parameter.requires_grad_(False)
        self.assertEqual(count_parameters(model),
                         {"trainable": total - frozen, "non_trainable": frozen, "total": total})

    def test_convolution_features_do_not_use_future_positions(self) -> None:
        torch.manual_seed(91)
        model = build_model("CNN1D", 14, 8, dropout=0).eval()
        observed = torch.randn(2, 14, 5)
        reference = model.encode(observed)
        self.assertEqual(reference.shape, (2, 8, 14))
        for cutoff in (1, 4, 9, 13):
            changed = observed.clone()
            changed[:, cutoff:, :] += torch.randn_like(changed[:, cutoff:, :]) * 100
            actual = model.encode(changed)
            self.assertTrue(torch.equal(reference[:, :, :cutoff], actual[:, :, :cutoff]), cutoff)

    def test_initialization_and_training_step_repeat_with_same_seed(self) -> None:
        values = torch.linspace(-1, 1, 4 * 7 * 5).reshape(4, 7, 5)
        targets = torch.tensor([.2, .7, 1.0, 1.3])
        for family in ("MLP", "LSTM", "CNN1D"):
            with self.subTest(family=family):
                snapshots = []
                for _ in range(2):
                    torch.manual_seed(77)
                    model = build_model(family, 7, 8).train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
                    initial = {name: value.clone() for name, value in model.state_dict().items()}
                    result = model(values)
                    loss = torch.nn.functional.mse_loss(result, targets)
                    loss.backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                    optimizer.step()
                    self.assertTrue(any(not torch.equal(initial[name], value)
                                        for name, value in model.state_dict().items()))
                    snapshots.append({name: value.clone() for name, value in model.state_dict().items()})
                for name, value in snapshots[0].items():
                    self.assertTrue(torch.equal(value, snapshots[1][name]), (family, name))

    def test_architecture_metadata_does_not_advance_random_generator(self) -> None:
        for family in ("MLP", "LSTM", "CNN1D"):
            model = build_model(family, 7, 8)
            before = torch.get_rng_state().clone()
            architecture_spec(model)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_invalid_contracts_are_rejected(self) -> None:
        for kwargs in ({"history_days": 0}, {"width": -1}, {"input_features": 0},
                       {"dropout": 1}, {"dropout": float("nan")}):
            config = {"history_days": 7, "width": 8, **kwargs}
            with self.subTest(config=config), self.assertRaises(ValueError):
                build_model("MLP", **config)
        with self.assertRaises(ValueError):
            build_model("Transformer", 7, 8)
        for family in ("MLP", "LSTM", "CNN1D"):
            model = build_model(family, 7, 8)
            for values in (torch.ones(2, 7), torch.ones(2, 8, 5),
                           torch.ones(2, 7, 4), torch.ones(0, 7, 5)):
                with self.subTest(family=family, shape=tuple(values.shape)), self.assertRaises(ValueError):
                    model(values)
            with self.assertRaises(TypeError):
                model(torch.ones(2, 7, 5, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
