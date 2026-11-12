"""Recipe identity, HPO opportunity and conditional NAS boundaries."""
import json
import os
from pathlib import Path
import sys
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import torch
import e2_published_registry as registry
import e2_training_models as historical


class PublishedRegistryTests(unittest.TestCase):
    def test_thirteen_neural_systems_nine_defined_top10_only_tide_nas(self):
        policy = registry.policy()
        rows = policy["models"]
        self.assertEqual(len(rows), 13)
        self.assertEqual([r["model_id"] for r in rows if r["nas_required"]], ["TiDE"])
        self.assertEqual(sum(not r["nas_required"] for r in rows[:10]), 9)
        public = registry.ROOT / "book_e2/_static/published_architectures_20261112/architecture_policy.json"
        self.assertEqual(json.loads(public.read_text()), policy)
        with self.assertRaises(ValueError):
            registry.model_policy("CNN-1D + LSTM + atención temporal")

    def test_restorations_keep_eighteen_optimizer_points_but_never_the_old_recipe_id(self):
        for name, (recipe_id, dropout) in registry.RESTORED.items():
            with self.subTest(model=name):
                old = historical.registered_trials(name)
                new = registry.registered_trials(name)
                self.assertEqual(len(new), 18)
                self.assertEqual(len({row["trial_id"] for row in new}), 18)
                for before, after in zip(old, new):
                    self.assertEqual(after["recipe_id"], recipe_id)
                    self.assertNotEqual(before["trial_id"], after["trial_id"])
                    for key in ("learning_rate", "weight_decay", "batch_size", "optimization_seeds"):
                        self.assertEqual(before[key], after[key])
                    self.assertEqual(after["dropout"], dropout)
                    self.assertNotIn("architecture_reference", after)
                    self.assertNotIn("num_kernels", after)
                    self.assertNotIn("top_k", after)

    def test_unchanged_recipes_include_tide_and_all_e1_references(self):
        for name in set(registry.NEURAL_NAMES) - set(registry.RESTORED):
            with self.subTest(model=name):
                self.assertEqual(registry.registered_trials(name), historical.registered_trials(name))

    def test_old_nas_row_overrides_and_implicit_seeds_fail_before_construction(self):
        name = "LSTM-autoencoder con predictor"
        with self.assertRaises(ValueError):
            registry.build_train_model(name, historical.registered_trials(name)[0], seed=1)
        with self.assertRaises(ValueError):
            registry.build_train_model(name, registry.registered_trials(name)[0],
                                       architecture={"widths": [16, 16, 16]}, seed=1)
        with self.assertRaises(ValueError):
            registry.build_train_model(name, registry.registered_trials(name)[0])

    def test_fixed_sae_widths_and_source_dropout_cannot_be_hpo_resized(self):
        trial = registry.registered_trials("LSTM-autoencoder con predictor")[0]
        before = torch.random.get_rng_state().clone()
        model = registry.build_train_model("LSTM-autoencoder con predictor", trial, seed=123)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual((model.encoder1.hidden_size, model.encoder2.hidden_size), (25, 26))
        self.assertEqual((model.dropout1.p, model.dropout2.p), (.3, .3))
        self.assertFalse(model.architecture_metadata["nas"])
        self.assertFalse(hasattr(model, "dense"))

    def test_tide_empty_covariate_probe_does_not_justify_an_exemption(self):
        """Transcribe the source residual equations, not a full TensorFlow model."""
        with torch.random.fork_rng(devices=[]), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")
            torch.manual_seed(27)
            hidden = torch.nn.Linear(0, 64)
            output = torch.nn.Linear(64, 4)
            skip = torch.nn.Linear(0, 4)
            for layer in (hidden, output, skip):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            optimizer = torch.optim.Adam([*hidden.parameters(), *output.parameters(), *skip.parameters()], lr=.01)
            x = torch.empty(8, 0)
            expected = [1.0, .9604000449180603, .9216222167015076, .8836802840232849]
            for score in expected:
                optimizer.zero_grad(set_to_none=True)
                prediction = output(torch.relu(hidden(x))) + skip(x)
                loss = (prediction - 1).square().mean()
                self.assertAlmostEqual(float(loss.detach()), score, places=6)
                loss.backward()
                self.assertEqual(float(hidden.bias.grad.abs().sum()), 0)
                self.assertEqual(float(output.weight.grad.abs().sum()), 0)
                self.assertGreater(float(output.bias.grad.abs().sum()), 0)
                optimizer.step()
        self.assertTrue(registry.model_policy("TiDE")["nas_required"])

    @unittest.skipUnless(os.environ.get("E2_PUBLISHED_SOURCE_ROOTS"), "Pinned source integration requires explicit roots")
    def test_all_thirteen_current_bodies_accept_only_observed_d7_windows(self):
        roots = json.loads(Path(os.environ["E2_PUBLISHED_SOURCE_ROOTS"]).read_text())
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for name in registry.NEURAL_NAMES:
                with self.subTest(model=name):
                    architecture = historical.reference_architecture(name) if name == "TiDE" else None
                    model = registry.build_train_model(name, registry.registered_trials(name)[0],
                        architecture=architecture, source_roots=roots, seed=271201).eval()
                    with torch.no_grad():
                        prediction = model(torch.randn(2, 7, 5))
                    self.assertEqual(tuple(prediction.shape), (2,))
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertTrue((prediction >= 0).all())
                    if name in registry.RESTORED:
                        self.assertEqual(model.architecture_metadata["training_recipe_id"], registry.RESTORED[name][0])
        finally:
            torch.set_num_threads(previous)


if __name__ == "__main__":
    unittest.main()
