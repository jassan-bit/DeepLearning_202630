"""Synthetic contracts for the portable current-model reproduction adapter."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np
import torch
import dl_models_gru_lstm_tcn as models
import run_gru_lstm_tcn as runner


class PublicModelsContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_current_families_only_and_registered_parameter_counts(self):
        expected = {"LSTM": (32, 5025), "GRU": (32, 3777), "TCN": (16, 4289)}
        for family, (width, count) in expected.items():
            model = models.build_model(family, 7, width)
            self.assertEqual(models.count_parameters(model)["trainable"], count)
            self.assertEqual(models.architecture_spec(model)["family"], family)
        for excluded in ("MLP", "CNN1D", "unknown"):
            with self.assertRaises(ValueError):
                models.build_model(excluded, 7, 16)

    def test_all_histories_finite_positive_outputs_and_gradients(self):
        for family, width in (("LSTM", 32), ("GRU", 32), ("TCN", 16)):
            for history in (7, 14, 21, 28):
                with self.subTest(family=family, history=history):
                    torch.manual_seed(4101)
                    model = models.build_model(family, history, width)
                    values = torch.randn(4, history, 5)
                    output = model(values)
                    self.assertEqual(output.shape, (4,))
                    self.assertTrue(torch.isfinite(output).all() and (output > 0).all())
                    ((output - .8) ** 2).mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters()))

    def test_gru_tcn_causality_and_tcn_full_history_support(self):
        for family in ("GRU", "TCN"):
            torch.manual_seed(921)
            model = models.build_model(family, 28, 8).eval()
            values = torch.randn(2, 28, 5)
            changed = values.clone()
            changed[:, 13:, :] += 100
            with torch.no_grad():
                a, b = model.encode(values), model.encode(changed)
            torch.testing.assert_close(a[:, :, :13], b[:, :, :13], rtol=0, atol=0)
        tcn = models.build_model("TCN", 28, 4, dropout=0).eval()
        with torch.no_grad():
            for parameter in tcn.parameters():
                parameter.fill_(.1)
        values = torch.ones(1, 28, 5, requires_grad=True)
        encoded = tcn.encode(values)
        encoded[:, :, 10].sum().backward(retain_graph=True)
        self.assertTrue(torch.equal(values.grad[:, 11:], torch.zeros_like(values.grad[:, 11:])))
        values.grad.zero_()
        encoded[:, :, -1].sum().backward()
        self.assertTrue((values.grad.abs().sum(dim=-1) > 0).all())
        self.assertEqual(tcn.receptive_field, 29)

    def test_state_checkpoint_roundtrip_and_historical_lstm_identity(self):
        from dl_models import SingleLayerLSTM
        self.assertIs(models.SingleLayerLSTM, SingleLayerLSTM)
        for family, width in (("LSTM", 32), ("GRU", 32), ("TCN", 16)):
            torch.manual_seed(4102)
            original = models.build_model(family, 14, width).eval()
            buffer = io.BytesIO()
            torch.save({"family": family, "history_days": 14, "width": width,
                        "dropout": .1, "state_dict": original.state_dict()}, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=True)
            restored = models.build_model(checkpoint["family"], checkpoint["history_days"],
                                          checkpoint["width"], checkpoint["dropout"]).eval()
            restored.load_state_dict(checkpoint["state_dict"], strict=True)
            values = torch.randn(5, 14, 5)
            torch.testing.assert_close(original(values), restored(values), rtol=0, atol=0)

    def test_legacy_optimizer_accepts_current_models_with_synthetic_data(self):
        import train_dl_benchmark as fitting
        previous = fitting.build_model, fitting._PROTOCOL
        rng = np.random.default_rng(456)
        x = rng.normal(size=(24, 7, 5)).astype(np.float32)
        y = (.7 + .1 * abs(x[:, -1, 3])).astype(np.float32)
        try:
            fitting.build_model = models.build_model
            fitting._PROTOCOL = {"training": {"dropout": .1, "weight_decay": .0001,
                "batch_size": 8, "max_epochs": 3, "gradient_clip_norm": 1.,
                "min_delta": .00001, "patience": 2}}
            for family in models.FAMILIES:
                job = {"model": family, "history_days": 7, "width": 8,
                       "seed": 4101, "learning_rate": .001}
                _, epoch, curve, _ = fitting.optimization_fit(job, x[:16], y[:16], x[16:], y[16:])
                self.assertTrue(1 <= epoch <= 3)
                self.assertTrue(all(np.isfinite(row["validation_mse_scaled_eval"]) for row in curve))
        finally:
            fitting.build_model, fitting._PROTOCOL = previous

    def test_bad_shapes_and_invalid_parameters_fail(self):
        for family in models.FAMILIES:
            model = models.build_model(family, 7, 8)
            for values in (torch.zeros(0, 7, 5), torch.zeros(2, 8, 5),
                           torch.zeros(2, 7, 5, dtype=torch.int64)):
                with self.assertRaises((TypeError, ValueError)):
                    model(values)
            with self.assertRaises(ValueError):
                models.build_model(family, 7, 8, float("nan"))


class PortableEvidenceContracts(unittest.TestCase):
    def test_manifest_restores_after_moving_its_entire_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            original = base / "first"
            original.mkdir()
            (original / "evidence.csv").write_text("x,y\n1,2\n")
            keys = ("artifact_sha256", "file_sha256", "files_sha256", "files")
            for key in keys:
                manifest = {key: {"evidence.csv": runner.digest(original / "evidence.csv")}}
                (original / f"manifest_{key}.json").write_text(json.dumps(manifest))
            moved = base / "relocated"
            original.rename(moved)
            for key in keys:
                result = runner.verify_evidence_manifest(moved / f"manifest_{key}.json")
                self.assertEqual(result["artifacts_verified"], 1)
                self.assertFalse(result["training_performed"])
            (moved / "evidence.csv").write_text("modified")
            for key in keys:
                with self.assertRaises(ValueError):
                    runner.verify_evidence_manifest(moved / f"manifest_{key}.json")

    def test_manifest_cannot_escape_root_or_follow_an_external_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "artifact.csv"
            source.write_text("public\n")
            inner = base / "inner"
            inner.mkdir()
            for name in ("../artifact.csv", str(source.resolve())):
                with self.assertRaises(ValueError):
                    runner.verify_mapping({name: runner.digest(source)}, inner)
            (inner / "escape.csv").symlink_to(source)
            with self.assertRaises(ValueError):
                runner.verify_mapping({"escape.csv": runner.digest(source)}, inner)

    def test_default_validation_does_not_fit_or_create_an_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new_reproduction"
            stream = io.StringIO()
            with mock.patch.object(sys, "argv", ["run_gru_lstm_tcn.py", "--project-root", temporary,
                                                  "--output-dir", str(output)]), \
                 mock.patch.object(runner, "validate_inputs", return_value={}) as validate, \
                 mock.patch.object(runner, "run_cv") as train, contextlib.redirect_stdout(stream):
                runner.main()
            validate.assert_called_once()
            train.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(json.loads(stream.getvalue())["training_performed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
