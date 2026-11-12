"""Scope boundaries: optional systems must be rejected before training."""
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_required_catalogue as required


class RequiredCatalogueTests(unittest.TestCase):
    def test_manifest_and_benchmark_template_share_exact_sixteen_ids(self):
        manifest = required.catalogue()
        public = required.ROOT / "book_e2/_static/published_architectures_20261112"
        self.assertEqual(json.loads((public / "catalogue.json").read_text()), manifest)
        with (public / "benchmark_A2_template.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(tuple(row["model_id"] for row in rows), required.REQUIRED)
        self.assertTrue(all(None not in row for row in rows))
        self.assertTrue(all(row["mean_rmse"] == "" for row in rows))
        self.assertEqual(len(set(required.TOP10_FAMILIES)), 4)

    def test_optional_hybrid_never_reaches_construction_or_selection(self):
        registry = SimpleNamespace(build_train_model=Mock())
        bindings = SimpleNamespace(run_neural_selection=Mock())
        with patch.dict(sys.modules, {"e2_published_registry": registry,
                                      "e2_published_training": bindings}):
            with self.assertRaises(ValueError):
                required.build_required_neural_model(required.OPTIONAL_HISTORICAL[0], {})
            with self.assertRaises(ValueError):
                required.run_required_neural_selection(None, required.OPTIONAL_HISTORICAL[0], None)
        registry.build_train_model.assert_not_called()
        bindings.run_neural_selection.assert_not_called()

    def test_required_neural_selection_keeps_recipe_arguments(self):
        bindings = SimpleNamespace(run_neural_selection=Mock(return_value="receipt"))
        with patch.dict(sys.modules, {"e2_published_training": bindings}):
            result = required.run_required_neural_selection("prepared", "TSMixer", "adapter",
                                                            hpo_trials=("trial",), k=40)
        self.assertEqual(result, "receipt")
        bindings.run_neural_selection.assert_called_once_with("prepared", "TSMixer", "adapter",
                                                               hpo_trials=("trial",), k=40)

    def test_optional_hybrid_cannot_reenter_through_final_refits(self):
        bindings = SimpleNamespace(run_selected_refits=Mock(return_value="fresh-fits"))
        selection = SimpleNamespace(winner=SimpleNamespace(payload={"model": required.OPTIONAL_HISTORICAL[0]}))
        with patch.dict(sys.modules, {"e2_published_training": bindings}):
            with self.assertRaises(ValueError):
                required.run_required_neural_refits(None, selection, None, (1, 2))
            bindings.run_selected_refits.assert_not_called()
            selection.winner.payload["model"] = "MLP16"
            self.assertEqual(required.run_required_neural_refits("prepared", selection, "adapter", (1, 2)),
                             "fresh-fits")
        bindings.run_selected_refits.assert_called_once_with("prepared", selection, "adapter", (1, 2))

    def test_required_classical_selection_cannot_request_ridge(self):
        module = SimpleNamespace(select_classical=Mock(return_value="selected"))
        with patch.dict(sys.modules, {"e2_required_classical_training": module}):
            self.assertEqual(required.select_required_classical("panel", "prepared", source_commit="source"), "selected")
            with self.assertRaises(ValueError):
                required.select_required_classical("panel", "prepared", include_ridge=True)
        module.select_classical.assert_called_once_with("panel", "prepared", include_ridge=False,
                                                        source_commit="source")

    def test_package_import_resolves_package_training_bindings(self):
        packaged = importlib.import_module("scripts.e2_required_catalogue")
        bindings = SimpleNamespace(run_neural_selection=Mock(return_value="package-receipt"))
        with patch.dict(sys.modules, {"scripts.e2_published_training": bindings}):
            self.assertEqual(packaged.run_required_neural_selection(None, "MLP16", None),
                             "package-receipt")
        bindings.run_neural_selection.assert_called_once_with(None, "MLP16", None)

    def test_historical_four_control_selection_cannot_refit_in_required_mode(self):
        module = SimpleNamespace(refit_classical=Mock())
        historical = SimpleNamespace(ridge_alpha=1.0, selected_scores={name: 1.0 for name in (*required.CLASSICAL, "Ridge")})
        valid = SimpleNamespace(ridge_alpha=None, selected_scores={name: 1.0 for name in required.CLASSICAL})
        with patch.dict(sys.modules, {"e2_required_classical_training": module}):
            with self.assertRaises(ValueError):
                required.refit_required_classical(None, None, historical)
            required.refit_required_classical("panel", "prepared", valid, source_commit="source")
        module.refit_classical.assert_called_once_with("panel", "prepared", valid, source_commit="source")

    def test_optional_duplicate_and_classical_neural_ids_are_rejected(self):
        for names in (("Ridge",), ("TSMixer", "TSMixer"), (), "TSMixer"):
            with self.assertRaises(ValueError):
                required.require_models(names)
        with self.assertRaises(ValueError):
            required.require_models(("ARIMA",), neural_only=True)

    def test_cli_only_reports_metadata_and_migration_is_explicit(self):
        result = subprocess.run([sys.executable, str(Path(required.__file__))],
                                check=True, capture_output=True, text=True)
        value = json.loads(result.stdout)
        self.assertEqual(value["counts"]["total"], 16)
        self.assertFalse(value["amendment"]["private_colab_session_modified"])
        self.assertEqual(value["amendment"]["required_selection_controller_migration"],
                         "pending_before_a_campaign_using_the_new_scope")


if __name__ == "__main__":
    unittest.main()
