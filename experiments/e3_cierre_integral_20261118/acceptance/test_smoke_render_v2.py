"""Acceptance-client contracts; no browser, service, or application imports.

The fixture copies the frozen serving artifacts so corruption tests exercise
the real reference loader, including the posterior sidecar's predictor binding.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("smoke_render_v2.py")
E3 = "CNN-LSTM-atencion-aditiva E3"
REVISION = "abcdef0123456789abcdef0123456789abcdef01"
INVALID = (ValueError, RuntimeError)


def load_client():
    spec = importlib.util.spec_from_file_location("e3_smoke_render_v2_contracts", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load_client()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def healthy(**changes):
    result = {"status": "ok", "service": "e3-dashboard", "catalogue_systems": 17,
              "source_commit": REVISION, "source_commit_source": "RENDER_GIT_COMMIT"}
    result.update(changes)
    return result


class ImportContracts(unittest.TestCase):
    def test_client_import_needs_neither_browser_nor_application(self):
        original_import = builtins.__import__
        forbidden = {"app", "dash", "e3_dashboard", "numpy", "onnxruntime", "playwright"}

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                raise AssertionError(f"Smoke client imported {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard):
            module = load_client()
        self.assertTrue(callable(module.load_references))


class FrozenReferenceContracts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        for relative in ("artifacts/e3_dashboard_v1", "artifacts/e3_serving_extensions_v2",
                         "artifacts/e3_serving_calibration_v2", "data/e3_dashboard_v1",
                         "experiments/e3_cierre_integral_20261118/serving/export_capabilities/tsmixer"):
            shutil.copytree(ROOT / relative, self.root / relative)

    def test_real_reference_set_contains_five_predictors_and_e3_posterior_interval(self):
        example, models = smoke.load_references(root=self.root)
        self.assertEqual(example["id"], "evaluation-0")
        self.assertEqual(example["target_date"], "2025-01-01")
        self.assertEqual(hashlib.sha256(example["csv_text"].encode()).hexdigest(), example["csv_sha256"])
        self.assertEqual({m["id"] for m in models}, {"MLP16", "LSTM32", "GRU32", E3, "TSMixer"})
        self.assertEqual(len(models), 5)
        for model in models:
            with self.subTest(model=model["id"]):
                self.assertTrue(math.isfinite(model["prediction"]))
                self.assertGreaterEqual(model["prediction"], 0)
                self.assertIsInstance(model["seed"], int)
                self.assertRegex(model["onnx_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(model["metadata_sha256"], r"^[0-9a-f]{64}$")
                self.assertTrue(model["has_interval"])
                self.assertEqual(model["interval_kind"], "posterior_90_percent" if model["id"] == E3 else "nominal_90_percent")

    def test_duplicate_model_id_across_base_and_extension_is_rejected(self):
        path = self.root / "artifacts/e3_serving_extensions_v2/manifest.json"
        manifest = read_json(path)
        manifest["models"][0]["id"] = "MLP16"
        write_json(path, manifest)
        with self.assertRaises(INVALID):
            smoke.load_references(root=self.root)

    def test_registry_path_traversal_is_rejected(self):
        path = self.root / "artifacts/e3_serving_extensions_v2/manifest.json"
        manifest = read_json(path)
        manifest["models"][0]["slug"] = "../e3_serving_extensions_v2/tsmixer"
        write_json(path, manifest)
        with self.assertRaises(INVALID):
            smoke.load_references(root=self.root)

    def test_tampered_metadata_is_rejected(self):
        path = self.root / "artifacts/e3_dashboard_v1/mlp16/metadata.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(INVALID):
            smoke.load_references(root=self.root)

    def test_tampered_onnx_is_rejected(self):
        path = self.root / "artifacts/e3_serving_extensions_v2/tsmixer/model.onnx"
        path.write_bytes(path.read_bytes() + b"\x00")
        with self.assertRaises(INVALID):
            smoke.load_references(root=self.root)

    def test_sidecar_wrong_seed_is_rejected_even_with_updated_manifest_hash(self):
        root = self.root / "artifacts/e3_serving_calibration_v2"
        manifest_path = root / "manifest.json"
        manifest = read_json(manifest_path)
        row = manifest["models"][0]
        path = root / row["file"]
        profile = read_json(path)
        profile["seed"] += 1
        write_json(path, profile)
        raw = path.read_bytes()
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        write_json(manifest_path, manifest)
        with self.assertRaises(INVALID):
            smoke.load_references(root=self.root)


class HealthIdentityContracts(unittest.TestCase):
    def test_public_identity_accepts_case_insensitive_full_revision(self):
        identity = smoke.validate_health(healthy(source_commit=REVISION.upper()), REVISION, public=True)
        self.assertEqual(identity["source_commit"], REVISION)
        self.assertTrue(identity["service_reported_commit_matches_operator"])
        self.assertFalse(identity["remote_commit_identity_verified"])
        self.assertFalse(identity["cryptographic_attestation"])

    def test_public_identity_requires_operator_revision(self):
        for expected in (None, ""):
            with self.subTest(expected=expected), self.assertRaises(INVALID):
                smoke.validate_health(healthy(), expected, public=True)

    def test_missing_revision_is_allowed_only_for_unpinned_local_preflight(self):
        health = healthy()
        health.pop("source_commit")
        health.pop("source_commit_source")
        identity = smoke.validate_health(health, None, public=False)
        self.assertFalse(identity["remote_commit_identity_verified"])
        for public, expected in ((True, REVISION), (False, REVISION)):
            with self.subTest(public=public), self.assertRaises(INVALID):
                smoke.validate_health(health, expected, public=public)

    def test_different_revision_is_rejected_even_for_local_pinned_check(self):
        for public in (True, False):
            with self.subTest(public=public), self.assertRaises(INVALID):
                smoke.validate_health(healthy(source_commit="0" * 40), REVISION, public=public)

    def test_health_must_identify_ready_service_and_complete_catalogue(self):
        for changes in ({"status": "starting"}, {"service": "another-service"},
                        {"catalogue_systems": 16}, {"catalogue_systems": "17"}):
            with self.subTest(changes=changes), self.assertRaises(INVALID):
                smoke.validate_health(healthy(**changes), REVISION, public=True)

    def test_nonhex_abbreviated_or_untrusted_revision_evidence_is_rejected(self):
        for changes in ({"source_commit": "a" * 39}, {"source_commit": "g" * 40},
                        {"source_commit_source": "operator_assertion"},
                        {"source_commit_source": None}):
            with self.subTest(changes=changes), self.assertRaises(INVALID):
                smoke.validate_health(healthy(**changes), REVISION, public=True)

    def test_malformed_operator_revision_is_rejected(self):
        for expected in ("a" * 39, "g" * 40):
            with self.subTest(expected=expected), self.assertRaises(INVALID):
                smoke.validate_health(healthy(), expected, public=True)

    def test_app_revision_environment_is_valid_identity_evidence(self):
        identity = smoke.validate_health(healthy(source_commit_source="APP_SOURCE_COMMIT"), REVISION, public=True)
        self.assertTrue(identity["service_reported_commit_matches_operator"])
        self.assertFalse(identity["remote_commit_identity_verified"])

    def test_health_cannot_self_attest_without_a_revision(self):
        health = healthy()
        health.pop("source_commit")
        health.pop("source_commit_source")
        health["remote_commit_identity_verified"] = True
        identity = smoke.validate_health(health, None, public=False)
        self.assertFalse(identity["remote_commit_identity_verified"])
        with self.assertRaises(INVALID):
            smoke.validate_health(health, REVISION, public=True)


class ExplanationContracts(unittest.TestCase):
    def explanation(self):
        return {"type": "bar", "labels": [f"t−{i}" for i in range(7, 0, -1)],
                "effects": [0.001, -0.002, 0.0, 0.003, -0.004, 0.005, 0.006]}

    def test_seven_ordered_lag_occlusions_are_accepted(self):
        smoke.validate_explanation(self.explanation())

    def test_missing_reordered_or_duplicated_lags_are_rejected(self):
        for labels in ([f"t−{i}" for i in range(6, 0, -1)],
                       [f"t−{i}" for i in range(1, 8)], ["t−7"] * 7):
            explanation = self.explanation()
            explanation["labels"] = labels
            with self.subTest(labels=labels), self.assertRaises(INVALID):
                smoke.validate_explanation(explanation)

    def test_nonfinite_incomplete_and_nonbar_explanations_are_rejected(self):
        for changes in ({"effects": [0.0] * 6}, {"effects": [float("nan")] + [0.0] * 6},
                        {"effects": [float("inf")] + [0.0] * 6}, {"type": "heatmap"}):
            explanation = self.explanation()
            explanation.update(changes)
            with self.subTest(changes=changes), self.assertRaises(INVALID):
                smoke.validate_explanation(explanation)


class IntervalTextContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, models = smoke.load_references(root=ROOT)
        cls.models = {m["id"]: m for m in models}
        cls.profile = read_json(ROOT / "artifacts/e3_serving_calibration_v2/e3_additive_271801.json")

    def interval_text(self, model_id, *, offset=0.0, upper_offset=0.0, posterior=None, coverage=True):
        model = self.models[model_id]
        if model_id == E3:
            q = self.profile["q_sigma7"]
        else:
            manifest_root = "e3_serving_extensions_v2" if model_id == "TSMixer" else "e3_dashboard_v1"
            manifest = read_json(ROOT / "artifacts" / manifest_root / "manifest.json")
            slug = next(m["slug"] for m in manifest["models"] if m["id"] == model_id)
            q = read_json(ROOT / "artifacts" / manifest_root / slug / "metadata.json")["calibration_q"]
        lower = max(0.0, model["prediction"] - q)
        upper = model["prediction"] + q
        posterior = model_id == E3 if posterior is None else posterior
        label = "INTERVALO POSTERIOR NOMINAL 90%" if posterior else "INTERVALO NOMINAL 90%"
        text = f"{label}\n[{lower + offset:.6f}, {upper + upper_offset:.6f}]"
        if coverage and model_id == E3:
            text += "\n" + self.profile["coverage_statement"]
        return text

    def test_all_five_displayed_intervals_match_frozen_quantiles(self):
        for model_id, model in self.models.items():
            with self.subTest(model=model_id):
                smoke.validate_interval_text(self.interval_text(model_id), model, 1e-6)

    def test_e3_must_use_posterior_label_and_full_coverage_scope(self):
        for text in (self.interval_text(E3, posterior=False), self.interval_text(E3, coverage=False)):
            with self.subTest(text=text), self.assertRaises(INVALID):
                smoke.validate_interval_text(text, self.models[E3], 1e-6)

    def test_interval_numeric_parity_rejects_material_endpoint_error(self):
        for model_id in (E3, "TSMixer"):
            for offsets in ({"offset": 1e-4}, {"upper_offset": 1e-4}):
                with self.subTest(model=model_id, offsets=offsets), self.assertRaises(INVALID):
                    smoke.validate_interval_text(self.interval_text(model_id, **offsets), self.models[model_id], 1e-6)

    def test_interval_label_alone_does_not_certify_endpoints(self):
        text = "INTERVALO POSTERIOR NOMINAL 90%\n" + self.profile["coverage_statement"]
        with self.assertRaises(INVALID):
            smoke.validate_interval_text(text, self.models[E3], 1e-6)


class ReceiptContracts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / "receipt.json"

    def args(self, public=True, revision=REVISION):
        options = ["--url", "https://e3-demo.onrender.com" if public else "http://localhost:8050",
                   "--output", str(self.output), "--deployment-commit", revision]
        if not public:
            options.append("--allow-local")
        return smoke.parser().parse_args(options)

    @staticmethod
    def ready(base, seconds, report, *, expected_commit, public):
        report["deployment_identity"] = smoke.validate_health(healthy(), expected_commit, public)

    @staticmethod
    def complete_browser(base, args, report, example, models):
        report["pages"] = [{"path": path, "HTTP": 200} for path in ("/", "/eda", "/modelos")]
        report["models"] = [{"model": model["id"], "status": "passed"} for model in models]

    def assert_receipt_saved(self, receipt):
        self.assertEqual(read_json(self.output), receipt)
        self.assertFalse(receipt["remote_commit_identity_verified"])
        self.assertEqual(receipt["public_deployment_smoke_passed"], receipt["render_public_smoke_passed"])
        self.assertEqual(receipt["new_fits"], 0)
        self.assertEqual(receipt["new_calibrations"], 0)

    def test_complete_public_run_records_operator_match_without_attestation(self):
        with patch.object(smoke, "wake_service", side_effect=self.ready), patch.object(smoke, "browser_smoke", side_effect=self.complete_browser):
            receipt = smoke.run(self.args())
        self.assert_receipt_saved(receipt)
        self.assertEqual(receipt["status"], "passed")
        self.assertTrue(receipt["render_public_smoke_passed"])
        self.assertTrue(receipt["deployment_identity"]["service_reported_commit_matches_operator"])
        self.assertFalse(receipt["deployment_identity"]["cryptographic_attestation"])
        self.assertEqual(set(receipt["expected_live_models"]), {"MLP16", "LSTM32", "GRU32", E3, "TSMixer"})

    def test_local_success_never_certifies_a_public_render_deployment(self):
        with patch.object(smoke, "wake_service", side_effect=self.ready), patch.object(smoke, "browser_smoke", side_effect=self.complete_browser):
            receipt = smoke.run(self.args(public=False, revision=""))
        self.assert_receipt_saved(receipt)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["scope"], "local_preflight_not_public_deployment")
        self.assertFalse(receipt["render_public_smoke_passed"])

    def test_unavailable_service_writes_failed_receipt_without_starting_browser(self):
        with patch.object(smoke, "wake_service", side_effect=RuntimeError("readiness deadline")), patch.object(smoke, "browser_smoke") as browser:
            receipt = smoke.run(self.args())
        self.assert_receipt_saved(receipt)
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["render_public_smoke_passed"])
        browser.assert_not_called()

    def test_four_predictions_cannot_satisfy_five_model_acceptance(self):
        def incomplete_browser(base, args, report, example, models):
            self.complete_browser(base, args, report, example, models[:-1])

        with patch.object(smoke, "wake_service", side_effect=self.ready), patch.object(smoke, "browser_smoke", side_effect=incomplete_browser):
            receipt = smoke.run(self.args())
        self.assert_receipt_saved(receipt)
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["render_public_smoke_passed"])

    def test_failed_running_prediction_is_preserved_as_failed_in_receipt(self):
        def failed_browser(base, args, report, example, models):
            report["models"] = [{"model": E3, "status": "running"}]
            raise RuntimeError("posterior interval differs")

        with patch.object(smoke, "wake_service", side_effect=self.ready), patch.object(smoke, "browser_smoke", side_effect=failed_browser):
            receipt = smoke.run(self.args())
        self.assert_receipt_saved(receipt)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["models"][0]["status"], "failed")
        self.assertFalse(receipt["render_public_smoke_passed"])

    def test_missing_or_malformed_public_revision_fails_before_any_io(self):
        for revision in ("", "a" * 39, "g" * 40):
            with self.subTest(revision=revision), patch.object(smoke, "load_references") as references, patch.object(smoke, "wake_service") as wake:
                with self.assertRaises(ValueError):
                    smoke.run(self.args(revision=revision))
                references.assert_not_called()
                wake.assert_not_called()
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
