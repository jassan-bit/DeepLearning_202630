"""Contracts of the public smoke client, without network or Playwright."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "experiments/e3_bloques_fg_20261117/smoke_render.py"
SPEC = importlib.util.spec_from_file_location("e3_smoke_render", SOURCE)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class RenderSmokeContracts(unittest.TestCase):
    def test_render_origin_and_explicit_loopback_have_different_scope(self):
        self.assertEqual(smoke.validate_url("https://e3-demo.onrender.com/"), ("https://e3-demo.onrender.com", True))
        self.assertEqual(smoke.validate_url("http://127.0.0.1:8050", True), ("http://127.0.0.1:8050", False))
        with self.assertRaises(ValueError):
            smoke.validate_url("http://127.0.0.1:8050")

    def test_public_target_cannot_be_replaced_by_credentials_or_private_destinations(self):
        for url in ("https://service.onrender.com.evil.example", "https://service.onrender.com@127.0.0.1",
                    "http://service.onrender.com", "https://service.onrender.com/?token=secret",
                    "https://service.onrender.com/path", "https://service.onrender.com:8050",
                    "https://169.254.169.254", "https://service.onrender.com/#fragment"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                smoke.validate_url(url)
        self.assertTrue(smoke.same_origin("https://service.onrender.com/eda", "https://service.onrender.com:443"))
        self.assertFalse(smoke.same_origin("https://other.onrender.com", "https://service.onrender.com"))
        self.assertFalse(smoke.same_origin("http://service.onrender.com", "https://service.onrender.com"))
        self.assertIsNone(smoke.NoHealthRedirect().redirect_request(None, None, 302, "Found", {}, "https://other.example"))

    def test_numeric_parity_rejects_nonfinite_negative_and_mismatched_forecasts(self):
        self.assertLess(smoke.validate_forecast(0.003, 0.0030004, 1e-6), 1e-6)
        for values in ((float("nan"), 0.003, 1e-6), (float("inf"), 0.003, 1e-6),
                       (-0.003, -0.003, 1e-6), (0.004, 0.003, 1e-6), (0.003, 0.003, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                smoke.validate_forecast(*values)

    def test_unavailable_service_still_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "receipt.json"
            args = SimpleNamespace(url="https://e3-demo.onrender.com", allow_local=False, deployment_commit="a" * 40,
                                   wake_seconds=180, navigation_seconds=30, prediction_seconds=30, tolerance=1e-6, output=output)
            with patch.object(smoke, "wake_service", side_effect=RuntimeError("Readiness deadline")):
                receipt = smoke.run(args)
            self.assertEqual(json.loads(output.read_bytes()), receipt)
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["render_public_smoke_passed"])
            self.assertFalse(receipt["remote_commit_identity_verified"])

    def test_success_on_loopback_cannot_certify_render(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(url="http://localhost:8050", allow_local=True, deployment_commit="",
                                   wake_seconds=180, navigation_seconds=30, prediction_seconds=30, tolerance=1e-6,
                                   output=Path(folder) / "receipt.json")
            with patch.object(smoke, "wake_service"), patch.object(smoke, "browser_smoke"):
                receipt = smoke.run(args)
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["scope"], "local_preflight_not_public_deployment")
            self.assertFalse(receipt["render_public_smoke_passed"])

    def test_public_check_requires_operator_deployment_revision(self):
        args = SimpleNamespace(url="https://e3-demo.onrender.com", allow_local=False, deployment_commit="")
        with self.assertRaises(ValueError):
            smoke.run(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
