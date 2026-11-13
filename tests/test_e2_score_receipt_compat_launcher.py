"""Offline receipt-only and complete preserved-launcher-chain checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("notebooks/run_e2_score_receipt_compat_colab.py", "test_receipt_launcher")
driver = load("scripts/run_e2_score_receipt_compat_delivery.py", "test_receipt_parent")
COMMIT = "a" * 40


class LauncherTests(unittest.TestCase):
    def test_dry_run_does_not_fetch_prepare_or_start_anything(self):
        with patch.object(launcher, "_prior_launcher") as prior, patch.object(launcher, "_materialize") as materialize:
            result = launcher.run_e2_score_receipt_compat_colab(COMMIT)
        prior.assert_not_called(); materialize.assert_not_called()
        self.assertEqual(result["requested_stages"], ["score"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["new_budget_authorization"])

    def test_actual_source_arima_caf_f80_q_h_g_chain_keeps_original_sources_and_consent(self):
        source = load("notebooks/run_e2_score_source_recovery_colab.py", "receipt_real_source")
        arima = load("notebooks/run_e2_arima_resume_colab.py", "receipt_real_arima")
        caf = load("notebooks/run_e2_final80_resume_colab.py", "receipt_real_caf")
        f80 = load("notebooks/run_e2_final_continuation_colab.py", "receipt_real_f80")
        q = load("notebooks/run_e2_a100_recovery_colab.py", "receipt_real_q")
        h = load("notebooks/run_e2_reduced_github72_colab.py", "receipt_real_h")
        g = load("notebooks/run_e2_reduced_github_colab.py", "receipt_real_g")
        approval = {"confirmed_text": "80", "sha256": "b" * 64,
            "observed_at_utc": "2026-11-13T09:54:17.605263+00:00"}
        calls = []
        g._stream = lambda command, *args: calls.append((command, args)) or 0
        def run_g(commit, *, namespace, execute):
            g._stream(["python", Path("/Q/scripts/run_e2_reduced_github_delivery.py"),
                "--engine-root", "/E", "--extension-root", "/Q", "--extension-commit", commit],
                "env", "cwd", "log")
            return {"engine_commit": launcher.ENGINE_COMMIT, "temporary_workspace": "/nonexistent"}
        g.run_reduced_github_colab = run_g
        h._base_launcher = lambda: vars(g)
        q._funding_launcher = lambda: vars(h)
        q._prepare_runtime = lambda namespace: None
        f80._prior_launcher = lambda: vars(q)
        f80._materialize = lambda commit, workspace: Path("/F80")
        caf._prior_launcher = lambda: vars(f80)
        caf._materialize = lambda commit, workspace: Path("/CAF")
        caf.existing_approval = lambda: approval
        arima._prior_launcher = lambda: vars(caf)
        arima._materialize = lambda commit, workspace: Path("/ARIMA")
        source._prior_launcher = lambda: vars(arima)
        source._materialize = lambda commit, workspace: Path("/SOURCE2434")
        with patch.object(launcher, "_prior_launcher", return_value=vars(source)), \
                patch.object(launcher, "_materialize", return_value=Path("/RECEIPT")):
            result = launcher.run_e2_score_receipt_compat_colab(COMMIT, execute=True)
        self.assertEqual(len(calls), 1)
        command, args = calls[0]
        self.assertEqual(command[1], Path("/RECEIPT/scripts/run_e2_score_receipt_compat_delivery.py"))
        self.assertEqual(command[command.index("--score-source-root") + 1], Path("/SOURCE2434"))
        self.assertEqual(command[command.index("--score-source-commit") + 1], launcher.PRIOR_SCORE_SOURCE_COMMIT)
        self.assertEqual(command[command.index("--arima-commit") + 1], launcher.PRIOR_ARIMA_COMMIT)
        self.assertEqual(command[command.index("--resume-commit") + 1], launcher.PRIOR_RESUME_COMMIT)
        self.assertEqual(command[command.index("--continuation-commit") + 1], launcher.FUNDING_CONTINUATION_COMMIT)
        self.assertEqual(command[command.index("--receipt-compat-commit") + 1], COMMIT)
        self.assertEqual(json.loads(command[command.index("--approval-receipt") + 1]), approval)
        self.assertEqual(args, ("env", "cwd", "log"))
        self.assertTrue(result["consent_reused"])
        self.assertFalse(result["predictor_refit_requested"])
        self.assertEqual(result["requested_stages"], ["score"])


class ParentTests(unittest.TestCase):
    def test_factory_keeps_original_registration_proof_and_wraps_only_callback(self):
        callback, wrapped, final = object(), object(), object()
        original_kwargs = {"registration_commit": driver.PRIOR_SCORE_SOURCE_COMMIT,
            "registration_root": Path("/SOURCE2434"), "engine_root": Path("/E")}
        factory = Mock(return_value=Mock(return_value="result"))
        compatibility = Mock(return_value=wrapped)
        scoring = driver.receipt_scoring_factory(factory, compatibility, COMMIT)(final, **original_kwargs)
        factory.assert_called_once_with(final, **original_kwargs)
        unchanged = {"supervisor": object(), "raw_root": Path("/raw"), "allow_download": True}
        self.assertEqual(scoring("project", persist_result=callback, **unchanged), "result")
        compatibility.assert_called_once_with(callback, registration_commit=COMMIT)
        factory.return_value.assert_called_once_with("project", persist_result=wrapped, **unchanged)

    def test_missing_callback_stays_missing(self):
        scoring = Mock()
        compatibility = Mock()
        run = driver.receipt_scoring_factory(lambda: scoring, compatibility, COMMIT)()
        run("project", supervisor="original")
        compatibility.assert_not_called()
        scoring.assert_called_once_with("project", supervisor="original")

    def test_parent_delegates_old_arguments_and_restores_helper_after_exception(self):
        helper = SimpleNamespace(make_run_final_scoring=Mock())
        original_factory = helper.make_run_final_scoring
        previous_module = sys.modules.get("e2_score_source_recovery")
        adapter = SimpleNamespace(compatible_persist_result=Mock())
        rest = ["--score-source-root", str(ROOT), "--score-source-commit", driver.PRIOR_SCORE_SOURCE_COMMIT,
            "--engine-root", str(ROOT), "--arima-commit", driver.PRIOR_ARIMA_COMMIT]
        def old_main(args):
            self.assertEqual(args, rest)
            self.assertIs(sys.modules["e2_score_source_recovery"], helper)
            self.assertIsNot(helper.make_run_final_scoring, original_factory)
            raise RuntimeError("preserved-parent-stop")
        prior = SimpleNamespace(_verify=Mock(), main=old_main)
        with patch.object(driver, "_verify", return_value=ROOT), \
                patch.object(driver, "_load", side_effect=[prior, helper, adapter]), \
                patch.object(driver.subprocess, "check_output", return_value=driver.ENGINE_COMMIT + "\n"), \
                patch.object(driver.subprocess, "run"):
            with self.assertRaisesRegex(RuntimeError, "preserved-parent-stop"):
                driver.main(["--receipt-compat-root", str(ROOT), "--receipt-compat-commit", COMMIT, *rest])
        prior._verify.assert_called_once_with(ROOT, driver.PRIOR_SCORE_SOURCE_COMMIT, ROOT)
        self.assertIs(helper.make_run_final_scoring, original_factory)
        self.assertIs(sys.modules.get("e2_score_source_recovery"), previous_module)

    def test_real_git_registration_binds_exact_new_runtime_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            extension = repo / "extension"
            files = {}
            for relative in driver.FILES:
                path = extension / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (ROOT / relative).read_bytes()
                path.write_bytes(raw)
                files[relative] = hashlib.sha256(raw).hexdigest()
            record = {"engine_commit": driver.ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
                "resource_commit": driver.RESOURCE_COMMIT, "funding_continuation_commit": driver.FUNDING_CONTINUATION_COMMIT,
                "prior_resume_commit": driver.PRIOR_RESUME_COMMIT, "prior_arima_commit": driver.PRIOR_ARIMA_COMMIT,
                "prior_score_source_commit": driver.PRIOR_SCORE_SOURCE_COMMIT, "files_sha256": files}
            raw = json.dumps(record).encode()
            (extension / "registration.json").write_bytes(raw)
            published = repo / driver.REGISTRATION_PATH
            published.parent.mkdir(parents=True)
            published.write_bytes(raw)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            env = {**os.environ, "GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@invalid",
                "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@invalid"}
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "offline source binding"], check=True, env=env)
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(driver._verify(extension, commit, repo), extension)
            (extension / driver.FILES[1]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "source differs"):
                driver._verify(extension, commit, repo)


if __name__ == "__main__":
    unittest.main()
