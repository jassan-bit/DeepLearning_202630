"""Reuse actual persisted consent and preserve E/H/P/Q/F80 routing without fits."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("notebooks/run_e2_final80_resume_colab.py", "test_final80_resume_launcher")
driver = load("scripts/run_e2_final80_resume_delivery.py", "test_final80_resume_driver")
COMMIT = "d" * 40


def sealed(value):
    value = deepcopy(value)
    value["sha256"] = hashlib.sha256(launcher._canonical(value)).hexdigest()
    return value


def authority_fixture():
    approval = sealed({"version": "e2-confirmed-budget-continuation-v1",
        "source": "explicit_colab_confirmation", "confirmed_text": "80",
        "approved_total_hours": 80, "previous_total_hours": 72,
        "continuation_registration_commit": launcher.FUNDING_CONTINUATION_COMMIT,
        "observed_at_utc": "2026-11-13T10:00:00+00:00"})
    anchor = sealed({"version": "e2-confirmed-budget-continuation-v1",
        "effective_total_hours": 80, "previous_total_hours": 72, "budget_reset": False,
        "continuation_registration_commit": launcher.FUNDING_CONTINUATION_COMMIT,
        "approval_sha256": approval["sha256"], "approval_receipt": approval})
    payloads = {launcher.ANCHOR_PATH: launcher._canonical(anchor),
        launcher.APPROVAL_PATH: launcher._canonical(approval)}
    head = {"version": "e2-github-primary-state-v1", "scientific_commit": launcher.ENGINE_COMMIT,
        "persistent_destination": "GitHub", "reason": "shared_budget_closed",
        "metadata_deltas": [], "files": {p: {"bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()} for p, raw in payloads.items()}}
    head["state_id"] = hashlib.sha256(launcher._canonical(head)).hexdigest()
    return head, payloads, approval


class ExistingAuthorityTests(unittest.TestCase):
    def test_dry_run_does_not_read_authority_or_prepare_or_prompt(self):
        with patch.object(launcher, "existing_approval") as approved, \
                patch.object(launcher, "_prior_launcher") as prior:
            result = launcher.run_e2_final80_resume_colab(COMMIT)
        approved.assert_not_called()
        prior.assert_not_called()
        self.assertFalse(result["training_started"])
        self.assertFalse(result["new_budget_authorization"])

    def test_existing_local_receipt_is_verified_against_remote_without_writing(self):
        head, payloads, approval = authority_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            for relative, raw in payloads.items():
                path = workspace / "github-primary-output-v1" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            with patch.object(launcher, "_download", return_value=launcher._canonical(head)) as download:
                result = launcher.existing_approval(workspace)
            self.assertEqual(result, approval)
            self.assertEqual(download.call_count, 1)
            self.assertIn("?e2_resume=", download.call_args.args[0])
            for relative, raw in payloads.items():
                self.assertEqual((workspace / "github-primary-output-v1" / relative).read_bytes(), raw)

    def test_missing_local_receipt_is_recovered_read_only_from_verified_delta(self):
        head, payloads, approval = authority_fixture()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, raw in payloads.items():
                archive.writestr(path, raw)
        raw_zip = buffer.getvalue()
        head.pop("state_id")
        head["metadata_deltas"] = [{"path": "control/delta.zip", "bytes": len(raw_zip),
            "sha256": hashlib.sha256(raw_zip).hexdigest()}]
        head["state_id"] = hashlib.sha256(launcher._canonical(head)).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            def fetch(url, *unused):
                return launcher._canonical(head) if "head.json" in url else raw_zip
            with patch.object(launcher, "_download", side_effect=fetch):
                result = launcher.existing_approval(Path(temporary))
            self.assertEqual(result, approval)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_bad_head_and_unclosed_campaign_block_before_parent_launch(self):
        for mutation in ("hash", "active"):
            head, unused, approval = authority_fixture()
            if mutation == "hash":
                head["state_id"] = "f" * 64
            else:
                head.pop("state_id")
                head["reason"] = "lease_reserved_before_worker"
                head["state_id"] = hashlib.sha256(launcher._canonical(head)).hexdigest()
            with patch.object(launcher, "_download", return_value=launcher._canonical(head)):
                with self.assertRaises(RuntimeError):
                    launcher.existing_approval(Path("/nonexistent"))

    def test_original_consent_and_authority_survive_actual_nested_launcher_chain(self):
        unused_head, unused_payloads, approval = authority_fixture()
        f80 = load("notebooks/run_e2_final_continuation_colab.py", "real_f80_resume_route")
        q = load("notebooks/run_e2_a100_recovery_colab.py", "real_q_resume_route")
        h = load("notebooks/run_e2_reduced_github72_colab.py", "real_h_resume_route")
        g = load("notebooks/run_e2_reduced_github_colab.py", "real_g_resume_route")
        calls = []
        g._stream = lambda command, *args: calls.append(command) or 0
        def run_g(commit, *, namespace, execute):
            g._stream(["python", Path("/Q/scripts/run_e2_reduced_github_delivery.py"),
                "--extension-root", "/Q", "--extension-commit", commit], "env", "cwd", "log")
            return {"engine_commit": launcher.ENGINE_COMMIT, "temporary_workspace": "/nonexistent"}
        g.run_reduced_github_colab = run_g
        h._base_launcher = lambda: vars(g)
        q._funding_launcher = lambda: vars(h)
        q._prepare_runtime = lambda namespace: None
        f80._prior_launcher = lambda: vars(q)
        f80._materialize = lambda commit, workspace: Path("/F80")
        with patch.object(launcher, "existing_approval", return_value=approval), \
                patch.object(launcher, "_prior_launcher", return_value=vars(f80)), \
                patch.object(launcher, "_materialize", return_value=Path("/FIX")):
            result = launcher.run_e2_final80_resume_colab(COMMIT, execute=True)
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertEqual(command[1], Path("/FIX/scripts/run_e2_final80_resume_delivery.py"))
        self.assertEqual(command[command.index("--extension-commit") + 1], launcher.PRIOR_CONTROLLER_COMMIT)
        self.assertEqual(command[command.index("--continuation-commit") + 1], launcher.FUNDING_CONTINUATION_COMMIT)
        self.assertEqual(command[command.index("--resume-commit") + 1], COMMIT)
        reused = json.loads(command[command.index("--approval-receipt") + 1])
        self.assertEqual(reused, approval)
        self.assertEqual(reused["observed_at_utc"], "2026-11-13T10:00:00+00:00")
        self.assertTrue(result["consent_reused"])
        self.assertFalse(result["new_budget_authorization"])


class AdmissionTests(unittest.TestCase):
    def test_resume_admission_is_separate_immutable_and_hash_bound(self):
        admission = sealed({"resume_registration_commit": COMMIT, "historical_files_modified": False})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "hardware_transition/historical_t4_profile.json"
            old.parent.mkdir()
            old.write_bytes(b"original")
            result = driver._persist_admission(root, COMMIT, admission)
            before = result.read_bytes()
            self.assertEqual(driver._persist_admission(root, COMMIT, admission), result)
            self.assertEqual(result.read_bytes(), before)
            self.assertEqual(old.read_bytes(), b"original")
            self.assertIn("hardware_transition_resume/" + COMMIT, str(result))
            changed = deepcopy(admission)
            changed["historical_files_modified"] = True
            with self.assertRaises(ValueError):
                driver._persist_admission(root, COMMIT, changed)


if __name__ == "__main__":
    unittest.main()
