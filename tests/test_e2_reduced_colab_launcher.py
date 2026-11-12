"""Offline checks for source binding and transfer of already spent local time."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "notebooks/run_e2_reduced_colab.py"
SPEC = importlib.util.spec_from_file_location("reduced_colab_launcher", SOURCE)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class ColabLauncherTests(unittest.TestCase):
    def test_prior_ledgers_are_transferred_but_current_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            e2 = Path(directory)
            target = e2 / "reduced-delivery-v1"
            current = target / "shared_budget/pilot_budget.jsonl"
            other = e2 / "another-pilot/pilot_budget.jsonl"
            for path in (current, other):
                path.parent.mkdir(parents=True)
                path.write_text("preserved")
            expected_absent = e2 / "amended-pilots-v1/ledger/pilot_budget.jsonl"
            self.assertEqual(set(launcher._inherited_ledgers(e2, target)),
                             {other, expected_absent})
            self.assertEqual(current.read_text(), "preserved")

    def _source(self, workspace, *, tamper=False):
        commit = "a" * 40
        repo = workspace / ("reduced-source-" + commit)
        helper = repo / "notebooks/comprobar_e2_reducido_colab.py"
        helper.parent.mkdir(parents=True)
        helper.write_bytes(b"frozen helper" if not tamper else b"changed")
        for relative in ("experiments/e2_reduced_protocol_20261112/git_registration.json",
                         "experiments/e2_protocol_amendment_20261112/git_registration.json"):
            path = repo / relative
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"registration_commit": "b" * 40}))
        record = repo / launcher.IMPLEMENTATION
        record.parent.mkdir(parents=True)
        record.write_text(json.dumps({"protocol_id": "e2-reduced-delivery-v1",
            "files_sha256": {helper.relative_to(repo).as_posix():
                hashlib.sha256(b"frozen helper").hexdigest()}}))
        return repo, commit

    @staticmethod
    def _git(command, env, **kwargs):
        if "rev-parse" in command:
            stdout = "a" * 40
        elif "get-url" in command:
            stdout = launcher.REPOSITORY
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout)

    def test_checkout_verifies_sources_and_fetches_registration_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repo, commit = self._source(workspace)
            with patch.object(launcher, "_run", side_effect=self._git) as run:
                self.assertEqual(launcher._checkout(commit, workspace, {}), repo)
            fetches = [call.args[0] for call in run.call_args_list if "fetch" in call.args[0]]
            self.assertEqual(len(fetches), 2)
            self.assertTrue(all("--depth" in command for command in fetches))

    def test_checkout_preserves_and_rejects_modified_frozen_source(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            repo, commit = self._source(workspace, tamper=True)
            with patch.object(launcher, "_run", side_effect=self._git):
                with self.assertRaisesRegex(RuntimeError, "Una fuente registrada cambió"):
                    launcher._checkout(commit, workspace, {})
            self.assertEqual((repo / "notebooks/comprobar_e2_reducido_colab.py").read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
