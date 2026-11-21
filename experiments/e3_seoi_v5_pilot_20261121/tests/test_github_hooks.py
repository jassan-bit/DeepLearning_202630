"""Persistence barriers and secret pipe tests; no network or optimizer steps."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import bootstrap_colab
from seoi_v5 import runner


class GithubHookTests(unittest.TestCase):
    def test_save_failure_stops_before_next_fit(self):
        plan = {"jobs": runner.jobs()}
        with patch.object(runner, "load_case", return_value=(plan, {}, Mock())), \
             patch.object(runner, "complete_job", return_value=None), \
             patch.object(runner, "fit_job", side_effect=runner.DurabilityError("save failed")) as fit, \
             patch.object(runner, "record_failure") as failure:
            with self.assertRaises(runner.DurabilityError):
                runner.run("fixture", "fixture", on_checkpoint=Mock())
            self.assertEqual(fit.call_count, 1)
            failure.assert_not_called()

    def test_job_is_saved_before_next_job_starts(self):
        events = []
        plan = {"jobs": runner.jobs()[:2]}
        def fit(*args, **kwargs): events.append(("fit", args[4]["job_id"]))
        def save(case, event):
            events.append(("save", event["job"]["job_id"]))
            raise runner.DurabilityError("stop before next job")
        with patch.object(runner, "load_case", return_value=(plan, {}, Mock())), \
             patch.object(runner, "complete_job", return_value=None), \
             patch.object(runner, "fit_job", side_effect=fit):
            with self.assertRaises(runner.DurabilityError):
                runner.run("fixture", "fixture", on_checkpoint=save)
        self.assertEqual([event[0] for event in events], ["fit", "save"])

    def test_checkpoint_callback_error_redacts_original_exception(self):
        with self.assertRaises(runner.DurabilityError) as error:
            runner.checkpoint_barrier(Mock(side_effect=ValueError("fake-private-value")), "fixture", {})
        self.assertNotIn("fake-private-value", str(error.exception))

    def test_remote_cost_unknown_record_is_permanent_in_history(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            self.assertFalse(runner.remote_compute_uncertain(p))
            runner.write_json(p / "recovery_history.json", [{"unobserved_remote_compute_possible": True}])
            runner.write_json(p / "remote_restore.json", {"unobserved_remote_compute_possible": False})
            self.assertTrue(runner.remote_compute_uncertain(p))

    def test_secrets_are_stdin_only_and_not_in_process_environment(self):
        settings = {"python": "/fixture/python", "source_root": "/fixture/source", "case_root": "/fixture/case", "data_file": "/fixture/data.npz"}
        with patch("subprocess.run") as run, patch.dict("os.environ", {"GITHUB_TOKEN": "fixture-token", "E3_V5_BACKUP_KEY": "fixture-key"}):
            bootstrap_colab.github_command(settings, "setup", source_commit="a" * 40,
                                           token="private-pat-fixture", backup_secret="private-key-fixture")
            args, kwargs = run.call_args
            self.assertNotIn("private-pat-fixture", " ".join(args[0]))
            self.assertNotIn("private-key-fixture", " ".join(args[0]))
            self.assertNotIn("GITHUB_TOKEN", kwargs["env"])
            self.assertNotIn("E3_V5_BACKUP_KEY", kwargs["env"])
            self.assertIn("private-pat-fixture", kwargs["input"])
            self.assertTrue(kwargs["text"])


if __name__ == "__main__":
    unittest.main()
