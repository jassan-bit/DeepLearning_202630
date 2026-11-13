"""Offline checks for the actual launcher chain and score-only dispatch."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("notebooks/run_e2_score_source_recovery_colab.py", "test_score_source_launcher")
driver = load("scripts/run_e2_score_source_recovery_delivery.py", "test_score_source_driver")
COMMIT = "a" * 40


class LauncherTests(unittest.TestCase):
    def test_dry_run_does_not_fetch_prepare_consent_or_start_a_parent(self):
        with patch.object(launcher, "_prior_launcher") as prior, patch.object(launcher, "_materialize") as materialize:
            result = launcher.run_e2_score_source_recovery_colab(COMMIT)
        prior.assert_not_called()
        materialize.assert_not_called()
        self.assertEqual(result["requested_stages"], ["score"])
        self.assertFalse(result["new_budget_authorization"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["predictor_refit_requested"])

    def test_actual_arima_caf_f80_q_h_g_chain_preserves_all_authorities_and_consent(self):
        arima = load("notebooks/run_e2_arima_resume_colab.py", "real_arima_source_route")
        caf = load("notebooks/run_e2_final80_resume_colab.py", "real_caf_source_route")
        f80 = load("notebooks/run_e2_final_continuation_colab.py", "real_f80_source_route")
        q = load("notebooks/run_e2_a100_recovery_colab.py", "real_q_source_route")
        h = load("notebooks/run_e2_reduced_github72_colab.py", "real_h_source_route")
        g = load("notebooks/run_e2_reduced_github_colab.py", "real_g_source_route")
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
        with patch.object(launcher, "_prior_launcher", return_value=vars(arima)), \
                patch.object(launcher, "_materialize", return_value=Path("/SOURCE")):
            result = launcher.run_e2_score_source_recovery_colab(COMMIT, execute=True)
        self.assertEqual(len(calls), 1)
        command, args = calls[0]
        self.assertEqual(command[1], Path("/SOURCE/scripts/run_e2_score_source_recovery_delivery.py"))
        self.assertEqual(command[command.index("--score-source-commit") + 1], COMMIT)
        self.assertEqual(command[command.index("--arima-commit") + 1], launcher.PRIOR_ARIMA_COMMIT)
        self.assertEqual(command[command.index("--resume-commit") + 1], launcher.PRIOR_RESUME_COMMIT)
        self.assertEqual(command[command.index("--continuation-commit") + 1], launcher.FUNDING_CONTINUATION_COMMIT)
        self.assertEqual(json.loads(command[command.index("--approval-receipt") + 1]), approval)
        self.assertEqual(args, ("env", "cwd", "log"))
        self.assertTrue(result["consent_reused"])
        self.assertEqual(result["requested_stages"], ["score"])
        self.assertFalse(result["new_budget_authorization"])
        self.assertFalse(result["predictor_refit_requested"])


class RoutingTests(unittest.TestCase):
    def test_actual_prior_router_preserves_all_child_args_and_replaces_only_score_worker(self):
        prior = load("scripts/run_e2_arima_resume_delivery.py", "real_arima_source_router")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, resource, arima, recovery = (root / name for name in ["E", "P", "ARIMA", "SOURCE"])
            spec = root / "worker_spec.json"
            spec.write_text(json.dumps({"kind": "final-score", "model": "all16"}))
            calls = []
            def supervisor(command, *args, **kwargs):
                calls.append((command, args, kwargs))
                return "unchanged-receipt"
            factory = driver.score_supervisor_factory(prior.repair_supervisor, recovery, COMMIT)
            route = factory(supervisor, engine, resource, arima, driver.PRIOR_ARIMA_COMMIT)
            original = ["python", "-u", engine / "scripts/e2_reduced_final.py", "--worker-spec", spec]
            self.assertEqual(route(original, "budget", "attempt", phases=1), "unchanged-receipt")
            command, args, kwargs = calls[0]
            self.assertEqual(command[2], recovery / "scripts/e2_score_source_worker.py")
            self.assertEqual(command[command.index("--arima-root") + 1], arima)
            self.assertEqual(command[command.index("--arima-commit") + 1], driver.PRIOR_ARIMA_COMMIT)
            self.assertEqual(command[command.index("--extension-root") + 1], resource)
            self.assertEqual(command[command.index("--extension-commit") + 1], driver.RESOURCE_COMMIT)
            self.assertEqual(command[command.index("--worker-spec") + 1], spec)
            self.assertEqual(command[command.index("--score-source-commit") + 1], COMMIT)
            self.assertEqual(args, ("budget", "attempt"))
            self.assertEqual(kwargs, {"phases": 1})
            for kind, model in [("classical", "ARIMA"), ("classical", "Persistence"), ("neural-final", "GRU32")]:
                spec.write_text(json.dumps({"kind": kind, "model": model}))
                with self.assertRaises(ValueError):
                    route(original)
            self.assertEqual(len(calls), 1)

    def test_score_only_context_restores_original_functions_even_on_failure(self):
        prior = load("scripts/run_e2_arima_resume_delivery.py", "real_arima_source_context")
        original = prior.FINAL_STAGES, prior.repair_supervisor
        final = SimpleNamespace(run_final_scoring=object())
        previous_scoring = final.run_final_scoring
        replacement = object()
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with driver.score_only_parent(prior, final, replacement, Path("/SOURCE"), COMMIT):
                self.assertEqual(prior.FINAL_STAGES, ("score",))
                self.assertIs(final.run_final_scoring, replacement)
                raise RuntimeError("stop")
        self.assertEqual((prior.FINAL_STAGES, prior.repair_supervisor), original)
        self.assertIs(final.run_final_scoring, previous_scoring)

    def test_actual_f80_stage_adapter_runs_score_only_and_preserves_selection_files(self):
        prior = load("scripts/run_e2_arima_resume_delivery.py", "real_arima_source_stages")
        f80 = load("scripts/run_e2_final_continuation_delivery.py", "real_f80_source_stages")
        from e2_reduced_protocol import NEURAL
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            selected = {name: {"model_id": name, "all_eighteen_slots_accounted_for": True,
                "validation_count": 249, "winner_configuration_id": "C01"} for name in NEURAL}
            path = output / "selection_complete.json"
            path.write_text(json.dumps({"status": "completed", "completed_models": list(NEURAL),
                "required_models": list(NEURAL), "selections": selected}))
            original = path.read_bytes()
            final = SimpleNamespace(run_final_scoring=object())
            with driver.score_only_parent(prior, final, object(), Path("/SOURCE"), COMMIT):
                # The preserved ARIMA parent assigns this exact value to its freshly loaded F80 module.
                f80.FINAL_STAGES = prior.FINAL_STAGES
                stages = []
                adapter = f80.final_stage_adapter(lambda args, *rest: stages.append(args.stage))
                adapter(SimpleNamespace(stage="all", models=None, output=output), "budget", "state")
            self.assertEqual(stages, ["score"])
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
