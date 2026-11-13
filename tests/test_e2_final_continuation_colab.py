"""Consent, pinned routing and final-stage admission boundaries without training."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("notebooks/run_e2_final_continuation_colab.py", "test_final_continuation_launcher")
driver = load("scripts/run_e2_final_continuation_delivery.py", "test_final_continuation_driver")
COMMIT = "a" * 40
MODELS = ["model-" + str(i) for i in range(13)]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()


class LauncherConsentTests(unittest.TestCase):
    def test_dry_run_neither_prompts_fetches_nor_prepares_runtime(self):
        with patch.object(launcher, "_prior_launcher") as prior:
            answer = Mock(side_effect=AssertionError("Unexpected prompt"))
            result = launcher.run_e2_final_continuation_colab(COMMIT, input_fn=answer)
        answer.assert_not_called()
        prior.assert_not_called()
        self.assertFalse(result["training_started"])
        self.assertFalse(result["budget_continuation_applied"])

    def test_cancel_and_nonexact_confirmation_do_not_touch_parent(self):
        for text in ("", "0", "72", "80.0", "sí"):
            with self.subTest(text=text), patch.object(launcher, "_prior_launcher") as prior:
                result = launcher.run_e2_final_continuation_colab(COMMIT,
                    execute=True, input_fn=lambda prompt: text)
            self.assertTrue(result["cancelled"])
            prior.assert_not_called()

    def test_confirmation_routes_only_new_driver_and_keeps_q_and_resource_arguments(self):
        calls = []
        base = {"CONTROLLERS": set(), "LOCAL_WORKSPACE": Path("/scratch"),
            "json": json, "_stream": lambda command, *args: calls.append(command) or 0}
        funding = {"_base_launcher": lambda: base}
        prior = {"_funding_launcher": lambda: funding}
        def run_prior(commit, *, namespace, execute):
            self.assertEqual(commit, launcher.PRIOR_COMMIT)
            self.assertTrue(execute)
            current = prior["_funding_launcher"]()["_base_launcher"]()
            current["_stream"](["python", Path("/Q/scripts/run_e2_a100_recovery_delivery.py"),
                "--engine-root", "/E", "--extension-root", "/Q", "--extension-commit", commit,
                "--resource-extension-root", "/P", "--execute"], "env", "cwd", "log")
            return {"temporary_workspace": "/nonexistent", "effective_total_budget_hours": 72}
        prior["run_e2_a100_recovery_colab"] = run_prior
        approval = {"effective_total_hours": 80, "sha256": "b" * 64}
        with patch.object(launcher, "_prior_launcher", return_value=prior), \
                patch.object(launcher, "_materialize", return_value=Path("/new")) as materialize, \
                patch.object(launcher, "_approval", return_value=approval) as consent:
            result = launcher.run_e2_final_continuation_colab(COMMIT,
                execute=True, input_fn=lambda prompt: "80")
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertEqual(command[1], Path("/new/scripts/run_e2_final_continuation_delivery.py"))
        self.assertEqual(command[command.index("--extension-commit") + 1], launcher.PRIOR_COMMIT)
        self.assertEqual(command[command.index("--resource-extension-root") + 1], "/P")
        self.assertEqual(command[command.index("--continuation-commit") + 1], COMMIT)
        self.assertEqual(json.loads(command[command.index("--approval-receipt") + 1]), approval)
        materialize.assert_called_once_with(COMMIT, Path("/scratch"))
        self.assertEqual(consent.call_args.args[2], "80")
        self.assertEqual(result["effective_total_budget_hours"], 72)
        self.assertFalse(result["budget_continuation_applied"])

    def test_summary_needs_verified_applied_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "shared_budget/budget_continuation_80.json"
            path.parent.mkdir()
            source = {"temporary_workspace": temporary, "effective_total_budget_hours": 72}
            self.assertFalse(launcher.verified_continuation_summary(source, COMMIT)["budget_continuation_applied"])
            anchor = {"effective_total_hours": 80, "continuation_registration_commit": COMMIT}
            anchor["sha256"] = hashlib.sha256(canonical(anchor)).hexdigest()
            path.write_text(json.dumps(anchor))
            result = launcher.verified_continuation_summary(source, COMMIT)
            self.assertTrue(result["budget_continuation_applied"])
            self.assertEqual(result["effective_total_budget_hours"], 80)
            anchor["effective_total_hours"] = 800
            path.write_text(json.dumps(anchor))
            self.assertFalse(launcher.verified_continuation_summary(source, COMMIT)["budget_continuation_applied"])

    def test_actual_q_h_g_routing_layers_keep_their_original_commits(self):
        prior = load("notebooks/run_e2_a100_recovery_colab.py", "actual_q_route")
        funding = load("notebooks/run_e2_reduced_github72_colab.py", "actual_h_route")
        base = load("notebooks/run_e2_reduced_github_colab.py", "actual_g_route")
        calls = []
        base._stream = lambda command, *args: calls.append(command) or 0
        def run_g(commit, *, namespace, execute):
            command = ["python", Path("/Q/scripts/run_e2_reduced_github_delivery.py"),
                "--engine-root", "/E", "--extension-root", "/Q", "--extension-commit", commit,
                "--execute"]
            base._stream(command, "environment", "cwd", "log")
            return {"engine_commit": launcher.ENGINE_COMMIT, "temporary_workspace": "/nonexistent"}
        base.run_reduced_github_colab = run_g
        funding._base_launcher = lambda: vars(base)
        prior._funding_launcher = lambda: vars(funding)
        prior._prepare_runtime = lambda namespace: None
        with patch.object(launcher, "_prior_launcher", return_value=vars(prior)), \
                patch.object(launcher, "_materialize", return_value=Path("/new")), \
                patch.object(launcher, "_approval", return_value={"confirmed_text": "80"}):
            launcher.run_e2_final_continuation_colab(COMMIT, namespace={},
                execute=True, input_fn=lambda prompt: "80")
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertEqual(command[1], Path("/new/scripts/run_e2_final_continuation_delivery.py"))
        self.assertEqual(command[command.index("--extension-commit") + 1], launcher.PRIOR_COMMIT)
        self.assertEqual(command[command.index("--resource-extension-root") + 1],
            base.LOCAL_WORKSPACE / ("github-primary-extension-" + prior.RESOURCE_COMMIT))


class PublishedRegistrationTests(unittest.TestCase):
    def test_real_file_bytes_and_prior_proofs_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes = {}
            for relative in driver.FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, path)
                hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            proposal = json.loads((root / driver.PROPOSAL_PATH).read_text())
            record = {"engine_commit": driver.ENGINE_COMMIT,
                "protocol_id": "e2-reduced-delivery-v1", "prior_controller_commit": driver.PRIOR_COMMIT,
                "prior_funding_commit": driver.FUNDING_COMMIT, "resource_commit": driver.RESOURCE_COMMIT,
                "files_sha256": hashes, "proposal_sha256": hashes[driver.PROPOSAL_PATH],
                "proposal_policy_sha256": proposal["sha256"]}
            raw = json.dumps(record).encode()
            (root / "registration.json").write_bytes(raw)
            with patch.object(driver.subprocess, "check_output", return_value=raw):
                actual, checked = driver._verify_extension(root, COMMIT, Path("/E"))
            self.assertEqual(actual, record)
            self.assertEqual(checked, proposal)
            (root / driver.PROPOSAL_PATH).write_text(json.dumps({"sha256": proposal["sha256"]}))
            with patch.object(driver.subprocess, "check_output", return_value=raw):
                with self.assertRaisesRegex(ValueError, "runtime source differs"):
                    driver._verify_extension(root, COMMIT, Path("/E"))


class FinalStageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.closure = {"status": "completed", "completed_models": MODELS,
            "required_models": MODELS, "selections": {model: {"model_id": model,
                "all_eighteen_slots_accounted_for": True, "validation_count": 249,
                "winner_configuration_id": "C01"} for model in MODELS}}
        self.path = self.root / "selection_complete.json"
        self.path.write_text(json.dumps(self.closure))
        self.args = SimpleNamespace(output=self.root, stage="all", models=None,
            allow_final_download=True)

    def test_existing_selection_is_preserved_and_builtin_final_stages_are_ordered(self):
        before = self.path.read_bytes()
        calls = []
        budget = object()
        state = object()
        def execute(args, used_budget, used_state):
            self.assertIs(used_budget, budget)
            self.assertIs(used_state, state)
            self.assertTrue(args.allow_final_download)
            calls.append(args.stage)
            return {"stage": args.stage}
        with patch.dict(sys.modules, {"e2_reduced_protocol": SimpleNamespace(NEURAL=MODELS)}):
            result = driver.final_stage_adapter(execute)(self.args, budget, state)
        self.assertEqual(calls, ["final-fit", "pilot", "classical", "freeze", "score"])
        self.assertEqual(result, {"stage": "score"})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.args.stage, "all")

    def test_failure_stops_before_test_evaluation(self):
        calls = []
        def execute(args, *unused):
            calls.append(args.stage)
            if args.stage == "pilot":
                raise RuntimeError("pilot unresolved")
        with patch.dict(sys.modules, {"e2_reduced_protocol": SimpleNamespace(NEURAL=MODELS)}):
            with self.assertRaisesRegex(RuntimeError, "pilot unresolved"):
                driver.final_stage_adapter(execute)(self.args, object(), object())
        self.assertEqual(calls, ["final-fit", "pilot"])

    def test_incomplete_cohort_and_recipe_block_all_final_stages(self):
        for invalid in ("cohort", "recipe"):
            value = json.loads(json.dumps(self.closure))
            if invalid == "cohort":
                value["completed_models"] = MODELS[:-1]
            else:
                value["selections"][MODELS[0]]["validation_count"] = 83
            self.path.write_text(json.dumps(value))
            execute = Mock()
            with patch.dict(sys.modules, {"e2_reduced_protocol": SimpleNamespace(NEURAL=MODELS)}):
                with self.assertRaises(ValueError):
                    driver.final_stage_adapter(execute)(self.args, object(), object())
            execute.assert_not_called()

    def test_subset_request_is_rejected(self):
        self.args.models = MODELS[:5]
        execute = Mock()
        with self.assertRaises(ValueError):
            driver.final_stage_adapter(execute)(self.args, object(), object())
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
