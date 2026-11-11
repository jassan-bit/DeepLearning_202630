"""Amendment tests without CUDA, market data, or credentials."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "schedule", Path(__file__).resolve().parents[1]/"scripts/e2_guide_schedule.py")
schedule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(schedule)


class ScheduleTests(unittest.TestCase):
    def test_budget(self):
        s = schedule.summary()
        self.assertEqual(s["hpo_attempts"]+s["nas1_attempts"]+
                         s["nas2_attempts_upper_bound"], 23202)
        self.assertAlmostEqual(s["reduction_fraction"], 8/11)

    def test_exact_original_plan_required(self):
        with self.assertRaisesRegex(ValueError, "exact_original"):
            schedule.validate_plan({}, lambda p: "wrong")

    def test_completion_is_not_full_eleven_origin_completion(self):
        old = {"status": "completed_selected_registered_domains",
               "coverage": {"selected_origins": 11, "partial_origins": False}}
        state = schedule.amended_report(lambda: copy.deepcopy(old))
        self.assertEqual(state["coverage"]["selected_origins"], 3)
        self.assertTrue(state["coverage"]["partial_origins"])
        self.assertEqual(state["status"], "completed_amended_three_origin_selection")
        self.assertEqual(old["coverage"]["selected_origins"], 11)

    def test_pause_is_preserved(self):
        state = schedule.amended_report(lambda: {
            "status": "reconciliation_required", "coverage": {}})
        self.assertEqual(state["status"], "reconciliation_required")

    def test_scheduler_reuses_cache_without_changing_plan_or_running_worker(self):
        plan = {"source_commit": schedule.SOURCE, "origins": [
            "2024-01-01", "2024-02-01", "2024-10-01", "2025-01-01"],
            "models": ["FCN-1D"], "remote_storage": {"branch": schedule.BRANCH},
            "seed_pools": {"hpo": [1, 2], "nas1": [3], "nas2": [4, 5, 6, 7, 8]}}
        original = copy.deepcopy(plan)
        prepared_origins, identities = [], []
        def prepare(panel, origin, end):
            prepared_origins.append(origin)
            split = types.SimpleNamespace(signature=origin)
            return types.SimpleNamespace(origin=origin, inner_train=split,
                                         inner_validation=split)
        candidate = types.SimpleNamespace()
        def select(prepared, name, adapter, **kwargs):
            self.assertEqual(kwargs["k"], 40)
            adapter.fit_candidate(candidate, 1, prepared.inner_train, prepared.inner_validation)
            winner = types.SimpleNamespace(candidate_id="winner", payload={})
            return types.SimpleNamespace(winner=winner, fixed_epochs={}, signature="sig",
                                         attempts=36, selection_rule="mean_rmse"), {}
        def job(full_plan, prepared, model, c, seed):
            self.assertEqual(full_plan, original)
            return {"stage": "hpo", "origin": prepared.origin}
        def digest(value):
            if value == original:
                return schedule.PLAN_SHA256
            self.assertEqual(value["plan_sha256"], schedule.PLAN_SHA256)
            return value["job"]["origin"]
        runner = types.SimpleNamespace(admission=types.SimpleNamespace(digest=digest),
                    job_payload=job, restore_trial=lambda v: v)
        registry = types.SimpleNamespace(MODEL_IDS={}, registered_trials=lambda n: [],
                    protocol=lambda: {"contract": {"delta": .01}})
        class Session:
            def get_result(self, identity, payload):
                identities.append(identity)
                return {"status": "completed"}
            def run_job(self, identity, payload, callback, **kwargs):
                return self.get_result(identity, payload)
        modules = {"run_e2_cuda_search": runner, "e2_training_models": registry,
            "e2_training_bindings": types.SimpleNamespace(run_neural_selection=select),
            "e2_temporal_training": types.SimpleNamespace(prepare_b=prepare)}
        with patch.dict(sys.modules, modules):
            selections = schedule.run_selection(plan, Session(), None,
                worker=lambda *a: self.fail("cached fit was repeated"))
        self.assertEqual(prepared_origins, list(schedule.ORIGINS))
        self.assertEqual(len(selections), 3)
        self.assertEqual(plan, original)
        self.assertTrue(set(identities) <= set(schedule.ORIGINS))

    def test_plan_only_has_no_training_dependencies(self):
        with patch("builtins.print"):
            self.assertEqual(schedule.main(["--plan-only"]), 0)

    def test_notebook_pins_scheduler_and_is_opt_in(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/"notebooks/entregable2_calendario_guia.ipynb").read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
        launch = "".join(next(c for c in notebook["cells"] if c["id"] == "guide-launch")["source"])
        self.assertIn("RUN_AMENDED_SCHEDULE = False", launch)
        self.assertIn(hashlib.sha256((root/"scripts/e2_guide_schedule.py").read_bytes()).hexdigest(), launch)
        self.assertNotIn("--retry-lost-once", launch)


if __name__ == "__main__":
    unittest.main()
