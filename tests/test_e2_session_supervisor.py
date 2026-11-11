"""Stdlib-only supervisor checks; no GPU, network or scientific callbacks."""
import ast
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "notebooks/supervise_e2_within_56_hours.py"
if not SCRIPT.is_file():
    SCRIPT = HERE / "supervise_e2_within_56_hours.py"
spec = importlib.util.spec_from_file_location("session_supervisor_subject", SCRIPT)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class Process:
    def __init__(self, code=0, active=False, interrupt=False):
        self.code = code
        self.active = active
        self.interrupt = interrupt
        self.waits = 0

    def poll(self):
        return None if self.active else self.code

    def wait(self):
        self.waits += 1
        if self.interrupt:
            raise KeyboardInterrupt()
        self.active = False
        return self.code

    def kill(self):
        raise AssertionError("Supervisor must never kill the existing controller")

    terminate = kill
    send_signal = kill


class SupervisorTests(unittest.TestCase):
    @contextmanager
    def campaign(self, *, count=100, state="paused_session_budget", total=56, active=False, code=0):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plan = {"remote_storage": {"branch": supervisor.BRANCH},
                    "legacy_cache": {"spent_seconds": 156553.03253392776}}
            digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":"),
                                               ensure_ascii=False).encode()).hexdigest()
            config = {"plan": plan, "plan_sha256": digest}
            output.joinpath("campaign.json").write_text(json.dumps(config))
            def write(*, new_count=count, new_state=state, new_total=total, remaining=None):
                limit = new_total * 3600 - plan["legacy_cache"]["spent_seconds"]
                charged = 8748.729611191999
                status = {"plan_sha256": digest, "remote_branch": supervisor.BRANCH,
                          "status": new_state, "completed_jobs": new_count,
                          "pause_reason": "session_budget" if new_state == "paused_session_budget" else None,
                          "combined_charged_seconds": plan["legacy_cache"]["spent_seconds"] + charged,
                          "budget": {"plan_sha256": digest, "completed_jobs": new_count,
                                     "campaign_budget_seconds": limit, "charged_wall_seconds": charged,
                                     "remaining_campaign_seconds": limit - charged if remaining is None else remaining}}
                output.joinpath("status.json").write_text(json.dumps(status))
                return status
            write()
            process = Process(code=code, active=active)
            namespace = {"OPTIMIZATION_OUTPUT": output, "E2_OPT_ACTIVE_PROCESS": process}
            with redirect_stdout(io.StringIO()):
                yield namespace, write, process, config

    def test_two_clean_session_pauses_resume_sequentially_then_complete(self):
        with self.campaign(active=True) as (namespace, write, process, config):
            original = namespace["OPTIMIZATION_OUTPUT"].joinpath("campaign.json").read_bytes()
            calls = []
            def invoke(ns):
                self.assertIs(ns, namespace)
                self.assertEqual(ns["E2_OPT_ACTIVE_PROCESS"].poll(), 0)
                calls.append(len(calls) + 1)
                write(new_count=100 + 10 * len(calls),
                      new_state="paused_session_budget" if len(calls) == 1 else
                      "completed_eleven_origin_selection_not_final_benchmark")
                ns["E2_OPT_ACTIVE_PROCESS"] = Process()
            result = supervisor.supervise(namespace, invoke)
            self.assertEqual(calls, [1, 2])
            self.assertEqual(process.waits, 1)
            self.assertEqual(result["automatic_resumptions"], 2)
            self.assertEqual(result["completed_jobs"], 120)
            self.assertEqual(namespace["OPTIMIZATION_OUTPUT"].joinpath("campaign.json").read_bytes(), original)

    def test_every_other_status_stops_without_invocation(self):
        states = ["paused_campaign_budget", "paused_github_sync", "reconciliation_required",
                  "blocked_runtime_failure", "interrupted", "paused_storage_budget",
                  "completed_eleven_origin_selection_not_final_benchmark", "unrecognized_status"]
        for state in states:
            with self.subTest(state=state), self.campaign(state=state) as (ns, write, process, config):
                result = supervisor.supervise(ns, lambda ns: self.fail("Unexpected restart"))
                self.assertEqual(result["stop_reason"], state)

    def test_nonzero_exit_stops_even_if_status_is_session_pause(self):
        with self.campaign(code=2) as (ns, write, process, config):
            result = supervisor.supervise(ns, lambda ns: self.fail("Unexpected restart"))
            self.assertEqual(result["stop_reason"], "controller_nonzero_exit")

    def test_budget_cannot_exceed_already_authorized_56_hours(self):
        with self.campaign(total=56.01) as (ns, write, process, config):
            with self.assertRaisesRegex(ValueError, "exceeds_authorized_56"):
                supervisor.supervise(ns, lambda ns: self.fail("Unexpected restart"))

    def test_no_new_results_after_resume_stops_instead_of_looping(self):
        with self.campaign() as (ns, write, process, config):
            calls = []
            def invoke(ns):
                calls.append(1)
                write(new_count=100)
                ns["E2_OPT_ACTIVE_PROCESS"] = Process()
            with self.assertRaisesRegex(RuntimeError, "no_new_progress"):
                supervisor.supervise(ns, invoke)
            self.assertEqual(calls, [1])

    def test_keyboard_interrupt_propagates_without_retry_or_signal(self):
        with self.campaign(active=True) as (ns, write, process, config):
            process.interrupt = True
            with self.assertRaises(KeyboardInterrupt):
                supervisor.supervise(ns, lambda ns: self.fail("Unexpected restart"))
            self.assertTrue(process.active)

    def test_stale_hash_nonfinite_or_inconsistent_budget_is_refused(self):
        mutations = [lambda s: s.update(plan_sha256="b" * 64),
                     lambda s: s["budget"].update(plan_sha256="b" * 64),
                     lambda s: s["budget"].update(campaign_budget_seconds=float("nan")),
                     lambda s: s["budget"].update(remaining_campaign_seconds=10),
                     lambda s: s.update(completed_jobs=True)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.campaign() as (ns, write, process, config):
                status = write()
                mutation(status)
                ns["OPTIMIZATION_OUTPUT"].joinpath("status.json").write_text(json.dumps(status))
                with self.assertRaises(ValueError):
                    supervisor.supervise(ns, lambda ns: self.fail("Unexpected restart"))

    def test_no_balance_stops_without_invoking(self):
        with self.campaign() as (ns, write, process, config):
            status = write()
            status["budget"]["charged_wall_seconds"] = status["budget"]["campaign_budget_seconds"]
            status["budget"]["remaining_campaign_seconds"] = 0
            status["combined_charged_seconds"] = 56 * 3600
            ns["OPTIMIZATION_OUTPUT"].joinpath("status.json").write_text(json.dumps(status))
            result = supervisor.supervise(ns, lambda ns: self.fail("Unexpected restart"))
            self.assertEqual(result["stop_reason"], "no_remaining_campaign_budget")

    def test_source_has_no_budget_extension_or_process_signal_calls(self):
        forbidden = {"extend_campaign_budget", "kill", "terminate", "send_signal"}
        for node in ast.walk(ast.parse(SCRIPT.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)


if __name__ == "__main__":
    unittest.main()
