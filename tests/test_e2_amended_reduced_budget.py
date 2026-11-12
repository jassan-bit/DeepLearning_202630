"""Shared reduced-worker accounting tests, synthetic records and fake clocks."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_reduced_budget as budget
import e2_amended_pilot_budget as old


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def reconciliation():
    value = {"version": old.VERSION, "status": "reconciled_before_new_pilot_reservations",
        "authorized_combined_hours": 64, "authorized_combined_seconds": 230400,
        "historical_charged_seconds": 222927.98951942776,
        "remaining_seconds_before_pilots": 7472.010480572237, "budget_reset": False}
    return value | {"sha256": budget.digest(value)}


def migration():
    value = {"version": budget.MIGRATION_VERSION, "authorized_total_hours": 64,
        "budget_reset": False, "budget_extension_authorized": False,
        "historical_reconciliation_sha256": reconciliation()["sha256"], "scope": "new_reduced_jobs_only",
        "stage_wall_seconds": 600, "worker_overhead_seconds": 60, "max_optimization_phases": 3,
        "max_epochs": 200, "patience_epochs": 20}
    return value | {"sha256": budget.digest(value)}


def make(path, clock, **kwargs):
    return budget.ReducedBudgetLedger(path, reconciliation(), migration=migration(),
        migration_registration_commit="a" * 40, clock=clock, **kwargs)


class ReducedBudgetTests(unittest.TestCase):
    def test_bound_reservations_preserve_epochs_and_shared_cap(self):
        with tempfile.TemporaryDirectory() as folder, make(Path(folder) / "shared", Clock()) as ledger:
            self.assertEqual(ledger.stage_wall_seconds, 600)
            self.assertEqual(ledger.reservation_for(), 660)
            self.assertEqual(ledger.reservation_for(3), 1860)
            self.assertEqual(ledger.worker_timeout_for(3), 1855)
            with self.assertRaises(budget.BudgetError): ledger.reserve("old", 7500)
            with self.assertRaises(budget.BudgetError): ledger.reserve("too-short", 10)
            self.assertAlmostEqual(ledger.remaining_seconds, 7472.010480572237)

    def test_actual_whole_worker_and_between_worker_overhead_are_charged(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock(); ledger = make(Path(folder) / "shared", clock)
            clock.advance(4)
            ledger.reserve("one", 660)
            clock.advance(10)
            ledger.charge("one", 2, status="completed", resource_use={"result_sha256": "c" * 64})
            clock.advance(3)
            ledger.reserve("two", 660)
            clock.advance(5)
            ledger.charge("two", 5, status="completed")
            clock.advance(2); ledger.close()
            self.assertAlmostEqual(ledger.report()["pilot_charged_seconds"], 24)
            with make(Path(folder) / "shared", clock) as again:
                self.assertAlmostEqual(again.report()["remaining_seconds"], 7472.010480572237-24)

    def test_successful_artifact_reuse_needs_its_cost_binding(self):
        with tempfile.TemporaryDirectory() as folder, make(Path(folder) / "shared", Clock()) as ledger:
            self.assertIsNone(ledger.completed_receipt("missing"))
            ledger.reserve("one", 660, metadata={"logical_job_id": "logical-one"})
            ledger.charge("one", 12, status="completed", resource_use={"result_sha256": "c" * 64})
            self.assertEqual(ledger.completed_receipt("one", expected_resource_use={"result_sha256": "c" * 64})["elapsed_seconds"], 12)
            with self.assertRaises(budget.BudgetReconciliationRequired):
                ledger.completed_receipt("one", expected_resource_use={"result_sha256": "d" * 64})
            with self.assertRaises(budget.BudgetReconciliationRequired):
                ledger.reserve("new-id", 660, metadata={"logical_job_id": "logical-one"})

    def test_unclean_death_preserves_reservation_and_known_overrun(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock(); ledger = make(Path(folder) / "shared", clock)
            ledger.reserve("unknown", 660)
            clock.advance(700); ledger.close()
            self.assertEqual(ledger.report()["reserved_seconds"], 700)
            self.assertAlmostEqual(ledger.report()["remaining_seconds"], 7472.010480572237-700)
            with self.assertRaises(budget.BudgetReconciliationRequired):
                make(Path(folder) / "shared", clock)

    def test_lease_excludes_hpo_and_final_controllers(self):
        with tempfile.TemporaryDirectory() as folder:
            clock = Clock(); path = Path(folder) / "shared"
            with make(path, clock):
                with self.assertRaises(budget.BudgetError): make(path, clock)
            with make(path, clock) as reopened:
                self.assertFalse(reopened.report()["reconciliation_required"])

    def test_old_local_cost_is_transferred_once_without_rewriting_history(self):
        with tempfile.TemporaryDirectory() as folder:
            earlier = Path(folder) / "earlier"
            with old.PilotBudgetLedger(earlier, reconciliation()) as prior:
                prior.reserve("old-fit", 200)
                prior.charge("old-fit", 30, status="completed")
                prior.charge_overhead(2)
            contents = (earlier / "pilot_budget.jsonl").read_bytes()
            clock = Clock(); shared = Path(folder) / "shared"
            with make(shared, clock, inherited_ledgers=[earlier]) as ledger:
                self.assertEqual(ledger.report()["inherited_local_ledger_seconds"], 32)
                self.assertAlmostEqual(ledger.remaining_seconds, 7472.010480572237-32)
            with make(shared, clock, inherited_ledgers=[earlier]) as reopened:
                self.assertAlmostEqual(reopened.remaining_seconds, 7472.010480572237-32)
            self.assertEqual(contents, (earlier / "pilot_budget.jsonl").read_bytes())

    def test_old_unknown_attempt_or_changed_anchor_cannot_become_fresh_credit(self):
        with tempfile.TemporaryDirectory() as folder:
            earlier = Path(folder) / "earlier"
            prior = old.PilotBudgetLedger(earlier, reconciliation()); prior.reserve("unknown", 300); prior.close()
            with self.assertRaises(budget.BudgetReconciliationRequired):
                make(Path(folder) / "shared", Clock(), inherited_ledgers=[earlier])
            valid = migration(); changed = {**valid, "authorized_total_hours": 72}
            changed["sha256"] = budget.digest({k: v for k, v in changed.items() if k != "sha256"})
            with self.assertRaises(budget.BudgetError):
                budget.ReducedBudgetLedger(Path(folder) / "other", reconciliation(), migration=changed,
                    migration_registration_commit="a" * 40)

    def test_insufficient_remaining_allowance_blocks_before_new_fit(self):
        with tempfile.TemporaryDirectory() as folder, make(Path(folder) / "shared", Clock()) as ledger:
            ledger.charge_overhead(7000)
            with self.assertRaises(budget.BudgetPause): ledger.reserve("cannot-fit", 660)
            self.assertIsNone(ledger.completed_receipt("cannot-fit"))

    def test_reservation_overrun_is_charged_and_cannot_supply_compliant_cache(self):
        with tempfile.TemporaryDirectory() as folder, make(Path(folder) / "shared", Clock()) as ledger:
            ledger.reserve("overrun", 660)
            with self.assertRaises(budget.BudgetPause):
                ledger.charge("overrun", 700, status="completed")
            self.assertAlmostEqual(ledger.report()["pilot_charged_seconds"], 700)
            with self.assertRaises(budget.BudgetReconciliationRequired):
                ledger.completed_receipt("overrun")


if __name__ == "__main__":
    unittest.main()
