"""Budget invariants for concurrent, failed and tampered continuation attempts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("correction_budget", Path(__file__).resolve().parents[1]/"scripts/e2_corrections_budget.py")
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = Path(self.tmp.name)/"ledger.jsonl"
        self.base = Path(self.tmp.name)/"base.json"
        self.base.write_text(json.dumps({"total_charged_seconds": 900, "remaining_seconds": 100,
            "approved_total_hours": 1000/3600, "closed_at_utc": "2026-11-14T00:00:00Z",
            "unresolved_reservations": {}}))
        self.kw = dict(path=self.ledger, base_path=self.base)

    def state(self):
        with budget.locked_ledger(self.ledger, self.base) as (rows, _):
            return budget.summarize(rows)

    def test_reservations_prevent_combined_oversubscription_and_release_unused(self):
        first = budget.reserve("a", 70, **self.kw)
        with self.assertRaises(ValueError):
            budget.reserve("b", 31, **self.kw)
        budget.finish(first, 20, 0, **self.kw)
        second = budget.reserve("b", 80, **self.kw)
        budget.finish(second, 10, 1, **self.kw)
        self.assertEqual(self.state()["remaining_seconds"], 70)
        self.assertEqual(self.state()["cumulative_charged_seconds"], 930)

    def test_failures_and_overruns_are_fully_charged(self):
        key = budget.reserve("failed", 10, **self.kw)
        budget.finish(key, 12, 124, **self.kw)
        self.assertEqual(self.state()["corrections_charged_seconds"], 12)
        with self.assertRaises(ValueError):
            budget.finish(key, 12, 124, **self.kw)

    def test_tampering_of_a_linked_record_rejected(self):
        key = budget.reserve("first", 10, **self.kw)
        budget.finish(key, 2, 0, **self.kw)
        raw = self.ledger.read_bytes().replace(b'"cap_seconds":10.0', b'"cap_seconds":11.0', 1)
        self.ledger.write_bytes(raw)
        with self.assertRaises(ValueError):
            self.state()

    def test_partial_writes_rejected(self):
        budget.reserve("first", 10, **self.kw)
        with self.ledger.open("ab") as stream:
            stream.write(b'{}')
        with self.assertRaises(ValueError):
            self.state()

    def test_changed_historical_account_rejected(self):
        budget.reserve("first", 10, **self.kw)
        self.base.write_text(self.base.read_text()+"\n")
        with self.assertRaises(ValueError):
            self.state()


if __name__ == "__main__":
    unittest.main()
