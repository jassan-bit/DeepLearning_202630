"""Synthetic checks of HPO aggregation and observed coverage; no fits."""
from pathlib import Path
import math
import hashlib
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_hpo_coverage_closure as closure


class HPOCoverageTests(unittest.TestCase):
    def test_ledger_zero_anchor_and_mutation_detection(self):
        first = {"sequence": 0, "previous_sha256": "0"*64, "kind": "overhead", "elapsed_seconds": 1.}
        first["sha256"] = hashlib.sha256(json.dumps(first, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(closure.verify_ledger([first]), first["sha256"])
        first["elapsed_seconds"] = 2.
        with self.assertRaises(ValueError): closure.verify_ledger([first])

    def test_pooled_rmse_is_not_the_arithmetic_fold_mean(self):
        self.assertAlmostEqual(closure.pooled_rmse([1., 3.], [1, 3]), math.sqrt(7))
        self.assertNotAlmostEqual(closure.pooled_rmse([1., 3.], [1, 3]), 2.5)
        with self.assertRaises(ValueError):
            closure.pooled_rmse([1., 3.], [1, 0])

    def test_coverage_counts_failed_unknown_without_imputing_performance(self):
        configs = [{"model": "synthetic", "configuration_id": "C%02d" % i,
                    "parent_trial_index": i+4, "recorded_winner": i == 2,
                    "eligible_for_selection": i != 1, "learning_rate": i*.01,
                    "weight_decay": 0., "batch_size": 16 if i % 2 else 32,
                    "dropout": .3, "warmup_fraction": 0.} for i in range(1, 7)]
        opportunities = [{"model": "synthetic", "configuration_id": c["configuration_id"],
                          "status": "known_failed" if i == 0 else "outcome_unknown" if i == 1 else "completed"}
                         for c in configs for i in range(3)]
        summary, levels = closure.parameter_coverage(configs, opportunities)
        self.assertEqual(next(r for r in summary if r["parameter"] == "learning_rate")["distinct_registered_values"], 6)
        fixed = next(r for r in summary if r["parameter"] == "dropout")
        self.assertEqual(fixed["variation"], "fixed")
        self.assertEqual(fixed["winner_configuration_id"], "C02")
        rows = next(r for r in levels if r["parameter"] == "dropout")
        self.assertEqual(rows["planned_fold_opportunities"], 18)
        self.assertEqual(rows["completed_fold_results"], 6)
        self.assertEqual(rows["known_failures"], 6)
        self.assertEqual(rows["unknown_outcomes"], 6)
        self.assertEqual(rows["complete_configurations"], 5)


if __name__ == "__main__":
    unittest.main()
