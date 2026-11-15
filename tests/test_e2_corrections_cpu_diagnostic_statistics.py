"""CPU-only domain and paired-block checks on synthetic inputs."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e2_corrections_cpu_diagnostic_statistics as cpu


class CPUDiagnosticTests(unittest.TestCase):
    def risks(self):
        return [{"model": model, "seed": str(seed), "train_count": "655", "calibration_count": "53",
                 "state_before_sha256": "a"*64, "state_after_sha256": "a"*64,
                 "device": "cpu", "batch_size": "1", "train_rmse": ".01",
                 "calibration_rmse": str(.0101+(j-2.5)*.00001),
                 "projection_alert": "True", "max_absolute_prediction_difference": "100"}
                for model in cpu.NEURAL for j, seed in enumerate(cpu.SEEDS)]

    def test_projection_alerts_do_not_filter_any_CPU_model(self):
        result = cpu.gaps_from_rows(self.risks(), .00025)
        self.assertEqual(len(result), 13)
        self.assertEqual({r["model"] for r in result}, set(cpu.NEURAL))
        for row in result:
            self.assertAlmostEqual(row["mean"], .0001)
            self.assertTrue(row["within_tolerance_conditionally"])

    def test_missing_predictor_blocks_global_statistics(self):
        with self.assertRaises(ValueError): cpu.gaps_from_rows(self.risks()[:-1], .00025)

    def test_no_GPU_risk_or_mutated_state_can_enter(self):
        rows = self.risks(); rows[0]["device"] = "cuda:0"
        with self.assertRaises(ValueError): cpu.gaps_from_rows(rows, .00025)
        rows = self.risks(); rows[0]["state_after_sha256"] = "b"*64
        with self.assertRaises(ValueError): cpu.gaps_from_rows(rows, .00025)

    def test_zero_seed_variance_remains_not_estimable(self):
        rows = self.risks()
        for row in rows: row["calibration_rmse"] = ".01"
        result = cpu.gaps_from_rows(rows, .00025)
        self.assertTrue(all(row["decision"] == "not_estimable" for row in result))
        self.assertTrue(all(row["p_Holm13"] is None for row in result))

    def test_cost_preserves_full_six_seed_blocks_and_units(self):
        rows = []
        for block in range(2):
            for j, seed in enumerate(cpu.SEEDS):
                for position, role in enumerate(("candidate", "reference")):
                    elapsed = 10+j if role == "candidate" else 20+j
                    rows.append({"candidate": "A", "baseline": "GRU32", "block_id": str(block), "seed": str(seed),
                                 "role": role, "model": "A" if role == "candidate" else "GRU32", "device": "cpu",
                                 "batch_size": "1", "pair_id": f"{block}:{seed}", "input_date": "2024-01-01",
                                 "order": "AB", "position": str(position), "elapsed_ns": str(elapsed),
                                 "start_perf_counter_ns": "100", "end_perf_counter_ns": str(100+elapsed)})
        plan = {"cost_analysis": {"candidates": ["A"], "blocks": 2, "bootstrap_seed": 17, "replicates": 99}}
        result, blocks = cpu.cost_from_rows(rows, plan)
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(result[0]["CPU_latency_difference_ms"], -1e-5)
        self.assertAlmostEqual(result[0]["difference_CI95_lower_ms"], -1e-5)
        self.assertAlmostEqual(result[0]["p_cost_lower_raw"], .01)
        self.assertNotIn("p_IUT", result[0])


if __name__ == "__main__": unittest.main()
