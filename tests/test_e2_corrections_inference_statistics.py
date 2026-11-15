"""Analytic boundaries and pairing requirements for the new diagnostics."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_corrections_inference_statistics as diagnostics


class StatisticsTests(unittest.TestCase):
    def test_lower_tail_boundary_and_direction(self):
        x=np.array([-3,-2,-1,1,2,3],dtype=float)
        at=diagnostics.six_seed_test(x+10,10)
        self.assertAlmostEqual(at["p_raw"],.5)
        below=diagnostics.six_seed_test(x-20,0)
        above=diagnostics.six_seed_test(x+20,0)
        self.assertLess(below["p_raw"],.001)
        self.assertGreater(above["p_raw"],.999)

    def test_zero_seed_variance_never_manufactures_significance(self):
        row=diagnostics.six_seed_test(np.full(6,.1),1)
        self.assertIsNone(row["p_raw"])
        self.assertIsNone(row["CI95_lower"])

    def test_family_keeps_missing_contrast(self):
        self.assertEqual(diagnostics.holm([.01,None,.04]),[.03,None,.08])

    def test_centering_uses_margin_and_tail(self):
        gain=diagnostics.centered_summary(.6,np.array([.5,.7]),.2)
        cost=diagnostics.centered_summary(-.6,np.array([-.7,-.5]),0,lower=True)
        self.assertAlmostEqual(gain["p_raw"],1/3)
        self.assertAlmostEqual(cost["p_raw"],1/3)

    def rows(self):
        rows=[]
        for block in range(2):
            for seed in (1,2):
                for pos,role in enumerate(("candidate","reference")):
                    elapsed=10+seed if role=="candidate" else 4+seed
                    rows.append(dict(candidate="A",baseline="GRU32",block_id=str(block),seed=str(seed),
                        role=role,model="A" if role=="candidate" else "GRU32",device="cpu",batch_size="1",
                        elapsed_ns=str(elapsed),start_perf_counter_ns="100",end_perf_counter_ns=str(100+elapsed),
                        pair_id=f"A:{block}:{seed}",input_date=f"2024-01-0{block+1}",order="AB",position=str(pos)))
        return rows

    def test_timing_pairs_preserve_seeds_and_complete_blocks(self):
        a,b=diagnostics.latency_matrices(self.rows(),blocks=2,seeds=(1,2),candidates=("A",))
        np.testing.assert_allclose(a-b,np.full((2,1),6e-9))
        with self.assertRaises(ValueError):
            diagnostics.latency_matrices(self.rows()[:-1],blocks=2,seeds=(1,2),candidates=("A",))

    def test_duplicate_and_mismatched_timing_rejected(self):
        rows=self.rows()
        with self.assertRaises(ValueError):
            diagnostics.latency_matrices(rows+[rows[0]],blocks=2,seeds=(1,2),candidates=("A",))
        rows[1]["input_date"]="2024-02-01"
        with self.assertRaises(ValueError):
            diagnostics.latency_matrices(rows,blocks=2,seeds=(1,2),candidates=("A",))


if __name__=="__main__":
    unittest.main()
