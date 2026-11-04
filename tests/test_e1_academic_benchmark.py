from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from plan_e1_academic_benchmark import directional_power,required_repetitions


class AcademicBoundaryPlanningTests(unittest.TestCase):
    def test_minimal_R_for_positive_separation(self):
        n=required_repetitions(.4,1.,.025,.9)
        self.assertGreaterEqual(directional_power(n,.4,1.,.025),.9)
        self.assertLess(directional_power(n-1,.4,1.,.025),.9)

    def test_no_power_claim_on_or_below_boundary(self):
        for gap in [0.,-.1]:
            with self.assertRaises(ValueError):required_repetitions(gap,1.,.025,.9)

    def test_more_variability_needs_more_R(self):
        self.assertGreater(required_repetitions(.4,2.,.025,.9),required_repetitions(.4,1.,.025,.9))


if __name__=="__main__":unittest.main()
