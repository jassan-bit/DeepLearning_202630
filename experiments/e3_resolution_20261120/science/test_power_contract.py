"""Software-only contract checks. No markets, model fits or power claims."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('contract',Path(__file__).with_name('power_contract.py'))
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
RIVALS=['r'+str(i) for i in range(19)]

def fixture():
    return [{'stage':'planning_simulation','dataset_id':'SOFTWARE_FIXTURE','R':'10','T':'100','replicate':str(k),'rival':r,'procedure':p,'p_raw':'0.000001'}
            for k in range(25) for r in RIVALS for p in c.PROCEDURES]

class Contracts(unittest.TestCase):
    def test_holm_monotone_and_original_order(self):
        self.assertEqual(c.holm([.02,.001,.03]),[.04,.003,.04])
    def test_one_missing_component_rejected(self):
        rows=fixture();rows.pop()
        with self.assertRaises(ValueError):c.summarize_rows(rows,RIVALS)
    def test_duplicate_rejected(self):
        rows=fixture();rows.append(rows[0])
        with self.assertRaises(ValueError):c.summarize_rows(rows,RIVALS)
    def test_mixed_calendars_or_budgets_rejected(self):
        rows=fixture();rows[0]['T']='200'
        with self.assertRaises(ValueError):c.summarize_rows(rows,RIVALS)
    def test_observed_confirmation_results_rejected(self):
        rows=fixture();rows[0]['stage']='confirmation'
        with self.assertRaises(ValueError):c.summarize_rows(rows,RIVALS)
    def test_one_weak_method_blocks_intersection(self):
        rows=fixture()
        for row in rows:
            if row['procedure']==c.PROCEDURES[2]:row['p_raw']='.7'
        report=c.summarize_rows(rows,RIVALS)
        self.assertEqual(report['conditional_joint_power_estimate'],0)
        self.assertTrue(all(r['conditional_power_estimate']==0 for r in report['rival_intersection_power']))
    def test_perfect_fixture_does_not_certify_power(self):
        report=c.summarize_rows(fixture(),RIVALS)
        self.assertEqual(report['conditional_joint_power_estimate'],1)
        self.assertFalse(report['all_rival_MC_lower_bounds_at_least90'])
        self.assertFalse(report['confirmation_enabled'])
    def test_invalid_pvalue_rejected(self):
        rows=fixture();rows[0]['p_raw']='NaN'
        with self.assertRaises(ValueError):c.summarize_rows(rows,RIVALS)
    def test_wilson_conservative_for_small_M(self):
        self.assertLess(c.wilson_lower(25,25,.05/19),.9)
        self.assertLess(c.wilson_lower(0,25,.05),1e-12)
    def test_exact_binomial_boundary_and_interior(self):
        self.assertEqual(c.clopper_pearson_lower(0,10,.05),0)
        self.assertAlmostEqual(c.clopper_pearson_lower(10,10,.05),.05**.1)
        self.assertLess(c.clopper_pearson_lower(5,10,.05),.5)
        self.assertAlmostEqual(c.clopper_pearson_lower(1,10,.05),1-.95**.1,places=10)
    def test_no_go_cannot_be_overridden_by_family_alone(self):
        config={'science':{},'family':{'rivals':RIVALS,'procedures':list(c.PROCEDURES)}}
        with tempfile.TemporaryDirectory() as tmp:report=c.readiness(config,Path(tmp))
        self.assertIn('renewed_scientific_rationale_review',report['blockers'])
        self.assertFalse(report['confirmation_enabled'])
    def test_evidence_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(c.bound({'evidence_bindings':{'x':{'path':'../other','sha256':'a'*64}}},Path(tmp),'x'))

if __name__=='__main__':unittest.main()
