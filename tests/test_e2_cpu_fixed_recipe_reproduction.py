"""No fits: admission completeness, source protection and per-worker charging."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parents[1]/'scripts/e2_cpu_fixed_recipe_reproduction.py'
S=importlib.util.spec_from_file_location('cpu_replica',P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)

class CPUReplicaGates(unittest.TestCase):
    def test_one_architecture_cannot_admit_thirteen(self):
        with self.assertRaisesRegex(ValueError,'All13'):
            m.timing_projection({'FCN-1D':23.2},['FCN-1D','TimesNet'])

    def test_runtime_projection_is_sum_of_six_fits_not_parallel_parent_wall(self):
        self.assertEqual(m.timing_projection({'a':10,'b':20},['a','b'],factor=1.5,overhead=1200),1470)
        with self.assertRaisesRegex(ValueError,'positive'):m.timing_projection({'a':float('nan')},['a'])

    def test_each_worker_is_charged_once_and_parent_is_not_added(self):
        class Budget:
            def __init__(self):self.finished=[]
            def finish(self,*args):self.finished.append(args)
        budget=Budget()
        class Base:
            def budget_module(self):return budget
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);planfile=root/'plan';planfile.write_text('immutable')
            args=argparse.Namespace(output=root,python=Path('/python'),plan=planfile,source_root=root,author_root=root,raw_root=root)
            plan={'runtime_policy':{'worker_cpu_sets':[[0,1],[2,3]]}}
            with patch.object(m.subprocess,'run',return_value=argparse.Namespace(returncode=0)),patch.object(m.time,'monotonic',side_effect=[100.,110.,200.,215.]):
                a=m.reserved_worker(args,plan,Base(),{'id':'a'},0,'reservation-a',20)
                b=m.reserved_worker(args,plan,Base(),{'id':'b'},1,'reservation-b',25)
            self.assertEqual([(r[0],r[1]) for r in budget.finished],[('reservation-a',10.),('reservation-b',15.)])
            self.assertEqual(a['charged_worker_wall_seconds']+b['charged_worker_wall_seconds'],25.)
            self.assertFalse(a['parent_wall_seconds_added']);self.assertFalse(b['parent_wall_seconds_added'])

    def test_resume_uses_unspent_group_cap_after_previous_jobs(self):
        self.assertEqual(m.remaining_budget_gate(10000,8000,2500,1000),2000)
        with self.assertRaisesRegex(ValueError,'Remaining'):
            m.remaining_budget_gate(10000,8000,1500,1000)
        with self.assertRaisesRegex(ValueError,'Remaining'):
            m.remaining_budget_gate(10000,8000,2500,2100)

    def test_unbudgeted_worker_cannot_fit(self):
        class Budget:
            LEDGER=Path('/unused')
            class Context:
                def __enter__(self):return [],None
                def __exit__(self,*args):pass
            def locked_ledger(self,path):return self.Context()
            def summarize(self,rows):return {'open_reservations':{}}
        class Base:
            def budget_module(self):return Budget()
        with self.assertRaisesRegex(ValueError,'active parent'):
            m.lease(argparse.Namespace(reservation_id=None,job='a'),Base())

if __name__=='__main__':unittest.main()
