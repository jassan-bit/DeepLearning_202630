import argparse
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_e2_reduced_delivery as delivery


class DeliveryTests(unittest.TestCase):
    def test_all_stages_share_budget_and_backup_then_score_only_after_freeze(self):
        events=[]
        class Budget:
            remaining_seconds=7472
            def report(self):return {'remaining_seconds':self.remaining_seconds}
        budget=Budget()
        class Backup:
            def persist_result(self,*a):events.append('backup')
        backup=Backup()
        def selection(args,current,*a,**kwargs):
            self.assertIs(current,budget);events.append('selection')
            (args.output/'selection_complete.json').write_text('{}')
            return {'selections':{'FCN-1D':{}}}
        def fitted(*args,**kwargs):
            self.assertIs(args[3],budget);events.append('fits');return {'pending':[]}
        def classical(*args,**kwargs):self.assertIs(args[2],budget);events.append('classical')
        def score(*args,**kwargs):self.assertIs(args[3],budget);events.append('score');return {'status':'completed','systems':16}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,data in [('runtime.json','{"runtime":{}}'),('roots.json','{}'),('audit.json','{}')]:
                (root/name).write_text(data)
            args=argparse.Namespace(output=root,stage='all',runtime_profile=root/'runtime.json',source_roots=root/'roots.json',
                cache_audit=root/'audit.json',reconciliation_root=root,data_root=root,models=None,python=sys.executable,
                daily_csv=None,raw_root=None,allow_final_download=False)
            with mock.patch.object(delivery,'_source',return_value={}), \
                 mock.patch('run_e2_reduced_selection.run_selection',side_effect=selection), \
                 mock.patch('e2_reduced_final.run_final_fits',side_effect=fitted), \
                 mock.patch('e2_reduced_final.run_classical',side_effect=classical), \
                 mock.patch('e2_reduced_final.close_pilot',side_effect=lambda *a,**k:events.append('power')), \
                 mock.patch('e2_reduced_final.freeze_final',side_effect=lambda *a,**k:events.append('freeze')), \
                 mock.patch('e2_reduced_final.run_final_scoring',side_effect=score):
                report=delivery.execute_delivery(args,budget,backup)
            self.assertEqual([e for e in events if e!='backup'],['selection','fits','power','classical','freeze','score'])
            self.assertLess(events.index('backup'),events.index('fits'))
            self.assertTrue(report['all_sixteen_systems_benchmark_complete'])

    def test_planning_is_read_only_and_inherited_ledgers_are_accepted(self):
        args=delivery.parser().parse_args(['--inherited-ledger','/tmp/first.jsonl','--inherited-ledger','/tmp/second.jsonl'])
        self.assertEqual(len(args.inherited_ledger),2)
        self.assertFalse(args.execute)
        self.assertEqual(delivery.main([]),0)


if __name__=='__main__':unittest.main()
