"""No learning: real input binding and no-bypass execution gates."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parents[1]/'scripts/e2_reproduce_from_raw.py'
S=importlib.util.spec_from_file_location('real_repro',P);m=importlib.util.module_from_spec(S);S.loader.exec_module(m)

class RealReproductionGates(unittest.TestCase):
    def test_changed_source_same_size_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'source.py';p.write_bytes(b'good');row=m.record(root,p)
            p.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError,'changed'):m.verify(root,row)

    def test_missing_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Missing'):m.verify(Path(d),{'path':'absent','bytes':0,'sha256':'0'*64})

    def test_no_direct_worker_without_active_parent_reservation(self):
        class Ledger:
            LEDGER=Path('/unused')
            class Context:
                def __enter__(self):return [],None
                def __exit__(self,*args):pass
            def locked_ledger(self,p):return self.Context()
            def summarize(self,rows):return {'open_reservations':{}}
        with patch.object(m,'budget_module',return_value=Ledger()):
            with self.assertRaisesRegex(ValueError,'active parent'):
                m.require_worker_lease(argparse.Namespace(reservation_id=None))

    def test_partial_job_never_treated_as_completed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'request.json').write_text('{"phase":"running"}')
            self.assertFalse(m.completed_job(root))

    def test_durable_real_api_is_used_without_CPU_test_override(self):
        import ast
        tree=ast.parse(P.read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden=[n for n in calls if any(k.arg=='allow_cpu_for_tests' and isinstance(k.value,ast.Constant) and k.value.value for k in n.keywords)]
        self.assertFalse(forbidden)
        self.assertEqual(78,len(m.NEURAL)*len(m.SEEDS));self.assertEqual(3,len(m.CLASSICAL))

if __name__=='__main__':unittest.main()
