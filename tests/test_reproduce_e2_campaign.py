"""The orchestrator must reject tampering and never pass omitted mandatory gates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

PROGRAM=Path(__file__).resolve().parents[1]/'scripts/reproduce_e2_campaign.py'
spec=importlib.util.spec_from_file_location('campaign_replay',PROGRAM)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReproductionSafety(unittest.TestCase):
    def test_changed_bytes_rejected_even_with_same_length(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'input.json';p.write_bytes(b'good')
            record={'path':'input.json','bytes':4,'sha256':hashlib.sha256(b'good').hexdigest()}
            self.assertEqual(m.checked_file(root,record),p)
            p.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError,'changed'):m.checked_file(root,record)

    def test_missing_input_is_never_verified(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Missing'):m.checked_file(Path(d),{'path':'absent','bytes':0,'sha256':'0'*64})

    def test_check_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);catalogue=root/'manifest.json';catalogue.write_text(json.dumps({'scientific_source_commit':m.SOURCE}))
            out=root/'existing';out.mkdir();sentinel=out/'receipt.json';sentinel.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'fresh or empty'):
                m.run(argparse.Namespace(inputs_manifest=catalogue,output=out,replay=False))
            self.assertEqual(sentinel.read_text(),'preserve')

    def test_nonexistent_stdlib_python_is_not_ready(self):
        self.assertFalse(m.dependencies(Path('/nonexistent-e2-python'),{})['ready'])

    def test_archive_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);z=root/'bad.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../outside','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe'):m.safe_extract(z,root/'output')
            self.assertFalse((root/'outside').exists())

    def test_missing_dependency_blocks_stage(self):
        result=m.dependencies(Path(sys.executable),{'e2-nonexistent-test-distribution':'1'})
        self.assertFalse(result['ready'])
        self.assertIsNone(result['mismatches']['e2-nonexistent-test-distribution']['observed'])

    def test_completed_analytics_do_not_pass_missing_training_gate(self):
        self.assertEqual(m.finish_status([{'status':'completed'}],[{'id':'training','status':'not_executed'}]),'partial')

    def test_ready_is_not_completed(self):
        self.assertEqual(m.finish_status([{'status':'ready'}],[]),'partial')

    def test_failed_stage_cannot_be_hidden_by_partial(self):
        self.assertEqual(m.finish_status([{'status':'completed'},{'status':'failed'}],[{'id':'missing_data'}]),'failed')

if __name__=='__main__':unittest.main()
