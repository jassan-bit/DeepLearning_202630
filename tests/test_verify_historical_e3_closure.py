"""Append-only and retained-integrity guards; fixtures do not run experiments."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('historical_closure',REPO/'scripts/verify_historical_e3_closure.py')
v=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(v)
OLD_SOURCE=(REPO/'experiments/e3_evidence_close_20261120/verify_closure.py').read_bytes()

def write(root,path,body):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(body)

def record(path,body,with_bytes=True):
    d={'path':path,'sha256':hashlib.sha256(body).hexdigest()}
    if with_bytes:d['bytes']=len(body)
    return d

class HistoricalClosureGuards(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.prefix=b'historical design record\n';self.append=b'new bounded development decision\n'
        scope={k:0 for k in ['new_forecasting_model_fits','new_forecasting_model_predictions',
               'new_statistical_resamples','new_confirmatory_superiority_tests']}
        scope.update(R1_accredited=False,R3_accredited=False,automatic_promotion_of_best_ablation=False)
        self.manifest={'closure_payloads':[record('verify_closure.py',OLD_SOURCE),record('result.txt',b'result')],
          'optional_downloads':[record('inventory/bundle.zip',b'zipfixture')],
          'edited_root_files':[record(v.APPEND_ONLY,self.prefix,False),record('app.py',b'oldapp',False)],
          'protected_existing_files':[record('models/frozen.bin',b'frozenweights',False)],'scope':scope}
        for folder in [str(v.CLOSURE),str(v.STATIC)]:
            write(self.root,folder+'/verify_closure.py',OLD_SOURCE)
            write(self.root,folder+'/result.txt',b'result')
            write(self.root,folder+'/inventory/bundle.zip',b'zipfixture')
            write(self.root,folder+'/closure_manifest.json',json.dumps(self.manifest).encode())
        write(self.root,v.APPEND_ONLY,self.prefix+self.append)
        write(self.root,'app.py',b'oldapp');write(self.root,'models/frozen.bin',b'frozenweights')
        self.patches=[patch.object(v,'MANIFEST_SHA256',v.sha(self.root/str(v.CLOSURE/'closure_manifest.json'))),
                      patch.object(v,'HISTORICAL_LOG_BYTES',len(self.prefix)),
                      patch.object(v,'HISTORICAL_LOG_SHA256',hashlib.sha256(self.prefix).hexdigest())]
        for item in self.patches:item.start()
    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.temp.cleanup()
    def test_append_passes_and_actual_inputs_are_unchanged(self):
        before={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result=v.verify_historical(self.root)
        self.assertEqual(result['current_design_log']['appended_bytes'],len(self.append))
        self.assertEqual(result['historical_component_scope_only']['protected_files_absent'],0)
        after={p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_no_append_also_passes(self):
        write(self.root,v.APPEND_ONLY,self.prefix)
        self.assertEqual(v.verify_historical(self.root)['current_design_log']['appended_bytes'],0)
    def test_prepend_fails(self):
        write(self.root,v.APPEND_ONLY,b'new'+self.prefix+self.append)
        with self.assertRaisesRegex(ValueError,'prefix changed'):v.verify_historical(self.root)
    def test_prefix_mutation_fails(self):
        write(self.root,v.APPEND_ONLY,b'H'+self.prefix[1:]+self.append)
        with self.assertRaisesRegex(ValueError,'prefix changed'):v.verify_historical(self.root)
    def test_truncation_fails(self):
        write(self.root,v.APPEND_ONLY,self.prefix[:-1])
        with self.assertRaisesRegex(ValueError,'truncated'):v.verify_historical(self.root)
    def test_other_edited_file_change_fails(self):
        write(self.root,'app.py',b'oldapp newtail')
        with self.assertRaisesRegex(ValueError,'Edited source differs'):v.verify_historical(self.root)
    def test_protected_corruption_fails(self):
        write(self.root,'models/frozen.bin',b'changedweights')
        with self.assertRaisesRegex(ValueError,'artifact changed'):v.verify_historical(self.root)
    def test_missing_protected_is_not_allowed(self):
        (self.root/'models/frozen.bin').unlink()
        with self.assertRaisesRegex(ValueError,'Missing source'):v.verify_historical(self.root)
    def test_mirror_corruption_still_fails(self):
        write(self.root,str(v.STATIC/'result.txt'),b'corrupt')
        with self.assertRaisesRegex(ValueError,'Published download differs'):v.verify_historical(self.root)
    def test_optional_mirror_corruption_still_fails(self):
        write(self.root,str(v.STATIC/'inventory/bundle.zip'),b'corrupt')
        with self.assertRaisesRegex(ValueError,'Optional published download differs'):v.verify_historical(self.root)
    def test_frozen_verifier_change_fails(self):
        write(self.root,str(v.CLOSURE/'verify_closure.py'),OLD_SOURCE+b'\n')
        with self.assertRaisesRegex(ValueError,'verifier changed'):v.verify_historical(self.root)
    def test_symlinked_file_fails(self):
        target=self.root/'models/frozen.bin';target.unlink();target.symlink_to(self.root/'app.py')
        with self.assertRaisesRegex(ValueError,'Symlinked'):v.verify_historical(self.root)
    def test_symlinked_ancestor_fails(self):
        (self.root/'models/frozen.bin').unlink();(self.root/'models').rmdir()
        (self.root/'models').symlink_to(self.root/'experiments',target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlinked'):v.verify_historical(self.root)
    def test_path_escapes_rejected(self):
        for value in ['../file','/absolute','a/../../escape','a\\b','a/./b']:
            with self.subTest(path=value),self.assertRaises(ValueError):v.member(self.root,value)

if __name__=='__main__':unittest.main(verbosity=2)
