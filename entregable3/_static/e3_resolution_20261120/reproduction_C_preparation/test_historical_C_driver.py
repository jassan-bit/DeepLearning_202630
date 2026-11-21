"""Prepared C execution guards; all tests use CPU stdlib and never train."""
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import historical_C_driver as c

HOME=Path(__file__).resolve().parent

def remove_case(root):
    if root.exists():
        for p in root.rglob('*'):
            try:p.chmod(0o755 if p.is_dir() else 0o644)
            except OSError:pass
        root.chmod(0o755);shutil.rmtree(root)

def prepared(root):
    case=root/'case'
    result=subprocess.run([sys.executable,str(HOME/'prepare_C_case.py'),'--case-root',str(case)],capture_output=True,text=True)
    if result.returncode:raise AssertionError(result.stderr)
    return case

def request_fixture():
    r=json.loads((HOME/'C_REQUEST.template.json').read_bytes())
    r.update(new_run_id='C-software-guard-fixture',allow_training=True,recorded_at_utc=datetime.now(timezone.utc).isoformat())
    r['operator']={'name':'SOFTWARE GUARD FIXTURE','relationship_to_design':'Synthetic schema fixture, not a person approval','independent_of_design':True}
    r['tolerances']['prediction_sigma7']={'abs_tol':1e-8,'rel_tol':0.,'rationale':'Synthetic guard fixture only; not recommended human C tolerance.'}
    r['acknowledgements']={key:True for key in r['acknowledgements']}
    return r

class Guards(unittest.TestCase):
    def setUp(self):self.root=Path(tempfile.mkdtemp(prefix='e3-C-guard-'))
    def tearDown(self):remove_case(self.root)
    def test_source_and_reference_byte_pins(self):
        case=prepared(self.root);manifest,report=c.inspect(case)
        self.assertEqual(report['source_and_reference_files_verified'],12)
        self.assertFalse(report['input_present_and_hash_verified'])
        self.assertEqual(len(manifest['fixed_jobs']),2)
    def test_default_dry_run_imports_no_scientific_runtime(self):
        case=prepared(self.root)
        code="import runpy,sys,json;sys.argv=['historical_C_driver.py','--case-root',"+repr(str(case))+"];\ntry:runpy.run_path("+repr(str(HOME/'historical_C_driver.py'))+",run_name='__main__')\nexcept SystemExit as e:assert e.code==0\nprint('MODULE_GUARD:'+json.dumps({'numpy':'numpy' in sys.modules,'torch':'torch' in sys.modules}))"
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        modules=json.loads(r.stdout.split('MODULE_GUARD:')[1]);self.assertEqual(modules,{'numpy':False,'torch':False})
        self.assertIn('prepared_C_subset_dry_run_only',r.stdout);self.assertFalse((case/'output').exists())
    def test_template_execution_blocked_before_runtime(self):
        case=prepared(self.root);r=subprocess.run([sys.executable,str(HOME/'historical_C_driver.py'),'--case-root',str(case),'--execute'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('Execution blocked before runtime import',r.stderr);self.assertFalse((case/'output').exists())
    def test_source_corruption_rejected(self):
        case=prepared(self.root);f=case/'input/frozen_sources/src/original_model.py';f.chmod(0o644);f.write_bytes(f.read_bytes()+b'\n#modified')
        with self.assertRaisesRegex(ValueError,'changed'):c.inspect(case)
    def test_writable_original_copy_rejected(self):
        case=prepared(self.root);f=case/'input/frozen_sources/src/original_model.py';f.chmod(0o644)
        with self.assertRaisesRegex(ValueError,'writable'):c.inspect(case)
    def test_bad_input_hash_rejected(self):
        case=prepared(self.root);f=case/'input/data/cohort.npz';f.parent.chmod(0o755) if f.parent.exists() else None
        (case/'input').chmod(0o755);f.parent.mkdir();f.write_bytes(b'not the pinned NPZ');f.chmod(0o444);f.parent.chmod(0o555);(case/'input').chmod(0o555)
        with self.assertRaisesRegex(ValueError,'NPZ input differs'):c.inspect(case)
    def test_old_master_and_resource_override_rejected(self):
        m=json.loads((HOME/'SOURCE_MANIFEST.json').read_bytes());r=request_fixture()
        self.assertEqual(c.request_issues(r,m),[])
        r['new_run_id']='C-'+c.OLD_IDS[0];r['resources']['max_nominal_fit_seconds']=3600
        issues=c.request_issues(r,m);self.assertIn('fresh C run ID',issues);self.assertIn('two120s nominal fit opportunities',issues)
    def test_scope_or_tolerance_missing_rejected(self):
        m=json.loads((HOME/'SOURCE_MANIFEST.json').read_bytes());r=request_fixture();r['tolerances']['prediction_sigma7']['abs_tol']=float('nan');r['acknowledgements']['not_confirmation']=False
        issues=c.request_issues(r,m);self.assertIn('finite nonnegative predeclared C tolerances',issues);self.assertIn('acknowledgement:not_confirmation',issues)
    def test_case_overwrite_rejected(self):
        case=prepared(self.root);r=subprocess.run([sys.executable,str(HOME/'prepare_C_case.py'),'--case-root',str(case)],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('New case directory required',r.stderr)
    def test_references_have_935_rows_and_fixed_110_subset(self):
        with (HOME/'references/assessment_predictions.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),935)
        neural=[r for r in rows if r['seed']!='analytic'];self.assertEqual(len(neural),825)
        for name in c.PAIR:self.assertEqual(sum(r['name']==name and r['seed']=='20261118' for r in rows),55)

if __name__=='__main__':unittest.main()
