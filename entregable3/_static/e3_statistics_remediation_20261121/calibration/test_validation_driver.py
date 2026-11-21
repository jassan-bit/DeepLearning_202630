"""Provenance/resource and reporting guards; no fresh validation is executed."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from run_validation import preflight, CP, screen, summarize, execute, RUNTIME_FILES


HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(root):
    files=[]
    for n in sorted(RUNTIME_FILES):
        p=root/n;files.append({'path':n,'bytes':p.stat().st_size,'sha256':digest(p)})
    (root/'CODE_MANIFEST.json').write_text(json.dumps({'kind':'frozen_before_new_controlled_validation','files':files}))
    anchor={'status':'MCS_bootstrap_t_code_and_plan_published_before_validation','observed_by_root_before_validation':True,
            'commit':'a'*40,'plan_SHA256':digest(root/'VALIDATION_PLAN.json'),'source_manifest_SHA256':digest(root/'CODE_MANIFEST.json')}
    (root/'MOCK_ANCHOR.json').write_text(json.dumps(anchor));return root/'MOCK_ANCHOR.json'


class DriverGuards(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='mcs-driver-software-test-');self.root=Path(self.temp.name)
        for n in RUNTIME_FILES:shutil.copyfile(HERE/n,self.root/n)
        self.anchor=freeze(self.root)
        self.env=patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'});self.env.start()

    def tearDown(self):self.env.stop();self.temp.cleanup()

    def alter_anchor(self,key,value):
        d=json.loads(self.anchor.read_text());d[key]=value;self.anchor.write_text(json.dumps(d))

    def test_preflight_software_mock_does_not_execute(self):
        plan,_,_=preflight(self.root,self.anchor);self.assertEqual(plan['planned_replicates'],640)
        self.assertFalse((self.root/'validation_fresh').exists())

    def test_anchor_status_is_required(self):
        self.alter_anchor('status','draft')
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_observed_public_anchor_is_required(self):
        self.alter_anchor('observed_by_root_before_validation',False)
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_exact_commit_is_required(self):
        self.alter_anchor('commit','main')
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_zero_sha_is_not_public_commit(self):
        self.alter_anchor('commit','0'*40)
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_source_change_rejected(self):
        with (self.root/'crossed_rmse_v2.py').open('a') as f:f.write('\n# mutation\n')
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_omitted_runtime_file_rejected(self):
        m=self.root/'CODE_MANIFEST.json';v=json.loads(m.read_text());v['files']=v['files'][1:];m.write_text(json.dumps(v))
        self.alter_anchor('source_manifest_SHA256',digest(m))
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_plan_change_rejected(self):
        p=self.root/'VALIDATION_PLAN.json';v=json.loads(p.read_text());v['max_wall_seconds']=301;p.write_text(json.dumps(v))
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_even_rehashed_budget_change_is_rejected(self):
        p=self.root/'VALIDATION_PLAN.json';v=json.loads(p.read_text());v['max_wall_seconds']=301;p.write_text(json.dumps(v));freeze(self.root)
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_old_validation_seed_rejected(self):
        p=self.root/'VALIDATION_PLAN.json';v=json.loads(p.read_text());v['validation_seed']=2026112107;p.write_text(json.dumps(v));freeze(self.root)
        with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_thread_profile_required(self):
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'8'}):
            with self.assertRaises(ValueError):preflight(self.root,self.anchor)

    def test_existing_output_never_overwritten(self):
        out=self.root/'existing';out.mkdir();(out/'proof.txt').write_text('preserve')
        with self.assertRaises(ValueError):execute(self.root,out,self.anchor)
        self.assertEqual((out/'proof.txt').read_text(),'preserve')

    def test_binomial_boundaries_are_finite(self):
        self.assertEqual(CP(0,128)['CI95'][0],0.);self.assertEqual(CP(128,128)['CI95'][1],1.)
        self.assertGreater(CP(0,128)['CI95'][1],0.)

    def test_screen_does_not_certify_nominal_or_market(self):
        x=screen(CP(0,128),.1);self.assertEqual(x['screen'],'exploratory_tolerance_screen_passed')
        self.assertFalse(x['exact_nominal_level_certified']);self.assertFalse(x['market_calibration_certified'])

    def test_unsupported_outputs_are_not_calibration_merit(self):
        names=['candidate','rival_01'];m={'initial_global_p':None,'retained':names}
        pair={'theta_hat':.1,'P1_equal_two_sided_p':1.,'P2_equal_two_sided_p':1.,'P1_Holm':1.,'P2_Holm':1.,
              'P1_CI95':None,'P2_CI95':None,'nondegenerate':False}
        result={'pairwise':[pair],**{k:m for k in ['MCS_bootstrap_t','MCS_fixed_studentizer_development_control','MCS_legacy_frozen_control_same_draws']}}
        case={'id':'software_guard','replicates':1,'R':12,'T':128,'M':2,'delta':0.,'theta':'equal'}
        out=summarize(case,[{'result':result}]);self.assertEqual(out['MCS_bootstrap_t']['initial_global_rejection']['screen'],'inconclusive_unsupported_replicates')
        self.assertEqual(out['known_two_method_subset_FWER_Holm38']['screen'],'inconclusive_unsupported_replicates')
        self.assertEqual(out['per_rival'][0]['methods']['P1']['pointwise_CI95_coverage']['rate'],0.)
        self.assertIsNone(out['per_rival'][0]['methods']['P1']['mean_CI_width'])
        self.assertTrue(all(x['exact_McNemar_two_sided_p']==1. and x['unsupported_pairs']==1
                            and 'inconclusive' in x['status'] for x in out['paired_variant_diagnostics']))

    def test_legacy_function_bodies_are_verbatim(self):
        refs=json.loads((HERE/'LEGACY_REFERENCE.json').read_text())
        for record in refs['references']:
            text=(HERE/record['new_filename']).read_text();lines=text.splitlines(keepends=True)
            nodes={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
            for item in record['functions']:
                n=nodes[item['name']];body=''.join(lines[n.lineno-1:n.end_lineno])
                self.assertEqual(hashlib.sha256(body.encode()).hexdigest(),item['verbatim_function_SHA256'])


if __name__=='__main__':unittest.main()
