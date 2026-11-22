"""Offline checks; generated diagnostics are not market evidence."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import warnings

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved_adequacy',HERE/'audit_saved_adequacy.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SoftwareTests(unittest.TestCase):
    def test_holm_keeps_all_family_positions_and_monotonicity(self):
        np.testing.assert_allclose(module.holm([.04,.001,.8,.02]),[.08,.004,.8,.06])
        np.testing.assert_array_equal(module.holm([1.,1.,1.]),[1.,1.,1.])

    def test_holm_rejects_nonfinite_or_out_of_range(self):
        for values in ([np.nan,.1],[-.1,.1],[1.01,.1]):
            with self.subTest(values=values),self.assertRaises(ValueError):module.holm(values)

    def test_primary_estimator_is_not_pooled_rmse_or_ensemble(self):
        errors=np.asarray([[0.,2.],[0.,-4.]])
        rmse,mae=module.mean_individual_metrics(errors)
        self.assertAlmostEqual(rmse,1.5*np.sqrt(2))
        self.assertAlmostEqual(mae,1.5)
        self.assertNotAlmostEqual(rmse,np.sqrt((errors**2).mean()))
        self.assertNotAlmostEqual(rmse,np.sqrt((errors.mean(axis=0)**2).mean()))

    def test_time_blocks_are_contiguous_and_bounded(self):
        n,block=55,14
        idx=module.circular_indices(np.random.default_rng(123),n,block,19)
        self.assertEqual(idx.shape,(19,55))
        self.assertTrue(np.all((idx>=0)&(idx<n)))
        for start in range(0,n,block):
            selected=idx[:,start:min(start+block,n)]
            self.assertTrue(np.all(np.diff(selected,axis=1)%n==1))

    def test_shared_time_indices_preserve_identical_model_contrast(self):
        errors=np.arange(275,dtype=float).reshape(5,55)/100
        idx=module.circular_indices(np.random.default_rng(4),55,7,37)
        a=module.mean_individual_metrics(errors,idx)
        b=module.mean_individual_metrics(errors.copy(),idx)
        np.testing.assert_array_equal(a[0]-b[0],np.zeros(37))
        np.testing.assert_array_equal(a[1]-b[1],np.zeros(37))

    def test_deterministic_control_has_one_trajectory(self):
        e=np.arange(55,dtype=float)[None,:]
        idx=module.circular_indices(np.random.default_rng(8),55,7,5)
        a=module.mean_individual_metrics(e,idx)
        self.assertEqual(a[0].shape,(5,))
        self.assertAlmostEqual(a[0][0],np.sqrt(np.mean(e[0,idx[0]]**2)))

    def test_original_zip_integrity_and_alignment(self):
        _,plan,_,dates,_,_,errors,trajectories=module.load_saved(module.ORIGINAL/'inputs/public_results.zip')
        self.assertEqual(len(trajectories),32)
        self.assertEqual(len(dates),55)
        self.assertEqual(len(plan['seeds']),5)
        self.assertEqual(errors['input_only_mse'].shape,(5,55))
        self.assertEqual(errors['floor_only'].shape,(1,55))
        c=module.mean_individual_metrics(errors['input_only_mse'])[0]
        g=module.mean_individual_metrics(errors['global_mse'])[0]
        self.assertAlmostEqual(c-g,-5.821987228066001e-5,places=16)

    def test_modified_source_zip_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'changed.zip';p.write_bytes(b'NOT THE ORIGINAL SCIENTIFIC EVIDENCE')
            with self.assertRaisesRegex(ValueError,'immutable'):module.load_saved(p)

    def test_unsupported_mz_stays_in_128_position_family(self):
        rng=np.random.default_rng(9);target=1.+rng.uniform(0,.1,55)
        trajectories=[]
        for i in range(32):
            p=np.ones(55)*(1.+i*.001)
            trajectories.append(dict(recipe='SYNTHETIC',seed=i,prediction=p,error=p-target))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rows,bias=module.diagnostic_family(trajectories,target,1.)
        mz=[r for r in rows if r['test']=='MZ_HAC14']
        self.assertEqual(len(rows),128);self.assertEqual(len(bias),64);self.assertEqual(len(mz),32)
        self.assertTrue(all(not r['supported'] and r['p_value']==1. and r['p_Holm128']==1. for r in mz))

    def test_existing_results_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sentinel=root/'original.txt';sentinel.write_text('PRESERVE')
            with self.assertRaisesRegex(AssertionError,'immutable'):
                module.run(root/'absent.zip',root/'absent',root,root/'absent-plan.json')
            self.assertEqual(sentinel.read_text(),'PRESERVE')

    def test_existing_outputs_verify_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'results';shutil.copytree(HERE/'results',target)
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                module.verify(module.ORIGINAL/'inputs/public_results.zip',target,HERE/'analysis_plan.json')
            self.assertTrue(json.loads(captured.getvalue())['no_resampling'])
            altered=target/'diagnostic_summary.csv';altered.write_bytes(altered.read_bytes()+b'changed')
            with self.assertRaises(AssertionError):
                module.verify(module.ORIGINAL/'inputs/public_results.zip',target,HERE/'analysis_plan.json')


if __name__=='__main__':unittest.main(verbosity=2)
