"""Neural fit/restore contracts on generated OHLCV, never market data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
def module(name):
    spec=importlib.util.spec_from_file_location('test_extended_'+name,HERE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
fitting=module('fitting');panels=module('data')
torch.set_num_threads(1)
CONFIG={'learning_rate':.001,'weight_decay':0.,'batch_size':8,
        'max_epochs':2,'patience':2,'gradient_clip':1.,'per_fit_seconds':60.}


def synthetic_panel():
    rng=np.random.default_rng(400)
    close=np.exp(4+np.cumsum(rng.normal(0,.015,70)))
    op=close*np.exp(rng.normal(0,.002,70))
    high=np.maximum(op,close)*1.005
    low=np.minimum(op,close)*.995
    volume=rng.lognormal(6,.3,70)
    raw=np.stack([op,high,low,close,volume],axis=1)
    dates=np.arange(np.datetime64('2000-01-01'),np.datetime64('2000-03-11')).astype('U10')
    g=panels.old_geometry().windows(raw)
    positions={'fit':np.arange(30),'validation':np.arange(37,47)}
    wd=np.asarray([dates[t-7:t] for t in range(7,len(dates))])
    return panels.package_panel(g['raw_X'],g['y'],g['A'],g['U'],dates[7:],positions,
        dataset='SYNTHETIC',raw_sha='b'*64,window_dates=wd,
        history=dict(zip(dates[7:],g['y'])),geometry={'clock':'synthetic_generated'})


class FittingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.data=synthetic_panel()
        self.identity={'source_commit':'a'*40,'protocol_sha256':'c'*64,
                       'dataset':'SYNTHETIC','real_data_fit':False}

    def tearDown(self):
        self.temporary.cleanup()

    def test_real_legacy_fit_restore_and_common_projection_preserve_frozen_scaler(self):
        scaler=fitting.common.digest(self.data['scaler'])
        model,meta=fitting.fit(fitting.LEGACY,73,self.data,self.root/'legacy',CONFIG,'cpu',
                               source_identity=self.identity)
        self.assertEqual(meta['parameters'],1210)
        self.assertEqual(meta['optimizer_steps'],8)
        self.assertFalse(meta['legacy_initial_bias_calibration_executed'])
        before=fitting.predict(model,fitting.LEGACY,self.data['val_raw_X'],self.data['scaler'],'cpu')
        restored,record=fitting.load_selected(fitting.LEGACY,self.root/'legacy/selected_model.pt',self.data,'cpu')
        after=fitting.predict(restored,fitting.LEGACY,self.data['val_raw_X'],self.data['scaler'],'cpu')
        for first,second in zip(before,after):np.testing.assert_array_equal(first,second)
        np.testing.assert_array_equal(before[1],np.maximum(before[0],before[2]))
        self.assertEqual(fitting.common.digest(self.data['scaler']),scaler)
        self.assertEqual(record['scaler_sha256'],scaler)
        expected=np.sqrt(np.mean((before[1]-self.data['val_y_original'])**2))
        self.assertAlmostEqual(meta['validation_projected_RMSE'],expected,places=14)

    def test_SAE_really_runs_three_phases_under_one_total_and_restores_without_pretraining(self):
        trial={'dropout':.3}
        rows={fitting.SAE:{'trial':trial}}
        model,meta=fitting.fit(fitting.SAE,19,self.data,self.root/'sae',CONFIG,'cpu',rows=rows,
                               source_identity=self.identity)
        self.assertEqual(set(meta['phase_resources']),{'ae1','ae2','supervised'})
        self.assertTrue(meta['SAE_pretraining_executed'])
        self.assertTrue(all(p['optimizer_steps']>0 for p in meta['phase_resources'].values()))
        self.assertEqual(sum(meta['phase_budget_seconds'].values()),CONFIG['per_fit_seconds'])
        self.assertEqual(meta['configuration']['pretraining_executed'],True)
        before=fitting.predict(model,fitting.SAE,self.data['val_raw_X'],self.data['scaler'],'cpu')
        with patch.object(fitting.bounded,'phase',side_effect=AssertionError('restore must not fit')):
            restored,record=fitting.load_selected(fitting.SAE,self.root/'sae/selected_model.pt',self.data,'cpu',rows=rows)
        after=fitting.predict(restored,fitting.SAE,self.data['val_raw_X'],self.data['scaler'],'cpu')
        np.testing.assert_array_equal(before[0],after[0])

    def test_corrupt_selected_bytes_or_changed_scaler_are_refused(self):
        model,meta=fitting.fit(fitting.LEGACY,31,self.data,self.root/'checks',CONFIG,'cpu',
                               source_identity=self.identity)
        altered=dict(self.data);altered['scaler']={**self.data['scaler'],'y_scale':self.data['scaler']['y_scale']*2}
        with self.assertRaisesRegex(ValueError,'scaler binding'):
            fitting.load_selected(fitting.LEGACY,self.root/'checks/selected_model.pt',altered,'cpu')
        path=self.root/'checks/selected_model.pt';path.write_bytes(path.read_bytes()+b'corruption')
        with self.assertRaisesRegex(ValueError,'identity or bytes'):
            fitting.load_selected(fitting.LEGACY,path,self.data,'cpu')

    def test_training_uses_raw_mse_even_when_projection_would_have_zero_gradient(self):
        class Toy(torch.nn.Module):
            def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,x):return self.value.expand(len(x))
        toy=Toy()
        with patch.object(fitting,'_factory',return_value=toy):
            model,meta=fitting.fit('MLP16',51,self.data,self.root/'raw_mse',CONFIG,'cpu',
                                   source_identity=self.identity)
        self.assertGreater(float(model.value.detach()),0.)
        self.assertEqual(meta['training_loss'],'raw_sigma7_MSE')
        self.assertEqual(meta['validation_selection'],'common_projected_sigma7_MSE')

    def test_prediction_normalization_is_independent_of_target_labels(self):
        model,meta=fitting.fit(fitting.LEGACY,42,self.data,self.root/'labels',CONFIG,'cpu',
                               source_identity=self.identity)
        before=fitting.predict(model,fitting.LEGACY,self.data['val_raw_X'],self.data['scaler'],'cpu')
        self.data['val_y_original']=self.data['val_y_original']*100
        after=fitting.predict(model,fitting.LEGACY,self.data['val_raw_X'],self.data['scaler'],'cpu')
        np.testing.assert_array_equal(before[0],after[0])


if __name__=='__main__':unittest.main()
