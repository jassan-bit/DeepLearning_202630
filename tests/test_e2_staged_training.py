"""Numerical stage semantics on artificial tensors, no data loading."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
from torch import nn
from torch.nn import functional as F
from e2_nas_common import NASConfig
from e2_staged_training import (
    TrainConfig, TrainingFailure, batch_indices, clone_state, fit_reconstruction,
    fit_sae, fit_supervised, learning_rate_multiplier, optimizer_groups, state_sha256,
)


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(35,8), nn.ReLU(),
                                 nn.Dropout(.2), nn.Linear(8,1))
    def forward(self,x):
        return F.softplus(self.net(x)).squeeze(-1)


class ConstantPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))
    def forward(self,x):
        return F.softplus(self.value).expand(len(x))


class StagedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
    def setUp(self):
        torch.manual_seed(291001)
        self.x = torch.randn(17,7,5)
        self.y = torch.linspace(.3,1.3,17)
        self.vx = torch.randn(8,7,5)
        self.vy = torch.linspace(.4,1.2,8)
        self.config = TrainConfig(.001,.0001,16,3,20,291002)

    def test_shuffle_keeps_every_example_and_redistributes_singleton_tail(self):
        for size in (2,15,16,17,31,32,33,49):
            chunks=batch_indices(size,16,generator=torch.Generator().manual_seed(1))
            indices=torch.cat(chunks)
            self.assertEqual(sorted(indices.tolist()),list(range(size)))
            self.assertEqual(len(set(indices.tolist())),size)
            self.assertTrue(all(2<=len(chunk)<=16 for chunk in chunks))
        self.assertEqual([len(c) for c in batch_indices(17,16,generator=torch.Generator().manual_seed(1))],[15,2])

    def test_fresh_optimizer_determinism_copied_checkpoint_and_original_unit_metrics(self):
        original=TinyPredictor()
        one,two=deepcopy(original),deepcopy(original)
        x_before,y_before=self.x.clone(),self.y.clone()
        a=fit_supervised(one,self.x,self.y,self.vx,self.vy,.02,self.config)
        b=fit_supervised(two,self.x,self.y,self.vx,self.vy,.02,self.config)
        self.assertEqual(a.selected_state_sha256,b.selected_state_sha256)
        self.assertEqual(a.optimizer_initial_state_entries,0)
        self.assertEqual(b.optimizer_initial_state_entries,0)
        self.assertEqual(a.optimizer_steps,6)
        self.assertEqual(a.scheduler_total_steps,6)
        self.assertEqual(state_sha256(one),a.selected_state_sha256)
        self.assertNotEqual(a.initial_state_sha256,a.selected_state_sha256)
        self.assertFalse(one.training)
        torch.testing.assert_close(self.x,x_before)
        torch.testing.assert_close(self.y,y_before)
        self.assertTrue(all(row['examples_seen']==17 for row in a.logs))
        with torch.no_grad():
            metric=float(F.mse_loss(one(self.vx),self.vy).sqrt())*.02
        self.assertAlmostEqual(metric,a.logs[a.best_epoch-1]['validation_score'],places=8)
        selected_hash=state_sha256(a.state_dict)
        with torch.no_grad():next(one.parameters()).add_(10)
        self.assertEqual(state_sha256(a.state_dict),selected_hash)
        json.dumps(a.to_dict(),allow_nan=False)

    def test_earliest_checkpoint_wins_exact_ties_and_patience_is_counted(self):
        model=ConstantPredictor()
        target=F.softplus(torch.tensor(0.0)).expand(17).clone()
        validation=F.softplus(torch.tensor(0.0)).expand(8).clone()
        config=TrainConfig(.001,0,16,10,1,19)
        result=fit_supervised(model,self.x,target,self.vx,validation,1.0,config)
        self.assertEqual((result.best_epoch,result.epochs_completed,result.stop_reason),(1,2,'patience'))
        self.assertEqual(result.logs[0]['validation_score'],0)
        self.assertEqual(result.logs[1]['validation_score'],0)

    def test_fixed_refit_has_no_validation_reads_or_outer_early_stopping(self):
        model=TinyPredictor()
        result=fit_supervised(model,self.x,self.y,None,None,.02,self.config,fixed_epochs=2)
        self.assertEqual((result.best_epoch,result.epochs_completed,result.stop_reason),(2,2,'fixed_epochs'))
        self.assertTrue(all(row['validation_score'] is None for row in result.logs))
        self.assertEqual(result.scheduler_total_steps,6)  # Original three-epoch horizon retained.
        self.assertEqual(result.optimizer_steps,4)
        with self.assertRaises(ValueError):
            fit_supervised(TinyPredictor(),self.x,self.y,object(),object(),.02,self.config,fixed_epochs=2)

    def test_scheduler_warmup_and_minimum_follow_declared_step_horizon(self):
        rates=[learning_rate_multiplier(step,20,.1) for step in range(20)]
        self.assertEqual(rates[:2],[.5,1.])
        self.assertEqual(rates[2],1.)
        self.assertGreater(rates[-1],0.)
        self.assertEqual(learning_rate_multiplier(20,20,.1),0.)
        self.assertTrue(all(a>=b for a,b in zip(rates[2:],rates[3:])))
        self.assertEqual(learning_rate_multiplier(0,1,0),1.)
        self.assertEqual(learning_rate_multiplier(1,1,0),0.)

    def test_single_batch_single_epoch_executes_a_positive_lr_weight_update(self):
        config=replace(self.config,max_epochs=1)
        model=TinyPredictor()
        result=fit_supervised(model,self.x[:16],self.y[:16],self.vx,self.vy,.02,config)
        self.assertEqual(result.optimizer_steps,1)
        self.assertEqual(result.logs[0]['learning_rate_first'],config.learning_rate)
        self.assertEqual(result.logs[0]['learning_rate_last'],config.learning_rate)
        self.assertEqual(result.logs[0]['learning_rate_after_epoch'],0.)
        self.assertNotEqual(result.initial_state_sha256,result.selected_state_sha256)

    def test_decay_exempts_biases_norm_and_vectors_but_not_matrix_weights(self):
        model=nn.Sequential(nn.Linear(3,4),nn.LayerNorm(4),nn.Linear(4,1))
        groups=optimizer_groups(model,.01)
        ids=[{id(p) for p in group['params']} for group in groups]
        self.assertIn(id(model[0].weight),ids[0])
        self.assertIn(id(model[2].weight),ids[0])
        self.assertIn(id(model[0].bias),ids[1])
        self.assertIn(id(model[1].weight),ids[1])
        self.assertIn(id(model[1].bias),ids[1])
        self.assertFalse(ids[0]&ids[1])

    def test_invalid_data_and_configs_fail_without_hidden_retries(self):
        for kwargs in ({'learning_rate':.1},{'weight_decay':1e-8},{'batch_size':1},
                       {'warmup_fraction':.2},{'max_epochs':0},{'seed':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                replace(self.config,**kwargs)
        x=self.x.clone();x[0,0,0]=float('nan')
        with self.assertRaises(ValueError):fit_supervised(TinyPredictor(),x,self.y,self.vx,self.vy,.02,self.config)
        with self.assertRaises(ValueError):fit_supervised(TinyPredictor(),self.x,self.y,self.vx,self.vy,0,self.config)
        model=TinyPredictor()
        with torch.no_grad():next(model.parameters()).fill_(float('nan'))
        with self.assertRaises(TrainingFailure) as raised:
            fit_supervised(model,self.x,self.y,self.vx,self.vy,.02,self.config)
        self.assertFalse(raised.exception.receipt['retry_performed'])
        self.assertEqual(raised.exception.receipt['optimizer_steps'],0)
        self.assertEqual(raised.exception.receipt['error_code'],'nonfinite')
        json.dumps(raised.exception.receipt,allow_nan=False)

    def test_soft_rss_gate_records_terminal_failure_before_first_optimizer_step(self):
        config=replace(self.config,memory_limit_bytes=1)
        with self.assertRaises(TrainingFailure) as raised:
            fit_supervised(TinyPredictor(),self.x,self.y,self.vx,self.vy,.02,config)
        self.assertEqual(raised.exception.receipt['error_type'],'MemoryError')
        self.assertEqual(raised.exception.receipt['error_code'],'resource_limit')
        self.assertEqual(raised.exception.receipt['optimizer_steps'],0)
        self.assertGreater(raised.exception.receipt['max_observed_rss_bytes'],1)

    def test_shape_absent_gradient_and_nonfinite_gradient_have_distinct_error_codes(self):
        class WrongShape(nn.Module):
            def __init__(self):
                super().__init__();self.value=nn.Parameter(torch.tensor(0.))
            def forward(self,x):return self.value.expand(len(x),1)
        class MissingGradient(nn.Module):
            def __init__(self):
                super().__init__();self.value=nn.Parameter(torch.tensor(0.));self.unused=nn.Parameter(torch.tensor(1.))
            def forward(self,x):return F.softplus(self.value).expand(len(x))
        class InfiniteGradient(nn.Module):
            def __init__(self):
                super().__init__();self.value=nn.Parameter(torch.tensor(0.))
            def forward(self,x):return F.softplus(torch.sqrt(self.value)).expand(len(x))
        for model,code in [(WrongShape(),'geometry'),(MissingGradient(),'missing_gradient'),
                           (InfiniteGradient(),'nonfinite')]:
            with self.subTest(code=code),self.assertRaises(TrainingFailure) as raised:
                fit_supervised(model,self.x,self.y,self.vx,self.vy,.02,self.config)
            self.assertEqual(raised.exception.receipt['error_code'],code)
            self.assertEqual(raised.exception.receipt['optimizer_steps'],0)
            self.assertFalse(raised.exception.receipt['retry_performed'])

    def test_sae_executes_three_fresh_stages_and_refits_without_validation(self):
        nas=NASConfig((16,32,16),(1,0,1))
        config=replace(self.config,max_epochs=2)
        selected=fit_sae(nas,self.x,self.y,self.vx,self.vy,.02,config,fold_id='artificial-inner')
        self.assertEqual(set(selected.phases),{'ae1','ae2','supervised'})
        self.assertEqual([r.optimizer_initial_state_entries for r in selected.phases.values()],[0,0,0])
        self.assertTrue(all(r.optimizer_steps==4 for r in selected.phases.values()))
        self.assertAlmostEqual(selected.auxiliary_scale,max(1,float(self.x.abs().max())))
        self.assertFalse(any('decoder' in name for name in selected.model.state_dict()))
        self.assertEqual(selected.phases['ae1'].selection_metric,'reconstruction_mse')
        self.assertEqual(selected.phases['supervised'].selection_metric,'rmse_original')
        chosen={name:result.best_epoch for name,result in selected.phases.items()}
        refit=fit_sae(nas,self.x,self.y,None,None,.02,config,fold_id='artificial-refit',fixed_epochs=chosen)
        self.assertTrue(all(r.fixed_epochs==chosen[name] for name,r in refit.phases.items()))
        self.assertTrue(all(row['validation_score'] is None for r in refit.phases.values() for row in r.logs))
        self.assertTrue(all(p.requires_grad for p in refit.model.parameters()))
        json.dumps(refit.to_dict(),allow_nan=False)
        with self.assertRaises(ValueError):
            fit_sae(nas,self.x,self.y,self.vx,self.vy,.02,config,fold_id='bad-refit',fixed_epochs=chosen)

    def test_sae_scale_never_refits_to_extreme_validation_and_invalid_scale_stops_early(self):
        nas=NASConfig((16,16,16),(0,0,0))
        config=replace(self.config,max_epochs=1)
        result=fit_sae(nas,self.x,self.y,self.vx*100,self.vy,.02,config,fold_id='artificial-scale')
        self.assertAlmostEqual(result.auxiliary_scale,float(self.x.abs().max()))
        with self.assertRaises(ValueError):
            fit_sae(nas,self.x,self.y,self.vx,self.vy,0,config,fold_id='invalid-scale')


if __name__=='__main__':
    unittest.main()
