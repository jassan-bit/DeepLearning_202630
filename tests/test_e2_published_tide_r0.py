"""Synthetic checks for the prospective, source-defined r0 TiDE adapter."""
from copy import deepcopy
from pathlib import Path
import os
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import torch
import e2_published_tide as tide
import e2_training_models as historical

ROOT = Path(os.environ['E2_TIDE_SOURCE_ROOT']) if os.environ.get('E2_TIDE_SOURCE_ROOT') else Path('/nonexistent/explicit-TiDE-source-required')
if os.environ.get('E2_PUBLISHED_SOURCE_ROOTS'):
    import json
    roots = json.loads(Path(os.environ['E2_PUBLISHED_SOURCE_ROOTS']).read_text())
    if 'TiDE' in roots:ROOT = Path(roots['TiDE'])


class TiDESourceContractTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.warn = warnings.catch_warnings()
        self.warn.__enter__()
        warnings.filterwarnings('ignore', message='Initializing zero-element tensors is a no-op')

    def tearDown(self):
        self.warn.__exit__(None, None, None)
        torch.set_num_threads(self.previous_threads)

    def test_new_identity_preserves_optimizer_points_and_rejects_nas_knobs(self):
        new = tide.registered_trials()
        old = historical.registered_trials('TiDE')
        self.assertEqual(len(new), 18)
        for a, b in zip(old, new):
            for k in ('learning_rate', 'weight_decay', 'batch_size', 'optimization_seeds'):
                self.assertEqual(a[k], b[k])
            self.assertEqual(b['recipe_id'], tide.RECIPE_ID)
            self.assertNotEqual(a['trial_id'], b['trial_id'])
            self.assertEqual(b['dropout'], 0)
            self.assertNotIn('architecture_reference', b)
            self.assertNotIn('hidden_dims', b)

    def test_empty_source_features_are_retained_with_disclosed_inactivity(self):
        m = tide.SourceTiDEDefaultR0()
        x = torch.linspace(-1, 1, 70).reshape(10,7).requires_grad_()
        y = m(x, torch.arange(5).repeat(2))
        self.assertEqual(tuple(y.shape), (10,1))
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in m.parameters()))
        self.assertEqual(float(m.time_encoder.lin_a.bias.grad.abs().sum()), 0)
        self.assertEqual(float(m.time_encoder.lin_b.weight.grad.abs().sum()), 0)
        self.assertGreater(float(m.linear.weight.grad.abs().sum()), 0)
        self.assertEqual(tuple(m.time_encoder.lin_a.weight.shape), (64,0))
        self.assertEqual(sum(p.numel() for p in m.parameters()), 163378)

    def test_source_cannot_be_skipped_or_replaced_with_unverified_tree(self):
        with self.assertRaises(ValueError):
            tide.verify_tide_sources({})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                tide.build_tide(seed=1, source_roots={'TiDE':directory})

    @unittest.skipUnless(ROOT.is_dir(), 'Explicit pinned TiDE source root is required')
    def test_full_d7_interface_and_rng_preservation(self):
        torch.manual_seed(53)
        before = torch.random.get_rng_state().clone()
        m = tide.build_tide(seed=271201, source_roots={'TiDE':ROOT}, trial=tide.registered_trials()[5])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        y = m(torch.linspace(-1,1,140).reshape(4,7,5))
        self.assertEqual(tuple(y.shape),(4,))
        self.assertTrue(bool((y > 0).all()))
        self.assertEqual(m.architecture_metadata['training_recipe_id'], tide.RECIPE_ID)
        self.assertFalse(m.architecture_metadata['all_parameters_expected_to_have_nonzero_gradients'])
        for shape in [(4,8,5),(4,7,6),(4,35),(0,7,5)]:
            with self.assertRaises(ValueError):m(torch.empty(shape))
        with self.assertRaises(ValueError):m(torch.full((4,7,5),float('nan')))

    @unittest.skipUnless(ROOT.is_dir(), 'Explicit pinned TiDE source root is required')
    def test_historical_nas_trial_is_not_accepted_as_source_trial(self):
        with self.assertRaises(ValueError):
            tide.build_tide(seed=271201, source_roots={'TiDE':ROOT}, trial=historical.registered_trials('TiDE')[5])
        changed = deepcopy(tide.registered_trials()[5]);changed['dropout']=.3
        with self.assertRaises(ValueError):
            tide.build_tide(seed=271201, source_roots={'TiDE':ROOT}, trial=changed)


class TiDEReducedSelectionTests(unittest.TestCase):
    def fixture(self):
        import hashlib,json
        from e2_reduced_protocol import FOLDS
        domain=json.loads((Path(__file__).resolve().parents[1]/'experiments/e2_tide_published_r0_20261112/configurations.json').read_text())['TiDE']
        rows=[]
        for index,configured in enumerate(domain):
            for j,fold in enumerate(FOLDS):
                value=([0.,0.,3.][j] if index==0 else 1.5 if index==1 else 5.+index)
                rows.append(dict(model_id='TiDE',configuration_id=configured['configuration_id'],
                    fold_origin=fold,seed=271201,parent_trial_id=configured['parent_trial_id'],
                    configuration_sha256=configured['configuration_sha256'],
                    run_receipt_sha256=hashlib.sha256((str(index)+fold).encode()).hexdigest(),
                    status='completed',validation_count=83,rmse_original=value,best_epoch=[2,5,9][j]))
        return domain,rows

    def test_pooled_dates_criteria_and_median_epochs_for_explicit_new_tide_recipe(self):
        domain,rows=self.fixture()
        result=tide.select_reduced_tide(rows,domain)
        self.assertEqual(result['winner_configuration_id'],'C02')
        self.assertEqual(result['validation_count'],249)
        self.assertEqual(result['fixed_epochs'],5)
        self.assertEqual(result['pooled_cv_RMSE'],1.5)
        self.assertEqual(result['model_id'],'TiDE')

    def test_missing_attempt_wrong_mask_and_unknown_receipt_are_rejected(self):
        domain,rows=self.fixture()
        with self.assertRaises(ValueError):tide.select_reduced_tide(rows[:-1],domain)
        changed=deepcopy(rows);changed[0]['validation_count']=82
        with self.assertRaises(ValueError):tide.select_reduced_tide(changed,domain)
        changed=deepcopy(rows);changed[0]['run_receipt_sha256']='unknown'
        with self.assertRaises(ValueError):tide.select_reduced_tide(changed,domain)
        changed=deepcopy(domain);changed[0]['trial']['dropout']=.3
        with self.assertRaises(ValueError):tide.select_reduced_tide(rows,changed)

    def test_failure_consumes_slot_without_replacement_and_fatal_blocks_selection(self):
        domain,rows=self.fixture()
        rows[3].update(status='failed',reason='resource cap')
        result=tide.select_reduced_tide(rows,domain)
        self.assertEqual(result['winner_configuration_id'],'C01')
        self.assertFalse(result['candidate_ledger'][1]['replacement_performed'])
        rows[3]['fatal']=True
        with self.assertRaises(ValueError):tide.select_reduced_tide(rows,domain)


if __name__ == '__main__':unittest.main()
