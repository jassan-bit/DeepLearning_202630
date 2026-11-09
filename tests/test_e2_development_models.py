"""Fixed development-pilot recipes are separate from HPO and fit only sealed2023 data."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import e2_development_models as dev
import e2_training_models as registry
import e2_search_plan as planner
from e2_temporal_training import DatedPanel,prepare_b
from e2_staged_training import TrainingFailure

SOURCE=Path(os.environ.get("E2_ITRANSFORMER_SOURCE_ROOT",str(ROOT/"work/author_sources/iTransformer")))


def fixture(*,origin="2023-10-01"):
    dates=pd.date_range("2023-01-01",periods=450,tz="UTC")
    n=np.arange(len(dates),dtype=float)
    close=100+.05*n+.5*np.sin(n/13)
    open_=close*.999
    features=np.column_stack((open_,close*1.003,open_*.997,close,1000+n))
    target=.01+.000001*n+.00005*np.cos(n/17)
    return prepare_b(DatedPanel(dates,features,target),origin,
                     (pd.Timestamp(origin,tz="UTC")+pd.Timedelta(days=4)).strftime("%Y-%m-%d"))


class DevelopmentPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.prepared=fixture()
        cls.sources={"iTransformer":str(SOURCE)}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_literal_controls_are_fixed_and_do_not_claim_hpo_membership(self):
        controls=dev.fixed_pilot_config(271601)
        self.assertEqual((controls.learning_rate,controls.weight_decay,controls.batch_size),
                         (.001,.0001,32))
        self.assertEqual((controls.max_epochs,controls.patience,controls.warmup_fraction),(200,20,0))
        for name in ("FCN-1D","ResNet-1D","Inception-1D","TSMixer","MLP16","LSTM32","GRU32"):
            model=dev.build_pilot_model(name,None,seed=271601)
            self.assertEqual(sum(v.numel() for v in model.parameters()),registry.FIXED_COUNTS[name])
            meta=model.architecture_metadata
            self.assertTrue(meta["pilot_fixed_not_hpo"])
            self.assertIsNone(meta["registered_trial_id"])
            self.assertIsNone(meta["training_hpo_record"])
            self.assertFalse(meta["training_axes_applied"])
            sites=[m.p for m in model.modules() if isinstance(m,torch.nn.Dropout)]
            self.assertTrue(all(p==(.0 if name.endswith("-1D") else .1) for p in sites))

    def test_registered_hpo_constructor_cannot_be_used_with_a_fabricated_pilot_trial(self):
        row=registry.registered_trials("MLP16")[0]
        row.update(learning_rate=.001,weight_decay=.0001,batch_size=32,dropout=.1)
        with self.assertRaises(ValueError):
            registry.build_train_model("MLP16",row,seed=271601)
        with patch.object(registry,"build_train_model",side_effect=AssertionError("HPO path called")):
            result=dev.build_pilot_model("MLP16",None,seed=271601)
        self.assertTrue(result.architecture_metadata["pilot_fixed_not_hpo"])

    def test_nas_reference_and_noise_patterns_have_exact_registered_identity(self):
        p=registry.protocol();patterns=planner.noise_pilot_architectures(p)
        for name in registry.MODEL_IDS:
            pattern=patterns[-1]
            model=dev.build_pilot_model(name,pattern,seed=271601)
            self.assertEqual(model.config.widths,tuple(pattern["widths"]))
            self.assertEqual(model.config.dropout_pattern,tuple(pattern["dropout_pattern"]))
            self.assertEqual(model.architecture_metadata["pilot_pattern"],pattern)
        reference=dev.build_pilot_model("TimesNet",p["nas"]["reference"],seed=271601)
        self.assertEqual(reference.num_kernels,6)
        self.assertEqual(reference.top_k,4)
        self.assertIsNone(reference.architecture_metadata["registered_trial_id"])
        bad=deepcopy(patterns[0]);bad["pattern_id"]+="tampered"
        with self.assertRaises(ValueError):
            dev.build_pilot_model("TimesNet",bad,seed=271601)
        with self.assertRaises(ValueError):
            dev.build_pilot_model("TimesNet",None,seed=271601)
        with self.assertRaises(ValueError):
            dev.build_pilot_model("MLP16",patterns[0],seed=271601)

    def test_model_initialization_is_fresh_reproducible_and_preserves_caller_rng(self):
        torch.manual_seed(99);before=torch.random.get_rng_state().clone()
        a=dev.build_pilot_model("GRU32",None,seed=271601)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        b=dev.build_pilot_model("GRU32",None,seed=271601)
        c=dev.build_pilot_model("GRU32",None,seed=271602)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertTrue(any(not torch.equal(v,c.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertTrue(all(x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters())))

    def test_fixed_pilot_fit_returns_original_unit_predictions_and_true_stage_costs(self):
        fitted=dev.fit_pilot("MLP16",None,271601,self.prepared,epoch_cap=1,wallcap=30)
        self.assertEqual(len(fitted.predictions_original),len(self.prepared.inner_validation.dates))
        truth=np.array(self.prepared.inner_validation.y_original)
        expected=float(np.sqrt(np.mean((np.array(fitted.predictions_original)-truth)**2)))
        self.assertAlmostEqual(fitted.rmse_original,expected,places=14)
        self.assertGreater(fitted.receipt["optimizer_steps"],0)
        self.assertEqual(set(fitted.phases),{"supervised"})
        self.assertTrue(fitted.receipt["nonprotocol_epoch_override"])
        self.assertFalse(fitted.receipt["outer_labels_used"])
        self.assertFalse(fitted.receipt["market_dataset_files_opened"])
        self.assertTrue(fitted.receipt["development_training_executed"])
        self.assertTrue(fitted.model.architecture_metadata["fixed_pilot_hyperparameters_applied"])
        json.dumps(fitted.to_dict(),allow_nan=False)

    def test_sae_fixed_pilot_runs_three_stages_and_preserves_scale_and_decoder_separation(self):
        fitted=dev.fit_pilot("LSTM-autoencoder con predictor",dev.reference_pattern(),
                             271601,self.prepared,epoch_cap=1,wallcap=30)
        self.assertEqual(set(fitted.phases),{"ae1","ae2","supervised"})
        self.assertEqual(fitted.receipt["optimizer_steps"],
                         sum(x.optimizer_steps for x in fitted.phases.values()))
        expected=max(1,float(self.prepared.inner_train.x.abs().max()))
        self.assertEqual(fitted.receipt["auxiliary_scale"],expected)
        self.assertFalse(fitted.receipt["decoder_parameters_in_predictor"])
        self.assertTrue(all(x.optimizer_initial_state_entries==0 for x in fitted.phases.values()))
        self.assertTrue(fitted.receipt["pilot_fixed_not_hpo"])

    def test_later_origins_mutated_preparation_and_invalid_seeds_fail_before_fit(self):
        later=fixture(origin="2024-01-01")
        with self.assertRaises(ValueError):
            dev.fit_pilot("MLP16",None,271601,later,epoch_cap=1)
        mutated=self.prepared.inner_train.x.clone()
        self.prepared.inner_train.x[0,0,0]+=1
        try:
            with self.assertRaises(ValueError):
                dev.fit_pilot("MLP16",None,271601,self.prepared,epoch_cap=1)
        finally:
            self.prepared.inner_train.x.copy_(mutated)
        for bad in (None,True,-1):
            with self.assertRaises(ValueError):
                dev.build_pilot_model("GRU32",None,seed=bad)

    def test_typed_training_failure_is_preserved_without_retry(self):
        failure=TrainingFailure("deliberate",{"optimizer_steps":2,"error_code":"resource_limit"})
        with patch.object(dev,"fit_supervised",side_effect=failure) as call:
            with self.assertRaises(TrainingFailure) as caught:
                dev.fit_pilot("MLP16",None,271601,self.prepared,epoch_cap=1)
            self.assertIs(caught.exception,failure)
            self.assertEqual(call.call_count,1)

    @unittest.skipUnless(SOURCE.is_dir(),"Explicit verified author-source tree not supplied")
    def test_fixed_itransformer_retains_pinned_body_with_existing_point_one_dropout(self):
        model=dev.build_pilot_model("iTransformer",None,seed=271601,source_roots=self.sources).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),795403)
        self.assertTrue(all(m.p==.1 for m in model.modules() if isinstance(m,torch.nn.Dropout)))
        with torch.no_grad():prediction=model(self.prepared.inner_validation.x[:1])
        self.assertEqual(tuple(prediction.shape),(1,))
        self.assertTrue(bool(torch.isfinite(prediction).all()))


if __name__=="__main__":
    unittest.main()

