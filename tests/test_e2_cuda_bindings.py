"""Scientific CUDA adapter contracts using explicit CPU fixtures and real CUDA skips."""
from pathlib import Path
from copy import deepcopy
import json
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import e2_cuda_bindings as bindings
import e2_training_models as registry
import e2_search_plan as planner
from e2_temporal_training import prepare_b,select_b,refit_b
from e2_staged_training import TrainingFailure,state_sha256
from run_e2_training_synthetic import artificial_panel


class CudaBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        cls.panel=artificial_panel()
        cls.prepared=prepare_b(cls.panel,cls.panel.calendar[220],cls.panel.calendar[222])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def adapter(self,name="MLP16",store=None,**kwargs):
        return bindings.CudaNeuralTrainingAdapter(name,store,device="cpu",allow_cpu_for_tests=True,
                                                 full_epochs=1,short_epochs=1,**kwargs)

    def candidate(self,name="MLP16",stage="hpo",index=0,architecture=None):
        if architecture is None and name in bindings.registry.MODEL_IDS:
            architecture=registry.reference_architecture(name)
        return bindings.candidate_for(name,registry.registered_trials(name)[index],architecture,stage)

    def fit(self,adapter,candidate=None,seed=271201):
        return adapter.fit_candidate(candidate or self.candidate(),seed,
            self.prepared.inner_train.clone(),self.prepared.inner_validation.clone())

    def test_cpu_requires_explicit_test_opt_in_and_cuda_never_falls_back(self):
        with self.assertRaises(ValueError):
            bindings.CudaNeuralTrainingAdapter("MLP16",device="cpu")
        with patch("e2_cuda_training.torch.cuda.is_available",return_value=False):
            with self.assertRaises(bindings.cuda.CudaUnavailableError):
                bindings.CudaNeuralTrainingAdapter("MLP16")
        self.assertEqual(self.adapter().device,"cpu")

    def test_registered_epoch_caps_and_hpo_axes_are_retained(self):
        adapter=bindings.CudaNeuralTrainingAdapter("MLP16",device="cpu",allow_cpu_for_tests=True)
        trial=registry.registered_trials("MLP16")[1]
        full=adapter._config(trial,271201,"hpo")
        short=adapter._config(trial,271301,"nas1")
        self.assertEqual((full.max_epochs,short.max_epochs,full.patience),(200,50,20))
        self.assertEqual((full.wall_limit_seconds,short.wall_limit_seconds),(7200,1800))
        for key in ("learning_rate","weight_decay","batch_size","warmup_fraction"):
            self.assertEqual(getattr(full,key),trial.get(key,0))

    def test_fit_identity_includes_candidate_payload_and_validation_split(self):
        adapter=self.adapter();candidate=self.candidate()
        original=adapter._run_id(candidate,271201,self.prepared.inner_train,"hpo",self.prepared.inner_validation)
        second=adapter._run_id(candidate,271201,self.prepared.inner_train,"hpo",SimpleNamespace(signature="another-validation"))
        altered=deepcopy(candidate.payload);altered["identity_test_marker"]=1
        from e2_temporal_training import CandidateSpec
        third=adapter._run_id(CandidateSpec(candidate.candidate_id,altered),271201,self.prepared.inner_train,"hpo",self.prepared.inner_validation)
        self.assertEqual(len({original,second,third}),3)

    def test_nondeterministic_production_cuda_is_rejected_before_any_fit(self):
        with patch.object(bindings.cuda,"_resolve_device",return_value=torch.device("cuda:0")):
            with self.assertRaisesRegex(ValueError,"registered deterministic"):
                bindings.CudaNeuralTrainingAdapter("MLP16",deterministic=False)

    def test_selection_scores_only_and_fresh_refit_has_cpu_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            store=bindings.CudaCheckpointStore(Path(directory)/"weights",max_bytes=1024*1024)
            adapter=self.adapter(store=store)
            candidates=[self.candidate(index=i) for i in (0,1)]
            chosen=select_b(self.prepared,candidates,[271201,271202],adapter.fit_candidate)
            self.assertEqual(list(store.directory.iterdir()),[])
            self.assertEqual(chosen.attempts,4)
            for row in adapter.run_records:
                resource=row["resource_use"]
                self.assertEqual(resource["checkpoint"]["retention"],"metadata_only")
                self.assertFalse(resource["checkpoint"]["weights_saved"])
                self.assertGreater(resource["optimizer_steps"],0)
                self.assertTrue(resource["test_only_cpu_execution"])
            runs=refit_b(self.prepared,chosen,[271801],adapter.refit_candidate)
            self.assertEqual(runs[0].status,"completed")
            self.assertTrue(runs[0].fresh_initialization)
            artifact=runs[0].artifact
            loaded=store.load(artifact.checkpoint)
            self.assertTrue(all(t.device.type=="cpu" for t in loaded["state_dict"].values()))
            self.assertEqual(loaded["phase_epochs"],chosen.fixed_epochs)
            self.assertEqual(state_sha256(loaded["state_dict"]),artifact.phase_records["supervised"]["selected_state_sha256"])
            self.assertFalse(artifact.checkpoint["optimizer_state_saved"])
            self.assertTrue(artifact.checkpoint["weights_saved"])
            initials=[r["resource_use"]["phase_records"]["supervised"]["initial_state_sha256"]
                      for r in adapter.run_records if r["status"]=="completed"]
            self.assertNotEqual(initials[-1],initials[0])

    def test_frozen_predictor_routes_declared_device_and_preserves_inference_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store=bindings.CudaCheckpointStore(Path(directory)/"weights",max_bytes=1024*1024)
            adapter=self.adapter(store=store)
            run=adapter.refit_candidate(self.candidate(),271801,self.prepared.refit_train.clone(),{"supervised":1})
            self.assertEqual(run.status,"completed")
            frozen=run.artifact.frozen_system();before=frozen.fingerprint()
            rng=torch.random.get_rng_state().clone()
            with patch.object(bindings.cuda,"predict_scaled",wraps=bindings.cuda.predict_scaled) as inference:
                value=frozen.predict(self.panel.calendar[220],self.panel.features[213:220])
                self.assertTrue(math.isfinite(value))
                self.assertEqual(inference.call_args.kwargs["device"],"cpu")
                self.assertTrue(inference.call_args.kwargs["allow_cpu_for_tests"])
            self.assertEqual(before,frozen.fingerprint())
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            run.artifact.model.train()
            self.assertNotEqual(before,frozen.fingerprint())

    def test_complete_tiny_hpo_nas1_nas2_uses_common_selection_contract(self):
        name="TiDE"
        reference=registry.reference_architecture(name)
        smaller=next(a for a in planner.enumerate_architectures(registry.protocol(),name)
                     if a["widths"]==[16,16,16] and a["dropout_pattern"]==[0,0,0])
        adapter=self.adapter(name)
        plan=registry.protocol()
        selection,stages=bindings.run_neural_selection(self.prepared,name,adapter,
            hpo_trials=registry.registered_trials(name)[:1],architectures=[reference,smaller],
            hpo_seeds=plan["common_hpo"]["optimization_seeds"],
            phase1_seeds=[plan["nas"]["phase1_seed"]],phase2_seeds=plan["nas"]["phase2_seeds"],
            k=2,delta=plan["contract"]["delta"],scope="synthetic_fixture")
        self.assertEqual(set(stages),{"hpo","nas_phase1","nas_phase2"})
        self.assertEqual([stages[k].attempts for k in ("hpo","nas_phase1","nas_phase2")],[2,2,10])
        self.assertEqual(selection.selection_rule,"nas_parsimony")
        self.assertEqual(len(adapter.run_records),14)
        self.assertTrue(all(row["status"]=="completed" for row in adapter.run_records))
        self.assertTrue(all(row["resource_use"]["checkpoint"]["retention"]=="metadata_only" for row in adapter.run_records))

    def test_sae_selection_and_fixed_refit_preserve_all_three_stages(self):
        name="LSTM-autoencoder con predictor"
        architecture=next(a for a in planner.enumerate_architectures(registry.protocol(),name)
                          if a["widths"]==[16,16,16] and a["dropout_pattern"]==[0,0,0])
        candidate=self.candidate(name,architecture=architecture)
        with tempfile.TemporaryDirectory() as directory:
            store=bindings.CudaCheckpointStore(Path(directory)/"weights",max_bytes=1024*1024)
            adapter=self.adapter(name,store=store)
            selected=select_b(self.prepared,[candidate],[271201],adapter.fit_candidate)
            self.assertEqual(set(selected.fixed_epochs),{"ae1","ae2","supervised"})
            runs=refit_b(self.prepared,selected,[271801],adapter.refit_candidate)
            self.assertEqual(runs[0].status,"completed")
            resource=runs[0].resource_use
            self.assertEqual(resource["optimization_stages"],3)
            self.assertEqual(resource["optimizer_steps"],sum(r["optimizer_steps"] for r in resource["phase_records"].values()))
            self.assertTrue(all(r["optimizer_initial_state_entries"]==0 for r in resource["phase_records"].values()))
            self.assertEqual(runs[0].phase_epochs,selected.fixed_epochs)

    def test_opt_in_selection_weight_failure_preserves_fit_cost_and_is_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            store=bindings.CudaCheckpointStore(Path(directory)/"weights",max_bytes=1024*1024)
            adapter=self.adapter(store=store,retain_selection_weights=True)
            with patch.object(store,"save",side_effect=OSError("synthetic storage failure")):
                run=self.fit(adapter)
            self.assertEqual(run.status,"failed");self.assertTrue(run.fatal)
            self.assertEqual(run.resource_use["error_code"],"checkpoint_persistence")
            self.assertGreater(run.resource_use["optimizer_steps"],0)
            self.assertEqual(run.resource_use["optimizer_steps"],run.resource_use["optimizer_steps_known_lower_bound"])
            self.assertEqual([r["status"] for r in adapter.run_records],["failed"])

    def test_validation_inference_failure_preserves_completed_fit_cost(self):
        adapter=self.adapter()
        with patch.object(bindings.cuda,"predict_scaled",side_effect=RuntimeError("synthetic inference failure")):
            run=self.fit(adapter)
        self.assertEqual(run.status,"failed");self.assertTrue(run.fatal)
        self.assertEqual(run.resource_use["error_code"],"validation_inference")
        self.assertGreater(run.resource_use["optimizer_steps"],0)
        self.assertFalse(run.resource_use["partial_optimizer_update_may_have_occurred"])
        self.assertEqual([r["status"] for r in adapter.run_records],["failed"])

    def test_validation_resource_and_numerical_failures_charge_candidate_without_retry(self):
        failures=(torch.OutOfMemoryError("synthetic OOM"),TimeoutError("synthetic timeout"),
                  bindings.cuda.cpu.TrainingNumericalError("synthetic nonfinite"))
        for error in failures:
            with self.subTest(error=type(error).__name__):
                adapter=self.adapter()
                with patch.object(bindings.cuda,"predict_scaled",side_effect=error) as infer:
                    run=self.fit(adapter)
                self.assertEqual(run.status,"failed");self.assertFalse(run.fatal)
                self.assertEqual(infer.call_count,1)
                self.assertIn(run.resource_use["error_code"],("resource_limit","nonfinite"))
                self.assertGreater(run.resource_use["optimizer_steps"],0)
                self.assertFalse(run.resource_use["retry_performed"])

    def test_validation_state_mutation_is_fatal_with_completed_cost_preserved(self):
        adapter=self.adapter()
        original=bindings.cuda.predict_scaled
        def mutate(model,*args,**kwargs):
            result=original(model,*args,**kwargs)
            with torch.no_grad():next(model.parameters()).add_(.25)
            return result
        with patch.object(bindings.cuda,"predict_scaled",side_effect=mutate):
            run=self.fit(adapter)
        self.assertEqual(run.status,"failed");self.assertTrue(run.fatal)
        self.assertEqual(run.resource_use["error_code"],"checkpoint")
        self.assertGreater(run.resource_use["optimizer_steps"],0)
        self.assertEqual([r["status"] for r in adapter.run_records],["failed"])

    def test_sae_partial_failure_retains_prior_stages_and_unknown_update(self):
        adapter=self.adapter()
        failure=TrainingFailure("partial",{"optimizer_steps":3,"completed_optimizer_steps":3,
            "partial_optimizer_update_may_have_occurred":True,"error_code":"resource_limit",
            "completed_phases":{"ae1":{"optimizer_steps":5}}})
        normalized=adapter._failed_resources(failure)
        self.assertIsNone(normalized["optimizer_steps"])
        self.assertEqual(normalized["optimizer_steps_known_lower_bound"],8)
        self.assertEqual(normalized["optimization_stages"],2)
        self.assertEqual(normalized["failed_phase_optimizer_steps"],3)
        self.assertFalse(normalized["retry_performed"])
        lower_only=adapter._failed_resources(TrainingFailure("unknown",{
            "optimizer_steps":None,"optimizer_steps_known_lower_bound":3,
            "completed_phases":{"ae1":{"optimizer_steps":5}},"error_code":"resource_limit"}))
        self.assertIsNone(lower_only["optimizer_steps"])
        self.assertEqual(lower_only["optimizer_steps_known_lower_bound"],8)

    def test_unreceipted_midfit_failure_is_unknown_but_construction_cost_is_zero(self):
        adapter=self.adapter()
        with patch.object(bindings.cuda,"fit_supervised",side_effect=RuntimeError("lost receipt")):
            run=self.fit(adapter)
        self.assertIsNone(run.resource_use["optimizer_steps"])
        self.assertEqual(run.resource_use["optimizer_steps_known_lower_bound"],0)
        self.assertTrue(run.resource_use["partial_optimizer_update_may_have_occurred"])
        second=self.adapter()
        with patch.object(bindings.registry,"build_train_model",side_effect=ValueError("construction failed")):
            before=self.fit(second)
        self.assertEqual(before.resource_use["optimizer_steps"],0)
        self.assertFalse(before.resource_use["partial_optimizer_update_may_have_occurred"])

    def test_duplicate_attempt_does_not_start_a_second_optimizer(self):
        adapter=self.adapter()
        self.assertEqual(self.fit(adapter).status,"completed")
        with patch.object(bindings.cuda,"fit_supervised") as fit:
            again=self.fit(adapter)
        fit.assert_not_called()
        self.assertEqual(again.status,"failed");self.assertTrue(again.fatal)
        self.assertIn("cannot be repeated",again.reason)

    def test_refit_requires_a_store_before_training(self):
        adapter=self.adapter()
        with patch.object(bindings.cuda,"fit_supervised") as fit:
            run=adapter.refit_candidate(self.candidate(),271801,self.prepared.refit_train.clone(),{"supervised":1})
        fit.assert_not_called()
        self.assertEqual(run.status,"failed")
        self.assertEqual(run.resource_use["optimizer_steps"],0)

    def test_checkpoint_persistence_requires_a_declared_byte_budget(self):
        class Unbounded:
            def save(self,*args):
                raise AssertionError("Unbounded store must not be used")
        with self.assertRaisesRegex(ValueError,"positive storage budget"):
            self.adapter(store=Unbounded())

    def test_checkpoint_budget_resume_corruption_and_exclusive_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"weights"
            store=bindings.CudaCheckpointStore(path,max_bytes=32*1024)
            payload={"state_dict":{"weight":torch.ones(2,2)},"optimizer_state_saved":False}
            receipt=store.save("first",payload)
            before={p.name:p.read_bytes() for p in path.iterdir()}
            with self.assertRaises(FileExistsError):store.save("first",payload)
            with self.assertRaises(Exception):store.save("large",{"state_dict":{"weight":torch.ones(100000)}})
            self.assertEqual({p.name:p.read_bytes() for p in path.iterdir()},before)
            resumed=bindings.CudaCheckpointStore(path,max_bytes=32*1024,resume=True)
            torch.testing.assert_close(resumed.load(receipt)["state_dict"]["weight"],payload["state_dict"]["weight"])
            (path/receipt["file"]).write_bytes(b"corrupt")
            with self.assertRaises(ValueError):bindings.CudaCheckpointStore(path,max_bytes=32*1024,resume=True)

    def test_interrupted_checkpoint_needs_reconciliation_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            path=base/"weights";path.mkdir()
            (path/"unfinished.pt").write_bytes(b"partial")
            with self.assertRaises(ValueError):bindings.CudaCheckpointStore(path,max_bytes=100000,resume=True)
            alias=base/"alias";alias.symlink_to(path,target_is_directory=True)
            with self.assertRaises(ValueError):bindings.CudaCheckpointStore(alias,max_bytes=100000,resume=True)


@unittest.skipUnless(torch.cuda.is_available(),"Real CUDA device unavailable; no CPU substitution")
class RealCudaBindingTests(unittest.TestCase):
    def test_real_cuda_selection_refit_frozen_inference_and_cpu_checkpoint(self):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        panel=artificial_panel();prepared=prepare_b(panel,panel.calendar[220],panel.calendar[222])
        for name in ("MLP16","LSTM-autoencoder con predictor"):
            with self.subTest(model=name),tempfile.TemporaryDirectory() as directory:
                store=bindings.CudaCheckpointStore(Path(directory)/"weights",max_bytes=4*1024**2)
                adapter=bindings.CudaNeuralTrainingAdapter(name,store,full_epochs=1,short_epochs=1)
                architecture=None
                if name in registry.MODEL_IDS:
                    architecture=next(a for a in planner.enumerate_architectures(registry.protocol(),name)
                        if a["widths"]==[16,16,16] and a["dropout_pattern"]==[0,0,0])
                candidate=bindings.candidate_for(name,registry.registered_trials(name)[0],architecture,"hpo")
                selected=select_b(prepared,[candidate],[271201],adapter.fit_candidate)
                refit=refit_b(prepared,selected,[271801],adapter.refit_candidate)[0]
                self.assertEqual(refit.status,"completed")
                self.assertTrue(all(p.device.type=="cuda" for p in refit.artifact.model.parameters()))
                self.assertTrue(all(v.device.type=="cpu" for v in store.load(refit.artifact.checkpoint)["state_dict"].values()))
                frozen=refit.artifact.frozen_system();before=frozen.fingerprint()
                self.assertTrue(math.isfinite(frozen.predict(panel.calendar[220],panel.features[213:220])))
                self.assertEqual(before,frozen.fingerprint())


if __name__=="__main__":unittest.main()
