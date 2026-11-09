"""Real tiny optimization bindings, atomic checkpoints and registered-domain gates."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_e2_training_synthetic import artificial_panel
from e2_temporal_training import prepare_b,select_b,refit_b
from e2_training_bindings import (CheckpointStore,NeuralTrainingAdapter,candidate_for,
                                  run_neural_selection,run_selected_refits)
from e2_staged_training import TrainingFailure,state_sha256
import e2_training_models as registry


class BindingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)
        panel=artificial_panel()
        cls.prepared=prepare_b(panel,panel.calendar[220],panel.calendar[222])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_checkpoint_is_exclusive_and_readable_without_custom_pickle_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=CheckpointStore(Path(temporary)/"checkpoints")
            payload={"state_dict":{"weight":torch.ones(2,2)},"scaler":{"target_scale":.02}}
            receipt=store.save("run",payload)
            loaded=torch.load(store.directory/receipt["file"],weights_only=True)
            torch.testing.assert_close(loaded["state_dict"]["weight"],payload["state_dict"]["weight"])
            with self.assertRaises(FileExistsError):store.save("run",payload)

    def test_real_inner_selection_and_refit_use_distinct_states_and_fixed_epochs(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=CheckpointStore(Path(temporary)/"checkpoints")
            adapter=NeuralTrainingAdapter("MLP16",store,full_epochs=1,short_epochs=1)
            candidate=candidate_for("MLP16",registry.registered_trials("MLP16")[0],None,"hpo")
            result=select_b(self.prepared,[candidate],[271201,271202],adapter.fit_candidate)
            runs=refit_b(self.prepared,result,[271801],adapter.refit_candidate)
            self.assertEqual(result.fixed_epochs,{"supervised":1})
            self.assertTrue(runs[0].fresh_initialization)
            self.assertEqual(runs[0].phase_epochs,result.fixed_epochs)
            records=[r for r in adapter.run_records if r["status"]=="completed"]
            self.assertEqual(len(records),3)
            hashes=[r["resource_use"]["phase_records"]["supervised"]["initial_state_sha256"] for r in records]
            self.assertEqual(len(set(hashes)),3)
            self.assertTrue(all(r["resource_use"]["optimizer_steps"]>0 for r in records))
            for record in records:
                loaded=torch.load(store.directory/record["resource_use"]["checkpoint"]["file"],weights_only=True)
                self.assertEqual(state_sha256(loaded["state_dict"]),
                    loaded["phase_records"]["supervised"]["selected_state_sha256"])
            system=runs[0].artifact.frozen_system()
            before=system.fingerprint()
            model=runs[0].artifact.model
            model.train()
            self.assertNotEqual(before,system.fingerprint())
            model.eval()
            with torch.no_grad():next(model.parameters()).add_(.1)
            self.assertNotEqual(before,system.fingerprint())

    def test_registered_search_rejects_fixture_reduction_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=CheckpointStore(Path(temporary)/"checkpoints")
            adapter=NeuralTrainingAdapter("MLP16",store,full_epochs=2,short_epochs=1)
            with self.assertRaises(ValueError):
                run_neural_selection(self.prepared,"MLP16",adapter,
                    hpo_trials=registry.registered_trials("MLP16")[:2],architectures=[],
                    hpo_seeds=[271201,271202],phase1_seeds=[271301],
                    phase2_seeds=[271401,271402,271403,271404,271405],k=2,delta=.0002)
            self.assertEqual(list(store.directory.iterdir()),[])

    def test_registered_margin_and_runtime_caps_are_checked_before_any_fit(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=CheckpointStore(Path(temporary)/"checkpoints")
            adapter=NeuralTrainingAdapter("MLP16",store)
            kwargs=dict(hpo_trials=registry.registered_trials("MLP16"),architectures=[],
                        hpo_seeds=[271201,271202],phase1_seeds=[271301],
                        phase2_seeds=[271401,271402,271403,271404,271405],k=20)
            with self.assertRaises(ValueError):
                run_neural_selection(self.prepared,"MLP16",adapter,delta=.0001,**kwargs)
            adapter.short_wall=7200
            with self.assertRaises(ValueError):
                run_neural_selection(self.prepared,"MLP16",adapter,
                    delta=registry.protocol()["contract"]["delta"],**kwargs)
            self.assertEqual(list(store.directory.iterdir()),[])

    def test_failed_checkpoint_preserves_completed_training_cost_and_blocks_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            store=CheckpointStore(Path(temporary)/"checkpoints")
            adapter=NeuralTrainingAdapter("MLP16",store,full_epochs=1,short_epochs=1)
            candidate=candidate_for("MLP16",registry.registered_trials("MLP16")[0],None,"hpo")
            with patch.object(store,"save",side_effect=OSError("injected disk failure")):
                run=adapter.fit_candidate(candidate,271201,self.prepared.inner_train.clone(),
                                          self.prepared.inner_validation.clone())
            self.assertEqual(run.status,"failed")
            self.assertTrue(run.fatal)
            self.assertGreater(run.resource_use["optimizer_steps"],0)
            self.assertEqual(run.resource_use["error_code"],"checkpoint_persistence")

    def test_sae_failed_receipt_aggregates_completed_and_current_stage_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter=NeuralTrainingAdapter("MLP16",CheckpointStore(Path(temporary)/"checkpoints"))
            error=TrainingFailure("failed",{"optimizer_steps":3,"completed_phases":{
                "ae1":{"optimizer_steps":5},"ae2":{"optimizer_steps":7}}})
            result=adapter._failed_resources(error)
            self.assertEqual(result["optimizer_steps"],15)
            self.assertEqual(result["optimization_stages"],3)

    def test_refit_failures_are_logged_and_ids_include_the_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter=NeuralTrainingAdapter("MLP16",CheckpointStore(Path(temporary)/"checkpoints"),full_epochs=1)
            candidate=candidate_for("MLP16",registry.registered_trials("MLP16")[0],None,"hpo")
            panel=artificial_panel()
            second=prepare_b(panel,panel.calendar[221],panel.calendar[223])
            with patch("e2_training_bindings.fit_supervised",side_effect=TrainingFailure(
                    "injected",{"optimizer_steps":3,"error_code":"nonfinite"})):
                a=adapter.refit_candidate(candidate,271801,self.prepared.refit_train.clone(),{"supervised":1})
                b=adapter.refit_candidate(candidate,271801,second.refit_train.clone(),{"supervised":1})
            self.assertNotEqual(a.run_id,b.run_id)
            self.assertEqual(len(adapter.run_records),2)
            self.assertTrue(all(r["resource_use"]["optimizer_steps"]==3 for r in adapter.run_records))

    def test_registered_refit_requires_power_count_and_separate_seed_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter=NeuralTrainingAdapter("MLP16",CheckpointStore(Path(temporary)/"checkpoints"))
            with self.assertRaises(ValueError):
                run_selected_refits(self.prepared,None,adapter,[271201],scope="synthetic_fixture")
            with self.assertRaises(ValueError):
                run_selected_refits(self.prepared,None,adapter,[271801])


if __name__=="__main__":unittest.main()

