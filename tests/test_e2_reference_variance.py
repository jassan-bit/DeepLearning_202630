"""Pure helper tests for the preregistered reference-variance runner; no fitting."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import torch
import run_e2_reference_variance as pilot
from e2_staged_training import state_sha256
from e2_training_bindings import CheckpointStore
from e2_real_executor import DatedJournal


class ReferenceVarianceHelpers(unittest.TestCase):
    def _fake_campaign(self, root, *, failed_selection=None):
        calls=[]
        registered={"resources":{"sampled_worker_group_rss_limit_bytes":12*1024**3},
                    "preselection_variance":{"external_worker_wall_seconds":180}}
        def supervise(command,output_dir,**kwargs):
            spec=json.loads(Path(command[-1]).read_text())
            calls.append(spec)
            output_dir=Path(output_dir);output_dir.mkdir()
            failed=(spec["phase"]=="epoch_selection" and
                    (spec["model"],spec["seed"])==failed_selection)
            worker={"status":"failed" if failed else "completed","model":spec["model"],
                    "seed":spec["seed"],"phase":spec["phase"],
                    "registration_file_sha256":pilot.REGISTRATION_SHA256,
                    "best_epoch":None if failed else (spec["seed"]-pilot.SEEDS[0]+1),
                    "optimizer_steps":0 if failed else 1}
            pilot.json_new(output_dir/"worker_profile.json",worker)
            result={"status":"worker_failure" if failed else "success","worker_profile":worker,
                    "elapsed_wall_seconds":.01,"sampled_peak_process_group_rss_bytes":1024}
            pilot.json_new(output_dir/"result.json",result)
            return result
        with patch.object(pilot,"registration",return_value=registered),\
             patch("profile_e2_training.run_isolated_profile",side_effect=supervise):
            result=pilot.run_reference_variance(Path(root)/"unused-data",
                Path(root)/"unused-registration",Path(root)/"campaign")
        return result,calls

    def test_campaign_orders_selection_before_refit_and_freezes_all_seed_medians(self):
        with tempfile.TemporaryDirectory() as root:
            result,calls=self._fake_campaign(root)
        self.assertEqual(len(calls),30)
        self.assertTrue(all(c["phase"]=="epoch_selection" for c in calls[:15]))
        self.assertTrue(all(c["phase"]=="fresh_refit" for c in calls[15:]))
        self.assertTrue(all(c["fixed_epochs"]==3 for c in calls[15:]))
        self.assertTrue(result["paired_3x5_complete"])
        self.assertFalse(result["final_R_set"])
        self.assertEqual(result["replacement_attempts"],0)

    def test_campaign_missing_seed_blocks_model_refits_without_replacement(self):
        with tempfile.TemporaryDirectory() as root:
            result,calls=self._fake_campaign(root,failed_selection=("TimeMixer",pilot.SEEDS[2]))
        self.assertEqual(len(calls),25)
        self.assertEqual(len(result["models"]["TimeMixer"]["selection"]),5)
        self.assertEqual(result["models"]["TimeMixer"]["refit"],[])
        self.assertIn("TimeMixer",result["blocked_refits"])
        self.assertFalse(result["paired_3x5_complete"])
        self.assertEqual(result["replacement_attempts"],0)

    def test_all_five_seed_epoch_records_are_required_and_median_is_frozen(self):
        records=[{"seed":seed,"status":"completed","best_epoch":epoch}
                 for seed,epoch in zip(pilot.SEEDS,[9,2,7,20,4])]
        self.assertEqual(pilot.median_epoch_count(records),7)
        for changed in [records[:-1],records+[deepcopy(records[0])]]:
            with self.assertRaises(ValueError):
                pilot.median_epoch_count(changed)
        failed=deepcopy(records);failed[2]["status"]="failed"
        with self.assertRaises(ValueError):
            pilot.median_epoch_count(failed)
        repeated=deepcopy(records);repeated[3]["seed"]=repeated[0]["seed"]
        with self.assertRaises(ValueError):
            pilot.median_epoch_count(repeated)

    def test_worker_domain_rejects_old_seeds_and_refit_without_epochs(self):
        spec={"version":pilot.VERSION,"model":"TimeMixer","seed":pilot.SEEDS[0],
              "phase":"epoch_selection","registration_commit":pilot.REGISTRATION_COMMIT,
              "fixed_epochs":None}
        pilot._validate_spec(spec)
        for key,value in [("seed",271701),("model","TimesNet"),("fixed_epochs",3)]:
            wrong=deepcopy(spec);wrong[key]=value
            with self.assertRaises(ValueError):
                pilot._validate_spec(wrong)
        refit={**spec,"phase":"fresh_refit","fixed_epochs":8}
        pilot._validate_spec(refit)
        with self.assertRaises(ValueError):
            pilot._validate_spec({**refit,"fixed_epochs":None})

    def test_exclusive_json_does_not_replace_an_existing_record(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"record.json"
            pilot.json_new(path,{"first":True})
            with self.assertRaises(FileExistsError):
                pilot.json_new(path,{"first":False})
            self.assertEqual(json.loads(path.read_text()),{"first":True})

    def test_safe_checkpoint_load_binds_file_and_tensor_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            store=CheckpointStore(Path(root)/"checkpoints")
            state={"weight":torch.arange(4,dtype=torch.float32).reshape(2,2)}
            record=store.save("unit-seed",{"state_dict":state,"optimizer_state_saved":False})
            path=store.directory/record["file"]
            expected=state_sha256(state)
            loaded=pilot.safe_load_checkpoint(path,record["sha256"],expected)
            torch.testing.assert_close(loaded["state_dict"]["weight"],state["weight"])
            with self.assertRaises(ValueError):
                pilot.safe_load_checkpoint(path,record["sha256"],"0"*64)
            with path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaises(ValueError):
                pilot.safe_load_checkpoint(path,record["sha256"],expected)

    def _journal(self,root,*,observation_first=False):
        journal=DatedJournal(Path(root)/"replay")
        date="2024-11-08T00:00:00+00:00"
        if observation_first:
            journal.append("observation",date=date,value=.1,prediction_event_sha256="0"*64)
        token=journal.append("predictions",date=date,eligible=True,
                             systems={"model/seed1":{"value":.09}})
        journal.append("observation",date=date,value=.1,prediction_event_sha256=token)
        journal.append("block_completed",result={"status":"complete"})
        journal.close()
        return Path(root)/"replay/events.jsonl",date

    def test_paired_vectors_require_durable_order_and_complete_dates(self):
        with tempfile.TemporaryDirectory() as root:
            path,date=self._journal(root)
            values=pilot.replay_vectors(path,"model/seed1",[date])
            self.assertEqual(values["predictions"],[.09])
            self.assertEqual(values["truths"],[.1])
            with self.assertRaises(ValueError):
                pilot.replay_vectors(path,"model/seed1",[date,"2024-11-09T00:00:00+00:00"])
        with tempfile.TemporaryDirectory() as root:
            path,date=self._journal(root,observation_first=True)
            with self.assertRaises(ValueError):
                pilot.replay_vectors(path,"model/seed1",[date])

    def test_paired_vector_extraction_rejects_hash_chain_changes(self):
        with tempfile.TemporaryDirectory() as root:
            path,date=self._journal(root)
            events=path.read_text().splitlines()
            first=json.loads(events[0]);first["systems"]["model/seed1"]["value"]=999
            events[0]=json.dumps(first)
            path.write_text("\n".join(events)+"\n")
            with self.assertRaises(ValueError):
                pilot.replay_vectors(path,"model/seed1",[date])


if __name__=="__main__":
    unittest.main()

