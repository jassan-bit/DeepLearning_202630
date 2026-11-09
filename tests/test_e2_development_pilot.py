"""Metadata-only guards for the registered development resource supervisor.

No development CSV, source archive, model factory or fit is opened by these
tests. The worker supervisor is mocked for plan-enumeration assertions.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_e2_development_pilot as runner


class DevelopmentPilotGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registration_path=runner._default_registration()
        cls.registration_bytes=cls.registration_path.read_bytes()

    def test_exact_registration_bytes_are_required_before_execution(self):
        self.assertEqual(hashlib.sha256(self.registration_bytes).hexdigest(),
                         runner.REGISTRATION_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"registration.json"
            path.write_bytes(self.registration_bytes)
            admitted=runner.registration(path)
            self.assertEqual(admitted["resources"]["seed"],271601)
            path.write_bytes(self.registration_bytes+b"\n")
            with self.assertRaisesRegex(ValueError,"Registration bytes differ"):
                runner.registration(path)

    def test_default_resource_domain_has_fifteen_distinct_jobs_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data=root/"metadata-only-data-root";data.mkdir()
            authors=root/"metadata-only-author-root";authors.mkdir()
            output=root/"out"
            observed=[]

            def failed_worker_stub(command,worker_dir,**kwargs):
                worker_dir=Path(worker_dir);worker_dir.mkdir()
                spec_path=Path(command[command.index("--worker-spec")+1])
                spec=json.loads(spec_path.read_text())
                observed.append((spec["model"],spec["case"],spec["seed"]))
                self.assertEqual(kwargs["timeout_seconds"],180)
                self.assertEqual(kwargs["memory_limit_bytes"],12*1024**3)
                return {"status":"worker_failure","returncode":1,
                        "worker_profile":None,"elapsed_wall_seconds":0.,
                        "sampled_peak_process_group_rss_bytes":0}

            with patch.object(runner,"run_isolated_profile",side_effect=failed_worker_stub):
                summary=runner.resources(data,authors,output,
                                         registration_path=self.registration_path)
            self.assertEqual(len(observed),15)
            self.assertEqual(len(set(observed)),15)
            self.assertEqual(sum(case=="reference" for _,case,_ in observed),14)
            self.assertEqual([row for row in observed if row[1]=="largest_timesnet"],
                             [("TimesNet","largest_timesnet",271601)])
            self.assertTrue(all(seed==271601 for _,_,seed in observed))
            self.assertTrue(summary["all_requested_jobs_attempted"])
            self.assertTrue(summary["full_registered_job_set"])
            self.assertFalse(summary["all_fits_completed"])
            self.assertEqual(summary["unknown_optimizer_step_workers"],15)
            self.assertTrue(all(row["optimizer_steps"] is None for row in summary["results"]))
            self.assertEqual(summary["known_optimizer_steps_sum"],0)

    def test_duplicate_or_unknown_models_are_rejected_before_any_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for models in ([],["FCN-1D","FCN-1D"],["unregistered-model"]):
                with self.subTest(models=models):
                    output=root/("out-"+str(len(models)))
                    with patch.object(runner,"run_isolated_profile") as launch:
                        with self.assertRaises(ValueError):
                            runner.resources(root,None,output,models=models,
                                             registration_path=self.registration_path)
                        launch.assert_not_called()
                    self.assertFalse(output.exists())

    def test_exclusive_json_and_event_files_cannot_overwrite_prior_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            path=root/"record.json"
            runner.write_json_new(path,{"original":True})
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.write_json_new(path,{"original":False})
            self.assertEqual(path.read_bytes(),original)
            events=root/"events.jsonl"
            writer=runner.DurableEvents(events);writer.append("start");writer.close()
            original=events.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.DurableEvents(events)
            self.assertEqual(events.read_bytes(),original)

    def test_hash_chain_detects_complete_record_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"events.jsonl"
            writer=runner.DurableEvents(path)
            writer.append("first",count=1);writer.append("second",count=2);writer.close()
            clean=runner.read_events(path)
            self.assertEqual(len(clean["events"]),2)
            lines=path.read_text().splitlines()
            changed=json.loads(lines[0]);changed["count"]=999
            lines[0]=json.dumps(changed)
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaisesRegex(ValueError,"event chain changed"):
                runner.read_events(path)

    def test_torn_final_event_is_reported_without_accepting_an_incomplete_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"events.jsonl"
            writer=runner.DurableEvents(path)
            writer.append("start");last=writer.append("completed");writer.close()
            with path.open("ab") as handle:
                handle.write(b'{"sequence":2,"kind":"torn')
            result=runner.read_events(path)
            self.assertTrue(result["incomplete_tail"])
            self.assertEqual(len(result["events"]),2)
            self.assertEqual(result["last_event_sha256"],last["sha256"])
            self.assertEqual(result["sha256"],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_partial_sae_costs_are_added_and_unknown_counts_remain_null(self):
        receipt={"optimizer_steps":3,
                 "completed_phases":{"ae1":{"optimizer_steps":7},
                                     "ae2":{"optimizer_steps":5}}}
        self.assertEqual(runner._partial_steps(receipt),15)
        self.assertEqual(runner._partial_steps({"optimizer_steps":0}),0)
        for receipt in (None,{},{"optimizer_steps":None},{"optimizer_steps":True},
                        {"optimizer_steps":-1},
                        {"optimizer_steps":3,"completed_phases":{"ae1":{}}}):
            with self.subTest(receipt=receipt):
                self.assertIsNone(runner._partial_steps(receipt))


if __name__=="__main__":
    unittest.main()

