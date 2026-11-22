"""Synthetic orchestration fixtures; no market acquisition/test or real GitHub.

Final workflow unit fixtures mock the method/power admission gate, whose actual
failed-report rejection is tested separately. They do not establish admission.
"""
import copy
import datetime as dt
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from engine import common as c, controller, data, models, training

ROOT=Path(__file__).resolve().parents[1]
SOURCE="c"*40


class FakeStore:
    def __init__(self):
        self.remote={"exists":False,"source_commit":SOURCE}
        self.calls=[]
        self.fail_at=None
        self.counter=0
    def status(self):
        self.calls.append("status")
        return dict(self.remote)
    def restore(self,case):
        return None
    def backup(self,case,public_files=None,*,stage="checkpoint"):
        self.calls.append(stage)
        self.counter+=1
        if self.fail_at is not None and self.counter>=self.fail_at:
            raise PermissionError("PRIVATE_CREDENTIAL_MUST_NEVER_APPEAR")
        plan=c.read_json(Path(case)/"plan.json")
        head=f"{self.counter:040x}"
        result={"remote_verified":True,"commit":head,"head_commit":head,"source_commit":SOURCE,
                "case_id":plan["case_id"],"plan_sha256":c.digest(plan)}
        c.write_json(Path(case)/".github_backup_receipt.json",result)
        self.remote={**result,"head":head,"exists":True}
        return result


class FakeWorker:
    def __init__(self):
        self.fits=[]
        self.forwards=[]
        self.crash_forward=False
    def complete(self,job,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        raw=c.canonical({"model":job["model"],"seed":job["seed"],"synthetic":True})
        filename="snapshot_"+"d"*32+".pt"
        c.atomic(directory/filename,raw)
        c.write_json(directory/"checkpoint.json",{"file":filename,"bytes":len(raw),"sha256":c.sha(directory/filename)})
        c.write_json(directory/"request.json",{"job_spec":job})
        result={"status":"completed_validated_fit" if job["model"] in models.NEURAL else "completed_deterministic_system",
                "checkpoint_file":filename,"checkpoint_sha256":c.sha(directory/filename),"selected_state_sha256":c.digest(job),
                "preprocessor":{"candidate":{"preprocessor":{"reference_rms":.01}}},"active_seconds_charged":.05,"compute_uncertain":False}
        c.write_json(directory/"done.json",result)
        return result
    def fit(self,job,fit,selection,directory,*,on_checkpoint,checkpoint_interval_seconds):
        on_checkpoint(directory,{"kind":"before_fit"})
        self.fits.append(job["id"])
        result=self.complete(job,directory)
        on_checkpoint(directory,{"kind":"job_completed"})
        return result
    def verify(self,job,directory):
        result=c.read_json(Path(directory)/"done.json")
        assert c.sha(Path(directory)/result["checkpoint_file"])==result["checkpoint_sha256"]
        assert c.read_json(Path(directory)/"request.json")["job_spec"]==job
        return result
    def predict(self,job,partition,directory):
        self.forwards.append((job["id"],tuple(str(d) for d in partition["dates"])))
        if self.crash_forward:raise RuntimeError("synthetic interrupted final query")
        floor=np.sqrt(partition["known_floor_A"])
        value=floor+.001
        result=self.verify(job,directory)
        return {"raw_sigma7":value,"projected_sigma7":value,"floor":floor,
                "checkpoint_sha256":result["checkpoint_sha256"],"selected_state_sha256":result["selected_state_sha256"]}


def synthetic_parts():
    n=180;t=np.arange(n)
    close=100*np.exp(.001*t+.01*np.sin(t/3))
    bars={"dates":np.asarray([(dt.date(2020,1,1)+dt.timedelta(days=int(i))).isoformat() for i in t]),
          "ohlcv":np.column_stack((close*.999,close*1.01,close*.99,close,1000+t)),"split":np.ones(n)}
    arrays,_=data.build_windows(bars)
    return data.split_by_support(arrays,{"fit":[0,39],"selection":[40,79],"planning":[80,119],"calibration":[120,159],"test":[160,179]},allow_test=True)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        self.source=self.base/"source"
        (self.source/"configs").mkdir(parents=True)
        (self.source/"engine").mkdir()
        for name in ("CANDIDATE_IDENTITY.json","FINAL_SEED_BANK.json"):
            shutil.copy(ROOT/"configs"/name,self.source/"configs"/name)
        for name in ("controller.py","common.py","data.py","models.py","training.py","storage.py","statistics.py"):
            shutil.copy(ROOT/"engine"/name,self.source/"engine"/name)
        (self.source/"vendor").mkdir()
        shutil.copy(ROOT/"vendor/SOURCE_MANIFEST.json",self.source/"vendor/SOURCE_MANIFEST.json")
        manifest={"files":{str(p.relative_to(self.source)):c.sha(p) for p in self.source.rglob("*") if p.is_file()}}
        c.write_json(self.source/"SOURCE_MANIFEST.json",manifest)
        c.write_json(self.source/"PUBLIC_SOURCE_ANCHOR.json",{"source_commit":SOURCE,"manifest_sha256":c.sha(self.source/"SOURCE_MANIFEST.json")})
        self.protocol=c.read_json(ROOT/"configs/protocol.json")
        self.protocol["training"].update(max_epochs=1,patience=1,batch_size=8,total_budget_seconds=60.,epoch_reservation_seconds=20.)
        self.protocolpath=self.base/"protocol.json";c.write_json(self.protocolpath,self.protocol)
        self.parts,self.audit=synthetic_parts()
        self.panels={}
        for panel in controller.PANELS:
            path=self.base/panel
            data.write_panel(path,{k:v for k,v in self.parts.items() if k!="test"},provenance={"synthetic_fixture":True},
                             rights={},exposure={},support_audit=self.audit,fixture=True)
            self.panels[panel]=path
        self.case=self.base/"case"
        self.store=FakeStore();self.worker=FakeWorker()
    def tearDown(self):self.tmp.cleanup()
    def prepare(self):
        return controller.prepare_case(self.source,self.case,self.panels,"planning",SOURCE,self.protocolpath,software_fixture=True)
    def worker_context(self):
        return mock.patch.multiple(training,fit_job=self.worker.fit,verify_job=self.worker.verify,predict_job=self.worker.predict,
                                   export_job_resources=lambda _: {})
    def complete_all(self):
        plan=c.read_json(self.case/"plan.json")
        for job in plan["jobs"]:self.worker.complete(job,self.case/"jobs"/job["id"])
        controller._backup(self.case,plan,self.store,None,"synthetic_completions")
        return plan
    def final_fixture(self):
        self.prepare()
        plan=c.read_json(self.case/"plan.json")
        final_protocol=c.read_json(self.case/"protocol.json")
        final_protocol["robustness_stress_plans"]=controller.default_stress_plans()
        c.write_json(self.case/"protocol.json",final_protocol)
        plan["protocol_sha256"]=c.sha(self.case/"protocol.json")
        bank=c.read_json(self.source/"configs/FINAL_SEED_BANK.json")["seed_ids"]
        plan.update(stage="final",case_id="r3-engine-final-software-unit",software_fixture=False,seeds=bank[:10])
        # These isolated query-lifecycle tests use four synthetic jobs. The
        # actual 21-system Cartesian gate is tested without patching below.
        plan["jobs"]=[next(j for j in plan["jobs"] if j["panel"]==p) for p in controller.PANELS]
        for job in plan["jobs"]:
            job["identity"]["case_id"]=plan["case_id"]
            job["protocol_sha256"]=plan["protocol_sha256"]
        c.write_json(self.case/"plan.json",plan)
        self.complete_all()
        selected={"jobs":[{"id":j["id"],"model":j["model"],"panel":j["panel"],"seed":j["seed"],
                           "checkpoint_file":self.worker.verify(j,self.case/"jobs"/j["id"])["checkpoint_file"],
                           "checkpoint_sha256":self.worker.verify(j,self.case/"jobs"/j["id"])["checkpoint_sha256"],
                           "selected_state_sha256":self.worker.verify(j,self.case/"jobs"/j["id"])["selected_state_sha256"]} for j in plan["jobs"]]}
        c.write_json(self.case/"selected_manifest.json",selected)
        c.write_json(self.case/"metrics/calibrators.json",{"sha256":c.digest({})})
        seal={"plan_sha256":c.digest(plan),"selected_manifest_sha256":c.sha(self.case/"selected_manifest.json"),
              "calibrators_sha256":c.sha(self.case/"metrics/calibrators.json"),"prediction_files":{}}
        c.write_json(self.case/"training_seal.json",seal)
        controller._backup(self.case,plan,self.store,None,"synthetic_seal")
        admissions={p:data.read_admission(path) for p,path in self.panels.items()}
        for p,a in admissions.items():
            a["partition_sha256"]["test"]="b"*64
        return plan,admissions

    def test_prepare_full_family_seed_tail_no_fit_or_test(self):
        with mock.patch.object(data,"load_partition",wraps=data.load_partition) as loader:
            result=self.prepare()
        self.assertEqual(result["jobs"],340)
        self.assertEqual(result["neural_fits"],320)
        self.assertFalse(any(call.args[1]=="test" for call in loader.call_args_list))
        plan=c.read_json(self.case/"plan.json")
        bank=c.read_json(self.source/"configs/FINAL_SEED_BANK.json")["seed_ids"]
        self.assertEqual(plan["seeds"],bank[4091:4096])
        self.assertEqual(sum(j["model"] in models.DETERMINISTIC for j in plan["jobs"]),20)
        controller._validate_jobs(plan)
        plan["jobs"].pop()
        with self.assertRaisesRegex(controller.ControllerError,"cartesian"):
            controller._validate_jobs(plan)

    def test_real_source_anchor_and_hash_mutation_rejected(self):
        self.prepare()
        (self.source/"engine/statistics.py").write_text("changed source")
        with self.assertRaises(ValueError):controller.status_case(self.source,self.case)

    def test_fixture_optin_and_current_final_closure_are_rejected(self):
        with self.assertRaisesRegex(controller.ControllerError,"software_fixture"):
            controller.prepare_case(self.source,self.case,self.panels,"planning",SOURCE,self.protocolpath)
        with self.assertRaisesRegex(controller.ControllerError,"fixtures_never"):
            controller.prepare_case(self.source,self.case,self.panels,"final",SOURCE,self.protocolpath,software_fixture=True)
        with self.assertRaisesRegex(controller.ControllerError,"final_protocol_not_closed"):
            controller._admit_final(ROOT,self.protocol,c.sha(self.protocolpath),None)
        failed_report=self.base/"executed_failed_method_unit_fixture.json"
        c.write_json(failed_report,{"status":"FAILED_FINITE_SAMPLE_VALIDATION","admitted":False})
        with self.assertRaisesRegex(controller.ControllerError,"temporal_methods_not_admitted"):
            controller._method_admission(ROOT,{"path":str(failed_report),"sha256":c.sha(failed_report)},{})

    def test_explicit_activation_and_mandatory_backup_before_fit(self):
        self.prepare()
        with self.assertRaisesRegex(controller.ControllerError,"activation"):
            controller.run_case(self.source,self.case,self.panels,self.store)
        with self.worker_context():
            with self.assertRaisesRegex(controller.ControllerError,"github_store_required"):
                controller.run_case(self.source,self.case,self.panels,None,activate=True,max_jobs=1)
        self.assertEqual(self.worker.fits,[])

    def test_backup_failure_stops_first_fit_and_sanitizes_credentials(self):
        self.prepare();self.store.fail_at=1
        with self.worker_context(),self.assertRaises(training.DurabilityError):
            controller.run_case(self.source,self.case,self.panels,self.store,activate=True,max_jobs=2)
        self.assertEqual(self.worker.fits,[])
        self.assertNotIn("PRIVATE_CREDENTIAL",(self.case/"failures.json").read_text())
        self.assertTrue(c.read_json(self.case/"session.json")["session_open"])

    def test_after_job_failure_stops_next_job(self):
        self.prepare()
        # before run, before job, worker-before, worker-after, controller-after
        self.store.fail_at=5
        with self.worker_context(),self.assertRaises(training.DurabilityError):
            controller.run_case(self.source,self.case,self.panels,self.store,activate=True,max_jobs=2)
        self.assertEqual(len(self.worker.fits),1)

    def test_resume_skips_verified_complete_and_checks_external_data_hash(self):
        self.prepare()
        with self.worker_context():
            first=controller.run_case(self.source,self.case,self.panels,self.store,activate=True,max_jobs=1)
            second=controller.run_case(self.source,self.case,self.panels,self.store,activate=True,max_jobs=1)
        self.assertEqual(first["jobs_executed_this_call"],1)
        self.assertEqual(second["jobs_skipped_verified"],1)
        self.assertEqual(len(self.worker.fits),2)
        (self.panels["F00"]/"fit.npz").write_bytes(b"bad reacquisition")
        with self.worker_context(),self.assertRaises(data.AdmissionError):
            controller.run_case(self.source,self.case,self.panels,self.store,activate=True,max_jobs=1)
        self.assertEqual(len(self.worker.fits),2)

    def test_stale_remote_receipt_blocks_training(self):
        self.prepare();plan=c.read_json(self.case/"plan.json")
        controller._backup(self.case,plan,self.store,None,"fixture")
        self.store.remote["head"]="f"*40
        with self.worker_context(),self.assertRaisesRegex(controller.ControllerError,"stale"):
            controller.run_case(self.source,self.case,self.panels,self.store,activate=True,max_jobs=1)
        self.assertFalse(self.worker.fits)

    def test_planning_export_requires_all_jobs_then_reuses_exact_CSV(self):
        self.prepare();plan=c.read_json(self.case/"plan.json")
        controller._backup(self.case,plan,self.store,None,"fixture")
        with self.worker_context(),self.assertRaises(FileNotFoundError):
            controller.predict_planning(self.source,self.case,self.panels,self.store)
        self.complete_all()
        with self.worker_context(),mock.patch.object(data,"load_partition",wraps=data.load_partition) as load:
            result=controller.predict_planning(self.source,self.case,self.panels,self.store)
            count=len(self.worker.forwards)
            again=controller.predict_planning(self.source,self.case,self.panels,self.store)
        self.assertEqual(count,340)
        self.assertEqual(len(self.worker.forwards),count)
        self.assertTrue(again["reused"])
        self.assertFalse(any(call.args[1]=="test" for call in load.call_args_list))
        receipt=c.read_json(result["receipt"])
        self.assertTrue(receipt["synthetic"])
        self.assertEqual(receipt["predictions_sha256"],c.sha(result["predictions"]))
        self.assertEqual(len(receipt["fit_receipts"]),340)
        self.assertEqual(self.store.remote["case_id"],plan["case_id"])

    def test_seal_uses_real_calibrators_and_dev_predictions_before_test(self):
        plan,admissions=self.final_fixture()
        (self.case/"training_seal.json").unlink()
        (self.case/"selected_manifest.json").unlink()
        (self.case/"metrics/calibrators.json").unlink()
        with self.worker_context(),mock.patch.object(controller,"_load_case",return_value=(self.case,plan,self.panels,admissions)),\
             mock.patch.object(controller,"_admit_final"),mock.patch.object(data,"load_partition",wraps=data.load_partition) as loader:
            result=controller.seal_training(self.source,self.case,self.panels,self.store)
        self.assertEqual(result["status"],"TRAINING_AND_CALIBRATION_SEALED")
        self.assertFalse(any(call.args[1]=="test" for call in loader.call_args_list))
        calibrators=c.read_json(self.case/"metrics/calibrators.json")
        self.assertEqual(len(calibrators["calibrators"]),4)
        self.assertTrue(all(item["status"]=="ok" for item in calibrators["calibrators"]))
        seal=c.read_json(self.case/"training_seal.json")
        self.assertEqual(seal["calibrators_sha256"],c.sha(self.case/"metrics/calibrators.json"))
        self.assertEqual(set(seal["prediction_files"]),{"fit","selection","calibration"})

    def test_corrupt_selection_seal_blocks_any_test_payload_or_forward(self):
        self.final_fixture()
        c.write_json(self.case/"selected_manifest.json",{"jobs":[]})
        with self.worker_context(),mock.patch.object(controller,"_validate_jobs"),mock.patch.object(controller,"_admit_final"),\
             mock.patch.object(data,"load_partition",side_effect=AssertionError("opened test")):
            with self.assertRaisesRegex(controller.ControllerError,"changed_since_seal"):
                controller.evaluate_once(self.source,self.case,self.panels,self.store,activate=True)
        self.assertFalse(self.worker.forwards)

    def test_final_query_cache_one_forward_and_reexport_no_data_load(self):
        plan,admissions=self.final_fixture()
        def loader(path,part,**kwargs):
            self.assertEqual(part,"test");self.assertTrue(kwargs["allow_test"])
            # Durable identity and pending marker are already remote.
            self.assertIn("before_single_final_forward_",self.store.calls[-2])
            return self.parts["test"]
        with self.worker_context(),mock.patch.object(controller,"_validate_jobs"),mock.patch.object(controller,"_admit_final"),\
             mock.patch.object(controller,"_final_panels",return_value=(self.panels,admissions)),mock.patch.object(data,"load_partition",side_effect=loader):
            result=controller.evaluate_once(self.source,self.case,self.panels,self.store,activate=True)
            count=len(self.worker.forwards)
        with self.worker_context(),mock.patch.object(controller,"_validate_jobs"),mock.patch.object(controller,"_admit_final"),\
             mock.patch.object(data,"load_partition",side_effect=AssertionError("reopened test")):
            again=controller.evaluate_once(self.source,self.case,self.panels,self.store,activate=True)
        self.assertEqual(count,12);self.assertEqual(len(self.worker.forwards),12)
        self.assertTrue(again["no_new_forwards"])
        receipt=c.read_json(result["evaluation_receipt"])
        self.assertTrue(receipt["sealed"])
        self.assertTrue(receipt["calibrators_frozen_before_test"])
        self.assertEqual(receipt["status"],"completed")

    def test_uncertain_final_forward_is_never_automatically_requeried(self):
        _,admissions=self.final_fixture();self.worker.crash_forward=True
        with self.worker_context(),mock.patch.object(controller,"_validate_jobs"),mock.patch.object(controller,"_admit_final"),\
             mock.patch.object(controller,"_final_panels",return_value=(self.panels,admissions)),mock.patch.object(data,"load_partition",return_value=self.parts["test"]):
            with self.assertRaises(RuntimeError):controller.evaluate_once(self.source,self.case,self.panels,self.store,activate=True)
            self.worker.crash_forward=False
            with self.assertRaisesRegex(controller.ControllerError,"uncertain_final_forward"):
                controller.evaluate_once(self.source,self.case,self.panels,self.store,activate=True)
        self.assertEqual(len(self.worker.forwards),1)

    def test_causal_stress_transforms_preserve_labels_close_floor_and_pairing(self):
        original=self.parts["test"]
        plans=controller.default_stress_plans()
        for name,spec in plans.items():
            first=controller.stress_partition(original,spec,"F00")
            again=controller.stress_partition(original,spec,"F00")
            self.assertTrue(np.array_equal(first["raw"],again["raw"]))
            self.assertFalse(np.array_equal(first["raw"],original["raw"]))
            self.assertTrue(np.array_equal(first["raw"][:,:,3],original["raw"][:,:,3]))
            self.assertTrue(np.array_equal(first["y"],original["y"]))
            self.assertTrue(np.array_equal(first["known_floor_A"],original["known_floor_A"]))
            self.assertTrue(np.all(first["raw"][:,:,1]>=np.maximum(first["raw"][:,:,0],first["raw"][:,:,3])))
            self.assertTrue(np.all(first["raw"][:,:,2]<=np.minimum(first["raw"][:,:,0],first["raw"][:,:,3])))
        invalid=copy.deepcopy(plans);invalid["input_noise"]["log_sd"]=.2
        with self.assertRaisesRegex(controller.ControllerError,"stress_scenario_identity"):
            controller._stress_plans({"robustness_stress_plans":invalid})

    def test_freeze_command_calls_strong_final_gate_before_any_output(self):
        from engine import cli
        power_path=self.base/"fake_power.json"
        design=c.read_json(ROOT/"configs/power_design.json")
        c.write_json(power_path,{"status":"DIMENSIONED_CONDITIONAL_EMPIRICAL_DGP","final_R":10,"final_T":768,"final_B":8191,"design_sha256":c.digest(design),"methods_sha256":c.sha(ROOT/"engine/statistics.py")})
        test_spec=self.base/"test_spec.json"
        c.write_json(test_spec,{p:{"start":"2030-01-01","end":"2032-02-06","count":768,"reserve_mode":"post_protocol",
                                  "source_identity_sha256":"a"*64,"exposure_evidence_sha256":"b"*64} for p in controller.PANELS})
        output=self.base/"final_protocol.json"
        report=ROOT/"qa/statistical_validation/verdict_v2.json"
        with mock.patch.object(controller,"_admit_final",side_effect=controller.ControllerError("actual_validation_rejected")) as gate:
            with self.assertRaisesRegex(controller.ControllerError,"actual_validation_rejected"):
                cli.freeze_protocol(self.protocolpath,power_path,report,test_spec,output)
        self.assertEqual(gate.call_count,1)
        self.assertFalse(output.exists())
        prospective=gate.call_args.args[1]
        self.assertEqual(prospective["adequacy"]["config_sha256"],c.sha(ROOT/"configs/adequacy.json"))
        self.assertEqual(set(prospective["robustness_stress_plans"]),{"input_noise","missingness"})

    def test_real_store_mock_Git_real_worker_and_authenticated_restore(self):
        from test_storage import FakeGit,TOKEN,SECRET
        from engine.storage import GithubStore
        self.prepare()
        git=FakeGit()
        store=GithubStore(TOKEN,SECRET,"owner/repo","r3-plan-fixture",source_commit=SOURCE,prefix="runs/r3/controller",sleep=lambda _:None)
        store._request=git.request
        result=controller.run_case(self.source,self.case,self.panels,store,activate=True,max_jobs=1)
        self.assertEqual(result["jobs_executed_this_call"],1)
        c.write_json(self.case/"metrics/assessment.json",{"software_fixture_only":True,"criteria":{}})
        c.write_json(self.case/"analysis_receipt.json",{"software_fixture_only":True,"assessment_sha256":c.sha(self.case/"metrics/assessment.json")})
        with mock.patch.object(data,"load_partition",side_effect=AssertionError("sync must not open input data")), \
             mock.patch.object(training,"predict_job",side_effect=AssertionError("sync/export must not query models")):
            synced=controller.sync_case(self.source,self.case,store)
            exported=controller.export_case(self.source,self.case,self.base/"saved.zip")
        self.assertEqual(synced["status"],"SAVED_CASE_REMOTE_VERIFIED")
        self.assertEqual(exported["new_forwards"],0)
        destination=self.base/"restored"
        restored=store.restore(destination)
        self.assertTrue(restored["remote_verified"])
        self.assertEqual(c.read_json(destination/"analysis_receipt.json")["assessment_sha256"],c.sha(destination/"metrics/assessment.json"))
        plan=c.read_json(destination/"plan.json");job=plan["jobs"][0]
        self.assertEqual(training.verify_job(job,destination/"jobs"/job["id"])["status"],"completed_validated_fit")
        self.assertFalse((destination/"inputs").exists())

    def test_real_candidate_worker_runs_one_synthetic_controller_job(self):
        self.prepare()
        result=controller.run_case(self.source,self.case,self.panels,self.store,activate=True,max_jobs=1)
        self.assertEqual(result["jobs_executed_this_call"],1)
        plan=c.read_json(self.case/"plan.json");job=plan["jobs"][0]
        done=training.verify_job(job,self.case/"jobs"/job["id"])
        self.assertGreater(done["optimizer_steps_committed"],0)
        self.assertIn("before_fit",self.store.calls)
        self.assertIn("job_completed",self.store.calls)


if __name__=="__main__":unittest.main()
