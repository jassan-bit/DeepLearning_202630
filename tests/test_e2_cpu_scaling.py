"""Synthetic-only checks for sealed CPU domains and instrumentation semantics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_e2_cpu_scaling as runner


def registration():
    return {"cpu_profile":{"thread_epochs":5,"scaling_epochs":2}}


def kernel_records(dense=10.,cropped=9.):
    return [{"job":job,"status":"success","elapsed_wall_seconds":
             dense if job["backend"]=="dense" else cropped,
             "worker_profile":{"status":"completed"}}
            for job in runner.build_jobs("kernels",registration())]


def thread_records(costs=None):
    costs=costs or {1:10.,2:10.,4:10.}
    return [{"job":job,"status":"success",
             "worker_profile":{"status":"completed","total_work_seconds":costs[job["threads"]],
                               "validation_RMSE":1000/job["threads"]}}
            for job in runner.build_jobs("threads",registration(),kernel_backend="cropped")]


class ProfileDomainTests(unittest.TestCase):
    def test_draft_registration_cannot_start_a_market_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"registration.json"
            path.write_text(json.dumps({"version":runner.REGISTRATION_VERSION,"status":"draft"}))
            with self.assertRaisesRegex(ValueError,"draft registration"):
                runner.load_registration(path)

    def test_sampling_weights_and_exact_job_domains(self):
        frame=runner.profile_sampling_frame()
        self.assertEqual(len(frame),12)
        self.assertEqual(sum(x["stratum_size"] for x in frame),1000)
        self.assertEqual(sum(x["sampling_role"]=="anchor" for x in frame),3)
        self.assertEqual([x["stratum_index"] for x in frame[3:]],list(range(9)))
        self.assertEqual(len({x["pattern"]["pattern_id"] for x in frame}),12)
        for phase,n in (("kernels",8),("threads",12),("scaling",86)):
            jobs=runner.build_jobs(phase,registration(),2 if phase=="scaling" else None)
            self.assertEqual(len(jobs),n)
            self.assertEqual(len({j["job_id"] for j in jobs}),n)
        scales=runner.build_jobs("scaling",registration(),2)
        self.assertEqual(sum(j["sampling"] is not None for j in scales),72)
        self.assertEqual(sum(j["cohort"]=="small" for j in scales),6)
        self.assertTrue(all(j["epochs"]==2 for j in scales))
        with self.assertRaises(ValueError):runner.build_jobs("scaling",registration(),3)

    def test_thread_selection_is_global_complete_and_runtime_only(self):
        rows=thread_records({1:10.4,2:10.,4:20.})
        selected=runner.select_thread_policy(rows)
        self.assertEqual(selected["selected_threads"],1)
        for r in rows:r["worker_profile"]["validation_RMSE"]=-999*r["job"]["threads"]
        self.assertEqual(selected,runner.select_thread_policy(rows))
        self.assertIsNone(runner.select_thread_policy(rows[:-1])["selected_threads"])
        rows[-1]["status"]="timeout"
        self.assertIsNone(runner.select_thread_policy(rows)["selected_threads"])
        with self.assertRaises(ValueError):runner.select_thread_policy(rows+rows[:1])

    def test_kernel_policy_requires_both_cases_parity_and_all_crops(self):
        rows=kernel_records()
        self.assertEqual(runner.select_kernel_policy(rows,parity_suite_passed=True)["selected_backend"],"cropped")
        self.assertEqual(runner.select_kernel_policy(rows,parity_suite_passed=False)["selected_backend"],"dense")
        rows[-1]["elapsed_wall_seconds"]=10.
        rows[-2]["elapsed_wall_seconds"]=10.
        # Changing both backends in one replicate leaves that case below the
        # required5% median gain, even though the other case is faster.
        for row in rows:
            if row["job"]["pattern"]["widths"]==[256]*3 and row["job"]["backend"]=="cropped":
                row["elapsed_wall_seconds"]=9.6
        self.assertEqual(runner.select_kernel_policy(rows,parity_suite_passed=True)["selected_backend"],"dense")
        rows=kernel_records()
        next(r for r in rows if r["job"]["backend"]=="cropped")["status"]="timeout"
        self.assertEqual(runner.select_kernel_policy(rows,parity_suite_passed=True)["selected_backend"],"dense")

    def test_kernel_dense_wall_censor_is_only_a_lower_bound(self):
        rows=kernel_records()
        dense=next(r for r in rows if r["job"]["backend"]=="dense")
        dense.update(status="timeout",elapsed_wall_seconds=240.,worker_profile=None)
        choice=runner.select_kernel_policy(rows,parity_suite_passed=True)
        self.assertEqual(choice["selected_backend"],"cropped")
        self.assertEqual(len(choice["dense_wall_censored_cases"]),1)
        dense.update(status="worker_failure",worker_profile={"error_code":"implementation_failure"})
        self.assertIsNone(runner.select_kernel_policy(rows,parity_suite_passed=True)["selected_backend"])

    def test_micro_fallback_retains_distinct_scope_and_seed(self):
        job=runner.build_jobs("scaling",registration(),2)[0]
        failed={"job":job,"status":"timeout","worker_profile":None}
        self.assertTrue(runner.eligible_micro_fallback(failed))
        micro=runner.micro_job(job)
        self.assertEqual(job["phase"],"scaling")
        self.assertEqual(micro["phase"],"micro_fallback")
        self.assertEqual(micro["parent_job_id"],job["job_id"])
        self.assertNotEqual(micro["seed"],job["seed"])
        self.assertEqual((micro["epochs"],micro["batch_windows"]),(1,32))
        failed.update(status="worker_failure",worker_profile={"error_code":"nonfinite"})
        self.assertFalse(runner.eligible_micro_fallback(failed))

    def test_runtime_policy_seals_both_registration_and_kernel_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            kpath,tpath=Path(directory)/"kernels.json",Path(directory)/"threads.json"
            plan={"registration_sha256":"a"*64,"registration_commit":"b"*40,"parity_suite_passed":True}
            runner.write_json_new(kpath,{"plan":plan,"results":kernel_records()})
            threadplan={**plan,"kernels_summary_sha256":runner.digest(kpath)}
            rows=thread_records()
            runner.write_json_new(tpath,{"plan":threadplan,"results":rows,
                                        "thread_policy":runner.select_thread_policy(rows)})
            persisted=json.loads(tpath.read_text())["thread_policy"]
            self.assertEqual(set(persisted["normalized_runtime_scores"]),{"1","2","4"})
            self.assertTrue(all(set(values)=={"1","2","4"}
                for values in persisted["family_median_work_seconds"].values()))
            policy=runner.load_runtime_policy(kpath,tpath)
            self.assertEqual((policy["threads"],policy["crop"]),(1,True))
            threadplan["registration_commit"]="c"*40
            tpath.write_text(json.dumps({"plan":threadplan,"results":thread_records()}))
            with self.assertRaises(ValueError):runner.load_runtime_policy(kpath,tpath)

    def test_large_calendar_placeholder_preserves_admission_and_619_cut(self):
        from test_e2_development_data import registered_cache
        with registered_cache("2024-12-30") as (root,_):
            cohort,receipt=runner.prepare_cohort(root,"large")
            self.assertEqual((len(cohort.inner_train.dates),len(cohort.inner_validation.dates)),(619,83))
            self.assertEqual(receipt["placeholder_dates"],["2024-12-31","2025-01-01"])
            self.assertEqual(receipt["train_end"],"2024-09-25")
            self.assertEqual(receipt["validation_end"],"2024-12-24")
            self.assertFalse(receipt["numeric_2025_market_values_read"])
            self.assertEqual(receipt["outer_observations_available"],0)


class InstrumentationTests(unittest.TestCase):
    def test_observer_preserves_steps_weights_losses_rng_and_restores_objects(self):
        import torch
        from torch import nn
        import e2_staged_training as staged
        torch.set_num_threads(1)
        config=staged.TrainConfig(.001,.0001,16,2,20,123)
        x=torch.arange(35*7*5,dtype=torch.float32).reshape(35,7,5)/1000
        y=torch.arange(35,dtype=torch.float32)/100
        def build():
            torch.manual_seed(98)
            class Tiny(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear=nn.Linear(35,1)
                def forward(self,value):
                    return self.linear(value.flatten(1)).flatten()
            return Tiny()
        base=build()
        expected=staged.fit_supervised(base,x,y,None,None,1.,config,fixed_epochs=2)
        expected_rng=torch.get_rng_state().clone()
        observed=build()
        names=("_fit","batch_indices","_mse","clone_state","state_sha256")
        original={name:getattr(staged,name) for name in names}
        with runner.observe_trainer_timings() as timing:
            actual=staged.fit_supervised(observed,x,y,None,None,1.,config,fixed_epochs=2)
        self.assertEqual(actual.optimizer_steps,expected.optimizer_steps)
        self.assertEqual(actual.optimizer_steps,6)
        self.assertEqual(actual.selected_state_sha256,expected.selected_state_sha256)
        self.assertEqual(actual.logs,expected.logs)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        self.assertTrue(all(getattr(staged,name) is value for name,value in original.items()))
        self.assertEqual(len(timing),1)
        t=timing[0]
        self.assertEqual(t["batch_calls"],3)
        self.assertAlmostEqual(t["total_seconds"],sum(t[k+"_seconds"] for k in ("setup","loop","finalization")),places=10)
        self.assertAlmostEqual(t["loop_seconds"],t["update_loop_seconds"]+t["train_evaluation_seconds"],places=10)
        self.assertTrue(all(v>=0 for k,v in t.items() if k.endswith("_seconds")))
        self.assertTrue(t["completed"])

    def test_observer_restores_functions_after_failure(self):
        import e2_staged_training as staged
        original=staged._fit
        with self.assertRaisesRegex(RuntimeError,"synthetic"):
            with runner.observe_trainer_timings():
                raise RuntimeError("synthetic")
        self.assertIs(staged._fit,original)


if __name__=="__main__":unittest.main()
