"""Versioned implementation optimization and coverage-first E2 selection.

Frozen A3 is loaded from a separate checkout. A separate campaign binds the
implementation, CUDA resource profile and exact legacy snapshot. No original
record is edited; reused outcomes are not booked as new fits or new GPU steps.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
ENGINE = "564ba4762f84a800b02fa4c734f5752a169f5223"
LEGACY_PLAN = "4eae2d2ff9ef6f83a2f9eb0dc5ba647208d695e1665a8e522d812fed67893bda"
REPOSITORY = "jassan-bit/DeepLearning_202630"
BRANCH = "campaigns/e2-cuda-crop-11-v2"
OLD_BRANCH = "campaigns/e2-cuda-selection-v1"
ORIGINS = tuple("2024-%02d-01" % month for month in range(1,11))+("2025-01-01",)
VERSION = "e2-cuda-crop-plan-v2"
REGISTRATION = ROOT/"experiments/e2_implementation_optimization_11_origins_20261111/registration.json"
FILES = ("scripts/e2_timesnet_exact_crop.py", "scripts/run_e2_optimized_selection_v2.py")


class AdmissionClock:
    """Charge successful admission/profile and cached replay, not just new fits."""
    def __init__(self, started):
        self.started, self.first = started, True
    def __call__(self):
        if self.first:
            self.first=False
            return self.started
        return time.monotonic()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_registration():
    raw = REGISTRATION.read_bytes()
    value = json.loads(raw)
    if value["status"] != "registered_before_new_market_fits":
        raise ValueError("registered_optimization_required")
    actual = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES}
    if value["implementation_sha256"] != actual:
        raise ValueError("optimization_implementation_changed")
    return value, hashlib.sha256(raw).hexdigest()


def install_public_schema(module):
    """An explicit extension; the frozen positive public schema stays intact."""
    if getattr(module, "_crop_schema_installed_v2", False):
        return
    original_plan = module._validate_plan
    def validate_plan(plan):
        if plan.get("version") != VERSION:
            return original_plan(plan)
        clean = dict(plan)
        opt, cache = clean.pop("optimization"), clean.pop("legacy_cache")
        module._keys(opt, {"profile", "profile_sha256", "fast_cases"},
                     {"profile", "profile_sha256", "fast_cases"})
        if digest(opt["profile"]) != opt["profile_sha256"]:
            raise ValueError("profile_hash_changed")
        validate_profile(opt["profile"])
        if (opt["profile"]["device"] != "cuda:0" or not opt["profile"]["synthetic_only"]
                or opt["profile"]["market_data_read"] or
                opt["fast_cases"] != opt["profile"]["fast_cases"]):
            raise ValueError("real_cuda_resource_profile_required")
        if any(len(case)!=2 or case[0] not in (16,32,64,128,256) or case[1] not in (16,32,64) for case in opt["fast_cases"]):
            raise ValueError("invalid_profiled_widths")
        module._keys(cache, {"commit", "plan_sha256", "completed_jobs", "spent_seconds", "limit_seconds"},
                     {"commit", "plan_sha256", "completed_jobs", "spent_seconds", "limit_seconds"})
        if (cache["plan_sha256"] != LEGACY_PLAN or len(cache["commit"]) != 40
                or type(cache["completed_jobs"]) is not int or cache["completed_jobs"] < 1802
                or not 0 <= cache["spent_seconds"] < cache["limit_seconds"]
                or cache["limit_seconds"] != 172800):
            raise ValueError("legacy_cache_binding_invalid")
        if (clean["origins"] != list(ORIGINS) or clean["k"] != 40
                or not math.isclose(clean["initial_controls"]["campaign_budget_seconds"],
                    cache["limit_seconds"]-cache["spent_seconds"],rel_tol=0,abs_tol=1e-6)
                or opt["profile"]["device_name"] != clean["hardware_binding"]["device_name"]):
            raise ValueError("scientific_schedule_or_transferred_budget_changed")
        original_plan(clean)
    module._validate_plan = validate_plan
    module.STATUS_KEYS.update({"reused_jobs", "total_available_unique_jobs", "combined_charged_seconds"})
    original_record = module.validate_record
    def validate_record(path, raw):
        value = original_record(path, raw)
        if path == "status.json":
            for key in ("reused_jobs", "total_available_unique_jobs"):
                if key in value and (type(value[key]) is not int or value[key] < 0):
                    raise ValueError("invalid_reuse_counter")
            if "combined_charged_seconds" in value and value["combined_charged_seconds"] < 0:
                raise ValueError("invalid_combined_budget")
        return value
    module.validate_record = validate_record
    module._crop_schema_installed_v2 = True


def validate_profile(profile):
    if (profile.get("version") != "timesnet-support-crop-v1"
            or profile.get("device") != "cuda:0" or profile.get("torch") != "2.8.0+cu126"
            or profile.get("repeats") != 5 or profile.get("synthetic_only") is not True
            or profile.get("market_data_read") is not False
            or profile.get("optimizer_included") is not False):
        raise ValueError("invalid_synthetic_cuda_profile")
    rows=profile.get("rows",[])
    if [(row.get("width"),row.get("batch")) for row in rows] != [(w,b) for w in (16,32,64,128,256) for b in (16,32,64)]:
        raise ValueError("incomplete_width_profile")
    for row in rows:
        before,after=row["original_seconds"],row["cropped_seconds"]
        if (row.get("num_kernels")!=6 or not all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in (before,after))
                or row.get("forward_and_gradient_parity") is not True
                or not math.isclose(row["speedup"],before/after,rel_tol=1e-12)):
            raise ValueError("invalid_profile_measurement")
    expected=[[row["width"],row["batch"]] for row in rows if row["speedup"]>=1.05]
    if profile.get("fast_cases")!=expected or profile.get("cuda_gate_passed")!=bool(expected):
        raise ValueError("profile_dispatch_rule_changed")


def load_engine(root):
    root = Path(root).resolve()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != ENGINE or root == ROOT.resolve():
        raise ValueError("separate_frozen_A3_checkout_required")
    sys.path.insert(0, str(root/"scripts"))
    runner = importlib.import_module("run_e2_cuda_search")
    if Path(runner.__file__).resolve().parent != root/"scripts":
        raise ValueError("engine_import_provenance_changed")
    runner.verify_source(ENGINE)
    return runner


def cached_outcome(legacy, legacy_plan, runner, prepared, model, candidate, seed):
    job = runner.job_payload(legacy_plan, prepared, model, candidate, seed)
    identity = runner.admission.digest({"plan_sha256": LEGACY_PLAN, "job": job})
    value = legacy.get_result(identity, job)
    if value is not None and value["resource_use"]["identity"] != runner.identity(job):
        raise ValueError("legacy_outcome_identity_changed")
    return identity, value


def require_no_previous_optimization(mirror):
    if mirror.inspect_remote() is not None:
        raise ValueError("previous_optimization_requires_explicit_budget_and_cache_migration")


def run_selection(plan, legacy_plan, legacy, session, panel, runner, worker, mirror, progress, reused):
    import e2_training_models as registry
    from e2_training_bindings import run_neural_selection
    from e2_temporal_training import prepare_b
    # No hidden pruning: complete the inexpensive fixed-body wave first,
    # then traverse every admitted NAS architecture at the same eleven origins.
    waves = ([n for n in plan["models"] if n not in registry.MODEL_IDS],
             [n for n in plan["models"] if n in registry.MODEL_IDS])
    def sync():
        paths = session.pending_record_paths
        if paths:
            mirror.push(session.directory, paths=paths, message="Confirm versioned optimized E2 selection")
            session.mark_records_synced(paths)
    for names in waves:
        for origin in plan["origins"]:
            prepared = prepare_b(panel, origin, origin)
            for model in names:
                class Adapter:
                    wall=7200;short_wall=1800;memory=12*1024**3
                    full_epochs=200;short_epochs=50;patience=20
                    def fit_candidate(self, candidate, seed, train, validation):
                        if (train.signature != prepared.inner_train.signature
                                or validation.signature != prepared.inner_validation.signature):
                            raise ValueError("selection_split_changed")
                        old_id, old = cached_outcome(legacy, legacy_plan, runner, prepared, model, candidate, seed)
                        if old is not None:
                            reused.add(old_id)
                            return runner.restore_trial(old)
                        job = runner.job_payload(plan, prepared, model, candidate, seed)
                        identity = digest({"plan_sha256": digest(plan), "job": job})
                        phases = 3 if model == "LSTM-autoencoder con predictor" else 1
                        seconds = (1800 if job["stage"] == "nas1" else 7200)*phases+300
                        cached = session.get_result(identity, job) is not None
                        if not cached:
                            mirror.check_capacity(required_requests=32)
                        def call(deadline):
                            sync()
                            print(json.dumps({"event":"new_fit_started","model":model,
                                "origin":origin,"stage":job["stage"],"seed":seed}),flush=True)
                            return worker(job, deadline)
                        result = session.run_job(identity, job, call, reservation_seconds=seconds)
                        sync()
                        if not cached:
                            print(json.dumps({"model":model,"origin":origin,"stage":job["stage"],
                                              "status":result["status"],"implementation":"support_crop_v1"}),flush=True)
                        return runner.restore_trial(result)
                p = registry.protocol()
                architectures = runner.admission.planner.enumerate_architectures(p, model) if model in registry.MODEL_IDS else []
                selected, _ = run_neural_selection(prepared, model, Adapter(),
                    hpo_trials=registry.registered_trials(model), architectures=architectures,
                    hpo_seeds=plan["seed_pools"]["hpo"], phase1_seeds=plan["seed_pools"]["nas1"],
                    phase2_seeds=plan["seed_pools"]["nas2"], k=40, delta=p["contract"]["delta"])
                progress.append({"model":model,"origin":origin,"winner_id":selected.winner.candidate_id,
                    "winner":selected.winner.payload,"fixed_epochs":selected.fixed_epochs,
                    "selection_signature":selected.signature,"attempts":selected.attempts,
                    "selection_rule":selected.selection_rule})


def worker_main(path, engine_root):
    registration, registration_sha = read_registration()
    runner = load_engine(engine_root)
    spec = json.loads(Path(path).read_text())
    outer = spec["outer_plan"]
    if digest(spec["plan"])!=LEGACY_PLAN:
        raise ValueError("worker_legacy_plan_changed")
    for key in ("models","protocol_sha256","seed_pools","k","stage_caps","execution"):
        if outer[key]!=spec["plan"][key]:
            raise ValueError("worker_scientific_domain_changed")
    import e2_github_campaign as public
    install_public_schema(public)
    public._validate_plan(outer)
    if spec["job_id"]!=digest({"plan_sha256":digest(outer),"job":spec["job"]}):
        raise ValueError("worker_job_binding_changed")
    if outer["registration_sha256"] != registration_sha:
        raise ValueError("worker_registration_changed")
    for name, sha in registration["implementation_sha256"].items():
        if outer["source_sha256"].get(name) != sha:
            raise ValueError("worker_implementation_changed")
    import e2_timesnet_exact_crop as crop
    crop.install(outer["optimization"]["fast_cases"])
    # The A3 worker still validates scientific files, split, candidate, GPU,
    # deterministic policy and source. Only the separately pinned kernel is
    # installed. The outer envelope records this implementation version.
    inner = {k:v for k,v in spec.items() if k != "outer_plan"}
    inner["plan"]=deepcopy(inner["plan"])
    inner["plan"]["hardware_binding"]=outer["hardware_binding"]
    with tempfile.TemporaryDirectory(prefix="e2-opt-inner-") as temporary:
        inner_path = Path(temporary)/"spec.json"
        runner.write_json(inner_path, inner)
        runner.run_worker(inner_path)
    output = Path(spec["output"])/"worker_profile.json"
    payload = json.loads(output.read_text())
    payload["plan_sha256"] = digest(outer)
    runner.write_json(output, payload)
    return 0


def isolated_trial(plan, legacy_plan, job, deadline, args, runner):
    from profile_e2_training import run_isolated_profile
    from e2_temporal_training import TrialRun
    job_id = digest({"plan_sha256":digest(plan),"job":job})
    phases = 3 if job["model"] == "LSTM-autoencoder con predictor" else 1
    cap = 1800 if job["stage"] == "nas1" else 7200
    with tempfile.TemporaryDirectory(prefix="e2-opt-worker-") as temporary:
        root = Path(temporary);output = root/"attempt"
        spec = {"outer_plan":plan,"plan":legacy_plan,"job":job,"job_id":job_id,
            "output":str(output),"data_root":str(args.data_root.resolve()),
            "source_root_itransformer":str(args.source_root_itransformer.resolve())}
        runner.write_json(root/"spec.json",spec)
        environment = dict(os.environ)
        for name in ("GITHUB_TOKEN","GH_TOKEN","PYTHONPATH"):
            environment.pop(name,None)
        environment.update(CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONNOUSERSITE="1",
            OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2")
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise RuntimeError("admitted_job_has_no_remaining_time")
        result = run_isolated_profile([sys.executable,str(Path(__file__).resolve()),
            "--worker-spec",str(root/"spec.json"),"--engine-root",str(args.engine_root)],output,
            timeout_seconds=min(phases*cap+120,remaining),memory_limit_bytes=12*1024**3,
            cwd=ROOT,environment=environment)
        payload = result.get("worker_profile")
        if (result["status"] == "success" and isinstance(payload,dict)
                and payload.get("job_id") == job_id and payload.get("plan_sha256") == digest(plan)):
            trial = runner.restore_trial(payload["trial_run"])
            if trial.resource_use.get("identity") != runner.identity(job):
                raise ValueError("optimized_worker_identity_changed")
            trial.resource_use.update(supervisor_elapsed_seconds=result["elapsed_wall_seconds"],
                supervisor_rss_bytes=result["sampled_peak_process_group_rss_bytes"])
            return runner.public_trial(trial)
        runner.retain_private_diagnostics(output,job_id)
        code = {"timeout":"external_timeout","memory_limit_observed":"external_memory"}.get(result["status"],"worker_result_invalid")
        lower=0
        if (isinstance(payload,dict) and payload.get("job_id")==job_id
                and payload.get("plan_sha256")==digest(plan)):
            try:
                partial=runner.restore_trial(payload["trial_run"])
                if partial.resource_use.get("identity")==runner.identity(job):
                    lower=partial.resource_use.get("optimizer_steps_known_lower_bound",0)
            except (ValueError,KeyError,TypeError):
                pass
        trial = TrialRun.failed(code,{"error_code":code,"optimizer_steps":None,
            "optimizer_steps_known_lower_bound":lower,"partial_optimizer_update_may_have_occurred":True,
            "identity":runner.identity(job)},"external-"+job_id,fatal=code=="worker_result_invalid")
        return runner.public_trial(trial)


def execute(args):
    started=time.monotonic()
    registration, registration_sha = read_registration()
    head = subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],check=True,
                          capture_output=True,text=True).stdout.strip()
    if head != args.controller_commit:
        raise ValueError("controller_checkout_pin_mismatch")
    changed=subprocess.run(["git","-C",str(ROOT),"diff","--name-only","HEAD"],check=True,
                           capture_output=True,text=True).stdout.strip()
    extras=subprocess.run(["git","-C",str(ROOT),"ls-files","--others","--exclude-standard","scripts"],
                          check=True,capture_output=True,text=True).stdout.splitlines()
    if changed or any(name.endswith('.py') for name in extras):
        raise ValueError("controller_checkout_has_unregistered_changes")
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError("github_secret_required")
    runner = load_engine(args.engine_root)
    hardware = runner.cuda_hardware()
    import e2_github_campaign as public
    from e2_github_git_transport import GitCampaignMirror
    from e2_search_session import SearchSession, SessionPaused, UnresolvedAttempt
    install_public_schema(public)
    mirrors=[];legacy=session=None;plan=None;reused=set();progress=[]
    status="not_started";exit_code=0;pause=None
    try:
        previous=GitCampaignMirror(REPOSITORY,"campaigns/e2-cuda-crop-v1","campaigns/e2-cuda-crop-v1")
        mirrors.append(previous)
        require_no_previous_optimization(previous)
        target=GitCampaignMirror(REPOSITORY,BRANCH,BRANCH);mirrors.append(target)
        target.restore(args.output)
        existing=args.output/"campaign.json"
        if existing.is_file():
            plan=json.loads(existing.read_text())["plan"]
            if plan["registration_sha256"]!=registration_sha or plan["hardware_binding"]!=hardware:
                raise ValueError("existing_optimized_campaign_binding_changed")
            public._validate_plan(plan)
        else:
            import e2_timesnet_exact_crop as crop
            print(json.dumps({"stage":"synthetic_cuda_resource_profile","cases":15,
                "market_training_started":False}),flush=True)
            profile=crop.benchmark(device="cuda:0",repeats=5)
            print(json.dumps({"resource_profile":profile},ensure_ascii=False),flush=True)
        old=GitCampaignMirror(REPOSITORY,OLD_BRANCH,OLD_BRANCH);mirrors.append(old)
        old.restore(args.legacy_cache)
        old_config=json.loads((args.legacy_cache/"campaign.json").read_text())
        legacy_plan=old_config["plan"]
        if digest(legacy_plan)!=LEGACY_PLAN:
            raise ValueError("legacy_scientific_plan_changed")
        ancestry=old.git_transport._run(["merge-base","--is-ancestor",registration["legacy_min_commit"],old._head],allow_failure=True)
        if ancestry.returncode:
            raise ValueError("legacy_snapshot_not_descendant_of_verified_recovery")
        legacy=SearchSession(args.legacy_cache,legacy_plan,session_budget_seconds=28800,
            campaign_budget_seconds=86400,max_storage_bytes=10*1024**3)
        if plan is None:
            remaining=legacy.campaign_budget_seconds-legacy.charged_wall_seconds
            if remaining<=0 or legacy.campaign_budget_seconds!=172800:
                raise ValueError("no_transferrable_registered_budget")
            plan=deepcopy(legacy_plan)
            plan.update(version=VERSION,registration_sha256=registration_sha,
                origins=list(ORIGINS),remote_storage={"repository":REPOSITORY,"branch":BRANCH,"prefix":BRANCH},
                scope="new_fits_only_eleven_origin_implementation_optimization",
                optimization={"profile":profile,"profile_sha256":digest(profile),"fast_cases":profile["fast_cases"]},
                legacy_cache={"commit":old._head,"plan_sha256":LEGACY_PLAN,
                    "completed_jobs":len(legacy.completed_job_ids),"spent_seconds":legacy.charged_wall_seconds,
                    "limit_seconds":legacy.campaign_budget_seconds})
            plan["source_sha256"].update(registration["implementation_sha256"])
            plan["initial_controls"]["campaign_budget_seconds"]=remaining
            plan["coverage"].update(selected_origins=11,partial_origins=False)
        if plan["legacy_cache"]["commit"]!=old._head:
            raise ValueError("legacy_snapshot_advanced_after_optimized_campaign_seal")
        public._validate_plan(plan)
        session=SearchSession(args.output,plan,session_budget_seconds=28800,
            campaign_budget_seconds=plan["initial_controls"]["campaign_budget_seconds"],max_storage_bytes=10*1024**3,
            clock=AdmissionClock(started))
        session._start_session()
        paths=session.pending_record_paths
        target.push(args.output,paths=paths,message="Seal optimized implementation and transferred remaining budget")
        session.mark_records_synced(paths)
        panel,_=runner.prepare_panel(args.data_root)
        run_selection(plan,legacy_plan,legacy,session,panel,runner,
            lambda job,deadline:isolated_trial(plan,legacy_plan,job,deadline,args,runner),target,progress,reused)
        status="completed_eleven_origin_selection_not_final_benchmark"
    except SessionPaused as error:
        status="paused_"+error.reason;pause=error.reason
    except UnresolvedAttempt:
        status="reconciliation_required";exit_code=3
    except public.CampaignSyncError:
        status="paused_github_sync";exit_code=4
    except KeyboardInterrupt:
        status="interrupted";exit_code=130
    except Exception:
        status="blocked_runtime_failure";exit_code=2
        # Detailed diagnostic remains on this executor, outside public records.
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        if session is not None:
            session.close()
            state=runner.report(plan,session,status,progress,REPOSITORY,BRANCH,pause_reason=pause)
            state.update(reused_jobs=len(reused),
                total_available_unique_jobs=plan["legacy_cache"]["completed_jobs"]+len(session.completed_job_ids),
                combined_charged_seconds=plan["legacy_cache"]["spent_seconds"]+session.charged_wall_seconds)
            runner.write_json(args.output/"status.json",state)
            target.push(args.output,paths=(*session.pending_record_paths,"status.json"),message="Save optimized selection outcome")
        if legacy is not None:
            legacy.close()
        for mirror in mirrors:
            mirror.close()
        print(json.dumps({"status":status,"new_completed_jobs":None if session is None else len(session.completed_job_ids),
            "reused_jobs":len(reused),"branch":BRANCH,"final_R":None}),flush=True)
    return exit_code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--worker-spec",type=Path)
    parser.add_argument("--engine-root",type=Path)
    parser.add_argument("--controller-commit")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--legacy-cache",type=Path)
    parser.add_argument("--data-root",type=Path)
    parser.add_argument("--source-root-itransformer",type=Path)
    args=parser.parse_args(argv)
    if args.worker_spec:
        return worker_main(args.worker_spec,args.engine_root)
    if args.plan_only:
        registration,sha=read_registration()
        print(json.dumps({"registration_sha256":sha,"origins":ORIGINS,"upper_bound":85074,
            "nas_exemptions_added":0,"architecture_changed":False,
            "cuda_profile_required":True,"budget_reset":False,"training_started":False},indent=2))
        return 0
    if not args.execute or any(v is None for v in (args.engine_root,args.output,args.legacy_cache,
            args.data_root,args.source_root_itransformer,args.controller_commit)):
        parser.error("explicit execution and all frozen/local paths required")
    return execute(args)


if __name__=="__main__":
    raise SystemExit(main())
