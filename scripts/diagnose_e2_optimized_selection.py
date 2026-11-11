"""Local diagnostic only: synthetic CUDA and remote metadata reads, never fits/pushes."""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def stage(name):
    print("DIAGNOSTICO:", name, flush=True)


def diagnose(args):
    controller = args.controller_root.resolve()
    stage("fuentes y registro")
    head = subprocess.run(["git", "-C", str(controller), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != args.controller_commit:
        raise ValueError("controller_checkout_pin_mismatch")
    sys.path.insert(0, str(controller / "scripts"))
    opt = importlib.import_module("run_e2_optimized_selection_v2")
    registration, registration_sha = opt.read_registration()
    runner = opt.load_engine(args.engine_root)
    stage("Python, dependencias y GPU CUDA")
    hardware = runner.cuda_hardware()
    print(json.dumps(hardware), flush=True)
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError("github_secret_required")
    import e2_github_campaign as public
    from e2_github_git_transport import GitCampaignMirror
    from e2_search_session import SearchSession, _json_value
    opt.install_public_schema(public)
    mirrors = []
    legacy = None
    # Restores/replay are exclusively in a disposable directory. No real local
    # campaign is changed; no session is started and no remote push is called.
    with tempfile.TemporaryDirectory(prefix="e2-diagnostic-only-") as directory:
        base = Path(directory)
        try:
            stage("consulta de ramas, sin escrituras remotas")
            previous = GitCampaignMirror(opt.REPOSITORY, "campaigns/e2-cuda-crop-v1",
                                         "campaigns/e2-cuda-crop-v1")
            mirrors.append(previous)
            opt.require_no_previous_optimization(previous)
            target = GitCampaignMirror(opt.REPOSITORY, opt.BRANCH, opt.BRANCH)
            mirrors.append(target)
            target.restore(base / "optimized")
            existing = base / "optimized/campaign.json"
            plan = None
            if existing.is_file():
                plan = json.loads(existing.read_text())["plan"]
                if plan["registration_sha256"] != registration_sha or plan["hardware_binding"] != hardware:
                    raise ValueError("existing_optimized_campaign_binding_changed")
                public._validate_plan(plan)
            else:
                stage("perfil sintetico CUDA; NO entrenamiento de mercado")
                import e2_timesnet_exact_crop as crop
                profile = opt.json_native_profile(crop.benchmark(device="cuda:0", repeats=5))
                opt.validate_profile(profile)
                print(json.dumps({"fast_cases": profile["fast_cases"]}), flush=True)
            stage("lectura y verificacion de copia temporal del historial")
            old = GitCampaignMirror(opt.REPOSITORY, opt.OLD_BRANCH, opt.OLD_BRANCH)
            mirrors.append(old)
            old.restore(base / "legacy")
            legacy_plan = json.loads((base / "legacy/campaign.json").read_text())["plan"]
            if opt.digest(legacy_plan) != opt.LEGACY_PLAN:
                raise ValueError("legacy_scientific_plan_changed")
            ancestry = old.git_transport._run(["merge-base", "--is-ancestor",
                registration["legacy_min_commit"], old._head], allow_failure=True)
            if ancestry.returncode:
                raise ValueError("legacy_snapshot_not_descendant_of_verified_recovery")
            legacy = SearchSession(base / "legacy", legacy_plan,
                session_budget_seconds=28800, campaign_budget_seconds=86400,
                max_storage_bytes=10 * 1024**3)
            remaining = legacy.campaign_budget_seconds - legacy.charged_wall_seconds
            if plan is None:
                if remaining <= 0 or legacy.campaign_budget_seconds != 172800:
                    raise ValueError("no_transferrable_registered_budget")
                plan = deepcopy(legacy_plan)
                plan.update(version=opt.VERSION, registration_sha256=registration_sha,
                    origins=list(opt.ORIGINS),
                    remote_storage={"repository": opt.REPOSITORY, "branch": opt.BRANCH, "prefix": opt.BRANCH},
                    scope="new_fits_only_eleven_origin_implementation_optimization",
                    optimization={"profile": profile, "profile_sha256": opt.digest(profile), "fast_cases": profile["fast_cases"]},
                    legacy_cache={"commit": old._head, "plan_sha256": opt.LEGACY_PLAN,
                        "completed_jobs": len(legacy.completed_job_ids),
                        "spent_seconds": legacy.charged_wall_seconds, "limit_seconds": legacy.campaign_budget_seconds})
                plan["source_sha256"].update(registration["implementation_sha256"])
                plan["initial_controls"]["campaign_budget_seconds"] = remaining
                plan["coverage"].update(selected_origins=11, partial_origins=False)
            if plan["legacy_cache"]["commit"] != old._head:
                raise ValueError("legacy_snapshot_advanced_after_optimized_campaign_seal")
            stage("validacion del plan antes de sellar")
            public._validate_plan(plan)
            _json_value(plan)  # Exercise the exact strict ledger boundary as well.
            print(json.dumps({"diagnostic": "passed_pre_seal_checks",
                "previous_completed_jobs": len(legacy.completed_job_ids),
                "remaining_seconds": remaining, "market_training_started": False,
                "campaign_session_started": False, "remote_writes": False}), flush=True)
        finally:
            with ExitStack() as cleanup:
                for mirror in mirrors:
                    cleanup.callback(mirror.close)
                if legacy is not None:
                    cleanup.callback(legacy.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--controller-commit", required=True)
    args = parser.parse_args()
    try:
        diagnose(args)
    except Exception:
        # No locals/env dumping. The caller additionally redacts known secrets.
        traceback.print_exc(file=sys.stdout)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
