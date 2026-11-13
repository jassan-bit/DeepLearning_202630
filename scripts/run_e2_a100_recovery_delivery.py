"""Registered funding continuation of the unchanged reduced scientific engine.

This adapter runs only after the earlier controller stops. Proven completed
local work is reconciled before restoration; incomplete work stays blocked. Funding
is validated from a separate explicit user authorization before opening a
budget; no published scientific source file or active process is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
VERSION = "e2-a100-recovery-controller-v1"
FUNDING_COMMIT = "f540227c23c572b9ca7fb36920588727b029484a"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
REGISTRATION_PATH = "experiments/e2_a100_recovery_20261113/registration.json"
RESOURCE_REGISTRATION_PATH = "experiments/e2_resource_continuation_20261112/registration.json"


def _engine_root(path):
    root = Path(path).absolute()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != ENGINE_COMMIT:
        raise ValueError("The registered scientific engine E must remain unchanged")
    return root


def _extension_root(path, commit):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the immutable approved funding-extension commit")
    root = Path(path).absolute()
    record = json.loads((root / "registration.json").read_text())
    if record.get("engine_commit") != ENGINE_COMMIT or record.get("protocol_id") != "e2-reduced-delivery-v1":
        raise ValueError("The funding extension is bound to another protocol/engine")
    required = ("scripts/run_e2_a100_recovery_delivery.py",
        "scripts/e2_expired_lease_recovery.py", "scripts/e2_a100_cache_selection.py")
    for relative in required:
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve()) or target.is_symlink()
                or hashlib.sha256(target.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A registered continuation source differs: " + relative)
    return root, record


def verify_published_registration(engine,root,commit,path):
    raw=(root/'registration.json').read_bytes()
    published=subprocess.check_output(['git','-C',str(engine),'show',commit+':'+path])
    if raw!=published:raise ValueError('The local extension registration differs from its immutable commit')
    record=json.loads(raw)
    if record.get('engine_commit')!=ENGINE_COMMIT or record.get('protocol_id')!='e2-reduced-delivery-v1':
        raise ValueError('The extension registration belongs to another engine/protocol')
    for relative,expected in record.get('files_sha256',{}).items():
        if not relative.startswith('scripts/') and relative!='experiments/e2_resource_continuation_20261112/resource_policy.json':
            continue
        target=root/relative
        if (not target.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (target,*target.parents) if p.is_relative_to(root))
                or hashlib.sha256(target.read_bytes()).hexdigest()!=expected):
            raise ValueError('A published runtime source differs: '+relative)
    return record


def funded_state_class(base_class, budget_constructor):
    """Flush known idle costs before publishing a GitHub control snapshot."""
    class FundedState(base_class):
        _live_funding_budget = None

        def persist_result(self, directory, receipt):
            # restore() already verified every remote manifest and artifact;
            # the frozen E loop checks their bytes again before this callback.
            # Only the exact sealed result may skip a duplicate REST backup.
            if getattr(self, "_restored", False) is True and not getattr(self, "_pending", None):
                try:
                    path = Path(directory).absolute()
                    relative = path.relative_to(self.output).as_posix()
                    previous = self._backups.get(relative)
                    bound = self._files.get(relative + "/result.json")
                    target = path / "result.json"
                    effective = receipt.get("result", receipt)
                    if previous and bound and not target.is_symlink():
                        raw = target.read_bytes()
                        if (len(raw) == bound.get("bytes")
                                and hashlib.sha256(raw).hexdigest() == bound.get("sha256")
                                and json.loads(raw) == effective):
                            return dict(previous)
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
            return super().persist_result(directory, receipt)

        def snapshot(self, reason, **kwargs):
            live = self._live_funding_budget
            if (live is not None and not getattr(live, "_closed", False)
                    and not live.report().get("reserved_seconds", 0)):
                # charge_overhead(0) flushes the measured pending interval;
                # it does not grant credit or charge an active worker twice.
                live.charge_overhead(0, description="github_control_idle_snapshot")
            return super().snapshot(reason, **kwargs)

        @contextmanager
        def budget_factory(self, reconciliation, **kwargs):
            if "budget_class" in kwargs:
                raise ValueError("The registered funding adapter owns its budget class")
            def construct(*args, **options):
                live = budget_constructor(*args, **options)
                self._live_funding_budget = live
                return live
            with super().budget_factory(reconciliation, budget_class=construct, **kwargs) as budget:
                yield budget
    return FundedState


def resource_worker_router(supervisor, engine, extension, extension_commit):
    """Route only registered E entrypoints to the prospective resource wrapper."""
    engine=Path(engine).resolve();extension=Path(extension).resolve()
    modes={engine/'scripts/run_e2_reduced_selection.py':'selection',
           engine/'scripts/e2_reduced_final.py':'final'}
    def dispatch(command,*args,**kwargs):
        command=list(command)
        if (len(command)!=5 or str(command[1])!='-u' or str(command[3])!='--worker-spec'
                or Path(command[2]).resolve() not in modes):
            raise ValueError('Unexpected scientific child invocation; nothing was launched')
        mode=modes[Path(command[2]).resolve()]
        routed=[command[0],'-u',extension/'scripts/e2_resource_worker.py',
            '--mode',mode,'--engine-root',engine,'--extension-root',extension,
            '--extension-commit',extension_commit,'--worker-spec',command[4]]
        return supervisor(routed,*args,**kwargs)
    return dispatch


def main(argv=None):
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--engine-root", type=Path, required=True)
    gate.add_argument("--extension-root", type=Path, required=True)
    gate.add_argument("--extension-commit", required=True)
    gate.add_argument("--resource-extension-root", type=Path, required=True)
    extension_args, remainder = gate.parse_known_args(argv)
    engine = _engine_root(extension_args.engine_root)
    extension, registration = _extension_root(extension_args.extension_root, extension_args.extension_commit)
    resource_extension=extension_args.resource_extension_root.absolute()
    sys.path.insert(0, str(resource_extension / "scripts"))
    sys.path.insert(0, str(extension / "scripts"))
    sys.path.insert(0, str(engine / "scripts"))
    import run_e2_reduced_delivery as scientific
    from e2_reduced_protocol import verify_reduced_registration
    verify_reduced_registration(engine, require_git=True)
    args = scientific.parser().parse_args(remainder)
    if not args.execute:
        print(json.dumps({"version": VERSION, "stage": args.stage,
            "persistent_destination": "github", "training_started": False,
            "funding_requested": registration.get("funding_policy"),
            "funding_applied": False}), flush=True)
        return 0
    required = ("output", "data_root", "source_roots", "runtime_profile", "budget_evidence",
                "migration", "cache_audit", "reconciliation_root", "migration_registration_commit")
    if any(getattr(args, name, None) is None for name in required):
        raise ValueError("Execution requires every registered source/data/cache/budget path")
    if args.migration_registration_commit != ENGINE_COMMIT:
        raise ValueError("Retain the original migration registration and prior costs")
    from e2_resilient_github_backup import ResilientGitHubBackup
    from e2_reduced_github_state import GitHubState
    from e2_registered_budget_extension import make_funded_budget_class
    from e2_registered_resource_extension import make_resource_budget_class
    from reconcile_e2_github_completed import reconcile_completed_local
    from e2_expired_lease_recovery import reconcile_expired_lease
    from e2_a100_cache_selection import prepare_historical_cache_audit
    try:
        # The constructor factory validates the prior explicit authorization;
        # a draft registration never reaches GitHub writes or a GPU worker.
        for registered in (FUNDING_COMMIT, RESOURCE_COMMIT, extension_args.extension_commit):
            subprocess.run(["git", "-C", str(engine), "fetch", "--filter=blob:none", "--depth", "1",
                "origin", registered], check=True, capture_output=True, text=True)
        registration=verify_published_registration(engine,extension,extension_args.extension_commit,REGISTRATION_PATH)
        prior=verify_published_registration(engine,resource_extension,RESOURCE_COMMIT,RESOURCE_REGISTRATION_PATH)
        if registration.get('funding_policy')!=prior.get('funding_policy') or registration.get('resource_policy')!=prior.get('resource_policy'):
            raise ValueError('Keep the unchanged H funding and P resource policies')
        runtime=scientific.read_json(args.runtime_profile)
        if 'A100' not in runtime.get('runtime',{}).get('device_name',''):
            raise ValueError('The actual registered current runtime is not A100')
        funded = make_funded_budget_class(prior.get("funding_policy"),
            funding_registration_commit=FUNDING_COMMIT, funding_git_root=engine)
        constructor = make_resource_budget_class(funded, prior.get("resource_policy"),
            registration_commit=RESOURCE_COMMIT, registration_root=engine,
            extension_root=resource_extension)
        State = funded_state_class(GitHubState, constructor)
        client = ResilientGitHubBackup()
        client.preflight()
        print('Recuperando el intento T4 pendiente con cargo conservador y sin repetirlo.',flush=True)
        recovery=reconcile_expired_lease(args.output,client,
            recovery_policy=registration.get('expired_lease_policy'),
            registration_commit=extension_args.extension_commit,registration_root=engine)
        print(json.dumps({'event':'expired_lease_reconciliation','result':recovery},ensure_ascii=False),flush=True)
        reconcile_completed_local(args.output, client)
        state = State(args.output, client)
        print('Restaurando desde GitHub los artefactos verificados; las descargas pueden tardar varios minutos.',flush=True)
        state.restore()
        with state.budget_factory(scientific.read_json(args.budget_evidence),
                migration=scientific.read_json(args.migration),
                migration_registration_commit=ENGINE_COMMIT,
                inherited_ledgers=args.inherited_ledger) as budget:
            prepared=prepare_historical_cache_audit(args,budget,
                registration_commit=extension_args.extension_commit,
                transition_directory=Path(args.output)/'hardware_transition')
            state.snapshot("hardware_transition_historical_cache_admitted")
            print(json.dumps({"event":"resource_policy_applied",
                "phase_wall_seconds":budget.stage_wall_seconds,
                "resource_registration_commit":RESOURCE_COMMIT,
                "hardware_transition_registration_commit":extension_args.extension_commit,
                "funding_registration_commit":FUNDING_COMMIT,
                "new_budget_credit":False}),flush=True)
            import run_e2_reduced_selection as selection
            from e2_a100_cache_selection import run_selection
            selection.supervise_command = resource_worker_router(selection.supervise_command,
                engine, resource_extension, RESOURCE_COMMIT)
            selection.run_selection = run_selection
            scientific.execute_delivery(args, budget, state)
        return 0
    except (KeyboardInterrupt, Exception) as error:
        print(json.dumps({"version": VERSION, "status": "paused_not_complete",
            "error_type": type(error).__name__, "reason": str(error),
            "persistent_destination": "github", "automatic_retry": False,
            "all_sixteen_systems_benchmark_complete": False}), file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
