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
VERSION = "e2-reduced-github-funding-controller-v1"


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
    required = ("scripts/run_e2_reduced_github72_delivery.py", "scripts/e2_registered_budget_extension.py",
                "scripts/e2_reduced_github_state.py", "scripts/e2_reduced_github_restore.py",
                "scripts/reconcile_e2_github_completed.py")
    for relative in required:
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve()) or target.is_symlink()
                or hashlib.sha256(target.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A registered continuation source differs: " + relative)
    return root, record


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


def main(argv=None):
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--engine-root", type=Path, required=True)
    gate.add_argument("--extension-root", type=Path, required=True)
    gate.add_argument("--extension-commit", required=True)
    extension_args, remainder = gate.parse_known_args(argv)
    engine = _engine_root(extension_args.engine_root)
    extension, registration = _extension_root(extension_args.extension_root, extension_args.extension_commit)
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
    from e2_reduced_github_backup import GitHubBackup
    from e2_reduced_github_state import GitHubState
    from e2_registered_budget_extension import make_funded_budget_class
    from reconcile_e2_github_completed import reconcile_completed_local
    try:
        # The constructor factory validates the prior explicit authorization;
        # a draft registration never reaches GitHub writes or a GPU worker.
        subprocess.run(["git", "-C", str(engine), "fetch", "--filter=blob:none", "--depth", "1",
            "origin", extension_args.extension_commit], check=True, capture_output=True, text=True)
        constructor = make_funded_budget_class(registration.get("funding_policy"),
            funding_registration_commit=extension_args.extension_commit, funding_git_root=engine)
        State = funded_state_class(GitHubState, constructor)
        client = GitHubBackup()
        client.preflight()
        reconcile_completed_local(args.output, client)
        state = State(args.output, client)
        state.restore()
        with state.budget_factory(scientific.read_json(args.budget_evidence),
                migration=scientific.read_json(args.migration),
                migration_registration_commit=ENGINE_COMMIT,
                inherited_ledgers=args.inherited_ledger) as budget:
            state.snapshot("registered_funding_continuation")
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
