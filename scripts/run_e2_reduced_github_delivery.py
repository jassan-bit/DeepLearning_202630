"""GitHub-primary persistence adapter for the unchanged reduced scientific engine.

This module changes transport/durability only. E's worker code, selection rules,
seeds, data admission and sixty-four-hour cap remain in its separate checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
VERSION = "e2-reduced-github-primary-controller-v1"


def _engine_root(path):
    root = Path(path).absolute()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != ENGINE_COMMIT:
        raise ValueError("The scientific engine must remain at its registered E commit")
    return root


def _extension_root(path, commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the immutable extension commit")
    root = Path(path).absolute()
    record = json.loads((root / "registration.json").read_text())
    if record.get("engine_commit") != ENGINE_COMMIT or record.get("protocol_id") != "e2-reduced-delivery-v1":
        raise ValueError("The transport extension is bound to another protocol or engine")
    required = ("scripts/run_e2_reduced_github_delivery.py", "scripts/e2_reduced_github_state.py",
                "scripts/e2_reduced_github_restore.py")
    for relative in required:
        target = root / relative
        if (not target.resolve().is_relative_to(root.resolve()) or target.is_symlink()
                or hashlib.sha256(target.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A registered transport source differs: " + relative)
    return root


def main(argv=None):
    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument("--engine-root", type=Path, required=True)
    gate.add_argument("--extension-root", type=Path, required=True)
    gate.add_argument("--extension-commit", required=True)
    extension_args, remainder = gate.parse_known_args(argv)
    engine = _engine_root(extension_args.engine_root)
    extension = _extension_root(extension_args.extension_root, extension_args.extension_commit)
    # E modules resolve from E only. New transport modules have distinct names.
    sys.path.insert(0, str(extension / "scripts"))
    sys.path.insert(0, str(engine / "scripts"))
    import run_e2_reduced_delivery as scientific
    from e2_reduced_protocol import verify_reduced_registration
    verify_reduced_registration(engine, require_git=True)
    parser = scientific.parser()
    args = parser.parse_args(remainder)
    if not args.execute:
        from e2_reduced_protocol import SYSTEMS
        print(json.dumps({"version": VERSION, "stage": args.stage,
            "systems": list(SYSTEMS), "persistent_destination": "github",
            "training_started": False, "budget_extended": False}), flush=True)
        return 0
    required = ("output", "data_root", "source_roots", "runtime_profile", "budget_evidence",
                "migration", "cache_audit", "reconciliation_root", "migration_registration_commit")
    if any(getattr(args, name, None) is None for name in required):
        parser.error("Execution requires every registered source/runtime/data/cache/budget path")
    if args.migration_registration_commit != ENGINE_COMMIT:
        raise ValueError("The original registered budget migration commit must be retained")
    from e2_reduced_github_backup import GitHubBackup
    from e2_reduced_github_state import GitHubState
    try:
        client = GitHubBackup()
        client.preflight()
        state = GitHubState(args.output, client)
        state.restore()
        with state.budget_factory(scientific.read_json(args.budget_evidence),
                migration=scientific.read_json(args.migration),
                migration_registration_commit=ENGINE_COMMIT,
                inherited_ledgers=args.inherited_ledger) as budget:
            state.snapshot("registered_github_primary_launch")
            scientific.execute_delivery(args, budget, state)
        return 0
    except (KeyboardInterrupt, Exception) as error:
        print(json.dumps({"version": VERSION, "status": "paused_not_complete",
            "error_type": type(error).__name__, "reason": str(error),
            "persistent_destination": "github", "automatic_retry": False,
            "budget_extended": False, "all_sixteen_systems_benchmark_complete": False}),
            file=sys.stderr, flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
