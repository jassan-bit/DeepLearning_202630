"""Continue registered E2 final stages under a separately confirmed budget cap.

The E/H/P/Q sources and prior funding anchors stay unchanged. This new parent
adapter only adds confirmed funding and selects existing final-stage commands.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
PRIOR_COMMIT = "72c04635b7a746d06066273ce61e891975feb510"
FUNDING_COMMIT = "f540227c23c572b9ca7fb36920588727b029484a"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
PRIOR_DRIVER_SHA256 = "42ae803ea26b8d7c3558e985e360d7693e8c3348ece03d6aafaa644713e85c23"
REGISTRATION_PATH = "experiments/e2_budget_continuation_20261113/registration.json"
PROPOSAL_PATH = "experiments/e2_budget_continuation_20261113/proposal.json"
FINAL_STAGES = ("final-fit", "pilot", "classical", "freeze", "score")
FILES = ("notebooks/run_e2_final_continuation_colab.py",
    "scripts/run_e2_final_continuation_delivery.py",
    "scripts/e2_confirmed_budget_continuation.py", PROPOSAL_PATH)


def _argument(argv, option):
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        raise ValueError("Missing original controller argument: " + option) from None


def _verify_extension(root, commit, engine):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("An immutable continuation commit is required")
    root = Path(root).absolute()
    raw = (root / "registration.json").read_bytes()
    published = subprocess.check_output(["git", "-C", str(engine), "show", commit + ":" + REGISTRATION_PATH])
    if raw != published:
        raise ValueError("The continuation registration differs from its published commit")
    record = json.loads(raw)
    expected = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "prior_controller_commit": PRIOR_COMMIT, "prior_funding_commit": FUNDING_COMMIT,
        "resource_commit": RESOURCE_COMMIT}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Retain the original scientific, resource and funding registrations")
    for relative in FILES:
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A continuation runtime source differs: " + relative)
    if record.get("proposal_sha256") != hashlib.sha256((root / PROPOSAL_PATH).read_bytes()).hexdigest():
        raise ValueError("The registered budget proposal differs")
    return record, json.loads((root / PROPOSAL_PATH).read_text())


def final_stage_adapter(execute_delivery):
    """Use unchanged E stage commands, preserving the already closed HPO files."""
    def run(args, budget, state):
        if args.stage != "all" or getattr(args, "models", None):
            raise ValueError("This continuation owns the full registered final cohort")
        closure = json.loads((Path(args.output) / "selection_complete.json").read_text())
        from e2_reduced_protocol import NEURAL
        expected = set(NEURAL)
        if (closure.get("status") != "completed"
                or set(closure.get("completed_models", [])) != expected
                or set(closure.get("required_models", [])) != expected
                or set(closure.get("selections", {})) != expected):
            raise ValueError("All thirteen audited selections must already be closed")
        for model, selected in closure["selections"].items():
            if (selected.get("model_id") != model
                    or selected.get("all_eighteen_slots_accounted_for") is not True
                    or selected.get("validation_count") != 249
                    or not selected.get("winner_configuration_id")):
                raise ValueError("A closed winning recipe is incomplete")
        result = None
        for stage in FINAL_STAGES:
            stage_args = copy(args)
            stage_args.stage = stage
            result = execute_delivery(stage_args, budget, state)
        return result
    return run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--continuation-commit", required=True)
    parser.add_argument("--approval-receipt", required=True)
    continuation, remainder = parser.parse_known_args(argv)
    engine = Path(_argument(remainder, "--engine-root")).absolute()
    prior_root = Path(_argument(remainder, "--extension-root")).absolute()
    if _argument(remainder, "--extension-commit") != PRIOR_COMMIT:
        raise ValueError("The original A100 controller must retain its published Q commit")
    if subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != ENGINE_COMMIT:
        raise ValueError("The scientific engine E differs")
    subprocess.run(["git", "-C", str(engine), "fetch", "--filter=blob:none", "--depth", "1",
        "origin", continuation.continuation_commit], check=True, capture_output=True, text=True)
    _, proposal = _verify_extension(continuation.continuation_root, continuation.continuation_commit, engine)
    prior_driver = prior_root / "scripts/run_e2_a100_recovery_delivery.py"
    if hashlib.sha256(prior_driver.read_bytes()).hexdigest() != PRIOR_DRIVER_SHA256:
        raise ValueError("The published A100 parent controller differs")
    spec = importlib.util.spec_from_file_location("e2_pinned_q_final_continuation", prior_driver)
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    sys.path.insert(0, str(continuation.continuation_root / "scripts"))
    sys.path.insert(0, str(engine / "scripts"))
    from e2_confirmed_budget_continuation import make_continued_budget_class
    import run_e2_reduced_delivery as scientific
    original_state = prior.funded_state_class
    original_execute = scientific.execute_delivery
    approval = json.loads(continuation.approval_receipt)
    def state_class(base_class, constructor):
        continued = make_continued_budget_class(constructor, proposal,
            registration_commit=continuation.continuation_commit,
            registration_git_root=engine, approval_receipt=approval)
        return original_state(base_class, continued)
    prior.funded_state_class = state_class
    scientific.execute_delivery = final_stage_adapter(original_execute)
    try:
        return prior.main(remainder)
    finally:
        scientific.execute_delivery = original_execute
        prior.funded_state_class = original_state


if __name__ == "__main__":
    raise SystemExit(main())
