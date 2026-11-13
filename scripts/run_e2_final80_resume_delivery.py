"""Resume the immutable 80-hour controller using sealed transition artifacts."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
FUNDING_CONTINUATION_COMMIT = "41da88cf600aadc0319214465c94a85335e2b1b7"
PRIOR_CONTROLLER_COMMIT = "72c04635b7a746d06066273ce61e891975feb510"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
REGISTRATION_PATH = "experiments/e2_final80_resume_fix_20261113/registration.json"
OLD_DRIVER_SHA256 = "75c0941e9f28ccb3ced98c1ca95865f45cd56d7a4133a11c92b7b10c7da62c10"
FILES = ("notebooks/run_e2_final80_resume_colab.py", "scripts/run_e2_final80_resume_delivery.py",
    "scripts/e2_sealed_transition_resume.py", "scripts/e2_verified_local_restore.py")


def _argument(argv, option):
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        raise ValueError("Missing preserved controller argument: " + option) from None


def _verify(root, commit, engine):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the immutable resume repair commit")
    root = Path(root).absolute()
    raw = (root / "registration.json").read_bytes()
    published = subprocess.check_output(["git", "-C", str(engine), "show", commit + ":" + REGISTRATION_PATH])
    if raw != published:
        raise ValueError("The resume repair registration differs")
    record = json.loads(raw)
    expected = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT,
        "prior_controller_commit": PRIOR_CONTROLLER_COMMIT}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Keep the original engine and existing funding authority")
    for relative in FILES:
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A registered repair source differs: " + relative)
    return root


def _persist_admission(output, commit, admission):
    value = dict(admission)
    checksum = value.pop("sha256", None)
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()
    if (not isinstance(checksum, str) or hashlib.sha256(canonical).hexdigest() != checksum
            or value.get("resume_registration_commit") != commit):
        raise ValueError("A bound sealed-transition admission receipt is required")
    root = Path(output).absolute()
    target = root / "hardware_transition_resume" / commit / (checksum + ".json")
    if (not target.resolve().is_relative_to(root.resolve())
            or any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(root))):
        raise ValueError("A resume admission path must remain inside the control root")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(admission, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode() + b"\n"
    if target.exists() and target.read_bytes() != payload:
        raise ValueError("An earlier immutable resume admission differs")
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(payload)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--resume-commit", required=True)
    repair, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    engine = Path(_argument(remainder, "--engine-root")).absolute()
    original = Path(_argument(remainder, "--continuation-root")).absolute()
    q_root = Path(_argument(remainder, "--extension-root")).absolute()
    resource = Path(_argument(remainder, "--resource-extension-root")).absolute()
    if (_argument(remainder, "--continuation-commit") != FUNDING_CONTINUATION_COMMIT
            or _argument(remainder, "--extension-commit") != PRIOR_CONTROLLER_COMMIT
            or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != ENGINE_COMMIT):
        raise ValueError("The E/Q/80-hour funding authorities must remain unchanged")
    subprocess.run(["git", "-C", str(engine), "fetch", "--filter=blob:none", "--depth", "1",
        "origin", repair.resume_commit], check=True, capture_output=True, text=True)
    root = _verify(repair.resume_root, repair.resume_commit, engine)
    driver = original / "scripts/run_e2_final_continuation_delivery.py"
    if hashlib.sha256(driver.read_bytes()).hexdigest() != OLD_DRIVER_SHA256:
        raise ValueError("The original 80-hour controller source differs")
    spec = importlib.util.spec_from_file_location("e2_unchanged_80_hour_controller", driver)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    for directory in (resource / "scripts", q_root / "scripts", engine / "scripts", root / "scripts"):
        sys.path.insert(0, str(directory))
    import e2_a100_cache_selection as cache
    import e2_reduced_github_state as state
    from e2_sealed_transition_resume import prepare_sealed_historical_cache
    from e2_verified_local_restore import verified_local_restore
    original_prepare = cache.prepare_historical_cache_audit
    original_state = state.GitHubState
    def sealed_prepare(*args, **kwargs):
        kwargs["resume_registration_commit"] = repair.resume_commit
        report = prepare_sealed_historical_cache(*args, **kwargs)
        path = _persist_admission(args[0].output, repair.resume_commit, report["resume_admission"])
        print(json.dumps({"event": "sealed_hardware_transition_reused",
            "resume_admission": str(path), "original_transition_unchanged": True,
            "new_budget_authorization": False}), flush=True)
        return report
    cache.prepare_historical_cache_audit = sealed_prepare
    state.GitHubState = verified_local_restore(original_state)
    try:
        return old.main(remainder)
    finally:
        cache.prepare_historical_cache_audit = original_prepare
        state.GitHubState = original_state


if __name__ == "__main__":
    raise SystemExit(main())
