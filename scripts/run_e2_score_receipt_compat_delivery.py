"""Adapt one metadata backup receipt without changing the registered score run."""
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
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
FUNDING_CONTINUATION_COMMIT = "41da88cf600aadc0319214465c94a85335e2b1b7"
PRIOR_RESUME_COMMIT = "caf650b67a68226d350ccd6a1f96b10940e0c6a9"
PRIOR_ARIMA_COMMIT = "f12cc0904efefde6854342ef18a6c447d5572b3e"
PRIOR_SCORE_SOURCE_COMMIT = "24344bd6db20f0546b26167eda4c3076a38c0f4b"
PRIOR_DRIVER_SHA256 = "2a2f795c4e907f98ab16b6b97909e0313eed514d2eae18ce59f46bc9ed5a5a04"
REGISTRATION_PATH = "experiments/e2_score_receipt_compat_20261113/registration.json"
FILES = ("notebooks/run_e2_score_receipt_compat_colab.py",
    "scripts/run_e2_score_receipt_compat_delivery.py", "scripts/e2_score_receipt_compat.py")


def _argument(argv, option):
    try:
        return argv[argv.index(option) + 1]
    except (ValueError, IndexError):
        raise ValueError("Missing preserved controller argument: " + option) from None


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify(root, commit, engine):
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("An immutable receipt compatibility registration is required")
    root = Path(root).absolute()
    raw = (root / "registration.json").read_bytes()
    if raw != subprocess.check_output(["git", "-C", str(engine), "show", commit + ":" + REGISTRATION_PATH]):
        raise ValueError("The receipt registration differs from its published commit")
    record = json.loads(raw)
    expected = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "resource_commit": RESOURCE_COMMIT, "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT,
        "prior_resume_commit": PRIOR_RESUME_COMMIT, "prior_arima_commit": PRIOR_ARIMA_COMMIT,
        "prior_score_source_commit": PRIOR_SCORE_SOURCE_COMMIT}
    if any(record.get(k) != v for k,v in expected.items()):
        raise ValueError("Keep every original scientific, source and funding authority")
    for relative in FILES:
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A registered receipt adapter source differs: " + relative)
    return root


def receipt_scoring_factory(factory, compatible_persist_result, registration_commit):
    """Keep the original factory/proof and adapt only the backup callback."""
    def make(*args, **kwargs):
        scoring = factory(*args, **kwargs)
        def run(*score_args, **score_kwargs):
            callback = score_kwargs.get("persist_result")
            if callback is not None:
                score_kwargs["persist_result"] = compatible_persist_result(
                    callback, registration_commit=registration_commit)
            return scoring(*score_args, **score_kwargs)
        return run
    return make


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--receipt-compat-root", type=Path, required=True)
    parser.add_argument("--receipt-compat-commit", required=True)
    own, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    engine = Path(_argument(remainder, "--engine-root")).absolute()
    source_root = Path(_argument(remainder, "--score-source-root")).absolute()
    if (_argument(remainder, "--score-source-commit") != PRIOR_SCORE_SOURCE_COMMIT
            or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != ENGINE_COMMIT):
        raise ValueError("Keep the original scoring source and scientific engine")
    for commit in (own.receipt_compat_commit, PRIOR_SCORE_SOURCE_COMMIT):
        subprocess.run(["git", "-C", str(engine), "fetch", "--filter=blob:none", "--depth", "1",
            "origin", commit], check=True, capture_output=True, text=True)
    root = _verify(own.receipt_compat_root, own.receipt_compat_commit, engine)
    driver = source_root / "scripts/run_e2_score_source_recovery_delivery.py"
    if (driver.is_symlink() or not driver.resolve().is_relative_to(source_root.resolve())
            or hashlib.sha256(driver.read_bytes()).hexdigest() != PRIOR_DRIVER_SHA256):
        raise ValueError("The original scoring parent source differs")
    prior = _load(driver, "e2_preserved_score_source_receipt_parent")
    prior._verify(source_root, PRIOR_SCORE_SOURCE_COMMIT, engine)
    helper = _load(source_root / "scripts/e2_score_source_recovery.py", "e2_preserved_score_source_receipt_helper")
    adapter = _load(root / "scripts/e2_score_receipt_compat.py", "e2_registered_score_receipt_adapter")
    original_factory = helper.make_run_final_scoring
    previous_module = sys.modules.get("e2_score_source_recovery")
    helper.make_run_final_scoring = receipt_scoring_factory(original_factory,
        adapter.compatible_persist_result, own.receipt_compat_commit)
    sys.modules["e2_score_source_recovery"] = helper
    try:
        print("Se corrige únicamente el recibo de respaldo; continúan la evaluación y los resultados congelados.", flush=True)
        return prior.main(remainder)
    finally:
        helper.make_run_final_scoring = original_factory
        if previous_module is None:
            sys.modules.pop("e2_score_source_recovery", None)
        else:
            sys.modules["e2_score_source_recovery"] = previous_module


if __name__ == "__main__":
    raise SystemExit(main())
