"""Run deterministic independent review checks and bind their source hashes."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import unittest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipt(here):
    root = here.parent
    receipt = json.loads((here / "REVIEW_RECEIPT.json").read_text())
    mismatches = []
    for category in ["reviewed_production_files", "review_check_files"]:
        for name, expected in receipt[category].items():
            path = root / name
            if not path.is_file() or sha(path) != expected:
                mismatches.append(name)
    current_tests = {str(path.relative_to(root)) for path in here.glob("test_*.py")}
    recorded_tests = set(receipt["review_check_files"])
    mismatches.extend(sorted(current_tests ^ recorded_tests))
    if receipt.get("review_runner_sha256") != sha(here / "run_independent_review.py"):
        mismatches.append("independent_review/run_independent_review.py")
    valid = receipt["status"] == "PASS" and not mismatches
    return {"status": "REVIEW_CURRENT" if valid else "REVIEW_STALE_OR_FAILED", "source_mismatches": mismatches,
            "recorded_tests_run": receipt["tests_run"], "recorded_at_utc": receipt["reviewed_at_utc"]}


def preserve_previous_receipt(here):
    path = here / "REVIEW_RECEIPT.json"
    if not path.is_file():
        return
    receipt = json.loads(path.read_text())
    identity = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    history = here / "history"
    history.mkdir(exist_ok=True)
    for old, suffix in [(path, "receipt.json"), (here / "TEST_OUTPUT.txt", "tests.txt")]:
        if old.is_file():
            destination = history / (identity + "_" + suffix)
            if destination.exists() and destination.read_bytes() != old.read_bytes():
                raise ValueError("Historical review identity collision")
            destination.write_bytes(old.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Fail if source/test bytes no longer match the passing review")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parent
    if args.verify_only:
        verification = verify_receipt(here)
        print(json.dumps(verification))
        return 0 if verification["status"] == "REVIEW_CURRENT" else 2
    preserve_previous_receipt(here)
    cpu_start = time.process_time()
    child_start = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(here), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    log = output.getvalue()
    (here / "TEST_OUTPUT.txt").write_text(log)
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {
        "schema_version": 1,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Independent deterministic functional, native arithmetic, API and decision-contract checks; not an outer Monte Carlo validation",
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "wall_seconds": time.perf_counter() - start,
        "parent_cpu_seconds": time.process_time() - cpu_start,
        "child_cpu_seconds": child_end.ru_utime + child_end.ru_stime - child_start.ru_utime - child_start.ru_stime,
        "python_version": sys.version,
        "reviewed_production_files": {str(path.relative_to(root)): sha(path) for path in [root / "methods/primary_rmse.py", root / "methods/primary_bootstrap.cpp"]},
        "review_check_files": {str(path.relative_to(root)): sha(path) for path in sorted(here.glob("test_*.py"))},
        "review_runner_sha256": sha(here / "run_independent_review.py"),
        "test_output_sha256": sha(here / "TEST_OUTPUT.txt"),
        "initial_blocking_finding": "GMM nuisance temporal derivative omitted division by T; reported before calibration and corrected by methods author; original finding is preserved",
        "current_mathematical_judgement": "The reviewed smooth-functional and nuisance derivatives match independently calculated pathwise derivatives; bootstrap reestimates roots and HAC; exact duplicate compression preserves outputs and all 120 Holm slots; the decision enforces the full ordered family and preserves the per-rival two-of-three policy",
        "limits": [
            "Weak dependence, joint temporal FCLT, fixed finite seed cohort and strictly positive seed risks remain assumptions",
            "GW conditional-functional vector null is stronger than scalar mean-RMSE equality; it is not a scalar directional p-value",
            "No finite-sample or universal market error control is established by these deterministic checks",
            "No grading decision, academic R1/R2 acceptance or R3 accreditation is made",
        ],
        "new_calibration_simulations": 0,
        "market_fits": 0,
        "market_forwards": 0,
        "reserved_test_opened": False,
        "AI_assistance_declared": True,
    }
    (here / "REVIEW_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(log, end="")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
