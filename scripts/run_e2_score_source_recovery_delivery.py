"""Continue only scoring, preserving frozen predictors and the failed attempt."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
FUNDING_CONTINUATION_COMMIT = "41da88cf600aadc0319214465c94a85335e2b1b7"
PRIOR_RESUME_COMMIT = "caf650b67a68226d350ccd6a1f96b10940e0c6a9"
PRIOR_ARIMA_COMMIT = "f12cc0904efefde6854342ef18a6c447d5572b3e"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
PRIOR_DRIVER_SHA256 = "92e8a85720b53abf41bbae40e9440558828848749d7cd92648e51d5675392e1f"
REGISTRATION_PATH = "experiments/e2_score_source_recovery_20261113/registration.json"
POLICY_PATH = "experiments/e2_score_source_recovery_20261113/policy.json"
FILES = ("notebooks/run_e2_score_source_recovery_colab.py",
    "scripts/run_e2_score_source_recovery_delivery.py",
    "scripts/e2_score_source_recovery.py", "scripts/e2_score_archive_transport.py",
    "scripts/e2_score_source_worker.py", POLICY_PATH)
FINAL_STAGES = ("score",)


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
        raise ValueError("An immutable score-source recovery registration is required")
    root = Path(root).absolute()
    raw = (root / "registration.json").read_bytes()
    published = subprocess.check_output(["git", "-C", str(engine), "show", commit + ":" + REGISTRATION_PATH])
    if raw != published:
        raise ValueError("The score-source registration differs from its published commit")
    record = json.loads(raw)
    expected = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT,
        "prior_resume_commit": PRIOR_RESUME_COMMIT, "prior_arima_commit": PRIOR_ARIMA_COMMIT,
        "resource_commit": RESOURCE_COMMIT}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Keep the existing scientific, resource and 80-hour authorities")
    for relative in FILES:
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A registered scoring source differs: " + relative)
    return root


def score_supervisor_factory(prior_factory, recovery_root, recovery_commit):
    """Preserve the old E/ARIMA/P router and replace its scoring child only."""
    recovery_root = Path(recovery_root).resolve()
    def factory(supervisor, engine, resource_root, arima_root, arima_commit):
        if arima_commit != PRIOR_ARIMA_COMMIT:
            raise ValueError("The original ARIMA authority must remain unchanged")
        arima_root = Path(arima_root).resolve()
        def redirect(command, *args, **kwargs):
            command = list(command)
            if (len(command) < 7 or command[1] != "-u"
                    or Path(command[2]).resolve() != arima_root / "scripts/e2_arima_compat_worker.py"
                    or _argument(command, "--arima-commit") != PRIOR_ARIMA_COMMIT
                    or Path(_argument(command, "--arima-root")).resolve() != arima_root
                    or json.loads(Path(_argument(command, "--worker-spec")).read_bytes()).get("kind") != "final-score"):
                raise ValueError("This continuation admits only the original frozen scoring child")
            command[2] = recovery_root / "scripts/e2_score_source_worker.py"
            command.extend(["--score-source-root", recovery_root, "--score-source-commit", recovery_commit])
            return supervisor(command, *args, **kwargs)
        return prior_factory(redirect, engine, resource_root, arima_root, arima_commit)
    return factory


@contextmanager
def score_only_parent(prior, final, scoring, recovery_root, recovery_commit):
    previous = prior.FINAL_STAGES, prior.repair_supervisor, final.run_final_scoring
    prior.FINAL_STAGES = FINAL_STAGES
    prior.repair_supervisor = score_supervisor_factory(previous[1], recovery_root, recovery_commit)
    final.run_final_scoring = scoring
    try:
        yield
    finally:
        prior.FINAL_STAGES, prior.repair_supervisor, final.run_final_scoring = previous


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--score-source-root", type=Path, required=True)
    parser.add_argument("--score-source-commit", required=True)
    own, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    engine = Path(_argument(remainder, "--engine-root")).absolute()
    arima_root = Path(_argument(remainder, "--arima-root")).absolute()
    if (_argument(remainder, "--arima-commit") != PRIOR_ARIMA_COMMIT
            or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != ENGINE_COMMIT):
        raise ValueError("The existing E and ARIMA authorities must remain unchanged")
    for commit in (own.score_source_commit, PRIOR_ARIMA_COMMIT):
        subprocess.run(["git", "-C", str(engine), "fetch", "--filter=blob:none", "--depth", "1",
            "origin", commit], check=True, capture_output=True, text=True)
    root = _verify(own.score_source_root, own.score_source_commit, engine)
    driver = arima_root / "scripts/run_e2_arima_resume_delivery.py"
    if (driver.is_symlink() or not driver.resolve().is_relative_to(arima_root.resolve())
            or hashlib.sha256(driver.read_bytes()).hexdigest() != PRIOR_DRIVER_SHA256):
        raise ValueError("The published ARIMA parent source differs")
    prior = _load(driver, "e2_preserved_arima_score_source_parent")
    prior._verify(arima_root, PRIOR_ARIMA_COMMIT, engine)
    for directory in (engine / "scripts", root / "scripts"):
        sys.path.insert(0, str(directory))
    import e2_reduced_final as final
    from e2_score_source_recovery import make_run_final_scoring
    scoring = make_run_final_scoring(final, registration_commit=own.score_source_commit,
        registration_root=root, engine_root=engine)
    with score_only_parent(prior, final, scoring, root, own.score_source_commit):
        print("Los 81 predictores permanecen congelados; continúa únicamente la evaluación final.", flush=True)
        return prior.main(remainder)


if __name__ == "__main__":
    raise SystemExit(main())
