"""Continue comparator opportunities after a recorded diagnostic-contract failure."""
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
PRIOR_RESUME_COMMIT = "caf650b67a68226d350ccd6a1f96b10940e0c6a9"
PRIOR_CONTROLLER_COMMIT = "72c04635b7a746d06066273ce61e891975feb510"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
REGISTRATION_PATH = "experiments/e2_arima_compat_20261113/registration.json"
POLICY_PATH = "experiments/e2_arima_compat_20261113/policy.json"
CAF_REGISTRATION_PATH = "experiments/e2_final80_resume_fix_20261113/registration.json"
F80_DRIVER_SHA256 = "75c0941e9f28ccb3ced98c1ca95865f45cd56d7a4133a11c92b7b10c7da62c10"
FINAL_STAGES = ("classical", "freeze", "score")
FILES = ("notebooks/run_e2_arima_resume_colab.py", "scripts/run_e2_arima_resume_delivery.py",
    "scripts/e2_arima_prediction_compat.py", "scripts/e2_arima_compat_worker.py",
    "scripts/e2_arima_classical_continuation.py", POLICY_PATH)


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
        raise ValueError("An immutable ARIMA repair registration is required")
    root = Path(root).absolute()
    raw = (root / "registration.json").read_bytes()
    published = subprocess.check_output(["git", "-C", str(engine), "show", commit + ":" + REGISTRATION_PATH])
    if raw != published:
        raise ValueError("The ARIMA repair registration differs from its published commit")
    record = json.loads(raw)
    expected = {"engine_commit": ENGINE_COMMIT, "protocol_id": "e2-reduced-delivery-v1",
        "funding_continuation_commit": FUNDING_CONTINUATION_COMMIT,
        "prior_resume_commit": PRIOR_RESUME_COMMIT, "resource_commit": RESOURCE_COMMIT}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("Keep the existing engine, resource and 80-hour authorities")
    for relative in FILES:
        path = root / relative
        if (not path.resolve().is_relative_to(root.resolve())
                or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
                or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("files_sha256", {}).get(relative)):
            raise ValueError("A registered ARIMA runtime source differs: " + relative)
    return root, json.loads((root / POLICY_PATH).read_bytes())


def completed_neural_guard(output):
    from e2_reduced_protocol import NEURAL
    from e2_reduced_final import FINAL_SEEDS, PILOT_SEEDS, slug
    output = Path(output) / "final"
    for model in NEURAL:
        for seed in FINAL_SEEDS:
            result = json.loads((output / "final-fit" / slug(model) / str(seed) / "result.json").read_bytes())
            if result.get("status") != "completed" or result.get("model") != model or result.get("seed") != seed:
                raise ValueError("All 78 existing neural final refits must remain completed")
    for model in ("iTransformer", "GRU32"):
        for seed in PILOT_SEEDS:
            result = json.loads((output / "pilot" / slug(model) / str(seed) / "result.json").read_bytes())
            if result.get("status") != "completed" or result.get("model") != model or result.get("seed") != seed:
                raise ValueError("All ten existing development pilots must remain completed")
    report = json.loads((output / "pilot_power_MDE.json").read_bytes())
    if report.get("status") != "development_pilot_completed_power_reported":
        raise ValueError("The original completed power/MDE report is required")


def repair_supervisor(supervisor, engine, resource_root, repair_root, repair_commit):
    engine, resource_root, repair_root = map(lambda p: Path(p).resolve(), (engine, resource_root, repair_root))
    def dispatch(command, *args, **kwargs):
        command = list(command)
        if (len(command) != 5 or str(command[1]) != "-u" or str(command[3]) != "--worker-spec"
                or Path(command[2]).resolve() != engine / "scripts/e2_reduced_final.py"):
            raise ValueError("Unexpected scientific child request; no worker was admitted")
        spec = json.loads(Path(command[4]).read_bytes())
        kind = spec.get("kind")
        if kind not in ("classical", "final-score"):
            raise ValueError("Only comparator and final scoring workers belong to this continuation")
        common = ["--mode", "final", "--engine-root", engine,
            "--extension-root", resource_root, "--extension-commit", RESOURCE_COMMIT,
            "--worker-spec", command[4]]
        if kind == "final-score" or spec.get("model") == "ARIMA":
            routed = [command[0], "-u", repair_root / "scripts/e2_arima_compat_worker.py",
                "--arima-root", repair_root, "--arima-commit", repair_commit, *common]
        else:
            routed = [command[0], "-u", resource_root / "scripts/e2_resource_worker.py", *common]
        return supervisor(routed, *args, **kwargs)
    return dispatch


def guarded_final_adapter(adapter_factory):
    """Keep the frozen stage adapter, requiring all earlier neural work first."""
    def factory(execute):
        adapter = adapter_factory(execute)
        def run(args, budget, client):
            completed_neural_guard(args.output)
            return adapter(args, budget, client)
        return run
    return factory


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--arima-root", type=Path, required=True)
    parser.add_argument("--arima-commit", required=True)
    own, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    caf_parser = argparse.ArgumentParser(add_help=False)
    caf_parser.add_argument("--resume-root", type=Path, required=True)
    caf_parser.add_argument("--resume-commit", required=True)
    caf_args, remainder = caf_parser.parse_known_args(remainder)
    engine = Path(_argument(remainder, "--engine-root")).absolute()
    original = Path(_argument(remainder, "--continuation-root")).absolute()
    q_root = Path(_argument(remainder, "--extension-root")).absolute()
    resource = Path(_argument(remainder, "--resource-extension-root")).absolute()
    if (caf_args.resume_commit != PRIOR_RESUME_COMMIT
            or _argument(remainder, "--continuation-commit") != FUNDING_CONTINUATION_COMMIT
            or _argument(remainder, "--extension-commit") != PRIOR_CONTROLLER_COMMIT
            or subprocess.check_output(["git", "-C", str(engine), "rev-parse", "HEAD"], text=True).strip() != ENGINE_COMMIT):
        raise ValueError("The existing E/Q/Caf and funding authorities must remain unchanged")
    for commit in (own.arima_commit, PRIOR_RESUME_COMMIT):
        subprocess.run(["git", "-C", str(engine), "fetch", "--filter=blob:none", "--depth", "1",
            "origin", commit], check=True, capture_output=True, text=True)
    root, policy = _verify(own.arima_root, own.arima_commit, engine)
    caf_root = caf_args.resume_root.absolute()
    caf_raw = (caf_root / "registration.json").read_bytes()
    if caf_raw != subprocess.check_output(["git", "-C", str(engine), "show", PRIOR_RESUME_COMMIT + ":" + CAF_REGISTRATION_PATH]):
        raise ValueError("The original sealed-transition repair registration differs")
    caf_record = json.loads(caf_raw)
    for relative in ("scripts/run_e2_final80_resume_delivery.py", "scripts/e2_sealed_transition_resume.py", "scripts/e2_verified_local_restore.py"):
        target = caf_root / relative
        if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != caf_record.get("files_sha256", {}).get(relative):
            raise ValueError("The original sealed-transition runtime source differs")
    caf = _load(caf_root / "scripts/run_e2_final80_resume_delivery.py", "e2_preserved_caf_controller")
    caf._verify(caf_root, PRIOR_RESUME_COMMIT, engine)
    driver = original / "scripts/run_e2_final_continuation_delivery.py"
    if driver.is_symlink() or hashlib.sha256(driver.read_bytes()).hexdigest() != F80_DRIVER_SHA256:
        raise ValueError("The original 80-hour final controller differs")
    old = _load(driver, "e2_preserved_f80_classical_controller")
    old.FINAL_STAGES = FINAL_STAGES
    for directory in (resource / "scripts", q_root / "scripts", engine / "scripts", caf_root / "scripts", root / "scripts"):
        sys.path.insert(0, str(directory))
    import e2_a100_cache_selection as cache
    import e2_reduced_github_state as state
    import e2_reduced_final as final
    from e2_reduced_supervision import supervise_command
    from e2_arima_classical_continuation import make_run_classical
    from e2_sealed_transition_resume import prepare_sealed_historical_cache
    from e2_verified_local_restore import verified_local_restore
    previous = (cache.prepare_historical_cache_audit, state.GitHubState,
        final.run_classical, final.run_final_scoring, old.final_stage_adapter)
    supervisor = repair_supervisor(supervise_command, engine, resource, root, own.arima_commit)
    def sealed_prepare(*args, **kwargs):
        kwargs["resume_registration_commit"] = own.arima_commit
        report = prepare_sealed_historical_cache(*args, **kwargs)
        caf._persist_admission(args[0].output, own.arima_commit, report["resume_admission"])
        return report
    original_scoring = final.run_final_scoring
    def scoring(*args, **kwargs):
        if kwargs.get("supervisor") is not None:
            raise ValueError("The registered repair owns final scoring supervision")
        kwargs["supervisor"] = supervisor
        return original_scoring(*args, **kwargs)
    classical = make_run_classical(final, policy,
        registration_commit=own.arima_commit, registration_git_root=engine, supervisor=supervisor)
    cache.prepare_historical_cache_audit = sealed_prepare
    state.GitHubState = verified_local_restore(state.GitHubState)
    final.run_classical = classical
    final.run_final_scoring = scoring
    old.final_stage_adapter = guarded_final_adapter(old.final_stage_adapter)
    try:
        print("Se conservan los 78 ajustes neuronales y los 10 pilotos; continúa la etapa de comparadores.", flush=True)
        return old.main(remainder)
    finally:
        cache.prepare_historical_cache_audit, state.GitHubState, final.run_classical, final.run_final_scoring, old.final_stage_adapter = previous


if __name__ == "__main__":
    raise SystemExit(main())
