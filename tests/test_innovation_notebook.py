"""Notebook transport, credential isolation and pinned-runtime checks; no fits."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/entregable3_innovacion_v2_colab.ipynb"


def _code_and_constants():
    notebook = json.loads(NOTEBOOK.read_bytes())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    tree = ast.parse(code)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return notebook, tree, code, constants


def test_notebook_has_empty_outputs_valid_source_pin_and_exact_dependency_hashes():
    notebook, _, code, values = _code_and_constants()
    assert notebook["nbformat"] == 4
    assert re.fullmatch(r"[0-9a-f]{40}", values["SOURCE_COMMIT"])
    assert values["SOURCE_COMMIT"] != "0" * 40
    assert values["RESULTS_BRANCH"] == "e3-desarrollo-resultados-v2"
    assert "drive.mount" not in code and "from google.colab import drive" not in code
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            compile("".join(cell["source"]), f"v2-cell-{index}", "exec")
    hashes = values["SOURCE_FILE_SHA256"]
    for path, digest in hashes.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    assert {"src/e3_innovation_model.py", "src/e3_innovation.py", "scripts/e3_innovation_github.py",
            "scripts/e3_run_innovation.py", "requirements-e3-innovation-gpu.lock.txt"} <= set(hashes)
    original_path = ROOT / "experiments/e3_ficha_cnn_lstm_atencion_v2_20261116/pilot_plan.json"
    original = json.loads(original_path.read_bytes())
    operational = json.loads((ROOT / "experiments/e3_innovacion_v2_20261116/plan.json").read_bytes())
    assert operational["design_provenance"]["sha256"] == hashlib.sha256(original_path.read_bytes()).hexdigest()
    for key in ("architecture", "data", "distribution", "loss", "controls", "candidate",
                "training", "proposed_budget", "decision"):
        assert operational[key] == original[key], key


def _downloader(payload):
    _, tree, _, _ = _code_and_constants()
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "download_pinned")
    fake = SimpleNamespace(request=SimpleNamespace(urlopen=lambda *_a, **_k: io.BytesIO(payload)))
    namespace = dict(Path=Path, PurePosixPath=PurePosixPath, re=re, hashlib=hashlib, os=os,
                     urllib=fake, REPOSITORY="jassan-bit/DeepLearning_202630")
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook-download", "exec"), namespace)
    return namespace["download_pinned"]


def test_download_rejects_bad_bytes_and_preserves_existing_local_file(tmp_path):
    good = b"pinned synthetic source\n"
    digest = hashlib.sha256(good).hexdigest()
    target = tmp_path / "source.py"
    fetch = _downloader(b"wrong remote payload")
    with pytest.raises(RuntimeError):
        fetch("src/model.py", "a" * 40, target, digest)
    assert not target.exists()
    fetch = _downloader(good)
    assert fetch("src/model.py", "a" * 40, target, digest) == digest
    target.write_bytes(b"local edit to preserve")
    with pytest.raises(RuntimeError):
        fetch("src/model.py", "a" * 40, target, digest)
    assert target.read_bytes() == b"local edit to preserve"


@pytest.mark.parametrize("path,commit", [("../escape.py", "a" * 40), ("/absolute.py", "a" * 40),
                                         ("src/model.py", "main")])
def test_downloader_requires_canonical_path_and_immutable_commit(tmp_path, path, commit):
    with pytest.raises(ValueError):
        _downloader(b"x")(path, commit, tmp_path / "out", hashlib.sha256(b"x").hexdigest())


def test_v2_bootstrap_uses_private_module_and_removes_credentials(monkeypatch, tmp_path):
    path = ROOT / "notebooks/preparar_e3_innovacion_runtime.py"
    spec = importlib.util.spec_from_file_location("test_v2_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SERVICE_PAT", "synthetic-private-value")
    monkeypatch.setenv("CUSTOM_SECRET_KEY", "synthetic-other-value")
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-token-value")
    bootstrap = module._bootstrap_module()
    assert bootstrap.LOCK_NAME == "requirements-e3-innovation-gpu.lock.txt"
    environment = bootstrap._environment(tmp_path)
    assert not {"SERVICE_PAT", "CUSTOM_SECRET_KEY", "GITHUB_TOKEN"} & environment.keys()
    assert environment["PYTHONNOUSERSITE"] == "1"
    pins = bootstrap._expected_packages(ROOT / bootstrap.LOCK_NAME)
    assert pins["torch"] == "2.8.0+cu126" and pins["numpy"] == "2.3.5" and pins["scipy"] == "1.17.0"


def test_worker_cli_requires_remote_claim_before_importing_or_loading_data(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/e3_run_innovation.py"), "run",
                             "--npz", str(tmp_path / "never_opened.npz"), "--expected-sha256", "a" * 64,
                             "--run-dir", str(tmp_path / "never_created")], capture_output=True, text=True)
    assert result.returncode == 2 and "--global-claim" in result.stderr
    assert not (tmp_path / "never_created").exists()


def test_isolated_worker_cli_initializes_pauses_and_resumes_synthetic_fixture(tmp_path):
    """Exercise the actual subprocess/argument boundary used by Colab."""
    path = ROOT / "tests/test_innovation_training.py"
    spec = importlib.util.spec_from_file_location("e3_synthetic_cli_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = module.fixture.__wrapped__(tmp_path)
    run_dir = tmp_path / "isolated_cli_run"
    plan_path, claim_path, initial_path = [tmp_path / name for name in
                                          ("plan.json", "claim.json", "initial.json")]
    plan_path.write_text(json.dumps(fixture["plan"]))
    claim_path.write_text(json.dumps(fixture["claim"]))
    initial_path.write_text(json.dumps(module._ledger(fixture["claim"])))
    common = ["--npz", str(fixture["npz"]), "--expected-sha256", fixture["sha"],
              "--run-dir", str(run_dir), "--global-claim", str(claim_path), "--device", "cpu"]
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    def invoke(command, extra):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/e3_run_innovation.py"),
                                 command, *common, *extra], env=environment,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr

    invoke("preflight", ["--plan", str(plan_path), "--initial-ledger", str(initial_path),
                         "--frozen-v1-dir", str(fixture["base"]),
                         "--snapshot-manifest", str(fixture["base"] / "snapshot_manifest.json")])

    def acknowledged_synthetic_reservation():
        ledger_path = run_dir / "ledger.json"
        ledger = json.loads(ledger_path.read_bytes())
        assert ledger["active_session"] is None
        ledger["active_session"] = {"session_id": "synthetic_cli_ack",
                                     "last_wall_utc": datetime.now(timezone.utc).isoformat()}
        ledger_path.write_text(json.dumps(ledger))

    acknowledged_synthetic_reservation()
    invoke("run", ["--stop-after-epochs", "1"])
    assert json.loads((run_dir / "summary.json").read_bytes())["status"] == "paused"
    acknowledged_synthetic_reservation()
    invoke("resume", ["--stop-after-epochs", "12"])
    summary = json.loads((run_dir / "summary.json").read_bytes())
    assert summary["status"] == "completed" and summary["comparison_complete"]
    assert len(summary["models"]) == 6
    assert summary["prior_charged_seconds"] == 369.4779596389991
    assert summary["global_charged_seconds"] == summary["prior_charged_seconds"] + summary["charged_seconds"]
    assert not summary["statistical_tests_performed"]


def test_colab_coordinator_and_real_worker_complete_synthetic_campaign(tmp_path, monkeypatch):
    """Combine remote ACKs, the actual worker subprocess and synthetic OHLCV."""
    def test_module(filename, alias):
        spec = importlib.util.spec_from_file_location(alias, ROOT / "tests" / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    training = test_module("test_innovation_training.py", "e3_colab_synthetic_training")
    storage_test = test_module("test_innovation_github.py", "e3_colab_memory_remote")
    from scripts import e3_innovation_github as storage

    fixture = training.fixture.__wrapped__(tmp_path)
    baseline_path = fixture["base"] / "snapshot_manifest.json"
    baseline = json.loads(baseline_path.read_bytes())
    baseline_files = {name: (fixture["base"] / name).read_bytes() for name in baseline["files"]}
    monkeypatch.setattr(storage, "V1_SNAPSHOT_SHA256", hashlib.sha256(baseline_path.read_bytes()).hexdigest())
    monkeypatch.setattr(storage, "_v1_terminal", lambda _transport: (baseline, baseline_files))
    plan_path = tmp_path / "coordinated_plan.json"
    plan_path.write_text(json.dumps(fixture["plan"]))
    remote = storage_test.MemoryRemote()
    run_dir = tmp_path / "coordinated_run"
    result = storage.run_github_innovation(
        token=storage_test.TOKEN, npz_path=fixture["npz"], expected_sha256=fixture["sha"],
        run_dir=run_dir, plan_path=plan_path, code_root=ROOT, python=sys.executable,
        environment=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        device="cpu", transport=remote, v1_transport=object(),
        frozen_v1_dir=fixture["base"], snapshot_manifest_path=baseline_path,
        max_chunks=4, stop_after_epochs=3)
    assert result["status"] == "completed"
    manifest, files = remote.read_snapshot(remote.head())
    assert manifest["registry"]["active_session"] is None
    summary = json.loads(files["summary.json"])
    assert summary["comparison_complete"] and len(summary["models"]) == 6
    assert 0 < summary["charged_seconds"] < 600
    assert summary["global_charged_seconds"] == 369.4779596389991 + summary["charged_seconds"]
    assert all(storage_test.TOKEN.encode() not in payload for payload in files.values())
    assert remote.events[0][1] is True and remote.events[-1][1] is False
