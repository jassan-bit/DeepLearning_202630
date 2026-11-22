"""Small offline source/isolation checks; no environment installation or MC."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("statistical_cpu_bootstrap_fixture", HERE / "bootstrap_cpu.py")
boot = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(boot)


def packet():
    files = {boot.LOCK:(HERE / "requirements-validation.lock.txt").read_bytes(),
             "joint_validation/cli.py":b"# Software fixture only, never executed\n"}
    manifest = dict(algorithm="sha256", files={p:boot.sha(body) for p,body in files.items()},
                    sizes={p:len(body) for p,body in files.items()})
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    return files, raw, boot.sha(raw)


def materialize(tmp_path, monkeypatch):
    files, manifest, digest = packet(); fetched=[]
    def get(url, maximum):
        fetched.append(url)
        relative = url.split(boot.PREFIX + "/", 1)[1]
        return manifest if relative == "SOURCE_MANIFEST.json" else files[relative]
    monkeypatch.setattr(boot, "download", get)
    return boot.acquire("a"*40, digest, tmp_path), digest, fetched


def test_materialization_is_hash_checked_and_fully_offline_idempotent(tmp_path, monkeypatch):
    source, digest, fetched = materialize(tmp_path, monkeypatch)
    assert len(fetched) == 3 and boot.verify(source, "a"*40, digest)["files"] == 2
    before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(boot, "download", lambda *a:pytest.fail("NO_REPEAT_NETWORK"))
    assert boot.acquire("a"*40, digest, tmp_path) == source
    assert {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()} == before


def test_altered_cached_source_is_preserved_and_rejected(tmp_path, monkeypatch):
    source, digest, _ = materialize(tmp_path, monkeypatch)
    path = source / "joint_validation/cli.py"; path.write_bytes(b"ALTERED")
    with pytest.raises(RuntimeError, match="size/hash"):
        boot.acquire("a"*40, digest, tmp_path)
    assert path.read_bytes() == b"ALTERED"


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", ".git/config", "x/../y.py"])
def test_manifest_rejects_unsafe_paths(name):
    value=dict(algorithm="sha256", files={name:"a"*64}, sizes={name:1})
    with pytest.raises(RuntimeError, match="Unsafe"):
        boot._manifest(json.dumps(value).encode())


def test_wrong_manifest_digest_never_writes_source(tmp_path, monkeypatch):
    _, raw, _ = packet()
    monkeypatch.setattr(boot, "download", lambda *a:raw)
    with pytest.raises(RuntimeError, match="manifest SHA256"):
        boot.acquire("a"*40, "b"*64, tmp_path)
    assert not list(tmp_path.iterdir())


def test_source_symlink_is_rejected_before_target_changes(tmp_path, monkeypatch):
    source, digest, _ = materialize(tmp_path, monkeypatch)
    path=source / "joint_validation/cli.py"; external=tmp_path / "external"
    external.write_bytes(path.read_bytes()); path.unlink(); path.symlink_to(external)
    original=external.read_bytes()
    with pytest.raises(RuntimeError, match="symlink"):
        boot.verify(source, "a"*40, digest)
    assert external.read_bytes() == original


def test_existing_interpreter_is_only_checked_never_installed(tmp_path, monkeypatch):
    source=tmp_path / "source"; (source / "colab").mkdir(parents=True)
    (source / boot.LOCK).write_bytes((HERE / "requirements-validation.lock.txt").read_bytes())
    seen=[]
    monkeypatch.setattr(boot, "check_python", lambda p:seen.append(p) or {"numerical_versions":boot.NUMERICAL_PINS})
    with patch("subprocess.run", side_effect=AssertionError("NO_PIP_EXISTING_ENV")), \
         patch("venv.EnvBuilder", side_effect=AssertionError("NO_VENV_EXISTING_ENV")):
        python, details, origin=boot.install(source, tmp_path / "cache", existing_python=sys.executable)
    assert seen == [sys.executable] and python == Path(sys.executable)
    assert origin == "verified_existing_interpreter_no_mutation"
    assert not (tmp_path / "cache").exists()


def test_unowned_existing_venv_cannot_trigger_pip_on_base_python(tmp_path, monkeypatch):
    source=tmp_path / "source"; (source / "colab").mkdir(parents=True)
    lock=(HERE / "requirements-validation.lock.txt").read_bytes(); (source / boot.LOCK).write_bytes(lock)
    directory=tmp_path / "cache" / ("venv-statistics-cpu-" + boot.sha(lock)[:16] +
                                   f"-py{sys.version_info.major}{sys.version_info.minor}")
    (directory / "bin").mkdir(parents=True); (directory / "bin/python").symlink_to(sys.executable)
    with patch("subprocess.run", side_effect=AssertionError("NO_BASE_ENV_MUTATION")), \
         pytest.raises(RuntimeError, match="venv metadata missing"):
        boot.install(source, tmp_path / "cache")


def test_cpu_env_strips_unneeded_credentials_and_pins_thread_counts(monkeypatch):
    for key in ("GITHUB_TOKEN", "E3_R3_BACKUP_SECRET", "OPENAI_API_KEY", "MY_PASSWORD", "PYTHONPATH"):
        monkeypatch.setenv(key, "SOFTWARE_FIXTURE_PRIVATE_VALUE")
    result=boot.cpu_env()
    assert all(k not in result for k in ("GITHUB_TOKEN", "E3_R3_BACKUP_SECRET", "OPENAI_API_KEY", "MY_PASSWORD", "PYTHONPATH"))
    assert result["CUDA_VISIBLE_DEVICES"] == ""
    assert result["OMP_NUM_THREADS"] == result["OPENBLAS_NUM_THREADS"] == "1"
    assert "GITHUB_TOKEN" in os.environ


@pytest.mark.parametrize("operation", ["train", "run-final", "collect-development", "evaluate", "predict-planning"])
def test_statistical_bootstrap_refuses_model_and_market_commands_before_process(operation):
    with patch("subprocess.Popen", side_effect=AssertionError("NO_MODEL_OR_MARKET_PROCESS")), \
         pytest.raises(RuntimeError, match="statistical validation CLI"):
        boot.command({}, [operation])
