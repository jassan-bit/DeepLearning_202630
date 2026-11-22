"""Pinned source and isolated CPU setup for statistical validation only.

Standard library in the notebook kernel. No market download, fitting, GPU,
credentials, automatic simulation, or mutation of an existing interpreter.
AI assistance: implementation and software review; repository code terms apply.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import venv

REPOSITORY = "jassan-bit/DeepLearning_202630"
PREFIX = "experiments/e3_primary_joint_repair_20261122"
LOCK = "colab/requirements-validation.lock.txt"
MAX_PACKET_BYTES = 64 * 1024 * 1024
NUMERICAL_PINS = {"numpy":"2.3.5", "scipy":"1.17.0"}
CLI_MODULE = "joint_validation.cli"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cpu_env():
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    for key in tuple(env):
        if key == "PYTHONPATH" or re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL", key, re.I):
            env.pop(key, None)
    return env


def _regular(path, root):
    path, root = Path(path), Path(root).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)) and
            path.resolve().is_relative_to(root.resolve()) and path.is_file(), "Source path/symlink differs")
    return path


def _new(path, raw):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Refuse symlink cache path")
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, "Immutable cached bytes differ; preserve this cache")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    try:
        require(not path.exists(), "Cache target appeared during write")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download(url, maximum):
    request = urllib.request.Request(url, headers={"User-Agent":"E3-statistical-validation-CPU-1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = response.read(maximum + 1)
    require(len(raw) <= maximum, "Pinned source download exceeds declared size")
    return raw


def _manifest(raw):
    value = json.loads(raw)
    require(value.get("algorithm") == "sha256" and isinstance(value.get("files"), dict) and
            isinstance(value.get("sizes"), dict) and 0 < len(value["files"]) <= 4096 and
            set(value["files"]) == set(value["sizes"]), "Invalid source manifest")
    for name, digest in value["files"].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and path.as_posix() == name and ".." not in path.parts and
                not any(part.startswith(".") for part in path.parts) and
                re.fullmatch(r"[0-9a-f]{64}", digest or ""), "Unsafe source manifest entry")
        require(type(value["sizes"][name]) is int and 0 <= value["sizes"][name] <= MAX_PACKET_BYTES,
                "Invalid source file size")
    require(sum(value["sizes"].values()) <= MAX_PACKET_BYTES, "Statistical source packet exceeds byte limit")
    require(LOCK in value["files"], "Pinned numerical lock missing")
    return value


def acquire(source_commit, manifest_sha256, cache):
    require(re.fullmatch(r"[0-9a-f]{40}", source_commit or "") and
            re.fullmatch(r"[0-9a-f]{64}", manifest_sha256 or ""), "Use immutable published source pins")
    root = Path(cache).absolute() / ("statistics-source-" + source_commit)
    manifest_path = root / "SOURCE_MANIFEST.json"
    if manifest_path.exists():
        raw = _regular(manifest_path, root).read_bytes()
    else:
        raw = download(f"https://raw.githubusercontent.com/{REPOSITORY}/{source_commit}/{PREFIX}/SOURCE_MANIFEST.json", 1024*1024)
    require(sha(raw) == manifest_sha256, "Source manifest SHA256 differs")
    manifest = _manifest(raw)
    base = f"https://raw.githubusercontent.com/{REPOSITORY}/{source_commit}/{PREFIX}/"
    for relative, expected in manifest["files"].items():
        target = root / relative
        if target.exists():
            body = _regular(target, root).read_bytes()
        else:
            body = download(base + urllib.parse.quote(relative, safe="/"), manifest["sizes"][relative])
        require(len(body) == manifest["sizes"][relative] and sha(body) == expected, "Source size/hash differs: " + relative)
        _new(target, body)
    _new(manifest_path, raw)
    anchor = dict(source_commit=source_commit, manifest_sha256=manifest_sha256,
                  scope="Immutable statistical package; source identity is not market confirmation")
    _new(root / "PUBLIC_SOURCE_ANCHOR.json", (json.dumps(anchor, sort_keys=True) + "\n").encode())
    return root


def verify(root, source_commit, manifest_sha256):
    root = Path(root).absolute()
    raw = _regular(root / "SOURCE_MANIFEST.json", root).read_bytes()
    require(sha(raw) == manifest_sha256, "Source manifest differs")
    manifest = _manifest(raw)
    anchor = json.loads(_regular(root / "PUBLIC_SOURCE_ANCHOR.json", root).read_text())
    require(anchor.get("source_commit") == source_commit and anchor.get("manifest_sha256") == manifest_sha256,
            "Source anchor differs")
    for relative, digest in manifest["files"].items():
        body = _regular(root / relative, root).read_bytes()
        require(len(body) == manifest["sizes"][relative] and sha(body) == digest, "Source file differs: " + relative)
    return dict(source_commit=source_commit, manifest_sha256=manifest_sha256, files=len(manifest["files"]))


_CHECK_PYTHON = r'''
import importlib.metadata as metadata, json, sys
import numpy, scipy
expected = json.loads(sys.argv[1])
actual = {key:metadata.version(key) for key in expected}
assert actual == expected, 'Numerical dependency versions differ'
assert numpy.__version__ == actual['numpy'] and scipy.__version__ == actual['scipy']
print(json.dumps({'python':sys.version.split()[0],'numerical_versions':actual,'numpy_scipy_CPU_only':True,'prefix':sys.prefix,'base_prefix':sys.base_prefix}))
'''


def check_python(python):
    require(Path(python).is_file(), "CPU interpreter missing")
    result = subprocess.run([str(python), "-B", "-c", _CHECK_PYTHON, json.dumps(NUMERICAL_PINS)],
                            env=cpu_env(), text=True, capture_output=True)
    require(result.returncode == 0, "Existing interpreter does not match the numerical lock")
    return json.loads(result.stdout)


def install(source, cache, existing_python=None):
    """Existing Python is checked only; pip can write only the owned new venv."""
    source, cache = Path(source), Path(cache).absolute()
    lock = _regular(source / LOCK, source)
    actual_pins = dict(line.strip().split("==") for line in lock.read_text().splitlines()
                       if line.strip() and not line.lstrip().startswith("#"))
    require(actual_pins == NUMERICAL_PINS, "Unexpected numerical dependency lock")
    if existing_python is not None:
        details = check_python(existing_python)
        return Path(existing_python), details, "verified_existing_interpreter_no_mutation"
    require((3, 11) <= sys.version_info[:2] <= (3, 14), "Python 3.11–3.14 required by this numerical lock")
    directory = cache / ("venv-statistics-cpu-" + sha(lock.read_bytes())[:16] +
                         f"-py{sys.version_info.major}{sys.version_info.minor}")
    require(not any(p.is_symlink() for p in (directory, *directory.parents)), "Refuse symlink venv")
    python = directory / "bin/python"
    def assert_owned():
        require((directory / "pyvenv.cfg").is_file(), "Owned venv metadata missing; refuse to write any interpreter")
        probe = subprocess.run([str(python), "-B", "-c",
            "import pathlib,sys; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve() and sys.prefix!=sys.base_prefix",
            str(directory)], env=cpu_env(), text=True, capture_output=True)
        require(probe.returncode == 0, "Interpreter is not the owned isolated venv; refuse pip mutation")
    ready = False
    if python.exists():
        assert_owned()
        try:
            check_python(python); ready = True
        except RuntimeError:
            pass
    if not ready:
        directory.parent.mkdir(parents=True, exist_ok=True)
        if not python.exists():
            venv.EnvBuilder(with_pip=False, system_site_packages=False, symlinks=True).create(directory)
        assert_owned()
        subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "install", "--disable-pip-version-check", "-r", str(lock)],
                       env=cpu_env(), check=True)
        subprocess.run([sys.executable, "-m", "pip", "--python", str(python), "check"], env=cpu_env(), check=True)
    return python, check_python(python), "owned_isolated_numerical_venv"


def setup(source_commit, manifest_sha256, cache, existing_python=None):
    source = acquire(source_commit, manifest_sha256, cache)
    python, environment, origin = install(source, cache, existing_python=existing_python)
    result = dict(source_root=str(source), source_commit=source_commit, manifest_sha256=manifest_sha256,
        python=str(python), cache_root=str(Path(cache).absolute()), environment=environment,
        interpreter_origin=origin, native_compiler_available=bool(shutil.which("c++") or shutil.which("g++")),
        market_acquisition_executed=False, model_training_executed=False, monte_carlo_executed=False)
    print(json.dumps(dict(status="statistical_CPU_sources_verified_no_simulation", **result)), flush=True)
    return result


def command(settings, args):
    """Run only the fixed statistical CLI; retain a bounded stdout tail."""
    require(isinstance(args, list) and all(isinstance(x, str) for x in args), "CLI args must be explicit strings")
    require(args and args[0] in ("freeze", "run", "status", "export", "restore"),
            "Only the independent statistical validation CLI is allowed")
    verify(settings["source_root"], settings["source_commit"], settings["manifest_sha256"])
    argv = [settings["python"], "-B", "-u", "-m", CLI_MODULE, *args]
    process = subprocess.Popen(argv, cwd=settings["source_root"], env=cpu_env(), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines, retained = deque(), 0
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line); retained += len(line)
            while retained > 512 * 1024 and len(lines) > 1:
                retained -= len(lines.popleft())
        result = subprocess.CompletedProcess(argv, process.wait(), stdout="".join(lines), stderr=None)
    except BaseException:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        raise
    finally:
        process.stdout.close()
    require(result.returncode == 0, "Statistical stage stopped; keep the case and its authenticated ledger")
    return result
