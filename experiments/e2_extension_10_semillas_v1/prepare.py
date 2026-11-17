"""Prepare pinned D02 source and a child runtime; never read market data or fit.

Only the standard library is imported here. The genuine constructors are checked
in a separate interpreter, on a fixed artificial [1, 7, 5] CPU tensor.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from zipfile import ZipFile

SOURCE_ZIP_SHA256 = "5423c85bf220729c826e47a1c37d79a49ce2ba8b0e8f764c47ab9c0796b4bcae"
SOURCE_COMMIT = "c4a4a1e68e9844cdfb81de0075177ced0cd96c07"
SOURCE_REPOSITORY_PATH = "book_e2/_static/entregables_independientes_v2_20261115/02_codigo_modelos.zip"
METADATA = (
    "experiments/e2_completion_20261106/design.json",
    "experiments/e2_published_architectures_20261112/architecture_policy.json",
    "book_e2/_static/search_protocol_20261109/protocol.json",
    "book_e2/_static/current_implementation_20261108/source_design.json",
    "book_e2/_static/e2_flops_final_audit_20261114/frozen_architecture_inputs.json",
    "book_e2/_static/e2_data_training_uncertainty_current_v1/model_training_register.csv",
    "experiments/dl_r2_cv_20261104/protocol.json",
)
AUTHOR_MODELS = frozenset(("TimesNet", "TimeMixer", "PatchTST", "iTransformer", "TSMixer", "TiDE"))
# Every large CUDA/Torch dependency is borrowed, never installed into the child.
SHARED_STACK = frozenset(("torch", "numpy", "filelock", "typing_extensions", "setuptools",
    "sympy", "mpmath", "networkx", "jinja2", "markupsafe", "fsspec", "triton"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    raw = canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("Existing immutable receipt differs: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def regular_path(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Symlink paths are not admitted: " + str(path))
    return path


def safe_member(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or ".." in path.parts or "\\" in name
            or ":" in name or "\x00" in name or path.as_posix() != name.rstrip("/")):
        raise ValueError("Unsafe ZIP path: " + name)
    return path


def selected_members(archive):
    """Positive selection; reject ambiguous paths and special files everywhere."""
    members = {}
    for info in archive.infolist():
        safe_member(info.filename)
        mode = info.external_attr >> 16
        if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("ZIP special files and symlinks are forbidden: " + info.filename)
        if info.filename in members:
            raise ValueError("Duplicate ZIP path: " + info.filename)
        members[info.filename] = info
    design = json.loads(archive.read("D02/code/" + METADATA[0]))
    authors = [r for r in design["author_sources"] if r["model"] in AUTHOR_MODELS]
    if len(authors) != 76 or {r["model"] for r in authors} != AUTHOR_MODELS:
        raise ValueError("Expected exactly 76 pinned files for six author subsets")
    selected = {n for n in members if n.startswith("D02/code/scripts/") and n.endswith(".py")}
    selected.update("D02/code/" + n for n in METADATA)
    selected.add("D02/requirements_cuda.txt")
    for row in authors:
        name = "D02/authors/" + row["model"] + "/" + row["path"]
        safe_member(name)
        raw = archive.read(name)
        if len(raw) != row["bytes"] or sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("Pinned author source differs: " + name)
        selected.add(name)
    if not selected or any(n not in members for n in selected):
        raise ValueError("Selected source member is missing")
    if any(members[n].is_dir() or members[n].file_size > 10 * 1024**2 for n in selected):
        raise ValueError("Selected member is not a small regular source file")
    if sum(members[n].file_size for n in selected) > 100 * 1024**2:
        raise ValueError("Selected source exceeds its size limit")
    return sorted(selected)


def clean_environment(parent_env=None):
    env = dict(os.environ if parent_env is None else parent_env)
    for key in list(env):
        if (key.startswith(("PIP_", "UV_", "GIT_"))
                or re.search(r"(?:TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY|ASKPASS|(?:^|_)PAT(?:_|$))", key.upper())
                or key in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}):
            env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    return env


def run_python(python, code, env):
    result = subprocess.run([str(python), "-B", "-c", code], env=env, check=True,
                            capture_output=True, text=True)
    # Some original author modules print diagnostics before our final JSON.
    return json.loads(result.stdout.strip().splitlines()[-1])


def source_preflight(package_root, python=sys.executable, env=None, smoke=False):
    root = str(regular_path(package_root))
    code = """
import json,sys
from pathlib import Path
root=Path(ROOT); sys.path.insert(0,str(root/'code/scripts'))
import e2_standalone_code as source
report,rows,roots=source.inventory(root,root/'code')
if SMOKE: report['smoke_cpu']=source.smoke_cpu(root/'code',rows,roots,source.MODEL_ORDER)
print(json.dumps(report,allow_nan=False))
""".replace("ROOT", repr(root)).replace("SMOKE", repr(smoke))
    return run_python(python, code, clean_environment(env))


def prepare_source(zip_path, workspace, expected_sha256=SOURCE_ZIP_SHA256):
    """Extract only audited source files, preserving every original byte."""
    zip_path, workspace = regular_path(zip_path), regular_path(workspace)
    if file_sha(zip_path) != expected_sha256:
        raise ValueError("Source ZIP SHA256 mismatch")
    bundle = workspace / "bundle"
    with ZipFile(zip_path) as archive:
        selected = selected_members(archive)
        source_root = bundle / "D02"
        if source_root.exists():
            for path in source_root.rglob("*"):
                if path.is_symlink() or (path.is_file() and path.relative_to(bundle).as_posix() not in selected):
                    raise ValueError("Unexpected file in source-only bundle: " + str(path))
        manifest = []
        for name in selected:
            raw = archive.read(name)
            destination = regular_path(bundle.joinpath(*PurePosixPath(name).parts))
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not destination.is_file() or destination.read_bytes() != raw:
                    raise ValueError("Existing source bytes differ: " + name)
            else:
                with destination.open("xb") as stream:
                    stream.write(raw)
            manifest.append({"path": name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()})
    binding = {"source_zip_sha256": expected_sha256, "source_commit": SOURCE_COMMIT,
        "source_repository_path": SOURCE_REPOSITORY_PATH, "members": manifest}
    report = {**binding, "version": "e2-four-seed-source-v1",
        "source_binding_sha256": sha256(canonical(binding)).hexdigest(),
        "package_root": str(source_root), "code_root": str(source_root / "code"),
        "inventory": source_preflight(source_root),
        "market_data_extracted": False, "historical_scores_extracted": False,
        "training_payloads_extracted": False}
    write_once(workspace / "source_manifest.json", report)
    return report


def read_lock(path):
    expected = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--index-url ", "--extra-index-url ")):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
        if not match or match[1].lower() in expected:
            raise ValueError("Only unique exact package pins are admitted: " + line)
        expected[match[1].lower()] = match[2]
    if len(expected) != 43 or expected.get("torch") != "2.8.0+cu126" or expected.get("numpy") != "2.3.5":
        raise ValueError("Unexpected D02 GPU package lock")
    return expected


def runtime_probe(python, env, expected, require_cuda=True):
    code = """
import importlib.metadata as m,json,platform,sys,sysconfig
from pathlib import Path
import torch,numpy
expected=EXPECTED
packages={n:None for n in expected}
for n in packages:
 try: packages[n]=m.version(n)
 except m.PackageNotFoundError: pass
modules={'torch':str(Path(torch.__file__).resolve()),'numpy':str(Path(numpy.__file__).resolve())}
if packages.get('scipy')==expected.get('scipy'):
 import scipy
 modules['scipy']=str(Path(scipy.__file__).resolve())
devices=[]
if torch.cuda.is_available():
 for i in range(torch.cuda.device_count()):
  p=torch.cuda.get_device_properties(i)
  devices.append({'index':i,'name':p.name,'total_memory_bytes':p.total_memory,
   'capability':list(torch.cuda.get_device_capability(i))})
if REQUIRE_CUDA:
 assert platform.python_version()=='3.12.12','Parent/child require Python 3.12.12'
 assert str(torch.__version__)=='2.8.0+cu126' and torch.version.cuda=='12.6','CUDA Torch differs'
 assert devices,'No visible CUDA GPU'
print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'prefix':sys.prefix,
 'base_prefix':sys.base_prefix,'site_paths':list(dict.fromkeys(p for p in sys.path if p and Path(p).name in ('site-packages','dist-packages'))),
 'packages':packages,'modules':modules,'torch':str(torch.__version__),'torch_cuda':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),'devices':devices}))
""".replace("EXPECTED", repr(expected)).replace("REQUIRE_CUDA", repr(require_cuda))
    return run_python(python, code, env)


def create_overlay(parent_python, child, parent_site_paths, env):
    """Create a venv whose own site precedes read-only borrowed parent packages."""
    subprocess.run([str(parent_python), "-B", "-m", "venv", "--without-pip", str(child)], env=env, check=True)
    python = child / "bin/python"
    paths = run_python(python, "import json,sysconfig;print(json.dumps(sysconfig.get_paths()))", env)
    site = regular_path(paths["purelib"])
    (site / "e2_parent_stack.pth").write_text("".join(p + "\n" for p in parent_site_paths), encoding="utf-8")
    return python, site


def prepare_runtime(workspace, parent_python, parent_env=None, *, require_cuda=True):
    """Return (child_python, environment, identity), without modifying the parent.

    The optional CPU mode is for local source smoke checks only. It never installs
    the CUDA lock or admits a production fit.
    """
    workspace = regular_path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    parent_python = Path(parent_python).absolute()  # Preserve a venv's executable path.
    env = clean_environment(parent_env)
    expected = read_lock(workspace / "bundle/D02/requirements_cuda.txt")
    parent = runtime_probe(parent_python, env, expected, require_cuda)
    protected = {n for n in expected if n in SHARED_STACK or n.startswith("nvidia-")}
    if require_cuda and any(parent["packages"][n] != expected[n] for n in protected):
        raise ValueError("Parent Torch/CUDA/NumPy stack differs from the frozen GPU lock")
    child = workspace / "runtime/venv-e2-extension"
    receipt = workspace / "runtime_identity.json"
    if child.exists() and not receipt.exists():
        raise ValueError("Partial child runtime preserved; use a fresh workspace: " + str(child))
    extras = {n: v for n, v in expected.items() if parent["packages"][n] != v}
    if not require_cuda:
        extras = {n: v for n, v in extras.items() if n not in protected}
    if not child.exists():
        child.parent.mkdir(parents=True, exist_ok=True)
        python, site = create_overlay(parent_python, child, parent["site_paths"], env)
        extra_lock = child.parent / "requirements-e2-extras.txt"
        write_once(child.parent / "overlay_plan.json", {"parent": parent, "extras": extras,
            "full_lock_sha256": file_sha(workspace / "bundle/D02/requirements_cuda.txt")})
        extra_lock.write_text("".join(n + "==" + v + "\n" for n, v in sorted(extras.items())), encoding="utf-8")
        if extras:
            subprocess.run([str(python), "-B", "-m", "pip", "install", "--target", str(site),
                "--no-deps", "--ignore-installed", "--no-cache-dir", "--disable-pip-version-check",
                "--no-input", "--index-url", "https://pypi.org/simple", "-r", str(extra_lock)], env=env, check=True)
    python = child / "bin/python"
    child_identity = runtime_probe(python, env, expected, require_cuda)
    admitted = expected if require_cuda else {n: v for n, v in expected.items() if n not in protected}
    if any(child_identity["packages"][n] != v for n, v in admitted.items()):
        raise ValueError("Child runtime does not satisfy its exact package lock")
    if child_identity["prefix"] != str(child) or any(
            child_identity["modules"].get(n) != p for n, p in parent["modules"].items()):
        raise ValueError("Child did not preserve the parent's shared scientific stack")
    for name in ("python", "torch", "torch_cuda", "devices", "base_prefix"):
        if child_identity[name] != parent[name]:
            raise ValueError("Parent and child runtime identities differ: " + name)
    smoke = source_preflight(workspace / "bundle/D02", python, env, smoke=True)
    if runtime_probe(parent_python, env, expected, require_cuda) != parent:
        raise ValueError("Parent runtime identity changed during child preparation")
    identity = {"version": "e2-four-seed-overlay-v1", "parent": parent, "child": child_identity,
        "child_python": str(python), "installed_extras": extras, "parent_modified": False,
        "full_gpu_lock_verified": require_cuda, "production_cuda_admitted": require_cuda,
        "lock_sha256": file_sha(workspace / "bundle/D02/requirements_cuda.txt"),
        "source_preflight": smoke, "fitting_performed": False, "market_data_read": False}
    write_once(receipt, identity)
    env["VIRTUAL_ENV"] = str(child)
    return python, env, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--parent-python", type=Path)
    args = parser.parse_args(argv)
    if args.source_zip:
        report = prepare_source(args.source_zip, args.workspace)
        print(json.dumps({"source_binding_sha256": report["source_binding_sha256"], "files": len(report["members"])}))
    if args.parent_python:
        python, _, report = prepare_runtime(args.workspace, args.parent_python)
        print(json.dumps({"child_python": str(python), "production_cuda_admitted": report["production_cuda_admitted"]}))


if __name__ == "__main__":
    main()
