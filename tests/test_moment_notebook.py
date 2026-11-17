"""Source pin, credential boundaries and robust notebook display; no real fits."""
from __future__ import annotations

import ast
import hashlib
import html
import io
import json
from pathlib import Path, PurePosixPath
import re
from types import SimpleNamespace

import pytest

from scripts import e3_build_moment_notebook as builder

ROOT = Path(__file__).resolve().parents[1]


def parts():
    notebook = json.loads(builder.OUTPUT.read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    tree = ast.parse(code)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return notebook, code, tree, constants


def test_notebook_reproduces_exact_sources_and_has_no_fabricated_outputs():
    notebook, code, _, values = parts()
    assert notebook == builder.build(values["SOURCE_COMMIT"])
    assert re.fullmatch(r"[0-9a-f]{40}", values["SOURCE_COMMIT"]) and values["SOURCE_COMMIT"] != "0" * 40
    assert values["RESULTS_BRANCH"] == "e3-desarrollo-resultados-momentos"
    assert values["RESULTS_PREFIX"] == "runs/momentos_v1"
    assert "drive.mount" not in code and "from google.colab import drive" not in code
    assert {"src/e3_moment_model.py", "src/e3_moment_training.py", "scripts/e3_run_moments.py",
            "scripts/e3_moment_github.py", builder.PLAN} <= set(values["SOURCE_FILE_SHA256"])
    for path, digest in values["SOURCE_FILE_SHA256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for index, c in enumerate(notebook["cells"]):
        if c["cell_type"] == "code":
            assert c["execution_count"] is None and not c["outputs"]
            compile("".join(c["source"]), f"moment-cell-{index}", "exec")


def downloader(payload):
    _, _, tree, _ = parts()
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "download_pinned")
    fake = SimpleNamespace(request=SimpleNamespace(urlopen=lambda *a, **kw: io.BytesIO(payload)))
    import os
    ns = dict(Path=Path, PurePosixPath=PurePosixPath, re=re, hashlib=hashlib,
              urllib=fake, os=os, REPOSITORY="jassan-bit/DeepLearning_202630")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "download-only", "exec"), ns)
    return ns["download_pinned"]


def test_download_hash_failure_never_writes_or_overwrites_sources(tmp_path):
    good = b"synthetic pinned code\n"
    digest = hashlib.sha256(good).hexdigest()
    path = tmp_path / "code.py"
    with pytest.raises(RuntimeError):
        downloader(b"altered")("src/model.py", "a" * 40, path, digest)
    assert not path.exists()
    assert downloader(good)("src/model.py", "a" * 40, path, digest) == digest
    path.write_bytes(b"local changes to preserve")
    with pytest.raises(RuntimeError):
        downloader(good)("src/model.py", "a" * 40, path, digest)
    assert path.read_bytes() == b"local changes to preserve"


@pytest.mark.parametrize("path", ["../file", "/absolute", "src/../file", "src\\file", ""])
def test_download_rejects_unpinned_path_forms(tmp_path, path):
    with pytest.raises(RuntimeError):
        downloader(b"data")(path, "a" * 40, tmp_path / "data", hashlib.sha256(b"data").hexdigest())


def test_stage_source_checks_precede_coordinator_and_secret_is_not_exported():
    notebook, code, _, _ = parts()
    for action in ("prepare_github_moments", "run_github_moments"):
        source = next("".join(c["source"]) for c in notebook["cells"] if action in "".join(c["source"]))
        assert source.strip().startswith("assert_fixed_sources()")
        assert "token=GITHUB_AUTH_TOKEN" in source or "**GITHUB_ARGS" in source
    assert 'device="cuda"' in code
    assert 'os.environ["GITHUB_TOKEN"]' not in code
    assert "print(GITHUB_AUTH_TOKEN" not in code


def test_display_handles_uninitialized_metrics_and_escapes_names(tmp_path):
    notebook, _, _, _ = parts()
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "summary_path = RUN_DIR" in "".join(c["source"]))
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    (tmp_path / "summary.json").write_text(json.dumps({"status": "budget_exhausted", "models": {
        "<script>unsafe</script>": {"parameters": 1236, "metrics": None, "complete": False}}}))
    displayed = []
    ns = dict(RUN_DIR=tmp_path, json=json, escape=html.escape, HTML=str, display=displayed.append)
    exec(compile(tree, "synthetic-display", "exec"), ns)
    assert len(displayed) == 1
    assert "&lt;script&gt;" in displayed[0] and "<script>" not in displayed[0]
    assert "—" in displayed[0]


def test_source_identity_check_detects_local_changes_before_fitting(tmp_path):
    _, _, tree, _ = parts()
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "assert_fixed_sources")
    path = tmp_path / "source.py"
    path.write_bytes(b"original")
    ns = dict(SOURCE_DIR=tmp_path, SOURCE_FILE_SHA256={"source.py": hashlib.sha256(b"original").hexdigest()},
              hashlib=hashlib)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "source-check-only", "exec"), ns)
    ns["assert_fixed_sources"]()
    path.write_bytes(b"modified")
    with pytest.raises(RuntimeError):
        ns["assert_fixed_sources"]()
    assert path.read_bytes() == b"modified"
