"""Offline operational guards. Synthetic metadata is not a human attestation.

No market requests, fitting, test opening, Git writes or source-tree copies.
The local source's unpublished QA anchor is substituted only in one read-only
unit test; production verification always requires the actual published anchor.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

OPS = Path(__file__).resolve().parent
sys.path.insert(0, str(OPS))
import resume_development as launcher
import resume_worker as worker

SOURCE = OPS.parents[1] / "e3_hypothesis_adequacy_20261122" / "runtime"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def bundle_namespace(tmp_path):
    bundle, private = tmp_path / "addon", tmp_path / "private"
    (bundle / "ops").mkdir(parents=True)
    files = {}
    for name in ("resume_development.py", "resume_worker.py"):
        path = bundle / "ops" / name
        shutil.copyfile(OPS / name, path)
        files["ops/" + name] = launcher.sha(path)
    manifest = dict(version=launcher.VERSION,
        parent_source_commit=launcher.SOURCE_COMMIT,
        parent_manifest_sha256=launcher.SOURCE_MANIFEST_SHA256,
        parent_data_sha256=launcher.SOURCE_DATA_SHA256,
        scope="resume_existing_F02_F03_development_only",
        preserved_physical_adapter_version="crypto-markets-bounded-archives-base-volume-v2.1",
        files=files)
    manifest_path = write_json(bundle / "ops" / "REPAIR_MANIFEST.json", manifest)
    namespace = dict(SOURCE=SOURCE, SOURCE_COMMIT=launcher.SOURCE_COMMIT,
        MANIFEST_SHA256=launcher.SOURCE_MANIFEST_SHA256,
        SETTINGS=dict(source_commit=launcher.SOURCE_COMMIT, source_root=str(SOURCE), python=sys.executable),
        PRIVATE=private, DEVELOPMENT_MAP=private / "development-panels.json", bootstrap=object(),
        definitions=[dict(panel=p, folder=private / "inputs" / p,
            raw=private / "raw" / p / "development-v1",
            panel_dir=private / "panels" / p / "development-v1") for p in worker.PANELS])
    return namespace, bundle, launcher.sha(manifest_path)


def test_original_304_source_files_are_verified_and_untouched(monkeypatch):
    manifest = json.loads((SOURCE / "SOURCE_MANIFEST.json").read_text())
    paths = [SOURCE / p for p in manifest["files"]]
    paths += [SOURCE / "SOURCE_MANIFEST.json", SOURCE / "PUBLIC_SOURCE_ANCHOR.json"]
    before = {str(p): (launcher.sha(p), p.stat().st_mtime_ns) for p in paths}
    read = worker.read
    # Unit fixture only: published S2 identity; no source file is edited.
    monkeypatch.setattr(worker, "read", lambda path: dict(source_commit=launcher.SOURCE_COMMIT,
        manifest_sha256=launcher.SOURCE_MANIFEST_SHA256) if Path(path).name == "PUBLIC_SOURCE_ANCHOR.json" else read(path))
    assert len(worker.verify_source(SOURCE)["files"]) == 304
    assert before == {str(p): (launcher.sha(p), p.stat().st_mtime_ns) for p in paths}


@pytest.mark.parametrize("change", ["source_commit", "manifest_pin", "settings_source", "settings_nondict", "source_root", "python_missing"])
def test_changed_source_or_interpreter_stops_before_subprocess_and_prompts(bundle_namespace, change):
    namespace, bundle, pin = bundle_namespace
    if change == "source_commit": namespace["SOURCE_COMMIT"] = "f" * 40
    elif change == "manifest_pin": namespace["MANIFEST_SHA256"] = "a" * 64
    elif change == "settings_source": namespace["SETTINGS"]["source_commit"] = "5" * 40
    elif change == "settings_nondict": namespace["SETTINGS"] = None
    elif change == "source_root": namespace["SETTINGS"]["source_root"] = "/tmp/foreign-source"
    elif change == "python_missing": namespace["SETTINGS"]["python"] = "/tmp/nonexistent-repair-python"
    with patch("subprocess.Popen", side_effect=AssertionError("NO_SUBPROCESS")), \
         patch("builtins.input", side_effect=AssertionError("NO_NEW_CONSENT")), \
         patch("getpass.getpass", side_effect=AssertionError("NO_SECRET")), \
         patch("urllib.request.urlopen", side_effect=AssertionError("NO_NETWORK")):
        with pytest.raises(RuntimeError): launcher.resume(namespace, bundle, pin)


def test_payload_validation_is_read_only_and_accepts_original_namespace(bundle_namespace):
    namespace, bundle, pin = bundle_namespace
    before = dict(namespace)
    payload = launcher.prepare_payload(namespace, bundle, pin)
    assert payload["source"] == str(SOURCE.resolve())
    assert namespace == before
    assert not namespace["PRIVATE"].exists()


def test_changed_addon_bytes_stops_before_subprocess(bundle_namespace):
    namespace, bundle, pin = bundle_namespace
    with (bundle / "ops/resume_worker.py").open("a") as stream: stream.write("\n# tampered fixture\n")
    with patch("subprocess.Popen", side_effect=AssertionError("NO_SUBPROCESS")):
        with pytest.raises(RuntimeError, match="addon diferente"):
            launcher.resume(namespace, bundle, pin)
    with pytest.raises(RuntimeError, match="Repair file differs"):
        worker.verify_bundle(dict(bundle=str(bundle), repair_manifest_sha256=pin))


@pytest.mark.parametrize("operation", ["prepare-final", "run-final", "train", "record-owner-evidence", "evaluate", "resume-F00"])
def test_worker_rejects_unallowed_operation_in_isolated_process_before_inputs(tmp_path, operation):
    token = "PRIVATE_TOKEN_SENTINEL_DO_NOT_PRINT"
    result = subprocess.run([sys.executable, "-B", str(OPS / "resume_worker.py")],
        input=json.dumps(dict(operation=operation, source=str(tmp_path / "not-created"), token=token)),
        capture_output=True, text=True, cwd=tmp_path,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=10)
    assert result.returncode == 1
    event = json.loads(result.stdout)
    assert event["status"] == "development_repair_preflight_or_execution_failed"
    assert event["new_market_fits"] == 0 and event["test_opened"] is False
    assert token not in result.stdout + result.stderr
    assert not (tmp_path / "not-created").exists()


def preservation_fixture(tmp_path):
    source, private, bundle = (tmp_path / p for p in ("source", "private", "bundle"))
    write_json(source / "SOURCE_MANIFEST.json", dict(files={}))
    write_json(source / "PUBLIC_SOURCE_ANCHOR.json", dict(software_fixture=True))
    write_json(bundle / "ops/REPAIR_MANIFEST.json", dict(files={}))
    rows = []
    for p in worker.PANELS:
        row = dict(panel=p, folder=str(private / "inputs" / p), raw=str(private / "raw" / p),
                   panel_dir=str(private / "panels" / p))
        for key in ("folder", "raw", "panel_dir"):
            folder = Path(row[key]); folder.mkdir(parents=True)
            (folder / "immutable.bin").write_bytes((p + key).encode())
        (Path(row["raw"]) / "acquisition_attempts.jsonl").write_bytes(b'{"synthetic":"original"}\n')
        rows.append(row)
    return rows, source, bundle


def test_snapshot_preserves_all_original_inputs_and_admissions(tmp_path):
    rows, source, bundle = preservation_fixture(tmp_path)
    before = worker.snapshot(rows, source, bundle)
    worker.assert_preserved(before)
    for row in rows:
        path = Path(row["raw"]) / "acquisition_attempts.jsonl"
        assert before[str(path)]["append_only"] is (row["panel"] in ("F02", "F03"))
    path = Path(rows[0]["panel_dir"]) / "immutable.bin"
    path.write_bytes(b"changed-original-admission")
    with pytest.raises(RuntimeError, match="immutable"):
        worker.assert_preserved(before)


def test_preservation_detects_touch_even_when_bytes_are_identical(tmp_path):
    rows, source, bundle = preservation_fixture(tmp_path)
    before = worker.snapshot(rows, source, bundle)
    path = Path(rows[1]["raw"]) / "immutable.bin"
    stamp = path.stat().st_mtime_ns
    os.utime(path, ns=(path.stat().st_atime_ns, stamp + 1_000_000))
    with pytest.raises(RuntimeError, match="immutable"):
        worker.assert_preserved(before)


def test_only_appended_F02_attempts_allowed_original_prefix_cannot_change(tmp_path):
    rows, source, bundle = preservation_fixture(tmp_path)
    before = worker.snapshot(rows, source, bundle)
    path = Path(rows[2]["raw"]) / "acquisition_attempts.jsonl"
    with path.open("ab") as stream: stream.write(b'{"synthetic":"resumed"}\n')
    worker.assert_preserved(before)
    path.write_bytes(b'{"synthetic":"modified"}\n')
    with pytest.raises(RuntimeError, match="log prefix changed"):
        worker.assert_preserved(before)


def test_F00_attempts_are_immutable_not_appendable(tmp_path):
    rows, source, bundle = preservation_fixture(tmp_path)
    before = worker.snapshot(rows, source, bundle)
    with (Path(rows[0]["raw"]) / "acquisition_attempts.jsonl").open("ab") as stream:
        stream.write(b'{"unexpected":"append"}\n')
    with pytest.raises(RuntimeError, match="immutable"):
        worker.assert_preserved(before)
