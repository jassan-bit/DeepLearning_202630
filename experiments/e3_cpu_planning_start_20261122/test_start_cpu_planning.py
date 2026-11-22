"""Offline activation ordering; all runtime and remote operations are mocks.

These software fixtures neither train models nor assert a human attestation.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("cpu_start_software_fixture", ROOT / "start_cpu_planning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remote_verified():
    return dict(status="SAVED_CASE_REMOTE_VERIFIED", backup=dict(remote_verified=True, commit="a" * 40))


def completed(value, *, prefix="", code=0):
    return subprocess.CompletedProcess([], code, prefix + json.dumps(value) + "\n", "")


def test_last_json_reads_last_structured_line_after_progress(helper):
    value = remote_verified()
    result = completed(value, prefix='progress: copying encrypted case\n{"status":"earlier"}\n')
    assert helper._last_json(result) == value


@pytest.mark.parametrize("stdout", ["", "progress only\n", "{malformed}\n"])
def test_last_json_never_invents_verified_backup(helper, stdout):
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        helper._last_json(subprocess.CompletedProcess([], 0, stdout, ""))


def orchestrator_fixture(helper, monkeypatch, tmp_path):
    calls = []
    context = dict(case=tmp_path / "case", source=tmp_path / "source", private=tmp_path / "private",
                   bundle=tmp_path / "repair", python="unused-software-fixture-python",
                   repository="jassan-bit/DeepLearning_202630", branch="fixture-branch", prefix="fixture-prefix",
                   plan={"case_id":"SOFTWARE_FIXTURE"}, plan_sha256="b" * 64)
    namespace = {"remote_args": lambda *a, **k: [], "DEVELOPMENT_MAP":tmp_path / "unused-map.json"}
    def validate(ns):
        assert ns is namespace
        calls.append("validate")
        return context
    def prepare(ns, ctx):
        assert ns is namespace and ctx is context
        calls.append("prepare")
    def capsule(ns, ctx):
        assert ns is namespace and ctx is context
        calls.append("capsule")
        path = tmp_path / "capsule.json"
        path.write_bytes(b"SYNTHETIC_CAPSULE_FIXTURE")
        return path
    def sync(ns, ctx):
        assert ns is namespace and ctx is context
        calls.append("sync")
        return remote_verified()
    def activate(args, **kwargs):
        calls.append(("run_cli", args, kwargs))
        return completed(dict(status="SOFTWARE_FIXTURE_COMPLETED_NO_MARKET"))
    namespace["run_cli"] = activate
    monkeypatch.setattr(helper, "_validate", validate)
    monkeypatch.setattr(helper, "_prepare", prepare)
    monkeypatch.setattr(helper, "_capsule", capsule)
    monkeypatch.setattr(helper, "_sync", sync)
    return namespace, context, calls


def test_activation_order_capsule_remote_receipt_then_literal_budget(helper, monkeypatch, tmp_path):
    namespace, context, calls = orchestrator_fixture(helper, monkeypatch, tmp_path)
    def accept(prompt):
        calls.append("input")
        assert "340" in prompt
        return "PLANIFICAR 340"
    with patch("builtins.input", side_effect=accept), \
         patch("getpass.getpass", side_effect=AssertionError("NO_SECRET_IN_SOFTWARE_FIXTURE")):
        helper.start_cpu_planning(namespace)
    assert calls[:5] == ["validate", "prepare", "capsule", "sync", "input"]
    run = calls[5]
    assert run[0] == "run_cli" and run[1][0] == "run"
    assert run[2].get("remote") is True
    assert len(calls) == 6
    assert not context["case"].exists()


def test_cancelled_literal_budget_never_calls_fit(helper, monkeypatch, tmp_path):
    namespace, context, calls = orchestrator_fixture(helper, monkeypatch, tmp_path)
    with patch("builtins.input", return_value="NO"):
        try:
            helper.start_cpu_planning(namespace)
        except (RuntimeError, AssertionError):
            pass
    assert calls == ["validate", "prepare", "capsule", "sync"]
    assert not context["case"].exists()


def test_remote_sync_failure_stops_before_budget_input_or_fit(helper, monkeypatch, tmp_path):
    namespace, context, calls = orchestrator_fixture(helper, monkeypatch, tmp_path)
    def failed_sync(ns, ctx):
        calls.append("sync_failed")
        raise RuntimeError("REMOTE_VERIFICATION_SOFTWARE_FIXTURE_FAILED")
    monkeypatch.setattr(helper, "_sync", failed_sync)
    with patch("builtins.input", side_effect=AssertionError("NO_BUDGET_AFTER_FAILED_BACKUP")), \
         pytest.raises(RuntimeError, match="REMOTE_VERIFICATION"):
        helper.start_cpu_planning(namespace)
    assert calls == ["validate", "prepare", "capsule", "sync_failed"]
    assert not context["case"].exists()


def test_validation_failure_precedes_prepare_capsule_sync_input_and_fit(helper, monkeypatch):
    def failed_validation(ns):
        raise RuntimeError("SOURCE_PIN_MISMATCH_SOFTWARE_FIXTURE")
    monkeypatch.setattr(helper, "_validate", failed_validation)
    with patch.object(helper, "_prepare", side_effect=AssertionError("NO_PREPARE")), \
         patch.object(helper, "_capsule", side_effect=AssertionError("NO_CAPSULE")), \
         patch.object(helper, "_sync", side_effect=AssertionError("NO_SYNC")), \
         patch("builtins.input", side_effect=AssertionError("NO_INPUT")), \
         pytest.raises(RuntimeError, match="SOURCE_PIN_MISMATCH"):
        helper.start_cpu_planning({"SOURCE_COMMIT": "unrelated"})


def test_real_source_pin_validation_fails_before_files_network_or_prompts(helper, tmp_path):
    # Populate required globals only. Invalid pin must fail before touching any
    # source/case file, remote credential or author prompt.
    names = ("SOURCE", "SOURCE_COMMIT", "MANIFEST_SHA256", "SETTINGS", "PRIVATE", "PLANNING_CASE",
        "DEVELOPMENT_MAP", "repair_dir", "CM_REPAIR_COMMIT", "bootstrap", "run_cli", "remote_args",
        "REPOSITORY", "PLANNING_BRANCH", "PLANNING_PREFIX", "CPU_PLANNING_START_COMMIT",
        "CPU_PLANNING_START_SHA256", "CPU_PLANNING_START_BYTES")
    namespace = {name: None for name in names}
    namespace["SOURCE_COMMIT"] = "unrelated-source"
    with patch("builtins.input", side_effect=AssertionError("NO_INPUT")), \
         patch("subprocess.run", side_effect=AssertionError("NO_PROCESS")), \
         patch("getpass.getpass", side_effect=AssertionError("NO_SECRET")), \
         patch.object(Path, "mkdir", side_effect=AssertionError("NO_WRITES")), \
         pytest.raises(RuntimeError, match="pins originales"):
        helper._validate(namespace)


@pytest.mark.parametrize("field,value", [
    ("status", "SAVED_BUT_UNVERIFIED"), ("remote_verified", False),
    ("source_commit", "f" * 40), ("case_id", "ANOTHER_CASE"),
    ("plan_sha256", "c" * 64), ("commit", "main")])
def test_sync_rejects_any_unverified_or_misbound_remote_receipt(helper, tmp_path, field, value):
    context = dict(case=tmp_path / "case", plan={"case_id":"SOFTWARE_FIXTURE"}, plan_sha256="b" * 64)
    receipt = dict(status="SAVED_CASE_REMOTE_VERIFIED", backup=dict(remote_verified=True,
        source_commit=helper.S2_COMMIT, case_id="SOFTWARE_FIXTURE", plan_sha256="b" * 64, commit="a" * 40))
    if field == "status": receipt[field] = value
    else: receipt["backup"][field] = value
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return completed(receipt)
    namespace = dict(run_cli=run, remote_args=lambda *a, **k: [], DEVELOPMENT_MAP=tmp_path / "unused")
    with pytest.raises(RuntimeError, match="respaldo remoto"):
        helper._sync(namespace, context)
    assert calls == [(["sync"], {"remote":True})]
    assert not context["case"].exists()


def test_sync_accepts_only_complete_receipt_bound_to_exact_case_plan_and_source(helper, tmp_path):
    context = dict(case=tmp_path / "case", plan={"case_id":"SOFTWARE_FIXTURE"}, plan_sha256="b" * 64)
    receipt = dict(status="SAVED_CASE_REMOTE_VERIFIED", backup=dict(remote_verified=True,
        source_commit=helper.S2_COMMIT, case_id="SOFTWARE_FIXTURE", plan_sha256="b" * 64, commit="a" * 40))
    namespace = dict(run_cli=lambda *a, **k:completed(receipt),
        remote_args=lambda *a, **k:[], DEVELOPMENT_MAP=tmp_path / "unused")
    assert helper._sync(namespace, context) == receipt


def test_existing_exact_plan_is_reused_without_rewrite_or_another_prepare(helper, tmp_path):
    case = tmp_path / "case"; case.mkdir()
    admissions = {p:dict(admission_sha256=chr(97+i) * 64) for i,p in enumerate(helper.PANELS)}
    plan = dict(stage="planning", source_commit=helper.S2_COMMIT, source_manifest_sha256=helper.S2_MANIFEST,
        original_case_root=str(case), software_fixture=False, test_opened=False,
        jobs=list(range(340)), seeds=list(range(5)),
        data_panels=[dict(id=p, data_sha256=admissions[p]["admission_sha256"]) for p in helper.PANELS])
    path = case / "plan.json"; path.write_text(json.dumps(plan))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    context = dict(case=case, source=tmp_path / "unused-source", admissions=admissions)
    namespace = dict(run_cli=lambda *a, **k:pytest.fail("NO_ANOTHER_PREPARE"))
    assert helper._prepare(namespace, context) == plan
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    assert context["plan_sha256"] == hashlib.sha256(helper._canonical(plan)).hexdigest()


def test_capsule_roundtrips_exact_overlay_receipts_own_code_and_is_idempotent(helper, tmp_path):
    bundle, private, case = (tmp_path / key for key in ("bundle", "private", "case"))
    originals = {}
    for relative, raw in (("ops/REPAIR_MANIFEST.json", b'{"SYNTHETIC":"manifest"}\n'),
                          ("parser/cm_header_parser.py", b"# synthetic parser bytes\n")):
        path = bundle / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        originals["addon/" + relative] = raw
    receipt = private / "operational-repairs" / helper.REPAIR_VERSION / "completed-software-fixture.json"
    receipt.parent.mkdir(parents=True); receipt.write_bytes(b'{"SYNTHETIC":"not a human signature"}\n')
    originals["operational_receipts/" + receipt.name] = receipt.read_bytes()
    case.mkdir(); plan_path = case / "plan.json"; plan_path.write_bytes(b'{"SYNTHETIC":"plan unchanged"}\n')
    before = (plan_path.read_bytes(), plan_path.stat().st_mtime_ns)
    own_bytes = (ROOT / "start_cpu_planning.py").read_bytes()
    namespace = dict(CPU_PLANNING_START_COMMIT="d" * 40,
        CPU_PLANNING_START_SHA256=hashlib.sha256(own_bytes).hexdigest(), CPU_PLANNING_START_BYTES=own_bytes)
    context = dict(bundle=bundle, private=private, case=case, receipts=[receipt],
        cm_manifest=dict(files={"parser/cm_header_parser.py":hashlib.sha256(originals["addon/parser/cm_header_parser.py"]).hexdigest()}),
        plan={"case_id":"SOFTWARE_FIXTURE"}, plan_sha256="b" * 64)
    first = helper._capsule(namespace, context)
    first_bytes, first_mtime = first.read_bytes(), first.stat().st_mtime_ns
    second = helper._capsule(namespace, context)
    assert second == first and second.read_bytes() == first_bytes and second.stat().st_mtime_ns == first_mtime
    value = json.loads(first_bytes)
    for name, expected in originals.items():
        bound = value["files"][name]
        decoded = base64.b64decode(bound["base64"], validate=True)
        assert decoded == expected and bound["bytes"] == len(expected)
        assert bound["sha256"] == hashlib.sha256(expected).hexdigest()
    helper_payloads = [item for name,item in value["files"].items() if "start_cpu_planning.py" in name]
    assert helper_payloads and any(base64.b64decode(item["base64"], validate=True) == own_bytes for item in helper_payloads)
    assert all(not ({"raw", "inputs", "datasets", "tokens", "secrets"} & set(Path(name).parts))
               for name in value["files"])
    assert before == (plan_path.read_bytes(), plan_path.stat().st_mtime_ns)
    assert value["new_fits_by_packaging"] == 0 and value["test_opened"] is False


def test_capsule_rejects_symlink_parent_before_creating_outside_paths(helper, tmp_path):
    bundle, private, case, outside = (tmp_path / key for key in ("bundle", "private", "case", "outside"))
    manifest = bundle / "ops/REPAIR_MANIFEST.json"; manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b'{"SYNTHETIC":"manifest"}\n')
    case.mkdir(); outside.mkdir()
    sentinel = outside / "sentinel.txt"; sentinel.write_bytes(b"UNCHANGED")
    (case / "receipts").symlink_to(outside, target_is_directory=True)
    own_bytes = (ROOT / "start_cpu_planning.py").read_bytes()
    namespace = dict(CPU_PLANNING_START_COMMIT="d" * 40, CPU_PLANNING_START_BYTES=own_bytes,
        CPU_PLANNING_START_SHA256=hashlib.sha256(own_bytes).hexdigest())
    context = dict(bundle=bundle, private=private, case=case, receipts=[], cm_manifest=dict(files={}),
        plan={"case_id":"SOFTWARE_FIXTURE"}, plan_sha256="b" * 64)
    with pytest.raises(RuntimeError, match="cápsula alterada"):
        helper._capsule(namespace, context)
    assert sentinel.read_bytes() == b"UNCHANGED" and list(outside.iterdir()) == [sentinel]


def test_original_storage_allows_json_capsule_but_rejects_standalone_python_or_raw():
    source = ROOT.parent / "e3_hypothesis_adequacy_20261122/runtime/engine/storage.py"
    spec = importlib.util.spec_from_file_location("original_s2_storage_fixture", source)
    storage = importlib.util.module_from_spec(spec); spec.loader.exec_module(storage)
    store = storage.GithubStore.__new__(storage.GithubStore)
    assert store._eligible("receipts/operational_addons/cm-header-" + "a" * 64 + ".json") is True
    assert store._eligible("receipts/operational_addons/start_cpu_planning.py") is False
    for name in ("raw/market.zip", "receipts/raw/archive.json", "inputs/owner.json", "secrets/private.json"):
        assert store._eligible(name) is False
