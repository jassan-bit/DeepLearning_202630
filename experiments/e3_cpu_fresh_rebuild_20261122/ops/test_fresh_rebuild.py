"""Offline fresh-session recovery fixtures; no acquisition, fit or remote calls."""
from __future__ import annotations

import datetime as dt
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parent


def actor_pair_digest(name, contact):
    normalized = dict(actor_name=" ".join(name.split()).casefold(), actor_contact=contact.strip().casefold())
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@pytest.fixture
def helper(monkeypatch):
    spec = importlib.util.spec_from_file_location("fresh_rebuild_software_fixture", ROOT / "fresh_rebuild.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The real prior actor's PII is neither available to fixtures nor published.
    # Pin a synthetic actor pair only inside this isolated unit-test instance.
    monkeypatch.setattr(module, "PRIOR_ACTOR_PAIR_DIGEST", actor_pair_digest(
        "SOFTWARE FIXTURE PERSON", "fixture@example.invalid"))
    return module


def callback_fixture():
    namespace = {"SOFTWARE_FIXTURE": True, "PRIVATE":Path("/__fresh_software_fixture_no_exports__")}
    states = {p: {"admitted":False} for p in ("F00", "F01", "F02", "F03")}
    calls = []
    def backup(ns, stage):
        assert ns is namespace
        calls.append(("backup", stage))
        return dict(status="SOFTWARE_FIXTURE_BACKED_UP", remote_verified=True)
    def worker(ns, operation, panel=None):
        assert ns is namespace
        calls.append(("worker", operation, panel))
        if operation in ("acquire_prepare", "reacquire"):
            assert panel in states
            states[panel] = dict(admitted=True)
        return {"panels": {p:dict(s) for p,s in states.items()}}
    return namespace, states, calls, backup, worker


def test_backup_precedes_collection_and_follows_each_admission(helper):
    ns, states, calls, backup, worker = callback_fixture()
    with patch("builtins.input", side_effect=AssertionError("NO_REPEATED_ACCEPTANCE")), \
         patch("getpass.getpass", side_effect=AssertionError("NO_SECRET_IN_OPERATIONAL_CALLBACKS")):
        result = helper.run_development(ns, backup, worker)
    assert calls == [("worker", "record", None), ("worker", "check", None),
        ("backup", "before_acquisition"),
        ("worker", "acquire_prepare", "F00"), ("backup", "after_admission_F00"),
        ("worker", "acquire_prepare", "F01"), ("backup", "after_admission_F01"),
        ("worker", "acquire_prepare", "F02"), ("backup", "after_admission_F02"),
        ("worker", "acquire_prepare", "F03"), ("backup", "after_admission_F03"),
        ("worker", "check", None)]
    assert all(s["admitted"] for s in result["panels"].values())


def test_backup_failure_before_acquisition_stops_all_provider_operations(helper):
    ns, states, calls, backup, worker = callback_fixture()
    def failed_backup(ns, stage):
        calls.append(("backup_failed", stage))
        raise RuntimeError("SOFTWARE_FIXTURE_REMOTE_FAILURE")
    with pytest.raises(RuntimeError, match="REMOTE_FAILURE"):
        helper.run_development(ns, failed_backup, worker)
    assert calls == [("worker", "record", None), ("worker", "check", None),
                     ("backup_failed", "before_acquisition")]
    assert all(not s["admitted"] for s in states.values())


def test_failure_to_back_up_first_admission_stops_next_panel(helper):
    ns, states, calls, backup, worker = callback_fixture()
    def backup_once(ns, stage):
        calls.append(("backup", stage))
        if stage == "after_admission_F00":
            raise RuntimeError("SOFTWARE_FIXTURE_POST_ADMISSION_BACKUP_FAILED")
        return dict(remote_verified=True)
    with pytest.raises(RuntimeError, match="POST_ADMISSION_BACKUP_FAILED"):
        helper.run_development(ns, backup_once, worker)
    assert states["F00"]["admitted"] is True
    assert all(not states[p]["admitted"] for p in ("F01", "F02", "F03"))
    assert calls[-1] == ("backup", "after_admission_F00")


def declaration_namespace(tmp_path):
    source = ROOT.parents[1] / "e3_hypothesis_adequacy_20261122/runtime"
    private = tmp_path / "private"
    proposals = json.loads((source / "configs/data_panels.json").read_text())["panels"]
    template = json.loads((source / "docs/OWNER_EXPOSURE_FORM.json").read_text())
    definitions = []
    for proposal in proposals:
        panel = proposal["id"]
        folder = private / "inputs" / panel; folder.mkdir(parents=True)
        license_path = source / proposal["rights_snapshot_path"]
        license_sha = hashlib.sha256(license_path.read_bytes()).hexdigest()
        owner = dict(template, panel_ids=[panel], reserve_mode="development_only",
            source_url=proposal["rights_source_url"], source_license_evidence_path=str(license_path),
            source_license_evidence_sha256=license_sha,
            dataset_scope=dict(panel=panel, provider=proposal["provider"], instrument=proposal["instrument"],
                start=proposal["proposed_development_start"], end=proposal["proposed_development_end"]))
        (folder / "owner-development.json").write_text(json.dumps(owner))
        (folder / "rights-DRAFT.json").write_text(json.dumps(dict(status="draft",
            evidence_path=str(license_path), evidence_sha256=license_sha,
            source_url=proposal["rights_source_url"], license=proposal["license"])))
        definitions.append(dict(panel=panel, proposal=proposal, folder=folder,
            raw=private / "raw" / panel / "development-v1",
            panel_dir=private / "panels" / panel / "development-v1"))
    return dict(SOURCE=source, PRIVATE=private, definitions=definitions)


def test_reused_authorization_has_truthful_current_record_and_no_new_signature(helper, tmp_path):
    namespace = declaration_namespace(tmp_path)
    proof_hash = helper.PRIOR_REPORT_SHA
    identity = dict(actor_name="SOFTWARE FIXTURE PERSON", actor_contact="fixture@example.invalid", actor_role="fixture")
    before = dt.datetime.now(dt.timezone.utc)
    with patch("builtins.input", side_effect=AssertionError("NO_ACEPTO_PROMPT")), \
         patch("getpass.getpass", side_effect=AssertionError("IDENTITY_ALREADY_SUPPLIED")):
        helper.create_declarations(namespace, proof_hash, identity_provider=lambda:identity)
    after = dt.datetime.now(dt.timezone.utc)
    for row in namespace["definitions"]:
        owner = json.loads((row["folder"] / "owner-development.json").read_text())
        assert owner["kind"] == "prior_authorization_reused_during_user_requested_rebuild"
        assert owner["AI_completed_signature"] is False
        assert owner["original_statement_utc"] is None
        assert "reconstruction" in owner["statement_utc_semantics"].lower()
        assert before <= dt.datetime.fromisoformat(owner["statement_utc"]) <= after
        assert proof_hash in json.dumps(owner)
        assert owner["actor_name"] == identity["actor_name"]
        assert owner["actor_contact"] == identity["actor_contact"]
        assert owner["authorize_test_preparation"] is False
        assert owner["metadata_reconstruction_utc"] == owner["statement_utc"]
        assert owner["statement_time_semantics"] == "current_user_requested_rebuild_record_not_original_consent_time"
        assert owner["prior_report_bytes_available_in_runtime"] is False
        cutoff = dt.datetime.fromisoformat(owner["observation_cutoff_utc"])
        assert cutoff.date().isoformat() == row["proposal"]["proposed_development_end"]
        assert (cutoff.hour, cutoff.minute, cutoff.second, cutoff.microsecond) == (23, 59, 59, 999000)
        assert cutoff.tzinfo is not None
        assert owner["observation_cutoff_semantics"] == "authorized_development_data_boundary_not_proof_of_historical_access"


def test_existing_valid_rebuild_declarations_are_byte_immutable_and_need_no_identity(helper, tmp_path):
    namespace = declaration_namespace(tmp_path)
    identity = dict(actor_name="SOFTWARE FIXTURE PERSON", actor_contact="fixture@example.invalid")
    helper.create_declarations(namespace, helper.PRIOR_REPORT_SHA, identity_provider=lambda:identity)
    files = [row["folder"] / name for row in namespace["definitions"]
             for name in ("owner-development.json", "rights.json")]
    before = {str(path):(path.read_bytes(), path.stat().st_mtime_ns) for path in files}
    with patch("builtins.input", side_effect=AssertionError("NO_REPEATED_ACCEPTANCE")), \
         patch("getpass.getpass", side_effect=AssertionError("NO_IDENTITY_REPROMPT")):
        helper.create_declarations(namespace, helper.PRIOR_REPORT_SHA,
            identity_provider=lambda:pytest.fail("NO_IDENTITY_PROVIDER_ON_RESUME"))
    assert before == {str(path):(path.read_bytes(), path.stat().st_mtime_ns) for path in files}


def test_unknown_identity_is_requested_only_through_private_fields_without_acceptance(helper, tmp_path, capsys, monkeypatch):
    namespace = declaration_namespace(tmp_path)
    identity = ["PRIVATE FIXTURE IDENTITY", "private-fixture@example.invalid"]
    monkeypatch.setattr(helper, "PRIOR_ACTOR_PAIR_DIGEST", actor_pair_digest(*identity))
    with patch.dict(sys.modules, {"google.colab":None}), \
         patch("builtins.input", side_effect=AssertionError("NO_ACEPTO_OR_PUBLIC_INPUT")), \
         patch("getpass.getpass", side_effect=identity) as private_prompt:
        helper.create_declarations(namespace, helper.PRIOR_REPORT_SHA)
    assert private_prompt.call_count == 2
    output = capsys.readouterr()
    assert all(value not in output.out + output.err for value in identity)
    for row in namespace["definitions"]:
        owner = json.loads((row["folder"] / "owner-development.json").read_text())
        assert owner["actor_name"] == identity[0] and owner["actor_contact"] == identity[1]
        assert owner["newly_obtained_authorization"] is False
        assert owner["prior_authorization_reused"] is True
        assert owner["human_identity_cryptographically_verified"] is False


def test_changed_prior_evidence_hash_stops_before_identity_or_new_records(helper, tmp_path):
    namespace = declaration_namespace(tmp_path)
    files = [row["folder"] / "owner-development.json" for row in namespace["definitions"]]
    before = {str(path):path.read_bytes() for path in files}
    with patch("getpass.getpass", side_effect=AssertionError("NO_PRIVATE_IDENTITY")), \
         pytest.raises(RuntimeError, match="autorización previa fijada"):
        helper.create_declarations(namespace, "0" * 64)
    assert before == {str(path):path.read_bytes() for path in files}
    assert not any((row["folder"] / "rights.json").exists() for row in namespace["definitions"])


def test_existing_declaration_with_unrelated_provenance_is_preserved_and_rejected(helper, tmp_path):
    namespace = declaration_namespace(tmp_path)
    path = namespace["definitions"][0]["folder"] / "owner-development.json"
    owner = json.loads(path.read_text())
    owner.update(actor_name="OTHER RECORD", kind="unrelated-old-declaration")
    path.write_text(json.dumps(owner))
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    with patch("builtins.input", side_effect=AssertionError("NO_PROMPT")), \
         patch("getpass.getpass", side_effect=AssertionError("NO_IDENTITY")), \
         pytest.raises(RuntimeError, match="otra reconstrucción"):
        helper.create_declarations(namespace, helper.PRIOR_REPORT_SHA)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_existing_admissions_are_not_downloaded_again(helper):
    ns, states, calls, backup, worker = callback_fixture()
    states["F00"]["admitted"] = states["F01"]["admitted"] = True
    helper.run_development(ns, backup, worker)
    assert ("worker", "acquire_prepare", "F00") not in calls
    assert ("worker", "acquire_prepare", "F01") not in calls
    assert ("worker", "acquire_prepare", "F02") in calls and ("worker", "acquire_prepare", "F03") in calls


def test_mismatching_prior_actor_pair_preserves_all_forms_before_any_write(helper, tmp_path):
    namespace = declaration_namespace(tmp_path)
    paths = [p for row in namespace["definitions"] for p in row["folder"].iterdir()]
    before = {str(path):(path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    identity = dict(actor_name="UNRELATED SYNTHETIC PERSON", actor_contact="other@example.invalid")
    with patch.object(Path, "mkdir", side_effect=AssertionError("NO_MKDIR_ON_ACTOR_MISMATCH")), \
         patch("builtins.input", side_effect=AssertionError("NO_ACCEPTANCE")), \
         pytest.raises(RuntimeError, match="no coinciden"):
        helper.create_declarations(namespace, helper.PRIOR_REPORT_SHA, identity_provider=lambda:identity)
    assert before == {str(path):(path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert not any((row["folder"] / "rights.json").exists() for row in namespace["definitions"])


@pytest.mark.parametrize("operation", ["run", "prepare-final", "evaluate"])
def test_fresh_worker_refuses_other_operations_in_isolated_process_before_source(tmp_path, operation):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "fresh_worker.py")],
        input=json.dumps(dict(operation=operation, source=str(tmp_path / "absent-source"))),
        text=True, capture_output=True, cwd=tmp_path,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=10)
    assert result.returncode == 1
    event = json.loads(result.stdout)
    assert event["status"] == "fresh_worker_failed"
    assert event["new_market_fits"] == 0 and event["test_opened"] is False
    assert not (tmp_path / "absent-source").exists()


@pytest.mark.parametrize("unverified", [None, {}, {"remote_verified":False}])
def test_unverified_backup_callback_return_blocks_acquisition(helper, unverified):
    ns, states, calls, backup, worker = callback_fixture()
    with pytest.raises(RuntimeError, match="Verificar respaldo"):
        helper.run_development(ns, lambda *a:unverified, worker)
    assert calls == [("worker", "record", None), ("worker", "check", None)]
    assert all(not s["admitted"] for s in states.values())


def initialize_book_fixture(helper, monkeypatch):
    """Only synthetic code executes; the production hash remains untouched."""
    codes = ["from pathlib import Path\nevents.append('setup0')\n"
             + "SOURCE_COMMIT = " + repr(helper.S2_COMMIT) + "\n"
             + "MANIFEST_SHA256 = " + repr(helper.S2_MANIFEST) + "\n",
             "events.append('setup1')\n",
             "PREPARE_CRYPTO_DEVELOPMENT = False\nevents.append('metadata')\n"
             + "definitions = " + repr([dict(panel=p) for p in helper.PANELS]) + "\n"]
    book = dict(metadata=dict(e3_source_pins=dict(source_commit=helper.S2_COMMIT,
        manifest_sha256=helper.S2_MANIFEST)), cells=[dict(cell_type="code", source=s) for s in codes])
    raw = json.dumps(book).encode()
    original_hash = helper._fresh_sha
    monkeypatch.setattr(helper, "_fresh_sha", lambda value:helper.T2_SHA if value == raw else original_hash(value))
    return raw


def test_initialization_runs_pre_metadata_callback_after_setup_before_metadata(helper, monkeypatch):
    raw = initialize_book_fixture(helper, monkeypatch)
    namespace = dict(events=[])
    def before_metadata(ns):
        assert ns is namespace and ns["SOURCE_COMMIT"] == helper.S2_COMMIT
        assert "definitions" not in ns
        ns["events"].append("callback")
    assert helper.initialize(namespace, raw, before_metadata_callback=before_metadata) is namespace
    assert namespace["events"] == ["setup0", "setup1", "callback", "metadata"]
    assert tuple(row["panel"] for row in namespace["definitions"]) == helper.PANELS


def test_failed_pre_metadata_callback_blocks_metadata_declarations_and_acquisition(helper, monkeypatch):
    raw = initialize_book_fixture(helper, monkeypatch)
    namespace = dict(events=[])
    def fail_before_metadata(ns):
        ns["events"].append("callback_failed")
        raise RuntimeError("SOFTWARE_FIXTURE_PRE_METADATA_VALIDATION_FAILED")
    with patch.object(helper, "create_declarations", side_effect=AssertionError("NO_DECLARATIONS")), \
         patch.object(helper, "run_development", side_effect=AssertionError("NO_ACQUISITION")), \
         pytest.raises(RuntimeError, match="PRE_METADATA_VALIDATION_FAILED"):
        helper.fresh_rebuild(namespace, raw, backup_callback=lambda *a:pytest.fail("NO_BACKUP_OR_MARKET"),
            before_metadata_callback=fail_before_metadata)
    assert namespace["events"] == ["setup0", "setup1", "callback_failed"]
    assert "definitions" not in namespace


def test_cached_initialization_still_validates_callback_without_reexecuting_cells(helper, monkeypatch, tmp_path):
    raw = initialize_book_fixture(helper, monkeypatch)
    namespace = dict(events=[], SOURCE_COMMIT=helper.S2_COMMIT, MANIFEST_SHA256=helper.S2_MANIFEST,
        SETTINGS=dict(source_commit=helper.S2_COMMIT), SOURCE=tmp_path,
        definitions=[dict(panel=p) for p in helper.PANELS], run_cli=object(), remote_args=object(),
        bootstrap=object(), PRIVATE=tmp_path / "private-unused")
    result = helper.initialize(namespace, raw, before_metadata_callback=lambda ns:ns["events"].append("cached_callback"))
    assert result is namespace and namespace["events"] == ["cached_callback"]
    assert not namespace["PRIVATE"].exists()


def test_saved_recovery_export_selects_reacquisition_only_after_verified_backup(helper, tmp_path):
    ns, states, calls, backup, worker = callback_fixture()
    ns["PRIVATE"] = tmp_path / "private"
    exports = ns["PRIVATE"] / "recovery-exports"; exports.mkdir(parents=True)
    (exports / "F00.json").write_bytes(b'{"SOFTWARE_FIXTURE":"recovery metadata"}\n')
    helper.run_development(ns, backup, worker)
    assert calls[:4] == [("worker", "record", None), ("worker", "check", None),
                         ("backup", "before_acquisition"), ("worker", "reacquire", "F00")]
    assert ("worker", "acquire_prepare", "F00") not in calls
    assert ("worker", "acquire_prepare", "F01") in calls
    assert calls[4] == ("backup", "after_admission_F00")


def test_recovery_export_does_not_bypass_failed_pre_acquisition_backup(helper, tmp_path):
    ns, states, calls, backup, worker = callback_fixture()
    ns["PRIVATE"] = tmp_path / "private"
    exports = ns["PRIVATE"] / "recovery-exports"; exports.mkdir(parents=True)
    (exports / "F00.json").write_bytes(b'{"SOFTWARE_FIXTURE":"metadata"}\n')
    with pytest.raises(RuntimeError, match="Verificar respaldo"):
        helper.run_development(ns, lambda *a:dict(remote_verified=False), worker)
    assert calls == [("worker", "record", None), ("worker", "check", None)]
    assert all(not s["admitted"] for s in states.values())


def reacquire_fixture(tmp_path):
    spec = importlib.util.spec_from_file_location("fresh_reacquire_fixture", ROOT / "fresh_worker.py")
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    private = tmp_path / "private"
    exported = private / "recovery-exports/F02.json"; exported.parent.mkdir(parents=True)
    exported.write_bytes(b'{"SYNTHETIC":"recovery export bytes only"}\n')
    row = dict(panel="F02", panel_dir=str(private / "panels/F02/development-v1"),
        raw=str(private / "raw/F02/development-v1"),
        proposal=dict(provider="binance_vision_cm", instrument="BTCUSD_PERP"))
    bundle = dict(original_panel_dir=row["panel_dir"], original_raw_dir=row["raw"])
    manifest = dict(row["proposal"])
    def require(condition, message):
        if not condition: raise RuntimeError(message)
    replay = Mock(return_value=dict(status="SOFTWARE_FIXTURE_RECOVERED"))
    read_bundle = Mock(return_value=(exported.read_bytes(), bundle, b"", b"", manifest, {}))
    data = SimpleNamespace(require=require, _read_recovery_bundle=read_bundle, reacquire_development=replay)
    return worker, private, exported, row, bundle, manifest, data


def test_reacquire_dispatch_binds_actual_export_hash_original_paths_and_source(tmp_path):
    worker, private, exported, row, bundle, manifest, data = reacquire_fixture(tmp_path)
    expected = hashlib.sha256(exported.read_bytes()).hexdigest()
    assert worker.reacquire_panel(data, row, private) == dict(status="SOFTWARE_FIXTURE_RECOVERED")
    data._read_recovery_bundle.assert_called_once_with(exported, expected)
    data.reacquire_development.assert_called_once_with(exported, row["panel_dir"], row["raw"], private.parent,
        enable_market_acquisition=True, expected_bundle_sha256=expected)
    assert not Path(row["panel_dir"]).exists() and not Path(row["raw"]).exists()


@pytest.mark.parametrize("field", ["original_panel_dir", "original_raw_dir", "provider", "instrument"])
def test_reacquire_rejects_changed_export_paths_or_source_before_provider_io(tmp_path, field):
    worker, private, exported, row, bundle, manifest, data = reacquire_fixture(tmp_path)
    if field in bundle: bundle[field] = str(tmp_path / "foreign-case")
    else: manifest[field] = "UNRELATED_SOURCE"
    with pytest.raises(RuntimeError, match="differs from fixed panel"):
        worker.reacquire_panel(data, row, private)
    data.reacquire_development.assert_not_called()


def test_reacquire_rejects_symlink_export_before_reading_bundle_or_provider_io(tmp_path):
    worker, private, exported, row, bundle, manifest, data = reacquire_fixture(tmp_path)
    outside = tmp_path / "outside.json"; outside.write_bytes(exported.read_bytes())
    exported.unlink(); exported.symlink_to(outside)
    with pytest.raises(RuntimeError, match="authentic metadata recovery export"):
        worker.reacquire_panel(data, row, private)
    data._read_recovery_bundle.assert_not_called()
    data.reacquire_development.assert_not_called()


def replay_companions_fixture(tmp_path):
    worker, private, exported, row, bundle, manifest, data = reacquire_fixture(tmp_path)
    source = ROOT.parents[1] / "e3_hypothesis_adequacy_20261122/runtime/engine/data.py"
    spec = importlib.util.spec_from_file_location("replay_original_metadata_primitives", source)
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    raw = Path(row["raw"]); raw.mkdir(parents=True)
    name = "BTCUSD_PERP-1d-2021-01.zip"
    archive_bytes = b"SYNTHETIC_ARCHIVE_BYTES_VERIFIED_BY_HASH_NOT_A_MARKET_DOWNLOAD"
    (raw / name).write_bytes(archive_bytes)
    events = raw / "reacquisition_events.jsonl"; events.write_bytes(b'{"synthetic":"new replay event"}\n')
    bundle["sources"] = [dict(path=name, sha256=hashlib.sha256(archive_bytes).hexdigest(), bytes=len(archive_bytes))]
    manifest["files"] = [dict(path=name, sha256=hashlib.sha256(archive_bytes).hexdigest())]
    manifest_bytes = json.dumps(manifest).encode()
    bundle["raw_manifest"] = dict(path=str(raw / "raw_manifest.json"),
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        content_base64=base64.b64encode(manifest_bytes).decode())
    companions = {
        "acquisition_pending.json": b'{"synthetic":"authentic pending metadata"}\n',
        name + ".CHECKSUM": (hashlib.sha256(archive_bytes).hexdigest() + "  " + name + "\n").encode(),
        "acquisition_attempts.jsonl": b'{"synthetic":"old immutable acquisition log"}\n',
    }
    bundle["raw_metadata_companions"] = [dict(relative_path=n, path=str(raw / n),
        sha256=hashlib.sha256(value).hexdigest(), content_base64=base64.b64encode(value).decode())
        for n,value in companions.items()]
    folder = private / "inputs/F02"; folder.mkdir(parents=True)
    row["folder"] = str(folder)
    for n,value in (("acquisition.json",dict(provider="binance_vision_cm",instrument="BTCUSD_PERP")),
                    ("rights.json",dict(SYNTHETIC="rights")),
                    ("exposure-development.json",dict(SYNTHETIC="exposure"))):
        (folder / n).write_text(json.dumps(value))
    data._safe_path = original._safe_path
    data._decoded_metadata = original._decoded_metadata
    data._restore_metadata_file = Mock(wraps=original._restore_metadata_file)
    data.inspect_binance_pending = Mock(return_value=dict(status="VALIDATED_PARTIAL_RAW_NOT_ADMITTED"))
    data._network_get = Mock(side_effect=AssertionError("NO_NETWORK_RESTORING_METADATA_COMPANIONS"))
    return worker, private, row, bundle, data, raw, name, archive_bytes, events, companions


def test_interrupted_raw_replay_restores_only_authentic_companions_and_is_idempotent(tmp_path):
    worker, private, row, bundle, data, raw, name, archive_bytes, events, companions = replay_companions_fixture(tmp_path)
    events_before = (events.read_bytes(), events.stat().st_mtime_ns)
    worker._restore_replay_companions(data, row, private)
    assert all((raw / n).read_bytes() == value for n,value in companions.items())
    assert (raw / name).read_bytes() == archive_bytes
    assert events_before == (events.read_bytes(), events.stat().st_mtime_ns)
    data.inspect_binance_pending.assert_called_once()
    assert not Path(row["panel_dir"]).exists()
    data.reacquire_development.assert_not_called(); data._network_get.assert_not_called()
    before = {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in raw.iterdir()}
    writes = data._restore_metadata_file.call_count
    worker._restore_replay_companions(data, row, private)
    assert before == {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in raw.iterdir()}
    assert data._restore_metadata_file.call_count == writes
    data.reacquire_development.assert_not_called(); data._network_get.assert_not_called()


def test_wrong_interrupted_zip_blocks_companion_writes_and_network(tmp_path):
    worker, private, row, bundle, data, raw, name, archive_bytes, events, companions = replay_companions_fixture(tmp_path)
    (raw / name).write_bytes(b"MUTATED_SYNTHETIC_ARCHIVE")
    before = {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in raw.iterdir()}
    with pytest.raises((RuntimeError, ValueError)):
        worker._restore_replay_companions(data, row, private)
    assert before == {str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in raw.iterdir()}
    data._restore_metadata_file.assert_not_called(); data.inspect_binance_pending.assert_not_called()
    data.reacquire_development.assert_not_called(); data._network_get.assert_not_called()
