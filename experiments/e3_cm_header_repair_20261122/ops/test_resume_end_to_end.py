"""Generated daily bars and fixture declarations only; never a market result."""
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import subprocess
from unittest.mock import patch
import zipfile

import pytest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "e3_hypothesis_adequacy_20261122" / "runtime"
if not SOURCE.exists():
    SOURCE = Path("/workspace/e3_hypothesis_adequacy_20261122/runtime")
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))
from engine import data
from package import crypto_development as helper, crypto_notebook_template as nb
import resume_worker as worker


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture_archive(item, cm=False, legacy=False):
    first = dt.date.fromisoformat(item["first_session"])
    last = dt.date.fromisoformat(item["last_session"])
    if legacy:
        header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
    else:
        header = "open_time,open,high,low,close,volume,close_time," + ("base_volume" if cm else "quote_volume") + ",count,taker_buy_volume,taker_buy_base_volume,ignore"
    lines = [header]
    for offset in range((last-first).days + 1):
        day = first + dt.timedelta(days=offset)
        start = int(dt.datetime.combine(day, dt.time(), dt.timezone.utc).timestamp() * 1000)
        close = 100 + .008 * day.toordinal() + math.sin(day.toordinal() / 11)
        lines.append(f"{start},{close},{close+2},{close-2},{close},123,{start+86400000-1},4.5,42,10,1.2,0")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(item["path"].replace(".zip", ".csv"), "\n".join(lines) + "\n")
    return output.getvalue()


def setup_case(tmp_path):
    # Execute metadata-only cells, with generated identifiers. This does not
    # authenticate a person and is never published as actual consent.
    private = tmp_path / "private"; private.mkdir()
    scope = dict(Path=Path, json=json, hashlib=hashlib, SOURCE=SOURCE,
        PRIVATE=private, WORK=tmp_path, SOURCE_COMMIT=worker.SOURCE_COMMIT)
    cells = [c["source"] for c in nb.notebook()["cells"] if c["cell_type"] == "code"]
    import subprocess
    from types import SimpleNamespace
    import bootstrap_colab
    scope.update(subprocess=subprocess, SETTINGS=dict(python=sys.executable, source_root=str(SOURCE), source_commit=worker.SOURCE_COMMIT),
                 bootstrap=SimpleNamespace(cpu_env=bootstrap_colab.cpu_env))
    with patch("builtins.print"):
        exec(compile(cells[1], "generated_metadata", "exec"), scope)
        exec(compile(next(s for s in cells if s.startswith("PREPARE_CRYPTO_DEVELOPMENT =")), "generated_metadata", "exec"), scope)
    rows = worker.rows_for_source(SOURCE, private)
    for row in rows:
        folder, p = Path(row["folder"]), row["proposal"]
        owner = dict(row["blank_owner"], kind="SOFTWARE_FIXTURE_NOT_A_REAL_AUTHOR_DECLARATION",
            actor_name="SOFTWARE FIXTURE ONLY", actor_contact="fixture@example.invalid", actor_role="fixture",
            statement_utc="2026-11-22T00:00:00+00:00", observation_cutoff_utc="2026-11-22T00:00:00+00:00",
            evidence_origin="Generated fixture, never a human signature", acknowledge_prior_D00_D03_exposed=True,
            all_data_declared_exposed_development=True, source_terms_accepted_by_real_actor=True,
            allowed_processing_scope=helper.SCOPE, allow_derived_publication=True,
            allow_raw_redistribution=False, AI_completed_signature=False, authorize_test_preparation=False)
        put(folder / "owner-development.json", owner)
        rights = dict(status="approved", actor_name=owner["actor_name"], actor_contact=owner["actor_contact"],
            evidence_path=owner["source_license_evidence_path"], evidence_sha256=owner["source_license_evidence_sha256"],
            scope=helper.SCOPE, source_url=p["rights_source_url"], license=p["license"],
            allow_derived_publication=True, allow_raw_redistribution=False,
            statement_utc=owner["statement_utc"], decision_origin="Generated software fixture")
        put(folder / "rights.json", rights)
        data.record_owner_evidence(folder / "owner-development.json", folder / "owner-development.json",
                                   folder / "exposure-development.json")
    return private, rows


def digest_tree(root):
    return {str(p): (data.sha256_bytes(p.read_bytes()), p.stat().st_mtime_ns)
            for p in Path(root).rglob("*") if p.is_file()}


def test_resume_partial_cm_and_new_um_preserves_originals_and_is_idempotent(tmp_path, monkeypatch):
    parser_path = HERE.parent / "parser/cm_header_parser.py"
    if not parser_path.exists():
        pytest.fail("Pinned parser extension must be present for the end-to-end repair test")
    overlay = worker.load(parser_path, "fixture_cm_overlay")
    private, rows = setup_case(tmp_path)
    responses = {}
    for row in rows:
        for item in data.binance_archive_plan(worker.read(Path(row["folder"]) / "acquisition.json")):
            payload = fixture_archive(item, cm=row["panel"] == "F02", legacy=row["panel"] == "F02" and "2022" in item["path"])
            responses[item["url"]] = payload
            responses[item["url"] + ".CHECKSUM"] = (hashlib.sha256(payload).hexdigest() + "  " + item["path"] + "\n").encode()
    fetched = []
    def get(url):
        fetched.append(url)
        return responses[url]
    monkeypatch.setattr(data, "_network_get", get)
    for row in rows[:2]:
        folder = Path(row["folder"])
        rights, exposure = (worker.read(folder / name) for name in ("rights.json", "exposure-development.json"))
        data.collect_development(worker.read(folder / "acquisition.json"), row["raw"], rights_receipt=rights,
                                 exposure_receipt=exposure, enable_market_acquisition=True)
        data.prepare_development(Path(row["raw"]) / "raw_manifest.json", worker.read(folder / "development-split.json"),
                                 row["panel_dir"], rights=rights, exposure=exposure)
    row = rows[2]; folder, raw = Path(row["folder"]), Path(row["raw"]); raw.mkdir(parents=True)
    config, rights, exposure = (worker.read(folder / name) for name in ("acquisition.json", "rights.json", "exposure-development.json"))
    put(raw / "acquisition_pending.json", data._pending_record(config, rights, exposure, "2026-11-22T00:00:00+00:00"))
    for item in data.binance_archive_plan(config)[:12]:
        (raw / item["path"]).write_bytes(responses[item["url"]])
        (raw / (item["path"] + ".CHECKSUM")).write_bytes(responses[item["url"] + ".CHECKSUM"])
    pending_before = (raw / "acquisition_pending.json").read_bytes()
    previous = {str(Path(r[k])): digest_tree(r[k]) for r in rows[:2] for k in ("raw", "panel_dir", "folder")}
    all_owner_forms = {str(Path(r["folder"])): digest_tree(r["folder"]) for r in rows}
    original_source = digest_tree(SOURCE)
    # Synthetic bundle provenance itself is not the published repair manifest.
    bundle = tmp_path / "bundle"; (bundle / "ops").mkdir(parents=True)
    put(bundle / "ops/REPAIR_MANIFEST.json", {"files": {}})
    payload = dict(source=str(SOURCE), private=str(private), bundle=str(bundle),
                   repair_manifest_sha256=worker.sha(bundle / "ops/REPAIR_MANIFEST.json"), operation="resume-F02-F03")
    fetched.clear()
    original_parser = data.parse_binance_zip
    monkeypatch.setattr(data, "parse_binance_zip", overlay.bind_parser(data))
    with patch("builtins.input", side_effect=AssertionError("NO_NEW_CONSENT")), \
         patch("getpass.getpass", side_effect=AssertionError("NO_TOKEN")):
        states = worker.perform(payload, data, helper, {"files": {"parser/cm_header_parser.py": worker.sha(parser_path)}})
    assert all(s["admitted"] for s in states.values())
    assert len(fetched) == (12 + 36) * 2
    assert all("BTCUSDT" not in url for url in fetched)
    assert all("BTCUSD_PERP-1d-2021" not in url for url in fetched)
    assert (raw / "acquisition_pending.json").read_bytes() == pending_before
    for root, before in {**previous, **all_owner_forms}.items():
        assert digest_tree(root) == before
    assert digest_tree(SOURCE) == original_source
    assert data.BINANCE_DATA_ADAPTER_VERSION == "crypto-markets-bounded-archives-base-volume-v2.1"
    receipts = list((private / "operational-repairs" / worker.VERSION).glob("*-completed-*.json"))
    assert len(receipts) == 1
    receipt = worker.read(receipts[0])
    assert receipt["preserved_existing_files_verified"] and "not pristine S2" in receipt["execution_identity"]
    assert len(receipt["normalized_header_events"]) == 12
    assert all(event["base_volume_column_zero_based"] == 7 and event["raw_source_bytes_rewritten"] is False
               for event in receipt["normalized_header_events"])
    assert set(receipt["admissions"]) == set(worker.PANELS)
    fetched.clear()
    with patch("builtins.input", side_effect=AssertionError("NO_NEW_CONSENT")):
        worker.perform(payload, data, helper, {"files": {"parser/cm_header_parser.py": worker.sha(parser_path)}})
    assert fetched == []
    assert len(list((private / "operational-repairs" / worker.VERSION).glob("*-completed-*.json"))) == 2
    assert data.parse_binance_zip is not original_parser
    # Success path in a fresh interpreter, without copying or changing SOURCE.
    # Only the local unpublished QA anchor READ is substituted by the test
    # harness; production has no fixture flag or source-validation bypass.
    harness = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import resume_worker as worker
original_read = worker.read
def qa_anchor_read(path):
    if Path(path).name == 'PUBLIC_SOURCE_ANCHOR.json':
        return dict(source_commit=worker.SOURCE_COMMIT, manifest_sha256=worker.SOURCE_MANIFEST_SHA256)
    return original_read(path)
worker.read = qa_anchor_read
worker.main()
"""
    isolated_payload = dict(payload, bundle=str(HERE.parent), operation="verify-only",
        repair_manifest_sha256=worker.sha(HERE / "REPAIR_MANIFEST.json"))
    child = subprocess.run([sys.executable, "-B", "-c", harness, str(HERE)],
        input=json.dumps(isolated_payload), text=True, capture_output=True)
    assert child.returncode == 0, child.stdout + child.stderr
    assert 'four_development_panels_verified_with_header_overlay' in child.stdout
    assert 'SOFTWARE FIXTURE ONLY' not in child.stdout and 'fixture@example.invalid' not in child.stdout
    assert digest_tree(SOURCE) == original_source


def test_failure_reason_retains_public_schema_and_redacts_private_values():
    reason = worker.safe_reason(RuntimeError("Wrong market-volume header quote_volume; https://example.invalid/?token=PRIVATE actor@example.invalid token=PRIVATE " + "x"*700))
    assert "Wrong market-volume header quote_volume" in reason
    assert "example.invalid" not in reason and "PRIVATE" not in reason
    assert len(reason) <= 500
