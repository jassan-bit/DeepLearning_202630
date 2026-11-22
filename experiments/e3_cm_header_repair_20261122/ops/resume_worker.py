"""One isolated CPU process; pin original S2, bind only a documented parser fix.

This is a development repair, not a general engine launcher. No command-line
payload can select final preparation, training, evaluation or new signatures.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
import sys
import threading
import uuid

sys.dont_write_bytecode = True
from resume_development import (VERSION, SOURCE_COMMIT, SOURCE_MANIFEST_SHA256,
                                SOURCE_DATA_SHA256, require, sha)

PARTS = ("fit", "selection", "planning", "calibration")
PANELS = ("F00", "F01", "F02", "F03")
RULE = dict(version="development-calendar-proportions-1", fit_fraction=.60, selection_fraction=.10,
            planning_fraction=.20, calibration_fraction=.10, embargo_bars=0,
            basis="expected exchange sessions only; no market prices, labels or performance used",
            support_rule="raw disjoint bounds; engine purges endpoints until all 15 closes fit the assigned support")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_reason(exc):
    value = str(exc)[:4000]
    value = re.sub(r"https?://[^\s\\\"<>]+", "[URL REDACTED]", value)
    value = re.sub(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "[EMAIL REDACTED]", value)
    value = re.sub(r"(?i)(token|secret|password|authorization)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    return value[:500]


def overlay_events(data):
    return list(getattr(data.parse_binance_zip, "__cm_header_repair_events__", []))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(**event):
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def verify_source(source):
    """Validate every S2 file and its unmodified published anchor before imports."""
    source = Path(source).resolve()
    require(sha(source / "SOURCE_MANIFEST.json") == SOURCE_MANIFEST_SHA256, "S2 manifest differs")
    anchor = read(source / "PUBLIC_SOURCE_ANCHOR.json")
    require(anchor.get("source_commit") == SOURCE_COMMIT and
            anchor.get("manifest_sha256") == SOURCE_MANIFEST_SHA256, "S2 anchor differs")
    manifest = read(source / "SOURCE_MANIFEST.json")
    require(len(manifest["files"]) == 304, "S2 source inventory differs")
    for relative, expected in manifest["files"].items():
        path = source / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(source) and sha(path) == expected,
                "S2 source file differs: " + relative)
    require(sha(source / "engine/data.py") == SOURCE_DATA_SHA256, "S2 data module differs")
    return manifest


def verify_bundle(payload):
    bundle = Path(payload["bundle"]).resolve()
    require(sha(bundle / "ops" / "REPAIR_MANIFEST.json") == payload["repair_manifest_sha256"], "Repair manifest differs")
    manifest = read(bundle / "ops" / "REPAIR_MANIFEST.json")
    require(manifest["version"] == VERSION and manifest["parent_source_commit"] == SOURCE_COMMIT and
            manifest["parent_manifest_sha256"] == SOURCE_MANIFEST_SHA256 and
            manifest["parent_data_sha256"] == SOURCE_DATA_SHA256 and
            manifest["scope"] == "resume_existing_F02_F03_development_only", "Repair parent/scope differs")
    for relative, expected in manifest["files"].items():
        path = bundle / relative
        require(not path.is_symlink() and path.resolve().is_relative_to(bundle) and sha(path) == expected,
                "Repair file differs: " + relative)
    require(sha(__file__) == manifest["files"]["ops/resume_worker.py"], "Worker not pinned")
    require(manifest.get("preserved_physical_adapter_version") ==
            "crypto-markets-bounded-archives-base-volume-v2.1", "Physical adapter identity differs")
    return manifest


def rows_for_source(source, private):
    proposals = {p["id"]: p for p in read(source / "configs/data_panels.json")["panels"]}
    require(set(proposals) == set(PANELS), "Original panel inventory differs")
    source_form = read(source / "docs/OWNER_EXPOSURE_FORM.json")
    rule = read(private / "development-split-rule.json")
    require(rule == RULE, "Original development calendar rule differs")
    rows = []
    for panel in PANELS:
        p = proposals[panel]
        license_path = source / p["rights_snapshot_path"]
        require(sha(license_path) == p["rights_snapshot_sha256"], "Primary-source snapshot differs")
        blank = dict(source_form, panel_ids=[panel], reserve_mode="development_only",
            source_url=p["rights_source_url"], source_license_evidence_path=str(license_path),
            source_license_evidence_sha256=sha(license_path), dataset_scope=dict(panel=panel,
                provider=p["provider"], instrument=p["instrument"],
                start=p["proposed_development_start"], end=p["proposed_development_end"]))
        rows.append(dict(panel=panel, proposal=p, blank_owner=blank,
            folder=str(private / "inputs" / panel), raw=str(private / "raw" / panel / "development-v1"),
            panel_dir=str(private / "panels" / panel / "development-v1")))
    require(read(private / "development-panels.json") == {r["panel"]: r["panel_dir"] for r in rows},
            "Original panel map differs")
    for row in rows:
        for key in ("folder", "raw", "panel_dir"):
            require(Path(row[key]).resolve().is_relative_to(private), "Private path escape/symlink")
    return rows


def check(data, helper, rows):
    """Use the original pinned helper's strict, read-only evidence contract."""
    stdin = sys.stdin
    output = io.StringIO()
    try:
        sys.stdin = io.StringIO(json.dumps(dict(rows=rows, rule=RULE)))
        with contextlib.redirect_stdout(output):
            exec(compile(helper._CHECK_PROGRAM, "<S2 pinned development preflight>", "exec"), {})
    finally:
        sys.stdin = stdin
    states = json.loads(output.getvalue())["panels"]
    require(all(not s["blank_owner"] and s["owner_receipt"] and s["rights_present"] for s in states.values()),
            "Original author rights/exposure receipts required; this repair cannot create or sign them")
    require(all(states[p]["admitted"] for p in ("F00", "F01")), "Preserve and verify the two original admissions first")
    return states


def snapshot(rows, source, bundle):
    """Snapshot existing bytes, including anchor; only F02/F03 attempt logs may append."""
    paths = []
    for row in rows:
        for key in ("folder", "raw", "panel_dir"):
            root = Path(row[key])
            if root.exists():
                paths.extend((p, row["panel"] in ("F02", "F03") and p.name == "acquisition_attempts.jsonl")
                             for p in root.rglob("*") if p.is_file())
    paths.extend((source / relative, False) for relative in read(source / "SOURCE_MANIFEST.json")["files"])
    paths.extend(((source / "SOURCE_MANIFEST.json", False), (source / "PUBLIC_SOURCE_ANCHOR.json", False)))
    paths.extend((bundle / relative, False) for relative in read(bundle / "ops" / "REPAIR_MANIFEST.json")["files"])
    paths.append((bundle / "ops" / "REPAIR_MANIFEST.json", False))
    result = {}
    for path, may_append in paths:
        require(not path.is_symlink(), "Preserved source/input symlink refused")
        stat = path.stat()
        result[str(path)] = dict(sha256=sha(path), size=stat.st_size,
                                 mtime_ns=stat.st_mtime_ns, append_only=may_append)
    return result


def assert_preserved(before):
    for value, state in before.items():
        path = Path(value)
        require(path.is_file() and not path.is_symlink(), "Previously existing file lost or relinked")
        if state["append_only"]:
            with path.open("rb") as stream:
                prefix = stream.read(state["size"])
            require(hashlib.sha256(prefix).hexdigest() == state["sha256"], "Original acquisition log prefix changed")
        else:
            require(sha(path) == state["sha256"] and path.stat().st_mtime_ns == state["mtime_ns"],
                    "An immutable source/input/admission was changed")


def write_event(directory, kind, body):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / (stamp + "-" + kind + "-" + uuid.uuid4().hex[:8] + ".json")
    value = dict(body, kind=kind, recorded_utc=utc())
    value["event_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def progress(row, operation):
    stopped = threading.Event()
    def report():
        while not stopped.wait(30):
            emit(status="development_repair_progress", panel=row["panel"], operation=operation,
                 local_archive_files=len(list(Path(row["raw"]).glob("*.zip"))),
                 new_market_fits=0, test_opened=False)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    return stopped, thread


def perform(payload, data, helper, repair_manifest):
    """Testable activated orchestration; imports/pins are handled before entry."""
    require(payload.get("operation") in ("verify-only", "resume-F02-F03"), "Unsupported repair operation")
    source, private, bundle = (Path(payload[k]).resolve() for k in ("source", "private", "bundle"))
    require(not source.is_relative_to(private) and not private.is_relative_to(source) and
            not bundle.is_relative_to(source) and not source.is_relative_to(bundle),
            "Keep source, private files and addon separate")
    rows = rows_for_source(source, private)
    states = check(data, helper, rows)
    # This preflight happens before any writes or provider network access.
    before = snapshot(rows, source, bundle)
    receipts = private / "operational-repairs" / VERSION
    provenance = dict(repair_version=VERSION, parent_source_commit=SOURCE_COMMIT,
        parent_manifest_sha256=SOURCE_MANIFEST_SHA256, parent_data_sha256=SOURCE_DATA_SHA256,
        repair_manifest_sha256=payload["repair_manifest_sha256"], overlay_files=repair_manifest["files"],
        operation=payload["operation"], scope="development_only", authorized_panels=["F02", "F03"],
        preserved_physical_adapter_version=data.BINANCE_DATA_ADAPTER_VERSION,
        execution_identity="S2 source plus separately pinned CM header parser overlay; not pristine S2 execution",
        new_market_fits=0, test_opened=False, normalized_header_events=overlay_events(data),
        original_author_receipts={r["panel"]:{name:sha(Path(r["folder"]) / name) for name in
            ("owner-development.json", "rights.json", "exposure-development.json")} for r in rows},
        original_F00_F01_admissions={r["panel"]:sha(Path(r["panel_dir"]) / "admission.json")
            for r in rows if r["panel"] in ("F00", "F01")},
        preserved_existing_files=before)
    started = write_event(receipts, "started", provenance)
    emit(status="development_header_overlay_preflight_verified", receipt=str(started),
         panels={p:{"admitted":s["admitted"],"raw_complete":s["raw_complete"]} for p,s in states.items()},
         new_market_fits=0, test_opened=False)
    try:
        if payload["operation"] == "resume-F02-F03":
            for row in rows:
                if row["panel"] not in ("F02", "F03") or states[row["panel"]]["admitted"]:
                    continue
                state, folder = states[row["panel"]], Path(row["folder"])
                stopped, thread = progress(row, "collect_and_prepare_development")
                try:
                    if not state["raw_complete"]:
                        # Reuses original pending/config/evidence; same source
                        # physical codec and archive policy, zero new consent.
                        data.collect_development(read(folder / "acquisition.json"), row["raw"],
                            rights_receipt=read(folder / "rights.json"),
                            exposure_receipt=read(folder / "exposure-development.json"),
                            enable_market_acquisition=True)
                    data.prepare_development(Path(row["raw"]) / "raw_manifest.json",
                        read(folder / "development-split.json"), row["panel_dir"],
                        rights=read(folder / "rights.json"), exposure=read(folder / "exposure-development.json"))
                finally:
                    stopped.set(); thread.join()
                emit(status="development_repair_panel_admitted", panel=row["panel"],
                     new_market_fits=0, test_opened=False)
        states = check(data, helper, rows)
        assert_preserved(before)
        require(all(s["admitted"] for s in states.values()), "Four real development admissions not yet complete")
        admissions = {r["panel"]:dict(admission_sha256=data.read_admission(r["panel_dir"])["admission_sha256"],
            admission_file_sha256=sha(Path(r["panel_dir"]) / "admission.json"),
            recovery_metadata_sha256=sha(Path(r["panel_dir"]) / "recovery_metadata.json")) for r in rows}
        finished = write_event(receipts, "completed", dict(provenance, parent_event_sha256=sha(started),
            preserved_existing_files_verified=True, admissions=admissions,
            normalized_header_events=overlay_events(data)))
        emit(status="four_development_panels_verified_with_header_overlay", receipt=str(finished),
             source_files_unchanged=304, original_admissions_preserved=True, panels=admissions,
             next_action="Execute original stage 3 only; planning remains separately disabled",
             new_market_fits=0, test_opened=False)
        return states
    except BaseException as exc:
        preserved = False
        try:
            assert_preserved(before)
            preserved = True
        finally:
            failure = write_event(receipts, "failed", dict(provenance, parent_event_sha256=sha(started),
                error_type=type(exc).__name__, reason=safe_reason(exc),
                normalized_header_events=overlay_events(data), preserved_existing_files_verified=preserved))
            emit(status="development_repair_stopped", error_type=type(exc).__name__, reason=safe_reason(exc),
                 receipt=str(failure), preserved_existing_files_verified=preserved,
                 new_market_fits=0, test_opened=False)
        raise


def main():
    payload = json.load(sys.stdin)
    require(payload.get("operation") in ("verify-only", "resume-F02-F03"), "Unsupported repair operation")
    source = Path(payload["source"]).resolve()
    verify_source(source)
    repair_manifest = verify_bundle(payload)
    # No notebook-kernel module or user PYTHONPATH can supply the engine.
    require("engine" not in sys.modules and "engine.data" not in sys.modules, "Unexpected prior engine import")
    sys.path.insert(0, str(source))
    from engine import data
    require(Path(data.__file__).resolve() == source / "engine/data.py", "Wrong engine import")
    helper = load(source / "package/crypto_development.py", "e3_s2_development_contract")
    overlay = load(Path(payload["bundle"]) / "parser/cm_header_parser.py", "e3_cm_header_parser_overlay")
    original_adapter = data.BINANCE_DATA_ADAPTER_VERSION
    original_parser = data.parse_binance_zip
    data.parse_binance_zip = overlay.bind_parser(data)
    require(data.parse_binance_zip is not original_parser and data.BINANCE_DATA_ADAPTER_VERSION == original_adapter,
            "Overlay must bind only parser acceptance; retain original physical adapter identity")
    perform(payload, data, helper, repair_manifest)
    verify_source(source)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        emit(status="development_repair_preflight_or_execution_failed", error_type=type(exc).__name__, reason=safe_reason(exc),
             new_market_fits=0, test_opened=False)
        raise SystemExit(1) from None
