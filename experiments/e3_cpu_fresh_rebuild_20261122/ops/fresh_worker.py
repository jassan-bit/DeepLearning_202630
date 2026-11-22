"""Whitelisted development-only worker over untouched S2 plus pinned CM parser."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path
import sys
import threading

sys.dont_write_bytecode = True


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def emit(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def _checked_export(data, row, private):
    panel, private = row["panel"], Path(private).resolve()
    data.require(panel in ("F00", "F01", "F02", "F03"), "Fixed development panel required")
    exported = private / "recovery-exports" / (panel + ".json")
    data.require(exported.is_file() and not any(p.is_symlink() for p in (exported, *exported.parents)),
                 "Preserve an authentic metadata recovery export")
    expected = sha(exported.read_bytes())
    _, bundle, _, _, manifest, _ = data._read_recovery_bundle(exported, expected)
    data.require(Path(bundle["original_panel_dir"]).resolve() == Path(row["panel_dir"]).resolve() and
                 Path(bundle["original_raw_dir"]).resolve() == Path(row["raw"]).resolve() and
                 manifest["provider"] == row["proposal"]["provider"] and
                 manifest["instrument"] == row["proposal"]["instrument"],
                 "Recovery export differs from fixed panel/source paths")
    return exported, expected, bundle, manifest


def _restore_replay_companions(data, row, private):
    """Repair only exact saved metadata after interrupted source reacquisition.

    S2 downloads ZIPs before its saved pending/checksum companions. An interrupt
    in that interval leaves authentic ZIPs which the ordinary pending inspector
    cannot yet bind. Reinstall the authenticated export's original companions;
    never generate a new pending, replace a payload, or edit current replay logs.
    """
    raw, private = Path(row["raw"]), Path(private).resolve()
    if not raw.exists():
        return 0
    _, _, bundle, manifest = _checked_export(data, row, private)
    data.require(raw.resolve().is_relative_to(private) and not any(p.is_symlink() for p in (raw, *raw.parents)),
                 "Keep replay metadata inside the original private directory")
    sources = {item["path"]:item for item in bundle["sources"]}
    allowed_companions = {item["path"] + ".CHECKSUM" for item in manifest["files"]} | {
        "acquisition_pending.json", "acquisition_attempts.jsonl"}
    companions = {}
    for item in bundle["raw_metadata_companions"]:
        name = item["relative_path"]
        data.require(name in allowed_companions and name not in companions, "Unexpected replay companion")
        companions[name] = data._decoded_metadata(item)
    data.require("acquisition_pending.json" in companions, "No original pending metadata in recovery export")
    original_manifest = data._decoded_metadata(bundle["raw_manifest"])
    # Reject any foreign/replaced bytes before writing even trusted metadata.
    for path in raw.iterdir():
        data.require(path.is_file() and not path.is_symlink(), "Unexpected replay directory/symlink")
        name = path.name
        if name in sources:
            original = sources[name]
            data.require(path.stat().st_size == original["bytes"] and sha(path.read_bytes()) == original["sha256"],
                         "Interrupted replay source bytes differ from the saved export")
        elif name in companions:
            data.require(path.read_bytes() == companions[name], "Interrupted replay companion bytes differ")
        elif name == "raw_manifest.json":
            data.require(path.read_bytes() == original_manifest, "Interrupted replay manifest differs")
        else:
            data.require(name == "reacquisition_events.jsonl", "Unknown interrupted replay file retained")
    if (raw / "acquisition_pending.json").exists():
        return 0
    for name, body in companions.items():
        data._restore_metadata_file(data._safe_path(raw, name), body, private.parent)
    folder = Path(row["folder"])
    data.inspect_binance_pending(json.loads((folder / "acquisition.json").read_text()), raw,
        rights_receipt=json.loads((folder / "rights.json").read_text()),
        exposure_receipt=json.loads((folder / "exposure-development.json").read_text()))
    return len(companions)


def reacquire_panel(data, row, private):
    """Replay an authenticated recovery EXPORT into its absent original target.

    A journal restores the export, never admission.json into an empty-data panel.
    The existing S2 replay code verifies and preserves that saved identity.
    """
    private = Path(private).resolve()
    exported, expected, _, _ = _checked_export(data, row, private)
    return data.reacquire_development(exported, row["panel_dir"], row["raw"], private.parent,
        enable_market_acquisition=True, expected_bundle_sha256=expected)


def main():
    payload = json.load(sys.stdin)
    if payload.get("operation") not in ("record", "check", "acquire_prepare", "reacquire"):
        raise RuntimeError("Only fixed development record/check/acquire_prepare/reacquire operations are allowed")
    if sha(Path(__file__).read_bytes()) != payload.get("worker_sha256"):
        raise RuntimeError("Fresh worker bytes differ")
    source, private, bundle = (Path(payload[k]).resolve() for k in ("source", "private", "bundle"))
    if source.is_relative_to(private) or private.is_relative_to(source) or bundle.is_relative_to(source) or source.is_relative_to(bundle):
        raise RuntimeError("Keep source, private state and addon separate")
    # Verify the addon bytes BEFORE importing its executable Python modules.
    cm_expected = "21ce8729f7db6c9e683adf57dfcb94e97eeb27c081c973a0791af5a39e9ac0a2"
    cm_path = bundle / "ops/REPAIR_MANIFEST.json"
    if sha(cm_path.read_bytes()) != cm_expected:
        raise RuntimeError("Pinned CM manifest differs")
    for relative, expected in json.loads(cm_path.read_text())["files"].items():
        path = bundle / relative
        if not path.resolve().is_relative_to(bundle) or any(p.is_symlink() for p in (path, *path.parents)) or sha(path.read_bytes()) != expected:
            raise RuntimeError("Pinned CM addon file differs: " + relative)
    # The old addon manifest independently fixes its own worker and parser.
    sys.path.insert(0, str(bundle / "ops"))
    from resume_worker import (verify_source, verify_bundle, rows_for_source, write_event, safe_reason,
                               VERSION as CM_VERSION, SOURCE_COMMIT, SOURCE_MANIFEST_SHA256)
    verify_source(source)
    cm_manifest_hash = "21ce8729f7db6c9e683adf57dfcb94e97eeb27c081c973a0791af5a39e9ac0a2"
    verify_bundle(dict(bundle=str(bundle), repair_manifest_sha256=cm_manifest_hash))
    sys.path.insert(0, str(source))
    from engine import data
    helper = load(source / "package/crypto_development.py", "fresh_pinned_development_contract")
    parser = load(bundle / "parser/cm_header_parser.py", "fresh_pinned_cm_parser")
    data.parse_binance_zip = parser.bind_parser(data)
    rows = rows_for_source(source, private)
    for row in rows:
        if (private / "recovery-exports" / (row["panel"] + ".json")).is_file():
            _restore_replay_companions(data, row, private)
    for row in rows:
        owner = json.loads((Path(row["folder"]) / "owner-development.json").read_text())
        data.require(owner.get("kind") == "prior_authorization_reused_during_user_requested_rebuild" and
            owner.get("prior_authorization_report_sha256") == "7e4dd2b4f2d90dc2c4b3eee528d09d99b45575d2529b1d6501d185263151e567" and
            owner.get("original_statement_utc") is None and owner.get("AI_completed_signature") is False,
            "Authentic prior-authorization reconstruction metadata required")
    def check():
        stdin, output = sys.stdin, io.StringIO()
        try:
            sys.stdin = io.StringIO(json.dumps(dict(rows=rows, rule=json.loads((private / "development-split-rule.json").read_text()))))
            with contextlib.redirect_stdout(output):
                exec(compile(helper._CHECK_PROGRAM, "<fixed S2 development check with explicit CM overlay>", "exec"), {})
        finally:
            sys.stdin = stdin
        return json.loads(output.getvalue())["panels"]
    stopped = threading.Event()
    def pulse():
        while not stopped.wait(30):
            emit(status="fresh_development_progress", operation=payload["operation"], panel=payload.get("panel"),
                 local_archives={r["panel"]:len(list(Path(r["raw"]).glob("*.zip"))) for r in rows},
                 new_market_fits=0, test_opened=False)
    thread = threading.Thread(target=pulse, daemon=True); thread.start()
    failed = None
    try:
        if payload["operation"] == "record":
            for row in rows:
                folder = Path(row["folder"]); receipt = folder / "exposure-development.json"
                if not receipt.exists():
                    data.record_owner_evidence(folder / "owner-development.json", folder / "owner-development.json", receipt)
        states = check()
        if payload["operation"] in ("acquire_prepare", "reacquire"):
            data.require(payload.get("panel") in ("F00", "F01", "F02", "F03"), "Fixed panel required")
            row = next(r for r in rows if r["panel"] == payload["panel"])
            folder, state = Path(row["folder"]), states[row["panel"]]
            if not state["admitted"]:
                if payload["operation"] == "reacquire":
                    reacquire_panel(data, row, private)
                else:
                    rights = json.loads((folder / "rights.json").read_text())
                    exposure = json.loads((folder / "exposure-development.json").read_text())
                    if not state["raw_complete"]:
                        data.collect_development(json.loads((folder / "acquisition.json").read_text()), row["raw"],
                            rights_receipt=rights, exposure_receipt=exposure, enable_market_acquisition=True)
                    data.prepare_development(Path(row["raw"]) / "raw_manifest.json",
                        json.loads((folder / "development-split.json").read_text()), row["panel_dir"], rights=rights, exposure=exposure)
            states = check()
        emit(status="fresh_worker_complete", panels=states, new_market_fits=0, test_opened=False)
    except BaseException as exc:
        failed = exc
        emit(status="fresh_development_stopped", error_type=type(exc).__name__, reason=safe_reason(exc),
             new_market_fits=0, test_opened=False)
        raise
    finally:
        stopped.set(); thread.join()
        write_event(private / "operational-repairs" / CM_VERSION, "fresh_development_progress", dict(
            fresh_rebuild_version="fresh-cpu-development-rebuild-1", parent_source_commit=SOURCE_COMMIT,
            parent_manifest_sha256=SOURCE_MANIFEST_SHA256, repair_manifest_sha256=cm_manifest_hash,
            fresh_worker_sha256=payload["worker_sha256"], operation=payload["operation"], panel=payload.get("panel"),
            normalized_header_events=list(data.parse_binance_zip.__cm_header_repair_events__),
            metadata_origin=("Exact replay of the journal-backed fresh-rebuild identity; no first-lost-VM hashes invented"
                if payload["operation"] == "reacquire" else
                "Fixed development metadata for fresh rebuild or its journal-backed replay; no first-lost-VM identity invented"),
            outcome="failed" if failed else "complete", error_type=type(failed).__name__ if failed else None,
            new_market_fits=0, test_opened=False))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        reason = re.sub(r"https?://\S+|[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "[PRIVATE REFERENCE REDACTED]", str(exc))[:500]
        emit(status="fresh_worker_failed", error_type=type(exc).__name__, reason=reason, new_market_fits=0, test_opened=False)
        raise SystemExit(1) from None
