"""Validate cached immutable receipt metadata and build a precise hydration index.

This is metadata validation, not full payload hydration, model execution or an
independent repetition of training. Declared AI assistance; no data alteration.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "private"
OUT = ROOT / "index"
REPO = "jassan-bit/DeepLearning_202630"
MASTER = "20261119T031104515520Z"
SOURCE = "5a07d6157a9796ca54416dc5dea1e1527e731597"
PROTOCOL = "5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a"
FINAL_COMMIT = "16a4504f6fb9208c7d6f89b65017194fa20470da"
V3_COMMIT = "8eaa581bc34b615a8507c9e384a3e4be29084720"
V3_RUN = "20261120T031424176339Z"


def meta(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    if sys.flags.optimize:
        raise RuntimeError("Run this metadata validator without Python -O; its assertions must remain enabled.")
    OUT.mkdir(parents=True, exist_ok=True)
    cursor = json.loads((PRIVATE / "chain_cursor.json").read_bytes())
    assert cursor["sequence"] == 0 and len(cursor["summaries"]) == 193
    catalogue = {row["sequence"]: row for row in cursor["summaries"]}
    assert set(catalogue) == set(range(1, 194))
    assert catalogue[193]["commit"] == FINAL_COMMIT
    latest = {}
    parts = []
    attempted = set()
    previous_receipt = previous_record = None
    checked_bytes = 0
    declarations = []
    for sequence in range(1, 194):
        record = catalogue[sequence]
        stem = PRIVATE / "chain_raw" / f"{sequence:03d}"
        receipt_raw = Path(str(stem) + ".receipt.json").read_bytes()
        storage_raw = Path(str(stem) + ".storage.json").read_bytes()
        commit_raw = Path(str(stem) + ".commit.json").read_bytes()
        receipt, storage, commit = map(json.loads, (receipt_raw, storage_raw, commit_raw))
        checked_bytes += len(receipt_raw) + len(storage_raw) + len(commit_raw)
        receipt_meta, storage_meta = meta(receipt_raw), meta(storage_raw)
        assert receipt_meta["git_blob_sha"] == record["receipt_git_blob_sha"]
        assert storage_meta["git_blob_sha"] == record["storage_git_blob_sha"]
        assert receipt["kind"] == "e3_closure_stage_receipt" and receipt["profile"] == "audit"
        assert receipt["repository"] == REPO and receipt["source_commit"] == SOURCE
        assert receipt["master_run_id"] == MASTER and receipt["protocol_sha256"] == PROTOCOL
        assert receipt["sequence"] == sequence and receipt["part_id"] == record["part"]
        assert commit["sha"] == record["commit"] and len(commit["parents"]) == 1
        if previous_record is None:
            assert receipt["previous_part"] is None and commit["parents"][0]["sha"] == SOURCE
        else:
            assert receipt["previous_part"] == {"commit_sha": previous_record["commit"],
                "part_id": previous_record["part"], "receipt_sha256": meta(previous_receipt)["sha256"]}
            assert commit["parents"][0]["sha"] == previous_record["commit"]
            assert previous_record["part"] < record["part"]
        assert storage["kind"] == "e3_closure_storage_manifest" and storage["profile"] == "audit"
        assert storage["run_id"] == record["part"] and storage["source_commit"] == SOURCE
        expected_storage = {"payload/" + name: item for name, item in receipt["files"].items()}
        expected_storage["stage_receipt.json"] = receipt_meta
        assert storage["files"] == expected_storage
        assert receipt["new_real_data_fits"] == len(receipt["new_attempted_jobs"])
        assert not attempted.intersection(receipt["new_attempted_jobs"])
        attempted.update(receipt["new_attempted_jobs"])
        prefix = "runs/e3_extended_audit/" + record["part"] + "/"
        for name, item in receipt["files"].items():
            latest[name] = {"group": "audit796", "logical_path": name,
                "storage_commit": record["commit"], "storage_path": prefix + "payload/" + name,
                "source_sequence": sequence, **item}
        declarations.append({"sequence": sequence, "controller_status": receipt["controller_status"]})
        parts.append({"sequence": sequence, "commit": record["commit"], "part_id": record["part"],
            "receipt_path": prefix + "stage_receipt.json", "receipt_bytes": len(receipt_raw),
            "receipt_sha256": receipt_meta["sha256"], "receipt_git_blob_sha": receipt_meta["git_blob_sha"],
            "storage_path": prefix + "storage_manifest.json", "storage_bytes": len(storage_raw),
            "storage_sha256": storage_meta["sha256"], "storage_git_blob_sha": storage_meta["git_blob_sha"],
            "payload_delta_files": len(receipt["files"]),
            "payload_delta_bytes": sum(item["bytes"] for item in receipt["files"].values()),
            "new_attempt_markers": len(receipt["new_attempted_jobs"])})
        previous_record, previous_receipt = record, receipt_raw
    v3_raw = (PRIVATE / "V3_snapshot_manifest.json").read_bytes()
    v3 = json.loads(v3_raw)
    assert meta(v3_raw)["git_blob_sha"] == "08591bddca026fb0a5bb958eb5d3916ad8f8445d"
    assert v3["kind"] == "e3_v3_development_git_snapshot" and v3["sequence"] == 93 and v3["run_id"] == V3_RUN
    rows = list(latest.values())
    rows += [{"group": "v3pilot30", "logical_path": name, "storage_commit": V3_COMMIT,
              "storage_path": "runs/e3_v3_development_pilot/" + V3_RUN + "/payload/" + name,
              "source_sequence": 93, "git_blob_sha": None, **item}
             for name, item in v3["files"].items()]
    rows.sort(key=lambda row: (row["group"], row["logical_path"]))
    chunks = []
    current = bytearray()
    count = 0
    def finish_chunk():
        nonlocal current, count
        if not current:
            return
        name = "locators-" + str(len(chunks) + 1).zfill(3) + ".jsonl"
        raw = bytes(current)
        (OUT / name).write_bytes(raw)
        chunks.append({"path": name, "bytes": len(raw), "sha256": meta(raw)["sha256"], "entries": count})
        current, count = bytearray(), 0
    for row in rows:
        line = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if len(current) + len(line) > 96_000:
            finish_chunk()
        current.extend(line)
        count += 1
    finish_chunk()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(parts[0]))
    writer.writeheader()
    writer.writerows(parts)
    catalogue_raw = stream.getvalue().encode()
    (OUT / "audit193_parts.csv").write_bytes(catalogue_raw)
    groups = {group: {"logical_files": sum(row["group"] == group for row in rows),
                      "logical_bytes": sum(row["bytes"] for row in rows if row["group"] == group)}
              for group in ("audit796", "v3pilot30")}
    manifest = {"schema_version": 1, "repository": REPO, "logical_files": len(rows), "chunks": chunks,
        "groups": groups, "audit_master_run_id": MASTER, "audit_terminal_commit": FINAL_COMMIT,
        "parts_catalogue": {"path": "audit193_parts.csv", "bytes": len(catalogue_raw), "sha256": meta(catalogue_raw)["sha256"]},
        "audit_part_count": 193, "V3_cumulative_snapshot_commit": V3_COMMIT, "V3_snapshot_sequence": 93,
        "validation_scope": "All193 receipt/storage bytes and their SHA256/Git blob bindings, parent links and source/protocol/master identities; V3 cumulative manifest Git blob; payload locators and declared hashes.",
        "all_indexed_payload_bytes_downloaded": False,
        "all_payload_Git_tree_entries_independently_checked": False,
        "license_granted_by_this_index": False}
    dump(OUT / "index_manifest.json", manifest)
    result = {"status": "metadata_chain_verified", "part_sequences": "1..193", "receipt_documents": 193,
        "storage_documents": 193, "commit_parent_documents": 193, "metadata_bytes_checked": checked_bytes,
        "receipt_and_storage_Git_SHA1_verified": True, "receipt_chain_SHA256_verified": True,
        "source_protocol_master_bindings_verified": True, "unique_attempt_markers": len(attempted),
        "attempt_marker_count_is_not_independent_completed_fit_replay": True,
        "logical_job_result_files": sum(name.startswith("jobs/") and name.endswith("/result.json") for name in latest),
        "logical_neural_fit_request_files": sum(name.startswith("jobs/") and name.endswith("/fit_request.json") for name in latest),
        "checkpoint_files_are_not_counted_as_distinct_fit_designs": True,
        "final_controller_declaration": declarations[-1]["controller_status"],
        "audit_final_delta_only_files": parts[-1]["payload_delta_files"],
        "groups": groups, "V3_manifest_sha256": meta(v3_raw)["sha256"],
        "V3_snapshot_is_cumulative": True, "audit_terminal_part_is_cumulative": False,
        "indexed_payload_bytes_verified_in_this_metadata_pass": 0,
        "independent_training_reproduction": False, "new_real_data_fits": 0,
        "all_original_payload_bytes_are_present_in_this_supplement": False}
    dump(OUT / "metadata_validation_receipt.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
