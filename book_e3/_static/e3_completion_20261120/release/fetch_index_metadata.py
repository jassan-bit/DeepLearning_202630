"""Restore metadata inputs for build_evidence_index.py, without scientific blobs.

Default: print a plan. --fetch additionally requires a byte ceiling. Optional
GITHUB_TOKEN is read from the environment only for api.github.com commit GETs;
it is never printed, persisted, or sent to raw.githubusercontent.com. Anonymous
GitHub API rate limits may prevent completion; a partial cache is resumable.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

REPO = "jassan-bit/DeepLearning_202630"
V3_COMMIT = "8eaa581bc34b615a8507c9e384a3e4be29084720"
V3_MANIFEST_PATH = "runs/e3_v3_development_pilot/20261120T031424176339Z/snapshot_manifest.json"
V3_MANIFEST_SHA = "e397eff5d4c170b27b5acd0debb93e2b0534c1057112079f10a0232f49807054"
V3_GIT_BLOB = "08591bddca026fb0a5bb958eb5d3916ad8f8445d"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(__file__).with_name("index") / "index_manifest.json")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.index.read_bytes())
    catalogue = manifest["parts_catalogue"]
    if catalogue["path"] != "audit193_parts.csv":
        raise SystemExit("Unexpected parts catalogue")
    raw = (args.index.parent / catalogue["path"]).read_bytes()
    if len(raw) != catalogue["bytes"] or sha(raw) != catalogue["sha256"]:
        raise SystemExit("Parts catalogue length/hash mismatch")
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    if len(rows) != 193 or [int(row["sequence"]) for row in rows] != list(range(1, 194)):
        raise SystemExit("Unexpected sequence catalogue")
    plan = {"receipt_metadata_documents": 193, "storage_metadata_documents": 193,
            "Git_commit_metadata_documents": 193, "V3_cumulative_manifest_documents": 1,
            "scientific_payload_blobs_requested": 0, "new_model_fits": 0,
            "mode": "fetch_metadata_only" if args.fetch else "metadata_fetch_plan_only"}
    if not args.fetch:
        print(json.dumps(plan, sort_keys=True))
        return
    if args.max_bytes is None or args.max_bytes <= 0:
        raise SystemExit("--fetch requires a positive explicit --max-bytes")
    target_root = Path(__file__).with_name("private")
    target_root.mkdir(exist_ok=True)
    if target_root.is_symlink():
        raise SystemExit("Symlink metadata cache refused")
    cache = target_root / "chain_raw"
    cache.mkdir(exist_ok=True)
    if cache.is_symlink():
        raise SystemExit("Symlink metadata part cache refused")
    spent = 0
    def get(url, target, expected_bytes=None, expected_sha=None, expected_blob=None):
        nonlocal spent
        if target.is_symlink():
            raise ValueError("Symlink metadata file refused")
        if target.exists():
            body = target.read_bytes()
        else:
            headers = {"User-Agent": "E3-read-only-metadata-hydrator/1"}
            if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
                headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
            remaining = args.max_bytes - spent
            if remaining <= 0:
                raise ValueError("Declared metadata byte ceiling exhausted")
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                body = response.read(min(1_000_000, remaining) + 1)
        spent += len(body)
        if spent > args.max_bytes:
            raise ValueError("Declared metadata byte ceiling exceeded")
        if ((expected_bytes is not None and len(body) != expected_bytes)
                or (expected_sha and sha(body) != expected_sha)
                or (expected_blob and blob(body) != expected_blob)):
            raise ValueError("Downloaded metadata does not match its pinned identity")
        target.write_bytes(body)
        return json.loads(body)
    summaries = []
    for row in rows:
        number = int(row["sequence"])
        prefix = f"https://raw.githubusercontent.com/{REPO}/{row['commit']}/"
        for kind in ("receipt", "storage"):
            get(prefix + row[kind + "_path"], cache / f"{number:03d}.{kind}.json",
                int(row[kind + "_bytes"]), row[kind + "_sha256"], row[kind + "_git_blob_sha"])
        commit = get(f"https://api.github.com/repos/{REPO}/git/commits/{row['commit']}",
                     cache / f"{number:03d}.commit.json")
        if commit.get("sha") != row["commit"]:
            raise ValueError("Git commit API identity mismatch")
        summaries.append({"sequence": number, "commit": row["commit"], "part": row["part_id"],
            "receipt_git_blob_sha": row["receipt_git_blob_sha"], "storage_git_blob_sha": row["storage_git_blob_sha"]})
        if number % 25 == 0:
            print(json.dumps({"metadata_parts_retrieved": number, "byte_budget_used": spent}), flush=True)
    get(f"https://raw.githubusercontent.com/{REPO}/{V3_COMMIT}/{V3_MANIFEST_PATH}",
        target_root / "V3_snapshot_manifest.json", expected_sha=V3_MANIFEST_SHA, expected_blob=V3_GIT_BLOB)
    (target_root / "chain_cursor.json").write_text(json.dumps({"sequence": 0, "summaries": summaries}) + "\n")
    plan.update(metadata_bytes_verified_or_retrieved=spent,
                next_command="python build_evidence_index.py; compare its result to the existing validation receipt")
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
