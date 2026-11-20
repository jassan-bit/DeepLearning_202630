"""Plan or hydrate pinned evidence bytes; never train, unpickle, deploy or publish.

The default is a local metadata verification and size plan. --hydrate requires
an explicit positive byte ceiling. Downloads are verified before atomic storage.
Declared AI assistance applies to this script and its documentary index.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.request import Request, urlopen

REPO = "jassan-bit/DeepLearning_202630"
SHA40 = re.compile(r"[0-9a-f]{40}")
SHA64 = re.compile(r"[0-9a-f]{64}")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid relative evidence path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(x in {"", ".", ".."} for x in path.parts) or path.as_posix() != value:
        raise ValueError("Unsafe or noncanonical evidence path")
    return path


def read_index(manifest_path):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("repository") != REPO or manifest.get("schema_version") != 1:
        raise ValueError("Unexpected evidence index identity")
    rows = []
    identities = set()
    for chunk in manifest["chunks"]:
        path = manifest_path.parent.joinpath(*relative(chunk["path"]).parts)
        if path.is_symlink():
            raise ValueError("Symlink in evidence index")
        raw = path.read_bytes()
        if len(raw) != chunk["bytes"] or sha256(raw) != chunk["sha256"]:
            raise ValueError("Evidence index chunk length/hash mismatch")
        parsed = [json.loads(line) for line in raw.splitlines()]
        if len(parsed) != chunk["entries"]:
            raise ValueError("Evidence index chunk count mismatch")
        for row in parsed:
            identity = (row["group"], row["logical_path"])
            if identity in identities:
                raise ValueError("Duplicate logical evidence identity")
            identities.add(identity)
            relative(row["logical_path"])
            relative(row["storage_path"])
            if (row["group"] not in {"audit796", "v3pilot30"}
                    or not SHA40.fullmatch(row["storage_commit"])
                    or not SHA64.fullmatch(row["sha256"])
                    or type(row["bytes"]) is not int or row["bytes"] < 0
                    or (row.get("git_blob_sha") is not None and not SHA40.fullmatch(row["git_blob_sha"]))):
                raise ValueError("Invalid pinned evidence locator")
        rows.extend(parsed)
    if len(rows) != manifest["logical_files"]:
        raise ValueError("Global evidence index count mismatch")
    return manifest, rows


def destination(output, row):
    path = output / row["group"]
    for component in relative(row["logical_path"]).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError("Evidence output may not traverse a symlink")
    return path


def check(raw, row):
    if len(raw) != row["bytes"] or sha256(raw) != row["sha256"]:
        raise ValueError("Evidence bytes do not match the pinned SHA-256 and length")
    if row.get("git_blob_sha") and git_blob(raw) != row["git_blob_sha"]:
        raise ValueError("Evidence bytes do not match the declared Git blob")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path(__file__).with_name("index") / "index_manifest.json")
    parser.add_argument("--group", choices=["audit796", "v3pilot30", "all"], default="all")
    parser.add_argument("--select-prefix", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hydrate", action="store_true")
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    _, rows = read_index(args.index)
    selected = [r for r in rows if (args.group == "all" or r["group"] == args.group)
                and r["logical_path"].startswith(args.select_prefix)]
    if not selected:
        raise SystemExit("Selection is empty; no request or output was created.")
    total = sum(r["bytes"] for r in selected)
    summary = {"mode": "hydrate" if args.hydrate else "metadata_plan_only", "selected_files": len(selected),
               "selected_bytes": total, "new_model_fits": 0, "bytes_unpickled": 0,
               "independent_training_reproduction": False}
    if not args.hydrate:
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.max_bytes is None or args.max_bytes <= 0 or total > args.max_bytes:
        raise SystemExit("Hydration requires an explicit positive --max-bytes covering the entire selection.")
    output = args.output.absolute()
    for parent in (output, *output.parents):
        if parent.is_symlink():
            raise SystemExit("Evidence output may not be a symlink or traverse one.")
    output.mkdir(parents=True, exist_ok=True)
    verified = downloaded = reused = 0
    for row in selected:
        target = destination(output, row)
        if target.exists():
            check(target.read_bytes(), row)
            reused += 1
        else:
            url = "https://raw.githubusercontent.com/" + REPO + "/" + row["storage_commit"] + "/" + row["storage_path"]
            with urlopen(Request(url, headers={"User-Agent": "E3-read-only-evidence-hydrator/1"}), timeout=60) as response:
                raw = response.read(row["bytes"] + 1)
            check(raw, row)
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".hydrating")
            if part.exists() or part.is_symlink():
                raise ValueError("Pending hydration file already exists; inspect it before retrying")
            with part.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            part.replace(target)
            downloaded += 1
        verified += 1
        if verified % 25 == 0:
            print(json.dumps({"verified_files": verified, "selection_total": len(selected)}), flush=True)
    summary.update(verified_files=verified, downloaded_files=downloaded, reused_files=reused,
                   all_selected_payload_bytes_verified=True,
                   no_claim_about_unselected_payload=True)
    receipt = output / "hydration_receipt.json"
    if receipt.is_symlink():
        raise ValueError("Receipt output is a symlink")
    receipt.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
