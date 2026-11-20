"""Assemble the reviewed distribution ZIP from small, hash-bound Git blobs.

Only byte transport is performed. No model/statistical code or network is used.
The complete generated ZIP is published as a static book download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import os


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def assemble(root):
    root = Path(root).resolve()
    prefix = Path("experiments/e3_evidence_close_20261120/release")
    release = root / prefix
    manifest = json.loads((release / "archive_parts_manifest.json").read_text())
    receipt = json.loads((release / (manifest["archive"] + ".manifest.json")).read_text())
    chunks = []
    for index, entry in enumerate(manifest["parts"]):
        rel = PurePosixPath(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "archive_parts":
            raise ValueError("Invalid archive part path")
        if entry["index"] != index:
            raise ValueError("Archive part order differs")
        path = release / str(rel)
        if path.is_symlink():
            raise ValueError("Archive part cannot be a symlink")
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise ValueError("Archive part differs: " + str(rel))
        chunks.append(raw)
    raw = b"".join(chunks)
    if len(raw) != receipt["bytes"] or digest(raw) != receipt["sha256"]:
        raise ValueError("Assembled distribution differs from its receipt")
    if manifest["sha256"] != receipt["sha256"] or manifest["bytes"] != receipt["bytes"]:
        raise ValueError("Archive transport manifests differ")
    targets = [release / manifest["archive"],
               root / "book_e3/_static/e3_evidence_close_20261120/release" / manifest["archive"]]
    for target in targets:
        if target.is_symlink():
            raise ValueError("Assembled archive target cannot be a symlink")
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("Existing archive differs; preserve it before retrying")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    return {"status": "reviewed_archive_reassembled_from_git_parts",
            "parts": len(chunks), "bytes": len(raw), "sha256": digest(raw),
            "documentation_source_commit": receipt["documentation_source_commit"],
            "new_forecasting_model_fits": 0, "new_forecasting_model_predictions": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    print(json.dumps(assemble(args.repo_root), sort_keys=True))
