"""Build the small, deterministic v1 scientific replay source archive.

Only the explicitly enumerated files are included. Forecast data, trained
weights, package wheels, timestamps and automatic execution receipts are not
part of this archive. No persistent identifier or redistribution right is
created by this operation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERSION = "1.0.0"
PREFIX = "e3-scientific-replay-v1"
ARCHIVE = HERE / "reproduction_bundle_v1.tar.gz"
MANIFEST = HERE / "reproduction_bundle_v1.manifest.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def included_paths():
    reference = json.loads((HERE / "reference_manifest.json").read_text())
    paths = set(reference["code"])
    paths.update(table["reference"] for table in reference["tables"])
    # The script's copy boundary is the same in a checkout and extracted bundle.
    paths.update(str((HERE / name).relative_to(ROOT)) for name in (
        "reproduce.py", "make_bundle.py", "input_manifest.json",
        "reference_manifest.json", "README.md"))
    paths.update((
        "deployment/science/Dockerfile",
        "deployment/science/Dockerfile.dockerignore",
        "deployment/science/requirements.lock.txt",
        "deployment/science/offline_packages/.gitignore",
        "deployment/science/offline_packages/.gitkeep",
        "tests/test_e3_scientific_replay.py",
        "DATA_STATEMENT.md",
    ))
    return sorted(paths)


def materialize():
    records = []
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for relative in included_paths():
            path = ROOT / relative
            if not path.is_file() or path.is_symlink():
                raise ValueError("Expected a regular, non-symlink file: " + relative)
            raw = path.read_bytes()
            info = tarfile.TarInfo(PREFIX + "/" + relative)
            info.size = len(raw)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(raw))
            records.append({"path": relative, "bytes": len(raw), "sha256": digest(raw)})
    gzip_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=gzip_buffer, mode="wb", filename="", mtime=0,
                       compresslevel=9) as compressed:
        compressed.write(tar_buffer.getvalue())
    archive_raw = gzip_buffer.getvalue()
    manifest = {
        "schema_version": 1,
        "package": PREFIX,
        "semantic_version": VERSION,
        "scope": "automated_CPU_aggregate_replay_from_frozen_forecasts",
        "reference_source_commit": json.loads((HERE / "reference_manifest.json").read_text())["source_commit"],
        "archive": {"name": ARCHIVE.name, "bytes": len(archive_raw), "sha256": digest(archive_raw)},
        "determinism": {"entry_order": "lexicographic", "tar_format": "USTAR", "mtime": 0,
                        "uid": 0, "gid": 0, "mode": "0644", "gzip_filename": "", "gzip_mtime": 0},
        "inputs": "Downloaded separately using input_manifest.json; pinned public commits, size and SHA-256.",
        "excluded": ["frozen_forecast_data", "raw_market_data", "trained_weights", "package_wheels",
                     "execution_receipts", "git_history", "personal_credentials"],
        "rights_status": "Source/data redistribution rights remain as documented in DATA_STATEMENT.md; no new license is granted.",
        "persistent_deposit_completed": False,
        "DOI": None,
        "human_independent_reproduction_completed": False,
        "files": records,
    }
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    return archive_raw, manifest_raw, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic reconstruction without replacing either file")
    args = parser.parse_args()
    archive_raw, manifest_raw, manifest = materialize()
    for path, raw in ((ARCHIVE, archive_raw), (MANIFEST, manifest_raw)):
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError("Existing v1 artifact differs; preserve it and create a new package version: " + str(path))
        elif args.check:
            raise ValueError("Missing artifact: " + str(path))
        else:
            path.write_bytes(raw)
    print(json.dumps({"status": "verified" if args.check else "created_or_identical",
                      "version": VERSION, "files": len(manifest["files"]), **manifest["archive"]}))


if __name__ == "__main__":
    main()
