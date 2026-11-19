"""Verify the published v1 bytes, independently of today's editable sources.

The v1 archive is a historical source snapshot. Current model/data cards may
evolve without rewriting that snapshot. This check pins the published archive
and external manifest, validates every archived member, and deterministically
repackages those frozen member bytes. It never imports or runs archived code.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path("experiments/e3_remediacion_20261118/reproduction")
ARCHIVE_NAME = "reproduction_bundle_v1.tar.gz"
MANIFEST_NAME = "reproduction_bundle_v1.manifest.json"
PREFIX = "e3-scientific-replay-v1"
SOURCE_COMMIT = "73d8958c50a69fbea46deb828f03f89385318789"
ARCHIVE_SHA256 = "c08265b7ff75793883359dc5bd84b2493975003c23848ec21d9f3de21d2509bd"
MANIFEST_SHA256 = "be60e16bcece6da5a02715d6d501640dc55a56212c3232a9f833a70d3ed9c960"
ARCHIVE_BYTES = 70337
MEMBER_COUNT = 31


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def member_payloads(raw):
    """Read regular, canonical members in memory; never extract to disk."""
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and "\\" not in member.name
                    and str(path) == member.name
                    and len(path.parts) >= 2 and path.parts[0] == PREFIX
                    and not any(part in {".", ".."} for part in path.parts),
                    "unsafe or noncanonical tar path: " + member.name)
            relative = str(PurePosixPath(*path.parts[1:]))
            require(relative not in result, "duplicate tar member: " + relative)
            require(member.isfile() and not member.issym() and not member.islnk(),
                    "non-regular tar member: " + relative)
            require(member.size <= 1_000_000, "oversized historical member")
            require((member.mtime, member.uid, member.gid, member.mode,
                     member.uname, member.gname) == (0, 0, 0, 0o644, "", "")
                    and not member.pax_headers,
                    "historical tar metadata changed: " + relative)
            stream = archive.extractfile(member)
            require(stream is not None, "missing tar member payload")
            payload = stream.read()
            require(len(payload) == member.size, "truncated tar member")
            result[relative] = payload
    require(list(result) == sorted(result), "tar member order changed")
    return result


def repack(payloads):
    """Use v1's published USTAR/gzip settings on its own member bytes."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for relative, raw in payloads.items():
            info = tarfile.TarInfo(PREFIX + "/" + relative)
            info.size, info.mtime, info.mode = len(raw), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(raw))
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", filename="", mtime=0,
                       compresslevel=9) as stream:
        stream.write(buffer.getvalue())
    return compressed.getvalue()


def verify(root=ROOT):
    folder = Path(root) / FOLDER
    archive_path, manifest_path = folder / ARCHIVE_NAME, folder / MANIFEST_NAME
    for path in (archive_path, manifest_path):
        require(path.is_file() and not path.is_symlink(), "expected regular historical file")
    raw, manifest_raw = archive_path.read_bytes(), manifest_path.read_bytes()
    require(len(raw) == ARCHIVE_BYTES and digest(raw) == ARCHIVE_SHA256,
            "published v1 archive size/SHA256 mismatch")
    require(digest(manifest_raw) == MANIFEST_SHA256,
            "published v1 external manifest SHA256 mismatch")
    manifest = json.loads(manifest_raw)
    require(manifest["schema_version"] == 1 and manifest["package"] == PREFIX
            and manifest["semantic_version"] == "1.0.0"
            and manifest["reference_source_commit"] == SOURCE_COMMIT,
            "historical manifest identity/source cut mismatch")
    require(manifest["archive"] == {"name": ARCHIVE_NAME, "bytes": ARCHIVE_BYTES,
                                    "sha256": ARCHIVE_SHA256},
            "historical manifest archive binding mismatch")
    require(manifest["persistent_deposit_completed"] is False and manifest["DOI"] is None
            and manifest["human_independent_reproduction_completed"] is False,
            "historical scope changed")
    payloads = member_payloads(raw)
    records = manifest["files"]
    require(len(records) == len(payloads) == MEMBER_COUNT, "historical member count mismatch")
    require([record["path"] for record in records] == list(payloads),
            "external manifest member list/order mismatch")
    for record in records:
        payload = payloads[record["path"]]
        require(len(payload) == record["bytes"] and digest(payload) == record["sha256"],
                "historical member hash/size mismatch: " + record["path"])
    reference = json.loads(payloads[str(FOLDER / "reference_manifest.json")])
    require(reference["source_commit"] == SOURCE_COMMIT,
            "embedded reference manifest source cut mismatch")
    require(repack(payloads) == raw, "historical deterministic repack mismatch")
    return {"status": "verified_published_historical_bundle", "name": ARCHIVE_NAME,
            "bytes": len(raw), "sha256": digest(raw), "files": len(payloads),
            "external_manifest_sha256": digest(manifest_raw),
            "reference_source_commit": SOURCE_COMMIT, "member_hashes_verified": True,
            "safe_regular_tar_paths_verified": True, "deterministic_repack_identical": True,
            "current_source_files_consulted": False, "model_fits": 0,
            "model_inferences": 0, "human_independent_reproduction_completed": False,
            "persistent_deposit_completed": False, "DOI": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
