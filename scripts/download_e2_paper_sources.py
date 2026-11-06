#!/usr/bin/env python3
"""Download hash-pinned E2 paper sources and ETTh1; never import ML or train.

The design contains dataset {url, sha256, filename} and sources keyed by
TimeMixer and controls. Each source declares repository, commit, archive_url,
archive_sha256, archive_filename, extract_directory and files [{path, sha256}].
The output is user-selected; original datasets and author code are not added to
the repository by this tool. Verified caches are reused without network access.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "experiments/e2_paper_guide_20261106/design.json"
MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024
MAX_DATASET_BYTES = 32 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 10000
RECEIPT_NAME = "source_acquisition_receipt.json"


def digest(payload):
    return sha256(payload).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def checked_hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("A lowercase SHA-256 is required for every pinned asset")
    return value


def safe_relative(value, single_component=False):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Unsafe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {"..", "."} or ":" in p for p in path.parts):
        raise ValueError("Unsafe relative path")
    if single_component and len(path.parts) != 1:
        raise ValueError("Asset filenames must be single path components")
    return path


def destination(root, relative):
    relative = safe_relative(relative)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlinks are not allowed in acquisition destinations")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Acquisition destination escapes the output directory")
    return current


def checked_url(url):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in
            {"raw.githubusercontent.com", "codeload.github.com"} or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Pinned assets must use direct HTTPS GitHub source URLs")
    return url


def validate_design(design):
    data = design["dataset"]
    safe_relative(data["filename"], single_component=True)
    checked_hash(data["sha256"]); checked_url(data["url"])
    sources = design["sources"]
    if not isinstance(sources, dict) or not {"TimeMixer", "controls"} <= set(sources):
        raise ValueError("TimeMixer and controls source receipts are required")
    destinations = {data["filename"], RECEIPT_NAME}
    for source in sources.values():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["repository"]):
            raise ValueError("Invalid repository identity")
        if not re.fullmatch(r"[0-9a-f]{40}", source["commit"]):
            raise ValueError("A complete immutable Git commit is required")
        expected = "https://codeload.github.com/{}/zip/{}".format(source["repository"], source["commit"])
        if checked_url(source["archive_url"]) != expected:
            raise ValueError("Source archive URL must identify its pinned repository and commit")
        checked_hash(source["archive_sha256"])
        for key in ("archive_filename", "extract_directory"):
            name = source[key]
            safe_relative(name, single_component=True)
            if name in destinations:
                raise ValueError("Acquisition destinations must be distinct")
            destinations.add(name)
        expected_files = source["files"]
        if not isinstance(expected_files, list) or not expected_files:
            raise ValueError("Each source must pin the files needed by the replica")
        names = []
        for receipt in expected_files:
            names.append(str(safe_relative(receipt["path"])))
            checked_hash(receipt["sha256"])
        if len(set(names)) != len(names):
            raise ValueError("Duplicate pinned source file")


def fetch_bytes(url):
    request = Request(url, headers={"User-Agent": "E2PaperSourceAcquisition/1.0"})
    with urlopen(request, timeout=45) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Download exceeds the allowed size")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Download exceeds the allowed size")
    return payload


def write_new_or_equal(path, payload):
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError("Existing file differs; refusing to overwrite: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def asset(root, filename, url, expected_hash, limit, fetcher):
    path = destination(root, filename)
    if path.exists():
        if not path.is_file():
            raise ValueError("Existing asset is not a regular file: " + filename)
        if path.stat().st_size > limit:
            raise ValueError("Cached asset exceeds the allowed size")
        payload = path.read_bytes()
    else:
        payload = fetcher(url)
    if not isinstance(payload, bytes) or len(payload) > limit:
        raise ValueError("Downloaded asset exceeds the allowed size")
    if digest(payload) != expected_hash:
        raise ValueError("SHA-256 mismatch for " + filename)
    write_new_or_equal(path, payload)
    return payload


def extract_verified(root, payload, source):
    """Validate all ZIP entries and cached destinations before writing sources."""
    prefix = source["extract_directory"]
    files = {}
    directories = set()
    total = 0
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError("Too many ZIP members")
        seen = set()
        for info in infos:
            relative = safe_relative(info.filename)
            name = str(relative)
            if relative.parts[0] != prefix:
                raise ValueError("ZIP member has an unexpected source root")
            if name in seen:
                raise ValueError("Duplicate ZIP member")
            seen.add(name)
            mode = stat.S_IFMT(info.external_attr >> 16)
            if mode not in {0, stat.S_IFREG, stat.S_IFDIR} or info.flag_bits & 1:
                raise ValueError("Symlink, special or encrypted ZIP member is forbidden")
            target = destination(root, name)
            if info.is_dir():
                if target.exists() and not target.is_dir():
                    raise ValueError("ZIP directory collides with an existing file")
                directories.add(name)
                continue
            if len(relative.parts) < 2 or info.file_size > MAX_MEMBER_BYTES:
                raise ValueError("Invalid source file or oversized ZIP member")
            total += info.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise ValueError("ZIP expands beyond the allowed size")
            with archive.open(info) as stream:
                content = stream.read(MAX_MEMBER_BYTES + 1)
            if len(content) != info.file_size or len(content) > MAX_MEMBER_BYTES:
                raise ValueError("ZIP member size does not match its declaration")
            if target.exists() and (not target.is_file() or target.read_bytes() != content):
                raise ValueError("Existing source differs; refusing to overwrite: " + name)
            files[name] = content
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents):
            raise ValueError("ZIP file/directory path collision")
    for name in directories:
        if name in files:
            raise ValueError("ZIP file/directory path collision")
    verified = []
    for receipt in source["files"]:
        name = prefix + "/" + str(safe_relative(receipt["path"]))
        if name not in files or digest(files[name]) != receipt["sha256"]:
            raise ValueError("Pinned source file is missing or differs: " + receipt["path"])
        verified.append({"path": name, "sha256": receipt["sha256"], "bytes": len(files[name])})
    for name in sorted(directories):
        destination(root, name).mkdir(parents=True, exist_ok=True)
    for name, content in sorted(files.items()):
        write_new_or_equal(destination(root, name), content)
    return {"extracted_file_count": len(files), "extracted_bytes": total,
            "verified_files": verified}


def acquire(design, output_dir, fetcher=None):
    validate_design(design)
    root = Path(output_dir)
    if root.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("Output directory is not a directory")
    fetcher = fetcher or fetch_bytes
    data = design["dataset"]
    payload = asset(root, data["filename"], data["url"], data["sha256"], MAX_DATASET_BYTES, fetcher)
    receipt = {"schema_version": 1, "scope": "Pinned source acquisition only; no model training",
               "dataset": {"path": data["filename"], "url": data["url"],
                           "sha256": data["sha256"], "bytes": len(payload)}, "sources": {},
               "model_fits": 0, "ML_packages_imported": False}
    for role, source in sorted(design["sources"].items()):
        payload = asset(root, source["archive_filename"], source["archive_url"],
                        source["archive_sha256"], MAX_DOWNLOAD_BYTES, fetcher)
        extracted = extract_verified(root, payload, source)
        receipt["sources"][role] = {"repository": source["repository"], "commit": source["commit"],
            "archive_path": source["archive_filename"], "archive_url": source["archive_url"],
            "archive_sha256": source["archive_sha256"], "archive_bytes": len(payload),
            "source_directory": source["extract_directory"], **extracted}
    write_new_or_equal(destination(root, RECEIPT_NAME), json_bytes(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = acquire(json.loads(args.design.read_bytes()), args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print("Source acquisition failed: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps({"dataset_sha256": receipt["dataset"]["sha256"],
                      "source_archives": len(receipt["sources"]), "model_fits": 0,
                      "receipt": str(args.output_dir / RECEIPT_NAME)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
