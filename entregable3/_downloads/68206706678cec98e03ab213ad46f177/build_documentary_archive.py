"""Build/check an explicit, hash-bound documentary supplement; never publish it.

The ZIP is a limited supplement, not a full repository/data/weight archive and
not a reproduction of the 796 jobs. No network, fitting or inference is used.
Rights statuses are retained per file; this program grants no license.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

NAME = "e3-documentary-closure-20261120-DRAFT"
LIMIT = 32 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".onnx", ".npz", ".npy", ".parquet", ".pdf", ".zip", ".tar", ".gz", ".key", ".pem"}
RIGHTS = {"author_license_selection_pending", "derived_results_author_review_required", "metadata_only_no_new_license_granted"}
FROZEN_INPUT_KIND = "frozen_documentary_input_archive"
FROZEN_EXPANDED_LIMIT = 16 * 1024 * 1024
INNER_TEXT_SUFFIXES = {".json", ".csv", ".md", ".py", ".txt"}
SECRET_PATTERN = re.compile(rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_input_path(value):
    """Only the project's reviewed text records/derived CSV extracts qualify."""
    path = PurePosixPath(value)
    return (len(path.parts) >= 4
            and path.parts[:2] == ("experiments", "e3_evidence_close_20261120")
            and path.parts[2] in {"inventory", "adequacy"}
            and (path.name == "sources.zip" or path.name.endswith(".csv.gz")))


def relative(value, *, kind=None):
    path = PurePosixPath(value)
    if (not value or str(path) != value or path.is_absolute() or ".." in path.parts
            or any(x.startswith(".") for x in path.parts) or "\\" in value):
        raise ValueError("Unsafe or noncanonical archive path")
    frozen = kind == FROZEN_INPUT_KIND and frozen_input_path(value)
    if (path.suffix.lower() in FORBIDDEN_SUFFIXES and not frozen) or path.name in {"LICENSE", "LICENSE.txt"}:
        raise ValueError("Weights, raw binary data, paper bodies, nested archives and inferred licenses are excluded")
    return path


def checked_text(raw):
    if SECRET_PATTERN.search(raw):
        raise ValueError("A credential-like value appears in a selected member; preserve and review it")
    if b"\0" in raw:
        raise ValueError("Frozen documentary inputs must contain text, not binary data")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Frozen documentary inputs must be UTF-8 text") from error


def csv_header(text):
    first_line = text.split("\n", 1)[0]
    if len(first_line) > 65536:
        raise ValueError("Frozen CSV header exceeds declared bound")
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error) as error:
        raise ValueError("Frozen CSV requires a readable header") from error
    if not header or len(header) > 256 or any(not name.strip() for name in header) or len(set(header)) != len(header):
        raise ValueError("Frozen CSV requires nonempty unique header columns")
    if {"open", "high", "low", "close", "volume"}.issubset({name.lower() for name in header}):
        raise ValueError("Raw OHLCV inputs are outside the documentary supplement")
    return header


def validate_frozen_input(source, raw):
    """Audit permitted compressed inputs after bounded decompression."""
    source = relative(str(source), kind=FROZEN_INPUT_KIND)
    if source.name.endswith(".csv.gz"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
                body = stream.read(FROZEN_EXPANDED_LIMIT + 1)
        except (OSError, EOFError) as error:
            raise ValueError("Frozen CSV gzip is invalid") from error
        if len(body) > FROZEN_EXPANDED_LIMIT:
            raise ValueError("Frozen CSV expanded size exceeds declared bound")
        header = csv_header(checked_text(body))
        return {"format": "gzip_csv", "expanded_bytes": len(body), "expanded_sha256": digest(body),
                "header": header, "expanded_text_and_credentials_checked": True}
    if source.name != "sources.zip":
        raise ValueError("Unrecognized frozen documentary input")
    members, names, expanded = [], set(), 0
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
            if not 1 <= len(zipped.infolist()) <= 512:
                raise ValueError("Frozen text ZIP member count exceeds declared bound")
            for entry in zipped.infolist():
                inner = relative(entry.filename)
                if (entry.is_dir() or inner.suffix.lower() not in INNER_TEXT_SUFFIXES
                        or entry.flag_bits & 1 or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError("Frozen ZIP admits only unencrypted regular text members, without nested archives")
                if str(inner) in names:
                    raise ValueError("Frozen ZIP duplicate member")
                names.add(str(inner))
                if not 0 <= entry.file_size <= 8 * 1024 * 1024:
                    raise ValueError("Frozen ZIP member size exceeds declared bound")
                expanded += entry.file_size
                if expanded > FROZEN_EXPANDED_LIMIT:
                    raise ValueError("Frozen ZIP expanded size exceeds declared bound")
                with zipped.open(entry) as stream:
                    body = stream.read(FROZEN_EXPANDED_LIMIT + 1)
                if len(body) != entry.file_size:
                    raise ValueError("Frozen ZIP member size differs")
                text = checked_text(body)
                header = csv_header(text) if inner.suffix.lower() == ".csv" else None
                members.append({"path": str(inner), "bytes": len(body), "sha256": digest(body),
                                **({"header": header} if header is not None else {})})
    except (zipfile.BadZipFile, OSError, EOFError) as error:
        raise ValueError("Frozen documentary ZIP is invalid") from error
    return {"format": "zip_text_only", "expanded_bytes": expanded, "members": members,
            "expanded_text_and_credentials_checked": True}


def contents(root, spec, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Exact documentation source commit required")
    root = Path(root).resolve()
    result, members = {}, []
    for entry in spec["entries"]:
        kind = entry.get("kind")
        source = relative(entry["source"], kind=kind)
        target = relative(entry["archive_path"], kind=kind)
        if str(target) in result or str(target) == "MANIFEST.json":
            raise ValueError("Duplicate or reserved archive path")
        path = root / str(source)
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Source symlink refused")
        if entry["rights_status"] not in RIGHTS or entry.get("kind") == "raw_market_data":
            raise ValueError("Explicit unresolved rights/category required; raw market inputs excluded")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Documentary member exceeds declared limit")
        raw = path.read_bytes()
        if SECRET_PATTERN.search(raw):
            raise ValueError("A credential-like value appears in a selected member; preserve and review it")
        if digest(raw) != entry["sha256"] or len(raw) != entry["bytes"]:
            raise ValueError("Source content differs from the reviewed input spec: " + str(source))
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Documentary member exceeds declared limit")
        validation = validate_frozen_input(source, raw) if kind == FROZEN_INPUT_KIND else None
        result[str(target)] = raw
        members.append({**entry, "source": str(source), "archive_path": str(target),
                        **({"frozen_input_validation": validation} if validation is not None else {})})
    if sum(map(len, result.values())) > LIMIT:
        raise ValueError("Documentary supplement exceeds32MiB; use the separate full-archive plan")
    manifest = {"schema_version": 1, "scope": "limited_documentary_supplement_not_full_training_reproduction",
                "documentation_source_commit": source_commit,
                "source_commit_binding": "supplied by publisher; each member independently bound by bytes and SHA256",
                "evidence_base_commit": spec["evidence_base_commit"],
                "external_evidence": spec["external_evidence"], "members": members,
                "included_raw_market_data": False, "included_trained_weights": False,
                "included_full_paper_bodies": False, "license_selected": False,
                "verified_compressed_documentary_inputs_included": any(m.get("kind") == FROZEN_INPUT_KIND for m in members),
                "DOI_reserved": "10.5281/zenodo.24356217", "DOI_publication_verified": False,
                "AI_assistance_declared": True, "new_fits": 0, "new_calibrations": 0}
    result["MANIFEST.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    return result, manifest


def archive_bytes(content):
    stream = io.BytesIO()
    # ZIP_STORED avoids zlib-version dependence. Metadata order/time/modes fixed.
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(content):
            info = zipfile.ZipInfo(NAME + "/" + name, date_time=(2026, 11, 20, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content[name])
    return stream.getvalue()


def build(root, spec_path, source_commit, output, *, check=False):
    spec = json.loads(Path(spec_path).read_text())
    content, manifest = contents(root, spec, source_commit)
    raw = archive_bytes(content)
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Archive symlink refused")
    if check or output.exists():
        if output.read_bytes() != raw:
            raise ValueError("Existing archive is different; preserve it instead of overwriting")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist() != [NAME + "/" + name for name in sorted(content)]:
            raise ValueError("Archive members differ")
        for name, body in content.items():
            if archive.read(NAME + "/" + name) != body:
                raise ValueError("Archive content differs")
    receipt = {"status": "prepared_and_verified_not_published", "archive": output.name,
               "bytes": len(raw), "sha256": digest(raw), "members": len(content),
               "scope": manifest["scope"], "documentation_source_commit": source_commit,
               "license_selected": False, "DOI_publication_verified": False,
               "full_796_campaign_reproduced": False, "new_fits": 0, "new_calibrations": 0}
    receipt_path = output.with_suffix(output.suffix + ".manifest.json")
    body = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if receipt_path.exists() and receipt_path.read_text() != body:
        raise ValueError("Existing manifest differs; preserve it")
    if not check:
        receipt_path.write_text(body)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.spec, args.source_commit, args.output, check=args.check), ensure_ascii=False))
