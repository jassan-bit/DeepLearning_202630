"""Lossless deterministic gzip packaging AFTER an audit has completed."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def pack(report_path, package_root):
    report_path, package_root = Path(report_path), Path(package_root)
    report = json.loads(report_path.read_text())
    plain, zipped = report_path.parent/"replicates.jsonl", report_path.parent/"replicates.jsonl.gz"
    if not plain.exists():
        raise FileNotFoundError("Plain local audit must exist before evidence packaging")
    digest, size = sha(plain), plain.stat().st_size
    if digest != report["raw_results_sha256"]:
        raise RuntimeError("Audit raw evidence hash mismatch before packaging")
    count = 0
    with plain.open("rb") as source, zipped.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0, compresslevel=9) as compressed:
            for line in source:
                json.loads(line)  # Packaging verifies, never filters, every record.
                count += 1
                compressed.write(line)
    check = hashlib.sha256()
    with gzip.open(zipped, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            check.update(block)
    if check.hexdigest() != digest:
        raise RuntimeError("Lossless gzip verification failed; plain evidence retained")
    report.setdefault("evidence", {})["replicates"] = {
        "path": str(zipped.relative_to(package_root)), "format": "jsonl+gzip",
        "sha256": sha(zipped), "uncompressed_sha256": digest,
        "bytes": zipped.stat().st_size, "uncompressed_bytes": size,
        "records": count, "gzip_mtime": 0,
    }
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n")
    folder = Path(tempfile.mkdtemp(prefix="temporal-audit-plain-"))
    destination = folder/f"{report_path.parent.name}-replicates.jsonl"
    shutil.move(str(plain), str(destination))
    return {"gzip": str(zipped), "local_plain": str(destination), "sha256": report["evidence"]["replicates"]["sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.report.resolve(), args.package_root.resolve()), sort_keys=True))
