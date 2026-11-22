"""Freeze addon bytes after parser and operational tests; never rewrite S2."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ("ops/resume_development.py", "ops/resume_worker.py", "parser/cm_header_parser.py",
         "REAL_CM_HEADER_DIAGNOSTIC.json", "ops/README.md")


def main():
    from resume_development import VERSION, SOURCE_COMMIT, SOURCE_MANIFEST_SHA256, SOURCE_DATA_SHA256
    files = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    manifest = dict(version=VERSION, parent_source_commit=SOURCE_COMMIT,
        parent_manifest_sha256=SOURCE_MANIFEST_SHA256, parent_data_sha256=SOURCE_DATA_SHA256,
        scope="resume_existing_F02_F03_development_only",
        preserved_physical_adapter_version="crypto-markets-bounded-archives-base-volume-v2.1",
        execution_identity="Explicit parser header overlay; not pristine S2 execution",
        AI_assistance="Implementation and software review; no signatures or new author declarations", files=files)
    path = ROOT / "ops/REPAIR_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), files=files)))


if __name__ == "__main__":
    main()
