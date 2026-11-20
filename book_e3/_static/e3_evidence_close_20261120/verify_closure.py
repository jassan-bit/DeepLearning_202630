"""Read-only integrity check for the existing-evidence closure; no model imports.

AI assistance declared. Archive-only mode reports the historical files that are
absent rather than treating a limited documentary supplement as a full archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, *, base_root=None, source_only=False, allow_missing_protected=False):
    root = Path(root).resolve()
    base_root = Path(base_root).resolve() if base_root else root
    prefix = Path("experiments/e3_evidence_close_20261120")
    static = Path("book_e3/_static/e3_evidence_close_20261120")
    manifest = json.loads((root / prefix / "closure_manifest.json").read_text())
    checked, mirrored = 0, 0
    for record in manifest["closure_payloads"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        source = root / prefix / relative
        if not source.is_file() or source.is_symlink() or sha(source) != record["sha256"]:
            raise ValueError("Closure payload differs: " + str(relative))
        if source.stat().st_size != record["bytes"]:
            raise ValueError("Closure payload size differs: " + str(relative))
        checked += 1
        if not source_only:
            mirror = root / static / relative
            if not mirror.is_file() or mirror.is_symlink() or sha(mirror) != record["sha256"]:
                raise ValueError("Published download differs: " + str(relative))
            mirrored += 1
    for record in manifest["edited_root_files"]:
        path = root / record["path"]
        if not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError("Edited source differs: " + record["path"])
    optional_checked = 0
    for record in manifest.get("optional_downloads", []):
        path = root / prefix / record["path"]
        if source_only and not path.is_file():
            continue
        if not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError("Optional convenience download differs: " + record["path"])
        if not source_only and sha(root / static / record["path"]) != record["sha256"]:
            raise ValueError("Optional published download differs: " + record["path"])
        optional_checked += 1
    preserved, absent = 0, []
    for record in manifest["protected_existing_files"]:
        path = root / record["path"]
        if not path.is_file():
            path = base_root / record["path"]
        if not path.is_file():
            absent.append(record["path"])
            continue
        if path.is_symlink() or sha(path) != record["sha256"]:
            raise ValueError("Existing evidence/model artifact changed: " + record["path"])
        preserved += 1
    if absent and not allow_missing_protected:
        raise ValueError("Missing protected evidence; a supplement is not the whole repository: " + absent[0])
    policy = manifest["scope"]
    for key in ("new_forecasting_model_fits", "new_forecasting_model_predictions",
                "new_statistical_resamples", "new_confirmatory_superiority_tests"):
        if policy[key] != 0:
            raise ValueError("Execution scope changed: " + key)
    if policy["R1_accredited"] or policy["R3_accredited"] or policy["automatic_promotion_of_best_ablation"]:
        raise ValueError("Unsupported scientific accreditation")
    return {"status": "verified_existing_evidence_closure_integrity",
            "closure_payloads": checked, "published_mirrors": mirrored,
            "edited_root_files": len(manifest["edited_root_files"]),
            "protected_files_verified": preserved, "protected_files_absent": len(absent),
            "protected_manifest_files_fully_verified": not absent,
            "optional_downloads_verified": optional_checked,
            "mode": "source_only" if source_only else "source_and_downloads", **policy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--base-files-root", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--allow-missing-protected", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root, base_root=args.base_files_root,
                            source_only=args.source_only,
                            allow_missing_protected=args.allow_missing_protected), sort_keys=True))
