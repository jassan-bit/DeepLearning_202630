"""Seal an explicit reviewed list for the limited documentary ZIP, with no crawl.

This does not select a license or make the package a complete final deliverable.
Pass every intended member explicitly via --include relative/path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_documentary_archive import FROZEN_INPUT_KIND, frozen_input_path, relative


def make(root, names):
    root = Path(root).resolve()
    if not names or len(names) != len(set(names)):
        raise ValueError("A nonempty explicit list without duplicates is required")
    entries = []
    for name in sorted(names):
        frozen = frozen_input_path(name)
        relative(name, kind=FROZEN_INPUT_KIND if frozen else None)
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Source symlink refused")
        raw = path.read_bytes()
        suffix = path.suffix.lower()
        kind = FROZEN_INPUT_KIND if frozen else "source_code" if suffix == ".py" else "derived_summary_or_report" if suffix in {".csv", ".json", ".png", ".svg"} else "documentation"
        rights = "derived_results_author_review_required" if kind in {"derived_summary_or_report", FROZEN_INPUT_KIND} else "author_license_selection_pending"
        entries.append({"source": name, "archive_path": name, "kind": kind,
                        "rights_status": rights, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"schema_version": 1, "scope": "limited_documentary_supplement_not_full_training_reproduction",
            "evidence_base_commit": "29f9a26f1fd05484c6581ea6538ad7cb5a4781cb",
            "external_evidence": [
                {"kind": "796job_audit_terminal_delta_not_whole_master", "commit": "16a4504f6fb9208c7d6f89b65017194fa20470da", "path": "runs/e3_extended_audit/20261119T234320461006Z", "sequence": 193},
                {"kind": "30fit_V3_cumulative_snapshot", "commit": "8eaa581bc34b615a8507c9e384a3e4be29084720", "path": "runs/e3_v3_development_pilot/20261120T031424176339Z", "sequence": 93}],
            "entries": entries, "rights_review": "No license is granted or inferred by this input specification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    body = json.dumps(make(args.root, args.include), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if target.exists() and target.read_text() != body:
        raise ValueError("Existing spec differs; preserve it and choose a versioned new name")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    print(json.dumps({"spec": str(target), "members": len(args.include)}))
