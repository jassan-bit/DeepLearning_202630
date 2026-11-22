"""Verify the unchanged S2 source; this command never runs a market model."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

S2_COMMIT = "db74afd0763596f71bcbafb4b9150a44a2e83339"
MANIFEST_SHA256 = "2c7dc052d2a53f212688340b50b65bed5fa367ff16fb5ed0a2474d417696715b"
CANDIDATE_SHA256 = "d07a106d1bfc8721edbf994b9e4227659d07d6d26a0d32d661c91a747e1592c2"
SEEDS_SHA256 = "137fe45eca358081307c041ab31dd94efb23f533202438e2eb6888e4e79836ee"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source: Path) -> dict:
    source = source.resolve(strict=True)
    manifest_path = source / "SOURCE_MANIFEST.json"
    if digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError("S2 manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    checked = []
    for name, expected in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe source member")
        path = (source / relative).resolve(strict=True)
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError("S2 member mismatch: " + name)
        checked.append(name)
    if len(checked) != 304:
        raise ValueError("S2 expected exactly 304 manifested source members")
    for name, expected in (
        ("configs/CANDIDATE_IDENTITY.json", CANDIDATE_SHA256),
        ("configs/FINAL_SEED_BANK.json", SEEDS_SHA256),
    ):
        if digest(source / name) != expected:
            raise ValueError("Immutable identity mismatch: " + name)
    candidate = json.loads((source / "configs/CANDIDATE_IDENTITY.json").read_text())
    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": S2_COMMIT,
        "manifest_sha256": MANIFEST_SHA256,
        "manifested_source_files_unchanged": len(checked),
        "candidate_id": candidate["candidate_id"],
        "candidate_identity_sha256": CANDIDATE_SHA256,
        "seed_bank_sha256": SEEDS_SHA256,
        "historical_rules_or_results_rewritten": False,
        "market_fits_executed_by_audit": 0,
        "model_forwards_executed_by_audit": 0,
        "reserved_test_opened_by_audit": False,
        "scope": "Byte identity verification only; no academic acceptance or experimental superiority is implied",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
