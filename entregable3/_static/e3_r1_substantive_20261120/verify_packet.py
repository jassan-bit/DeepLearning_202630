"""Verify this documentary R1 review without loading or importing models.

AI assistance declared. Integrity and algebra checks do not certify novelty.
The recorded technical verdict is a separately reasoned review, not this code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy


PREFIX = Path("experiments/e3_r1_substantive_20261120")
MIRROR = Path("book_e3/_static/e3_r1_substantive_20261120")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regular_files(root):
    return {str(path.relative_to(root)) for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix != ".pyc"}


def verify(root, *, check_receipt=True):
    root = Path(root).resolve()
    packet = root / PREFIX
    manifest_path = packet / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    excluded = {"packet_manifest.json", "verification.json"}
    expected = {entry["path"] for entry in manifest["payloads"]}
    if regular_files(packet) != expected | excluded:
        raise ValueError("Packet inventory differs from its explicit manifest")
    if regular_files(root / MIRROR) != expected | excluded:
        raise ValueError("Published inventory differs from the source packet")
    for entry in manifest["payloads"]:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe manifest path")
        for prefix in [PREFIX, MIRROR]:
            path = root / prefix / str(relative)
            if path.is_symlink() or not path.is_file():
                raise ValueError("Expected regular documentary payload: " + str(path))
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError("Documentary payload changed: " + str(path))
    for name in excluded:
        left, right = packet / name, root / MIRROR / name
        if left.is_symlink() or right.is_symlink() or left.read_bytes() != right.read_bytes():
            raise ValueError("Source/publication metadata differs: " + name)

    historical = root / "experiments/e3_evidence_close_20261120"
    if digest(historical / "closure_manifest.json") != manifest["historical_closure_manifest_sha256"]:
        raise ValueError("Historical closure manifest changed")
    old_receipt = runpy.run_path(str(historical / "verify_closure.py"))["verify"](root)
    theory = runpy.run_path(str(packet / "code_theory/verify_static_audit.py"))["verify"](
        root, packet / "code_theory")

    review = json.loads((packet / "independent_review/technical_verdict.json").read_text())
    decision = review["decision"]
    if decision["reference_additive_R1_supported"] or not decision["V3_R1_technically_supported"]:
        raise ValueError("The versioned review decision differs from this packet's scope")
    for name in ["institutional_accreditation_claimed", "R2_student_defense_accredited",
                 "R3_accredited", "project_grade_or_cap_assigned"]:
        if decision[name]:
            raise ValueError("Review scope changed: " + name)
    if review["pilot"]["continuation_heuristic_passed"] or review["pilot"]["automatic_promotion_of_best_ablation"]:
        raise ValueError("The adverse pilot decision changed")
    for name in ["new_model_fits", "new_forecasting_predictions", "new_statistical_resamples"]:
        if review[name] != 0:
            raise ValueError("Review execution scope changed: " + name)

    seeds, edges = set(), 0
    for module in ["literature_application", "literature_distribution"]:
        with (packet / module / "citation_followup.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            seed = row.get("seed_id") or row.get("seed")
            if not seed:
                raise ValueError("Citation follow-up lacks its seed identity")
            seeds.add((module, seed))
        edges += len(rows)
    if len(seeds) != 5 or edges != 7:
        raise ValueError("Citation seeds/edges changed")

    result = {"status": "documentary_R1_packet_integrity_and_preservation_verified",
            "packet_manifest_sha256": digest(manifest_path),
            "AI_assistance_declared": True,
            "payloads_verified": len(expected), "published_mirrors_verified": len(expected),
            "historical_payloads_preserved": old_receipt["closure_payloads"],
            "historical_protected_files_preserved": old_receipt["protected_files_verified"],
            "historical_root_files_preserved": old_receipt["edited_root_files"],
            "theory_inputs_verified": theory["inputs_verified"],
            "exact_identity_coefficients": theory["identity"]["monomial_coefficients_verified"],
            "previous_finite_law_cases_checked": theory["prior_finite_law_rows_consistency_checked"],
            "citation_seeds": len(seeds), "new_followup_edges": edges,
            "recorded_IA_review": "favorable_scoped_for_V3_only",
            "novelty_decided_by_this_verifier": False,
            "institutional_accreditation_claimed": False,
            "new_model_fits": 0, "new_forecasting_predictions": 0,
            "new_statistical_resamples": 0, "model_modules_imported": 0}
    if check_receipt and json.loads((packet / "verification.json").read_text()) != result:
        raise ValueError("Saved verification receipt differs from the current read-only check")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root), ensure_ascii=False, sort_keys=True))
