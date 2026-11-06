#!/usr/bin/env python3
"""Validate and package a frozen editorial title/abstract screening ledger.

The decisions are reviewed judgments in decisions.csv, not predictions generated
by this script. Rebuilding reproduces the summaries/flow, not the searches or a
new independent editorial review. Raw abstracts remain outside the repository.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = ROOT / "references/literature_e1/editorial_screening_20261106"
NATIVE = ROOT / "references/literature_e1/native_search_20261106"
DEFAULT_MIRROR = ROOT / "book/_static/literature_e1/editorial_screening_20261106"
DECISIONS = {"exclude_core", "contextual_only", "seek_full_text", "full_text_documented_elsewhere"}
PUBLIC_FILES = {"decisions.csv", "identity_review.csv", "README.md", "summary.json", "flow.svg"}
LINKED_FICHES = {"E1N037": "S8", "E1N038": "S1", "E1N073": "PR_CVI", "E1N111": "PR_GARCH"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def validate(stage: Path, native: Path = NATIVE) -> tuple[list[dict], list[dict]]:
    records = {row["group_id"]: row for row in read_csv(native / "records.csv")}
    occurrences = read_csv(native / "occurrences.csv")
    rows = read_csv(stage / "decisions.csv")
    identifiers = [row["group_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(records):
        raise ValueError("The editorial ledger must cover each native group exactly once")
    occurrence_ids = {row["occurrence_id"] for row in occurrences}
    covered = set()
    for row in rows:
        record = records[row["group_id"]]
        if row["decision"] not in DECISIONS or not row["rationale"].strip():
            raise ValueError("Missing/invalid editorial decision or rationale")
        if row["title"] != record["title"] or row["doi"] != record["doi"]:
            raise ValueError("Bibliographic identity differs from native ledger")
        if row["evidence_abstract_sha256"] != record["abstract_sha256"]:
            raise ValueError("Abstract fingerprints differ from native export parsing")
        if (row["full_text_required"] == "true") != (row["decision"] == "seek_full_text"):
            raise ValueError("Full-text requirement contradicts decision")
        if row["decision"] == "full_text_documented_elsewhere" and row["group_id"] not in LINKED_FICHES:
            raise ValueError("Unsubstantiated link to an existing full-text fiche")
        is_future = record["nominal_year_after_received_year"] == "true"
        if is_future != (row["effective_date_status"] == "pending_effective_publication_date"):
            raise ValueError("Nominal future year must retain effective-date uncertainty")
        row_occurrences = set(row["evidence_occurrence_ids"].split(";"))
        expected = {o["occurrence_id"] for o in occurrences if o["group_id"] == row["group_id"]}
        if row_occurrences != expected:
            raise ValueError("Editorial evidence does not match native occurrences")
        covered |= row_occurrences
    if covered != occurrence_ids:
        raise ValueError("Editorial evidence does not cover native occurrences")
    identities = read_csv(stage / "identity_review.csv")
    expected_identity_ids = {r["group_id"] for r in records.values() if "title_link" in r["identity_status"]}
    if {r["group_id"] for r in identities} != expected_identity_ids:
        raise ValueError("Every provisional title group requires an identity review entry")
    return rows, identities


def json_write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(stage: Path, mirror: Path | None, native: Path = NATIVE) -> dict:
    rows, identities = validate(stage, native)
    counts = dict(sorted(Counter(row["decision"] for row in rows).items()))
    future_ids = [r["group_id"] for r in rows if r["effective_date_status"] == "pending_effective_publication_date"]
    native_summary = json.loads((native / "summary.json").read_text())
    summary = {
        "schema_version": 1,
        "screening_date": "2026-11-06",
        "reviewer": "Codex; single AI-assisted editorial pass, parallel reading batches; no independent human dual review",
        "evidence_scope": "Titles and actual exported abstracts; existing fiches are linked, not claimed as newly read originals",
        "decisions_are": "Frozen editorial judgments; script validates and summarizes them, not an automatic eligibility classifier",
        "native_occurrences": sum(int(r["occurrence_count"]) for r in read_csv(native / "records.csv")),
        "native_working_groups": len(rows),
        "screened_working_groups": len(rows),
        "decision_counts": counts,
        "core_exclusion_reason_counts": dict(sorted(Counter(r["reason_code"] for r in rows if r["decision"] == "exclude_core").items())),
        "awaiting_full_text_ids": [r["group_id"] for r in rows if r["decision"] == "seek_full_text"],
        "existing_full_text_fiche_links": LINKED_FICHES,
        "new_full_texts_read_in_this_screening_pass": 0,
        "new_final_full_text_inclusions_in_this_pass": 0,
        "historical_N7_unchanged": True,
        "historical_N7": 7,
        "new_primary_fulltexts_previously_documented": 2,
        "denominator_rule": "Four links comprise two historical studies and two separately documented new primary fiches. Do not add them again to N7; do not mix price/RV/CVI/hybrid strata into a new common frequency denominator.",
        "historical_eligibility_not_reassessed": True,
        "historical_overlap_by_doi": {r["group_id"]: r["previous_e1_study_ids_by_doi"] for r in rows if r["previous_e1_study_ids_by_doi"]},
        "global_overlap_reconciliation": "Not completed across earlier search stages; this flow covers only four native exports received on 2026-11-06",
        "nominal_post_2026_groups": future_ids,
        "nominal_post_2026_occurrences": sum(int(r["occurrence_count"]) for r in read_csv(native / "records.csv") if r["group_id"] in future_ids),
        "year_rule": "No exclusion for nominal 2027; effective publication date remains pending, including records excluded for an independently observed target mismatch",
        "title_identity_review_counts": dict(sorted(Counter(r["editorial_identity_decision"] for r in identities).items())),
        "identity_uncertainty": "E1N097 remains unresolved; working groups are not a definitive count of unique studies",
        "full_text_eligibility_criteria": [
            "Verify numerical cryptocurrency volatility target and separate RV, rolling dispersion, implied CVI, risk/regime and price constructs",
            "Verify deep-learning comparator roles independently from hybrid components",
            "Verify architecture, metrics, temporal protocol, repetitions and quality before final inclusion",
        ],
        "remaining_search_evidence": "Exact executed strings/dates/totals/export ranges, IEEE consultation and five effective publication dates remain unverified by this editorial stage",
        "source_records_sha256": digest(native / "records.csv"),
        "source_occurrences_sha256": digest(native / "occurrences.csv"),
        "source_manifest_sha256": digest(native / "source_manifest.json"),
        "source_original_export_sha256": {s["source_key"]: s["sha256"] for s in json.loads((native / "source_manifest.json").read_text())["sources"]},
        "source_native_summary_sha256": digest(native / "summary.json"),
        "editorial_ledger_sha256": digest(stage / "decisions.csv"),
        "identity_review_sha256": digest(stage / "identity_review.csv"),
        "raw_exports_and_abstract_text_republished": False,
        "selection_chronology": "This editorial review is later than both experimental phases; it does not preregister or retrospectively justify their selection independently",
    }
    json_write(stage / "summary.json", summary)
    branches = [("exclude_core", "Excluidos del núcleo", "#fee2e2"), ("contextual_only", "Contexto / revisiones / volúmenes", "#e2e8f0"), ("seek_full_text", "Requieren original", "#fef3c7"), ("full_text_documented_elsewhere", "Fichas completas ya documentadas", "#dcfce7")]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="510" viewBox="0 0 1100 510" role="img" aria-label="Flujo del cribado editorial de 182 grupos provisionales">', '<rect width="1100" height="510" fill="white"/>', '<g font-family="sans-serif" fill="#0f172a" text-anchor="middle">', '<text x="550" y="32" font-size="22">Cribado editorial por título y resumen · 6 noviembre 2026</text>', '<rect x="350" y="52" width="400" height="58" rx="10" fill="#dbeafe"/><text x="550" y="87" font-size="18">244 ocurrencias · cuatro exportaciones</text>', '<path d="M550 110 V135" stroke="#64748b" stroke-width="2"/>', '<rect x="350" y="135" width="400" height="58" rx="10" fill="#dbeafe"/><text x="550" y="170" font-size="18">182 grupos provisionales · 182 examinados</text>']
    for i, (key, label, color) in enumerate(branches):
        x = 18 + i * 273
        svg += [f'<path d="M550 193 V220 H{x+125} V250" fill="none" stroke="#64748b" stroke-width="2"/>', f'<rect x="{x}" y="250" width="250" height="105" rx="10" fill="{color}"/>', f'<text x="{x+125}" y="290" font-size="30">{counts.get(key, 0)}</text>', f'<text x="{x+125}" y="323" font-size="13">{label}</text>']
    svg += ['<text x="550" y="395" font-size="17">4 enlaces = 2 históricos + 2 fichas nuevas previas; 0 originales nuevos leídos en esta pasada</text>', '<text x="550" y="430" font-size="16">N=7 histórico sin cambios · 22 originales por evaluar · identidad E1N097 pendiente</text>', '<text x="550" y="466" font-size="15">Flujo de esta etapa; no PRISMA acumulado ni corpus final de estudios incluidos</text>', '</g></svg>']
    (stage / "flow.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
    for name in PUBLIC_FILES:
        if not (stage / name).is_file():
            raise ValueError("Missing public stage file: " + name)
    manifest = {"schema_version": 1, "scope": "Public editorial metadata and original rationales; no export abstracts/contact fields", "files": {name: {"sha256": digest(stage / name), "bytes": (stage / name).stat().st_size} for name in sorted(PUBLIC_FILES)}}
    json_write(stage / "manifest.json", manifest)
    if mirror is not None:
        mirror.mkdir(parents=True, exist_ok=True)
        for name in PUBLIC_FILES | {"manifest.json"}:
            shutil.copyfile(stage / name, mirror / name)
    return summary


def verify(stage: Path, native: Path = NATIVE) -> None:
    validate(stage, native)
    manifest = json.loads((stage / "manifest.json").read_text())
    if set(manifest["files"]) != PUBLIC_FILES:
        raise ValueError("Unexpected public manifest entries")
    for name, metadata in manifest["files"].items():
        path = stage / name
        if digest(path) != metadata["sha256"] or path.stat().st_size != metadata["bytes"]:
            raise ValueError("Public stage integrity failure: " + name)
    summary = json.loads((stage / "summary.json").read_text())
    rows, _ = validate(stage, native)
    if summary["decision_counts"] != dict(Counter(r["decision"] for r in rows)):
        raise ValueError("Summary does not match editorial ledger")
    if summary["editorial_ledger_sha256"] != digest(stage / "decisions.csv"):
        raise ValueError("Summary fingerprint does not match editorial ledger")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.stage)
        print("PASS: editorial coverage, identities and public hashes")
    else:
        summary = build(args.stage, None if args.no_mirror else args.mirror)
        verify(args.stage)
        print(json.dumps(summary["decision_counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
