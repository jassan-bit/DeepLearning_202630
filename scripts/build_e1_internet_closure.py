#!/usr/bin/env python3
"""Compile reviewed E1 recovery fiches; never search, infer eligibility or fit models.

The explicit adjudications below distinguish reading a document from closing its
strict eligibility decision. Source fiches are retained verbatim as metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil

from refresh_e1_native_downloads import validate_public_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "references/literature_e1"
STAGE = BASE / "internet_closure_20261106"
MIRROR = ROOT / "book/_static/literature_e1/internet_closure_20261106"
EDITORIAL = BASE / "editorial_screening_20261106/decisions.csv"
NATIVE = BASE / "native_search_20261106/records.csv"
SEARCH = BASE / "search_recovery_20261106"
SEARCH_MIRROR = ROOT / "book/_static/literature_e1/search_recovery_20261106"
PUBLICATION_SCHEMA = "e1_public_metadata_allowlist_v1"
NATIVE_IDS = frozenset("E1N007 E1N053 E1N057 E1N074 E1N075 E1N080 E1N091 E1N118 E1N120 E1N121 E1N124 E1N129 E1N133 E1N138 E1N142 E1N152 E1N163 E1N169 E1N170 E1N173 E1N175 E1N177".split())
COMPLEMENTARY_IDS = frozenset("E1X001 E1X002 E1X003 E1X004".split())
ORIGINAL_READ_IDS = frozenset("E1N053 E1N075 E1N118 E1N163 E1X003".split())
TRANSCRIPTION_IDS = frozenset({"E1N091"})
CLOSED_NATIVE_IDS = frozenset("E1N053 E1N075 E1N091 E1N163 E1N170".split())
INPUT_FILES = (
    "part_a/README.md", "part_a/read_originals.bib", "part_a/reviews.json",
    "part_b/README.md", "part_b/reviews.json",
    "part_c/README.md", "part_c/manifest.sha256.json", "part_c/public_web_query_log.json", "part_c/reviews.json",
    "part_d/README.md", "part_d/consultations.json", "part_d/read_sources.bib", "part_d/reviews.json",
    "part_e/README.md", "part_e/manifest.sha256.json", "part_e/public_web_query_log.json", "part_e/reviews.json",
    "part_root/README.md", "part_root/public_web_query_log.json", "part_root/reviews.json",
)
OUTPUT_FILES = frozenset({"summary.json", "decisions.csv", "reviews.json", "model_evidence.csv", "README.md"})
PUBLIC_FILES = frozenset(INPUT_FILES) | OUTPUT_FILES
SEARCH_FILES = frozenset({"README.md", "search_log.json", "source_reconciliation.csv", "query_plan.csv", "publication_dates.json", "related_lead.json", "summary.json"})
REQUIRED = frozenset({"group_id", "title", "doi", "source_urls", "access_level", "full_text_read", "evidence_locations", "target", "models_and_roles", "architecture_details", "protocol", "quality_issues", "eligibility_decision", "eligibility_reason", "metrics"})

# Reviewed current decisions, not automatically classified from titles/abstracts.
CLOSED_REASONS = {
    "E1N053": "Original leído; clasificación y pérdida no definida, sin objetivo/receta operacional suficientes para el núcleo de regresión de volatilidad.",
    "E1N075": "Manuscrito de autor leído; excluir del núcleo estricto actual y conservar como contexto intradía por proxy y receta no conciliadas e inconsistencias de reporte. Puede reabrirse ante nueva evidencia.",
    "E1N091": "Transcripción completa de espejo, no PDF editorial; insuficiencia observada de objetivo/receta permite cierre estricto contextual sujeto a revisión si aparece original.",
    "E1N163": "Original leído; definición del objetivo de volatilidad y protocolo insuficientes; contexto solamente.",
    "E1N170": "Método primario parcial resuelve objetivo de precio del día siguiente; cierre de pertinencia sin declarar lectura íntegra.",
}

# (GRU, LSTM, TCN) autonomous-presence evidence. None increments historical N7.
MODEL_STATUS = {
    "E1N007": ("announced_only", "announced_only", "not_verified"),
    "E1N053": ("not_present_in_comparison", "hybrid_only", "independent_original_target_unresolved"),
    "E1N057": ("announced_price_role_unresolved", "announced_price_role_unresolved", "not_verified"),
    "E1N074": ("announced_only", "announced_only", "not_verified"),
    "E1N075": ("independent_original_intraday", "independent_original_intraday", "not_present_in_comparison"),
    "E1N080": ("hybrid_announced_only", "not_verified", "not_verified"),
    "E1N091": ("independent_mirror_transcription", "independent_mirror_transcription", "component_only"),
    "E1N118": ("not_present_in_comparison", "independent_original_target_pending", "independent_original_target_pending"),
    "E1N120": ("hybrid_announced_only", "hybrid_announced_only", "not_verified"),
    "E1N121": ("announced_only", "announced_only", "not_verified"),
    "E1N124": ("hybrid_announced_only", "hybrid_announced_only", "not_verified"),
    "E1N129": ("not_verified", "announced_only", "not_verified"),
    "E1N133": ("announced_only", "announced_only", "not_verified"),
    "E1N138": ("announced_only", "announced_only", "not_verified"),
    "E1N142": ("not_verified", "announced_only", "announced_only"),
    "E1N152": ("announced_only", "announced_only", "not_verified"),
    "E1N163": ("independent_original_target_unresolved", "independent_original_target_unresolved", "component_only"),
    "E1N169": ("not_verified", "not_verified", "title_mention_only"),
    "E1N170": ("independent_primary_partial_price", "independent_primary_partial_price", "not_verified"),
    "E1N173": ("announced_only", "announced_only", "not_verified"),
    "E1N175": ("not_verified", "not_verified", "announced_only"),
    "E1N177": ("announced_only", "announced_only", "not_verified"),
    "E1X001": ("not_present_in_comparison", "independent_primary_main_text_tables_pending", "not_present_in_comparison"),
    "E1X002": ("not_present_in_comparison", "independent_author_code_version_pending", "independent_author_code_version_pending"),
    "E1X003": ("not_present_in_comparison", "independent_original_daily_rv", "component_only"),
    "E1X004": ("not_verified", "announced_only", "announced_only"),
}
STRATA = {"E1N053": "volatility_states_target_insufficient", "E1N075": "intraday_variance_proxy_unreconciled", "E1N091": "risk_factor_target_insufficient", "E1N118": "multioutput_volatility_target_pending", "E1N163": "risk_target_insufficient", "E1N170": "price", "E1X001": "six_hour_realized_variance", "E1X002": "daily_squared_centered_return", "E1X003": "daily_realized_variance", "E1X004": "realized_variance_abstract_unverified"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(path: Path) -> dict:
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: str(v).lower() if type(v) is bool else v for k, v in row.items()})
    return output.getvalue().encode("utf-8")


def checked_file(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(directory)):
        raise ValueError("Expected nonsymlink public file: " + str(path))
    validate_public_bytes(name, path.read_bytes())
    return path


def load_inputs(stage: Path, editorial: Path, native: Path) -> list[dict]:
    found = {p.relative_to(stage).as_posix() for p in stage.glob("part_*/*") if p.is_file()}
    if found != set(INPUT_FILES):
        raise ValueError("Subpart inventory differs from explicit reviewed public inputs")
    for name in INPUT_FILES:
        checked_file(stage, name)
    for part in ("part_c", "part_e"):
        local = json.loads((stage / part / "manifest.sha256.json").read_text())
        if set(local) != {"README.md", "reviews.json", "public_web_query_log.json"}:
            raise ValueError("Unexpected subpart hash manifest")
        if any(digest(stage / part / name) != sha for name, sha in local.items()):
            raise ValueError("Subpart hash manifest mismatch: " + part)
    queue = {r["group_id"] for r in read_csv(editorial) if r["decision"] == "seek_full_text"}
    if queue != NATIVE_IDS:
        raise ValueError("Native queue must equal the exact frozen 22 editorial IDs")
    native_records = {r["group_id"]: r for r in read_csv(native)}
    all_rows = []
    for name in INPUT_FILES:
        if name.endswith("/reviews.json"):
            rows = json.loads((stage / name).read_text())
            if not isinstance(rows, list):
                raise ValueError("Review input must be an array: " + name)
            for row in rows:
                if not REQUIRED <= set(row) or type(row["full_text_read"]) is not bool:
                    raise ValueError("Invalid review record: " + name)
                item = copy.deepcopy(row)
                item["_input_file"] = name
                all_rows.append(item)
    identifiers = [r["group_id"] for r in all_rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != NATIVE_IDS | COMPLEMENTARY_IDS:
        raise ValueError("Review coverage must be exactly 22 unique native and four complementary IDs")
    native_dois = {r["doi"].lower() for r in native_records.values() if r["doi"]}
    for row in all_rows:
        group = row["group_id"]
        doi = (row["doi"] or "").lower()
        if group in NATIVE_IDS and doi != native_records[group]["doi"].lower():
            raise ValueError("DOI identity differs from native group: " + group)
        if group in COMPLEMENTARY_IDS and doi and doi in native_dois:
            raise ValueError("Complementary DOI already occurs in native corpus: " + group)
        expected_complete = group in ORIGINAL_READ_IDS | TRANSCRIPTION_IDS
        if row["full_text_read"] != expected_complete:
            raise ValueError("Source reading state changed; explicit adjudication needs review: " + group)
    return sorted(all_rows, key=lambda r: r["group_id"])


def compile_reviews(rows: list[dict]) -> list[dict]:
    compiled = []
    for source in rows:
        group = source["group_id"]
        native = group in NATIVE_IDS
        if native:
            status = "closed_excluded_context" if group in CLOSED_NATIVE_IDS else "pending_target_supplement" if group == "E1N118" else "pending_original"
            reason = CLOSED_REASONS.get(group, source["eligibility_reason"])
        else:
            status = {"E1X001": "complementary_pending_tables_recipe", "E1X002": "complementary_pending_original_version", "E1X003": "complementary_rv_context_reviewed", "E1X004": "complementary_pending_original"}[group]
            reason = source["eligibility_reason"]
        original = group in ORIGINAL_READ_IDS
        source_copy = {k: v for k, v in source.items() if k != "_input_file"}
        compiled.append({
            "group_id": group, "title": source["title"], "doi": source["doi"],
            "scope": "native_22" if native else "complementary_outside_22",
            "input_file": source["_input_file"], "source_access_level": source["access_level"],
            "source_full_text_read": source["full_text_read"],
            "original_read": original,
            "original_read_version": "accepted_author_manuscript_html" if group == "E1N075" else "publisher_original_public_representation" if original else None,
            "complete_mirror_transcription_read": group in TRANSCRIPTION_IDS,
            "complete_document_read": source["full_text_read"],
            "strict_eligibility_closed": native and group in CLOSED_NATIVE_IDS,
            "strict_eligibility_status": status,
            "strict_eligibility_decision": "exclude_current_core_retain_context" if group in CLOSED_NATIVE_IDS else "pending" if native else "outside_native_strict_corpus",
            "current_decision_reason": reason,
            "source_eligibility_decision": source["eligibility_decision"],
            "target_stratum": STRATA.get(group, "target_not_verified"),
            "recipe_for_simple_e1_comparison_verified": False,
            "historical_N7_inclusion": False, "strict_recurrence_increment": 0,
            "source_review": source_copy,
        })
    return compiled


def model_evidence(reviews: list[dict]) -> list[dict]:
    rows = []
    for review in reviews:
        group = review["group_id"]
        for model, status in zip(("GRU", "LSTM", "TCN"), MODEL_STATUS[group]):
            rows.append({"group_id": group, "scope": review["scope"], "model": model,
                "autonomous_presence_evidence": status,
                "autonomy_verified_from_original": status.startswith("independent_original"),
                "autonomy_verified_in_author_code": status == "independent_author_code_version_pending",
                "target_stratum": review["target_stratum"],
                "recipe_for_simple_e1_comparison_verified": False,
                "strict_eligibility_status": review["strict_eligibility_status"],
                "strict_volatility_recurrence_increment": 0,
                "historical_N7_inclusion": False, "evidence_file": review["input_file"]})
    return rows


def summarize(reviews: list[dict]) -> dict:
    native = [r for r in reviews if r["scope"] == "native_22"]
    extra = [r for r in reviews if r["scope"] != "native_22"]
    ids = lambda rows, field: [r["group_id"] for r in rows if r[field]]
    return {"schema_version": 1, "review_date_america_bogota": "2026-11-06",
        "purpose": "Consolidación reproducible de fichas revisadas; el script no busca, lee originales de nuevo ni clasifica por palabras clave.",
        "native_queue_size": len(native), "native_ids": [r["group_id"] for r in native],
        "native_originals_read": len(ids(native, "original_read")), "native_originals_read_ids": ids(native, "original_read"),
        "native_complete_document_read": len(ids(native, "complete_document_read")), "native_complete_document_read_ids": ids(native, "complete_document_read"),
        "native_complete_mirror_transcriptions": len(ids(native, "complete_mirror_transcription_read")), "native_complete_mirror_transcription_ids": ids(native, "complete_mirror_transcription_read"),
        "native_strict_decisions_closed": len(ids(native, "strict_eligibility_closed")), "native_strict_closed_ids": ids(native, "strict_eligibility_closed"),
        "native_strict_pending": sum(not r["strict_eligibility_closed"] for r in native),
        "native_strict_pending_ids": [r["group_id"] for r in native if not r["strict_eligibility_closed"]],
        "native_pending_supplement_ids": ["E1N118"], "native_new_strict_inclusions": 0,
        "native_status_counts": dict(sorted(Counter(r["strict_eligibility_status"] for r in native).items())),
        "complementary_sources": len(extra), "complementary_ids": [r["group_id"] for r in extra],
        "complementary_originals_read": len(ids(extra, "original_read")), "complementary_originals_read_ids": ids(extra, "original_read"),
        "complementary_status_counts": dict(sorted(Counter(r["strict_eligibility_status"] for r in extra).items())),
        "historical_N7": 7, "historical_N7_changed": False, "strict_recurrence_increment": {"GRU": 0, "LSTM": 0, "TCN": 0},
        "reading_does_not_imply_inclusion": True,
        "E1N075_current_adjudication": CLOSED_REASONS["E1N075"],
        "E1N091_scope": "Completa transcripción secundaria pública; no se declara original editorial recuperado.",
        "E1N170_scope": "Cierre del objetivo desde método primario parcial; no lectura íntegra.",
        "native_searches_recovered_by_this_stage": 0, "ieee_native_consultation_accredited_by_this_stage": False,
        "search_evidence_stage": "../search_recovery_20261106/summary.json",
        "native_search_requirement_closed": False,
        "reviewer": "Una revisión editorial asistida por IA, con lecturas paralelas y consolidación explícita; no revisión humana dual independiente.",
        "chronology": "Evidencia posterior a los experimentos; no prerregistro retrospectivo ni nueva justificación ex ante de selección.",
        "raw_abstracts_contact_fields_or_pdfs_republished": False,
        "reviewed_subpart_files": len(INPUT_FILES),
    }


def readme(summary: dict) -> str:
    pending = ", ".join(summary["native_strict_pending_ids"])
    return f"""# Cierre bibliográfico por internet · E1

La cola congelada contiene **22 grupos nativos**. Se revisó evidencia para los 22: **4 originales leídos**, **1 transcripción completa de espejo**, **5 decisiones estrictas cerradas** y **17 pendientes**. Las cuatro fuentes E1X001–004 se mantienen separadas; solo E1X003 tiene lectura de original completa.

| Dimensión nativa | Cantidad | Identificadores |
| --- | ---: | --- |
| Original de autor/editorial leído | 4 | E1N053, E1N075, E1N118, E1N163 |
| Transcripción completa, no original editorial | 1 | E1N091 |
| Decisión estricta cerrada: excluido/contexto | 5 | E1N053, E1N075, E1N091, E1N163, E1N170 |
| Pendiente de original | 16 | Véase decisions.csv |
| Original leído, suplemento/objetivo pendiente | 1 | E1N118 |

Estas dimensiones se superponen; no se suman como un flujo de artículos. E1N075 se leyó como manuscrito aceptado de autor, pero se excluye del núcleo estricto actual por proxy/receta no conciliadas e inconsistencias; se conserva como contexto y puede reabrirse ante evidencia nueva. E1N091 es una transcripción secundaria completa, sin PDF editorial recuperado. E1N170 cierra pertinencia desde método primario parcial que identifica precio; no suma lectura íntegra. E1N118 necesita confirmar en suplemento la salida de volatilidad evaluada.

Pendientes estrictos: {pending}.

**N7 permanece en 7; cero nuevas inclusiones nativas y cero incrementos de recurrencia estricta GRU/LSTM/TCN.** model_evidence.csv distingue presencia autónoma, objetivo, receta simple y elegibilidad. Una fila de TCN en clasificación, una salida multivariable sin volatilidad aislada o un módulo convolucional no resuelven conjuntamente esos criterios. E1X002 confirma un builder TCN autónomo en código de autor, con versión/tablas pendientes; E1X003 respalda LSTM en RV diaria, sin receta final simple equivalente a E1. No se combinan proxies ni horizontes.

reviews.json conserva íntegra cada ficha derivada bajo source_review y añade la adjudicación vigente, sin reescribir las fuentes. decisions.csv es el índice de estados; summary.json ofrece conteos e identificadores. Los {len(INPUT_FILES)} archivos de las subpartes son entradas públicas explícitamente revisadas, con huellas en manifest.json. No se incluyen PDFs, textos originales completos, resúmenes nativos ni contactos.

Reproducción: python scripts/build_e1_internet_closure.py. Verificación sin escritura: python scripts/build_e1_internet_closure.py --verify. El compilador valida cobertura exacta, identidad DOI, estados, hashes, privacidad de campos y espejo público. Empaqueta evidencia guardada: no ejecuta búsquedas, descarga originales ni entrena modelos. Las consultas públicas no acreditan historial nativo WoS/Scopus/IEEE; los faltantes de cadenas literales, fechas y totales siguen en search_recovery_20261106.

Esta consolidación es posterior a los experimentos y preserva su cronología; no constituye un prerregistro retrospectivo. El espejo de publicación está en book/_static/literature_e1/internet_closure_20261106.
"""


def mirror_files(stage: Path, mirror: Path, names: set[str]) -> None:
    if mirror.is_symlink() or any(parent.is_symlink() for parent in mirror.parents):
        raise ValueError("Symlink in public mirror destination")
    if mirror.exists() and any(path.is_symlink() for path in mirror.rglob("*")):
        raise ValueError("Symlink in public mirror contents")
    existing = {p.relative_to(mirror).as_posix() for p in mirror.rglob("*") if p.is_file()} if mirror.exists() else set()
    if existing - names:
        raise ValueError("Unapproved neighbouring public mirror file")
    for name in sorted(names):
        destination = mirror / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checked_file(stage, name), destination)


def update_search_manifest(stage: Path, mirror: Path | None) -> None:
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if set(manifest.get("files", {})) != SEARCH_FILES:
        raise ValueError("Search recovery public inventory must remain unchanged")
    for name in SEARCH_FILES:
        path = checked_file(stage, name)
        if metadata(path) != manifest["files"][name]:
            raise ValueError("Search recovery evidence changed after its review: " + name)
    manifest["publication_schema"] = PUBLICATION_SCHEMA
    manifest["approved_public_files"] = sorted(SEARCH_FILES)
    json_write(manifest_path, manifest)
    if mirror is not None:
        mirror_files(stage, mirror, set(SEARCH_FILES) | {"manifest.json"})


def build(stage: Path = STAGE, mirror: Path | None = MIRROR, editorial: Path = EDITORIAL, native: Path = NATIVE,
          search_stage: Path | None = SEARCH, search_mirror: Path | None = SEARCH_MIRROR) -> dict:
    rows = load_inputs(stage, editorial, native)
    reviews = compile_reviews(rows)
    summary = summarize(reviews)
    decisions = [{k: r[k] for k in ("group_id", "title", "doi", "scope", "source_access_level", "source_full_text_read", "original_read", "complete_mirror_transcription_read", "strict_eligibility_closed", "strict_eligibility_status", "strict_eligibility_decision", "current_decision_reason", "target_stratum", "recipe_for_simple_e1_comparison_verified", "historical_N7_inclusion", "input_file")} for r in reviews]
    json_write(stage / "reviews.json", reviews)
    json_write(stage / "summary.json", summary)
    (stage / "decisions.csv").write_bytes(csv_bytes(decisions))
    (stage / "model_evidence.csv").write_bytes(csv_bytes(model_evidence(reviews)))
    (stage / "README.md").write_text(readme(summary), encoding="utf-8")
    for name in PUBLIC_FILES:
        checked_file(stage, name)
    manifest = {"schema_version": 1, "publication_schema": PUBLICATION_SCHEMA,
        "approved_public_files": sorted(PUBLIC_FILES),
        "scope": "Reviewed derived metadata only; no PDF, original text, native abstract or contact payloads.",
        "reviewed_public_inputs": sorted(INPUT_FILES),
        "native_queue_evidence": {"editorial_decisions_sha256": digest(editorial), "native_records_sha256": digest(native)},
        "compiler": "scripts/build_e1_internet_closure.py",
        "files": {name: metadata(stage / name) for name in sorted(PUBLIC_FILES)}}
    json_write(stage / "manifest.json", manifest)
    if mirror is not None:
        mirror_files(stage, mirror, set(PUBLIC_FILES) | {"manifest.json"})
    if search_stage is not None:
        update_search_manifest(search_stage, search_mirror)
    verify(stage, mirror, editorial, native)
    return summary


def verify(stage: Path = STAGE, mirror: Path | None = MIRROR, editorial: Path = EDITORIAL, native: Path = NATIVE) -> None:
    expected_reviews = compile_reviews(load_inputs(stage, editorial, native))
    manifest = json.loads((stage / "manifest.json").read_text())
    if (manifest.get("publication_schema") != PUBLICATION_SCHEMA or manifest.get("approved_public_files") != sorted(PUBLIC_FILES)
            or set(manifest.get("files", {})) != PUBLIC_FILES or manifest.get("reviewed_public_inputs") != sorted(INPUT_FILES)):
        raise ValueError("Public approval manifest differs from exact reviewed inventory")
    for name in PUBLIC_FILES:
        path = checked_file(stage, name)
        if manifest["files"][name] != metadata(path):
            raise ValueError("Public evidence hash/size mismatch: " + name)
    if manifest["native_queue_evidence"] != {"editorial_decisions_sha256": digest(editorial), "native_records_sha256": digest(native)}:
        raise ValueError("Native queue provenance changed")
    if json.loads((stage / "reviews.json").read_text()) != expected_reviews:
        raise ValueError("Compiled reviews differ from explicit source adjudications")
    summary = summarize(expected_reviews)
    if json.loads((stage / "summary.json").read_text()) != summary:
        raise ValueError("Summary differs from reviewed evidence")
    if (stage / "model_evidence.csv").read_bytes() != csv_bytes(model_evidence(expected_reviews)):
        raise ValueError("Model evidence matrix differs from reviewed roles")
    expected_decisions = [{k: r[k] for k in ("group_id", "title", "doi", "scope", "source_access_level", "source_full_text_read", "original_read", "complete_mirror_transcription_read", "strict_eligibility_closed", "strict_eligibility_status", "strict_eligibility_decision", "current_decision_reason", "target_stratum", "recipe_for_simple_e1_comparison_verified", "historical_N7_inclusion", "input_file")} for r in expected_reviews]
    if (stage / "decisions.csv").read_bytes() != csv_bytes(expected_decisions) or (stage / "README.md").read_text() != readme(summary):
        raise ValueError("Decision ledger or explanation differs from reviewed evidence")
    if mirror is not None:
        found = {p.relative_to(mirror).as_posix() for p in mirror.rglob("*") if p.is_file()}
        if found != PUBLIC_FILES | {"manifest.json"}:
            raise ValueError("Public mirror inventory mismatch")
        for name in found:
            if checked_file(stage, name).read_bytes() != checked_file(mirror, name).read_bytes():
                raise ValueError("Public mirror byte mismatch: " + name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        print("PASS: exact 22+4 coverage, reading/eligibility distinction, roles, hashes and public mirror")
    else:
        summary = build()
        print(json.dumps({key: summary[key] for key in ("native_queue_size", "native_originals_read", "native_complete_mirror_transcriptions", "native_strict_decisions_closed", "native_strict_pending", "complementary_sources")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
