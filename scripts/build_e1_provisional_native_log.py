#!/usr/bin/env python3
"""Compile an explicitly provisional native-search worksheet from saved metadata.

This does not execute searches or recover historical queries, dates or displayed
totals. Export counts are exact; the separate export-compatible bounds, reference
dates and proposed queries never become verified native-search/PRISMA evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from refresh_e1_native_downloads import safe_member, validate_public_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "references/literature_e1"
STAGE = BASE / "provisional_native_log_20261107"
MIRROR = ROOT / "book/_static/literature_e1/provisional_native_log_20261107"
PUBLICATION_SCHEMA = "e1_public_metadata_allowlist_v1"
PUBLIC_FILES = frozenset({"README.md", "provisional_log.csv", "summary.json", "provenance.json"})
INPUT_SHA256 = {
    "search_recovery_20261106/source_reconciliation.csv": "71647142baf633ad9377ca431b2e35b823d72170507dd5e58790ac410ecf1b98",
    "native_search_20261106/source_manifest.json": "ae80451bcea7686d6fa3875c3fa9f8a9a0a4063336a15a0828561a359cf1cb72",
    "native_search_context_20261106.json": "580b53a3352bc3aa1601021f5bff32decaf96133d4c95748bfe9798278364061",
    "native_search_20261106/occurrences.csv": "005199f0d4088563d188db6b0d265021700245d5c705ec64c705348c51e2da48",
    "search_recovery_20261106/query_plan.csv": "1dd3679566e7b7eee63c6773b9eb7888e574f0d964bc2ef9dfda762e39d7a85e",
    "native_search_20261106/summary.json": "6474360911be95ecf956f4b9e984adb7ebde54ecda20cb14055dc6e0f7accbf9",
    "crypto_extension_20261105/evidence_matrix.json": "4b5eb36f0db801f15df64a5f04cc053a107165a86ff89f755afe9e548ec72f82",
}
SOURCE_ORDER = ("W02", "W01", "S01", "S02")
CSV_FIELDS = (
    "log_id", "database", "source_key", "source_filename", "query_family",
    "status", "editable", "exported_occurrences_verified", "distinct_source_ids_verified",
    "export_compatible_lower_bound", "lower_bound_display", "lower_bound_scope",
    "native_total_verified", "native_query_literal_verified", "native_query_literal_recovered",
    "native_search_date_verified", "native_export_range_verified", "native_execution_accredited_this_stage",
    "date_reference", "date_reference_display", "date_reference_timezone", "date_reference_scope",
    "publication_year_filter", "publication_year_filter_status", "query_proposed", "query_proposal_status",
    "query_proposal_source_ids", "interface", "field_scope",
    "exclude_from_verified_native_search_totals", "exclude_from_prisma_totals", "pending_verification",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata(data: bytes) -> dict:
    return {"sha256": digest(data), "bytes": len(data)}


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def csv_read(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def csv_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ("true" if value is True else "false" if value is False else value)
                         for key, value in row.items()})
    return output.getvalue().encode("utf-8")


def checked_file(base: Path, name: str) -> bytes:
    safe_member(name)
    path = base / name
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Symlink in source/destination directory")
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Expected regular file within directory: " + name)
    return path.read_bytes()


def inventory(base: Path) -> set[str]:
    if base.is_symlink() or any(parent.is_symlink() for parent in base.parents):
        raise ValueError("Symlink in source/destination directory")
    names = set()
    if base.exists():
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlink in source/destination directory")
            if path.is_file():
                name = path.relative_to(base).as_posix()
                safe_member(name)
                names.add(name)
    return names


def load_inputs(base: Path) -> dict[str, bytes]:
    raw = {name: checked_file(base, name) for name in INPUT_SHA256}
    for name, data in raw.items():
        if digest(data) != INPUT_SHA256[name]:
            raise ValueError("Frozen source hash mismatch: " + name)
    return raw


def query_parts(query: str, database: str) -> tuple[str, str, str]:
    """Separate common terms/family from the unchanged native-plan wrapper."""
    prefix, suffix = {"WoS": ("TS=(", ")"),
                      "Scopus": ("TITLE-ABS-KEY(", ") AND PUBYEAR > 2018 AND PUBYEAR < 2027"),
                      "IEEE": ("", "")}[database]
    if not query.startswith(prefix) or (suffix and not query.endswith(suffix)):
        raise ValueError("Unexpected query-plan syntax")
    body = query[len(prefix):len(query) - len(suffix) if suffix else None]
    common, family = body.rsplit(" AND (", 1)
    if not family.endswith(")"):
        raise ValueError("Unbalanced query-plan family")
    return common, family[:-1], suffix


def proposed_query(plans: dict[str, dict], database: str, families: tuple[str, ...]) -> dict:
    proposals = [plans[f"{database}_{family}"] for family in families]
    if any(p["status"] != "proposed_not_executed" or p["executed_date"] or p["displayed_total"]
           or p["exported_range"] or p["publication_years_requested"] != "2019–2026" for p in proposals):
        raise ValueError("Query proposal is not an unexecuted 2019–2026 plan")
    parts = [query_parts(p["query_proposed"], database) for p in proposals]
    if len({(part[0], part[2]) for part in parts}) != 1:
        raise ValueError("Proposals have inconsistent common terms or year syntax")
    prefix = {"WoS": "TS=(", "Scopus": "TITLE-ABS-KEY(", "IEEE": ""}[database]
    common, _, suffix = parts[0]
    query = prefix + common + " AND (" + " OR ".join(part[1] for part in parts) + ")" + suffix
    if query.count("(") != query.count(")"):
        raise ValueError("Unbalanced combined query proposal")
    return {"query_proposed": query, "query_proposal_status": "proposed_not_historical",
            "query_proposal_source_ids": ";".join(p["proposal_id"] for p in proposals),
            "interface": proposals[0]["interface"], "field_scope": proposals[0]["field_scope"]}


def derive_rows(raw: dict[str, bytes]) -> tuple[list[dict], dict]:
    reconciliation = csv_read(raw["search_recovery_20261106/source_reconciliation.csv"])
    manifest = json.loads(raw["native_search_20261106/source_manifest.json"])
    context = json.loads(raw["native_search_context_20261106.json"])
    occurrences = csv_read(raw["native_search_20261106/occurrences.csv"])
    native_summary = json.loads(raw["native_search_20261106/summary.json"])
    plans = {p["proposal_id"]: p for p in csv_read(raw["search_recovery_20261106/query_plan.csv"])}
    sources = {s["source_key"]: s for s in manifest["sources"]}
    reconciliation = {s["filename"]: s for s in reconciliation}
    families = {s["filename"]: s["query_family"] for s in context["sources"]}
    if (set(sources) != set(SOURCE_ORDER) or len(reconciliation) != 4 or len(families) != 4
            or context["exact_query_confirmation"] != "not_confirmed"
            or context["publication_years"] != {"start": 2019, "end": 2026}
            or manifest["received_date"] != "2026-11-06"
            or context["evidence"] != {"kind": "user_confirmation", "date_local": "2026-11-06", "timezone": "America/Bogota"}):
        raise ValueError("Unexpected native source/context scope")
    if (len(occurrences) != 244 or native_summary["exported_occurrences"] != 244
            or len({r["group_id"] for r in occurrences}) != 182
            or native_summary["candidate_groups_provisional"] != 182
            or native_summary["historical_e1"]["crypto_full_text_denominator_preserved"] != 7
            or native_summary["historical_e1"]["frequency_update_performed"] is not False
            or native_summary["historical_e1"]["sha256"] != INPUT_SHA256["crypto_extension_20261105/evidence_matrix.json"]):
        raise ValueError("Historical import/N7 scope changed")
    rows = []
    for key in SOURCE_ORDER:
        source = sources[key]
        filename = source["filename"]
        recon = reconciliation[filename]
        family = families[filename]
        observed = [row for row in occurrences if row["source_key"] == key]
        count = len(observed)
        if (not observed or any(not r["source_id"] or r["database"] != source["database"] for r in observed)
                or count != source["exported_occurrences"] or count != int(recon["exported_occurrences"])
                or source["database"] != recon["database"] or source["sha256"] != recon["sha256"]
                or source["bytes"] != int(recon["bytes"]) or source["query_family_association"] != family
                or recon["query_family_user_reported"] != family
                or any(recon[field] != "False" for field in ("exact_query_recovered", "native_displayed_total_recovered",
                        "native_export_range_recovered", "query_execution_date_recovered"))
                or any(source[field] is not None for field in ("displayed_result_total", "executed_query", "exported_result_range"))):
            raise ValueError("Source reconciliation or identifier evidence differs: " + key)
        # Count distinct native IDs within each export; repeated rows cannot inflate a bound.
        lower_bound = len({r["source_id"] for r in observed})
        scopus = source["database"] == "Scopus"
        reference_date = filename.rsplit("_", 1)[1][:10] if scopus else manifest["received_date"]
        if reference_date != "2026-11-06":
            raise ValueError("Unexpected reference date; never substitute execution date")
        row = common_row()
        row.update({"log_id": "E1-PROV-" + key, "database": source["database"], "source_key": key,
                    "source_filename": filename, "query_family": family, "status": "provisional_lower_bound",
                    "exported_occurrences_verified": count, "distinct_source_ids_verified": lower_bound,
                    "export_compatible_lower_bound": lower_bound, "lower_bound_display": "≥" + str(lower_bound),
                    "lower_bound_scope": "Distinct native IDs in this export; compatible bound only; native total unknown",
                    "date_reference": reference_date, "date_reference_display": "06/11/2026 (referencia)",
                    "date_reference_timezone": None if scopus else "America/Bogota (context only)",
                    "date_reference_scope": "export_filename_date_unknown_timezone" if scopus else "received_and_user_confirmation_date_not_execution_date",
                    "publication_year_filter_status": "user_reported_not_independently_verified_in_export"})
        row.update(proposed_query(plans, "Scopus" if scopus else "WoS", ("TCN",) if family == "TCN" else ("GRU", "LSTM")))
        rows.append(row)
    ieee = common_row()
    ieee.update({"log_id": "E1-PROV-IEEE", "database": "IEEE Xplore", "source_key": None,
                 "source_filename": None, "query_family": "TCN/GRU/LSTM", "status": "planned_not_executed",
                 "exported_occurrences_verified": None, "distinct_source_ids_verified": None,
                 "export_compatible_lower_bound": None, "lower_bound_display": "Sin base para estimar",
                 "lower_bound_scope": "No native export or accredited execution; unknown does not mean zero",
                 "date_reference": None, "date_reference_display": None, "date_reference_timezone": None,
                 "date_reference_scope": "planned_not_executed_no_native_date",
                 "publication_year_filter_status": "planned_not_executed"})
    ieee.update(proposed_query(plans, "IEEE", ("TCN", "GRU", "LSTM")))
    rows.append(ieee)
    return rows, native_summary


def common_row() -> dict:
    return {"editable": True, "native_total_verified": None, "native_query_literal_verified": None,
            "native_query_literal_recovered": False, "native_search_date_verified": None,
            "native_export_range_verified": None, "native_execution_accredited_this_stage": False,
            "publication_year_filter": "2019–2026", "exclude_from_verified_native_search_totals": True,
            "exclude_from_prisma_totals": True,
            "pending_verification": "Historical literal query, execution date/time, interface/collection, applied filters, displayed total and exported range"}


def summarize(rows: list[dict], native_summary: dict) -> dict:
    return {"schema_version": 1, "log_status": "PROVISIONAL_EDITABLE", "compiled_date_utc": "2026-11-07",
            "objective_3_1_closed": False,
            "row_count": len(rows), "status_counts": dict(Counter(r["status"] for r in rows)),
            "new_native_search_executions": 0, "verified_native_search_totals_added": 0,
            "verified_native_search_dates_added": 0, "recovered_historical_query_literals_added": 0,
            "provisional_export_compatible_lower_bounds_by_source_key": {r["source_key"]: r["export_compatible_lower_bound"] for r in rows if r["source_key"]},
            "ieee": {"status": rows[-1]["status"], "export_compatible_lower_bound": None,
                     "native_total_verified": None, "native_search_date_verified": None, "execution_accredited": False},
            "preserved_native_import": {"exported_occurrences": native_summary["exported_occurrences"],
                                       "candidate_groups_provisional": native_summary["candidate_groups_provisional"]},
            "historical_e1": {"crypto_full_text_denominator_preserved": 7, "frequency_update_performed": False,
                              "experiments_changed": False},
            "excluded_from_verified_native_search_totals": True, "excluded_from_prisma_totals": True,
            "exclusion_scope": "Provisional export-compatible bounds, reference dates and proposed queries; exact historical import counts remain unchanged",
            "interpretation": "Bounds describe distinct IDs in each saved export. They are neither native total estimates nor verified displayed totals; proposed queries are not recovered historical literals.",
            "rows": rows}


def readme(rows: list[dict]) -> str:
    table = []
    for row in rows:
        table.append("| " + " | ".join([row["database"], row["query_family"], row["date_reference_display"] or "No ejecutada",
                    str(row["exported_occurrences_verified"]) if row["exported_occurrences_verified"] is not None else "Desconocido",
                    row["lower_bound_display"], row["status"]]) + " |")
    return '''# Bitácora nativa PROVISIONAL y editable · E1

Este registro completa una hoja de trabajo provisional a partir de evidencia guardada. **Sus cotas, fechas de referencia y cadenas propuestas quedan pendientes de actualización y no acreditan totales nativos, consultas históricas ni nuevas ejecuciones.** El registro se compila el 07/11/2026 (UTC); esa fecha tampoco es una fecha de búsqueda.

| Base | Familia | Fecha de referencia | Filas exportadas exactas | Cota compatible con el export | Estado |
| --- | --- | --- | ---: | --- | --- |
''' + "\n".join(table) + '''

Las cotas ≥8, ≥59, ≥19 y ≥158 se apoyan en IDs nativos distintos dentro de cada archivo, contados por `source_key`; en estos cuatro exports los IDs son únicos y ninguno está vacío. Cada cota expresa únicamente el mínimo compatible con ese export, **no un estimador del total nativo ni un total visible real**. No se suman como total de resultados o estudios. Las 244 ocurrencias exportadas y los 182 grupos candidatos conservan su significado y valores exactos del registro anterior; los grupos siguen sujetos a revisión de identidad.

Para Scopus, «06/11/2026 (referencia)» procede del nombre del archivo: la zona horaria y la fecha/hora real de ejecución no están establecidas. Para WoS, la misma etiqueta corresponde a recepción y confirmación contextual del usuario el 06/11/2026 (America/Bogota), no a la ejecución de la consulta. No se reconstruyen horas. Las familias TCN y GRU/LSTM y el filtro 2019–2026 están confirmados por el usuario, sin verificación independiente del filtro en el export; se conservan también las cinco ocurrencias nominales de 2027 del registro original.

`provisional_log.csv` contiene las cinco filas y sus cadenas propuestas completas. Las propuestas se derivan de `search_recovery_20261106/query_plan.csv`: TCN conserva la cadena del plan; GRU/LSTM une ambos bloques con OR dentro del mismo campo; IEEE reúne TCN/GRU/LSTM en All Metadata. WoS solicita el filtro de años aparte; Scopus conserva PUBYEAR > 2018 AND PUBYEAR < 2027; IEEE solicita el filtro aparte. **Ninguna propuesta reconstruye la cadena histórica ni se atribuye a una ejecución observada.** La interfaz y el campo describen la propuesta, no una interfaz histórica recuperada.

En JSON, `native_total_verified`, `native_query_literal_verified`, `native_search_date_verified` y `native_export_range_verified` permanecen `null`, y `native_query_literal_recovered=false`; en CSV los valores `null` son celdas vacías. IEEE está `planned_not_executed`, no ejecutada/no acreditada: sus conteos son `null` (desconocidos), nunca cero. Las cotas, las referencias de fecha y las propuestas llevan `exclude_from_verified_native_search_totals=true` y `exclude_from_prisma_totals=true`; no se incorporan como evidencia verificada de identificación/PRISMA.

Hay **0 nuevas ejecuciones nativas**, 0 totales verificados añadidos, 0 fechas nativas verificadas añadidas y 0 cadenas históricas recuperadas. Se conservan N7=7, las frecuencias históricas y todos los experimentos. La actualización futura deberá aportar captura nativa de cadena literal, fecha/hora y zona, colección/interfaz, filtros, total y rango exportado antes de cambiar el estado de verificación.

`provenance.json` fija hashes y tamaños de las fuentes de metadatos anteriores; este compilador no lee ni publica exports originales, PDF, textos completos, resúmenes nativos o contactos. `manifest.json` usa una lista pública cerrada (`e1_public_metadata_allowlist_v1`): sólo README, CSV provisional, resumen y procedencia, con SHA256 y bytes, más el manifiesto. El espejo público conserva exactamente los mismos bytes.

Reproducción: `python scripts/build_e1_provisional_native_log.py`. Verificación sin escritura: `python scripts/build_e1_provisional_native_log.py --verify`. No construye HTML, no crea ZIP, no hace commits y no altera las fuentes originales.
'''


def artifacts(base: Path = BASE) -> tuple[dict[str, bytes], dict]:
    raw = load_inputs(base)
    rows, native_summary = derive_rows(raw)
    summary = summarize(rows, native_summary)
    provenance = {"schema_version": 1, "compiler": "scripts/build_e1_provisional_native_log.py",
                  "source_policy": "Frozen saved metadata only; source files are never modified",
                  "original_exports_read_this_stage": False, "original_exports_republished": False,
                  "new_native_search_executions": 0,
                  "input_files": {name: metadata(raw[name]) for name in sorted(raw)},
                  "lower_bound_method": "Distinct nonempty source_id values grouped by source_key, cross-checked with source_manifest and source_reconciliation; never equated with native displayed totals",
                  "query_method": "Proposed plan syntax; merge family terms within their common field; historical literal remains unknown",
                  "date_method": "Scopus filename date (timezone unknown); WoS receipt/user-confirmation date (not execution)",
                  "public_metadata_only": True}
    files = {"README.md": readme(rows).encode("utf-8"), "provisional_log.csv": csv_bytes(rows),
             "summary.json": json_bytes(summary), "provenance.json": json_bytes(provenance)}
    for name, data in files.items():
        validate_public_bytes(name, data)
    manifest = {"schema_version": 1, "publication_schema": PUBLICATION_SCHEMA,
                "approved_public_files": sorted(PUBLIC_FILES),
                "scope": "Provisional metadata only; bounds, dates and queries excluded from verified native-search/PRISMA totals",
                "files": {name: metadata(files[name]) for name in sorted(files)}}
    files["manifest.json"] = json_bytes(manifest)
    validate_public_bytes("manifest.json", files["manifest.json"])
    return files, summary


def build(stage: Path = STAGE, mirror: Path | None = MIRROR, base: Path = BASE) -> dict:
    files, summary = artifacts(base)
    # Validate both complete destinations before the first write; reject neighbours.
    for destination in (stage, mirror):
        if destination is not None and inventory(destination) - set(files):
            raise ValueError("Unapproved neighbouring output file")
    for destination in (stage, mirror):
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                (destination / name).write_bytes(data)
    verify(stage, mirror, base)
    return summary


def verify(stage: Path = STAGE, mirror: Path | None = MIRROR, base: Path = BASE) -> None:
    files, _ = artifacts(base)
    for destination in (stage, mirror):
        if destination is not None:
            if inventory(destination) != set(files):
                raise ValueError("Public inventory differs from the closed allowlist")
            for name, data in files.items():
                if checked_file(destination, name) != data:
                    raise ValueError("Compiled metadata or mirror hash differs: " + name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        print("PASS: five provisional rows; source-ID bounds, unknown native evidence, frozen sources, closed manifest and identical mirror")
    else:
        summary = build()
        print(json.dumps({key: summary[key] for key in ("log_status", "row_count", "status_counts", "new_native_search_executions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
