#!/usr/bin/env python3
"""Import native Scopus/WoS exports into a small, public E1 audit ledger.

Raw exports and abstract text are input only.  The output is a provisional
discovery/screening tier, never a full-text inclusion decision or a new N.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "references/literature_e1/native_search_20261106"
DEFAULT_MIRROR = ROOT / "book/_static/literature_e1/native_search_20261106"
HISTORICAL_EVIDENCE = ROOT / "references/literature_e1/crypto_extension_20261105/evidence_matrix.json"
ALGORITHM_VERSION = "1.0"
RECEIVED_DATE = "2026-11-06"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_doi(value: str) -> str:
    value = clean(value).lower()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
    return value.rstrip(" .;")


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    return "".join(c for c in value if c.isalnum())


@dataclass(frozen=True)
class Record:
    occurrence_id: str
    source_key: str
    database: str
    source_id: str
    title: str
    doi: str = ""
    year: str = ""
    venue: str = ""
    document_type: str = ""
    abstract: str = ""
    keywords: str = ""


def parse_scopus(text: str, source_key: str) -> list[Record]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames or "Title" not in reader.fieldnames:
        raise ValueError("Scopus CSV requires a Title column")
    records = []
    for number, row in enumerate(reader, 1):
        if not any(clean(v) for v in row.values() if isinstance(v, str)):
            continue
        records.append(Record(
            f"{source_key}:{number:04d}", source_key, "Scopus", clean(row.get("EID")),
            clean(row.get("Title")), normalize_doi(row.get("DOI") or ""),
            clean(row.get("Year")), clean(row.get("Source title")),
            clean(row.get("Document Type")), clean(row.get("Abstract")),
            clean((row.get("Author Keywords") or "") + " " + (row.get("Index Keywords") or "")),
        ))
    return records


def parse_wos(text: str, source_key: str) -> list[Record]:
    """Read field-tagged WoS text, including wrapped and repeated fields.

    ER closes a record. EF/EOF also flushes an unfinished last record so a
    missing terminator does not silently lose an otherwise usable record.
    Bibliographic field contents are data, never executable instructions.
    """
    records: list[Record] = []
    fields: dict[str, list[str]] = defaultdict(list)
    current = ""

    def flush() -> None:
        nonlocal fields, current
        if not fields:
            return
        if not any(k in fields for k in ("PT", "TI", "UT", "DI")):
            fields, current = defaultdict(list), ""
            return
        get = lambda tag: clean(" ".join(fields.get(tag, [])))
        records.append(Record(
            f"{source_key}:{len(records) + 1:04d}", source_key, "Web of Science",
            get("UT"), get("TI"), normalize_doi(get("DI")), get("PY"), get("SO"),
            get("DT") or get("PT"), get("AB"), get("DE") + " " + get("ID"),
        ))
        fields, current = defaultdict(list), ""

    for raw in text.lstrip("\ufeff").splitlines():
        if raw[:2] in {"ER", "EF"} and not raw[2:].strip():
            flush()
            continue
        if raw.startswith("   ") and current:
            fields[current][-1] += " " + raw.strip()
            continue
        if re.match(r"^[A-Z0-9]{2} ", raw):
            tag, content = raw[:2], raw[3:]
            if tag in {"FN", "VR"}:
                continue
            if tag == "PT" and fields:
                flush()
            current = tag
            fields[tag].append(content)
    flush()
    if not records:
        raise ValueError("No WoS field-tagged records found")
    return records


def deduplicate(records: list[Record]) -> tuple[list[list[Record]], list[dict]]:
    """Return candidate groups plus an explicit linking/conflict ledger.

    DOI and same database ID links precede title links.  A normalized title
    alone is always provisional.  Distinct nonempty DOI sets are not merged.
    Every occurrence remains independently addressable in occurrences.csv.
    """
    parent = list(range(len(records)))
    decisions: list[dict] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def dois(i: int) -> set[str]:
        return {r.doi for j, r in enumerate(records) if find(j) == find(i) and r.doi}

    def link(left: int, right: int, method: str, provisional: bool) -> None:
        a, b = find(left), find(right)
        if a == b:
            return
        conflict = bool(dois(a) and dois(b) and dois(a).isdisjoint(dois(b)))
        decisions.append({
            "left_occurrence": records[left].occurrence_id,
            "right_occurrence": records[right].occurrence_id,
            "method": method,
            "decision": "retain_separate_doi_conflict" if conflict else "candidate_group_link",
            "provisional": str(provisional or conflict).lower(),
            "reason": "conflicting_nonempty_doi" if conflict else (
                "normalized_title_is_not_final_identity" if provisional else method),
        })
        if not conflict:
            parent[b] = a

    for method, key in (
        ("exact_doi", lambda r: r.doi),
        ("exact_database_id", lambda r: f"{r.database}:{r.source_id}" if r.source_id else ""),
        ("normalized_title_provisional", lambda r: normalize_title(r.title)),
    ):
        seen: dict[str, list[int]] = defaultdict(list)
        for i, record in enumerate(records):
            value = key(record)
            if value:
                for previous in seen[value]:
                    link(previous, i, method, method == "normalized_title_provisional")
                seen[value].append(i)
    groups: dict[int, list[Record]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record)
    return sorted(groups.values(), key=lambda g: (normalize_title(g[0].title), g[0].doi, g[0].occurrence_id)), decisions


CRYPTO = re.compile(r"bitcoin|ethereum|cryptocurrenc|cryptoasset|crypto.asset|crypto.market|digital currenc|\bbtc\b", re.I)
VOLATILITY = re.compile(r"volatilit|volatilidad|realized variance|realised variance", re.I)
PRICE_RETURN_RISK = re.compile(r"price|return|risk|trading|portfolio", re.I)
FORECAST = re.compile(r"forecast|predict|estimat", re.I)
MODELS = {
    "GRU": r"\bgru\b|gated recurrent unit",
    "LSTM": r"\blstm\b|long.short.term memory",
    "TCN": r"\btcn\b|temporal convolution|dilated causal convolution",
    "Transformer": r"transformer",
    "MLP": r"\bmlp\b|multi.layer perceptron",
    "CNN": r"\bcnn\b|convolutional neural",
    "RNN": r"\brnn\b|recurrent neural",
    "GARCH": r"garch",
}


def screen_group(group: list[Record]) -> dict:
    title = " ".join(r.title for r in group)
    body = " ".join(r.abstract + " " + r.keywords for r in group)
    full = title + " " + body
    crypto_title, crypto_any = bool(CRYPTO.search(title)), bool(CRYPTO.search(full))
    vol_title, vol_any = bool(VOLATILITY.search(title)), bool(VOLATILITY.search(full))
    price_title = bool(PRICE_RETURN_RISK.search(title))
    mixed = bool(re.search(r"price and volatility|price.*volatility|volatility.*price|volatility.robust|volatility.regime", title, re.I))
    if crypto_title and vol_title and not mixed:
        category, priority, reason = "crypto_volatility_title", "1", "Titulo vincula cripto y volatilidad; comprobar variable objetivo y comparadores en texto completo."
    elif crypto_any and vol_title:
        category, priority, reason = "volatility_title_target_ambiguous", "1", "Titulo menciona volatilidad y registro menciona cripto; verificar objetivo mixto, robustez o activos."
    elif re.search(r"review|survey|bibliometric", title, re.I):
        category, priority, reason = "contextual_review", "3", "Posible revision contextual; no es evidencia experimental independiente por mencion de modelos."
    elif crypto_any and price_title:
        category, priority, reason = "price_return_risk_title", "2", "Titulo apunta a precio, retorno, riesgo o trading; comprobar si existe objetivo separado de volatilidad."
    elif crypto_any and vol_any and FORECAST.search(full):
        category, priority, reason = "abstract_volatility_check", "2", "Volatilidad y pronostico aparecen en metadatos; no se ha establecido que volatilidad sea el objetivo."
    elif crypto_any:
        category, priority, reason = "crypto_scope_unclear", "3", "Registro menciona cripto; objetivo de volatilidad no establecido por titulo/resumen."
    else:
        category, priority, reason = "noncrypto_or_scope_unclear", "3", "Cripto no identificado por las reglas de titulo/resumen; verificar posible transferencia o alcance."
    return {
        "scope_category": category, "review_priority": priority,
        "screening_status": "provisional_title_abstract_only", "reason": reason,
        "crypto_signal_title": str(crypto_title).lower(),
        "crypto_signal_any_metadata": str(crypto_any).lower(),
        "volatility_signal_title": str(vol_title).lower(),
        "volatility_signal_any_metadata": str(vol_any).lower(),
        "model_mentions_unverified": ";".join(k for k, pattern in MODELS.items() if re.search(pattern, full, re.I)),
        "full_text_review_by_importer": "not_performed",
        "eligibility_decision": "undetermined",
    }


def csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def read_historical(path: Path | None) -> tuple[dict[str, str], dict]:
    if path is None:
        return {}, {"reference": None, "sha256": None, "crypto_full_text_denominator_preserved": 7}
    raw = path.read_bytes()
    data = json.loads(raw)
    mapping = {normalize_doi(s.get("DOI") or ""): s["study_id"] for s in data["studies"] if s.get("DOI")}
    return mapping, {
        "reference": "references/literature_e1/crypto_extension_20261105/evidence_matrix.json",
        "sha256": sha256(raw),
        "crypto_full_text_denominator_preserved": data["scope_and_denominators"]["crypto_admissible_fulltexts"],
        "frequency_update_performed": False,
    }


def import_sources(source_specs: list[tuple[str, Path]], output_dir: Path,
                   mirror_dir: Path | None = None, historical_path: Path | None = HISTORICAL_EVIDENCE) -> dict:
    records: list[Record] = []
    sources = []
    db_counts: Counter = Counter()
    for kind, path in source_specs:
        db_counts[kind] += 1
        key = ("S" if kind == "scopus" else "W") + f"{db_counts[kind]:02d}"
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        parsed = parse_scopus(text, key) if kind == "scopus" else parse_wos(text, key)
        records.extend(parsed)
        sources.append({
            "source_key": key, "database": parsed[0].database if parsed else kind,
            "filename": path.name, "sha256": sha256(raw), "bytes": len(raw),
            "exported_occurrences": len(parsed), "doi_present": sum(bool(r.doi) for r in parsed),
            "abstract_present": sum(bool(r.abstract) for r in parsed),
            "executed_query": None, "executed_query_status": "not_embedded_in_received_export",
            "query_family_association": None, "applied_filters": None,
            "displayed_result_total": None, "exported_result_range": None,
            "completeness": "not_established_by_exported_row_count",
        })
    groups, decisions = deduplicate(records)
    historical_map, historical = read_historical(historical_path)
    occurrence_to_group = {}
    grouped_rows, screening_rows, occurrence_rows = [], [], []
    provisional_group_ids, identity_ambiguous_group_ids = [], []
    for number, group in enumerate(groups, 1):
        group_id = f"E1N{number:03d}"
        for r in group:
            occurrence_to_group[r.occurrence_id] = group_id
        ids = {r.occurrence_id for r in group}
        title_links = [d for d in decisions if d["left_occurrence"] in ids and d["right_occurrence"] in ids and d["method"] == "normalized_title_provisional" and d["decision"] == "candidate_group_link"]
        years = sorted({r.year for r in group if r.year})
        db_ids: dict[str, set] = defaultdict(set)
        for r in group:
            if r.source_id:
                db_ids[r.database].add(r.source_id)
        identity_ambiguous = bool(title_links and any(len(v) > 1 for v in db_ids.values()))
        if title_links:
            provisional_group_ids.append(group_id)
        if identity_ambiguous:
            identity_ambiguous_group_ids.append(group_id)
        dois = sorted({r.doi for r in group if r.doi})
        row = {
            "group_id": group_id, "title": group[0].title,
            "doi": ";".join(dois), "years_reported": ";".join(years),
            "venues_reported": " | ".join(sorted({r.venue for r in group if r.venue})),
            "document_types_reported": ";".join(sorted({r.document_type for r in group if r.document_type})),
            "databases": ";".join(sorted({r.database for r in group})),
            "source_keys": ";".join(sorted({r.source_key for r in group})),
            "source_ids": ";".join(sorted({r.source_id for r in group if r.source_id})),
            "occurrence_count": len(group),
            "identity_status": "provisional_title_link_distinct_ids" if identity_ambiguous else (
                "provisional_title_link" if title_links else "doi_or_source_id_group" if len(group) > 1 else "single_occurrence"),
            "abstract_available_occurrences": sum(bool(r.abstract) for r in group),
            "abstract_sha256": ";".join(sorted({sha256(r.abstract.encode("utf-8")) for r in group if r.abstract})),
            "previous_e1_study_ids_by_doi": ";".join(sorted({historical_map[d] for d in dois if d in historical_map})),
            "nominal_year_after_received_year": str(any(y.isdigit() and int(y) > int(RECEIVED_DATE[:4]) for y in years)).lower(),
        }
        grouped_rows.append(row)
        screening_rows.append({"group_id": group_id, "title": group[0].title, "doi": row["doi"], **screen_group(group)})
    for r in records:
        occurrence_rows.append({
            "occurrence_id": r.occurrence_id, "group_id": occurrence_to_group[r.occurrence_id],
            "source_key": r.source_key, "database": r.database, "source_id": r.source_id,
            "title": r.title, "doi": r.doi, "year_reported": r.year,
            "abstract_present": str(bool(r.abstract)).lower(),
            "abstract_sha256": sha256(r.abstract.encode("utf-8")) if r.abstract else "",
        })
    for d in decisions:
        d["left_group_id"] = occurrence_to_group[d["left_occurrence"]]
        d["right_group_id"] = occurrence_to_group[d["right_occurrence"]]
    category_counts = Counter(r["scope_category"] for r in screening_rows)
    priority_counts = Counter(r["review_priority"] for r in screening_rows)
    summary = {
        "schema_version": "1.0", "algorithm_version": ALGORITHM_VERSION, "received_date": RECEIVED_DATE,
        "source_count": len(sources), "exported_occurrences": len(records),
        "candidate_groups_provisional": len(groups),
        "occurrences_linked_within_candidate_groups": len(records) - len(groups),
        "groups_with_title_only_links": len(provisional_group_ids),
        "title_only_group_ids": provisional_group_ids,
        "groups_with_distinct_ids_requiring_identity_review": len(identity_ambiguous_group_ids),
        "identity_review_group_ids": identity_ambiguous_group_ids,
        "groups_with_both_databases": sum(len({r.database for r in g}) > 1 for g in groups),
        "screening_scope_categories": dict(sorted(category_counts.items())),
        "review_priority_counts": dict(sorted(priority_counts.items())),
        "all_groups_require_human_review": True,
        "full_texts_reviewed_by_importer": 0, "new_final_inclusions": None,
        "final_exclusions": None, "historical_e1": historical,
        "previous_e1_doi_matches": [{"group_id": r["group_id"], "study_ids": r["previous_e1_study_ids_by_doi"]} for r in grouped_rows if r["previous_e1_study_ids_by_doi"]],
        "interpretation": "The candidate-group count is not a validated study denominator; raw occurrences and uncertain identities are retained. Model mentions are not independent comparator evidence.",
    }
    source_manifest = {
        "schema_version": "1.0", "received_date": RECEIVED_DATE, "sources": sources,
        "queries_and_filters": "No exact query, time filter, search date, result total or export range is established from these received files alone. Suggested searches are not executed-query evidence.",
        "source_exports_republished": False,
        "public_metadata_policy": "Minimal bibliographic identity, controlled screening labels and hashes only; no abstract text, author emails, affiliations, or raw cited-reference fields.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        "source_manifest.json": json_bytes(source_manifest),
        "summary.json": json_bytes(summary),
        "records.csv": csv_bytes(grouped_rows, list(grouped_rows[0]) if grouped_rows else ["group_id"]),
        "occurrences.csv": csv_bytes(occurrence_rows, list(occurrence_rows[0]) if occurrence_rows else ["occurrence_id"]),
        "deduplication.csv": csv_bytes(decisions, ["left_occurrence", "right_occurrence", "method", "decision", "provisional", "reason", "left_group_id", "right_group_id"]),
        "screening.csv": csv_bytes(screening_rows, list(screening_rows[0]) if screening_rows else ["group_id"]),
        "review_queue.csv": csv_bytes(sorted(screening_rows, key=lambda r: (r["review_priority"], r["group_id"])), list(screening_rows[0]) if screening_rows else ["group_id"]),
        "README.md": make_readme(summary).encode("utf-8"),
        "flow.svg": make_flow(summary).encode("utf-8"),
    }
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    manifest = {"schema_version": "1.0", "algorithm_version": ALGORITHM_VERSION,
                "generator": {"script": "scripts/import_e1_literature.py", "sha256": sha256(Path(__file__).read_bytes())},
                "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in sorted(files.items())},
                "note": "Manifest hashes cover public outputs; source hashes are in source_manifest.json. Raw exports are required only to regenerate, not to verify these public artifacts."}
    (output_dir / "manifest.json").write_bytes(json_bytes(manifest))
    verify_public(output_dir)
    if mirror_dir:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        for name in [*files, "manifest.json"]:
            shutil.copyfile(output_dir / name, mirror_dir / name)
        verify_public(mirror_dir)
    return summary


def make_readme(summary: dict) -> str:
    return f"""# Búsquedas nativas recibidas el 6 de noviembre de 2026

Esta capa documenta cuatro exportaciones recibidas: {summary['exported_occurrences']} ocurrencias y {summary['candidate_groups_provisional']} grupos candidatos provisionales. No sustituye el corpus histórico de E1 ni acredita exhaustividad de las búsquedas. El denominador histórico de textos completos cripto sigue siendo N=7; esta importación no modifica sus frecuencias.

## Archivos

- `source_manifest.json`: nombre, tamaño, SHA-256 y cobertura observable de cada exportación. Consulta exacta, filtros, total mostrado y rango exportado quedan sin establecer.
- `occurrences.csv`: las ocurrencias originales por archivo, sin borrar identidades de origen.
- `records.csv`: grupos candidatos por DOI, identificador de base o título normalizado; conserva años e identificadores discrepantes.
- `deduplication.csv`: cada enlace o conflicto entre ocurrencias. Un enlace por título es siempre provisional.
- `screening.csv` y `review_queue.csv`: primera pasada automática por título, resumen y palabras clave, con reglas en el script, razones y prioridades. Todas las filas requieren revisión humana.
- `summary.json`: recuentos, incertidumbres y enlaces por DOI a la evidencia anterior.
- `flow.svg`: flujo de descubrimiento provisional; no es un PRISMA de selección final.
- `manifest.json`: integridad de los archivos públicos derivados.

## Identidades y decisiones pendientes

Los {summary['occurrences_linked_within_candidate_groups']} enlaces internos no equivalen a duplicados definitivamente descartados. Hay {summary['groups_with_title_only_links']} grupos con enlaces por título; {summary['groups_with_distinct_ids_requiring_identity_review']} contiene identificadores diferentes de la misma base. En particular, el título *Enhancing Cryptocurrency Price Prediction Using Hybrid Regression and RNN Models* aparece con dos EID Scopus y años 2024/2025: ambos registros y el registro WoS se conservan como ocurrencias distintas dentro de un grupo candidato pendiente de validación. El total definitivo de documentos únicos sigue sin resolverse.

Las prioridades 1, 2 y 3 ordenan la lectura; ninguna es inclusión o exclusión final. La prioridad 1 señala volatilidad en el título y contexto cripto, pero aún puede corresponder a objetivos mixtos, índices, clasificación de regímenes o robustez de precios. La prioridad 2 revisa precio/retorno/riesgo/trading y señales de resumen. La prioridad 3 conserva revisiones, contexto y alcance incierto. Las menciones de GRU, LSTM, TCN u otros modelos no acreditan comparadores independientes. No se revisaron textos completos mediante este importador. Los años nominales posteriores a 2026 requieren comprobar publicación anticipada o metadatos; no se excluyen automáticamente.

## Reproducir y verificar

Requiere Python 3.10 o posterior, sin paquetes externos. Desde la raíz del repositorio:

```sh
python scripts/import_e1_literature.py \\
  --scopus '/ruta/export_4a31d458-6ce4-4435-9507-df9ee7cc2881_2026-11-06T220747.614885599.csv' \\
  --scopus '/ruta/export_694623e0-7e5c-4597-aca5-c8fc7cb8b431_2026-11-06T220948.394330608.csv' \\
  --wos '/ruta/savedrecs (4).txt' \\
  --wos '/ruta/savedrecs (3).txt'
python scripts/import_e1_literature.py --verify-public references/literature_e1/native_search_20261106
python -m unittest discover -s tests -p 'test_e1_literature_import.py'
```

El orden de los cuatro argumentos fija los identificadores S01/S02/W01/W02; el orden de salida y las reglas son deterministas. El script lee la matriz histórica `references/literature_e1/crypto_extension_20261105/evidence_matrix.json` únicamente para enlazar DOI y registrar su huella. `--output-dir` y `--mirror-dir` permiten otros destinos; `--no-mirror` evita la copia estática. `--verify-public` verifica hashes, recuentos, cobertura de ocurrencias y campos permitidos sin necesitar las exportaciones originales.

Los resúmenes se usan en memoria y solo se publica su disponibilidad y huella del texto normalizado por el parser. No se redistribuyen exportaciones completas, resúmenes, correos, afiliaciones ni referencias citadas brutas. Las fuentes originales son necesarias para repetir el cribado; sus SHA-256 permiten comprobar que son los mismos archivos.
"""


def make_flow(summary: dict) -> str:
    labels = [
        ("Exportaciones recibidas", f"{summary['source_count']} archivos · {summary['exported_occurrences']} ocurrencias"),
        ("Agrupación candidata", f"{summary['candidate_groups_provisional']} grupos provisionales · identidades pendientes"),
        ("Primera pasada de metadatos", "Título/resumen · todas las decisiones por revisar"),
        ("Textos completos e inclusión", "No determinados por este importador · N histórico = 7"),
    ]
    boxes = []
    for i, (title, subtitle) in enumerate(labels):
        y = 24 + i * 108
        boxes.append(f'<rect x="30" y="{y}" width="700" height="76" rx="10" fill="#edf3fa" stroke="#2c5378"/><text x="380" y="{y+29}" text-anchor="middle" font-size="19" font-weight="bold">{escape(title)}</text><text x="380" y="{y+54}" text-anchor="middle" font-size="16">{escape(subtitle)}</text>')
        if i < 3:
            boxes.append(f'<path d="M380 {y+77} V{y+105}" stroke="#2c5378" stroke-width="2" marker-end="url(#arrow)"/>')
    return '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="460" viewBox="0 0 760 460" role="img" aria-labelledby="title desc"><title id="title">Flujo provisional de las exportaciones recibidas</title><desc id="desc">No es un flujo de selección final ni actualiza el denominador histórico de E1.</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#2c5378"/></marker></defs><g font-family="Arial, sans-serif" fill="#17324b">' + "".join(boxes) + '</g></svg>\n'


PUBLIC_COLUMNS = {
    "records.csv": {"group_id", "title", "doi", "years_reported", "venues_reported", "document_types_reported", "databases", "source_keys", "source_ids", "occurrence_count", "identity_status", "abstract_available_occurrences", "abstract_sha256", "previous_e1_study_ids_by_doi", "nominal_year_after_received_year"},
    "occurrences.csv": {"occurrence_id", "group_id", "source_key", "database", "source_id", "title", "doi", "year_reported", "abstract_present", "abstract_sha256"},
    "deduplication.csv": {"left_occurrence", "right_occurrence", "method", "decision", "provisional", "reason", "left_group_id", "right_group_id"},
    "screening.csv": {"group_id", "title", "doi", "scope_category", "review_priority", "screening_status", "reason", "crypto_signal_title", "crypto_signal_any_metadata", "volatility_signal_title", "volatility_signal_any_metadata", "model_mentions_unverified", "full_text_review_by_importer", "eligibility_decision"},
}
PUBLIC_COLUMNS["review_queue.csv"] = PUBLIC_COLUMNS["screening.csv"]


def verify_public(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    allowed_files = set(PUBLIC_COLUMNS) | {"source_manifest.json", "summary.json", "README.md", "flow.svg"}
    if set(manifest["files"]) != allowed_files:
        raise ValueError("Unexpected public artifact names in manifest")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if expected != actual:
        raise ValueError(f"Unexpected/missing public files: {expected ^ actual}")
    for name, identity in manifest["files"].items():
        raw = (directory / name).read_bytes()
        if sha256(raw) != identity["sha256"] or len(raw) != identity["bytes"]:
            raise ValueError(f"Public artifact hash/size mismatch: {name}")
    tables = {}
    for name, allowed in PUBLIC_COLUMNS.items():
        with (directory / name).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if set(reader.fieldnames or []) != allowed:
                raise ValueError(f"Unexpected public CSV fields: {name}")
            tables[name] = list(reader)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    sources = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))["sources"]
    occurrences, groups = tables["occurrences.csv"], tables["records.csv"]
    if len(occurrences) != summary["exported_occurrences"] or len(groups) != summary["candidate_groups_provisional"]:
        raise ValueError("Summary/table counts differ")
    ids = {r["occurrence_id"] for r in occurrences}
    group_ids = {r["group_id"] for r in groups}
    if len(ids) != len(occurrences) or len(group_ids) != len(groups):
        raise ValueError("Duplicate occurrence or group IDs")
    counts = Counter(r["group_id"] for r in occurrences)
    if set(counts) != group_ids or any(counts[r["group_id"]] != int(r["occurrence_count"]) for r in groups):
        raise ValueError("Candidate groups do not cover occurrences exactly")
    source_counts = Counter(r["source_key"] for r in occurrences)
    if source_counts != Counter({s["source_key"]: s["exported_occurrences"] for s in sources}):
        raise ValueError("Source manifest/occurrence counts differ")
    for name in ("screening.csv", "review_queue.csv"):
        rows = tables[name]
        if len(rows) != len(groups) or {r["group_id"] for r in rows} != group_ids:
            raise ValueError("Screening/review queue coverage differs")
        if any(r["eligibility_decision"] != "undetermined" or r["full_text_review_by_importer"] != "not_performed" for r in rows):
            raise ValueError("Automatic import must not claim final eligibility or full-text reading")
    if summary["new_final_inclusions"] is not None or summary["final_exclusions"] is not None:
        raise ValueError("Final screening decisions are outside this import tier")
    return {"status": "verified", "public_files": len(expected), "occurrences": len(occurrences), "candidate_groups": len(groups)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scopus", type=Path, action="append", default=[])
    parser.add_argument("--wos", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mirror-dir", type=Path, default=DEFAULT_MIRROR)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--verify-public", type=Path)
    args = parser.parse_args(argv)
    if args.verify_public:
        print(json.dumps(verify_public(args.verify_public), sort_keys=True))
        return 0
    if len(args.scopus) != 2 or len(args.wos) != 2:
        parser.error("provide exactly two --scopus and two --wos received exports, or --verify-public DIR")
    summary = import_sources([*(('scopus', p) for p in args.scopus), *(('wos', p) for p in args.wos)], args.output_dir, None if args.no_mirror else args.mirror_dir)
    print(json.dumps({k: summary[k] for k in ("exported_occurrences", "candidate_groups_provisional", "groups_with_distinct_ids_requiring_identity_review", "screening_scope_categories")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
