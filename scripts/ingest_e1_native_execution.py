#!/usr/bin/env python3
"""Ingest supplied evidence of fresh E1 native searches; never execute a search.

Only allowlisted bibliographic metadata and evidence hashes leave the private
inputs. Field-level review assertions are supplied by the person who inspected
the evidence; this program checks their integrity, links and consistency. It
does not infer screenshot contents, a search total, or a historical query.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import re

from import_e1_literature import (Record, clean, csv_bytes, deduplicate, json_bytes,
                                  normalize_doi, parse_scopus, parse_wos, sha256)

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "references/literature_e1/native_execution_20261107"
HISTORY = ROOT / "references/literature_e1/native_search_20261106"
NOT_BEFORE = datetime(2026, 11, 7, tzinfo=timezone.utc)
DATABASES = {"Web of Science", "Scopus", "IEEE Xplore"}
FIELDS = {"executed_at", "query_literal", "interface", "field_scope", "collection",
          "editions", "filters", "displayed_total"}
OPTIONAL_FIELDS = {"executed_on", "execution_date_timezone", "exported_all_results", "query_literal_received", "displayed_filter_controls"}
REVIEW_FIELDS = FIELDS | OPTIONAL_FIELDS | {"export_range", "access_error"}
VISUAL_KINDS = {"screenshot", "native_history", "screen_recording", "conversation_image"}
FORMATS = {"wos_tagged": "Web of Science", "scopus_csv": "Scopus", "ieee_csv": "IEEE Xplore", "ieee_plain_text": "IEEE Xplore"}
HISTORY_SHA256 = {
    "occurrences.csv": "005199f0d4088563d188db6b0d265021700245d5c705ec64c705348c51e2da48",
    "source_manifest.json": "ae80451bcea7686d6fa3875c3fa9f8a9a0a4063336a15a0828561a359cf1cb72",
    "summary.json": "6474360911be95ecf956f4b9e984adb7ebde54ecda20cb14055dc6e0f7accbf9",
}
HISTORICAL_FULLTEXTS = ROOT / "references/literature_e1/crypto_extension_20261105/evidence_matrix.json"
HISTORICAL_FULLTEXTS_SHA256 = "4b5eb36f0db801f15df64a5f04cc053a107165a86ff89f755afe9e548ec72f82"
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
HASH = re.compile(r"^[0-9a-f]{64}$")
EMAIL = re.compile(rb"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
OCCURRENCE_COLUMNS = ["occurrence_id", "group_id", "historical_group_id", "source_key",
                      "database", "source_id", "title", "doi", "year_reported", "origin"]
RECORD_COLUMNS = ["group_id", "occurrence_count", "historical_group_ids", "occurrence_ids",
                  "title", "dois", "databases", "identity_status"]
DECISION_COLUMNS = ["left_occurrence", "right_occurrence", "method", "decision", "provisional", "reason"]


def keys(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError("Invalid or unknown object keys; expected " + ", ".join(sorted(required)))


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid identifier")
    return value


def timestamp(value, now):
    if not isinstance(value, str):
        raise ValueError("Timestamp must include a timezone offset")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone offset")
    parsed = parsed.astimezone(timezone.utc)
    if parsed < NOT_BEFORE or parsed > now:
        raise ValueError("Timestamp predates the new campaign or is in the future")
    return parsed


def private_bytes(path_value, expected_hash):
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("Private inputs require absolute paths outside the checkout")
    path = Path(path_value).resolve()
    if path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Private inputs must be regular files outside the checkout")
    if not isinstance(expected_hash, str) or not HASH.fullmatch(expected_hash):
        raise ValueError("Expected SHA-256 is required")
    raw = path.read_bytes()
    if sha256(raw) != expected_hash:
        raise ValueError("Private input SHA-256 mismatch")
    return raw


def validate_value(field, value, now):
    if field == "executed_at":
        timestamp(value, now)
    elif field == "executed_on":
        try:
            day = date.fromisoformat(value)
        except (ValueError, TypeError) as error:
            raise ValueError("Execution day must be a real ISO calendar date") from error
        # A local day may precede the UTC campaign day. Preserve the supplied day;
        # never manufacture an execution time or offset from receipt metadata.
        if day < NOT_BEFORE.date() - timedelta(days=1) or day > now.date():
            raise ValueError("Execution day is outside the fresh search campaign")
    elif field == "exported_all_results":
        if type(value) is not bool:
            raise ValueError("Export-all confirmation must be a boolean")
    elif field == "displayed_total":
        if type(value) is not int or value < 0:
            raise ValueError("Displayed total must be a nonnegative integer")
    elif field in {"editions", "displayed_filter_controls"}:
        if (not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value)
                or len(value) != len(set(value))):
            raise ValueError("Editions/controls must be an explicit list (empty when inapplicable)")
    elif field == "filters":
        keys(value, {"publication_year_start", "publication_year_end", "document_types", "other"})
        start, end = value["publication_year_start"], value["publication_year_end"]
        if type(start) is not int or type(end) is not int or not 1900 <= start <= end <= 2100:
            raise ValueError("Invalid publication-year filter")
        for key in ("document_types", "other"):
            if (not isinstance(value[key], list) or any(not isinstance(v, str) or not v.strip() for v in value[key])
                    or len(value[key]) != len(set(value[key]))):
                raise ValueError("Filters require explicit lists")
        if not value["document_types"]:
            raise ValueError("Document types must explicitly say all or name the applied types")
    elif not isinstance(value, str) or not value.strip():
        raise ValueError("Verified text values must be nonempty")


def parse_ieee(text, source_key):
    """Read native IEEE CSV; Document Identifier is a type, not an article ID."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames or "Document Title" not in reader.fieldnames:
        raise ValueError("IEEE CSV requires a Document Title column")
    records = []
    for row in reader:
        if not any(clean(v) for v in row.values() if isinstance(v, str)):
            continue
        native_id = clean(row.get("Article Number") or row.get("Article ID"))
        if not native_id:
            links = " ".join(row.get(k) or "" for k in ("PDF Link", "Document Link", "Document URL", "URL"))
            match = re.search(r"(?:[?&]arnumber=|/document/)(\d+)", links)
            native_id = match.group(1) if match else ""
        records.append(Record(
            f"{source_key}:{len(records) + 1:04d}", source_key, "IEEE Xplore", native_id,
            clean(row.get("Document Title")), normalize_doi(row.get("DOI") or ""),
            clean(row.get("Publication Year")), clean(row.get("Publication Title")),
            clean(row.get("Document Identifier")),
        ))
    return records


def parse_ieee_plain_text(text, source_key):
    """Parse native IEEE citation text; discard author and keyword payloads."""
    records = []
    for block in re.split(r"\n\s*\n", text.lstrip("\ufeff").strip()):
        citation = " ".join(line.strip() for line in block.splitlines()
                            if not re.match(r"^\s*keywords\s*:", line, re.I))
        if not citation:
            continue
        match = re.search(r'"(.+?)"\s*,?\s*(.*?)\bdoi:\s*(10\.\d{4,9}/[^\s,;]+)', citation, re.I)
        if not match:
            raise ValueError("Unrecognized IEEE plain-text citation block")
        title, tail, doi = match.groups()
        years = re.findall(r"\b(?:19|20)\d{2}\b", tail)
        records.append(Record(f"{source_key}:{len(records) + 1:04d}", source_key,
                              "IEEE Xplore", "", clean(title.rstrip(",")),
                              normalize_doi(doi), years[-1] if years else ""))
    if not records:
        raise ValueError("No IEEE plain-text citations found")
    return records


def load_history():
    # Snapshot every existing public historical file, not just the three pinned inputs.
    snapshot = {p.name: sha256(p.read_bytes()) for p in HISTORY.iterdir() if p.is_file()}
    for name, expected in HISTORY_SHA256.items():
        if snapshot.get(name) != expected:
            raise ValueError("Frozen historical input hash mismatch: " + name)
    if sha256(HISTORICAL_FULLTEXTS.read_bytes()) != HISTORICAL_FULLTEXTS_SHA256:
        raise ValueError("Frozen historical full-text evidence hash mismatch")
    rows = list(csv.DictReader(io.StringIO((HISTORY / "occurrences.csv").read_text(encoding="utf-8"))))
    if len(rows) != 244 or len({r["group_id"] for r in rows}) != 182:
        raise ValueError("Historical 244-occurrence/182-group scope changed")
    records = [Record(r["occurrence_id"], r["source_key"], r["database"], r["source_id"],
                      r["title"], r["doi"], r["year_reported"]) for r in rows]
    return rows, records, snapshot


def evidence_links(ids, field, execution_id, evidence, *, allow_transcription=False):
    if (not isinstance(ids, list) or not ids or any(not isinstance(e, str) for e in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Verified fields require distinct reviewed evidence IDs")
    for item in ids:
        if not isinstance(item, str) or item not in evidence:
            raise ValueError("Unknown evidence ID")
        entry = evidence[item]
        allowed_kinds = VISUAL_KINDS | ({"conversation_transcription"} if allow_transcription else set())
        if (entry["execution_id"] != execution_id or field not in entry["verified_fields"]
                or entry["reviewed_at"] is None or entry["kind"] not in allowed_kinds):
            raise ValueError("Evidence has not been reviewed for this execution and field")


def ingest(manifest_path: Path, output_dir: Path, *, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Ingestion clock requires a timezone")
    now = now.astimezone(timezone.utc)
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if manifest_path.is_relative_to(ROOT.resolve()):
        raise ValueError("The private manifest must be outside the checkout")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be empty; historical outputs are never overwritten")
    if (output_dir.is_relative_to(HISTORY.resolve())
            or output_dir.is_relative_to(HISTORICAL_FULLTEXTS.parent.resolve())):
        raise ValueError("Output cannot be inside a protected historical evidence directory")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    keys(manifest, {"schema_version", "campaign_id", "evidence", "executions"})
    if manifest["schema_version"] != "1.0":
        raise ValueError("Unsupported manifest schema")
    identifier(manifest["campaign_id"])
    historical_rows, records, historical_hashes = load_history()
    historical_ids = {r.occurrence_id for r in records}
    used_sources = {r.source_key for r in records}
    evidence = {}
    if not isinstance(manifest["evidence"], list):
        raise ValueError("Evidence must be a list")
    for entry in manifest["evidence"]:
        keys(entry, {"evidence_id", "execution_id", "path", "sha256", "kind", "collected_at", "reviewed_at", "verified_fields"},
             {"conversation_ref", "review_provenance"})
        eid = identifier(entry["evidence_id"])
        identifier(entry["execution_id"])
        if eid in evidence or not isinstance(entry["kind"], str) or entry["kind"] not in VISUAL_KINDS | {"browser_error", "conversation_transcription"}:
            raise ValueError("Duplicate evidence ID or unsupported evidence kind")
        collected = timestamp(entry["collected_at"], now)
        fields = entry["verified_fields"]
        if not isinstance(fields, list) or any(not isinstance(f, str) or f not in REVIEW_FIELDS for f in fields) or len(fields) != len(set(fields)):
            raise ValueError("Unknown or repeated reviewed field")
        if entry["reviewed_at"] is None:
            if fields:
                raise ValueError("Unreviewed evidence cannot verify fields")
        elif timestamp(entry["reviewed_at"], now) < collected:
            raise ValueError("Evidence review precedes collection")
        evidence[eid] = {k: entry[k] for k in entry if k != "path"}
        if entry["kind"] in {"conversation_image", "conversation_transcription"}:
            if (entry["path"] is not None or entry["sha256"] is not None
                    or not isinstance(entry.get("conversation_ref"), str) or not entry["conversation_ref"].strip()):
                raise ValueError("Conversation evidence requires an honest source reference and null path/hash")
            evidence[eid]["bytes"] = None
            evidence[eid]["integrity_status"] = ("conversation_image_witnessed_no_binary_hash_available"
                                                  if entry["kind"] == "conversation_image" else
                                                  "user_transcribed_native_interface_no_binary_hash_available")
        else:
            if "conversation_ref" in entry:
                raise ValueError("A local evidence file cannot use a conversation-only source reference")
            raw = private_bytes(entry["path"], entry["sha256"])
            evidence[eid]["bytes"] = len(raw)
            evidence[eid]["integrity_status"] = "binary_sha256_checked"
        if "review_provenance" in entry and (not isinstance(entry["review_provenance"], str) or not entry["review_provenance"].strip()):
            raise ValueError("Review provenance must be nonempty text")
    executions = manifest["executions"]
    if not isinstance(executions, list) or not 1 <= len(executions) <= 3:
        raise ValueError("Supply one to three distinct native database executions")
    seen_databases, seen_executions, used_hashes = set(), set(), set()
    public_executions, new_records = [], []
    for execution in executions:
        keys(execution, {"execution_id", "database", "status", "verified", "user_declared", "exports", "attempt_evidence_ids"},
             {"reported_native_total_evidence_ids", "execution_day_context"})
        xid = identifier(execution["execution_id"])
        database, status = execution["database"], execution["status"]
        if (not isinstance(database, str) or database not in DATABASES
                or xid in seen_executions or database in seen_databases):
            raise ValueError("Execution IDs and databases must be distinct")
        if not isinstance(status, str) or status not in {"executed", "blocked", "not_executed"}:
            raise ValueError("Unknown execution status")
        seen_databases.add(database)
        seen_executions.add(xid)
        keys(execution["verified"], FIELDS, OPTIONAL_FIELDS)
        keys(execution["user_declared"], set(), FIELDS | OPTIONAL_FIELDS)
        for field, value in execution["user_declared"].items():
            validate_value(field, value, now)
        verified = execution["verified"]
        for field, claim in verified.items():
            if claim is None:
                continue
            keys(claim, {"value", "evidence_ids"})
            validate_value(field, claim["value"], now)
            evidence_links(claim["evidence_ids"], field, xid, evidence)
        if database == "Web of Science" and verified["editions"] is not None and not verified["editions"]["value"]:
            raise ValueError("WoS requires the selected editions, not an empty edition list")
        attempts = execution["attempt_evidence_ids"]
        if (not isinstance(attempts, list)
                or any(not isinstance(e, str) or e not in evidence or evidence[e]["execution_id"] != xid for e in attempts)
                or len(attempts) != len(set(attempts))):
            raise ValueError("Unknown or mismatched attempt evidence")
        if status == "blocked" and not any(evidence[e]["reviewed_at"] is not None
                                           and "access_error" in evidence[e]["verified_fields"] for e in attempts):
            raise ValueError("A blocked attempt requires reviewed access_error evidence")
        if status != "executed" and (any(v is not None for v in verified.values()) or execution["exports"] or "displayed_total" in execution["user_declared"]):
            raise ValueError("A blocked or unexecuted search has no verified execution or native total")
        executed_time = timestamp(verified["executed_at"]["value"], now) if verified["executed_at"] else None
        if executed_time:
            for claim in verified.values():
                if claim is not None and any(timestamp(evidence[e]["collected_at"], now) < executed_time for e in claim["evidence_ids"]):
                    raise ValueError("Execution postdates its cited evidence")
        verified_total = verified["displayed_total"]["value"] if verified["displayed_total"] else None
        reported_refs = execution.get("reported_native_total_evidence_ids", [])
        if not isinstance(reported_refs, list):
            raise ValueError("Reported native-total evidence IDs must be a list")
        if reported_refs:
            evidence_links(reported_refs, "displayed_total", xid, evidence, allow_transcription=True)
            if ("displayed_total" not in execution["user_declared"]
                    or any(evidence[e]["kind"] != "conversation_transcription" for e in reported_refs)):
                raise ValueError("A reported native total needs its user-transcribed interface evidence")
        total = verified_total if verified_total is not None else execution["user_declared"].get("displayed_total") if reported_refs else None
        total_source = ("visually_witnessed_native_result" if verified_total is not None else
                        "user_transcribed_native_result_heading" if reported_refs else None)
        if not isinstance(execution["exports"], list):
            raise ValueError("Exports must be a list")
        ranges, public_exports = [], []
        for export in execution["exports"]:
            keys(export, {"export_id", "path", "sha256", "format", "range_start", "range_end", "range_evidence_ids"})
            source_key = identifier(export["export_id"])
            if (source_key in used_sources or not isinstance(export["format"], str)
                    or export["format"] not in FORMATS or FORMATS[export["format"]] != database):
                raise ValueError("Export source key collision or wrong native format")
            used_sources.add(source_key)
            raw = private_bytes(export["path"], export["sha256"])
            if export["sha256"] in used_hashes:
                raise ValueError("The same export bytes cannot be counted twice")
            used_hashes.add(export["sha256"])
            text = raw.decode("utf-8-sig")
            parser = {"scopus_csv": parse_scopus, "wos_tagged": parse_wos, "ieee_csv": parse_ieee,
                      "ieee_plain_text": parse_ieee_plain_text}[export["format"]]
            parsed = parser(text, source_key)
            if not parsed or any(not r.title or not (r.source_id or r.doi) for r in parsed):
                raise ValueError("An export needs nonempty rows, titles and native IDs or DOIs")
            start, end = export["range_start"], export["range_end"]
            if (start is None) != (end is None):
                raise ValueError("Unknown export range needs both endpoints null")
            if start is not None:
                if type(start) is not int or type(end) is not int or not 1 <= start <= end or end - start + 1 != len(parsed):
                    raise ValueError("Export range cardinality must equal its parsed row count")
                if total is not None and end > total:
                    raise ValueError("Export range exceeds the verified displayed total")
                if any(start <= right and left <= end for left, right in ranges):
                    raise ValueError("Export ranges overlap within an execution")
                ranges.append((start, end))
            refs = export["range_evidence_ids"]
            if refs:
                if start is None:
                    raise ValueError("An unknown export range cannot have range-verification evidence")
                evidence_links(refs, "export_range", xid, evidence)
                if executed_time and any(timestamp(evidence[e]["collected_at"], now) < executed_time for e in refs):
                    raise ValueError("Export range evidence precedes execution")
            elif not isinstance(refs, list):
                raise ValueError("Export range evidence IDs must be a list")
            # Strip private payloads before even combining them with historical metadata.
            new_records.extend(Record(r.occurrence_id, r.source_key, r.database, r.source_id,
                                      r.title, r.doi, r.year) for r in parsed)
            public_exports.append({k: export[k] for k in export if k != "path"} | {
                "bytes": len(raw), "parsed_occurrences": len(parsed), "range_verified": bool(refs),
                "original_filename": Path(export["path"]).name,
            })
        count = sum(e["parsed_occurrences"] for e in public_exports)
        if total is not None and count > total:
            raise ValueError("Parsed export rows exceed the documented native total")
        export_complete = (total is not None and count == total and all(e["range_verified"] for e in public_exports)
                           and (total == 0 or (ranges and min(r[0] for r in ranges) == 1 and max(r[1] for r in ranges) == total)))
        metadata_complete = status == "executed" and all(verified[f] is not None for f in FIELDS)
        if total == 0 and execution["exports"]:
            raise ValueError("A verified zero-result search cannot have exported rows")
        disagreements = [f for f, v in execution["user_declared"].items() if verified.get(f) is not None and v != verified[f]["value"]]
        filters = verified["filters"]["value"] if verified["filters"] else None
        query_source = ("reviewed_native_evidence" if verified["query_literal"] is not None else
                        "user_supplied_literal" if execution["user_declared"].get("query_literal") else None)
        date_source = ("reviewed_native_evidence" if verified.get("executed_on") or verified["executed_at"] else
                       "user_confirmed_day" if execution["user_declared"].get("executed_on") else
                       "user_confirmed_instant" if execution["user_declared"].get("executed_at") else None)
        day_context = execution.get("execution_day_context")
        if day_context is not None:
            keys(day_context, {"date", "basis", "source_export_id"}, {"timezone"})
            validate_value("executed_on", day_context["date"], now)
            if day_context["basis"] != "native_export_filename_and_current_session_receipt":
                raise ValueError("Unsupported contextual execution-day basis")
            source = next((e for e in public_exports if e["export_id"] == day_context["source_export_id"]), None)
            match = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})\.\d{1,2}\.\d{1,2}\.\d{1,2}", source["original_filename"]) if source else None
            if not match or date(*map(int, match.groups())).isoformat() != day_context["date"]:
                raise ValueError("Contextual date does not match its received export filename")
            if "timezone" in day_context:
                validate_value("execution_date_timezone", day_context["timezone"], now)
            date_source = date_source or "native_export_filename_and_current_session_receipt"
        guide_complete = bool(status == "executed" and total is not None and query_source and date_source)
        year_outliers = [r.occurrence_id for r in new_records if r.source_key in {e["export_id"] for e in public_exports}
                         and r.year.isdigit() and filters and not filters["publication_year_start"] <= int(r.year) <= filters["publication_year_end"]]
        public_executions.append({
            "execution_id": xid, "database": database, "status": status,
            "verified": verified, "user_declared": execution["user_declared"],
            "declaration_disagreements": disagreements, "attempt_evidence_ids": attempts,
            "reported_native_total_evidence_ids": reported_refs,
            "native_displayed_total": total, "native_displayed_total_provenance": total_source,
            "execution_day_context": day_context,
            "exports": public_exports, "native_metadata_complete": metadata_complete,
            "native_export_complete": bool(metadata_complete and export_complete),
            "verified_zero_result_execution": bool(guide_complete and verified_total == 0),
            "guide_3_1_search_log_complete": guide_complete,
            "guide_3_1_provenance": {"literal_query": query_source, "execution_date": date_source,
                                     "native_displayed_total": total_source},
            "exported_row_count_equals_native_total": count == total if total is not None else None,
            "export_all_results_user_confirmed": execution["user_declared"].get("exported_all_results"),
            "new_occurrences_outside_verified_year_filter": year_outliers,
            "publication_year_filter_matches_2019_2026": (filters["publication_year_start"] == 2019 and filters["publication_year_end"] == 2026) if filters else None,
        })
    if any(e["execution_id"] not in seen_executions for e in evidence.values()):
        raise ValueError("Evidence points to an execution absent from the manifest")
    records.extend(new_records)
    if len({r.occurrence_id for r in records}) != len(records):
        raise ValueError("Occurrence identifier collision")
    groups, decisions = deduplicate(records)
    fresh_groups, fresh_decisions = deduplicate(new_records)
    exact_records = [Record(r.occurrence_id, r.source_key, r.database, r.source_id, "", r.doi, r.year) for r in records]
    fresh_exact_groups, fresh_exact_decisions = deduplicate([r for r in exact_records if r.occurrence_id not in historical_ids])
    global_exact_groups, _ = deduplicate(exact_records)
    old_group = {r["occurrence_id"]: r["group_id"] for r in historical_rows}
    exact_overlap_old_groups = {old_group[r.occurrence_id] for g in global_exact_groups
                                if any(r.occurrence_id not in historical_ids for r in g)
                                for r in g if r.occurrence_id in historical_ids}
    occurrence_group, record_rows = {}, []
    for group in groups:
        ids = sorted(r.occurrence_id for r in group)
        group_id = "E1G-" + sha256(json_bytes(ids))[:24]
        for occurrence_id in ids:
            occurrence_group[occurrence_id] = group_id
        title_link = any(d["method"] == "normalized_title_provisional" and d["decision"] == "candidate_group_link" and d["left_occurrence"] in ids and d["right_occurrence"] in ids for d in decisions)
        record_rows.append({
            "group_id": group_id, "occurrence_count": len(group),
            "historical_group_ids": ";".join(sorted({old_group[i] for i in ids if i in old_group})),
            "occurrence_ids": ";".join(ids), "title": group[0].title,
            "dois": ";".join(sorted({r.doi for r in group if r.doi})),
            "databases": ";".join(sorted({r.database for r in group})),
            "identity_status": "provisional_title_link" if title_link else "doi_or_native_id_link" if len(group) > 1 else "single_occurrence",
        })
    occurrence_rows = [{
        "occurrence_id": r.occurrence_id, "group_id": occurrence_group[r.occurrence_id],
        "historical_group_id": old_group.get(r.occurrence_id, ""), "source_key": r.source_key,
        "database": r.database, "source_id": r.source_id, "title": r.title, "doi": r.doi,
        "year_reported": r.year, "origin": "historical_export" if r.occurrence_id in historical_ids else "new_native_export",
    } for r in records]
    summary = {
        "schema_version": "1.0", "campaign_id": manifest["campaign_id"],
        "new_exported_occurrences": len(new_records), "global_occurrences": len(records),
        "global_candidate_groups_provisional": len(groups),
        "new_occurrences_by_database": dict(sorted(Counter(r.database for r in new_records).items())),
        "verified_metadata_executions": sum(e["native_metadata_complete"] for e in public_executions),
        "all_three_native_metadata_verified": seen_databases == DATABASES and all(e["native_metadata_complete"] for e in public_executions),
        "guide_3_1_search_log_complete": seen_databases == DATABASES and all(e["guide_3_1_search_log_complete"] for e in public_executions),
        "native_execution_documentation_complete": seen_databases == DATABASES and all(e["guide_3_1_search_log_complete"] for e in public_executions),
        "three_databases_consulted": seen_databases == DATABASES and all(e["status"] == "executed" for e in public_executions),
        "all_three_native_exports_complete": seen_databases == DATABASES and all(e["native_export_complete"] for e in public_executions),
        "all_three_requested_year_filters_verified": seen_databases == DATABASES and all(e["publication_year_filter_matches_2019_2026"] is True for e in public_executions),
        "historical_occurrences_preserved": 244, "historical_group_ids_preserved": 182,
        "historical_crypto_fulltext_denominator_preserved": 7,
        "new_final_inclusions": None, "fulltext_selection_performed": False,
        "training_performed": False, "historical_search_queries_reconstructed": False,
        "new_round_flow": {
            "branch": "broad_native_search_reexecution_audit_not_a_new_systematic_review",
            "native_result_occurrences_by_database": {e["database"]: e["native_displayed_total"] for e in public_executions},
            "sum_native_result_occurrences_not_unique_studies": sum(e["native_displayed_total"] for e in public_executions) if all(e["native_displayed_total"] is not None for e in public_executions) else None,
            "received_exported_occurrences": len(new_records),
            "native_occurrences_not_in_received_exports": sum(e["native_displayed_total"] - sum(x["parsed_occurrences"] for x in e["exports"]) for e in public_executions) if all(e["native_displayed_total"] is not None for e in public_executions) else None,
            "exact_doi_or_native_id_groups": len(fresh_exact_groups),
            "exact_link_decisions": len(fresh_exact_decisions),
            "provisional_title_link_decisions": sum(d["method"] == "normalized_title_provisional" and d["decision"] == "candidate_group_link" for d in fresh_decisions),
            "candidate_groups_provisional_from_received_exports": len(fresh_groups),
            "evaluated_for_new_fulltext_selection": None, "new_studies_included": None,
            "screening_or_inclusion_decisions_made_in_this_audit": False,
            "scope_note": "Deduplication covers received exports only. Native result sums are database occurrences with the actual recorded platform filters; they are not unique studies or a harmonized selected corpus.",
        },
        "global_overlap": {
            "historical_groups_with_exact_doi_or_native_id_overlap": len(exact_overlap_old_groups),
            "historical_only_groups": sum(all(r.occurrence_id in historical_ids for r in g) for g in groups),
            "fresh_only_groups": sum(all(r.occurrence_id not in historical_ids for r in g) for g in groups),
            "historical_group_mergers": sum(len({old_group[r.occurrence_id] for r in g if r.occurrence_id in historical_ids}) > 1 for g in groups),
        },
    }
    run = {
        "schema_version": "1.0", "campaign_id": manifest["campaign_id"],
        "ingested_at": now.isoformat(), "new_search_campaign_not_historical_reconstruction": True,
        "private_manifest_sha256": sha256(raw_manifest), "evidence": list(evidence.values()),
        "executions": public_executions, "historical_public_file_sha256": historical_hashes,
        "historical_fulltext_evidence_sha256": HISTORICAL_FULLTEXTS_SHA256,
        "raw_exports_republished": False, "raw_evidence_republished": False,
        "verification_scope": "File hashes, explicit field-review assertions, chronology, ranges and row counts checked. Evidence content must have been inspected by the manifest author; this script does not interpret screenshots.",
    }
    files = {
        "README.md": (STAGE / "README.md").read_bytes(),
        "schema.json": (STAGE / "schema.json").read_bytes(),
        "execution.json": json_bytes(run), "summary.json": json_bytes(summary),
        "occurrences.csv": csv_bytes(occurrence_rows, OCCURRENCE_COLUMNS),
        "records.csv": csv_bytes(record_rows, RECORD_COLUMNS),
        "deduplication.csv": csv_bytes(decisions, DECISION_COLUMNS),
    }
    for name, data in files.items():
        if EMAIL.search(data):
            raise ValueError("Contact text detected in allowlisted public metadata: " + name)
    # Complete validation before creating the output or writing a single artifact.
    if historical_hashes != {p.name: sha256(p.read_bytes()) for p in HISTORY.iterdir() if p.is_file()}:
        raise ValueError("Historical evidence changed during ingestion")
    if sha256(HISTORICAL_FULLTEXTS.read_bytes()) != HISTORICAL_FULLTEXTS_SHA256:
        raise ValueError("Historical full-text evidence changed during ingestion")
    files["manifest.json"] = json_bytes({
        "schema_version": "1.0", "publication_schema": "e1_public_metadata_allowlist_v1",
        "execution_schema": "e1_native_execution_metadata_v1",
        "approved_public_files": sorted(files),
        "files": {name: {"sha256": sha256(data), "bytes": len(data)} for name, data in files.items()},
        "generator_sha256": sha256(Path(__file__).read_bytes()),
        "identity_importer_sha256": sha256((ROOT / "scripts/import_e1_literature.py").read_bytes()),
        "input_schema_sha256": sha256((STAGE / "schema.json").read_bytes()),
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = ingest(args.manifest, args.output_dir)
    except (ValueError, OSError, UnicodeError, csv.Error) as error:
        parser.exit(2, "Ingestion rejected: " + str(error) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
