#!/usr/bin/env python3
"""Compile the reviewed E1 access amendment, without searching or reading papers.

Reading, access route, reuse licence, target, model role and strict eligibility
remain separate. Inputs are explicit reviewed metadata, pinned by SHA256; the
historical selection and its statuses are never rewritten.
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
from xml.sax.saxutils import escape

from refresh_e1_native_downloads import safe_member, validate_public_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "references/literature_e1"
STAGE = BASE / "open_access_replacement_20261107"
MIRROR = ROOT / "book/_static/literature_e1/open_access_replacement_20261107"
PUBLICATION_SCHEMA = "e1_public_metadata_allowlist_v1"
NATIVE_IDS = frozenset("E1N007 E1N053 E1N057 E1N074 E1N075 E1N080 E1N091 E1N118 E1N120 E1N121 E1N124 E1N129 E1N133 E1N138 E1N142 E1N152 E1N163 E1N169 E1N170 E1N173 E1N175 E1N177".split())
UPLOADS = {"E1N057": "part_upload_pricevol/review.json", "E1N118": "part_upload_multitarget/review.json", "E1N138": "part_upload_psogru/review.json"}
CLOSED_NATIVE_IDS = frozenset("E1N053 E1N057 E1N075 E1N091 E1N118 E1N138 E1N163 E1N170".split())
NOT_RETRIEVED_IDS = NATIVE_IDS - CLOSED_NATIVE_IDS
NATIVE_ORIGINAL_IDS = frozenset("E1N053 E1N057 E1N075 E1N118 E1N138 E1N163".split())
WEB_IDS = frozenset("E1OA-T01 E1OA-T02 E1OA-T03 E1T3 E1X002 E1OA-R01 E1OA-R02 E1OA-R03 E1OA-R04 E1X001".split())
FREE_FULLTEXT_URLS = {
    "E1OA-R01": "https://arxiv.org/html/2311.04727v2",
    "E1OA-R02": "https://piotr.fiszeder.prac.umk.pl/Dudek_Fiszeder_Kobus_Orzeszko_ASC.pdf",
    "E1OA-R03": "https://journals.sagepub.com/doi/pdf/10.3233/FAIA250121?download=true",
    "E1OA-R04": "https://arxiv.org/html/2508.15922v1",
    "E1X001": "https://link.springer.com/article/10.1007/s10690-024-09510-6",
    "E1OA-T01": "https://pdfs.semanticscholar.org/1b82/b28c49e99e9a216708516dc5d99eca0e5578.pdf",
    "E1OA-T02": "https://projekter.aau.dk/projekter/files/484145663/thesis.pdf",
    "E1OA-T03": "https://link.springer.com/article/10.1007/s41060-024-00562-5",
    "E1T3": "https://arxiv.org/pdf/2210.04797v3",
}
INPUT_FILES = (
    "protocol.json",
    "part_upload_pricevol/README.md", "part_upload_pricevol/review.json", "part_upload_pricevol/reference.bib",
    "part_upload_psogru/README.md", "part_upload_psogru/review.json", "part_upload_psogru/reference.bib",
    "part_upload_multitarget/README.md", "part_upload_multitarget/review.json", "part_upload_multitarget/reference.bib",
    "part_open_tcn/README.md", "part_open_tcn/reviews.json", "part_open_tcn/references.bib", "part_open_tcn/search_log.json",
    "part_open_recurrent/README.md", "part_open_recurrent/reviews.json", "part_open_recurrent/references.bib", "part_open_recurrent/search_log.json",
)
OUTPUT_FILES = frozenset({"README.md", "summary.json", "reviews.json", "sources.csv", "native_dispositions.csv", "replacement_map.csv", "model_evidence.csv", "flow.svg"})
PUBLIC_FILES = frozenset(INPUT_FILES) | OUTPUT_FILES
# Filled only after each owning reviewer declares its public input set READY.
REVIEWED_INPUT_SHA256 = {
    "part_open_recurrent/README.md": "ff2dbe5a520eb0f7e769b5f9795d0ce3017d75ed4132656a0380c7f30c250a10",
    "part_open_recurrent/references.bib": "ad410b0066a7755a803314a02e9717fe41e60115081454436efb07c27e937363",
    "part_open_recurrent/reviews.json": "71985a9924626516a33110913dd70daa6767dfba53da936c378ff651cba384c2",
    "part_open_recurrent/search_log.json": "677eb87b1f795afba61f899b86d47736732ebe100c2bb233ce78627e1ac3391f",
    "part_open_tcn/README.md": "9df37d7fcffbb867a3bfb34f3bd1ee17bcc55bf7f2bb8cb8afa1222093a35962",
    "part_open_tcn/references.bib": "294add976b8aae100c9876fca244cd5f53a5c4c842b2f0c73251ee5b2c6a102a",
    "part_open_tcn/reviews.json": "2e1cea4266446e0b722dfddf672e41081e6ae8f0e450f2794a10c8a6be62c056",
    "part_open_tcn/search_log.json": "78079683df00797a7df443d7a94bf0924794d9c8eb6d92563c1966c8aaa249c7",
    "part_upload_multitarget/README.md": "2e15e479ecdbd4be0ac06c629f7c0c848f8faf08ec9d2c33d6162843fd14dce4",
    "part_upload_multitarget/reference.bib": "6698b9d04716551418a5a28166d2e8e5ad2858dc3ff1962bb7b932af785178b4",
    "part_upload_multitarget/review.json": "33431cc2240e5964a1f308e5b9563052f04e593cf4a6e4452ef38ac7206a034d",
    "part_upload_pricevol/README.md": "d42d5b92a9617b8ae6a1f1a1dfd11d2fce917cdd345a80bf05a2083c31c89328",
    "part_upload_pricevol/reference.bib": "31be360c59b2c146462ca2e69ccb27c1cd91bc849863a9d73d65611ff6a308d0",
    "part_upload_pricevol/review.json": "beb00527c9e92d4fa2caa541977db36751c19f199769a1a5b095f66fb4f77870",
    "part_upload_psogru/README.md": "8e491109e20de69d1928f02554e144981a6203e6e60a5356205f59cfdae111c5",
    "part_upload_psogru/reference.bib": "1ca3b86b2d707c92a291f317187f60fb193383801ce58335d08c001ab02cdb63",
    "part_upload_psogru/review.json": "c12118382b5708b0810374cf3d0ed04f1c7925cfc13c0eb7b3cac2c5a67bab3e",
    "protocol.json": "78437a737b4ee753325811886a4847edf62af9f9282daba243f6b2425996a882"
}
FROZEN_FILES = {
    "crypto_extension_20261105/evidence_matrix.json": "4b5eb36f0db801f15df64a5f04cc053a107165a86ff89f755afe9e548ec72f82",
    "editorial_screening_20261106/decisions.csv": "72d39cc335c06f89e6b04939fa12e30eba2d4dc8082c69eaa747aeb0e09ee96e",
    "internet_closure_20261106/decisions.csv": "6fed5b3161670f210fc61884a957ada0765b68931cb70189dcef53086e0542fb",
    "internet_closure_20261106/manifest.json": "e7bcd2bfe077fa3fbd4edd72e870f274bcde2f25e67f1b4b3249c36cc7405d57",
    "internet_closure_20261106/model_evidence.csv": "13dc83ec011717da86190e9349e3fa4f56423a3deaeb1603bd14f0b581286fdf",
    "internet_closure_20261106/part_a/README.md": "f7690bdbb98105db0db80fdb0c793ad42ce675d55b38b45d0446eb409841bb2a",
    "internet_closure_20261106/part_a/read_originals.bib": "672db45ae44f3da4f8573e93f115176006c9d010e7209ae76f666911cf732a6c",
    "internet_closure_20261106/part_a/reviews.json": "70322747e7174b4f6fea0922eea10fc690ab0c7c4f2c8ee7bf2437a27a29a463",
    "internet_closure_20261106/part_b/README.md": "ad3963781b021cdaa52a52992d744b8c9c43241e8f8fe010dabd9e523af7f259",
    "internet_closure_20261106/part_b/reviews.json": "0b6d641003c321d1f997d60c930914e3ab5927bfc5b746f600d6cf40ca16645d",
    "internet_closure_20261106/part_c/README.md": "fbaa619c87166b7297f7d80ef811172a681f5052f0820c1d424e24e95d1589ca",
    "internet_closure_20261106/part_c/manifest.sha256.json": "004353043c1a17157e4e44d1284c3c2202e6fac78b6349d8d301161513ff19c4",
    "internet_closure_20261106/part_c/public_web_query_log.json": "12f1d58e4d748652c8d3ee1616d9cc935191fc2594b391632ed8c8fbc38e7476",
    "internet_closure_20261106/part_c/reviews.json": "d71415dfcdb681d4c26496dbefc282a1a0fb2ecc76e74550c083415093d9d7db",
    "internet_closure_20261106/part_d/README.md": "165aef287b06f7f4b5e296404119fef3d055192b97bd276217efb786a0c62ea2",
    "internet_closure_20261106/part_d/consultations.json": "bc06f517235789c8dfe19385f07320139d79ba737594093d5eeeb362411c06cc",
    "internet_closure_20261106/part_d/read_sources.bib": "d9d34441a95c68e58ecd0d9686f24b5e8e9000acc497cbd36ce2ad96142fdd94",
    "internet_closure_20261106/part_d/reviews.json": "f028c820d2a90d0a93bb50cfc229d048f0cf4f4133c3f1063d67c0ef9084cc11",
    "internet_closure_20261106/part_e/README.md": "8cca25acbe320cb48c475ea70301803dbdda4eebec78fe2c76dfcdbd1740ef64",
    "internet_closure_20261106/part_e/manifest.sha256.json": "09063e7bcb60d610c209d804fcbcc5339d00f4182d79f6f1574ba63565c894c6",
    "internet_closure_20261106/part_e/public_web_query_log.json": "d421e346890c8c8b498de40ec2571b3daa8fa7915830a3d7e7976552b8ed45f5",
    "internet_closure_20261106/part_e/reviews.json": "5b40f590cfe27af5ffc86412686435d4c169ad347614076ec2f11ab78855e315",
    "internet_closure_20261106/part_root/README.md": "8e26a83b5f0b5f9f002a9947ab4c3f4ceca10b58252ab2b8d255724990a908ab",
    "internet_closure_20261106/part_root/public_web_query_log.json": "51f6ace9c337d83e142a6f4c90bf87d79c19686e53d5f4291d05d34b03501720",
    "internet_closure_20261106/part_root/reviews.json": "cdfb2730f589b9f4bd2f1a96745a90126e66f1d3d8a013b0fcd23c6ddf31dc42",
    "internet_closure_20261106/reviews.json": "d1f8d811a261946e7ab85ab211120669cdf995d00370e4659822294fbbd1241b",
    "internet_closure_20261106/summary.json": "443992af498f838c1375989afb62c7654daa2de7abeec358dda0c1668b155aa2",
    "native_search_20261106/records.csv": "1a3d2072f233e2ed5d927642d9c2cb85c39ab4bf41f89f1f11fa6c375cb753b7",
    "native_search_20261106/summary.json": "6474360911be95ecf956f4b9e984adb7ebde54ecda20cb14055dc6e0f7accbf9"
}
# Explicit adjudication registry. No model/target classification from keywords.
WEB_ADJUDICATIONS = {
    "E1OA-T01": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": True, "study_identity": "new_accessible_source", "new_unique_reference_global": True, "include_in_new_stratum": True, "families": ["TCN", "LSTM"], "model_roles": {"TCN": "autonomous_original_transfer_major_protocol_reservations", "LSTM": "autonomous_original_with_baidu_predictor_transfer"}},
    "E1OA-T02": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": False, "study_identity": "new_accessible_source", "new_unique_reference_global": True, "include_in_new_stratum": True, "families": ["DCCNN", "MLP"], "model_roles": {"DCCNN": "autonomous_grey_crypto_har_predictors_not_residual_tcn", "MLP": "autonomous_grey_crypto_eight_layers_not_minimal", "TCN": "not_verified_as_standalone_residual_tcn"}},
    "E1OA-T03": {"full_text_read": True, "full_tables_read": False, "open_reuse_license_verified": True, "study_identity": "new_accessible_source", "new_unique_reference_global": True, "include_in_new_stratum": False, "families": [], "model_roles": {"TCN": "hybrid_tcn_lstm_component_only", "LSTM": "hybrid_tcn_lstm_component_only"}},
    "E1T3": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": False, "study_identity": "existing_source_upgrade", "new_unique_reference_global": False, "include_in_new_stratum": True, "families": ["LSTM", "MLP"], "model_roles": {"DCCNN": "proposed_dilated_causal_convolution_with_attention", "TCN": "not_verified_as_standalone_simple_tcn", "LSTM": "autonomous_original_transfer", "MLP": "autonomous_original_transfer"}},
    "E1X002": {"full_text_read": False, "full_tables_read": False, "open_reuse_license_verified": False, "study_identity": "existing_partial_source_unchanged", "new_unique_reference_global": False, "include_in_new_stratum": False, "families": [], "model_roles": {"LSTM": "author_code_only_original_version_unverified", "TCN": "author_code_only_original_version_unverified"}},
    "E1OA-R01": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": False, "study_identity": "new_accessible_source", "new_unique_reference_global": True, "include_in_new_stratum": True, "families": ["LSTM"], "model_roles": {"LSTM": "autonomous_original_crypto_one_layer_two_or_four_units_recipe_partial"}},
    "E1OA-R02": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": False, "study_identity": "new_accessible_source", "new_unique_reference_global": False, "include_in_new_stratum": True, "families": ["LSTM", "MLP"], "dependence_note": "Existing E2 DOI; new full reading for E1, not a new global reference.", "model_roles": {"LSTM": "autonomous_original_crypto_selected_width_missing", "MLP": "autonomous_original_crypto_selected_width_missing"}},
    "E1OA-R03": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": True, "study_identity": "new_accessible_source", "new_unique_reference_global": True, "include_in_new_stratum": False, "families": ["GRU", "LSTM"], "model_roles": {"GRU": "autonomous_original_crypto_quality_context_recipe_insufficient", "LSTM": "autonomous_original_crypto_quality_context_recipe_insufficient"}},
    "E1OA-R04": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": True, "study_identity": "new_accessible_source", "new_unique_reference_global": True, "include_in_new_stratum": False, "families": ["LSTM", "MLP"], "dependence_note": "Reuses base data and point forecasts of E1OA-R02; not an independent dataset replication or new LSTM recipe.", "model_roles": {"LSTM": "inherited_autonomous_base_forecast_from_R02", "MLP": "inherited_autonomous_base_forecast_from_R02"}},
    "E1X001": {"full_text_read": True, "full_tables_read": True, "open_reuse_license_verified": True, "study_identity": "existing_source_upgrade", "new_unique_reference_global": False, "include_in_new_stratum": True, "families": ["LSTM", "MLP"], "model_roles": {"LSTM": "autonomous_original_crypto_recipe_insufficient", "MLP": "autonomous_original_crypto_recipe_insufficient"}},
}
REPLACEMENT_ROLES = {
    "E1N007": ("Contraste recurrente y volatilidad cripto; contagio no sustituido", "E1OA-R01;E1X001", "No demuestra equivalencia con un estudio de spillover."),
    "E1N074": ("Volatilidad cripto y familias recurrentes/convolucionales", "E1OA-R01;E1X001;E1T3", "DeepVol es transferencia de otros activos; no réplica cripto."),
    "E1N080": ("Volatilidad multiactivo y modelos híbridos como contexto", "E1OA-R01;E1OA-R03", "No sustituye cobertura cambiaria ni dependencia BEKK."),
    "E1N120": ("Ventanas y protocolo de pronóstico de volatilidad", "E1X001;E1OA-R02;E1T3", "Las ventanas y horizontes son específicos del proxy; no se igualan a sigma7."),
    "E1N121": ("Objetivo continuo de volatilidad Bitcoin", "E1OA-R01;E1X001", "No reemplaza una prueba de contribución de sentimiento social."),
    "E1N124": ("Separación de componentes híbridos y comparadores autónomos", "E1OA-R01;E1OA-R03;E1T3", "No reconstruye el canal NASDAQ ni los horizontes del original no obtenido."),
    "E1N129": ("Volatilidad de Ethereum y otros criptoactivos", "E1OA-R01;E1X001", "El original anuncia precio; estas fuentes cubren volatilidad, sin afirmar mismo target."),
    "E1N133": ("Predicción multihorizonte y modelos recurrentes", "E1OA-R01;E1OA-R02;E1OA-R04", "R02/R04 comparten base de datos; no son réplicas independientes."),
    "E1N142": ("Comparadores temporales y arquitectura convolucional", "E1T3;E1OA-T01;E1OA-T02", "No sustituye Transformer/Optuna; T01 y T02 conservan sus restricciones de transferencia/calidad."),
    "E1N152": ("Evidencia recurrente de volatilidad cripto", "E1OA-R01;E1X001", "No infiere que fluctuaciones del original signifiquen volatilidad."),
    "E1N169": ("Convolución temporal, datos de alta frecuencia y volatilidad", "E1T3;E1OA-T01;E1OA-T02;E1X002", "X002 sólo código parcial; DCCNN no se renombra TCN simple; no acredita target del original."),
    "E1N173": ("Volatilidad directa y transferencia VIX/otros activos", "E1OA-R01;E1OA-R02;E1OA-R04;E1OA-T03", "VIX queda separado de volatilidad realizada cripto; T03 no respalda métricas ausentes."),
    "E1N175": ("Familia convolucional temporal y objetivo de volatilidad", "E1T3;E1OA-T02;E1X002", "No cubre STORJ específicamente ni convierte su precio en volatilidad."),
    "E1N177": ("Definición operacional de volatilidad y baselines recurrentes", "E1OA-R01;E1X001", "RV no equivale automáticamente al rango máximo-mínimo de 24h."),
}


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata(data: bytes) -> dict:
    return {"sha256": digest(data), "bytes": len(data)}


def csv_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: str(v).lower() if type(v) is bool else v for k, v in row.items()})
    return output.getvalue().encode("utf-8")


def csv_read(data: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))


def checked_file(directory: Path, name: str) -> bytes:
    relative = safe_member(name)
    path = directory / relative
    if directory.is_symlink() or any(parent.is_symlink() for parent in directory.parents):
        raise ValueError("Symlink in evidence directory")
    if any((directory / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError("Symlink in public evidence: " + name)
    if not path.is_file():
        raise ValueError("Missing reviewed public file: " + name)
    data = path.read_bytes()
    validate_public_bytes(name, data)
    return data


def inventory(directory: Path) -> set[str]:
    if directory.is_symlink() or any(parent.is_symlink() for parent in directory.parents):
        raise ValueError("Symlink in evidence directory")
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Symlink in evidence inventory")
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}


def load_inputs(stage: Path, frozen_base: Path) -> tuple[dict, dict, list[dict], dict]:
    found = inventory(stage)
    if found - (PUBLIC_FILES | {"manifest.json"}) or not set(INPUT_FILES) <= found:
        raise ValueError("Stage inventory differs from explicit reviewed public inputs/outputs")
    if set(REVIEWED_INPUT_SHA256) != set(INPUT_FILES):
        raise ValueError("Input set is not READY and hash-locked")
    raw = {name: checked_file(stage, name) for name in INPUT_FILES}
    for name, data in raw.items():
        if digest(data) != REVIEWED_INPUT_SHA256[name]:
            raise ValueError("Reviewed input changed; explicit re-review required: " + name)
    frozen = {}
    for name, sha in FROZEN_FILES.items():
        data = checked_file(frozen_base, name)
        if digest(data) != sha:
            raise ValueError("Frozen historical evidence changed: " + name)
        frozen[name] = data
    protocol = json.loads(raw["protocol.json"])
    prior = json.loads(frozen["internet_closure_20261106/reviews.json"])
    native = {r["group_id"]: r for r in prior if r["scope"] == "native_22"}
    if set(native) != NATIVE_IDS or len(native) != 22:
        raise ValueError("Frozen native queue must contain exactly 22 unique IDs")
    if {r["group_id"] for r in csv_read(frozen["editorial_screening_20261106/decisions.csv"]) if r["decision"] == "seek_full_text"} != NATIVE_IDS:
        raise ValueError("Frozen editorial/native queue mismatch")
    uploads = {group: json.loads(raw[name]) for group, name in UPLOADS.items()}
    web = []
    for name in ("part_open_tcn/reviews.json", "part_open_recurrent/reviews.json"):
        rows = json.loads(raw[name])
        if not isinstance(rows, list):
            raise ValueError("Web review inputs must be arrays: " + name)
        web.extend({"source_review": row, "input_file": name} for row in rows)
    validate_records(uploads, web, native, protocol)
    all_native_dois = {r["doi"].lower() for r in csv_read(frozen["native_search_20261106/records.csv"]) if r["doi"]}
    if any(str(item["source_review"].get("doi") or "").lower() in all_native_dois for item in web):
        raise ValueError("Accessible web source duplicates an existing native 182-group DOI")
    return raw, frozen, web, uploads


def validate_records(uploads: dict, web: list[dict], native: dict, protocol: dict) -> None:
    ids = [item["source_review"]["group_id"] for item in web]
    if len(ids) != len(set(ids)) or set(ids) != WEB_IDS or set(WEB_ADJUDICATIONS) != WEB_IDS:
        raise ValueError("Web review coverage/explicit adjudications must equal ten unique approved IDs")
    all_dois = [str(item["source_review"].get("doi") or "").lower().removeprefix("https://doi.org/") for item in web]
    if len([x for x in all_dois if x]) != len(set(x for x in all_dois if x)):
        raise ValueError("Duplicate web DOI; upgrades must keep one source identity")
    native_dois = {str(r["doi"] or "").lower() for r in native.values()}
    if any(x in native_dois for x in all_dois if x):
        raise ValueError("Web source duplicates a native DOI")
    for group, source in uploads.items():
        source_id = source.get("group_id", source.get("record_id"))
        if source_id != group or source["doi"].lower() != native[group]["doi"].lower():
            raise ValueError("Uploaded original identity differs from frozen native record: " + group)
        if not upload_metadata(group, source)["full_text_read"]:
            raise ValueError("Uploaded original reading state changed: " + group)
    for item in web:
        source = item["source_review"]
        spec = WEB_ADJUDICATIONS[source["group_id"]]
        if source.get("full_text_read") is not spec["full_text_read"]:
            raise ValueError("Web reading state differs from reviewed adjudication")
    if set(protocol["other_originals_not_obtained_at_this_cut"]) != NOT_RETRIEVED_IDS:
        raise ValueError("Protocol must retain exactly 14 not-obtained originals")
    if set(protocol["initial_native_pending_ids"]) != NOT_RETRIEVED_IDS | set(UPLOADS):
        raise ValueError("Protocol must reconcile the previous 17 as 14+3")
    if protocol["native_cutoff_disposition"]["active_seek_count"] != 0:
        raise ValueError("No active seek queue remains at this cutoff")


def upload_metadata(group: str, source: dict) -> dict:
    if group == "E1N057":
        return {"full_text_read": source["access"]["original_full_read"], "access_route": source["access"]["tier"],
                "reuse_license": source["access"]["license_evidence"], "open_reuse_license_verified": False,
                "free_fulltext_web_read_this_stage": False, "citation_key": "Poudel2023CryptocurrencyPriceVolatility",
                "decision": source["eligibility"]["decision"], "reason": "; ".join(source["eligibility"]["reasons"]),
                "target_stratum": "price_return_volatility_target_ambiguous", "families": ["GRU", "LSTM"],
                "urls": [source["doi_url"]], "original_sha256": source["access"]["original_file_sha256"]}
    return {"full_text_read": source["full_text_read"], "access_route": source["access_level"],
            "reuse_license": source.get("license_reported", "Licencia de reutilización abierta no verificada; original aportado por usuario."),
            "open_reuse_license_verified": group == "E1N118", "free_fulltext_web_read_this_stage": False,
            "citation_key": source.get("bibliography_key_existing", source.get("recommended_bibtex_key")),
            "decision": source["eligibility_decision"], "reason": source["eligibility_reason"],
            "target_stratum": "final_continuous_volatility_target_not_demonstrated" if group == "E1N118" else "crypto_daily_realized_variation_insufficient_recipe_protocol",
            "families": ["LSTM", "TCN"] if group == "E1N118" else ["GRU", "LSTM"],
            "urls": source["source_urls"], "original_sha256": source["original_file_sha256"]}


def web_metadata(source: dict) -> dict:
    group = source["group_id"]
    spec = copy.deepcopy(WEB_ADJUDICATIONS[group])
    return {**spec, "citation_key": source["recommended_bibtex_key"], "access_route": source["access_level"],
            "free_fulltext_web_read_this_stage": spec["full_text_read"], "reuse_license": source["license"],
            "target_stratum": source.get("stratum", source.get("comparability_stratum", "daily_squared_centered_return_partial_code")),
            "strict_eligibility_status": source.get("eligibility_decision", source.get("decision")),
            "eligible_strict_reproducible_recipe": False, "simple_recipe_families": [],
            "quantitative_extraction_complete": spec["full_tables_read"], "urls": source["source_urls"]}


def compile_reviews(raw: dict, frozen: dict, web: list[dict], uploads: dict) -> list[dict]:
    old = {r["group_id"]: r for r in json.loads(frozen["internet_closure_20261106/reviews.json"])}
    compiled = []
    for group in sorted(NATIVE_IDS):
        previous = old[group]
        uploaded = group in uploads
        source = copy.deepcopy(uploads[group] if uploaded else previous["source_review"])
        info = upload_metadata(group, source) if uploaded else None
        missing = group in NOT_RETRIEVED_IDS
        compiled.append({"group_id": group, "title": previous["title"], "doi": previous["doi"], "scope": "native_22",
            "input_file": UPLOADS.get(group, "../internet_closure_20261106/reviews.json"),
            "source_access_level": info["access_route"] if info else previous["source_access_level"],
            "source_full_text_read": info["full_text_read"] if info else previous["source_full_text_read"],
            "original_read": group in NATIVE_ORIGINAL_IDS, "complete_document_read": group in NATIVE_ORIGINAL_IDS or group == "E1N091",
            "complete_mirror_transcription_read": group == "E1N091", "new_unique_original_read_this_stage": group in {"E1N057", "E1N138"},
            "reread_original_this_stage": group == "E1N118", "strict_eligibility_closed": not missing,
            "strict_eligibility_status": "not_retrieved_user_reported_paywall" if missing else "closed_excluded_context",
            "strict_eligibility_decision": "not_assessed_original_not_obtained" if missing else "exclude_current_core_retain_context",
            "retrieval_disposition_closed_at_cutoff": True, "active_seek_at_cutoff": False,
            "exclusion_for_quality_without_fulltext": False,
            "current_decision_reason": "Original no obtenido al corte solicitado; el usuario informa acceso de pago. No es exclusión por calidad ni verificación independiente del paywall." if missing else info["reason"] if info else previous["current_decision_reason"],
            "target_stratum": info["target_stratum"] if info else previous["target_stratum"],
            "citation_key": info["citation_key"] if info else "",
            "historical_N7_inclusion": False, "strict_recurrence_increment": 0,
            "previous_cut_adjudication": copy.deepcopy(previous), "source_review": source})
    for item in sorted(web, key=lambda r: r["source_review"]["group_id"]):
        source = copy.deepcopy(item["source_review"])
        spec = web_metadata(source)
        compiled.append({"group_id": source["group_id"], "title": source["title"], "doi": source.get("doi"),
            "scope": "accessible_web_outside_native_22", "input_file": item["input_file"],
            **copy.deepcopy(spec), "original_read": spec["full_text_read"], "complete_document_read": spec["full_text_read"],
            "complete_mirror_transcription_read": False, "new_unique_original_read_this_stage": spec["full_text_read"],
            "reread_original_this_stage": False, "historical_N7_inclusion": False, "strict_recurrence_increment": 0,
            "previous_cut_adjudication": copy.deepcopy(old.get(source["group_id"])), "source_review": source})
    return compiled


def sources(reviews: list[dict]) -> list[dict]:
    result = []
    for row in reviews:
        group, source = row["group_id"], row["source_review"]
        if group not in UPLOADS and group not in WEB_IDS:
            continue
        info = upload_metadata(group, source) if group in UPLOADS else web_metadata(source)
        result.append({"group_id": group, "title": row["title"], "doi": row["doi"], "citation_key": info["citation_key"],
            "source_scope": "user_supplied_original" if group in UPLOADS else "public_web_source",
            "access_route": info["access_route"], "free_fulltext_web_read_this_stage": info["free_fulltext_web_read_this_stage"],
            "free_full_text_url": FREE_FULLTEXT_URLS.get(group, ""),
            "full_text_read": row["complete_document_read"], "open_reuse_license_verified": info["open_reuse_license_verified"],
            "full_tables_read": True if group in UPLOADS else info["full_tables_read"],
            "reuse_license": info["reuse_license"], "strict_eligibility_status": row["strict_eligibility_status"],
            "target_stratum": row["target_stratum"], "new_study_vs_upgrade": info.get("study_identity", "native_reread" if group == "E1N118" else "native_access_upgrade"),
            "fulltext_or_primary_urls": ";".join(info["urls"]), "evidence_file": row["input_file"]})
    return result


def native_dispositions(reviews: list[dict]) -> list[dict]:
    fields = ("group_id", "title", "doi", "source_access_level", "original_read", "complete_document_read", "complete_mirror_transcription_read", "new_unique_original_read_this_stage", "reread_original_this_stage", "strict_eligibility_closed", "strict_eligibility_status", "strict_eligibility_decision", "retrieval_disposition_closed_at_cutoff", "active_seek_at_cutoff", "exclusion_for_quality_without_fulltext", "current_decision_reason", "input_file")
    return [{**{k: row[k] for k in fields}, "previous_cut_status": row["previous_cut_adjudication"]["strict_eligibility_status"], "historical_N7_inclusion": False} for row in reviews if row["scope"] == "native_22"]


def model_evidence(reviews: list[dict], frozen: dict) -> list[dict]:
    previous = {(r["group_id"], r["model"]): r for r in csv_read(frozen["internet_closure_20261106/model_evidence.csv"])}
    result = []
    for review in reviews:
        group = review["group_id"]
        families = upload_metadata(group, review["source_review"])["families"] if group in UPLOADS else WEB_ADJUDICATIONS[group]["families"] if group in WEB_IDS else []
        for model in ("GRU", "LSTM", "TCN", "DCCNN", "MLP", "CNN_1D", "RNN_simple"):
            old = previous.get((group, model), {})
            if group in UPLOADS:
                role = "autonomous_original_target_or_protocol_insufficient" if model in families else "not_confirmed_in_comparison"
            elif group in WEB_IDS:
                role = WEB_ADJUDICATIONS[group]["model_roles"].get(model, "not_confirmed_in_comparison")
            else:
                role = old.get("autonomous_presence_evidence", "not_assessed_in_this_stage")
            result.append({"group_id": group, "scope": review["scope"], "model": model, "role_evidence": role,
                "full_original_read": review["original_read"], "target_stratum": review["target_stratum"],
                "strict_eligibility_status": review["strict_eligibility_status"],
                "recipe_for_simple_e1_comparison_verified": group in WEB_IDS and model in WEB_ADJUDICATIONS[group].get("simple_recipe_families", []),
                "same_data_replication_caveat": WEB_ADJUDICATIONS.get(group, {}).get("dependence_note", ""),
                "strict_volatility_recurrence_increment": 0, "historical_N7_inclusion": False, "evidence_file": review["input_file"]})
    return result


def replacement_map(reviews: list[dict]) -> list[dict]:
    lookup = {r["group_id"]: r for r in reviews}
    rows = []
    for group, (role, alternatives, limit) in sorted(REPLACEMENT_ROLES.items()):
        candidates = alternatives.split(";")
        if any(x not in lookup for x in candidates):
            raise ValueError("Replacement role references an unknown reviewed source")
        rows.append({"native_id": group, "native_title": lookup[group]["title"], "native_retrieval_disposition": "not_retrieved_user_reported_paywall",
            "role_to_cover": role, "accessible_source_ids": alternatives,
            "citation_keys": ";".join(lookup[x]["citation_key"] for x in candidates),
            "source_reading_states": ";".join(f"{x}:{'original_read' if lookup[x]['original_read'] else 'partial_only'}" for x in candidates),
            "mapping_unit": "substantive_role_not_one_to_one_paper", "equivalent_original_replacement_claimed": False,
            "limitations": limit, "active_original_seek_at_cutoff": False})
    return rows


def summarize(reviews: list[dict], raw: dict) -> dict:
    native = [r for r in reviews if r["scope"] == "native_22"]
    web = [r for r in reviews if r["scope"] != "native_22"]
    ids = lambda rows, key: sorted(r["group_id"] for r in rows if r[key])
    return {"schema_version": 1, "stage": "open_access_replacement_20261107", "cutoff_date_utc": "2026-11-07", "review_date_america_bogota": "2026-11-06",
        "historical_exported_occurrences": 244, "historical_candidate_groups": 182, "historical_N7": 7, "historical_N7_changed": False,
        "historical_family_counts": {"LSTM": 5, "GRU": 3, "TCN": 1, "Transformer_encoder": 2, "MLP": 1, "CNN_generic": 1, "CNN_1D_explicit": 0, "RNN_simple": 0},
        "initial_native_pending": 17, "initial_pending_partition": {"uploaded_or_reprovided": 3, "not_obtained": 14},
        "native_queue_size": 22, "native_ids": sorted(NATIVE_IDS), "native_status_counts": dict(sorted(Counter(r["strict_eligibility_status"] for r in native).items())),
        "native_closed_excluded_context": 8, "native_closed_excluded_context_ids": sorted(CLOSED_NATIVE_IDS),
        "native_not_retrieved": 14, "native_not_retrieved_ids": sorted(NOT_RETRIEVED_IDS), "native_quality_excluded_without_reading": 0,
        "native_active_seek_at_cutoff": 0, "native_active_seek_ids": [], "native_pending_supplement_ids": [],
        "native_originals_read": len(ids(native, "original_read")), "native_originals_read_ids": ids(native, "original_read"),
        "native_complete_document_read": len(ids(native, "complete_document_read")), "native_complete_document_read_ids": ids(native, "complete_document_read"),
        "native_complete_mirror_transcriptions": 1, "native_complete_mirror_transcription_ids": ["E1N091"],
        "native_new_unique_originals_read_this_stage": 2, "native_new_unique_original_ids": ["E1N057", "E1N138"], "native_original_reread_ids": ["E1N118"],
        "web_catalog_sources": len(web), "web_catalog_ids": sorted(WEB_IDS), "web_full_originals_read": len(ids(web, "original_read")), "web_full_original_read_ids": ids(web, "original_read"),
        "web_partial_only_ids": sorted(r["group_id"] for r in web if not r["original_read"]),
        "web_original_body_and_tables_read": sum(r["original_read"] and r["full_tables_read"] for r in web),
        "web_original_body_only_without_tables_ids": sorted(r["group_id"] for r in web if r["original_read"] and not r["full_tables_read"]),
        "web_status_counts": dict(sorted(Counter(r["strict_eligibility_status"] for r in web).items())),
        "web_existing_source_upgrades": sorted(r["group_id"] for r in web if r["study_identity"] == "existing_source_upgrade"),
        "web_new_distinct_source_ids": sorted(r["group_id"] for r in web if r["study_identity"] == "new_accessible_source"),
        "web_new_global_reference_ids": sorted(r["group_id"] for r in web if r["new_unique_reference_global"]),
        "web_existing_E2_reference_new_E1_read_ids": ["E1OA-R02"],
        "catalogue_sources_including_uploads": len(sources(reviews)), "replacement_map_native_cases": 14, "replacement_map_is_bijective_paper_substitution": False,
        "native_new_strict_inclusions": 0, "strict_recurrence_increment": {"GRU": 0, "LSTM": 0, "TCN": 0}, "no_cross_stratum_metric_pooling": True,
        "reading_does_not_imply_inclusion": True, "user_supplied_does_not_imply_open_licence": True,
        "native_searches_recovered_by_this_stage": 0, "ieee_native_consultation_accredited_by_this_stage": False, "native_search_requirement_closed": False,
        "new_model_fits": 0, "new_predictions": 0, "experiments_changed": False,
        "chronology": "Ampliación documental posterior a los experimentos; no prerregistro retrospectivo ni justificación ex ante de selección.",
        "privacy": "Metadatos y análisis derivados; sin PDFs, texto completo, resúmenes nativos ni contactos.",
        "prior_summary_sha256": FROZEN_FILES["internet_closure_20261106/summary.json"],
        "reviewed_input_files": sorted(INPUT_FILES)}


def readme(summary: dict) -> str:
    return f'''# Acceso y sustitución documental por funciones · E1

La cola nativa de 22 informes queda adjudicada al corte de 2026-11-07 UTC (2026-11-06 en America/Bogota): **8 excluidos del núcleo/conservados como contexto, 14 originales no obtenidos y 0 informes en cola activa de recuperación al corte**. Los 14 no son exclusiones por calidad: el usuario informa acceso de pago, sin que esta etapa acredite una comprobación independiente de cada muro de acceso.

La cola previa de 17 se reconcilia como tres PDF aportados/reaportados y 14 no obtenidos. E1N057 y E1N138 añaden dos lecturas únicas; E1N118 es una relectura. En total nativo hay seis originales completos y una transcripción completa de espejo; E1N170 cerró pertinencia mediante método primario parcial. Lectura, acceso y elegibilidad son dimensiones distintas. N118 cierra como contexto porque el original no demuestra volatilidad continua finalmente evaluada; un suplemento futuro podría reabrirlo y no constituye una obligación de esta etapa.

El catálogo nuevo contiene 13 fichas de ronda: tres originales aportados/reaportados y {summary['web_catalog_sources']} fuentes web. Hay {summary['web_full_originals_read']} cuerpos originales web leídos: {summary['web_original_body_and_tables_read']} con contenido y tablas inspeccionadas y uno (Hortúa/VIX) cualitativo sin tablas numéricas recuperadas; E1X002 conserva evidencia parcial de código. Estas cifras no significan 12 originales nuevos. E1T3 y E1X001 actualizan fuentes ya identificadas; R02 ya era referencia E2 y sólo añade lectura primaria E1. Las reservas de extracción/eligibilidad se conservan en cada wrapper. R02 y R04 comparten base de datos y no son dos réplicas independientes. DCCNN se mantiene distinta de una TCN simple; VIX, proxies de otros activos y volatilidad realizada cripto se estratifican.

`replacement_map.csv` cubre los 14 casos mediante funciones documentales y limitaciones explícitas: no afirma 14 papers distintos, equivalencia temática completa ni una sustitución uno-a-uno. `sources.csv` separa ruta de acceso, lectura web efectiva y licencia. El adjunto E1N118 sí declara CC BY 4.0; el hecho de aportar un PDF no acredita por sí solo una licencia abierta para los otros dos.

**Se conservan 244 ocurrencias, 182 grupos, N7=7 y las frecuencias históricas.** No se reescriben los estados del corte anterior, que permanecen bajo `previous_cut_adjudication`. Los wrappers mantienen literal cada ficha derivada bajo `source_review`. La evidencia de esta etapa no se incorpora como un nuevo ranking ni se agrupan métricas entre targets, escalas o activos. No hay entrenamiento, nuevas predicciones, consulta nativa IEEE acreditada ni reconstrucción de cadenas históricas. La ampliación es posterior a los experimentos.

El compilador usa entradas explícitas revisadas y fijadas por SHA256, la validación de privacidad ya existente y una lista pública cerrada. `manifest.json` autentica cada archivo aprobado con hash y tamaño; el espejo conserva exactamente esos bytes. No se publican originales, textos completos, resúmenes nativos ni contactos.

Reproducción documental: `python scripts/build_e1_open_access_replacement.py`; verificación sin escritura: `python scripts/build_e1_open_access_replacement.py --verify`. Estas órdenes no construyen el libro, no crean ZIP y no hacen commits.
'''


def flow_svg(summary: dict) -> bytes:
    lines = ["E1 · Enmienda de acceso documental", "Histórico intacto: 244 ocurrencias · 182 grupos · N7=7",
             "Cola nativa congelada: 22 informes", "Corte anterior: 5 contexto/excluidos + 17 pendientes",
             "17 = 3 PDF aportados/reaportados + 14 originales no obtenidos",
             "Corte actual: 8 contexto/excluidos + 14 no obtenidos + 0 seek activos",
             "Lectura nativa: 6 originales únicos + 1 espejo completo; 118 es relectura",
             "Catálogo web: 9 cuerpos leídos = 8 con tablas + 1 sin tablas; 1 sólo parcial",
             "14 casos cubiertos por funciones; no 14 sustituciones uno-a-uno",
             "No acceso ≠ exclusión por calidad · Lectura ≠ inclusión · Sin cambios de experimentos"]
    body = ''.join(f'<text x="450" y="{42+i*44}" text-anchor="middle">{escape(line)}</text>' for i, line in enumerate(lines))
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="900" height="470" viewBox="0 0 900 470" role="img" aria-label="Flujo documental y dimensiones de lectura separadas">'
            '<rect width="900" height="470" fill="#f5f8fc"/><g font-family="sans-serif" font-size="15" fill="#183047">'+body+'</g></svg>\n').encode('utf-8')


def artifacts(stage: Path, frozen_base: Path) -> tuple[dict[str, bytes], dict]:
    raw, frozen, web, uploads = load_inputs(stage, frozen_base)
    reviews = compile_reviews(raw, frozen, web, uploads)
    summary = summarize(reviews, raw)
    generated = {"reviews.json": json_bytes(reviews), "summary.json": json_bytes(summary), "sources.csv": csv_bytes(sources(reviews)),
        "native_dispositions.csv": csv_bytes(native_dispositions(reviews)), "replacement_map.csv": csv_bytes(replacement_map(reviews)),
        "model_evidence.csv": csv_bytes(model_evidence(reviews, frozen)), "README.md": readme(summary).encode("utf-8"), "flow.svg": flow_svg(summary)}
    files = {**raw, **generated}
    for name, data in files.items():
        validate_public_bytes(name, data)
    manifest = {"schema_version": 1, "publication_schema": PUBLICATION_SCHEMA, "approved_public_files": sorted(PUBLIC_FILES),
        "scope": "Reviewed derived metadata only; no original PDFs, full text, native abstracts or contact fields.",
        "reviewed_public_inputs": sorted(INPUT_FILES), "reviewed_input_sha256": REVIEWED_INPUT_SHA256,
        "frozen_historical_sources_sha256": FROZEN_FILES, "compiler": "scripts/build_e1_open_access_replacement.py",
        "files": {name: metadata(files[name]) for name in sorted(files)}}
    return {**files, "manifest.json": json_bytes(manifest)}, summary


def build(stage: Path = STAGE, mirror: Path | None = MIRROR, frozen_base: Path = BASE) -> dict:
    files, summary = artifacts(stage, frozen_base)
    if mirror is not None and mirror.exists() and inventory(mirror) - set(files):
        raise ValueError("Unapproved neighbouring public mirror file")
    for name in OUTPUT_FILES | {"manifest.json"}:
        path = stage / name
        if path.is_symlink():
            raise ValueError("Symlink in output destination")
        path.write_bytes(files[name])
    if mirror is not None:
        # Reject a symlink destination before creating/writing any public file.
        inventory(mirror)
        for name, data in files.items():
            destination = mirror / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    verify(stage, mirror, frozen_base)
    return summary


def verify(stage: Path = STAGE, mirror: Path | None = MIRROR, frozen_base: Path = BASE) -> None:
    files, _ = artifacts(stage, frozen_base)
    if inventory(stage) != set(files):
        raise ValueError("Public stage inventory mismatch")
    for name, data in files.items():
        if checked_file(stage, name) != data:
            raise ValueError("Compiled evidence/hash differs from reviewed derivation: " + name)
    if mirror is not None:
        if inventory(mirror) != set(files):
            raise ValueError("Public mirror inventory mismatch")
        for name, data in files.items():
            if checked_file(mirror, name) != data:
                raise ValueError("Public mirror byte mismatch: " + name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        print("PASS: 22 dispositions, 14+3 reconciliation, reviewed source roles, frozen history, hashes/privacy and exact public mirror")
    else:
        summary = build()
        print(json.dumps({key: summary[key] for key in ("native_queue_size", "native_closed_excluded_context", "native_not_retrieved", "native_active_seek_at_cutoff", "native_originals_read", "web_full_originals_read")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
