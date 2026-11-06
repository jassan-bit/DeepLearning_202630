#!/usr/bin/env python3
"""Compile the E2 review from frozen, curated facts, never from published PDFs.

First ingestion accepts --extraction-dir and --paper-manifest. It discards private
paths, source excerpts and raw text before freezing source_inputs.json. Subsequent
runs require only the public frozen inputs and use no network or model training.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "references/literature_e2/fulltext_20261105"
DEFAULT_STATIC = REPO_ROOT / "book_e2/_static/literature_fulltext"
GROUPS = ("mixers.json", "time_patch.json", "informer_autoformer.json",
          "timesnet_smamba.json", "i_cross.json")
ORDER = ("TSMixer", "TiDE", "TimeMixer", "PatchTST", "Informer", "Autoformer",
         "TimesNet", "S-Mamba", "iTransformer", "Crossformer")
# The row identities are deliberate review choices, not an optimization/ranking.
# Pinning them prevents a different variant, an averaged horizon or a copied
# baseline column from silently becoming an original-paper result.
ROWS = {
    "TSMixer": ("0.361", "0.392", 96, 512, "3", 10),
    "TiDE": ("0.375", "0.398", 96, 720, "2", 8),
    "TimeMixer": ("0.375", "0.400", 96, 96, "13", 17),
    "PatchTST": ("0.370", "0.400", 96, 512, "3", 7),
    "Informer": ("0.577", "0.549", 24, None, "2", 6),
    "Autoformer": ("0.384", "0.425", 24, 96, "5", 13),
    "TimesNet": ("0.384", "0.402", 96, 96, "13", 21),
    "S-Mamba": ("0.386", "0.405", 96, 96, "3", 7),
    "iTransformer": ("0.386", "0.405", 96, 96, "10", 23),
    "Crossformer": ("0.305", "0.367", 24, None, "1", 7),
}
VARIANTS = {
    "TSMixer": ("chen_tsmixer_basic_ltsf", "TSMixer básico multivariante"),
    "TiDE": ("tide_full_revin", "TiDE, Table2; ETTh1 settings Table8 including RevIN=True, all residuals present; NOT TiDE(no res.)"),
    "TimeMixer": ("timemixer_unified_m3_four_scales", "TimeMixer oficial, configuración unificada"),
    "PatchTST": ("patchtst64_supervised_p16_s8", "PatchTST/64 supervisado"),
    "Informer": ("informer_probsparse_distilling", "Informer con ProbSparse y distilling; columna Informer, no Informer†."),
    "Autoformer": ("autoformer_speedup", "Autoformer speedup (todos los resultados del paper), 2 encoder + 1 decoder, d_model512, 8 heads, moving average25; c exacto por fila NReportado."),
    "TimesNet": ("timesnet_shared_inception", "TimesNet (Ours), backbone Inception compartido final"),
    "S-Mamba": ("s_mamba_bidirectional_vc_ffn_td", "Simple-Mamba (S-Mamba): bi-Mamba VC + FFN TD; columna S-Mamba principal"),
    "iTransformer": ("itransformer_original_full", "Original complete iTransformer backbone, iTransformer (Ours) column"),
    "Crossformer": ("crossformer_full_dsw_tsa_hed_no_covariates", "Original full Crossformer (DSW+TSA+HED) without covariates"),
}
DROP_KEY = re.compile(r"private|excerpt|verbatim|quot(?:e|ation)|raw.?text|full.?text_content|abstract_text|"
                      r"first_page|local_pdf|text_path|text_filename|user_file|file_id", re.I)
PRIVATE_STRING = re.compile(r"(?:/workspace/|/tmp/|/mnt/data/|sediment://|file[_-][A-Za-z0-9]{8})")
STATUS = {
    "reported": "Located in the examined complete PDF; scope and locator retained.",
    "partial": "Some reporting exists, but does not establish the selected row or every condition.",
    "NR": "Not reported after complete reading of this exact PDF, including appendices; never zero.",
    "NV": "Not independently verified; outside this documentary review or another source is required.",
    "proposed": "Analyst adaptation, not an original-paper experiment or a completed E2 run.",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sanitize(value):
    """Keep paraphrased facts/metrics/locators, exclude private and raw content."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if DROP_KEY.search(key):
                continue
            if key.lower() in {"quote", "quotes", "quotation", "quotations", "abstract", "ab"}:
                continue
            if isinstance(item, str) and PRIVATE_STRING.search(item):
                continue
            result[key] = sanitize(item)
        return result
    if isinstance(value, list):
        return [sanitize(x) for x in value if not (isinstance(x, str) and PRIVATE_STRING.search(x))]
    return value


def validate_public(value):
    serialized = json.dumps(value, ensure_ascii=False)
    if PRIVATE_STRING.search(serialized):
        raise ValueError("Public facts contain a private path or uploaded-file identifier")
    if isinstance(value, dict):
        for key, item in value.items():
            if DROP_KEY.search(key) or key.lower() in {"abstract", "ab"}:
                raise ValueError(f"Public facts contain excluded source field {key}")
            validate_public(item)
    elif isinstance(value, list):
        for item in value:
            validate_public(item)


def validate_pages(value, total, trail=""):
    """Validate physical PDF pages only; printed page labels are a separate axis."""
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{trail}.{key}"
            if key in {"pdf_page", "pdf_pages", "read_pdf_pages", "reviewed_pdf_pages",
                       "pdf_pages_read", "rendered_and_inspected_pdf_pages", "pdf_page_range",
                       "scope_checked_pdf_pages"}:
                pages = item if isinstance(item, list) else [item]
                for page in pages:
                    if isinstance(page, int) and not 1 <= page <= total:
                        raise ValueError(f"PDF page {page} outside 1..{total}: {here}")
            validate_pages(item, total, here)
    elif isinstance(value, list):
        for item in value:
            validate_pages(item, total, trail)


def validate_pdf_identity(value, receipt):
    """Check PDF identities in every heterogeneous group schema, excluding code/text hashes."""
    if isinstance(value, dict):
        filename = first(value, "filename", "pdf_filename", "version_filename")
        is_pdf_record = isinstance(filename, str) and filename.endswith(".pdf")
        for key in (("pdf_sha256", "sha256") if is_pdf_record else ("pdf_sha256",)):
            if key in value and value[key] != receipt["sha256"]:
                raise ValueError(f"PDF hash differs from receipt: {receipt['model']}")
        if is_pdf_record:
            if filename != receipt["filename"]:
                raise ValueError(f"PDF filename differs from receipt: {receipt['model']}")
            for key in ("bytes", "pdf_pages", "pdf_page_count", "actual_pdf_pages", "manifest_pdf_pages"):
                if key in value and value[key] != receipt["bytes" if key == "bytes" else "pdf_pages"]:
                    raise ValueError(f"PDF {key} differs from receipt: {receipt['model']}")
        for item in value.values():
            validate_pdf_identity(item, receipt)
    elif isinstance(value, list):
        for item in value:
            validate_pdf_identity(item, receipt)


def make_manifest(private_manifest):
    papers = []
    seen = set()
    source = private_manifest.get("papers", []) if isinstance(private_manifest, dict) else private_manifest
    for paper in source:
        name = paper["model"]
        if name in seen:
            raise ValueError(f"Duplicate paper receipt: {name}")
        seen.add(name)
        filename = paper["filename"]
        if Path(filename).name != filename or not filename.endswith(".pdf"):
            raise ValueError("Receipt filename must be a plain PDF basename")
        sha = paper["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", sha):
            raise ValueError(f"Invalid SHA256 for {name}")
        byte_count, page_count = paper["bytes"], paper["pdf_pages"]
        if not isinstance(byte_count, int) or byte_count <= 0 or not isinstance(page_count, int) or page_count <= 0:
            raise ValueError(f"Invalid byte/page count for {name}")
        # A private ingestion validates the receipt against the actual binary. A
        # public rerun uses the immutable receipt, with no distributed source PDF.
        private_path = paper.get("path")
        if private_path:
            content = Path(private_path).read_bytes()
            if len(content) != byte_count or digest(content) != sha:
                raise ValueError(f"PDF hash or byte size mismatch for {name}")
        if name == "Crossformer":
            url, version = "https://openreview.net/forum?id=vSVLM2j9eie", "ICLR 2023 conference PDF"
        else:
            version = filename.removesuffix(".pdf")
            if not re.fullmatch(r"\d{4}\.\d{4,5}v\d+", version):
                raise ValueError(f"Missing exact arXiv version for {name}")
            url = "https://arxiv.org/abs/" + version
        papers.append({"model": name, "filename": filename, "sha256": sha,
                       "bytes": byte_count, "pdf_pages": page_count,
                       "page_count_method": "pdfinfo; physical PDF pages, not formfeed blocks",
                       "source_url": url, "source_version": version,
                       "receipt_kind": "user_provided",
                       "publication_scope": "Receipt and curated facts only; source PDF not redistributed"})
    if seen != set(ORDER):
        raise ValueError("Paper manifest must contain exactly the ten distinct reviewed models")
    return {"schema_version": "e2.fulltext.receipts.v1", "papers": sorted(papers, key=lambda x: ORDER.index(x["model"])),
            "paper_count": len(papers), "physical_page_total": sum(x["pdf_pages"] for x in papers),
            "scope": "User-provided exact-version PDF receipts; no PDF binaries or extracted source text included"}


def code_receipts(registry):
    rows = []
    for model in registry["models"]:
        if model["name"] not in ORDER:
            continue
        rows.append({"model": model["name"], "repository": model["repo"],
                     "code_source": model["code_source"], "readme_source": model["readme_source"],
                     "license": model["license"],
                     "evidence_stage": "historical_author_code_retrieval_before_PDF_review",
                     "paper_full_text_retrieved_at_that_stage": False,
                     "scope": "Immutable author code/README evidence; availability is not successful reproduction"})
    if {x["model"] for x in rows} != set(ORDER) or len(rows) != 10:
        raise ValueError("Author code receipts must cover ten distinct models")
    return sanitize(rows)


def ingest(extraction_dir, paper_manifest, author_registry):
    groups = []
    for filename in GROUPS:
        data = read_json(Path(extraction_dir) / filename)
        groups.append({"filename": filename, "facts": sanitize(data)})
    frozen = {"schema_version": "e2.fulltext.curated_inputs.v1",
              "scope": "Curated paraphrased facts, original numeric results and locators; no PDF or source text",
              "groups": groups, "author_code_receipts": code_receipts(read_json(author_registry))}
    validate_public(frozen)
    return frozen, make_manifest(read_json(paper_manifest))


def original_pair(model):
    for key in ("canonical_metric_pair", "exact_metric_pair"):
        if key in model:
            return model[key]
    for key in ("verified_metric_pairs", "numeric_evidence", "exact_metric_pairs"):
        if key in model:
            # All group extractors deliberately put their visually checked,
            # original-model ETTh1 row first. ROWS validates the complete identity.
            return model[key][0]
    raise ValueError(f"No canonical original metric pair for {model['model']}")


def first(data, *keys, default=None):
    for key in keys:
        if key in data:
            return data[key]
    return default


def metric_string(pair, metric):
    explicit = pair.get("decimal_strings", {}).get(metric, pair.get("exact_printed_" + metric))
    value = first(pair, metric, metric.lower())
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Missing original numeric {metric}; NR/NV must never become zero")
    result = explicit or f"{value:.3f}"
    if float(result) != value:
        raise ValueError(f"Printed {metric} and extracted value disagree")
    return result


def normalize_pair(model, receipt):
    pair = original_pair(model)
    locators = pair.get("locators") or [first(pair, "locator", "source")]
    locator = locators[0]
    page = first(locator, "pdf_page", default=(locator.get("pdf_pages") or [None])[0])
    table = str(first(pair, "table", default=first(locator, "table", "printed_table", "section_or_table", default="")))
    match = re.search(r"(?:Table\s*)?(\d+)", table)
    table = match.group(1) if match else table
    context = first(pair, "lookback", "input_length", "context", "context_L", "L")
    horizon = first(pair, "horizon", "output_horizon", "horizon_H", "H")
    variant = first(pair, "model_variant", "variant", default="")
    same_row = first(pair, "same_row_pair", "same_row", "same_row_verified")
    if same_row is None and pair.get("full_row"):
        own_cells = [c for c in pair["full_row"] if c.get("model") == model["model"]]
        same_row = len(own_cells) == 1 and all(own_cells[0].get(k) == pair.get(k) for k in ("MSE", "MAE"))
    mse, mae = metric_string(pair, "MSE"), metric_string(pair, "MAE")
    actual = (mse, mae, horizon, context, table, page)
    if pair.get("dataset") != "ETTh1" or actual != ROWS[model["model"]] or same_row is not True:
        raise ValueError(f"Canonical row/variant/horizon/context mismatch for {model['model']}: {actual}")
    variant_id, expected_variant = VARIANTS[model["model"]]
    if variant != expected_variant:
        raise ValueError(f"Canonical variant mismatch for {model['model']}: {variant}")
    if model["model"] == "PatchTST" and (pair.get("patch_length"), pair.get("stride")) != (16, 8):
        raise ValueError("PatchTST/64 canonical pair requires P16/S8")
    if model["model"] == "TimeMixer" and (pair.get("M"), pair.get("scale_lengths")) != (3, [96, 48, 24, 12]):
        raise ValueError("TimeMixer unified pair requires four scales with M=3")
    validate_pages(pair, receipt["pdf_pages"])
    scaling = first(pair, "scaling", "normalization", "scale_status", "metric_scale", default="NR")
    return {"dataset": "ETTh1", "task": "multivariate forecasting of the seven ETTh1 variables",
            "variant": variant, "variant_id": variant_id, "horizon_steps": horizon, "context_steps": context,
            "context_status": "reported" if context is not None else "NR",
            "context_detail": first(pair, "L_status", "input_length_status", default="Selected context not reported" if context is None else "Fixed context of selected row"),
            "MSE": mse, "MAE": mae, "same_row_pair": True,
            "table": table, "pdf_page": page, "printed_page": locator.get("printed_page", "NR"),
            "source_version": receipt["source_version"], "source_url": receipt["source_url"],
            "source_sha256": receipt["sha256"], "scaling_status": scaling,
            "visual_verified": True, "locators": locators,
            "scope": "Original-paper external result; not a BTC metric, reproduced run, or cross-paper ranking"}


def fact(status, value, evidence, note=""):
    if status not in STATUS:
        raise ValueError(status)
    return {"status": status, "value": value, "evidence": evidence, "note": note}


def normalize_quality(model):
    """Uniform selected-row reporting with source dictionaries kept as evidence."""
    name = model["model"]
    protocol = first(model, "experimental_protocol", "protocol", "data_and_protocol", "experiments", "original_forecasting_protocol")
    if name == "TSMixer":
        split, scale = protocol["splits"], protocol["scaling"]
        random = protocol["randomness_and_statistics"]
        repetitions, seed, variance = ("NR", None), ("NR", None), ("NR", None)
        tests, fairness = random["statistical_tests"], protocol["fairness_and_baselines"]
    elif name == "TiDE":
        split, scale = protocol["split"], protocol["scaling"]
        random = protocol["repetitions_and_uncertainty"]
        repetitions, seed = ("reported", 5), ("NR", None)
        variance = ("reported", {"kind": "standard_error", "MSE": "0.0003", "MAE": "0.0002"})
        tests, fairness = random["formal_tests"], protocol["fairness"]
    elif name in {"TimeMixer", "PatchTST"}:
        split = protocol["split_and_scaling"]
        scale = protocol["split_and_scaling"]
        random, fairness = protocol["repetitions_uncertainty"], protocol["baseline_fairness"]
        repetitions = ("reported", 3 if name == "TimeMixer" else 1)
        seed = ("NR", None) if name == "TimeMixer" else ("reported", [2021])
        variance = ("NR", None)
        tests = random.get("statistical_report", random.get("formal_statistical_tests"))
    elif name in {"Informer", "Autoformer"}:
        split, scale = protocol["datasets"], protocol["scaling"]
        random, fairness = protocol["repetitions_and_statistics"], protocol["search_and_equity"]
        repetitions = ("reported", 5 if name == "Informer" else 3)
        seed, variance = ("NR", None), ("NR", None)
        tests = random["statistical_tests"]
    elif name in {"TimesNet", "S-Mamba"}:
        split, scale = protocol["splits"], protocol["scaling"]
        random = {key: protocol[key] for key in ("repetitions", "seed_values", "confidence_intervals", "significance_tests")}
        variance = ("NR", None)
        repetitions = ("reported", 3) if name == "TimesNet" else ("NR", None)
        seed = ("NR", None)
        tests = protocol["significance_tests"]
        fairness = protocol.get("HPO_equality", "NR")
    else:
        split, scale = {"split": protocol["split"], "datasets": protocol["datasets"]}, protocol["scaling"]
        random, fairness = protocol["randomness"], protocol["hpo_equity"]
        repetitions = ("NV", None) if name == "iTransformer" else ("reported", 5)
        seed, variance = ("NR", None), ("NR", None)
        tests = random["statistical_tests"]
    split_status = "NR" if name in {"PatchTST", "S-Mamba"} else "reported"
    scaling_status = "NR" if name in {"Autoformer", "TimeMixer"} else "partial" if name in {"PatchTST", "TimesNet", "S-Mamba", "iTransformer"} else "reported"
    train_scaler = "reported" if name in {"TiDE", "Crossformer"} else "NR"
    return {
        "scope": "Selected canonical ETTh1 row unless a different scope is explicit in evidence",
        "split": fact(split_status, split if split_status != "NR" else None, split),
        "scaling": fact(scaling_status, scale if scaling_status != "NR" else None, scale),
        "train_only_scaler": fact(train_scaler, True if train_scaler == "reported" else None, scale,
                                  "Internal model normalization does not establish train-only dataset scaling."),
        "repetitions": fact(*repetitions, random, "iTransformer reports five robustness runs for six other datasets; selected ETTh1 repetitions not independently established." if name == "iTransformer" else "Main selected variant; robustness studies can use a different variant or scope."),
        "seed_ids": fact(*seed, random),
        "variance_selected_pair": fact(*variance, random, "SD for another variant, dataset, or horizon aggregate is not uncertainty for this selected row."),
        "formal_tests": fact("partial" if name == "TimeMixer" else "NR", None, tests,
                             "TimeMixer prints significance/CI claims but no named test, p-values or interval endpoints." if name == "TimeMixer" else "No verified formal inferential comparison for the selected row."),
        "baseline_effort": fact("partial" if name != "S-Mamba" else "NR", fairness if name != "S-Mamba" else None, fairness),
        "equal_hpo_budget": fact("NR", None, fairness, "Shared context or architecture controls do not prove equal search trials/compute."),
        "benchmark_data_download_validated": fact("NV", None, "This task inspected documents, not downloaded benchmark datasets."),
        "original_paper_numerically_reproduced": fact("NV", None, "No model fitting or inference was performed in this documentary review."),
    }


def normalize_efficiency(model):
    details = first(model, "computational_cost", "compute_evidence", "efficiency", "reported_efficiency", "resource_costs")
    # Preserve full condition-bearing dictionaries. In particular a runtime for
    # imputation or M5 must never be relabelled as ETTh1 forecasting/BTC runtime.
    runtime = first(details, "wall_clock", "training_and_inference_time", "reported_memory_time", "reported_runtime", "runtime", "reported_training_time_memory", "reported_time_memory", "reported_measured", default="NR")
    return {"parameters": first(details, "parameters", "parameter_count", "parameters_LTSF", default="NR"),
            "parameter_count_selected_forecast": {"status": "NR", "value": None,
                "note": "No numeric parameter count is reported for this canonical ETTh1 forecasting configuration. Counts for M5 and parameter sizes in MB for imputation have separate scopes in the source evidence."},
            "FLOPs": first(details, "FLOPs", "flops", default="NR"),
            "asymptotic": first(details, "asymptotic", "asymptotic_reported", "asymptotic_complexity", "complexity", "reported_attention_complexity", "reported_time_space_complexity", "full_model_asymptotic_formula", default="NR"),
            "runtime": runtime,
            "memory": first(details, "GPU_memory", "reported_memory_time", "reported_training_time_memory", "reported_time_memory", "memory", "peak_memory_numeric", default="NR"),
            "reported_measurements_and_conditions": details,
            "BTC_measured_parameters_FLOPs_time": {"status": "NV", "value": None,
                "note": "No D7×5/BTC model instantiated or benchmarked; source-paper costs do not transfer."}}


def compile_models(frozen, manifest):
    validate_public(frozen)
    validate_public(manifest)
    receipts = {p["model"]: p for p in manifest["papers"]}
    if len(manifest["papers"]) != 10 or set(receipts) != set(ORDER):
        raise ValueError("Exactly ten distinct paper receipts required")
    models = {}
    for group in frozen["groups"]:
        facts = group["facts"]
        for model in facts.get("models", facts.get("papers", [])):
            name = model["model"]
            if name in models:
                raise ValueError(f"Duplicate model extraction: {name}")
            if name not in ORDER or model.get("verified_fulltext") is not True:
                raise ValueError(f"Incomplete/unrecognized fulltext extraction: {name}")
            receipt = receipts[name]
            validate_pages(model, receipt["pdf_pages"], name)
            validate_pdf_identity(model, receipt)
            identity = first(model, "identity", "paper", "bibliography", default=model)
            models[name] = {"model": name, "fulltext_verified": True,
                "source_version": receipt["source_version"], "source_receipt": receipt,
                "title": first(identity, "title", "full_title"), "authors": identity.get("authors"),
                "publication_year": first(identity, "year", "publication_year", "initial_preprint_year"),
                "version_description": first(model, "version", default=identity.get("version", receipt["source_version"])),
                "canonical_metric_pair": normalize_pair(model, receipt),
                "quality": normalize_quality(model), "efficiency": normalize_efficiency(model),
                "extraction_group": group["filename"], "extraction": copy.deepcopy(model)}
            if name == "TimesNet":
                models[name]["review_notes"] = [{
                    "evidence_type": "separate_author_code_audit_and_proposed_E2_adapter_guard",
                    "source": "https://raw.githubusercontent.com/thuml/Time-Series-Library/4e938a1767106324dd753b2a44832bf870a0252e/models/TimesNet.py",
                    "lines": [9, 18],
                    "observation": "FFT_for_Period averages spectral magnitudes over batch and channels before selecting shared frequency indices, masks DC with zero, then uses integer division T//frequency.",
                    "interpretation": "If an inference batch combines different forecast origins, shared selection can depend on histories unavailable at earlier origins. This is an adapter risk identified from this code version, not a claim about leakage in published experiments.",
                    "proposed_guard": "Use inference batch size 1 or independently select periods per origin/window; restrict top-k to positive frequency indices and k<=floor(L_eff/2). A zero-valued DC mask alone does not exclude DC when all amplitudes are zero.",
                    "status": "proposed; no model executed or trained"}]
    if set(models) != set(ORDER):
        raise ValueError("Exactly ten distinct fulltext-reviewed models required")
    codes = frozen["author_code_receipts"]
    if len(codes) != 10 or {c["model"] for c in codes} != set(ORDER):
        raise ValueError("Exactly ten distinct author-code receipts required")
    return {"schema_version": "e2.fulltext.review.v1", "administrative_date": "2026-11-05",
            "status_vocabulary": STATUS,
            "scope": "Complete original PDFs and appendices, paraphrased facts and original metrics; separate historical author-code evidence. No new training, backtest, BTC ranking or reproduction.",
            "counts": {"fulltext_verified_current": len(models), "source_PDF_receipts": len(receipts),
                "physical_PDF_pages": sum(p["pdf_pages"] for p in receipts.values()),
                "author_code_available_from_immutable_prior_receipts": len(codes),
                "fulltext_retrieved_in_historical_code_only_stage": 0,
                "models_with_benchmark_descriptions": len(models),
                "benchmark_downloads_validated_this_review": 0,
                "new_model_fits_this_review": 0, "original_papers_numerically_reproduced_this_review": 0,
                "new_BTC_results_this_review": 0},
            "author_code_evidence": {"stage": "historical; predates current fulltext review", "receipts": codes},
            "models": [models[name] for name in ORDER]}


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else "NR" if value is None else value for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def report(review):
    lines = ["# E2: revisión de diez originales completos", "",
        "Se leyeron diez PDFs completos, incluidas referencias y apéndices: 215 páginas físicas. Los recibos fijan versión, SHA256 y tamaño. Las fuentes completas y el texto extraído no se redistribuyen.", "",
        "La disponibilidad de diez implementaciones de autor se documentó antes en un registro de código. Su estado histórico de cero PDFs pertenece a esa etapa. El estado actual es diez originales completos verificados. En esta revisión hubo cero ajustes de modelos, cero reproducciones numéricas, cero resultados BTC y cero descargas de benchmarks validadas. Describir un dataset en un artículo no valida una descarga.", "",
        "Cada par siguiente pertenece a una misma fila y variante. Hay diferencias de horizonte, contexto, split y escalado: estas cifras externas no ordenan modelos para BTC σ7 ni para D7×5,h1.", "",
        "| Modelo / variante | Dataset | H | L | MSE | MAE | Tabla; página PDF / impresa | Versión |",
        "|---|---|---:|---:|---:|---:|---|---|"]
    for m in review["models"]:
        p = m["canonical_metric_pair"]
        lines.append("| " + " | ".join(cell(x) for x in (m["model"] + ": " + p["variant"], p["dataset"], p["horizon_steps"], p["context_steps"] if p["context_steps"] is not None else "NR", p["MSE"], p["MAE"], f"{p['table']}; {p['pdf_page']} / {p['printed_page']}", p["source_version"])) + " |")
    lines += ["", "## Calidad de reporte para la fila seleccionada", "",
        "NR significa no reportado tras leer la versión completa; no equivale a cero. NV significa no verificado. Una DE de otros horizontes, datasets o variantes no se traslada a la fila seleccionada. Los campos completos y sus evidencias están en models.json y quality_matrix.csv.", "",
        "| Modelo | Split | Escalado | Scaler train-only | Repeticiones | Semillas | Incertidumbre de esta fila | Test formal | Presupuesto HPO igual |",
        "|---|---|---|---|---|---|---|---|---|"]
    for m in review["models"]:
        q = m["quality"]
        values = [m["model"]]
        for field in ("split", "scaling", "train_only_scaler", "repetitions", "seed_ids", "variance_selected_pair", "formal_tests", "equal_hpo_budget"):
            f = q[field]
            values.append(f["status"] + (": " + json.dumps(f["value"], ensure_ascii=False) if field in {"repetitions", "seed_ids"} and f["value"] is not None else ""))
        lines.append("| " + " | ".join(cell(x) for x in values) + " |")
    lines += ["", "## Controles críticos de interpretación", "",
        "- Informer y Crossformer: el original ofrece ETTh1 H24; no atribuirle ETTh1 H96. El contexto ganador de esa fila es NR.",
        "- Autoformer: ETTh1 H24 aparece en Tabla 5, PDF p13; ETT* H96 del cuerpo es ETTm2. Su operador Roll es circular.",
        "- TimeMixer: el par canónico es la configuración unificada L96, M=3, cuatro escalas; la búsqueda optimizada es otra condición. La Tabla 6 imprime 15min para ETTh1/h2: conservar la inconsistencia para verificarla, sin corregirla por costumbre.",
        "- PatchTST: /64 supervisado L512/P16/S8 se distingue de /42, pretraining, fine-tuning y linear probing.",
        "- TiDE: el artículo declara 7:1:2 para todos los datasets long-term, incluido ETT; no sustituirlo por otro split conocido. Sus errores estándar no son desviaciones estándar.",
        "- TSMixer: Table 2 imprime 7:2:1 para Weather/Electricity/Traffic; Table 8 contiene rangos/columnas incongruentes. Son anomalías de reporte observadas; no se infiere una conducta impropia.",
        "- TimesNet: la fórmula del paper usa ceil(T/f), mientras el código inspeccionado usa floor(T/f); mantener distintas las fuentes. El código selecciona periodos compartidos por batch: para combinar distintos orígenes, la adaptación propuesta exige inferencia B1 o selección independiente por origen, con top-k sólo sobre frecuencias positivas y k≤floor(L_eff/2). Es un riesgo del adaptador, no una acusación sobre experimentos publicados. Costes de imputation no son costes del forecast seleccionado.",
        "- S-Mamba: versión arXiv v3 de 2024, 14 páginas físicas; no es el cuerpo editorial de 2025. Optimizador, splits y semillas son NR. Costes de Fig5 llevan su propio batch/hardware/tarea.",
        "- iTransformer: cinco runs de robustez en Table 5 cubren otros seis datasets; no documentan repeticiones ni dispersión de la fila ETTh1 elegida.",
        "", "## Eficiencia y reproducción", "",
        "efficiency_matrix.csv y models.json conservan unidades, hardware, batch, longitudes y tareas. No se convierten MB en cantidades de pesos, barras sin etiquetas en tiempos exactos, ni costes de M5/imputation en costes BTC. FLOPs no reportados permanecen NR. Las fórmulas del paper, las derivaciones del analista y el código se identifican como fuentes distintas.", "",
        "Los datos públicos congelados permiten regenerar este paquete sin acceso a adjuntos privados: `python scripts/compile_e2_fulltext_review.py --check`. provenance.json y SHA256SUMS registran los hashes de las entradas y salidas públicas.", ""]
    return "\n".join(lines).encode("utf-8")


def build_outputs(frozen, manifest):
    review = compile_models(frozen, manifest)
    metrics, quality, efficiency = [], [], []
    for m in review["models"]:
        pair = m["canonical_metric_pair"]
        metrics.append({"model": m["model"], **{k: pair[k] for k in ("dataset", "task", "variant", "variant_id", "horizon_steps", "context_steps", "context_status", "MSE", "MAE", "same_row_pair", "table", "pdf_page", "printed_page", "source_version", "source_url", "source_sha256", "scaling_status")}})
        quality.append({"model": m["model"], "source_version": m["source_version"], **{k: v for k, v in m["quality"].items() if k != "scope"}})
        efficiency.append({"model": m["model"], "source_version": m["source_version"], **m["efficiency"]})
    outputs = {"source_inputs.json": json_bytes(frozen), "fulltext_manifest.json": json_bytes(manifest),
               "models.json": json_bytes(review), "metrics.csv": csv_bytes(metrics, list(metrics[0])),
               "quality_matrix.csv": csv_bytes(quality, list(quality[0])),
               "efficiency_matrix.csv": csv_bytes(efficiency, list(efficiency[0])), "report.md": report(review)}
    provenance = {"schema_version": "e2.fulltext.provenance.v1", "compiler": "scripts/compile_e2_fulltext_review.py",
                  "compiler_sha256": digest(Path(__file__).read_bytes()),
                  "method": "Deterministic compilation of curated frozen facts and immutable PDF receipts; no PDF binaries distributed",
                  "input_hashes": {name: digest(outputs[name]) for name in ("source_inputs.json", "fulltext_manifest.json")},
                  "output_hashes": {name: digest(content) for name, content in sorted(outputs.items())},
                  "counts": review["counts"], "private_inputs_required_for_public_rerun": False}
    outputs["provenance.json"] = json_bytes(provenance)
    outputs["SHA256SUMS"] = "".join(f"{digest(content)}  {name}\n" for name, content in sorted(outputs.items())).encode("utf-8")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction-dir", type=Path, help="Private first-ingestion group JSON directory")
    parser.add_argument("--paper-manifest", type=Path, help="Private first-ingestion immutable PDF manifest")
    parser.add_argument("--author-code-registry", type=Path, default=REPO_ROOT / "references/literature_e2/top10_20261105/primary_source_registry.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC)
    parser.add_argument("--check", action="store_true", help="Compare frozen regeneration with both public copies, without writing")
    args = parser.parse_args(argv)
    if bool(args.extraction_dir) != bool(args.paper_manifest):
        parser.error("--extraction-dir and --paper-manifest must be provided together")
    if args.extraction_dir:
        frozen, manifest = ingest(args.extraction_dir, args.paper_manifest, args.author_code_registry)
    else:
        frozen = read_json(args.output_dir / "source_inputs.json")
        manifest = make_manifest(read_json(args.output_dir / "fulltext_manifest.json"))
    outputs = build_outputs(frozen, manifest)
    for directory in (args.output_dir, args.static_dir):
        if not args.check:
            directory.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            target = directory / name
            if args.check:
                if not target.exists() or target.read_bytes() != content:
                    raise ValueError(f"Public frozen output differs: {target}")
            else:
                target.write_bytes(content)
    print(f"{'Verified' if args.check else 'Compiled'} 10 fulltext reviews, 215 physical pages, 0 model fits; {len(outputs)} public artifacts in each destination.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(f"Fulltext review validation failed: {exc}", file=sys.stderr)
        sys.exit(1)
