#!/usr/bin/env python3
"""Compile a new cumulative E1 literature cut without rewriting earlier cuts.

Consumes reviewed public metadata, never article payloads or market data.
The new search was purposively targeted by architecture; its requested quotas
are not estimates of architecture prevalence.
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

import build_e1_homogeneous_synthesis as previous

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "references/literature_e1/homogeneous_synthesis_20261107"
EXPANSION = ROOT / "references/literature_e1/free_model_expansion_20261107"
STAGE = ROOT / "references/literature_e1/homogeneous_expansion_20261107"
MIRROR = ROOT / "book/_static/literature_e1/homogeneous_expansion_20261107"
PRIORITY = ["LSTM", "MLP", "GRU", "TCN", "RNN_simple", "CNN_generic", "Transformer_encoder"]
CONTEXTUAL_NEURAL = {"RBFN", "GFN", "MFN", "ANFIS", "CANFIS", "PNN"}
HISTORICAL_SOURCE_COMMIT = "b21279284998baab32e5326356ada3e17ce6b0f4"
FROZEN_HISTORICAL_INPUTS = {
    "references/literature_e1/homogeneous_synthesis_20261107/part_corpus/adjudications.json": "82d97124b7e829570cd1804b9be8f4f5d2edb57d080a9a4afdd2b8efda5d4c7f",
    "references/literature_e1/homogeneous_synthesis_20261107/part_corpus/protocol.json": "3ea1491a1b29e0fc712e0ccbc5c4f174b5f4449f12306fad7bc335bcdcac7fc6",
    "references/literature_e1/homogeneous_synthesis_20261107/part_mechanisms/mechanisms.json": "2a5b22c0fb1035b834389825dd7b4f173c61587f5138a9af44a636003710fa9f",
    "scripts/build_e1_homogeneous_synthesis.py": "5ab4cdc12f87a4e9d28c202663e249f1d1780a3315c1fc8a53df8523fed86e71"
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def csv_encoded(fields, rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def read(path):
    return json.loads(path.read_text())


def normalized_doi(value):
    return str(value or "").strip().lower().removeprefix("https://doi.org/")


def merge_reviewed(historical, evidence):
    """Add only ten new works, keeping all historical records byte-equivalent."""
    old = historical["adjudications"]
    merged = copy.deepcopy(old)
    by_key = {r["identity"]["canonical_key"]: r for r in old}
    by_doi = {normalized_doi(r["doi"]): r for r in old if r["doi"]}
    old_ids = {r["id"] for r in old}
    selected = evidence["records"]
    if len(selected) != 13 or evidence["selected_count"] != 13:
        raise ValueError("This reviewed campaign must contain thirteen selected works")
    seen, aliases, additions = set(), [], []
    for record in selected:
        doi = normalized_doi(record["doi"])
        key = record["identity"]["canonical_key"]
        if not doi or key != "doi:" + doi or key in seen:
            raise ValueError("Selected work identity missing or duplicated")
        seen.add(key)
        existing = by_key.get(key) or by_doi.get(doi)
        if existing:
            if existing["id"] != record["id"]:
                raise ValueError("Existing work must retain its historical identifier")
            aliases.append({"selected_id": record["id"], "historical_id": existing["id"], "doi": doi})
            continue
        if record["id"] in old_ids:
            raise ValueError("New DOI reuses a historical identifier")
        additions.append(copy.deepcopy(record))
    if len(aliases) != 3 or len(additions) != 10:
        raise ValueError("Thirteen selected works are three existing and ten new, not thirteen additions")
    if evidence["existing_count"] != 3 or evidence["new_count"] != 10:
        raise ValueError("Input new/existing summary disagrees")
    merged.extend(additions)
    if len(merged) != 57 or merged[:len(old)] != old:
        raise ValueError("Historical ledger was modified or cumulative count is wrong")
    return merged, aliases, additions


def member(rows, name):
    if name == "financial_transfer_primary_all":
        return [r for r in rows if r["stratum"].startswith("transfer_")
                and r["decision"]["primary_document_read"]
                and r["identity"]["independent_base_experiment"]]
    if name == "financial_transfer_reference_extractable":
        return [r for r in member(rows, "financial_transfer_primary_all")
                if r["decision"]["baseline_role_verified"] and r["independentfamilies"]]
    if name.startswith("transfer_"):
        return [r for r in rows if r["stratum"] == name and r["decision"]["primary_document_read"]]
    return previous.membership(rows, name)


def build_frequencies(rows, old_frequencies):
    families = sorted(set().union(*(set(f["family_counts"]) for f in old_frequencies),
                                  *(set(r["independentfamilies"] or []) for r in rows)))
    definitions = {f["id"]: f["description"] for f in old_frequencies}
    excluded_reporting = sorted(r["id"] for r in rows if r["decision"]["frequency_eligible"] and not r["decision"]["guide31_reporting_sensitivity_eligible"])
    definitions["crypto_construct_reference_reporting_sensitivity"] = "Estrato actual elegible por suficiencia de descripción de referencia/protocolo (§3.1); exclusiones simétricas por reporte insuficiente: " + ", ".join(excluded_reporting) + ". No certifica ausencia de fuga ni calidad global."
    for row in rows:
        if row["decision"]["continuous_crypto_construct_corpus"]:
            definitions.setdefault("crypto_target_" + row["target"]["quantity"], "Estrato por magnitud objetivo operacional")
        if row["stratum"].startswith("transfer_"):
            definitions.setdefault(row["stratum"], "Transferencia financiera, sin sumarla al corpus cripto")
    definitions["financial_transfer_primary_all"] = "Textos primarios de transferencia financiera con experimentos independientes; no corpus cripto"
    definitions["financial_transfer_reference_extractable"] = "Transferencia financiera con al menos una referencia neuronal autónoma verificada"
    output = []
    for name, description in definitions.items():
        selected = member(rows, name)
        groups = [r["identity"]["dependence_group"] for r in selected]
        if len(groups) != len(set(groups)):
            raise ValueError("Dependent experiment counted twice in " + name)
        values = {}
        for family in families:
            positive = sorted(r["id"] for r in selected if r["decision"]["baseline_role_verified"]
                              and family in (r["independentfamilies"] or []))
            values[family] = {"count": len(positive), "denominator": len(selected), "experiments": positive}
        output.append({"id": name, "description": description, "denominator": len(selected),
                       "denominator_unit": "independent_base_experiments", "records": [r["id"] for r in selected],
                       "family_counts": values,
                       "interpretation": "Ocurrencias comparativas positivas verificadas; no prevalencia global, rendimiento ni elegibilidad automática de toda familia"})
    return output


def summary_counts(rows):
    return {"identities_in_documentary_ledger": len(rows),
            "primary_document_read_identities": sum(r["decision"]["primary_document_read"] for r in rows),
            "complete_mirror_only_identities": sum(r["stratum"] == "complete_mirror_only_crypto_risk" for r in rows),
            "crypto_construct_independent_experiments": len(member(rows, "crypto_construct_all")),
            "crypto_reference_extractable_independent_experiments": len(member(rows, "crypto_construct_reference_extractable")),
            "crypto_reference_reporting_sensitivity_experiments": len(member(rows, "crypto_construct_reference_reporting_sensitivity")),
            "unresolved_crypto_continuous_construct_identities": sum(r["stratum"] == previous.UNRECONCILED for r in rows),
            "historical_N7_unchanged": True,
            "stratum_identities": dict(Counter(r["stratum"] for r in rows))}


def audit_new(rows, historical, additions, frequencies):
    ids = [r["id"] for r in rows]
    keys = [r["identity"]["canonical_key"] for r in rows]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError("Duplicate cumulative work")
    if rows[:len(historical["adjudications"])] != historical["adjudications"]:
        raise ValueError("Historical adjudications changed")
    for row in additions:
        d = row["decision"]
        if d["continuous_crypto_construct_corpus"]:
            if not (row["stratum"] == previous.CORE and d["primary_document_read"]
                    and row["target"]["construct_verified"] and row["target"]["future_forecast_verified"]):
                raise ValueError("Crypto construct lacks primary operational evidence")
        if d["frequency_eligible"] and not (d["continuous_crypto_construct_corpus"] and d["baseline_role_verified"]):
            raise ValueError("Crypto frequency candidate lacks independent comparator")
        if d["guide31_reporting_sensitivity_eligible"] and not (d["frequency_eligible"]
                and d["minimum_reporting_adequate_for_candidate_appraisal"]):
            raise ValueError("Strict reporting inclusion lacks descriptive eligibility or sufficient report")
        if row["independentfamilies"] and not d["baseline_role_verified"]:
            raise ValueError("Unverified standalone family cannot receive a positive vote")
        if not row["identity"]["independent_base_experiment"] and d["continuous_crypto_construct_corpus"]:
            raise ValueError("Reused base experiment cannot add a crypto vote")
    mapping = {f["id"]: f for f in frequencies}
    if not set(mapping["crypto_construct_reference_reporting_sensitivity"]["records"]) <= set(mapping["crypto_construct_reference_extractable"]["records"]):
        raise ValueError("Reporting subset escaped descriptive corpus")
    for name in ["crypto_construct_all", "crypto_construct_reference_extractable", "crypto_construct_reference_reporting_sensitivity"]:
        if any(not next(r for r in rows if r["id"] == key)["stratum"] == previous.CORE for key in mapping[name]["records"]):
            raise ValueError("Financial transfer entered crypto frequencies")
    return {"passed": True, "old_identities_preserved": len(historical["adjudications"]),
            "new_works_added": len(additions), "selected_works_not_automatically_new": 13,
            "strata_recomputed": len(frequencies), "native_searches_executed_by_compiler": 0,
            "new_fits": 0, "external_metrics_pooled": False, "missing_values_imputed": False,
            "historical_cuts_unchanged": True}


def compile_outputs(root=ROOT):
    history = root / "references/literature_e1/homogeneous_synthesis_20261107"
    extension = root / "references/literature_e1/free_model_expansion_20261107"
    paths = {"historical_adjudications": history / "part_corpus/adjudications.json",
             "historical_protocol": history / "part_corpus/protocol.json",
             "historical_mechanisms": history / "part_mechanisms/mechanisms.json",
             "historical_builder": root / "scripts/build_e1_homogeneous_synthesis.py",
             "new_reviewed_readings": extension / "readings.json"}
    for name, wanted in FROZEN_HISTORICAL_INPUTS.items():
        if digest((root / name).read_bytes()) != wanted:
            raise ValueError("Frozen historical input changed: " + name)
    old = read(paths["historical_adjudications"])
    old_protocol = read(paths["historical_protocol"])
    mechanisms = read(paths["historical_mechanisms"])
    previous.audit(old, old_protocol, mechanisms, root)
    rows, duplicates, additions = merge_reviewed(old, read(paths["new_reviewed_readings"]))
    frequencies = build_frequencies(rows, old["frequencies"])
    validation = audit_new(rows, old, additions, frequencies)
    counts = summary_counts(rows)
    corpus = {"schema_version": "1.0", "summary": counts, "adjudications": rows, "frequencies": frequencies}
    by_name = {f["id"]: f for f in frequencies}
    main = by_name["crypto_construct_reference_reporting_sensitivity"]
    descriptive = by_name["crypto_construct_reference_extractable"]
    positive = sorted((f for f, v in main["family_counts"].items() if v["count"] and f not in CONTEXTUAL_NEURAL),
                      key=lambda f: (-main["family_counts"][f]["count"], PRIORITY.index(f) if f in PRIORITY else len(PRIORITY), f))
    ranking = []
    for family in positive:
        value = main["family_counts"][family]["count"]
        if ranking and ranking[-1]["count"] == value:
            ranking[-1]["families"].append(family)
        else:
            ranking.append({"count": value, "families": [family]})
    protocol = {"schema_version": "1.0", "cutoff_date": "2026-11-07", "cutoff_timezone": "America/Bogota",
                "purpose": "Corte acumulado posterior a una búsqueda web dirigida por arquitectura; sin reescribir cortes previos",
                "eligibility_policy": "Reglas del protocolo histórico aplicadas simétricamente; hipótesis, ámbitos y calidad separados",
                "source_inputs": {k: {"path": p.relative_to(root).as_posix(), "sha256": digest(p.read_bytes()), "bytes": p.stat().st_size} for k,p in paths.items()},
                "historical_source_commit": HISTORICAL_SOURCE_COMMIT,
                "historical_denominators": {"N7": 7, "construct_N12": 12, "descriptive_N10": 10, "reporting_eligible_N8": 8},
                "source_policy": "No publisher PDFs, full texts, abstracts, market data or user raw native exports are republished",
                "selection_bias": "La búsqueda solicitó 7 TCN y 6 MLP; los cupos son una decisión de recuperación, no una frecuencia bibliométrica. La muestra y el orden de candidatos quedan condicionados a esta búsqueda dirigida.",
                "experimental_models_unchanged": ["GRU", "LSTM", "TCN"], "new_fits": 0}
    summary = {"schema_version": "1.0", "cutoff_date": "2026-11-07", "cutoff_timezone": "America/Bogota",
               "denominators": counts, "selected_works": 13, "existing_selected_works": 3, "new_works": 10,
               "historical_reference_mapping": duplicates, "main_stratum": main,
               "main_stratum_role": "Current primary synthesis: reporting-eligible crypto reference experiments",
               "guide31_reporting_eligible_stratum": main, "descriptive_stratum": descriptive,
               "all_construct_stratum": by_name["crypto_construct_all"],
               "financial_transfer_stratum": by_name["financial_transfer_primary_all"],
               "frequency_order_with_ties": ranking,
               "frequency_top_three_with_tie_policy": positive[:3],
               "retained_registered_trio": ["GRU", "LSTM", "TCN"],
               "registered_recipe_parameters": {"GRU": 1121, "LSTM": 1489, "TCN": 1185},
               "historical_MLP_recipe_parameters": 593,
               "parameter_scope": "Same own D7 feature/output shapes; separate historical/current recipes, not evidence of equal HPO or universal architecture minimum",
               "historical_cuts": {"path": history.relative_to(root).as_posix(), "denominators": old["summary"], "unchanged": True},
               "selection_bias": protocol["selection_bias"],
               "contextual_neural_families": {f: "Comparador neuronal autónomo; clasificación como modelo Deep Learning básico no adjudicada en esta síntesis" for f in sorted(CONTEXTUAL_NEURAL) if f in main["family_counts"]},
               "claims": {"all_57_originals_read": False, "all_13_selected_new": False,
                          "all_7_TCN_papers_crypto_realized_volatility": False,
                          "quota_as_prevalence": False, "transfer_papers_added_to_crypto_frequency": False,
                          "registered_trio_is_frequency_top_three": False, "scientific_optimality_certified": False,
                          "historical_experimental_results_relabelled": False},
               "validation": validation}
    ledger = []
    quality = []
    for row in rows:
        d = row["decision"]
        ledger.append({"id": row["id"], "title": row["title"], "doi": row["doi"] or "", "genre": row["genre"],
                       "stratum": row["stratum"], "primary_document_read": d["primary_document_read"],
                       "target_quantity": row["target"]["quantity"], "baseline_role_verified": d["baseline_role_verified"],
                       "independent_comparator_families": "unknown" if row["independentfamilies"] is None else ";".join(row["independentfamilies"]),
                       "reporting_eligible": d["guide31_reporting_sensitivity_eligible"],
                       "dependence_group": row["identity"]["dependence_group"], "reason": row["reason"]})
        quality.append({"id": row["id"], "primary_document_read": d["primary_document_read"],
                        "crypto_construct": d["continuous_crypto_construct_corpus"],
                        "baseline_role_verified": d["baseline_role_verified"],
                        "minimum_reporting_adequate": d["minimum_reporting_adequate_for_candidate_appraisal"],
                        "reporting_eligible": d["guide31_reporting_sensitivity_eligible"],
                        "protocol_quality_status": row.get("protocolquality",{}).get("status","unknown"),
                        "reason": row["reason"]})
    frequency_rows = [{"stratum": f["id"], "denominator": f["denominator"], "family": family,
                       "positive_reference_experiments": value["count"], "experiment_ids": ";".join(value["experiments"])}
                      for f in frequencies for family,value in f["family_counts"].items()
                      if value["count"] or family in PRIORITY]
    table_families = sorted(set(PRIORITY) | {f for f,v in main["family_counts"].items() if v["count"]},
                            key=lambda f: (-main["family_counts"][f]["count"], PRIORITY.index(f) if f in PRIORITY else len(PRIORITY), f))
    table = "\n".join(f"| {f} | {main['family_counts'][f]['count']}/{main['denominator']} | {descriptive['family_counts'][f]['count']}/{descriptive['denominator']} | {by_name['crypto_construct_all']['family_counts'][f]['count']}/{by_name['crypto_construct_all']['denominator']} |" for f in table_families)
    readme = f"""# Síntesis acumulada E1 · ampliación gratuita del 7 de noviembre de 2026

Este corte conserva las 47 obras del expediente anterior y añade diez obras nuevas:
los 13 trabajos seleccionados contienen tres obras ya registradas. El registro
acumulado tiene **{counts['identities_in_documentary_ledger']} identidades**, **{counts['primary_document_read_identities']} lecturas de fuentes primarias completas**
y **{counts['complete_mirror_only_identities']} transcripción espejo completa**. Los demás niveles de acceso
siguen separados; el original de Vo no se considera leído íntegramente a partir
de fragmentos editoriales indexados.

## Frecuencia por un criterio común

| Familia | Cripto elegible por reporte actual | Cripto descriptivo actual | Todo el constructo cripto actual |
|---|---:|---:|---:|
{table}

El denominador principal es **N={main['denominator']} elegible por suficiencia de reporte**;
N={descriptive['denominator']} es extracción descriptiva y N={by_name['crypto_construct_all']['denominator']} es el constructo operativo
continuo realizado/histórico. No se exige mismo activo, exchange, frecuencia,
ventana o transformación que E1. La suficiencia se aplica simétricamente a
fuentes históricas y nuevas, sin convertir un ancho final ausente en ausencia
universal del candidato cuando estructura y protocolo son suficientes.
Los tres estudios cripto nuevos son García-Medina/Aguayo-Moreno,
Korkusuz/Sahiner y Lensky/Hao. Se revisan **todas** sus familias comparativas,
no únicamente MLP: RNN simple y CNN genérica conservan sus etiquetas propias.
Los módulos internos, variantes y componentes híbridos no suman bases autónomas.
Independiente significa experimento base no reutilizado; no independencia
estadística de mercados, muestras o errores compartidos entre publicaciones.
RBFN, GFN, MFN y otras redes comparativas permanecen como familias distintas;
sus ocurrencias no certifican que cada una satisfaga la condición de modelo
Deep Learning básico de la guía y se excluyen del ranking de candidatos DL.

Los cortes históricos N7, N12/N10/N8 permanecen en sus carpetas, con sus propios
identificadores y hashes. El N10 descriptivo anterior y el N actual elegible por
reporte pueden tener el mismo número y distinta composición: no se intercambian.
Las publicaciones/grises, unidades de varianza/desviación e índices implícitos
conservan estratos en `frequencies.csv`. Los trabajos financieros se registran
como transferencia; sus ocurrencias no incrementan los numeradores cripto.
El preprint de Ge no se transforma en publicación revisada, ni el preprint y
la publicación IEEE de Lensky se cuentan como dos estudios.

## Selección y alcance

La terna experimental registrada **GRU–LSTM–TCN** permanece como contraste entre
reglas de estado recurrente y filtros causales. Las frecuencias y el tamaño de
la MLP histórica se muestran con el mismo criterio; no se afirma que la terna
sea la más frecuente ni un mínimo global. Las recetas propias conservan
GRU 1.121, LSTM 1.489 y TCN 1.185 parámetros; MLP 593 pertenece a otro experimento
histórico y no a la reserva final de 465 semillas. La comparación de cantidades
no certifica igualdad de capacidad, coste, presupuesto ni error.

La búsqueda nueva fue **dirigida por arquitectura**, solicitando siete TCN y seis
MLP. Esa cuota organiza recuperación, sin estimar prevalencia ni convertir
los 13 trabajos en13 inclusiones homogéneas. No se declara revisión exhaustiva.
Los 14 informes nativos no recuperados siguen como disposición de acceso al
corte anterior, sin compras obligatorias ni evaluación de calidad inventada.
Este compilador no consulta bases nativas, entrena, emite pronósticos ni
modifica datos, selección o resultados experimentales.

## Verificación

`python scripts/build_e1_homogeneous_expansion.py --check`

El programa comprueba el expediente histórico, deduplicación por DOI,
47 identidades conservadas, diez adiciones, todas las familias verificadas,
subconjuntos de reporte, grupos de dependencia y ámbitos. `protocol.json`
conserva hashes de los insumos públicos. `adjudications.json` permite revisar
los juicios; `comparator_quality.csv` separa calidad, acceso y rol de referencia.
El manifiesto permite únicamente estos metadatos, sin artículos, resúmenes,
contactos, datos de mercado o predicciones individuales.
"""
    outputs = {"README.md": readme.encode(), "summary.json": encoded(summary), "protocol.json": encoded(protocol),
               "adjudications.json": encoded(corpus), "adjudications.csv": csv_encoded(list(ledger[0]),ledger),
               "comparator_quality.csv": csv_encoded(list(quality[0]),quality),
               "frequencies.csv": csv_encoded(list(frequency_rows[0]),frequency_rows), "validation.json": encoded(validation)}
    manifest = {"publication_schema": "e1_public_metadata_allowlist_v1", "approved_public_files": sorted(outputs),
                "files": {name: {"sha256": digest(data), "bytes": len(data)} for name,data in sorted(outputs.items())},
                "scope": "Cumulative reviewed documentary metadata; earlier cuts unchanged; no full articles, private exports or new fits"}
    outputs["manifest.json"] = encoded(manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = compile_outputs()
    for destination in (STAGE, MIRROR):
        for name,data in outputs.items():
            target = destination / name
            if args.check:
                if not target.is_file() or target.read_bytes() != data:
                    raise SystemExit("Missing or stale generated output: " + str(target))
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data)
    result = json.loads(outputs["summary.json"])
    print(json.dumps({"status": "checked" if args.check else "written", "identities": result["denominators"]["identities_in_documentary_ledger"], "new_works": 10, "public_files": len(outputs), "new_fits": 0}))


if __name__ == "__main__":
    main()
