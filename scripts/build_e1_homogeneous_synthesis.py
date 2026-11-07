#!/usr/bin/env python3
"""Audit and publish the E1 homogeneous documentary synthesis.

This consolidates existing primary-review decisions. It performs no search,
training or quality imputation and never changes historical denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "references/literature_e1/homogeneous_synthesis_20261107"
MIRROR = ROOT / "book/_static/literature_e1/homogeneous_synthesis_20261107"
CORE = "crypto_realized_or_historical_continuous_construct"
UNRECONCILED = "crypto_continuous_volatility_claim_unreconciled"
IMPLIED = "crypto_implied_volatility_index"
SOURCE_NAMES = (
    "part_corpus/protocol.json", "part_corpus/adjudications.json", "part_corpus/README.md",
    "part_mechanisms/mechanisms.json", "part_mechanisms/README.md",
)
FAMILY_LABELS = {"CNN_generic": "CNN genérica", "Transformer_encoder": "Encoder Transformer"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def membership(rows: list[dict], name: str) -> list[dict]:
    core = [r for r in rows if r["decision"]["continuous_crypto_construct_corpus"]]
    if name == "crypto_construct_all":
        return core
    if name == "crypto_construct_reference_extractable":
        return [r for r in core if r["decision"]["frequency_eligible"]]
    if name == "crypto_construct_reference_reporting_sensitivity":
        return [r for r in core if r["decision"]["guide31_reporting_sensitivity_eligible"]]
    if name == "crypto_declared_target_inclusive_sensitivity":
        return [r for r in rows if r in core or r["stratum"] == UNRECONCILED]
    if name == "crypto_all_continuous_constructs_including_implied":
        return [r for r in rows if r in core or r["stratum"] == IMPLIED]
    if name in {"crypto_construct_publication", "crypto_construct_grey"}:
        return [r for r in core if r["genre"] == name.removeprefix("crypto_construct_")]
    if name.startswith("crypto_scale_"):
        return [r for r in core if r["target"].get("scale_family") == name.removeprefix("crypto_scale_")]
    if name.startswith("crypto_target_"):
        return [r for r in core if r["target"]["quantity"] == name.removeprefix("crypto_target_")]
    if name in {IMPLIED, "transfer_equity_realized", "transfer_equity_implied", "crypto_volatility_states"}:
        return [r for r in rows if r["stratum"] == name and r["decision"]["primary_document_read"]]
    raise ValueError("Unknown documentary stratum: " + name)


def audit(adjudications: dict, protocol: dict, mechanisms: dict, root: Path = ROOT) -> dict:
    rows = adjudications["adjudications"]
    ids = [r["id"] for r in rows]
    keys = [r["identity"]["canonical_key"] for r in rows]
    if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
        raise ValueError("Repeated identity: cannot count versions twice")
    expected_summary = {
        "identities_in_documentary_ledger": len(rows),
        "primary_document_read_identities": sum(r["decision"]["primary_document_read"] for r in rows),
        "complete_mirror_only_identities": sum(r["stratum"] == "complete_mirror_only_crypto_risk" for r in rows),
        "crypto_construct_independent_experiments": len(membership(rows, "crypto_construct_all")),
        "crypto_reference_extractable_independent_experiments": len(membership(rows, "crypto_construct_reference_extractable")),
        "crypto_reference_reporting_sensitivity_experiments": len(membership(rows, "crypto_construct_reference_reporting_sensitivity")),
        "unresolved_crypto_continuous_construct_identities": sum(r["stratum"] == UNRECONCILED for r in rows),
        "historical_N7_unchanged": True,
        "stratum_identities": dict(Counter(r["stratum"] for r in rows)),
    }
    if adjudications["summary"] != expected_summary:
        raise ValueError("Document/read denominators do not reconcile")
    for source in protocol["source_inputs"]:
        if digest((root / source["path"]).read_bytes()) != source["sha256"]:
            raise ValueError("Historical source changed: " + source["path"])
    for frequency in adjudications["frequencies"]:
        selected = membership(rows, frequency["id"])
        selected_ids = {r["id"] for r in selected}
        if set(frequency["records"]) != selected_ids or len(frequency["records"]) != len(selected):
            raise ValueError("Wrong stratum membership: " + frequency["id"])
        groups = [r["identity"]["dependence_group"] for r in selected]
        if len(groups) != len(set(groups)) or any(not r["identity"]["independent_base_experiment"] for r in selected):
            raise ValueError("Dependent experiment adds a vote: " + frequency["id"])
        if frequency["denominator"] != len(selected):
            raise ValueError("Wrong denominator: " + frequency["id"])
        for family, values in frequency["family_counts"].items():
            positive = sorted(r["id"] for r in selected
                              if r["decision"]["baseline_role_verified"]
                              and family in (r["independentfamilies"] or []))
            if (values["experiments"] != positive or values["count"] != len(positive)
                    or values["denominator"] != len(selected)):
                raise ValueError("Invalid positive comparator count: " + frequency["id"] + "/" + family)
    final = mechanisms["current_phase_II_models"]
    for model in final:
        if sum(model["parameter_decomposition"].values()) != model["trainable_parameters"]:
            raise ValueError("Parameter arithmetic mismatch: " + model["family"])
    if {m["family"]: m["trainable_parameters"] for m in final} != {"GRU": 1121, "LSTM": 1489, "TCN": 1185}:
        raise ValueError("Registered recipes changed")
    return {"passed": True, "identities": len(rows), "strata_recomputed": len(adjudications["frequencies"]),
            "historical_inputs_verified": len(protocol["source_inputs"]), "new_searches": 0, "new_fits": 0,
            "external_metrics_pooled": False, "missing_values_imputed": False,
            "historical_N7_unchanged": True, "dependent_experiments_counted_twice": 0}


def compile_outputs(base: Path = STAGE, root: Path = ROOT) -> dict[str, bytes]:
    raw = {name: (base / name).read_bytes() for name in SOURCE_NAMES}
    corpus = json.loads(raw["part_corpus/adjudications.json"])
    protocol = json.loads(raw["part_corpus/protocol.json"])
    mechanisms = json.loads(raw["part_mechanisms/mechanisms.json"])
    validation = audit(corpus, protocol, mechanisms, root)
    frequencies = {f["id"]: f for f in corpus["frequencies"]}
    main = frequencies["crypto_construct_reference_extractable"]
    priority = ["LSTM", "GRU", "MLP", "TCN", "CNN_generic", "Transformer_encoder"]
    order = sorted(priority, key=lambda f: (-main["family_counts"][f]["count"], priority.index(f)))
    summary = {"schema_version": "1.0", "cutoff_utc_date": "2026-11-07",
               "purpose": "Homogeneous adjudication of existing primary-review records, after the registered experiments",
               "denominators": corpus["summary"], "main_stratum": main,
               "main_stratum_role": "Descriptive extraction, not all eligible under guide3.1 reporting criterion",
               "all_construct_stratum": frequencies["crypto_construct_all"],
               "reporting_sensitivity": frequencies["crypto_construct_reference_reporting_sensitivity"],
               "guide31_reporting_eligible_stratum": frequencies["crypto_construct_reference_reporting_sensitivity"],
               "frequency_order_with_ties": [["LSTM"], ["GRU"], ["MLP"], ["CNN_generic", "TCN", "Transformer_encoder"]],
               "parameter_order_registered_final_recipes": [{"family": "GRU", "parameters": 1121}, {"family": "TCN", "parameters": 1185}, {"family": "LSTM", "parameters": 1489}],
               "historical_MLP_recipe_parameters": 593,
               "parameter_comparison_scope": "Current own recipes only; no universal architecture ranking or identical representational capacity",
               "retained_registered_trio": ["GRU", "LSTM", "TCN"],
               "decision": "Retain the completed registered cell-versus-causal-filter comparison; disclose the MLP counterexample and the frequency/capacity tradeoff",
               "frequency_top_three": ["LSTM", "GRU", "MLP"],
               "claims": {"three_most_frequent_selected": False, "unique_minimum_complexity_trio_demonstrated": False,
                          "three_macro_families_required_by_guide": False, "selection_made_before_original_fits": False,
                          "functional_biases_described_pairwise": True, "historical_results_relabelled": False,
                          "scientific_optimality_certified": False},
               "validation": validation,
               "source_files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in raw.items()}}
    frequency_rows = []
    for f in corpus["frequencies"]:
        for family, values in f["family_counts"].items():
            if values["count"] or family in priority:
                frequency_rows.append({"stratum": f["id"], "denominator": f["denominator"], "family": family,
                                       "positive_reference_experiments": values["count"], "experiment_ids": ";".join(values["experiments"])})
    ledger = []
    for r in corpus["adjudications"]:
        d = r["decision"]
        ledger.append({"id": r["id"], "title": r["title"], "doi": r["doi"] or "", "genre": r["genre"],
                       "stratum": r["stratum"], "primary_document_read": d["primary_document_read"],
                       "target_quantity": r["target"]["quantity"], "baseline_role_verified": d["baseline_role_verified"],
                       "independent_comparator_families": "unknown" if r["independentfamilies"] is None else ";".join(r["independentfamilies"]),
                       "reporting_sensitivity_eligible": d["guide31_reporting_sensitivity_eligible"],
                       "dependence_group": r["identity"]["dependence_group"],
                       "independent_base_experiment": r["identity"]["independent_base_experiment"],
                       "reason": r["reason"]})
    table = "\n".join(f"| {FAMILY_LABELS.get(f, f)} | {main['family_counts'][f]['count']}/10 | {frequencies['crypto_construct_all']['family_counts'][f]['count']}/12 | {frequencies['crypto_construct_reference_reporting_sensitivity']['family_counts'][f]['count']}/8 |" for f in order)
    readme = f"""# Síntesis homogénea E1 · 7 de noviembre de 2026

Este expediente consolida fichas ya revisadas. No ejecuta nuevas búsquedas,
recupera originales, entrena redes ni cambia los resultados del proyecto.
La unidad es la obra; la frecuencia utiliza experimentos base independientes.
El registro contiene 47 identidades, 26 originales primarios leídos y una
transcripción espejo completa separada. No son 47 artículos completos leídos.

## Frecuencia y sensibilidad

| Familia | Referencias extraíbles | Todo el constructo realizado/histórico | Reporte elegible §3.1 |
|---|---:|---:|---:|
{table}

N12 reúne constructos operativos continuos realizados/históricos de cripto;
N10 contiene al menos una referencia autónoma verificable: es extracción
descriptiva, no diez inclusiones que cumplan íntegramente §3.1. Los dos restantes
de N12 no son negativos demostrados. El estrato N8 elegible por suficiencia de
reporte §3.1 excluye juntas S8 y PSO–GRU por descripción insuficiente.
La escala, las publicaciones/grises y el objetivo implícito
conservan estratos distintos en `frequencies.csv`. Dudek 2025 reutiliza la base
de 2024 y no suma voto. T3/DeepVol es una identidad. Los híbridos y componentes
internos no se descomponen en familias. N7 y sus tablas históricas se conservan.

## Decisión y límites

Las tres frecuencias mayores son LSTM, GRU y MLP. El estudio registrado
GRU–LSTM–TCN se conserva como contraste de reglas recurrentes de estado y
filtros temporales causales. La revisión actual es posterior a sus ajustes.
No prueba que esa terna sea top3, mínima ni única. La guía exige sesgos
inductivos diferentes, sin imponer tres macrofamilias: la diferencia entre
interpolación GRU y celda/salida LSTM se explica junto a su recurrencia común.
Ambas empiezan sin estado previo a cada ventana D7; TCN tampoco ve datos
anteriores a esos siete días aunque su campo receptivo teórico sea29.

Las recetas finales propias suman GRU1121, TCN1185 y LSTM1489 parámetros.
Existe una MLP histórica propia con593 y HPO declarado antes de sus ajustes;
no se descarta por falta de receta. Su menor número de pesos no demuestra
mejor pronóstico. Los tamaños de recetas publicadas no forman una comparación
común sin supuestos de entrada/salida, bloque y ancho. Se mantienen los
ausentes; no se rellenan defaults para decidir a favor de una familia.

Los14 informes nativos no recuperados al corte son un estado terminal de
acceso y una limitación de cobertura; no se eliminan del flujo ni se excluyen
por calidad sin leerlos. El estrato de17 registros parciales/no obtenidos del
universo ampliado es otra unidad y no17 tareas obligatorias de recuperación;
no equivale a todos los registros sin lectura primaria: también hay código y
otros niveles de acceso.
La bitácora nativa se acredita aparte: esta síntesis no prueba consultar IEEE.

## Reproducción documental

`python scripts/build_e1_homogeneous_synthesis.py --check`

El compilador verifica las fuentes históricas, identidades, dependencias,
denominadores,{validation['strata_recomputed']} estratos y las sumas de parámetros. `adjudications.csv` permite
auditar lectura, objetivo, comparadores y razones en47 filas; los JSON de
`part_corpus` conservan localizadores y juicio crítico. `part_mechanisms`
contiene ecuaciones, recetas, contraste por pares y alternativas. El manifiesto
permite únicamente estos metadatos revisados; no redistribuye originales,
resúmenes exportados, datos de mercado ni predicciones.
""".replace("\n+", "\n")
    outputs = dict(raw)
    outputs.update({"summary.json": json_bytes(summary), "README.md": readme.encode(),
                    "validation.json": json_bytes(validation),
                    "frequencies.csv": csv_bytes(list(frequency_rows[0]), frequency_rows),
                    "adjudications.csv": csv_bytes(list(ledger[0]), ledger)})
    manifest = {"publication_schema": "e1_public_metadata_allowlist_v1", "approved_public_files": sorted(outputs),
                "files": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in sorted(outputs.items())},
                "scope": "Existing reviewed documentary metadata only; no primary payloads, new fits or native-search execution claims"}
    outputs["manifest.json"] = json_bytes(manifest)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated files without changing them")
    args = parser.parse_args()
    outputs = compile_outputs()
    for destination in (STAGE, MIRROR):
        for name, data in outputs.items():
            target = destination / name
            if args.check:
                if not target.is_file() or target.read_bytes() != data:
                    raise SystemExit("Missing or stale public metadata: " + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    print(json.dumps({"status": "checked" if args.check else "written", "public_files": len(outputs),
                      "identities": 47, "native_searches_executed": 0, "new_fits": 0}))


if __name__ == "__main__":
    main()
