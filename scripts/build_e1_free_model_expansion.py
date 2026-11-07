"""Compile the reviewed public TCN/MLP intake without searching or training.

Only committed, curated metadata is read. Publisher PDFs, abstracts, contacts,
native exports and the private research workspace are never copied.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "references/literature_e1/free_model_expansion_20261107"
MIRROR = ROOT / "book/_static/literature_e1/free_model_expansion_20261107"
SCHEMA = "e1_public_metadata_allowlist_v1"
NAMES = {"inputs.json", "readings.json", "references.csv", "references.bib",
         "search_log.json", "README.md", "summary.json", "validation.json"}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(data):
    rows = data["records"]
    if len(rows) != 13:
        raise ValueError("Intake must preserve thirteen selected works")
    if len({r["doi"].casefold() for r in rows}) != 13:
        raise ValueError("A publication and its preprint cannot receive two votes")
    ids = [r["expansion_intake_id"] for r in rows]
    if set(ids) != {f"TCN{i:02}" for i in range(1, 8)} | {f"MLP{i:02}" for i in range(1, 7)}:
        raise ValueError("Seven/six are intake groups, not extracted frequencies")
    existing = [r for r in rows if r["existing_identity"]]
    if len(existing) != 3 or {r["id"] for r in existing} != {"S1", "PR_CVI", "E1OA-T01"}:
        raise ValueError("Existing DOI identities must be preserved")
    new = [r for r in rows if not r["existing_identity"]]
    if len(new) != 10 or len({r["id"] for r in rows}) != 13:
        raise ValueError("New work and identity counts disagree")
    by = {r["id"]: r for r in rows}
    for r in rows:
        if not r["free_fulltext_url"].startswith("https://"):
            raise ValueError("A public scholarly text route is required")
        if len(r["independentfamilies"]) != len(set(r["independentfamilies"])):
            raise ValueError("Variants cannot add another family occurrence")
        if r["decision"]["continuous_crypto_construct_corpus"] and not r["stratum"].startswith("crypto_"):
            raise ValueError("Other-market evidence cannot inflate the crypto denominator")
        if r["identity"]["dependence_group"] != r["id"] and not r["existing_identity"]:
            raise ValueError("New dependence claims require explicit adjudication")
    if by["E1F-T05"]["decision"]["primary_document_read"]:
        raise ValueError("Indexed Vo excerpts are not a complete original read")
    if "TCN" in by["E1F-T07"]["independentfamilies"] or by["E1F-T07"]["decision"]["baseline_role_verified"]:
        raise ValueError("Agal's hybrid cannot become an autonomous basic-TCN vote")
    if by["E1F-M03"]["decision"]["guide31_reporting_sensitivity_eligible"]:
        raise ValueError("Lensky missing MLP recipe must retain the same reporting sensitivity")
    if by["E1F-M03"]["doi"] != "10.1109/CAI59869.2024.00154" or by["E1F-M03"]["year"] != 2024:
        raise ValueError("Lensky final conference identity must supersede preliminary preprint status")
    expected = {
        "E1F-M01": {"MLP", "LSTM"},
        "E1F-M02": {"MLP", "LSTM", "RNN_simple", "RBFN", "GFN", "MFN"},
        "E1F-M03": {"MLP", "LSTM", "CNN_generic"},
    }
    if any(set(by[k]["independentfamilies"]) != v for k, v in expected.items()):
        raise ValueError("All verified comparative families must be retained, not just the searched label")
    if by["E1F-M02"].get("dependency_check", {}).get("result") != "distinct_described_experiment; no_positive_forecast_or_fit_reuse_documented":
        raise ValueError("Korkusuz/Brauneis dependency check is required")
    if sum(r["genre"] == "grey" for r in rows) != 1 or by["E1F-T06"]["genre"] != "grey":
        raise ValueError("Ge remains the only selected unverified-publication preprint")
    return rows


def outputs(data, search):
    rows = validate(data)
    new = [r for r in rows if not r["existing_identity"]]
    new_crypto = [r for r in new if r["decision"]["continuous_crypto_construct_corpus"]]
    summary = {
        "schema_version": 1, "search_date_colombia": data["search_date_colombia"],
        "search_type": data["search_type"], "selected_works": len(rows),
        "selected_intake_quotas_not_prevalence": {"TCN": 7, "MLP": 6},
        "existing_identities_not_new_studies": [r["id"] for r in rows if r["existing_identity"]],
        "new_identities": [r["id"] for r in new], "new_identity_count": len(new),
        "new_primary_documents_read": sum(r["decision"]["primary_document_read"] for r in new),
        "new_indexed_primary_only": [r["id"] for r in new if not r["decision"]["primary_document_read"]],
        "publication_identities": sum(r["genre"] == "publication" for r in rows),
        "preprint_identities": [r["id"] for r in rows if r["genre"] == "grey"],
        "new_crypto_construct_experiments": [r["id"] for r in new_crypto],
        "new_crypto_reporting_eligible": [r["id"] for r in new_crypto if r["decision"]["guide31_reporting_sensitivity_eligible"]],
        "new_crypto_comparators": {r["id"]: r["independentfamilies"] for r in new_crypto},
        "new_TCNS_are_seven_new_bitcoin_realized_volatility_studies": False,
        "native_index_queries_executed_by_this_builder": 0,
        "new_model_fits": 0, "new_forecasts": 0,
        "historical_experiments_unchanged": True,
        "cumulative_frequencies": "Computed separately by homogeneous_expansion_20261107 using the same historical eligibility policy.",
        "intake_sha256": digest(encoded(data)),
    }
    csvout = io.StringIO(newline="")
    names = ["id", "intake_id", "citation_key", "title", "authors", "year", "online_year", "doi",
             "free_fulltext_url", "stratum", "target_quantity", "independent_comparator_families",
             "hybrid_families", "primary_document_read", "reporting_eligible_crypto",
             "existing_identity", "publication_type", "source_version"]
    writer = csv.DictWriter(csvout, fieldnames=names)
    writer.writeheader()
    bib = []
    for r in rows:
        writer.writerow({"id": r["id"], "intake_id": r["expansion_intake_id"],
                         "citation_key": r["citation_key"], "title": r["title"],
                         "authors": "; ".join(r["authors"]), "year": r["year"],
                         "online_year": r.get("online_year"), "doi": r["doi"],
                         "free_fulltext_url": r["free_fulltext_url"], "stratum": r["stratum"],
                         "target_quantity": r["target"]["quantity"],
                         "independent_comparator_families": ";".join(r["independentfamilies"]),
                         "hybrid_families": ";".join(r["hybridfamilies"]),
                         "primary_document_read": r["decision"]["primary_document_read"],
                         "reporting_eligible_crypto": r["decision"]["guide31_reporting_sensitivity_eligible"],
                         "existing_identity": r["existing_identity"],
                         "publication_type": r["publication"]["type"], "source_version": r["source_version"]})
        pub = dict(r["publication"])
        kind = {"journal": "article", "conference": "inproceedings", "preprint": "misc"}[pub.pop("type")]
        fields = {"author": " and ".join(r["authors"]), "title": r["title"], "year": str(r["year"]),
                  "doi": r["doi"], "url": r["free_fulltext_url"], **pub}
        bib.append("@" + kind + "{" + r["citation_key"] + ",\n" +
                   ",\n".join("  " + k + " = {" + v + "}" for k, v in fields.items()) + "\n}\n")
    readme = """# Ampliación gratuita: siete fuentes TCN y seis MLP

La búsqueda web complementaria del 07/11/2026 reúne trece obras distintas:
tres ya estaban identificadas en E1 y diez se incorporan como referencias nuevas.
Son nueve artículos de revista, tres publicaciones de congreso y un preprint
(Ge et al.). La identidad IEEE CAI 2024 de Lensky–Hao se verificó al finalizar:
el DOI IEEE y la versión arXiv corresponden a una sola obra.

## Acceso y lectura

Doce de las trece obras seleccionadas cuentan con fuente primaria completa
consultada: tres lecturas anteriores y nueve nuevas. Vo (2025) es acceso abierto
editorial; las secciones y tablas indexadas se consultaron, pero las aperturas
automatizadas devolvieron HTTP 429. Se conserva como lectura primaria parcial,
sin inventar descarga, texto íntegro leído o licencia de reutilización de archivos.
Se publican únicamente metadatos y enlaces: ningún PDF, resumen exportado,
contacto, texto íntegro o contenido privado se redistribuye.

## Papel de cada arquitectura

Las cuotas siete/seis organizan esta selección dirigida; no representan frecuencia
de uso en toda la literatura. Las fuentes de VIX, bolsa, divisas y materias primas
son transferencia financiera, separada de volatilidad realizada/histórica cripto.
Nguyen estudia CVI implícita y conserva otros componentes del modelo al sustituir
LSTM por TCN. Agal utiliza TCN dentro de FinAnalyzeNet: mencionar una TCN base
sin recuperar su receta ni fila numérica no añade un voto autónomo básico.

Las tres lecturas cripto nuevas verifican MLP y también otras bases: García-Medina
y Aguayo-Moreno, MLP/LSTM; Korkusuz y Sahiner, MLP/LSTM/RNN simple y RBFN/GFN/MFN
como tipos distintos; Lensky y Hao, MLP/LSTM/CNN. Las variantes y cabezas internas
no añaden estudios. Lensky conserva reporte incompleto de la receta MLP y queda
fuera de la sensibilidad estricta, aplicando la misma regla a fuentes antiguas y
nuevas. Korkusuz describe otro protocolo que Brauneis–Sahiner: diferente fuente,
frecuencia, activos, horizonte y ventana de evaluación. No se documentó reutilización
de ajustes o pronósticos; esto no acredita independencia estadística de mercados.

## Productos y reproducción

- `inputs.json` conserva adjudicación, versiones, calidad, datos desconocidos y procedencia.
- `readings.json` contiene las mismas trece fichas, compatibles con el compilador acumulado.
- `references.csv` y `references.bib` contienen autores, DOI y rutas gratuitas.
- `search_log.json` conserva consultas web efectivamente comunicadas por los investigadores.
- `summary.json`, `validation.json` y `manifest.json` verifican unidades e integridad.

`python scripts/build_e1_free_model_expansion.py --check` recompone estos metadatos.
No ejecuta consultas nativas, entrenamiento, pronósticos o lectura de reservas.
Los conteos acumulados se calculan en la etapa homogénea posterior, sin cambiar
los cortes históricos ni atribuir retrospectivamente estas fuentes a la selección
experimental. Desconocido no significa cero o ausencia. La receta y los resultados
de los autores no se reproducen por compilar sus referencias.
"""
    validation = {"schema_version": 1, "status": "PASS", "selected_unique_works": 13,
                  "distinct_dois": 13, "existing": 3, "new": 10,
                  "all_comparator_families_retained": True,
                  "crypto_transfer_and_hybrid_roles_separated": True,
                  "indexed_primary_not_relabelled_as_full_read": True,
                  "publication_and_author_version_not_duplicated": True,
                  "historical_cut_not_rewritten": True,
                  "new_fits": 0, "new_forecasts": 0}
    result = {"inputs.json": encoded(data), "readings.json": encoded(data),
              "references.csv": csvout.getvalue().encode(), "references.bib": "\n".join(bib).encode(),
              "search_log.json": encoded(search), "README.md": readme.encode(),
              "summary.json": encoded(summary), "validation.json": encoded(validation)}
    assert set(result) == NAMES
    manifest = {"publication_schema": SCHEMA, "approved_public_files": sorted(result),
                "files": {name: {"sha256": digest(content), "bytes": len(content)} for name, content in sorted(result.items())}}
    result["manifest.json"] = encoded(manifest)
    return result


def build(check=False):
    data = json.loads((STAGE / "inputs.json").read_text())
    search = json.loads((STAGE / "search_log.json").read_text())
    products = outputs(data, search)
    for directory in (STAGE, MIRROR):
        if check:
            if {p.name for p in directory.iterdir() if p.is_file()} != set(products):
                raise ValueError("Unexpected public stage member")
            for name, content in products.items():
                if (directory / name).read_bytes() != content:
                    raise ValueError("Stale or modified metadata: " + str(directory / name))
        else:
            directory.mkdir(parents=True, exist_ok=True)
            for name, content in products.items():
                (directory / name).write_bytes(content)
    return {"status": "checked" if check else "built", "files": len(products), "works": 13, "new": 10, "new_fits": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    print(json.dumps(build(parser.parse_args().check), ensure_ascii=False))
