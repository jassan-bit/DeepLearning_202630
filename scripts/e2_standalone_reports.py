#!/usr/bin/env python3
"""Generate an offline report for E2 packages D04/D05/D07/D09/D11.

This file is copied to ``run.py`` inside each standalone ZIP. Only the Python
standard library is used. Inputs are declared in the adjacent package.json:

  {"id":"D11", "title":"...", "source_commit":"<40 hex characters>",
   "documents":[{"title":"...", "path":"documents/report.md"}],
   "tables":[{"title":"...", "path":"data/appendix_B_current.csv"}],
   "json_files":[{"title":"...", "path":"data/appendix_B_current.json"}]}

An entry may also declare sha256 and bytes. An optional manifest.json or
source_manifest.json accepts {"files":[{"path":"...", "sha256":"..."}]}.
No models, remote resources, predictive metrics or statistical tests are run.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import date, timedelta
import hashlib
import html
import json
import math
import mimetypes
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import sys
from collections import Counter
from urllib.parse import quote, unquote, urlsplit


VERSION = "e2-standalone-documentary-reports-v1"
IDS = {"D04", "D05", "D07", "D09", "D11"}
NEURAL = {
    "FCN-1D", "ResNet-1D", "Inception-1D",
    "LSTM-autoencoder con predictor", "TimesNet", "TSMixer", "TiDE",
    "TimeMixer", "PatchTST", "iTransformer", "MLP16", "LSTM32", "GRU32",
}
CLASSICAL = {"Persistence", "SeasonalNaive", "ARIMA"}
SEEDS = list(range(271801, 271807))
DELTA = 0.00025772842664981106


class PackageError(ValueError):
    """The declared package cannot be read as a standalone input."""


def local_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise PackageError("La ruta debe ser relativa y usar separadores '/'.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ":" in relative.parts[0]:
        raise PackageError(f"Ruta no relativa: {name}")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise PackageError(f"La ruta sale del paquete: {name}")
    if not result.is_file():
        raise PackageError(f"No se encuentra el archivo incluido: {name}")
    return result


def load_json(path: Path):
    def nonfinite(value):
        raise PackageError(f"JSON contiene un número no finito: {path.name}: {value}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"JSON inválido: {path.name}: {exc}") from exc


def load_csv(path: Path):
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or len(set(fields)) != len(fields):
                raise PackageError(f"Cabecera CSV vacía o duplicada: {path.name}")
            rows = list(reader)
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise PackageError(f"Número de columnas inconsistente: {path.name}")
        return fields, rows
    except (UnicodeError, csv.Error) as exc:
        raise PackageError(f"CSV inválido: {path.name}: {exc}") from exc


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class Review:
    def __init__(self):
        self.checks = []
        self.summary = []

    def check(self, label, passed, detail, source=""):
        self.checks.append({"control": label, "correcto": bool(passed),
                            "detalle": detail, "fuente": source})

    def note(self, label, value):
        self.summary.append({"concepto": label, "valor": value})

    @property
    def passed(self):
        return bool(self.checks) and all(item["correcto"] for item in self.checks)


def find_json(inputs, predicate, description):
    matches = [item for item in inputs if item["kind"] == "json_files" and predicate(item["data"])]
    if not matches:
        raise PackageError(f"Falta el JSON incluido para {description}.")
    return matches[0]


def find_table(inputs, fields, description):
    for item in inputs:
        if item["kind"] == "tables" and set(fields).issubset(item["fields"]):
            return item
    raise PackageError(f"Falta la tabla incluida para {description}.")


def validate_freeze(inputs, review):
    item = find_json(inputs, lambda x: isinstance(x, dict) and
                     {"analysis_frozen", "final_seed_ids", "models"}.issubset(x),
                     "la congelación de los 81 predictores")
    data, source = item["data"], item["path"]
    models = data.get("models", [])
    counts = Counter(row.get("model") for row in models if isinstance(row, dict))
    expected = {**{name: 6 for name in NEURAL}, **{name: 1 for name in CLASSICAL}}
    review.check("Catálogo congelado", counts == expected and len(models) == 81,
                 "Trece redes por seis semillas y tres comparadores deterministas.", source)
    review.check("Análisis y predictores congelados",
                 data.get("analysis_frozen") is True and
                 data.get("weights_scalers_and_calibrators_frozen_before_final_data") is True,
                 "Flags registrados antes del scoring de la campaña.", source)
    review.check("Semillas finales", data.get("final_seed_ids") == SEEDS,
                 "Identificadores 271801–271806, sin réplicas clásicas artificiales.", source)
    identities = [(row.get("model"), row.get("seed")) for row in models]
    review.check("Identidades de predictores", len(set(identities)) == 81,
                 "No hay pares modelo/semilla duplicados.", source)
    for name in NEURAL:
        values = sorted(row.get("seed") for row in models
                        if row.get("model") == name and isinstance(row.get("seed"), int))
        review.check(f"Semillas de {name}", values == SEEDS, "Seis semillas registradas.", source)
    review.note("Predictores congelados", len(models))


def validate_preregistration(inputs, review):
    item = find_json(inputs, lambda x: isinstance(x, dict) and
                     x.get("protocol_id") == "e2-reduced-delivery-v1" and
                     {"hpo", "inference", "replications"}.issubset(x),
                     "el protocolo reducido vigente")
    data, source = item["data"], item["path"]
    contract, inference = data["contract"], data["inference"]
    systems = data.get("systems", [])
    review.check("Catálogo registrado", len(systems) == 16 and set(systems) == NEURAL | CLASSICAL,
                 "Dieciséis sistemas con identidades conocidas.", source)
    review.check("Contrato temporal", contract.get("input_shape") == [7, 5] and
                 contract.get("horizon_days") == 1 and contract.get("embargo_days") == 7,
                 "Ventana D7×5, horizonte de un día y embargo registrado de siete días.", source)
    review.check("Margen y métrica", contract.get("delta") == DELTA and
                 "individual_seed_RMSE" in contract.get("primary_metric", ""),
                 "Margen heredado y media de RMSE individuales en unidades originales.", source)
    pairs = inference.get("primary_contrasts", [])
    expected = {(name, baseline) for name in NEURAL - {"MLP16", "LSTM32", "GRU32"}
                for baseline in ("GRU32", "Persistence")}
    actual = {(row.get("model"), row.get("baseline")) for row in pairs}
    review.check("Familia primaria", len(pairs) == 20 and actual == expected and
                 inference.get("family_size") == 20 and inference.get("family_wise_alpha") == .05 and
                 "Holm" in inference.get("multiplicity", ""),
                 "Diez candidatos frente a GRU32 y Persistence; Holm20 a α=0,05.", source)
    review.check("Repeticiones registradas", data["replications"].get("operative_R") == 6 and
                 data["replications"].get("seed_ids") == SEEDS,
                 "Seis semillas finales declaradas, no seis mercados.", source)
    evaluation = data["evaluation"]
    review.check("Periodo y exposición declarados", evaluation.get("calendar_days") == 608 and
                 evaluation.get("calibration_calendar_days") == 53 and
                 evaluation.get("previous_exposure_disclosed") is True and
                 evaluation.get("selection_or_retuning_from_final_scores_allowed") is False,
                 "El test es heredado de E1 y se prohíbe seleccionar desde los resultados finales.", source)
    review.note("Comparaciones primarias registradas", len(pairs))
    review.note("Alcance", "Registro y congelación de la campaña actual; calendario conocido desde E1.")
    validate_freeze(inputs, review)


def validate_route(inputs, review):
    item = find_json(inputs, lambda x: isinstance(x, dict) and
                     "hpo_fold_table" in x and "final_partition" in x,
                     "la auditoría de particiones F2a")
    data, source = item["data"], item["path"]
    folds = data["hpo_fold_table"]
    review.check("Tres folds internos", len(folds) == 3,
                 "Orígenes de selección internos; no se cuentan como tres mercados.", source)
    observed = [(row["inner_train"]["count"], row["validation"]["count"]) for row in folds]
    review.check("Tamaños internos", observed == [(253, 83), (435, 83), (527, 83)],
                 "Train 253/435/527 y validación83 en los tres folds.", source)
    for fold in folds:
        review.check(f"Orden temporal de {fold['fold_origin']}",
                     fold["inner_train"]["last"] < fold["validation"]["first"] and
                     fold.get("train_validation_objective_ID_intersection") == 0,
                     "Train precede a validación y sus IDs de objetivo son disjuntos.", source)
    parts, ids = data["final_partition"], {}
    for key, expected in (("training", 655), ("calibration", 53), ("evaluation", 608)):
        value = parts[key]
        sample = value.get("training_sample_IDs", value.get("sample_IDs", []))
        valid = (value.get("count") == expected and len(sample) == expected and
                 len(set(sample)) == expected and sample == sorted(sample) and
                 sample[0] == value.get("first") and sample[-1] == value.get("last"))
        review.check(f"Soporte {key}", valid,
                     f"{expected} identificadores únicos y ordenados, consistentes con los extremos.", source)
        ids[key] = set(sample)
    disjoint = all(not ids[a] & ids[b] for a, b in
                   (("training", "calibration"), ("training", "evaluation"), ("calibration", "evaluation")))
    review.check("IDs finales disjuntos", disjoint,
                 "No implica ausencia de historia causal compartida o independencia temporal.", source)
    dates = sorted(ids["evaluation"])
    continuous = len(dates) == 608 and all(date.fromisoformat(b) - date.fromisoformat(a) == timedelta(days=1)
                                           for a, b in zip(dates, dates[1:]))
    review.check("Calendario de evaluación", continuous and dates[0] == "2025-01-01" and
                 dates[-1] == "2026-08-31", "608 días sin comprimir huecos.", source)
    review.check("Separación de las etapas", parts["training"]["last"] < parts["calibration"]["first"] and
                 parts["calibration"]["last"] < parts["evaluation"]["first"],
                 "Fechas de resultados ordenadas; se conserva el puente causal del31/12.", source)
    review.note("Train / calibración / prueba", "655 / 53 / 608")
    review.note("Folds de selección", observed)
    review.note("Ruta", "F2a; controles causales de covariables complementarios. Una serie de mercado.")


def finite_number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("valor no finito")
    return number


def validate_training(inputs, review):
    item = find_table(inputs, {"model", "R_final", "final_seeds_json", "primary_RMSE_per_seed_json"},
                      "el registro de las trece redes")
    rows, source = item["rows"], item["path"]
    names = [row["model"] for row in rows]
    review.check("Identidad de las trece redes", len(rows) == 13 and set(names) == NEURAL,
                 "Diez candidatos y tres referencias neuronales de E1, sin duplicados.", source)
    for row in rows:
        try:
            seeds = json.loads(row["final_seeds_json"])
            values = [finite_number(x) for x in json.loads(row["primary_RMSE_per_seed_json"])]
            valid = int(row["R_final"]) == 6 and seeds == SEEDS and len(values) == 6 and min(values) >= 0
            aggregation = math.isclose(statistics.mean(values),
                                       finite_number(row["primary_RMSE_mean_individual_seeds"]),
                                       rel_tol=1e-12, abs_tol=1e-14)
            dispersion = math.isclose(statistics.stdev(values),
                                      finite_number(row["primary_RMSE_seed_SD"]),
                                      rel_tol=1e-10, abs_tol=1e-14)
            design = int(row["n_common_dates"]) == 608 and int(row["hpo_configurations"]) == 6 and \
                int(row["hpo_folds_per_configuration"]) == 3
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            review.check(f"Registro de {row.get('model', '?')}", False, str(exc), source)
            continue
        review.check(f"Semillas de {row['model']}", valid, "Seis valores finitos con las semillas registradas.", source)
        review.check(f"Agregación de {row['model']}", aggregation and dispersion,
                     "Media y DE muestral publicadas coinciden con las seis RMSE incluidas.", source)
        review.check(f"Diseño de {row['model']}", design,
                     "608 fechas, seis configuraciones y tres folds; no significa igual cómputo.", source)
    review.note("Redes en el registro", len(rows))
    review.note("Ajustes finales neuronales registrados", "13 × 6 = 78")
    review.note("Agregación comprobada", "Media y DE de RMSE ya publicadas; no se generan predicciones.")


def validate_nas(inputs, review):
    item = find_json(inputs, lambda x: isinstance(x, dict) and "architecture_search_applicability" in x,
                     "la adjudicación de arquitectura definida")
    data, source = item["data"], item["path"]
    value = data["architecture_search_applicability"]
    review.check("Aplicabilidad NAS", value.get("NAS1000_two_stage") == "no_aplica" and
                 value.get("NAS_is_current_pending_experiment") is False,
                 "La búsqueda de1000 diseños y dos fases es condicional a cuerpos indefinidos.", source)
    review.check("HPO distinto de NAS", value.get("six_HPO_slots_are_NAS") is False,
                 "Las seis plazas de hiperparámetros no se presentan como la búsqueda estructural.", source)
    review.check("Identidad TiDE", value.get("includes_TiDE") == "TIDE-D7-PUBLISHED-R0-v2",
                 "La adjudicación conserva la receta publicada r=0.", source)
    review.check("Sin redistribuir la rúbrica", data.get("weights_redistributed_for_NAS_non_applicability") is False and
                 data.get("total_weight_percent") == 100,
                 "La no aplicabilidad no elimina el criterio de implementación y rigor.", source)
    for candidate in inputs:
        values = candidate.get("data")
        if candidate["kind"] != "json_files":
            continue
        frozen_name = candidate["file"].name == "frozen_architecture_inputs.json"
        frozen_shape = isinstance(values, list) and len(values) == 13 and \
            all(isinstance(row, dict) and "model" in row for row in values)
        if frozen_name or frozen_shape:
            valid = isinstance(values, list) and len(values) == 13 and \
                all(isinstance(row, dict) and "model" in row for row in values)
            review.check("Cuerpos congelados incluidos", valid and {row["model"] for row in values} == NEURAL,
                         "Las trece identidades coinciden con el catálogo neuronal.", candidate["path"])
    review.note("NAS1000 / dos fases", "No aplica a los cuerpos actuales definidos.")
    review.note("Regla condicional", "Un cuerpo nuevo indefinido necesitaría su protocolo; no se ejecuta búsqueda aquí.")


def validate_appendix(inputs, review):
    documents = [item for item in inputs if item["kind"] == "json_files" and
                 isinstance(item["data"], dict) and "items" in item["data"] and
                 (item["file"].name.startswith("appendix_B") or
                  isinstance(item["data"]["items"], list) and
                  any(isinstance(row, dict) and re.fullmatch(r"B\d\d", str(row.get("id", "")))
                      for row in item["data"]["items"]))]
    if documents:
        item = documents[0]
        rows, source = item["data"]["items"], item["path"]
    else:
        item = find_table(inputs, {"id", "requirement", "status", "remaining_limit"}, "el Apéndice B literal")
        rows, source = item["rows"], item["path"]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        review.check("Esquema de ítems B", False, "items debe ser una lista de registros.", source)
        return
    identities = [row.get("id") for row in rows]
    expected = {f"B{i:02d}" for i in range(1, 22)}
    review.check("Veintiún ítems literales", len(rows) == 21 and set(identities) == expected,
                 "B01–B21 aparecen una vez cada uno.", source)
    states = {row.get("id"): str(row.get("status", "")).strip().replace(" ", "_") for row in rows}
    allowed = {"completo", "parcial", "no_cumple", "no_aplica"}
    review.check("Estados reconocidos", set(states.values()).issubset(allowed),
                 "Los estados se conservan; no se convierten en una nota.", source)
    review.check("B14 y B15 condicionales", states.get("B14") == states.get("B15") == "no_aplica",
                 "La NAS no aplica al catálogo actual de cuerpos definidos.", source)
    review.check("Evidencia y límites", all(row.get("requirement") and row.get("evidence") and
                                            row.get("remaining_limit") for row in rows),
                 "Cada ítem conserva requisito, evidencia y límite explícitos.", source)
    counts = Counter(states.values())
    if documents and isinstance(item["data"].get("counts"), dict):
        declared = item["data"]["counts"]
        review.check("Recuentos consistentes", all(counts.get(key, 0) == declared.get(key, 0) for key in allowed),
                     "Los recuentos coinciden con los estados incluidos; no son porcentajes de nota.", source)
    if documents:
        for table in inputs:
            if table["kind"] == "tables" and {"id", "requirement", "status", "remaining_limit"}.issubset(table["fields"]):
                csv_rows = {row["id"]: row for row in table["rows"]}
                same = len(table["rows"]) == 21 and all(
                    key in csv_rows and all(str(csv_rows[key].get(field)) == str(row.get(field))
                                           for field in table["fields"])
                    for row in rows for key in [row["id"]])
                review.check("CSV y JSON del Apéndice B", same,
                             "Los21 registros coinciden entre sus dos formatos.", table["path"])
    review.note("Ítems de reproducibilidad", len(rows))
    review.note("Estados conservados", dict(sorted(counts.items())))
    review.note("Interpretación", "Completar la lista no acredita todas sus condiciones ni calcula una calificación.")


VALIDATORS = {"D04": validate_preregistration, "D05": validate_route,
              "D07": validate_training, "D09": validate_nas, "D11": validate_appendix}


def manifest_entries(value):
    if isinstance(value, dict):
        value = value.get("files", value)
        if isinstance(value, dict):
            return [{"path": name, **(entry if isinstance(entry, dict) else {"sha256": entry})}
                    for name, entry in value.items()]
    if not isinstance(value, list):
        raise PackageError("El manifiesto debe contener una lista o mapa de archivos.")
    return value


def verify_integrity(root, package, inputs, review):
    expected = [{"path": item["path"], **{key: item[key] for key in ("sha256", "bytes") if key in item}}
                for item in inputs if "sha256" in item or "bytes" in item]
    manifests = [package["manifest_path"]] if package.get("manifest_path") else [
        name for name in ("manifest.json", "source_manifest.json") if (root / name).is_file()]
    for name in manifests:
        expected.extend(manifest_entries(load_json(local_path(root, name))))
    seen = set()
    for entry in expected:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise PackageError("Entrada de manifiesto sin path válido.")
        path = local_path(root, entry["path"])
        identity = (entry["path"], entry.get("sha256"), entry.get("bytes"))
        if identity in seen:
            continue
        seen.add(identity)
        if entry.get("sha256") is not None:
            review.check("SHA256 de " + entry["path"], digest(path) == entry["sha256"],
                         "Identidad del archivo incluido frente al manifiesto.", entry["path"])
        if entry.get("bytes") is not None:
            review.check("Bytes de " + entry["path"], path.stat().st_size == entry["bytes"],
                         "Tamaño frente al inventario incluido.", entry["path"])


def load_package(root, package_path):
    package = load_json(local_path(root, package_path))
    if not isinstance(package, dict) or package.get("id") not in IDS:
        raise PackageError("package.json debe identificar D04,D05,D07,D09 oD11.")
    if not isinstance(package.get("title"), str) or not package["title"]:
        raise PackageError("Falta el título del paquete.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(package.get("source_commit", ""))):
        raise PackageError("source_commit debe ser el SHA de40 caracteres del repositorio fuente.")
    inputs, seen = [], set()
    for kind in ("documents", "tables", "json_files"):
        entries = package.get(kind, [])
        if not isinstance(entries, list):
            raise PackageError(f"{kind} debe ser una lista.")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise PackageError(f"Entrada inválida en {kind}.")
            name = entry["path"]
            if name in seen:
                raise PackageError(f"Ruta declarada más de una vez: {name}")
            seen.add(name)
            path = local_path(root, name)
            item = {**entry, "title": entry.get("title", path.name), "kind": kind,
                    "file": path, "actual_sha256": digest(path), "actual_bytes": path.stat().st_size}
            if kind == "tables":
                item["fields"], item["rows"] = load_csv(path)
            elif kind == "json_files":
                item["data"] = load_json(path)
            else:
                item["text"] = path.read_text(encoding="utf-8-sig")
            inputs.append(item)
    if not inputs:
        raise PackageError("El paquete no declara documentos, tablas o JSON incluidos.")
    return package, inputs


def text(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def escaped(value):
    return html.escape(text(value), quote=True)


def inline_markdown(value, document_path, root, links):
    value = value.replace("\x00", "")
    tokens = []

    def token(content):
        tokens.append(content)
        return f"\x00{len(tokens)-1}\x00"

    def link(label, target, image=False):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            return token(f'<span title="{escaped(target)}">{escaped(label)}</span>')
        candidate = (PurePosixPath(document_path).parent / unquote(parsed.path)).as_posix()
        try:
            path = local_path(root, candidate)
        except PackageError:
            return token(escaped(label))
        relative = path.relative_to(root).as_posix()
        if image:
            mime = mimetypes.guess_type(path.name)[0]
            if mime and mime.startswith("image/"):
                uri = "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")
                return token(f'<figure><img src="{uri}" alt="{escaped(label)}"><figcaption>{escaped(label)}</figcaption></figure>')
            return token(escaped(label))
        destination = links.get(relative)
        if destination:
            return token(f'<a href="{escaped(destination)}">{escaped(label)}</a>')
        return token(escaped(label))

    value = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: link(m[1], m[2], True), value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: link(m[1], m[2]), value)
    def download(match):
        if " <" not in match[1]:
            return token(escaped(match[1]))
        label, target = match[1].rsplit(" <", 1)
        return link(label, target.removesuffix(">"))

    value = re.sub(r"\{download\}`([^`]+)`", download, value)
    value = re.sub(r"\{(?:ref|doc)\}`([^`]+)`", lambda m: m[1].split(" <", 1)[0], value)
    value = re.sub(r"\{cite(?::[a-z]+)?\}`([^`]+)`", lambda m: "[" + m[1] + "]", value)
    value = html.escape(value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
    return re.sub(r"\x00(\d+)\x00", lambda m: tokens[int(m[1])], value)


def markdown_document(body, document_path, root, links, prefix):
    lines, output, index = body.splitlines(), [], 0
    inline = lambda value: inline_markdown(value, document_path, root, links)
    while index < len(lines):
        line = lines[index].strip()
        if not line or re.fullmatch(r"\([^)]*\)=", line):
            index += 1
            continue
        fence = re.match(r"^(`{3,}|~{3,})(.*)$", line)
        if fence:
            marker, label = fence.groups()
            block, index = [], index + 1
            while index < len(lines) and lines[index].strip() != marker:
                block.append(lines[index]); index += 1
            if label.startswith("{dropdown}"):
                output.append("<details><summary>" + escaped(label[len("{dropdown}"):].strip()) + "</summary>" +
                              markdown_document("\n".join(block), document_path, root, links, prefix) + "</details>")
            elif label.startswith(("{note}", "{tip}", "{important}", "{warning}")):
                output.append('<aside>' + markdown_document("\n".join(block), document_path, root, links, prefix) + '</aside>')
            else:
                output.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = min(len(heading[1]) + 1, 6)
            slug = re.sub(r"[^\w-]+", "-", heading[2].lower()).strip("-")
            output.append(f'<h{level} id="{escaped(prefix + "-" + slug)}">{inline(heading[2])}</h{level}>')
            index += 1
            continue
        if index + 1 < len(lines) and "|" in line and re.fullmatch(r"[\s|:\-]+", lines[index + 1]):
            headers = [cell.strip() for cell in line.strip("|").split("|")]
            records, index = [], index + 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                values = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                records.append(values); index += 1
            output.append('<div class="table-scroll"><table><thead><tr>' +
                          ''.join('<th>' + inline(cell) + '</th>' for cell in headers) + '</tr></thead><tbody>' +
                          ''.join('<tr>' + ''.join('<td>' + inline(cell) + '</td>' for cell in values) + '</tr>'
                                  for values in records) + '</tbody></table></div>')
            continue
        listing = re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)(.+)$", line)
        if listing:
            values = []
            while index < len(lines):
                match = re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.+)$", lines[index])
                if not match:
                    break
                values.append(match[1]); index += 1
            output.append('<ul>' + ''.join('<li>' + inline(value) + '</li>' for value in values) + '</ul>')
            continue
        if re.fullmatch(r"[-*_]{3,}", line):
            output.append('<hr>'); index += 1; continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(
                r"^\s*(?:#{1,6}\s|`{3,}|~{3,}|[-*+]\s|\d+[.)]\s|\([^)]*\)=)", lines[index]):
            if index + 1 < len(lines) and "|" in lines[index] and re.fullmatch(r"[\s|:\-]+", lines[index + 1]):
                break
            paragraph.append(lines[index].strip()); index += 1
        output.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    return '\n'.join(output)


CSS = """
:root{color-scheme:light;--ink:#17304b;--line:#d8e2ec;--accent:#0b5c91}
*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:1250px;margin:auto;padding:28px}section{background:white;border:1px solid var(--line);border-radius:9px;padding:24px;margin:22px 0}
h1,h2,h3,h4{line-height:1.25;color:#123553}h1{font-size:2rem}h2{margin-top:0}h3,h4{margin-top:26px}
a{color:var(--accent)}nav{background:#e9f0f6;padding:16px;border-radius:8px}nav ul{columns:2;padding-left:22px}nav li{break-inside:avoid;margin:6px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f9;padding:15px;border-radius:6px;font-size:.85rem}code{font-size:.9em}
.table-scroll{overflow:auto;max-height:600px;border:1px solid var(--line);margin:14px 0}table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--line);min-width:110px;max-width:450px;overflow-wrap:anywhere}
th{position:sticky;top:0;background:#e8f0f7;z-index:1}tbody tr:nth-child(even){background:#f7fafc}
input[type=search]{width:100%;padding:11px;border:1px solid #a6bccd;border-radius:5px;font:inherit}
.ok{color:#165d34}.bad{color:#9f2630}.subtle{font-size:.9rem;color:#49627b}.exports{display:flex;gap:16px;flex-wrap:wrap}details,aside{padding:12px;background:#f5f8fb;margin:12px 0}summary{cursor:pointer;font-weight:600}
img{max-width:100%;height:auto}figure{margin:18px 0}figcaption{font-size:.9rem}.stamp{font-family:monospace;overflow-wrap:anywhere}
@media(max-width:650px){main{padding:12px}section{padding:15px}nav ul{columns:1}}
@media print{body{background:white}main{max-width:none;padding:0}section{border:0;break-inside:avoid}input,nav{display:none}.table-scroll{overflow:visible;max-height:none}th{position:static}}
"""
JS = """
document.querySelectorAll('[data-filter]').forEach(function(input){
 input.addEventListener('input',function(){var q=input.value.toLocaleLowerCase();
 document.querySelectorAll('#'+input.dataset.filter+' tbody tr').forEach(function(row){
 row.hidden=!row.textContent.toLocaleLowerCase().includes(q);});});});
"""


def render_table(fields, rows, table_id, limit):
    visible = rows[:limit]
    return (f'<label for="filter-{table_id}">Buscar en las filas mostradas</label>'
            f'<input id="filter-{table_id}" type="search" data-filter="{table_id}" placeholder="Modelo, estado o texto">'
            f'<p class="subtle">{len(rows)} filas incluidas; {len(visible)} mostradas. La descarga CSV conserva todas las filas.</p>'
            f'<div class="table-scroll"><table id="{table_id}"><thead><tr>' +
            ''.join('<th>' + escaped(field) + '</th>' for field in fields) + '</tr></thead><tbody>' +
            ''.join('<tr>' + ''.join('<td>' + escaped(row.get(field, '')) + '</td>' for field in fields) + '</tr>'
                    for row in visible) + '</tbody></table></div>')


def generate(root, package, inputs, review, output, table_limit):
    output.mkdir(parents=True, exist_ok=True)
    links = {}
    inventory = []
    for index, item in enumerate(inputs):
        destination = output / "export" / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["file"], destination)
        export = "export/" + quote(item["path"], safe="/")
        links[item["file"].relative_to(root).as_posix()] = export
        item["export"] = export
        item["anchor"] = f"input-{index}"
        inventory.append({"title": item["title"], "path": item["path"], "kind": item["kind"],
                          "sha256": item["actual_sha256"], "bytes": item["actual_bytes"],
                          "export": export, **({"rows": len(item["rows"]), "columns": len(item["fields"])}
                                                if item["kind"] == "tables" else {})})
    for item in inputs:
        if item["kind"] == "documents":
            links[item["file"].relative_to(root).as_posix()] = "#" + item["anchor"]
    report = {"version": VERSION, "package_id": package["id"], "title": package["title"],
              "source_commit": package["source_commit"],
              "status": "verified" if review.passed else "validation_failed",
              "scope": "Local schema, identity and recorded-aggregation checks; original scientific states are preserved.",
              "python": sys.version.split()[0], "summary": review.summary,
              "checks": review.checks, "inputs": inventory,
              "operations": {"model_fits": 0, "model_inferences": 0, "predictive_scores_computed": 0,
                             "statistical_tests": 0, "bootstrap_draws": 0, "network_requests": 0},
              "calculated_grade": None}
    (output / "resultado.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    title = escaped(package["title"])
    nav = '<nav aria-label="Contenido"><strong>Contenido del informe</strong><ul><li><a href="#checks">Controles ejecutados</a></li>' + ''.join(
        f'<li><a href="#{item["anchor"]}">{escaped(item["title"])}</a></li>' for item in inputs) + '</ul></nav>'
    count = sum(row["correcto"] for row in review.checks)
    state = "Comprobaciones correctas" if review.passed else "Hay comprobaciones que no coinciden"
    sections = [f'<header><p>{escaped(package["id"])} · Informe independiente</p><h1>{title}</h1>'
                f'<p class="stamp">Fuente: {escaped(package["source_commit"])}</p></header>', nav,
                '<section><h2>Resultado de la ejecución</h2>' +
                f'<p class="{"ok" if review.passed else "bad"}"><strong>{state}: {count}/{len(review.checks)}.</strong></p>' +
                '<p>Se comprobaron los archivos y la coherencia de los registros incluidos. Los estados científicos completos, parciales y no aplicables se conservan en sus tablas.</p>' +
                '<dl>' + ''.join('<dt><strong>' + escaped(row['concepto']) + '</strong></dt><dd>' + escaped(row['valor']) + '</dd>'
                                for row in review.summary) + '</dl><p><a href="resultado.json">Descargar resultado estructurado JSON</a></p></section>',
                '<section id="checks"><h2>Controles ejecutados</h2>' +
                '<div class="table-scroll"><table><thead><tr><th>Control</th><th>Resultado</th><th>Detalle</th><th>Fuente incluida</th></tr></thead><tbody>' +
                ''.join('<tr><td>' + escaped(row['control']) + '</td><td class="' + ('ok' if row['correcto'] else 'bad') + '">' +
                        ('Correcto' if row['correcto'] else 'No coincide') + '</td><td>' + escaped(row['detalle']) + '</td><td>' + escaped(row['fuente']) + '</td></tr>'
                        for row in review.checks) + '</tbody></table></div></section>']
    for index, item in enumerate(inputs):
        body = f'<section id="{item["anchor"]}"><h2>{escaped(item["title"])}</h2>' + \
            f'<p class="exports"><a href="{escaped(item["export"])}">Descargar archivo incluido</a></p>'
        if item["kind"] == "documents":
            body += markdown_document(item["text"], item["path"], root, links, item["anchor"])
        elif item["kind"] == "tables":
            body += render_table(item["fields"], item["rows"], f"table-{index}", table_limit)
        else:
            value = item["data"]
            if isinstance(value, dict):
                flat = [{"campo": key, "valor": text(val)} for key, val in value.items()
                        if not isinstance(val, (dict, list))]
                if flat:
                    body += render_table(["campo", "valor"], flat, f"table-{index}", table_limit)
                for key, val in value.items():
                    if isinstance(val, (dict, list)):
                        body += '<details><summary>' + escaped(key) + '</summary><pre>' + escaped(json.dumps(val, ensure_ascii=False, indent=2)) + '</pre></details>'
            else:
                body += '<details><summary>Datos JSON completos</summary><pre>' + escaped(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre></details>'
        sections.append(body + '</section>')
    (output / "informe.html").write_text('<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>' + title + '</title><style>' + CSS + '</style></head><body><main>' +
        '\n'.join(sections) + '</main><script>' + JS + '</script></body></html>\n', encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generar informe local del entregable E2 con Python estándar.")
    parser.add_argument("--output", default="output", help="Carpeta de salida; por defecto output junto a run.py.")
    parser.add_argument("--package", default="package.json", help="Descriptor relativo al directorio de run.py.")
    parser.add_argument("--table-limit", type=int, default=300, help="Filas mostradas por tabla; los CSV exportados se conservan completos.")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    if args.table_limit < 1:
        parser.error("--table-limit debe ser positivo.")
    try:
        package, inputs = load_package(root, args.package)
        output = Path(args.output)
        output = (root / output).resolve() if not output.is_absolute() else output.resolve()
        if output == root or root.is_relative_to(output) or any(item["file"].is_relative_to(output) for item in inputs):
            raise PackageError("La salida debe ser una carpeta separada de los archivos de entrada.")
        review = Review()
        verify_integrity(root, package, inputs, review)
        try:
            VALIDATORS[package["id"]](inputs, review)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            review.check("Esquema específico del entregable", False, str(exc))
        report = generate(root, package, inputs, review, output, args.table_limit)
        print(f"{package['id']}: {report['status']}")
        print(f"Informe: {output / 'informe.html'}")
        print(f"Datos y controles: {output / 'resultado.json'}")
        return 0 if review.passed else 2
    except (PackageError, OSError, UnicodeError) as exc:
        print(f"No se pudo generar el informe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
