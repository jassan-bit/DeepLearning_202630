"""Join existing E2 corrective results into new delivery readouts, without statistics.

Use a fresh --output directory for each later state. Original B, A2 and primary
contrasts are read only. Failed/partial frozen-inference outputs are never used
as completed gap or RQ3 evidence. This exporter does not reserve compute time.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VERSION = "e2-corrections-delivery-readouts-v1"
BASE = "book_e2/_static/e2_agregacion_entrega_current_v1"
CORRECTIONS = "book_e2/_static/e2_corrections_20261115"
PACKAGE_SNAPSHOT = "0bdbc42bd724c6ffbbb1dec67564327adedb9883"
EFFECT_FIELDS = ("primary_standardized_effect_size", "primary_standardized_effect_CI95", "primary_effect_size_status")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), "Do not create a fictitious empty result table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.records = {}

    def relative(self, path):
        path = Path(path).resolve()
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return "external_input/" + path.name

    def raw(self, path, expected=None, role="existing published or corrective evidence"):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        require(path.is_file() and not path.is_symlink(), "Required regular input missing: " + self.relative(path))
        raw = path.read_bytes()
        sha = checksum(raw)
        require(expected is None or sha == expected, "Input hash differs: " + self.relative(path))
        key = self.relative(path)
        self.records[key] = {"path": key, "bytes": len(raw), "sha256": sha, "role": role}
        return raw

    def json(self, path, **kwargs):
        return json.loads(self.raw(path, **kwargs))

    def csv(self, path, **kwargs):
        return list(csv.DictReader(io.StringIO(self.raw(path, **kwargs).decode("utf-8"))))

    def manifest_files(self, directory, names):
        directory = Path(directory)
        manifest = self.json(directory / "manifest.json")
        for name in names:
            require(Path(name).name == name, "Only explicit result files are admitted here")
            record = manifest["files"][name]
            raw = self.raw(directory / name, expected=record["sha256"], role="result bound by its producer manifest")
            require(len(raw) == record["bytes"], "Manifest byte count differs: " + name)
        return manifest


def keyed(rows, left="model", right="baseline"):
    result = {(r[left], r[right]): r for r in rows}
    require(len(result) == len(rows), "Duplicate contrast key")
    return result


def finite_fields(row, names):
    try:
        return all(row.get(k) not in (None, "") and math.isfinite(float(row[k])) for k in names)
    except (TypeError, ValueError):
        return False


def validate_bound_report(inputs, path, stage):
    report = inputs.json(path)
    require(report.get("status") == "passed" and report.get("stage") == stage,
            "A completed passed inference report is required")
    for name, record in report["all_output_files"].items():
        require(Path(name).name == name, "Unsafe inference report binding")
        raw = inputs.raw(path.parent / name, expected=record["sha256"], role="output bound by completed inference report")
        require(len(raw) == record["bytes"], "Inference output byte count differs")
    require(report.get("states_unchanged") is True and report.get("new_model_fits") == 0,
            "Frozen state/no-fit guarantee is missing")
    if stage == "predictions":
        require(report.get("predictors_completed") == report.get("calibration_checks_passed") == 78,
                "All 78 predictor parity checks must pass")
        require(report.get("calibration_absolute_tolerance") == 1e-7
                and report.get("maximum_calibration_absolute_difference", math.inf) <= 1e-7,
                "The fixed calibration parity tolerance must not be relaxed")
        require(report.get("train_prediction_rows") == 78 * 655
                and report.get("calibration_prediction_rows") == 78 * 53,
                "Complete current training/calibration prediction rows required")
        agreements = inputs.csv(path.parent / "calibration_agreement.csv")
        require(len(agreements) == len({(r["model"], r["seed"]) for r in agreements}) == 78
                and all(r["passed"].lower() == "true" and float(r["absolute_tolerance"]) == 1e-7
                        and math.isfinite(float(r["max_absolute_prediction_difference"]))
                        and float(r["max_absolute_prediction_difference"]) <= 1e-7 for r in agreements),
                "All 78 actual calibration agreement rows must pass the fixed gate")
    return report


def matching_report(inputs, directory, name, expected):
    paths = [p for p in directory.rglob(name) if p.is_file() and checksum(p.read_bytes()) == expected]
    require(len(paths) == 1, "A unique SHA-bound completed report is required: " + name)
    return paths[0]


def inference_state(inputs, directory):
    """Admit completed evidence only; preserve failed attempts as provenance."""
    base = inputs.root / CORRECTIONS
    frozen = base / "frozen_inference"
    state = {"status": "PARTIAL", "gap_reporting_valid": False, "RQ3_reporting_valid": False,
             "all78_parity_valid": False, "reason": "No completed statistical summary with valid78-predictor parity",
             "historical_failed_attempts": []}
    for path in sorted(frozen.rglob("failure_report.json")):
        failure = inputs.json(path, role="preserved failed inference/timing attempt")
        state["historical_failed_attempts"].append({"path": inputs.relative(path),
            "status": failure.get("status"), "stage": failure.get("stage"),
            "error_type": failure.get("error_type"), "error": failure.get("error")})
    plan_path = base / "inference_statistics/statistical_plan.json"
    if plan_path.is_file():
        inputs.raw(plan_path, role="post-benchmark statistical plan, not proof of execution")
    if not (directory / "summary.json").is_file():
        return state, {}
    try:
        summary = inputs.json(directory / "summary.json")
        receipt = inputs.json(directory / "execution_receipt.json")
        inputs.manifest_files(directory, ["summary.json", "execution_receipt.json", "delta_g_Holm13.csv",
                                          "validation_test_Holm13.csv"])
        report_path = matching_report(inputs, frozen, "inference_report.json", receipt["inference_report_sha256"])
        inference = validate_bound_report(inputs, report_path, "predictions")
        gaps = inputs.csv(directory / "delta_g_Holm13.csv")
        consistency = inputs.csv(directory / "validation_test_Holm13.csv")
        require(summary.get("gap_tests") == summary.get("period_consistency_tests") == 13
                and len(gaps) == len(consistency) == 13, "Complete 13-model result tables required")
        require(len({r["model"] for r in gaps}) == len({r["model"] for r in consistency}) == 13,
                "The 13-model contrast tables have duplicate IDs")
        require(summary.get("primary_A2_changed") is False and summary.get("primary_RQ1_changed") is False,
                "Corrective statistics must preserve primary A2/RQ1")
        require(all(finite_fields(r, ("p_raw", "p_Holm13", "CI95_lower", "CI95_upper")) for r in gaps)
                and all(finite_fields(r, ("p_raw", "p_Holm13", "CI95_lower", "CI95_upper")) for r in consistency),
                "Finite p/CI required for literal B11 completion")
        state.update(status="VERIFIED_FOR_GAP_REPORTING", gap_reporting_valid=True, all78_parity_valid=True,
                     reason="Completed 13-model gap/period reporting bound to passed 78-predictor parity",
                     summary=summary, inference_report=inputs.relative(report_path),
                     inference_spec_sha256=inference["spec_sha256"])
        admitted = {"delta_g_Holm13.csv": directory / "delta_g_Holm13.csv",
                    "validation_test_Holm13.csv": directory / "validation_test_Holm13.csv"}
        # RQ3 is a separately specified question, not an extra universal B11 condition.
        try:
            inputs.manifest_files(directory, ["RQ3_cpu_IUT_Holm10.csv"])
            latency_path = matching_report(inputs, frozen, "latency_report.json", receipt["latency_report_sha256"])
            latency = validate_bound_report(inputs, latency_path, "latency")
            require(latency["spec_sha256"] == inference["spec_sha256"], "Inference/timing specification mismatch")
            require(latency.get("raw_timed_operations") == 12000 and latency.get("blocks") == 100,
                    "Complete paired CPU timing design required")
            rq3 = inputs.csv(directory / "RQ3_cpu_IUT_Holm10.csv")
            require(summary.get("RQ3_tests") == len(rq3) == 10
                    and len({r["candidate"] for r in rq3}) == 10
                    and all(finite_fields(r, ("p_IUT", "p_IUT_Holm10")) for r in rq3), "Complete valid RQ3 results required")
            state.update(RQ3_reporting_valid=True, latency_report=inputs.relative(latency_path))
            admitted["RQ3_cpu_IUT_Holm10.csv"] = directory / "RQ3_cpu_IUT_Holm10.csv"
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            state["RQ3_reason"] = str(exc)
        return state, admitted
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        state["reason"] = "Completed gap evidence was not admitted: " + str(exc)
        return state, {}


def load_inputs(root, inference_directory):
    inputs = Inputs(root)
    original_b = inputs.json(BASE + "/appendix_B_current.json")
    b_csv = inputs.csv(BASE + "/appendix_B_current.csv")
    require(b_csv == [{k: str(v) for k, v in r.items()} for r in original_b["items"]], "Original B CSV/JSON differ")
    require([r["id"] for r in b_csv] == [f"B{i:02d}" for i in range(1, 22)], "Exactly 21 literal B items required")
    a3 = inputs.csv(BASE + "/a3_primary_contrasts_readout.csv")
    require(len(a3) == len(keyed(a3)) == 20, "Exactly 20 original primary contrasts required")
    stats = root / CORRECTIONS / "statistics"
    inputs.manifest_files(stats, ["summary.json", "execution_receipt.json", "analysis_plan.json",
                                 "standardized_effects_20.csv", "rq2_primary_Holm10.csv", "rq2_support.json"])
    effect_summary = inputs.json(stats / "summary.json")
    effects = inputs.csv(stats / "standardized_effects_20.csv")
    rq2 = inputs.csv(stats / "rq2_primary_Holm10.csv")
    require(len(effects) == effect_summary["effect_pairs"] == 20
            and set(keyed(effects, "candidate", "reference")) == set(keyed(a3)), "Effect/primary pair mismatch")
    require(len(rq2) == effect_summary["RQ2_primary_pairs"] == 10, "Complete recorded RQ2 family required")
    hpo = root / CORRECTIONS / "hpo"
    inputs.manifest_files(hpo, ["summary.json", "execution_receipt.json", "opportunities_234.csv",
                               "configurations_78.csv", "model_reconciliation_13.csv"])
    hpo_summary = inputs.json(hpo / "summary.json")
    require(hpo_summary["opportunities"] == 234 and hpo_summary["configuration_count"] == 78
            and hpo_summary["recorded_winners_reproduced"] == 13, "Current HPO summary scope differs")
    data = inputs.json(CORRECTIONS + "/data_audit/audit_result.json")
    require(dict(Counter(r["status"] for r in data["controls"])) == data["control_counts"], "Data-audit status counts differ")
    require(not data["control_counts"].get("FAIL"), "A failed data criterion requires explicit review before export")
    readiness_path = root / CORRECTIONS / "controls/readiness_report.json"
    readiness = inputs.json(readiness_path) if readiness_path.is_file() else None
    frozen_state, admitted = inference_state(inputs, inference_directory)
    package_raw = None
    package_path = "book_e2/_static/entregables_independientes_v1/execution_report.json"
    try:
        package_raw = subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "show",
                                               PACKAGE_SNAPSHOT + ":" + package_path], stderr=subprocess.DEVNULL)
        package = json.loads(package_raw)
        require(package.get("archives_executed") == package.get("passed") == 12,
                "The cited published package execution did not pass all 12")
    except (subprocess.CalledProcessError, ValueError, KeyError):
        package_raw = None
    return inputs, original_b, a3, effects, rq2, effect_summary, hpo_summary, data, readiness, frozen_state, admitted, package_raw


def build_tables(original_b, original_a3, effects, effect_summary, hpo, data, frozen):
    b = deepcopy(original_b)
    b.update(version=VERSION, scope="Current 21 literal items with corrective evidence; documentary export, no grade or percentage.")
    b["base_B_version"] = original_b["version"]
    b["reviewed_source_commit_scope"] = "Inherited base B metadata; corrective working-tree inputs are separately bound by the new manifest hashes."
    b["percentage_of_guide_complete_reported"] = False
    by_id = {r["id"]: r for r in b["items"]}
    def update(key, evidence, paths, remaining, status="parcial"):
        by_id[key].update(status=status, evidence=evidence, evidence_paths=";".join(paths), remaining_limit=remaining)
    droot, sroot, hroot = CORRECTIONS + "/data_audit", CORRECTIONS + "/statistics", CORRECTIONS + "/hpo"
    counts = data["control_counts"]
    cohorts = ", ".join(f"{k}={v}" for k, v in data["actual_cohorts_audited"].items())
    update("B03", f"La auditoría real de datos informa {counts.get('PASS', 0)} PASS, {counts.get('PARTIAL', 0)} PARTIAL y {counts.get('NA', 0)} NA, sin fallos observados; cohortes numéricas verificadas: {cohorts}. Reconstruye σ7, ventanas causales, firmas y escaladores. Las guardas unitarias se identifican aparte.",
           [droot + "/audit_result.json", droot + "/controls.csv", CORRECTIONS + "/controls/readiness_report.json"],
           "Los PASS se limitan a sus criterios. El contenido OHLCV de evaluación 608 y los controles reales adversarial/permutación siguen sujetos a sus recibos; no se certifica ausencia global de fuga ni cada port desde fixtures o hashes.")
    update("B05", "La preparación final común se contrastó con datos reales: 655 train / 53 cal, aritmética del escalado, transformación relativa y firmas de los 78 ajustes. Se cotejaron 81 JSON de escaladores físicos con el freeze original; cambiar observaciones posteriores al 31/10 no modifica train o sus escalas.",
           [droot + "/audit_result.json", droot + "/physical_scaler_checks.json", droot + "/reconstructed_training_scaler.json"],
           "La verificación común no ejecuta ni certifica todo comportamiento aprendido de los 13 ports. Las transformaciones de evaluación requieren su entrada real; el ensemble posterior de calibración conserva otra identidad.")
    update("B06", f"La recuperación conserva {hpo['opportunities']} oportunidades HPO: {hpo['completed']} completadas, {hpo['known_failed']} fallidas y {hpo['outcome_unknown']} desconocida; 78 configuraciones y 13 ganadores verificados. Distingue {hpo['reused_outcomes']} resultados reutilizados, {hpo['measured_phase_time_receipts']} recibos de fase medida y {hpo['campaign_budget_charge_receipts']} cargos de campaña, con sus fuentes.",
           [hroot + "/summary.json", hroot + "/opportunities_234.csv", hroot + "/configurations_78.csv", hroot + "/model_reconciliation_13.csv"],
           "Las 13 redes comparten plazas, pero SAE conserva tres fases y los clásicos otros dominios. Duración de fase, cargo de campaña y coste del ajuste cacheado tienen ámbitos distintos; no se suman para afirmar un presupuesto homogéneo ni se convierten cargos desconocidos en tiempo medido.")
    gap_valid = frozen["gap_reporting_valid"]
    if gap_valid:
        gap_text = "Los 13 contrastes de brecha operacional y 13 diagnósticos de consistencia entre periodos están reportados con p, Holm13 e IC, ligados a 78 verificaciones de paridad y estados congelados."
        remaining = "Completo en el alcance literal de reporte de las Tablas A/B/C; no exige p favorables ni certifica ausencia universal de sobreajuste. La calibración 53 ya se utilizó y la inferencia t es condicional a periodos y semillas; drift puede producir diferencias. La descomposición mixta no es una condición universal de este literal."
    else:
        gap_text = "La prueba comparable de brecha actual no tiene una ejecución completa admitida con 78 paridades; no se utilizan CSV parciales del intento fallido."
        remaining = "δg tenía registro previo, pero falta completar su contraste comparable actual con un recibo válido de 78 predictores. Los efectos d_z no lo sustituyen. No se añade una descomposición mixta como obligación universal del literal B11 ni se interpreta el fallo de paridad como sobreajuste."
    update("B11", f"Los 20 pares conservan ganancias, IC y p primarios sin cambios; se añaden 20 d_z pareados con IC marginales entre seis semillas. RQ2 exploratoria reporta {effect_summary['RQ2_primary_pairs']} pares y {effect_summary['RQ2_rejections_Holm10']} rechazos de erosión con Holm10. " + gap_text,
           [CORRECTIONS + "/delivery/a3_primary_contrasts_current.csv", sroot + "/standardized_effects_20.csv",
            sroot + "/rq2_primary_Holm10.csv", CORRECTIONS + "/delivery/inference_evidence_state.json",
            "book_e2/sections/49_diagnosticos_f2a.md"], remaining, "completo" if gap_valid else "parcial")
    later = ("Las familias posteriores de brecha Holm13 y consistencia Holm13 tienen planes y resultados propios. "
             + ("RQ3 IUT-Holm10 conserva su ejecución separada validada." if frozen["RQ3_reporting_valid"]
                else "RQ3 mantiene su estado específico, sin atribuirle una ejecución validada.")
             if gap_valid else "Las familias de brecha/consistencia/RQ3 tienen plan posterior; un plan o un intento parcial no acredita su ejecución completa.")
    update("B12", "La primaria conserva 20 pares, δ y Holm por cada uno de los tres procedimientos originales, congelados antes del scoring. El cálculo posterior de d_z/IC y RQ2-Holm10 se identifica como exploratorio y tiene un plan anterior a su propio cálculo. " + later,
           ["book_e2/sections/09_protocolo_registrado.md", sroot + "/analysis_plan.json", sroot + "/execution_receipt.json",
            CORRECTIONS + "/delivery/inference_evidence_state.json"],
           "No se retrofecha la nueva evidencia: conocer el test heredado o registrar antes del cálculo no equivale a preregistrar antes de todos los ajustes. No hay potencia universal demostrada para estas familias; un no rechazo no establece equivalencia ni estabilidad frente a drift.")
    update("B21", "Doce entregables tienen puntos de entrada independientes y la versión pública citada conserva la ejecución de sus 12 ZIP extraídos. Los nuevos readouts y recibos pueden incorporarse en sus áreas de corrección sin cambiar A2 o la familia primaria; código y entornos permanecen identificados.",
           [CORRECTIONS + "/delivery/standalone_execution_snapshot.json", "scripts/package_e2_standalone_deliverables.py",
            "book_e2/sections/08_reproducibilidad.md"],
           "Empaquetar o ejecutar por defecto los informes no reproduce todos los ajustes históricos ni todo el entrenamiento con un solo comando. B21 permanece parcial; cada nueva versión de los ZIP necesita su propio recibo de ejecución y no hereda automáticamente el de otro snapshot.")
    require(all(a["requirement"] == c["requirement"] and a["id"] == c["id"]
                for a, c in zip(b["items"], original_b["items"])), "Literal B text changed")
    require(by_id["B14"]["status"] == by_id["B15"]["status"] == "no_aplica", "Defined-architecture NAS condition changed")
    b["counts"] = {key: sum(r["status"] == key for r in b["items"])
                   for key in ("completo", "parcial", "no_cumple", "no_aplica")}
    b["operations"] = {"new_model_fits": 0, "new_model_inference_calls": 0, "new_predictive_scores": 0,
                       "new_hypothesis_tests": 0, "new_bootstrap_draws": 0, "original_scientific_files_modified": False}
    effect_by_key = keyed(effects, "candidate", "reference")
    joined = []
    for old in original_a3:
        effect = effect_by_key[old["model"], old["baseline"]]
        row = dict(old)
        effect_ci = ("[" + effect["dz_CI95_lower"] + ", " + effect["dz_CI95_upper"] + "]"
                     if effect["dz_CI95_lower"] and effect["dz_CI95_upper"] else "")
        row.update({"primary_standardized_effect_size": effect["paired_Cohen_dz"],
                    "primary_standardized_effect_CI95": effect_ci,
                    "primary_effect_size_status": "posthoc paired Cohen dz, conditional on six optimization seeds",
                    "paired_Cohen_dz": effect["paired_Cohen_dz"], "dz_CI95_lower": effect["dz_CI95_lower"],
                    "dz_CI95_upper": effect["dz_CI95_upper"], "SD_seed_gains": effect["SD_seed_gains"],
                    "mean_gain_RMSE_for_effect": effect["mean_gain_RMSE"], "effect_status": effect["status"],
                    "effect_bootstrap_draws": effect["bootstrap_draws"], "effect_valid_draws": effect["valid_draws"],
                    "effect_invalid_draws": effect["invalid_draws"], "effect_bootstrap_seed": effect["bootstrap_seed"],
                    "effect_n_seed_pairs": effect["n_seed_pairs"], "effect_scope": effect["scope"],
                    "effect_CI_method": effect["CI_method"],
                    "effect_source": sroot + "/standardized_effects_20.csv", "primary_p_and_decision_unchanged": "True"})
        for field in EFFECT_FIELDS:
            row["previous_" + field] = old[field]
        require(all(row[key] == old[key] for key in old if key not in EFFECT_FIELDS),
                "An original primary field was modified")
        require(math.isclose(float(old["gain"]), float(effect["mean_gain_RMSE"]), rel_tol=0, abs_tol=1e-15)
                and old["delta"] == effect["delta_RMSE"], "Effect and primary estimands/margins differ")
        joined.append(row)
    return b, joined


def export(root, output, inference_directory, preview=False):
    loaded = load_inputs(root, inference_directory)
    inputs, original_b, original_a3, effects, rq2, stats, hpo, data, readiness, frozen, admitted, package_raw = loaded
    b, joined = build_tables(original_b, original_a3, effects, stats, hpo, data, frozen)
    state = {"version": VERSION, "appendix_B_counts": b["counts"], "primary_pairs": len(joined),
             "primary_raw_and_Holm_p_preserved": True, "all_original_primary_fields_preserved_except_filled_effect_placeholders": True,
             "effects": stats, "HPO": hpo,
             "data_audit": {"status": data["status"], "control_counts": data["control_counts"],
                            "actual_cohorts_audited": data["actual_cohorts_audited"],
                            "data_ready_for_bounded_controls": data["data_ready_for_bounded_controls"]},
             "control_readiness": {"status": readiness.get("status") if readiness else "not_available",
                                   "new_fits_executed": readiness.get("operations", {}).get("new_model_fits", 0) if readiness else None},
             "frozen_inference_statistics": frozen, "percentage_complete_or_grade_reported": False,
             "limits": ["Post-benchmark documentary joins do not restore prospective preregistration or blindness.",
                        "One market series; initialization uncertainty is not uncertainty across markets or all partitions.",
                        "The 12.3 percent pilot power is conditional on its two representatives/alternative, not universal power.",
                        "dz and its seed CI do not replace the temporal practical-margin contrast or prove gain greater than delta.",
                        "RQ2 rejection means erosion of advantage; non-rejection is not robustness/equivalence.",
                        "Period differences can reflect drift; operational gap tests do not certify universal absence of overfitting.",
                        "Default standalone ZIP execution is not full historical market training replay."]}
    if preview:
        return state
    require(not output.exists(), "Use a fresh output directory; preserve earlier delivery readouts")
    output.mkdir(parents=True)
    if package_raw is not None:
        (output / "standalone_execution_snapshot.json").write_bytes(package_raw)
    else:
        raise ValueError("The historical standalone execution source cited by B21 is unavailable")
    b["correction_evidence_state"] = state
    b["base_B_preserved"] = BASE + "/appendix_B_current.json"
    write_csv(output / "appendix_B_current.csv", b["items"])
    write_json(output / "appendix_B_current.json", b)
    write_csv(output / "a3_primary_contrasts_current.csv", joined)
    write_json(output / "a3_primary_contrasts_current.json", {
        "version": VERSION, "rows": len(joined), "columns": len(joined[0]), "items": joined,
        "base_A3": BASE + "/a3_primary_contrasts_readout.csv",
        "original_primary_fields_preserved": [k for k in original_a3[0] if k not in EFFECT_FIELDS],
        "effect_definition": "mean of six raw paired seed gains divided by their sample SD(ddof1), not (gain-delta)/SD",
        "effect_CI_scope": "Existing 20k paired-seed percentile bootstrap conditional on 608 dates; undefined draws retained in counts",
        "no_new_primary_p_or_test": True, "no_temporal_effect_CI_claimed": True})
    (output / "rq2_exploratory_current.csv").write_bytes(inputs.raw(CORRECTIONS + "/statistics/rq2_primary_Holm10.csv"))
    for name, source in admitted.items():
        (output / name).write_bytes(inputs.raw(source))
    write_json(output / "inference_evidence_state.json", frozen)
    write_json(output / "current_readout_state.json", state)
    write_json(output / "source_provenance.json", list(inputs.records.values()))
    write_json(output / "execution_receipt.json", {
        "version": VERSION, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "program_sha256": checksum(Path(__file__).read_bytes()), "source_file_count": len(inputs.records),
        "standalone_execution_snapshot_source_commit": PACKAGE_SNAPSHOT,
        "standalone_execution_snapshot_sha256": checksum(package_raw),
        "metadata_join_only": True, "new_model_fits": 0, "new_model_inference": False,
        "new_hypothesis_tests": 0, "new_bootstrap_draws": 0, "primary_A2_or_RQ1_modified": False})
    (output / "README.md").write_text(
        "# Lectura actual de reproducibilidad y contrastes\n\n"
        "Este directorio reúne valores existentes sin ejecutar estadísticas. Los 21 literales de B y los veinte p primarios quedan conservados.\n\n"
        "A3 añade d_z e IC de semillas publicados posteriormente: la incertidumbre es condicional a seis pares y al calendario. No cambia Holm20, la ganancia o la decisión primaria. RQ2 conserva otra familia exploratoria.\n\n"
        "Los resultados de brecha/RQ3 sólo se incorporan si existen resumen final, recibos/hash coherentes y las 78 paridades a tolerancia 1e−7. Un fallo o CSV parcial queda como procedencia, nunca como resultado completo. RQ3 no se convierte en una condición universal de B11.\n\n"
        "El recibo de los 12 ZIP se conserva como snapshot de su versión pública; no es replay integral de entrenamiento. No se calcula porcentaje de cumplimiento o nota.\n\n"
        "Repetición documental en una carpeta NUEVA:\n\n"
        "```sh\npython scripts/export_e2_corrections_readouts.py --output /tmp/e2-corrections-readouts-new\n```\n",
        encoding="utf-8")
    output_files = {p.name: {"bytes": p.stat().st_size, "sha256": checksum(p.read_bytes())}
                    for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", {"version": VERSION, "files": output_files,
        "inputs": list(inputs.records.values()), "scope": "documentary exact-copy joins, not new scientific computation",
        "original_static_artifacts_modified": False, "primary_A2_replaced": False,
        "new_model_fits": 0, "new_statistics": False})
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / CORRECTIONS / "delivery")
    parser.add_argument("--inference-statistics-dir", type=Path, default=ROOT / CORRECTIONS / "inference_statistics/results")
    parser.add_argument("--preview", action="store_true", help="Read/validate current source state without creating output files")
    args = parser.parse_args(argv)
    state = export(args.root.resolve(), args.output.resolve(), args.inference_statistics_dir.resolve(), args.preview)
    print(json.dumps({"B_counts": state["appendix_B_counts"], "primary_pairs": state["primary_pairs"],
                      "primary_p_preserved": state["primary_raw_and_Holm_p_preserved"],
                      "gap_reporting_valid": state["frozen_inference_statistics"]["gap_reporting_valid"],
                      "RQ3_reporting_valid": state["frozen_inference_statistics"]["RQ3_reporting_valid"],
                      "preview_only": args.preview}, ensure_ascii=False))


if __name__ == "__main__":
    main()
