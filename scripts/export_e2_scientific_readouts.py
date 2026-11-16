#!/usr/bin/env python3
"""Export new E2 delivery readouts using metadata only and preserving old bytes.

No fit, forward pass, bootstrap or new statistical test is executed. A.3 and
RQ2 are copied byte for byte from their 15-November snapshots. The 21 literal
Appendix B requirements retain their identities while evidence is updated.
An optional changing CPU status is sampled once and saved as a dated snapshot;
the active compute ledger and model weights are never consumed by this export.
Always use a NEW output directory, including when a running study later ends.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OLD = Path("book_e2/_static/e2_corrections_20261115")
SCIENCE = Path("book_e2/_static/e2_scientific_closure_20261116")
VERSION = "e2-scientific-delivery-readouts-v1"
CPU_PLAN = SCIENCE / "cpu_fixed_recipe_reproduction/plan_v2/analysis_plan.json"
BASE_PINS = {
    "delivery/a3_primary_contrasts_current.csv": "73be27eddc715dbe6f9fba5bdb575f1ece9a9658b7e1c5a96992397360186d62",
    "delivery/a3_primary_contrasts_current.json": "7e81c87b8140785708e147613609cf7ce38d50283639c1dc9daf33ccfda2fa1c",
    "delivery/appendix_B_current.csv": "45a86dc5cfe2d09a4f154bab9b912366c0478ace1750d30cbf49866f03b0f35a",
    "delivery/appendix_B_current.json": "ddb4398dd518fa0a5fddf9e72575a0f24454481a031541993a13545e6e2f7dae",
    "delivery/rq2_exploratory_current.csv": "d6956fbe2c6e1b88bf574a73363de549153aceb9a799c78dde28955018a267b9",
    "statistics/standardized_effects_20.csv": "720d2ce06bb2fc4c2ccda4f3dabddbc8b6fcb1488c7127f6491d1ab4671fca38",
    "statistics/rq2_primary_Holm10.csv": "d6956fbe2c6e1b88bf574a73363de549153aceb9a799c78dde28955018a267b9",
    "statistics/summary.json": "42240749c5a3f6daeaf7c17527488d865db834f9e216fa9f0955e5c3becf5542",
    "statistics/rq2_support.json": "21547dc521b790593056509d37ad46e93f0c7b253772c59219a54c6b951f8b82",
}
COMPLETE = {"B02", "B04", "B05", "B07", "B08", "B10", "B11", "B12", "B13", "B16", "B17", "B18", "B19", "B20"}
NA = {"B14", "B15"}
PARTIAL = {"B01", "B03", "B06", "B09", "B21"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_csv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, path, expected=None):
        path = Path(path)
        require(not path.is_absolute() and ".." not in path.parts, "Stable input must be repository-relative")
        absolute = ROOT / path
        require(absolute.is_file() and not absolute.is_symlink(), f"Missing regular input: {path}")
        raw = absolute.read_bytes()
        record = {"sha256": sha(raw), "bytes": len(raw)}
        require(expected is None or record["sha256"] == expected, f"Source hash changed: {path}")
        key = path.as_posix()
        require(key not in self.records or self.records[key] == record, f"Input changed while exporting: {path}")
        self.records[key] = record
        return raw

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def reverify(self):
        for path, record in list(self.records.items()):
            self.read(path, record["sha256"])


def validate_b_identity(original, updated):
    require(len(original) == len(updated) == 21, "Appendix B must retain exactly 21 requirements")
    expected_ids = [f"B{i:02d}" for i in range(1, 22)]
    require([row["id"] for row in original] == [row["id"] for row in updated] == expected_ids,
            "Appendix B item identities/order changed")
    for old, new in zip(original, updated):
        require(all(old[key] == new[key] for key in ("id", "requirement", "guide_pdf_page")),
                "A literal Appendix B requirement was edited")
    require(next(row for row in updated if row["id"] == "B21")["status"] == "parcial",
            "A CPU fixed-recipe repetition cannot close the whole historical campaign requirement B21")


def validate_primary_copy(original, exported):
    require(original == exported, "A.3 byte preservation failed; no primary field may change")
    rows = parse_csv(exported)
    require(len(rows) == 20 and len({(row["model"], row["baseline"]) for row in rows}) == 20,
            "A.3 must preserve the 20 original pairs")
    require(all(row["primary_p_and_decision_unchanged"] == "True" and
                row["original_result_replaced"] == "False" for row in rows),
            "Primary p-values/decisions were replaced")


def validate_inference_distinction(state):
    require(state["original_GPU_parity_plan"]["all78_parity_valid"] is False,
            "The failed original GPU parity plan cannot be promoted by a separate CPU study")
    require(state["posterior_CPU_test_IUT"]["full_CPU_IUT_executed"] is True,
            "Completed CPU RQ3 evidence must retain its separate identity")
    require(state["posterior_CPU_test_IUT"]["original_GPU_parity_gate_passed"] is False,
            "CPU IUT does not certify original GPU parity")


def univariate_metadata(inputs):
    directory = SCIENCE / "univariate_attributes"
    summary = inputs.json(directory / "result_v1/summary.json")
    plan = inputs.json(directory / "plan_v1/analysis_plan.json", summary["plan_sha256"])
    qa = inputs.json(directory / "qa_before_run.json", summary["QA_sha256"])
    receipt = inputs.json(directory / "result_v1/execution_receipt.json")
    manifest = inputs.json(directory / "result_v1/manifest.json")
    require(qa["approved_for_univariate_attribute_audit"] is True
            and qa["plan_sha256"] == receipt["plan_sha256"] == summary["plan_sha256"]
            and qa["script_sha256"] == receipt["script_sha256"] == summary["script_sha256"],
            "Univariate execution lacks matching prospective plan/QA identities")
    require(datetime.fromisoformat(plan["registered_at_utc"]) <= datetime.fromisoformat(qa["reviewed_at_utc"])
            < datetime.fromisoformat(receipt["completed_at_utc"]), "Univariate plan/QA chronology changed")
    require((summary["coordinates"], summary["scored_rows"], summary["OLS_fits"],
             summary["flagged_coordinate_cohorts"]) == (35, 105, 35, 0)
            and summary["suspicious_R2_threshold"] == 0.95
            and summary["all_availability_offsets_strictly_past"] is True
            and summary["calibration_or_test_used_for_fit"] is False,
            "Univariate finite screen or train-only scope changed")
    require(len(manifest["inputs"]) == 11, "Unexpected univariate source-input manifest")
    # Byte hashes only: do not load the NPZ, fit OLS, score attributes or rerun tests.
    for record in manifest["inputs"]:
        raw = inputs.read(Path(record["path"]), record["sha256"])
        require(len(raw) == record["bytes"], "Univariate source input size changed")
    for record in manifest["files"]:
        raw = inputs.read(directory / "result_v1" / record["path"], record["sha256"])
        require(len(raw) == record["bytes"], "Univariate result file size changed")
    require(manifest["analysis_plan_sha256"] == summary["plan_sha256"]
            and manifest["QA_sha256"] == summary["QA_sha256"], "Univariate manifest plan/QA differs")
    return {"summary": summary, "plan_path": (directory / "plan_v1/analysis_plan.json").as_posix(),
            "QA_path": (directory / "qa_before_run.json").as_posix(),
            "registered_plan_QA_commit": "ea60ec3c806a290f1c00deeec952d6fdae672308",
            "registration_commit_recorded_at_utc": "2026-11-16T02:48:23Z",
            "registration_reference_scope": "Pinned plan/QA blob hashes were checked against this recorded Git commit during export development; this stdlib exporter rechecks their file hashes and recorded chronology, without requiring Git or treating Git time as an independent timestamp authority.",
            "source_input_files_reverified": 11, "result_manifest_verified": True,
            "prospective_plan_and_QA_binding_verified": True,
            "scope": "All declared attribute-audit procedures executed; finite descriptive screen, not universal nonlinear/multivariate leakage exclusion."}


def cpu_status_snapshot(inputs, path):
    plan = inputs.json(CPU_PLAN)
    require(plan["HPO_repeated"] is False and plan["full_original_campaign_passed"] is False,
            "CPU plan acquired an unsupported full-campaign claim")
    # Only the adapter's identity is checked; no model source or weight is frozen/exported here.
    inputs.read(Path("scripts/e2_cpu_fixed_recipe_reproduction.py"), plan["script_sha256"])
    base_path = Path(plan["fixed_recipe_plan"]["path"]).resolve().relative_to(ROOT)
    base = inputs.json(base_path, plan["fixed_recipe_plan"]["sha256"])
    index = {job["id"]: job for job in base["jobs"]}
    require(len(index) == 81 and Counter(j["kind"] for j in index.values()) == {"neural": 78, "classical": 3},
            "CPU fixed-recipe job catalogue changed")
    state = {"scope": plan["scope"], "plan_path": CPU_PLAN.as_posix(),
        "plan_sha256": inputs.records[CPU_PLAN.as_posix()]["sha256"],
        "runtime_policy_declared_in_plan": plan["runtime_policy"],
        "planned_neural_fits": 78, "planned_classical_reconstructions": 3,
        "completed_neural_workers": 0, "completed_classical_workers": 0,
        "completed_scoring_stages": 0, "status": "not_observed_by_this_export",
        "CPU_study_completed": False, "original_CUDA_parity": False,
        "full_original_campaign_passed": False, "HPO_repeated": False,
        "NAS": "not_applicable_defined_bodies", "dynamic_status_sampled_once": True,
        "live_ledger_read_or_frozen": False}
    snapshots = {}
    if path is None:
        return state, snapshots
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "Missing CPU status snapshot input")
    raw = path.read_bytes()
    status = json.loads(raw)
    snapshots["cpu_status_at_export.json"] = raw
    jobs = status.get("jobs", [])
    require(len({job["id"] for job in jobs}) == len(jobs), "Duplicate CPU worker receipt in sampled status")
    require(status.get("original_CUDA_parity") is False and status.get("HPO_repeated") is False,
            "CPU status must retain its separate scientific scope")
    completed = []
    for job in jobs:
        require(job["CPU_study_plan_sha256"] == state["plan_sha256"], "CPU status binds another plan")
        require(job["id"] in index or job["id"] in ("CPU-freeze-raw-score", "CPU-primary20000"),
                "Unknown CPU worker identity")
        if job["status"] == "completed":
            require(job["returncode"] == 0, "Completed CPU worker has a failure return code")
            completed.append(job["id"])
    neural = sum(job_id in index and index[job_id]["kind"] == "neural" for job_id in completed)
    classical = sum(job_id in index and index[job_id]["kind"] == "classical" for job_id in completed)
    scoring = sum(job_id in ("CPU-freeze-raw-score", "CPU-primary20000") for job_id in completed)
    complete = status.get("status") == "completed_new_CPU_fixed_recipe_fit_cal_freeze_raw_test_A2_scope"
    require(not complete or (neural, classical, scoring) == (78, 3, 2), "CPU completion lacks all required worker receipts")
    state.update(status=status["status"], completed_neural_workers=neural,
        completed_classical_workers=classical, completed_scoring_stages=scoring,
        CPU_study_completed=complete, sampled_status_source=str(path),
        sampled_status_sha256=sha(raw), sampled_status_bytes=len(raw),
        sampled_at_utc=datetime.now(timezone.utc).isoformat(),
        status_snapshot_path="source_snapshots/cpu_status_at_export.json",
        claimed_charged_worker_seconds_at_snapshot=status.get("charged_worker_seconds"),
        worker_accounting_scope="status-reported sum of worker wall seconds; not the full project ledger or training-only time")
    identity_path = path.parent / "run_identity.json"
    if identity_path.is_file():
        identity_raw = identity_path.read_bytes()
        identity = json.loads(identity_raw)
        require(identity["plan_sha256"] == state["plan_sha256"] and identity["old_weights_or_predictions_used"] is False,
                "CPU run identity differs from its true-target plan")
        snapshots["cpu_run_identity_at_export.json"] = identity_raw
        state["run_identity_sha256"] = sha(identity_raw)
    completed_neural = next((index[job_id] for job_id in completed
                             if job_id in index and index[job_id]["kind"] == "neural"), None)
    if completed_neural is not None:
        metadata_path = path.parent / completed_neural["directory"] / "result.json"
        metadata_raw = metadata_path.read_bytes()
        metadata = json.loads(metadata_raw)
        require(metadata["status"] == "completed" and metadata["CPU_study_plan_sha256"] == state["plan_sha256"]
                and metadata["original_CUDA_reproduction_claimed"] is False
                and metadata["label_permutation_used"] is False, "Observed completed CPU fit has another scope")
        snapshots["sample_completed_CPU_worker_metadata.json"] = metadata_raw
        state["sample_observed_worker_runtime_evidence"] = {
            "source": str(metadata_path), "sha256": sha(metadata_raw),
            "observed_phase_backend_labels": sorted({phase["backend"] for phase in metadata["phase_records"].values()}),
            "training_label_end": metadata["training_label_end"],
            "scope": "One completed worker metadata sample, not independent attestation of every runtime package or all81workers; plan runtime settings remain declared policy."}
    return state, snapshots


def update_b(original, cpu):
    rows = copy.deepcopy(original)
    updates = {
      "B01": ("Protocolo y motor reducidos anteriores a los 170 HPO nuevos (206 diarios verificados) y a los 78 ajustes finales (90 diarios); δ/δg y enmiendas conservan su linaje.",
        "preregistration/summary.json;final_registration_chronology/audit_results/summary.json",
        "Los 61 resultados HPO cacheados y ajustes anteriores conservan otra historia. No se certifican todas las decisiones antes del primer entrenamiento del proyecto ni cegamiento del test E1; los tiempos registrados no son sellos externos independientes."),
      "B03": ("Todos los procedimientos declarados están ejecutados: entradas reales 655/53/608, 78 ports/390 condiciones de desarrollo, 36 controles causales clásicos, 35 OLS univariadas/105 resultados sin R²≥0,95, adversarial y 13 entrenamientos con etiquetas permutadas. La deriva AUC=0,95253867, p=0,001 se conserva, explica y trata.",
        "data_audit/audit_result.json;port_audit/readout.json;classical_causality/readout.json;univariate_attributes/result_v1/summary.json;univariate_attributes/plan_v1/analysis_plan.json;univariate_attributes/qa_before_run.json;controls/negative_summary.json;controls/adversarial_result.json;drift_explanation/treatment_evidence.json",
        "La aceptación de rendimiento cercano al azar bajo etiquetas permutadas no está establecida: iTransformer/TSMixer conservan ρ=0,52230/0,49253 y ganancias RMSE positivas; p conjuntos nominales=0,00740/0,04460 y Holm13=0,09619/0,53515. Cero alertas Holm no demuestra equivalencia al azar. La parcialidad corresponde a este criterio de aceptación, no a procedimientos pendientes ni a una demostración de fuga."),
      "B05": ("Completo en el pipeline auditado: las cohortes 655/53/608 mantienen transformaciones relativas, ventanas causales y escaladores ajustados sólo en train. Firmas de 78 ajustes y 81 escaladores cotejados; 390 condiciones del port conservan parámetros/buffers y los clásicos avanzan sin refits de coeficientes. Los 35 atributos usan información disponible hasta t−1.",
        "data_audit/audit_result.json;data_audit/physical_scaler_checks.json;port_audit/readout.json;classical_causality/readout.json;univariate_attributes/result_v1/summary.json",
        "Alcance: preparación train-only y predictores/consultas auditados, sin extenderlo a historias arbitrarias, otros ports o todas las plataformas. Calibrar intervalos en cal53, actualizar el estado causal de ARIMA y el ensemble posterior con identidad propia no equivalen a reajustar parámetros con test."),
      "B06": ("234 oportunidades: 231 completas, dos fallos conocidos y una desconocida; 61 cachés y 170 ajustes nuevos completos. Hay 180/231 desgloses de fase. Los 233 cargos de pared observados mutuamente excluyentes suman 25.833,898918 s; 1.860 s de asignación desconocida quedan separados.",
        "hpo_cost_reconciliation/summary.json;catalogue_charge_accounting/summary.json;catalogue_charge_accounting/catalogue_charge_opportunities_234.csv",
        "Mismas plazas no implica mismo presupuesto efectivo: SAE tiene tres fases, los clásicos otros dominios y los relojes/hardware difieren. Los 51 desgloses de fase faltantes sí conservan cargos originales. No se suman fases/callbacks/cargos anidados ni se interpreta la reserva desconocida como tiempo observado o el subtotal como todo el historial."),
      "B08": ("Piloto de cinco pares iTransformer/GRU32 en 53 fechas: R=6, α=0,0025, ganancia 2δ, potencia 0,122953 frente a 0,80 y MDE=0,000878674. Se justifica R=6 por recursos y se reporta la limitación conforme a la excepción explícita de la guía 9.2.",
        None,
        "Completo en el alcance literal de análisis y reporte con justificación/MDE. No se alcanzó potencia 0,80 ni se acredita potencia temporal o exacta de Holm para todos los candidatos; no autoriza ampliar R tras observar prueba."),
      "B11": ("Completo en el reporte de las Tablas A/B/C: 20 contrastes primarios con ganancias, IC y p exactos/ajustados; 20 d_z con IC entre semillas; RQ2 con diez pares y siete rechazos Holm10; 13 pruebas CPU de δg, 13 comparaciones de periodos originales y diez IUT CPU de RQ3. Las unidades, familias y supuestos están identificados.",
        "period_comparison/summary.json;cpu_test_iut/statistics/summary.json;cpu_test_iut/verification_report.json",
        "Reporte completo no significa adecuación favorable: se conservan resultados adversos, dos tolerancias condicionales, falta de equivalencia, puerta antigua de paridad CUDA no aprobada y adecuación conjunta A100 no establecida. Calibración no es validación HPO independiente; los d_z entre semillas no sustituyen incertidumbre temporal."),
      "B12": ("Completo para las familias declaradas: los 20 pares primarios, α y Holm por procedimiento constan en el protocolo reducido anterior a los 170 HPO nuevos y a los 78 ajustes finales. Efectos/RQ2, brecha CPU, periodos e IUT RQ3 tienen planes, familias y fechas propios anteriores a sus cálculos y se etiquetan como suplementos posteriores.",
        "period_comparison/analysis_plan.json;cpu_test_iut/analysis_plan.json;cpu_test_iut/statistics/summary.json",
        "El alcance no atribuye preregistro a todo el historial o a los 61 HPO cacheados. Conocer el test E1 limita confirmación prospectiva; no convierte por sí solo una familia explícita, corregida y fechada en infracción del literal. No se retrofechan los suplementos ni se infiere equivalencia desde no rechazo."),
      "B13": ("Completo en la estimación operativa reportada conforme a §12.1: entropía ordinal como predictibilidad intrínseca y referencias empíricas de ensembles, con referencia ideal y ensemble uniforme adverso conservados. El proxy ajustado sólo en cal53 obtiene RMSE=0,003689532 y γ=0,287574, IC [−1,49210; 2,37128].",
        None,
        "La incertidumbre amplia, el orden favorable en 6.588/10.000 réplicas y las dos ganancias no concluyentes Holm2 limitan la interpretación del margen. Se reconoce la referencia y γ reportados sin exigir resultado favorable ni identificar techo óptimo, riesgo Bayes o piso de ruido."),
      "B18": ("El ciclo E2 documentado conserva 81 predictores/escaladores/calibradores/reglas fijados y el registro de no refit/retuning tras scoring. La cronología recuperada vincula 78 pesos evaluados con sus ajustes originales anteriores al fin reportado de admisión.",
        "final_registration_chronology/audit_results/summary.json",
        "Completo en el ciclo documentado; el test E1 ya era conocido. El fin de admisión no prueba la fecha del primer acceso externo ni un sello independiente de cegamiento. Los suplementos posteriores tienen otra identidad."),
    }
    for row in rows:
        item = row["id"]
        row["status"] = "completo" if item in COMPLETE else ("no_aplica" if item in NA else "parcial")
        if item in updates:
            evidence, paths, limit = updates[item]
            row.update(evidence=evidence, remaining_limit=limit)
            if paths:
                current_paths = ";".join((SCIENCE / path).as_posix() for path in paths.split(";"))
                if item in {"B01", "B03", "B05", "B06", "B18"}:
                    row["evidence_paths"] = current_paths
                else:
                    retained = [path for path in row["evidence_paths"].split(";")
                                if not path.endswith("/delivery/inference_evidence_state.json")]
                    row["evidence_paths"] = ";".join(retained) + ";" + current_paths
        if item == "B21":
            status_word = ("completada en su alcance CPU" if cpu["CPU_study_completed"]
                           else "no completada en este corte; estado del recibo: " + cpu["status"])
            row["evidence"] = ("Doce ZIP del 15/11 conservan sus propios puntos de entrada y recibos. Nueva repetición de recetas finales desde cero: "
                f"{cpu['completed_neural_workers']}/78 trabajadores neuronales y {cpu['completed_classical_workers']}/3 clásicos, "
                f"{cpu['completed_scoring_stages']}/2 etapas de scoring/análisis; {status_word}. "
                "Las auditorías, estadísticas e inferencias CPU ya terminadas conservan manifiestos propios.")
            row["evidence_paths"] += ";" + CPU_PLAN.as_posix()
            if "status_snapshot_path" in cpu:
                row["evidence_paths"] += ";" + cpu["status_snapshot_path"]
            row["remaining_limit"] = ("La repetición CPU de recetas finales no repite HPO, robustez ni toda la campaña histórica y no prueba paridad CUDA. "
                "Los comandos predeterminados de informes no repiten todos los ajustes; B21 permanece parcial. Cada versión de ZIP necesita sus propios recibos y no hereda los de otro snapshot.")
    validate_b_identity(original, rows)
    return rows


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_bytes(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(output, cpu_status):
    started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
    require(not output.exists(), "Existing evidence is immutable: choose a NEW --output directory")
    inputs = Inputs()
    script_sha = sha(inputs.read(Path(__file__).resolve().relative_to(ROOT)))
    raw_base = {relative: inputs.read(OLD / relative, expected) for relative, expected in BASE_PINS.items()}
    old_b = parse_csv(raw_base["delivery/appendix_B_current.csv"])
    old_b_json = json.loads(raw_base["delivery/appendix_B_current.json"])
    stats = json.loads(raw_base["statistics/summary.json"])
    require((stats["effect_pairs"], stats["effects_with_estimable_CI"], stats["RQ2_primary_pairs"], stats["RQ2_rejections_Holm10"]) == (20, 20, 10, 7),
            "Preserved effects/RQ2 family changed")
    sources = {name: inputs.json(SCIENCE / relative) for name, relative in {
        "real_data": "data_audit/audit_result.json", "ports": "port_audit/readout.json",
        "classical": "classical_causality/readout.json", "negative": "controls/negative_summary.json",
        "adversarial": "controls/adversarial_result.json", "drift": "drift_explanation/summary.json",
        "drift_treatment": "drift_explanation/treatment_evidence.json", "period": "period_comparison/summary.json",
        "CPU_IUT": "cpu_test_iut/statistics/summary.json", "CPU_IUT_verification": "cpu_test_iut/verification_report.json",
        "HPO_chronology": "preregistration/summary.json", "final_chronology": "final_registration_chronology/audit_results/summary.json",
        "HPO_phase_cost": "hpo_cost_reconciliation/summary.json", "catalogue_charges": "catalogue_charge_accounting/summary.json"}.items()}
    old_inference = inputs.json(OLD / "delivery/inference_evidence_state.json")
    diagnostic = inputs.json(OLD / "cpu_diagnostic/statistics/results/summary.json")
    univariate = univariate_metadata(inputs)
    operational_ceiling = {
        "reference_report": inputs.json(Path("book_e2/_static/e2_error_ceiling_final_audit_20261114/ceiling_gamma_references.json")),
        "calibration_ensemble": inputs.json(Path("book_e2/_static/e2_calibration_ensemble_20261114/summary.json")),
        "scope": "Allowed operational intrinsic-predictability/ensemble references and reported gamma with uncertainty; no Bayes risk or optimal ceiling identified."}
    inputs.read(Path("book_e2/_static/e2_benchmark_final_20261113/pilot_power_MDE.json"))
    inputs.read(Path("book_e2/_static/e2_repro_prereg_current_v1/power_design_readout.json"))
    cpu, snapshots = cpu_status_snapshot(inputs, cpu_status)
    require(sources["real_data"]["control_counts"] == {"PASS": 12, "PARTIAL": 3, "NA": 1}
            and sources["real_data"]["actual_cohorts_audited"] == {"training": 655, "calibration": 53, "evaluation": 608},
            "Real-cohort audit scope changed")
    require((sources["ports"]["predictors"], sources["ports"]["conditions_passed"], sources["classical"]["branch_checks"]) == (78, 390, 36),
            "Bounded implementation audit scope changed")
    require(len(sources["negative"]["rows"]) == 13 and not any(r["flag"] for r in sources["negative"]["rows"]),
            "Negative-control family disposition changed")
    require(sources["drift"]["original_drift_flag_retained"] is True
            and sources["adversarial"]["reject_alpha05"] is True, "Observed distribution shift was suppressed")
    require((sources["period"]["neural_period_tests"], sources["period"]["t_rejections_Holm13"]) == (13, 6),
            "Original-period comparison scope changed")
    require(sources["CPU_IUT_verification"]["unique_test_model_seed_dates"] == 47424
            and sources["CPU_IUT_verification"]["all_cost_IUT_Holm_p_equal_one"] is True, "CPU IUT verification changed")
    require(sources["catalogue_charges"]["observed_charge_receipts_union"] == 233
            and sources["catalogue_charges"]["unmeasured_conservative_allocation_seconds_not_added"] == 1860,
            "Exclusive catalogue charge account changed")
    new_b = update_b(old_b, cpu)
    counts = dict(Counter(row["status"] for row in new_b))
    require(counts == {"parcial": 5, "completo": 14, "no_aplica": 2}
            and {row["id"] for row in new_b if row["status"] == "parcial"} == PARTIAL,
            "Appendix B final literal scope counts changed")
    inference = {"version": VERSION, "status": "separate_completed_posterior_CPU_studies_original_GPU_gate_still_failed",
        "original_GPU_parity_plan": old_inference,
        "posterior_CPU_train_calibration_diagnostic": diagnostic,
        "original_saved_period_comparison": sources["period"],
        "posterior_CPU_test_IUT": sources["CPU_IUT"],
        "posterior_CPU_test_IUT_verification": sources["CPU_IUT_verification"],
        "new_true_target_CPU_fixed_recipe_repetition": cpu,
        "original_A2_or_RQ1_replaced": False, "new_original_GPU_parity_claim": False,
        "generic_parity_or_adequacy_flag_created": False,
        "scope": "Completed CPU RQ3 is a separate posterior study; failed historical 714/GPU-parity plan remains failed. Calibration is not independent HPO validation."}
    validate_inference_distinction(inference)
    state = {"version": VERSION, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "appendix_B_counts": {**counts, "no_cumple": 0}, "percentage_complete_or_grade_reported": False,
        "primary_pairs": 20, "A3_15November_bytes_preserved": True,
        "primary_p_and_decisions_unchanged": True, "A2_replaced": False, "RQ1_replaced": False,
        "effects_and_RQ2_preserved": stats,
        "real_cohort_audit": {"status": sources["real_data"]["status"], "control_counts": sources["real_data"]["control_counts"],
            "actual_cohorts_audited": sources["real_data"]["actual_cohorts_audited"]},
        "bounded_development_port_audit": sources["ports"], "bounded_classical_causality_audit": sources["classical"],
        "univariate_attribute_audit": univariate,
        "declared_audit_procedures_executed": True,
        "B03_remaining_acceptance_criterion": "Closeness to chance under permuted labels is not established; absence of Holm13 alerts is not an equivalence result. This is an acceptance limitation, not an unexecuted procedure or proof of leakage.",
        "negative_training_controls": {"family_size": 13, "Holm_flags": 0,
            "close_to_chance_acceptance_established": False,
            "retained_nominal_warning_rows": [row for row in sources["negative"]["rows"]
                                              if row["model"] in ("iTransformer", "TSMixer")],
            "scope": "One registered block-label permutation fit per model; no flag does not certify absence of leakage."},
        "adversarial_drift": {"AUC": sources["adversarial"]["AUC_OOF"],
            "p_block_permutation_with_refit": sources["adversarial"]["p_block_permutation_with_refit"],
            "flag_retained": True, "explanation": sources["drift"], "treatment": sources["drift_treatment"]},
        "original_period_comparison": sources["period"], "posterior_CPU_test_IUT": sources["CPU_IUT"],
        "HPO_chronology": sources["HPO_chronology"], "final_registration_chronology": sources["final_chronology"],
        "HPO_phase_breakdown_separate_scope": sources["HPO_phase_cost"],
        "original_reduced_catalogue_observed_charges_separate_scope": sources["catalogue_charges"],
        "operational_performance_ceiling_and_gamma": operational_ceiling,
        "new_true_target_CPU_fixed_recipe_repetition": cpu,
        "B08_reporting_scope": "Analysis, resource justification and MDE reported under guide 9.2; target power 0.80 was not achieved.",
        "B05_reporting_scope": "Complete within the audited train-only pipeline and predictors/queries; no universal claim across arbitrary histories or platforms.",
        "B11_reporting_scope": "Reported tests and effects are complete; adverse adequacy outcomes and the failed original CUDA gate are retained as scientific results/limits, not missing reporting.",
        "B12_reporting_scope": "Complete declaration/control of primary families and separately dated exploratory supplements; inherited E1 exposure limits prospective confirmation, not the existence of explicit families.",
        "B13_reporting_scope": "Complete operational intrinsic-predictability/ensemble reference and uncertain gamma report under allowed guide12.1 references; no favorable outcome or Bayes identification required or claimed.",
        "B21_scope": "Partial even after a successful new CPU fixed-recipe repetition; all historical HPO and campaign stages are not automated/repeated.",
        "operations_this_export": {"new_model_fits": 0, "new_inference": False, "new_predictive_scores": 0,
                                  "new_hypothesis_tests": 0, "new_bootstrap_draws": 0, "active_ledger_read": False}}
    b_json = {k: copy.deepcopy(v) for k, v in old_b_json.items()
              if k not in ("items", "counts", "correction_evidence_state", "operations", "version", "reviewed_source_commit_scope")}
    b_json.update(version=VERSION, counts={**counts, "no_cumple": 0}, items=new_b,
        base_B_preserved=(OLD / "delivery/appendix_B_current.json").as_posix(),
        base_B_version=old_b_json["version"],
        base_B_reviewed_source_commit=old_b_json.get("reviewed_source_commit"),
        reviewed_source_commit=None,
        previous_versions_preserved=old_b_json.get("previous_versions_preserved", []) +
            [(OLD / "delivery/appendix_B_current.csv").as_posix(), (OLD / "delivery/appendix_B_current.json").as_posix()],
        reviewed_source_commit_scope="New metadata export; stable working-tree inputs are SHA-bound in source_provenance; no historical source is rewritten.",
        correction_evidence_state=state, operations=state["operations_this_export"])
    inputs.reverify()
    output.mkdir(parents=True)
    copies = {"a3_primary_contrasts_current.csv": "delivery/a3_primary_contrasts_current.csv",
        "a3_primary_contrasts_current.json": "delivery/a3_primary_contrasts_current.json",
        "a3_current.csv": "delivery/a3_primary_contrasts_current.csv", "a3_current.json": "delivery/a3_primary_contrasts_current.json",
        "rq2_exploratory_current.csv": "delivery/rq2_exploratory_current.csv",
        "rq2_primary_Holm10.csv": "statistics/rq2_primary_Holm10.csv", "rq2_support.json": "statistics/rq2_support.json",
        "standardized_effects_20.csv": "statistics/standardized_effects_20.csv", "effects_rq2_original_summary.json": "statistics/summary.json"}
    for destination, source in copies.items():
        write_bytes(output / destination, raw_base[source])
    validate_primary_copy(raw_base["delivery/a3_primary_contrasts_current.csv"],
                          (output / "a3_primary_contrasts_current.csv").read_bytes())
    write_csv(output / "appendix_B_current.csv", new_b)
    write_json(output / "appendix_B_current.json", b_json)
    write_bytes(output / "B_current.csv", (output / "appendix_B_current.csv").read_bytes())
    write_bytes(output / "B_current.json", (output / "appendix_B_current.json").read_bytes())
    write_json(output / "current_readout_state.json", state)
    write_json(output / "inference_evidence_state.json", inference)
    if snapshots:
        (output / "source_snapshots").mkdir()
        for name, raw in snapshots.items():
            write_bytes(output / "source_snapshots" / name, raw)
    provenance = {"version": VERSION, "stable_inputs": inputs.records,
        "copied_15November_bytes": {name: {"original": (OLD / relative).as_posix(), "sha256": sha(raw_base[relative]),
                                         "bytes": len(raw_base[relative])} for name, relative in copies.items()},
        "sampled_dynamic_CPU_status": cpu,
        "univariate_attribute_plan_QA_and_11_source_inputs": univariate,
        "snapshot_hashes": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in snapshots.items()},
        "A3_original_embedded_source_paths_preserved": True,
        "original_files_modified": False, "active_compute_ledger_read_or_frozen": False,
        "weights_or_model_source_exported": False}
    write_json(output / "source_provenance.json", provenance)
    readme = ("# Lecturas científicas actuales de E2\n\n"
        "A.3 conserva exactamente los bytes del 15/11: 20 pares, 20 efectos con IC y decisiones/p originales. "
        "RQ2 conserva diez pares y siete rechazos Holm10. Los alias a3_current y B_current sólo facilitan el acceso. "
        "Los JSON A.3 conservan deliberadamente sus rutas de procedencia históricas.\n\n"
        "Apéndice B conserva los 21 requisitos literales: catorce completos, cinco parciales y dos no aplica (NAS). "
        "B08 está completo en el reporte del análisis, excepción de recursos y MDE; la potencia 0,80 no se alcanzó. "
        "B05 reconoce el pipeline auditado; B11/B12/B13 reconocen el reporte de pruebas, familias y referencia operativa/γ, conservando resultados adversos e incertidumbre. "
        "B01/B03/B06/B09/B21 son parciales. B03 conserva un criterio concreto sin aceptación demostrada: rendimiento cercano al azar en controles permutados; cero alertas Holm no prueba equivalencia.\n\n"
        "El auditor real verifica 655/53/608 (12 PASS, 3 PARTIAL y 1 NA); los 78 ports de desarrollo, 36 controles clásicos, "
        "35 OLS univariadas/105 resultados, 13 controles negativos y el adversarial están ejecutados y tienen ámbitos separados. "
        "Los once insumos del univariado y su plan/QA anterior se verifican por SHA sin repetir cálculos. La deriva detectada se explica y se conserva. "
        "Los 13 contrastes de periodos originales producen seis rechazos t-Holm13; calibración no es validación HPO independiente. "
        "La RQ3 CPU posterior usa 47.424 forwards y diez IUT, p Holm=1; la puerta antigua de paridad CUDA sigue fallida.\n\n"
        "Los 233 cargos observados del catálogo reducido suman 25.833,898918 s; 1.860 s reservados desconocidos quedan separados. "
        "Los 180 desgloses de fases y 51 ausencias no se suman a cargos anidados. No se afirma coste histórico completo ni GPU activa.\n\n"
        f"La nueva repetición CPU de recetas finales se observa como {cpu['status']}: "
        f"{cpu['completed_neural_workers']}/78 trabajadores neuronales y {cpu['completed_classical_workers']}/3 clásicos. "
        "Su snapshot conservaSHA/fecha y no congela el diario activo. B21 sigue parcial: no se repite HPO ni toda la campaña.\n\n"
        "Este exportador sólo une metadatos y copia archivos; no ajusta modelos, predice, remuestrea ni recalcula contrastes. "
        "Cada nueva exportación exige un directorio nuevo. source_provenance.json y manifest.json permiten verificar las fuentes y los bytes.\n")
    write_bytes(output / "README.md", readme.encode())
    write_json(output / "execution_receipt.json", {"version": VERSION, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_metadata_export_seconds": time.monotonic() - clock,
        "script_sha256": script_sha, "stable_inputs_reverified": len(inputs.records),
        "source_provenance_sha256": sha((output / "source_provenance.json").read_bytes()),
        "old_primary_bytes_unchanged": True, "B_requirement_identities_preserved": True,
        **state["operations_this_export"], "compute_charge": "metadata-only export; no scientific calculation or ledger mutation"})
    write_json(output / "manifest.json", {"version": VERSION, "files": {
        p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
        for p in sorted(output.rglob("*")) if p.is_file()}, "original_files_modified": False})
    print(json.dumps({"output": str(output), "B_counts": counts, "primary_pairs": 20,
        "CPU_repetition_status": cpu["status"], "CPU_completed_neural": cpu["completed_neural_workers"],
        "new_statistics": 0, "stable_input_files": len(inputs.records)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-status", type=Path)
    args = parser.parse_args()
    export(args.output, args.cpu_status)


if __name__ == "__main__":
    main()
