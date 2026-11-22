"""Assemble auditable references, guide matrix and source hashes without training."""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def span(path, name):
    tree = ast.parse(path.read_text())
    matches = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    assert len(matches) == 1, (path, name)
    return matches[0].lineno, matches[0].end_lineno


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=HERE.parents[1] / "e3_input_only_candidate_20261121")
    args = parser.parse_args()
    wall, cpu = time.perf_counter(), time.process_time()
    candidate = args.candidate_root.resolve()
    identity = json.loads((candidate / "CANDIDATE_IDENTITY.json").read_text())
    old_tests = json.loads((candidate / "qa/CANDIDATE_SOFTWARE_TESTS.json").read_text())
    verdict = json.loads((candidate / "docs/R1_VERDICT.json").read_text())
    new_checks = json.loads((HERE / "EXECUTION_RECEIPT.json").read_text())
    assert identity["candidate_id"] == "input-only-overlap-mse-1"
    assert old_tests["status"] == "PASS" and old_tests["tests"] == 8
    assert verdict["decision"] == "CONTINUAR" and verdict["substantive_adaptation_supported"] is True
    assert new_checks["status"] == "PASS" and new_checks["new_checks"] == 6
    assert new_checks["script_sha256"] == sha(HERE / "defense_exercises.py")
    assert new_checks["candidate_identity_sha256"] == sha(candidate / "CANDIDATE_IDENTITY.json")
    assert {k: sha(candidate / k) for k in identity["runtime_source_sha256"]} == identity["runtime_source_sha256"]
    assert new_checks["candidate_sources_verified"] == identity["runtime_source_sha256"]
    assert not any([new_checks["training_optimizer_steps"], new_checks["new_forecasting_fits"],
                    new_checks["market_arrays_loaded"], new_checks["market_predictions_computed"],
                    new_checks["student_defense_observed"], new_checks["teacher_acceptance_claimed"],
                    new_checks["R3_accredited"], new_checks["novel_designs_added"]])

    specs = [
        ("C01", "A=Σ(r−m6)²/7; RMS and relative logprices", "runtime/vendor/seoi_v5/data.py", "known_geometry", "Population geometry; no future return is an input"),
        ("C02", "s=max(RMS,epsilon*r_fit); volume log1p", "runtime/vendor/seoi_v5/data.py", "raw_features", "Origin causal, not prefix causal normalization"),
        ("C03", "Frozen fit channel mean/SD and reference_RMS", "runtime/vendor/seoi_v5/data.py", "fit_preprocessor", "Explicitly supplied fit rows; source/date admission separate"),
        ("C04", "Use frozen coordinate statistics; FP32 X", "runtime/vendor/seoi_v5/data.py", "transform", "Input-only uses local coordinates"),
        ("C05", "U scales from label identity and frozen means", "runtime/candidate.py", "fit_inputs", "Labels used only for supplied fit scales; no forward U input"),
        ("C06", "Own candidate wrapper and identity", "runtime/candidate.py", "InputOnlyOverlapModel", "Selects historical input_only_mse; no new architecture or training"),
        ("C07", "Causal convolution5→8,k3; tanh; LSTM8→12", "runtime/vendor/seoi_v5/frozen_original.py", "CausalEncoder", "Inherited own encoder; old additive-head properties not inherited"),
        ("C08", "Attention; ordered atoms; simplex; mu=sum p*hypot(sqrtA,sqrtc*u)", "runtime/vendor/seoi_v5/model.py", "forward_components", "Input-only factor=1, no output local-scale restitution"),
        ("C09", "MSE in global fit Y units; lambda_CRPS=0", "runtime/vendor/seoi_v5/model.py", "loss_components", "MSE does not identify latent distribution"),
    ]
    correspondence = []
    for key, equation, relative, symbol, limitation in specs:
        first, last = span(candidate / relative, symbol)
        correspondence.append({"id": key, "formulation": equation, "candidate_file": relative,
                               "symbol": symbol, "first_line": first, "last_line": last,
                               "sha256": sha(candidate / relative), "scope": limitation})
    with (HERE / "code_correspondence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=correspondence[0].keys())
        writer.writeheader(); writer.writerows(correspondence)
    write_json(HERE / "code_correspondence.json", correspondence)

    # Exact short quotations from the user-supplied original PDF (not a new rule).
    quotes = [
        {"id": "G-R1", "pages": "7", "section": "6 R1", "quote": "su diferencia frente al trabajo cercano es de mecanismo, de función de pérdida o de esquema de entrenamiento, no solo de aplicación"},
        {"id": "G-R2", "pages": "7–8", "section": "6 R2", "quote": "Debe existir un componente central programado por el estudiante"},
        {"id": "G-R2-LIVE", "pages": "7–8", "section": "6 R2", "quote": "que el estudiante pueda explicar y defender línea por línea"},
        {"id": "G-PRIMITIVES", "pages": "8", "section": "6 R2", "quote": "Se permite el uso de PyTorch, TensorFlow o JAX como motor de cálculo tensorial y de diferenciación automática"},
        {"id": "G-NOT-PREPROCESS", "pages": "8", "section": "6.1", "quote": "no únicamente el preprocesamiento o los hiperparámetros"},
        {"id": "G-THEORY", "pages": "9", "section": "6.2 Cuadro1", "quote": "demostración formal de una propiedad"},
        {"id": "G-LITERATURE", "pages": "12", "section": "6.3", "quote": "Repetir la búsqueda antes de la entrega final y registrar la fecha"},
        {"id": "G-THIRD", "pages": "12", "section": "6.3", "quote": "otra persona ajena al diseño repite al menos dos de las cadenas de búsqueda"},
        {"id": "G-HYPOTHESIS", "pages": "44", "section": "14.5.1", "quote": "Hipótesis del mecanismo falsable, con predicciones secundarias declaradas antes de los experimentos finales"},
        {"id": "G-THEORY-NUMERIC", "pages": "44", "section": "14.5.1", "quote": "Al menos una propiedad teórica demostrada formalmente, con supuestos explícitos y verificada numéricamente"},
        {"id": "G-AI", "pages": "48", "section": "15", "quote": "El estudiante es responsable de comprender y poder defender cada línea entregada, incluida la demostración formal"},
        {"id": "G-WEIGHTS", "pages": "60", "section": "Cuadro9", "quote": "Modelo original: revisión y diseño8%; implementación y demostración10%; Sustentación7%", "kind": "transcription_of_parallel_rubric_cells_not_contiguous_quote"},
    ]
    write_json(HERE / "GUIDE_CITATIONS.json", {"guide_filename": "Entregable3DeepLearningProject.pdf",
        "sha256": "d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364", "citations": quotes,
        "AI_review_modality": "Existing authorized modality preserved; G-THIRD wording is not silently rewritten as a human signature"})

    rows = [
        ("R1-01", "G-R1;G-NOT-PREPROCESS", "Revisión/diseño8%", "Diferencia sustantiva del algoritmo completo", "docs/R1_VERDICT.json;docs/R1_THEORY_LITERATURE.md;docs/R1_INDEPENDENT_AI_REVIEW.json", "SUSTENTADO_ACOTADO_PREVIO_REUTILIZADO", "No es novedad de normalización sola ni de retirar CRPS", "Valorar académicamente la argumentación; no nuevo entrenamiento"),
        ("R1-02", "G-R1;G-HYPOTHESIS", "Revisión/diseño8%", "Limitación y cuantificación con hipótesis falsable", "analysis/AUDIT_SUMMARY.json;R1_R2_DOSSIER.md§1,3,5", "EXPOSICION_CONSOLIDADA_SIN_NUEVO_RESULTADO_DE_MERCADO", "2.7595% descriptivo postselección no confirma causa o superioridad", "Congelar hipótesis/reglas para finales sólo tras batería admitida"),
        ("R1-03", "G-LITERATURE;G-THIRD", "Revisión/diseño8%", "Matriz, búsqueda fechada y revisión separada", "docs/R1_SEARCH_RECEIPT.json;docs/R1_REQUIREMENTS_MATRIX.csv;docs/R1_INDEPENDENT_AI_REVIEW.json", "CORPUS_AUTENTICO_REUTILIZADO_CON_LIMITES", "Nueve externos+V3; actualización focal no tres bases nativas nuevas; IA no firma humana", "Reutilizar modalidad IA ya autorizada; repetir sólo ante hueco o necesidad real"),
        ("R1-04", "G-THEORY;G-THEORY-NUMERIC", "Implementación/demostración10%", "Propiedad formal con supuestos y verificación numérica", "docs/R1_THEORY_LITERATURE.md;qa/CANDIDATE_SOFTWARE_TESTS.json;EXECUTION_RECEIPT.json D02-D06", "PRUEBA_FORMAL_CONSOLIDADA_8_TESTS_PREVIOS_6_CHECKS_NUEVOS", "Soporte/ley implementada e invariancia acotada; salida no equivariante; no ley identificada", "Explicar demostración y límites; no otro piloto para probar álgebra"),
        ("R2-01", "G-R2;G-PRIMITIVES", "Implementación/demostración10%", "Implementación propia y correspondencia con formulación", "runtime/candidate.py;runtime/vendor/ORIGIN.json;code_correspondence.csv;EXECUTION_RECEIPT.json", "FUENTES_Y_CORRESPONDENCIA_VERIFICADAS", "7SHA runtime y estado intactos; no prueba de comprensión humana", "Estudiante explica núcleo línea por línea ante evaluador"),
        ("R2-02", "G-R2-LIVE;G-AI", "Implementación/demostración10%;Sustentación7%", "Modificación razonada y demostración en vivo", "defense_exercises.py;DEFENSE_GUIDE.md;EXECUTION_RECEIPT.json D03-D04", "MODIFICACION_DIDACTICA_EJECUTADA_POR_IA_ACTO_HUMANO_NO_OBSERVADO", "Refactor equivalente ya presente en candidata; no nueva mejora algorítmica o market fit", "Autor realiza explicación/demostración y responde preguntas; aceptación docente auténtica"),
        ("R2-03", "G-AI;G-R2", "Revisión/diseño8%;Implementación/demostración10%", "Procedencia, derechos y asistencia declaradas", "docs/RIGHTS_AND_AI.md;runtime/vendor/ORIGIN.json;R1_R2_DOSSIER.md§6,7;SOURCE_MANIFEST.json", "PROCEDENCIA_DOCUMENTAL_CONSERVADA", "Sin licencia global nueva; los conceptos IA no son firmas o aprobaciones", "Mantener declaración IA del último capítulo y responsabilidad del autor"),
        ("SEPARACION", "G-R1;G-R2", "Sin puntuación nueva", "Distinguir técnica, defensa y acreditación", "CANDIDATE_IDENTITY.json;EXECUTION_RECEIPT.json;README.md", "SIN_CAMBIO_NOTAS_SIN_ACREDITACION_R3", "Sustentación4.5 fue valor de escenario no acto observado; tope condicional no cambia", "Evaluador emite el juicio real; fase2 corrige métodos sin abrir reserva"),
    ]
    fields = ("id", "guide_citation_ids", "rubric_component_and_weight", "requirement", "evidence", "status", "limit", "next_authentic_action")
    matrix = [dict(zip(fields, row)) for row in rows]
    with (HERE / "REQUIREMENT_EVIDENCE_MATRIX.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(matrix)
    write_json(HERE / "REQUIREMENT_EVIDENCE_MATRIX.json", {"schema_version": 1,
        "source_rule": "Guide original, existing candidate-specific reviews and genuinely executed checks",
        "paths_in_evidence": "Candidate-relative unless a new dossier filename is given; no third-party human act inferred",
        "new_grade": None, "teacher_acceptance": None, "student_defense_observed": False,
        "R3_accredited": False, "rows_are_not_independent_weights": True, "requirements": matrix})

    candidate_sources = ["CANDIDATE_IDENTITY.json", "docs/R1_VERDICT.json", "docs/R1_THEORY_LITERATURE.md",
        "docs/R1_REQUIREMENTS_MATRIX.csv", "docs/R1_INDEPENDENT_AI_REVIEW.json", "docs/R1_SEARCH_RECEIPT.json",
        "docs/R1_SOURCE_MANIFEST.json", "docs/RIGHTS_AND_AI.md", "runtime/vendor/ORIGIN.json",
        "qa/CANDIDATE_SOFTWARE_TESTS.json", "qa/INDEPENDENT_SCIENTIFIC_REVIEW.json",
        "tests/test_candidate.py", "analysis/AUDIT_SUMMARY.json", *identity["runtime_source_sha256"]]
    sources = [{"path": "../../e3_input_only_candidate_20261121/"+rel, "bytes": (candidate / rel).stat().st_size,
                "sha256": sha(candidate / rel), "kind": "unchanged_existing_candidate_evidence"} for rel in candidate_sources]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ("SOURCE_MANIFEST.json", "AUDIT_RECEIPT.json"):
            sources.append({"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p), "kind": "new_phase1_dossier"})
    write_json(HERE / "SOURCE_MANIFEST.json", {"schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(), "candidate_id": identity["candidate_id"],
        "candidate_identity_unchanged": True, "guide": {"filename": "Entregable3DeepLearningProject.pdf",
        "sha256": "d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364", "redistributed": False},
        "files": sources, "AI_assistance_declared": True,
        "new_literature_queries": 0, "new_forecasting_fits": 0, "new_designs": 0})
    write_json(HERE / "AUDIT_RECEIPT.json", {"status": "PASS", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha(HERE / "SOURCE_MANIFEST.json"),
        "audit_script_sha256": sha(__file__), "source_files_hashed": len(sources),
        "code_equation_rows": len(correspondence), "requirement_rows": len(matrix), "guide_citations": len(quotes),
        "old_candidate_tests_reused": 8, "new_didactic_checks_executed": 6,
        "candidate_runtime_files_verified_unchanged": 7,
        "cpu_seconds": time.process_time()-cpu, "wall_seconds": time.perf_counter()-wall,
        "no_model_import_in_this_audit": True, "new_forecasting_fits": 0, "market_predictions": 0,
        "academic_acceptance_claimed": False, "R3_accredited": False, "new_grade": None})
    print(json.dumps({"status": "PASS", "code_equation_rows": len(correspondence), "requirement_rows": len(matrix), "source_files_hashed": len(sources)}))


if __name__ == "__main__":
    main()
