"""Preserve the 390-item hierarchy and add the seven delivery products."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import shutil

from literature_crypto_stage import crypto_delivery_text, load_crypto_stage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filipe_multihistory_20261104"
BASE = ROOT / "references/audits/revision_items_libro_20261104"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    previous = json.loads(BASE.with_suffix(".json").read_text())
    expected = [row["item"] for row in previous["items"] if not row["item"].startswith("8.")]
    items, counts, hashes = [], {}, {}
    pattern = re.compile(r"^#{2,6}\s+([1-8](?:\.\d+)+)\.?\s+(.+)$")
    for path in sorted((ROOT / "book/sections").glob("0[1-8]_*.md")):
        rows = []
        for line, text in enumerate(path.read_text().splitlines(), 1):
            match = pattern.match(text)
            if match:
                rows.append({"item": match.group(1).rstrip("."), "titulo": match.group(2),
                             "archivo": path.relative_to(ROOT).as_posix(), "linea": line,
                             "revision": "revisado; contenido actualizado o verificado",
                             "cierre_del_requisito": "consultar evidencia y límites del apartado; revisión no equivale a cumplimiento total"})
        items.extend(rows)
        counts[path.name] = len(rows)
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    original = [row["item"] for row in items if not row["item"].startswith("8.")]
    added = [row["item"] for row in items if row["item"].startswith("8.")]
    if len(original) != 390 or original != expected:
        raise ValueError("The original 390 numbered items must retain their exact hierarchy")
    if added != [f"8.{number}" for number in range(1,8)] or len(items) != 397:
        raise ValueError("The delivery chapter must add exactly the seven requested items")
    freeze_path = ROOT / "experiments/filipe_multihistory_20261104/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    cv = json.loads((OUT / "analysis_cv/summary.json").read_text())
    test = json.loads((OUT / "test/analysis/summary.json").read_text())
    execution = json.loads((OUT / "test/execution_summary.json").read_text())
    independent = ROOT / "references/audits/revision_filipe_multihistory_20261104.json"
    if not independent.exists() or cv["neural_forecasts"] != 43920 or execution["model_runs"] != 120:
        raise ValueError("Finish the full experiment and its independent audit first")
    independent_report = json.loads(independent.read_text())
    if independent_report.get("status") != "PASS" or independent_report.get("phase") != "test":
        raise ValueError("The independent final-test audit must pass before refreshing the book inventory")
    report = {"reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Revisión de los capítulos1–8: 390 incisos previos conservados y siete productos de entrega desarrollados.",
              "reviewed_numbered_items": 397, "original_numbered_items_preserved":390,
              "delivery_items_added":7,"numbering_preserved": True, "chapter_counts": counts,
              "delivery_section_execution": {"scope": "This inventory refresh only; completed new pilot training is described separately in new_registered_pilot", "new_fits": 0, "new_forecasts": 0,
                                             "raw_market_archive_reads": 0},
              "distinction": "Revisado o actualizado no significa que todos los requisitos científicos o bibliográficos estén completamente cumplidos.",
              "implemented": {"input_days": [7, 14, 21, 28], "target_return_days": 7, "horizon_days": 1,
                              "CV_primary": "120 trayectorias, 1440 segmentos modelo-fold; D7 reutilizado y 90 trayectorias nuevas D14/21/28.",
                              "new_CV_project_fits": 2160, "final_development_fits": 120,
                              "final_test": execution, "CV_BDS": cv["bds"], "test_BDS": test["bds"],
                              "metrics": "MAPE, MAE, RMSE, MSE, R2 y MASE por fecha/semilla/mes; resúmenes con media, DE e IC95 entre semillas.",
                              "selection": freeze["selection"],
                              "delivery_section": "8.1–8.7: informe, código, dataset, literatura/flujo, comparación/inferencia, configuraciones y declaración cronológica, con límites explícitos.",
                              "test_months": "20 intervalos de reporte con los mismos pesos finales; cero reajustes durante test."},
              "historical_D7_supplement": {"protocol": "experiments/filipe_statistics_20261104/protocol.json",
                                          "scope": "Holm12/DM6/bootstrap y diagnósticos anuales pertenecen exclusivamente a la CV inicial D7/2024; no se trasladan a los demás D o al test."},
              "outstanding": {"prospective_pilot_and_definitive_power": "Nuevo piloto registrado antes de60 trayectorias/1440 ajustes sobre2024; R futuro calculado. Repeticiones definitivas, potencia real y evaluación temporal independiente no ejecutadas; no repara el benchmark histórico.",
                              "applied_comparator_literature": "CorpusBTC6 y39 referencias. Miura2019 conserva MLP/CNN1/6 y dimensiónCNN desconocida; T1 añade original financiero FNN de revista, T2 código autor Conv1D y T3 identidad editorial sin original. No aumentan frecuenciasBTC ni prueban réplica de las recetas o indexación individual. Revisión posterior al benchmark.",
                              "future_prospective_validation": "E1 exige definir margen/MDE y sus límites a partir del baseline. El criterio futuro del5% cumple ese propósito documental sin ser margen previo del test histórico. El piloto propio de un avanzado y otra evaluación independiente son condiciones si después se implementa esa propuesta, no entrenamientos obligatorios de E1.",
                              "external_measurements": "Ruido externo de precios/etiquetas y mecanismo MCAR/MAR/MNAR no identificados."},
              "new_execution_in_this_extension": {"CV_fits": 2160, "final_fits": 120,
                                                   "raw_test_archives_opened": execution["raw_archives_opened"],
                                                   "test_checkpoint_forward_calls": execution["actual_checkpoint_forward_calls"],
                                                   "test_period_refits": execution["test_period_model_refits"]},
              "protocol_sha256": sha(freeze_path.with_name("protocol.json")), "freeze_sha256": sha(freeze_path),
              "independent_audit_sha256": sha(independent), "source_book_sha256": hashes, "items": items}
    closure = ROOT / "references/audits/cierre_auditoria_e1_20261104.json"
    if closure.exists():
        updated = json.loads(closure.read_text())
        report["audit_improvements"] = {
            "scope": "Suplemento post-hoc de pronósticos guardados y mejoras documentales; ningún nuevo ajuste ni pronóstico.",
            "closure_matrix": closure.relative_to(ROOT).as_posix(),
            "closure_matrix_sha256": sha(closure),
            "main_open_requirements": updated["main_open_requirements"],
            "academic_operational_criterion": updated["academic_operational_criterion"],
            "equivalence": "Condicional, no requisito pendiente sin afirmación de equivalencia.",
        }
    literature = ROOT / "references/literature_e1/recovery_20261105/selection_update.json"
    if literature.exists():
        report["current_literature_review"] = {
            "source": literature.relative_to(ROOT).as_posix(),
            "source_sha256": sha(literature),
            "timing": "Posterior al benchmark; no reescribe la selección histórica ni los registros de auditoría anteriores.",
            "finding": "Una nueva fuente primaria verifica MLP/CNN independientes; seis estudios en el corpus ampliado. La dimensionalidad CNN no se declara en el original.",
        }
    journals = ROOT / "references/literature_e1/journal_update_20261105/methodological_sources.json"
    if journals.exists():
        report["historical_applied_literature_review"] = report.get("current_literature_review")
        report["current_literature_review"] = {
            "source": journals.relative_to(ROOT).as_posix(), "source_sha256": sha(journals),
            "bibliographic_references": 39, "BTC_full_text_studies": 6,
            "new_methodological_full_texts": 1, "new_author_code_sources": 1,
            "new_publication_identity_only_sources": 1, "new_BTC_inclusions": 0,
            "finding": "T1 Bucci original de revista con FNN; T2 HARNet código Conv1D; T3 DeepVol identidad editorial, original pendiente. Niveles distintos, sin aumentar recurrencia BTC o acreditar registros individuales/Q1.",
            "timing": "Recuperación posterior al benchmark; no altera resultados o cronología de entrenamiento.",
        }
    literature_note = (
        "La recuperación posterior de Miura2019 verifica comparadores MLP y CNN independientes en Bitcoin: "
        "cada familia aparece en1/6 textos completos. No acredita dimensionalidad explícita CNN1D, "
        "réplica exacta de las recetas ni uso amplio en el campo.")
    crypto = load_crypto_stage(ROOT)
    if crypto:
        counts = crypto["counts"]
        summary = crypto["summary"]
        report["historical_journal_literature_review"] = report.get("current_literature_review")
        report["current_literature_review"] = {
            "source": (crypto["directory"] / "summary.json").relative_to(ROOT).as_posix(),
            "source_sha256": sha(crypto["directory"] / "summary.json"),
            "stage_id": summary["stage_id"], **counts,
            "study_ids": summary["study_ids"], "new_study_ids": summary["new_study_ids"],
            "finding": crypto_delivery_text(crypto)["review"],
            "harnet_erratum": (crypto["directory"] / summary["harnet_erratum_file"]).relative_to(ROOT).as_posix(),
            "metadata_source_sha256": crypto["source_sha256"],
            "timing": summary["timing"], "experiment_results_modified": False,
            "new_model_fits": 0, "new_forecasts": 0,
        }
        report["outstanding"]["applied_comparator_literature"] = (
            f"Corpus cripto {counts['crypto_full_text_studies']} ({counts['BTC_full_text_studies']} con BTC), "
            f"{counts['bibliographic_references']} referencias y {counts['studies_with_non_BTC_crypto']} estudio con otros criptoactivos. "
            f"Baselines MLP {counts['MLP_baseline_studies']}/{counts['crypto_full_text_studies']}, "
            f"CNN genérica {counts['CNN_generic_baseline_studies']}/{counts['crypto_full_text_studies']} y "
            f"CNN1D explícita {counts['CNN1D_explicit_baseline_studies']}/{counts['crypto_full_text_studies']}. "
            "S9 es un original no publicado; su LSTM propuesto no cuenta como baseline. "
            "HARNet conserva nivel código con errata de escalado y DeepVol identidad con original pendiente. "
            "La dimensión CNN de S8, recetas exactas, recurrencia amplia e indexación individual conservan límites. "
            "Extensión documental posterior al benchmark sin nuevos ajustes ni pronósticos.")
        literature_note = (
            f"La revisión documental vigente comprende {counts['crypto_full_text_studies']} textos completos cripto "
            f"({counts['BTC_full_text_studies']} con Bitcoin; {counts['studies_with_non_BTC_crypto']} también con otros criptoactivos) "
            f"y {counts['bibliographic_references']} referencias. Los baselines MLP y CNN genérica aparecen, respectivamente, "
            f"en{counts['MLP_baseline_studies']}/{counts['crypto_full_text_studies']} y "
            f"{counts['CNN_generic_baseline_studies']}/{counts['crypto_full_text_studies']} textos; "
            f"CNN1D explícita permanece en{counts['CNN1D_explicit_baseline_studies']}/{counts['crypto_full_text_studies']}. "
            "S9 añade un original no publicado de varios criptoactivos y su LSTM propuesto no suma un baseline. "
            "Los registros históricos de seis textos y39 referencias se conservan como cortes anteriores. "
            "La errata HARNet retira una atribución de fuga no sustentada; no constituye auditoría integral del código. "
            "No se acredita dimensión CNN de S8, réplica exacta, uso amplio ni indexación individual.")
    pilot = ROOT / "results/e1_filipe_pilot_20261105/analysis/summary.json"
    if pilot.exists():
        report["new_registered_pilot"] = {
            "source": pilot.relative_to(ROOT).as_posix(), "source_sha256": sha(pilot),
            "scope": "Registro antes de nuevas aleatorizaciones, condicionado al calendario2024 ya examinado; benchmark anterior conservado.",
            "analysis": json.loads(pilot.read_text()),
        }
    conditional_note = "Las repeticiones definitivas del estudio D7 nuevo todavía no se declaran completadas."
    conditional = ROOT / "results/e1_filipe_confirmatory_d7_20261105/analysis/summary.json"
    conditional_audit = ROOT / "references/audits/revision_filipe_confirmatory_d7_20261105.json"
    if conditional.exists() and conditional_audit.exists():
        verified = json.loads(conditional_audit.read_text())
        if verified.get("status") != "PASS" or verified.get("phase") != "definitive":
            raise ValueError("The new D7 study must pass its independent final audit")
        study = conditional.parent.parent
        lock = json.loads((study / "locked_design.json").read_text())
        definitive = json.loads((study / "definitive/execution_summary.json").read_text())
        r = lock["common_R"]
        if definitive["trajectories"] != 3*r:
            raise ValueError("The locked number of definitive D7 trajectories must be complete")
        report["registered_conditional_D7_study"] = {
            "source": conditional.relative_to(ROOT).as_posix(), "source_sha256": sha(conditional),
            "independent_audit": conditional_audit.relative_to(ROOT).as_posix(),
            "independent_audit_sha256": sha(conditional_audit),
            "variance_extension_seeds_per_model": 20, "definitive_seeds_per_model": r,
            "definitive_trajectories": 3*r, "definitive_fits": 72*r,
            "primary_family_size": 3, "new_temporal_evaluation": False,
            "historical_benchmark_repaired": False, "achieved_power_claim": False,
            "scope": "Protocolo publicado antes de la ampliación; R publicado antes de las nuevas semillas definitivas. Inferencia condicionada a las recetas y al calendario2024 ya conocido.",
        }
        sensitivity = ROOT / "book/_static/e1_filipe_confirmatory_d7_20261105/planning/posthoc_design_sensitivity.json"
        report["registered_conditional_D7_study"]["normality_warning_pairs"] = json.loads(conditional.read_text())["normality_warning_pairs"]
        if sensitivity.exists():
            posterior = json.loads(sensitivity.read_text())
            report["registered_conditional_D7_study"]["posthoc_design_sensitivity"] = {
                "source": sensitivity.relative_to(ROOT).as_posix(), "source_sha256": sha(sensitivity),
                "MDE_exceeds_fixed_delta_pairs": sum(row["MDE_exceeds_fixed_delta"] for row in posterior["pairs"]),
                "scope": "Posterior normal-reference sensitivity with fixed delta and R; not achieved power, not a new primary test.",
            }
        report["outstanding"]["prospective_pilot_and_definitive_power"] = (
            f"Estudio D7 nuevo completado: ampliación20/modelo y fase definitivaR={r}/modelo, con auditoría independiente. "
            "La sensibilidad posterior MLP–CNN1D tiene MDE mayor que delta y se declara como límite. "
            "No repara la cronología anterior ni acredita potencia alcanzada o otra trayectoria temporal.")
        conditional_note = (
            f"El estudio D7 nuevo publicó protocolo antes de su ampliación de20 semillas por modelo y R={r} antes "
            f"de sus {3*r} trayectorias definitivas ({72*r} ajustes). Su familia primaria de tres pares y su "
            "auditoría independiente quedan separadas del suplemento histórico. Es prospectivo respecto de "
            "las nuevas aleatorizaciones sobre2024 conocido; no es otro test temporal ni prueba de potencia alcanzada.")
    BASE.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    with BASE.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(items)
    BASE.with_suffix(".md").write_text(f"""# Revisión integral de ítems del libro — cuatro entradas y test final

Se conservaron las **390 subsecciones numeradas** de los capítulos 1–7 y su jerarquía. La sección 8 desarrolla los siete productos solicitados en **8.1–8.7**, para un total de **397 incisos**. El inventario CSV/JSON identifica cada ítem, título, archivo y línea. «Revisado» no declara cierre de todos los requisitos científicos.

La CV de Filipe evalúa D=7,14,21,28 días con objetivo σ7 y horizonte diario h=1. D7 conserva sus treinta trayectorias originales; se añadieron noventa trayectorias y 2.160 ajustes de desarrollo. La configuración se cerró con 2024 antes del ajuste final de 120 modelos y de abrir los veinte archivos del test enero2025–agosto2026.

El test tiene {test['outcome_days']} fechas elegibles comunes y {test['neural_forecasts']:,} pronósticos. Sus veinte meses son intervalos de reporte de pesos congelados, sin reajustes ni selección por desempeño del test. MAPE, MAE, RMSE y MSE, además de R² y MASE, se publican por entrada, modelo, semilla y mes.

BDS se calculó por fold y semilla: {cv['bds']['planned_hypotheses']} hipótesis previstas en CV y {test['bds']['planned_hypotheses']} en test. Las medias de p-valores son descriptivas; la referencia asintótica en meses cortos tiene limitaciones. El suplemento anual previo de D7/2024 conserva su alcance histórico.

Se añadieron tablas anuales, descomposición descriptiva de variabilidad, segmentos y residuos, inferencia final explícitamente post-hoc, diagramas y una ruta de restauración privada verificada. El criterio operativo académico futuro es una mejora superior al5% frente a MLP/D7 y persistencia; no se presenta como definido antes del test ya conocido ni como utilidad financiera.

{literature_note} El piloto de cuatro historias se conserva por separado. {conditional_note} La generalización a otro periodo y las mediciones externas de ruido/faltantes conservan sus límites. TOST/ROPE es condicional a una afirmación de equivalencia, no un pendiente obligatorio. No se asigna nota ni porcentaje de cumplimiento.
""")
    public = ROOT / "book/_static/e1"
    public.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".csv", ".md"):
        shutil.copyfile(BASE.with_suffix(suffix), public / BASE.with_suffix(suffix).name)
    for path in (independent, independent.with_suffix(".md")):
        shutil.copyfile(path, public / path.name)
    print(json.dumps({"reviewed_numbered_items":397,"original_numbered_items_preserved":390,
                      "delivery_items_added":7,"numbering_preserved": True,
                      "test_outcomes": test["outcome_days"], "test_forecasts": test["neural_forecasts"]}))


if __name__ == "__main__":
    main()
