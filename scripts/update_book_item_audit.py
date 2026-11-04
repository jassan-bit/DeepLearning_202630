"""Refresh the 390-item inventory after the completed four-history experiment."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/filipe_multihistory_20261104"
BASE = ROOT / "references/audits/revision_items_libro_20261104"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    previous = json.loads(BASE.with_suffix(".json").read_text())
    expected = [row["item"] for row in previous["items"]]
    items, counts, hashes = [], {}, {}
    pattern = re.compile(r"^#{2,6}\s+([1-7](?:\.\d+)+)\.?\s+(.+)$")
    for path in sorted((ROOT / "book/sections").glob("0[1-7]_*.md")):
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
    if len(items) != 390 or [row["item"] for row in items] != expected:
        raise ValueError("The original 390 numbered items must retain their exact hierarchy")
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
              "scope": "Revisión de todos los incisos del libro con CV de Filipe para cuatro entradas y evaluación del test tras cierre de configuración.",
              "reviewed_numbered_items": 390, "numbering_preserved": True, "chapter_counts": counts,
              "distinction": "Revisado o actualizado no significa que todos los requisitos científicos o bibliográficos estén completamente cumplidos.",
              "implemented": {"input_days": [7, 14, 21, 28], "target_return_days": 7, "horizon_days": 1,
                              "CV_primary": "120 trayectorias, 1440 segmentos modelo-fold; D7 reutilizado y 90 trayectorias nuevas D14/21/28.",
                              "new_CV_project_fits": 2160, "final_development_fits": 120,
                              "final_test": execution, "CV_BDS": cv["bds"], "test_BDS": test["bds"],
                              "metrics": "MAPE, MAE, RMSE, MSE, R2 y MASE por fecha/semilla/mes; resúmenes con media, DE e IC95 entre semillas.",
                              "selection": freeze["selection"],
                              "test_months": "20 intervalos de reporte con los mismos pesos finales; cero reajustes durante test."},
              "historical_D7_supplement": {"protocol": "experiments/filipe_statistics_20261104/protocol.json",
                                          "scope": "Holm12/DM6/bootstrap y diagnósticos anuales pertenecen exclusivamente a la CV inicial D7/2024; no se trasladan a los demás D o al test."},
              "outstanding": {"prospective_pilot_and_definitive_power": "No realizado; sensibilidad retrospectiva no lo sustituye.",
                              "applied_comparator_literature": "Respaldo aplicado independiente de MLP y CNN1D simple sigue incompleto en el corpus incluido.",
                              "future_prospective_validation": "Criterio académico operativo del5% definido ahora para comparaciones futuras; no registrado antes del test ya observado. Falta piloto y evaluación independiente nuevos.",
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
    BASE.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    with BASE.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(items)
    BASE.with_suffix(".md").write_text(f"""# Revisión integral de ítems del libro — cuatro entradas y test final

Se revisaron **390 subsecciones numeradas** de los capítulos 1–7 y se conservó exactamente su jerarquía. El inventario CSV/JSON identifica cada ítem, título, archivo y línea. «Revisado» no declara cierre de todos los requisitos científicos.

La CV de Filipe evalúa D=7,14,21,28 días con objetivo σ7 y horizonte diario h=1. D7 conserva sus treinta trayectorias originales; se añadieron noventa trayectorias y 2.160 ajustes de desarrollo. La configuración se cerró con 2024 antes del ajuste final de 120 modelos y de abrir los veinte archivos del test enero2025–agosto2026.

El test tiene {test['outcome_days']} fechas elegibles comunes y {test['neural_forecasts']:,} pronósticos. Sus veinte meses son intervalos de reporte de pesos congelados, sin reajustes ni selección por desempeño del test. MAPE, MAE, RMSE y MSE, además de R² y MASE, se publican por entrada, modelo, semilla y mes.

BDS se calculó por fold y semilla: {cv['bds']['planned_hypotheses']} hipótesis previstas en CV y {test['bds']['planned_hypotheses']} en test. Las medias de p-valores son descriptivas; la referencia asintótica en meses cortos tiene limitaciones. El suplemento anual previo de D7/2024 conserva su alcance histórico.

Se añadieron tablas anuales, descomposición descriptiva de variabilidad, segmentos y residuos, inferencia final explícitamente post-hoc, diagramas y una ruta de restauración privada verificada. El criterio operativo académico futuro es una mejora superior al5% frente a MLP/D7 y persistencia; no se presenta como definido antes del test ya conocido ni como utilidad financiera.

Permanecen abiertos el respaldo aplicado de MLP/CNN1D, el piloto/dimensionamiento prospectivo de una evaluación futura y las mediciones externas de ruido/faltantes. TOST/ROPE es condicional a una afirmación de equivalencia, no un pendiente obligatorio. No se asigna nota ni porcentaje de cumplimiento.
""")
    public = ROOT / "book/_static/e1"
    public.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".csv", ".md"):
        shutil.copyfile(BASE.with_suffix(suffix), public / BASE.with_suffix(suffix).name)
    for path in (independent, independent.with_suffix(".md")):
        shutil.copyfile(path, public / path.name)
    print(json.dumps({"reviewed_numbered_items": 390, "numbering_preserved": True,
                      "test_outcomes": test["outcome_days"], "test_forecasts": test["neural_forecasts"]}))


if __name__ == "__main__":
    main()
