"""Track concrete audit improvements without certifying unperformed research."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"references/audits/auditoria_e1_pdf_baseline_20261104.json"
OUT=ROOT/"references/audits/cierre_auditoria_e1_20261104"
RESULTS="results/e1_audit_improvements_20261104"
STATES={
 "A01":("Parcial", "Se amplió la recuperación dirigida; faltan manuscritos aplicados completos de MLP y CNN1D.", "references/literature_e1/recovery_20261104/summary.json"),
 "A02":("Criterio académico futuro definido", "5 % frente a MLP/D7 y persistencia; 36 sensibilidades iniciales y 27 escenarios de frontera práctica. No es margen previo al test actual ni utilidad financiera.", "experiments/e1_audit_improvements_20261104/academic_margin_policy.json"),
 "A03":("Análisis completado con límites", "Comparaciones, efectos, IC, DM/Ljung–Box y bootstrap finales calculados y verificados como suplemento post-hoc.", "references/audits/revision_e1_final_inference_20261104.json"),
 "A04":("Pendiente prospectivo", "El piloto histórico se conserva; la plantilla del nuevo estudio es un borrador no registrado ni ejecutado.", "experiments/e1_next_study_pilot_template.json"),
 "A05":("Ruta privada implementada y verificada", "BUILD vigente; paquete y restauración aislada con auditoría PASS, sin repetir ajuste o pronósticos.", RESULTS+"/reproducibility/summary.json"),
 "A06":("Trazabilidad documentada", "Se documentó la búsqueda histórica de 96 corridas y la transferencia de recetas; no se reoptimizó usando test.", "book/sections/04_implementacion_evaluacion.md"),
 "A07":("Recuperación ampliada; cobertura parcial", "13 cadenas y 18 páginas; candidatos de código identificados sin convertirlos en textos científicos incluidos.", "references/literature_e1/recovery_20261104/search_log.json"),
 "A08":("Comparabilidad documentada; coincidencia no acreditada", "Matriz de siete fuentes distingue Spot/Futures, frecuencia y objetivo; ninguna acredita igualdad con la versión σ7.", "references/literature_e1/recovery_20261104/dataset_comparability.csv"),
 "A09":("Límite abierto", "El mecanismo de faltantes no puede clasificarse sin evidencia; exclusiones y consecuencias permanecen documentadas.", "book/sections/02_eda.md"),
 "A10":("Descriptivo completado post-hoc", "Distribuciones por 2023/2024/2025/2026 parcial calculadas, sin cambiar preprocesamiento o selección.", RESULTS+"/descriptive/distribution_by_period.csv"),
 "A11":("Descomposición descriptiva completada", "Componentes por modelo, semilla, mes e interacciones calculados; no se atribuye causalidad ni independencia de mercados.", RESULTS+"/descriptive/variance_components.csv"),
 "A12":("Diagnóstico descriptivo completado", "Bins de residuos, sesgo, dispersión y meses difíciles; celdas con menos de diez fechas suprimidas.", RESULTS+"/descriptive/residual_bins_summary.csv"),
 "A13":("Diferencias calculadas", "Diferencias absolutas y relativas de RMSE/MAE con referencias y signos explícitos.", RESULTS+"/inference/relative_differences.csv"),
 "A14":("Tablas anuales completadas", "2025:365 fechas; enero–agosto2026:243. R² usa denominador propio de cada periodo.", RESULTS+"/descriptive/annual_metrics_summary.csv"),
 "A15":("Corregido", "Se distingue el control de dos representaciones D7 de la representación relativa principal en cuatro historias.", "book/sections/03_estado_arte.md"),
 "A16":("Alcances históricos diferenciados", "Snapshots y configuración vigente se identifican por experimento; diez semillas actuales no se mezclan con dos históricas.", "book/sections/03_estado_arte.md"),
 "A17":("Diagramas realizados", "Tres SVG muestran dimensiones, capas y parámetros coherentes con los modelos existentes.", "book/_static/e1_architectures_20261104/mlp.svg"),
 "A18":("Interpretación reforzada; calibración adicional no realizada", "BDS mensual sigue exploratorio; nuevos DM/LB y diagnóstico agregado no demuestran calibración finita del BDS.", "book/sections/05_benchmark.md"),
 "A19":("Política descrita; alternativa no entrenada", "Épocas de los doce cortes resumidas, con comparación de diciembre; no se cambió la regla por resultados de test.", RESULTS+"/traceability/epoch_policy_summary.csv"),
 "A20":("Segmentación y estabilidad calculadas", "Errores por rangos y meses analizados; perturbaciones y ruido externo siguen como mejoras opcionales no ejecutadas.", RESULTS+"/descriptive/severity_metrics_summary.csv"),
 "A21":("Referencia contextual conservada", "Ridge sigue como diagnóstico EDA; no se presenta como una comparación final homogénea no ejecutada.", "book/sections/06_procedimiento.md"),
 "A22":("Calendario real dibujado", "SVG de doce cortes por D, historia inicial, purga y test, separado del ejemplo oficial de la librería.", "book/_static/e1_cv_calendar_20261104/cv_calendar.svg"),
 "A23":("Condicional; no pendiente obligatorio", "TOST/ROPE solo corresponde si se afirma equivalencia; no se usa ausencia de significancia para afirmarla.", "book/sections/07_limitaciones_entregables.md"),
}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    baseline=json.loads(BASE.read_text())
    rows=[]
    for item in baseline["findings"]:
        status,change,evidence=STATES[item["id"]]
        if not (ROOT/evidence).exists():raise FileNotFoundError(evidence)
        rows.append({"id":item["id"],"priority":item["priority"],"criterion_type":item["criterion_type"],
            "title":item["title"],"previous_status":item["status"],"current_status":status,
            "concrete_change":change,"evidence":evidence,"evidence_sha256":digest(ROOT/evidence),
            "remaining_acceptance":item["acceptance"],"pdf_pages":item["pdf_pages"]})
    if len(rows)!=23:raise ValueError("The audit's 23 findings must remain identifiable")
    report={"updated_at_utc":datetime.now(timezone.utc).isoformat(),"original_book_commit":baseline["book_source_commit"],
        "guide":baseline["guide"],"audit_baseline_sha256":digest(BASE),
        "scope":"Concrete documentation and saved-forecast analysis improvements; no new training, forecast or raw-market read.",
        "new_model_fits":0,"new_checkpoint_forward_calls":0,"raw_market_archive_reads":0,
        "statistical_scope":"Exploratory post-hoc supplement, not prospective confirmation; MLP/D7 baseline unchanged.",
        "main_open_requirements":["Applied full-text support for the MLP/CNN1D baselines","Prospective pilot/dimensioning and independent evaluation for a future study"],
        "academic_operational_criterion":"Future improvement above5percent versus frozen MLP/D7 and persistence; recorded after the current test, with scenario planning and no financial utility claim.",
        "conditional_not_required":"Equivalence/TOST/ROPE is not required without an equivalence claim.",
        "public_scope":"Aggregates and metadata only; complete restoration package remains private.","findings":rows}
    OUT.with_suffix(".json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    with OUT.with_suffix(".csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    lines=["# Cierre de mejoras de la auditoría E1","",
        "Se mantienen visibles los 23 hallazgos originales. Registrar una mejora no convierte un análisis retrospectivo en prospectivo ni acredita criterios que aún no tienen evidencia.","",
        "| ID | Hallazgo | Estado actual | Mejora y evidencia |","|---|---|---|---|"]
    for row in rows:lines.append(f"| {row['id']} | {row['title']} | {row['current_status']} | {row['concrete_change']} (`{row['evidence']}`) |")
    lines += ["","Permanecen abiertos el respaldo aplicado de MLP/CNN1D y el piloto/dimensionamiento prospectivo de una evaluación futura independiente. El criterio académico operativo del 5 % queda definido para el futuro, sin atribuirlo al protocolo previo al test ya observado. La réplica privada verifica la evidencia histórica y no constituye otro experimento ni un test nuevo.",""]
    OUT.with_suffix(".md").write_text("\n".join(lines))
    for path in [ROOT/"book/_static/e1/decisiones.json",ROOT/"book/_static/literature_e1/guide_aligned_proposal_20261104.json",ROOT/"book/_static/literature_e1/selection_evidence.json"]:
        data=json.loads(path.read_text())
        data["audit_improvements_20261104"]={"scope":"post-hoc saved-forecast supplement and documentary improvements",
            "closure_matrix":"../e1_audit_improvements_20261104/cierre_auditoria_e1_20261104.json",
            "new_model_fits":0,"new_checkpoint_forward_calls":0,"baseline_selection_changed":False,
            "statistical_inference":"Final three D7 configurations: paired six contrasts, temporalDM6, LB9, paired bootstrap; explicitly exploratory.",
            "annual_tables":"2025=365days;2026Jan-Aug=243days, all4D",
            "independent_replica":"Private package restoration+independent audit PASS; no new forecasts.",
            "equivalence":"Conditional, not an unfulfilled obligation without an equivalence claim",
            "main_open_requirements":report["main_open_requirements"]}
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"audit_findings_tracked":len(rows),"new_fits":0,"new_forecasts":0,"main_open_requirements":len(report["main_open_requirements"])}))


if __name__ == "__main__":main()
