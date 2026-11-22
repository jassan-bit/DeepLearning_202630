"""Audit the original guide against hash-bound existing evidence.

No markets, model fits/forwards, Monte Carlo, calibration, or final data access.
AI-assisted documentary/statistical review; not academic R1/R2/R3 acceptance.
Rows are overlapping requirements, not separately weighted grading components.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
GUIDE_SHA = 'd6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364'
ENGINE_PREFIX = 'experiments/e3_confirmatory_engine_20261121/'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def norm(text):
    return ' '.join(text.lower().split())


CITATIONS = {
 'rubric': (60, 'Cuadro 9', 'Tres pruebas por contraste, supuestos verificados'),
 'unit': (25, '8.6', 'la unidad de remuestreo debe ser la unidad de independencia'),
 'assumptions': (25, '8.6', 'Verificar los supuestos de cada prueba'),
 'three': (26, 'Cuadro 4', 'Se aplican al menos tres pruebas distintas por contraste'),
 'raw_adjusted': (25, '8.6', 'reportar siempre el p-valor crudo y el ajustado'),
 'holm': (27, '8.7.1', 'controlar el FWER con Holm'),
 'all_reject': (27, '8.7.1', 'todas deben rechazar'),
 'pilot': (27, '8.7.2', 'Estudio piloto con al menos 5 semillas'),
 'power': (24, '8.5', 'se exige una potencia de al menos 0.90 por contraste'),
 'joint': (24, '8.5', 'El equipo debe reportar la potencia conjunta'),
 'seeds': (45, '14.5.3', 'Mínimo 10 semillas para todos los modelos de la familia'),
 'effect': (24, '8.4.3', 'el límite inferior del intervalo de confianza del 95 % de la diferencia es mayor que cero'),
 'adequacy': (24, '8.4.5', 'supera al baseline trivial'),
 'gap': (24, '8.4.5', 'brecha train-validación menor que el margen'),
 'residuals': (26, 'Cuadro 4/F2', 'Ljung-Box, ARCH-LM, BDS y'),
 'mcs': (45, '14.5.3', 'Diagrama de diferencia crítica o MCS que incluya el modelo propio'),
 'mcs_forecast': (29, '8.9', 'la tabla del MCS (pronóstico y regresión)'),
 'between': (45, '14.5.3', 'Prueba entre conjuntos de datos'),
 'sensitivity': (45, '14.5.3', 'Análisis de sensibilidad de la conclusión a la prueba'),
 'coverage': (46, '14.5.5', 'incluida la cobertura por subgrupo'),
 'same_task': (23, '8.3', 'mantener la misma tarea y la misma métrica primaria'),
 'two_axes': (23, '8.3', 'cada conjunto debe diferir del principal en al menos dos ejes'),
 'sealed': (25, '8.6', 'la prueba se evalúa una sola vez, al final'),
 'levels': (43, '14.4', 'ausencia de análisis de potencia'),
 'no_power': (42, '14.1', 'un resultado no concluyente sin análisis de potencia'),
}

# Namespace is the source of bytes, not a claim that every file is in the same
# local checkout. repo paths refer to the complete book source; engine paths to
# the immutable source package; delivery paths to this new scoped audit.
EVIDENCE = {
 'baseline': ('repo','book_e3/_static/e3_btc_scope_assessment_20261121/assessment.json'),
 'protocol': ('engine','configs/protocol.json'),
 'power_design': ('engine','configs/power_design.json'),
 'adequacy_config': ('engine','configs/adequacy.json'),
 'methods': ('engine','docs/TEMPORAL_BATTERY.md'),
 'statistics': ('engine','engine/statistics.py'),
 'assessment': ('engine','engine/assessment.py'),
 'conditional_validation': ('engine','qa/statistical_validation/v4_precision/report.json'),
 'method_probe': ('engine','qa/actual_method_admission_probe.json'),
 'input_only_previous': ('repo','experiments/e3_input_only_candidate_20261121/analysis/AUDIT_SUMMARY.json'),
 'historical_v4': ('repo','experiments/e3_statistics_remediation_20261121/retrospective/results/analysis_summary.json'),
 'historical_aditive_adequacy': ('repo','experiments/e3_evidence_close_20261120/adequacy/results/adequacy_receipt.json'),
 'historical_aditive_mcs': ('repo','book_e3/_static/e3_estadistica_posterior_20261117/mcs.json'),
 'new_retrospective': ('delivery','retrospective/results/summary.json'),
 'new_diagnostics': ('delivery','retrospective/results/diagnostics_Holm128.csv'),
 'new_temporal_intervals': ('delivery','retrospective/results/temporal_RMSE_MAE_intervals.csv'),
 'new_analysis_code': ('delivery','retrospective/audit_saved_adequacy.py'),
 'new_analysis_plan': ('delivery','retrospective/analysis_plan.json'),
 'methods_input_audit': ('delivery','methods_power/current_input_audit_v2.json'),
 'scientific_final_gate': ('delivery','methods_power/final_gate.json'),
 'candidate_secondary_mcs': ('delivery','mcs/results/summary.json'),
 'candidate_mcs_results': ('delivery','mcs/results/mcs_full_results.json'),
}


def row(id, requirement, citations, evidence, status, limitation, action, command, stage):
    return dict(id=id,requirement=requirement,citation_ids=citations.split(','),evidence_ids=evidence.split(','),
        status=status,limitation=limitation,analysis_or_action_needed=action,resolution_command=command,
        command_stage=stage,candidate_requirement_closed=False)


def matrix():
    check = 'cd "$ENGINE_SOURCE" && "$ENGINE_PYTHON" -m engine.cli --source-root "$ENGINE_SOURCE" check'
    assess = 'cd "$ENGINE_SOURCE" && "$ENGINE_PYTHON" -m engine.cli assess --case "$FINAL_CASE" --protocol "$FINAL_PROTOCOL"'
    dimension = 'cd "$ENGINE_SOURCE" && "$ENGINE_PYTHON" -m engine.cli dimension --design configs/power_design.json --predictions "$PLANNING_PREDICTIONS" --receipts "$PLANNING_RECEIPT" --output "$POWER_RECEIPT"'
    return [
 row('H01','Estimando, unidad temporal, pareado y familia completa declarados','rubric,unit','protocol,methods,assessment,methods_input_audit,scientific_final_gate','PRIMARY_ENDPOINT_ALIGNMENT_UNRESOLVED',
     'La media de RMSE individuales no es la raíz del MSE agregado ni el RMSE del ensamble. Una prueba sobre MSE no se convierte en inferencia sobre la primaria heredada por exigir además dirección favorable. El contraejemplo positivo conservado invierte el ranking de ambos funcionales. Fechas correlacionadas y semillas no crean R×T unidades independientes.',
     'Preservar la primaria heredada y validar inferencia para ese funcional, o documentar una enmienda prospectiva realmente aceptada y su alcance; no se fabrica esa aceptación ni se cambia el resultado antiguo. El preflight nuevo mantiene bloqueada la confirmación.',
     '"$ENGINE_PYTHON" experiments/e3_hypothesis_adequacy_20261122/methods_power/audit_power_inputs.py --source-root "$ENGINE_SOURCE" --verify-ledgers --output /tmp/e3-power-input-recheck.json','executed_scientific_preflight_no_automatic_resolution'),
 row('H02','Tres procedimientos distintos y pertinentes por contraste','rubric,three,all_reject','methods,conditional_validation,method_probe','CONDITIONALLY_VALIDATED_METHOD_ONLY',
     'DM-HLN/HAC, bootstrap estacionario-t y GW son distintos; variantes de bloques no cuentan como procedimientos nuevos. GW rechaza momentos de forma no direccional.',
     'Aplicar las tres reglas predefinidas a la familia completa después de datos, potencia y reserva admitidos; no usar sólo la prueba favorable.',assess,'future_saved_final_predictions_only'),
 row('H03','Supuestos, dependencia temporal y alcance del validador','rubric,assumptions','conditional_validation,method_probe,statistics','CONDITIONAL_DGP_EVIDENCE_REUSED',
     'La validación es por panel y ajustes R10/T768/B8191/HAC48/bloque32/alpha0.025, con cuatro escenarios previos y un nulo nuevo; no prueba validez para cualquier mercado ni un Monte Carlo conjunto de cuatro paneles.',
     'Auditar estacionariedad/dependencia, FCLT/momentos/instrumentos/no singularidad en desarrollo; si cambian ajustes o surge una brecha concreta fijar antes nueva validación. No repetir MC sólo para sumar evidencia.',check,'reuse_executed_method_evidence'),
 row('H04','P crudos y ajustados, Holm y familias declaradas','rubric,raw_adjusted,holm','protocol,methods,new_diagnostics,methods_input_audit,scientific_final_gate','FULL_COMPOSITE_ERROR_CONTROL_UNRESOLVED',
     'Holm128 es una familia exploratoria, distinta de principal120 y adicionalEPA180. Holm no corrige p marginales inválidos. Las cotas puntuales del validador establecen q≤0.05 por panel, no q≤0.025 nominal; la composición positiva2/3 admite la cota conservadora2q. Con el peor límite CP95 actual0.0478608731,2q=0.0957217463 no certifica error global≤0.05. Esto no prueba inflación de la regla conjunta.',
     'Aportar validación prefijada o una composición formal pertinente de la regla completa y sus supuestos. Mantener alpha/regla2de3 intactos; la opción más estricta all3 ensayada lógicamente no fue activada. Separar familias, guardar p crudos/ajustados y no afirmar FWER global5% desde cotas marginales insuficientes.',
     '"$ENGINE_PYTHON" experiments/e3_hypothesis_adequacy_20261122/methods_power/audit_power_inputs.py --source-root "$ENGINE_SOURCE" --verify-ledgers --output /tmp/e3-power-input-recheck.json','executed_scientific_preflight_no_new_MC'),
 row('H05','Piloto de al menos cinco semillas para estimar dispersión pareada de cada rival','pilot','power_design,input_only_previous,new_retrospective','PLANNING_INPUTS_INCOMPLETE',
     'V5 ofrece cinco semillas y55fechas para seis recetas neuronales/dos controles; no cubre los veinte rivales ni los cuatro paneles del nuevo diseño. No sirve como sizing suficiente.',
     'Generar y respaldar las predicciones de planificación de todos los21 sistemas en cuatro paneles admitidos, con≥5 semillas neuronales y≥80fechas útiles por panel.',
     'cd "$ENGINE_SOURCE" && "$ENGINE_PYTHON" -m engine.cli predict-planning --case "$PLANNING_CASE" --source-commit "$ENGINE_COMMIT" --panels "$DEVELOPMENT_MAP" --repository "$REPOSITORY" --branch "$PLANNING_BRANCH" --prefix "$PLANNING_PREFIX"',
     'future_after_explicitly_activated_planning_training'),
 row('H06','Dimensionamiento suficiente: potencia≥0.90 por contraste, conjunta y MDE; mínimo diez semillas finales','power,joint,seeds,levels','power_design,protocol','NOT_DIMENSIONED_WITH_REAL_INPUTS',
     'final_R/T/B siguen nulos. La potencia512/512 de una alternativa sintética fuerte no es potencia de mercado. La potencia conjunta debe reportarse; la guía no le impone un mínimo universal0.90.',
     'Estimar el diseño sobre predicciones reales completas de planificación, contabilizando error Monte Carlo, semilla/tiempo y la regla final efectiva; aumentar semillas o informar MDE si no alcanza.',dimension,'future_real_planning_inputs_required'),
 row('H07','Tamaño del efecto, intervalo y relevancia práctica frente a delta','effect','adequacy_config,input_only_previous,new_temporal_intervals','REAL_RETROSPECTIVE_PARTIAL',
     'Las ganancias puntuales del piloto son desarrollo. Los IC temporales nuevos vs global contienen cero en ambos bloques; no prueban superar el margen prospectivo. Efectos sobre cinco semillas/calendario fijo no cubren variabilidad temporal futura.',
     'Conservar efectos descriptivos actuales; en confirmación reportar diferencia, IC y efecto pertinente para cada rival y panel con el margen original y el prospectivo distinguidos.',assess,'future_confirmatory_effect_analysis'),
 row('H08','Intervalos de confianza del95% con remuestreo apropiado a datos temporales','rubric,effect,unit','new_temporal_intervals,new_analysis_plan,methods','REAL_RETROSPECTIVE_PARTIAL',
     'CBB7/14 de3999réplicas cada uno es puntual, exploratorio y condicionado a cinco trayectorias guardadas. No tiene cobertura finita ni simultánea validada; los bloques son sensibilidad del mismo método.',
     'Reutilizar esos IC con sus límites. La confirmación necesita la regla temporal admitida, cobertura/alcance declarados y niveles95/99 sin renombrarlos.',
     '"$ENGINE_PYTHON" experiments/e3_hypothesis_adequacy_20261122/retrospective/audit_saved_adequacy.py --verify-only',
     'existing_saved_outputs_verification_no_resampling'),
 row('H09','Tablas A/B: supera baseline trivial, brecha ajuste-validación menor que margen y consistencia con prueba','adequacy,gap','new_retrospective,adequacy_config,historical_aditive_adequacy','NOT_RECONSTRUCTIBLE_FROM_CURRENT_PILOT',
     'No hay predicciones punto a punto de fit/selection del checkpoint seleccionado ni riesgo final de ajuste. Las curvas mean-batch no son ese riesgo. La auditoría aditiva no certifica input-only.',
     'Guardar fit/selection/calibration y test desde los mismos checkpoints seleccionados; fijar márgenes antes de la prueba y evaluar todos los criterios. No fabricar una brecha retrospectiva.',assess,'future_same_selected_state_split_predictions_required'),
 row('H10','Diagnósticos temporales de residuos y calibración lineal pertinentes','rubric,residuals','new_diagnostics,new_retrospective,historical_v4','REAL_RETROSPECTIVE_EXECUTED_NOT_ADEQUACY_PASS',
     'Holm128 tiene27rechazosMZ, incluidos2/5input-only; LB/ARCH sin rechazos no certifican estacionariedad, homocedasticidad ni adecuación. Con55fechas las referencias clásicas/HAC son asintóticas, no calibración finita.',
     'Reportar todos los diagnósticos, tamaños, supuestos, sesgos e IC actuales; aplicar los criterios positivos prospectivos de tolerancia/residuos sólo con el diseño admitido. BDS/JB se justifican por pertinencia, no por acumular pruebas.',assess,'new_retrospective_available_future_confirmatory_assessment'),
 row('H11','Calibración apropiada a regresión e intervalos con cobertura por subgrupos','coverage,adequacy','new_retrospective,adequacy_config,assessment','NO_SAVED_CANDIDATE_INTERVALS',
     'La cabeza MSE y los átomos puntuales no definen densidad calibrada. No hay cuantiles/ley guardada para reconstruir cobertura o PIT; ECE/Brier de clasificación no se trasladan mecánicamente a sigma7.',
     'Ajustar radios/intervalos sólo en calibración antes de test; guardar cuantiles, cobertura nominal/empírica, ancho y subgrupos prefijados. Declarar dependencia/no garantía de intercambiabilidad. No recalibrar sobre el piloto expuesto para certificarlo.',assess,'future_calibration_then_sealed_coverage_required'),
 row('H12','Tabla MCS o diagrama pertinente que incluya el modelo propio','mcs,mcs_forecast','historical_aditive_mcs,assessment,methods,candidate_secondary_mcs,candidate_mcs_results','PARTIAL_EXECUTED_SECONDARY_FIXED_COHORT_MCS',
     'El nuevo Tmax estándar incluye input-only en un MCS90% de8recetas/32trayectorias/55fechas expuestas, con bloques7/14 y1999sorteos cada uno. En ambos retiene7recetas y excluye sólo floor_only; no identifica ganador único. Su pérdida es MSEmedio por fecha con5semillas fijas, no la primaria mediaRMSE ni toda la familia21. La cobertura finita y estacionariedad no se validan por ejecutar el algoritmo; no es la vieja extensión fallida.',
     'Reutilizar el complemento en su alcance secundario, mostrando supervivientes con p individual no definido. Para el criterio completo falta familia/estimando pertinentes y datos reales reservados, con nivel, supuestos y membresía reportados sin p direccionales inventados. No cuenta como tercer contraste primario ni justifica otro aumento de nota por inclusión.',None,'secondary_saved_pilot_diagnostic_executed_not_primary_confirmation'),
 row('H13','Prueba entre conjuntos, resultados por conjunto y consistencia adicional','between,two_axes,same_task','protocol,assessment','IMPLEMENTED_NOT_REAL_FOUR_PANEL_EXECUTED',
     'Friedman/Iman-Davenport y posthoc están implementados, pero las cuatro propuestas carecen de admisiones reales. Paneles cripto/perpetuos pueden ser económicamente dependientes; enumeración de signos exige independencia/simetría no certificadas.',
     'Justificar dos ejes por adicional antes del desempeño; admitir datos, ejecutar familia completa y reportar diferencias/IC/efectos separados. Justificar validez entre conjuntos; no promediar métricas de escalas distintas ni afirmar independencia por ticker.',assess,'future_admitted_four_panel_evaluation'),
 row('H14','Sensibilidad de la conclusión a prueba, métrica secundaria y subconjunto de semillas','sensitivity,all_reject','input_only_previous,new_temporal_intervals,adequacy_config','DESCRIPTIVE_SENSITIVITY_REUSED_AND_EXTENDED',
     'Leave-out/mes/semillas previos y bloques7/14 nuevos son desarrollo; MAE y RMSE no siempre coinciden. No existen aún una decisión confirmatoria propia ni su estabilidad frente a toda la familia.',
     'Reutilizar sensibilidades sin escoger regiones favorables. Tras congelar, repetir la conclusión completa con las alternativas prefijadas; justificar Nadeau-Bengio no aplicable a una única reserva cronológica sin particiones repetidas.',assess,'future_frozen_sensitivity_analysis'),
 row('H15','Prueba única, exposición y procedencia de datos/checkpoints sin fuga','sealed,unit','protocol,assessment,new_analysis_plan','SOFTWARE_GUARDS_EXECUTED_AUTHENTIC_FINAL_INPUTS_ABSENT',
     'Las pruebas de software acreditan guardas y recuperación, no una reserva real. El calendarioV5 se examinó y no puede reclasificarse como independiente. Elegir otros archivos/tickers no recupera cegamiento.',
     'Documentar exposición real, datos/licencias/hash y checkpoints seleccionados; cerrar protocolo y reserva válida antes de activación final. Mantener la consulta única y conservar fallos/adversos.',check,'source_checks_only_then_authentic_final_evidence'),
 row('H16','Reporte completo por contraste y conclusión coherente con todas las pruebas','rubric,raw_adjusted,all_reject','new_diagnostics,new_temporal_intervals,assessment','REAL_EXPLORATORY_TABLES_AVAILABLE_CONFIRMATION_ABSENT',
     'Las tablas nuevas separan p diagnósticos de IC de diferencias; no inventan p de superioridad ni familias completas actuales. El resultado más favorable no determina la decisión R3.',
     'Conservar estadístico/nulo/supuestos/pcrudo/pajustado/IC/efecto/tamaño y estados no disponibles por contraste; mantener la decisión sobre la peor prueba exigida y todos los rivales.',assess,'future_sealed_report_plus_existing_exploratory_tables'),
 ]


def run(args):
    started=now()
    if sha(args.guide_pdf)!=GUIDE_SHA:
        raise ValueError('The original guide PDF hash differs')
    pages=subprocess.run(['pdftotext','-layout',str(args.guide_pdf),'-'],check=True,capture_output=True,text=True).stdout.split('\f')
    citation_records=[]
    for id,(page,section,quote) in CITATIONS.items():
        found=norm(quote) in norm(pages[page-1])
        # PDF line-end discretionary hyphenation is preserved in some builds.
        if not found:
            found=norm(quote) in norm(re.sub(r'-\s*\n\s*','',pages[page-1]))
        if not found:
            raise ValueError('Guide citation not found at stated page: '+id)
        citation_records.append(dict(id=id,page=page,section=section,quote=quote,verified=True))
    roots=dict(repo=args.repo_root,engine=args.engine_root,delivery=args.delivery_root)
    evidence={}
    for id,(namespace,relative) in EVIDENCE.items():
        p=roots[namespace]/relative
        if not p.is_file():
            raise ValueError('Evidence file missing: '+str(p))
        evidence[id]=dict(namespace=namespace,path=relative,bytes=p.stat().st_size,sha256=sha(p))
    baseline=json.loads((args.repo_root/EVIDENCE['baseline'][1]).read_text())
    component=[c for c in baseline['components'] if c['weight_percent']==15]
    assert len(component)==1 and component[0]['score_0_to_5']=='2.8'
    assert baseline['oral_defense_author_provided']=='4.5' and not baseline['oral_defense_observed']
    assert sum(c['weight_percent'] for c in baseline['components'])==100
    method=json.loads((args.engine_root/EVIDENCE['conditional_validation'][1]).read_text())
    assert method['status']=='PASSED_PREDECLARED_VALIDATION'
    assert method['conditional_scope_admitted'] is True and method['general_validation_established'] is False
    power=json.loads((args.engine_root/EVIDENCE['power_design'][1]).read_text())
    assert power['status']=='NOT_DIMENSIONED' and all(power[k] is None for k in ('final_R','final_T','final_B'))
    retrospective=json.loads((args.delivery_root/EVIDENCE['new_retrospective'][1]).read_text())
    assert retrospective['n_dates']==55 and retrospective['n_trajectories']==32 and retrospective['n_point_predictions']==1760
    assert retrospective['candidate_diagnostic_rejections']['MZ_HAC14']==2
    assert retrospective['full_Tables_A_B_reconstructed'] is False and retrospective['predictive_interval_coverage_reconstructed'] is False
    assert retrospective['R3_accredited'] is False
    records=matrix()
    assert len({r['id'] for r in records})==len(records)
    assert all(set(r['evidence_ids'])<=set(evidence) and set(r['citation_ids'])<=set(CITATIONS) for r in records)
    previous=args.previous_matrix
    assert previous.is_file(), 'Preserve the original matrix as results_v1 before this version'
    result=dict(schema_version=2,audit_version='guide-evidence-audit-2',scope='Executed guide/evidence concordance for the15percent component; no academic accreditation or new grade',
        started_utc=started,finished_utc=now(),guide_sha256=GUIDE_SHA,reviewed_source_commit='fdf697aa870571b4e333a32198aa97fec69c9bad',
        previous_matrix_path='results_v1/requirements_matrix.json',previous_matrix_sha256=sha(previous),audit_script_sha256=sha(__file__),
        baseline_component_score='2.8',component_weight_percent=15,author_oral_defense_value='4.5',oral_defense_observed=False,
        new_component_score=None,official_grade=None,R3_accredited=False,
        matrix_rows_overlap_not_weights=True,requirements=records,citations=citation_records,evidence=evidence,
        academic_guardrails=['A posterior analysis is not a preregistration or independent confirmation',
            'No seed/date pseudoreplication; no absence-of-rejection adequacy certificate',
            'The guide requires joint power reporting, not universal joint power>=0.90',
            'Actions/Nasdaq/1.5x size ratio are not guide mandates',
            'Adverse results and authorized declared AI review are not grading penalties',
            'No mandatory independent full retraining is invented for this15percent component'],
        executed_operations=dict(guide_citations_checked=len(citation_records),artifact_hashes_checked=len(evidence),
            new_market_downloads=0,new_model_fits=0,new_model_forwards=0,new_bootstrap_replicates=0,new_final_test_accesses=0),
        AI_assistance_declared=True)
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/'requirements_matrix.json'
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    with (args.output/'requirements_matrix.csv').open('w',newline='') as stream:
        fields=list(records[0]);writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,list) else v for k,v in record.items()})
    receipt=dict(status='GUIDE_AND_EVIDENCE_AUDIT_EXECUTED',audit_version='guide-evidence-audit-2',finished_utc=now(),requirements=len(records),
        citations_checked=len(citation_records),artifacts_hashed=len(evidence),matrix_sha256=sha(output),
        previous_matrix_sha256=sha(previous),audit_script_sha256=sha(__file__),
        retrospective_evidence_reused=True,method_validation_reused=True,new_stochastic_operations=0,
        R3_accredited=False,new_grade_assigned=False,AI_assistance_declared=True)
    (args.output/'execution_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--guide-pdf',type=Path,default=Path('/workspace/attachments/69c1e483-d404-4ff8-90e9-6e019a764510/Entregable3DeepLearningProject.pdf'))
    parser.add_argument('--repo-root',type=Path,default=Path('/workspace/e3_completion_worktree'))
    parser.add_argument('--engine-root',type=Path,default=Path('/workspace/e3_confirmatory_engine_20261121'))
    parser.add_argument('--delivery-root',type=Path,default=HERE.parent)
    parser.add_argument('--previous-matrix',type=Path,default=HERE/'results_v1/requirements_matrix.json')
    parser.add_argument('--output',type=Path,default=HERE/'results')
    run(parser.parse_args())
