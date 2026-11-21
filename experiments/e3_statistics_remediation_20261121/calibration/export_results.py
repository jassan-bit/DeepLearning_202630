"""Read-only postprocessing of the fixed validation; no RNG, fits or forecasts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')


def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def export(root):
    root=Path(root);run=root/'validation_fresh';receipt=json.loads((run/'execution_receipt.json').read_text())
    summary=json.loads((run/'validation_summary.json').read_text());cases=summary['cases']
    tables=root/'tables';tables.mkdir(exist_ok=True)
    nulls=[c for c in cases if c['global_EPA_null_true']]
    new='MCS_bootstrap_t';null_status=[c[new]['initial_global_rejection']['screen'] for c in nulls]
    complete=receipt['status']=='completed' and receipt['completed_replicates']==640 and len(cases)==4
    if not complete:status='partial_no_calibration_conclusion'
    elif any(c[new]['unsupported_replicates'] for c in nulls):status='unsupported_do_not_use'
    elif 'inflation_detected' in null_status:status='inflation_detected_do_not_use'
    elif all(x=='exploratory_tolerance_screen_passed' for x in null_status):status='exploratory_tolerance_screen_passed_declared_nulls_only'
    else:status='inconclusive_no_nominal_level_certificate'
    mcs=[];coverage=[];family=[];paired=[]
    for c in cases:
        for variant in [new,'MCS_fixed_studentizer_development_control','MCS_legacy_frozen_control_same_draws']:
            v=c[variant];p=v['initial_global_rejection'];ci=p['CI95']
            mcs.append({'case':c['id'],'global_EPA_null_true':c['global_EPA_null_true'],'variant':variant,
                        'events':p['events'],'replicates':p['replicates'],'rejection_rate':p['rate'],
                        'individual_CP95_lower':ci[0],'individual_CP95_upper':ci[1],
                        'exploratory_screen':p.get('screen','alternative_power_not_size'),
                        'unsupported_replicates':v['unsupported_replicates'],
                        'all_true_best_retained_rate':v['all_true_best_retained']['rate'],
                        'unique_candidate_set_rate':v['unique_candidate_set']['rate']})
        f=c.get('known_two_method_subset_FWER_Holm38');p=c.get('two_method_joint_alternative_rejection_Holm38')
        family.append({'case':c['id'],'replicates':c['completed_replicates'],
                        'two_method_Holm38_FWER':None if f is None else f['rate'],
                        'FWER_individual_CP95_lower':None if f is None else f['CI95'][0],
                        'FWER_individual_CP95_upper':None if f is None else f['CI95'][1],
                        'FWER_exploratory_screen':None if f is None else f['screen'],
                        'two_method_joint_alternative_power':None if p is None else p['rate'],
                        'joint_power_individual_CP95_lower':None if p is None else p['CI95'][0],
                        'joint_power_individual_CP95_upper':None if p is None else p['CI95'][1],
                        'full_three_method_power':False})
        for r in c['per_rival']:
            for method,v in r['methods'].items():
                p=v['pointwise_CI95_coverage']
                coverage.append({'case':c['id'],'rival':r['rival'],'method':method,
                                  'truth_theta':r['true_theta'],'bias':r['bias_against_longrun_truth'],
                                  'coverage':p['rate'],'individual_CP95_lower':p['CI95'][0],
                                  'individual_CP95_upper':p['CI95'][1],
                                  'unsupported_intervals':v['unsupported_intervals'],
                                  'Holm38_rejection':v['Holm38_rejection']['rate'],
                                  'raw_rejection':v['raw_rejection']['rate']})
        for p in c.get('paired_variant_diagnostics',[]):paired.append({'case':c['id'],**p})
    csv_file(tables/'mcs_calibration_comparison.csv',mcs)
    csv_file(tables/'p1_p2_coverage_by_rival.csv',coverage)
    csv_file(tables/'family_fwer_and_subset_power.csv',family)
    csv_file(tables/'paired_mechanism_comparisons.csv',paired)
    decision={'kind':'POST_RESULT_BOUNDED_CONTROLLED_VALIDATION_DECISION_NOT_PREREGISTRATION',
               'new_method':new,'controlled_screen_status':status,
               'actual_completed_replicates':receipt['completed_replicates'],
               'actual_crossed_bootstrap_draws':receipt['actual_crossed_bootstrap_draws'],
               'legacy_control_extra_draws':receipt['old_control_extra_bootstrap_draws'],
               'actual_elapsed_seconds':receipt['elapsed_seconds'],
               'public_source_and_plan_anchor':receipt['public_anchor_commit'],
               'source_files_unchanged_after_results':receipt['source_hashes_still_match'],
               'null_case_screens':{c['id']:c[new]['initial_global_rejection'] for c in nulls},
               'two_method_Holm38_FWER_screens':{c['id']:c['known_two_method_subset_FWER_Holm38'] for c in nulls},
               'primary_pairwise_battery_status':'not_approved_empirical_false_positive_inflation' if any(
                   c['known_two_method_subset_FWER_Holm38']['screen']=='inflation_detected' for c in nulls)
                   else 'not_complete_or_not_fully_validated',
               'alternative_diagnostics':{c['id']:{
                   'MCS_global_rejection':c[new]['initial_global_rejection'],
                   'MCS_unique_candidate_set':c[new]['unique_candidate_set'],
                   'two_method_joint_Holm38_power':c.get('two_method_joint_alternative_rejection_Holm38'),
                   'two_method_joint_power_scope':'Intersection of adjusted P1/P2 equality-test p-values only; excludes the full practical-margin/effect/adequacy decision rule'}
                   for c in cases if not c['global_EPA_null_true']},
               'individual_MC_intervals_not_simultaneous_guarantee':True,
               'a_passed_tolerance_screen_is_not_exact_nominal_level_certification':True,
               'methodological_admissibility_for_primary_market_question_certified':False,
               'third_primary_pairwise_method_accredited':False,
               'complete_three_method_power_computed':False,'final_market_power_certified':False,
               'R3_accredited':False,'confirmation_allowed':False,'new_market_fits':0,'new_market_predictions':0,
               'V3_NO_GO_unchanged':True,'V4_abandonment_unchanged':True,
               'statistical_methods_retuned_after_fresh_results':False,
               'AI_assistance_declared':True,
               'raw_validation_log_SHA256':hashlib.sha256((run/'per_replicate.jsonl.gz').read_bytes()).hexdigest()}
    write(root/'RESULTS_DECISION.json',decision)
    lines=['# Resultado posterior de la validación controlada nueva','',
           'Este documento es posterior a los resultados; no reemplaza el plan ni sus fuentes congeladas.',
           f"Se completaron **{receipt['completed_replicates']}** réplicas y **{receipt['actual_crossed_bootstrap_draws']}** remuestreos cruzados; "
           f"el control antiguo añadió **{receipt['old_control_extra_bootstrap_draws']}** muestras. Tiempo del controlador: **{receipt['elapsed_seconds']:.3f} s**.",
           f"Plan y código públicos antes de ejecutar: `{receipt['public_anchor_commit']}`. Fuentes intactas: `{receipt['source_hashes_still_match']}`.",
           '', '| Caso nulo | Nuevo bootstrap-t | IC95 % individual CP | Cribado de desarrollo |',
           '|---|---:|---|---|']
    for c in nulls:
        v=c[new]['initial_global_rejection'];lo,hi=v['CI95']
        lines.append(f"| {c['id']} | {v['events']}/{v['replicates']} = {100*v['rate']:.2f} % | {100*lo:.2f}–{100*hi:.2f} % | {v['screen']} |")
    lines += ['',f"**Decisión del cribado controlado: `{status}`.** El umbral era límite superior CP ≤0,13 frente a nominal0,10. "
               'Es una tolerancia exploratoria individual; no certifica nivel exacto, garantía simultánea o validez universal.',
              '', '**Resultado adverso que permanece:** el subconjunto primario de dos métodos no queda aprobado. '
              'Sus tasas de falso positivo familiar corregidas por Holm se conservan a continuación; no se elimina '
              'después el método o rival que provoca el problema.',
              '', '| Caso nulo | FWER de P1/P2 con Holm38 | IC95 % individual CP | Decisión exploratoria |',
              '|---|---:|---|---|']
    for c in nulls:
        v=c['known_two_method_subset_FWER_Holm38'];lo,hi=v['CI95']
        lines.append(f"| {c['id']} | {v['events']}/{v['replicates']} = {100*v['rate']:.2f} % | {100*lo:.2f}–{100*hi:.2f} % | {v['screen']} |")
    for c in cases:
        if c['global_EPA_null_true']:continue
        g=c[new]['initial_global_rejection'];u=c[new]['unique_candidate_set'];p=c['two_method_joint_alternative_rejection_Holm38']
        lines += ['',f"En la alternativa predeclarada, el MCS nuevo rechaza globalmente en **{g['events']}/{g['replicates']}** "
                  f"({100*g['rate']:.2f} %), pero retiene sólo al candidato en **{u['events']}/{u['replicates']}** "
                  f"({100*u['rate']:.2f} %). La potencia conjunta del subconjunto de dos métodos es **{p['events']}/{p['replicates']}** "
                  f"({100*p['rate']:.2f} %; IC95 % {100*p['CI95'][0]:.2f}–{100*p['CI95'][1]:.2f} %). "
                  'La cifra conjunta cuenta sólo la intersección de los p-valores ajustados P1/P2; '
                  'no incorpora toda la regla de márgenes, efectos y adecuación. No es potencia de una batería completa '
                  'ni satisface por sí sola el diseño prospectivo de mercado.']
    lines += [
              '', 'Las tres variantes de MCS son comparaciones de una extensión metodológica, no tres pruebas primarias distintas. '
              'El MCS original no se altera ni se declara erróneo por usar studentizadores fijos. Las diferencias entre v1 y v2 incluyen '
              'cambio del estimador de covarianza y re-studentización; el control nuevo de studentizador fijo distingue esta última intervención.',
              '', 'Los CSV de `tables/` conservan todos los casos, variantes, intervalos por rival, FWER del subconjunto de dos métodos '
              'y seis comparaciones pareadas corregidas por Holm. El escenario alternativo informa potencia sintética de ese subconjunto, '
              'sin certificar ≥0,90 por contraste o potencia conjunta de una batería primaria completa.',
              '', '**No se habilita confirmación ni se acredita R3.** Faltan admisibilidad y calibración suficiente de tres procedimientos '
              'primarios pertinentes, candidata viable, paneles admitidos y dimensionamiento prospectivo real. V3 conserva NO-GO y V4 sigue abandonada.',
              '', 'No se retocaron métodos, parámetros, bloques, semillas o criterios después de esta validación. Asistencia de IA declarada.']
    (root/'RESULTS_DECISION.md').write_text('\n'.join(lines)+'\n')
    return decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args()
    print(json.dumps(export(a.root),sort_keys=True,ensure_ascii=False),flush=True)
