"""Version the requested orientative component reassessment, without grading R3."""
from copy import deepcopy
import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
BASELINE=Path('/workspace/e3_completion_worktree/book_e3/_static/e3_btc_scope_assessment_20261121/assessment.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(baseline_path=BASELINE):
    baseline=json.loads(baseline_path.read_text())
    old=next(c for c in baseline['components'] if c['weight_percent']==15)
    assert old['component']=='Pruebas de hipótesis y adecuación' and old['score_0_to_5']=='2.8'
    assert baseline['weighted_score_exact']=='4.092' and baseline['oral_defense_author_provided']=='4.5'
    assert baseline['oral_defense_observed'] is False and baseline['official_grade'] is None
    retrospective=json.loads((ROOT/'retrospective/results/summary.json').read_text())
    assert retrospective['n_point_predictions']==1760 and retrospective['n_dates']==55
    assert retrospective['candidate_diagnostic_rejections']['MZ_HAC14']==2 and not retrospective['R3_accredited']
    after=deepcopy(baseline['components'])
    new=next(c for c in after if c['weight_percent']==15)
    new.update(score_0_to_5='3.0',contribution_0_to_5='0.45',
        reason='Los nuevos diagnósticos de residuos de la candidata con Holm128 y sus intervalos temporales pareados mejoran análisis e interpretación; potencia real, inferencia de la primaria heredada, regla conjunta y adecuación final siguen sin establecerse.',
        evidence='experiments/e3_hypothesis_adequacy_20261122/retrospective/results/summary.json')
    old_total=sum(Decimal(str(c['weight_percent']))*Decimal(c['score_0_to_5'])/100 for c in baseline['components'])
    new_total=sum(Decimal(str(c['weight_percent']))*Decimal(c['score_0_to_5'])/100 for c in after)
    assert old_total==Decimal('4.092') and new_total==Decimal('4.122') and new_total-old_total==Decimal('.030')
    assert sum(c['weight_percent'] for c in after)==100
    assert all(a==b for a,b in zip(after,baseline['components']) if a['weight_percent']!=15)
    stamp=dt.datetime.now(dt.timezone.utc)
    refs={name:dict(path=relative,sha256=sha(ROOT/relative)) for name,relative in {
        'candidate_residuals_and_intervals':'retrospective/results/summary.json',
        'candidate_diagnostic_family':'retrospective/results/diagnostics_Holm128.csv',
        'temporal_intervals':'retrospective/results/temporal_RMSE_MAE_intervals.csv',
        'independent_review':'guide_audit/retrospective_independent_review.json',
        'guide_matrix':'guide_audit/results/requirements_matrix.json',
        'method_power_preflight':'methods_power/current_input_audit_v2.json',
        'final_scientific_gate':'methods_power/final_gate.json',
        'secondary_MCS':'mcs/results/summary.json'}.items()}
    result=dict(schema_version=1,assessment_version='hypothesis-adequacy-evidence-delta-20261122-1',
        assessed_utc=stamp.isoformat(),date_local=stamp.astimezone(ZoneInfo('America/Bogota')).date().isoformat(),timezone='America/Bogota',
        scope='Independent orientative AI reassessment requested by author; not an official teacher grade or academic acceptance',
        baseline_assessment_version=baseline['assessment_version'],baseline_assessment_sha256=sha(baseline_path),
        component=old['component'],weight_percent=15,previous_score_0_to_5='2.8',new_score_0_to_5='3.0',
        component_delta='0.2',old_weighted_contribution='0.42',new_weighted_contribution='0.45',global_weighted_delta='0.030',
        old_weighted_grade_exact=str(old_total),new_weighted_grade_exact=str(new_total),new_weighted_grade_display='4.12',
        oral_defense_author_provided='4.5',oral_defense_observed=False,oral_defense_contribution='0.315',
        all_weights_unchanged=True,all_other_component_scores_unchanged=True,components=after,
        cap_if_R1_R2_accepted_R3_not_accredited='4.0',old_conditional_capped_grade='4.00',new_conditional_capped_grade='4.00',
        cap_if_R1_or_R2_assessed_not_satisfied='3.0',official_grade=None,R1_academic_acceptance='pending',R2_academic_acceptance='pending',R3_accredited=False,
        judgement_level='en desarrollo; no official automatic numeric mapping of guide levels',
        evidence_justifying_delta=['New candidate-specific 128 residual diagnostic positions with Holm128 and64 pointwise HAC bias intervals on32 saved trajectories',
            'New paired chronological RMSE/MAE intervals across3999draws each atblock7/14, conditional on5storedseed trajectories, preserving uncertainty and all7pilot comparators',
            'Independent archive arithmetic and Holm cross-check; explicit adverse/null findings and limits'],
        no_additional_credit_for=['Writing the matrix or updating documentation','Prepared acquisition, notebook, tests or unexecuted planning',
            'Additional method calibration reuse as if it were new market evidence','Being retained in secondary8-recipe MCS; it reinforces reporting without another numeric increment'],
        adverse_findings_preserved={'candidate_MZ_Holm128_rejections':'2 of5 trajectories','other_LB_ARCH_nonrejections':'not adequacy certificates',
            'primary_mean_RMSE_difference_vs_global':'-0.00005821987228066001',
            'block7_CI95':[-0.00013727536780667175,0.00002937366637729736],
            'block14_CI95':[-0.00011955730570656036,0.000019106373050022486],
            'interval_interpretation':'Both include zero; not proof of superiority, equality, power or unconditional seed-population performance'},
        unresolved=['Inherited mean individual-seed RMSE versus tested aggregateMSE endpoint alignment',
            'Error control for the exact composite primary/additional2-of3decision, beyond pointwise one-panel calibration',
            'Real complete four-panel21-system planning and sufficient power/MDE before final closure',
            'Complete selected-checkpoint fit/selection point forecasts for TablesA/B and calibration/coverage',
            'Current full-family primary-endpoint MCS or pertinent product and justified between-panel assumptions',
            'Real reserved confirmation, effects, robustness and authentic academic R1/R2 acceptance'],
        evidence=refs,forecast_fits_executed_this_review=0,new_model_forwards=0,new_market_downloads=0,new_stochastic_draws=0,
        AI_assistance_declared=True)
    (HERE/'COMPONENT_REASSESSMENT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    md='''# Revaloración orientativa del componente de hipótesis y adecuación

La valoración independiente solicitada cambia **2,8 → 3,0/5** para el componente
del **15 %**, exclusivamente por nueva evidencia numérica ejecutada sobre la
candidata actual. Es un juicio documental por IA; no es nota oficial ni una
equivalencia automática de los niveles de la guía. El componente permanece en
desarrollo.

| Magnitud | Anterior | Revisión actual |
|---|---:|---:|
| Componente hipótesis/adecuación | 2,8 | **3,0** |
| Aporte ponderado de ese componente | 0,420 | **0,450** |
| Puntaje ponderado completo, sin redondear | 4,092 | **4,122** |
| Puntaje mostrado antes del tope | 4,09 | **4,12** |
| Sustentación indicada por el autor | 4,5 | **4,5** |
| Nota bajo R1/R2 aceptadas y R3 no acreditada | 4,00 | **4,00** |

El cambio global es exactamente **0,030 puntos**. Todos los pesos y las otras
diez notas permanecen idénticos. La sustentación aporta 0,315, se trata como valor
del autor y no como defensa observada o R2 acreditada. La nota oficial sigue
vacía. Si se determina incumplimiento de R1 o R2, el tope pertinente sería 3,0;
una evaluación pendiente no se convierte automáticamente en ese incumplimiento.

## Evidencia que sí justifica el incremento limitado

Sobre 1760 predicciones ya guardadas, 32 trayectorias, 55 fechas expuestas y cinco
semillas se ejecutaron 128 posiciones de diagnósticos con Holm128, 64 intervalos
HAC puntuales de sesgo y dos remuestreos temporales pareados de 3999 sorteos
cada uno. Se conservaron los siete comparadores del piloto y la media de RMSE
individuales, con fechas comunes y semillas fijas. La lectura e integridad se
verificaron contra el ZIP original; una revisión separada confirmó la aritmética
y Holm mediante una biblioteca independiente. Estos diagnósticos/IC propios de
la candidata faltaban en el corte valorado en 2,8.

La mejora reconoce ejecución y calidad de interpretación, **no un resultado
favorable**. Hay 2/5 rechazos Mincer–Zarnowitz de input-only tras Holm128. Los
intervalos RMSE contra global incluyen cero con ambos bloques: bloque 7,
[-0,0001372754;0,0000293737], y bloque 14,
[-0,0001195573;0,0000191064]. Se reportan puntuales, exploratorios y condicionados
a las trayectorias guardadas, sin cobertura finita validada. Los no rechazos de
Ljung–Box/ARCH no demuestran adecuación; los IC que incluyen cero no prueban
equivalencia ni insuficiencia de la candidata en toda aplicación futura.

El complemento MCS estándar retiene a input-only entre siete recetas del piloto,
para pérdida MSE secundaria. Se incorpora como reporte ejecutado de alcance
limitado, **sin otro aumento por inclusión**, sin ganador único y sin sustituir
la primaria media-RMSE o la familia de 21 sistemas. La guía/matriz y las preparaciones de datos,
notebook y software no reciben crédito como experimentos de mercado.

## Límites que mantienen la valoración en desarrollo

El preflight identifica que la inferencia sobre MSE/raíz agregada no es una
prueba de la primaria heredada media-RMSE y que las cotas marginales por panel no
certifican por sí solas la regla compuesta adicional2/3 a error global 0,05. La
cota conservadora 2q=0,0957217463 no establece ese objetivo, sin demostrar por sí
misma inflación real. No se cambian automáticamente alpha, endpoint o reglas.

Faltan predicciones completas de planificación de 21 sistemas × 4 paneles y un dimensionamiento real,
predicciones fit/selection del mismo checkpoint para Tablas A/B, calibración y
cobertura de la candidata, producto MCS pertinente de familia completa y una
confirmación reservada con sus efectos y robustez. La nueva revisión no acredita
R3 ni transforma datos de desarrollo en prueba independiente.

Los hashes, fuentes y cálculos exactos están en COMPONENT_REASSESSMENT.json.
Asistencia de IA declarada; no se atribuye esta revisión a un tercero humano.
'''
    (HERE/'COMPONENT_REASSESSMENT.md').write_text(md)
    print(json.dumps({k:result[k] for k in ('previous_score_0_to_5','new_score_0_to_5','old_weighted_grade_exact','new_weighted_grade_exact','new_conditional_capped_grade','R3_accredited')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    args=parser.parse_args()
    run(args.baseline)
