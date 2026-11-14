#!/usr/bin/env python3
"""Rebuild an explicit, retrospective adjudication of 30 named recovery targets.

The decision constants are an individual evidence review, not rules that screen
the 9,559 bibliographic groups. No manuscript reading is claimed by this script.
"""
import collections
import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, help='Repository with book_e2/ and references/.')
parser.add_argument('--output', type=Path, help='Destination; defaults to this script directory.')
args = parser.parse_args()
repository_candidates = [Path.cwd(), *Path.cwd().parents,
                         Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
ROOT = args.root.resolve() if args.root else next(
    (p for p in repository_candidates if (p / 'book_e2').is_dir() and (p / 'references').is_dir()), None)
if ROOT is None:
    parser.error('Repository not found among ancestors; pass --root /path/to/repository.')
OUT = args.output.resolve() if args.output else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
DATE = '2026-11-14'
INPUTS = set()


def read(rel):
    INPUTS.add(rel)
    return json.loads((ROOT / rel).read_text())


def rows(rel):
    INPUTS.add(rel)
    with (ROOT / rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def write_csv(name, data):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        for obj in data:
            w.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                        if isinstance(v, (list, dict)) else v for k, v in obj.items()})


E2 = 'references/literature_e2/'
E1 = 'references/literature_e1/'
native_path = E2 + 'native_20261108/native/cribado_metadatos.csv'
native = rows(native_path)
native_id = {x['study_id']: x for x in native}
native_doi = {x['doi'].lower(): x for x in native if x['doi']}
nodes = rows(E2 + 'final_audit_20261113/source_nodes.csv')
groups = {}
for x in nodes:
    if x['doi']:
        groups.setdefault(x['doi'].lower(), set()).add(x['unified_group_id'])

completion_path = E2 + 'completion_20261106/applied_fulltext_recovery_status.json'
completion = read(completion_path)
follow_path = E2 + 'screening_followup_20261108/source_reviews.json'
follow = read(follow_path)['studies']
second_path = E2 + 'recovery_second_20261108/source_reviews.json'
second = read(second_path)['studies']
second_summary_path = E2 + 'recovery_second_20261108/summary.json'
second_summary = read(second_summary_path)
primary_path = E1 + 'native_620_screening_20261108/primary_assessments.json'
primary = read(primary_path)['reviews']
primary_doi = {x['doi'].lower(): x for x in primary if x.get('doi')}
miura_path = E1 + 'selection_20261105/decision.json'
miura = read(miura_path)['evidence_and_adaptation']['Miura2019']
miura_receipt_path = E2 + 'completion_20261106/inherited_E1_body_reverification.json'
miura_receipt = next(x for x in read(miura_receipt_path) if x['study_id'] == 'S8')
pso_path = E1 + 'open_access_replacement_20261107/part_upload_psogru/review.json'
pso = read(pso_path)
crypto_path = E1 + 'internet_closure_20261106/part_a/reviews.json'
crypto = next(x for x in read(crypto_path)
              if (x.get('doi') or '').lower() == '10.1109/ojcs.2026.3651226')

# Membership is fixed by the union of three public lists, before any decisions.
targets = {}


def add(doi, title, membership):
    key = doi.lower()
    n = native_doi.get(key, {})
    t = targets.setdefault(key, {
        'study_id': n.get('study_id', 'doi:' + key),
        'doi': key,
        'title': n.get('title') or title,
        'cohort_memberships': [],
        'unified_group_ids': sorted(groups.get(key, set())),
    })
    t['cohort_memberships'].append(membership)


for x in completion:
    add(x['DOI'], x['title'], 'completion18:' + x['set'])
for sid in second_summary['new_examined_ids']:
    n = native_id[sid]
    add(n['doi'], n['title'], 'recovery_second10')
for x in follow:
    add(x['doi'], x['title'], 'followup5_complete_source_reviews')
assert len(targets) == 30
assert sum(len(x['cohort_memberships']) for x in targets.values()) == 33

# Individual decisions. A target mismatch with E2 is never by itself exclusion.
# Values: disposition, narrative stratum, substantive eligibility rationale.
DECISIONS = {
 '10.3390/app10144768': ('excluido', 'comparador_neuronal_no_profundo',
     'El original describe ANN/HONN de una capa oculta (Figura 2; tablas 4–10). Se excluye del núcleo DL definido por arquitectura profunda o recurrente/convolucional/atención; se conserva como comparador neuronal de volatilidad BTC. La exclusión no obedece al objetivo de cinco días ni a la calidad de sus resultados.'),
 '10.30798/makuiibf.1530528': ('incluido', 'proxy_retorno_diario_cuadrado_hibrido_GARCH',
     'BTC/ETH, proxy de retorno diario al cuadrado y BPNN con dos capas ocultas/DBN-DNN documentadas. Las redes emplean pronósticos GARCH como entradas: inclusión aplicada híbrida; no voto de red autónoma ni comparación de sigma7.'),
 '10.3390/risks10120237': ('contexto_elegibilidad_pendiente', 'volatilidad_declarada_construccion_inconsistente',
     'El estudio evalúa híbridos GARCH-LSTM/GRU/BiLSTM sobre BTC. Se admite para síntesis narrativa de métodos y problemas de reporte. La fórmula intradía no conciliada con cierres diarios impide una conclusión cuantitativa sobre el constructo y cualquier pooling; no se repara por inferencia.'),
 '10.1186/s40854-025-00881-x': ('incluido', 'varianza_realizada_intradia',
     'Estudio empírico de varianza realizada cripto a partir de retornos de cinco minutos con comparación predictiva HAR/ARMA/LSTM. La parte de conectividad no sustituye el experimento de forecast. Admisible narrativamente; no convertir el promedio de pérdidas heterogéneas en RMSE.'),
 '10.1145/3539597.3570387': ('incluido', 'raiz_varianza_realizada_intradia_preprint',
     'La versión arXiv v2 leída predice volatilidad BTC obtenida de retornos de 15 minutos y compara LSTM/TCN con AR-RV. Se admite esa versión identificada; no se atribuyen sus cifras a una versión editorial no cotejada.'),
 '10.1080/1351847x.2025.2553053': ('incluido', 'indice_volatilidad_implicita_opciones',
     'LSTM y modelos econométricos pronostican BitVol BTC derivado de opciones. Se conserva como estrato contextual de volatilidad implícita, separado de volatilidad realizada/histórica. El tenor de 30 días del índice no es el horizonte de forecast.'),
 '10.1109/access.2025.3584243': ('incluido', 'desviacion_historica_transformada_version_editorial',
     'La ficha primaria E1 de la versión final IEEE verifica objetivo de dispersión anualizada transformada, comparadores LSTM/Transformer y split cronológico. La receta numérica final no se transcribió íntegramente: limita reproducibilidad, no admisibilidad narrativa. El SSRN 2022 es una versión previa del mismo estudio, no otra inclusión.'),
 '10.1007/s42521-026-00229-9': ('incluido', 'volatilidad_condicional_modelo_GARCH',
     'La ficha primaria más específica verifica en §3.2 el proxy de desviación condicional GARCH y en §3.5 una LSTM de dos capas 64/32. Se incluye en un estrato condicional, con cifras de tablas pendientes; no se presenta como volatilidad realizada.'),
 '10.1007/978-3-030-22796-8_18': ('incluido', 'varianza_realizada_tres_horas',
     'El DOI exacto enlaza S8/E2-0671 al original recibido y verificado: MLP de cuatro capas, CNN y LSTM/GRU independientes para RV de tres horas de BTC. Falta receta CNN completa; no se inventa Conv1D. El acceso parcial de una tanda E2 no anula el recibo previo del cuerpo original.'),
 '10.1016/j.mlwa.2023.100465': ('incluido', 'desviacion_historica_30_dias_multiactivo',
     'Las secciones primarias definen desviación poblacional de logretornos de 30 días, incluyen BTC entre 27 criptomonedas y comparan DFFNN/LSTM autónomas e híbridas. Seis lags, rejilla de capas/anchos y resultados están localizados. Se admite como evidencia aplicada de otra definición de volatilidad.'),
 '10.1016/j.irfa.2023.102914': ('incluido', 'desviacion_intradia_ventanas_1_7_30_dias',
     'Las ecuaciones 1–3 y tabla 7 describen BTC/ETH/LTC/XRP con respuestas de desviación de retornos de 30 minutos y una LSTM autónoma. La coincidencia de una ventana de siete días no implica igualdad con sigma7 diaria ni de información disponible.'),
 '10.1007/s40822-023-00232-0': ('excluido', 'logretorno_diario_no_volatilidad',
     'La ecuación 14 y §3.4 definen BTC-USD-RET como logretorno diario; §4/tabla 2 pronostican esa serie. El término volatilidad en el título no establece un objetivo de dispersión/varianza. Exclusión por variable dependiente, no por ausencia de semillas.'),
 '10.1016/j.intfin.2024.102064': ('incluido', 'varianza_realizada_10_min_preaveraging',
     'Métodos y tabla 6 verifican varianza realizada BTC con retornos de diez minutos y preaveraging, LSTM autónoma y CNN-LSTM híbrida a horizontes 1/7/14/30/60 días. La receta y la comparación son suficientes para narrativa; faltan semillas y validación independiente inequívoca.'),
 '10.1186/s43093-023-00200-9': ('excluido', 'constructo_y_rol_DL_no_establecidos',
     'Las secciones primarias alternan retorno y volatilidad sin fórmula de etiqueta de dispersión verificada; describen NNETAR/GMDH sin establecer una arquitectura DL del predictor de volatilidad. Se excluye del núcleo con constructo y DL verificables y se conserva contexto metodológico. No se afirma que esos elementos estén ausentes de todo el artículo.'),
 '10.1016/j.asoc.2023.111132': ('incluido', 'varianza_realizada_5_min_diaria_semanal',
     'El original recuperado E1OA-R02 define RV diaria/semanal BTC/ETH/LTC/XMR, ventana móvil de 23 meses y comparación autónoma LSTM/MLP/HAR. La falta de anchos finales no excluye narrativa; las cifras de BTC muestran que la LSTM no domina estos comparadores.'),
 '10.1007/s10614-023-10373-8': ('incluido', 'varianza_intrahoraria_cartera_cripto',
     'La revisión primaria E1F-M01 verifica varianza empírica de retornos cinco minutos, nueve criptoactivos, ventanas cronológicas y LSTM/MLP autónomas junto al híbrido. Se incluye como evidencia cripto/cartera; no se convierte en resultado de BTC individual si la celda no lo identifica.'),
 '10.1007/s10614-023-10392-5': ('incluido', 'varianza_realizada_retornos_porcentuales_5_min',
     'El original de 23 páginas recibido define RV BTC como suma de retornos porcentuales cinco minutos al cuadrado y compara PSO-GRU, GRU, LSTM y HAR-RV. Se incluye como aplicación; sus recetas incompletas, posibles fugas y discordancias MSE/RMSE impiden avalar magnitudes o superioridad general.'),
 '10.1109/ojcs.2026.3651226': ('contexto_elegibilidad_pendiente', 'volatilidad_intradia_multiples_proxies_sin_mapa',
     'El manuscrito aceptado de 18 páginas y apéndices fue leído en E1N075 y contiene experimentos de volatilidad BTC/cripto con SSM, GRU/LSTM y atención. Se incluye sólo como contexto descriptivo con constructo/escala por tabla no conciliados. No se considera la versión editorial leída ni evidencia cuantitativa de superioridad.'),
 '10.3390/en18102434': ('excluido', 'otra_poblacion_fotovoltaica',
     'El registro identifica generación fotovoltaica como población/resultado; está fuera de BTC/cripto-volatilidad. Exclusión por título/registro, sin afirmar lectura del original. Puede ser evidencia de transferencia arquitectónica si se recupera en otro estrato.'),
 '10.1016/j.epsr.2025.112196': ('excluido', 'otra_poblacion_eolica',
     'El registro identifica potencia eólica como población/resultado; está fuera de BTC/cripto-volatilidad. Exclusión por título/registro; no es exclusión por calidad ni lectura de fulltext.'),
 '10.1016/j.physa.2024.130185': ('excluido', 'otra_poblacion_trafico',
     'El registro identifica velocidad de tráfico en autopista como resultado; está fuera de BTC/cripto-volatilidad. Puede orientar transferencia iTransformer, pero no entra en evidencia aplicada cripto.'),
 '10.1016/j.energy.2025.138706': ('excluido', 'otra_poblacion_eolica',
     'El registro identifica potencia eólica como resultado; está fuera de BTC/cripto-volatilidad. Se excluye por registro, sin convertir su falta de acceso al cuerpo en juicio de calidad.'),
 '10.1016/j.jfds.2024.100143': ('no_recuperado', 'volatilidad_financiera_poblacion_por_confirmar',
     'El recibo completion documenta identidad y README de autor, no el cuerpo. El título no establece BTC/cripto ni una comparación elegible. No se adjudican métodos/métricas desde el repositorio y no se excluye por calidad.'),
 '10.1016/j.eswa.2024.124851': ('no_recuperado', 'volatilidad_financiera_poblacion_por_confirmar',
     'El recibo documenta DOI y README de código TimesNet para volatilidad de acciones; no recuperó el manuscrito. No basta para adjudicar toda su población ni su diseño. Se mantiene no recuperado en este corte.'),
 '10.1108/cfri-01-2024-0032': ('no_recuperado', 'forecast_financiero_constructo_por_confirmar',
     'El título financiero y la identidad bibliográfica no suministran experimento primario verificable. No hay cuerpo recuperado en los recibos examinados para decidir población, target y rol neuronal.'),
 '10.1016/j.physa.2022.127158': ('no_recuperado', 'candidato_cripto_DL_volatilidad',
     'El resumen/editor y depósitos restringidos permiten identificar candidato de volatilidad cripto, pero no las secciones primarias del target y experimento. Las fichas E1 y E2 coinciden en falta de recuperación completa.'),
 '10.1016/j.physa.2021.126613': ('no_recuperado', 'candidato_residuos_LSTM_intradia',
     'El abstract propone LSTM para errores multiplicativos de volatilidad BTC; no suministra el experimento primario completo. Se conserva como no recuperado, con elegibilidad no resuelta.'),
 '10.1016/j.future.2024.04.043': ('no_recuperado', 'candidato_CEEMDAN_RF_LSTM',
     'La tanda2 identifica el artículo CEEMDAN-RF-LSTM y lo deja en recuperación parcial. No se publicó ficha primaria de método/target/resultados entre las fuentes examinadas; no se asigna resultado ni inclusión por título.'),
 '10.1007/s10614-024-10713-2': ('no_recuperado', 'candidato_stacking_volatilidad_VaR',
     'El abstract confirma volatilidad/VaR BTC con datos intradía, pero la ficha primaria documenta preview de suscripción. Se necesita el experimento de volatilidad para separar VaR y roles de stacking; no se excluye por mencionar VaR.'),
 '10.3390/jrfm19090688': ('no_recuperado', 'secciones_indexadas_y_codigo_parcial',
     'Existen secciones primarias indexadas y código autor fijado: apoyan candidato ARCH-LSTM y TCN. No hay lectura continua ni tablas completas del original. Se mantiene no recuperado completo; no se rebaja el código parcial a cero evidencia ni se lo eleva a fulltext.'),
}
assert set(DECISIONS) == set(targets)

follow_doi = {x['doi'].lower(): x for x in follow}
second_doi = {native_id[x['study_id']]['doi'].lower(): x for x in second}
metadata_excluded = {'10.3390/en18102434', '10.1016/j.epsr.2025.112196',
                     '10.1016/j.physa.2024.130185', '10.1016/j.energy.2025.138706'}
complete_expected = set(DECISIONS) - metadata_excluded - {
    k for k, v in DECISIONS.items() if v[0] == 'no_recuperado'}


def evidence(t):
    d = t['doi']
    ptr = []
    loc = []
    target = ''
    architecture = ''
    report = ''
    quality = []
    source_url = 'https://doi.org/' + d
    source_version = 'no verificada a partir del cuerpo'
    if d in follow_doi:
        x = follow_doi[d]
        ptr.append(follow_path + '#' + x['study_id'])
        loc.append(x['locator'])
        target = x['target_and_data']
        report = x['reported_result']
        quality.extend([x['limits']] + x['unverified_fields'])
        source_url = x['source_url']
        source_version = 'editorial, versión según recibo fulltext previo'
        if x['study_id'] == 'E2-0867':
            source_version = 'arXiv 2110.14317v2, 2022-12-06; equivalencia editorial no establecida aquí'
    if d in second_doi:
        x = second_doi[d]
        ptr.append(second_path + '#' + x['study_id'])
        loc.append(x['locator'])
        target = x['target']
        report = x['reported_result']
        quality.append(x['limits'])
        source_url = x['source_url']
        source_version = x['version']
    x = primary_doi.get(d)
    if x:
        ptr.append(primary_path + '#doi=' + d)
        loc.extend(x['source_locators'])
        if x['primary_sections_assessed']:
            target = x['target_quantity']
            architecture = x['architecture_details']
            if x['metrics_reported']:
                report = x['metrics_reported']
            quality.extend([x['limitations'], x['leakage_risk'],
                            'Protocolo: ' + str(x['protocol_clarity']),
                            'Repeticiones: ' + str(x['repetitions'])])
            if d == '10.1109/access.2025.3584243':
                source_version = 'IEEE Access 2025 final; secciones de PDF 20 páginas según ficha primaria E1'
                source_url = x['retrieval_attempts'][1]['url']
                quality.append('No usar cifras SSRN 2022 .101/.463 como si fueran IEEE final .103/.451; tablas finales de arquitectura no transcritas íntegramente.')
            elif d not in follow_doi and d not in second_doi:
                source_version = 'Versión primaria identificada en ficha E1; no nueva recuperación ni lectura hoy'
                source_url = x['source_urls'][0]
        else:
            quality.append(x['reason'])
    if d == '10.1007/978-3-030-22796-8_18':
        ptr.extend([miura_path + '#evidence_and_adaptation/Miura2019', miura_receipt_path + '#S8'])
        loc.append(miura['evidence_pages'])
        target = 'RV de tres horas de BTC, formada con retornos de precios minuto; varianza realizada'
        architecture = miura['verified']
        quality.append(miura['not_verified'])
        source_version = 'Capítulo editorial ISNN 2019, pp165–172; cuerpo SHA256 ' + miura_receipt['body_sha256']
        source_url = miura['consulted_source']
    if d == '10.1007/s10614-023-10392-5':
        ptr.append(pso_path)
        loc.extend(pso['evidence_locations'])
        source_version = pso['metadata_verification']['version']
        source_url = pso['source_urls'][0]
        report = [x for x in pso['metrics']['mean_rows'] if x['horizon'] == 1 and x['model'] in ['PSO–GRU', 'GRU', 'LSTM', 'HAR-RV']]
        quality.append(pso['metrics']['status'])
        quality.append('No combinar medias de tablas inconsistentes como estimación de efecto; split/scaler/HPO y presupuesto de baselines requieren aclaración.')
    if d == '10.1109/ojcs.2026.3651226':
        ptr.append(crypto_path + '#E1N075')
        loc.extend(crypto['evidence_locations'])
        target = crypto['target']
        architecture = crypto['architecture_details']
        source_url = crypto['source_urls'][-1]
        source_version = crypto['version_read']
        report = 'Tablas1–2 y5 verificadas en manuscrito aceptado; sin importar métricas a eficacia E2 ni agrupar proxies no conciliadas.'
        quality.extend(crypto['quality_issues'])
    if not ptr:
        ptr.append(completion_path + '#DOI=' + d)
        loc.append('Título/registro individual del objetivo de recuperación; no lectura de resumen/editor ni fulltext afirmada hoy.')
    if d in native_doi:
        ptr.append(native_path + '#' + t['study_id'])
    if d == '10.3390/jrfm19090688':
        ptr.append(E1 + 'internet_closure_20261106/part_d/reviews.json#E1X002')
        INPUTS.add(E1 + 'internet_closure_20261106/part_d/reviews.json')
        quality.append('Código estático/secciones indexadas no equivalen a artículo completo; no recuperación total acreditada.')
    return dict(evidence_pointers=ptr, source_locators=loc, source_url=source_url,
                source_version=source_version, target_extracted=target,
                neural_architecture_extracted=architecture, reported_result=report,
                reporting_and_bias_limits=quality)


applied = []
for d, t in sorted(targets.items(), key=lambda kv: (kv[1]['study_id'], kv[0])):
    disposition, stratum, reason = DECISIONS[d]
    obj = dict(t)
    obj.update(decision=disposition, narrative_stratum=stratum,
               decision_reason=reason,
               adjudicated_source_role='mecanismo_contextual_constructo_no_resuelto'
                   if disposition == 'contexto_elegibilidad_pendiente' else
                   'otra_poblacion_sin_lectura_primaria' if d in metadata_excluded else
                   'aplicada_volatilidad_implicita_condicional' if d in {
                       '10.1080/1351847x.2025.2553053', '10.1007/s42521-026-00229-9'} else
                   'aplicada_volatilidad_retornos' if disposition == 'incluido' else
                   'excluido_nucleo_conservado_comparador' if disposition == 'excluido' else
                   'acceso_incompleto_elegibilidad_no_resuelta',
               applied_eligibility='incluido' if disposition == 'incluido' else
                   'excluido' if disposition == 'excluido' else 'pendiente',
               assessed_primary_sections_available=d in complete_expected,
               decision_stage='registro_titulo' if d in metadata_excluded else
                   'no_recuperado_completo' if disposition == 'no_recuperado' else 'ficha_secciones_primarias',
               direct_E2_effectiveness_established=False,
               numerical_pooling_permitted=False,
               new_manuscript_reading_claimed=False,
               eligibility_adjudication_date=DATE,
               independent_human_double_review=False)
    obj.update(evidence(obj))
    applied.append(obj)
dump('applied_target_decisions.json', applied)
write_csv('applied_target_decisions.csv', applied)
focused_ids = set(second_summary['editorial_fulltext_ids'] + second_summary['preprint_fulltext_ids'])
focused = [x for x in applied if x['study_id'] in focused_ids]
assert len(focused) == 8
write_csv('focused8_decisions.csv', focused)

# Keep architectural source inclusion entirely separate from applied eligibility.
current_path = 'book_e2/_static/e2_literature_closure_20261113/current10_extraction.json'
current = read(current_path)
mechanism = {}
for x in current:
    d = x['study_id'].lower()
    m = mechanism.setdefault(d, dict(source_id=x['study_id'], title=x['title'],
        models=[], current_Top10_system_source=True, decision='incluido',
        role='mecanismo_arquitectonico',
        rationale='Fuente primaria del mecanismo, evidencia de arquitectura y resultado localizada y código de autor identificado. Se admite para transferencia explícita, no para eficacia BTC.',
        source_url=x['source_url'], source_version=x['source_version'],
        evidence_pointers=[current_path + '#study_id=' + x['study_id']],
        evidence_locators=x['metric_locator'],
        source_read_level=x['read_level'], quality=x['quality'],
        limitations=x['critical_limit'], foundational_2016_exception=x['original_year'] == 2016,
        direct_E2_effectiveness_established=False, new_manuscript_reading_claimed=False,
        adjudication_date=DATE, independent_human_double_review=False))
    m['models'].append(x['model'])
old_elig_path = E2 + 'completion_20261106/architectural_originals_eligibility.csv'
old_elig = rows(old_elig_path)
old_manifest_path = E2 + 'fulltext_20261105/fulltext_manifest.json'
old_manifest = read(old_manifest_path)
old_metrics = rows(E2 + 'fulltext_20261105/metrics.csv')
for x in old_elig:
    if x['model'] in {v['model'] for v in current}:
        continue
    r = next(v for v in old_manifest['papers'] if v['model'] == x['model'])
    d = x['study_identity']
    mechanism[d.lower()] = dict(source_id=d, title=d if not d.startswith('10.') else x['model'] + ' original',
        models=[x['model']], current_Top10_system_source=False, decision='incluido',
        role='comparador_mecanismo_historico',
        rationale='Original completo y resultado documentados; se conserva como comparador arquitectónico disponible, sin incorporarlo al Top10 vigente ni excluirlo por no ocupar un puesto.',
        source_url=r['source_url'], source_version=r['source_version'],
        evidence_pointers=[old_elig_path + '#model=' + x['model'], old_manifest_path + '#model=' + x['model']],
        evidence_locators='Fila modelo en metrics.csv y quality_matrix.csv del paquete fulltext_20261105',
        source_read_level='Recibo previo de PDF completo y apéndices; no lectura nueva hoy',
        quality=json.loads(x['methodological_quality_statuses']),
        limitations='Dominio/dataset/horizonte/escala distintos de BTC sigma7; límites de reporte preservados en quality_matrix.csv. No prueba inferioridad por haber sido desplazado del catálogo.',
        foundational_2016_exception=False, direct_E2_effectiveness_established=False,
        new_manuscript_reading_claimed=False, adjudication_date=DATE,
        independent_human_double_review=False)
mechanism_rows = sorted(mechanism.values(), key=lambda x: x['source_id'])
assert len(mechanism_rows) == 13
assert sum(x['current_Top10_system_source'] for x in mechanism_rows) == 9
assert sum(len(x['models']) for x in mechanism_rows if x['current_Top10_system_source']) == 10
dump('mechanism_source_decisions.json', mechanism_rows)
write_csv('mechanism_source_decisions.csv', mechanism_rows)

# Explicit individual source-role adjudications, after reading all43 linked fichas.
# This does not import E1's strict eligibility or screen9559 rows by keywords.
ROLE43 = {
    'aplicada_cripto_volatilidad_de_retornos': [
        '10.1109/access.2026.3709184', '10.3390/jrfm14070337',
        '10.1134/s1995080224600705', '10.1007/s10614-024-10694-2',
        '10.1109/access.2025.3584243', '10.1016/j.mlwa.2023.100465',
        '10.1016/j.irfa.2023.102914', '10.1016/j.intfin.2024.102064',
        '10.30798/makuiibf.1530528', '10.1007/978-3-030-22796-8_18',
        '10.1145/3539597.3570387', '10.1007/978-3-031-38074-7_41',
        '10.1016/j.asoc.2023.111132', '10.1186/s40854-025-00768-x',
        '10.1007/s10614-023-10373-8', '10.1007/s10614-023-10392-5'],
    'aplicada_constructo_implicito_condicional_o_parametro': [
        '10.1080/1351847x.2025.2553053', '10.1007/s42521-026-00229-9',
        '10.3934/math.2023712'],
    'comparador_neuronal_profundidad_no_establecida_para_DL': [
        '10.3390/app10144768', '10.15240/tul/001/2022-2-007',
        '10.1186/s40854-023-00520-3'],
    'mecanismo_neuronal_covariable_no_predictor_volatilidad': [
        '10.1007/s42521-020-00018-y', '10.1007/978-3-030-66891-4_13'],
    'contexto_primario_constructo_no_resuelto': [
        '10.3390/jrfm14070308', '10.1186/s43093-023-00200-9',
        '10.3233/faia250121', '10.1057/s41270-023-00239-1',
        '10.3390/risks10120237'],
    'otro_objetivo_precio_retorno_clasificacion_o_riesgo': [
        '10.1109/access.2021.3063243', '10.1016/j.physa.2024.130140',
        '10.46354/i3m.2021.mas.010', '10.1016/j.neucom.2024.128816',
        '10.34021/ve.2026.09.01(2)', '10.1155/jama/9089827',
        '10.11591/ijece.v15i1.pp614-623', '10.1109/access.2023.3317899',
        '10.1049/joe.2019.1236', '10.1109/access.2026.3702666',
        '10.1145/3583780.3614790', '10.1007/s40822-023-00232-0',
        '10.7717/peerj-cs.1750', '10.1080/08839514.2026.2684165'],
}
role_by_doi = {doi: role for role, dois in ROLE43.items() for doi in dois}
assert len(role_by_doi) == sum(map(len, ROLE43.values())) == 43
primary_linked = [x for x in primary if x['primary_sections_assessed']
                  and (x.get('doi') or '').lower() in groups]
assert len(primary_linked) == 43
assert {(x.get('doi') or '').lower() for x in primary_linked} == set(role_by_doi)
role_rows = []
for x in sorted(primary_linked, key=lambda x: x['doi'].lower()):
    d = x['doi'].lower()
    role_rows.append(dict(doi=d, E2_group_ids=sorted(groups[d]), title=x['title'],
        assigned_source_role=role_by_doi[d],
        target_primary_ficha=x['target_quantity'],
        neural_role_and_architecture_primary_ficha=x['architecture_details'],
        role_rationale_primary_ficha=x['reason'],
        prior_E1_strict_disposition_preserved=x['disposition'],
        limitations_primary_ficha=x['limitations'],
        protocol_primary_ficha=x['protocol_clarity'],
        metrics_primary_ficha=x['metrics_reported'],
        source_locators=x['source_locators'], source_urls=x['source_urls'],
        evidence_pointer=primary_path + '#doi=' + d,
        adjudication_date=DATE, assignment_kind='source_role_only_not_global_eligibility',
        new_original_manuscript_reading=False,
        global_E2_final_eligibility_assigned=False,
        overlaps_focal30=d in targets,
        direct_E2_effectiveness_established=False))
dump('historical43_source_roles.json', role_rows)
write_csv('historical43_source_roles.csv', role_rows)
dump('historical43_summary.json', dict(
    scope='Roles de43 grupos E2 con fichas primarias E1 enlazadas por DOI exacto; no nueva búsqueda o lectura del original ni PRISMA43.',
    source_records=43, source_role_counts={k: len(v) for k, v in ROLE43.items()},
    overlaps_focal30=sum(x['overlaps_focal30'] for x in role_rows),
    not_additive_to_focal30=True, historical_strict_E1_eligibility_not_imported=True,
    global_E2_review_completed=False, independent_human_reviewers=0))

protocol = dict(
    version='e2-focused-eligibility-v2', adjudication_date=DATE,
    temporal_status='Retrospectivo; posterior al catálogo y resultados E2; no preregistro ni selección ciega.',
    research_question='Qué muestran los informes ya identificados y documentados sobre métodos DL/autónomos o híbridos para pronosticar volatilidad de BTC/cripto, separando constructos y sin trasladar su eficacia a sigma7 E2.',
    target_cohort='Unión exacta de completion18, recovery_second10 y followup5, deduplicada por DOI normalizado; 33 apariciones,30 estudios/objetivos bibliográficos.',
    rationale_for_scope='Auditoría de la recuperación dirigida realmente documentada, con límites de acceso y disponibilidad. Incluye todos sus objetivos identificables, no sólo los favorables o los modelos implementados. No representa una búsqueda sistemática ni estima prevalencia o exhaustividad.',
    inclusion=dict(population='BTC o criptomonedas; cartera/cripto separada de BTC individual.',
        prediction='Pronóstico empírico de varianza/desviación/proxy de dispersión. Volatilidad implícita, condicional, realizada e histórica se retienen en estratos distintos.',
        neural_role='Red profunda (dos o más capas ocultas de transformación), red recurrente LSTM/GRU, CNN/atención/SSM, o híbrido con ese predictor verificado. Una ANN/HONN de una capa no acredita el núcleo DL por sí sola.',
        dates_and_versions='2019–2026; preprint o manuscrito aceptado admisibles con versión explícita. Un DOI/estudio una sola vez aunque se lean varias versiones.',
        evidence='Informe primario/ficha de secciones primarias documentada suficiente para verificar población, rol neuronal y estudio de volatilidad. Construcción inconsistente sólo permite contexto de método/riesgo; elegibilidad aplicada queda pendiente y no se cuenta como incluido aplicado.',
        quantitative_effect='Eficacia comparable requiere adicionalmente mismo constructo, unidades, horizonte, información/split y protocolo. Ningún estudio se incorpora a pooling E2.'),
    exclusions=dict(metadata='Cuatro objetivos explícitos fotovoltaica/viento/tráfico fuera de población.',
        fulltext='Un estudio predice logretorno, uno ANN/HONN sólo una capa, uno no establece constructo de dispersión y predictor DL en secciones evaluadas.',
        missing_report='no_recuperado es una disposición de acceso; elegibilidad científica pendiente, nunca exclusión por calidad.'),
    reporting_quality='Semillas, R5, incertidumbre, receta exacta, código, validación temporal y presupuesto HPO se evalúan como límites; no se usan como exclusiones automáticas de narrativa.',
    review_method='Un adjudicador IA revisa registros y fichas públicas previas por fuente; cero originales nuevos leídos hoy; cero revisores humanos independientes; κ no calculado.',
    preserved_historical_scope='No cambia decisiones estrictas E1, cohortes1078/40/9559, fechas históricas ni selección Top10. Se decide hoy una admisibilidad narrativa diferente y se identifica toda diferencia.',
    prohibition='No etiquetar los9559 registros como lectura individual IA a partir de reglas deterministas.')
dump('protocol.json', protocol)
counts = collections.Counter(x['decision'] for x in applied)
main_counts = collections.Counter(x['decision'] for x in focused)
flow = dict(
    version=protocol['version'], date=DATE,
    global_received_groups=9559, global_eligible=None, global_included=None,
    global_systematic_review_complete=False,
    focused1078_complete=False, original40_complete=False,
    focal_targets=dict(identified_occurrences=33, exact_DOI_duplicate_occurrences_removed=3,
        unique_targets=30, metadata_population_exclusions=4,
        potentially_in_scope_reports=26, reports_not_completely_recovered=counts['no_recuperado'],
        primary_reports_with_assessed_sections=18,
        fulltext_stage_exclusions=3, applied_included=counts['incluido'],
        applied_return_volatility_included=11,
        applied_implied_or_conditional_included=2,
        context_mechanism_only_applied_eligibility_pending=2,
        total_scientific_exclusions=counts['excluido'],
        accounting_identity='30 =4 registro +8 no recuperado completo +18 primarios;18=13 incluidos aplicados+2 contexto de mecanismo/constructo pendiente+3 excluidos',
        historical_retrieval_sequence_not_rewritten=True,
        eligibility_for_unrecovered_reports='pending; not scientific exclusions',
        complete_manuscript_readings_today=0,
        complete_primary_readings_newly_inferred_from_metadata=0),
    original_public8=dict(n=8, applied_included=main_counts['incluido'], excluded=main_counts['excluido'],
        applied_eligibility_pending_context_only=main_counts['contexto_elegibilidad_pendiente'],
        no_new_fulltext_reading=True,
        version_update='E2-0285 final IEEE source sections already documented in E1; SSRN figures not transferred to final.'),
    architectural_sources=dict(current_model_systems=10, current_unique_primary_studies=9,
        current_included_for_mechanism=9, historical_additional_comparator_sources=4,
        total_unique_available_mechanism_sources=13,
        included_for_mechanism=13, included_for_BTC_effectiveness=0,
        model_systems_not_independent_studies=True),
    scope_status='Corte documental focal:16 decisiones de elegibilidad sobre fichas primarias,2 constructos pendientes y8 accesos incompletos; roles documentados. Revisión sistemática global incompleta.',
    independent_human_reviewers=0, kappa=None)
assert counts == {'incluido': 13, 'excluido': 7, 'no_recuperado': 8, 'contexto_elegibilidad_pendiente': 2}, counts
assert len(complete_expected) == 18
assert counts['incluido'] + counts['contexto_elegibilidad_pendiente'] + 3 == 18
dump('flow.json', flow)

# Provenance is for all source evidence used, not just study inclusion counts.
input_manifest = {rel: {'sha256': hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(),
                        'bytes': (ROOT / rel).stat().st_size} for rel in sorted(INPUTS)}
dump('evidence_manifest.json', {'repository': str(ROOT), 'input_files': input_manifest,
    'new_source_bodies_in_output': 0, 'sources_redistributed': False,
    'new_external_access_attempts': 'access_attempts.json; all five requests failed with proxy403',
    'interpretation': 'Hash de ficha pública fija la evidencia consultada; no equivale a hash ni relectura del manuscrito original.'})
dump('validation.json', dict(status='PASS',
    assertions=['33 membership occurrences minus3 exact DOI duplicate occurrences=30 targets',
                '30=4 metadata exclusions+8 incomplete/unrecovered reports+18 primary assessments',
                '18=13 applied inclusions+2 mechanism-only context with unresolved construct+3 fulltext-stage exclusions',
                'all30 targets have individual disposition, rationale, version and source pointer',
                'public focused8=6 applied included+1 mechanism-only context unresolved+1 excluded strictDL',
                'current10 model systems map to9 primary studies; Wang counted once',
                '13 mechanism sources=9 current+4 historical comparators',
                'zero fulltext readings or human independent reviewers manufactured'],
    applied_decision_counts=dict(counts), focused8_counts=dict(main_counts)))
(OUT / 'SHA256SUMS').write_text(''.join(
    hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n'
    for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'))
print(json.dumps({'status': 'PASS', 'applied': dict(counts), 'primary_assessments': 18,
                  'focused8': dict(main_counts), 'mechanism_sources': len(mechanism_rows)}, ensure_ascii=False))
