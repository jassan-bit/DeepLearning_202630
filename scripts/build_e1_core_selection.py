#!/usr/bin/env python3
"""Compile the current E1 literature-based core selection, without fitting models."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'references/literature_e1/core_selection_20261107'
MIRROR = ROOT / 'book/_static/literature_e1/core_selection_20261107'
SUMMARY_SOURCE = 'references/literature_e1/homogeneous_expansion_20261107/summary.json'
LEDGER_SOURCE = 'references/literature_e1/homogeneous_expansion_20261107/adjudications.json'
PUBLICATION_SCHEMA = 'e1_public_metadata_allowlist_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def csv_encoded(fields, rows):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return text.getvalue().encode()


def read_json(path):
    return json.loads(path.read_text())


def ranked_candidates(stratum, eligible, count):
    """Rank verified DL candidates; a tie at the selection boundary is unresolved."""
    if len(eligible) != len(set(eligible)):
        raise ValueError('Eligible candidate families must be unique')
    denominator = stratum['denominator']
    if type(denominator) is not int or denominator <= 0:
        raise ValueError('Positive report-eligible denominator required')
    records = stratum['records']
    if len(records) != denominator or len(set(records)) != denominator:
        raise ValueError('Report-eligible denominator does not match unique records')
    rows = []
    for family in eligible:
        raw = stratum['family_counts'][family]
        experiments = raw['experiments']
        if (raw['denominator'] != denominator or raw['count'] != len(set(experiments))
                or len(experiments) != len(set(experiments)) or not set(experiments) <= set(records)):
            raise ValueError('Invalid family count or unit: ' + family)
        rows.append({'family': family, 'frequency': raw['count'], 'denominator': denominator,
                     'experiments': sorted(experiments)})
    rows.sort(key=lambda row: (-row['frequency'], row['family']))
    if not 0 < count < len(rows):
        raise ValueError('Selection count requires candidates beyond the chosen set')
    if rows[count - 1]['frequency'] == rows[count]['frequency']:
        raise ValueError('Tie at selected-set boundary needs an explicit additional rule')
    for row in rows:
        row['frequency_rank'] = 1 + sum(other['frequency'] > row['frequency'] for other in rows)
        row['selected_current_core'] = row in rows[:count]
        row['display_order_note'] = 'Equal-frequency rows are displayed alphabetically, not ranked by unknown complexity.'
    return rows


def parameters(family, width, days=7, features=5):
    """Scalar-head recipes with no trainable activation/dropout parameters."""
    if any(type(x) is not int or x <= 0 for x in [width, days, features]):
        raise ValueError('Positive integer dimensions required')
    if family == 'MLP':
        return width * (days * features + 2) + 1
    if family == 'GRU':
        return 3 * width * (features + width + 2) + width + 1
    if family == 'LSTM':
        return 4 * width * (features + width + 2) + width + 1
    raise ValueError('No parameter formula for unselected template ' + family)


def verify_sources(inputs, root=ROOT):
    for relative, expected in inputs['source_sha256'].items():
        actual = digest((root / relative).read_bytes())
        if actual != expected:
            raise ValueError('Frozen documentary source changed: ' + relative)


def build(root=ROOT):
    inputs = read_json(root / 'references/literature_e1/core_selection_20261107/inputs.json')
    verify_sources(inputs, root)
    source = read_json(root / SUMMARY_SOURCE)
    ledger = read_json(root / LEDGER_SOURCE)['adjudications']
    by_id = {row['id']: row for row in ledger}
    if len(by_id) != len(ledger):
        raise ValueError('Repeated source identity')
    ranked = ranked_candidates(source['main_stratum'], inputs['eligible_dl_candidates'], inputs['selected_count'])
    selected = [row['family'] for row in ranked if row['selected_current_core']]
    if set(selected) != set(inputs['planned_templates']):
        raise ValueError('Planned templates do not match literature-derived selected set')
    evidence = []
    for family in selected:
        family_row = next(row for row in ranked if row['family'] == family)
        anchors = inputs['source_anchors'][family]
        if set(anchors['core_frequency_sources']) != set(family_row['experiments']):
            raise ValueError('Frequency anchors differ from verified family occurrences: ' + family)
        for identifier in family_row['experiments']:
            row = by_id[identifier]
            if (family not in row['independentfamilies'] or not row['decision']['primary_document_read']
                    or not row['decision']['guide31_reporting_sensitivity_eligible']):
                raise ValueError('Selected-family occurrence lacks common eligibility: ' + identifier)
        for identifier in anchors['template_sources']:
            row = by_id[identifier]
            evidence.append({'family': family, 'source_id': identifier, 'title': row['title'],
                             'doi': row['doi'], 'source_urls': row['primary_source_urls'],
                             'template_relevance': anchors['extraction'],
                             'scope': row['stratum'],
                             'contributes_to_crypto_frequency': identifier in family_row['experiments']})
    contract = inputs['problem_contract']; parameter_rows = []
    for family in selected:
        template = inputs['planned_templates'][family]
        for width in template['widths']:
            parameter_rows.append({'family': family, 'depth': template['depth'], 'width_candidate': width,
                                   'input_days': contract['history_days'], 'input_features': contract['input_features'],
                                   'output_units': contract['output_units'],
                                   'parameters': parameters(family, width, contract['history_days'], contract['input_features']),
                                   'status': 'prefit_architecture_candidate_not_selected_final_hyperparameters'})
    summary = {'schema_version': 1, 'decision_date_colombia': inputs['decision_date_colombia'],
               'decision_timezone': inputs['decision_timezone'], 'selection_context': inputs['selection_context'],
               'selected_current_core_in_frequency_order': selected,
               'complementary_historical_model': 'TCN',
               'source_report_eligible_crypto_denominator': source['main_stratum']['denominator'],
               'ranking': ranked, 'parameter_candidates': parameter_rows,
               'historical_MLP_593_parameters_reused_as_current_final_recipe': False,
               'same_information_contract': contract, 'planned_common_search': inputs['planned_common_search'],
               'selection_matches_three_highest_frequencies_in_this_corpus': True,
               'selection_boundary_has_no_frequency_tie': True,
               'complexity_assessment': 'One hidden-layer MLP and one recurrent-layer GRU/LSTM; matched-order parameter ranges under the same D7/5-input/scalar-output contract. Published templates anchor the family and depth; own capacity/activation/search adaptations are explicit.',
               'global_minimum_parameter_count_or_global_prevalence_proven': False,
               'inductive_bias_assessment': {'MLP': 'Position-specific nonlinear mixing of the entire flattened D7 window; no recurrent state or temporal weight sharing.',
                                            'GRU': 'Shared recurrent transition with reset and update gates; one evolving hidden vector over the supplied window.',
                                            'LSTM': 'Shared recurrent transition with a separate additive cell state and input/forget/output gating; differs from GRU despite shared recurrent family.'},
               'all_models_observe_same_7_days': True,
               'recurrent_state_carried_before_input_window': False,
               'error_independence_or_predictive_complementarity_proven_by_architecture': False,
               'selected_hyperparameters_and_new_scores': None,
               'new_model_fits_performed_by_this_builder': 0,
               'new_forecasts_performed_by_this_builder': 0,
               'reserve_was_already_observed': True,
               'fresh_unobserved_holdout_created_by_this_selection': False,
               'historical_selection_and_metrics_rewritten': False,
               'experimental_execution_report': 'book/sections/10_nucleo_mlp.md',
               'source_sha256': inputs['source_sha256']}
    readme = '''# Selección vigente de los tres modelos base de E1

La selección vigente es **LSTM–MLP–GRU**, derivada del estrato cripto con reporte
suficiente: **10/10, 4/10 y 2/10**, respectivamente. TCN registra 1/10 y permanece
como comparación complementaria histórica. La actualización del 7 de noviembre de 2026
se realiza después de conocer los resultados anteriores y por solicitud del autor:
no convierte la selección previa ni la reserva conocida en un estudio prospectivo.

## Criterio conjunto de la guía §3.2

La frecuencia se extrae del corpus congelado `homogeneous_expansion_20261107`.
Se consideran candidatos DL; RBFN/GFN/MFN y otras ANN contextuales no se convierten
automáticamente en modelos Deep Learning elegibles. Los tres puestos tienen frecuencias
10, 4 y 2; el siguiente grupo tiene 1, por lo que no hace falta resolver un empate
en el límite para favorecer una terna. Las filas empatadas posteriores se muestran
alfabéticamente y no como un ranking de complejidad desconocida.

La simplicidad se concreta en **una capa oculta MLP** y **una capa recurrente
unidireccional GRU/LSTM**, con cabezas escalares. Dudek, Brauneis y García-Medina
aportan MLP autónomas de una capa oculta; Korkusuz publica una búsqueda con una/dos
capas y anchos 32/64/128. Zhang añade una variante MLP ReLU de una capa en su
Apéndice B, como transferencia financiera y sin aumentar la frecuencia cripto.
GRU/LSTM se apoyan en comparadores publicados y se adaptan al mismo contrato.
No se afirma réplica exacta de las configuraciones ganadoras de los autores.

Las anchuras MLP 32/128 se prefijan como candidatos por su orden de capacidad bajo
D7 y cinco entradas: **1.185/4.737 parámetros**, frente a GRU 16/32 **1.121/3.777**
y LSTM 16/32 **1.489/5.025**. La antigua MLP 16 de 593 parámetros conserva su propia
historia; no es la receta final de esta actualización. Los candidatos no son una
configuración seleccionada ni una puntuación observada. La búsqueda experimental
separada debe seleccionar por desarrollo y publicar su estado y resultados.

MLP mezcla rezagos con pesos específicos por posición, GRU comparte una transición
recurrente con reset/actualización y LSTM añade una celda separada y compuerta de
salida. Son diferencias de sesgo funcional, sin exigir tres macrofamilias ni
inferir errores independientes. Todas reciben los mismos siete días; GRU/LSTM
reinician estado en cada ventana y no acceden a memoria anterior.

## Alcance y cronología

Este cambio resuelve la discrepancia documental de conservar TCN mientras la regla
de recurrencia favorecía MLP. No demuestra mínimos universales, prevalencia global
o superioridad predictiva. La búsqueda previa era dirigida por arquitectura; las
frecuencias son las del corpus evaluado, con sus límites de cobertura y calidad.
La versión anterior GRU–LSTM–TCN, sus recetas, curvas y métricas permanecen intactas.

La ejecución nueva tiene protocolo y planificación propios en el capítulo 10.
La reserva de septiembre–octubre de 2026 ya fue observada: una extensión sobre ella
es exploratoria respecto de selección/modelado, aunque las nuevas reglas de ajuste
se publiquen antes de sus ajustes. Este compilador no entrena ni abre datos de
mercado, no publica pesos/predicciones y no inventa resultados MLP.

`python scripts/build_e1_core_selection.py --check` verifica hashes de cinco
insumos documentales, ranking, procedencia de cada ocurrencia, parámetros y
salidas públicas. El manifiesto aprueba exclusivamente los metadatos de la selección.
'''
    frequency_rows = [{k: (';'.join(row[k]) if k == 'experiments' else row[k]) for k in
                       ['family', 'frequency', 'denominator', 'frequency_rank', 'selected_current_core', 'experiments']} for row in ranked]
    evidence_rows = [{**row, 'source_urls': ';'.join(row['source_urls'])} for row in evidence]
    outputs = {'inputs.json': (root / 'references/literature_e1/core_selection_20261107/inputs.json').read_bytes(),
               'README.md': readme.encode(), 'summary.json': encoded(summary),
               'ranking.csv': csv_encoded(['family','frequency','denominator','frequency_rank','selected_current_core','experiments'],frequency_rows),
               'parameter_candidates.csv': csv_encoded(list(parameter_rows[0]),parameter_rows),
               'source_anchors.csv': csv_encoded(list(evidence_rows[0]),evidence_rows),
               'validation.json': encoded({'frozen_sources_verified':len(inputs['source_sha256']),
                                           'selection_derived_from_report_eligible_occurrences':True,
                                           'templates_match_derived_selected_set':True,
                                           'source_occurrences_eligible_and_autonomous':True,
                                           'historical_source_payloads_modified':0,
                                           'new_model_fits':0,'new_forecasts':0})}
    manifest = {'publication_schema':PUBLICATION_SCHEMA,'approved_public_files':sorted(outputs),
                'files':{name:{'sha256':digest(payload),'bytes':len(payload)} for name,payload in sorted(outputs.items())}}
    outputs['manifest.json'] = encoded(manifest)
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    outputs=build()
    for directory in [STAGE,MIRROR]:
        if args.check:
            actual={p.name for p in directory.iterdir() if p.is_file()}
            if actual != set(outputs): raise ValueError('Public file set differs: '+str(directory))
            for name,payload in outputs.items():
                if (directory/name).read_bytes()!=payload: raise ValueError('Stale or changed selection output: '+name)
        else:
            directory.mkdir(parents=True,exist_ok=True)
            for name,payload in outputs.items():(directory/name).write_bytes(payload)
    print(json.dumps({'status':'checked' if args.check else 'built','public_files':len(outputs),
                      'selected_core':['LSTM','MLP','GRU'],'new_fits':0}))

if __name__=='__main__':main()
