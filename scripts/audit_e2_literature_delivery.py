"""Verify the focused E2 literature delivery without network or new readings."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'references/literature_e2/delivery_closure_20261114_v2'


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def verify(folder, root=ROOT):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, receipt in manifest['files'].items():
        raw = (folder/name).read_bytes()
        if len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('Changed delivery file: '+name)
    for name, expected in manifest['repository_inputs_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed repository evidence: '+name)
    rows = read_csv(folder/'eligibility/applied_target_decisions.csv')
    counts = Counter(r['decision'] for r in rows)
    if len(rows) != 30 or counts != {'incluido':13,'excluido':7,'no_recuperado':8,'contexto_elegibilidad_pendiente':2}:
        raise ValueError('The documented focal disposition changed')
    if len({r['doi'].casefold() for r in rows}) != 30:
        raise ValueError('A source DOI is counted twice in the focal cohort')
    flow = json.loads((folder/'eligibility/flow.json').read_text())
    if flow['global_systematic_review_complete'] or flow['global_eligible'] is not None or flow['global_included'] is not None:
        raise ValueError('A focused delivery cannot close the unreviewed global corpus')
    if flow['focal_targets']['primary_reports_with_assessed_sections'] != 18:
        raise ValueError('Primary receipt count changed')
    scores = read_csv(folder/'citation_score/conditional_score_by_scenario.csv')
    selected = [r for r in scores if r['scenario']=='posthoc_intrapaper_9_primary']
    if len(selected) != 10 or sum(r['P_scenario']!='' for r in selected)!=9:
        raise ValueError('The documented P9 contextual scenario changed')
    if sum(r['C_preserved']!='' for r in selected)!=6 or sum(int(r['known_component_count']) for r in selected)!=45:
        raise ValueError('A missing bibliometric component was invented')
    if sum(r['complete_under_this_scenario']=='True' for r in selected)!=6:
        raise ValueError('Conditional score completeness changed')
    if any(r['original_P_overwritten']!='False' or r['final_test_A2_used']!='False' or r['global_definitive_score']!='False' for r in selected):
        raise ValueError('Contextual support was misrepresented as original selection')
    if len(read_csv(folder/'eligibility/historical43_source_roles.csv')) != 43:
        raise ValueError('Historical receipt-role coverage changed')
    return {'status':'verified', 'focal_targets':30, 'applied_included':13,
            'excluded':7, 'unrecovered':8, 'context_only_eligibility_pending':2,
            'P_contextual_systems':9, 'C_known_systems':6, 'conditional_components_known':45,
            'conditional_complete_scores':6, 'historical_primary_roles':43,
            'global_PRISMA_complete':False, 'historical_selection_rewritten':False,
            'new_manuscript_readings':0, 'new_model_fits':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=DEFAULT)
    p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    print(json.dumps(verify(a.bundle,a.root),ensure_ascii=False,indent=2))
