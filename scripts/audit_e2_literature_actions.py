"""Audit the dated E2 literature amendment; no network or model execution."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'references/literature_e2/delivery_closure_20261114_v3'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def check(folder=DEFAULT, root=ROOT):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = folder/name
        assert path.stat().st_size == expected['bytes'], name
        assert digest(path) == expected['sha256'], name
    for name, expected in manifest['repository_inputs_sha256'].items():
        assert digest(root/name) == expected, name
    receipt = json.loads((folder/'identification/receipt.json').read_text())
    received = rows(folder/'identification/records.csv')
    assert len(received) == len({r['EID'] for r in received}) == receipt['received_records'] == 1104
    assert receipt['native_total_shown'] == 1104
    assert receipt['unreceived_from_this_search'] == 0
    assert receipt['global_PRISMA_closed'] is False
    assert receipt['historical_374_missing_records_recovered_by_this_new_search'] is False
    assert 'Abstract' not in received[0]
    counts = Counter(r['decision'] for r in rows(folder/'eligibility.csv'))
    assert counts == {'incluido':14, 'excluido':10, 'no_recuperado':4, 'contexto_elegibilidad_pendiente':2}, counts
    decisions = json.loads((folder/'eligibility.json').read_text())
    assert len(decisions) == len({r['doi'].casefold() for r in decisions}) == 30
    assert all(r['numerical_pooling_permitted'] is False for r in decisions)
    assert sum(r['new_manuscript_reading_claimed'] for r in decisions) == 3
    citation = rows(folder/'citation_snapshot/records.csv')
    assert len(citation) == len({r['EID'] for r in citation}) == 9
    assert all(r['Source'] == 'Scopus' and r['Cited by'].isdigit() for r in citation)
    components = rows(folder/'citation_snapshot/model_components.csv')
    assert len(components) == len({r['model'] for r in components}) == 10
    by_eid = {r['EID']:r for r in citation}
    for r in components:
        source = by_eid[r['EID']]
        assert int(r['citations']) == int(source['Cited by'])
        age = 2026-int(r['earliest_study_year'])+1
        assert int(r['citation_age_years']) == age
        expected = min(1., max(0., int(r['citations'])/age/487.))
        assert abs(float(r['C_new'])-expected) < 1e-14
        assert r['old_C_replaced'] == 'False'
    # Recompute P for a single network directly from the two author result files.
    from statistics import mean
    single=rows(folder/'inception_single/author_raw/results-inception-128.csv')
    ensemble=rows(folder/'inception_single/author_raw/results-InceptionTime-128.csv')
    assert len(single)==640 and len(ensemble)==128
    assert {r['dataset_name'] for r in single}=={r['dataset_name'] for r in ensemble}
    cells=Counter((r['dataset_name'],float(r['iteration'])) for r in single)
    assert len(cells)==640 and all(n==1 for n in cells.values())
    for name in {r['dataset_name'] for r in single}:
        assert {float(r['iteration']) for r in single if r['dataset_name']==name}=={0,1,2,3,4}
    assert all(r['classifier_name']=='inception' and r['archive_name']=='UCRArchive_2018' for r in single)
    assert all(r['classifier_name']=='InceptionTime' and r['archive_name']=='UCRArchive_2018' for r in ensemble)
    single_loss=mean(1-float(r['accuracy']) for r in single)
    ensemble_loss=mean(1-float(r['accuracy']) for r in ensemble)
    candidate=json.loads((folder/'inception_single/candidate.json').read_text())['candidate']
    assert abs(candidate['single_mean_error']-single_loss)<1e-14
    assert abs(candidate['ensemble5_mean_error']-ensemble_loss)<1e-14
    assert candidate['P']==(max(single_loss,ensemble_loss)-single_loss)/abs(single_loss-ensemble_loss)==0
    quality = rows(folder/'quality/current_nine_quality.csv')
    bridge = rows(folder/'quality/model_study_bridge.csv')
    assert len(quality) == 9 and len(bridge) == 10
    access=rows(folder/'quality/current_nine_data_access.csv')
    assert len(access)==9 and all(r['public_data_availability_status']=='public_release_identified' for r in access)
    applied = json.loads((folder/'quality/quality_extraction_matrix_14.json').read_text())
    # The source bundle explicitly declares whether its JSON is rows or a wrapper.
    applied_rows = applied if isinstance(applied, list) else applied['studies']
    assert len(applied_rows) == 14
    scoring = rows(folder/'citation_snapshot/conditional_scores.csv')
    scenario_counts = Counter(r['scenario'] for r in scoring)
    assert scenario_counts and all(n == 10 for n in scenario_counts.values())
    new_C = {r['model']: float(r['C_new']) for r in components}
    for r in scoring:
        assert abs(float(r['C'])-new_C[r['model']]) < 1e-14
        values = [float(r[k]) if r[k] else None for k in ['C','U','P','A','F']]
        weights = [.15,.15,.25,.25,.20]
        low = 100*sum(w*v for w,v in zip(weights,values) if v is not None)
        high = low+100*sum(w for w,v in zip(weights,values) if v is None)
        assert abs(float(r['score_lower'])-low) < 1e-10
        assert abs(float(r['score_upper'])-high) < 1e-10
        assert int(r['known_components']) == sum(v is not None for v in values)
        assert r['controlled_global_rank'] == r['original_selection_rewritten'] == 'False'
    return {'status':'verified','search_records_received':1104,'focal_targets':30,
            'focal_decisions':dict(counts),'citation_studies':9,'citation_models':10,
            'quality_current_studies':9,'quality_applied_studies':14,
            'score_scenarios':dict(scenario_counts),'global_PRISMA_closed':False,
            'historical_sources_preserved':True,'new_model_fits':0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=DEFAULT)
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    print(json.dumps(check(args.bundle,args.root),ensure_ascii=False,indent=2))
