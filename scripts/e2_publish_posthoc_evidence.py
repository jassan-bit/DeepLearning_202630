#!/usr/bin/env python3
"""Export verified *recorded* E2 posthoc evidence to a new documentary folder.

Despite the historical publish name this command is entirely local. It neither
contacts GitHub nor executes pipeline modules, inference, fitting or bootstrap.
Incomplete campaigns are useful inputs: missing receipts remain explicitly
missing, and target-dependent statistics require a verified producer summary.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import tempfile

VERSION = 'e2-posthoc-documentary-export-v1'
DIAGNOSTIC_VERSION = 'e2-frozen-posthoc-inference-v1'
FIT_VERSION = 'e2-posthoc-clean-fit-reproduction-v1'
ENGINE_COMMIT = '1e8745497b484518f766b6da5c19be8362d9add7'
SAE = 'LSTM-autoencoder con predictor'
NEURAL = ('FCN-1D', 'ResNet-1D', 'Inception-1D', SAE, 'TimesNet', 'TSMixer', 'TiDE',
          'TimeMixer', 'PatchTST', 'iTransformer', 'MLP16', 'LSTM32', 'GRU32')
SEEDS = tuple(range(271801, 271807))
VARIANTS = ('remove_auxiliary_pretraining', 'remove_second_encoder', 'dropout_zero')
FEATURES = ('open', 'high', 'low', 'close', 'volume')
SEVERITIES = {'noise':(.1,.3,.5), 'outlier':(.01,.05,.1),
              'missing':(.05,.15,.3), 'structural_break':(.25,.5,1.)}
EXPECTED_DIAGNOSTICS = {(name, seed) for name in NEURAL for seed in SEEDS} | {('ARIMA', 'deterministic')}
EXPECTED_REPRODUCTION = {(name, seed) for name in NEURAL for seed in SEEDS} | {(name, 'deterministic') for name in ('ARIMA', 'Persistence', 'SeasonalNaive')}
EXPECTED_ABLATIONS = {(name, seed) for name in VARIANTS for seed in SEEDS}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def regular(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink paths are not documentary evidence')
    return path


def inside(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Artifact path escapes its receipt directory')
    target = regular(Path(root).joinpath(*relative.parts))
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact path escapes its receipt directory')
    return target


def file_hash(path):
    checksum = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    # No pickle, checkpoints, tensors, source execution or network transport.
    return json.loads(regular(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def vector(value, length=None):
    if not isinstance(value, list) or (length is not None and len(value) != length) or not all(finite(v) for v in value):
        raise ValueError('A finite recorded vector with the expected support is required')
    return value


def verified_receipt(path):
    """Verify local result layout or restored runs/<attempt>/receipt layout.

    Missing claimed artifacts produce an explicit nonadmitted record. A hash or
    path mismatch is corruption and stops export, even for a partial campaign.
    Weights, if claimed, are streamed for SHA only and are never deserialized or
    copied into the documentary output.
    """
    receipt = read_json(path)
    if not isinstance(receipt, dict) or receipt.get('version') not in (DIAGNOSTIC_VERSION, FIT_VERSION):
        return None
    if receipt.get('status') not in ('completed', 'paused_clean'):
        return {'path': path, 'receipt': receipt, 'admitted': False, 'missing': [], 'reason': 'nonterminal_or_failed_receipt'}
    artifacts = receipt.get('artifacts_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        return {'path': path, 'receipt': receipt, 'admitted': False, 'missing': [], 'reason': 'missing_artifact_hash_inventory'}
    base = path.parent/'artifacts' if path.name == 'receipt.json' and (path.parent/'artifacts').is_dir() else path.parent
    missing, hashes = [], {}
    for relative, expected in artifacts.items():
        target = inside(base, relative)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('Invalid claimed artifact hash')
        if not target.is_file():
            missing.append(relative)
        elif file_hash(target) != expected:
            raise ValueError('Claimed posthoc artifact SHA256 mismatch: '+str(target))
        else:
            hashes[relative] = {'sha256': expected, 'bytes': target.stat().st_size}
    manifest_path = path.parent/'manifest.json'
    remote = None
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest.get('receipt_sha256') != digest(canonical(receipt))
                or manifest.get('attempt_id') != receipt.get('attempt_id')
                or manifest.get('fit_status') != receipt['status']):
            raise ValueError('Restored GitHub manifest does not bind its receipt')
        descriptors = {r['path']:r for r in manifest['artifacts']}
        for relative, expected in artifacts.items():
            if relative not in descriptors or descriptors[relative].get('sha256') != expected:
                raise ValueError('Restored manifest artifact inventory differs from receipt')
            if relative in hashes and descriptors[relative].get('bytes') != hashes[relative]['bytes']:
                raise ValueError('Restored manifest artifact byte count differs')
        remote = {'manifest_sha256': file_hash(manifest_path), 'attempt_id': manifest['attempt_id']}
    return {'path': path, 'base': base, 'receipt': receipt, 'admitted': not missing,
            'missing': missing, 'reason': 'verified_receipt_and_all_claimed_artifacts' if not missing else 'claimed_artifacts_not_restored',
            'artifacts': hashes, 'remote_manifest': remote}


def _mean(values):
    return statistics.mean(values) if values else None


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, i = [0.]*len(values), 0
    while i < len(order):
        j = i+1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        for k in range(i,j):
            result[order[k]] = (i+j-1)/2+1
        i = j
    return result


def _spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError('Explanation dimensions differ across seeds')
    a, b = _average_ranks(left), _average_ranks(right)
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = sum((v-ma)**2 for v in a), sum((v-mb)**2 for v in b)
    return None if va == 0 or vb == 0 else sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(va*vb)


def diagnostic_record(record):
    member = read_json(record['base']/'diagnostics.json')
    receipt = record['receipt']
    identity = (member.get('model'), member.get('seed'))
    if (identity not in EXPECTED_DIAGNOSTICS or receipt['status'] != 'completed'
            or member.get('run_id') != receipt.get('run_id') or member.get('baseline_projection_passed') is not True
            or member.get('training_performed') is not False):
        raise ValueError('Recorded diagnostic identity/projection/role differs')
    vector(member['baseline_predictions'],608)
    expected_perturbations = 0 if identity[0] == 'ARIMA' else 12
    perturbations = member.get('perturbations')
    if not isinstance(perturbations,list) or len(perturbations) != expected_perturbations:
        raise ValueError('Completed predictor lacks its recorded perturbations')
    seen = set()
    for item in perturbations:
        key = (item['kind'],item['severity'])
        if key in seen:
            raise ValueError('Duplicate recorded perturbation')
        seen.add(key); vector(item['predictions'],608)
    if identity[0] != 'ARIMA' and seen != {(kind,level) for kind,levels in SEVERITIES.items() for level in levels}:
        raise ValueError('Recorded perturbations differ from the registered severity grid')
    return member


def explanatory_tables(members):
    cells, curves, fidelity, stability, feature_rows = [], [], [], [], []
    groups = defaultdict(list)
    for member in members:
        explanation = member.get('explanation')
        if explanation is None:
            continue
        model, seed = member['model'], member['seed']
        magnitude = vector(explanation['mean_absolute_effect'])
        expected_cells = 7 if model == 'ARIMA' else 35
        if len(magnitude) != expected_cells:
            raise ValueError('Explanation coordinate count differs from its input scope')
        signed = explanation['signed_effects']
        base = vector(explanation['baseline_predictions'])
        if len(signed) != len(base):
            raise ValueError('Signed explanation support differs')
        for row in signed:
            vector(row,expected_cells)
        groups[model].append((seed,magnitude))
        for cell,value in enumerate(magnitude):
            cells.append({'model':model,'seed':seed,'coordinate_index':cell,
                'lag':7-cell if model=='ARIMA' else 7-cell//5,
                'feature':'sigma7_target_history' if model=='ARIMA' else FEATURES[cell%5],
                'mean_absolute_recorded_occlusion_effect':value,
                'mean_signed_recorded_occlusion_effect':_mean([r[cell] for r in signed]),'n_explanation_dates':len(base),
                'scope':explanation.get('input_scope',explanation.get('fidelity_scope'))})
        for dimension,key in (('feature','feature_mean_absolute_effect'),('lag','lag_mean_absolute_effect')):
            for name,value in explanation.get(key,{}).items():
                if not finite(value): raise ValueError('Nonfinite recorded attribution')
                feature_rows.append({'model':model,'seed':seed,'dimension':dimension,'name':name,'recorded_mean_absolute_effect_sum':value})
        fractions = vector(explanation['fractions'])
        for method,key in (('ranked_deletion','deletion_predictions'),('ranked_insertion','insertion_predictions'),
                           ('random_deletion','random_deletion_predictions'),('random_insertion','random_insertion_predictions')):
            values = explanation.get(key)
            if values is None:
                continue
            if len(values) != len(fractions): raise ValueError('Fidelity fraction support differs')
            for fraction,predictions in zip(fractions,values):
                vector(predictions,len(base))
                fidelity.append({'model':model,'seed':seed,'fraction':fraction,'method':method,
                    'mean_absolute_distance_to_original_prediction':_mean([abs(p-b) for p,b in zip(predictions,base)]),
                    'RMSE':None,'RMSE_scope':'requires_verified_producer_summary_with_targets',
                    'scope':'descriptive_recorded_predictor_response_not_causal_market_effect'})
                for sample,(prediction,original) in enumerate(zip(predictions,base)):
                    curves.append({'model':model,'seed':seed,'sample_index_within_record':sample,'fraction':fraction,
                                   'method':method,'recorded_prediction':prediction,'original_prediction':original})
    for model,rows in groups.items():
        pairs = [_spearman(a[1],b[1]) for i,a in enumerate(rows) for b in rows[:i]]
        values = [v for v in pairs if v is not None]
        expected = 1 if model=='ARIMA' else 6
        stability.append({'model':model,'R_available':len(rows),'R_planned':expected,
                          'all_planned_seeds_available':len(rows)==expected,'seed_pairs':len(pairs),'defined_pairs':len(values),
                          'mean_spearman':_mean(values),'minimum_spearman':min(values) if values else None,
                          'scope':'descriptive_available_frozen_seed_rank_stability_no_market_replication'})
    return {'explanation_cells.csv':cells,'explanation_feature_lag.csv':feature_rows,
            'explanation_curve_predictions.csv':curves,'fidelity_prediction_response.csv':fidelity,
            'explanation_stability.csv':stability}


def _flatten(rows):
    # Keep vector-valued confidence intervals without losing their stated scope.
    return [{k:(json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v)
             for k,v in row.items()} for row in rows]


def write_csv(path, rows):
    rows = _flatten(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['evidence_status']
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def export_evidence(source, destination, *, require_complete=False):
    source, destination = regular(source), regular(destination)
    if not source.is_dir() or destination.exists():
        raise ValueError('An existing source and a NEW export destination are required')
    if destination.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(destination.resolve()):
        raise ValueError('Documentary destination must be separate from the source evidence')
    records, seen_receipts, diagnostic_members, fit_jobs, summaries, incomplete = [], set(), {}, {}, [], []
    # Ignore unknown original/transport receipt versions. Never execute them.
    paths = sorted(set(source.rglob('result.json')) | set(source.rglob('receipt.json')))
    for path in paths:
        verified = verified_receipt(regular(path))
        if verified is None:
            continue
        receipt = verified['receipt']
        receipt_hash = digest(canonical(receipt))
        if receipt_hash in seen_receipts:
            continue
        records.append(verified)
        if not verified['admitted']:
            incomplete.append({'receipt':str(path.relative_to(source)),'status':receipt.get('status'),
                               'reason':verified['reason'],'missing_artifacts':verified['missing']})
            continue
        seen_receipts.add(receipt_hash)
        artifacts = receipt['artifacts_sha256']
        if 'diagnostics.json' in artifacts:
            member = diagnostic_record(verified)
            key = (member['model'],member['seed'])
            if key in diagnostic_members:
                raise ValueError('Conflicting diagnostic receipts: isolate one campaign')
            diagnostic_members[key] = member
        elif 'summary.json' in artifacts:
            summary = read_json(verified['base']/'summary.json')
            summaries.append((verified,summary))
        elif receipt.get('kind') in ('reproduction','ablation'):
            kind, model, seed = receipt['kind'],receipt.get('model'),receipt.get('seed')
            key = (kind,model,seed,receipt.get('condition'))
            if key in fit_jobs: raise ValueError('Conflicting completed fit receipts')
            if (kind=='reproduction' and (model,seed) not in EXPECTED_REPRODUCTION
                    or kind=='ablation' and (receipt.get('condition'),seed) not in EXPECTED_ABLATIONS
                    or kind=='reproduction' and receipt.get('condition')!='full'
                    or kind=='ablation' and model!=SAE
                    or receipt['status']!='completed'):
                raise ValueError('Unexpected completed fit identity')
            if ('test_predictions.json' not in artifacts or receipt.get('scientific_commit')!=ENGINE_COMMIT
                    or receipt.get('original_freeze_unchanged') is not True
                    or receipt.get('new_winner_selection_performed') is not False
                    or receipt.get('metadata',{}).get('fresh_fit') is not True):
                raise ValueError('Completed fit lacks the actual producer role/prediction schema')
            predictions = read_json(verified['base']/'test_predictions.json')
            vector(predictions['prediction_original'],608)
            fit_jobs[key] = receipt
    members = [diagnostic_members[k] for k in sorted(diagnostic_members,key=lambda k:(k[0],str(k[1])))]
    run_ids = {m['run_id'] for m in members}
    if len(run_ids)>1: raise ValueError('Diagnostic records belong to different runs')
    fit_configurations = {r.get('configuration_commit') for r in fit_jobs.values()}
    if len(fit_configurations)>1:
        raise ValueError('Fit records belong to different configuration commits')
    repro = [r for r in fit_jobs.values() if r['kind']=='reproduction']
    ablations = [r for r in fit_jobs.values() if r['kind']=='ablation']
    diagnostic_summaries = [(r,s) for r,s in summaries if r['receipt']['version']==DIAGNOSTIC_VERSION]
    fit_summaries = [(r,s) for r,s in summaries if r['receipt']['version']==FIT_VERSION]
    if len(diagnostic_summaries)>1: raise ValueError('Multiple diagnostic root summaries')
    diagnostic_summary = diagnostic_summaries[0][1] if diagnostic_summaries else None
    if diagnostic_summary:
        root_record = diagnostic_summaries[0][0]
        root_run = root_record['receipt'].get('run_id')
        if (root_record['receipt']['status'] != 'completed' or not root_run
                or diagnostic_summary.get('run_id') != root_run or (run_ids and run_ids != {root_run})):
            raise ValueError('Diagnostic root summary belongs to a different or incomplete run')
        required = {'summary.json','registration.json','robustness.csv'}
        if not required.issubset(root_record['receipt']['artifacts_sha256']):
            raise ValueError('Diagnostic root receipt lacks the actual producer artifact schema')
        registration = read_json(root_record['base']/'registration.json')
        if ('posthoc-'+digest(canonical(registration))[:24] != root_run
                or registration.get('version') != DIAGNOSTIC_VERSION):
            raise ValueError('Diagnostic root registration does not bind the actual run identity')
    diagnostic_complete = bool(diagnostic_summary and len(members)==79
        and diagnostic_summaries[0][0]['receipt']['status']=='completed'
        and diagnostic_summary.get('predictor_records')==79
        and diagnostic_summary.get('all_baseline_projections_passed') is True)
    if diagnostic_complete:
        expected_grid={(model,kind,level,block) for model in NEURAL for kind,levels in SEVERITIES.items()
                       for level in levels for block in (7,14,28)}
        rows=diagnostic_summary.get('robustness',[])
        actual_grid={(r['model'],r['kind'],r['severity'],r['block_length']) for r in rows if r.get('R')==6}
        if len(rows)!=len(expected_grid) or actual_grid!=expected_grid:
            raise ValueError('Completed diagnostic summary lacks the full registered robustness grid')
    reproduction_complete = len(repro)==81
    ablation_complete = len(ablations)==18
    # A paused_clean summary can describe real progress, never complete it.
    fit_root_complete = any(r['receipt']['status']=='completed' and s.get('status')=='completed'
                            and s.get('completed_reproduction_fits')==81 and s.get('completed_ablation_fits')==18
                            for r,s in fit_summaries)
    complete = diagnostic_complete and reproduction_complete and ablation_complete and fit_root_complete
    if require_complete and not complete:
        raise ValueError('Complete pipeline receipts are unavailable; partial export is allowed without --require-complete')
    tables = explanatory_tables(members)
    tables['diagnostic_predictors.csv'] = [{'model':m['model'],'seed':m['seed'],'run_id':m['run_id'],
        'checkpoint_sha256':m.get('checkpoint_sha256'),'baseline_projection_passed':m['baseline_projection_passed'],
        'baseline_projection_maximum_absolute_error':m.get('max_absolute_error'),
        'recorded_perturbations':len(m['perturbations']),'explanation_recorded':m.get('explanation') is not None,
        'profile_recorded':m.get('profile') is not None,'elapsed_wall_seconds':m.get('elapsed_wall_seconds')} for m in members]
    profiles = [{'model':m['model'],'seed':m['seed'],**m['profile']} for m in members if m.get('profile') is not None]
    tables['inference_profiles_per_predictor.csv'] = profiles
    grouped = defaultdict(list)
    for row in profiles: grouped[row['model']].append(row)
    tables['inference_profiles_by_model.csv'] = []
    for model,rows in sorted(grouped.items()):
        table = {'model':model,'R_available':len(rows),'R_planned':6,'all_six_seeds_available':len(rows)==6,
                 'aggregation_scope':'mean_of_recorded_seed_level_metrics_not_pooled_latency_samples',
                 'recorded_latency_scopes':sorted({r.get('latency_scope') for r in rows if r.get('latency_scope')}),
                 'device_name_from_verified_root_summary':(diagnostic_summary or {}).get('runtime',{}).get('device_name'),
                 'hardware_scope':'individual_profile_receipts_do_not_record_device_name'}
        for field in ('latency_gpu_median_ms','latency_gpu_p95_ms','throughput_serial_forecasts_per_second',
                      'peak_allocated_bytes','peak_reserved_bytes','incremental_allocated_peak_bytes'):
            values = [r[field] for r in rows if finite(r.get(field))]
            table['mean_'+field] = _mean(values)
            table['n_observed_'+field] = len(values)
        tables['inference_profiles_by_model.csv'].append(table)
    tables['robustness.csv'] = diagnostic_summary.get('robustness',[]) if diagnostic_summary else []
    tables['normalized_degradation_areas.csv'] = diagnostic_summary.get('normalized_degradation_areas',[]) if diagnostic_summary else []
    recorded_fidelity = []
    if diagnostic_summary:
        for explanation in diagnostic_summary.get('interpretability',[]):
            for index,row in enumerate(explanation['fidelity_per_seed']):
                recorded_fidelity.append({'model':explanation['model'],'summary_seed_order_index':index,
                    'seed_identity_scope':'producer_summary_has_order_but_no_explicit_seed_field',**row})
    tables['fidelity_producer_summary.csv'] = recorded_fidelity
    tables['reproduction.csv'] = []
    for row in sorted(repro,key=lambda r:(r['model'],str(r['seed']))):
        metadata = row['metadata']
        tables['reproduction.csv'].append({'model':row['model'],'seed':row['seed'],'attempt_id':row['attempt_id'],
            'recorded_clean_reproduction_passed':metadata.get('clean_reproduction_passed'),
            'rmse_original':row.get('rmse_original'),
            'test608_maximum_absolute_difference':metadata.get('test608_maximum_absolute_difference'),
            'calibration53_maximum_absolute_difference':metadata.get('calibration53_maximum_absolute_difference'),
            'state_comparison':metadata.get('state_comparison'),'scientific_commit':row.get('scientific_commit'),
            'configuration_commit':row.get('configuration_commit'),'new_clean_environment_recorded':bool(row.get('new_clean_environment'))})
    tables['ablation_fits.csv'] = [{'model':r['model'],'seed':r['seed'],'condition':r['condition'],
        'attempt_id':r['attempt_id'],'rmse_original':r.get('rmse_original'),'fresh_fit_recorded':r['metadata'].get('fresh_fit'),
        'new_winner_selection_performed':r.get('new_winner_selection_performed')} for r in sorted(ablations,key=lambda r:(r['condition'],r['seed']))]
    comparisons = {}
    for receipt,summary in fit_summaries:
        for row in summary.get('ablation_comparisons') or []:
            condition = row['condition']
            if condition not in VARIANTS:
                raise ValueError('Unknown producer ablation comparison')
            if condition in comparisons and canonical(comparisons[condition]['recorded_comparison'])!=canonical(row):
                raise ValueError('Conflicting immutable producer ablation comparisons')
            item = comparisons.setdefault(condition, {'recorded_comparison':row,'source_summary_receipts_sha256':[]})
            item['source_summary_receipts_sha256'].append(digest(canonical(receipt['receipt'])))
    # Repeated paused-clean snapshots may embed the same actual comparisons.
    tables['ablation_comparisons.csv'] = [{**item['recorded_comparison'],
        'supporting_18_ablation_and_6_full_SAE_receipts_available':ablation_complete and sum(r['model']==SAE for r in repro)==6,
        'source_summary_receipts_sha256':item['source_summary_receipts_sha256']}
        for condition,item in sorted(comparisons.items())]
    status = {'version':VERSION,'export_status':'completed_evidence' if complete else 'partial_evidence',
        'diagnostics':{'planned_predictors':79,'verified_completed_predictors':len(members),
            'root_summary_receipt_verified':bool(diagnostic_summary),'completed':diagnostic_complete,
            'missing_predictors':[{'model':m,'seed':s} for m,s in sorted(EXPECTED_DIAGNOSTICS-set(diagnostic_members),key=lambda k:(k[0],str(k[1])))],
            'robustness_statistics_status':'verified_producer_summary' if diagnostic_summary else 'unavailable_no_completed_root_summary_no_recomputation'},
        'reproduction':{'planned_fits':81,'verified_completed_fits':len(repro),'completed':reproduction_complete,
            'all_81_recorded_checks_passed':reproduction_complete and all(r['metadata'].get('clean_reproduction_passed') is True for r in repro)},
        'ablations':{'planned_fits':18,'verified_completed_fits':len(ablations),'completed':ablation_complete,
            'producer_comparison_rows':len(tables['ablation_comparisons.csv'])},
        'fit_root_completed_receipt_verified':fit_root_complete,'nonadmitted_receipts':incomplete,
        'orphan_diagnostics_without_result_receipt':[str(p.relative_to(source)) for p in sorted(source.rglob('diagnostics.json'))
            if not any(r.get('base')==p.parent and 'diagnostics.json' in r['receipt'].get('artifacts_sha256',{}) for r in records)],
        'scope':{'new_inference':False,'new_training':False,'new_bootstrap':False,'network_or_publication':False,
            'original_A2_replaced':False,'partial_seed_counts_are_explicit':True,
            'target_dependent_statistics':'copied_only_from_receipt_bound_producer_summaries',
            'derived_tables':'descriptive_recorded_prediction_response_and_available_seed_rank_stability'},
        'limitations':['A completed producer receipt is required before evidence is admitted',
            'Missing claimed artifacts are excluded; modified bytes or unsafe paths fail export',
            'Conditional historical-calendar diagnostics do not establish prospective market performance',
            'Incomplete seed groups do not represent the registered six-seed scope',
            'No raw inputs, model weights, environment trees or secrets are copied']}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        for name,rows in tables.items(): write_csv(temp/name,rows)
        (temp/'status.json').write_bytes(canonical(status)+b'\n')
        (temp/'source_receipts').mkdir(); (temp/'producer_summaries').mkdir()
        sources = []
        for record in records:
            receipt_hash = digest(canonical(record['receipt']))
            (temp/'source_receipts'/(receipt_hash+'.json')).write_bytes(canonical(record['receipt'])+b'\n')
            sources.append({'receipt_source_path':str(record['path'].relative_to(source)),
                'receipt_sha256':file_hash(record['path']),'canonical_receipt_sha256':receipt_hash,
                'admitted':record['admitted'],'reason':record['reason'],'artifacts':record.get('artifacts',{}),
                'remote_manifest':record.get('remote_manifest')})
        for record,summary in summaries:
            (temp/'producer_summaries'/(digest(canonical(record['receipt']))+'.json')).write_bytes(canonical(summary)+b'\n')
        evidence = {'version':VERSION,'status':status['export_status'],'exporter_sha256':file_hash(Path(__file__)),
            'diagnostic_run_ids':sorted(run_ids),'verified_source_receipts':sources,
            'source_directory_modified':False,'original_scientific_outputs_modified':False,
            'outputs':{str(p.relative_to(temp)):{'sha256':file_hash(p),'bytes':p.stat().st_size}
                for p in sorted(temp.rglob('*')) if p.is_file()}}
        (temp/'source_evidence.json').write_bytes(canonical(evidence)+b'\n')
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp); raise
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True,help='Restored posthoc evidence directory; read only')
    parser.add_argument('--destination',type=Path,required=True,help='A NEW static documentary directory; no upload')
    parser.add_argument('--require-complete',action='store_true',help='Reject partial campaigns instead of documenting available evidence')
    args = parser.parse_args()
    status = export_evidence(args.source_root,args.destination,require_complete=args.require_complete)
    print(json.dumps({k:status[k] for k in ('export_status','diagnostics','reproduction','ablations')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
