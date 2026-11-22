"""Read-only audit of frozen E3 temporal methods and real planning inputs.

This program never fits models, queries a market provider, opens test data or
runs new Monte Carlo calibration. It reuses archived validation and performs
deterministic arithmetic/proof checks. AI assistance: implementation/review.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import time

SOURCE_COMMIT = 'c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c'
SOURCE_MANIFEST_SHA256 = 'cd96def4f0a792d0d26d82f94048b0dbc9bf10b35cc7d3a561cf0e0db6d5b12b'
VERSION = 'e3-planning-input-and-estimand-audit-1'
SCIENTIFIC_FILES = {
    'engine/statistics.py':'a14e20b74f668404835f0f15e1fe555c5ac718ab886125b2dea20d531df06e95',
    'engine/power.py':'975ef3b0027f15e4b7cfaa60035a16a8e3ed7a2d54a56482e24090a1ad1bc49a',
    'engine/models.py':'b5a68fc1c12e42296510550b9ab02969fb0b7dfc08be3f1266bc5bcfe74b2446',
    'engine/training.py':'db87c41dd0fc30231950cba686881c402a6da5ab23fcaf5865b5446425444638',
    'engine/recipes.json':'b28e102c585794dc7df79ac8755898d4ffb65a823a0ad57cd91f863c57c0ba7a',
    'configs/CANDIDATE_IDENTITY.json':'d07a106d1bfc8721edbf994b9e4227659d07d6d26a0d32d661c91a747e1592c2',
    'configs/FINAL_SEED_BANK.json':'137fe45eca358081307c041ab31dd94efb23f533202438e2eb6888e4e79836ee',
    'configs/protocol.json':'ce567ec69c2879335432ee2123d4b0dbee2a5612f19b1008c97072e662cb1304',
}
SCIENTIFIC_DESIGN_SHA256 = '46d6c9f4850bc97962e55f43cc6d165ba1210f7618688285d3292463d6c37263'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    if path.exists():
        raise ValueError('Audit receipts are retained: choose a new output/version rather than overwrite '+str(path))
    with path.open('xb') as handle:
        handle.write(encoded)


def frozen_modules(source_root,expected_manifest=SOURCE_MANIFEST_SHA256,expected_commit=SOURCE_COMMIT):
    source = Path(source_root).resolve()
    manifest = source/'SOURCE_MANIFEST.json'
    if not re.fullmatch('[0-9a-f]{64}',expected_manifest) or not re.fullmatch('[0-9a-f]{40}',expected_commit):
        raise ValueError('Explicit immutable source and manifest pins required')
    if digest(manifest) != expected_manifest:
        raise ValueError('Frozen source manifest differs')
    files = read(manifest)['files']
    for relative, expected in files.items():
        path = source/relative
        if not path.resolve().is_relative_to(source) or digest(path) != expected:
            raise ValueError('Frozen source file differs: '+relative)
    for relative,expected in SCIENTIFIC_FILES.items():
        if digest(source/relative) != expected:
            raise ValueError('This data-only audit requires unchanged scientific identity: '+relative)
    sys.path.insert(0,str(source))
    power = importlib.import_module('engine.power')
    statistics = importlib.import_module('engine.statistics')
    data = importlib.import_module('engine.data')
    if Path(power.__file__).resolve() != source/'engine/power.py':
        raise ValueError('Another engine is already imported; use a fresh interpreter')
    return source, power, statistics, data, files


def endpoint_counterexample():
    """Positive seed risks show the two functionals have opposite orderings."""
    candidate = [1.21,1.21]
    rival = [.01,4.]
    mean_root = lambda values: sum(math.sqrt(x) for x in values)/len(values)
    root_mean = lambda values: math.sqrt(sum(values)/len(values))
    return dict(kind='DETERMINISTIC_COUNTEREXAMPLE_NOT_MARKET',
        seed_MSE_candidate=candidate,seed_MSE_rival=rival,
        historical_mean_seed_RMSE_candidate=mean_root(candidate),
        historical_mean_seed_RMSE_rival=mean_root(rival),
        prospective_pooled_RMSE_candidate=root_mean(candidate),
        prospective_pooled_RMSE_rival=root_mean(rival),
        historical_candidate_better=mean_root(candidate)<mean_root(rival),
        prospective_candidate_better=root_mean(candidate)<root_mean(rival),
        statement='sqrt is monotone in aggregate MSE, but average sqrt(seed MSE) is another functional. A favorable observed direction is not a hypothesis test for that other functional.',
        source_policy='The original protocol explicitly declares a prospective endpoint amendment, not academic acceptance or a rewrite of the historical pilot.')


def positive_rule(bits, additional_required=2):
    return bool(bits[0] and sum(bits[1:]) >= additional_required)


def deterministic_composition_check():
    """Exhaust the one-witness reduction for the complete-family conjunction.

    If the asserted joint claim is false, fix one violating rival (or primary
    panel) according to population truth, never according to observed p-values.
    Main false: acceptance requires a false claim in that one panel. Additional
    <=1 true: acceptance requires a false claim in at least one of two fixed null
    panels. Union bound is 2q. The adverse Holm screen only reduces acceptance.
    For a prospective all-three-additional guard, one fixed false panel suffices
    and the bound is q. This program does NOT activate that stricter rule.
    """
    witnesses = []
    checks = 0
    for truth in itertools.product((False,True),repeat=4):
        if positive_rule(truth):
            continue
        if not truth[0]:
            witness = (0,)
        else:
            witness = tuple(i for i in (1,2,3) if not truth[i])[:2]
        for observed in itertools.product((False,True),repeat=4):
            if positive_rule(observed):
                assert any(observed[i] and not truth[i] for i in witness)
            checks += 1
        witnesses.append(dict(population_truth=list(truth),fixed_null_panel_witness=list(witness)))
    all_guard_checks = 0
    for truth in itertools.product((False,True),repeat=4):
        if positive_rule(truth,3):
            continue
        witness = next(i for i in range(4) if not truth[i])
        for observed in itertools.product((False,True),repeat=4):
            if positive_rule(observed,3):
                assert observed[witness] and not truth[witness]
            all_guard_checks += 1
    return dict(kind='EXHAUSTIVE_DETERMINISTIC_BOOLEAN_PROOF',
        current_two_of_three_checks=checks,current_rule_null_witnesses=witnesses,
        current_rule_bound='max(q_main, max(q_i+q_j for two fixed null additional panels)); <=2q for uniform q',
        nominal_alpha_per_panel=.025,nominal_bound_if_valid_marginal_size=.05,
        Holm180_role='Additional adverse EPA screen can remove positive decisions; it cannot enlarge the positive decision event or establish missing marginal super-uniformity.',
        alternative_all_three_guard_checks=all_guard_checks,
        alternative_bound='q for an intersection-union requirement of all panels/all rivals',
        alternative_is_implemented=False,
        caveat='Bounds require valid marginal family error assumptions for the actual DGP; finite pointwise simulation intervals do not establish that assumption for arbitrary markets or simultaneous confidence over scenarios.')


def method_scope(report):
    scenarios = []
    for row in report['scenarios']:
        fwer = row.get('fwer')
        if not fwer:
            continue
        q = fwer['cp95'][1]
        scenarios.append(dict(id=row['id'],count=fwer['count'],trials=fwer['n'],
            observed_FWER=fwer['estimate'],pointwise_CP95_upper=q,
            two_panel_union_upper_from_this_component=min(1.,2*q),
            establishes_marginal_size_at_nominal_025=q<=.025,
            establishes_component_bound_05=q<=.05))
    return dict(status=report['status'],scope=report['scope'],confidence_scope=report['confidence_scope'],
        scenarios=scenarios,new_calibration_replicates_this_audit=0,
        composite_no_pooling=report['no_pooling'],
        earlier_full_null_preserved_excluded=report['original_V4_null_excluded_from_new_null_counts'],
        calibration_upper_threshold=.05,operational_nominal_alpha=.025,
        worst_pointwise_upper=max(s['pointwise_CP95_upper'] for s in scenarios),
        worst_two_panel_union_diagnostic=min(1.,2*max(s['pointwise_CP95_upper'] for s in scenarios)),
        joint_error_control_certified_by_these_diagnostics=False,
        note='A 2q union diagnostic above .05 does not prove inflation of the actual joint rule; it means these marginal bounds alone do not certify the desired joint bound. Pointwise CP95 bounds are not pooled or presented as a simultaneous95% statement.')


def panel_status(panel_map, data):
    if not panel_map:
        return dict(provided=False,development_verified=[],missing=list(('F00','F01','F02','F03')),
                    errors=[],test_opened=False)
    mapped = read(panel_map)
    if set(mapped) != {'F00','F01','F02','F03'}:
        raise ValueError('Panel map must contain exactly F00–F03')
    valid, missing, errors = [], [], []
    for panel, location in mapped.items():
        path = Path(location)
        if not (path/'admission.json').is_file():
            missing.append(panel)
            continue
        try:
            admission = data.read_admission(path)
            if admission['status'] != 'development_admitted' or set(admission['partition_paths']) != {'fit','selection','planning','calibration'}:
                raise ValueError('Real development-only admission required; test/fixtures are forbidden')
            for part in ('fit','selection','planning','calibration'):
                data.load_partition(path,part)
            valid.append(dict(panel=panel,admission_sha256=admission['admission_sha256'],
                              planning_count=admission['partition_counts']['planning']))
        except (ValueError,OSError,KeyError) as exc:
            errors.append(dict(panel=panel,reason=str(exc)))
    return dict(provided=True,development_verified=valid,missing=missing,errors=errors,test_opened=False)


def contract(design,power):
    seeds = power.planning_seed_ids(design,design['minimum_planning_seeds'])
    jobs = []
    for panel in design['panels']:
        for system in design['systems']:
            for seed in seeds if system['kind']=='neural' else ['deterministic']:
                jobs.append(dict(panel=panel,system=system['id'],kind=system['kind'],seed=str(seed)))
    count_dates = design['minimum_within_panel_planning_dates']
    b = power.budget(design,R=len(seeds),T=count_dates)
    requirements = power.resampling_budget(alpha=design['alpha'])
    return dict(schema_version=1,kind='REAL_PLANNING_INPUT_CONTRACT_NOT_ACHIEVED_POWER',
        candidate=design['candidate'],candidate_identity=design['candidate_identity'],panels=design['panels'],
        ordered_systems=[s['id'] for s in design['systems']],planning_seed_ids=seeds,
        expected_successful_jobs=len(jobs),planning_jobs=jobs,
        minimum_dates_per_panel=count_dates,minimum_physical_CSV_rows=b['prediction_rows'],
        CSV_fields=['panel','system','seed','date','y','prediction','known_floor_A','rfit'],
        receipt_kind='development_planning_predictions',receipt_partition='planning',
        receipt_required_fields=['final_test_opened=false','synthetic=false','is_fixture=false',
            'predictions_sha256','seed_bank_sha256','protocol_sha256','methods_sha256',
            'protocol_file:{path,sha256}','methods_file:{path,sha256}',
            'source_manifest_file:{path,sha256}','planning_data_file:{path,sha256}',
            'planning_partition_purged=true','instruments_available_at_origin=true','fit_receipts'],
        successful_job_fields=['panel','system','seed','phase=planning','success=true','active_seconds',
            'checkpoint:{path,sha256}'],
        invariants=['All21systems on each of4panels; neural methods have5same preallocated planning seeds.',
            'Deterministic trajectories are stored once and are not five independent seed replicates.',
            'All methods/seeds share dates,y,known_floor_A and fit-only rfit within a panel.',
            'At least80chronological dates per panel; labels/predictions respect sqrt(known_floor_A).',
            'GW instruments: constant and sqrt(known_floor_A)/rfit with nonconstant full rank.',
            'Actual hash-bound checkpoints/fit receipts; failed attempts retained; no inference from old pilot identities.',
            'Coincident-calendar blocks share actual dates; disjoint groups are not made simultaneous by row numbers.',
            'Fixtures, exposed historical test or incomplete candidate-only pilots do not dimension final protocol.'],
        nominal_planning_budget=b,resampling_resolution=requirements,
        final_grid_available_under_existing_calibration=design['grid'],
        final_grid_is_not_dimensioned=True,final_R=None,final_T=None,final_B=None,
        hypotheses=dict(epa='E_t[mean_fixed_seed loss_rival-loss_candidate]=0',
            practical='E_t[(1-margin)^2 mean_fixed_seed loss_rival - mean_fixed_seed loss_candidate]=0',
            margin_relative_pooled_RMSE=design['margin_relative_rmse'],
            alternatives_relative_pooled_RMSE_gain=design['alternatives_relative_rmse_gain']),
        minimum_final_seeds=10,per_contrast_power_target=design['target_per_contrast_power'],
        joint_power_must_be_reported=True,joint_power_target_090_is_not_universal_guide_requirement=True)


def audit(args):
    start = time.process_time()
    source,power,statistics,data,source_files = frozen_modules(args.source_root,args.expected_manifest_sha256,args.expected_source_commit)
    design_path = Path(args.design) if args.design else source/'configs/power_design.json'
    design = read(design_path)
    scientific_design = dict(design)
    for key in ('version','panel_instrument_proposals','data_design_amendment','status'):
        scientific_design.pop(key,None)
    design_identity = hashlib.sha256(json.dumps(scientific_design,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if design_identity != SCIENTIFIC_DESIGN_SHA256:
        raise ValueError('Scientific power settings differ; retain this version and audit the new design separately')
    report_path = source/design['temporal_validation_report']['path']
    if digest(report_path) != design['temporal_validation_report']['sha256']:
        raise ValueError('Temporal report identity differs')
    archived = read(report_path)
    expected = contract(design,power)
    assert tuple(power.METHODS) == tuple(statistics.METHODS) == ('dm_hln_hac','stationary_bootstrap','gw_conditional')
    cells = []
    for R,T,B in itertools.product(design['grid']['R'],design['grid']['T'],design['grid']['B']):
        cell = dict(R=R,T=T,B=B)
        if args.verify_ledgers:
            admitted,reason = power._calibration(design,statistics.analyze,cell)
        else:
            admitted,reason = None,'Full archived ledger verification not requested; no new admission claimed.'
        cells.append(dict(cell=cell,archived_calibration_verified=admitted,reason=reason))
    provided = dict(predictions=args.predictions,receipts=args.receipts)
    planning = None
    input_error = None
    try:
        planning = power.load_planning(provided,design)
    except (ValueError,OSError,KeyError,json.JSONDecodeError) as exc:
        input_error = str(exc)
    data_status = panel_status(args.panel_map,data)
    missing = []
    if planning is None:
        missing.append(dict(code='REAL_COMPLETE_PLANNING_PREDICTIONS',reason=input_error,
            required_jobs=expected['expected_successful_jobs'],minimum_CSV_rows=expected['minimum_physical_CSV_rows']))
    if len(data_status['development_verified']) != 4:
        missing.append(dict(code='FOUR_VERIFIED_REAL_DEVELOPMENT_ADMISSIONS',details=data_status))
    endpoint = endpoint_counterexample()
    result = dict(version=VERSION,created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        kind='READ_ONLY_INPUT_METHODS_AND_SCOPE_AUDIT',source_commit=args.expected_source_commit,
        source_manifest_sha256=args.expected_manifest_sha256,source_files_verified=len(source_files),
        scientific_identity_base_commit=SOURCE_COMMIT,scientific_files_verified=SCIENTIFIC_FILES,
        scientific_code_unchanged=True,source_root=str(source),design_sha256=digest(design_path),
        scientific_power_design_sha256=design_identity,audit_source_sha256=digest(__file__),
        statistics_sha256=digest(source/'engine/statistics.py'),power_sha256=digest(source/'engine/power.py'),
        report_sha256=digest(report_path),methods=list(statistics.METHODS),
        method_definitions=[dict(id='dm_hln_hac',object='scalar mean temporal squared-loss differential',
            inference='HLN-adjusted HAC statistic against frozen Gaussian fixed-b scalar reference; not an exact finite-sample law'),
            dict(id='stationary_bootstrap',object='paired temporal loss-mean differential',
            inference='stationary-block studentized bootstrap with replicate HAC; common blocks preserve all fixed seed/rival pairing'),
            dict(id='gw_conditional',object='vector conditional loss moments',
            inference='HAC Wald statistic with predictable nonconstant instruments and frozen fixed-b Gaussian vector reference; rejection itself is not directional superiority')],
        family=dict(rivals=20,methods=3,nulls_per_method=2,Holm_positive_slots_per_panel=120,
            additional_EPA_Holm_slots=180,alpha=design['alpha'],direction_required=True,
            historical_mean_seed_RMSE_direction_also_required=True,
            temporal_units='chronological dates after averaging losses over the fixed trained-seed panel; never R*T iid observations',
            fixed_seed_temporal_inference_is_not_unconditional_training_seed_inference=True),
        cells=cells,archived_validation_scope=method_scope(archived),
        deterministic_composition_check=deterministic_composition_check(),
        endpoint_non_equivalence=endpoint,planning_contract=expected,
        planning_inputs_valid=planning is not None,missing_inputs=missing,
        planning_input_evidence=None if planning is None else planning['evidence'],
        panel_admissions=data_status,
        final_status='NOT_DIMENSIONED' if planning is None else 'INPUTS_READY_FOR_CONDITIONAL_EMPIRICAL_DGP_POWER_STUDY',
        final_R=None,final_T=None,final_B=None,achieved_power=None,MDE_achieved=None,
        confirmation_external_gates=[dict(code='PRIMARY_ENDPOINT_ALIGNMENT_NOT_ESTABLISHED',
            reason='MSE/pooled-RMSE inference is not a test of inherited mean individual-seed RMSE. Preserve the explicit prospective amendment and resolve guide alignment before claiming R3.'),
            dict(code='JOINT_DECISION_VALIDATION_NOT_ESTABLISHED',
            reason='Conditional one-panel error bounds do not alone certify the current two-of-three joint positive decision at .05. Do not silently tighten rules or alpha based on observed calibration outcomes.'),
            dict(code='CANDIDATE_MCS_OR_CRITICAL_DIFFERENCE_EVIDENCE',
            reason='Guide requires the own model in MCS or critical-difference evidence. Historical additive MCS and failed later MCS cannot be transferred. Between-panel assumptions also need justification.'),
            dict(code='REAL_POWER_AND_FINAL_DATA_EVIDENCE',reason='A method-calibration ledger, seed minimum, T768 or B8191 do not establish market power or an independent reserve.')],
        new_market_acquisitions=0,new_model_fits=0,new_model_predictions=0,new_calibration_simulations=0,
        reserved_test_opened=False,R3_accredited=False,CPU_seconds=time.process_time()-start)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',required=True)
    parser.add_argument('--expected-source-commit',default=SOURCE_COMMIT)
    parser.add_argument('--expected-manifest-sha256',default=SOURCE_MANIFEST_SHA256)
    parser.add_argument('--design')
    parser.add_argument('--predictions',nargs='*',default=[])
    parser.add_argument('--receipts',nargs='*',default=[])
    parser.add_argument('--panel-map')
    parser.add_argument('--verify-ledgers',action='store_true')
    parser.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    result = audit(args)
    save(args.output,result)
    print(json.dumps(dict(status=result['final_status'],planning_inputs_valid=result['planning_inputs_valid'],
        jobs_required=result['planning_contract']['expected_successful_jobs'],
        minimum_rows=result['planning_contract']['minimum_physical_CSV_rows'],
        final_R=None,final_T=None,final_B=None,output=args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
