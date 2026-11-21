#!/usr/bin/env python3
"""Verify the R3 route supplement without executing scientific procedures.

AI assistance declared. This module imports no forecasting/statistical implementation,
draws no samples and treats an honest partial validation as partial evidence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

BASE = '2fbab63f7c4f19edcd7e9927c0c99d7ddae5d9f5'
REQUIRED_COMPONENTS = {'statistical_revision', 'guide_candidate', 'design_gates',
                       'methods_admissibility', 'bundle_tools'}
REQUEST_SHA256 = '86dd4889912b9dd726ca25c5ba809aead933b9bd2554d0d4bf6831043a74f705'
CODE_MANIFEST_SHA256 = 'ff316a8ad68b7621a99aa5274576b749a656bf859e52f2ab2dd707a6d20e7f01'
VALIDATION_PLAN_SHA256 = '5379b30a632f20494599389ad9af5aa8e30fd826a306d148dd21ccddb9bdde8f'
VALIDATION_PUBLIC_COMMIT = '374fc00f50da4454522c04a7973d0ef590d5e192'
FORBIDDEN_PARTS = {'__pycache__', '.git', 'private_cases', 'private_case_current',
                   'source-checkout', 'wheelhouse', 'book_wheelhouse', 'before'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.npz', '.npy', '.whl', '.pyc', '.pyo',
                      '.onnx', '.safetensors', '.ckpt', '.pdf'}


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def safe_path(value):
    require(isinstance(value, str) and value and '\x00' not in value,
            'empty or invalid member path')
    p = PurePosixPath(value)
    require(p.parts and not p.is_absolute() and '..' not in p.parts and p.as_posix() == value
            and '\\' not in value, 'unsafe/noncanonical member: ' + value)
    require(not value.startswith('guide_candidate/primary/') and not FORBIDDEN_PARTS.intersection(p.parts)
            and p.suffix.lower() not in FORBIDDEN_SUFFIXES,
            'excluded private/model/cache payload: ' + value)
    return p


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decode_json(text):
    def unique(pairs):
        d = {}
        for key, value in pairs:
            require(key not in d, 'duplicate JSON key: ' + key)
            d[key] = value
        return d
    def reject(value):
        raise VerificationError('nonfinite JSON: ' + value)
    def finite_float(value):
        number=float(value)
        require(math.isfinite(number),'nonfinite JSON numeric overflow: '+value)
        return number
    return json.loads(text, object_pairs_hook=unique, parse_constant=reject,parse_float=finite_float)


def read_json(root, relative):
    p = root.joinpath(*safe_path(relative).parts)
    require(p.is_file() and not p.is_symlink(), 'missing/linked JSON: ' + relative)
    value = decode_json(p.read_text(encoding='utf-8'))
    require(isinstance(value, dict), 'JSON object required: ' + relative)
    return value


def actual_members(root):
    found = set()
    for p in root.rglob('*'):
        rel = p.relative_to(root).as_posix()
        require(not p.is_symlink(), 'symlink in immutable packet: ' + rel)
        if p.is_file():
            safe_path(rel)
            found.add(rel)
    return found


def check_entries(root, entries, prefix=''):
    require(isinstance(entries, list) and entries, 'manifest has no payload list')
    names, total = set(), 0
    for e in entries:
        require(isinstance(e, dict), 'invalid payload record')
        rel = e.get('path'); safe_path(rel)
        full = prefix + rel; safe_path(full)
        require(full not in names, 'duplicate member: ' + full)
        require(type(e.get('bytes')) is int and e['bytes'] >= 0,
                'invalid payload size: ' + full)
        require(isinstance(e.get('sha256'), str)
                and re.fullmatch('[0-9a-f]{64}', e['sha256']), 'invalid hash: ' + full)
        p = root.joinpath(*PurePosixPath(full).parts)
        require(p.is_file() and not p.is_symlink(), 'missing/linked member: ' + full)
        require(p.stat().st_size == e['bytes'] and digest(p) == e['sha256'],
                'payload bytes/hash differ: ' + full)
        names.add(full); total += e['bytes']
    return names, total


def unchanged_false(d, keys, label):
    for k in keys:
        require(d.get(k) is False, label + ' unsupported assertion: ' + k)


def zero_fields(d, keys, label):
    for k in keys:
        require(type(d.get(k)) is int and d[k] == 0, label + ' nonzero/absent counter: ' + k)


def check_initial(root, packet):
    require(packet.get('source_base_commit') == BASE, 'packet source base differs')
    require(packet.get('request_sha256') == REQUEST_SHA256, 'packet request binding differs')
    zero_fields(packet, ['new_forecasting_model_fits', 'new_forecasting_model_forward_passes',
                         'new_market_predictions'], 'packet')
    unchanged_false(packet, ['R3_accredited', 'confirmation_enabled', 'human_execution_claimed'], 'packet')
    require(packet.get('AI_assistance_declared') is True, 'AI assistance declaration absent')
    initial = read_json(root, 'initial_state.json')
    require(initial.get('source_commit') == BASE and initial.get('request_sha256') == REQUEST_SHA256,
            'initial source/request differs')
    zero_fields(initial, ['new_market_fits', 'new_market_predictions', 'new_controlled_replicates'],
                'initial counters before this route')
    unchanged_false(initial, ['confirmation_enabled', 'R3_accredited', 'oral_defense_observed'],
                    'initial scope')
    require(initial.get('oral_grade_author_provided') == 4.5
            and initial.get('AI_assistance_declared') is True, 'oral score/declaration scope differs')
    saved = initial.get('preserved', {})
    for key, value in [('additive_jobs',796),('V3_fits',30),('transfer_rows',20400),
                       ('V4_fits',20),('V4_rows',1210),('last_controlled_replicates',640),
                       ('last_bootstrap_draws',639360)]:
        require(saved.get(key) == value, 'historical count changed: ' + key)
    require(saved.get('V3_NO_GO') is True and saved.get('V4_abandoned') is True,
            'historical scientific stops changed')
    bound=initial.get('observed_prior_packet_bindings', {}).get('previous_statistics_packet', {})
    require(bound.get('sha256') == 'af84932e2bf6fb4f4a63ca6c18a0e7cd036998affd1485fcb8b705d3233a525e',
            'previous statistics supplement binding differs')


def check_design_gates(root):
    protocol=read_json(root,'design_gates/protocol.DRAFT.json')
    require(protocol.get('kind') == 'DRAFT_NOT_PREREGISTERED', 'draft falsely labelled preregistered')
    require(protocol.get('source_main_commit') == BASE, 'draft source base differs')
    unchanged_false(protocol,['confirmation_registered','R3_accredited'],'draft protocol')
    zero_fields(protocol,['new_model_fits','new_model_predictions','new_MC_replicates'],'draft protocol')
    design=protocol.get('design',{});sampling=design.get('sampling',{})
    require(design.get('candidate') is None and design.get('family') is None
            and sampling.get('final_R') is None and sampling.get('final_T') is None
            and protocol.get('campaign_authorization') is None,
            'unassigned candidate/family/R/T/authorization became an execution permit')
    design_hash=hashlib.sha256(json.dumps(design,sort_keys=True,ensure_ascii=False,
                                       separators=(',',':'),allow_nan=False).encode()).hexdigest()
    for filename,phase,n in [('CURRENT_PREFLIGHT.json','confirmation',9),
                             ('CURRENT_PLANNING_PREFLIGHT.json','planning',7)]:
        state=read_json(root,'design_gates/'+filename)
        require(state.get('phase') == phase and state.get('protocol_status') == 'DRAFT_NOT_PREREGISTERED',
                'preflight phase/draft status differs')
        require(state.get('design_sha256') == design_hash,'preflight draft design binding differs')
        unchanged_false(state,['technical_preflight_ready','execution_permit_conditionally_ready',
                               'R3_accredited','receipt_authenticity_verified_by_program',
                               'power_certified_by_this_program'],'design preflight')
        require(state.get('scientific_accreditation') == 'NOT_ASSESSED_BY_SOFTWARE',
                'technical controller claimed scientific accreditation')
        zero_fields(state,['new_model_fits','new_model_predictions','new_MC_replicates',
                           'new_model_forward_passes','new_market_data_downloads','new_market_p_values',
                           'new_statistical_resamples'],'design preflight')
        checks=state.get('checks',{})
        require(len(checks) == n and all(value is False for value in checks.values())
                and set(state.get('blockers',[])) == set(checks)
                and state.get('execution_manifest') == [], 'blocked preflight became runnable')
        require(state.get('controller_source_sha256') == digest(root/'design_gates/controller.py'),
                'preflight controller source binding differs')
    cost=read_json(root,'design_gates/COST_INPUTS.json')
    require(cost.get('proposed_current_cell_count') is None,
            'historical dimensions promoted to current proposed cells')
    require(cost.get('actual_admitted_panels') == 0 and cost.get('panel_count_if_admitted_P') == 4,
            'unadmitted panel proposals became actual evidence')
    historical=cost.get('historical_only',{})
    require(historical.get('V3_panel_rival_cells_unfilled') == 76
            and historical.get('V4_panel_rival_cells_unfilled') == 84
            and historical.get('V4_nominal_340_authorized') is False,
            'historical cells or nominal fits were promoted/changed')
    require(all(cost.get(k) is None for k in ['measured_per_fit_seconds','elapsed_seconds_estimate',
                                              'currency_cost_estimate']),
            'unmeasured resource costs became reported measurements')
    unchanged_false(cost,['execution_authorized','R3_accredited'],'cost scenario')


def check_methods(root):
    methods=read_json(root,'methods_admissibility/METHOD_ADMISSIBILITY.json')
    prior=methods.get('prior_synthetic_study_dimensions',{})
    require(methods.get('baseline_commit') == BASE and prior.get('rivals') == 19
            and prior.get('models') == 20, 'historical method population/family differs')
    future=methods.get('prospective_family',{})
    require(future.get('rival_count') is None and future.get('total_model_count') is None,
            'historical synthetic dimensions became the future roster')
    unchanged_false(future,['roster_frozen','abandoned_candidate_inclusion_authorized'],
                    'prospective method family')
    require(methods.get('estimand') == 'rho_m=E_seed[sqrt(E_time[squared_error_m(seed,time)])]'
            and methods.get('equality_null') == 'theta_j=0', 'mean-individual-RMSE equality target changed')
    require(methods.get('distinct_pairwise_candidate_families') == 2
            and methods.get('distinct_global_candidate_families') == 1
            and methods.get('confirmed_admissible_and_calibrated_primary_procedures') == 0,
            'candidate procedure families promoted to accredited procedures')
    unchanged_false(methods,['three_pairwise_procedures_justified',
                             'global_route_currently_admissible_for_confirmation',
                             'confirmatory_gate_open','R3_accredited'],'method admissibility')
    zero_fields(methods,['new_random_draws','new_statistical_resamples','new_MC_replicates',
                         'new_model_fits','new_market_predictions'],'documentary methods review')
    procedures=methods.get('procedures',[])
    require(len(procedures) == 3 and {x.get('id') for x in procedures} == {'P1','P2','P3'},
            'three explicitly scoped procedure records required')
    for procedure in procedures:
        require(procedure.get('counts_as_accredited_primary') is False,
                'unaccredited primary procedure promoted')
    p2=next(x for x in procedures if x['id']=='P2');p3=next(x for x in procedures if x['id']=='P3')
    require(p2.get('classical_DM_HLN_theorem_claimed') is False,
            'classical DM-HLN theorem misapplied to crossed delta-HAC')
    require(p3.get('role') == 'global_optimal_set_coverage_and_candidate_singleton'
            and p3.get('pairwise_p_values_provided') is False
            and p3.get('singleton_is_margin_test') is False,
            'MCS global singleton converted into pairwise p-values/margin inference')
    multiplicity=methods.get('multiplicity',{})
    unchanged_false(multiplicity,['MCS_alpha_0_10_is_FWER_0_05','guide_requires_MCS_alpha_0_10',
                                   'prior_alpha_0_10_screens_establish_R3_control_at_0_05'],
                    'MCS exploratory alpha interpretation')
    require(multiplicity.get('prospective_MCS_alpha') is None,
            'exploratory alpha became the future scientific rule')
    plan=read_json(root,'methods_admissibility/THIRD_PROCEDURE_PLAN.json')
    unchanged_false(plan,['old_initial_EPA_screen_is_set_coverage_validation',
                          'new_validation_authorized_by_this_file','confirmatory_gate_open',
                          'R3_accredited'],'third-procedure plan')
    zero_fields(plan,['new_random_draws','new_MC_replicates','new_model_fits'],'third-procedure plan')


def check_guide_candidate(root):
    candidate=read_json(root,'guide_candidate/CANDIDATE_VIABILITY_AND_NEXT_DECISION.json')
    require(candidate.get('current_candidate_for_confirmation') is None
            and candidate.get('verdict') == 'NO_CURRENT_CANDIDATE_ENABLED_FOR_CONFIRMATION',
            'documentary candidate audit promoted an unselected model')
    rows=candidate.get('candidate_rows',[])
    require(len(rows) == 5 and len({r.get('candidate') for r in rows}) == 5
            and all(r.get('viable_for_current_confirmation') is False for r in rows),
            'candidate inventory or current viability changed')
    v3=next((r for r in rows if r.get('candidate')=='V3_atom_crps_1262'),{})
    v4=next((r for r in rows if r.get('candidate')=='V4_sigma_crps_1262'),{})
    require(v3.get('development_fits') == 30 and v3.get('development_decision') == 'NO_GO_PRESERVED'
            and 0.027 < v3.get('relative_RMSE_increase_from_existing_receipt',0) < 0.028,
            'V3 historical adverse outcome or stop changed')
    require(v4.get('development_fits_in_new_CPU_pilot_total') == 20
            and v4.get('development_decision') == 'ABANDON_CURRENT_V4_RECIPE'
            and 0.014 < v4.get('relative_RMSE_increase_from_existing_receipt',0) < 0.015,
            'V4 historical adverse outcome or stop changed')
    relation=candidate.get('R1_R2_R3_relation',{})
    require(relation.get('R3_evaluated_only_when_R1_R2_met') is True,'guide prerequisite misrepresented')
    unchanged_false(relation,['poor_development_utility_implies_R1_no',
                              'unobserved_student_defense_implies_copying_or_R2_no',
                              'pending_equals_demonstrated_no','new_candidate_inherits_R1_or_R2_or_R3',
                              'automatic_grade_or_cap_assigned'],'academic interpretation')
    execution=candidate.get('execution',{})
    zero_fields(execution,['new_model_fits','new_predictions','new_MC_replicates',
                           'new_statistical_resamples'],'candidate documentary audit')
    unchanged_false(execution,['confirmation_enabled','V5_started','R3_accredited'],'candidate execution')
    next_decision=candidate.get('next_decision',{})
    require(all(next_decision.get(k) is None for k in ['new_candidate_selection','final_R','final_T',
                                                      'final_budget']),
            'documentary next decision became an executed/final campaign')
    matrix=read_json(root,'guide_candidate/REQUIREMENT_PRIMARY_EVIDENCE_GAP_GATE_MATRIX.json')
    zero_fields(matrix,['new_model_fits','new_predictions','new_Monte_Carlo_replicates',
                        'new_statistical_resamples','new_literature_searches'],'guide matrix')
    unchanged_false(matrix,['human_execution_claimed','R3_accredited'],'guide matrix')
    inputs=read_json(root,'guide_candidate/INPUT_BYTES_MANIFEST.json')['inputs']
    require(inputs.get('guide',{}).get('sha256') ==
            'd6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364',
            'guide primary PDF bytes differ')
    requirements=matrix.get('rows',[]);ids=[r.get('requirement_id') for r in requirements]
    require(len(requirements) >= 30 and len(set(ids)) == len(ids),'guide matrix is incomplete/duplicated')
    for row in requirements:
        require(row.get('primary_sha256') == inputs['guide']['sha256']
                and row.get('primary_verbatim_fragment') and row.get('gap') and row.get('gate'),
                'guide row lost a primary excerpt or gap/gate')
        pages=row.get('PDF_pages',[])
        require(pages and all(type(n) is int and n>0 for n in pages),'guide PDF-page labels absent')
        evidence=row.get('evidence',[])
        require({e.get('ref') for e in evidence} == set(row.get('evidence_refs',[])),
                'guide evidence references differ')
        for entry in evidence:
            ref=entry['ref'];require(ref in inputs,'unknown guide evidence reference')
            require({k:entry.get(k) for k in ['path','bytes','sha256']} == inputs[ref],
                    'guide evidence byte binding differs: '+ref)
    with (root/'guide_candidate/REQUIREMENT_PRIMARY_EVIDENCE_GAP_GATE_MATRIX.csv').open(newline='') as stream:
        csv_ids=[row['requirement_id'] for row in csv.DictReader(stream)]
    require(csv_ids == ids,'guide JSON/CSV requirement order/identity differs')
    return len(requirements)


def check_component_semantics(root,packet):
    return check_statistical_revision(root)


def check_closure(root,packet,counts):
    closure=read_json(root,'CLOSURE_STATE.json')
    require(closure.get('source_base_commit')==packet['source_base_commit'],
            'closure source-base binding differs')
    for key in ['new_forecasting_model_fits','new_forecasting_model_forward_passes','new_market_predictions']:
        require(type(closure.get(key)) is int and closure[key]==packet[key],
                'closure counter contradicts packet: '+key)
    for key in ['new_controlled_replicates','new_completed_replicate_bootstrap_draws']:
        require(type(closure.get(key)) is int and closure[key]==counts[key],
                'closure counter contradicts the verified raw execution: '+key)
    for key in ['R3_accredited','confirmation_enabled','human_execution_claimed','AI_assistance_declared']:
        require(closure.get(key) is packet[key],'closure assertion contradicts packet: '+key)


def timestamp(value):
    instant=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(instant.tzinfo is not None,'timezone required in execution/publication receipt')
    return instant


def binomial_rate(record,events,n,label):
    require(type(record.get('events')) is int and record['events'] == events
            and record.get('replicates') == n and n>0, 'raw event count differs: '+label)
    require(type(record.get('rate')) in (int,float)
            and abs(record['rate']-events/n) < 1e-14, 'event rate differs: '+label)
    interval=record.get('CI95',[])
    require(isinstance(interval,list) and len(interval) == 2
            and all(type(x) in (int,float) and math.isfinite(x) for x in interval)
            and 0<=interval[0]<=events/n<=interval[1]<=1
            and record.get('simultaneous_interval') is False,
            'individual binomial interval was lost/promoted: '+label)


def tolerance_screen(record,nominal,unsupported,label):
    lo,hi=record['CI95']
    expected=('inflation_detected' if lo>nominal else
              'exploratory_tolerance_screen_passed' if hi<=nominal+.03 else 'inconclusive')
    if unsupported:expected='inconclusive_unsupported_replicates'
    require(record.get('nominal')==nominal and record.get('exploratory_tolerance')==.03
            and record.get('screen')==expected,'screen contradicts its saved bounds/support: '+label)
    unchanged_false(record,['exact_nominal_level_certified','market_calibration_certified'],label)


def coverage_screen(record,label):
    lo,hi=record['CI95']
    expected=('undercoverage_detected' if hi<.95 else
              'exploratory_tolerance_screen_passed' if lo>=.92 else 'inconclusive')
    require(record.get('nominal')==.95 and record.get('exploratory_tolerance')==.03
            and record.get('screen')==expected,'coverage screen contradicts its saved bounds: '+label)
    require(record.get('simultaneous_coverage_claimed') is False,
            'pointwise coverage promoted to simultaneous coverage')


def check_statistical_revision(root):
    prefix='statistical_revision/'
    plan=read_json(root,prefix+'VALIDATION_PLAN.json')
    code=read_json(root,prefix+'CODE_MANIFEST.json')
    require(digest(root/prefix/'CODE_MANIFEST.json') == CODE_MANIFEST_SHA256
            and digest(root/prefix/'VALIDATION_PLAN.json') == VALIDATION_PLAN_SHA256,
            'reviewed/frozen source or plan bytes differ')
    expected_runtime={'crossed_rmse_v2_frozen.py','revised_primary.py','dgp_revision.py',
                      'run_validation.py','test_deterministic.py','VALIDATION_PLAN.json',
                      'LEGACY_REFERENCE.json','deterministic_diagnosis.py'}
    runtime,_=check_entries(root,code.get('files'),prefix)
    require(code.get('kind') == 'frozen_before_new_primary_validation'
            and runtime == {prefix+p for p in expected_runtime},'frozen runtime inventory differs')
    require(digest(root/prefix/'crossed_rmse_v2_frozen.py') ==
            '849f3152676f58736141476bb6fe52d90675e4c821f67030476d13e24bd06b2a',
            'legacy adverse-method control changed')
    cases=plan.get('cases',[])
    require(plan.get('baseline_commit') == BASE and len(cases) == 7
            and [c.get('replicates') for c in cases] == [256]*6+[128]
            and plan.get('planned_replicates') == 1664
            and plan.get('planned_crossed_bootstrap_draws') == 1662336,
            'bounded case allocation/draw plan differs')
    require(plan.get('validation_seed') == 2026112121
            and plan['validation_seed'] not in plan.get('prior_validation_seeds_not_reused',[])
            and plan.get('threads') == 2 and plan.get('max_wall_seconds') == 600
            and plan.get('settings') == {'B':999,'mean_block':12.,'hac_lag':8,
                                         'alpha':.05,'mcs_alpha':.10,'batch':16},
            'frozen development settings/seed/resource cap differ')
    unchanged_false(plan,['R3_accredited','confirmation_allowed','third_primary_method_accredited'],
                    'development validation plan')
    zero_fields(plan,['new_market_fits','new_market_forwards','new_market_predictions'],
                'development validation plan')
    receipt=read_json(root,prefix+'validation_new/execution_receipt.json')
    decision=read_json(root,prefix+'validation_new/RESULTS_DECISION.json')
    status=receipt.get('status');n=receipt.get('completed_replicates')
    require(status in {'completed','resource_stop','error','not_run'} and type(n) is int
            and 0<=n<=1664,'execution has not stopped honestly')
    unchanged_false(receipt,['R3_accredited'],'controlled execution')
    zero_fields(receipt,['new_market_fits','new_market_forwards','new_market_predictions'],
                'controlled execution')
    unchanged_false(decision,['primary_market_admissibility_certified','exact_nominal_level_certified',
                              'third_primary_method_accredited','complete_three_method_power_computed',
                              'confirmed_valid_for_market','confirmation_allowed','R3_accredited'],
                    'controlled-development decision')
    zero_fields(decision,['new_market_fits','new_market_predictions','new_market_forwards'],
                'controlled-development decision')
    require(decision.get('V3_NO_GO_unchanged') is True
            and decision.get('V4_abandonment_unchanged') is True,'controlled study revived old candidates')
    if status == 'not_run':
        require(n == 0 and decision.get('complete') is False and receipt.get('reason'),
                'not-run branch has results or lacks its reason')
        return {'new_controlled_replicates':0,'new_completed_replicate_bootstrap_draws':0,
                'controlled_validation_status':'not_run'}
    anchor=read_json(root,prefix+'PUBLIC_ANCHOR.json')
    require(anchor.get('status') == 'new_primary_code_and_plan_published_before_validation'
            and anchor.get('observed_by_root_before_validation') is True
            and anchor.get('commit') == VALIDATION_PUBLIC_COMMIT,'actual prior public anchor differs')
    require(anchor.get('plan_SHA256') == digest(root/prefix/'VALIDATION_PLAN.json')
            and anchor.get('source_manifest_SHA256') == digest(root/prefix/'CODE_MANIFEST.json')
            and receipt.get('plan_SHA256') == anchor['plan_SHA256']
            and receipt.get('source_manifest_SHA256') == anchor['source_manifest_SHA256']
            and receipt.get('public_anchor_commit') == anchor['commit'],
            'executed code/plan does not match the prior public source')
    require(timestamp(anchor['observed_at_utc']) <= timestamp(receipt['started_at_utc']) <=
            timestamp(receipt['finished_at_utc']),'public source was not observed before execution')
    attempted=receipt.get('attempted_replicates')
    require(type(attempted) is int and n<=attempted<=1664,'invalid attempted replication count')
    failed=attempted-n
    require(receipt.get('planned_replicates') == 1664
            and receipt.get('planned_crossed_bootstrap_draws') == 1662336
            and receipt.get('completed_replicate_bootstrap_draws') == n*999
            and receipt.get('actual_crossed_bootstrap_draws') == (None if failed else n*999)
            and receipt.get('failed_attempt_bootstrap_draws_exact') == (None if failed else 0)
            and receipt.get('failed_attempt_bootstrap_draws_upper_bound') == failed*999
            and receipt.get('control_extra_bootstrap_draws') == 0
            and receipt.get('source_hashes_still_match') is True
            and receipt.get('source_implementation_changed_after_results') is False,
            'draw accounting or immutable source binding differs')
    require(receipt.get('runtime',{}).get('OPENBLAS_NUM_THREADS') == '2'
            and receipt.get('runtime',{}).get('OMP_NUM_THREADS') == '2','execution thread cap differs')
    require(receipt.get('raw_log_SHA256') == digest(root/prefix/'validation_new/per_replicate.jsonl.gz'),
            'raw replication log receipt hash differs')
    summary=read_json(root,prefix+'validation_new/validation_summary.json')
    require(summary.get('status') == status and summary.get('complete') is (status=='completed')
            and summary.get('public_anchor_commit') == anchor['commit']
            and summary.get('source_manifest_SHA256') == anchor['source_manifest_SHA256'],
            'overall validation receipt differs')
    unchanged_false(summary,['three_primary_pairwise_tests_accredited','R3_accredited'],
                    'overall validation')
    zero_fields(summary,['new_market_fits','new_market_predictions','new_market_forwards'],
                'overall validation')
    require(decision.get('complete') is (status=='completed')
            and (status!='completed' or n==1664),'partial/error study promoted to complete')
    verify_raw_validation(root,plan,receipt,summary)
    screens=[case.get('known_true_null_component_FWER_Holm38',{}).get('screen')
             for case in summary.get('cases',[]) if case.get('true_null_rivals')]
    expected_status=('incomplete_no_family_approval' if status!='completed' else
                     'new_development_family_not_approved_inflation_detected'
                     if 'inflation_detected' in screens else
                     'bounded_component_FWER_tolerance_screens_passed_not_accreditation'
                     if len(screens)==6 and all(s=='exploratory_tolerance_screen_passed' for s in screens)
                     else 'new_development_family_inconclusive_not_approved')
    require(decision.get('known_true_null_FWER_screens') == screens
            and decision.get('status') == expected_status,'adverse/incomplete development decision changed')
    return {'new_controlled_replicates':n,'new_completed_replicate_bootstrap_draws':n*999,
            'controlled_validation_status':status,
            'failed_attempt_draw_accounting_requires_execution_receipt':bool(receipt.get('errors'))}


def verify_raw_validation(root,plan,receipt,summary):
    """Recount saved decisions; no p-values, intervals, samples or models recomputed."""
    prefix='statistical_revision/validation_new/'
    specs={c['id']:c for c in plan['cases']};seen=set();counts={k:0 for k in specs}
    events={k:{'new':0,'old':0,'P1':0,'P2':0,'intersection':0,'old_only':0,'new_only':0,
               'all_best':0,'any_best_excluded':0,'candidate_retained':0,'unique':0,'false_unique':0,
               'unsupported_P1':0,'unsupported_P2':0,'initial_EPA':0,'unsupported_MCS':0}
            for k in specs}
    coverage={k:[[0,0] for _ in range(19)] for k in specs};attempted=0;errors=0
    with gzip.open(root/prefix/'per_replicate.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            record=decode_json(line);case_id=record.get('case');replicate=record.get('replicate')
            require(case_id in specs and type(replicate) is int
                    and 0<=replicate<specs[case_id]['replicates']
                    and (case_id,replicate) not in seen,'unknown/duplicate/out-of-budget replicate')
            seen.add((case_id,replicate));attempted+=1
            if record.get('error'):
                errors+=1;continue
            case=specs[case_id];counts[case_id]+=1;result=record.get('result',{})
            truth=[x-case['bases'][0] for x in case['bases'][1:]]
            require(record.get('bases') == case['bases']
                    and record.get('analytic_longrun_theta') == truth,
                    'analytic population truth changed from the frozen DGP')
            require(result.get('shape') == [case['R'],case['T'],case['M']]
                    and result.get('bootstrap_studentizers_recomputed') == 999
                    and result.get('pairwise_methods') == 2
                    and result.get('P1_CI_absolute_pivot_order_rank') == 951,
                    'raw result axes/method/draw allocation differs')
            zero_fields(result,['new_market_fits','new_market_predictions'],'raw controlled replicate')
            unchanged_false(result,['third_primary_method_accredited','confirmed_valid_for_market',
                                     'R3_accredited'],'raw controlled replicate')
            pairs=result.get('pairwise',[]);names=[f'rival_{i:02d}' for i in range(1,20)]
            require(len(pairs) == 19 and [p.get('rival') for p in pairs] == names,
                    'raw primary family lost/reordered rival slots')
            true_indices=[i for i,t in enumerate(truth) if abs(t)<=1e-12]
            new_events=[];old_events=[];p1_events=[];p2_events=[];joint_events=[]
            supported_rep=[True,True]
            for j,pair in enumerate(pairs):
                adjusted=[pair.get('P1_Holm'),pair.get('P2_Holm')]
                old=[pair.get('P1_basic_frozen_control_Holm38'),pair.get('P2_frozen_control_Holm38')]
                require(all(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1
                            for p in adjusted+old),'raw adjusted p-value missing/nonfinite/outside range')
                for index,method in enumerate(['P1','P2']):
                    interval=pair.get(method+'_CI95')
                    supported=pair.get(method+'_supported')
                    require(type(supported) is bool and (interval is not None) == supported,
                            'unsupported method promoted to an interval')
                    supported_rep[index]=supported_rep[index] and supported
                    if interval is not None:
                        require(len(interval)==2 and interval[0]<=interval[1],'invalid saved interval')
                        coverage[case_id][j][index]+=int(interval[0]<=truth[j]<=interval[1])
                    else:require(adjusted[index] == 1,'unsupported family slot did not retain p=1')
                if j in true_indices:
                    new_events.append(any(p<.05 for p in adjusted));old_events.append(any(p<.05 for p in old))
                    p1_events.append(adjusted[0]<.05);p2_events.append(adjusted[1]<.05)
                    joint_events.append(all(p<.05 for p in adjusted))
            new=any(new_events);old=any(old_events);event=events[case_id]
            for key,value in [('new',new),('old',old),('P1',any(p1_events)),('P2',any(p2_events)),
                              ('intersection',any(joint_events)),('old_only',old and not new),
                              ('new_only',new and not old),('unsupported_P1',not supported_rep[0]),
                              ('unsupported_P2',not supported_rep[1])]:event[key]+=int(value)
            mcs=result.get('MCS_bootstrap_t_frozen_control',{});retained=mcs.get('retained',[])
            global_p=mcs.get('initial_global_p')
            require(global_p is None or type(global_p) in (int,float) and 0<=global_p<=1,
                    'invalid saved MCS global EPA p-value')
            event['initial_EPA']+=int(global_p is not None and global_p<.10)
            event['unsupported_MCS']+=int(global_p is None)
            all_names=['candidate']+names
            require(retained and len(set(retained)) == len(retained) and set(retained)<=set(all_names),
                    'MCS retained set is invalid')
            best={all_names[i] for i,value in enumerate(case['bases'])
                  if abs(value-min(case['bases']))<=1e-12}
            contains=best<=set(retained);unique=retained==['candidate']
            for key,value in [('all_best',contains),('any_best_excluded',not contains),
                              ('candidate_retained','candidate' in retained),('unique',unique),
                              ('false_unique',unique and best!={'candidate'})]:event[key]+=int(value)
    require(sum(counts.values()) == receipt['completed_replicates']
            and attempted == receipt.get('attempted_replicates')
            and errors == len(receipt.get('errors',[])),'raw completed/attempt/error accounting differs')
    summaries=summary.get('cases',[])
    require(len({s['id'] for s in summaries}) == len(summaries)
            and all(s['id'] in specs for s in summaries),'case summary inventory differs')
    require({case_id for case_id,_ in seen} <= {s['id'] for s in summaries},
            'raw replication case was omitted from the reported summaries')
    if receipt['status']=='completed':
        require(len(summaries)==7 and errors==0,'completed study omitted a case or error')
    for item in summaries:
        case_id=item['id'];n=counts[case_id];case=specs[case_id];event=events[case_id]
        frozen=read_json(root,prefix+case_id+'_summary.json')
        require(frozen == item and item.get('completed_replicates') == n
                and item.get('complete') is (n==case['replicates']),
                'case JSON/overall summary/raw replicate count differs')
        unchanged_false(item,['R3_accredited','final_market_power_certified'],'case summary')
        if not n:continue
        truth=[x-case['bases'][0] for x in case['bases'][1:]]
        null_names=[f'rival_{i+1:02d}' for i,t in enumerate(truth) if abs(t)<=1e-12]
        require(item.get('true_null_rivals') == null_names,'false/equality-null labels changed')
        if null_names:
            for key,count_key in [('known_true_null_component_FWER_Holm38','new'),
                                  ('old_basic_control_known_true_null_FWER_Holm38','old'),
                                  ('known_true_null_two_method_intersection_event','intersection')]:
                binomial_rate(item[key],event[count_key],n,case_id+'/'+key)
            tolerance_screen(item['known_true_null_component_FWER_Holm38'],.05,
                             event['unsupported_P1']>0 or event['unsupported_P2']>0,case_id+'/primaryFWER')
            tolerance_screen(item['old_basic_control_known_true_null_FWER_Holm38'],.05,
                             event['unsupported_P2']>0,case_id+'/oldcontrolFWER')
            for method in ['P1','P2']:
                binomial_rate(item['known_true_null_FWER_by_component_within_same_Holm38'][method],
                              event[method],n,case_id+'/'+method)
            paired=item['paired_old_basic_mechanism_diagnostic']
            require(paired.get('old_only_FWER') == event['old_only']
                    and paired.get('new_only_FWER') == event['new_only']
                    and paired.get('causal_explanation_or_superiority_certified') is False,
                    'paired control diagnostic became causal certification or changed counts')
        require(len(item.get('per_rival',[])) == 19,'pointwise coverage family lost slots')
        for j,rival in enumerate(item['per_rival']):
            for index,method in enumerate(['P1','P2']):
                record=rival['methods'][method]['pointwise_CI95_coverage']
                binomial_rate(record,coverage[case_id][j][index],n,case_id+'/'+rival['rival']+'/'+method)
                coverage_screen(record,case_id+'/'+rival['rival']+'/'+method)
        mcs=item['MCS_bootstrap_t_frozen_control']
        binomial_rate(mcs['initial_global_rejection'],event['initial_EPA'],n,case_id+'/initial_EPA')
        require(mcs.get('unsupported_replicates')==event['unsupported_MCS'],
                'unsupported MCS count differs')
        best=sorted((['candidate']+[f'rival_{i:02d}' for i in range(1,20)])[i]
                    for i,value in enumerate(case['bases']) if abs(value-min(case['bases']))<=1e-12)
        require(mcs.get('true_best_models')==best,'MCS population-optimal set labels differ')
        if len(null_names)==19:
            tolerance_screen(mcs['initial_global_rejection'],.10,event['unsupported_MCS']>0,
                             case_id+'/initial_EPA')
        for key,count_key in [('all_true_best_retained','all_best'),('any_true_best_excluded','any_best_excluded'),
                              ('candidate_retained','candidate_retained'),('unique_candidate_set','unique'),
                              ('false_unique_candidate','false_unique')]:
            binomial_rate(mcs[key],event[count_key],n,case_id+'/'+key)
        require(mcs.get('third_pairwise_p_provided') is False
                and item.get('complete_three_method_power_computed') is False
                and item.get('simultaneous_CI95_claimed') is False,
                'MCS/pointwise diagnostics promoted into primary/full-rule claims')


def verify_package(root):
    root=Path(root)
    require(root.is_dir() and not root.is_symlink(), 'packet must be a real directory')
    packet=read_json(root,'PACKET_MANIFEST.json')
    require(packet.get('schema_version') == 1, 'unsupported packet schema')
    names,total=check_entries(root,packet.get('files'))
    require(actual_members(root) == names | {'PACKET_MANIFEST.json'},
            'unexpected or missing packet members')
    descriptors=packet.get('components')
    require(isinstance(descriptors,list) and len(descriptors) == len(REQUIRED_COMPONENTS),
            'five component descriptors required')
    seen=set()
    for descriptor in descriptors:
        name=descriptor.get('name');require(name in REQUIRED_COMPONENTS and name not in seen,
                                             'unexpected/duplicate component')
        seen.add(name);manifest_path=descriptor.get('manifest');safe_path(manifest_path)
        require(manifest_path.startswith(name+'/') and manifest_path in names,
                'component manifest outside its namespace')
        require(descriptor.get('sha256') == digest(root/manifest_path),
                'component manifest hash differs: '+name)
        manifest=read_json(root,manifest_path)
        members,_=check_entries(root,manifest.get('files'),name+'/')
        component_actual={p for p in names if p.startswith(name+'/')}
        require(component_actual == members | {manifest_path},
                'component contains members beyond its frozen allowlist: '+name)
    check_initial(root,packet)
    check_design_gates(root)
    check_methods(root)
    check_guide_candidate(root)
    counts=check_component_semantics(root,packet)
    check_closure(root,packet,counts)
    return {'status':'PASS','files':len(names)+1,'payload_bytes':total,
            'packet_manifest_sha256':digest(root/'PACKET_MANIFEST.json'),
            'new_forecasting_model_fits':0,'new_forecasting_model_forward_passes':0,
            'new_market_predictions':0,'R3_accredited':False,'confirmation_enabled':False,
            **counts}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.report:
        require(not args.report.resolve().is_relative_to(args.packet.resolve()),
                'verification report cannot mutate its packet')
    result=verify_package(args.packet)
    text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if args.report:args.report.write_text(text)
    print(text,end='')


