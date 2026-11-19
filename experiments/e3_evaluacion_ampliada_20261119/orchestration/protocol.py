"""Bind an executable, predeclared audit to admitted bytes and an annotated tag.

This is an exploratory multi-dataset audit. It is not a claim of R3 power,
novelty, blind BTC evaluation, or superiority. Prepared with declared AI help.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


CORE_FAMILY = ['FCN-1D','ResNet-1D','Inception-1D',
    'LSTM-autoencoder con predictor','TimesNet','TSMixer','TiDE','TimeMixer',
    'PatchTST','iTransformer','MLP16','LSTM32','GRU32','Persistence',
    'SeasonalNaive','ARIMA','CNN-LSTM-atencion-aditiva E3']
FAMILY = CORE_FAMILY + ['floor_only','floor_plus_mean_excess']
CANDIDATE = 'CNN-LSTM-atencion-aditiva E3'
NPZ_SHA = '186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'
AUXILIARY_SHA = ['bf6bb5160af924aef6a15c3684aeae932df3017e8639888bb7e91ce787900171',
                 '9563ef557e180f34ee37dce3b1480c38aaefd7063df21c031644973310e28deb']
TAG_PREFIX = 'e3-extended-audit-protocol-'


def canonical(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def source_sha(source):
    require(isinstance(source,str) and re.fullmatch('[0-9a-f]{40}',source) is not None,
            'A full immutable source SHA is required')
    return source


def build(admission_path,source_commit):
    """Read admission receipts and file hashes, never reserved target labels."""
    source_sha(source_commit)
    path=Path(admission_path)
    report=json.loads(path.read_bytes())
    require(report.get('kind')=='e3_additional_OHLCV_admission'
            and report.get('datasets_required')==3 and report.get('datasets_admitted')==3,
            'Three complete physical file admissions are required before the audit')
    require(report.get('datasets_evaluated_by_models')==0
            and report.get('target_values_computed') is False,
            'Admission must precede model evaluation and reserved-target construction')
    rows=report.get('datasets',[])
    require([r.get('id') for r in rows]==['D01','D02','D03'],
            'The complete D01/D02/D03 inventory is required')
    hashes={'D00':NPZ_SHA}
    provenance={}
    for row in rows:
        identity=row['id'];relative='datasets/'+identity+'.csv';file=path.parent/relative
        require(row.get('admitted') is True and row.get('path')==relative
                and row.get('target_values_computed') is False,
                'Each additional dataset must be physically admitted')
        require(file.is_file() and not file.is_symlink(),'A regular admitted CSV is required')
        raw=file.read_bytes()
        require(len(raw)==row.get('bytes') and digest(raw)==row.get('sha256'),
                'Admitted file bytes changed')
        hashes[identity]=digest(raw)
        provenance[identity]={k:row[k] for k in ('rows','clock','first','last','license','volume_units') if k in row}
    return {'schema_version':1,'kind':'e3_extended_evaluation_protocol',
        'source_commit':source_commit,'family':list(FAMILY),'candidate':CANDIDATE,
        'analytic_controls':['floor_only','floor_plus_mean_excess'],
        'seeds':list(range(20261118,20261128)),'hpo_seed':20261117,
        'datasets':hashes,'dataset_provenance':provenance,
        'auxiliary_sources':list(AUXILIARY_SHA),
        'admission_receipt_sha256':digest(path.read_bytes()),
        'admission_source_commit':report['source_commit'],
        'split':{'ratios':[.55,.15,.10],'purge':7,
                 'D00':'original532fit_116validation_53calibration_608known_test'},
        'fit_diagnostic_burn_in':7,
        'fit_diagnostic_mask':'burnin7_and_classical_lag_availability',
        'training':{'lr_grid':[.0003,.001,.003],'max_epochs':40,'patience':8,
                    'batch_size':32,'gradient_clip':1.0,'per_fit_seconds':120.0,
                    'optimizer':'AdamW','weight_decay':0.0,
                    'SAE_three_phases_share_one_fit_budget':True,
                    'training_loss':'raw normalized sigma MSE',
                    'epoch_and_HPO_selection':'common projected validation sigma MSE'},
        'projection':{'policy':'max(raw_prediction,sqrt(A_available_input))',
                      'applies_to':'all nineteen systems, HPO and final panels',
                      'keep_raw_prediction':True},
        'information_contract':{'neural':'seven past OHLCV observations',
                                 'classical':'causal history of past sigma labels; broader than seven bars',
                                 'identical_information_set_claimed':False},
        'panel_layout':'one dataset/model CSV, immutable hash manifest; no monolithic panel',
        'hpo_sensitivity':{'models':'all fourteen neural models','seed':20261117,
                           'trials':'all three fixed learning rates',
                           'test_access':'only after validation winner is durably recorded',
                           'test_result_changes_winner':False,'additional_optimizer_fits':0},
        'interval':{'alpha':.10,'calibration':'absolute residual quantile on unperturbed calibration only'},
        'interval_alpha':.10,
        'analysis':{'block_lengths':[7,14,28],'bootstrap_repetitions':9999,
                    'bootstrap_seed':2026111902,
                    'scenarios':['unperturbed','ohlcv_observation_noise_1',
                                 'ohlcv_observation_noise_2','ohlcv_observation_noise_3'],
                    'perturbation_severities':[.25,.5,1.0],
                    'target_under_perturbation':'original clean sigma7',
                    'calibrator_refitted_under_stress':False,
                    'between_dataset_rank_inference':'descriptive; dependent financial domains'},
        'learning_curves':{'models':[CANDIDATE,'TSMixer'],'fractions':[.25,.5,1.0],
                           'seeds':list(range(20261118,20261121)),
                           'subset':'nested chronological prefixes of fit only',
                           'hyperparameters':'fixed winner from full fit/validation HPO',
                           'full_fraction_reuses_final_fit':True,
                           'calendar_span_confounding_disclosed':True},
        'evidence_mode':'prespecified_exploratory_audit',
        'planning_status':'not_dimensioned_for_full_temporal_multidataset_battery',
        'R1_accredited':False,'R3_accredited':False,
        'globally_blind_market_calendar_claimed':False,
        'additional_validation_use':'automated prefixed HPO only; no mechanism redesign',
        'failure_policy':'retain failed jobs and exact common test dates; no substitute model or test-row deletion',
        'domain_dependence':report.get('dependence_and_exposure',[])}


def freeze(protocol_path,source_commit,master_run_id,token,api):
    """Create one immutable annotated protocol tag before worker model fits."""
    source_sha(source_commit)
    require(re.fullmatch('[0-9]{8}T[0-9]{12}Z',master_run_id or '') is not None,
            'A full UTC master run ID is required')
    raw=Path(protocol_path).read_bytes();protocol=json.loads(raw)
    require(raw==canonical(protocol),'Protocol must use canonical bytes')
    require(protocol.get('kind')=='e3_extended_evaluation_protocol'
            and protocol.get('source_commit')==source_commit
            and protocol.get('family')==FAMILY
            and protocol.get('evidence_mode')=='prespecified_exploratory_audit'
            and protocol.get('R3_accredited') is False
            and protocol.get('compute_profile',{}).get('device')=='cuda:0',
            'Unexpected protocol scope or missing GPU cohort')
    tag=TAG_PREFIX+master_run_id
    record={'kind':'e3_extended_audit_protocol_tag','source_commit':source_commit,
            'protocol_sha256':digest(raw),'master_run_id':master_run_id,
            'protocol':protocol,'scope':'prespecified exploratory audit; R3 not accredited'}
    message=canonical(record).decode()
    root='/repos/'+api.REPOSITORY
    ref_path=root+'/git/ref/tags/'+tag
    existing=api._request('GET',ref_path,token,allow_missing=True)
    if existing is None:
        obj=api._request('POST',root+'/git/tags',token,
            {'tag':tag,'message':message,'object':source_commit,'type':'commit'})
        require(re.fullmatch('[0-9a-f]{40}',obj.get('sha','')) is not None,
                'An annotated tag object SHA is required')
        try:
            api._request('POST',root+'/git/refs',token,{'ref':'refs/tags/'+tag,'sha':obj['sha']})
        except api.StorageError:
            if api._request('GET',ref_path,token,allow_missing=True) is None:
                raise
    ref=api._request('GET',ref_path,token)
    require(ref.get('ref')=='refs/tags/'+tag and ref.get('object',{}).get('type')=='tag',
            'An annotated protocol tag is required, never a lightweight tag')
    obj=api._request('GET',root+'/git/tags/'+ref['object']['sha'],token)
    require(obj.get('sha')==ref['object']['sha'] and obj.get('tag')==tag
            and obj.get('object',{}).get('type')=='commit'
            and obj['object'].get('sha')==source_commit and obj.get('message')==message,
            'Existing protocol tag binds different immutable code or data')
    return {'kind':'e3_extended_audit_freeze_receipt','verified':True,
            'source_commit':source_commit,'protocol_sha256':digest(raw),
            'datasets':dict(protocol['datasets']),
            'annotated_tag':tag,'tag_object_sha':obj['sha'],
            'tagger_date':obj.get('tagger',{}).get('date'),
            'tag_url':'https://github.com/'+api.REPOSITORY+'/tree/'+tag,
            'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'targets_not_previously_evaluated_claimed':False,
            'R3_accredited':False,'mode':'prespecified_exploratory_audit'}


def verify_receipt(protocol_path,freeze_receipt,source_commit):
    """Verify local bindings to the tag already checked by the controller API.

    This does not turn a JSON receipt into a cryptographic server attestation;
    the public tag object is independently queryable by its immutable SHA.
    """
    source_sha(source_commit)
    raw=Path(protocol_path).read_bytes();p=json.loads(raw)
    f=freeze_receipt if isinstance(freeze_receipt,dict) else json.loads(Path(freeze_receipt).read_bytes())
    require(raw==canonical(p) and p.get('source_commit')==source_commit
            and p.get('family')==FAMILY,'Protocol bytes, family or source changed')
    require(f.get('kind')=='e3_extended_audit_freeze_receipt' and f.get('verified') is True
            and f.get('source_commit')==source_commit and f.get('protocol_sha256')==digest(raw)
            and f.get('R3_accredited') is False and f.get('mode')=='prespecified_exploratory_audit'
            and re.fullmatch('[0-9a-f]{40}',f.get('tag_object_sha','')) is not None
            and re.fullmatch(TAG_PREFIX+'[0-9]{8}T[0-9]{12}Z',f.get('annotated_tag','')) is not None,
            'An unchanged annotated audit tag receipt is required')
    return f


def compute_profile():
    """Collect the hardware/software cohort for wall-time fitting caps."""
    import subprocess
    import torch
    require(torch.cuda.is_available(),'This audit requires an available Colab CUDA GPU')
    props=torch.cuda.get_device_properties(0)
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version',
                                    '--format=csv,noheader'],text=True,timeout=15).splitlines()[0].strip()
    return {'device':'cuda:0','GPU':props.name,'total_memory_bytes':int(props.total_memory),
            'capability':list(torch.cuda.get_device_capability(0)),
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'driver':driver}


def verify_compute_profile(protocol,actual=None):
    expected=protocol.get('compute_profile')
    require(isinstance(expected,dict) and expected.get('device')=='cuda:0',
            'A frozen GPU compute profile is required for the common time budget')
    observed=compute_profile() if actual is None else actual
    require(observed==expected,
            'GPU/software cohort changed: preserve this run and select the same GPU/runtime before resuming')
    return observed
