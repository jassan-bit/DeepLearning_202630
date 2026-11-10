"""Immutable CUDA search admission; standard library only, no fits or data reads."""
from __future__ import annotations
import hashlib,json,math,re
from pathlib import Path
import e2_search_plan as planner

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'book_e2/_static/search_protocol_20261109/protocol.json'
PROTOCOL_SHA256='e0030c990a4afa15665aec254211b0d9e680e9ac5d92dfb8094e43e965152abe'
REGISTRATION=ROOT/'experiments/e2_cuda_search_20261109/registration.json'
VERSION='e2-cuda-search-plan-v1'
REPOSITORY='jassan-bit/DeepLearning_202630'
ORIGINS=tuple('2024-%02d-01'%month for month in range(1,11))+('2025-01-01',)
EXECUTION={'device':'cuda:0','dtype':'float32','amp':False,'tf32':False,'deterministic':True,
           'cpu_threads':2,'interop_threads':1,'gpu_memory_fraction':.8,'gpu_memory_cap_bytes':12*1024**3}

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def positive(value,label):
    if isinstance(value,bool) or type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError(label+' must be positive and finite')
    return float(value)

def protocol():
    if file_sha(PROTOCOL)!=PROTOCOL_SHA256:
        raise ValueError('normalized_registered_protocol_required')
    value=json.loads(PROTOCOL.read_text())
    planner.validate_protocol(value)
    return value

def source_hashes():
    registered={}
    if REGISTRATION.is_file():
        registered=json.loads(REGISTRATION.read_text()).get('implementation_sha256',{})
    if registered:
        if any(not (re.fullmatch(r'scripts/[A-Za-z0-9_]+\.py',name) or name=='requirements-e2-training-gpu.lock.txt') for name in registered):
            raise ValueError('invalid_implementation_manifest_path')
        files=[ROOT/name for name in sorted(registered)]
    else:
        files=sorted((ROOT/'scripts').glob('*.py'))+[ROOT/'requirements-e2-training-gpu.lock.txt']
    return {p.relative_to(ROOT).as_posix():file_sha(p) for p in files}

def selected(values,allowed,label):
    requested=list(allowed if values is None else values)
    if not requested or len(set(requested))!=len(requested) or any(v not in allowed for v in requested):
        raise ValueError('invalid_'+label)
    return [v for v in allowed if v in requested]

def build_plan(*,source_commit,models=None,origins=None,session_hours=8,campaign_hours=24,
               storage_gib=10,device='cuda:0',hardware=None,expected_registration_sha256=None,
               github_branch='campaigns/e2-cuda-selection-v1',github_prefix='campaigns/e2-cuda-selection-v1'):
    if not isinstance(source_commit,str) or not re.fullmatch('[0-9a-f]{40}',source_commit):
        raise ValueError('immutable_source_commit_required')
    if device!='cuda:0':
        raise ValueError('registered_device_is_cuda_0')
    if (not re.fullmatch(r'campaigns/[a-z0-9][a-z0-9._-]{0,79}',github_branch)
            or github_prefix!=github_branch):
        raise ValueError('campaign_branch_and_prefix_must_match')
    p=protocol();profiles=p['models']+p['neural_reference_profiles']
    names=[m['model'] for m in profiles]
    names=selected(models,names,'models');origins=selected(origins,list(ORIGINS),'origins')
    session=positive(session_hours,'session_hours')*3600
    campaign=positive(campaign_hours,'campaign_hours')*3600
    storage=int(positive(storage_gib,'storage_gib')*1024**3)
    if campaign<session:raise ValueError('campaign_budget_below_session_budget')
    if storage<2*1024**2:raise ValueError('storage_budget_too_small')
    registration_sha=file_sha(REGISTRATION)
    if expected_registration_sha256 is not None and registration_sha!=expected_registration_sha256:
        raise ValueError('registration_hash_mismatch')
    reg=json.loads(REGISTRATION.read_text())
    if (reg.get('version')!='e2-cuda-search-admission-v1' or reg.get('protocol_sha256')!=PROTOCOL_SHA256
            or reg.get('repository')!=REPOSITORY):
        raise ValueError('invalid_cuda_search_admission')
    if (session!=reg['initial_budget']['session_seconds'] or campaign!=reg['initial_budget']['campaign_seconds']):
        raise ValueError('initial_budget_must_match_published_admission_use_explicit_extension')
    sources=source_hashes()
    if reg.get('implementation_sha256') and sources!=reg['implementation_sha256']:
        raise ValueError('implementation_hash_mismatch')
    domains={}
    for name in names:
        profile=next(m for m in profiles if m['model']==name)
        trials=planner.generate_hpo(p,name)
        architectures=planner.enumerate_architectures(p,name) if profile['nas'] else []
        domains[name]={'hpo':digest(trials),'nas':digest(architectures)}
    return {'version':VERSION,'protocol_sha256':PROTOCOL_SHA256,'registration_sha256':registration_sha,
        'source_commit':source_commit,'source_sha256':sources,'domain_sha256':domains,'models':names,
        'remote_storage':{'repository':REPOSITORY,'branch':github_branch,'prefix':github_prefix},
        'origins':origins,'k':40,'seed_pools':{'hpo':p['common_hpo']['optimization_seeds'],
            'nas1':[p['nas']['phase1_seed']],'nas2':p['nas']['phase2_seeds']},
        'execution':dict(EXECUTION),'checkpoint_policy':'selection_metadata_only_refit_separate',
        'scope':'registered_neural_selection_only',
        'coverage':{'selected_models':len(names),'selected_origins':len(origins),'full_registry_models':14,
            'full_registry_origins':11,'partial_models':len(names)!=14,'partial_origins':len(origins)!=11,
            'final_refit_included':False,'classical_models_included':False,'final_market_values_read':False},
        'hardware_binding':hardware,'initial_controls':{'session_budget_seconds':session,
            'campaign_budget_seconds':campaign,'storage_bytes':storage},
        'stage_caps':{'short_epochs':50,'full_epochs':200,'patience':20,'short_seconds':1800,'full_seconds':7200}}

def plan_summary(plan):
    p=protocol();nas={m['model'] for m in p['models'] if m['nas']}
    n=len(plan['origins']);searched=sum(name in nas for name in plan['models'])
    hpo=len(plan['models'])*18*2*n
    nas1=searched*1000*n;nas2=searched*41*5*n
    return {'version':VERSION,'plan_sha256':digest(plan),'models':plan['models'],'origins':plan['origins'],
        'hpo_attempts':hpo,'nas1_attempts':nas1,'nas2_attempts_upper_bound':nas2,
        'selection_attempts_upper_bound':hpo+nas1+nas2,'nas_finalists':'40 or 41 including reference',
        'session_hours':plan['initial_controls']['session_budget_seconds']/3600,
        'initial_campaign_hours':plan['initial_controls']['campaign_budget_seconds']/3600,
        'storage_gib':plan['initial_controls']['storage_bytes']/1024**3,
        'checkpoint_policy':plan['checkpoint_policy'],'scope':plan['scope'],'coverage':plan['coverage'],
        'scientific_HPO_or_NAS_executed':False,'final_market_values_read':False,
        'final_R':None,'full_campaign_wall_hours':None,
        'budget_interpretation':'Operational execution ceilings, not a full-campaign duration forecast. Pending jobs prevent stage winners.'}
