"""Read-only fixed-state development measurements, never training or forecasts.

The CLI preflight loads authenticated checkpoints without model forwards.
Execution requires a public plan/code anchor and writes only a fresh output case.
AI assistance declared. No claim of R1/R2/R3 or causal generalization.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import time

VERSION='e3-fixed-state-gradients-1'
RECIPES=('sigma_crps_v4','u_crps_control','mse_only_control')
SEEDS=tuple(range(20261121,20261126))
BATCHES=(('fit_first32','train',0,32),('fit_middle32','train',250,282),
         ('fit_last32','train',500,532),('selection_first27','selection',0,27),
         ('selection_last27','selection',27,54))
MAX_FORWARD=150
MAX_POINTS=4500
MAX_AUTOGRAD=300
MAX_SECONDS=300
MAX_RSS_BYTES=2*1024**3

def require(ok,message):
    if not ok: raise ValueError(message)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(d):return hashlib.sha256(canonical(d)).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def file_record(p):
    p=Path(p);require(p.is_file() and not p.is_symlink(),'Regular file required')
    return {'bytes':p.stat().st_size,'sha256':sha(p)}

def gradient_stats(gm,ga,lam,indices,torch):
    def flat(gs):return torch.cat([(torch.zeros_like(gm[i] if gm[i] is not None else ga[i]) if gs[i] is None else gs[i]).detach().reshape(-1).to(torch.float64) for i in indices])
    # A branch-None uses the other branch's tensor shape. Both-None is unsupported
    # by this fixed-model scope and is explicitly rejected by execute().
    a=flat(gm);b=flat(ga)
    nm=float(torch.linalg.vector_norm(a));na=float(torch.linalg.vector_norm(b));dot=float(torch.dot(a,b))
    cosine=dot/(nm*na) if nm>0 and na>0 else None
    return {'MSE_gradient_norm':nm,'aux_unweighted_gradient_norm':na,'aux_effective_gradient_norm':lam*na,
            'dot_MSE_aux_unweighted':dot,'cosine_unweighted':cosine,
            'cosine_actual_auxiliary':cosine if lam>0 else None,
            'actual_auxiliary_supported':lam>0 and cosine is not None,
            'unweighted_aux_is_counterfactual':lam==0,
            'MSE_unused_tensors':sum(gm[i] is None for i in indices),'aux_unused_tensors':sum(ga[i] is None for i in indices)}

def combine(gm,ga,lam):
    result=[]
    for m,a in zip(gm,ga):
        if lam==0:result.append(None if m is None else m.detach().clone())
        elif m is None and a is None:result.append(None)
        elif m is None:result.append(a.detach()*lam)
        elif a is None:result.append(m.detach().clone())
        else:result.append(m.detach()+lam*a.detach())
    return result

def adam_direction(grads,params,groups,states,torch,max_norm=1.):
    """Analytical hypothetical AdamW direction in clones; no optimizer object/step."""
    norms=[torch.linalg.vector_norm(g.detach(),2) for g in grads if g is not None]
    # Stack promotes mixed native dtypes as Torch clipping does; never hide overflow.
    total_tensor=torch.linalg.vector_norm(torch.stack(norms),2) if norms else torch.tensor(0.)
    require(bool(torch.isfinite(total_tensor)),'Nonfinite global gradient norm')
    coefficient_tensor=torch.clamp(max_norm/(total_tensor+1e-6),max=1.)
    total=float(total_tensor);coefficient=float(coefficient_tensor)
    delta=[torch.zeros_like(p) for p in params]
    seen=[];steps=[]
    for group in groups:
        require(group['weight_decay']==0 and not group['amsgrad'] and not group['maximize'],
                'Only saved weight_decay0, amsgradFalse, maximizeFalse supported')
        require(not group.get('capturable',False) and not group.get('differentiable',False),'Unsupported optimizer mode')
        b1,b2=group['betas'];lr=group['lr'];eps=group['eps']
        for idx in group['params']:
            require(type(idx) is int and 0<=idx<len(params),'Optimizer index differs from frozen natural parameter order')
            require(idx not in seen,'Duplicated optimizer index');seen.append(idx)
            g=grads[idx]
            if g is None:continue
            require(g.shape==params[idx].shape and g.dtype==params[idx].dtype,'Gradient/parameter dtype or shape differs')
            s=states[idx];require(set(s)>={'step','exp_avg','exp_avg_sq'},'Adam state incomplete')
            t=float(s['step'].item() if torch.is_tensor(s['step']) else s['step'])
            require(t>=0 and t.is_integer(),'Noninteger parameter step')
            for key in ['exp_avg','exp_avg_sq']:
                require(s[key].shape==params[idx].shape and s[key].dtype==params[idx].dtype,'Moment/parameter alignment differs')
                require(bool(torch.isfinite(s[key]).all()),'Nonfinite stored moment')
            require(bool((s['exp_avg_sq']>=0).all()),'Negative second moment')
            clipped=g.detach()*coefficient_tensor
            m=s['exp_avg'].detach().clone().lerp_(clipped,1-b1)
            v=s['exp_avg_sq'].detach().clone().mul_(b2).addcmul_(clipped,clipped,value=1-b2)
            denominator=v.sqrt().div_(math.sqrt(1-b2**(t+1))).add_(eps)
            delta[idx]=m.div(denominator).mul_(-lr/(1-b1**(t+1)))
            require(bool(torch.isfinite(delta[idx]).all()),'Nonfinite hypothetical displacement');steps.append(t)
    require(seen==list(range(len(params))),'Frozen model.parameters() optimizer order differs')
    return delta,{'global_gradient_norm_before_clip':total,'clip_coefficient':coefficient,
                  'active_tensors':sum(g is not None for g in grads),'omitted_None_tensors':sum(g is None for g in grads),
                  'parameter_step_min':min(steps) if steps else None,'parameter_step_max':max(steps) if steps else None}

def update_stats(gm,full,mse,params,indices,torch):
    def flatten(gs):return torch.cat([(torch.zeros_like(params[i]) if gs[i] is None else gs[i]).detach().reshape(-1).to(torch.float64) for i in indices])
    g=flatten(gm);a=flatten(full);b=flatten(mse);p=flatten(params)
    na=float(torch.linalg.vector_norm(a));nb=float(torch.linalg.vector_norm(b));pn=float(torch.linalg.vector_norm(p))
    return {'MSE_dot_Adam_full_displacement':float(torch.dot(g,a)),
            'MSE_dot_Adam_MSE_counterfactual_displacement':float(torch.dot(g,b)),
            'MSE_dot_displacement_difference':float(torch.dot(g,a-b)),
            'Adam_full_displacement_norm':na,'Adam_MSE_counterfactual_displacement_norm':nb,
            'Adam_displacement_difference_norm':float(torch.linalg.vector_norm(a-b)),
            'Adam_full_to_parameter_norm_ratio':na/pn if pn>0 else None,
            'Adam_direction_is_hypothetical_not_historical_step':True}

def state_equal(a,b,torch):
    return list(a)==list(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and torch.equal(a[k],b[k]) for k in a)

def safe_load(path,torch,np):
    allowed_names={'numpy.dtype','numpy.ndarray','numpy._core.multiarray._reconstruct'}
    require(set(torch.serialization.get_unsafe_globals_in_checkpoint(path))<=allowed_names,'Unexpected checkpoint globals')
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.UInt32DType]):
        return torch.load(path,map_location='cpu',weights_only=True)

def load_inputs(source,case,output):
    driver=module('e3_fixed_state_original_driver',source/'cpu_pilot_driver.py')
    source_manifest=source/'pilot_source_manifest.json'
    manifest_hash=driver.verify_sources(source,source_manifest)
    npz=driver.validate_npz(case/'inputs/current_cohort_arrays.npz')
    runtime=driver.scientific_runtime(source);np,torch=runtime[:2]
    observed_keys=[];original_np_load=np.load
    class GuardedNPZ:
        def __init__(self,inner):self.inner=inner
        @property
        def files(self):return self.inner.files
        def __getitem__(self,key):
            require(key in ['training_raw','training_y_original','training_dates'],'Forbidden NPZ array access')
            observed_keys.append(key);return self.inner[key]
        def __enter__(self):return self
        def __exit__(self,*args):self.inner.close()
    def guarded_np_load(path,*aa,**kk):
        require(Path(path).resolve()==npz.resolve() and kk.get('allow_pickle') is False,'Unexpected NPZ source or pickle mode')
        return GuardedNPZ(original_np_load(path,*aa,**kk))
    np.load=guarded_np_load
    try:data,split,admission=driver.admit(runtime,npz)
    finally:np.load=original_np_load
    require(observed_keys==['training_raw','training_y_original','training_dates'],'NPZ access audit differs')
    original_plan=read(case/'cpu_pilot_plan.json')
    require(driver.digest(original_plan)==driver.digest(read(source/'cpu_pilot_results/cpu_pilot_plan.json')),'Private/public pilot plans differ')
    require(manifest_hash==original_plan['source_manifest_sha256'],'Source manifest differs from original plan')
    require(runtime[7]==original_plan['cohort'],'Declared recorded CPU cohort fields differ')
    require(driver.digest(admission)==original_plan['admission_sha256'],'Admission differs from prior case')
    public_hashes=read(source/'cpu_pilot_results/LOCAL_CHECKPOINT_HASHES_ONLY.json')
    lookup={x['job']['job_id']:x for x in public_hashes['checkpoints']}
    loaded=[];input_hashes={'NPZ':file_record(npz),'source_manifest':file_record(source_manifest)}
    for seed in SEEDS:
        for recipe in RECIPES:
            jobid=f'seed{seed}/{recipe}';info=lookup[jobid]
            p=driver.no_symlink_path(case/info['local_private_relative_path'])
            require(file_record(p)=={'bytes':info['bytes'],'sha256':info['sha256']},'Checkpoint file differs')
            receipt=read(p.with_name('receipt.json'));marker=p.with_name('phase_started.json')
            s=safe_load(p,torch,np)
            require(s['completed'] and s['steps']>0 and s['best_score'] is not None,'Uncompleted/untrained checkpoint')
            require(s['request_sha256']==receipt['request_sha256'] and s['phase_marker_sha256']==sha(marker),'Request/phase marker differs')
            require(s['epoch']==receipt['epochs'] and s['best_epoch']==receipt['best_epoch'],'Saved epoch identity differs')
            model=driver.factory(runtime,source,recipe,seed,data)
            model.eval();named=list(model.named_parameters());params=[p for _,p in named]
            require(sum(p.numel() for p in params)==1262 and len(params)==12,'Model parameter identity differs')
            for snapshot in ['best','latest']:
                require(list(s[snapshot])==list(model.state_dict()),'State tensor names/order differ')
                for name,t in s[snapshot].items():
                    expected=model.state_dict()[name]
                    require(t.dtype==expected.dtype and t.shape==expected.shape and bool(torch.isfinite(t).all()),'State dtype/shape/finite differs')
            model.load_state_dict(s['latest'],strict=True)
            ids=[i for group in s['optimizer']['param_groups'] for i in group['params']]
            require(ids==list(range(len(params))),'Optimizer natural ordering differs')
            for idx,p0 in enumerate(params):
                for key in ['exp_avg','exp_avg_sq']:
                    v=s['optimizer']['state'][idx][key]
                    require(v.shape==p0.shape and v.dtype==p0.dtype,'Optimizer tensor differs')
            shared=[i for i,(name,_) in enumerate(named) if name.startswith('encoder.') or name in ['query_raw','temperature_raw']]
            head=[i for i in range(len(named)) if i not in shared]
            require(sum(params[i].numel() for i in shared)==1197 and sum(params[i].numel() for i in head)==65,'Parameter partitions differ')
            loaded.append({'jobid':jobid,'recipe':recipe,'seed':seed,'state':s,'model':model,'named':named,
                'groups':{'all':list(range(len(params))),'shared':shared,'head':head},
                'best_latest_equal':state_equal(s['best'],s['latest'],torch),'checkpoint_path':p})
            input_hashes[jobid]=file_record(p)
    batches=[];select=split['select']
    for label,splitname,start,end in BATCHES:
        if splitname=='train':
            ii=np.arange(start,end);prefix='train'
        else:ii=select[start:end];prefix='val'
        dates=data[prefix+'_dates'][ii].tolist()
        require(not set(dates)&set(admission['assessment_dates']),'Assessment date entered measurement batch')
        require(len(dates)==(end-start),'Batch length differs')
        batches.append({'label':label,'split':splitname,'dates':dates,
            'X':torch.as_tensor(data[prefix+'_X'][ii],dtype=torch.float32),
            'A':torch.as_tensor(data[prefix+'_A'][ii],dtype=torch.float64),
            'Y':torch.as_tensor(data[prefix+'_y_original'][ii],dtype=torch.float64),
            'U':torch.as_tensor(data[prefix+'_U'][ii],dtype=torch.float64)})
    metadata={'source_manifest_sha256':manifest_hash,'runtime':runtime[7],
        'historical_runtime':original_plan['cohort'],'runtime_equals_complete_historical_record':runtime[7]==original_plan['cohort'],
        'matching_primary_versions_only':'Python3.12.14/Torch2.8.0+cpu/NumPy2.3.5; platform and ancillary packages separately recorded',
        'admission_digest':driver.digest(admission),'NPZ_keys_read':admission['keys_read'],
        'NPZ_array_accesses_observed':observed_keys,
        'forbidden_array_families_loaded':False,'assessment_dates_measured':False,
        'input_files':input_hashes,'batches':[{k:v for k,v in x.items() if k in ['label','split','dates']} for x in batches],
        'checkpoints':[{'job_id':x['jobid'],'epoch':x['state']['epoch'],'best_epoch':x['state']['best_epoch'],
            'best_latest_equal':x['best_latest_equal'],
            'parameter_order':[{'index':i,'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'elements':p.numel()} for i,(n,p) in enumerate(x['named'])],
            'optimizer_groups':[{k:v for k,v in g.items() if k!='params'} for g in x['state']['optimizer']['param_groups']]} for x in loaded],
        'safe_checkpoint_loading':'weights_onlyTrue with static NumPy reconstruction allowlist; hashes checked first',
        'new_training_jobs':0,'new_forward_calls':0,'new_autograd_calls':0,'AI_assistance_declared':True}
    return driver,runtime,loaded,batches,metadata

def max_rss_bytes():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

def execute(args,plan):
    output=args.output;require(not output.exists(),'Output case already exists; preserve it')
    output.mkdir(parents=True)
    start=time.monotonic();counts={'forward_attempts':0,'forward_completed':0,'internal_point_evaluations':0,
        'autograd_attempts':0,'autograd_completed':0,'optimizer_steps':0,'training_jobs':0,
        'new_out_of_sample_market_forecasts':0,'new_MC_replicates':0,'new_statistical_resamples':0,'p_values':0}
    def alarm(signum,frame):raise TimeoutError('Frozen300second budget reached')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    status='failed';error=None;metadata=None;allrows=[];permutations=[]
    try:
        anchor=read(args.anchor)
        require(anchor['public_readback_verified'] and anchor['code_sha256']==sha(__file__) and anchor['plan_sha256']==sha(args.plan),
                'Public plan/code binding required before measurement')
        require(anchor['plan_id']==plan['plan_id'] and anchor['commit'],'Anchor identity differs')
        driver,runtime,loaded,batches,metadata=load_inputs(args.source_root,args.case_root,output)
        np,torch=runtime[:2]
        write(output/'DATA_AND_STATE_PREFLIGHT.json',metadata)
        for job in loaded:
            model=job['model'];named=job['named'];params=[p for _,p in named];s=job['state']
            configurations=model.configuration();lam=float(configurations['auxiliary_coefficient'])
            require(lam==(0. if job['recipe']=='mse_only_control' else .1),'Effective coefficient differs')
            # No deduplication: both labeled snapshots are measured, even if equal.
            for snapshot in ['best','latest']:
                model.load_state_dict(s[snapshot],strict=True);model.eval()
                before={k:v.detach().clone() for k,v in model.state_dict().items()}
                epoch=s['best_epoch'] if snapshot=='best' else s['epoch']
                for batch in batches:
                    require(max_rss_bytes()<=MAX_RSS_BYTES,'RSS ceiling exceeded')
                    require(counts['forward_attempts']<MAX_FORWARD,'Forward cap reached')
                    counts['forward_attempts']+=1
                    outputs=model.forward_components(batch['X'],batch['A'])
                    counts['forward_completed']+=1;counts['internal_point_evaluations']+=len(batch['Y'])
                    require(counts['internal_point_evaluations']<=MAX_POINTS,'Point-evaluation cap exceeded')
                    loss=model.loss_components(outputs,batch['Y'],batch['U'])
                    mse=loss['mse_scaled']
                    aux=loss['crps_sigma'] if job['recipe']=='sigma_crps_v4' else loss['crps_u']
                    require(bool(torch.allclose(loss['loss'],mse+lam*aux,rtol=1e-12,atol=1e-15)),'Loss/normalization identity differs')
                    counts['autograd_attempts']+=1
                    gm=torch.autograd.grad(mse,params,retain_graph=True,allow_unused=True)
                    counts['autograd_completed']+=1;counts['autograd_attempts']+=1
                    ga=torch.autograd.grad(aux,params,retain_graph=False,allow_unused=True)
                    counts['autograd_completed']+=1
                    require(counts['autograd_attempts']<=MAX_AUTOGRAD,'Autograd cap exceeded')
                    for m,a,p in zip(gm,ga,params):
                        require(m is not None or a is not None,'Both gradient branches unused; unsupported declared parameter')
                        require(all(g is None or bool(torch.isfinite(g).all()) for g in [m,a]),'Nonfinite gradient')
                    probabilities=outputs['probabilities'].detach();atoms=outputs['atoms'].detach()
                    sig=outputs['sigma_atoms'].detach();pred=outputs['prediction'].detach()
                    require(bool((atoms>=0).all() and (atoms[:,1:]>=atoms[:,:-1]).all()),'Invalid ordered atoms')
                    require(bool(torch.isfinite(probabilities).all()) and bool((probabilities>=0).all()),'Invalid probabilities')
                    require(bool(torch.allclose(probabilities.sum(-1),torch.ones(len(pred),dtype=probabilities.dtype),rtol=1e-12,atol=1e-12)),'Masses do not sum1')
                    require(bool(torch.allclose((probabilities*sig).sum(-1),pred,rtol=1e-12,atol=1e-12)),'Output-law mean identity differs')
                    floor_margin=float((pred-torch.sqrt(batch['A'])).min())
                    require(floor_margin>=-1e-12,'Support floor violated')
                    entropy=float((-torch.special.xlogy(probabilities,probabilities).sum(-1)).mean())
                    require(math.isfinite(entropy),'Nonfinite probability entropy')
                    total=combine(gm,ga,lam)
                    dfull=dmse=None;clipfull=clipmse={}
                    if snapshot=='latest':
                        dfull,clipfull=adam_direction(total,params,s['optimizer']['param_groups'],s['optimizer']['state'],torch)
                        dmse,clipmse=adam_direction(combine(gm,ga,0.),params,s['optimizer']['param_groups'],s['optimizer']['state'],torch)
                    common={'job_id':job['jobid'],'recipe':job['recipe'],'seed':job['seed'],'snapshot':snapshot,'epoch':epoch,
                        'best_latest_equal':job['best_latest_equal'],'batch':batch['label'],'split':batch['split'],'n_examples':len(pred),
                        'lambda_actual':lam,'auxiliary_coordinate':'sigma' if job['recipe']=='sigma_crps_v4' else 'U',
                        'MSE_scaled':float(mse.detach()),'CRPS_aux_unweighted_scaled':float(aux.detach()),'actual_total_loss':float(loss['loss'].detach()),
                        'CRPS_U_scaled':float(loss['crps_u'].detach()),'CRPS_sigma_scaled':float(loss['crps_sigma'].detach()),
                        'min_atom_U':float(atoms.min()),'min_probability':float(probabilities.min()),'max_probability':float(probabilities.max()),
                        'mean_probability_entropy':entropy,
                        'min_prediction_minus_floor':floor_margin,'mean_atom_range_U':float((atoms[:,-1]-atoms[:,0]).mean()),
                        'Adam_optimizer_matched_to_snapshot':snapshot=='latest',
                        'Adam_full_clip':clipfull.get('clip_coefficient'),'Adam_MSE_clip':clipmse.get('clip_coefficient')}
                    for label,indices in job['groups'].items():
                        row={**common,'parameter_partition':label,**gradient_stats(gm,ga,lam,indices,torch)}
                        if dfull is not None:row.update(update_stats(gm,dfull,dmse,params,indices,torch))
                        allrows.append(row)
                    require(state_equal(before,model.state_dict(),torch),'Model state mutated')
                    require(all(p.grad is None for p in params),'Parameter .grad buffer unexpectedly written')
                # Loaded states/optimizer are never written, restored, resumed or stepped.
        require(counts['forward_completed']==150 and counts['internal_point_evaluations']==4500 and counts['autograd_completed']==300,'Incomplete measurement counters')
        for job in loaded:
            require(file_record(job['checkpoint_path'])==metadata['input_files'][job['jobid']],'Checkpoint bytes changed')
        require(file_record(args.case_root/'inputs/current_cohort_arrays.npz')==metadata['input_files']['NPZ'],'NPZ bytes changed')
        require(driver.verify_sources(args.source_root,args.source_root/'pilot_source_manifest.json')==metadata['source_manifest_sha256'],'Source changed')
        for item in plan['bound_source_files']:
            require(file_record(args.source_root/item['path'])=={'bytes':item['bytes'],'sha256':item['sha256']},'Bound source changed during measurement')
        require(sha(__file__)==plan['code_sha256'],'Measurement code changed during execution')
        require(max_rss_bytes()<=MAX_RSS_BYTES,'RSS ceiling exceeded')
        write(output/'MEASUREMENTS.json',{'rows':allrows,'scope':'Fixed-state development geometry only','AI_assistance_declared':True,'R3_accredited':False})
        keys=list(dict.fromkeys(k for row in allrows for k in row))
        with (output/'measurements.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(allrows)
        status='completed'
    except Exception as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
        if allrows:write(output/'PARTIAL_MEASUREMENTS.json',{'rows':allrows,'incomplete':True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        receipt={'status':status,'error':error,'version':VERSION,'counts':counts,'rows':len(allrows),
            'elapsed_seconds':time.monotonic()-start,'max_RSS_bytes':max_rss_bytes(),
            'completed_at_UTC':datetime.now(timezone.utc).isoformat(),'code_sha256':sha(__file__),'plan_sha256':sha(args.plan),
            'old_checkpoints_or_admission_files_written':False,'optimizer_step_called':False,
            'fixed_model_parameters_unchanged_during_measurements':status=='completed',
            'historical_campaign_repeated':False,'new_candidate_selected':False,'R3_accredited':False,'AI_assistance_declared':True}
        write(output/'EXECUTION_RECEIPT.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False))

def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['preflight','execute']);a.add_argument('--source-root',type=Path,required=True)
    a.add_argument('--case-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--plan',type=Path);a.add_argument('--anchor',type=Path);args=a.parse_args()
    if args.mode=='preflight':
        require(not args.output.exists(),'Preflight output already exists; preserve it');args.output.mkdir(parents=True)
        _,_,_,_,metadata=load_inputs(args.source_root,args.case_root,args.output)
        write(args.output/'PREFLIGHT.json',metadata);print(json.dumps({'status':'preflight_passed','checkpoints_loaded':15,'model_forwards':0,'autograd_calls':0,'training_jobs':0}))
    else:
        require(args.plan and args.anchor,'Plan and public anchor required')
        plan=read(args.plan);require(plan['code_sha256']==sha(__file__),'Frozen code hash differs')
        require(plan['max_forward_calls']==MAX_FORWARD and plan['max_point_evaluations']==MAX_POINTS and plan['max_autograd_calls']==MAX_AUTOGRAD,'Plan counters differ')
        require(tuple(plan['recipes'])==RECIPES and tuple(plan['seeds'])==SEEDS,'Plan identity differs')
        require(plan['max_wall_seconds']==MAX_SECONDS and plan['max_RSS_bytes']==MAX_RSS_BYTES,'Plan ceilings differ')
        for item in plan['bound_source_files']:
            require(file_record(args.source_root/item['path'])=={'bytes':item['bytes'],'sha256':item['sha256']},'Frozen measurement source binding differs')
        execute(args,plan)

if __name__=='__main__':main()
