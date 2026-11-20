"""Reproduce one frozen V3 checkpoint and export a separate experimental graph.

AI assistance declared. No optimizer/training/calibration path is called.
The fixed minimum final seed was chosen before reading individual performance.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,sys,time,warnings
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from replay_common import *


class StableHypotReadout(torch.nn.Module):
    """Same frozen graph, with the non-exportable hypot expressed by scaled primitives.

For finite valid inputs, max(x,y)*sqrt(1+(min(x,y)/max(x,y))²) equals hypot.
The one-zero denominator fallback is irrelevant to positive learned atoms and
also makes the zero case explicit. Inputs are validated by replay_common.
"""
    def __init__(self, model):super().__init__();self.model=model
    def forward(self, values,A):
        h=self.model._context(values).to(torch.float64)
        z=self.model.head(h)
        # ORT1.23 CPU has no double Softplus kernel. The stable primitive
        # identity preserves the mathematical softplus and Torch's threshold20.
        positive=torch.clamp(z,min=0)
        softplus=torch.where(z>20,z,positive+torch.log(1+torch.exp(-torch.abs(z))))
        atoms=(self.model.u_scale*(1e-4+softplus)).cumsum(-1)
        logits=self.model.probability_head(h)
        p=torch.softmax(torch.cat([logits,torch.zeros_like(logits[:,:1])],-1),-1)
        x=torch.sqrt(A)[:,None];y=math.sqrt(6./49.)*atoms
        high=torch.maximum(x,y);low=torch.minimum(x,y)
        denominator=torch.where(high>0,high,torch.ones_like(high))
        sigma=high*torch.sqrt(1+(low/denominator).square())
        return torch.maximum((p*sigma).sum(-1),torch.sqrt(A))


def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(root,npz):
    root=Path(root).resolve();out=root/'artifacts/atom_crps_seed20261118';out.mkdir(exist_ok=True,parents=True)
    plan=json.loads((root/'reference/plan.json').read_text())
    if min(plan['final_seeds'])!=SEED or plan['config']['lambda_crps']!=.1:raise ValueError('Frozen representative/recipe changed')
    for name,h in plan['identity']['source_files'].items():
        if sha(root/'source'/name)!=h:raise ValueError('Frozen source changed:'+name)
    checkpoint=out/'checkpoint.pt'
    if sha(checkpoint)!=CHECKPOINT_SHA256:raise ValueError('Checkpoint SHA256 differs')
    raw,X,A,rows,admission=development_inputs(root,npz)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    def no_step(*args,**kwargs):raise RuntimeError('Training is forbidden in checkpoint reproduction')
    for optimizer in [torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD]:optimizer.step=no_step
    sys.path.insert(0,str(root/'source'))
    from src.e3_overlap_model_v3 import OverlapDistributionModel
    model=OverlapDistributionModel('atom_crps',u_scale=admission['scaler']['u_scale'],y_scale=admission['scaler']['y_scale'],seed=SEED,lambda_crps=.1)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if not state['completed'] or state['steps']<1:raise ValueError('Checkpoint was not a completed trained phase')
    model.load_state_dict(state['best'],strict=True);model.eval()
    if sum(p.numel() for p in model.parameters())!=1262:raise ValueError('Capacity changed')
    if float(model.u_scale)!=admission['scaler']['u_scale'] or float(model.y_scale)!=admission['scaler']['y_scale']:raise ValueError('Frozen scale differs')
    tx,ta=torch.from_numpy(X),torch.from_numpy(A)
    with torch.inference_mode():
        batched=model(tx,ta).numpy()
        single=np.concatenate([model(tx[i:i+1],ta[i:i+1]).numpy() for i in range(len(X))])
        wrapper=StableHypotReadout(model).eval();adapted=np.concatenate([wrapper(tx[i:i+1],ta[i:i+1]).numpy() for i in range(len(X))])
    frozen=np.asarray([float(r['prediction']) for r in rows]);target=np.asarray([float(r['target']) for r in rows])
    measures={'CPU_batch55_vs_frozen_GPU':float(np.max(np.abs(batched-frozen))),
              'CPU_batch1_vs_frozen_GPU':float(np.max(np.abs(single-frozen))),
              'CPU_batch1_vs_CPU_batch55':float(np.max(np.abs(single-batched))),
              'export_adapter_vs_original_CPU':float(np.max(np.abs(adapted-single)))}
    if max(measures.values())>ABS_TOL_SIGMA7:raise ValueError('Pre-export parity failed:'+str(measures))
    model_path=out/'model.onnx'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.onnx.export(wrapper,(tx[:1],ta[:1]),model_path,input_names=['values','A'],output_names=['sigma7'],opset_version=17,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(str(model_path))
    sess=ort.InferenceSession(str(model_path),sess_options=session_options(ort),providers=['CPUExecutionProvider'])
    predicted=np.concatenate([sess.run(None,{'values':X[i:i+1],'A':A[i:i+1]})[0] for i in range(len(X))])
    measures['ORT_batch1_vs_CPU_original']=float(np.max(np.abs(predicted-single)))
    measures['ORT_batch1_vs_frozen_GPU']=float(np.max(np.abs(predicted-frozen)))
    if max(measures.values())>ABS_TOL_SIGMA7:raise ValueError('ORT parity failed:'+str(measures))
    if not np.isfinite(predicted).all() or not (predicted+1e-12>=np.sqrt(A)).all():raise ValueError('Invalid exported prediction')
    # Numerical validation, not a market experiment. Absolute tolerance is kept
    # for real data; artificial huge magnitudes additionally have a fixed rtol.
    edge_rows=[]
    for floor in [0.,1e-18,1e-8,1.,1e100,1e300]:
        ex=X[:1].copy();ea=np.asarray([floor],dtype=np.float64)
        with torch.inference_mode():ref=model(torch.from_numpy(ex),torch.from_numpy(ea)).numpy()
        got=sess.run(None,{'values':ex,'A':ea})[0]
        if not np.isfinite(got).all() or not np.allclose(got,ref,atol=ABS_TOL_SIGMA7,rtol=1e-12):raise ValueError('Synthetic hypot edge failed')
        edge_rows.append({'A':floor,'absolute_difference':float(np.max(np.abs(got-ref))),'relative_difference':float(np.max(np.abs(got-ref)/np.maximum(np.abs(ref),1e-300)))})
    with (out/'roundtrip_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['date','target','frozen_GPU','reproduced_CPU','exported_ORT','ORT_minus_frozen']);w.writeheader()
        for r,y,p,c,o in zip(rows,target,frozen,single,predicted):w.writerow(dict(date=r['date'],target=float(y),frozen_GPU=float(p),reproduced_CPU=float(c),exported_ORT=float(o),ORT_minus_frozen=float(o-p)))
    profile={'profile_id':PROFILE,'AI_assistance_declared':True,'variant':'atom_crps','seed':SEED,
        'representative_policy':'minimum final seed from the frozen plan; independent of assessment metrics',
        'source_commit':plan['identity']['source_commit'],'results_commit':'8eaa581bc34b615a8507c9e384a3e4be29084720',
        'run_id':plan['master_run_id'],'source_checkpoint_sha256':CHECKPOINT_SHA256,'source_state':'best validated state, epoch37; not latest epoch40',
        'onnx_sha256':sha(model_path),'onnx_bytes':model_path.stat().st_size,'trainable_parameters':1262,
        'input_contract':{'values':{'dtype':'float32','shape':[1,7,5],'meaning':'frozen relative OHLCV standardization'},'A':{'dtype':'float64','shape':[1],'meaning':'known population variance component from six returns; sigma7 squared units'}},
        'output_contract':{'sigma7':{'dtype':'float64','shape':[1],'units':'daily unannualized population sigma7 ddof0'}},
        'scaler':{k:v for k,v in admission['scaler'].items() if k!='fit_dates'},
        'calibration':{'status':'not_calibrated','quantile':None,'inherited_additive_calibrator':False},
        'scope':'experimental development-only representative; no serving promotion','default_serving_enabled':False,
        'pilot_continuation_heuristic_passed':False,'automatic_ablation_promotion':False,'R3_accredited':False,
        'parity_tolerance_absolute_sigma7':ABS_TOL_SIGMA7,'real_dates':len(rows),'dates':[r['date'] for r in rows],
        'software':{'python':platform.python_version(),'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,'numpy':np.__version__},
        'operator_adapter':'hypot represented by max/min/scaled sqrt; float64 Softplus by stable exp/log primitives with threshold20; no weight or learned-parameter change',
        'runtime_input_validation_required':True,'new_model_fits':0,'optimizer_steps_executed_by_reproduction':0,'new_calibrations':0}
    write_json(out/'metadata.json',profile)
    receipt={'status':'passed_frozen_weight_CPU_and_ONNX_reproduction','AI_assistance_declared':True,
        'level':'B:one frozen V3 representative and55development predictions; not full796training reproduction',
        'checkpoint_sha256':CHECKPOINT_SHA256,'original_npz_sha256':NPZ_SHA256,'NPZ_members_opened':['training_raw','training_dates'],
        'calibration_or_test_arrays_opened':False,'dates_verified':len(rows),'seed':SEED,'minimum_seed_policy':True,
        'primary_absolute_tolerance_sigma7':ABS_TOL_SIGMA7,'max_absolute_differences':measures,'synthetic_hypot_edges':edge_rows,
        'synthetic_edges_relative_tolerance':1e-12,'onnx_sha256':sha(model_path),'onnx_bytes':model_path.stat().st_size,
        'warnings':sorted(set(str(w.message) for w in caught)),
        'new_training':0,'new_optimizer_steps':0,'new_calibration':0,'new_confirmatory_claim':False,'human_reproduction_observed':False}
    write_json(out/'roundtrip.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--npz',required=True,type=Path);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();main(args.root,args.npz)
