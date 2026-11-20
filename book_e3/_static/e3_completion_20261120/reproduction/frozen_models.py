"""Hash-bound final pilot models, inference only; AI assistance declared."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from replay_common import sha

NEURAL=('additive_positive_head','atom_crps','atom_mse_only','fixed_probs','atom_plugin')
SEEDS=(20261118,20261119,20261120)


def initialize(root):
    root=Path(root);plan=json.loads((root/'reference/plan.json').read_text())
    for relative,expected in plan['identity']['source_files'].items():
        if sha(root/'source'/relative)!=expected:raise ValueError('Frozen scientific source changed: '+relative)
    plugin=root/'source/experiments/e3_v3_development_pilot_20261120/plugin_control.py'
    if sha(plugin)!=plan['identity']['plugin_sha256']:raise ValueError('Frozen plug-in source changed')
    sys.path.insert(0,str(root/'source'))
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    def refuse_step(*args,**kwargs):raise RuntimeError('Optimizer steps prohibited in frozen-weight inference')
    for optimizer in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):optimizer.step=refuse_step
    spec=importlib.util.spec_from_file_location('e3_frozen_plugin',plugin)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    admission=json.loads((root/'reference/admission.json').read_text())
    snapshot=json.loads((root/'reference/snapshot_manifest.json').read_text())
    return torch,module,admission,snapshot


def load_model(root,name,seed,torch,plugin,admission,snapshot):
    from src.e3_innovation_model import InnovationModel
    from src.e3_overlap_model_v3 import OverlapDistributionModel
    args={'seed':seed,'u_scale':admission['scaler']['u_scale'],'y_scale':admission['scaler']['y_scale']}
    if name=='additive_positive_head':model=InnovationModel(name,**args)
    elif name=='atom_plugin':model=plugin.PluginSecondMomentModel(**args,lambda_crps=.1)
    else:model=OverlapDistributionModel(name,**args,lambda_crps=.1)
    relative=f'jobs/final/{name}/seed{seed}_lr0.0030000/supervised/checkpoint.pt'
    path=Path(root)/'final_checkpoints'/relative.removeprefix('jobs/final/')
    entry=snapshot['files'][relative]
    if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('Frozen checkpoint bytes differ')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if not state['completed']:raise ValueError('Incomplete final checkpoint')
    model.load_state_dict(state['best'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,{'checkpoint_sha256':entry['sha256'],'best_epoch':state['best_epoch'],'completed_epoch':state['epoch'],'state_used':'best','seed':seed,'name':name}


def infer(model,name,X,A,U,torch):
    from src.e3_overlap_model_v3 import discrete_crps
    values=[];crps=[]
    with torch.no_grad():
        for start in range(0,len(X),128):
            x=torch.from_numpy(X[start:start+128]);a=torch.from_numpy(A[start:start+128])
            if name=='additive_positive_head':p=model(x,a);c=None
            else:
                out=model.forward_components(x,a);p=out['prediction']
                c=discrete_crps(out['atoms'],out['probabilities'],torch.from_numpy(U[start:start+128]))
            values.append(p.cpu().numpy())
            if c is not None:crps.append(c.cpu().numpy())
    return np.concatenate(values),None if not crps else np.concatenate(crps)
