"""Replay all 15 final pilot checkpoints on existing development dates only."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from replay_common import development_inputs,ABS_TOL_SIGMA7
from frozen_models import initialize,load_model,infer,NEURAL,SEEDS

def main():
    p=argparse.ArgumentParser();p.add_argument('--npz',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    raw,X,A,_,admission=development_inputs(root,a.npz)
    torch,plugin,admission,snapshot=initialize(root)
    rows=list(csv.DictReader((root/'reference/assessment_predictions.csv').open()))
    # Targets from the archived assessment table, not calibration/test arrays.
    dates=[r['date'] for r in rows if r['name']=='atom_crps' and int(r['seed'])==SEEDS[0]]
    target=np.asarray([float(r['target']) for r in rows if r['name']=='atom_crps' and int(r['seed'])==SEEDS[0]])
    U=np.sqrt(np.maximum(target*target-A,0)/(6/49))
    receipts=[]
    for name in NEURAL:
        for seed in SEEDS:
            model,meta=load_model(root,name,seed,torch,plugin,admission,snapshot)
            got,_=infer(model,name,X,A,U,torch)
            ref=[r for r in rows if r['name']==name and int(r['seed'])==seed]
            if [r['date'] for r in ref]!=dates:raise ValueError('Reference calendar mismatch')
            errors=np.abs(got-np.asarray([float(r['prediction']) for r in ref]))
            if not np.isfinite(got).all() or np.any(errors>ABS_TOL_SIGMA7):raise ValueError('Frozen checkpoint replay failed')
            receipts.append({**meta,'rows':55,'max_absolute_error_sigma7':float(errors.max()),'tolerance_absolute_sigma7':ABS_TOL_SIGMA7,'status':'passed'})
    receipt={'status':'passed','AI_assistance_declared':True,'level':'B:15 final frozen pilot checkpoints and825 saved development predictions','checkpoints':receipts,'new_model_fits':0,'optimizer_steps':0,'new_calibrations':0,'new_domain_evaluations':0,'human_reproduction_observed':False,'not_training_reproduction':True}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'checkpoints':len(receipts),'saved_development_predictions':825,'max_absolute_error_sigma7':max(x['max_absolute_error_sigma7'] for x in receipts)}))

if __name__=='__main__':main()
