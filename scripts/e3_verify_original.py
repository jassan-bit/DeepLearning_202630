"""Record synthetic numerical checks for E3's development candidate.

No market dataset, benchmark, HPO or confirmatory comparison is evaluated.
The finite numerical checks support, but do not replace, the formal proof.
--check reads stored evidence and source hashes without importing PyTorch.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

DATA = Path('experiments/e3_bloque_c_20261115')
REPORT = DATA/'software_verification.json'
SOURCE_PATHS = ['src/original_model.py', 'tests/test_original_model.py',
                'scripts/e3_verify_original.py', 'requirements-e3-original-cpu.lock.txt',
                str(DATA/'theory.md')]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_junit(path):
    xml=ET.parse(path).getroot()
    suites=[xml] if xml.tag=='testsuite' else list(xml.iter('testsuite'))
    counts={key:sum(int(s.attrib.get(key,0)) for s in suites)
            for key in ['tests','failures','errors','skipped']}
    if not counts['tests'] or any(counts[k] for k in ['failures','errors','skipped']):
        raise ValueError('A completed non-skipped successful pytest run is required')
    return counts


def run(root, junit):
    import torch
    sys.path.insert(0,str(root))
    from src.original_model import CandidateConfig, OriginalModel
    torch.set_num_threads(1)
    config=CandidateConfig()
    model=OriginalModel(config,seed=20261115).double().eval()
    head=model.mechanism
    generator=torch.Generator().manual_seed(20261115)
    magnitude=torch.logspace(-3,3,256,dtype=torch.float64)
    states=torch.randn(256,7,config.hidden_size,generator=generator,dtype=torch.float64)*magnitude[:,None,None]
    other=states+torch.randn(states.shape,generator=generator,dtype=torch.float64)*1e-4
    with torch.no_grad():
        difference=(head(states)-head(other)).abs()
        distance=torch.linalg.matrix_norm(states-other,ord='fro')
        gain=difference/distance
        bound=head.latent_lipschitz_bound()
        if not torch.all(difference<=bound*distance+1e-12):
            raise ValueError('A sampled pair violates the latent bound')
        values=head.attention_components(states)['values']
        value_norm=torch.linalg.vector_norm(values,dim=-1)
        if float(value_norm.max())>config.radius+1e-12:
            raise ValueError('Radial values exceed the stated radius')
        pars=head.effective_parameters()
        q,w=float(torch.linalg.vector_norm(pars['q'])),float(torch.linalg.vector_norm(pars['w']))
        penalty,temperature=float(pars['penalty']),float(pars['temperature'])
    point=torch.randn(1,7,config.hidden_size,generator=generator,dtype=torch.float64,requires_grad=True)
    derivative,=torch.autograd.grad(head(point).sum(),point)
    grad_norm=float(torch.linalg.vector_norm(derivative))
    if grad_norm>bound+1e-12:
        raise ValueError('Latent input gradient exceeds the stated bound')
    # Independent derivative from the displayed equations, not autograd itself.
    with torch.no_grad():
        h=point[0]
        components=head.attention_components(point)
        a=components['weights'][0]
        v=components['values'][0]
        c=components['context'][0]
        pars=head.effective_parameters()
        q,w=pars['q'],pars['w']
        squared=config.radius**2+h.square().sum(dim=-1)
        direct=(config.radius/squared.sqrt())[:,None]*w[None,:]
        direct-=((config.radius*(h@w)/squared.pow(1.5))[:,None]*h)
        gs=a*(v@w-c@w)/pars['temperature']
        operator=torch.eye(7,dtype=h.dtype)-head.P
        residual=operator@h
        distances=components['distance'][0]
        manual=a[:,None]*direct+gs[:,None]*q[None,:]
        manual-=pars['penalty']*(operator.T@((gs/distances)[:,None]*residual))
        manual*=torch.sigmoid(c@w+head.bias)
        gradient_error=float((manual-derivative[0]).abs().max())
        if gradient_error>1e-11:
            raise ValueError('Manual latent derivative differs from automatic differentiation')
    proof_source={p:sha256(root/p) for p in SOURCE_PATHS}
    report={
        'version':'e3-original-development-v1',
        'executed_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'synthetic software and latent-head property verification',
        'configuration':model.configuration(),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,
                       'pytest':importlib.metadata.version('pytest'),
                       'platform':platform.system(),'device':'CPU','torch_threads':1},
        'pytest':parse_junit(junit),
        'parameters':sum(p.numel() for p in model.parameters()),
        'latent_verification':{
            'unit_test_seed':20261115,'pairs':256,
            'state_magnitude_min':0.001,'state_magnitude_max':1000.0,
            'perturbation_SD':1e-4,'dtype':'float64',
            'formal_upper_bound':bound,
            'maximum_sampled_gain':float(gain.max()),
            'maximum_sampled_gain_over_bound':float(gain.max())/bound,
            'sampled_violations':int((difference>bound*distance+1e-12).sum()),
            'maximum_value_norm':float(value_norm.max()),
            'single_point_gradient_norm':grad_norm,
            'manual_gradient_maximum_absolute_error':gradient_error,
            'effective_query_norm':float(torch.linalg.vector_norm(q)),
            'effective_readout_norm':float(torch.linalg.vector_norm(w)),
            'effective_penalty':penalty,'effective_temperature':temperature,
            'guarantee_scope':'H to positive scaled prediction, fixed P and inference parameters',
            'full_OHLCV_encoder_bound_claimed':False,
            'sampling_proves_global_property':False},
        'source_sha256':proof_source,
        'real_market_data_used':False,'HPO_performed':False,
        'confirmatory_protocol_registered':False,
        'novelty_verified':False,'student_defence_verified':False,
        'statistical_superiority_verified':False,
        'warning_scope':'TorchScript smoke test requires an external shape/finiteness guard; no trained candidate export is delivered.'}
    directory=root/DATA;directory.mkdir(parents=True,exist_ok=True)
    (root/REPORT).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def check(root):
    report=json.loads((root/REPORT).read_text())
    if set(report['source_sha256'])!=set(SOURCE_PATHS):
        raise ValueError('Missing source provenance')
    for path,expected in report['source_sha256'].items():
        if sha256(root/path)!=expected:
            raise ValueError('Verification source changed: '+path)
    counts=report['pytest']
    if counts['tests']<1 or any(counts[k] for k in ['failures','errors','skipped']):
        raise ValueError('Invalid pytest evidence')
    property_check=report['latent_verification']
    numeric_fields=['formal_upper_bound','maximum_sampled_gain',
                    'maximum_sampled_gain_over_bound','maximum_value_norm',
                    'single_point_gradient_norm','manual_gradient_maximum_absolute_error',
                    'effective_query_norm','effective_readout_norm','effective_penalty',
                    'effective_temperature']
    if any(isinstance(property_check[k],bool) or not isinstance(property_check[k],(int,float))
           or not math.isfinite(property_check[k]) or property_check[k]<0 for k in numeric_fields):
        raise ValueError('Invalid numeric verification evidence')
    if (property_check['pairs']!=256 or property_check['sampled_violations']!=0
            or not math.isfinite(property_check['formal_upper_bound'])
            or property_check['formal_upper_bound']<=0
            or property_check['maximum_sampled_gain']>property_check['formal_upper_bound']):
        raise ValueError('Invalid latent-property verification')
    config=report['configuration']['config']
    if (property_check['maximum_value_norm']>config['radius']+1e-12
            or property_check['effective_query_norm']>config['query_bound']+1e-12
            or property_check['effective_readout_norm']>config['readout_bound']+1e-12
            or property_check['effective_penalty']>config['penalty_bound']+1e-12
            or property_check['effective_temperature']<config['temperature_min']
            or property_check['manual_gradient_maximum_absolute_error']>1e-11
            or property_check['full_OHLCV_encoder_bound_claimed'] is not False
            or property_check['sampling_proves_global_property'] is not False):
        raise ValueError('Property constraints or scope do not match the stated verification')
    for key in ['real_market_data_used','HPO_performed','confirmatory_protocol_registered',
                'novelty_verified','student_defence_verified','statistical_superiority_verified']:
        if report[key] is not False:
            raise ValueError('Reassess the development report when its scope changes')
    return report


def illustration(root):
    """Analytical n=2, lambda=0 example, separate from the candidate benchmark."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    size=np.logspace(-3,3,121)
    delta=1e-4
    ordinary=(np.logaddexp(0,size*np.tanh(delta))-np.log(2))/(math.sqrt(2)*delta)
    rows=[]
    fig,ax=plt.subplots(figsize=(7.2,4.1),layout='constrained')
    ax.loglog(size,ordinary,color='#a44337',label='Valores sin saturar')
    for radius,color in [(0.5,'#436b8d'),(1.0,'#1b6b50'),(2.0,'#8065a4')]:
        saturated=radius*size/np.sqrt(radius**2+size**2+delta**2)
        gain=(np.logaddexp(0,saturated*np.tanh(delta))-np.log(2))/(math.sqrt(2)*delta)
        ax.loglog(size,gain,color=color,label=f'Saturación radial B={radius:g}')
        for r,plain,bounded in zip(size,ordinary,gain):
            rows.append({'state_magnitude':float(r),'radius':radius,
                         'identity_value_gain':float(plain),'bounded_value_gain':float(bounded),
                         'perturbation':delta})
    ax.set(xlabel='Magnitud del estado latente R',ylabel='Cambio de salida / norma de perturbación',
           title='Ejemplo analítico de sensibilidad de la atención')
    ax.grid(True,which='both',alpha=.18)
    ax.legend(fontsize=9)
    folder=root/'book_e3/_static/e3_bloque_c_20261115';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'radial_sensitivity.png',dpi=180)
    plt.close(fig)
    with (root/DATA/'radial_sensitivity_example.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--junit',type=Path)
    parser.add_argument('--illustration',action='store_true')
    args=parser.parse_args()
    if args.check:
        report=check(args.root)
    else:
        if args.junit is None:parser.error('--junit is required for a new numerical verification')
        report=run(args.root,args.junit)
    if args.illustration:illustration(args.root)
    print(json.dumps({'pytest_passed':report['pytest']['tests'],
                      'latent_pairs':report['latent_verification']['pairs'],
                      'sampled_violations':report['latent_verification']['sampled_violations'],
                      'novelty_verified':report['novelty_verified'],
                      'statistical_superiority_verified':report['statistical_superiority_verified']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
