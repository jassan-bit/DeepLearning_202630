"""Export scientific figures from the bound, actually computed audit tables.

No model, data acquisition, fit or statistical resampling occurs here. RMSE
bars carry the existing conditional calendar-block intervals; seed dispersion
is explicitly labelled separately. Synthetic software fixtures are watermarked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def require(value,message):
    if not value:raise ValueError(message)


def _write(path,raw):
    path=Path(path)
    require(not any(p.is_symlink() for p in (path,*path.parents)),'Figure symlinks are refused')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(path.read_bytes()==raw,'Existing figure differs; preserve and choose another output')
    else:
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.partial-'+path.name+'.',delete=False) as stream:
                temporary=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
            try:os.link(temporary,path)
            except FileExistsError:
                require(path.read_bytes()==raw,'Existing figure differs; preserve and choose another output')
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)


def load_tables(directory):
    directory=Path(directory)
    receipt_raw=(directory/'analysis_receipt.json').read_bytes()
    receipt=json.loads(receipt_raw)
    manifest_raw=(directory/'output_manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    require(manifest['protocol_sha256']==receipt['protocol_sha256'],'Analysis protocol binding changed')
    tables={}
    for name,record in manifest['files'].items():
        require(Path(name).name==name and name not in {'.','..'},'Analysis output path is unsafe')
        path=directory/name
        require(path.is_file() and not path.is_symlink(),'A regular analysis output is required')
        raw=path.read_bytes()
        require(len(raw)==record['bytes'] and sha(raw)==record['sha256'],'Analysis table differs from its manifest')
        if name.endswith('.csv'):tables[name[:-4]]=list(csv.DictReader(io.StringIO(raw.decode())))
    require('calendar_CIs' in tables and 'model_metrics' in tables,'RMSE tables and conditional intervals are required')
    return receipt,tables,{'analysis_receipt_sha256':sha(receipt_raw),'analysis_manifest_sha256':sha(manifest_raw)}


def run(analysis_dir,output):
    receipt,tables,binding=load_tables(analysis_dir)
    config=receipt['configuration'];output=Path(output)
    code_sha=sha(Path(__file__).read_bytes())
    if (output/'figures_manifest.json').is_file():
        previous=json.loads((output/'figures_manifest.json').read_bytes())
        require(previous['inputs']==binding and previous['plot_code_sha256']==code_sha,'Existing figures bind different analysis or plotting code')
        for name,record in previous['files'].items():
            raw=(output/name).read_bytes()
            require(len(raw)==record['bytes'] and sha(raw)==record['sha256'],'Existing figure bytes changed')
        return previous
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'svg.hashsalt':receipt['protocol_sha256'],'axes.spines.top':False,'axes.spines.right':False})
    datasets=config['datasets'];family=config['family'];candidate=config['candidate']
    synthetic=config.get('synthetic') is True
    scope='IC temporal condicional a trayectorias ajustadas; no remuestrea semillas.'
    watermark='DATOS SINTÉTICOS · PRUEBA DE SOFTWARE' if synthetic else 'Auditoría preespecificada exploratoria · no acredita R3'
    files={};descriptions={}

    def save(fig,name,description):
        fig.text(.01,.006,watermark,fontsize=7,color='#9b2020' if synthetic else '#555555')
        fig.tight_layout(rect=(0,.03,1,.97))
        for extension in ('png','svg'):
            target=name+'.'+extension;stream=io.BytesIO()
            metadata={'Title':description,'Description':watermark}
            if extension=='svg':metadata['Date']=None
            fig.savefig(stream,format=extension,dpi=170,metadata=metadata)
            raw=stream.getvalue();_write(output/target,raw)
            files[target]={'bytes':len(raw),'sha256':sha(raw)}
        descriptions[name]=description
        plt.close(fig)

    # One chart per dataset avoids pooling RMSE across incompatible price tasks.
    for dataset in datasets:
        rows={r['model']:r for r in tables['calendar_CIs'] if r['dataset']==dataset
              and r['scenario']=='unperturbed' and int(r['block_observations'])==7}
        require(set(rows)==set(family),'The primary RMSE figure must include the entire fixed family')
        points=np.asarray([float(rows[m]['RMSE_mean_individual']) for m in family])
        bounds=np.asarray([json.loads(rows[m]['RMSE_CI']) for m in family])
        require(bounds.shape==(len(family),2) and np.isfinite(bounds).all(),'Finite existing RMSE confidence bounds required')
        fig,ax=plt.subplots(figsize=(10,max(4,.30*len(family))))
        positions=np.arange(len(family));colors=['#a52b2b' if m==candidate else '#467895' for m in family]
        ax.barh(positions,points,color=colors,alpha=.78)
        # Render the supplied interval endpoints directly: a basic bootstrap
        # interval is not guaranteed to contain its point estimate.
        ax.hlines(positions,bounds[:,0],bounds[:,1],color='black',linewidth=1)
        ax.vlines(bounds[:,0],positions-.1,positions+.1,color='black',linewidth=.8)
        ax.vlines(bounds[:,1],positions-.1,positions+.1,color='black',linewidth=.8)
        ax.set_yticks(positions,family);ax.invert_yaxis()
        ax.set_xlabel('Media de RMSE individuales (σ₇, sin anualizar)')
        ax.set_title(f'{dataset} · familia completa e IC nominal {100*config["confidence"]:.0f}% · bloques de 7 observaciones')
        fig.suptitle(scope,fontsize=8,y=.995)
        ax.grid(axis='x',alpha=.2);save(fig,'rmse_'+dataset,scope+' Todos los sistemas; IC nominal, no simultáneo.')

    chosen=[m for m in [candidate,'TSMixer','floor_only','Persistence'] if m in family]
    for dataset in datasets:
        fig,ax=plt.subplots(figsize=(8,4.2))
        for model in chosen:
            values=[]
            for scenario in config['scenarios']:
                row=next(r for r in tables['model_metrics'] if r['dataset']==dataset and r['split']=='test'
                         and r['scenario']==scenario and r['model']==model)
                values.append(float(row['RMSE_mean_individual']))
            ax.plot(np.arange(len(values)),values,marker='o',label=model)
        ax.set_xticks(np.arange(len(config['scenarios'])),['Limpio','Nivel 1','Nivel 2','Nivel 3'][:len(config['scenarios'])])
        ax.set_ylabel('Media de RMSE individuales (σ₇)');ax.set_xlabel('Perturbación causal D7; objetivo limpio; calibrador fijo')
        ax.set_title(dataset+' · sensibilidad a errores de observación');ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,'noise_'+dataset,'Comparadores preseleccionados. La historia cruda de los clásicos permanece limpia; cambia la proyección común.')

    if tables.get('quarter_metrics'):
        for dataset in datasets:
            rows=[r for r in tables['quarter_metrics'] if r['dataset']==dataset and r['scenario']=='unperturbed' and r['model']==candidate]
            quarters=sorted({r['subgroup'] for r in rows});require(quarters,'Candidate coverage quarters are missing')
            coverage=[np.mean([float(r['coverage']) for r in rows if r['subgroup']==q]) for q in quarters]
            fig,ax=plt.subplots(figsize=(max(8,.38*len(quarters)),4.2))
            ax.plot(quarters,coverage,marker='o',label='Media entre semillas del candidato')
            ax.axhline(1-config['alpha'],color='black',linestyle='--',label='Nivel nominal')
            ax.set_ylim(0,1.03);ax.set_ylabel('Cobertura empírica');ax.set_xlabel('Trimestre del calendario')
            ax.tick_params(axis='x',rotation=45);ax.set_title(dataset+' · cobertura descriptiva por trimestre')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
            save(fig,'coverage_'+dataset,'Cobertura descriptiva por trimestre; no garantiza cobertura futura o por grupo.')

    if tables.get('sample_size_metrics'):
        for dataset in datasets:
            fig,ax=plt.subplots(figsize=(8,4.2))
            models=[m for m in [candidate,'TSMixer'] if m in family]
            for model in models:
                rows=[r for r in tables['sample_size_metrics'] if r['dataset']==dataset and r['model']==model]
                fractions=sorted({float(r['train_fraction']) for r in rows})
                require(fractions==[.25,.5,1.],'Actual three-fraction learning curves are required')
                grouped=[[float(r['RMSE']) for r in rows if float(r['train_fraction'])==f] for f in fractions]
                require(all(len(v)==3 for v in grouped),'The learning plot requires its three prespecified seeds')
                ax.errorbar(np.asarray(fractions)*100,[np.mean(v) for v in grouped],
                            yerr=[np.std(v,ddof=1) for v in grouped],marker='o',capsize=3,label=model)
            ax.set_xlabel('Fracción del ajuste (%) · prefijos anidados');ax.set_ylabel('RMSE media ± DE entre 3 semillas (σ₇)')
            ax.set_title(dataset+' · ajustes realmente redimensionados');ax.legend(fontsize=8);ax.grid(alpha=.2)
            save(fig,'learning_'+dataset,'DE entre tres semillas, no IC temporal; tamaño y extensión del calendario se modifican conjuntamente.')

    result={'schema_version':1,'kind':'e3_extended_scientific_figures','inputs':binding,
            'plot_code_sha256':code_sha,'matplotlib':matplotlib.__version__,
            'synthetic_fixture':synthetic,'R3_accredited':False,'new_model_fits':0,
            'RMSE_error_bars':'Existing nominal calendar-block confidence intervals; not seed SD or simultaneous Holm intervals',
            'learning_error_bars':'Sample SD across the three declared initializations',
            'descriptions':descriptions,'files':files}
    _write(output/'figures_manifest.json',canonical(result))
    print(json.dumps({'figures':len(descriptions),'files':len(files),'synthetic_fixture':synthetic,'new_model_fits':0}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.analysis_dir,args.output)
