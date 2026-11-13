"""Plot recorded frozen-predictor diagnostics; no model or target recomputation."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

SUMMARY_SHA='e9df7b0f89370f807d6387574d8195161353b8d5fe4da40e99b60d9a2c1587bd'
SOURCE_COMMIT='12cccdec2c454fd1510c1ac940edb61c115f0b39'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'e2-posthoc-documentary-v1'})

def draw(source,destination):
    source,destination=Path(source),Path(destination)
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SUMMARY_SHA:raise ValueError('Verified producer summary required')
    value=json.loads(raw)
    if value['predictor_records']!=79 or not value['all_baseline_projections_passed']:
        raise ValueError('Complete frozen-predictor summary required')
    destination.mkdir(parents=True,exist_ok=True)
    made=[]
    def save(fig,name):
        for suffix in ['svg','png']:
            path=destination/(name+'.'+suffix)
            if path.exists():raise FileExistsError(path)
            fig.savefig(path,bbox_inches='tight',dpi=180,
                        metadata={'Date':None} if suffix=='svg' else {'Software':'E2 recorded diagnostic plots'})
            made.append(path)
        plt.close(fig)
    names=sorted({r['model'] for r in value['normalized_degradation_areas']})
    kinds=['noise','outlier','missing','structural_break']
    areas=np.array([[100*next(r['normalized_degradation_area'] for r in value['normalized_degradation_areas']
                             if r['model']==name and r['kind']==kind) for kind in kinds] for name in names])
    fig,ax=plt.subplots(figsize=(10,7.5))
    mesh=ax.imshow(areas,aspect='auto',cmap='RdBu_r',norm=SymLogNorm(linthresh=10,vmin=-10,vmax=1000))
    ax.set_xticks(range(4),['Ruido','Outliers','Faltantes','Desplazamiento'])
    ax.set_yticks(range(len(names)),[n.replace('LSTM-autoencoder con predictor','LSTM-SAE') for n in names])
    for i in range(len(names)):
        for j in range(4):ax.text(j,i,f'{areas[i,j]:.1f}%',ha='center',va='center',fontsize=9,
                                 color='white' if areas[i,j]>350 else 'black')
    fig.colorbar(mesh,ax=ax,label='Área normalizada de degradación RMSE (%) · escala simlog',ticks=[-10,0,10,100,1000])
    ax.set_title('Sensibilidad de los 13 predictores neuronales congelados\nÁrea relativa, integrada sobre las severidades registradas')
    fig.text(.03,.01,'R=6; menor área indica menor degradación relativa. El error basal difiere entre modelos.\n'
             'Estrés de entradas estandarizadas independientes; objetivos y calibradores conservados.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,1]);save(fig,'robustness_areas')
    groups={name:[r for r in value['profiles'] if r['model']==name] for name in names}
    timing=sorted((np.mean([r['latency_gpu_median_ms'] for r in rows]),name,rows) for name,rows in groups.items())
    means=np.array([x[0] for x in timing]);low=np.array([min(r['latency_gpu_median_ms'] for r in x[2]) for x in timing])
    high=np.array([max(r['latency_gpu_median_ms'] for r in x[2]) for x in timing])
    fig,ax=plt.subplots(figsize=(10,6.5))
    ax.barh(range(len(timing)),means,xerr=np.array([means-low,high-means]),color='#4258a8',capsize=3)
    ax.set_yticks(range(len(timing)),[x[1].replace('LSTM-autoencoder con predictor','LSTM-SAE') for x in timing])
    ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Milisegundos por forward, batch=1 (escala log)')
    ax.set_title('A100: media de las seis medianas de latencia\nLínea: mínimo–máximo de esas medianas entre semillas')
    fig.text(.03,.01,'Eventos CUDA; entrada residente 1×7×5; 32 calentamientos y 100 repeticiones por predictor.\n'
             'Excluye transferencia, preprocesamiento y latencia de servicio. No es el tiempo de entrenamiento.',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,1]);save(fig,'inference_latency')
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True)
    for ax,item in zip(axes,value['interpretability']):
        rows=item['fidelity_per_seed'];fractions=rows[0]['fractions']
        ranked=np.mean([r['insertion_mean_absolute_distance_to_original_prediction'] for r in rows],axis=0)
        random=np.mean([r['random_insertion_mean_absolute_distance_to_original_prediction'] for r in rows],axis=0)
        ax.plot(fractions,ranked,'o-',label='Orden por oclusión',color='#4258a8')
        ax.plot(fractions,random,'s--',label='Orden aleatorio fijo',color='#c15b20')
        ax.set_title(item['model'].replace('LSTM-autoencoder con predictor','LSTM-SAE')+' · R='+str(item['R']))
        ax.set_xlabel('Fracción de coordenadas restauradas');ax.grid(alpha=.25)
    axes[0].set_ylabel('Distancia absoluta media a la predicción original');axes[-1].legend(fontsize=9)
    fig.suptitle('Fidelidad de respuesta del predictor: inserción en 24 fechas fijas')
    fig.text(.03,.01,'Menor distancia indica mejor reconstrucción de la respuesta. Las redes usan 35 celdas D7/OHLCV;\n'
             'ARIMA usa siete rezagos del objetivo y mantiene fijo el estado anterior. No mide causalidad del mercado.',fontsize=9)
    fig.tight_layout(rect=[0,.12,1,.95]);save(fig,'explanation_fidelity')
    manifest={'version':'e2-recorded-posthoc-figures-v1','source_commit':SOURCE_COMMIT,'producer_summary_sha256':SUMMARY_SHA,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_inference':False,
              'new_training':False,'new_bootstrap':False,'figures':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
              'bytes':p.stat().st_size} for p in made}}
    (destination/'presentation_manifest.json').write_text(json.dumps(manifest,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    print(json.dumps(draw(a.summary,a.destination),ensure_ascii=False))
