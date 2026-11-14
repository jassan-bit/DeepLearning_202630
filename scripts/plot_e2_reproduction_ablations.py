"""Plot receipt-bound ablation effects; no predictions or bootstrap are computed."""
from pathlib import Path
import argparse,csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(folder):
 folder=Path(folder);table=folder/'ablation_comparisons.csv';manifest=json.loads((folder/'manifest.json').read_bytes())
 digest=hashlib.sha256(table.read_bytes()).hexdigest()
 if digest!=manifest['files'][table.name]['sha256']:raise ValueError('Verified recorded comparisons required')
 with table.open(newline='')as f:rows=list(csv.DictReader(f))
 labels={'remove_auxiliary_pretraining':'Sin preentrenamiento auxiliar','remove_second_encoder':'Sin segundo encoder','dropout_zero':'Sin dropout supervisado'}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'e2-recorded-ablation-effects-20261113'})
 fig,ax=plt.subplots(figsize=(10,4.5))
 for i,r in enumerate(rows):
  effect=float(r['delta_RMSE_ablated_minus_full']);low=float(r['CI95_low_unadjusted']);high=float(r['CI95_high_unadjusted'])
  color='#b45309'if r['reject_Holm05']=='True'else'#1d4ed8'
  ax.errorbar(effect,i,xerr=[[effect-low],[high-effect]],fmt='o',color=color,capsize=5,markersize=7)
 ax.axvline(0,color='#374151',linestyle='--',linewidth=1)
 ax.set_yticks(range(len(rows)),[labels[r['condition']]for r in rows]);ax.invert_yaxis()
 ax.set_xlabel('RMSE de la ablación menos RMSE del SAE completo')
 ax.set_title('Ablaciones entrenadas: diferencias de RMSE e IC95 % sin ajuste')
 ax.grid(axis='x',alpha=.2)
 fig.text(.03,.02,'Seis semillas emparejadas; 608 fechas; bootstrap de 10.000 réplicas, bloques de 14 días.\n'
          'Naranja: rechazo tras Holm en tres contrastes. Análisis posterior; no cambia la receta original.',fontsize=9)
 fig.tight_layout(rect=[0,.12,1,1])
 target=folder/'ablation_effects.svg';fig.savefig(target,bbox_inches='tight',metadata={'Date':None});plt.close(fig)
 proof={'version':'e2-recorded-ablation-figure-v1','input_csv_sha256':digest,'figure_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_training':False,'new_inference':False,'new_bootstrap':False}
 (folder/'figure_manifest.json').write_text(json.dumps(proof,sort_keys=True,indent=2)+'\n')
 manifest['files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for p in sorted(folder.iterdir())if p.is_file()and p.name!='manifest.json'}
 (folder/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n')
 return proof
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--folder',type=Path,required=True);a=p.parse_args();print(json.dumps(draw(a.folder)))
