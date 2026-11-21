"""Plot deterministic CSV diagnostics; no scientific resampling or model calls."""
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/e3-diagnostic-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/e3-diagnostic-cache')
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
p=root/'results/date_error_contributions.csv'
with p.open() as f: rows=list(csv.DictReader(f))
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
for ax,(cohort,label) in zip(axes,[('V3_GPU','V3 GPU: 3 semillas'),('V4_CPU','V4 CPU: 5 semillas')]):
    rr=[r for r in rows if r['cohort']==cohort]
    x=[datetime.fromisoformat(r['date']) for r in rr]
    v=[float(r['mean_squared_error_difference'])/55*1e6 for r in rr]
    acc=[];s=0
    for y in v:s+=y;acc.append(s)
    ax.bar(x,v,color=['#b94b41' if y>=0 else '#287c8e' for y in v],alpha=.55,label='Contribución de cada fecha / 55')
    ax.plot(x,acc,color='#202a44',linewidth=1.8,label='Contribución acumulada')
    ax.axhline(0,color='#333333',linewidth=.7)
    ax.set_title(label+' frente a su control MSE-only',loc='left',fontsize=11)
    ax.set_ylabel('ΔMSE · 10⁶\n(σ₇²)')
    ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8,loc='upper left')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
axes[-1].set_xlabel('55 fechas de desarrollo ya expuestas, 2024; cohortes separadas')
fig.suptitle('Dónde se acumula el exceso de error cuadrático observado',fontsize=13)
matplotlib.rcParams['svg.hashsalt']='e3-diagnosis-20261121'
fig.savefig(root/'excess_mse_by_date.svg',metadata={'Date':None})
preview=Path('/workspace/e3_failure_diagnosis_delivery');preview.mkdir(exist_ok=True)
fig.savefig(preview/'figure_preview.png',dpi=140)
receipt={'status':'completed','matplotlib':matplotlib.__version__,
         'input_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),
         'SVG_SHA256':hashlib.sha256((root/'excess_mse_by_date.svg').read_bytes()).hexdigest(),
         'code_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'new_fits':0,'new_forwards':0,'new_predictions':0,'new_resamples':0,
         'scope':'Figure from previously computed descriptive error contributions; not a new significance or uncertainty analysis.'}
(root/'results/FIGURE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
