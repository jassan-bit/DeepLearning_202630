from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mpl-e2-paper-evidence')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/cache-e2-paper-evidence')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'book_e2/_static/paper_guide_evidence'
out.mkdir(parents=True,exist_ok=True)
plt.rcParams['svg.hashsalt']='e2-paper-guide-timemixer-20261106'
fig,ax=plt.subplots(figsize=(12,5));ax.set(xlim=(-.03,1.03),ylim=(-.03,1.03));ax.axis('off')
def box(x,y,w,h,label,color='#edf4fb'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#285c8c'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
ys=[.77,.56,.35,.14]
for i,(y,length) in enumerate(zip(ys,[96,48,24,12])):
 box(.01,y,.17,.13,f'Escala {i}: {length} horas\n7 canales observados')
 box(.27,y,.31,.13,'Norm. por instancia + calendario\n2 bloques PDM · dimensión16')
 box(.67,y,.14,.13,'Predictor\n96 × 7')
 ax.annotate('',xy=(.25,y+.065),xytext=(.19,y+.065),arrowprops={'arrowstyle':'->','color':'#285c8c'})
 ax.annotate('',xy=(.65,y+.065),xytext=(.60,y+.065),arrowprops={'arrowstyle':'->','color':'#285c8c'})
 ax.plot([.83,.86],[y+.065,y+.065],color='#285c8c')
 if i<3:
  ax.annotate('',xy=(.09,ys[i+1]+.145),xytext=(.09,y-.01),arrowprops={'arrowstyle':'->','color':'#285c8c'})
  ax.annotate('',xy=(.37,ys[i+1]+.145),xytext=(.37,y-.01),arrowprops={'arrowstyle':'->','color':'#238565'})
  ax.annotate('',xy=(.49,y-.01),xytext=(.49,ys[i+1]+.145),arrowprops={'arrowstyle':'->','color':'#cb7b27'})
ax.plot([.86,.86],[ys[-1]+.065,ys[0]+.065],color='#285c8c');box(.90,.40,.11,.17,'FMM\nSuma 4\npronósticos','#e7f4eb');ax.annotate('',xy=(.885,.485),xytext=(.86,.485),arrowprops={'arrowstyle':'->','color':'#285c8c'})
ax.text(.37,.035,'↓ estacional',ha='center',fontsize=9,color='#238565');ax.text(.52,.035,'↑ tendencia',ha='center',fontsize=9,color='#cb7b27')
fig.suptitle('TimeMixer · réplica externa ETTh1 unificada L96/H96',fontsize=14)
fig.text(.02,.015,'Esquema propio de Jassan Arteta basado en el paper y código de autores. Canales con pesos compartidos; no es el adaptador Bitcoin D7.',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(out/'timemixer_replication_architecture.png',dpi=160,bbox_inches='tight');fig.savefig(out/'timemixer_replication_architecture.svg',bbox_inches='tight',metadata={'Date':None});plt.close(fig)
p=out/'timemixer_replication_architecture.svg';p.write_text('\n'.join(l.rstrip() for l in p.read_text().splitlines())+'\n')
