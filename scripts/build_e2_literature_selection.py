"""Build the proposed E2 model catalogue from curated evidence and source records.

No search, paper-text inference, model fitting or benchmark scoring occurs here.
WoS citation counts belong to representative studies, not automatically to the
architectural source. An unverified performance component remains an interval.
"""
from __future__ import annotations
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import textwrap
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-e2-literature')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/cache-e2-literature')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
matplotlib.rcParams['svg.hashsalt'] = 'e2-top10-20261105'

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'references/literature_e2/top10_20261105'
FULLTEXT = ROOT/'references/literature_e2/fulltext_20261105'
OUT = ROOT/'book_e2/_static/literature_selection'
HISTORICAL_REVIEW_DATE = '2026-11-05'
PAPER_GUIDE_URL = 'https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/05_paper_guia.html'
HISTORICAL_EXECUTION_SCOPE = (
 'Historical literature-review stage dated 2026-11-05 only. Zero fit and numerical-reproduction '
 'counters are not current project-wide totals. Subsequent external ETTh1 execution is documented in chapter 5; '
 'the Bitcoin D7 benchmark and its adapters remain separate.'
)
WEIGHTS = {'relevance': .35, 'code': .20, 'recency': .15, 'citations': .15, 'performance': .15}
FORMULAS = {
 'TSMixer': r'T=X+\mathcal M_t(\mathcal N(X)^\top)^\top,\qquad Z=T+\mathcal M_c(\mathcal N(T)).',
 'TiDE': r'z=\operatorname{Enc}(\operatorname{vec}(X)),\qquad r=\operatorname{Dec}(z),\qquad \widehat y=g_\sigma(r).',
 'TimeMixer': r'X_s=\operatorname{Down}_s(X),\quad (S_s,T_s)=\operatorname{Decomp}(X_s),\quad \widehat y=g_\sigma(\{\operatorname{Mix}_s(S_s,T_s)\}_s).',
 'TimesNet': r'p_i=\lfloor L/f_i\rfloor,\quad Z=\sum_i a_i\operatorname{Reshape}_{p_i}^{-1}(\operatorname{Conv2D}_i(\operatorname{Reshape}_{p_i}(X))).',
 'PatchTST': r'P_c=\operatorname{Patch}(X_{:,c}),\quad Z_c=\operatorname{Encoder}(P_cW_p+E),\quad \widehat y=g_\sigma(Z_1,\ldots,Z_5).',
 'Informer': r'\operatorname{Attn}(\overline Q,K,V)=\operatorname{softmax}(\overline QK^\top/\sqrt d)V,\qquad \widehat y=g_\sigma(Z).',
 'Autoformer': r'R_{QK}=\operatorname{IFFT}(\operatorname{FFT}(Q)\overline{\operatorname{FFT}(K)}),\quad Z=\sum_{\tau\in\operatorname{TopK}(R)}a_\tau\operatorname{Roll}_{\tau}(V).',
 'iTransformer': r'H^{(0)}=X^\top W_e,\qquad H^{(\ell+1)}=\operatorname{Encoder}(H^{(\ell)}),\qquad \widehat y=g_\sigma(H).',
 'Crossformer': r'H_{c,s}=\operatorname{DSW}(X_{s:s+p,c}),\quad Z=\operatorname{Attn}_{\rm dim}(\operatorname{Attn}_{\rm time}(H)),\quad\widehat y=g_\sigma(Z).',
 'S-Mamba': r'h_i=\overline A_i h_{i-1}+\overline B_i e_i,\quad z_i=C_i h_i,\quad Z=\operatorname{Mamba}(E)+\operatorname{Flip}(\operatorname{Mamba}(\operatorname{Flip}(E))).',
}
CORES = {
 'TSMixer':'MLP temporal residual\nMLP de variables residual', 'TiDE':'Encoder y decoder densos\nAdaptación covariate-only',
 'TimeMixer':'Descomposición multiescala\nMezcla y fusión de predictores', 'TimesNet':'FFT y periodos compatibles\nConvoluciones sobre variación 2D',
 'PatchTST':'Parches cortos por canal\nEncoder con atención', 'Informer':'Selección ProbSparse\nEncoder y decoder observado',
 'Autoformer':'Descomposición + autocorrelación\nTendencia proyectada al objetivo', 'iTransformer':'Tokens de las cinco variables\nEncoder con atención entre variables',
 'Crossformer':'Segmentos por variable\nAtención temporal y dimensional', 'S-Mamba':'Tokens de variables observadas\nEstados selectivos bidireccionales',
}

# These are scoped conclusions from the complete-PDF extraction, rather than
# extrapolations from another variant's error bars or from a public repository.
QUALITY_NOTES = {
 'TSMixer': 'La fila LTSF seleccionada no declara repeticiones ni dispersión. Las cinco semillas y media±DE de M5 pertenecen a otra tarea. El PDF declara estandarización global y RevIN, pero no explicita el ajuste del scaler sólo en entrenamiento; parte de los comparadores se importa de otro artículo.',
 'TiDE': 'Declara cinco ejecuciones independientes. La tabla 5 (PDF p.19) proporciona errores estándar para ETTh1/H96: MSE 0.0003 y MAE 0.0002; no son desviaciones estándar. La normalización con media/DE del periodo de entrenamiento está explícita. Los presupuestos HPO de todos los comparadores no se igualan documentalmente.',
 'TimeMixer': 'Declara tres repeticiones. La media±DE de la tabla 9 corresponde al promedio de cuatro horizontes, no a esta fila H96. Sus afirmaciones de IC/significancia no incluyen prueba identificada, p-valores ni extremos reproducibles. La configuración unificada alinea contexto, batch y épocas, sin demostrar igual presupuesto de búsqueda.',
 'TimesNet': 'Declara tres repeticiones; la dispersión de esta fila H96 no está reportada. Series Stationarization es una normalización interna y no acredita por sí sola un scaler global ajustado sólo en entrenamiento. Los controles compartidos de entrada/salida no demuestran igual presupuesto HPO.',
 'PatchTST': 'La tabla principal usa semilla 2021. La robustez con cinco semillas 2019–2023 y media±DE se estudia para la variante /42; no se atribuye su dispersión a /64. La normalización de instancia está descrita, pero el scaler global y su población de ajuste no quedan explícitos en el PDF.',
 'Informer': 'Declara cinco ejecuciones con desplazamiento de los conjuntos de entrenamiento/validación; no equivale a cinco semillas independientes sobre una única partición fija. No reporta dispersión para la fila seleccionada. La estandarización está declarada, sin precisar ajuste sólo en entrenamiento ni inversión al evaluar.',
 'Autoformer': 'Declara tres ejecuciones y presenta DE para otros casos, como ETTm2/H96; no se transfiere esa incertidumbre a ETTh1/H24. La normalización global del benchmark no está reportada en el PDF. Los resultados corresponden a la versión speedup.',
 'iTransformer': 'El análisis de robustez incluye cinco semillas en otros datasets; su tabla 5 no contiene ETTh1 y no acredita incertidumbre de esta fila. La atención entre variables tiene coste cuadrático en el número de variables, no en el número de fechas. Un contexto común no prueba igualdad de presupuesto HPO.',
 'Crossformer': 'Declara media de cinco ejecuciones y normalización con media/DE de entrenamiento; no proporciona IDs de semillas ni dispersión para esta fila. El contexto ganador exacto no está reportado y no se infiere del ejemplo del README. Algunos comparadores se importan de Informer y utilizan covariables calendario distintas.',
 'S-Mamba': 'En las 14 páginas del preprint suministrado no se reportan repeticiones, semillas, dispersión, prueba formal, optimizador ni presupuesto HPO. La normalización dentro del encoder no demuestra un scaler global train-only. Se leyó el preprint v3 de 2024, no el cuerpo de la publicación editorial de 2025.',
}

ADAPTER_NOTES = {
 'TimesNet': 'La ecuación 1 del artículo usa techo para el periodo y el código examinado usa división entera (piso); la fórmula anterior describe ese código. FFT_for_Period promedia amplitudes sobre el batch. Para no compartir información entre orígenes temporales, la adaptación propuesta debe seleccionar periodos por ventana o inferir con batch=1, excluir explícitamente la frecuencia cero y limitar k≤⌊L_eff/2⌋. Son controles propuestos, aún no ejecutados.',
 'Autoformer': 'Roll es una operación circular sobre representaciones de historia observada, no una garantía de causalidad por posición. La rama de tendencia en unidades de covariables también debe proyectarse de forma coherente al objetivo σ₇ antes de aplicar la escala Y de entrenamiento.',
 'TiDE': 'La formulación original incorpora historia del objetivo y covariables futuras; la versión covariate-only bajo el contrato vigente es una adaptación explícita que necesita validación, no una reproducción idéntica.',
 'S-Mamba': 'La bidireccionalidad recorre tokens de las cinco variables ya observadas, no fechas futuras. El código registra módulos que no participan en forward; al implementarlo habrá que distinguir parámetros registrados y activos. No se verificó una licencia raíz que autorice reutilizar ese código.',
}
COMPLEXITY_ES = {
 'TSMixer': 'Estimación estructural propia por bloque O(C L² + L C F), más la proyección; L=7 y C=5. No es un recuento medido de FLOPs.',
 'TiDE': 'Estimación propia Σℓ O(dℓ dℓ₊₁) sobre capas densas, más proyecciones de variables. Los anchos no están fijados; no se infieren parámetros ni FLOPs numéricos.',
 'TimeMixer': 'Estimación propia O(Σₛ C Lₛ² + costes de FFN de variables), sumando las escalas. No es una medición de FLOPs.',
 'TimesNet': 'Estimación propia O(L d log L) para FFT más O(Σᵢ Aᵢ kᵢ² d_in d_out) para convoluciones 2D sobre rejillas con relleno. No se ha medido el coste numérico.',
 'PatchTST': 'Estimación propia O(C P² d + C P d²), con P parches. Reducir tokens puede ahorrar atención, pero con L=7 no demuestra una ventaja.',
 'Informer': 'El mecanismo original propone atención O(L log L); el modelo completo añade proyecciones/FFN O(L d²) y decoder. No se deduce superioridad de velocidad con L=7.',
 'Autoformer': 'Correlación basada en FFT O(L d log L), más proyecciones/FFN O(L d²). Es una síntesis estructural propia, no un coste medido del modelo completo.',
 'iTransformer': 'Estimación propia O(C L d + C² d + C d²) para embedding/bloque: atención entre C variables, no entre L fechas.',
 'Crossformer': 'Estimación propia O(C S² d + S C R d) para atención temporal y dimensional con routers; S segmentos y R routers. Proyecciones/FFN añaden coste.',
 'S-Mamba': 'Embedding O(C L d) y scan selectivo lineal en C tokens de variables para estado/ancho fijos, más FFN/proyecciones. Esta implementación no recorre simplemente L tokens cronológicos.',
}
RISK_ES = {
 'TSMixer': 'La fuente elegida es Google/Chen All-MLP TSMixer, distinta de variantes IBM con nombre parecido. Las capas y la receta del adaptador aún deben fijarse antes de entrenarlo.',
 'TiDE': 'La interfaz original no se ejecuta sin cambios bajo el contrato heredado. La cabeza/skip escalar y retirar historia del objetivo deben declararse como adaptaciones.',
 'TimeMixer': 'Las escalas y longitudes predeterminadas para historias de 96/336 pasos no sirven automáticamente para D7. No se añaden variables futuras ni se invierte la escala OHLCV como si el objetivo fuera una de sus columnas.',
 'TimesNet': 'El README original deriva la implementación a Time-Series-Library: se conservan ambos repositorios. Periodos inferidos por FFT fuera del soporte observado no prueban memoria de largo plazo.',
 'PatchTST': 'Los parches predeterminados de 16/32 pasos exceden D7. Una nueva cabeza que combine canales es una adaptación que debe declararse y validarse.',
 'Informer': 'La licencia raíz observada es Apache-2.0 y el badge del README indica CC BY-NC-SA: se distinguen código, datos y badge. Debe comprobarse que el distilling sea válido con siete pasos.',
 'Autoformer': 'Fijar c_out=1 puede difundir una tendencia OHLCV en unidades incompatibles con σ₇. El regresor de volatilidad basado sólo en covariables es una adaptación explícita.',
 'iTransformer': 'La ventaja asintótica depende del eje de atención y no acredita una mejora para sólo cinco variables. La lectura escalar debe preceder a cualquier desnormalización por canal.',
 'Crossformer': 'Segmentos de seis pasos y merges de cuatro requieren rediseñar una jerarquía válida con pocos segmentos. El DOI editorial del original no está identificado; se conserva su enlace OpenReview.',
 'S-Mamba': 'No se verificó licencia raíz. Un docstring copia una referencia a iTransformer; la identidad se contrasta con README y registro editorial. Los módulos registrados pero inactivos exigen una política explícita al contar parámetros.',
}
VARIANT_ES = {
 'TSMixer':'TSMixer básico multivariante', 'TiDE':'TiDE completo, RevIN activado, todos los residuales; tabla 2 con configuración de tabla 8',
 'TimeMixer':'TimeMixer unificado, M=3 y cuatro escalas 96/48/24/12', 'TimesNet':'TimesNet con Inception compartido',
 'PatchTST':'PatchTST/64 supervisado, parche 16 y salto 8', 'Informer':'Informer con ProbSparse y distilling, columna Informer sin daga',
 'Autoformer':'Autoformer speedup, dos capas de encoder y una de decoder', 'iTransformer':'iTransformer completo, columna original Ours',
 'Crossformer':'Crossformer original completo DSW+TSA+HED sin covariables', 'S-Mamba':'S-Mamba principal con bi-Mamba VC y FFN TD',
}
SCALING_ES = {
 'TSMixer':'estandarización global por covariable y RevIN; se evalúa sin invertir la escala global, cuyo ajuste train-only no se explicita',
 'TiDE':'estandarización con media/DE del periodo de entrenamiento y RevIN activado para ETTh1',
 'TimeMixer':'escala/unidades exactas de las métricas no reportadas en el PDF',
 'TimesNet':'Series Stationarization declarada; scaler global y unidades exactas de estas métricas no reportados',
 'PatchTST':'normalización de instancia declarada; scaler global y unidades exactas de las métricas no reportados',
 'Informer':'estandarización media cero/DE uno declarada; ajuste train-only e inversión al evaluar no explicitados',
 'Autoformer':'normalización global del benchmark no reportada en el PDF',
 'iTransformer':'preprocesamiento del benchmark remitido a TimesNet; no son errores de precio bruto ni de Bitcoin',
 'Crossformer':'normalización con media/DE de entrenamiento declarada en §4.1; la fila se conserva en la escala del benchmark',
 'S-Mamba':'normalización interna del encoder declarada; scaler global y escala final de estas métricas no reportados',
}

def digest(path): return sha256(Path(path).read_bytes()).hexdigest()
def write_json(value,name): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(rows,name):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def diagram(model):
 fig,ax=plt.subplots(figsize=(12,2.3));ax.set(xlim=(-.025,1.025),ylim=(0,1));ax.axis('off')
 labels=['Historia conocida\n7 días × 5 OHLCV',CORES[model],'Representación latente\nantes de desnormalizar OHLCV','Lectura escalar σ₇\nescala Y de entrenamiento']
 xs=[.01,.25,.51,.76];widths=[.20,.23,.22,.22]
 for i,(x,w,label) in enumerate(zip(xs,widths,labels)):
  ax.add_patch(FancyBboxPatch((x,.27),w,.46,boxstyle='round,pad=0.012',facecolor='#edf4fb',edgecolor='#285c8c'))
  ax.text(x+w/2,.50,label,ha='center',va='center',fontsize=10)
  if i<3:ax.annotate('',xy=(xs[i+1]-.014,.5),xytext=(x+w+.012,.5),arrowprops={'arrowstyle':'->','color':'#285c8c'})
 ax.set_title(model+' · esquema propio de adaptación al contrato de Bitcoin',fontsize=13)
 fig.text(.02,.03,'Esquema: Jassan Arteta. Fuente arquitectónica: código de autores citado en la ficha. Adaptación aún no implementada.',fontsize=8)
 fig.tight_layout(rect=(0,.07,1,1));folder=OUT/'diagrams';folder.mkdir(exist_ok=True)
 fig.savefig(folder/(model+'.png'),dpi=150,bbox_inches='tight')
 svg=folder/(model+'.svg');fig.savefig(svg,bbox_inches='tight',metadata={'Date':None});svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n');plt.close(fig)

def receipt_flow():
 fig,ax=plt.subplots(figsize=(10,5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
 items=[('8.802 ocurrencias recibidas','WoS:1.302 + Scopus:7.500'),('7.833 grupos bibliográficos','969 duplicaciones agrupadas; procedencia conservada'),('10 originales completos leídos · 215 páginas','Subcorpus arquitectónico dirigido; código de autores verificado'),('Cribado global pendiente de cierre','No declarar los 7.833 grupos como elegibles o incluidos')]
 for i,(title,detail) in enumerate(items):
  y=.80-i*.23
  ax.add_patch(FancyBboxPatch((.12,y-.055),.76,.15,boxstyle='round,pad=.014',facecolor='#edf4fb',edgecolor='#285c8c'))
  ax.text(.5,y+.04,title,ha='center',va='center',fontsize=12,weight='bold');ax.text(.5,y-.01,detail,ha='center',va='center',fontsize=9)
  if i<3:ax.annotate('',xy=(.5,y-.14),xytext=(.5,y-.075),arrowprops={'arrowstyle':'->','color':'#285c8c'})
 fig.suptitle('Flujo documentado de recepción, deduplicación y propuesta',fontsize=13)
 fig.text(.04,.018,'Cobertura Scopus:7.500/7.874; 374 registros no recibidos. El total nativo de Q2 en WoS permanece desconocido.',fontsize=8)
 fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(OUT/'receipt_flow.png',dpi=150,bbox_inches='tight')
 svg=OUT/'receipt_flow.svg';fig.savefig(svg,bbox_inches='tight',metadata={'Date':None});svg.write_text('\n'.join(l.rstrip() for l in svg.read_text().splitlines())+'\n');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 proposal=json.loads((SOURCE/'selection_inputs.json').read_text());sources=json.loads((SOURCE/'primary_source_registry.json').read_text())
 review=json.loads((FULLTEXT/'models.json').read_text())
 fulltexts={m['model']:m for m in review['models']}
 assert len(fulltexts)==10 and all(m['fulltext_verified'] for m in fulltexts.values())
 assert review['counts']['physical_PDF_pages']==215
 by_name={x['name']:x for x in sources['models']};assert len(by_name)==10
 with (ROOT/'book_e2/_static/literature_scopus/original_model_records.csv').open() as stream:
  original_citations={r['model']:r for r in csv.DictReader(stream)}
 rows=[];studies={};cards=[];full=[]
 cite_rates=[]
 for model in proposal['models']:
  c=original_citations[model['model']]
  cite_rates.append(float(c['Cited_by_Scopus_per_calendar_year']) if c['Cited_by_Scopus_per_calendar_year'] else None)
 max_rate=max(x for x in cite_rates if x is not None)
 for i,(m,rate) in enumerate(zip(proposal['models'],cite_rates),1):
  name=m['model'];p=by_name[name];s=m['representative_wos_study'];assert p.get('code_source')
  relevance=.8 if m['evidence_scope']=='applied_financial_volatility' else .5
  recency=max(0.,min(1.,(p['paper_year']-2019)/7))
  original=original_citations[name];citation_component=min(1.,rate/max_rate) if rate is not None else None
  lower=100*(WEIGHTS['relevance']*relevance+WEIGHTS['code']+WEIGHTS['recency']*recency+WEIGHTS['citations']*(citation_component if citation_component is not None else 0.))
  upper=lower+100*(WEIGHTS['performance']+(WEIGHTS['citations'] if citation_component is None else 0.))
  reviewed=fulltexts[name];metric=reviewed['canonical_metric_pair']
  rows.append({'model':name,'family':m['family_code'],'recommendation_status':'proposed_not_Bitcoin_validated','representative_UT':s['ut'],'representative_DOI':s['doi'],'support_scope':m['evidence_scope_spanish'],'representative_TC':s['times_cited_in_WoS_export'],'original_Scopus_EID':original['EID'],'original_Scopus_cited_by':original['Cited_by_Scopus'],'original_citations_per_calendar_year':rate,'original_source_year':p['paper_year'],'relevance_component':relevance,'verified_code_component':1,'recency_component':recency,'normalized_citation_component':citation_component,'comparable_performance_component':'not_comparable_to_Bitcoin_contract','preliminary_score_lower':lower,'preliminary_score_upper':upper,'family_constraint':'at least4families in10','code_commit':p['code_source']['commit'],'code_sha256':p['code_source']['sha256'],'code_url':p['code_source']['url'],'original_identifier':p.get('doi') or 'OpenReview:vSVLM2j9eie','full_manuscript_verified':True,'source_pdf_version':reviewed['source_version'],'source_pdf_sha256':reviewed['source_receipt']['sha256'],'source_pdf_pages':reviewed['source_receipt']['pdf_pages'],'target_adapter_implemented':False})
  studies[s['ut']]={k:s[k] for k in ['ut','doi','title','source','publication_year','times_cited_in_WoS_export','index_field_WE']}
  merged={**m,'primary_code_historical_code_only_stage':p,'preliminary_priority':rows[-1],'full_manuscript_verified':True,'fulltext_current_review':reviewed}
  merged['code_verified_in_initial_abstract_review']=merged.pop('code_verified_in_this_assessment',False)
  merged['architecture_identity_status_initial_abstract_review']=merged.pop('architecture_original_bibliography_status','not_verified')
  merged['code_verified_current_stage']=True
  merged['architecture_identity_status_current_stage']='Full original PDF and primary author code identified; adapted Bitcoin model not implemented'
  merged['execution_counter_scope']=HISTORICAL_EXECUTION_SCOPE
  merged['historical_literature_review_date']=HISTORICAL_REVIEW_DATE
  merged['later_external_execution_reference']=PAPER_GUIDE_URL
  full.append(merged)
  identifier='https://doi.org/'+p['doi'] if p.get('doi') else 'https://openreview.net/forum?id=vSVLM2j9eie'
  context=str(metric['context_steps']) if metric['context_steps'] is not None else 'NR (contexto ganador no reportado)'
  quality=reviewed['quality']
  selected_status='; '.join(label+': '+quality[key]['status'] for label,key in [('repeticiones','repetitions'),('semillas','seed_ids'),('dispersión de esta fila','variance_selected_pair'),('scaler train-only','train_only_scaler'),('prueba formal','formal_tests'),('igual presupuesto HPO','equal_hpo_budget')])
  cards.extend([f'### 3.3.{i} {name}', '',f"**Familia:** {m['family']}.", '',f"**Fuente arquitectónica:** [{p['paper_title']}]({identifier}), {p['paper_year']}. Identificador: {p.get('doi') or 'OpenReview vSVLM2j9eie; DOI no verificado'}. La naturaleza editorial o de preprint se conserva en el registro de fuentes.", '',f"**Texto completo leído:** [{reviewed['source_version']}]({metric['source_url']}), {reviewed['source_receipt']['pdf_pages']} páginas PDF; SHA-256 `{reviewed['source_receipt']['sha256']}`. Se distingue esta versión del registro editorial.", '',f"**Código primario:** [implementación fijada por commit]({p['code_source']['url']}), SHA-256 `{p['code_source']['sha256']}`. Licencia observada: {p.get('license','no verificada')}.", '',f'![Esquema propio de {name}](diagrams/{name}.png)', '', '**Síntesis matemática propia, basada en el código examinado:**', '', '$$',FORMULAS[name],'$$','',f"**Resultado publicado rastreable:** {metric['dataset']} multivariante; variante **{VARIANT_ES[name]}**; contexto L={context}, horizonte H={metric['horizon_steps']}; **MSE {metric['MSE']} y MAE {metric['MAE']}**, misma fila de la tabla {metric['table']}, **página PDF {metric['pdf_page']}**. Escala: {SCALING_ES[name]}. Son resultados externos, no una corrida de Bitcoin ni un ranking comparable entre estos diez artículos.", '',f"**Calidad y reproducibilidad:** {QUALITY_NOTES[name]}", '',f"Estado para la fila seleccionada: {selected_status}. `NR` significa no reportado en el PDF para ese alcance; `NV` significa no verificado y `partial`, evidencia parcial. El detalle y los localizadores están en la [matriz metodológica](../literature_fulltext/quality_matrix.csv).", '',f"**Respaldo aplicado adicional:** [{s['title']}](https://doi.org/{s['doi']}), `{s['ut']}`. Alcance: {m['evidence_scope_spanish']}. Sus {s['times_cited_in_WoS_export']} citas corresponden a ese estudio representativo, no automáticamente al original arquitectónico. El registro de apoyo procede de resumen/metadatos, salvo el preprint original de S-Mamba también leído.", '',f"**Adaptación propuesta:** {m['adaptation_D7_C5_to_sigma7_h1']}", '',f"**Complejidad:** {COMPLEXITY_ES[name]} El número de parámetros y FLOPs de la configuración ETTh1 seleccionada no está reportado; los tiempos/memorias de otras condiciones se conservan en la [matriz de eficiencia](../literature_fulltext/efficiency_matrix.csv), sin trasladarlos a D7 ni a Bitcoin.", '',f"**Limitación técnica:** {RISK_ES[name]}", '', *( [ADAPTER_NOTES[name], ''] if name in ADAPTER_NOTES else []), f'**Alcance histórico de ejecución:** la revisión bibliográfica del {HISTORICAL_REVIEW_DATE} no ajustó modelos ni efectuó reproducciones numéricas. La adaptación D7×5→σ₇ y su evaluación en Bitcoin siguen pendientes. La [réplica externa de TimeMixer y el piloto de tres modelos en ETTh1]({PAPER_GUIDE_URL}) pertenecen a la etapa posterior del capítulo 5.', ''])
  diagram(name)
 receipt_flow()
 write_csv(rows,'model_selection.csv');write_csv(list(studies.values()),'representative_studies.csv')
 write_json({'models':full,'scope':'Ten full original PDFs plus immutable author-code receipts; proposed adaptations, not a global closed systematic review or Bitcoin benchmark','historical_literature_review_date':HISTORICAL_REVIEW_DATE,'execution_counter_scope':HISTORICAL_EXECUTION_SCOPE,'later_external_execution_reference':PAPER_GUIDE_URL},'model_cards.json')
 (OUT/'model_cards.md').write_text('\n'.join(cards))
 write_json({**sources,'registry_scope':'Historical author-code-only receipt: zero PDF acquisitions at that earlier stage. Current complete-PDF evidence is in ../literature_fulltext/models.json and the model_cards fulltext_current_review fields.','current_fulltext_review_reference':'../literature_fulltext/models.json','current_fulltext_verified_models':10,'historical_literature_review_date':HISTORICAL_REVIEW_DATE,'later_external_execution_reference':PAPER_GUIDE_URL},'primary_source_registry.json')
 summary={'model_count':10,'families':{'MLP/mixing':3,'Convolution':1,'Attention':5,'State-space':1},'unique_representative_WoS_studies':len(studies),'primary_backbones_code_verified':10,'complete_manuscripts_verified_in_this_stage':10,'physical_PDF_pages_reviewed':215,'native_received_occurrences':8802,'deduplicated_bibliographic_groups':7833,'Scopus_received_records':7500,'Scopus_displayed_results':7874,'Scopus_unreceived_records':374,'original_architectural_records_matched_in_partial_Scopus_export':8,'new_deep_learning_fits':0,'original_papers_numerically_reproduced':0,'new_Bitcoin_results':0,'target_adapters_implemented':0,'minimum_benchmark_configurations':15,'inherited_references':['GRU','LSTM','TCN'],'additional_references':['Ridge','Persistence'],'weights':WEIGHTS,'score_scope':'Priority among ten proposed mechanisms, not a ranking of 7833 grouped records. External performance is extracted for all ten but not comparable to the Bitcoin contract: this contributes 15 points of interval, plus 15 if original citation counts are missing. C is capped at 1 against the maximum observed among eight original Scopus records.','citations_scope':'Original architectural-study Scopus Cited by where identified; citations remain distinct from WoS TC and representative applied-study counts; missing original counts are not zero','native_database_export_sources':2,'targeted_fulltext_architecture_review_complete':True,'full_systematic_review_closed':False,'input_hashes':{p.relative_to(ROOT).as_posix():digest(p) for p in [SOURCE/'selection_inputs.json',SOURCE/'primary_source_registry.json',ROOT/'book_e2/_static/literature_scopus/original_model_records.csv',FULLTEXT/'models.json',FULLTEXT/'fulltext_manifest.json']}}
 summary.update({'historical_literature_review_date':HISTORICAL_REVIEW_DATE,'top_level_execution_counters_scope':HISTORICAL_EXECUTION_SCOPE,'historical_execution_counters':{'new_deep_learning_fits':0,'original_papers_numerically_reproduced':0,'new_Bitcoin_results':0,'target_adapters_implemented':0},'later_external_execution_reference':PAPER_GUIDE_URL})
 write_json(summary,'summary.json')
 (OUT/'README.md').write_text('# Selección bibliográfica de diez modelos\n\nReconstruir desde la raíz: `python scripts/build_e2_literature_selection.py`, con el entorno de `requirements-eda.lock.txt`. Las decisiones curadas están en references/literature_e2/top10_20261105 y la extracción completa en references/literature_e2/fulltext_20261105. Este generador no ejecuta búsquedas ni entrena modelos. Los contadores de cero ajustes y reproducciones describen exclusivamente la etapa bibliográfica histórica del 05/11/2026; no son totales actuales del proyecto. La réplica externa de TimeMixer y el piloto de tres modelos en ETTh1 se documentan después en el [capítulo 5]('+PAPER_GUIDE_URL+'). No equivalen al benchmark Bitcoin D7.\n\nSe leyeron diez originales completos (215 páginas PDF), con versión, hash, métricas de una misma fila, páginas/tablas y evaluación metodológica. El registro primary_source_registry conserva como históricos los estados de la etapa anterior de código sin PDF; no representa la cobertura actual. La matriz actual está en ../literature_fulltext. Los PDF no se redistribuyen.\n\nEl puntaje es preliminar: las métricas externas ya están extraídas, pero no son comparables con el contrato de Bitcoin; el componente de desempeño conserva un intervalo de 15 puntos. Se añaden otros 15 puntos de incertidumbre cuando no se encontró el recuento de citas del original en la exportación parcial de Scopus (no equivale a cero citas).\n\nWoS y Scopus aportan 8.802 ocurrencias y 7.833 grupos bibliográficos. Scopus recibido: 7.500 de 7.874 resultados mostrados; faltan 374, que no son exclusiones. El cribado global y el PRISMA de elegibilidad no están cerrados. Los diez modelos son una selección dirigida con adaptación D7 pendiente, no un ranking exhaustivo ni ganadores de Bitcoin. Código público no equivale automáticamente a permiso de redistribución; las licencias observadas constan en el registro.\n')
 hashes={p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['provenance.json','literature_review_bundle.zip']}
 write_json({'generator':'scripts/build_e2_literature_selection.py','generator_sha256':digest(__file__),'source_inputs':summary['input_hashes'],'output_hashes':hashes,'source_code_copied':False,'raw_WoS_exports_copied':False},'provenance.json')
 print(json.dumps({'models':10,'families':4,'representative_studies':len(studies),'diagrams':10}))

if __name__=='__main__':main()
