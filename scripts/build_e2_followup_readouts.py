"""Consolidate follow-up evidence without fitting or scientific recalculation."""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import e2_corrections_budget as budget
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'book_e2/_static/e2_closure_followup_20261116'
OLD=ROOT/'book_e2/_static/e2_scientific_closure_20261116/delivery/appendix_B_current.json'
def read(p):return json.loads(p.read_text())
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=BASE/'delivery'
 old=read(OLD); current=json.loads(json.dumps(old))
 inf=BASE/'inference/recovery_v3/result_v1/summary.json'
 neg=BASE/'audit/summary.json'
 short=BASE/'shortcut/study_v1/statistics/summary.json'
 hpo=BASE/'hpo/descriptive_summary.json'
 rep=BASE/'reproduction/summary.json'
 for p in (inf,neg,short,hpo,rep):
  if not p.is_file():raise ValueError('Follow-up component missing: '+str(p))
 by={r['id']:r for r in current['items']}
 rel=lambda p:p.relative_to(ROOT).as_posix()
 by['B03']['evidence']+=' Suplemento posterior: revisión estática de dos redes, seis nuevos ajustes con etiquetas permutadas y dos funciones de inicialización evaluadas. Todos los nuevos resultados se conservan.'
 by['B03']['evidence_paths']+=';'+rel(neg)+';'+rel(BASE/'audit/trace_review.json')+';'+rel(BASE/'audit/initialization/result.json')
 by['B03']['remaining_limit']='La equivalencia media finita de RMSE con la constante dentro de ±δ no se establece en TSMixer ni iTransformer (TOST/Holm2 p=1). Persiste señal; no se certifica rendimiento cercano al azar de las trece redes. No se encontró una vía explícita de etiquetas futuras en el código revisado, pero eso no prueba ausencia general de fuga. La selección HPO no se repitió bajo permutación.'
 by['B06']['evidence']+=' Las seis configuraciones comunes por red sí acreditan la alternativa de igualdad de plazas de §11.1. Se añaden gráficos de puntuaciones y asociaciones descriptivas con identificabilidad delimitada.'
 by['B06']['evidence_paths']+=';'+rel(hpo)
 by['B06']['remaining_limit']='La guía permite igual número de configuraciones o igual tiempo: no se exige igualdad simultánea de ambos. Permanecen el desenlace histórico desconocido, los límites de contabilidad global y comparabilidad con clásicos, y la amplitud/variación de selección HPO no acreditada. La heterogeneidad de segundos por sí sola no invalida la igualdad de plazas. Las asociaciones de seis tuplas no identifican importancia independiente.'
 by['B11']['evidence']+=' Se ejecutan comparación condicional OOF249–prueba608, descomposición finita de pérdidas con IC y diagnóstico de dependencia de grupos de entradas.'
 by['B11']['evidence_paths']+=';'+rel(inf)+';'+rel(short)
 by['B11']['remaining_limit']+=' La comparación OOF conserva selección condicionada y entrenamientos distintos; los componentes finitos no estiman variación entre mercados o particiones nuevas. La perturbación de volumen no identifica causalidad o señal espuria.'
 by['B21']['evidence']+=' Nuevo paquete con fuentes, datos y pesos permite reanálisis e inferencia real de 81 predictores en CPU fuera del repositorio, con recibo propio; separa el modo de entrenamiento de recetas fijas.'
 by['B21']['evidence_paths']=by['B21']['evidence_paths'].replace('source_snapshots/cpu_status_at_export.json','book_e2/_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/result_v1/export_receipt.json')+';'+rel(rep)
 by['B21']['remaining_limit']='El alcance portátil demostrado se lee en su recibo; no se atribuye una repetición completa nueva de HPO, robustez, ablaciones o toda la campaña histórica. Las pruebas específicas de entrenamiento no representan 78 refits ni una nueva A2 completa. B21 conserva alcance parcial y las versiones ZIP anteriores mantienen su snapshot.'
 for item in current['items']:
  for part in item['evidence_paths'].split(';'):
   if part and not (ROOT/part).exists():raise ValueError('Missing evidence '+part)
 counts=Counter(r['status'] for r in current['items'])
 current.update(version='e2-followup-delivery-readouts-v1',reviewed_source_commit=None,reviewed_source_commit_scope='Content and component hashes bound by follow-up manifests; no invented commit identity',base_B_version=old['version'],base_B_preserved=rel(OLD),counts={k:counts.get(k,0) for k in current['statuses']},operations={'scientific_recalculation':False,'new_fits':0,'new_forwards':0,'historical_bytes_overwritten':False})
 current['previous_versions_preserved']=list(dict.fromkeys(current['previous_versions_preserved']+[rel(OLD)]))
 dump(out/'appendix_B_current.json',current)
 with (out/'appendix_B_current.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(current['items'][0]));w.writeheader();w.writerows(current['items'])
 with budget.locked_ledger(BASE/'compute_ledger.jsonl') as (rows,_):account=budget.summarize(rows)
 if account['open_reservations']:raise ValueError('Settle scientific reservations before final export')
 dump(out/'budget_final.json',account)
 state={'version':'e2-followup-delivery-readouts-v1','observed_at_utc':datetime.now(timezone.utc).isoformat(),'appendix_B_counts':current['counts'],'new_negative_control_fits':read(neg)['completed_fits'],'original_13_model_negative_acceptance':False,'conditional_inference':read(inf),'input_dependency':read(short),'HPO_descriptive_scope':read(hpo),'HPO_timing_pilot':read(BASE/'hpo/pilot_v3/pilot_summary.json'),'portable_reproduction':read(rep),'original_A2_changed':False,'original_RQ1_changed':False,'NAS_not_applicable':True,'native_original_bibliography_recovery_closed':False,'budget':account,'scope':'Component results, including adverse findings, are retained. No percentage complete or grade is inferred.'}
 dump(out/'closure_state.json',state)
 files={p.relative_to(BASE).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(BASE.rglob('*')) if p.is_file() and p.name!='followup_manifest.json'}
 dump(out/'followup_manifest.json',{'version':'e2-followup-content-manifest-v1','files':files,'original_A2_RQ1_replaced':False})
 print(json.dumps({'files':len(files),'B_counts':current['counts'],'remaining_seconds':account['remaining_seconds']},ensure_ascii=False))
if __name__=='__main__':main()
