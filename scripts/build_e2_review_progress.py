"""Build aggregate-only review progress from private tables, or validate public output."""
from pathlib import Path
import argparse, collections, csv, hashlib, json, random, re
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'references/literature_e2/title_abstract_review_20261108'
DEST=ROOT/'book_e2/_static/literature_review_progress_20261108'
FILES=['summary.json','method.json','alerts.json','next_phases.json','README.md']
DECISIONS={
 'priority': {'CANDIDATO_DIRECTO_A_TEXTO_COMPLETO':40,'INCIERTO_CONSERVAR_A_TEXTO_COMPLETO':17,'SECUNDARIO_NO_EVIDENCIA_DIRECTA':2},
 'audit': {'MANTENER_EXCLUSION_DEL_NUCLEO_DIRECTO':40,'RESCATAR_SECUNDARIO_VOLATILIDAD':1,'RESCATAR_INCIERTO':1}}

def dump(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate():
 s=json.loads((SOURCE/'summary.json').read_text())
 assert s['priority_review']['reviewed']==sum(s['priority_review']['decisions'].values())==59
 assert s['priority_review']['decisions']==DECISIONS['priority']
 a=s['exclusion_audit'];assert a['universe']==337 and a['reviewed']==42 and a['not_individually_reviewed']==295
 assert a['reviewed']+a['not_individually_reviewed']==a['universe']
 assert a['decisions']==DECISIONS['audit'] and sum(a['decisions'].values())==42
 assert a['sampling']==dict(random_without_replacement=30,risk_title_matches=15,overlap=3,additional_risk=12,seed=20261108)
 assert 30+15-3==a['reviewed']
 assert a['rescued_secondary_ids']==['E2-0807'] and a['rescued_uncertain_ids']==['E2-0993']
 assert s['final_fulltext_inclusions'] is None
 assert s['review_method']=='AI_assisted_individual_title_abstract_review'
 assert not s['human_independent_double_screening'] and not s['review_complete']
 method=json.loads((SOURCE/'method.json').read_text())
 assert method['mixed_sample_error_rate_extrapolation_allowed'] is False
 phases=json.loads((SOURCE/'next_phases.json').read_text())
 assert phases['status']=='editorial_plan_not_preregistration' and not phases['market_training_started_by_this_update']
 assert {p.name for p in SOURCE.iterdir() if p.is_file()}==set(FILES+['manifest.json']) or {p.name for p in SOURCE.iterdir() if p.is_file()}==set(FILES)
 return s

def build_from_private(folder):
 folder=Path(folder).resolve()
 assert not folder.is_relative_to(ROOT), 'Private inputs must remain outside checkout'
 specs=[('priority','59_decisiones_titulo_resumen.csv'),('audit','42_auditoria_exclusiones.csv')]
 provenance={};data={}
 for label,name in specs:
  path=folder/name
  with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
  assert len({r['study_id'] for r in rows})==len(rows)
  counts=dict(collections.Counter(r['decision_preliminar'] for r in rows));assert counts==DECISIONS[label]
  data[label]=rows;provenance[label]={'sha256':digest(path),'bytes':path.stat().st_size,'rows':len(rows),'source_body_private':True}
 assert dict(collections.Counter(r['muestreo'] for r in data['audit']))=={'ALEATORIA_SEMILLA_20261108':30,'ADICIONAL_RIESGO_TITULO':12}
 with (ROOT/'references/literature_e2/native_20261108/native/cribado_metadatos.csv').open(encoding='utf-8-sig',newline='') as f: initial=list(csv.DictReader(f))
 assert {r['study_id'] for r in data['priority']}=={r['study_id'] for r in initial if r['screening_class']=='PRIORIDAD_PRINCIPAL'}
 excluded=[r for r in initial if r['screening_class']=='EXCLUSION_PROPUESTA_PRECIO']
 sampled={r['study_id'] for r in random.Random(20261108).sample(excluded,30)}
 risk={r['study_id'] for r in excluded if re.search('volatility|variance|fluctuation|risk',r['title'],re.I)}
 assert sampled=={r['study_id'] for r in data['audit'] if r['muestreo']=='ALEATORIA_SEMILLA_20261108'}
 assert len(risk)==15 and len(sampled & risk)==3
 assert sampled | risk == {r['study_id'] for r in data['audit']}

 resc=lambda decision:sorted(r['study_id'] for r in data['audit'] if r['decision_preliminar']==decision)
 s={'review_date':'2026-11-08','review_method':'AI_assisted_individual_title_abstract_review','human_independent_double_screening':False,'review_complete':False,'final_fulltext_inclusions':None,'priority_review':{'reviewed':59,'decisions':DECISIONS['priority']},'exclusion_audit':{'universe':337,'reviewed':42,'not_individually_reviewed':295,'sampling':dict(random_without_replacement=30,risk_title_matches=15,overlap=3,additional_risk=12,seed=20261108),'decisions':DECISIONS['audit'],'rescued_secondary_ids':resc('RESCATAR_SECUNDARIO_VOLATILIDAD'),'rescued_uncertain_ids':resc('RESCATAR_INCIERTO')},'private_source_fingerprints':provenance,'prior_screening_ledger_modified':False,'native_corpus_counts_changed':False,'external_bibliographic_verification_performed':False,'fulltext_review_performed_in_this_step':False}
 (SOURCE/'summary.json').write_bytes(dump(s))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private-dir',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
 if a.private_dir:
  assert not a.check,'Use private build and public validation as separate steps'
  build_from_private(a.private_dir)
 validate()
 m={'schema':'e2_aggregate_review_allowlist_v1','approved_public_files':sorted(FILES),'files':{n:{'bytes':(SOURCE/n).stat().st_size,'sha256':digest(SOURCE/n)} for n in sorted(FILES)}}
 if a.check:assert (SOURCE/'manifest.json').read_bytes()==dump(m)
 else:(SOURCE/'manifest.json').write_bytes(dump(m))
 for n in FILES+['manifest.json']:
  b=(SOURCE/n).read_bytes();q=DEST/n
  if a.check:assert q.read_bytes()==b,n
  else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 assert {p.name for p in DEST.iterdir()}==set(FILES+['manifest.json'])
 print('PASS: 59 priority reviews; 42/337 exclusions audited; 295 pending; no final inclusion claim.')
if __name__=='__main__':main()
