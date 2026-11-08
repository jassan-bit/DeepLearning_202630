"""Publish only aggregate screening progress and curated brief source extractions."""
from pathlib import Path
import argparse,collections,csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'references/literature_e2/screening_followup_20261108'
DEST=ROOT/'book_e2/_static/literature_screening_followup_20261108'
FILES=['summary.json','source_reviews.json','next_phases.json','README.md']
COUNTS={'EXCLUSION_NUCLEO_PRECIO':184,'SECUNDARIO_TRANSFERIBLE':98,'INCIERTO_CONSERVAR':12,'CANDIDATO_DIRECTO':1}
FULL={'E2-0012','E2-0025','E2-0059','E2-0253','E2-0867'}
def dump(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def from_private(base):
 base=base.resolve();assert not base.is_relative_to(ROOT)
 paths={'remaining':base/'revision_restantes_privada/295_decisiones_titulo_resumen.csv','previous':base/'revision_prioritarios_privada/42_auditoria_exclusiones.csv','recovery':base/'texto_completo_privado/registro_40_candidatos.csv','priority':base/'revision_prioritarios_privada/59_decisiones_titulo_resumen.csv'}
 r,p,f,q=[rows(paths[k]) for k in ['remaining','previous','recovery','priority']]
 initial=rows(ROOT/'references/literature_e2/native_20261108/native/cribado_metadatos.csv')
 ids=lambda x:{a['study_id'] for a in x}
 assert len(r)==len(ids(r))==295 and len(p)==len(ids(p))==42 and not ids(r)&ids(p)
 assert ids(r)|ids(p)=={x['study_id'] for x in initial if x['screening_class']=='EXCLUSION_PROPUESTA_PRECIO'}
 assert dict(collections.Counter(x['decision'] for x in r))==COUNTS
 assert {x['study_id'] for x in r if x['decision']=='CANDIDATO_DIRECTO'}=={'E2-0449'}
 assert next(x for x in r if x['study_id']=='E2-0538')['decision']=='INCIERTO_CONSERVAR'
 assert len(f)==len(ids(f))==40 and ids(f)=={x['study_id'] for x in q if x['decision_preliminar']=='CANDIDATO_DIRECTO_A_TEXTO_COMPLETO'}
 assert {x['study_id'] for x in f if x['estado'].startswith('TEXTO_COMPLETO')}==FULL
 assert sum(x['estado']=='PENDIENTE_NO_BUSCADO' for x in f)==28
 s={'review_date':'2026-11-08','method':'individual_title_abstract_reading_assisted_by_AI','human_independent_double_review':False,'global_review_complete':False,'final_included_studies':None,'remaining_exclusions':{'reviewed':295,'decisions':COUNTS,'conserved_for_some_role':111,'direct_candidate_ids':['E2-0449'],'priority_uncertain_ids':['E2-0538']},'cumulative_exclusion_review':{'initial_universe':337,'previous_review':42,'subsequent_review':295,'individually_unread':0,'secondary_role_criteria_harmonized_between_rounds':False,'global_error_rate_estimated':False},'fulltext_recovery':{'original_candidate_cohort':40,'examined_for_recovery':12,'fulltexts_reviewed':5,'partial_or_not_recovered':7,'not_individually_searched':28,'fulltext_reviewed_ids':sorted(FULL),'new_rescued_candidate_in_this_cohort':False,'reproduction_or_code_validation_performed':False},'private_source_fingerprints':{k:{'sha256':sha(v),'bytes':v.stat().st_size} for k,v in paths.items()},'previous_ledgers_modified':False,'source_abstracts_or_originals_published':False}
 (SOURCE/'summary.json').write_bytes(dump(s))
def validate():
 s=json.loads((SOURCE/'summary.json').read_text());a=s['remaining_exclusions'];c=s['cumulative_exclusion_review'];f=s['fulltext_recovery']
 assert a['reviewed']==sum(a['decisions'].values())==295 and a['decisions']==COUNTS
 assert a['conserved_for_some_role']==98+12+1==111
 assert c['initial_universe']==c['previous_review']+c['subsequent_review']==337 and c['individually_unread']==0
 assert not c['secondary_role_criteria_harmonized_between_rounds'] and not c['global_error_rate_estimated']
 assert f['fulltexts_reviewed']+f['partial_or_not_recovered']==f['examined_for_recovery']==12
 assert f['examined_for_recovery']+f['not_individually_searched']==f['original_candidate_cohort']==40
 assert set(f['fulltext_reviewed_ids'])==FULL and not f['new_rescued_candidate_in_this_cohort']
 assert not s['global_review_complete'] and s['final_included_studies'] is None and not s['human_independent_double_review']
 reviews=json.loads((SOURCE/'source_reviews.json').read_text())['studies'];assert {x['study_id'] for x in reviews}==FULL
 allowed={'study_id','title','doi','source_url','access_level','target_and_data','reported_result','locator','limits','unverified_fields'}
 for x in reviews:assert set(x)==allowed and x['locator'] and x['access_level']=='fulltext_method_empirical_review_not_replication'
 assert json.loads((SOURCE/'next_phases.json').read_text())['status']=='editorial_plan_not_preregistration'
 assert {p.name for p in SOURCE.iterdir() if p.is_file()}<=set(FILES+['manifest.json'])
 return s

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private-base',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
 if a.private_base:assert not a.check;from_private(a.private_base)
 validate();m={'schema':'e2_aggregate_followup_allowlist_v1','approved_public_files':sorted(FILES),'files':{n:{'sha256':sha(SOURCE/n),'bytes':(SOURCE/n).stat().st_size} for n in sorted(FILES)}}
 if a.check:assert (SOURCE/'manifest.json').read_bytes()==dump(m)
 else:(SOURCE/'manifest.json').write_bytes(dump(m))
 for n in FILES+['manifest.json']:
  q=DEST/n;b=(SOURCE/n).read_bytes()
  if a.check:assert q.read_bytes()==b,n
  else:q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
 assert {p.name for p in DEST.iterdir()}==set(FILES+['manifest.json'])
 print('PASS: 295 subsequent +42 prior exclusions; recovery12=5 complete+7 partial;28 not searched; final eligibility open.')
if __name__=='__main__':main()
