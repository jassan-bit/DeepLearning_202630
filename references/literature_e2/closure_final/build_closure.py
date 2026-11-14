"""Build a compact public index from public prior decisions and own brief reviews."""
import csv,json
from pathlib import Path
B=Path(__file__).resolve().parent;R=B.parents[2]
def load(p):return json.loads(p.read_text())
old=load(R/'references/literature_e2/delivery_closure_20261114_v3/eligibility.json')
reviews={d['doi']:(p.name,d) for p in (B/'original_reviews').glob('*.json') for d in [load(p)]}
rows=[]
for x in old:
 name,d=reviews.get(x['doi'],(None,None));decision=d['decision'] if d else x['decision']
 rows.append({'study_id':x['study_id'],'doi':x['doi'],'title':x['title'],'decision':decision,'decision_reason':d['reason'] if d else x['decision_reason'],'source_url':'https://doi.org/'+x['doi'],'source_locators':d['locators'] if d else x['source_locators'],'public_evidence':('original_reviews/'+name) if d else 'references/literature_e2/delivery_closure_20261114_v3/eligibility.json#doi='+x['doi'],'outstanding_access_task':False,'outstanding_review_task':False,'role_adjudication':'closed_context' if decision=='contexto_elegibilidad_pendiente' else 'closed','applied_effect_identification':'unverified' if decision=='contexto_elegibilidad_pendiente' else 'not_pooled','author_clarification_optional':decision=='contexto_elegibilidad_pendiente','numerical_pooling_permitted':False})
(B/'eligibility.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
with (B/'eligibility.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
 for x in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in x.items()})
print('Public focal rows:',len(rows))
