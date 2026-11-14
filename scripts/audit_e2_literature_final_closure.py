"""Verify focal literature closure and comparability without network or training."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'references/literature_e2/closure_final'
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(folder=DEFAULT,root=ROOT):
 b,r=Path(folder),Path(root);m=json.loads((b/'manifest.json').read_text())
 for name,item in m['files'].items():
  p=b/name;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],name
 for name,expected in m['repository_inputs_sha256'].items():assert sha(r/name)==expected,name
 d=json.loads((b/'eligibility.json').read_text());counts=Counter(x['decision'] for x in d)
 assert len(d)==len({x['doi'].casefold() for x in d})==30
 assert counts=={'incluido':17,'excluido':11,'contexto_elegibilidad_pendiente':2},counts
 assert all(not x['outstanding_access_task'] and not x['outstanding_review_task'] for x in d)
 assert all(x['numerical_pooling_permitted'] is False for x in d)
 assert all(x['role_adjudication']=='closed_context' and x['applied_effect_identification']=='unverified' for x in d if x['decision']=='contexto_elegibilidad_pendiente')
 received=json.loads((b/'received_originals.json').read_text())
 expected={'framework':'06f1c5a0fede379e3c0443346df20f913bf0c794c4baa9efe9418c2b1e44eccc','ceemdan':'6694ac423f35cfaee75f1dcf3b93b980e2bd4c4bdb7d2982d12fc0fdf25452b1','stacking':'145b71cd43c81eb50d8e53d27142f3ca951d9d5a5201fe737fc38e42fdeac442','jordan':'b906d2d5ec105e886baea5b79e94cae41d9fd00255d265a4524f743c99a6c3b0'}
 assert {x['key']:x['pdf_sha256'] for x in received}==expected
 q=json.loads((b/'quality/quality_extraction_matrix_17.json').read_text())['studies']
 assert len(q)==17 and {x['doi'].casefold() for x in q}=={x['doi'].casefold() for x in d if x['decision']=='incluido'}
 previous=json.loads((r/'references/literature_e2/delivery_closure_20261114_v3/quality/quality_extraction_matrix_14.json').read_text())['studies']
 assert [x['doi'] for x in q[:14]]==[x['doi'] for x in previous]
 for current,old in zip(q[:14],previous):
  for dim in ['target','data','horizon','split','metrics','models','repetitions','hpo','tests','code','leakage','fairness']:
   assert current['fields'][dim]['status']==old[dim].get('status','no_verificado')
 for path in b.rglob('*'):
  if path.is_file() and path.suffix in {'.json','.csv','.md','.txt'}:
   assert '/workspace/' not in path.read_text(), 'Local provenance path in public derivative: '+str(path)
 bounds=read(b/'selection/robust_global_P_bounds.csv')
 for x in bounds:
  w=float(x['P_weight']);base=100*(1-w)/.75*(.15*float(x['C'])+.15*float(x['U'])+.25*float(x['A'])+.20*float(x['F']))
  assert abs(base-float(x['score_lower']))<1e-10
  assert abs(base+100*w-float(x['score_upper']))<1e-10
 primary=[x for x in bounds if float(x['P_weight'])==.25]
 assert len(primary)==10 and all(int(x['best_rank'])==1 and int(x['worst_rank'])==10 for x in primary)
 strata=read(b/'selection/stratified_scores.csv')
 for x in strata:
  if x['P_within_stratum']:
   low,high=float(x['normalization_min']),float(x['normalization_max'])
   assert high>low
   assert abs((high-float(x['source_loss']))/(high-low)-float(x['P_within_stratum']))<1e-12
  else:assert x['P_unidentified_reason']
 history=read(b/'search_history/git_query_history_ledger.csv')
 assert {'WoS_Q1','WoS_Q2'}<={x['query_id'] for x in history}
 for x in history:
  if x['query_id'] in {'WoS_Q1','WoS_Q2'}:assert x['query'] and x['evidence_commit']=='f8fbbe6af3e91202f2eb6971d60102624ef38446'
 flow=json.loads((b/'flow.json').read_text())
 assert flow['global_PRISMA_closed'] is False and flow['global_included'] is None
 assert flow['originals_pending_access']==flow['documentary_reviews_pending']==0
 assert flow['current_abstracts_available_for_historical_groups']+flow['historical_groups_without_currently_linked_abstracts']==9559
 return {'status':'verified','focal_targets':30,'applied_included':17,'excluded_applied_core':11,'closed_contexts_unidentified_effect':2,'pending_access':0,'pending_focal_documentary_reviews':0,'quality_rows':17,'original_queries_recovered':True,'global_PRISMA_closed':False,'no_global_performance_rank_identified':True,'original_A2_preserved':True,'new_market_model_fits':0}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=DEFAULT);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
 print(json.dumps(verify(a.bundle,a.root),ensure_ascii=False,indent=2))
