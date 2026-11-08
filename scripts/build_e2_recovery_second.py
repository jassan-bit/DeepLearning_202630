"""Validate and mirror public aggregate recovery evidence; never read private abstracts."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'references/literature_e2/recovery_second_20261108'
DEST=ROOT/'book_e2/_static/literature_recovery_second_20261108'
FILES=['summary.json','source_reviews.json','next_phases.json','README.md']
def dump(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
def validate():
 s=json.loads((SOURCE/'summary.json').read_text())
 assert s['cohort']==40 and s['examined_cumulative']==22 and s['unsearched']==18
 assert s['new_examined']==s['new_editorial_fulltext']+s['new_preprint_fulltext']+s['new_partial']==10
 assert s['editorial_fulltext_cumulative']==6 and s['preprint_fulltext_editorial_version_pending']==2 and s['partial_cumulative']==14
 assert s['examined_cumulative']==6+2+14 and s['cohort']==s['examined_cumulative']+s['unsearched']
 assert len(set(s['new_examined_ids']))==10 and len(set(s['editorial_fulltext_ids']))==6 and s['preprint_fulltext_ids']==['E2-0285','E2-0867']
 assert not set(s['preprint_fulltext_ids'])&set(s['editorial_fulltext_ids'])
 for k in ['new_rescued_E2_0449_in_cohort','final_inclusion_decided','human_independent_double_review','global_review_complete','code_reproduced']:assert s[k] is False
 old=json.loads((ROOT/'references/literature_e2/screening_followup_20261108/summary.json').read_text())['fulltext_recovery']
 assert s['examined_cumulative']==old['examined_for_recovery']+s['new_examined']
 assert set(s['editorial_fulltext_ids'])-set(old['fulltext_reviewed_ids'])=={'E2-0064','E2-0593'}
 studies=json.loads((SOURCE/'source_reviews.json').read_text())['studies']
 assert {x['study_id'] for x in studies}=={'E2-0064','E2-0285','E2-0593'}
 allowed={'study_id','source_url','version','target','reported_result','locator','limits'}
 for x in studies:assert set(x)==allowed and x['source_url'].startswith('https://') and x['locator']
 assert json.loads((SOURCE/'next_phases.json').read_text())['status']=='editorial_plan_not_preregistration'
 assert {x.name for x in SOURCE.iterdir()}<=set(FILES+['manifest.json'])
 return s

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args();validate()
 m={'schema':'e2_recovery_second_allowlist_v1','approved_public_files':sorted(FILES),'files':{n:{'sha256':hashlib.sha256((SOURCE/n).read_bytes()).hexdigest(),'bytes':(SOURCE/n).stat().st_size} for n in sorted(FILES)}}
 if a.check:assert (SOURCE/'manifest.json').read_bytes()==dump(m)
 else:(SOURCE/'manifest.json').write_bytes(dump(m))
 DEST.mkdir(parents=True,exist_ok=True)
 for n in FILES+['manifest.json']:
  if a.check:assert (DEST/n).read_bytes()==(SOURCE/n).read_bytes(),n
  else:(DEST/n).write_bytes((SOURCE/n).read_bytes())
 assert {x.name for x in DEST.iterdir()}==set(FILES+['manifest.json'])
 print('PASS: recovery22/40 =6 editorial +2 preprint +14 partial;18 unsearched; no final inclusion.')
if __name__=='__main__':main()
