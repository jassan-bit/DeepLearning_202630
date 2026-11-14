"""Restore title/abstract material to fixed groups. No eligibility classification."""
import argparse,collections,csv,hashlib,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-map',type=Path,required=True,help='Private JSON mapping source_id to supplied original file path')
parser.add_argument('--output-dir',type=Path,required=True,help='Private output directory outside this repository')
args=parser.parse_args()
OUT=args.output_dir.resolve()
if OUT.is_relative_to(ROOT):
    parser.error('Choose a private output directory outside the public repository')
OUT.mkdir(parents=True,exist_ok=True)
input_map=json.loads(args.input_map.read_text())
PRIVATE=OUT/'PRIVATE';PRIVATE.mkdir(exist_ok=True);os.chmod(PRIVATE,0o700)
sys.path.insert(0,str(ROOT/'scripts'))
from import_e2_wos import parse_wos_text,normalize_doi,normalize_title
from import_e2_scopus import read_scopus
BASE=ROOT/'references/literature_e2'
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def genuine_abstract(value):
    """A nonempty database placeholder is not an available source abstract."""
    value=value.strip()
    return bool(value) and not re.fullmatch(r'\[?\s*(?:no abstract (?:available|provided)|abstract not available|not available|n/?a)\s*\]?',value,re.I)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def writecsv(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
sources=[
('Scopus_global','Scopus','export_dcb22fe7-ce2a-4457-ab38-2a1021da5766_2026-11-06T011951.464806988.csv'),
('WoS0','WoS','savedrecs.txt'),
('WoS1','WoS','savedrecs (1).txt'),
('WoS2','WoS','savedrecs (2).txt'),
('WoS3','WoS','savedrecs (3).txt'),
('WoS4','WoS','savedrecs (4).txt'),
('WoS5','WoS','savedrecs (5).txt'),
('WoS6','WoS','savedrecs (6).txt'),
('WoS7','WoS','savedrecs (7).txt'),
('Scopus_current1104','Scopus','export_b93b2bae-2696-49cc-ae91-a4e4a8593990_2026-11-14T140618.516834071.csv'),
('Scopus_partial10','Scopus','export_57821d91-12c5-4736-bb98-0b9877c8c5d2_2026-11-14T140142.292225634.csv'),
('Scopus_primaries9','Scopus','export_3c909002-9f63-49c0-b7a5-78a7dbf48c16_2026-11-14T142202.018124612.csv')]
allgroups=csvrows(BASE/'final_audit_20261113/reconciled_groups.csv')
received={'native_20261105','native_20261108','arxiv_20261108'}
groups=[g for g in allgroups if received.intersection(json.loads(g['corpora']))]
assert len(groups)==9559
gids={g['group_id'] for g in groups}
nodes=csvrows(BASE/'final_audit_20261113/source_nodes.csv')
nodegid={n['node_id']:n['unified_group_id'] for n in nodes}
indices={k:collections.defaultdict(set) for k in ['doi','ut','eid','arxiv','title']}
groupdois={};groupyears=collections.defaultdict(set)
for g in groups:
    gid=g['group_id'];groupdois[gid]={normalize_doi(d) for d in json.loads(g['DOIs'])}
    for d in groupdois[gid]:indices['doi'][d].add(gid)
    for a in json.loads(g['arxiv_ids']):indices['arxiv'][re.sub(r'v\d+$','',a)].add(gid)
    indices['title'][normalize_title(g['representative_title'])].add(gid)
for n in nodes:
    gid=n['unified_group_id']
    if gid not in gids:continue
    indices['title'][normalize_title(n['title'])].add(gid)
    if n['year']:groupyears[gid].add(n['year'])
for r in csvrows(BASE/'scopus_20261105/combined_records.csv'):
    gid=nodegid.get('native_20261105:'+r['record_id'])
    if gid not in gids:continue
    for k,c in [('ut','UTs'),('eid','EIDs')]:
        for v in json.loads(r[c]):indices[k][v].add(gid)

expected={}
for path in ['scopus_20261105/source_manifest.json','wos_20261105/source_manifest.json']:
    d=json.loads((BASE/path).read_text());items=[d['source_file']] if isinstance(d.get('source_file'),dict) else d['sources']
    for v in items:expected[v['sha256']]=v.get('filename',v.get('source_file'))
focus=json.loads((BASE/'native_20261108/native/resumen.json').read_text())
for name,v in focus['files'].items():expected[v['sha256']]='focused/'+name
expected.update({
 '9f938405a118af505e08f934431bf0a8dff980227c4a14b2a19ac2a3c285f968':'E1/native_search/W02_8_records',
 '7e5851e9a5a853a520bbb3f006a71ba793c6e12f8b7ff5635853db53636a10bd':'E1/native_search/W01_59_records',
 'cd439243f2e47ff5b8f9232050bff56b9012ded7829ffdbaab55496dbec40fd0':'E1/native_execution/E1NEW-WOS-GENERAL_271_records'})

records=[];inventory=[]
for sid,db,filename in sources:
    path=Path(input_map[sid])
    digest=sha(path)
    if db=='Scopus':
        rr,_=read_scopus(path)
        parsed=[{'source_id':sid,'source_sha256':digest,'record_index':r['record_index'],'title':r['title'],'year':str(r['year'] or ''),'doi':r['doi'],'ut':'','eid':r['eid'],'arxiv':'','abstract':r['abstract']} for r in rr]
        integrity='validated_csv'
    else:
        rr,meta=parse_wos_text(path.read_text(encoding='utf-8-sig'))
        parsed=[{'source_id':sid,'source_sha256':digest,'record_index':i,'title':r.get('TI',''),'year':r.get('PY',''),'doi':normalize_doi(r.get('DI','')),'ut':r.get('UT',''),'eid':'','arxiv':'','abstract':r.get('AB','')} for i,r in enumerate(rr,1)]
        integrity='PT_ER_passed_EF_present'
    for r in parsed:
        d=r['doi']
        if d.startswith('10.48550/arxiv.'):r['arxiv']=re.sub(r'v\d+$','',d.split('arxiv.',1)[1])
    records.extend(parsed)
    inventory.append({'source_id':sid,'database':db,'private_path':str(path),'sha256':digest,'bytes':path.stat().st_size,'records':len(parsed),'with_abstract':sum(genuine_abstract(r['abstract']) for r in parsed),'nonempty_abstract_cells':sum(bool(r['abstract'].strip()) for r in parsed),'integrity':integrity,'exact_historical_identity':expected.get(digest)})

material=collections.defaultdict(list);conflicts=[];unmatched=[];links=[];nonempty_groups=set()
for r in records:
    hits={key:indices[key].get(r.get(key,''),set()) for key in ['doi','ut','eid','arxiv'] if r.get(key)}
    candidates=set().union(*hits.values()) if hits else set()
    rule='exact_identifier'
    if len(candidates)>1:
        conflicts.append({k:r[k] for k in ['source_id','record_index','title','doi','ut','eid']}|{'candidate_group_ids':json.dumps(sorted(candidates)),'reason':'identifiers_point_to_multiple_groups'})
        continue
    if not candidates:
        titlehits=indices['title'].get(normalize_title(r['title']),set())
        compatible={g for g in titlehits if (not r['doi'] or not groupdois[g] or r['doi'] in groupdois[g]) and r['year'] and r['year'] in groupyears[g]}
        candidates=compatible;rule='exact_normalized_title_year_no_doi_conflict'
        if titlehits and not compatible:
            conflicts.append({k:r[k] for k in ['source_id','record_index','title','doi','ut','eid']}|{'candidate_group_ids':json.dumps(sorted(titlehits)),'reason':'title_match_but_doi_or_year_conflict'})
    if len(candidates)!=1:
        unmatched.append(r|{'candidate_group_ids':sorted(candidates),'pending_reason':'multiple_compatible_exact_titles' if len(candidates)>1 else 'no_unique_compatible_identifier_or_title_year_match'})
        continue
    gid=next(iter(candidates));abstract=r['abstract'].strip()
    if abstract:nonempty_groups.add(gid)
    links.append({'source_id':r['source_id'],'record_index':r['record_index'],'group_id':gid,'match_rule':rule,'abstract_present':genuine_abstract(abstract)})
    if genuine_abstract(abstract):material[gid].append(r|{'match_rule':rule,'abstract_sha256':hashlib.sha256(abstract.encode()).hexdigest()})

coverage=[]
with (PRIVATE/'group_materials.jsonl').open('w') as out:
    for g in groups:
        gid=g['group_id'];versions=collections.defaultdict(list)
        for r in material[gid]:versions[r['abstract_sha256']].append(r)
        abstracts=[]
        for digest,rr in versions.items():
            abstracts.append({'abstract':rr[0]['abstract'],'abstract_sha256':digest,'sources':[{k:r[k] for k in ['source_id','source_sha256','record_index','doi','ut','eid','match_rule']} for r in rr]})
        out.write(json.dumps({'group_id':gid,'title':g['representative_title'],'DOIs':json.loads(g['DOIs']),'arxiv_ids':json.loads(g['arxiv_ids']),'corpora':json.loads(g['corpora']),'member_node_ids':json.loads(g['member_node_ids']),'abstracts':abstracts,'material_status':'abstract_available' if abstracts else 'title_metadata_only','screening_decision':'not_assessed_by_this_task'},ensure_ascii=False)+'\n')
        coverage.append({'group_id':gid,'title':g['representative_title'],'DOIs':g['DOIs'],'arxiv_ids':g['arxiv_ids'],'corpora':g['corpora'],'abstract_available':bool(abstracts),'distinct_abstract_versions':len(abstracts),'source_ids':';'.join(sorted({r['source_id'] for r in material[gid]})),'screening_decision':'not_assessed_by_this_task'})
os.chmod(PRIVATE/'group_materials.jsonl',0o600)
with (PRIVATE/'unmatched_records.jsonl').open('w') as out:
    for r in unmatched:out.write(json.dumps(r,ensure_ascii=False)+'\n')
os.chmod(PRIVATE/'unmatched_records.jsonl',0o600)
focused_rows=csvrows(BASE/'native_20261108/native/cribado_metadatos.csv')
focus_by_id={r['study_id']:r for r in focused_rows}
group_focus=collections.defaultdict(list)
for n in nodes:
    if n['corpus']=='native_20261108' and n['source_id'] in focus_by_id:
        group_focus[n['unified_group_id']].append(focus_by_id[n['source_id']])
only_focal=[]
for r in coverage:
    if not r['abstract_available'] and set(json.loads(r['corpora']))=={'native_20261108'}:
        f=group_focus[r['group_id']]
        only_focal.append({'group_id':r['group_id'],'study_ids':';'.join(x['study_id'] for x in f),'title':r['title'],'DOIs':r['DOIs'],'arxiv_ids':r['arxiv_ids'],'years':';'.join(sorted(groupyears[r['group_id']])),'providers':';'.join(sorted({x['databases'] for x in f})),'native_record_ids':';'.join(x['record_ids'] for x in f)})
writecsv(PRIVATE/'solo_focal_without_abstract.csv',only_focal)
with (PRIVATE/'group_materials.jsonl').open() as source,(PRIVATE/'screening_ready.jsonl').open('w') as target:
    for line in source:
        item=json.loads(line)
        if not item['abstracts']:continue
        chosen=max(item['abstracts'],key=lambda a:len(a['abstract']))
        target.write(json.dumps({'group_id':item['group_id'],'title':item['title'],'DOIs':item['DOIs'],'arxiv_ids':item['arxiv_ids'],'corpora':item['corpora'],'member_node_ids':item['member_node_ids'],'abstract':chosen['abstract'],'abstract_sha256':chosen['abstract_sha256'],'source_locators':chosen['sources'],'all_distinct_abstract_versions':len(item['abstracts']),'other_versions_available_in':'group_materials.jsonl','screening_decision':'not_assessed_by_this_task'},ensure_ascii=False)+'\n')
os.chmod(PRIVATE/'screening_ready.jsonl',0o600)
os.chmod(PRIVATE/'solo_focal_without_abstract.csv',0o600)
dump(PRIVATE/'input_inventory.json',inventory)
writecsv(OUT/'group_coverage.csv',coverage)
writecsv(OUT/'identifier_links.csv',links)
writecsv(OUT/'identity_conflicts.csv',conflicts)
writecsv(OUT/'remaining_groups.csv',[r for r in coverage if not r['abstract_available']])
safe_inventory=[{k:v for k,v in r.items() if k!='private_path'} for r in inventory]
writecsv(OUT/'source_coverage.csv',safe_inventory)
summary={'received_groups_fixed':9559,'directed_only_groups_separate':6,'groups_with_abstract':sum(r['abstract_available'] for r in coverage),'groups_without_abstract':sum(not r['abstract_available'] for r in coverage),'input_occurrences':len(records),'linked_occurrences':len(links),'unmatched_occurrences':len(unmatched),'conflict_occurrences':len(conflicts),'safe_csv_has_abstracts':False,'safe_csv_has_local_paths':False,'eligibility_classified':False,'original_query_universe_extended':False,'new_records_not_added_to_fixed_universe':True,'source_sha256_matches':{r['source_id']:r['exact_historical_identity'] for r in inventory if r['exact_historical_identity']}}
summary['coverage_by_received_corpus']={c:{'groups':sum(c in json.loads(r['corpora']) for r in coverage),'with_abstract':sum(c in json.loads(r['corpora']) and r['abstract_available'] for r in coverage)} for c in sorted(received)}
summary['remaining_by_corpora']=dict(collections.Counter(r['corpora'] for r in coverage if not r['abstract_available']))
summary['identity_conflicts_are_not_exclusions']=True
summary['identity_conflicts_may_overlap_unmatched_records']=True
summary['nonempty_abstract_groups_before_placeholder_check']=len(nonempty_groups)
summary['placeholder_only_groups_not_abstracts']=len(nonempty_groups)-summary['groups_with_abstract']
summary['group_ledger_source_sha256']=sha(BASE/'final_audit_20261113/reconciled_groups.csv')
summary['node_ledger_source_sha256']=sha(BASE/'final_audit_20261113/source_nodes.csv')
dump(OUT/'coverage_summary.json',summary)
assert len(coverage)==len({r['group_id'] for r in coverage})==9559
assert summary['groups_with_abstract']+summary['groups_without_abstract']==9559
print(json.dumps(summary,ensure_ascii=False))
