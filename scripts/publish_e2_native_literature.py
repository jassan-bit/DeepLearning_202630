"""Validate and mirror the public E2 20261108 evidence. Never read private exports."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'references/literature_e2/native_20261108'
DEST = ROOT / 'book_e2/_static/literature_native_20261108'
FILES = [
 'README.md', 'focal_source_reviews.json',
 'native/LEEME.md', 'native/cribar.py', 'native/crear_derivados_publicos.py',
 'native/ajustes_revision.json', 'native/cribado_metadatos.csv',
 'native/conflictos_duplicados.csv', 'native/relaciones_duplicados.csv',
 'native/resumen.json', 'native/validacion.json',
 'arxiv/README.md', 'arxiv/summary.json', 'arxiv/query_manifest.json',
 'arxiv/records.csv', 'arxiv/query_membership.csv', 'arxiv/screening_initial.csv',
]

CSV_COLUMNS = {
 'conflictos_duplicados.csv': ['record_a','record_b','reason'],
 'relaciones_duplicados.csv': ['record_a','record_b','rule'],
 'cribado_metadatos.csv': ['study_id','title','year','years','doi','dois','record_ids','databases','duplicate_count','document_type','period_status','initial_class','screening_class','reason','targeted_audit','full_text_inclusion'],
 'query_membership.csv': ['query_id','arxiv_id','version_id'],
 'records.csv': ['arxiv_id','version_id','title','authors','url','published','updated','doi','journal_ref','categories'],
 'screening_initial.csv': ['arxiv_id','title','url','screening_status','eligibility_final','scope_note'],
}

def read_csv(path):
 with path.open(encoding='utf-8-sig', newline='') as f:
  reader = csv.DictReader(f)
  assert path.name in CSV_COLUMNS, path
  assert reader.fieldnames == CSV_COLUMNS[path.name], path
  return list(reader)

def validate(source=SOURCE):
 for csv_path in source.rglob('*.csv'):
  read_csv(csv_path)
 summary = json.loads((source/'native/resumen.json').read_text())
 rows = read_csv(source/'native/cribado_metadatos.csv')
 assert len(rows) == summary['unique_total'] == 1078
 assert len({r['study_id'] for r in rows}) == len(rows)
 assert dict(collections.Counter(r['screening_class'] for r in rows)) == summary['classes']
 assert summary['raw_total'] - len(rows) == summary['duplicates_removed'] == 556
 assert all(r['full_text_inclusion']=='NO DETERMINADA' for r in rows)
 assert sum(r['targeted_audit'].startswith('SÍ') for r in rows) == 21
 record_ids = [x.strip() for r in rows for x in r['record_ids'].split(';')]
 assert len(record_ids) == len(set(record_ids)) == 1634
 assert sum(r['period_status']=='2019–2026' for r in rows) == 1054
 assert sum(not r['doi'].strip() for r in rows) == 50
 assert sum(not r['dois'].strip() for r in rows) == 49
 quality = json.loads((source/'native/validacion.json').read_text())
 assert quality['missing_doi_representatives'] == 50
 assert quality['groups_without_any_doi'] == 49
 arxiv = read_csv(source/'arxiv/records.csv')
 screened = read_csv(source/'arxiv/screening_initial.csv')
 membership = read_csv(source/'arxiv/query_membership.csv')
 ids={r['arxiv_id'] for r in arxiv}
 assert len(arxiv) == len(ids) == len(screened) == 1004
 assert ids == {r['arxiv_id'] for r in screened} == {r['arxiv_id'] for r in membership}
 assert len(membership) == 1199
 assert sum(r['screening_status']=='pending_screening' for r in screened)==983
 sets={q:{r['arxiv_id'] for r in membership if r['query_id']==q} for q in ['btc','crypto','architectures','har_hybrid']}
 assert {q:len(v) for q,v in sets.items()} == dict(btc=181,crypto=391,architectures=605,har_hybrid=22)
 assert sets['btc'] <= sets['crypto']
 assert all(r['eligibility_final']=='not_determined' for r in screened)
 for path in source.rglob('*'):
  if path.is_file():
   assert path.relative_to(source).as_posix() in FILES+['manifest.json'], path
 return {'native_groups':len(rows),'native_occurrences':1634,'arxiv_ids':len(ids),'arxiv_occurrences':1199,'final_eligibility_closed':False}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
 counts=validate()
 payload={'schema':'e2_public_metadata_allowlist_v1','scope':'Bibliographic update only; historical and experimental files unchanged','counts':counts,'approved_public_files':sorted(FILES),'files':{name:{'bytes':(SOURCE/name).stat().st_size,'sha256':hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()} for name in sorted(FILES)}}
 manifest=(json.dumps(payload,ensure_ascii=False,indent=2)+'\n').encode()
 if args.check:
  assert (SOURCE/'manifest.json').read_bytes()==manifest, 'Source manifest stale'
 else:(SOURCE/'manifest.json').write_bytes(manifest)
 for name in FILES+['manifest.json']:
  data=(SOURCE/name).read_bytes();target=DEST/name
  if args.check:assert target.read_bytes()==data, f'Stale public mirror: {name}'
  else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 assert {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()} == set(FILES+['manifest.json'])
 print(json.dumps(counts))

if __name__=='__main__':main()
