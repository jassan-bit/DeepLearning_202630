#!/usr/bin/env python3
"""Verify the frozen7500cohort with standard Python; no repository/network needed."""
import argparse,csv,hashlib,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FIELDS=['original_row_index','EID','title','DOI','year','source_title']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--original-export',type=Path,help='Optional private original CSV; never required for the default check')
 a=p.parse_args();manifest=json.loads((ROOT/'manifest.json').read_text())
 for row in manifest['files']:
  path=ROOT/row['path']
  if not path.is_file() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Changed package file: '+row['path'])
 with (ROOT/'inventory_7500.csv').open(encoding='utf-8',newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames!=FIELDS:raise ValueError('Inventory schema changed')
  rows=list(reader)
 if len(rows)!=7500 or len({r['EID'] for r in rows if r['EID'].strip()})!=7500:raise ValueError('7500unique nonempty EIDs required')
 if [int(r['original_row_index']) for r in rows]!=list(range(1,7501)):raise ValueError('Original row order changed')
 decision=json.loads((ROOT/'scope_decision.json').read_text());state=json.loads((ROOT/'closure_state.json').read_text())
 flow=json.loads((ROOT/'received_flow_baseline.json').read_text())
 baseline=(flow['received_groups_screened'],flow['unique_studies_included_applied_narrative'],flow['report_not_retrieved_rows'])
 if baseline!=(9559,42,174):raise ValueError('Preserved PRISMA denominator changed')
 if sha(ROOT/'received_flow_baseline.json')!=state['preserved_PRISMA']['public_flow_sha256']:raise ValueError('Preserved flow bytes changed')
 if not state['operational_scope_closed'] or state['active_Scopus_export_recovery_tasks']!=[]:raise ValueError('Scope decision state changed')
 raw_checked=False
 if a.original_export:
  if sha(a.original_export)!=decision['selected_original_export']['sha256']:raise ValueError('Wrong raw export: use original dcb22 cohort')
  with a.original_export.open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
  compact=[{'original_row_index':i,'EID':r['EID'],'title':r['Title'],'DOI':r['DOI'],'year':r['Year'],'source_title':r['Source title']} for i,r in enumerate(original,1)]
  stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(compact)
  if hashlib.sha256(stream.getvalue().encode('utf-8')).hexdigest()!=sha(ROOT/'inventory_7500.csv'):raise ValueError('Raw-to-inventory projection differs')
  raw_checked=True
 dois=[r['DOI'].lower().strip() for r in rows if r['DOI'].strip()]
 result={'status':'passed','inventory_rows':len(rows),'unique_EIDs':len({r['EID'] for r in rows}),
  'nonempty_DOI_rows':len(dois),'unique_nonempty_DOIs':len(set(dois)),
  'preserved_PRISMA_counts':list(baseline),'operational_scope_closed':True,'active_Scopus_export_recovery_tasks':[],
  'original_private_export_raw_bytes_reverified_this_run':raw_checked,
  'inventory_and_bundled_flow_hashes_verified':True,'new_screening_performed':False,'new_records_incorporated':0,
  'original_native_query_exhaustiveness_established':False,'model_operations':0,'network_calls':0}
 (ROOT/'verification_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
