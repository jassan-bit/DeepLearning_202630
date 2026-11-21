"""Reaggregate the additive19x4 audit from fixed, costly per-seed results.

This is arithmetic reconstruction from4,060 saved metric records, not a replay
of1,874,560 predictions, a training run, bootstrap or confirmatory analysis.
"""
from __future__ import annotations
import csv, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path

PREFIX='book_e3/_static/e3_audit_completed_20261119T234320461006Z'
CANDIDATE='CNN-LSTM-atencion-aditiva E3'
INPUT_HASHES={
 'analysis/per_seed_metrics.csv':'a81d0827c77f56ec68bcd0aec3649e6e8e1de41199af7e2a4837a8f99ce5c94b',
 'analysis/interval_metrics.csv':'23876ca6a22d8e8eec2be3445d867463026513fd10264315b01a8eb48e39dcb8',
 'analysis/analysis_receipt.json':'0048bdf3e8376ed5cefa57e57b31eabbc2e5785bd5acf7235a544fa5f7265efc',
 'analysis/model_metrics.csv':'4b311d972ae0f6051e29868ea9b497ef70ee961ebcedbd05a297fb4c5d488b9a',
 'analysis/cross_dataset_ranks.csv':'c698878be35a9fe82edcab98c0a8cdf3bdec5b5f5cb5115f3222e04ac91f021c',
 'clean_test_summary.csv':'ec3707d3520ecd8bfeea933cd02498ce81ac2861a3047f595ab10347c84aeba8'}
MODEL_FIELDS=['dataset','split','scenario','model','seed_count','RMSE_mean_individual','RMSE_seed_sample_SD','MAE_mean_individual','raw_RMSE_mean_individual_secondary','raw_MAE_mean_individual_secondary','sMAPE_percent_mean_individual','R2_mean_individual','dates','ensembling_performed','classical_history_stressed']
MODEL_NUMERIC=MODEL_FIELDS[5:12]
RANK_FIELDS=['scope','dataset','scenario','model','rank_RMSE','RMSE_mean_individual','formal_between_dataset_p_value','independent_markets_assumed','pooled_original_unit_RMSE_computed']
SUMMARY_FIELDS=['dataset','test_dates','candidate_RMSE_mean_individual','candidate_RMSE_seed_sample_SD','candidate_rank_among_19','best_rival','best_rival_RMSE','relative_RMSE_reduction_vs_best_rival_percent','nominal_coverage','candidate_empirical_coverage_mean_across_10_seeds','candidate_interval_mean_width']
SUMMARY_NUMERIC=SUMMARY_FIELDS[2:4]+SUMMARY_FIELDS[6:]
NUMERIC=sorted(set(MODEL_NUMERIC+['rank_RMSE']+SUMMARY_NUMERIC))
UNITS={field:'unannualized sigma7 in its own dataset; no cross-dataset pooling' for field in NUMERIC}
UNITS.update({'sMAPE_percent_mean_individual':'percent','R2_mean_individual':'dimensionless R2','rank_RMSE':'dimensionless descriptive average rank','relative_RMSE_reduction_vs_best_rival_percent':'percent','nominal_coverage':'fraction in0..1','candidate_empirical_coverage_mean_across_10_seeds':'fraction in0..1'})

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def table(path):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
    require(rows and all(None not in r and None not in r.values() for r in rows),'Malformed/empty audit CSV');return reader.fieldnames,rows
def write_csv(path,fields,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:('' if r.get(k) is None else r.get(k,'')) for k in fields} for r in rows)
def finite(value):
    number=float(value);require(math.isfinite(number),'Nonfinite cached metric');return number
def average_ranks(values):
    ordered=sorted(values,key=lambda k:(values[k],k));out={};i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and values[ordered[j]]==values[ordered[i]]:j+=1
        for model in ordered[i:j]:out[model]=(i+1+j)/2
        i=j
    return out

def compare_csv(reference,actual,keys,numeric,tolerances):
    ef,e=table(reference);af,a=table(actual);require(ef==af,'Audit schema mismatch')
    def index(rows):
        result={}
        for r in rows:
            k=tuple(r[f] for f in keys);require(k not in result,'Duplicate audit row key');result[k]=r
        return result
    ei,ai=index(e),index(a);require(ei.keys()==ai.keys(),'Audit row support mismatch')
    report={field:{'unit':UNITS.get(field,'caller-declared fixture unit'),'max_absolute':0.0,'max_relative':0.0,'maximum_at_key':None} for field in numeric};failures=[]
    for key,er in ei.items():
        ar=ai[key]
        for field in ef:
            ev,av=er[field],ar[field]
            if field not in numeric or ev=='' or av=='':
                if ev!=av:failures.append({'key':key,'field':field,'reason':'exact/blank/support differs'})
                continue
            ev,av=finite(ev),finite(av);absolute=abs(av-ev);relative=absolute/abs(ev) if ev else (0.0 if absolute==0 else None)
            record=report[field]
            if absolute>record['max_absolute']:record.update(max_absolute=absolute,maximum_at_key=key)
            if relative is not None:record['max_relative']=max(record['max_relative'],relative)
            t=tolerances[field]
            if absolute>t['absolute']+t['relative']*abs(ev):failures.append({'key':key,'field':field,'absolute':absolute,'relative':relative})
    return {'status':'pass' if not failures else 'fail','rows':len(e),'per_field':report,'failures':failures,'max_absolute_discrepancy':max(x['max_absolute'] for x in report.values()),'global_max_interpretation':'Mechanical maximum across mixed-unit fields; substantive assessment uses the per-field discrepancy, unit and row key.','tolerances':{f:tolerances[f] for f in numeric}}

def validate_tolerances(tolerances):
    require(isinstance(tolerances,dict) and set(tolerances)==set(NUMERIC),'Declare all audit floating-field tolerances')
    for values in tolerances.values():
        require(isinstance(values,dict) and set(values)=={'absolute','relative'},'Audit tolerance schema')
        require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values.values()),'Audit tolerances must be finite nonnegative, not null')

def regenerate(source,out,tolerances,*,install_in_book=False):
    source,out=Path(source),Path(out);base=source/PREFIX;validate_tolerances(tolerances);out.mkdir(parents=True,exist_ok=True)
    for name,h in INPUT_HASHES.items():require(sha(base/name)==h,'Fixed audit input/reference SHA256 differs: '+name)
    cfg=json.loads((base/'analysis/analysis_receipt.json').read_text())['configuration']
    datasets=cfg['datasets'];family=cfg['family'];seeds=set(map(str,cfg['seeds']));det=set(cfg['deterministic']);scenarios=cfg['scenarios'];splits=cfg['splits']
    require(datasets==['D00','D01','D02','D03'] and len(family)==19 and len(set(family))==19 and len(seeds)==10 and len(det)==5,'Historical family/configuration differs')
    columns,per_seed=table(base/'analysis/per_seed_metrics.csv');groups=defaultdict(list);seen=set()
    require(len(per_seed)==4060,'Full cached per-seed metric support must contain4060rows')
    for row in per_seed:
        key=tuple(row[f] for f in ['dataset','split','scenario','model']);seed=row['seed'];require((*key,seed) not in seen,'Duplicate per-seed metric');seen.add((*key,seed))
        require(row['dataset'] in datasets and row['model'] in family and row['split'] in splits,'Unexpected cached metric scope')
        require(row['scenario'] in (scenarios if row['split']=='test' else ['unperturbed']),'Unexpected split/scenario')
        require(seed=='-1' if row['model'] in det else seed in seeds,'Unexpected cached seed')
        require(row['R2_status']=='defined' and int(row['n'])>0,'Missing/undefined historical metric')
        for f in ['RMSE','MAE','sMAPE_percent','R2','raw_RMSE_secondary','raw_MAE_secondary']:finite(row[f])
        groups[key].append(row)
    expected={(d,sp,sc,m) for d in datasets for sp in splits for sc in (scenarios if sp=='test' else ['unperturbed']) for m in family}
    require(set(groups)==expected and len(expected)==532,'Full19x4 split/scenario support differs')
    aggregate=[]
    for d in datasets:
      for sp in splits:
       for sc in (scenarios if sp=='test' else ['unperturbed']):
        for model in family:
            rows=groups[(d,sp,sc,model)];require({r['seed'] for r in rows}==({'-1'} if model in det else seeds),'Incomplete registered seed support')
            require(len({r['n'] for r in rows})==1,'Per-seed date counts differ');rmses=[finite(r['RMSE']) for r in rows]
            values={'dataset':d,'split':sp,'scenario':sc,'model':model,'seed_count':len(rows),'RMSE_mean_individual':statistics.mean(rmses),'RMSE_seed_sample_SD':statistics.stdev(rmses) if len(rows)>1 else None,'dates':int(rows[0]['n']),'ensembling_performed':False,'classical_history_stressed':False if model in cfg['history_models'] and sc!='unperturbed' else None}
            for dest,src in [('MAE_mean_individual','MAE'),('raw_RMSE_mean_individual_secondary','raw_RMSE_secondary'),('raw_MAE_mean_individual_secondary','raw_MAE_secondary'),('sMAPE_percent_mean_individual','sMAPE_percent'),('R2_mean_individual','R2')]:values[dest]=statistics.mean(finite(r[src]) for r in rows)
            aggregate.append(values)
    write_csv(out/'model_metrics_regenerated.csv',MODEL_FIELDS,aggregate)
    ranks=[];lookup={(r['dataset'],r['split'],r['scenario'],r['model']):r for r in aggregate};ranks_by={}
    for sc in scenarios:
        for d in datasets:
            values={m:lookup[(d,'test',sc,m)]['RMSE_mean_individual'] for m in family};rr=average_ranks(values);ranks_by[(d,sc)]=rr
            ranks.extend({'scope':'within_dataset','dataset':d,'scenario':sc,'model':m,'rank_RMSE':rr[m],'RMSE_mean_individual':values[m]} for m in family)
        for label,ids in [('all_contexts',datasets),('additional_tasks',[d for d in datasets if d!='D00'])]:
            ranks.extend({'scope':label,'dataset':','.join(ids),'scenario':sc,'model':m,'rank_RMSE':statistics.mean(ranks_by[(d,sc)][m] for d in ids),'independent_markets_assumed':False,'pooled_original_unit_RMSE_computed':False} for m in family)
    write_csv(out/'cross_dataset_ranks_regenerated.csv',RANK_FIELDS,ranks)
    _,intervals=table(base/'analysis/interval_metrics.csv');require(len(intervals)==2320,'Full cached interval support differs');summary=[]
    for d in datasets:
        candidate=lookup[(d,'test','unperturbed',CANDIDATE)];rival=min((lookup[(d,'test','unperturbed',m)] for m in family if m!=CANDIDATE),key=lambda r:(r['RMSE_mean_individual'],r['model']))
        irows=[r for r in intervals if r['dataset']==d and r['scenario']=='unperturbed' and r['model']==CANDIDATE];require(len(irows)==10 and {r['seed'] for r in irows}==seeds,'Candidate cached interval seeds differ')
        require({int(r['n']) for r in irows}=={candidate['dates']},'Interval/metric date support differs')
        nominal={finite(r['nominal_coverage']) for r in irows};require(nominal=={0.9},'Nominal coverage differs')
        rmse=candidate['RMSE_mean_individual'];rv=rival['RMSE_mean_individual']
        summary.append(dict(zip(SUMMARY_FIELDS,[d,candidate['dates'],rmse,candidate['RMSE_seed_sample_SD'],int(ranks_by[(d,'unperturbed')][CANDIDATE]),rival['model'],rv,100*(rv-rmse)/rv,0.9,statistics.mean(finite(r['coverage']) for r in irows),statistics.mean(finite(r['mean_width']) for r in irows)])))
    write_csv(out/'clean_test_summary_regenerated.csv',SUMMARY_FIELDS,summary)
    clean=[dict(r,rank_RMSE=ranks_by[(r['dataset'],'unperturbed')][r['model']]) for r in aggregate if r['split']=='test' and r['scenario']=='unperturbed'];require(len(clean)==76,'Main76row support differs')
    write_csv(out/'principal_additive_19x4.csv',MODEL_FIELDS+['rank_RMSE'],clean)
    comparisons={
        'model_metrics_532':compare_csv(base/'analysis/model_metrics.csv',out/'model_metrics_regenerated.csv',['dataset','split','scenario','model'],MODEL_NUMERIC,tolerances),
        'ranks_456':compare_csv(base/'analysis/cross_dataset_ranks.csv',out/'cross_dataset_ranks_regenerated.csv',['scope','dataset','scenario','model'],['rank_RMSE','RMSE_mean_individual'],tolerances),
        'clean_summary_4':compare_csv(base/'clean_test_summary.csv',out/'clean_test_summary_regenerated.csv',['dataset'],SUMMARY_NUMERIC,tolerances)}
    require(all(r['status']=='pass' for r in comparisons.values()),'Audit regeneration differs; no tolerance adjustment performed')
    published=check_markdown_summary(source/'book_e3/sections/08_verificacion_estadistica.md',summary)
    installed=[]
    if install_in_book:
        for name in ['model_metrics_regenerated.csv','cross_dataset_ranks_regenerated.csv','clean_test_summary_regenerated.csv','principal_additive_19x4.csv']:
            extra=source/'book_e3/_static/human_reproduction_19x4'/name;extra.parent.mkdir(parents=True,exist_ok=True);extra.write_bytes((out/name).read_bytes());installed.append(extra.relative_to(source).as_posix())
    result={'status':'full_additive_audit_reaggregated_from_fixed_cached_seed_metrics','input_metric_rows':4060,'main_clean_metric_rows':580,'aggregate_rows':532,'main_clean_rows':76,'rank_rows':456,'summary_rows':4,'datasets':4,'systems':19,'neural_seeds_each':10,'deterministic_systems':5,'comparisons':comparisons,'published_summary':published,'installed_generated_paths':installed,'original_audit_payloads_modified':False,'main76_csv_sha256':sha(out/'principal_additive_19x4.csv'),'all_input_hashes':INPUT_HASHES,'scope':'Reaggregation from saved per-seed metrics and interval summaries; no recomputation of per-seed metrics from1874560points, no bootstrap, fits, forecasts, CRPS or confirmatory decisions.','new_model_fits':0,'new_model_predictions':0,'human_execution_claimed':False}
    (out/'audit_regeneration.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result

def check_markdown_summary(path,summary):
    text=Path(path).read_text();require('## 8.18' in text and '## 8.19' in text,'Published audit sections missing')
    text=text.split('## 8.18',1)[1].split('## 8.19',1)[0];rows={r['dataset']:r for r in summary};seen=set()
    for line in text.splitlines():
        cells=[v.strip() for v in line.strip().split('|')[1:-1]]
        if len(cells)!=6 or cells[0] not in rows or cells[-1]!='1/19':continue
        r=rows[cells[0]];require(cells[0] not in seen,'Duplicate published audit row');seen.add(cells[0])
        require(cells[1]==str(r['test_dates']),'Published date count differs')
        for i,key in [(2,'candidate_RMSE_mean_individual'),(3,'best_rival_RMSE')]:require(cells[i].replace(',','.')==f'{r[key]:.9f}','Published audit RMSE differs')
        require(cells[4].replace(',','.').replace(' %','')==f'{r["relative_RMSE_reduction_vs_best_rival_percent"]:.2f}','Published audit reduction differs')
    require(seen==set(rows),'All four visible audit summary rows must match')
    return {'status':'pass','datasets':4,'display_RMSE_decimals':9,'display_gain_decimals':2,'book_section':'8.18'}
