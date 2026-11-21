"""Regenerate the published development V3 principal table from saved points.

Standard library arithmetic only. No model imports, fits, forward pass, market
download, CRPS reconstruction, selection update, or claim of confirmation.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, math, statistics
from collections import defaultdict
from pathlib import Path

NEURAL = ('additive_positive_head', 'atom_crps', 'atom_mse_only', 'fixed_probs', 'atom_plugin')
CONTROLS = ('floor_only', 'floor_plus_mean_excess')
SEEDS = {'20261118', '20261119', '20261120'}
FIELDS = ['name', 'kind', 'trajectories', 'dates_per_trajectory', 'mean_individual_RMSE', 'SD_individual_RMSE', 'mean_individual_MAE', 'mean_bias_prediction_minus_target', 'RMSE_rank_of_7']
FLOATS = FIELDS[4:8]
PREDICTION_SHA256 = '0399c7c80f46ac2fdfccaf218a2866fe110f7ec5051d8484e488299c679ad193'

def require(ok, message):
    if not ok: raise ValueError(message)

def regenerate(source, output, *, verify_hash=True):
    source, output = Path(source), Path(output)
    raw = source.read_bytes()
    if verify_hash: require(hashlib.sha256(raw).hexdigest() == PREDICTION_SHA256, 'Saved prediction SHA256 mismatch')
    reader = csv.DictReader(raw.decode('utf-8').splitlines())
    require(reader.fieldnames == ['name','seed','date','target','prediction'], 'Prediction schema mismatch')
    groups, target_by_date, seen = defaultdict(list), {}, set()
    start, stop = datetime.date(2024,9,7), datetime.date(2024,10,31)
    dates = [(start+datetime.timedelta(days=d)).isoformat() for d in range((stop-start).days+1)]
    count = 0
    for row in reader:
        require(set(row) == set(reader.fieldnames) and None not in row.values(), 'Malformed row')
        name, seed, date = row['name'], row['seed'], row['date']
        require(name in NEURAL+CONTROLS, 'Unexpected recipe')
        require(seed in SEEDS if name in NEURAL else seed == 'analytic', 'Unexpected seed')
        require(date in dates, 'Unexpected assessment date')
        key = (name,seed,date);require(key not in seen, 'Duplicate point');seen.add(key)
        target, prediction = float(row['target']), float(row['prediction'])
        require(math.isfinite(target) and math.isfinite(prediction), 'Nonfinite point')
        require(target >= 0 and prediction >= 0, 'Negative volatility point')
        require(date not in target_by_date or target_by_date[date] == target, 'Targets differ across recipes')
        target_by_date[date] = target
        groups[(name,seed)].append((date,prediction-target));count += 1
    require(count == 935 and len(groups) == 17, 'Incomplete 935-point / 17-trajectory support')
    per_seed = defaultdict(list)
    for (name,seed), items in sorted(groups.items()):
        require(sorted(x[0] for x in items) == dates, 'Incomplete date support')
        errors = [x[1] for x in sorted(items)]
        per_seed[name].append((math.sqrt(math.fsum(e*e for e in errors)/55), math.fsum(abs(e) for e in errors)/55, math.fsum(errors)/55))
    rows = []
    for name in sorted(NEURAL+CONTROLS):
        values = per_seed[name];n = len(values);require(n == (3 if name in NEURAL else 1), 'Trajectory count differs')
        rmses, maes, biases = zip(*values)
        rows.append(dict(zip(FIELDS[:-1], [name, 'neural' if name in NEURAL else 'analytical', n, 55,
            statistics.mean(rmses), statistics.stdev(rmses) if n>1 else '', statistics.mean(maes), statistics.mean(biases)])))
    ranks = {r['name']: i+1 for i,r in enumerate(sorted(rows,key=lambda r:(r['mean_individual_RMSE'],r['name'])))}
    for row in rows: row[FIELDS[-1]] = ranks[row['name']]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    by_name = {r['name']:r for r in rows}
    worse = (by_name['atom_crps']['mean_individual_RMSE']/by_name['atom_mse_only']['mean_individual_RMSE']-1)*100
    return {'status':'saved_point_primary_table_regenerated', 'prediction_sha256': hashlib.sha256(raw).hexdigest(),
        'prediction_rows':935,'trajectories':17,'recipes':7,'dates':55,'SD_ddof':1,
        'primary_metric':'mean of individual-seed RMSE; not ensemble RMSE',
        'atom_crps_percent_worse_than_atom_mse_only':worse,'original_two_percent_rule_passed':False,
        'source_mean_CRPS_U_recomputed':False,'development_exposed_calendar':True,
        'new_model_fits':0,'new_model_predictions':0,'new_training_campaign':False,'independent_confirmation':False,
        'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}

def read_table(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream);require(reader.fieldnames == FIELDS,'Principal-table schema mismatch')
        rows=list(reader)
    require(len(rows)==7 and {r['name'] for r in rows}==set(NEURAL+CONTROLS),'Principal-table row support mismatch')
    require(len({r['name'] for r in rows})==len(rows),'Duplicate recipe')
    for row in rows:
        require(None not in row and None not in row.values(),'Malformed principal table')
        for key in FLOATS:
            if row[key]=='': require(key=='SD_individual_RMSE' and row['name'] in CONTROLS,'Missing numeric field')
            else: require(math.isfinite(float(row[key])),'Nonfinite principal field')
    return {r['name']:r for r in rows}

def compare(reference, actual, tolerances):
    require(isinstance(tolerances,dict) and set(tolerances)==set(FLOATS),'Each floating field needs a declared tolerance')
    for field,values in tolerances.items():
        require(isinstance(values,dict) and set(values)=={'absolute','relative'},'Tolerance schema mismatch')
        for value in values.values(): require(type(value) in (int,float) and math.isfinite(value) and value>=0,'Tolerance must be a finite nonnegative number; null is not approval')
    expected, observed = read_table(reference),read_table(actual)
    discrepancies={field:{'max_absolute':0.0,'max_relative':0.0,'maximum_at_recipe':None} for field in FLOATS}
    failures=[]
    for name,e in expected.items():
        a=observed[name]
        for field in FIELDS:
            if field not in FLOATS or e[field]=='' or a[field]=='':
                if e[field]!=a[field]: failures.append({'recipe':name,'field':field,'reason':'exact/support/blank mismatch'})
                continue
            ev,av=float(e[field]),float(a[field]);absolute=abs(av-ev);relative=absolute/abs(ev) if ev else (0.0 if absolute==0 else None)
            d=discrepancies[field]
            if absolute>d['max_absolute']:d.update(max_absolute=absolute,maximum_at_recipe=name)
            if relative is not None:d['max_relative']=max(d['max_relative'],relative)
            t=tolerances[field]
            if absolute>t['absolute']+t['relative']*abs(ev):failures.append({'recipe':name,'field':field,'absolute':absolute,'relative':relative})
    return {'status':'pass' if not failures else 'fail','rows':7,'floating_fields':4,'exact_fields':5,'units':'daily unannualized volatility; bias prediction minus target',
        'per_field':discrepancies,'max_absolute_discrepancy':max(d['max_absolute'] for d in discrepancies.values()),
        'tolerances':tolerances,'failures':failures,'CRPS_U_in_comparison':False}

def check_published_table(markdown, actual):
    """Check the seven visible rounded values against regenerated full precision."""
    text=Path(markdown).read_text();require('## 8.19' in text,'Published V3 section missing')
    text=text.split('## 8.19',1)[1].split('## 8.20',1)[0]
    rows=read_table(actual);checked=0
    for line in text.splitlines():
        cells=[v.strip().replace('`','') for v in line.strip().split('|')[1:-1]]
        if not cells or len(cells)!=5:continue
        name={'Aditiva':'additive_positive_head','atom_crps —V3 completa—':'atom_crps'}.get(cells[0],cells[0])
        if name not in rows:continue
        row=rows[name]
        # The principal Markdown table lists name, trajectories, RMSE, SD, MAE.
        for idx,field in ((2,'mean_individual_RMSE'),(3,'SD_individual_RMSE'),(4,'mean_individual_MAE')):
            cell=cells[idx].replace(',','.')
            if row[field]=='':require(cell in ('NA','N/A','—','–','','No aplica'), 'Published analytic SD differs');continue
            # Some Markdown editions include a rank column; do not guess schema.
            require(cell == f'{float(row[field]):.9f}',f'Published rounded value differs: {row["name"]}/{field}: {cell}')
        checked+=1
    require(checked==7,'Exactly seven published principal rows must match')
    return {'status':'pass','published_rows':7,'display_decimals':9,'book_section':'8.19'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(json.dumps(regenerate(a.predictions,a.output),sort_keys=True))
if __name__=='__main__':main()
