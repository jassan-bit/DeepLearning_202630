"""Inspect a pinned public-domain OHLCV candidate; no fitting or forecasting."""
from __future__ import annotations
import csv
from datetime import date, datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_SHA = '5abadf99b2d394f920120a27de6f0df129859d71d4cfb0bf661bcc99446ab1fd'
CSV_SHA = 'e73fe05d106b7da1abe8f947264c514afd02cef5c9db4864b9437c84d6bd6069'
PROVIDER_COMMIT = '3e831d2779ca74c087c7eee05d538015c97b1e52'
FIELDS = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def inspect_aapl(rows):
    chosen = [r for r in rows if r['Ticker'] == 'AAPL']
    if not chosen:
        raise ValueError('No AAPL rows in the source')
    parsed = []
    missing = invalid = 0
    for r in chosen:
        day = date.fromisoformat(r['Date'])
        try:
            o, h, l, c, v = [float(r[k]) for k in ['Open','High','Low','Close','Volume']]
        except (ValueError, TypeError):
            missing += 1
            continue
        if not all(math.isfinite(x) for x in (o,h,l,c,v)):
            missing += 1
            continue
        if not (min(o,h,l,c) > 0 and v >= 0 and l <= min(o,c) <= max(o,c) <= h):
            invalid += 1
        parsed.append((day, o, h, l, c, v))
    if not parsed:
        raise ValueError("No finite OHLCV rows to inspect")
    dates = [r[0] for r in parsed]
    jumps = []
    for before, after in zip(parsed, parsed[1:]):
        if min(before[4], after[4]) <= 0:
            continue
        ratio = after[4] / before[4]
        if ratio < .70 or ratio > 1.30:
            jumps.append({'date': after[0].isoformat(), 'previous': before[0].isoformat(), 'close_ratio': ratio})
    return {'rows': len(chosen), 'first': min(dates).isoformat(), 'last': max(dates).isoformat(),
            'unique_dates': len(set(dates)), 'strictly_sorted': all(b > a for a,b in zip(dates,dates[1:])),
            'weekend_rows': sum(d.weekday() >= 5 for d in dates),
            'max_calendar_gap_days': max((b-a).days for a,b in zip(dates,dates[1:])),
            'missing_numeric_rows': missing, 'invalid_OHLCV_rows': invalid,
            'close_ratio_flags': jumps, 'rows_removed': 0, 'adjusted_close_used': False,
            'flags_are_not_automatic_split_labels': True,
            'trading_calendar_verified': False,
            'target_values_or_model_metrics_computed': False}

def run(output, source_commit):
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Full repository source commit required')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output directory; preserve old results')
    raw = (HERE/'inputs/stocks.csv.gz').read_bytes()
    if len(raw) != 294246 or sha(raw) != DATA_SHA:
        raise ValueError('Pinned input bytes differ')
    csv_raw = gzip.decompress(raw)
    if len(csv_raw) != 797881 or sha(csv_raw) != CSV_SHA:
        raise ValueError('Uncompressed source bytes differ')
    reader = csv.DictReader(io.StringIO(csv_raw.decode('utf-8')))
    if reader.fieldnames != FIELDS:
        raise ValueError('Source schema differs')
    source_rows = list(reader)
    qc = inspect_aapl(source_rows)
    if (qc['rows'],qc['first'],qc['last'],qc['unique_dates']) != (3270,'2000-03-01','2013-03-01',3270):
        raise ValueError('Pinned AAPL coverage differs')
    if not qc['strictly_sorted'] or qc['missing_numeric_rows'] or qc['invalid_OHLCV_rows']:
        raise ValueError('AAPL basic OHLCV checks failed')
    provenance = json.loads((HERE/'provenance.json').read_text())
    license_entry = provenance['stock_license_entry']
    if provenance['commit'] != PROVIDER_COMMIT or license_entry['recorded_license'] != 'Public domain' or license_entry['status'] != 'source-verified':
        raise ValueError('Source rights record differs')
    report = {'kind':'dataset_admission_audit','repository_HEAD':source_commit,
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Source and basic data quality inspection; no target tensor, fit, prediction or statistical test.',
        'datasets':[
            {'id':'D01','instrument':'AAPL','admitted':False,'technical_file_inspected':True,
             'basic_OHLCV_checks_passed':True,'quality':qc,
             'source':{'repository':'bokeh/bokeh_sampledata','commit':PROVIDER_COMMIT,
                 'path':'src/bokeh_sampledata/_data/stocks.csv.gz','bytes':294246,'sha256':DATA_SHA,
                 'uncompressed_sha256':CSV_SHA,'declared_license':'Public domain',
                 'source_rights_statement_status':'verified_primary_distributor_statement'},
             'prospective_task_contract':{'clock':'trading_session','window':7,'horizon':'next_trading_session',
                 'target':'population standard deviation of 7 close log returns; ddof=0; not annualized',
                 'UTC_calendar_equivalence_claimed':False,'target_not_built_in_this_step':True},
             'remaining_gates':['corporate-action policy with event evidence; keep genuine crashes',
                 'exchange-session calendar validation','common partitions and final prospective protocol']},
            {'id':'D02','instrument':'SPY','admitted':False,'technical_file_inspected':False,
             'remaining_gates':['provider account/access, raw version and rights conditions','causal cleaning and session aggregation']},
            {'id':'D03','instrument':'Bitstamp BTCUSD','admitted':False,'technical_file_inspected':False,
             'remaining_gates':['immutable data version and upstream terms','UTC aggregation and coverage']}
        ],'datasets_admitted':0,'datasets_with_file_inspected':1,
        'raw_input_preserved_in_source_repository':True,'full_source_price_file_opened_by_automated_audit':True,
        'forecast_performance_used_for_selection':False,'new_real_data_fits':0,
        'new_market_predictions':0,'new_hypothesis_tests':0,'R1_R3_verified':False,
        'real_campaign_allowed':False}
    decision_path=HERE.parent/'e3_d01_admission_20261118/admission_decision.json'
    if decision_path.is_file():
        decision=json.loads(decision_path.read_text())
        if not (decision.get('admitted') is True and decision['input']['sha256']==DATA_SHA
                and decision['policy']['id']=='RAW_KEEP_ALL'
                and decision['policy']['adjusted_close_used'] is False
                and decision['calendar']['ordered_sessions_matched']==3270
                and decision['calendar']['missing']==decision['calendar']['extra']==[]):
            raise ValueError('D01 admission decision does not match the inspected file/task')
        calendar_path=decision_path.parent/'NASDAQ_sessions_20000301_20130301.csv'
        calendar_raw=calendar_path.read_bytes()
        if sha(calendar_raw) != decision['calendar']['csv_sha256']:
            raise ValueError('Calendar reference bytes differ')
        sessions=[r['date'] for r in csv.DictReader(io.StringIO(calendar_raw.decode()))]
        if sessions != [r['Date'] for r in source_rows if r['Ticker']=='AAPL']:
            raise ValueError('Dataset dates differ from admitted sessions')
        report['datasets'][0]['quality']['trading_calendar_verified']=True
        report['datasets'][0]['admitted']=True
        report['datasets'][0]['remaining_gates']=[]
        report['datasets'][0]['admission_decision']='experiments/e3_d01_admission_20261118/admission_decision.json'
        report['datasets'][0]['task_contract_status']='admitted_raw_session_definition; model protocol still pending'
        report['datasets_admitted']=1
        report['remaining_campaign_gates']=decision['remaining_campaign_gates']
        report['initial_zero_admission_report_preserved']=decision['evidence']['reported_audit_commit']
    output.mkdir(parents=True)
    for name, value in [('dataset_audit.json',report),('source_evidence.json',provenance)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'audit.log').write_text('D01 AAPL: 3270 rows; 0 basic OHLCV failures; no filtering, adjusted-close use or modelling.\n')
    print(json.dumps({'D01_rows':3270,'D01_basic_OHLCV':'passed','sources_inspected':1,
        'datasets_admitted':report['datasets_admitted'],'new_real_data_fits':0,'new_market_predictions':0,'output':str(output)}))
    return report
