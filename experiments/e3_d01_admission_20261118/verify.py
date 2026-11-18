"""Verify D01 input, calendar snapshot and explicit RAW admission without models."""
from __future__ import annotations
import argparse
import csv
from datetime import date
import gzip
import hashlib
import io
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
DATA_SHA='5abadf99b2d394f920120a27de6f0df129859d71d4cfb0bf661bcc99446ab1fd'
CAL_SHA='05911b2c78202d985f55a16f1a8919a8526fa48ccd82cffd6bf7eaa72ec0be60'

def verify():
    data=HERE.parent/'e3_dataset_admission_20261118/inputs/stocks.csv.gz'
    raw=data.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==DATA_SHA
    rows=[r for r in csv.DictReader(io.StringIO(gzip.decompress(raw).decode())) if r['Ticker']=='AAPL']
    dates=[r['Date'] for r in rows]
    cal=(HERE/'NASDAQ_sessions_20000301_20130301.csv').read_bytes()
    assert hashlib.sha256(cal).hexdigest()==CAL_SHA
    sessions=[r['date'] for r in csv.DictReader(io.StringIO(cal.decode()))]
    assert dates==sessions and len(dates)==len(set(dates))==3270
    assert dates[0]=='2000-03-01' and dates[-1]=='2013-03-01'
    assert all(date.fromisoformat(d).weekday()<5 for d in dates)
    closed=['2001-09-11','2001-09-12','2001-09-13','2001-09-14','2004-06-11','2007-01-02','2012-10-29','2012-10-30']
    assert not set(closed)&set(dates)
    check=json.loads((HERE/'calendar_check.json').read_text())
    assert check['ordered_dates_equal'] and check['missing_sessions']==check['extra_dates']==[]
    assert check['version']=='4.11.3' and check['requested_calendar']=='NASDAQ' and check['resolved_calendar']=='XNYS'
    decision=json.loads((HERE/'admission_decision.json').read_text())
    assert decision['admitted'] is True and decision['datasets_admitted_after_review']==1
    assert decision['input']['sha256']==DATA_SHA and decision['calendar']['csv_sha256']==CAL_SHA
    assert decision['policy']['id']=='RAW_KEEP_ALL' and decision['policy']['adjusted_close_used'] is False
    assert decision['policy']['rows_removed']==decision['policy']['rows_filled']==0
    assert decision['policy']['all_large_moves_retained'] is True
    assert decision['task']['window_bars']==decision['task']['target_return_count']==7
    assert decision['task']['horizon_sessions']==1 and decision['task']['possible_window_target_pairs_by_structure']==len(rows)-7
    assert decision['task']['target_values_computed_for_admission'] is False
    assert decision['real_campaign_allowed'] is False and decision['R1_R3_verified'] is False
    assert decision['historical_E2_E3_training_preserved'] and not decision['automatic_retraining_requested']
    assert decision['new_real_data_fits']==decision['new_market_predictions']==decision['new_hypothesis_tests']==0
    result={'status':'passed','D01_admitted_for_declared_task':True,'datasets_admitted':1,'datasets_required':3,
            'ordered_calendar_matches':3270,'missing_sessions':0,'extra_dates':0,
            'rows_removed_or_filled':0,'target_values_computed':False,'new_model_fits':0,
            'real_campaign_allowed':False}
    print(json.dumps(result))
    return result

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    verify()
