"""Verify published historical receipts. This does not repeat their work."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCE='7db8040619f602d7c52ae0c8bf54f7b6f5b5110e'
BASE='https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'+SOURCE+'/'
REFERENCES=[
    ('A','book_e3/_static/e3_remediacion_20261118/reproduction/reproduction_report.json','6a05cb1810979b07f5af4743cdcfad6183d201fb072094a3652b0178ec17a4ae'),
    ('B','experiments/e3_completion_20261120/reproduction/fifteen_replay/receipt.json','d4049e76c74e184706bc434dd032a5e52002dccca3d4fad6716f1ee4d2ea99b4')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    reports=[]
    for level,path,expected in REFERENCES:
        req=urllib.request.Request(BASE+path,headers={'User-Agent':'E3-receipt-audit/1.0'})
        with urllib.request.urlopen(req,timeout=60) as response:
            raw=response.read(512001)
            if response.status!=200 or len(raw)>512000:raise ValueError('Invalid receipt response')
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=expected:raise ValueError('Historical receipt hash differs')
        data=json.loads(raw)
        if data['status']!='passed':raise ValueError('Historical receipt was not passed')
        if level=='A':
            tables=data['tables']
            if len(tables)!=11 or not all(t['status']=='passed' for t in tables):raise ValueError('Eleven passed tables required')
            details={'tables':[{'name':t['name'],'rows':t['rows'],'bytes_identical':t['bytes_identical'],
                       'absolute_tolerance':t['absolute_tolerance'],'relative_tolerance':t['relative_tolerance'],
                       'discrepancies_by_field':t['field_discrepancies']} for t in tables],
                     'original_execution_started':data['started_at_utc'],'original_execution_finished':data['finished_at_utc']}
        else:
            if len(data['checkpoints'])!=15 or sum(x['rows'] for x in data['checkpoints'])!=825:raise ValueError('Fifteen checkpoints/825 rows required')
            details={'checkpoints':15,'saved_prediction_rows':825,'state_used':'best',
                     'absolute_tolerance_sigma7':1e-6,'max_absolute_error_sigma7':max(x['max_absolute_error_sigma7'] for x in data['checkpoints'])}
        reports.append({'level':level,'source_commit':SOURCE,'url':BASE+path,'HTTP':200,'bytes':len(raw),'sha256':digest,
                        'receipt_verified':True,'work_reexecuted_by_this_audit':False,'details':details})
    out={'status':'passed','AI_assistance_declared':True,'observed_at_UTC':datetime.now(timezone.utc).isoformat(),
         'reports':reports,'level_C_training_reproduced':False,'human_independent_reproduction_observed':False,
         'new_model_fits':0,'model_inferences':0,'new_calibrations':0,
         'scope':'Published historical receipt verification, not reexecution of A/B and not C.'}
    Path(args.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'passed','verified_receipts':len(reports),'tables_A':11,'checkpoints_B':15,'saved_rows_B':825,'work_reexecuted':False}))


if __name__=='__main__':main()
