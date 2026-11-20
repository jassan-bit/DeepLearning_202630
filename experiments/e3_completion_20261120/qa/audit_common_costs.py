"""Read-only review of saved serial ORT measurements; never execute a model."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT=Path('/workspace/e3_completion_delivery')
R=ROOT/'costs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    receipt_path=R/'measured/receipt.json'
    receipt=json.loads(receipt_path.read_text())
    registry=json.loads((R/'runtime_registry_private.json').read_text())
    assert receipt['registry_sha256']==sha(R/'runtime_registry_private.json')
    assert receipt['input_json_sha256']==sha(R/'input_fixture.json')
    assert len(receipt['models'])==len(registry['models'])==15
    rows=[]
    for row,ref in zip(receipt['models'],registry['models']):
        assert row['model']==ref['model'] and row['cpu']==receipt['cpu_logical_id']
        assert row['ort_threads']==row['batch_size']==1 and row['warmups']==10 and row['timed_calls']==100
        assert row['model_sha256']==sha(ref['model_path'])==ref['model_sha256']
        assert row['metadata_sha256']==sha(ref['metadata_path'])==ref['metadata_sha256']
        assert row['software']=={'python':'3.12.14','numpy':'2.3.5','onnxruntime':'1.23.2'}
        calls=row['calls_ms']
        assert len(calls)==100 and all(math.isfinite(v) and v>0 for v in calls)
        assert statistics.median(calls)==row['median_ms']
        values=sorted(calls)
        p95=values[94]+.05*(values[95]-values[94])
        assert abs(p95-row['p95_ms'])<1e-12
        assert row['process_peak_RSS_bytes']>=row['process_baseline_RSS_bytes']>0
        rows.append({k:row[k] for k in ['model','model_sha256','metadata_sha256','median_ms','p95_ms','process_peak_RSS_bytes']})
    with (R/'measured/costs.csv').open(newline='') as f:
        table=list(csv.DictReader(f))
    assert len(table)==15
    for a,b in zip(table,receipt['models']):
        assert a['model']==b['model'] and float(a['median_ms'])==b['median_ms'] and float(a['p95_ms'])==b['p95_ms']
    report={'status':'common_ORT_cost_measurements_provenance_and_aggregation_QA_passed',
        'AI_assistance_declared':True,'new_QA_model_fits':0,'new_QA_model_predictions':0,
        'receipt_sha256':sha(receipt_path),'models_verified':15,
        'profile':'Fresh sequential process per model; same raw window, one logical CPU affinity, ORT CPU one intra/inter thread, batch1,10 warmups and100 timed calls; frozen model-specific scaler; no Torch in benchmark worker.',
        'measurements_scope':'Normalized-input ORT forward, session load and first call; separately logged total process RSS. These measurements do not establish Dash, transport, explanations, incremental memory, physical-core isolation, energy or FLOPs. Filesystem caches were not flushed; loads are not certified cold.',
        'median_latency_range_ms':[min(x['median_ms'] for x in rows),max(x['median_ms'] for x in rows)],
        'total_process_peak_RSS_range_MiB':[min(x['process_peak_RSS_bytes'] for x in rows)/1048576,max(x['process_peak_RSS_bytes'] for x in rows)/1048576],
        'timing_model_calls_separate_from_diagnostic_predictions':15*111,
        'serving_activation_or_Render_capacity_proven':False,
        'prior_metadata_race':'Earlier measurements are preserved in costs/attempts/pre_metadata_freeze; current receipt matches every final ONNX and metadata file.',
        'records':rows}
    (ROOT/'qa/common_cost_review.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    result=audit()
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
