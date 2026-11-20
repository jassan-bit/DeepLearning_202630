"""Verify the source-bound documentary inventory without market computation."""
from __future__ import annotations
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parent

def rows(name):
    raw=(ROOT/name).read_bytes()
    if name.endswith('.gz'): raw=gzip.decompress(raw)
    return list(csv.DictReader(io.StringIO(raw.decode())))

def main():
    generated=['design_inventory.csv','design_aliases.csv','selection_opportunities.csv',
               'attempt_ledger.csv.gz','benchmark_bank.csv','design_inventory.json','reconciliation.json']
    with tempfile.TemporaryDirectory(prefix='e3-documentary-inventory-') as temporary:
        target=Path(temporary)
        for name in ('build_inventory.py','sources.zip'):
            shutil.copyfile(ROOT/name,target/name)
        subprocess.run([sys.executable,str(target/'build_inventory.py')],check=True,capture_output=True)
        for name in generated:
            assert (target/name).read_bytes()==(ROOT/name).read_bytes(),name
    designs=rows('design_inventory.csv');jobs=rows('attempt_ledger.csv.gz');trials=rows('selection_opportunities.csv')
    assert sum(int(r['completed_neural_fits_in_named_sources']) for r in designs)==124
    assert next(r for r in designs if r['design_id']=='N06')['completed_neural_fits_in_named_sources']=='93'
    assert sum(r['scope']=='extended_audit_796' for r in jobs)==796
    assert sum(r['scope']=='v3_controlled_development_30' for r in jobs)==30
    assert sum(r['scope']=='legacy_development_or_replicates' for r in jobs)==18
    with zipfile.ZipFile(ROOT/'sources.zip') as archive:
        assert not any(n.lower().endswith(('.npz','.pt','.onnx','.pdf')) for n in archive.namelist())
        manifest=json.loads(archive.read('source_manifest.json'))
        assert len(manifest['sources'])==21
        for record in manifest['sources']:
            body=archive.read(record['archive_name'])
            assert hashlib.sha256(body).hexdigest()==record['sha256']
            assert hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()==record['git_blob_sha']
        resources=list(csv.DictReader(io.StringIO(archive.read('audit_resources.csv').decode())))
        assert len({tuple(r[k] for k in ('dataset','model','seed','kind','phase')) for r in resources})==900
        assert sum(r['unclosed_crash_compute_unknown'].lower()=='true' for r in resources)==3
        old_trials=list(csv.DictReader(io.StringIO(archive.read('audit_hpo_metrics.csv').decode())))
        assert all(r['test_based_reselection_performed'].lower()=='false' for r in old_trials)
    groups={}
    for r in trials:
        groups.setdefault((r['scope'],r['dataset'],r['model']),[]).append(r)
    for values in groups.values():
        assert len(values)==3
        winner=min(values,key=lambda r:(float(r['validation_RMSE']),float(r['learning_rate'])))
        selected=[r for r in values if r['selected'].lower()=='true']
        assert selected==[winner]
    remote=json.loads((ROOT/'remote_source_verification.json').read_text())
    assert remote['verified_source_blobs']==21
    print(json.dumps({'status':'verified_source_bounded_documentary_reconciliation',
      'design_entries':len(designs),'HPO_trial_jobs':len(trials),'completed_job_records':len(jobs),
      'completed_neural_fit_records':824,'candidate_family_neural_fit_records':124,
      'source_records':21,'retrospective_not_preregistered':True,'new_model_fits':0,
      'new_market_forecasts':0,'R1_accredited':False,'R3_accredited':False}))

if __name__=='__main__': main()
