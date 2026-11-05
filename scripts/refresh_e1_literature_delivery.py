"""Refresh public literature delivery metadata without private model artifacts."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
LITERATURE=ROOT/'references/literature_e1/journal_update_20261105'
PUBLIC=ROOT/'book/_static/e1_deliverables_20261104'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    keys=re.findall(r'@\w+\s*\{\s*([^,]+),',(ROOT/'book/references.bib').read_text())
    required={'bucci2020realized','reisenhofer2022harnet','morenopino2024deepvol'}
    if len(keys)!=39 or len(set(keys))!=39 or not required.issubset(keys):
        raise ValueError('The bibliography must preserve36 entries and add only the three documented sources')
    names=['methodological_sources.json','methodological_comparison.csv',
           'bucci2020_primary_extraction.json','bucci2020_metrics.csv',
           'harnet2022_primary_code.json','deepvol2024_publication.json',
           'journal_index_matches.json','editorial_candidates.json',
           'references_access_39.json','references_access_39.csv','manifest.json']
    for name in names:
        if not (LITERATURE/name).is_file():raise FileNotFoundError(name)
    previous=read(PUBLIC/'public_manifest.json')
    # Existing configuration catalogues remain byte-identical to the audited
    # historical delivery; this update does not regenerate configurations.
    for name,count in [('cv_configuration_catalog.csv',1440),('final_configuration_catalog.csv',120)]:
        path=PUBLIC/name
        if sha(path)!=previous['files_sha256'][name]:
            raise ValueError('Historical public configuration catalogue changed: '+name)
        if len(list(csv.DictReader(path.open(newline=''))))!=count:
            raise ValueError('Wrong historical configuration count: '+name)
    now=datetime.now(timezone.utc).isoformat()
    extension={'updated_at_utc':now,'bibliographic_references':39,'BTC_full_text_studies':6,
        'new_methodological_full_texts':1,'new_methodological_author_code_sources':1,
        'new_publication_identity_only_sources':1,'new_BTC_full_text_inclusions':0,
        'T1':'Bucci2020 journal original; standalone feedforward reference; S&P500 monthly',
        'T2':'HARNet2022 author Conv1D code; original/journal not consulted',
        'T3':'DeepVol2024 QuantitativeFinance identity/DOI in author CV; original not consulted',
        'BTC_MLP_study_count':1,'BTC_generic_CNN_study_count':1,'BTC_explicit_shallow_CNN1D_study_count':0,
        'journal_index_evidence':'Independent secondary journal lists only; no individual article or Q1 verification',
        'original_benchmark_changed':False,'new_model_fits':0,'new_forecasts':0,
        'source_sha256':{str((LITERATURE/name).relative_to(ROOT)):sha(LITERATURE/name) for name in names}}
    write(PUBLIC/'literature_extension.json',extension)
    manifest=read(PUBLIC/'delivery_manifest.json')
    product=next(r for r in manifest['products'] if r['item']=='8.4')
    product.update(status='39 referencias; seis textos BTC y respaldo metodológico con niveles explícitos',
        evidence='references/literature_e1/journal_update_20261105/methodological_sources.json',
        limit='T1 original FNN de revista, T2 código autor Conv1D, T3 identidad editorial con original pendiente. No aumentan frecuencias BTC ni acreditan registros individuales o Q1.')
    manifest['literature_review']='39 references; six BTC full texts and separate T1 full paper, T2 author code, T3 publication identity only'
    manifest['literature_extension']=extension
    write(PUBLIC/'delivery_manifest.json',manifest)
    with (PUBLIC/'delivery_manifest.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifest['products'][0]),lineterminator='\n')
        writer.writeheader();writer.writerows(manifest['products'])
    declaration=read(PUBLIC/'declaration_and_chronology.json')
    declaration['literature_extension']=extension
    declaration['source_sha256'].update(extension['source_sha256'])
    write(PUBLIC/'declaration_and_chronology.json',declaration)
    previous['literature_refresh_at_utc']=now
    previous['literature_refresh_script_sha256']=sha(__file__)
    previous['source_sha256'].update(extension['source_sha256'])
    previous['files_sha256']={p.relative_to(PUBLIC).as_posix():sha(p) for p in sorted(PUBLIC.rglob('*'))
        if p.is_file() and p.name!='public_manifest.json'}
    write(PUBLIC/'public_manifest.json',previous)
    print(json.dumps({'references':39,'BTC_studies':6,'new_methodological_paper':1,
                     'new_models_fitted':0,'new_forecasts':0,'historical_configurations_preserved':1560}))


if __name__=='__main__':main()
