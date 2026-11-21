"""Read-only saved-result reconstruction of the final d6 E3 edition.

AI assistance declared. Original v1/v1.1 source bytes remain unchanged. This
version explicitly changes their runtime edition/profile context, not their
reconstruction mathematics. No training, forward pass or resampling occurs.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,statistics,sys
from pathlib import Path

SOURCE_COMMIT='d6cf7f1af6800ec1ac85a26347bdde52380e362f'
BASE_MANIFEST='a35a5f56d6288ad5640f347b59415f00968277dceb96096fda0dc83db75b0206'
ADDON_MANIFEST='6818420f3efa61eb38b456f2bc2a9da3bf4052e48b9d210c45b18ede886e116a'
HERE=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def load_context(case):
    kit=case/'kit';addon=case/'addon'
    require(digest(kit/'MANIFEST.json')==BASE_MANIFEST,'Original base manifest differs')
    require(digest(addon/'MANIFEST.json')==ADDON_MANIFEST,'Original addon manifest differs')
    sys.path.insert(0,str(kit));sys.path.insert(0,str(addon))
    import prepare_external
    prepare_external.inspect_kit(kit);prepare_external.inspect_kit(addon)
    # Explicit edition context override after exact verification. No frozen
    # source file, input reference, mathematics or old receipt is edited.
    prepare_external.SOURCE_COMMIT=SOURCE_COMMIT
    spec=importlib.util.spec_from_file_location('frozen_extended_current_context',addon/'run_extended_case.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.HERE=HERE
    require(module.SOURCE_COMMIT==SOURCE_COMMIT,'Edition override failed')
    return module
def regenerate_V4(source,out,mode,request):
    prefix=source/'experiments/e3_next_evidence_20261121/development/cpu_pilot_results'
    input_path=prefix/'assessment_predictions.csv'
    require(digest(input_path)=='3de0b8cad079c4b03f3d0cd7f7d35848b0acc9807a91e88c563bd34cf2a4c2c0','V4 saved points differ')
    with input_path.open(newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==1210,'V4 saved-row count differs')
    grouped={};seen=set()
    for r in rows:
        k=(r['name'],r['seed']);key=(*k,r['date']);require(key not in seen,'Duplicate V4 row');seen.add(key)
        y,p=float(r['target']),float(r['prediction']);require(math.isfinite(y) and math.isfinite(p),'Nonfinite V4 row')
        grouped.setdefault(k,[]).append((p-y,r['date']))
    require(len(grouped)==22 and all(len(v)==55 for v in grouped.values()),'V4 trajectories/calendar differ')
    recipes={}
    for (name,seed),values in grouped.items():
        errors=[x[0] for x in values];rmse=math.sqrt(math.fsum(e*e for e in errors)/len(errors));mae=math.fsum(abs(e) for e in errors)/len(errors)
        recipes.setdefault(name,[]).append((rmse,mae))
    with (prefix/'recipe_summary.csv').open(newline='') as f:reference={r['name']:r for r in csv.DictReader(f)}
    require(set(recipes)==set(reference) and len(recipes)==6,'V4 recipe identities differ')
    tolerance=request.get('V4_tolerance')
    require(isinstance(tolerance,dict) and set(tolerance)=={'absolute','relative'},'Declare V4 tolerances before execution')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in tolerance.values()),'V4 tolerance values incomplete')
    computed=[];comparisons=[]
    for name,values in recipes.items():
        row={'name':name,'trajectories':len(values),'mean_RMSE':statistics.mean(v[0] for v in values),'mean_MAE':statistics.mean(v[1] for v in values),'sample_SD_RMSE':statistics.stdev(v[0] for v in values) if len(values)>1 else None}
        for field in ('mean_RMSE','mean_MAE','sample_SD_RMSE'):
            expected=reference[name][field];actual=row[field]
            if expected=='':require(actual is None,'Analytical-control SD must remain absent');continue
            expected=float(expected);error=abs(actual-expected)
            require(error<=tolerance['absolute']+tolerance['relative']*abs(expected),'V4 discrepancy exceeds prior tolerance')
            comparisons.append({'name':name,'field':field,'unit':'unannualized sigma7','absolute_discrepancy':error})
        computed.append(row)
    target=out/'current_V4_recipe_summary.csv'
    with target.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(computed[0]));w.writeheader();w.writerows(computed)
    decision=json.loads((prefix/'pilot_result.json').read_text())
    receipt={'status':'pass','source_commit':SOURCE_COMMIT,'saved_predictions':1210,'trajectories':22,'recipes':6,'comparisons':comparisons,'max_absolute_discrepancy_sigma7':max(r['absolute_discrepancy'] for r in comparisons),'tolerances_declared_before_execution':tolerance,'input_SHA256':digest(input_path),'derived_csv_SHA256':digest(target),'original_V4_decision':'ABANDON_CURRENT_V4_RECIPE','new_model_fits':0,'new_model_predictions':0,'new_statistical_resamples':0,'human_execution_claimed':False}
    (out/'current_V4_comparison.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
def run(case,mode):
    case=Path(case).resolve();module=load_context(case)
    profile=HERE/'SOFTWARE_PROFILE.json' if mode=='software-validation' else case/'HUMAN_REQUEST.json'
    request=json.loads(profile.read_text())
    require(request.get('source_commit')==SOURCE_COMMIT,'Current source required')
    module.validate_request(request,mode)
    require(request.get('V4_development_scope_accepted') is True,'Accept the additional current V4 development-table scope')
    # Validate the added tolerance before any book operation.
    v=request.get('V4_tolerance');require(isinstance(v,dict) and set(v)=={'absolute','relative'} and all(type(n) in (int,float) and math.isfinite(n) and n>=0 for n in v.values()),'Complete V4 tolerances before execution')
    receipt=module.run(case,mode)
    v4=regenerate_V4(case/'source',case/'output',mode,request)
    receipt.update({'status':'final_edition_book_additive19x4_V3_and_V4_saved_results_passed','final_edition_wrapper_version':'1.2','previous_f5_receipt_relabelled':False,'original_frozen_sources_byte_unchanged':True,'runtime_context_override_declared':{'source_commit':SOURCE_COMMIT,'profile_and_manifest_directory':str(HERE)},'additional_V4_saved_result_comparison':v4,'human_execution_claimed':False,'human_attestation_still_required':True})
    (case/'output/CURRENT_EDITION_EXECUTION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--mode',choices=['software-validation','human-external'],required=True);a=p.parse_args()
    try:print(json.dumps(run(a.case,a.mode),sort_keys=True))
    except Exception as e:print(type(e).__name__+': '+str(e),file=sys.stderr);raise SystemExit(2)
