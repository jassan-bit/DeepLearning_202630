"""Optional read-only recovery and evaluation of authenticated original V5 weights.

Only GET operations are allowed remotely. No setup/run/assess/backup, optimizer,
training, recalibration or new reserved data. Outputs stay in a private local
folder. AI assistance declared. Source/data rights remain their original terms.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import time
import zipfile

SOURCE_COMMIT='c4ca86f1431afeaa958ff0b544b453ab17a4d7b8'
SOURCE_MANIFEST_SHA='4706f20a00508e492a54e25873b4885cead3fa7660d43bdb18c5aeda4fce2509'
RESULT_COMMIT='05c1a637f13356dd1df6b8c5aa8f4676dda61c12'
PUBLIC_ZIP_SHA='beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c'
PUBLIC_ZIP_BYTES=124067
DATA_SHA='186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683'
DATA_BYTES=460090
CIPHER_SHA='54800d43632f341576e8f1737b15a28b96fa6d2098ea5c8d6fc365c180824607'
CIPHER_BYTES=1328694
PLAN_SHA='fac6a2003e8a315754bbfc3f413072b97f861830b4c5143d4ae2733629a88e43'
CASE_ID='seoi-v5-dev-3daac6f1124c4e028a97d9768e8f64e5'
REPOSITORY='jassan-bit/DeepLearning_202630'
BRANCH='e3-seoi-v5-run-01'
PREFIX='runs/e3_seoi_v5/run-01'
SEEDS=tuple(range(20261127,20261132))
BEST_EPOCHS={20261127:39,20261128:3,20261129:2,20261130:31,20261131:19}
RECIPE='input_only_mse'
VERSION='original-v5-frozen-weight-fit-selection-reanalysis-1'
# Fixed before any real-key run. A mismatch stops; never adjust after observing it.
PARITY_ATOL=1e-8
PARITY_RTOL=1e-5


class ReanalysisError(RuntimeError):
    pass


def require(condition,code):
    if not condition:raise ReanalysisError(code)


def sha(raw):return hashlib.sha256(raw).hexdigest()
def canon(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())


def safe(path):
    path=Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path,*path.parents)),'symlink_path_refused')
    return path


def verify_source(source_root):
    source_root=safe(source_root)
    manifest_raw=(source_root/'SOURCE_MANIFEST.json').read_bytes()
    require(sha(manifest_raw)==SOURCE_MANIFEST_SHA,'original_source_manifest_differs')
    files=json.loads(manifest_raw)['files']
    seen=set()
    for item in files:
        name=item['path'];p=PurePosixPath(name)
        require(not p.is_absolute() and p.as_posix()==name and '..' not in p.parts and name not in seen,
                'unsafe_or_duplicate_original_source_member')
        seen.add(name);path=safe(source_root/name);raw=path.read_bytes()
        require(len(raw)==item['bytes'] and sha(raw)==item['sha256'],'original_source_file_differs')
    required={'seoi_v5/runner.py','seoi_v5/model.py','seoi_v5/data.py','seoi_v5/github_store.py','protocol.json'}
    require(required<=seen,'original_source_incomplete')
    return source_root


def import_original(source_root):
    sys.path.insert(0,str(source_root))
    from seoi_v5 import runner,github_store
    for module in (runner,github_store):
        require(Path(module.__file__).resolve().is_relative_to(source_root.resolve()),'foreign_cached_V5_module')
    return runner,github_store


def pinned_readonly_store(store_module,token,backup_secret,**kwargs):
    """The inherited restore client is frozen to an immutable result commit."""
    class ReadOnlyFrozenStore(store_module.GithubStore):
        def _ref(self):
            return {'object':{'sha':RESULT_COMMIT}}
        def _request(self,method,path,payload=None,**options):
            require(method=='GET' and payload is None,'remote_write_forbidden_in_weight_reanalysis')
            self.get_operations+=1
            return super()._request(method,path,payload,**options)
    instance=ReadOnlyFrozenStore(token=token,backup_secret=backup_secret,
        repository=REPOSITORY,branch=BRANCH,source_commit=SOURCE_COMMIT,prefix=PREFIX,**kwargs)
    instance.get_operations=0
    return instance


def recover(store_module,restore_dir,credentials):
    restore_dir=safe(restore_dir)
    if restore_dir.exists():
        require(restore_dir.is_dir(),'existing_case_not_directory')
        return dict(restored=False,remote_GET_operations=0,scope='Existing case will be independently authenticated by frozen hashes')
    token=credentials.get('token','');key=credentials.get('backup_secret','')
    require(isinstance(token,str) and len(token)>=20,'private_GitHub_read_token_required')
    require(isinstance(key,str) and len(key)>=32 and key!=token,'original_independent_V5_backup_key_required')
    store=pinned_readonly_store(store_module,token,key)
    try:
        status=store.status()
        require(status.get('head')==RESULT_COMMIT and status.get('checkpoint_sha256')==CIPHER_SHA and
                status.get('checkpoint_bytes')==CIPHER_BYTES and status.get('case_id')==CASE_ID and
                status.get('plan_sha256')==PLAN_SHA and status.get('completed_neural_fits')==30 and
                status.get('session_open') is False,'frozen_encrypted_backup_identity_differs')
        receipt=store.restore(restore_dir)
        require(receipt.get('remote_verified') is True and receipt.get('commit')==RESULT_COMMIT,
                'frozen_restore_not_verified')
        return dict(restored=True,remote_GET_operations=store.get_operations,
                    encrypted_backup_sha256=CIPHER_SHA,result_commit=RESULT_COMMIT)
    finally:
        store._secret=None;store._token=None


def public_evidence(path):
    raw=safe(path).read_bytes()
    require(len(raw)==PUBLIC_ZIP_BYTES and sha(raw)==PUBLIC_ZIP_SHA,'original_public_results_differs')
    archive=zipfile.ZipFile(io.BytesIO(raw))
    require(len(archive.namelist())==len(set(archive.namelist()))==110,'original_public_inventory_differs')
    plan=json.loads(archive.read('plan.json'))
    require(sha(canon(plan))==PLAN_SHA and plan.get('case_id')==CASE_ID and tuple(plan.get('seeds',()))==SEEDS and
            plan.get('source_manifest_sha256')==SOURCE_MANIFEST_SHA,
            'original_public_plan_differs')
    jobs={}
    inventories={r['job']['seed']:r for r in json.loads(archive.read('checkpoint_inventory.json')) if r['job']['recipe']==RECIPE}
    require(set(inventories)==set(SEEDS),'original_candidate_checkpoint_inventory_differs')
    for seed in SEEDS:
        name=f'jobs/seed{seed}/{RECIPE}/job.json'
        job=json.loads(archive.read(name));inventory=inventories[seed]
        require(job.get('status')=='completed_validated_fit' and job['job']==inventory['job'] and
                job['checkpoint_sha256']==inventory['checkpoint_sha256'] and
                job['best_epoch']==BEST_EPOCHS[seed] and job['plan_sha256']==PLAN_SHA,
                'original_selected_checkpoint_evidence_differs')
        jobs[seed]=job
    rows=list(csv.DictReader(io.StringIO(archive.read('predictions.csv').decode())))
    assessment={seed:sorted([r for r in rows if r['recipe']==RECIPE and r['seed']==str(seed)],key=lambda r:r['date']) for seed in SEEDS}
    require(all(len(v)==55 for v in assessment.values()),'original_assessment_predictions_incomplete')
    return dict(plan=plan,plan_raw=archive.read('plan.json'),jobs=jobs,assessment=assessment)


def validate_case(runner,case_dir,data_file,evidence):
    case_dir=safe(case_dir);data_file=safe(data_file)
    raw=data_file.read_bytes()
    require(len(raw)==DATA_BYTES and sha(raw)==DATA_SHA,'original_cohort_transport_differs')
    plan=read(case_dir/'plan.json');prepared=read(case_dir/'prepared.json')
    require((case_dir/'plan.json').read_bytes()==evidence['plan_raw'] and sha(canon(plan))==PLAN_SHA and
            prepared.get('plan_sha256')==PLAN_SHA and prepared.get('plan_file_sha256')==sha(evidence['plan_raw']),
            'restored_scientific_plan_differs')
    require(sha((case_dir/'protocol.json').read_bytes())==plan['protocol_sha256'],'restored_protocol_differs')
    fingerprints={}
    for seed,reference in evidence['jobs'].items():
        directory=safe(case_dir/'jobs'/reference['job']['job_id'])
        job=read(directory/'job.json');pointer=read(directory/'checkpoint.json')
        require(job==reference,'restored_selected_job_differs_from_public_evidence')
        name=reference['checkpoint_file']
        require(PurePosixPath(name).name==name and name.startswith('snapshot_') and name.endswith('.pt'),
                'unsafe_selected_checkpoint_filename')
        path=safe(directory/name);snapshot=path.read_bytes()
        require(pointer.get('file')==name and pointer.get('sha256')==reference['checkpoint_sha256'] and
                pointer.get('bytes')==len(snapshot) and sha(snapshot)==reference['checkpoint_sha256'] and
                pointer.get('request_sha256')==reference['request_sha256'],'authentic_selected_snapshot_hash_differs')
        fingerprints[str(path)]=sha(snapshot)
    return plan,fingerprints


def evaluate_frozen(runner,model_module,data,case_dir,evidence,*,load_snapshot=None,fixture=False):
    """Pure evaluation core; fixtures are never labelled authentic evidence."""
    import numpy as np
    import torch
    torch.set_num_threads(2)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:require(torch.get_num_interop_threads()==1,'inference_thread_cohort_differs')
    torch.use_deterministic_algorithms(True)
    loader=load_snapshot or runner.load_snapshot
    all_rows=[];metric_rows=[];parities=[];pending_models=[]
    for seed in SEEDS:
        ref=evidence['jobs'][seed]
        state,pointer=loader(Path(case_dir)/'jobs'/ref['job']['job_id'],ref['request_sha256'])
        require(state.get('best_epoch')==BEST_EPOCHS[seed] and state.get('request_sha256')==ref['request_sha256'] and
                state.get('best') is not None and pointer['sha256']==ref['checkpoint_sha256'],'selected_state_identity_differs')
        scales=data['scalers'][RECIPE]
        model=model_module.build_model(RECIPE,seed,u_scale=scales['u_scale'],y_scale=scales['y_scale']).cpu()
        model.load_state_dict(state['best']);model.eval()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        def predict(split,_model=model):
            with torch.inference_mode():
                p=_model.forward_components(torch.as_tensor(split['X'][RECIPE]),torch.as_tensor(split['A']),
                    torch.as_tensor(split['scale']))['prediction'].detach().cpu().numpy()
            require(p.shape==split['y'].shape and np.isfinite(p).all() and
                    np.all(p+1e-12>=np.sqrt(split['A'])),'invalid_or_below_support_prediction')
            return p
        selection=predict(data['selection']);assessment=predict(data['assessment'])
        original=evidence['assessment'][seed]
        require([str(x) for x in data['assessment']['dates']]==[r['date'] for r in original],
                'assessment_calendar_identity_differs')
        observed_target=np.asarray([float(r['target']) for r in original])
        saved_prediction=np.asarray([float(r['prediction']) for r in original])
        require(np.array_equal(data['assessment']['y'],observed_target),'assessment_target_identity_differs')
        max_error=float(np.max(np.abs(assessment-saved_prediction)))
        require(np.allclose(assessment,saved_prediction,rtol=PARITY_RTOL,atol=PARITY_ATOL),
                'frozen_assessment_parity_failed_do_not_relax_tolerance')
        selection_rmse=float(np.sqrt(np.mean((selection-data['selection']['y'])**2)))
        require(np.isclose(selection_rmse,ref['selection_RMSE'],rtol=PARITY_RTOL,atol=PARITY_ATOL),
                'frozen_selection_metric_parity_failed_do_not_relax_tolerance')
        parities.append(dict(seed=seed,best_epoch=ref['best_epoch'],checkpoint_sha256=pointer['sha256'],
            assessment_points=55,assessment_max_abs_difference=max_error,selection_RMSE=selection_rmse,
            saved_selection_RMSE=ref['selection_RMSE'],rtol=PARITY_RTOL,atol=PARITY_ATOL,passed=True))
        pending_models.append((seed,model,before,selection,predict))
    # All source-assessment/selection checks pass before any fit results are produced.
    for seed,model,before,selection,predict in pending_models:
        fit=predict(data['train'])
        for split_name,split,prediction in (('fit',data['train'],fit),('selection',data['selection'],selection)):
            e=prediction-split['y']
            metric_rows.append(dict(seed=seed,split=split_name,n=len(e),RMSE=float(np.sqrt(np.mean(e*e))),
                MSE=float(np.mean(e*e)),MAE=float(np.mean(np.abs(e))),mean_prediction_minus_target=float(e.mean())))
            all_rows.extend(dict(recipe=RECIPE,seed=seed,split=split_name,date=str(date),target=float(y),prediction=float(p))
                for date,y,p in zip(split['dates'],split['y'],prediction))
        require(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()),'weights_changed_during_readonly_inference')
    require(len(all_rows)==2930 and len(metric_rows)==10 and len(parities)==5,'reanalysis_geometry_differs')
    return all_rows,metric_rows,parities,dict(status='SOFTWARE_FIXTURE_ONLY' if fixture else 'AUTHENTIC_FROZEN_WEIGHT_REANALYSIS',
        fit_selection_new_forecast_points=2930,assessment_parity_forward_points=275,total_forward_points=3205,
        forecast_training_steps=0,optimizer_instantiated=False,weights_updated=False,final_reserved_test_opened=False,
        retrospective_development_only=True,R3_accredited=False)


def write_outputs(output_dir,rows,metrics,parities,receipt):
    output_dir=safe(output_dir);require(not output_dir.exists(),'existing_outputs_preserved_use_verify_only')
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.v5-evaluation-',dir=output_dir.parent))
    try:
        os.chmod(staging,0o700)
        for name,values in (('fit_selection_predictions_PRIVATE.csv',rows),('fit_selection_metrics_PRIVATE.csv',metrics),
                            ('frozen_assessment_and_selection_parity.csv',parities)):
            with (staging/name).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
        (staging/'execution_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        manifest={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in staging.iterdir() if p.is_file()}
        (staging/'OUTPUT_MANIFEST.json').write_text(json.dumps(dict(private_local_outputs=True,
            automatic_publication=False,files=manifest),indent=2)+'\n')
        for p in staging.iterdir():os.chmod(p,0o600)
        require(not output_dir.exists(),'output_target_appeared_preserve_existing')
        os.rename(staging,output_dir)
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging)


def verify_outputs(output_dir):
    output_dir=safe(output_dir);manifest=read(output_dir/'OUTPUT_MANIFEST.json')
    require(manifest.get('private_local_outputs') is True and manifest.get('automatic_publication') is False,
            'output_privacy_marker_differs')
    for name,record in manifest['files'].items():
        require(PurePosixPath(name).name==name,'unsafe_output_member')
        raw=safe(output_dir/name).read_bytes()
        require(len(raw)==record['bytes'] and sha(raw)==record['sha256'],'existing_output_hash_differs')
    return dict(status='EXISTING_PRIVATE_OUTPUTS_VERIFIED',files=len(manifest['files']),new_forward_points=0,
                new_training_steps=0,remote_writes=0)


def run(source_root,data_file,public_zip,case_dir,output_dir,credentials):
    require(not safe(output_dir).exists(),'existing_outputs_preserved_use_verify_only')
    case_path,output_path=safe(case_dir),safe(output_dir)
    require(not output_path.is_relative_to(case_path) and not case_path.is_relative_to(output_path),
            'private_analysis_outputs_must_be_separate_from_original_case')
    started=utc();wall=time.perf_counter();cpu=time.process_time()
    source_root=verify_source(source_root)
    require(not output_path.is_relative_to(source_root),'analysis_must_not_write_into_original_source')
    raw=safe(data_file).read_bytes()
    require(len(raw)==DATA_BYTES and sha(raw)==DATA_SHA,'original_cohort_transport_differs')
    runner,store_module=import_original(source_root)
    evidence=public_evidence(public_zip)
    plan=evidence['plan']
    require(sha((source_root/'protocol.json').read_bytes())==plan['protocol_sha256'],'pinned_protocol_differs')
    import numpy as np
    import torch
    require(torch.__version__=='2.8.0+cpu' and np.__version__=='2.3.5','use_pinned_CPU_inference_runtime')
    data_module,model_module=runner._modules(source_root)
    data=data_module.load_data(Path(data_file))
    require(runner.data_identity(data)==plan['data_identities'] and runner.digest(data['admission'])==plan['admission_sha256'] and
            runner.digest(data['scalers'])==plan['scalers_sha256'],'pinned_arrays_or_preprocessing_differ')
    restoration=recover(store_module,case_dir,credentials)
    plan,fingerprints=validate_case(runner,case_dir,data_file,evidence)
    rows,metrics,parities,counts=evaluate_frozen(runner,model_module,data,case_dir,evidence)
    require(all(sha(Path(path).read_bytes())==digest for path,digest in fingerprints.items()),'source_checkpoint_changed')
    observed=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',
        cryptography=importlib.metadata.version('cryptography'),
        torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    receipt=dict(version=VERSION,started_utc=started,finished_utc=utc(),wall_seconds=time.perf_counter()-wall,
        cpu_seconds=time.process_time()-cpu,source_commit=SOURCE_COMMIT,result_commit=RESULT_COMMIT,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,public_results_sha256=PUBLIC_ZIP_SHA,cohort_NPZ_sha256=DATA_SHA,
        original_plan_sha256=PLAN_SHA,case_id=CASE_ID,checkpoint_hashes=fingerprints,
        original_training_cohort=plan['cohort_binding'],observed_inference_cohort=observed,
        training_resume_claimed=False,cohort_policy='Independent frozen-weight inference; actual cohort recorded, original resume guard unchanged; parity gates mandatory',
        restoration=restoration,remote_writes=0,**counts,AI_assistance_declared=True,
        adequacy_approved=False,calibration_approved=False,prospective_gap_margin_created=False,
        outputs_private=True,automatic_publication=False)
    write_outputs(output_dir,rows,metrics,parities,receipt)
    return dict(status=counts['status'],private_output_dir=str(safe(output_dir)),
        fit_selection_points=2930,assessment_parity_points=275,training_steps=0,remote_writes=0,
        adequacy_approved=False,calibration_approved=False,R3_accredited=False)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--data-file',type=Path)
    parser.add_argument('--public-zip',type=Path)
    parser.add_argument('--case-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();credentials={}
    try:
        if args.verify_only:result=verify_outputs(args.output_dir)
        else:
            require(all((args.source_root,args.data_file,args.public_zip,args.case_dir)),'explicit_local_paths_required')
            supplied=sys.stdin.read(65537);require(len(supplied)<=65536,'credentials_stdin_too_large')
            credentials=json.loads(supplied or '{}')
            require(isinstance(credentials,dict) and set(credentials)<= {'token','backup_secret'},'invalid_private_credentials_object')
            result=run(args.source_root,args.data_file,args.public_zip,args.case_dir,args.output_dir,credentials)
        print(json.dumps(result,allow_nan=False));return 0
    except KeyboardInterrupt:
        print(json.dumps(dict(status='INTERRUPTED_PRIVATE_FILES_PRESERVED',new_training_steps=0)));return 130
    except Exception as exc:
        code=str(exc) if isinstance(exc,ReanalysisError) else 'reanalysis_stopped_preserve_private_files'
        print(json.dumps(dict(status='STOPPED',diagnostic=code,error_type=type(exc).__name__,
            remote_writes=0,new_training_steps=0)));return 1
    finally:credentials.clear()


if __name__=='__main__':raise SystemExit(main())
