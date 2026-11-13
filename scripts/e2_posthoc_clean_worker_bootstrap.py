"""Execute the unchanged clean worker with its original ARIMA name shim."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys

OLD_SHA='5d8ce059f96f86a35b7e2d2530e3fb776827240fcb29df920b4d05374f139fe9'
REGISTRATION='experiments/e2_posthoc_environment_20261113/registration.json'
MODELS=('FCN-1D','ResNet-1D','Inception-1D','LSTM-autoencoder con predictor','TimesNet','TSMixer',
        'TiDE','TimeMixer','PatchTST','iTransformer','MLP16','LSTM32','GRU32')

def verified_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def execute(old_controller,expected_sha,spec_path,output):
    root=Path(__file__).resolve().parents[1]
    registration=json.loads((root/REGISTRATION).read_bytes())
    for relative,expected in registration['files_sha256'].items():
        p=root/relative
        if any(x.is_symlink() for x in (p,*p.parents)) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise ValueError('Registered environment adapter changed: '+relative)
    controller=Path(old_controller).resolve()
    if expected_sha!=OLD_SHA or hashlib.sha256(controller.read_bytes()).hexdigest()!=OLD_SHA:
        raise ValueError('Original worker bytes must be preserved')
    job=json.loads(Path(spec_path).read_bytes())
    if job.get('controller_source_sha256')!=OLD_SHA:
        raise ValueError('Worker specification belongs to a different controller')
    clean=Path(job['clean_runtime']['clean_source'])
    sys.path[:0]=[str(clean/'scripts'),str(controller.parent)]
    old=verified_module(controller,'e2_unchanged_posthoc_clean_worker')
    if job['model']=='ARIMA' and job['kind']=='reproduction':
        shim=verified_module(root/'scripts/e2_arima_prediction_compat.py','e2_registered_arima_name_compat')
        admission=shim.install_prediction_raw_alias(clean)
        admission.update(environment_adapter_registration_sha256=hashlib.sha256((root/REGISTRATION).read_bytes()).hexdigest(),
                         bootstrap_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         original_worker_source_sha256=OLD_SHA)
        old.immutable_json(Path(output)/'arima_name_compatibility.json',admission)
        print(json.dumps({'event':'original_ARIMA_read_only_name_compatibility_reused',
                          'numeric_predictions_changed':False}),flush=True)
    return old.worker(Path(spec_path),Path(output))

def preflight(old_controller,expected_sha,runtime_path,original_output):
    """One zero-input forward per fixed body; no market observations or fits."""
    root=Path(__file__).resolve().parents[1];runtime_path=Path(runtime_path)
    runtime_raw=runtime_path.read_bytes();runtime=json.loads(runtime_raw)
    controller=Path(old_controller).resolve()
    if expected_sha!=OLD_SHA or hashlib.sha256(controller.read_bytes()).hexdigest()!=OLD_SHA:
        raise ValueError('Original preflight controller bytes differ')
    clean=Path(runtime['clean_source']);sys.path[:0]=[str(clean/'scripts'),str(controller.parent)]
    old=verified_module(controller,'e2_unchanged_posthoc_preflight');old.verify_clean_runtime(runtime)
    import torch
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):raise ValueError('Preserved A100 required')
    torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    import e2_reduced_final as final
    from e2_published_pilot_training import predict_pilot
    output=Path(original_output);freeze_raw=(output/'final/final_freeze.json').read_bytes()
    if hashlib.sha256(freeze_raw).hexdigest()!='c0e9e2589dca9fa03b341bb72f87d8b541d33b4721715622daeae4438dbd60b4':
        raise ValueError('Frozen catalogue metadata changed')
    freeze=json.loads(freeze_raw);accepted=[]
    for name in MODELS:
        row=next(r for r in freeze['models'] if r['model']==name and r['seed']==271801)
        checkpoint=output/'final'/row['checkpoint']
        spec=json.loads((checkpoint.parent.parent/'worker_spec.json').read_bytes())
        model=final._trial_builder(name,spec['selection']['winner_trial'],271801,spec.get('source_roots'))
        predicted=predict_pilot(model,torch.zeros((1,7,5),dtype=torch.float32),batch_size=1,device='cuda:0')
        if predicted.shape!=(1,) or not torch.isfinite(predicted).all():raise ValueError('Synthetic forward shape/finite check failed: '+name)
        accepted.append({'model':name,'seed':271801,'input_shape':[1,7,5],'forward_shape':[1],
                         'parameters':sum(p.numel() for p in model.parameters()),'market_values_read':False})
        del model;torch.cuda.empty_cache()
        print(json.dumps({'event':'clean_synthetic_body_passed','model':name,'optimizer_steps':0}),flush=True)
    shim=verified_module(root/'scripts/e2_arima_prediction_compat.py','e2_registered_preflight_arima_compat')
    shim.install_prediction_raw_alias(clean)
    import e2_temporal_controls as controls
    probe=controls.Prediction('2000-01-01',-1.,0.,True)
    if probe.raw_value!=probe.raw:raise ValueError('Historical ARIMA field alias differs')
    receipt={'version':'e2-posthoc-synthetic-acceptance-v1','status':'passed',
             'clean_runtime_sha256':hashlib.sha256(runtime_raw).hexdigest(),'old_controller_sha256':OLD_SHA,
             'bootstrap_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'model_count':len(accepted),'models':accepted,'market_data_read':False,
             'training_performed':False,'optimizer_steps':0,'original_ARIMA_name_alias_validated':True,
             'scope':'One zero-input D7 CUDA forward per fixed body; not acceptance of every possible model kernel'}
    old.immutable_json(runtime_path.parent/'synthetic_acceptance.json',receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-controller',type=Path,required=True)
    p.add_argument('--old-controller-sha256',required=True)
    p.add_argument('--worker-spec',type=Path)
    p.add_argument('--worker-output',type=Path)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--clean-runtime',type=Path)
    p.add_argument('--original-output',type=Path)
    a=p.parse_args()
    if a.preflight:
        if a.clean_runtime is None or a.original_output is None:p.error('Preflight needs runtime and output metadata')
        preflight(a.old_controller,a.old_controller_sha256,a.clean_runtime,a.original_output)
    else:
        if a.worker_spec is None or a.worker_output is None:p.error('Worker needs its original specification/output')
        execute(a.old_controller,a.old_controller_sha256,a.worker_spec,a.worker_output)
