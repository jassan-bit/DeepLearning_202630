"""Read-only audit of development artifacts; no training or raw/test reads."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/dl_benchmark_20261104'
sys.path.insert(0, str(ROOT / 'scripts'))
from dl_data import load_development, split_training, fit_preprocessing, transform_windows, mase_scale
from dl_models import build_model, count_parameters

failures, passed, replay = [], {}, []

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    return json.loads((OUT / name).read_text())

def require(value, description):
    if not value:
        failures.append(description)

def close(actual, expected, description, rtol=1e-10, atol=1e-13):
    require(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True), description)

data = load_development(ROOT)
calendar = data['calendar']
manifest = read('code_manifest_before_search.json')
initial, final = read('protocol_initial.json'), read('protocol_final.json')
selected, execution = read('selected_configuration.json'), read('execution_summary.json')
pilot, audit = read('pilot_summary.json'), read('data_audit.json')
runtime = read('runtime.json')
runs = pd.read_csv(OUT / 'runs.csv')
for source, expected in manifest['training_code_sha256'].items():
    require(sha(ROOT / source) == expected, f'Signed script changed: {source}')
require(sha(ROOT / manifest['protocol_path']) == manifest['protocol_sha256'], 'Original protocol digest')
require(sha(OUT / 'protocol_initial.json') == manifest['protocol_sha256'], 'Copied initial protocol digest')
require(sha(OUT / 'selected_configuration.json') == final['selected_configuration_sha256'], 'Selected configuration digest')
require(sha(OUT / 'pilot_summary.json') == final['pilot_summary_sha256'], 'Pilot summary digest')
require(data['source_hashes'] == audit['source_hashes'], 'Data audit source hashes')
require(execution['test_files_opened'] == final['test_files_opened'] == selected['test_files_opened'] == pilot['test_files_opened'] == manifest['project_test_files_opened'] == audit['raw_test_files_opened'] == 0, 'All recorded test-read counters must be zero')
require(runtime['device'] == 'cpu' and not runtime['cuda_available'], 'CPU-only runtime')
require(runtime['deterministic_algorithms'] and runtime['torch_threads_per_worker'] == 1, 'Registered deterministic single-thread runtime')
require(len(runs) == 201 and runs.run_id.nunique() == 201, '201 unique runs')
expected_counts = {'search':96, 'pilot':15, 'development_benchmark':90}
require(runs.groupby('phase').size().to_dict() == expected_counts, 'Planned phase counts')
require(execution['completed_project_fit_count'] == 2*len(runs) == 402, '402 recorded fits from two stages per run')
run_dirs = {p.name for p in (OUT/'runs').iterdir() if p.is_dir()}
require(run_dirs == set(runs.run_id), 'Run directory list equals runs.csv exactly')

splits = {d:split_training(data,d) for d in [7,14,21,28]}
scalers = {(d,s):fit_preprocessing(data,d,splits[d][key])
           for d in splits for s,key in [('inner','inner_train_indices'),('refit','train_indices')]}
files_checked = prediction_rows = curve_rows = 0
phase_frames = {p:[] for p in expected_counts}
registration = pd.Timestamp(manifest['at_utc'])
selection_closed = pd.Timestamp(selected['selected_at_utc'])
final_frozen = pd.Timestamp(final['frozen_at_utc'])
for row in runs.to_dict('records'):
    rid, d, phase = row['run_id'], int(row['history_days']), row['phase']
    directory = OUT/'runs'/rid
    summary = json.loads((directory/'summary.json').read_text())
    config = json.loads((directory/'configuration.json').read_text())
    preproc = json.loads((directory/'preprocessing.json').read_text())
    split = json.loads((directory/'splits.json').read_text())
    expected_split = splits[d]
    indices = expected_split['selection_indices' if phase=='search' else 'calibration_indices']
    require(summary['code_signature'] == config['code_signature'] == manifest['code_signature'], f'{rid}: code signature')
    require(config['source_hashes'] == data['source_hashes'], f'{rid}: sources')
    require(config['test_files_opened'] == 0, f'{rid}: test counter')
    require(config['training'] == initial['training'], f'{rid}: immutable training settings')
    require(config['inner_initialization_seed'] == row['seed'] and config['refit_initialization_seed'] == row['seed']+10000000, f'{rid}: independent stage seeds')
    completed = pd.Timestamp(summary['completed_at_utc'])
    require(completed > registration, f'{rid}: fit after code registration')
    if phase=='search':
        require(completed <= selection_closed, f'{rid}: search before selection freeze')
    elif phase=='pilot':
        require(selection_closed < completed <= final_frozen, f'{rid}: pilot chronology')
    else:
        require(completed > final_frozen, f'{rid}: benchmark after final freeze')
    expected_names = {'configuration.json','preprocessing.json','splits.json','predictions.csv','history_inner.csv','history_refit.csv'}
    if phase!='search': expected_names.add('model.pt')
    require(set(summary['artifact_sha256']) == expected_names, f'{rid}: artifact names')
    require({p.name for p in directory.iterdir()} == expected_names|{'summary.json'}, f'{rid}: no extra or missing run artifacts')
    for name, expected in summary['artifact_sha256'].items():
        require(sha(directory/name) == expected, f'{rid}: digest {name}')
        files_checked += 1
    for key, expected in summary['metrics'].items():
        if isinstance(expected,(int,float)) and not isinstance(expected,bool):
            close(row[key], expected, f'{rid}: runs.csv field {key}')
        else: require(row[key] == expected, f'{rid}: runs.csv field {key}')
    for name, key in [('train_outcome_indices','train_indices'),('inner_train_outcome_indices','inner_train_indices'),('inner_validation_outcome_indices','inner_validation_indices')]:
        require(np.array_equal(split[name],expected_split[key]), f'{rid}: {name}')
        require((calendar[np.asarray(split[name])].year == 2023).all(), f'{rid}: labels only 2023')
    for key,value in expected_split['split_metadata'].items():
        require(split[key] == value, f'{rid}: split metadata {key}')
    require(np.array_equal(split['evaluation_outcome_indices'],indices), f'{rid}: evaluation indices')
    for stage in ['inner','refit']:
        require(preproc[stage] == scalers[(d,stage)], f'{rid}: recomputed {stage} preprocessing')
        input_indices = np.asarray(preproc[stage]['input_day_indices'])
        require((calendar[input_indices].year == 2023).all(), f'{rid}: preprocessing only 2023 inputs')
        require(len(input_indices) == len(np.unique(input_indices)), f'{rid}: unique input-day scaler fit')
    pred = pd.read_csv(directory/'predictions.csv')
    phase_frames[phase].append(pred)
    prediction_rows += len(pred)
    require(len(pred) == len(indices) and pred.outcome_day.nunique() == len(indices), f'{rid}: prediction completeness')
    require((pred.run_id == rid).all() and (pred.phase == phase).all(), f'{rid}: prediction identity')
    require(pred.outcome_day.tolist() == calendar[indices].strftime('%Y-%m-%d').tolist(), f'{rid}: prediction calendar')
    require(pred.input_first_day.tolist() == calendar[indices-d].strftime('%Y-%m-%d').tolist(), f'{rid}: causal input first day')
    require(pred.input_last_day.tolist() == calendar[indices-1].strftime('%Y-%m-%d').tolist(), f'{rid}: input strictly before outcome')
    require(pd.DatetimeIndex(pd.to_datetime(pred.prediction_origin_utc,utc=True)).equals(calendar[indices]), f'{rid}: origin timestamps')
    require(pd.DatetimeIndex(pd.to_datetime(pred.target_available_at_utc,utc=True)).equals(calendar[indices]+pd.Timedelta(days=1)), f'{rid}: label availability timestamps')
    close(pred.y_true, data['targets'][indices], f'{rid}: actual target alignment')
    close(pred.persistence_pred, data['persistence'][indices], f'{rid}: persistence alignment')
    require(np.isfinite(pred.y_pred).all() and (pred.y_pred > 0).all(), f'{rid}: finite positive predictions')
    err = pred.y_true.to_numpy()-pred.y_pred.to_numpy()
    close(np.sqrt(np.mean(err**2)), row['rmse'], f'{rid}: RMSE')
    close(np.mean(err**2), row['mse'], f'{rid}: MSE')
    close(np.mean(np.abs(err)), row['mae'], f'{rid}: MAE')
    scale = mase_scale(data,expected_split['train_indices'])
    close(scale, row['mase_train_scale'], f'{rid}: training-only MASE scale')
    close(np.mean(np.abs(err))/scale, row['mase'], f'{rid}: MASE')
    inner, refit = [pd.read_csv(directory/n) for n in ['history_inner.csv','history_refit.csv']]
    curve_rows += len(inner)+len(refit)
    require(len(refit) == row['refit_epochs'] == row['best_epoch'] and len(inner) >= row['best_epoch'], f'{rid}: refit epochs equal selected inner epoch')
    require(inner.epoch.tolist()==list(range(1,len(inner)+1)) and refit.epoch.tolist()==list(range(1,len(refit)+1)), f'{rid}: epoch sequences')
    require(np.isfinite(inner[['train_mse_scaled_eval','validation_mse_scaled_eval']]).all().all(), f'{rid}: finite inner curves')
    require(np.isfinite(refit.train_mse_scaled_eval).all() and refit.validation_mse_scaled_eval.isna().all(), f'{rid}: refit never evaluates 2024 labels')

for phase,frames in phase_frames.items():
    aggregate = pd.read_csv(OUT/'predictions'/f'{phase}.csv').sort_values(['run_id','outcome_day']).reset_index(drop=True)
    combined = pd.concat(frames,ignore_index=True).sort_values(['run_id','outcome_day']).reset_index(drop=True)
    try: pd.testing.assert_frame_equal(aggregate,combined,check_exact=False,rtol=1e-12,atol=1e-15)
    except AssertionError: failures.append(f'{phase}: aggregate prediction CSV differs from run files')

# Independently recompute the registered common-history/configuration choice.
search = runs.loc[runs.phase=='search']
group=['model','history_days','config_id','width','learning_rate']
scores=search.groupby(group,as_index=False).agg(rmse=('rmse','mean'),parameters=('parameters','first'),seeds=('seed','nunique'))
require((scores.seeds==2).all() and len(scores)==48, 'Equal search budget: four configs x four histories x three families, two seeds each')
best=scores.sort_values(['rmse','parameters','config_id']).groupby(['model','history_days'],as_index=False).first()
joint=best.groupby('history_days',as_index=False).rmse.mean().sort_values(['rmse','history_days'])
require(int(joint.iloc[0].history_days)==selected['selected_common_history_days']==7, 'Reproduced common D selection')
for family in ['MLP','LSTM','CNN1D']:
    winner=best.loc[(best.model==family)&(best.history_days==7)].iloc[0]
    require(winner.config_id==selected['model_configurations'][family]['config_id'], f'{family}: reproduced selected configuration')
    benchmark=runs.loc[(runs.phase=='development_benchmark')&(runs.model==family)]
    require(sorted(benchmark.seed)==final['benchmark_seeds'], f'{family}: 30 declared benchmark seeds')
    require(benchmark.config_id.nunique()==1 and benchmark.config_id.iloc[0]==winner.config_id, f'{family}: configuration remains frozen')

# Evaluation-only replay of a saved checkpoint per family; never fit/train.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
for family in ['MLP','LSTM','CNN1D']:
    row=runs.loc[(runs.phase=='development_benchmark')&(runs.model==family)&(runs.seed==1001)].iloc[0]
    directory=OUT/'runs'/row.run_id
    checkpoint=torch.load(directory/'model.pt',map_location='cpu',weights_only=True)
    model=build_model(checkpoint['family'],checkpoint['history_days'],checkpoint['width'],checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    model.eval()
    preproc=json.loads((directory/'preprocessing.json').read_text())['refit']
    x,_=transform_windows(data,7,splits[7]['calibration_indices'],preproc)
    with torch.inference_mode():
        actual=model(torch.from_numpy(x)).cpu().numpy().astype(np.float64)*preproc['y_scale']
    expected=pd.read_csv(directory/'predictions.csv').y_pred.to_numpy()
    max_abs=float(np.max(np.abs(actual-expected)))
    close(actual,expected,f'{family}: checkpoint reproduces 184 saved predictions',rtol=1e-12,atol=1e-15)
    require(count_parameters(model)['trainable']==row.parameters, f'{family}: checkpoint parameter count')
    replay.append({'family':family,'seed':1001,'predictions_replayed':len(actual),'max_absolute_difference':max_abs,'trainable_parameters':int(row.parameters)})

passed.update({'runs':len(runs),'phase_counts':expected_counts,'recorded_model_fits':2*len(runs),'per_run_artifact_hashes_checked':files_checked,'prediction_rows_checked':prediction_rows,'training_curve_rows_checked':curve_rows,'checkpoint_replays':replay,'source_hashes':data['source_hashes'],'code_signature':manifest['code_signature'],'recorded_test_files_opened':0,'audit_raw_test_files_opened':0,'audit_model_fits':0})
report={'audited_at_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if not failures else 'FAIL','failures':failures,'checks':passed,'runtime':runtime,'planning_limit':{'requested_power':pilot['power_target'],'max_required_R':pilot['max_required_R'],'actual_R':final['benchmark_R'],'cap_limits_power':pilot['cap_limits_power']},'scope':'2023 static training; 2024 development selection/calibration; no test/raw reads or training during audit'}
Path('/tmp/dl_actual_artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
