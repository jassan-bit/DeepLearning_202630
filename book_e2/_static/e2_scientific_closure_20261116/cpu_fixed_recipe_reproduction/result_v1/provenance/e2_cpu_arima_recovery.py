"""Prepare immutable recovery metadata and80 verified reused predictors; no fits.

The original run remains untouched. This program never starts the CPU driver or
opens a budget reservation. Root must review/register the recovery before run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'book_e2/_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction'
ORIGINAL_PLAN_SHA = '53704931391109bf2179985f5026af0316eacc892b8ebe8cfe62b87cd5aa5c7f'
ORIGINAL_DRIVER_SHA = '0a8d5bfc58bdb8810f80493b0092fbe71f50ddb2299e89829a8dc0b9a1ea5c9a'
FAILED_ID = '6c395251bcaf9062-deterministic'
REMAINING = [FAILED_ID, 'CPU-freeze-raw-score', 'CPU-primary20000']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def tree(root):
    result = []
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'Symlink in preserved run: ' + str(path))
        if path.is_file():
            result.append({'path': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
    return result


def prepare(args):
    require(not args.output.exists() and not args.recovery.exists(), 'Use fresh output and recovery directories')
    plan_path = STUDY / 'plan_v2/analysis_plan.json'
    driver = ROOT / 'scripts/e2_cpu_fixed_recipe_reproduction.py'
    require(sha(plan_path) == ORIGINAL_PLAN_SHA and sha(driver) == ORIGINAL_DRIVER_SHA,
            'Original registered plan or driver changed')
    plan = read(plan_path)
    fixed = read(plan['fixed_recipe_plan']['path'])
    require(sha(plan['fixed_recipe_plan']['path']) == plan['fixed_recipe_plan']['sha256'], 'Fixed recipes changed')
    status, identity = read(args.prior / 'status.json'), read(args.prior / 'run_identity.json')
    require(status['status'] == 'partial_stopped' and identity['plan_sha256'] == ORIGINAL_PLAN_SHA,
            'Expected the stoppedV3 run')
    receipt_paths = sorted((args.prior / 'accounting').glob('*.json'))
    require(len(receipt_paths) == 81, 'Exactly81 stopped-run accounting receipts required')
    receipts = [read(p) for p in receipt_paths]
    require({r['id'] for r in receipts} == {j['id'] for j in fixed['jobs']}, 'Stopped roster differs')
    require(sorted(receipts, key=lambda r:r['id']) == sorted(status['jobs'], key=lambda r:r['id']), 'Stopped status/receipts differ')
    success = [r for r in receipts if r['returncode'] == 0 and r['status'] == 'completed']
    failed = [r for r in receipts if r['id'] == FAILED_ID and r['returncode'] == 1 and r['status'] == 'failed']
    require(len(success) == 80 and len(failed) == 1 and len(receipts) == len(success) + len(failed),
            'Recovery must reuse exactly80 successes and one failedARIMA')
    require(all(r['CPU_study_plan_sha256'] == ORIGINAL_PLAN_SHA for r in receipts), 'Mixed CPU study')
    spec = importlib.util.spec_from_file_location('_recovery_export_validation', ROOT / 'scripts/export_e2_cpu_refit_evidence.py')
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    ledger_raw, _, finishes = validation.closed_ledger(Path(plan['active_ledger_path']))
    for row in receipts:
        entry = finishes.get(row['reservation_id'])
        require(entry and entry['phase'] == validation.PHASE + row['id']
                and entry['returncode'] == row['returncode']
                and entry['elapsed_seconds'] == row['charged_worker_wall_seconds'], 'Actual ledger completion differs')
        validation.verified(args.prior, 'logs/' + row['id'] + '.log', row['log_sha256'])
    require(b"'Prediction' object has no attribute 'raw_value'" in (args.prior / 'logs' / (FAILED_ID + '.log')).read_bytes(),
            'Different failure needs a separate recovery design')
    for row in success:
        job = next(j for j in fixed['jobs'] if j['id'] == row['id'])
        directory = args.prior / job['directory']
        result = read(directory / 'result.json')
        require(result['status'] == 'completed' and result['CPU_study_plan_sha256'] == ORIGINAL_PLAN_SHA
                and result['model'] == job['model'] and result['seed'] == job['seed'], 'Reused predictor identity differs')
        for relative, expected in result['artifacts_sha256'].items():
            validation.verified(directory, relative, expected)
        for key in ['checkpoint', 'scaler', 'calibrator']:
            validation.verified(directory, result[key], result[key + '_sha256'])
        calibration = read(directory / result['calibrator'])
        require(calibration['predictor_sha256'] == result['checkpoint_sha256']
                and calibration['sha256'] == validation.digest({k:v for k,v in calibration.items() if k != 'sha256'}),
                'Reused calibration binding differs')
    old_inventory = tree(args.prior)
    prior_failure_seconds = read(STUDY / 'runtime_recovery_v2/recovery_decision.json')['previous_failed_worker_seconds_charged']
    cap = identity['global_cap_sum_worker_seconds'] - failed[0]['charged_worker_wall_seconds']
    require(abs(cap + prior_failure_seconds + failed[0]['charged_worker_wall_seconds'] - 10000) < 1e-8,
            'Recovery cannot reset the original10000s study cap')
    wrapper = ROOT / 'scripts/e2_cpu_arima_worker.py'
    compat = ROOT / 'scripts/e2_arima_prediction_compat.py'
    new_plan = {'version': 'new-CPU-ARIMA-name-compat-recovery-v1',
        'original_plan': {'path': str(plan_path), 'sha256': ORIGINAL_PLAN_SHA},
        'original_driver': {'path': str(driver), 'sha256': ORIGINAL_DRIVER_SHA},
        'wrapper': {'path': str(wrapper), 'sha256': sha(wrapper)},
        'compatibility_adapter': {'path': str(compat), 'sha256': sha(compat)},
        'preparer': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
        'python': '/workspace/work/e2-opt-venv/bin/python',
        'frozen_source_root': plan['frozen_source_root'], 'prior_output': str(args.prior), 'new_output': str(args.output),
        'prior_run_identity_sha256': sha(args.prior / 'run_identity.json'),
        'prior_status_sha256': sha(args.prior / 'status.json'),
        'reused_job_ids': [r['id'] for r in success], 'remaining_job_ids': REMAINING,
        'reused_predictors': 80, 'reused_neural_predictors': 78, 'reused_naive_predictors': 2,
        'new_neural_fits_permitted': 0, 'remaining_ARIMA_estimation_attempts': 1,
        'remaining_shared_stages': 2, 'numeric_training_or_forecast_method_changed': False,
        'adaptation': 'Audited read-only Prediction.raw_value property returns existing Prediction.raw; numeric value and causal forecast-before-observe unchanged',
        'failed_ARIMA_estimation_may_have_run_before_receipt_error': True,
        'failed_attempt_is_not_a_successful_predictor': True,
        'historical_test_exposure': True, 'original_CUDA_parity': False, 'HPO_repeated': False,
        'new_output_worker_cap_seconds': cap, 'original_study_cap_seconds': 10000,
        'priorV2_failed_worker_seconds': prior_failure_seconds,
        'priorV3_failed_ARIMA_worker_seconds': failed[0]['charged_worker_wall_seconds'],
        'reused_successful_worker_seconds': sum(r['charged_worker_wall_seconds'] for r in success),
        'budget_reset': False, 'parent_wall_seconds_added': False,
        'copied_original_receipts_keep_original_output_paths': True,
        'QA_and_source_commit_required_before_recovery_run': True}
    args.recovery.mkdir(parents=True, exist_ok=False)
    write(args.recovery / 'preserved_v3_inventory.json', {'root': str(args.prior), 'all_files': old_inventory})
    new_plan['preserved_v3_inventory_sha256'] = sha(args.recovery / 'preserved_v3_inventory.json')
    for name in ['status.json', 'run_identity.json']:
        shutil.copyfile(args.prior / name, args.recovery / ('prior_' + name))
    shutil.copyfile(args.prior / 'accounting' / (FAILED_ID + '.json'), args.recovery / 'failed_ARIMA_accounting.json')
    shutil.copyfile(args.prior / 'logs' / (FAILED_ID + '.log'), args.recovery / 'failed_ARIMA.log')
    failed_job = next(j for j in fixed['jobs'] if j['id'] == FAILED_ID)
    shutil.copytree(args.prior / failed_job['directory'], args.recovery / 'failed_ARIMA_partial_artifacts')
    (args.recovery / 'closed_ledger_at_preparation.jsonl').write_bytes(ledger_raw)
    args.output.mkdir(parents=True, exist_ok=False)
    for row in success:
        job = next(j for j in fixed['jobs'] if j['id'] == row['id'])
        shutil.copytree(args.prior / job['directory'], args.output / job['directory'])
        for folder, suffix in [('accounting', '.json'), ('logs', '.log')]:
            destination = args.output / folder / (row['id'] + suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.prior / folder / destination.name, destination)
    copied_inventory = tree(args.output)
    original_map = {r['path']:r for r in old_inventory}
    require(all(original_map[r['path']] == r for r in copied_inventory), 'Reused copies differ from original bytes')
    write(args.recovery / 'reused80_inventory.json', {'prior_output': str(args.prior), 'new_output': str(args.output),
        'files': copied_inventory, 'all_original_bytes_preserved': True})
    new_plan['reused80_inventory_sha256'] = sha(args.recovery / 'reused80_inventory.json')
    write(args.recovery / 'recovery_plan.json', new_plan)
    new_identity = dict(identity, global_cap_sum_worker_seconds=cap,
        recovery_plan_sha256=sha(args.recovery / 'recovery_plan.json'),
        recovered_from_output=str(args.prior), reused_successful_predictors=80,
        fresh_neural_fits_in_this_retry=0, remaining_jobs=REMAINING,
        excluded_failed_worker_seconds=failed[0]['charged_worker_wall_seconds'])
    write(args.output / 'run_identity.json', new_identity)
    write(args.recovery / 'QA_candidate.json', {'status': 'awaiting_independent_root_review_before_execution',
        'recovery_plan_sha256': sha(args.recovery / 'recovery_plan.json'),
        'original_plan_unchanged': True, 'original_driver_unchanged': True,
        '80_results_checkpoints_scalers_calibrators_and_receipts_verified': True,
        'v3_bytes_preserved': tree(args.prior) == old_inventory,
        'no_new_fit_or_forward_run_by_preparer': True, 'recovery_run_executed': False})
    command = [str(Path('/workspace/work/e2-opt-venv/bin/python')), str(driver), '--run', '--resume',
        '--plan', str(plan_path), '--source-root', plan['frozen_source_root'],
        '--author-root', '/workspace/work/e2architecture_sources', '--raw-root',
        str(ROOT / 'work/e2-scientific-market/raw'), '--output', str(args.output), '--python', str(wrapper),
        '--admission', str(STUDY / 'admission_v2/timing_admission.json'), '--qa', str(STUDY / 'qa_v2.json'),
        '--cap-seconds', repr(cap)]
    write(args.recovery / 'proposed_command.json', {'argv': command, 'execute_only_after_root_QA_and_registration': True})
    (args.recovery / 'README.md').write_text(
        '# Recuperación de compatibilidad ARIMA\n\n'
        'V3 se conserva íntegra.80predictores exitosos se copian con resultados,recibos y logs byteidénticos '
        '(78redes+2ingenuos). Sus comandos originales siguen señalandoV3 como procedencia. '
        'El intento ARIMA falló al construir un recibo que pide raw_value; puede haber estimado '
        'coeficientes,pero no produjo predictor completo. Se conserva su coste real.\n\n'
        'V4 permite sólo ARIMA,congelamiento/admisión/score y primaria20000. Un wrapper verifica '
        'el código original y añade la propiedad de sólolectura raw_value→raw ya auditada; no '
        'cambia números,recetas,escalado,reglas causales ni pesos de las78redes. Se conservan '
        'el plan5370 y driver0a8. El cap nuevo descuenta el falloV3 y el falloV2 ya estaba descontado. '
        'No se reinicia84h ni se atribuyen83ajustes nuevos a este retry.\n\n'
        'QA_candidate no autoriza ejecución: root debe revisar,congelar y publicar las fuentes '
        'y el nuevo plan antes de ejecutar proposed_command.json.\n', encoding='utf-8')
    print(json.dumps({'new_output': str(args.output), 'recovery': str(args.recovery), 'recovery_plan_sha256': sha(args.recovery / 'recovery_plan.json'),
        'reused_predictors':80,'new_fits_executed':0,'new_cap_seconds':cap}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, default=Path('/workspace/work/e2-true-target-cpu-reproduction-v3'))
    parser.add_argument('--output', type=Path, default=Path('/workspace/work/e2-true-target-cpu-reproduction-v4'))
    parser.add_argument('--recovery', type=Path, default=STUDY / 'runtime_recovery_v3')
    args = parser.parse_args()
    for name in ['prior','output','recovery']:
        setattr(args,name,getattr(args,name).absolute())
    prepare(args)


if __name__ == '__main__':
    main()
