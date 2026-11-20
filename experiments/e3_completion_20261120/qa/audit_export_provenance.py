"""Independent, read-only provenance review; no model imports or forwards."""
from pathlib import Path
import hashlib
import json

ROOT = Path('/workspace/e3_completion_delivery')
QA = ROOT / 'qa'
FROZEN = Path('/workspace/work/e3_ab/metadata/freeze')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    frozen_predictions = ROOT / 'export_standard/inputs_private/combined_predictors_133.json'
    assert sha(frozen_predictions) == 'df5206e531ad9c4e5718fb8f17d86390bb1267bcd1d724c2333cacd6c9903655'
    original = json.loads(frozen_predictions.read_text())
    standard = json.loads((ROOT/'export_standard/payload/input_manifest.json').read_text())
    temporal = json.loads((ROOT/'export_temporal/source_input_manifest.json').read_text())
    assets = {a['model']: (a, 'export_standard') for a in standard['checkpoints']}
    assets.update({a['model']: (a, 'export_temporal') for a in temporal['checkpoints']})
    records = []
    for group in ('export_standard', 'export_temporal'):
        for receipt_path in sorted((ROOT/group/'payload').glob('*/roundtrip.json')):
            folder = receipt_path.parent
            receipt = json.loads(receipt_path.read_text())
            metadata = json.loads((folder/'metadata.json').read_text())
            name = metadata['model']
            asset, expected_group = assets[name]
            assert group == expected_group
            checkpoint = ROOT/group/'inputs_private'/asset['path'] if group == 'export_standard' else ROOT/group/'inputs_private'/(name+'.pt')
            assert sha(checkpoint) == asset['sha256']
            assert checkpoint.stat().st_size == asset['bytes']
            assert metadata['scientific_source']['checkpoint_sha256'] == asset['sha256']
            model = folder/'model.onnx'
            digest = sha(model)
            assert metadata['model_sha256'] == digest
            assert receipt.get('onnx_sha256', receipt.get('model_sha256')) == digest
            assert receipt['onnx_bytes'] == model.stat().st_size
            assert metadata['seed'] == receipt['seed'] == 271801
            assert receipt['evaluation_dates'] == 608 and receipt['calibration_dates'] == 53
            assert receipt['declared_absolute_tolerance_original_sigma7'] == 1e-6
            assert receipt['within_tolerance'] is True
            errors = {k: v for k, v in receipt.items() if k.startswith('maximum_') and 'error' in k}
            assert errors and max(errors.values()) <= 1e-6
            assert receipt['new_training_runs'] == receipt['new_calibration_fits'] == 0
            assert receipt['live_registry_updated'] is False
            assert metadata['scientific_source']['checkpoint_sha256'] == metadata['calibrator_source']['checkpoint_sha256']
            old_scaler_path = FROZEN/Path(asset['path']).parent.parent/'scaler.json'
            old_calibrator_path = old_scaler_path.with_name('calibrator.json')
            scaler = json.loads(old_scaler_path.read_text())
            calibrator = json.loads(old_calibrator_path.read_text())
            assert sha(old_calibrator_path) == metadata['calibrator_source']['file_sha256']
            for key in ('mean','std','target_scale'):
                assert metadata['preprocessing'][key] == scaler[key]
            assert metadata['output_scale'] == scaler['target_scale']
            assert metadata['calibration_q'] == calibrator['q'] == metadata['calibrator_source']['q']
            reference = [r for r in original['predictors'] if r['model'] == name and str(r['seed']) == '271801']
            assert len(reference) == 1
            assert reference[0]['checkpoint_sha256'] == asset['sha256']
            assert reference[0]['calibrator_q'] == calibrator['q']
            records.append({'model': name, 'group': group, 'checkpoint_sha256': asset['sha256'],
                'ONNX_sha256': digest, 'ONNX_bytes': model.stat().st_size,
                'metadata_sha256': sha(folder/'metadata.json'), 'roundtrip_receipt_sha256': sha(receipt_path),
                'evaluation_rows': 608, 'calibration_rows': 53,
                'receipt_reported_maximum_error_sigma7': max(errors.values()),
                'original_scaler_values_verified': True, 'original_calibrator_q_verified': True,
                'live_registry_updated': False})
    assert len(records) == 9 and len({r['model'] for r in records}) == 9
    report = {'status': 'nine_old_checkpoint_exports_provenance_and_receipts_independently_verified',
        'AI_assistance_declared': True,
        'scope': 'QA hashed actual checkpoint and ONNX bytes, compared frozen scaler/q/reference identities and read recorded parity; QA did not rerun models.',
        'new_QA_model_fits': 0, 'new_QA_model_predictions': 0,
        'checkpoint_seed': 271801, 'models_verified': 9,
        'receipt_reported_total_parity_rows': 9*661,
        'maximum_receipt_reported_error_sigma7': max(r['receipt_reported_maximum_error_sigma7'] for r in records),
        'total_ONNX_bytes': sum(r['ONNX_bytes'] for r in records),
        'tolerance_sigma7': 1e-6,
        'export_implies_public_serving_or_R3': False,
        'records': records}
    (QA/'nine_export_provenance_review.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    result = audit()
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))
