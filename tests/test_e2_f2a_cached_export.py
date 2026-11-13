"""Synthetic read-only F2a export checks; no network, training, or market reads."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical(value) + b'\n'
    path.write_bytes(raw)
    return sha(raw)


def synthetic_campaign(root):
    """All81 identities with tiny, explicitly synthetic derived arrays."""
    import math
    from datetime import date, timedelta
    from e2_reduced_protocol import NEURAL, CLASSICAL, SYSTEMS
    output = root / 'campaign'
    final = output / 'final'
    score = final / 'score-worker-source-recovery-v1'
    science = score / 'scientific_outputs'
    dates = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(608)]
    calibration_dates = [(date(2024, 11, 8) + timedelta(days=i)).isoformat() for i in range(53)]
    truth = [.01 + .0001 * (i % 11) for i in range(608)]
    calibration_truth = [.01 + .0001 * (i % 7) for i in range(53)]
    calibration_hash = sha(canonical(calibration_truth))
    mask = [True] * 608
    rows, frozen, metric_rows = [], {}, []
    artifacts = {}
    for model in SYSTEMS:
        per_seed = []
        seeds = range(271801, 271807) if model in NEURAL else ['deterministic']
        slug = hashlib.sha256(model.encode()).hexdigest()[:16]
        for index, seed in enumerate(seeds):
            prefix = f'final-fit/{slug}/{seed}' if model in NEURAL else f'classical-final/{slug}'
            checkpoint = prefix + ('/fit/selected_model.pt' if model in NEURAL else '/predictor.json')
            checkpoint_sha = sha(f'synthetic checkpoint identity: {model}/{seed}'.encode())
            prediction = [v + .00001 * (index + 1) for v in truth]
            cp = [v + .00002 * (index + 1) for v in calibration_truth]
            per_seed.append(math.sqrt(sum((p-y)**2 for p, y in zip(prediction, truth)) / 608))
            calibrator = {'version': 'synthetic-calibrator', 'dates': calibration_dates,
                'n': 53, 'q': abs(cp[0] - calibration_truth[0]), 'rank': 49, 'predictor_sha256': checkpoint_sha,
                'calibration_predictions_sha256': sha(canonical(cp)),
                'calibration_targets_sha256': calibration_hash}
            calibrator['sha256'] = sha(canonical(calibrator))
            calibrator_path = prefix + '/calibrator.json'
            calibrator_sha = write_json(final / calibrator_path, calibrator)
            frozen.update({checkpoint: checkpoint_sha, calibrator_path: calibrator_sha})
            calibration = {'model': model, 'seed': seed, 'dates': calibration_dates,
                'prediction_original': cp, 'target_original': calibration_truth,
                'checkpoint_sha256': checkpoint_sha, 'same_predictor_as_final': True,
                'used_for_HPO': False}
            calibration_path = prefix + ('/calibration_predictions.json' if model in NEURAL else '/predictions.json')
            calibration_sha = write_json(final / calibration_path, calibration)
            if model in CLASSICAL:
                write_json(final / prefix / 'result.json', {'model': model, 'status': 'completed',
                    'checkpoint_sha256': checkpoint_sha,
                    'artifacts_sha256': {'predictions.json': calibration_sha}})
            pred_path = f'predictions_{slug}_{seed}.json'
            prediction_record = {'model': model, 'seed': seed, 'dates': dates,
                'prediction_original': prediction, 'checkpoint_sha256': checkpoint_sha,
                'mask_sha256': sha(canonical(mask))}
            artifacts['scientific_outputs/' + pred_path] = write_json(science / pred_path, prediction_record)
            rows.append({'model': model, 'seed': seed, 'checkpoint': checkpoint,
                'checkpoint_sha256': checkpoint_sha, 'calibrator': calibrator_path,
                'calibrator_sha256': calibrator_sha})
        metric_rows.append({'model': model, 'RMSE_per_seed': per_seed,
            'RMSE': sum(per_seed) / len(per_seed), 'R': len(per_seed), 'n_common_dates': 608})
    freeze = {'models': rows, 'systems': list(SYSTEMS), 'final_seed_ids': list(range(271801, 271807)),
        'frozen_files_sha256': frozen, 'pilot_power_MDE_closed': True, 'analysis_frozen': True}
    freeze_sha = write_json(final / 'final_freeze.json', freeze)
    cache = final / 'private-known-E1-data' / 'known_E1_daily.csv'
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'SYNTHETIC PRIVATE INPUT: must never be backed up\n')
    cache_sha = sha(cache.read_bytes())
    support_sha = 'b' * 64
    admission = {'daily_market_sha256': cache_sha, 'freeze_sha256': sha(canonical(freeze)),
        'calendar_days': 608, 'common_eligible_count': 608, 'common_eligible_dates': dates,
        'support_sha256': support_sha, 'development_bridge': {'source_bytes_unchanged': True}}
    admission_sha = write_json(science / 'data_admission.json', admission)
    write_json(cache.parent / 'known_E1_data_receipt.json', admission)
    artifacts['scientific_outputs/data_admission.json'] = admission_sha
    analysis = {'table_A2': metric_rows, 'n_common_dates': 608, 'common_mask_sha256': sha(canonical(mask))}
    artifacts['scientific_outputs/benchmark_analysis.json'] = write_json(science / 'benchmark_analysis.json', analysis)
    result = {'status': 'completed', 'artifacts_sha256': artifacts, 'freeze_sha256': sha(canonical(freeze)),
        'actual_new_E2_final_metrics': True, 'systems': 16, 'retuning_performed': False}
    result_sha = write_json(score / 'result.json', result)
    return {'output': output, 'final': final, 'score': score, 'science': science,
        'dates': dates, 'truth': truth, 'mask': mask, 'support': support_sha,
        'pins': {'freeze': freeze_sha, 'result': result_sha, 'admission': admission_sha, 'cache': cache_sha},
        'calibration_hash': calibration_hash}


class CachedExportTests(unittest.TestCase):
    def setUp(self):
        self.module = load('scripts/export_e2_f2a_cached_data.py', 'test_cached_export_module')
        temporary = tempfile.TemporaryDirectory(prefix='e2-f2a-synthetic-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_hash_gate_rejects_changed_cache_without_writing(self):
        cache = self.root / 'synthetic_cache.csv'
        cache.write_bytes(b'synthetic cached bytes')
        checksum = sha(cache.read_bytes())
        self.assertEqual(self.module.read_pinned(cache, checksum), cache.read_bytes())
        cache.write_bytes(b'changed cached bytes')
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with self.assertRaises(ValueError):
            self.module.read_pinned(cache, checksum)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_existing_source_rejects_wrong_loader_identity(self):
        source = self.root / 'original-E'
        scripts = source / 'scripts'
        scripts.mkdir(parents=True)
        (scripts / 'e2_reduced_final_data.py').write_bytes(b'changed-loader-source')
        (scripts / 'e2_reduced_github_backup.py').write_bytes(b'changed-backup-source')
        from types import SimpleNamespace
        with patch.object(self.module.subprocess, 'check_output', return_value=self.module.ENGINE_COMMIT), \
                patch.object(self.module.subprocess, 'run', return_value=SimpleNamespace(returncode=0)):
            with self.assertRaises(ValueError):
                self.module.validate_source(source)
        self.assertEqual(sorted(p.name for p in scripts.iterdir()),
                         ['e2_reduced_final_data.py', 'e2_reduced_github_backup.py'])

    def campaign(self):
        fixture = synthetic_campaign(self.root)
        self.fixture = fixture
        for item in [patch.dict(self.module.PIN_SHA, fixture['pins']),
                     patch.object(self.module, 'CALIBRATION_TARGET_SHA256', fixture['calibration_hash']),
                     patch.object(self.module, 'validate_source'),
                     patch.object(self.module, 'derive_targets', return_value=(
                         fixture['dates'], fixture['truth'], fixture['mask'],
                         {'source_bytes_unchanged': True}, fixture['support']))]:
            item.start()
            self.addCleanup(item.stop)
        return fixture

    def test_all81_existing_arrays_validate_before_any_write(self):
        fixture = self.campaign()
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        bundle = self.module.build_bundle(self.root / 'original-E', fixture['output'])
        self.assertEqual(bundle['dates'], fixture['dates'])
        self.assertEqual(bundle['y_true'], fixture['truth'])
        self.assertEqual(len(bundle['predictors']), 81)
        self.assertTrue(bundle['provenance']['all_81_original_seed_RMSE_verified'])
        self.assertFalse(bundle['provenance']['model_inference'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_rmse_mismatch_blocks_bundle_before_destination_or_backup(self):
        fixture = self.campaign()
        path = fixture['science'] / 'benchmark_analysis.json'
        analysis = json.loads(path.read_bytes())
        analysis['table_A2'][0]['RMSE_per_seed'][0] += .01
        checksum = write_json(path, analysis)
        result_path = fixture['score'] / 'result.json'
        result = json.loads(result_path.read_bytes())
        result['artifacts_sha256']['scientific_outputs/benchmark_analysis.json'] = checksum
        self.module.PIN_SHA['result'] = write_json(result_path, result)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(self.module, 'publish_export') as publish:
            with self.assertRaisesRegex(ValueError, 'RMSE'):
                self.module.build_bundle(self.root / 'original-E', fixture['output'])
            publish.assert_not_called()
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_changed_calibration_cannot_be_exported(self):
        fixture = self.campaign()
        path = next(fixture['final'].glob('final-fit/*/*/calibration_predictions.json'))
        record = json.loads(path.read_bytes())
        record['prediction_original'][0] += 1.
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, 'calibration'):
            self.module.build_bundle(self.root / 'original-E', fixture['output'])
        self.assertFalse((self.root / 'export').exists())

    def test_changed_cache_receipt_blocks_target_derivation(self):
        fixture = self.campaign()
        receipt = fixture['final'] / 'private-known-E1-data/known_E1_data_receipt.json'
        write_json(receipt, {'daily_market_sha256': 'different'})
        with patch.object(self.module, 'derive_targets') as derive:
            with self.assertRaisesRegex(ValueError, 'receipt'):
                self.module.build_bundle(self.root / 'original-E', fixture['output'])
            derive.assert_not_called()

    def test_actual_backup_publishes_only_new_run_metadata_and_preserves_control(self):
        from test_e2_amended_reduced_github_backup import FakeGitHub
        import e2_reduced_github_backup as backup
        fixture = self.campaign()
        bundle = self.module.build_bundle(self.root / 'original-E', fixture['output'])
        client = FakeGitHub()
        client.branch_exists = True
        initial_tree = client.trees[client.commits[client.head]['tree']['sha']]
        pointer = b'{"existing_control_pointer":"unchanged"}\n'
        pointer_sha = backup.git_blob_sha(pointer)
        client.blobs[pointer_sha] = pointer
        initial_tree['control/latest.json'] = pointer_sha
        source_before = {p: p.read_bytes() for p in fixture['output'].rglob('*') if p.is_file()}
        destination = self.root / 'f2a-readonly-export-v1'
        report = self.module.publish_export(bundle, destination, client)
        current = client.trees[client.commits[client.head]['tree']['sha']]
        self.assertEqual(report['status'], 'backed_up')
        self.assertEqual(current['control/latest.json'], pointer_sha)
        added = set(current) - {'control/latest.json'}
        prefix = 'runs/' + report['attempt_id'] + '/'
        self.assertTrue(added)
        self.assertTrue(all(path.startswith(prefix) for path in added))
        self.assertFalse(client.remote_assets)
        receipt = json.loads(client.blobs[current[prefix + 'receipt.json']])
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['kind'], 'posthoc_input_export')
        self.assertFalse(receipt['scientific_fit'])
        self.assertFalse(receipt['benchmark_repeated'])
        artifact_paths = [p for p in added if '/artifacts/' in p]
        self.assertEqual(artifact_paths, [prefix + 'artifacts/e2_f2a_input.json'])
        uploaded = json.loads(client.blobs[current[artifact_paths[0]]])
        self.assertEqual(len(uploaded['predictors']), 81)
        self.assertFalse(uploaded['provenance']['raw_market_data_included'])
        private_marker = b'SYNTHETIC PRIVATE INPUT'
        self.assertTrue(all(private_marker not in client.blobs[current[p]] for p in added))
        self.assertEqual(source_before, {p: p.read_bytes() for p in fixture['output'].rglob('*') if p.is_file()})
        second = self.module.publish_export(bundle, destination, client)
        self.assertTrue(second['existing_identical_result'])
        changed = dict(bundle, unexpected_changed_export=True)
        before_calls = len(client.calls)
        with self.assertRaisesRegex(ValueError, 'existing export differs'):
            self.module.publish_export(changed, destination, client)
        self.assertEqual(len(client.calls), before_calls)

    def test_original_constructor_derives608_sigma7_targets_from_toy_cache(self):
        import numpy as np
        import pandas as pd
        from types import SimpleNamespace
        import e2_reduced_final_data as original
        calendar = pd.date_range('2023-01-01', '2026-08-31', tz='UTC')
        index = np.arange(len(calendar), dtype=float)
        closes = np.exp(.001 * index + .01 * np.sin(index / 13.))
        frame = pd.DataFrame({'open': closes, 'high': closes * 1.01,
            'low': closes * .99, 'close': closes, 'volume': np.ones(len(calendar))}, index=calendar)
        development_frame = frame.loc[:'2024-12-31']
        development = original._panel_from_frames(development_frame)
        receipt = {'synthetic_development': True}
        loader = SimpleNamespace(FEATURES=original.FEATURES,
            load_development_bridge=Mock(return_value=(development, receipt)),
            _panel_from_frames=original._panel_from_frames)
        cache = self.root / 'synthetic_daily.csv'
        evaluation = frame.loc['2025-01-01':].copy()
        evaluation.index.name = 'market_day_utc'
        evaluation.to_csv(cache)
        before = cache.read_bytes()
        with patch.object(self.module, '_original_module', return_value=loader):
            dates, targets, mask, admitted, support = self.module.derive_targets(
                ROOT, self.root / 'development', cache)
        self.assertEqual(dates, self.module.FINAL_DATES)
        self.assertEqual(len(targets), 608)
        self.assertTrue(all(mask))
        self.assertEqual(admitted, receipt)
        self.assertEqual(len(support), 64)
        log_prices = np.log(closes)
        expected = [float(np.std(np.diff(log_prices[i-7:i+1]), ddof=0))
                    for i in range(len(development_frame), len(frame))]
        np.testing.assert_allclose(targets, expected, rtol=1e-9, atol=1e-13)
        self.assertEqual(cache.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['synthetic_daily.csv'])


class CachedExportLauncherTests(unittest.TestCase):
    def setUp(self):
        self.launcher = load('notebooks/export_e2_f2a_cached_colab.py', 'test_cached_export_launcher')
        temporary = tempfile.TemporaryDirectory(prefix='e2-f2a-launcher-synthetic-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        changed = patch.object(self.launcher, 'LOCAL_WORKSPACE', self.root)
        changed.start()
        self.addCleanup(changed.stop)

    def test_dry_run_never_materializes_queries_secret_or_starts_runtime(self):
        with patch.object(self.launcher, '_materialize') as materialize, \
                patch.object(self.launcher.subprocess, 'run') as run:
            result = self.launcher.run_e2_f2a_cached_export_colab('a' * 40)
        materialize.assert_not_called()
        run.assert_not_called()
        self.assertEqual(result['status'], 'dry_run')
        self.assertFalse(result['runtime_installation'])
        self.assertFalse(result['restore_requested'])
        self.assertFalse(result['scientific_fit'])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_launcher_passes_token_only_stdin_using_existing_python(self):
        from types import SimpleNamespace
        python = self.root / 'existing-python'
        python.write_bytes(b'synthetic executable identity only')
        source = self.root / ('reduced-source-' + self.launcher.ENGINE_COMMIT)
        source.mkdir()
        (self.root / 'github-primary-output-v1/final').mkdir(parents=True)
        export_source = self.root / 'new-export-source'
        token = 'synthetic-token-in-memory-only'
        completed = SimpleNamespace(returncode=0, stdout=json.dumps({'status': 'backed_up',
            'attempt_id': 'f2a-fixture-posthoc', 'remote_commit': 'b' * 40,
            'manifest_path': 'runs/f2a-fixture-posthoc/manifest.json'}), stderr='')
        with patch.object(self.launcher, '_materialize', return_value=export_source) as materialize, \
                patch.object(self.launcher.subprocess, 'run', return_value=completed) as run, \
                patch.dict(self.launcher.os.environ, {'GITHUB_TOKEN': token, 'GH_TOKEN': token,
                                                      'PYTHONPATH': 'must-be-removed'}):
            report = self.launcher.run_e2_f2a_cached_export_colab('a' * 40,
                namespace={'GPU_PYTHON': python, 'GITHUB_TOKEN': token}, execute=True)
        materialize.assert_called_once_with('a' * 40)
        command = run.call_args.args[0]
        options = run.call_args.kwargs
        self.assertEqual(command[0], str(python))
        self.assertEqual(command[1], '-B')
        self.assertEqual(command[command.index('--source-root') + 1], str(source))
        self.assertEqual(command[command.index('--destination') + 1], str(self.root / 'f2a-readonly-export-v1'))
        self.assertNotIn(token, ' '.join(command))
        self.assertEqual(options['input'], token + '\n')
        self.assertNotIn('GITHUB_TOKEN', options['env'])
        self.assertNotIn('GH_TOKEN', options['env'])
        self.assertNotIn('PYTHONPATH', options['env'])
        self.assertTrue(options['capture_output'])
        self.assertFalse(report['scientific_fit'])
        self.assertFalse(report['benchmark_repeated'])

    def test_source_hash_mismatch_leaves_no_export_sources(self):
        files = {name: b'synthetic registered source' for name in self.launcher.FILES}
        registration = {'protocol_id': 'e2-reduced-delivery-v1',
            'engine_commit': self.launcher.ENGINE_COMMIT,
            'files_sha256': {name: sha(raw) for name, raw in files.items()}}
        payloads = {**files, self.launcher.REGISTRATION_PATH: canonical(registration)}
        payloads[self.launcher.FILES[-1]] = b'tampered downloaded source'
        with patch.object(self.launcher, '_download', side_effect=lambda commit, relative: payloads[relative]):
            with self.assertRaisesRegex(RuntimeError, 'registro'):
                self.launcher._materialize('a' * 40)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
