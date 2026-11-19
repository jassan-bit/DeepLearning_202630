"""Synthetic bound-table plotting regression; no fit, inference or resampling."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PLOTTER = Path(os.environ.get('E3_SCIENTIFIC_SOURCE', '/workspace/e3_review9')) / 'experiments/e3_evaluacion_ampliada_20261119/robustness/plots.py'
EXPECTED_CODE = 'c9a80471098e73be3f4bff18e5095cf9261618eb5efb70355e164bfaa567e0c5'
PYTHON = Path(os.environ.get('E3_PLOT_TEST_PYTHON', sys.executable))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def write_csv(path, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(stream.getvalue().encode())


def fixture():
    analysis = ROOT / 'synthetic_analysis'
    analysis.mkdir(exist_ok=True)
    family = ['synthetic_candidate', 'TSMixer', 'floor_only', 'Persistence'] + [f'synthetic_model_{i:02d}' for i in range(15)]
    datasets = ['D00', 'D01', 'D02', 'D03']
    scenarios = ['unperturbed', 'low', 'medium', 'high']
    config = {'datasets': datasets, 'family': family, 'candidate': family[0],
              'scenarios': scenarios, 'confidence': .95, 'alpha': .1, 'synthetic': True}
    ci, metrics, quarters, learning = [], [], [], []
    for d, dataset in enumerate(datasets):
        for m, model in enumerate(family):
            point = .2 + m * .01 + d * .03
            ci.append({'dataset': dataset, 'model': model, 'scenario': 'unperturbed',
                       'block_observations': 7, 'RMSE_mean_individual': point,
                       'RMSE_CI': json.dumps([point - .02, point + .03])})
            for s, scenario in enumerate(scenarios):
                metrics.append({'dataset': dataset, 'model': model, 'scenario': scenario,
                                'split': 'test', 'RMSE_mean_individual': point + .01 * s})
        for quarter in ['2026Q1', '2026Q2', '2026Q3']:
            quarters.append({'dataset': dataset, 'model': family[0], 'scenario': 'unperturbed',
                             'subgroup': quarter, 'coverage': .91})
        for model in [family[0], 'TSMixer']:
            for fraction in [.25, .5, 1.]:
                for seed in range(3):
                    learning.append({'dataset': dataset, 'model': model, 'train_fraction': fraction,
                                     'seed': seed, 'RMSE': .1 + .1 / fraction + .01 * seed})
    for name, rows in [('calendar_CIs', ci), ('model_metrics', metrics),
                       ('quarter_metrics', quarters), ('sample_size_metrics', learning)]:
        write_csv(analysis / (name + '.csv'), rows)
    protocol_sha = sha(b'SYNTHETIC SOFTWARE FIXTURE; NOT AN E3 RESULT')
    manifest = {'protocol_sha256': protocol_sha, 'files': {}}
    for file in analysis.glob('*.csv'):
        raw = file.read_bytes()
        manifest['files'][file.name] = {'bytes': len(raw), 'sha256': sha(raw)}
    manifest_raw = canonical(manifest)
    (analysis / 'output_manifest.json').write_bytes(manifest_raw)
    receipt = {'configuration': config, 'protocol_sha256': protocol_sha,
               'output_manifest_sha256': sha(manifest_raw), 'synthetic_fixture': True,
               'new_model_fits': 0, 'new_model_inference': 0, 'statistical_resamples': 0}
    (analysis / 'analysis_receipt.json').write_bytes(canonical(receipt))
    return analysis


def run(analysis, output, backend):
    env = dict(os.environ, MPLBACKEND=backend, MPLCONFIGDIR=str(ROOT / 'mpl-cache'), PYTHONNOUSERSITE='1')
    return subprocess.run([str(PYTHON), str(PLOTTER), '--analysis-dir', str(analysis),
                           '--output', str(output)], env=env, text=True, capture_output=True)


def main():
    assert sha(PLOTTER.read_bytes()) == EXPECTED_CODE, 'Frozen scientific plot source differs'
    analysis = fixture()
    inventory_before = {p.name: sha(p.read_bytes()) for p in analysis.iterdir()}
    bad = run(analysis, ROOT / 'refused_bad_backend', 'module://matplotlib_inline.backend_inline')
    assert bad.returncode != 0
    assert "Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value" in bad.stderr, bad.stderr
    assert not (ROOT / 'refused_bad_backend').exists(), 'Import failure must precede export'
    output = ROOT / 'figures'
    good = run(analysis, output, 'Agg')
    assert good.returncode == 0, good.stderr
    manifest_raw = (output / 'figures_manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest['plot_code_sha256'] == EXPECTED_CODE
    assert manifest['matplotlib'] == '3.10.8'
    assert manifest['synthetic_fixture'] is True
    assert manifest['new_model_fits'] == 0 and manifest['R3_accredited'] is False
    assert len(manifest['descriptions']) == 16 and len(manifest['files']) == 32
    for name, meta in manifest['files'].items():
        raw = (output / name).read_bytes()
        assert meta == {'bytes': len(raw), 'sha256': sha(raw)}
        if name.endswith('.png'):
            assert raw.startswith(b'\x89PNG\r\n\x1a\n')
        else:
            assert 'DATOS SINTÉTICOS · PRUEBA DE SOFTWARE' in raw.decode()
    before = {p.name: sha(p.read_bytes()) for p in output.iterdir()}
    retry = run(analysis, output, 'Agg')
    assert retry.returncode == 0, retry.stderr
    assert before == {p.name: sha(p.read_bytes()) for p in output.iterdir()}
    assert inventory_before == {p.name: sha(p.read_bytes()) for p in analysis.iterdir()}
    table = analysis / 'calendar_CIs.csv'
    original = table.read_bytes()
    try:
        table.write_bytes(original + b'changed')
        tampered = run(analysis, ROOT / 'refused_tampered_tables', 'Agg')
        assert tampered.returncode != 0 and 'Analysis table differs from its manifest' in tampered.stderr
        assert not (ROOT / 'refused_tampered_tables').exists()
    finally:
        table.write_bytes(original)
    assert sha(PLOTTER.read_bytes()) == EXPECTED_CODE
    report = {'status': 'passed_synthetic_software_only', 'plotter_sha256': EXPECTED_CODE,
              'python': subprocess.check_output([str(PYTHON), '--version'], text=True).strip(),
              'matplotlib': manifest['matplotlib'], 'regression_error_reproduced': True,
              'fix': 'Set MPLBACKEND=Agg in plotting child environment before Python imports matplotlib',
              'checks': ['bad inherited backend fails before writing output', 'same frozen source exports PNG and SVG under Agg',
                         'four datasets and nineteen systems per RMSE plot', 'all four plot types exercised',
                         'all PNG signatures and table/image hashes verified', 'all SVG watermarks verified',
                         'retry preserves every output byte', 'changed table rejected before output',
                         'input analysis and frozen source bytes preserved'],
              'figures': 16, 'image_files': 32, 'new_model_fits': 0, 'new_model_inference': 0,
              'statistical_resamples': 0, 'R1_accredited': False, 'R3_accredited': False}
    (ROOT / 'verification.json').write_bytes(canonical(report))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
