# Operational finalization adapter; AI assistance declared.
VERIFIER_SOURCE = '"""Read-only completion gate for this already published E3 campaign.\n\nOnly the original, completed evaluation may bypass its worker.  Statistical\nanalysis remains the frozen analyze.py, whose existing final-receipt reuse path\nchecks every input and table. This verifier never imports model code, fits,\npredicts, writes an artifact, changes a receipt, or accredits R1/R3.\n"""\nfrom __future__ import annotations\n\nimport hashlib\nimport json\nfrom pathlib import Path\n\n\nSOURCE_COMMIT = \'5a07d6157a9796ca54416dc5dea1e1527e731597\'\nPROTOCOL_SHA256 = \'5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a\'\nEVALUATION_CODE_SHA256 = \'55dd1cbde85f565b1aca9d74d44b44b27ab84d2c016677d15223cee5a94f49f3\'\nSCOPE = \'experiments/e3_evaluacion_ampliada_20261119\'\n# Bytes independently read from the saved final evaluation at d8133ce...,\n# part 189. Its successor, part 190, contains the completed statistical analysis.\nFINAL_ARTIFACT_SHA256 = {\n    \'evaluation_receipt.json\': \'67e27e5324b6feb3cb0a03fcbd05e196449b3100ee85efb68f0f8a085bcce110\',\n    \'run_manifest.json\': \'288eadf7430e97e1df76eb96efa9ecdf1f697dba834771f5544ee5be5f99acc1\',\n    \'panel_manifest.json\': \'5f91f7622b5eb160dd5de14cb38edbc872f0fd9963c94afa24450250bec2aeb5\',\n}\nDATASETS = (\'D00\', \'D01\', \'D02\', \'D03\')\nDETERMINISTIC = {\'Persistence\', \'SeasonalNaive\', \'ARIMA\', \'floor_only\', \'floor_plus_mean_excess\'}\nEXPECTED_COUNTS = {\n    \'dataset_count\': 4, \'additional_datasets_evaluated\': 3,\n    \'new_real_data_fit_attempts\': 780, \'new_real_data_fits\': 780,\n    \'neural_fit_jobs_attempted\': 776, \'ARIMA_fit_jobs_attempted\': 4,\n    \'completed_job_results\': 796, \'panel_rows\': 1874560,\n    \'panel_files\': 76, \'sample_curve_rows\': 32544,\n}\n\n\ndef _require(value, message):\n    if not value:\n        raise ValueError(\'completed_evaluation_verification: \' + message)\n\n\ndef _regular(path):\n    path = Path(path).absolute()\n    _require(not any(item.is_symlink() for item in (path, *path.parents)),\n             \'symlink refused: \' + str(path))\n    _require(path.is_file(), \'regular file missing: \' + str(path))\n    return path\n\n\ndef _read(path):\n    return _regular(path).read_bytes()\n\n\ndef _sha(raw):\n    return hashlib.sha256(raw).hexdigest()\n\n\ndef _json(path):\n    return json.loads(_read(path))\n\n\ndef _file_proof(path):\n    path = _regular(path)\n    before = path.stat()\n    digest = hashlib.sha256()\n    size = newlines = 0\n    last = b\'\'\n    with path.open(\'rb\') as stream:\n        for chunk in iter(lambda: stream.read(1024 * 1024), b\'\'):\n            digest.update(chunk)\n            size += len(chunk)\n            newlines += chunk.count(b\'\\n\')\n            last = chunk[-1:]\n    after = path.stat()\n    _require((before.st_size, before.st_mtime_ns) ==\n             (after.st_size, after.st_mtime_ns) and size == before.st_size,\n             \'file changed while hashing: \' + str(path))\n    # The immutable executor CSV writer emits exactly one newline per record,\n    # without embedded newlines. Exact published byte hashes bind this format.\n    _require(last == b\'\\n\', \'published CSV must end with newline: \' + str(path))\n    return {\'sha256\': digest.hexdigest(), \'bytes\': size, \'rows\': newlines - 1}\n\n\ndef verify_completed_evaluation(output, protocol_path, source):\n    """Return proof, or raise; ``source`` is the frozen source checkout path.\n\n    This campaign-specific gate requires the published final receipt/manifest\n    byte identities and validates the restored panel payloads. A missing or\n    changed artifact is an error, never a request to rerun the training worker.\n    """\n    output = Path(output).absolute()\n    source = Path(source).absolute()\n    protocol_raw = _read(protocol_path)\n    protocol = json.loads(protocol_raw)\n    _require(_sha(protocol_raw) == PROTOCOL_SHA256, \'protocol differs from frozen campaign\')\n    _require(protocol.get(\'source_commit\') == SOURCE_COMMIT, \'protocol source differs\')\n    _require(_sha(_read(source / SCOPE / \'evaluation/run.py\')) == EVALUATION_CODE_SHA256,\n             \'evaluation worker source differs\')\n    artifacts = {}\n    for name, expected in FINAL_ARTIFACT_SHA256.items():\n        raw = _read(output / name)\n        _require(_sha(raw) == expected, \'published final artifact differs: \' + name)\n        artifacts[name] = json.loads(raw)\n    receipt = artifacts[\'evaluation_receipt.json\']\n    manifest = artifacts[\'run_manifest.json\']\n    panel = artifacts[\'panel_manifest.json\']\n    _require(receipt.get(\'kind\') == \'e3_extended_evaluation_receipt\' and\n             receipt.get(\'status\') == \'completed\' and receipt.get(\'failures\') == [],\n             \'evaluation did not complete without failures\')\n    for name, expected in EXPECTED_COUNTS.items():\n        _require(type(receipt.get(name)) is int and receipt[name] == expected,\n                 \'final evaluation count differs: \' + name)\n    for data in (receipt, manifest, panel):\n        _require(data.get(\'source_commit\') == SOURCE_COMMIT and\n                 data.get(\'protocol_sha256\') == PROTOCOL_SHA256,\n                 \'final artifact source/protocol binding differs\')\n    for data in (receipt, manifest):\n        _require(data.get(\'R1_accredited\') is False and data.get(\'R3_accredited\') is False,\n                 \'exploratory accreditation flags changed\')\n    _require(manifest.get(\'kind\') == \'e3_extended_audit_run_manifest\' and\n             manifest.get(\'panel_manifest\') == \'panel_manifest.json\' and\n             manifest.get(\'panel_sha256\') == FINAL_ARTIFACT_SHA256[\'panel_manifest.json\'],\n             \'run/panel manifest binding differs\')\n    _require(manifest.get(\'datasets\') == {k: {\'raw_sha256\': v}\n             for k, v in protocol.get(\'datasets\', {}).items()} and\n             set(protocol.get(\'datasets\', {})) == set(DATASETS), \'dataset binding differs\')\n    _require(manifest.get(\'family\') == panel.get(\'family\') == protocol.get(\'family\') and\n             len(protocol[\'family\']) == 19 and len(set(protocol[\'family\'])) == 19,\n             \'complete fixed model family differs\')\n    _require(manifest.get(\'seeds\') == panel.get(\'seeds\') == protocol.get(\'seeds\') and\n             protocol[\'seeds\'] == list(range(20261118, 20261128)), \'final seed roster differs\')\n    _require(manifest.get(\'test_targets_used_for_selection\') is False and\n             manifest.get(\'additional_datasets_used_for_mechanism_redesign\') is False and\n             manifest.get(\'prospective_power_certified\') is False and\n             manifest.get(\'evidence_mode\') == protocol.get(\'evidence_mode\') ==\n             \'prespecified_exploratory_audit\', \'exploratory evaluation policy differs\')\n    freeze_raw = _read(output / \'freeze_receipt.json\')\n    freeze = json.loads(freeze_raw)\n    _require(_sha(freeze_raw) == manifest.get(\'freeze_receipt_sha256\') and\n             freeze.get(\'verified\') is True and freeze.get(\'source_commit\') == SOURCE_COMMIT and\n             freeze.get(\'protocol_sha256\') == PROTOCOL_SHA256 and\n             freeze.get(\'annotated_tag\') == manifest.get(\'protocol_tag\'),\n             \'freeze receipt binding differs\')\n    _require(panel.get(\'kind\') == \'e3_extended_panel_manifest\' and\n             type(panel.get(\'files\')) is list and len(panel[\'files\']) == 76 and\n             panel.get(\'total_rows\') == receipt[\'panel_rows\'], \'panel completion differs\')\n    expected_pairs = {(dataset, model) for dataset in DATASETS for model in protocol[\'family\']}\n    actual_pairs = set()\n    paths = set()\n    total_rows = total_bytes = 0\n    for entry in panel[\'files\']:\n        name = entry.get(\'path\')\n        _require(type(name) is str and name, \'missing shard path\')\n        relative = Path(name)\n        _require(not relative.is_absolute() and \'..\' not in relative.parts and\n                 \'\\\\\' not in name and relative.as_posix() == name and\n                 relative.suffix == \'.csv\' and name not in paths, \'unsafe/duplicate shard path\')\n        paths.add(name)\n        pair = (entry.get(\'dataset\'), entry.get(\'model\'))\n        _require(pair in expected_pairs and pair not in actual_pairs, \'unknown/duplicate dataset-model panel\')\n        actual_pairs.add(pair)\n        expected_seeds = [-1] if pair[1] in DETERMINISTIC else protocol[\'seeds\']\n        _require(entry.get(\'observed_seeds\') == expected_seeds, \'panel seed roster differs\')\n        target = output / relative\n        _require(target.resolve().is_relative_to(output.resolve()), \'shard escaped output\')\n        proof = _file_proof(target)\n        _require(all(type(entry.get(field)) is int and entry[field] > 0\n                     for field in (\'bytes\', \'rows\')), \'invalid shard sizes\')\n        _require(all(proof[field] == entry.get(field) for field in (\'sha256\', \'bytes\', \'rows\')),\n                 \'panel payload differs from final manifest: \' + name)\n        total_rows += proof[\'rows\']\n        total_bytes += proof[\'bytes\']\n    _require(actual_pairs == expected_pairs and total_rows == receipt[\'panel_rows\'],\n             \'incomplete final panel roster/rows\')\n    _require(_json(output / \'failures.json\') == [], \'failure list is nonempty\')\n    # Only saved forecasts are needed after the worker completes; no checkpoint\n    # is deserialized or used for another forward pass by this completion gate.\n    return {\'status\': \'published_completed_evaluation_verified\',\n            \'source_commit\': SOURCE_COMMIT, \'protocol_sha256\': PROTOCOL_SHA256,\n            \'evaluation_receipt_sha256\': FINAL_ARTIFACT_SHA256[\'evaluation_receipt.json\'],\n            \'run_manifest_sha256\': FINAL_ARTIFACT_SHA256[\'run_manifest.json\'],\n            \'panel_manifest_sha256\': FINAL_ARTIFACT_SHA256[\'panel_manifest.json\'],\n            \'completed_job_results\': receipt[\'completed_job_results\'],\n            \'verified_panel_files\': len(paths), \'verified_panel_rows\': total_rows,\n            \'verified_panel_bytes\': total_bytes, \'new_model_fits\': 0,\n            \'new_model_inference\': 0, \'R1_accredited\': False, \'R3_accredited\': False}\n'

"""Finish the completed E3 audit with a noninteractive plotting backend.

AI assistance declared. This adapter does not execute the evaluation worker.
It requires the exact saved completion artifacts, delegates cached-analysis
validation and plotting to the unchanged scientific source, and retains the
existing Git PACK publisher and original controller finalization.
"""
import hashlib
import json
from pathlib import Path
import os
import tempfile
import urllib.request

SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
MASTER = '20261119T031104515520Z'
PROTOCOL = '5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a'
SCOPE = 'experiments/e3_evaluacion_ampliada_20261119'
GIT_HELPER_SHA256 = 'dfdab57f0816a50c58f3c966b4f22c333e33474365d448ce84dec3f8c7677623'
ANALYSIS_RECEIPT_SHA256 = '0048bdf3e8376ed5cefa57e57b31eabbc2e5785bd5acf7235a544fa5f7265efc'
ANALYSIS_MANIFEST_SHA256 = 'f2e2d2ea5cc9c5e117aad583c95ea7cd0ac22dfd3f10f4b0a30cacc4746d396f'
ANALYZE_SHA256 = '9b2d38a89480756901aad7fd97d09a728653842014f9059a01b0e355f67482bb'
PLOTS_SHA256 = 'c9a80471098e73be3f4bff18e5095cf9261618eb5efb70355e164bfaa567e0c5'
RUN_SHA256 = '55dd1cbde85f565b1aca9d74d44b44b27ab84d2c016677d15223cee5a94f49f3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Evidence symlink refused')
    require(path.is_file(), 'Required saved file is missing: ' + path.name)
    return path


def sha(path):
    return hashlib.sha256(regular(path).read_bytes()).hexdigest()


def argument(command, flag):
    values = [str(value) for value in command[2:]]
    require(values.count(flag) == 1, 'A single original argument is required: ' + flag)
    index = values.index(flag)
    require(index + 1 < len(values), 'Missing original argument value: ' + flag)
    return values[index + 1]


def immutable_json(path, value):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Recovery receipt symlink refused')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if path.exists():
        require(regular(path).read_bytes() == raw, 'Existing recovery declaration differs')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-plot-backend-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(regular(path).read_bytes() == raw, 'Concurrent recovery declaration differs')
    finally:
        temporary.unlink(missing_ok=True)


def verify_saved_analysis(output):
    directory = output / 'analysis'
    require(sha(directory / 'analysis_receipt.json') == ANALYSIS_RECEIPT_SHA256,
            'The final saved analysis receipt differs; preserve this run for review')
    require(sha(directory / 'output_manifest.json') == ANALYSIS_MANIFEST_SHA256,
            'The final saved analysis manifest differs; no resampling will be attempted')
    receipt = json.loads((directory / 'analysis_receipt.json').read_bytes())
    manifest = json.loads((directory / 'output_manifest.json').read_bytes())
    require(receipt['source_commit'] == SOURCE and receipt['protocol_sha256'] == PROTOCOL
            and receipt['output_manifest_sha256'] == ANALYSIS_MANIFEST_SHA256
            and receipt['analysis_code_sha256'] == ANALYZE_SHA256
            and receipt['new_model_fits'] == 0 and receipt['new_model_inference'] == 0
            and receipt['R3_accredited'] is False, 'Saved analysis identity differs')
    require(manifest['protocol_sha256'] == PROTOCOL, 'Analysis manifest protocol differs')
    for name, record in manifest['files'].items():
        require(Path(name).name == name and name not in {'.', '..'}, 'Unsafe analysis table path')
        path = regular(directory / name)
        require(path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                'Saved analysis table differs: ' + name)
    return {'receipt_sha256': ANALYSIS_RECEIPT_SHA256, 'manifest_sha256': ANALYSIS_MANIFEST_SHA256,
            'tables_verified': len(manifest['files']), 'bootstrap_recomputed': False}


def verify_figures(output):
    directory = output / 'figures'
    manifest = json.loads(regular(directory / 'figures_manifest.json').read_bytes())
    expected = {kind + '_' + dataset for kind in ('rmse', 'noise', 'coverage', 'learning')
                for dataset in ('D00', 'D01', 'D02', 'D03')}
    require(manifest['inputs'] == {'analysis_receipt_sha256': ANALYSIS_RECEIPT_SHA256,
                                 'analysis_manifest_sha256': ANALYSIS_MANIFEST_SHA256}
            and manifest['plot_code_sha256'] == PLOTS_SHA256
            and manifest['synthetic_fixture'] is False and manifest['new_model_fits'] == 0
            and manifest['R3_accredited'] is False, 'Figures bind another analysis or plotting source')
    require(set(manifest['descriptions']) == expected, 'The complete sixteen-figure set is required')
    require(set(manifest['files']) == {name + ext for name in expected for ext in ('.png', '.svg')},
            'The complete PNG and SVG output set is required')
    for name, record in manifest['files'].items():
        path = regular(directory / name)
        require(path.stat().st_size == record['bytes'] and record['bytes'] > 0
                and sha(path) == record['sha256'], 'Figure differs from its manifest: ' + name)
    return {'figures': len(expected), 'files': len(manifest['files'])}


def install_plot_finalization(controller, checkout, base, verifier):
    checkout, base = Path(checkout).absolute(), Path(base).absolute()
    output = base / ('e3-extended-audit-' + MASTER)
    worker = regular(checkout / SCOPE / 'evaluation/run.py')
    analyze = regular(checkout / SCOPE / 'robustness/analyze.py')
    plots = regular(checkout / SCOPE / 'robustness/plots.py')
    require(sha(worker) == RUN_SHA256 and sha(analyze) == ANALYZE_SHA256 and sha(plots) == PLOTS_SHA256,
            'Frozen scientific executable bytes differ')
    original_stream = controller.stream
    stages = {'worker_verified': False, 'analysis_reused': False}

    # This is a finalization-only adapter. The controller's ordinary live
    # publisher rewrites panel_manifest.json with a running-view annotation.
    # Preserve the completed producer's exact bytes even if a later save is
    # interrupted and restored before final controller completion.
    original_view = controller.stable_view

    def completed_view(current_output, view, *, finished=False):
        require(Path(current_output).absolute() == output
                and Path(view).absolute() == base / ('e3-extended-persist-view-' + MASTER),
                'Publisher selected another campaign or view')
        verifier(output, output / 'protocol.json', checkout)
        verify_saved_analysis(output)
        return original_view(current_output, view, finished=True)

    controller.stable_view = completed_view

    def stream(command, cwd, env, log, on_job=None):
        target = Path(command[1]).absolute() if len(command) > 1 else None
        if target == worker:
            require(Path(argument(command, '--output')).absolute() == output
                    and argument(command, '--source-commit') == SOURCE,
                    'A different evaluation campaign was selected')
            protocol = Path(argument(command, '--protocol')).absolute()
            require(protocol == output / 'protocol.json' and sha(protocol) == PROTOCOL,
                    'Frozen protocol identity differs')
            proof = verifier(output, protocol, checkout)
            analysis = verify_saved_analysis(output)
            declaration = {
                'kind': 'e3_completed_audit_plot_backend_recovery', 'version': 1,
                'AI_assistance_declared': True, 'source_commit': SOURCE,
                'master_run_id': MASTER, 'protocol_sha256': PROTOCOL,
                'reason': 'Colab MPLBACKEND was rejected at matplotlib import before the plotter could select Agg.',
                'selected_child_plot_backend': 'Agg', 'evaluation_completion': proof,
                'saved_analysis': analysis, 'evaluation_worker_executed': False,
                'analysis_action': 'Original cached-analysis reuse with all original input/output checks',
                'new_model_fits': 0, 'new_market_predictions': 0,
                'protocol_changed': False, 'scientific_source_changed': False,
                'R1_accredited': False, 'R3_accredited': False,
            }
            immutable_json(output / 'plot_backend_recovery/recovery_v1.json', declaration)
            require(callable(on_job), 'The original durable save callback is required')
            on_job()
            stages['worker_verified'] = True
            print('Los 796 trabajos completos están verificados: se omite el trabajador; cero entrenamientos y predicciones nuevos.', flush=True)
            return 796
        if target == analyze:
            require(stages['worker_verified'], 'Completed evaluation must be verified before analysis reuse')
            require(Path(argument(command, '--output')).absolute() == output / 'analysis'
                    and Path(argument(command, '--panel')).absolute() == output / 'panel_manifest.json'
                    and Path(argument(command, '--protocol')).absolute() == output / 'protocol.json',
                    'Analysis selected another saved audit')
            verify_saved_analysis(output)
            print('Validando el análisis terminado para reutilizarlo; no se repiten los 9.999 remuestreos.', flush=True)
            result = original_stream(command, cwd, env, log, on_job)
            # The original analyzer returns the same final receipt on reuse.
            # A rewritten analysis receipt, even with similar values, is refused.
            verify_saved_analysis(output)
            stages['analysis_reused'] = True
            return result
        if target == plots:
            require(stages['worker_verified'] and stages['analysis_reused'], 'Verified cached analysis is required before plotting')
            require(Path(argument(command, '--analysis-dir')).absolute() == output / 'analysis'
                    and Path(argument(command, '--output')).absolute() == output / 'figures',
                    'Plotter selected another saved audit')
            plot_env = dict(env)
            plot_env['MPLBACKEND'] = 'Agg'
            print('Generando figuras con MPLBACKEND=Agg en el proceso aislado.', flush=True)
            result = original_stream(command, cwd, plot_env, log, on_job)
            summary = verify_figures(output)
            print('Figuras verificadas:', json.dumps(summary, ensure_ascii=False), flush=True)
            return result
        return original_stream(command, cwd, env, log, on_job)

    controller.stream = stream
    return controller


def main(namespace, *, base=None):
    base = Path('/content') if base is None else Path(base).absolute()
    url = ('https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/'
           'c3d099998c88fde990642783edcfe96f0250fb77/'
           'runs/e3_git_push_resume/20261119T183321564878Z/resume_cell.py')
    with urllib.request.urlopen(url, timeout=60) as response:
        helper = response.read()
    require(hashlib.sha256(helper).hexdigest() == GIT_HELPER_SHA256, 'Verified Git publisher helper differs')
    tools = {'__name__': 'e3_verified_git_plot_recovery'}
    exec(compile(helper, 'e3_verified_git_publisher', 'exec'), tools)
    original_install = tools['install_controller']
    verifier_namespace = {'__name__': 'e3_completed_evaluation_verifier'}
    exec(compile(VERIFIER_SOURCE, 'e3_completed_evaluation_verifier', 'exec'), verifier_namespace)
    verifier = verifier_namespace['verify_completed_evaluation']

    def install(controller, checkout, token, context):
        controller = original_install(controller, checkout, token, context)
        return install_plot_finalization(controller, checkout, base, verifier)
    tools['install_controller'] = install
    if namespace.get('_E3_GITHUB_TOKEN'):
        tools['_E3_GITHUB_TOKEN'] = namespace['_E3_GITHUB_TOKEN']
    print('Cierre de figuras: reutiliza los trabajos y el análisis ya terminados; conserva el mismo master.', flush=True)
    result = tools['main'](tools, base=base)
    namespace['_E3_EXTENDED_AUDIT_RESULT'] = result
    return result


if __name__ == '__main__':
    main(globals())
