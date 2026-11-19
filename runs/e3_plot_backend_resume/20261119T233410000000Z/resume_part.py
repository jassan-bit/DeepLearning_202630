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
