"""Resume with a disclosed, data-only D00 calendar parser amendment.

The worker adapter is recorded and preflighted before any new fit. Completed
jobs, model recipes, input bytes, protocol and source checkout are preserved.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

SOURCE = '5a07d6157a9796ca54416dc5dea1e1527e731597'
MASTER = '20261119T031104515520Z'
PROTOCOL = '5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a'
RUN_SHA256 = '55dd1cbde85f565b1aca9d74d44b44b27ab84d2c016677d15223cee5a94f49f3'
GIT_HELPER_SHA256 = 'dfdab57f0816a50c58f3c966b4f22c333e33474365d448ce84dec3f8c7677623'
SCOPE = 'experiments/e3_evaluacion_ampliada_20261119'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Parser amendment refuses symlinks')
    return path


def immutable_file(path, raw):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file() and path.read_bytes() == raw, 'Existing parser amendment file differs: ' + path.name)
        return path
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.partial-parser-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.is_file() and path.read_bytes() == raw, 'Concurrent parser amendment differs')
    finally:
        temporary.unlink(missing_ok=True)
    return path


def argument(arguments, flag):
    require(arguments.count(flag) == 1, 'A single original worker argument is required: ' + flag)
    index = arguments.index(flag)
    require(index + 1 < len(arguments), 'Original worker argument is missing its value')
    return arguments[index + 1]


def amendment(output, worker_raw):
    folder = safe_path(output / 'parser_compatibility')
    worker_sha = hashlib.sha256(worker_raw).hexdigest()
    fixed = {
        'kind': 'e3_disclosed_post_freeze_D00_parser_amendment', 'version': 1,
        'source_commit': SOURCE, 'master_run_id': MASTER, 'protocol_sha256': PROTOCOL,
        'effective_executable_amended': True, 'AI_assistance_declared': True,
        'scope': 'Replace only the in-memory auxiliary BTC calendar history reader; use the existing strict frozen daily parser and the original NumPy sigma7 arithmetic.',
        'reason': "Frozen data.py attempted float('') on the already documented all-missing 2023-03-24 calendar bar.",
        'worker_source_sha256': worker_sha, 'frozen_evaluation_run_sha256': RUN_SHA256,
        'raw_CSV_or_NPZ_changed': False, 'calendar_compressed': False, 'missing_values_imputed': False,
        'cohort_rows_removed': 0, 'model_loss_training_or_budget_changed': False,
        'original_job_request_identity_changed': False, 'protocol_tag_changed': False,
        'R1_accredited': False, 'R3_accredited': False,
    }
    receipt = safe_path(folder / 'amendment_v1.json')
    if receipt.exists():
        saved = json.loads(receipt.read_bytes())
        require(all(saved.get(k) == v for k, v in fixed.items()), 'Existing parser amendment has another executable or scope')
    else:
        completed = []
        for path in (output / 'jobs').rglob('result.json'):
            path = safe_path(path)
            if json.loads(path.read_bytes()).get('status') == 'completed':
                completed.append(path.relative_to(output).as_posix())
        d00 = sum(name.startswith('jobs/D00/') for name in completed)
        require(d00 == 0, 'Existing D00 results predate this declared amendment; preserve them for review')
        saved = {**fixed, 'declared_at_utc': datetime.now(timezone.utc).isoformat(),
                 'completed_jobs_at_first_declaration': len(completed),
                 'D00_completed_jobs_at_first_declaration': d00,
                 'new_real_data_fits_at_declaration': 0}
        immutable_file(receipt, (json.dumps(saved, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode())
    immutable_file(folder / 'worker_source_v1.txt', worker_raw)
    return saved


def install_worker_compatibility(controller, source, base, worker_source_text):
    source, base = Path(source).absolute(), Path(base).absolute()
    frozen_run = safe_path(source / SCOPE / 'evaluation/run.py')
    require(hashlib.sha256(frozen_run.read_bytes()).hexdigest() == RUN_SHA256, 'Frozen evaluation coordinator differs')
    raw = worker_source_text.encode('utf-8') if isinstance(worker_source_text, str) else worker_source_text
    require(isinstance(raw, bytes) and raw, 'An exact worker adapter source is required')
    worker_sha = hashlib.sha256(raw).hexdigest()
    worker = safe_path(base / ('e3-d00-parser-worker-' + worker_sha + '.py'))
    original_stream = controller.stream

    def stream(command, cwd, env, log, on_job=None):
        if len(command) < 2 or Path(command[1]).resolve() != frozen_run.resolve():
            return original_stream(command, cwd, env, log, on_job)
        arguments = [str(value) for value in command[2:]]
        require(argument(arguments, '--source-commit') == SOURCE, 'Worker source identity differs')
        output = safe_path(argument(arguments, '--output'))
        require(output == base / ('e3-extended-audit-' + MASTER), 'Worker selected another campaign')
        protocol = safe_path(argument(arguments, '--protocol'))
        require(hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL, 'Frozen protocol bytes differ')
        require(callable(on_job), 'A durable pre-fit amendment save callback is required')
        immutable_file(worker, raw)
        report = amendment(output, raw)
        print('Corrección de lectura D00 declarada:', json.dumps({
            'worker_sha256': worker_sha,
            'completed_jobs_preserved': report['completed_jobs_at_first_declaration'],
            'new_real_data_fits': 0, 'post_freeze_parser_amendment': True}, ensure_ascii=False), flush=True)
        compatibility = [command[0], worker, '--frozen-run', frozen_run]
        preflight = [*compatibility, '--preflight-only', *arguments]
        print('Verificando D00 contra los CSV y el NPZ originales antes de nuevos ajustes.', flush=True)
        original_stream(preflight, cwd, env, output / 'parser_compatibility/preflight.log', None)
        # This uses the original controller's publisher and records the source
        # and data-only proof BEFORE the first corrected D00 optimization.
        on_job()
        print('Corrección y preflight guardados; se reanuda el trabajador original con sus trabajos completos.', flush=True)
        started = time.time()
        try:
            return original_stream([*compatibility, *arguments], cwd, env, log, on_job)
        except Exception:
            receipt = output / 'evaluation_receipt.json'
            try:
                if receipt.is_file() and receipt.stat().st_mtime >= started:
                    current = json.loads(receipt.read_bytes())
                    if (current.get('source_commit') == SOURCE and current.get('protocol_sha256') == PROTOCOL
                            and current.get('failures')):
                        print('Causa registrada por el trabajador:', json.dumps(current['failures'][:20], ensure_ascii=False), flush=True)
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            raise
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
    tools = {'__name__': 'e3_verified_git_parser_resume'}
    exec(compile(helper, 'e3_verified_git_publisher', 'exec'), tools)
    original_install = tools['install_controller']

    def install(controller, checkout, token, context):
        controller = original_install(controller, checkout, token, context)
        return install_worker_compatibility(controller, checkout, base, WORKER_SOURCE)
    tools['install_controller'] = install
    if namespace.get('_E3_GITHUB_TOKEN'):
        tools['_E3_GITHUB_TOKEN'] = namespace['_E3_GITHUB_TOKEN']
    result = tools['main'](tools, base=base)
    namespace['_E3_EXTENDED_AUDIT_RESULT'] = result
    return result


if __name__ == '__main__':
    main(globals())
