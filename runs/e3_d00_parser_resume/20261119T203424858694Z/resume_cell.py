WORKER_SOURCE = '"""Disclosed post-freeze D00 parser compatibility amendment.\n\nThe immutable scientific checkout and input bytes are never edited.  Only the\nin-memory auxiliary BTC-history parser is replaced.  The original CLI,\ncoordinator, model recipes, completed-job cache, cohorts and budget remain in\nforce.  A blank complete calendar bar stays absent; it is never imputed.\n"""\nfrom __future__ import annotations\n\nimport argparse\nfrom datetime import date, timedelta\nimport hashlib\nimport importlib.util\nimport json\nfrom pathlib import Path\nimport sys\n\nSOURCE = "5a07d6157a9796ca54416dc5dea1e1527e731597"\nMASTER = "20261119T031104515520Z"\nPROTOCOL_SHA256 = "5d3e5c0f7440efaa4993e193526d0d53628b7bb053d87a2324975d673379136a"\nTAG_OBJECT = "ba8b5c6fde85cef3deb57b072ab3570a53585b29"\nSCOPE = "experiments/e3_evaluacion_ampliada_20261119"\nFROZEN_FILE_SHA256 = {\n    "evaluation/run.py": "55dd1cbde85f565b1aca9d74d44b44b27ab84d2c016677d15223cee5a94f49f3",\n    "evaluation/data.py": "22615ba4c1b3ffbb6523036b14d98b7459da290955a8ea416fcf6249c0872bbf",\n    "admission/d00_history.py": "a603bc32b71323b060f36985f4adeb248ca3fc191b222dfcecd02cf93fb6709e",\n}\n\n\ndef sha(path):\n    return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n\n\ndef canonical(value):\n    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)\n\n\ndef require(condition, message):\n    if not condition:\n        raise ValueError(message)\n\n\ndef regular(path):\n    path = Path(path).absolute()\n    require(not any(p.is_symlink() for p in (path, *path.parents)),\n            "Parser compatibility refuses symlinked source/input paths")\n    require(path.is_file(), "Parser compatibility requires an existing regular file")\n    return path\n\n\ndef load_frozen(path):\n    """Verify the three parser/coordinator files before importing any of them."""\n    run_path = regular(path)\n    require(run_path.name == "run.py" and run_path.parent.name == "evaluation",\n            "Expected immutable evaluation/run.py")\n    scope = run_path.parent.parent\n    checked = {}\n    for relative, expected in FROZEN_FILE_SHA256.items():\n        p = regular(scope / relative)\n        require(sha(p) == expected, "Frozen source digest differs: " + relative)\n        checked[relative] = expected\n    spec = importlib.util.spec_from_file_location("e3_d00_compat_frozen_run", run_path)\n    original = importlib.util.module_from_spec(spec)\n    sys.modules[spec.name] = original\n    spec.loader.exec_module(original)\n    history = original.load(scope / "admission/d00_history.py", "e3_d00_compat_pinned_history")\n    require(sha(original.data_api.__file__) == FROZEN_FILE_SHA256["evaluation/data.py"],\n            "Imported data module differs from pinned frozen file")\n    original._d00_compat_history = history\n    original._d00_compat_source_checks = checked\n    return original\n\n\ndef install_parser(original):\n    """Change one function in memory; use the original NumPy arithmetic."""\n    history_api = original._d00_compat_history\n    data_api = original.data_api\n    np = data_api.np\n\n    def btc_history(paths, expected_hashes):\n        if len(paths) != 2 or sorted(data_api.sha(p) for p in paths) != sorted(expected_hashes):\n            raise ValueError("Both pinned daily BTC source files are required for classical causal history")\n        # This frozen validator checks hashes, complete ordered calendars,\n        # headers, coherent finite bars, and rejects partially missing bars.\n        # Its records retain the all-missing 2023-03-24 calendar entry.\n        records = history_api.history(paths)\n        closes = {r["date"]: r["values"][3] for r in records if r["values"] is not None}\n        history = {}\n        for day in sorted(closes):\n            end = date.fromisoformat(day)\n            support = [(end - timedelta(days=j)).isoformat() for j in range(7, -1, -1)]\n            if all(d in closes for d in support):\n                history[day] = float(np.std(np.diff(np.log([closes[d] for d in support])), ddof=0))\n        return history\n\n    original.data_api.btc_history = btc_history\n    return btc_history\n\n\ndef parse_forwarded(arguments):\n    """Validate required preflight inputs without consuming original CLI args."""\n    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)\n    for name in ("output", "source-commit", "admission", "protocol", "freeze-receipt", "npz"):\n        parser.add_argument("--" + name, required=True)\n    parser.add_argument("--btc-history", nargs=2, required=True)\n    parser.add_argument("--device", default="cuda:0")\n    parser.add_argument("--package-path")\n    return parser.parse_args(arguments)\n\n\ndef _job_inventory(output):\n    result = {}\n    if (output / "jobs").exists():\n        for p in sorted((output / "jobs").rglob("result.json")):\n            regular(p)\n            result[p.relative_to(output).as_posix()] = sha(p)\n    raw = canonical(result).encode()\n    return {"completed_job_results": len(result), "result_inventory_sha256": hashlib.sha256(raw).hexdigest()}\n\n\ndef preflight(original, args):\n    """Data-only checks. Never construct, fit or predict with a model."""\n    require(args.source_commit == SOURCE, "Compatibility amendment is limited to its frozen source commit")\n    protocol_path = regular(args.protocol)\n    freeze_path = regular(args.freeze_receipt)\n    npz_path = regular(args.npz)\n    regular(args.admission)\n    paths = [regular(p) for p in args.btc_history]\n    require(sha(protocol_path) == PROTOCOL_SHA256, "Frozen protocol SHA256 differs")\n    p, freeze = original.protocol(protocol_path, freeze_path, SOURCE)\n    require(freeze["tag_object_sha"] == TAG_OBJECT and\n            freeze["annotated_tag"] == "e3-extended-audit-protocol-" + MASTER,\n            "Annotated protocol tag differs from this master campaign")\n    output = Path(args.output).absolute()\n    require(not any(x.is_symlink() for x in (output, *output.parents)),\n            "Compatibility output cannot traverse symlinks")\n    require(output.is_dir(), "Existing campaign output required; no new campaign is created")\n    before = _job_inventory(output)\n    input_hashes = {str(path): sha(path) for path in [npz_path, *paths]}\n    history_api = original._d00_compat_history\n    records = history_api.history(paths)\n    parity = history_api.verify_npz(records, npz_path)\n    data = original.data_api.btc(npz_path, p, evaluation=True, history_paths=paths)\n    cohorts = {k: len(v) for k, v in data["positions"].items()}\n    require(cohorts == {"fit": 532, "validation": 116, "calibration": 53, "test": 608},\n            "D00 frozen cohort geometry changed")\n    missing_bars = [r["date"] for r in records if r["values"] is None]\n    require(missing_bars == ["2023-03-24"], "Pinned D00 missing-bar calendar differs")\n    axis = original.classical._axis(data)\n    missing_labels = [day for day in axis if day not in data["history"]]\n    expected_missing = [(date(2023, 3, 24) + timedelta(days=k)).isoformat() for k in range(8)]\n    require(missing_labels == expected_missing, "Missing daily sigma7 states were altered")\n    lookup = {day: i for i, day in enumerate(axis)}\n    required = {str(day) for split in ("val", "calibration", "test")\n                for day in data[split + "_dates"]}\n    unavailable = []\n    for day in sorted(required):\n        i = lookup[day]\n        for lag in (1, 7):\n            if i < lag or not original.data_api.np.isfinite(data["history"].get(axis[i-lag], float("nan"))):\n                unavailable.append({"date": day, "lag": lag})\n    require(not unavailable, "A required held-out temporal-control lag is unavailable")\n    fit_diagnostics, excluded = original.classical.fit_diagnostic_mask(data)\n    require(before == _job_inventory(output), "Completed-job evidence changed during data-only preflight")\n    require(input_hashes == {str(path): sha(path) for path in [npz_path, *paths]},\n            "Input bytes changed during data-only preflight")\n    report = {\n        "kind": "e3_disclosed_D00_parser_compatibility_preflight",\n        "status": "passed_data_only", "source_commit": SOURCE, "master_run_id": MASTER,\n        "protocol_sha256": PROTOCOL_SHA256, "protocol_tag_object_sha": TAG_OBJECT,\n        "worker_sha256": sha(__file__), "frozen_source_sha256": original._d00_compat_source_checks,\n        "post_freeze_execution_amendment": True, "AI_assistance_declared": True,\n        "amendment_scope": "In-memory auxiliary BTC history parser accepts the already documented all-missing calendar bar; immutable source and input bytes stay unchanged.",\n        "arithmetic": "Original numpy.std(numpy.diff(numpy.log(eight consecutive observed closes)), ddof=0)",\n        "numpy_version": original.data_api.np.__version__,\n        "calendar_rows": len(records), "missing_OHLCV_dates": missing_bars,\n        "retained_missing_sigma7_calendar_states": missing_labels,\n        "daily_axis_rows": len(axis), "observed_sigma7_dates": len(data["history"]),\n        "cohorts": cohorts, "fit_diagnostic_rows": len(fit_diagnostics),\n        "predeclared_fit_diagnostic_exclusions": len(excluded),\n        "missing_required_control_lags": unavailable, "cohort_parity": parity,\n        "input_files": [{"file": x.name, "sha256": sha(x), "bytes": x.stat().st_size}\n                        for x in [npz_path, *paths]],\n        "existing_jobs_before_and_after": before,\n        "rows_filled": 0, "cohort_rows_removed": 0, "new_real_data_fits": 0,\n        "new_market_predictions": 0, "models_constructed": 0,\n        "new_campaign_started": False, "R1_accredited": False, "R3_accredited": False,\n    }\n    destination = output / "parser_compatibility" / "d00_preflight.json"\n    require(not any(x.is_symlink() for x in (destination, *destination.parents)),\n            "Compatibility report cannot traverse symlinks")\n    destination.parent.mkdir(exist_ok=True)\n    raw = (canonical(report) + "\\n").encode()\n    if destination.exists():\n        existing = json.loads(regular(destination).read_bytes())\n        # Keep the first preflight as immutable evidence. New successful jobs\n        # may legitimately have accumulated before a later resume.\n        old_binding = {k: v for k, v in existing.items() if k != "existing_jobs_before_and_after"}\n        new_binding = {k: v for k, v in report.items() if k != "existing_jobs_before_and_after"}\n        require(old_binding == new_binding,\n                "Existing parser preflight bindings differ; preserve it for explicit reconciliation")\n        report = existing\n    else:\n        with destination.open("xb") as stream:\n            stream.write(raw)\n    print(canonical({"event": "d00_parser_compatibility_preflight", "status": report["status"],\n                     "cohorts": cohorts, "existing_completed_jobs": before["completed_job_results"],\n                     "missing_required_control_lags": len(unavailable),\n                     "new_real_data_fits": 0, "R3_accredited": False}), flush=True)\n    return report\n\n\ndef main(argv=None):\n    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)\n    parser.add_argument("--frozen-run", required=True)\n    parser.add_argument("--preflight-only", action="store_true")\n    own, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)\n    original = load_frozen(own.frozen_run)\n    install_parser(original)\n    if own.preflight_only:\n        return preflight(original, parse_forwarded(remaining))\n    # The amended parser has no independent evaluation implementation. Every\n    # original flag (and its order/value) is forwarded without alteration.\n    sys.argv = [str(Path(own.frozen_run)), *remaining]\n    return original.main()\n\n\nif __name__ == "__main__":\n    main()\n'

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
