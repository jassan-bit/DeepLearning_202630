"""CPU-only data admission and public Render acceptance; no scientific fits."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys


RELATIVE = Path('experiments/e3_cierre_integral_20261118')
RENDER_URL = 'https://deeplearning-e3-dashboard.onrender.com'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def environment():
    result = dict(os.environ)
    for name in list(result):
        if (re.search(r'(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD|CREDENTIAL)(?:_|$)', name.upper())
                or name in {'PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'} or name.startswith(('PIP_','UV_'))):
            result.pop(name, None)
    result.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull,
                  PIP_DISABLE_PIP_VERSION_CHECK='1')
    return result


def streamed(command, cwd, log, env):
    """Preserve a console transcript without shell expansion or credentials."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        with subprocess.Popen([str(v) for v in command], cwd=cwd, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = process.wait()
    if code:
        raise RuntimeError('subprocess_failed_exit_'+str(code))


def prepare_browser(workspace, log):
    """Isolate Playwright from the Colab kernel; no Torch dependency."""
    workspace = Path(workspace).absolute()
    if any(p.is_symlink() for p in (workspace,*workspace.parents)):
        raise ValueError('browser_runtime_symlink_refused')
    workspace.mkdir(parents=True, exist_ok=True)
    python = workspace/'venv/bin/python'
    env = environment()
    env['PLAYWRIGHT_BROWSERS_PATH'] = str(workspace/'browsers')
    expected = {'playwright':'1.56.0','pyee':'13.0.1',
                'greenlet':'3.5.6','typing_extensions':'4.15.0'}
    if not python.is_file():
        # Colab may omit ensurepip. The existing kernel pip can target a
        # pip-less venv with --python, without installing into the kernel.
        streamed([sys.executable,'-m','venv','--without-pip',str(workspace/'venv')],workspace,log,env)
    probe = ('import importlib.metadata as m,json\nexpected='+repr(expected)
             + '\nassert all(m.version(k)==v for k,v in expected.items())\n'
             + 'print(json.dumps(expected))')
    check = subprocess.run([str(python),'-c',probe],env=env,stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,text=True)
    if check.returncode:
        print('Preparando Playwright aislado; el kernel y Torch de Colab no cambian.',flush=True)
        streamed([sys.executable,'-m','pip','--python',python,'install','--index-url','https://pypi.org/simple',
                  *[k+'=='+v for k,v in expected.items()]],workspace,log,env)
    subprocess.run([str(python),'-c',probe],env=env,check=True,stdout=subprocess.PIPE,text=True)
    # The browser's own revision is selected by the pinned Playwright package.
    ready = workspace/'browser_ready.json'
    if not ready.is_file():
        streamed([python,'-m','playwright','install','--with-deps','chromium'],workspace,log,env)
    verify = ('from playwright.sync_api import sync_playwright\nimport json\n'
              'with sync_playwright() as p:\n'
              ' b=p.chromium.launch(headless=True,args=["--no-sandbox","--disable-dev-shm-usage"])\n'
              ' print(json.dumps({"chromium":b.version}))\n b.close()\n')
    browser = json.loads(subprocess.check_output([str(python),'-c',verify],env=env,text=True))
    receipt = {'kind':'e3_cpu_browser_runtime','packages':expected,**browser,
               'kernel_python':sys.version,'isolated_python':str(python),
               'kernel_restarted':False,'Torch_changed':False}
    ready.write_bytes(canonical(receipt))
    return python,env,receipt


def run(checkout, output, source_commit, *, browser_workspace=None,
        collector=None, browser_setup=None, execute=None, render_url=RENDER_URL):
    """A completed receipt is immutable on retry; save it before creating a new run."""
    if re.fullmatch(r'[0-9a-f]{40}',source_commit or '') is None:
        raise ValueError('exact_source_commit_required')
    checkout,output = Path(checkout).absolute(),Path(output).absolute()
    if any(p.is_symlink() for p in (output,*output.parents)):
        raise ValueError('stage_symlink_refused')
    output.mkdir(parents=True,exist_ok=True)
    receipt_path = output/'stage_receipt.json'
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_bytes())
        if previous.get('source_commit')!=source_commit or previous.get('profile')!='cpu':
            raise ValueError('existing_stage_source_mismatch')
        print('Evidencia local terminada: se conserva y sólo se reintenta su guardado.',flush=True)
        return previous
    start = datetime.now(timezone.utc).isoformat()
    data = None
    data_error = None
    print('Etapa CPU: verificando bytes y calendario de D01, D02 y D03. No se entrenan modelos.',flush=True)
    try:
        collect = collector or module(checkout/RELATIVE/'data/collect.py','e3_closure_collect').run
        data = collect(output/'data',source_commit)
    except Exception as error:
        data_error = type(error).__name__+': '+str(error)
        (output/'data_failure.json').write_bytes(canonical({'error':data_error}))
    report_path = output/'render/public_smoke.json'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    setup_receipt = None
    print('Etapa CPU: comprobando Render con navegador y cinco inferencias.',flush=True)
    try:
        setup = browser_setup or prepare_browser
        python,env,setup_receipt = setup(browser_workspace or output.parent/'e3-closure-browser-v1',
                                        output/'render/browser_setup.log')
        (output/'render/browser_runtime.json').write_bytes(canonical(setup_receipt))
        command = [str(python),str(checkout/RELATIVE/'acceptance/smoke_render_v2.py'),
                   '--url',render_url,'--deployment-commit',source_commit,
                   '--output',str(report_path),'--screenshots-dir',str(output/'render/screenshots'),
                   '--wake-seconds','180']
        (execute or streamed)(command,checkout,output/'render/public_smoke.log',env)
    except Exception as error:
        # Preserve the browser's detailed failed report if it already exists.
        if not report_path.is_file():
            report_path.write_bytes(canonical({'status':'blocked_before_browser_acceptance',
                'public_deployment_smoke_passed':False,
                'error':type(error).__name__+': '+str(error)}))
    smoke = json.loads(report_path.read_bytes())
    # Read a positive boolean set by the strict V2 client, never infer success
    # from an HTTP healthcheck, a process exit, or a screenshot.
    public_ok = smoke.get('public_deployment_smoke_passed') is True
    receipt = {'kind':'e3_closure_stage_receipt','profile':'cpu',
               'source_commit':source_commit,'started_at_utc':start,
               'completed_at_utc':datetime.now(timezone.utc).isoformat(),
               'status':'cpu_evidence_collected','data_error':data_error,
               'datasets_admitted':data.get('datasets_admitted',0) if data else 0,
               'datasets_evaluated_by_models':0,
               'public_deployment_smoke_passed':public_ok,
               'render_report':'render/public_smoke.json',
               'new_real_data_fits':0,'new_scientific_market_predictions':0,
               'serving_check_is_not_generalization_evaluation':True,
               'new_statistical_contrasts':0,'R1_accredited':False,'R3_accredited':False,
               'protocol_frozen':False,'real_confirmatory_campaign_allowed':False}
    receipt_path.write_bytes(canonical(receipt))
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt
