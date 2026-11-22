"""Real byte-bound development declarations on generated software identities.

No real author consent is claimed, no network is used and no fitting occurs.
"""
from __future__ import annotations
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import bootstrap_colab
from engine import data
from package import crypto_development as helper
from package import crypto_notebook_template as nb

ROOT=Path(__file__).resolve().parents[1]


def snapshot(root):
    return {str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def namespace(tmp_path):
    anchor=json.loads((ROOT/'PUBLIC_SOURCE_ANCHOR.json').read_text())
    private=tmp_path/'private';private.mkdir()
    scope=dict(Path=Path,json=json,hashlib=hashlib,subprocess=subprocess,
        WORK=tmp_path,PRIVATE=private,SOURCE=ROOT,SOURCE_COMMIT=anchor['source_commit'],
        MANIFEST_SHA256=hashlib.sha256((ROOT/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
        SETTINGS=dict(python=sys.executable,source_root=str(ROOT),source_commit=anchor['source_commit']),
        bootstrap=SimpleNamespace(cpu_env=bootstrap_colab.cpu_env))
    cells=[c['source'] for c in nb.notebook()['cells'] if c['cell_type']=='code']
    with redirect_stdout(io.StringIO()):
        exec(compile(cells[1],'crypto_helpers_fixture','exec'),scope)
        exec(compile(next(s for s in cells if s.startswith('PREPARE_CRYPTO_DEVELOPMENT =')),
                     'crypto_metadata_fixture','exec'),scope)
    return scope


def test_cancellation_preserves_exact_private_metadata_and_never_acquires(tmp_path):
    scope=namespace(tmp_path)
    before=snapshot(scope['PRIVATE'])
    scope['run_data']=lambda *a,**k:pytest.fail('No acquisition/evidence mutation on cancellation')
    with patch('builtins.input',side_effect=['SOFTWARE FIXTURE','fixture@example.invalid','software fixture','NO']), \
         patch('urllib.request.urlopen',side_effect=AssertionError('NO_NETWORK')), \
         redirect_stdout(io.StringIO()):
        result=helper.prepare_development(scope)
    assert all(s['blank_owner'] for s in result.values())
    assert snapshot(scope['PRIVATE'])==before


def test_acceptance_makes_real_engine_receipts_and_rerun_preserves_them(tmp_path):
    scope=namespace(tmp_path)
    original=scope['run_data'];calls=[]
    def run(args,*,provider=None):
        calls.append(args[0])
        if args[0]=='collect-development':raise RuntimeError('SYNTHETIC stop before any network')
        assert args[0]=='record-owner-evidence'
        return original(args,provider=provider)
    scope['run_data']=run
    answers=['SOFTWARE FIXTURE ONLY','fixture@example.invalid','software fixture','ACEPTO DESARROLLO']
    with patch('builtins.input',side_effect=answers),redirect_stdout(io.StringIO()), \
         patch('urllib.request.urlopen',side_effect=AssertionError('NO_NETWORK')):
        with pytest.raises(RuntimeError,match='descarga detenida'):helper.prepare_development(scope)
    assert calls==['record-owner-evidence']*4+['collect-development']
    for panel in helper.PANELS:
        folder=scope['PRIVATE']/'inputs'/panel
        owner=json.loads((folder/'owner-development.json').read_text())
        receipt=json.loads((folder/'exposure-development.json').read_text())
        rights=json.loads((folder/'rights.json').read_text())
        data.verify_evidence(receipt,'audited');data.verify_evidence(rights,'approved')
        assert receipt['form']==owner and receipt['form_sha256']==data.identity(owner)
        assert owner['actor_name']=='SOFTWARE FIXTURE ONLY'
        assert owner['AI_completed_signature'] is False and owner['authorize_test_preparation'] is False
        assert receipt['final_reserve_claim'] is False and owner['allow_raw_redistribution'] is False
    before=snapshot(scope['PRIVATE'])
    calls.clear()
    with patch('builtins.input',side_effect=['ACEPTO DESARROLLO']),redirect_stdout(io.StringIO()):
        with pytest.raises(RuntimeError,match='descarga detenida'):helper.prepare_development(scope)
    assert calls==['collect-development']
    assert snapshot(scope['PRIVATE'])==before


def test_helper_rejects_source_pin_or_old_equity_definition_before_actor_input(tmp_path):
    scope=namespace(tmp_path)
    scope['MANIFEST_SHA256']='0'*64
    with patch('builtins.input',side_effect=AssertionError('NO_ACTOR_INPUT')), \
         pytest.raises(RuntimeError,match='Ancla|Manifest'):
        helper.prepare_development(scope)


def test_completed_admission_short_circuit_does_not_collect_or_prompt(tmp_path):
    scope=namespace(tmp_path)
    complete={panel:dict(admitted=True,raw_complete=True,blank_owner=False,owner_receipt=True,rights_present=True)
              for panel in helper.PANELS}
    scope['run_data']=lambda *a,**k:pytest.fail('Already complete must not acquire')
    with patch.object(helper,'_check',return_value=complete),patch('builtins.input',side_effect=AssertionError('NO_PROMPT')),redirect_stdout(io.StringIO()):
        assert helper.prepare_development(scope)==complete
