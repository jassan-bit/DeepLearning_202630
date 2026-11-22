"""Real controller/worker and encrypted mock REST, generated source observations.

No human statement, market datum, final training, test label or real network is
used. Genuine-shaped authorization records are explicit software fixtures.
"""
import copy
import datetime as dt
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

from engine import common as c, controller, data, models, training
from engine.storage import GithubStore
import test_controller as controller_fixtures
from test_controller import SOURCE
from test_storage import FakeGit, TOKEN, SECRET


def store(git,branch,prefix):
    obj=GithubStore(TOKEN,SECRET,'owner/repo',branch,source_commit=SOURCE,prefix=prefix,sleep=lambda _:None)
    obj._request=git.request
    return obj


@pytest.fixture
def runtime():
    helper=controller_fixtures.ControllerTests();helper.setUp()
    try:yield helper
    finally:helper.tearDown()


def generated_dataset(helper):
    root=helper.base;raw_dir=root/'raw';raw_dir.mkdir()
    evidence=root/'evidence';evidence.mkdir()
    proof=evidence/'software-only.txt';proof.write_text('Generated unit fixture. Not an actual owner declaration or provider license.\n')
    base={'evidence_path':str(proof),'evidence_sha256':c.sha(proof),'actor_name':'SOFTWARE FIXTURE','actor_contact':'fixture@example.invalid'}
    rights={**base,'status':'approved','allow_derived_publication':False,'allow_raw_redistribution':False,
            'license':'generated-software-fixture-only','scope':'generated observations, not market data','source_url':'https://data.binance.vision/'}
    exposure={**base,'status':'audited','form':{'reserve_mode':'development_only','software_fixture_only':True},'final_reserve_claim':False}
    n=180;t=np.arange(n);close=100*np.exp(.001*t+.01*np.sin(t/3))
    output=io.BytesIO()
    rows=[]
    for i in range(n):
        date=dt.datetime(2020,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=i)
        start=int(date.timestamp())*1000
        rows.append(f'{start},{close[i]*.999},{close[i]*1.01},{close[i]*.99},{close[i]},{1000+i},{start+86400000-1},0,1,0,0,0\n')
    with zipfile.ZipFile(output,'w') as archive:archive.writestr('generated.csv',''.join(rows))
    payload=output.getvalue();(raw_dir/'generated.zip').write_bytes(payload)
    source_url='https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1d/generated.zip'
    manifest={'provider':'binance_vision','instrument':'BTCUSDT','scope':'development_only','acquired_utc':'2026-11-20T00:00:00+00:00',
              'rights':rights,'exposure':exposure,'files':[{'path':'generated.zip','url':source_url,'sha256':data.sha256_bytes(payload)}]}
    manifest['manifest_sha256']=data.identity(manifest)
    c.write_json(raw_dir/'raw_manifest.json',manifest)
    split={'raw_start':'2020-01-01','raw_end':(dt.date(2020,1,1)+dt.timedelta(days=n-1)).isoformat(),'calendar':'24/7',
           'raw_bounds':{'fit':[0,39],'selection':[40,79],'planning':[80,119],'calibration':[120,179]}}
    for panel,path in helper.panels.items():
        shutil.rmtree(path)
        data.prepare_development(raw_dir/'raw_manifest.json',split,path,rights=rights,exposure=exposure)
    return payload,source_url,raw_dir,evidence


def test_real_worker_recovery_replays_exact_metadata_and_skips_completed_job(runtime):
    h=runtime;payload,url,raw_dir,evidence=generated_dataset(h)
    controller.prepare_case(h.source,h.case,h.panels,'planning',SOURCE,h.protocolpath,software_fixture=True)
    git=FakeGit();remote=store(git,'planning-fixture','runs/planning')
    controller.run_case(h.source,h.case,h.panels,remote,activate=True,max_jobs=1)
    plan=c.read_json(h.case/'plan.json');first=plan['jobs'][0]
    selected_before=training.verify_job(first,h.case/'jobs'/first['id'])['checkpoint_sha256']
    original_admissions={panel:(path/'admission.json').read_bytes() for panel,path in h.panels.items()}
    original_protocol=h.protocolpath.read_bytes()
    power=h.base/'private-power.json';c.write_json(power,{'status':'NOT_DIMENSIONED','final_R':None,'final_T':None,'final_B':None,'software_fixture_only':True})
    power_bytes=power.read_bytes()
    controller.archive_evidence(h.source,h.case,power,h.base)
    controller.sync_case(h.source,h.case,remote)
    # Simulate lost local VM observations/configs. Sources are reinstalled by
    # bootstrap in real use, represented here by their still-pinned copy.
    shutil.rmtree(h.case);shutil.rmtree(raw_dir);shutil.rmtree(evidence)
    for path in h.panels.values():shutil.rmtree(path)
    h.protocolpath.unlink();power.unlink()
    with patch.object(data,'_network_get',side_effect=AssertionError('Metadata restore must not fetch observations')):
        restored=controller.restore_case(h.source,h.case,remote,h.base)
    assert restored['market_acquisition_executed'] is False
    assert h.protocolpath.read_bytes()==original_protocol and power.read_bytes()==power_bytes
    assert c.read_json(h.case/'plan.json')==plan
    downloads=[]
    def fake_http(actual):
        assert actual==url;downloads.append(actual);return payload
    with patch.object(data,'_network_get',side_effect=fake_http):
        for panel,record in [(p['id'],p) for p in plan['data_panels']]:
            binding=record['recovery_metadata'];bundle=h.case/binding['path'];meta=c.read_json(bundle)
            data.reacquire_development(bundle,record['panel_dir'],meta['original_raw_dir'],h.base,
                                      enable_market_acquisition=True,expected_bundle_sha256=binding['sha256'])
    assert downloads==[url]
    for panel,path in h.panels.items():assert (path/'admission.json').read_bytes()==original_admissions[panel]
    assert '2026-11-20T00:00:00+00:00' in (raw_dir/'raw_manifest.json').read_text()
    assert (raw_dir/'reacquisition_events.jsonl').is_file()
    with patch.object(training,'fit_job',wraps=training.fit_job) as fitter:
        resumed=controller.run_case(h.source,h.case,h.panels,remote,activate=True,max_jobs=1)
    assert resumed['jobs_skipped_verified']==1 and resumed['jobs_executed_this_call']==1
    assert fitter.call_args.args[0]['id']!=first['id']
    assert training.verify_job(first,h.case/'jobs'/first['id'])['checkpoint_sha256']==selected_before


def test_final_restore_recovers_planning_dependency_and_original_external_evidence(runtime):
    h=runtime;payload,url,raw_dir,evidence=generated_dataset(h)
    controller.prepare_case(h.source,h.case,h.panels,'planning',SOURCE,h.protocolpath,software_fixture=True)
    pgit=FakeGit();planning=store(pgit,'planning-fixture','runs/planning')
    controller.run_case(h.source,h.case,h.panels,planning,activate=True,max_jobs=1)
    planning_plan=c.read_json(h.case/'plan.json')
    power=h.base/'external-power.json';c.write_json(power,{'status':'NOT_DIMENSIONED','final_R':None,'final_T':None,'final_B':None,'software_fixture_only':True})
    controller.archive_evidence(h.source,h.case,power,h.base);controller.sync_case(h.source,h.case,planning)
    final_root=h.base/'final';final_root.mkdir()
    final_protocol=h.base/'external-final-protocol.json'
    proto=c.read_json(h.protocolpath);proto['final']['power_receipt']={'path':str(power),'sha256':c.sha(power)}
    c.write_json(final_protocol,proto)
    # Metadata-only fixture, deliberately not an admitted final campaign. The
    # full Cartesian job identity is genuine-shaped; no final fit can start.
    plan=copy.deepcopy(planning_plan);plan.update(stage='final',case_id='r3-engine-final-recovery-unit',software_fixture=False,
              software_test_only=True,original_case_root=str(final_root),original_protocol_path=str(final_protocol),
              protocol_sha256=c.sha(final_protocol),seeds=c.read_json(h.source/'configs/FINAL_SEED_BANK.json')['seed_ids'][:10])
    plan['dependency_cases']=[{'case_root':str(h.case),'case_id':planning_plan['case_id'],'plan_sha256':c.digest(planning_plan),'source_commit':SOURCE}]
    plan['jobs']=[]
    for panel in controller.PANELS:
        for mi,model in enumerate(models.METHODS):
            for si,seed in enumerate(plan['seeds'] if model in models.NEURAL else plan['seeds'][:1]):
                template=copy.deepcopy(planning_plan['jobs'][0]);template.update(id=f'{panel}-m{mi:02d}-s{si:04d}',panel=panel,model=model,seed=seed,protocol_sha256=plan['protocol_sha256'])
                template['identity']['case_id']=plan['case_id'];plan['jobs'].append(template)
    for panel in plan['data_panels']:
        binding=panel['recovery_metadata'];target=final_root/binding['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(h.case/binding['path'],target)
    c.write_json(final_root/'plan.json',plan);c.atomic(final_root/'protocol.json',final_protocol.read_bytes())
    c.atomic(final_root/'receipts/power.json',power.read_bytes())
    fgit=FakeGit();final_store=store(fgit,'final-fixture','runs/final');final_store.backup(final_root)
    shutil.rmtree(h.case);shutil.rmtree(final_root);power.unlink();final_protocol.unlink();h.protocolpath.unlink()
    with patch.object(data,'_network_get',side_effect=AssertionError('Restore must not download observations')):
        restored=controller.restore_case(h.source,final_root,final_store,h.base,planning_store=planning)
    assert restored['status']=='CASE_AND_ORIGINAL_EVIDENCE_RESTORED'
    assert len(restored['dependencies'])==1
    assert c.read_json(h.case/'plan.json')==planning_plan
    assert final_protocol.read_bytes()==(final_root/'protocol.json').read_bytes()
    assert c.sha(power)==proto['final']['power_receipt']['sha256']
    with pytest.raises(controller.ControllerError,match='final_protocol_not_closed'):
        controller._admit_final(h.source,proto,c.sha(final_protocol),power)


def test_original_case_restore_preserves_unlocked_state_and_refuses_locked_state(runtime):
    h=runtime;h.prepare();git=FakeGit();remote=store(git,'planning-fixture','runs/planning')
    remote.backup(h.case)
    (h.case/'retained-user-note.txt').write_text('Unuploaded local evidence must survive restore.')
    with c.case_lock(h.case):
        with pytest.raises(RuntimeError,match='Another process owns'):
            controller.restore_case(h.source,h.case,remote,h.base)
    assert (h.case/'retained-user-note.txt').exists()
    result=controller.restore_case(h.source,h.case,remote,h.base)
    preserved=Path(result['preserved_existing_case'])
    assert (preserved/'retained-user-note.txt').is_file()
    assert not (h.case/'retained-user-note.txt').exists()
    assert c.read_json(h.case/'plan.json')['original_case_root']==str(h.case)


def test_recovery_never_writes_missing_evidence_outside_supplied_workspace(tmp_path):
    workspace=tmp_path/'work';workspace.mkdir()
    target=tmp_path/'outside.txt';raw=b'bound original metadata'
    with pytest.raises(controller.ControllerError,match='outside_authorized_workspace'):
        controller._install_recovery_file(target,raw,data.sha256_bytes(raw),workspace)
    assert not target.exists()
