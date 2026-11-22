"""Metadata coupling fixtures; no reserve data, author declarations or models."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from engine import common as c, controller, data


def fixture(tmp_path):
    inventory=tmp_path/'independent-prior-inventory.txt'
    inventory.write_text('Software fixture of a prior exposure inventory, not author evidence.\n')
    inventory_sha=c.sha(inventory)
    proto_sha='b'*64;selected_sha='c'*64
    c.write_json(tmp_path/'training_seal.json',{'selected_manifest_sha256':selected_sha})
    plan={'protocol_sha256':proto_sha,'data_panels':[]}
    protocol={'final':{'test_panels':{}}}
    admissions={}
    paths={}
    for panel in controller.PANELS:
        provenance={'provider':'fixture_provider','instrument':panel,'calendar':'24/7','bar_index_origin':'2020-01-01','bar_index_convention':'zero_based_complete_observed_calendar_sessions_v1'}
        plan['data_panels'].append({'id':panel,'provenance':provenance,'development':{'fit':{'support':[0,10]}}})
        target={'reserve_mode':'documented_historical_reserve','count':10,'start':'2030-01-01','end':'2030-01-10',
                'source_identity_sha256':controller.source_identity_sha256(provenance),
                'exposure_evidence_sha256':inventory_sha}
        protocol['final']['test_panels'][panel]=target
        admissions[panel]={'provenance':provenance,'partition_paths':{'test':'unused.npz'},'partition_counts':{'test':10},
                           'partition_date_ranges':{'test':['2030-01-01','2030-01-10']},'partition_support_bounds':{'test':[11,30]},
                           'exposure':{'form':{'reserve_mode':'documented_historical_reserve','exposure_inventory_sha256':inventory_sha}},
                           'final_authorization':{'authorize_test_preparation':True,'protocol_sha256':proto_sha,
                                'selected_checkpoints_sha256':selected_sha,'sources_methods_size_frozen':True,'development_bar_index_origin':'2020-01-01',
                                'reserve_inventory':{'path':str(inventory),'sha256':inventory_sha}}}
        paths[panel]=tmp_path/panel
    c.write_json(tmp_path/'protocol.json',protocol)
    return plan,protocol,admissions,paths,inventory


def invoke(tmp_path,plan,admissions,paths):
    with patch.object(controller,'_admissions',return_value=(paths,admissions)), \
         patch.object(data,'verify_evidence'), \
         patch.object(data,'load_partition',side_effect=AssertionError('Metadata verification must not open test')):
        return controller._final_panels(tmp_path,plan,paths)


def test_source_identity_is_known_before_future_bytes_and_protocol(tmp_path):
    provenance={'provider':'a','instrument':'b','calendar':'XNYS'}
    expected=controller.source_identity_sha256(provenance)
    assert expected==controller.source_identity_sha256({**provenance,'raw_manifest_sha256':'0'*64,'protocol_sha256':'1'*64})
    assert expected!=controller.source_identity_sha256({**provenance,'instrument':'different'})


def test_final_metadata_verifies_independent_prior_source_and_exposure_files(tmp_path):
    plan,_,admissions,paths,_=fixture(tmp_path)
    actual_paths,actual_admissions=invoke(tmp_path,plan,admissions,paths)
    assert actual_paths==paths and actual_admissions==admissions


@pytest.mark.parametrize('change,diagnostic',[
    ('source','released_source_identity_differs'),
    ('origin','released_test_bar_index_origin_differs'),
    ('inventory','reserve_inventory_evidence_differs'),
    ('mode','released_reserve_mode_differs'),
    ('file','evidence_file_hash_differs'),
])
def test_released_source_or_reserve_inventory_cannot_differ_from_fixed_spec(tmp_path,change,diagnostic):
    plan,protocol,admissions,paths,inventory=fixture(tmp_path)
    if change=='source':
        protocol['final']['test_panels']['F00']['source_identity_sha256']='0'*64
        c.write_json(tmp_path/'protocol.json',protocol)
    elif change=='origin':
        admissions['F00']['provenance']={**admissions['F00']['provenance'],'bar_index_origin':'2029-01-01'}
    elif change=='mode':
        admissions['F00']['exposure']['form']['reserve_mode']='post_protocol'
    elif change=='inventory':
        admissions['F00']['exposure']['form']['exposure_inventory_sha256']='0'*64
    else:inventory.write_text('Changed exposure record after protocol closure\n')
    with pytest.raises(controller.ControllerError,match=diagnostic):
        invoke(tmp_path,plan,admissions,paths)


def test_progress_contains_only_safe_identity_and_status_fields(capsys):
    plan={'case_id':'unit-case','jobs':[{}]}
    job={'id':'F00-m00-s0000','panel':'F00','model':'input_only_mse','seed':123,'unprinted_private':'secret'}
    controller._progress('job_complete_backed_up',plan,job=job,status='complete_verified',remote_commit='a'*40)
    import json
    event=json.loads(capsys.readouterr().out)
    assert set(event)=={'event','case_id','jobs_total','job_id','panel','system','seed','status','remote_commit'}
    assert 'secret' not in str(event)


def test_final_preparation_requires_observed_origin_and_future_test_dates():
    records={panel:{"provenance":{"bar_index_origin":"2000-01-01","bar_index_convention":"zero_based_complete_observed_calendar_sessions_v1"},
                    "partition_date_ranges":{part:["2000-02-01","2018-03-27"] for part in controller.DEVELOPMENT}}
             for panel in controller.PANELS}
    protocol={"final":{"test_panels":{panel:{"start":"2018-03-28"} for panel in controller.PANELS}}}
    controller._prospective_final_support(protocol,records)
    protocol["final"]["test_panels"]["F00"]["start"]="2018-03-27"
    with pytest.raises(controller.ControllerError,match="not_after_all_development"):
        controller._prospective_final_support(protocol,records)
    protocol["final"]["test_panels"]["F00"]["start"]="2018-03-28"
    del records["F00"]["provenance"]["bar_index_origin"]
    with pytest.raises(controller.ControllerError,match="common_observed_bar_index_origin"):
        controller._prospective_final_support(protocol,records)
