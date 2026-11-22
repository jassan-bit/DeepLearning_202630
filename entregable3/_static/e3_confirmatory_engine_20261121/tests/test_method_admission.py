"""Archive-integrity fixtures only, not executed statistical calibration."""
import gzip
import hashlib
import json
from pathlib import Path
from copy import deepcopy

import pytest

from engine import common as c, controller, statistics

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def archive(tmp_path):
    settings={"R":10,"T":768,"B":8191,"block_length":32,"hac_lags":48,
              "horizon":1,"margin":.02,"reference_reps":49999,"reference_seed":2026112103,"alpha":.025}
    scenarios=[{"id":"complete","null_rivals":20,"boundary_null":"epa","overlap":1},
               {"id":"partial","null_rivals":10,"boundary_null":"epa","overlap":1},
               {"id":"overlap","null_rivals":20,"boundary_null":"epa","overlap":5},
               {"id":"boundary","null_rivals":20,"boundary_null":"practical","overlap":1},
               {"id":"strong","null_rivals":0,"boundary_null":None,"overlap":1}]
    plan={"frozen_before_execution":True,"n_rivals":20,"panels_per_replicate":1,"n_fixed_seeds":10,"n_dates":768,
          "bootstrap_reps":8191,"block_length":32,"hac_lags":48,"replicates_per_stochastic_scenario":100,
          **{k:settings[k] for k in ("horizon","margin","reference_reps","reference_seed","alpha")},"scenarios":scenarios,
          "software_fixture_only":True}
    c.write_json(tmp_path/"plan.json",plan)
    c.atomic(tmp_path/"runner.py",b'# archive-integrity unit fixture; not a calibration runner\n')
    freeze={"plan_sha256":c.sha(tmp_path/"plan.json"),"runner_sha256":c.sha(tmp_path/"runner.py"),
            "statistics_sha256":c.sha(ROOT/"engine/statistics.py")}
    c.write_json(tmp_path/"freeze.json",freeze)
    rows=[]
    for scenario in scenarios:
        strong=scenario["null_rivals"]==0
        reps=[settings["B"] if (i//2)%3==1 else settings["reference_reps"] for i in range(120)]
        counts=[0 if strong else n for n in reps]
        raw=[(k+1)/(n+1) for k,n in zip(counts,reps)]
        for index in range(100):
            rows.append({"scenario":scenario["id"],"replicate":index,"false_reject_any_null":False,
                         "method_false_reject":{name:False for name in statistics.METHODS},
                         "per_rival_gate":[strong]*20,"joint_gate":strong,
                         "truth_mse":2. if strong else 0.,"truth_gain":.10 if strong else 0.,
                         "ci99_mse":[1.,3.] if strong else [-1.,1.],"ci99_gain":[.05,.15] if strong else [-.1,.1],
                         "coverage_ci99_mse":True,"coverage_ci99_gain":True,
                         "test_vectors":{"raw_p":raw,"adjusted_p":statistics.holm(raw),"status":["ok"]*120,
                                         "mc_tail_count":counts,"mc_reps":reps,"mc_p_rule":["absolute_tail_plus_one"]*120}})
    report={"status":"PASSED_PREDECLARED_VALIDATION","admitted":True,"protocol_closure_permitted":True,
            "family_rivals":20,"validation_scope":"single_panel_temporal_family","panels_per_replicate":1,"final_panels_application":4,
            "all_three_methods":True,"complete_and_partial_nulls":True,"extreme_tail_validated":True,"coverage_validated":True,
            "validated_settings":[settings],"statistics_source_sha256":freeze["statistics_sha256"],"software_fixture_only":True}
    def save():
        raw=b''.join((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode() for r in rows)
        c.atomic(tmp_path/"replicates.jsonl.gz",gzip.compress(raw,mtime=0))
        report["replicates_uncompressed_sha256"]=hashlib.sha256(raw).hexdigest()
        report["evidence"]={name:{"path":filename,"sha256":c.sha(tmp_path/filename)} for name,filename in
                            (("plan","plan.json"),("freeze","freeze.json"),("runner","runner.py"),("replicates","replicates.jsonl.gz"))}
        c.write_json(tmp_path/"report.json",report)
        return {"path":str(tmp_path/"report.json"),"sha256":c.sha(tmp_path/"report.json")}
    return settings,rows,report,save,tmp_path


def test_gzip_archive_counters_are_verified_without_cloning_four_panels(archive):
    settings,_,_,save,_=archive
    result=controller._method_admission(ROOT,save(),settings)
    assert set(result["verified_bundle"])=={"plan","freeze","runner","replicates"}


@pytest.mark.parametrize('change,diagnostic',[
    (lambda rows: rows[0].update(false_reject_any_null=True),'false_rejection_counter'),
    (lambda rows: rows[0].update(coverage_ci99_mse=False),'coverage_counter'),
    (lambda rows: rows[0].update(panel='F00'),'scenario_or_replicate'),
    (lambda rows: rows[0]['test_vectors']['mc_reps'].__setitem__(0,8191),'MC_tail_identity'),
    (lambda rows: rows[0]['test_vectors']['raw_p'].__setitem__(0,.01),'MC_tail_p_mismatch'),
    (lambda rows: rows[0]['test_vectors']['adjusted_p'].__setitem__(0,.01),'Holm_adjustment'),
    (lambda rows: rows.append(deepcopy(rows[0])),'duplicate_replicate'),
    (lambda rows: rows.pop(),'replicates_incomplete'),
    (lambda rows: rows[0]['test_vectors']['raw_p'].pop(),'120_tail_vectors'),
    (lambda rows: rows[0].update(per_rival_gate=[True]*20,joint_gate=True),'power_gate_contradicts'),
])
def test_archive_tampering_or_mismatched_counters_stop_admission(archive,change,diagnostic):
    settings,rows,_,save,_=archive
    change(rows)
    with pytest.raises(controller.ControllerError,match=diagnostic):
        controller._method_admission(ROOT,save(),settings)


def test_plaintext_hash_is_checked_after_gzip_decompression(archive):
    settings,_,report,save,path=archive
    save();report['replicates_uncompressed_sha256']='0'*64
    c.write_json(path/'report.json',report)
    with pytest.raises(controller.ControllerError,match='plaintext_hash'):
        controller._method_admission(ROOT,{'path':str(path/'report.json'),'sha256':c.sha(path/'report.json')},settings)


def test_failure_bounds_cannot_be_overruled_by_passed_report_boolean(archive):
    settings,rows,_,save,_=archive
    for row in rows[:10]:
        v=row['test_vectors'];v['mc_tail_count']=[0]*120
        v['raw_p']=[1/(n+1) for n in v['mc_reps']];v['adjusted_p']=statistics.holm(v['raw_p'])
        row['false_reject_any_null']=True;row['method_false_reject']={name:True for name in statistics.METHODS}
    with pytest.raises(controller.ControllerError,match='FWER_not_demonstrated'):
        controller._method_admission(ROOT,save(),settings)


def composite_archive(archive):
    """Small fake provenance ledgers; no new statistical simulation."""
    settings,old_rows,old_report,save,folder=archive
    old_plan=c.read_json(folder/'plan.json');old_plan['seed']=101;c.write_json(folder/'plan.json',old_plan)
    old_freeze=c.read_json(folder/'freeze.json');old_freeze['plan_sha256']=c.sha(folder/'plan.json');c.write_json(folder/'freeze.json',old_freeze)
    # The preserved old full-null rows intentionally do not satisfy the FWER
    # upper bound; the new independent ledger must replace them, never pool.
    template=deepcopy(old_rows[0])
    for row in old_rows[:10]:
        v=row['test_vectors'];v['mc_tail_count']=[0]*120
        v['raw_p']=[1/(n+1) for n in v['mc_reps']];v['adjusted_p']=statistics.holm(v['raw_p'])
        row['false_reject_any_null']=True;row['method_false_reject']={name:True for name in statistics.METHODS}
    old_report.update(status='INCONCLUSIVE_BOUNDED_VALIDATION',admitted=False,protocol_closure_permitted=False,validated_settings=[])
    save()
    current=folder/'focal';current.mkdir()
    new_rows=[{**deepcopy(template),'replicate':i} for i in range(120)]
    plan={**old_plan,'seed':202,'focal_scenario':'complete','replicates_per_stochastic_scenario':120,
          'replicates_per_scenario':{s['id']:120 if s['id']=='complete' else 100 for s in old_plan['scenarios']}}
    c.write_json(current/'plan.json',plan)
    c.atomic(current/'runner.py',b'# fixed independent provenance-composite unit fixture\n')
    mapping={'frozen_before_focal_execution':True,'no_pooling':True,'statistics_source_sha256':c.sha(ROOT/'engine/statistics.py'),
             'focal_scenario':'complete','prior_report_sha256':c.sha(folder/'report.json'),
             'scenario_sources':{s['id']:{'component':'replicates' if s['id']=='complete' else 'prior_replicates',
                                       'replicates':plan['replicates_per_scenario'][s['id']]} for s in plan['scenarios']},
             'preserved_but_excluded':{'prior_replicates':['complete']}}
    c.write_json(current/'map.json',mapping)
    freeze={'plan_sha256':c.sha(current/'plan.json'),'runner_sha256':c.sha(current/'runner.py'),
            'statistics_sha256':c.sha(ROOT/'engine/statistics.py'),'scenario_source_map_sha256':c.sha(current/'map.json')}
    c.write_json(current/'freeze.json',freeze)
    report={**old_report,'status':'PASSED_PREDECLARED_VALIDATION','admitted':True,'protocol_closure_permitted':True,
            'validated_settings':[settings],'report_type':'provenance_composite'}
    def save_composite():
        raw=b''.join((json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode() for row in new_rows)
        c.atomic(current/'replicates.jsonl.gz',gzip.compress(raw,mtime=0))
        report['replicates_uncompressed_sha256']=hashlib.sha256(raw).hexdigest()
        report['evidence']={key:{'path':str(path),'sha256':c.sha(path)} for key,path in [
            ('plan',current/'plan.json'),('freeze',current/'freeze.json'),('runner',current/'runner.py'),
            ('replicates',current/'replicates.jsonl.gz'),('prior_replicates',folder/'replicates.jsonl.gz'),
            ('prior_report',folder/'report.json'),('scenario_source_map',current/'map.json')]}
        report['evidence']['prior_replicates']['uncompressed_sha256']=old_report['replicates_uncompressed_sha256']
        c.write_json(current/'report.json',report)
        return {'path':str(current/'report.json'),'sha256':c.sha(current/'report.json')}
    return settings,new_rows,mapping,freeze,report,save_composite,current


def test_composite_counts_each_scenario_from_its_frozen_source_without_pooling(archive):
    settings,_,_,_,_,save,folder=composite_archive(archive)
    result=controller._method_admission(ROOT,save(),settings)
    assert result['report_type']=='provenance_composite'
    assert result['replicates_per_scenario']['complete']==120
    assert result['verified_ledgers']['replicates']['selected_rows']==120
    assert result['verified_ledgers']['prior_replicates']['selected_rows']==400
    assert result['verified_ledgers']['prior_replicates']['preserved_excluded_rows']==100
    assert c.read_json(folder.parent/'report.json')['status']=='INCONCLUSIVE_BOUNDED_VALIDATION'


@pytest.mark.parametrize('change,diagnostic',[
    ('changed_map','source_map_not_frozen'),
    ('pool','exclusion_map_invalid'),
    ('missing_new','replicates_incomplete'),
    ('missing_old_excluded','original_inconclusive_rows_not_preserved'),
    ('changed_parent','prior_report_or_focal_identity'),
])
def test_composite_rejects_changed_map_pooling_or_incomplete_component(archive,change,diagnostic):
    settings,rows,mapping,freeze,report,save,folder=composite_archive(archive)
    if change=='changed_map':
        mapping['scenario_sources']['complete']['replicates']=220;c.write_json(folder/'map.json',mapping)
    elif change=='pool':
        mapping['preserved_but_excluded']={};c.write_json(folder/'map.json',mapping)
        freeze['scenario_source_map_sha256']=c.sha(folder/'map.json');c.write_json(folder/'freeze.json',freeze)
    elif change=='missing_new':rows.pop()
    elif change=='missing_old_excluded':
        prior_path=folder.parent/'replicates.jsonl.gz'
        old=gzip.decompress(prior_path.read_bytes()).splitlines(keepends=True);raw=b''.join(old[1:])
        c.atomic(prior_path,gzip.compress(raw,mtime=0))
        old_report=c.read_json(folder.parent/'report.json')
        old_report['evidence']['replicates']['sha256']=c.sha(prior_path)
        old_report['replicates_uncompressed_sha256']=hashlib.sha256(raw).hexdigest()
        c.write_json(folder.parent/'report.json',old_report)
        mapping['prior_report_sha256']=c.sha(folder.parent/'report.json');c.write_json(folder/'map.json',mapping)
        freeze['scenario_source_map_sha256']=c.sha(folder/'map.json');c.write_json(folder/'freeze.json',freeze)
        # The closure's original old_report object supplies this field.
        report['evidence']={}
        original_save=save
        def save():
            spec=original_save()
            obj=c.read_json(folder/'report.json');obj['evidence']['prior_replicates']['uncompressed_sha256']=hashlib.sha256(raw).hexdigest()
            c.write_json(folder/'report.json',obj)
            return {'path':str(folder/'report.json'),'sha256':c.sha(folder/'report.json')}
    elif change=='changed_parent':
        obj=c.read_json(folder.parent/'report.json');obj['admitted']=True;c.write_json(folder.parent/'report.json',obj)
    with pytest.raises(controller.ControllerError,match=diagnostic):
        controller._method_admission(ROOT,save(),settings)
