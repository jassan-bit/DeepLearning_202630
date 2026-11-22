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
