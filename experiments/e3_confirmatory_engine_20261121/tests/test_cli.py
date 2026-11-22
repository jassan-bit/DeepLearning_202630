"""CLI safety and evidence binding; no market access or real GitHub writes."""
import io
import json
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import pytest

from engine import cli, controller


def call(args):
    output=io.StringIO()
    with redirect_stdout(output):code=cli.main(args)
    return code,json.loads(output.getvalue().splitlines()[-1])


def test_credentials_private_stdin_strict_and_redacted():
    value=cli.private_credentials(io.StringIO(json.dumps({"token":"a token/$","backup_secret":"long private key"})))
    assert value["token"]=="a token/$"
    assert cli.redact("a token/$ a%20token%2F%24 long private key",list(value.values()))=="[REDACTED] [REDACTED] [REDACTED]"
    with pytest.raises(ValueError):cli.private_credentials(io.StringIO('{"token":"x"}'))


def test_inactive_run_never_requests_credentials_or_trains(tmp_path):
    with patch.object(cli,"private_credentials",side_effect=AssertionError("Credentials must not be read")):
        code,out=call(["run","--case",str(tmp_path/"case"),"--source-commit","a"*40,"--panels",str(tmp_path/"panels.json")])
    assert code==2 and out["status"]=="blocked_or_stopped"
    assert "disabled" in out["message"] and not (tmp_path/"case").exists()


def test_inactive_evaluation_never_opens_test(tmp_path):
    with patch.object(cli,"private_credentials",side_effect=AssertionError("No credentials")):
        code,out=call(["evaluate","--case",str(tmp_path/"case"),"--source-commit","a"*40,"--panels",str(tmp_path/"panels.json")])
    assert code==2 and "disabled" in out["message"]


def test_panel_map_requires_whole_family_of_panels(tmp_path):
    p=tmp_path/"panels.json";p.write_text(json.dumps({"F00":"x"}))
    with pytest.raises(ValueError):cli.panel_map(p)
    p.write_text(json.dumps({f"F0{i}":str(tmp_path/f"p{i}") for i in range(4)}))
    assert len(cli.panel_map(p))==4


def test_failed_statistical_validation_cannot_freeze_final_protocol(tmp_path):
    p=tmp_path/"failed.json";p.write_text(json.dumps({"admitted":False,"protocol_closure_permitted":False}))
    with pytest.raises(ValueError,match="Exact final statistical settings"):
        cli.method_report(p)
    with pytest.raises(controller.ControllerError,match="not_admitted"):
        cli.method_report(p,{})


def test_wrong_method_hash_cannot_admit(tmp_path):
    p=tmp_path/"invented.json"
    report={"status":"PASSED_PREDECLARED_VALIDATION","admitted":True,"protocol_closure_permitted":True,
            "family_rivals":20,"validation_scope":"single_panel_temporal_family","panels_per_replicate":1,"final_panels_application":4,
            "all_three_methods":True,"complete_and_partial_nulls":True,"extreme_tail_validated":True,"coverage_validated":True,
            "validated_settings":[{}],"statistics_source_sha256":"0"*64}
    p.write_text(json.dumps(report))
    with pytest.raises(controller.ControllerError,match="statistics_source_differs"):
        cli.method_report(p,{})


def test_power_draft_never_creates_closed_protocol(tmp_path):
    base=tmp_path/"base.json";base.write_text('{}')
    power=tmp_path/"power.json";power.write_text('{"status":"NOT_DIMENSIONED"}')
    with patch.object(cli,"method_report",return_value={"admitted":True}):
        with pytest.raises(ValueError,match="not dimensioned"):
            cli.freeze_protocol(base,power,tmp_path/"method.json",tmp_path/"test.json",tmp_path/"closed.json")
    assert not (tmp_path/"closed.json").exists()


def test_assessment_persists_result_and_explicit_durability_receipt(tmp_path):
    case=tmp_path/"case";protocol=tmp_path/"protocol.json";protocol.write_text('{}')
    def assess(case_arg,protocol_arg,*,output_path):
        assert case_arg==case and protocol_arg==protocol
        assert output_path==case/"metrics/assessment.json"
        output_path.parent.mkdir(parents=True)
        result={"r3_scientific_criteria_satisfied":False,"software_fixture_only":True}
        output_path.write_text(json.dumps(result))
        return result
    with patch("engine.assessment.assess_case",side_effect=assess):
        code,result=call(["assess","--case",str(case),"--protocol",str(protocol)])
    assert code==0 and result["r3_scientific_criteria_satisfied"] is False
    receipt=json.loads((case/"analysis_receipt.json").read_text())
    assert receipt["assessment_file"]=="metrics/assessment.json"
    assert receipt["requires_sync_before_remote_durability_claim"] is True
    assert receipt["new_model_forwards"]==0
