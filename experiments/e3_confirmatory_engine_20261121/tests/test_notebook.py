"""Colab artifact contracts, executed without network, credentials or fitting."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from package import notebook_template as nb

ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_FLAGS = {
    "RECORD_OWNER_EVIDENCE", "ACQUIRE_MARKET_DEVELOPMENT", "PREPARE_MARKET_DEVELOPMENT",
    "RUN_PLANNING_TRAINING", "EXPORT_PLANNING_PREDICTIONS", "RUN_DIMENSIONING",
    "FREEZE_FINAL_PROTOCOL", "RUN_FINAL_TRAINING", "SEAL_FINAL_TRAINING",
    "REGISTER_FINAL_EXPOSURE", "PREPARE_RESERVED_TEST", "OPEN_RESERVED_EVALUATION",
    "ASSESS_SAVED_RESULTS", "SYNC_SAVED_RESULTS", "EXPORT_SAVED_RESULTS",
    "VERIFY_REMOTE_GITHUB", "RESTORE_FROM_GITHUB",
}


def generated():
    return nb.notebook("a"*40, "b"*64, "c"*64)


def code_cells(notebook):
    return [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]


def test_notebook_has_eight_operational_stages_and_clean_outputs(tmp_path):
    target = nb.write_notebook(tmp_path/"book.ipynb")
    notebook = json.loads(target.read_text())
    headings = [c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown"
                and re.match(r"## [1-8]\. ", c["source"])]
    assert len(headings) == 8
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert notebook["metadata"]["e3_market_training_default"] is False
    assert notebook["metadata"]["e3_test_opening_default"] is False


def test_all_sensitive_stage_flags_default_off():
    found = {}
    for source in code_cells(generated()):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ACTIVATION_FLAGS:
                        found[target.id] = node.value.value
    assert set(found) == ACTIVATION_FLAGS
    assert all(value is False for value in found.values())


def test_pins_validate_and_all_embedded_programs_compile():
    notebook = generated()
    assert notebook["metadata"]["e3_source_pins"]["source_commit"] == "a"*40
    with pytest.raises(ValueError):
        nb.notebook("main", "b"*64, "c"*64)
    programs = set()
    for source in code_cells(notebook):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_program"):
                        ast.parse(node.value.value)
                        programs.add(target.id)
    assert programs >= {"split_program", "verification_program", "remote_program"}


def test_default_notebook_executes_all_cells_with_mocked_pinned_http(tmp_path, monkeypatch):
    # The bootstrap's own tests verify its acquisition/install path. Here only
    # setup/check are mocked; the calendar-only split subprocess is real.
    fake_bootstrap = f'''
from types import SimpleNamespace
from bootstrap_colab import cpu_env
commands=[]
def setup(commit, manifest, cache, run_tests=True):
    assert commit == {'a'*40!r} and manifest == {'b'*64!r} and run_tests
    return {{"python":{sys.executable!r},"source_root":{str(ROOT)!r},"source_commit":commit}}
def command(settings,args,credentials=None,allow_blocked=False):
    assert credentials is None
    commands.append(args)
    return SimpleNamespace(returncode=0,stdout="{{}}")
'''.encode()
    notebook = nb.notebook("a"*40, "b"*64, hashlib.sha256(fake_bootstrap).hexdigest())
    calls = []

    def urlopen(url, **kwargs):
        calls.append(str(url))
        return io.BytesIO(fake_bootstrap)

    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: False if str(self) == "/content" else original_is_dir(self))
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__notebook_qa__"}
    with patch("urllib.request.urlopen", side_effect=urlopen), \
         patch("getpass.getpass", side_effect=AssertionError("Default notebook must not request secrets")):
        for source in code_cells(notebook):
            exec(compile(source, "generated_colab_cell", "exec"), namespace)
    assert calls == ["https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"+"a"*40+
                     "/experiments/e3_confirmatory_engine_20261121/bootstrap_colab.py"]
    assert namespace["bootstrap"].commands == [["check"]]
    assert namespace["missing"] == ["F00", "F01", "F02", "F03"]
    assert not namespace["PLANNING_CASE"].exists() and not namespace["FINAL_CASE"].exists()
    assert namespace["PLANNING_BRANCH"] != namespace["FINAL_BRANCH"]
    assert namespace["PLANNING_PREFIX"] != namespace["FINAL_PREFIX"]
    for pid in namespace["PANELS"]:
        folder = namespace["PRIVATE"] / "inputs" / pid
        form = json.loads((folder/"owner-development.json").read_text())
        assert form["actor_name"] is None and form["all_data_declared_exposed_development"] is False
        assert form["reserve_mode"] == "development_only"
        assert json.loads((folder/"rights-DRAFT.json").read_text())["status"] == "requires_real_actor"
        assert not (folder/"rights.json").exists()
        split = json.loads((folder/"development-split.json").read_text())
        bounds = list(split["raw_bounds"].values())
        assert bounds[0][0] == 0
        assert all(bounds[i][1]+1 == bounds[i+1][0] for i in range(3))
        assert bounds[-1][1]+1 == split["metadata_only_expected_raw_bars"]
        assert bounds[2][1]-bounds[2][0]+1-14 >= 80
        assert "test" not in split["raw_bounds"]
        assert split["prospective_split_rule"]["basis"].startswith("expected exchange sessions only")
    assert not (namespace["PRIVATE"] / "raw").exists()


def test_remote_credentials_go_to_stdin_adapter_not_arguments(tmp_path):
    helpers = code_cells(generated())[1]
    calls = []
    bootstrap = SimpleNamespace(command=lambda settings,args,**kw: calls.append((args,kw)) or "done")
    namespace = {"WORK": tmp_path, "SOURCE_COMMIT": "a"*40, "Path": Path, "json": json,
                 "PRIVATE": tmp_path/"private", "hashlib": hashlib, "bootstrap": bootstrap,
                 "SETTINGS": {}, "subprocess": __import__("subprocess")}
    exec(helpers, namespace)
    namespace["github_credentials"] = lambda: {"token":"private token", "backup_secret":"private secret"}
    assert namespace["run_cli"](["run", "--activate"], remote=True) == "done"
    args, kw = calls[0]
    assert args == ["run", "--activate"]
    # The dictionary is cleared when the child adapter returns. Secrets are not
    # copied to arguments/environment or left in the notebook globals.
    assert kw["credentials"] == {}
    assert "private" not in repr(args)
    assert "token" not in namespace


def test_method_bundle_selected_from_current_protocol_not_v2_filename():
    source = next(s for s in code_cells(generated()) if "FREEZE_FINAL_PROTOCOL =" in s)
    assert '["method_validation"]' in source
    assert 'method_binding["path"]' in source
    assert "verdict_v2.json" not in source


def test_assessment_is_synced_and_export_has_no_evaluation_call():
    source = next(s for s in code_cells(generated()) if "OPEN_RESERVED_EVALUATION =" in s)
    tree = ast.parse(source)
    sections = [node for node in tree.body if isinstance(node, ast.If)]
    export = next(node for node in sections if ast.unparse(node.test) == "EXPORT_SAVED_RESULTS")
    assert '"evaluate"' not in ast.get_source_segment(source, export)
    assert 'if ASSESS_SAVED_RESULTS or SYNC_SAVED_RESULTS:' in source
    assert '["sync", *remote_args' in source
    assert 'metrics' not in ast.get_source_segment(source, export)  # no fabricated assessment fallback


def _final_draft_fixture(tmp_path):
    from engine.controller import source_identity_sha256
    admissions={}
    inventory_files={}
    targets={}
    for pid in ("F00","F01","F02","F03"):
        folder=tmp_path/"inputs"/pid;folder.mkdir(parents=True)
        inventory=folder/"reserve-inventory.json"
        inventory.write_text(json.dumps({"fixture":"independent reserve inventory, not an authorization", "panel":pid}))
        inventory_files[pid]=inventory
        provenance={"provider":"software_fixture","instrument":pid,"calendar":"24/7","bar_index_origin":"2020-01-01"}
        admissions[pid]={"provenance":provenance}
        targets[pid]={"start":"2020-09-01","end":"2020-09-30","count":30,"reserve_mode":"external_custodian",
                      "source_identity_sha256":source_identity_sha256(provenance),
                      "exposure_evidence_sha256":hashlib.sha256(inventory.read_bytes()).hexdigest()}
    protocol=tmp_path/"protocol.json"
    protocol.write_text(json.dumps({"final":{"test_panels":targets},"closed_at_utc":"2020-08-01T00:00:00+00:00"}))
    for name in ("selected.json","seal.json"):
        (tmp_path/name).write_text(json.dumps({"software_fixture":True}))
    payload={"protocol":str(protocol),"panels":{pid:pid for pid in admissions},"inputs":str(tmp_path/"inputs"),
             "selected":str(tmp_path/"selected.json"),"seal":str(tmp_path/"seal.json"),
             "form":str(ROOT/"docs/OWNER_EXPOSURE_FORM.json")}
    program=None
    for source in code_cells(generated()):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="final_split_program" for t in node.targets):
                program=node.value.value
    assert program
    return program,payload,admissions,inventory_files


def test_final_drafts_bind_independent_inventory_and_full_development_origin(tmp_path):
    from engine import data
    program,payload,admissions,inventories=_final_draft_fixture(tmp_path)
    with patch.object(data,"read_admission",side_effect=lambda path:admissions[path]), \
         patch.object(data,"load_partition",side_effect=AssertionError("Metadata draft must not load labels")), \
         patch.object(sys,"stdin",io.StringIO(json.dumps(payload))):
        exec(compile(program,"final_metadata_draft_fixture","exec"),{})
    for pid in admissions:
        folder=tmp_path/"inputs"/pid
        split=json.loads((folder/"final-split-DRAFT.json").read_text())
        auth=json.loads((folder/"final-authorization-DRAFT.json").read_text())
        form=json.loads((folder/"owner-final-DRAFT.json").read_text())
        assert split["raw_start"]==auth["development_bar_index_origin"]=="2020-01-01"
        assert split["raw_bounds"]["test"]==[230,273]
        assert split["fixed_test_count"]==30 and split["metadata_only_target_sessions"]==30
        assert auth["status"]=="requires_real_actor" and auth["authorize_test_preparation"] is False
        assert auth["reserve_inventory"]["sha256"]==form["exposure_inventory_sha256"]==hashlib.sha256(inventories[pid].read_bytes()).hexdigest()
        assert form["protocol_sha256"]==hashlib.sha256(Path(payload["protocol"]).read_bytes()).hexdigest()
        assert form["protocol_sha256"]!=form["exposure_inventory_sha256"]
        assert form["actor_name"] is None and form["custodian_seal_sha256"] is None


@pytest.mark.parametrize("changed",["inventory","source"])
def test_final_draft_refuses_changed_independent_bindings_before_writing(tmp_path,changed):
    from engine import data
    program,payload,admissions,inventories=_final_draft_fixture(tmp_path)
    if changed=="inventory":
        inventories["F00"].write_text("changed after protocol lock")
    else:
        admissions["F00"]["provenance"]["instrument"]="changed-instrument"
    with patch.object(data,"read_admission",side_effect=lambda path:admissions[path]), \
         patch.object(data,"load_partition",side_effect=AssertionError("Must not read labels")), \
         patch.object(sys,"stdin",io.StringIO(json.dumps(payload))),pytest.raises(AssertionError):
        exec(compile(program,"adverse_final_metadata_draft_fixture","exec"),{})
    assert not list((tmp_path/"inputs").rglob("*-DRAFT.json"))
