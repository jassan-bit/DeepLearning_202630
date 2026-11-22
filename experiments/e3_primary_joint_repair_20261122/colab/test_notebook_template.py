"""Offline notebook state/activation checks. No numerical package installation."""
import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("statistical_notebook_fixture", HERE / "notebook_template.py")
nb = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(nb)


def code_cells(book):
    return [cell["source"] for cell in book["cells"] if cell["cell_type"] == "code"]


def test_notebook_has_resolved_pins_and_no_model_or_credential_flow():
    book=nb.notebook("a"*40, "b"*64, "c"*64)
    assert book["metadata"]["e3_monte_carlo_default"] is False
    assert book["metadata"]["e3_market_training_enabled"] is False
    assert book["metadata"]["e3_credentials_required"] is False
    for source in code_cells(book):
        ast.parse(source)
        assert "getpass" not in source and "userdata" not in source and "drive.mount" not in source
        assert "--activate" not in source and "predict-planning" not in source and "torch" not in source
    with pytest.raises(ValueError):
        nb.notebook("main", "b"*64, "c"*64)


def test_unresolved_template_stops_before_any_network_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("urllib.request.urlopen", side_effect=AssertionError("NO_UNPINNED_NETWORK")), \
         patch.object(Path, "mkdir", side_effect=AssertionError("NO_UNPINNED_WRITES")), \
         pytest.raises(AssertionError, match="pins completos"):
        exec(code_cells(nb.notebook())[0], {})


def default_namespace(tmp_path, monkeypatch, prior=None):
    fake_source=tmp_path / "source-fixture"; fake_source.mkdir()
    bootstrap=f'''
import subprocess
checked=[]
commands=[]
def check_python(python):
    checked.append(str(python))
    return {{"numerical_versions":{{"numpy":"2.3.5","scipy":"1.17.0"}}}}
def setup(commit,manifest,cache,existing_python=None):
    return {{"source_root":{str(fake_source)!r},"source_commit":commit,"manifest_sha256":manifest,
            "python":{sys.executable!r},"candidate_received":existing_python}}
def command(settings,args):
    commands.append(args)
    return subprocess.CompletedProcess([],0,stdout='{{"status":"SOFTWARE_FIXTURE_ONLY"}}')
'''.encode()
    book=nb.notebook("a"*40,"b"*64,hashlib.sha256(bootstrap).hexdigest())
    original=Path.is_dir
    monkeypatch.setattr(Path,"is_dir",lambda self:False if str(self)=="/content" else original(self))
    monkeypatch.chdir(tmp_path)
    old_source, old_work=tmp_path / "ORIGINAL_SOURCE",tmp_path / "ORIGINAL_WORK"
    ns=dict(__name__="__software_notebook_fixture__",SOURCE=old_source,WORK=old_work,
            SOURCE_COMMIT="ORIGINAL_SOURCE_PIN",SETTINGS=prior)
    with patch("urllib.request.urlopen",return_value=io.BytesIO(bootstrap)), \
         patch("subprocess.Popen",side_effect=AssertionError("NO_DEFAULT_PROCESS")), \
         patch("builtins.input",side_effect=AssertionError("NO_DEFAULT_INPUT")), \
         redirect_stdout(io.StringIO()):
        for source in code_cells(book):
            exec(compile(source,"statistical_notebook_fixture","exec"),ns)
    assert ns["SOURCE"] == old_source and ns["WORK"] == old_work and ns["SOURCE_COMMIT"] == "ORIGINAL_SOURCE_PIN"
    assert ns["SETTINGS"] is prior
    return ns,book


def test_default_executes_setup_only_and_preserves_other_experiments(tmp_path,monkeypatch):
    prior=dict(source_commit="unrelated_V5_source",python=sys.executable)
    ns,_=default_namespace(tmp_path,monkeypatch,prior)
    assert ns["statistics_bootstrap"].checked == []
    assert ns["statistics_bootstrap"].commands == []
    assert not ns["STATISTICS_CASE"].exists()
    assert ns["RUN_STATISTICAL_VALIDATION"] is False
    assert ns["PREPARE_VALIDATION_CASE"] is False
    assert ns["RESTORE_VALIDATION_CHECKPOINT"] is False


@pytest.mark.parametrize("commit",["db74afd0763596f71bcbafb4b9150a44a2e83339","c54a2fd3123a1ea8f98c2e56c2acfbe491c8672c","a"*40])
def test_only_declared_known_source_can_offer_previous_interpreter(tmp_path,monkeypatch,commit):
    ns,_=default_namespace(tmp_path,monkeypatch,dict(source_commit=commit,python=sys.executable))
    assert ns["statistics_bootstrap"].checked == [sys.executable]
    assert ns["VALIDATION_SETTINGS"]["candidate_received"] == sys.executable


def test_explicit_prepare_freezes_source_protocol_without_simulating(tmp_path,monkeypatch):
    ns,book=default_namespace(tmp_path,monkeypatch)
    source=code_cells(book)[1].replace("PREPARE_VALIDATION_CASE = False","PREPARE_VALIDATION_CASE = True",1)
    with redirect_stdout(io.StringIO()):
        exec(source,ns)
    commands=ns["statistics_bootstrap"].commands
    assert len(commands)==1 and commands[0][0]=="freeze"
    assert commands[0][-2:]==["--source-commit","a"*40]
    assert "run" not in commands[0] and "--protocol" in commands[0]


def test_activation_only_runs_bounded_resume_and_status(tmp_path,monkeypatch):
    ns,book=default_namespace(tmp_path,monkeypatch)
    ns["STATISTICS_CASE"].mkdir(parents=True)
    source=code_cells(book)[2].replace("RUN_STATISTICAL_VALIDATION = False","RUN_STATISTICAL_VALIDATION = True",1)
    with redirect_stdout(io.StringIO()):
        exec(source,ns)
    commands=ns["statistics_bootstrap"].commands
    assert [row[0] for row in commands] == ["status", "run", "status"]
    assert "--resume" in commands[1] and commands[1][-4:]==["--max-replicates","8","--seconds-budget","300"]
    assert not any(x in commands[1] for x in ("--alpha","--bootstrap-reps","--replicates","--train"))


def test_restore_is_explicit_and_keeps_existing_case(tmp_path,monkeypatch):
    ns,book=default_namespace(tmp_path,monkeypatch)
    ns["STATISTICS_CASE"].mkdir(parents=True)
    source=code_cells(book)[1].replace("RESTORE_VALIDATION_CHECKPOINT = False","RESTORE_VALIDATION_CHECKPOINT = True",1)
    with patch("builtins.__import__", wraps=__import__) as imported, \
         pytest.raises(AssertionError,match="Conservar el caso existente"):
        exec(source,ns)
    assert ns["statistics_bootstrap"].commands == []


def test_complete_checkpoint_is_inspected_without_running_new_replicas(tmp_path,monkeypatch):
    ns,book=default_namespace(tmp_path,monkeypatch)
    ns["STATISTICS_CASE"].mkdir(parents=True)
    calls=[]
    def completed(args):
        calls.append(args)
        return subprocess.CompletedProcess([],0,stdout='{"status":"completed","accepted":false}')
    ns["statistics_bootstrap"].command=lambda settings,args:completed(args)
    source=code_cells(book)[2].replace("RUN_STATISTICAL_VALIDATION = False","RUN_STATISTICAL_VALIDATION = True",1)
    with redirect_stdout(io.StringIO()):
        exec(source,ns)
    assert [args[0] for args in calls] == ["status"]


def test_published_complete_result_is_imported_without_rerunning_or_changing_its_verdict(tmp_path,monkeypatch):
    ns,_=default_namespace(tmp_path,monkeypatch)
    raw=b"EXPORTED_SOFTWARE_FIXTURE_CHECKPOINT_BYTES"
    checkpoint=dict(commit="d"*40,path="experiments/e3_primary_joint_repair_20261122/results/complete.zip",
                    sha256=hashlib.sha256(raw).hexdigest())
    book=nb.notebook("a"*40,"b"*64,"c"*64,published_checkpoint=checkpoint)
    calls=[]
    def command(settings,args):
        calls.append(args)
        result={"status":"completed","accepted":False} if args[0]=="status" else {"status":"restored_without_new_replicas"}
        return subprocess.CompletedProcess([],0,stdout=json.dumps(result))
    ns["statistics_bootstrap"].command=command
    with patch("urllib.request.urlopen",return_value=io.BytesIO(raw)),redirect_stdout(io.StringIO()):
        exec(code_cells(book)[1],ns)
    assert [args[0] for args in calls]==["restore","status"]
    assert ns["published_state"]["accepted"] is False
    assert book["metadata"]["e3_published_checkpoint"]==checkpoint
    assert ns["LOAD_PUBLISHED_CHECKPOINT"] is True and ns["RUN_STATISTICAL_VALIDATION"] is False


def test_published_import_preserves_existing_case_without_download(tmp_path,monkeypatch):
    ns,_=default_namespace(tmp_path,monkeypatch)
    ns["STATISTICS_CASE"].mkdir(parents=True)
    marker=ns["STATISTICS_CASE"] / "existing.json";marker.write_bytes(b"PRESERVE EXISTING CASE")
    before=(marker.read_bytes(),marker.stat().st_mtime_ns)
    checkpoint=dict(commit="d"*40,path="results/complete.zip",sha256="e"*64)
    book=nb.notebook("a"*40,"b"*64,"c"*64,published_checkpoint=checkpoint)
    with patch("urllib.request.urlopen",side_effect=AssertionError("NO_REPLACEMENT_DOWNLOAD")),redirect_stdout(io.StringIO()):
        exec(code_cells(book)[1],ns)
    assert [args[0] for args in ns["statistics_bootstrap"].commands]==["status"]
    assert (marker.read_bytes(),marker.stat().st_mtime_ns)==before


def test_published_checkpoint_bad_hash_stops_before_import_or_case_creation(tmp_path,monkeypatch):
    ns,_=default_namespace(tmp_path,monkeypatch)
    checkpoint=dict(commit="d"*40,path="results/complete.zip",sha256="e"*64)
    book=nb.notebook("a"*40,"b"*64,"c"*64,published_checkpoint=checkpoint)
    with patch("urllib.request.urlopen",return_value=io.BytesIO(b"WRONG_BYTES")), \
         pytest.raises(AssertionError,match="ZIP publicado distinto"):
        exec(code_cells(book)[1],ns)
    assert not ns["STATISTICS_CASE"].parent.exists()
    assert ns["statistics_bootstrap"].commands==[]
    with pytest.raises(ValueError,match="safe"):
        nb.notebook("a"*40,"b"*64,"c"*64,published_checkpoint=dict(checkpoint,path="../escape.zip"))
