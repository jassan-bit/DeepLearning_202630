"""Recover a lost T4 lease and continue new opportunities on the allocated A100.

Earlier setup and resource-limit opportunities remain failed. No extra optimizer attempt,
budget reset, seed change, signal or Drive destination is introduced.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import subprocess
import urllib.request

ENGINE_COMMIT = "1e8745497b484518f766b6da5c19be8362d9add7"
FUNDING_COMMIT = "f540227c23c572b9ca7fb36920588727b029484a"
FUNDING_LAUNCHER = "notebooks/run_e2_reduced_github72_colab.py"
FUNDING_LAUNCHER_SHA256 = "6a970b13cd04efa176293ab54c30f2f73608e86769fa1f75463937eb3044c875"
FUNDING_POLICY_SHA256 = "a1cf19c7a4c88f525afa5d7369546b4e31a898447cee56830ec520bd542e350f"
METADATA_PATH = "book_e2/_static/current_implementation_20261108/source_design.json"
METADATA_SHA256 = "ce0a76304186fabf922f8e1351ea2e9d44db49f1714f44cd1fe2df04baeb3dbe"
METADATA_BYTES = 18811
EXTENSION_REGISTRATION = "experiments/e2_a100_recovery_20261113/registration.json"
RESOURCE_COMMIT = "b352ce7f27cd7b55e4141fbfb1548f506ed80987"
RESOURCE_REGISTRATION = "experiments/e2_resource_continuation_20261112/registration.json"
RESOURCE_FILES = ("scripts/e2_reduced_github_state.py", "scripts/e2_reduced_github_restore.py",
    "scripts/e2_registered_budget_extension.py", "scripts/run_e2_reduced_github72_delivery.py",
    "scripts/reconcile_e2_github_completed.py", "scripts/e2_known_setup_failure_selection.py",
    "scripts/e2_resource_failure_selection.py", "scripts/e2_registered_resource_extension.py",
    "scripts/e2_resource_worker.py", "scripts/e2_resilient_github_backup.py",
    "experiments/e2_resource_continuation_20261112/resource_policy.json")
EXTENSION_FILES = ("scripts/run_e2_a100_recovery_delivery.py",
    "scripts/e2_expired_lease_recovery.py", "scripts/e2_a100_cache_selection.py")
# Filled from the separately published preparation-only commit before Q runs.
PREPARATION_COMMIT = "d62a616a576fca304dcf987efefbba29ab990541"
PREPARATION_PATH = "notebooks/preparar_e2_a100_runtime_colab.py"
PREPARATION_SHA256 = "1b7d69c5ef568491067829652540bd6b0e992aa19526a1d3f5d2ac5f6cfd230b"



def verified_funding_summary(result):
    """Describe applied funding only from the existing, checksummed H anchor."""
    if not isinstance(result, dict) or result.get("engine_commit") != ENGINE_COMMIT:
        return result
    result.pop("budget_extended", None)  # G's historical hardcoded field is ambiguous.
    result.pop("effective_total_budget_hours", None)
    result["budget_extension_applied"] = False
    result["funding_summary_status"] = "local_anchor_not_verified"
    try:
        workspace = Path(result["temporary_workspace"])
        raw = (workspace / "shared_budget/funding_extension.json").read_bytes()
        anchor = json.loads(raw)
        unsigned = dict(anchor); checksum = unsigned.pop("sha256")
        canonical = json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
        proof = anchor["registration_proof"]
        if (hashlib.sha256(canonical).hexdigest() != checksum
                or anchor.get("version") != "e2-registered-budget-extension-v1"
                or anchor.get("funding_policy_sha256") != FUNDING_POLICY_SHA256
                or anchor.get("initial_total_hours") != 64
                or anchor.get("effective_total_hours") != 72
                or anchor.get("budget_reset") is not False
                or proof.get("registration_commit") != FUNDING_COMMIT
                or proof.get("registration_path") != "experiments/e2_budget_extension_20261112/registration.json"
                or proof.get("funding_policy_sha256") != FUNDING_POLICY_SHA256):
            return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return result
    result.update(effective_total_budget_hours=72, budget_extension_applied=True,
        funding_summary_status="verified_from_local_anchor", funding_registration_commit=FUNDING_COMMIT)
    return result


def materialize_pinned_metadata(repo, *, opener=None):
    """Copy only exact author-verification metadata; never decode market data."""
    repo = Path(repo).absolute()
    target = repo / METADATA_PATH
    if (not target.resolve().is_relative_to(repo.resolve())
            or any(p.is_symlink() for p in (target, *target.parents) if p.is_relative_to(repo))):
        raise RuntimeError("The pinned metadata path must not resolve through a symlink")
    if target.exists():
        raw = target.read_bytes()
        reused = True
    else:
        url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + ENGINE_COMMIT + "/" + METADATA_PATH
        with (urllib.request.urlopen if opener is None else opener)(url, timeout=30) as response:
            raw = response.read(METADATA_BYTES + 1)
        reused = False
    if len(raw) != METADATA_BYTES or hashlib.sha256(raw).hexdigest() != METADATA_SHA256:
        raise RuntimeError("The existing/downloaded author metadata differs; nothing was replaced")
    if not reused:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".source-design-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            # The launcher's exclusive lock prevents another local controller.
            if target.exists():
                if target.read_bytes() != raw:
                    raise RuntimeError("Metadata appeared with different bytes; preserved")
            else:
                os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {"path": METADATA_PATH, "bytes": len(raw), "sha256": METADATA_SHA256,
            "engine_commit": ENGINE_COMMIT, "reused": reused,
            "market_values_opened": False}


def _funding_launcher():
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + FUNDING_COMMIT + "/" + FUNDING_LAUNCHER
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or hashlib.sha256(raw).hexdigest() != FUNDING_LAUNCHER_SHA256:
        raise RuntimeError("The registered funding launcher differs")
    module = {"__name__": "e2_registered_funding_launcher"}
    exec(compile(raw, "e2_registered_funding_launcher", "exec"), module)
    return module


def _prepare_runtime(namespace):
    python=Path(namespace.get("GPU_PYTHON") or
        "/content/entregable2_gpu_workspace/reduced-runtime-py312-cu126-v1/venv-py312-cu126/bin/python")
    if python.is_file():
        env=dict(os.environ)
        for key in ('GITHUB_TOKEN','GH_TOKEN','PYTHONPATH'):env.pop(key,None)
        probe="import sys,torch; assert sys.version_info[:3]==(3,12,12); assert torch.__version__=='2.8.0+cu126'; assert torch.version.cuda=='12.6'; assert torch.cuda.is_available(); assert 'A100' in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))"
        actual=subprocess.run([str(python),'-c',probe],env=env,capture_output=True,text=True,check=True,timeout=30).stdout.strip()
        namespace["GPU_PYTHON"]=str(python)
        print('Entorno aislado existente verificado en:',actual,flush=True)
        return
    if not re.fullmatch(r"[0-9a-f]{40}",PREPARATION_COMMIT):
        raise RuntimeError("The preparation-only commit has not been bound")
    url="https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/"+PREPARATION_COMMIT+"/"+PREPARATION_PATH
    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(2*1024**2+1)
    if len(raw)>2*1024**2 or hashlib.sha256(raw).hexdigest()!=PREPARATION_SHA256:
        raise RuntimeError("The pinned A100 environment preparation source differs")
    module={"__name__":"e2_a100_runtime_preparation_only"}
    exec(compile(raw,"e2_a100_runtime_preparation_only","exec"),module)
    module["prepare_e2_a100_runtime"](namespace)


def run_e2_a100_recovery_colab(extension_commit, *, namespace=None, execute=False):
    if not isinstance(extension_commit,str) or not re.fullmatch(r"[0-9a-f]{40}",extension_commit):
        raise ValueError("Use the immutable A100-recovery extension commit")
    if type(execute) is not bool:raise TypeError("execute must be boolean")
    namespace=globals() if namespace is None else namespace
    _prepare_runtime(namespace)
    if not execute:
        print("Entorno preparado; recuperación y entrenamiento todavía no solicitados.",flush=True)
        return {"training_started":False,"recovery_requested":False,
            "GPU_PYTHON":namespace["GPU_PYTHON"],"persistent_destination":"github"}
    funding=_funding_launcher()
    funding["EXTENSION_REGISTRATION"]=EXTENSION_REGISTRATION
    funding["EXTENSION_FILES"]=EXTENSION_FILES
    original_base=funding["_base_launcher"]
    def recovered_base():
        base=original_base()
        checkout=base["_checkout"]
        materialize=base["_extension"]
        stream=base["_stream"]
        def checked_checkout(commit,workspace,env):
            if commit!=ENGINE_COMMIT:raise ValueError("Retain the unchanged scientific engine E")
            repo=checkout(commit,workspace,env)
            report=materialize_pinned_metadata(repo)
            print("Metadata de TimesNet/TimeMixer verificada:",report["sha256"],flush=True)
            return repo
        def dual_extension(commit,workspace):
            if commit!=extension_commit:raise ValueError("The A100 extension commit changed")
            selected_registration=base["EXTENSION_REGISTRATION"]
            selected_files=base["EXTENSION_FILES"]
            try:
                base["EXTENSION_REGISTRATION"]=RESOURCE_REGISTRATION
                base["EXTENSION_FILES"]=RESOURCE_FILES
                materialize(RESOURCE_COMMIT,workspace)
            finally:
                base["EXTENSION_REGISTRATION"]=selected_registration
                base["EXTENSION_FILES"]=selected_files
            return materialize(commit,workspace)
        def recovered_stream(command,*args):
            command=list(command)
            if Path(command[1]).name!="run_e2_reduced_github72_delivery.py":
                raise ValueError("Unexpected parent-controller launch route")
            command[1]=Path(command[1]).with_name("run_e2_a100_recovery_delivery.py")
            resource=base["LOCAL_WORKSPACE"]/("github-primary-extension-"+RESOURCE_COMMIT)
            command.extend(["--resource-extension-root",resource])
            return stream(command,*args)
        if "json" in base:
            original_json=base["json"]
            class SummaryJSON:
                def __getattr__(self,name):return getattr(original_json,name)
                def dumps(self,value,*args,**kwargs):
                    if isinstance(value,dict) and "temporary_workspace" in value:
                        value=verified_funding_summary(value)
                    return original_json.dumps(value,*args,**kwargs)
            base["json"]=SummaryJSON()
        base["_checkout"]=checked_checkout
        base["_extension"]=dual_extension
        base["_stream"]=recovered_stream
        base["CONTROLLERS"].add("run_e2_a100_recovery_delivery.py")
        return base
    funding["_base_launcher"]=recovered_base
    result=funding["run_reduced_github72_colab"](extension_commit,namespace=namespace,execute=True)
    return verified_funding_summary(result)
