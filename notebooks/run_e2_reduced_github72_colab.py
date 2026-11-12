"""Draft continuation launcher; funding requires a separately approved registration.

Queue this cell after the running GitHub-primary controller ends cleanly. It
never interrupts, signals or modifies an active worker. The scientific engine E
and the stable /content scratch location remain unchanged.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import urllib.request

BASE_COMMIT = "7ad290ada4ef9908aecfe042b610251206d9b099"
BASE_PATH = "notebooks/run_e2_reduced_github_colab.py"
BASE_SHA256 = "47d43540a7345135e731fc831a603b939ba324f2072f7af929f84b50836d0e85"
EXTENSION_REGISTRATION = "experiments/e2_budget_extension_20261112/registration.json"
EXTENSION_FILES = ("scripts/e2_reduced_github_state.py", "scripts/e2_reduced_github_restore.py",
    "scripts/e2_registered_budget_extension.py", "scripts/run_e2_reduced_github72_delivery.py")


def _base_launcher():
    url = "https://raw.githubusercontent.com/jassan-bit/DeepLearning_202630/" + BASE_COMMIT + "/" + BASE_PATH
    with urllib.request.urlopen(url, timeout=30) as response:
        code = response.read(2 * 1024**2 + 1)
    if len(code) > 2 * 1024**2 or hashlib.sha256(code).hexdigest() != BASE_SHA256:
        raise RuntimeError("The pinned GitHub-primary launcher bytes changed")
    module = {"__name__": "e2_registered_github_base_launcher"}
    exec(compile(code, "e2_github_primary_registered_base", "exec"), module)
    return module


def run_reduced_github72_colab(extension_commit, *, namespace=None, execute=False):
    """Continue only after the prior controller closes its fit and lease."""
    if not isinstance(extension_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", extension_commit):
        raise ValueError("An immutable funding-extension commit is required")
    if type(execute) is not bool:
        raise TypeError("execute must be boolean")
    namespace = globals() if namespace is None else namespace
    base = _base_launcher()
    base["EXTENSION_REGISTRATION"] = EXTENSION_REGISTRATION
    base["EXTENSION_FILES"] = EXTENSION_FILES
    base["CONTROLLERS"].add("run_e2_reduced_github72_delivery.py")
    stream = base["_stream"]
    def continued_stream(command, *args):
        command = list(command)
        if Path(command[1]).name != "run_e2_reduced_github_delivery.py":
            raise ValueError("Unexpected scientific-controller launch route")
        command[1] = Path(command[1]).with_name("run_e2_reduced_github72_delivery.py")
        return stream(command, *args)
    base["_stream"] = continued_stream
    return base["run_reduced_github_colab"](extension_commit, namespace=namespace, execute=execute)
