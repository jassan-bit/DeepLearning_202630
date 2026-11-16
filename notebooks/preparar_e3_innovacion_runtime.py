"""Prepare the pinned v2 Colab runtime; no data loading, fitting or credentials.

Reuse the audited isolated Python bootstrap without modifying its v1 source.
The private module instance uses the v2 lock, including SciPy for nominal
Student-t intervals and the synthetic quadrature check.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re


LOCK_NAME = "requirements-e3-innovation-gpu.lock.txt"


def _bootstrap_module():
    path = Path(__file__).with_name("preparar_e3_desarrollo_runtime.py")
    spec = importlib.util.spec_from_file_location("_e3_v2_private_runtime_bootstrap", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar el bootstrap aislado fijado.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LOCK_NAME = LOCK_NAME
    original_environment = module._environment

    def credential_free_environment(workspace):
        environment = original_environment(workspace)
        return {key: value for key, value in environment.items()
                if not re.search(r"(?:^|_)(?:TOKEN|PAT|SECRET|PASSWORD)(?:_|$)", key.upper())}

    module._environment = credential_free_environment
    return module


def preparar_entorno(workspace, device="cuda"):
    """Return an isolated interpreter/environment/receipt for the v2 lock.

    Installation and hardware detection are preparation. This function never
    creates an experiment, imports a model or reserves execution budget.
    """
    return _bootstrap_module().preparar_entorno(workspace, device=device)
