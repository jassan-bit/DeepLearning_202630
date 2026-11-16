"""Versioned execution adapter for the existing, unchanged real-control engine.

Preparation binds this adapter, its policy and the original engine before fits.
Only evidence/work destinations change; scientific methods remain in the engine.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import e2_corrections_controls as engine

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "book_e2/_static/e2_scientific_closure_20261116/controls"
WORK = ROOT / "work/e2-scientific-controls"
POLICY = PUBLIC / "execution_adapter_policy.json"
ENGINE_SHA = "1d5606526f676e9aef0d83ebb95150ba8ee117d3ac3715de22c81e5a2b63129c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(mode):
    path = ROOT / "scripts/e2_corrections_controls.py"
    if sha(path) != ENGINE_SHA:
        raise ValueError("The original registered control engine changed")
    if mode == "prepare" and not POLICY.exists():
        PUBLIC.mkdir(parents=True, exist_ok=True)
        policy = {"version": "e2-scientific-control-adapter-v1",
                  "registered_at_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "new posterior CPU controls on restored real 655/53/608 cohorts",
                  "engine": {"path": path.relative_to(ROOT).as_posix(), "sha256": ENGINE_SHA},
                  "adapter": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha(__file__)},
                  "public_directory": PUBLIC.relative_to(ROOT).as_posix(),
                  "work_directory": WORK.relative_to(ROOT).as_posix(),
                  "method_changed_by_adapter": False, "new_blind_test_claimed": False,
                  "original_A2_RQ1_or_recipes_changed": False,
                  "budget": "existing additive 84h account; an open matching reservation is mandatory for fits/statistics"}
        with POLICY.open("x", encoding="utf-8") as stream:
            json.dump(policy, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    if not POLICY.is_file():
        raise ValueError("Prepare and bind the real-control specification before execution")
    policy = json.loads(POLICY.read_text())
    if policy["engine"]["sha256"] != ENGINE_SHA or policy["adapter"]["sha256"] != sha(__file__):
        raise ValueError("Engine/adapter identity changed after preparation")
    original = engine.source_bindings

    def bindings():
        result = original()
        result[str(Path(__file__).resolve())] = sha(__file__)
        result[str(POLICY)] = sha(POLICY)
        return result

    engine.source_bindings = bindings
    engine.PUBLIC, engine.WORK = PUBLIC, WORK


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in {"-h", "--help"}:
        configure(sys.argv[1])
    engine.main()
