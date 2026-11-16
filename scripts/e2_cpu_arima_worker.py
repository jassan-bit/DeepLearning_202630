#!/workspace/work/e2-opt-venv/bin/python
"""Pinned interpreter adapter for the three remaining CPU recovery workers.

Only adds the audited read-only Prediction.raw_value -> Prediction.raw alias.
The original registered worker is executed unchanged using runpy.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT / 'book_e2/_static/e2_scientific_closure_20261116/cpu_fixed_recipe_reproduction/runtime_recovery_v3/recovery_plan.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def check_command(argv, plan):
    require(argv and argv[0] == plan['original_driver']['path'], 'Only the original registered CPU driver is permitted')
    require('--worker' in argv, 'This adapter permits worker invocation only')
    def option(name):
        require(argv.count(name) == 1, 'One explicit ' + name + ' is required')
        return argv[argv.index(name) + 1]
    require(option('--job') in plan['remaining_job_ids'], 'No neural refitting is permitted in recovery')
    require(option('--output') == plan['new_output'], 'Recovery output identity differs')
    require(option('--plan') == plan['original_plan']['path'], 'Original CPU plan path differs')
    require(option('--source-root') == plan['frozen_source_root'], 'Frozen source root differs')


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    plan = json.loads(RECOVERY.read_text())
    require(sha(__file__) == plan['wrapper']['sha256'], 'Recovery wrapper changed')
    require(sys.executable == plan['python'], 'Use the registered CPU Python interpreter')
    for key in ['original_driver', 'original_plan', 'compatibility_adapter']:
        require(sha(plan[key]['path']) == plan[key]['sha256'], 'Recovery source/plan changed: ' + key)
    identity = json.loads((Path(plan['new_output']) / 'run_identity.json').read_text())
    require(identity['recovery_plan_sha256'] == sha(RECOVERY), 'Recovery plan is not bound to the new output')
    check_command(args, plan)
    source = Path(plan['frozen_source_root'])
    sys.path.insert(0, str(source / 'scripts'))
    spec = importlib.util.spec_from_file_location('_pinned_arima_name_compat', plan['compatibility_adapter']['path'])
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    receipt = compat.install_prediction_raw_alias(source)
    require(receipt['numeric_predictions_changed'] is False and receipt['alias_is_read_only'] is True,
            'Unexpected numeric compatibility behavior')
    print(json.dumps({'CPU_recovery_plan_sha256': sha(RECOVERY), 'compatibility': receipt}), flush=True)
    sys.argv = args
    runpy.run_path(args[0], run_name='__main__')


if __name__ == '__main__':
    main()
