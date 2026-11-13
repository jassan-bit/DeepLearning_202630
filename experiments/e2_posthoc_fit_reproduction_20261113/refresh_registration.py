"""Refresh hashes after executable sources stabilize; no fitting or publication."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
FILES=(
    'scripts/e2_posthoc_fit_reproduction.py',
    'scripts/e2_frozen_posthoc_gpu.py',
    'scripts/export_e2_f2a_cached_data.py',
    'scripts/e2_native_transverse_colab.py',
    'scripts/e2_reduced_github_backup.py',
    'scripts/e2_reduced_github_state.py',
    'scripts/e2_reduced_github_restore.py',
    'notebooks/run_e2_posthoc_fit_reproduction_colab.py',
)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
record={
    'version':'e2-posthoc-clean-fit-reproduction-registration-v1',
    'scientific_commit':'1e8745497b484518f766b6da5c19be8362d9add7',
    'protocol_id':'e2-reduced-delivery-v1',
    'original_ledger_sha256':'27aa2336518432d1d873098a3b2e0e5dc039dc501e15caa17599fce7d1cc6c30',
    'total_authorized_hours':80,'new_budget_authorization':False,'budget_reset':False,
    'status':'prepared_for_actual_Colab_execution_not_market_executed',
    'entrypoint':'notebooks/run_e2_posthoc_fit_reproduction_colab.py:run_e2_posthoc_fit_reproduction_colab',
    'files_sha256':{relative:sha(ROOT/relative) for relative in FILES},
    'test_sha256':sha(ROOT/'tests/test_e2_posthoc_fit_reproduction.py'),
    'analysis_plan_sha256':sha(Path(__file__).parent/'analysis_plan.json'),
    'validation_sha256':sha(Path(__file__).parent/'validation.json'),
}
(Path(__file__).parent/'registration.json').write_text(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print('Refreshed8executable hashes; no fitting or publication')
