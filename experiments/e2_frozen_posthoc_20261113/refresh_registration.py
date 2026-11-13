"""Refresh the immutable-source registration before committing the launcher."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "scripts/e2_frozen_posthoc_gpu.py",
    "scripts/export_e2_f2a_cached_data.py",
    "scripts/e2_posthoc_fit_reproduction.py",
    "notebooks/run_e2_frozen_posthoc_colab.py",
    "tests/test_e2_frozen_posthoc_gpu.py",
    "experiments/e2_frozen_posthoc_20261113/README.md",
    "experiments/e2_frozen_posthoc_20261113/refresh_registration.py",
)


def main():
    value = {
        "version": "e2-frozen-posthoc-registration-v1",
        "engine_commit": "1e8745497b484518f766b6da5c19be8362d9add7",
        "original_results_commit": "50cb634e9eb163b09f9607982842144ba9f74c87",
        "protocol_id": "e2-reduced-delivery-v1",
        "training_performed": False,
        "new_model_inference": True,
        "original_score_modified": False,
        "budget_extended": False,
        "wall_cap_seconds": 5400,
        "original_ledger_sha256": "27aa2336518432d1d873098a3b2e0e5dc039dc501e15caa17599fce7d1cc6c30",
        "original_remaining_seconds": 18169.090972095262,
        "persistent_destination": "github",
        "files_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES},
    }
    path = ROOT / "experiments/e2_frozen_posthoc_20261113/registration.json"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    print(path)


if __name__ == "__main__":
    main()
