"""Check the final E2 disposition offline, without running models or statistics."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent

def validate_state(state):
    if state.get("management_closed") is not True or state.get("active_corrective_tasks") != []:
        raise ValueError("The frozen scope must have a recorded terminal management decision")
    for key in ("full_guide_compliance_certified", "full_audit_accepted", "full_historical_campaign_retrained_by_this_closure", "historical_preregistration_integral_certified", "independent_external_validation_established", "leakage_demonstrated_by_negative_control"):
        if state.get(key) is not False:
            raise ValueError("Unsupported scientific acceptance: " + key)
    if state.get("NAS_required") is not False or state.get("Scopus_original_records") != 7500:
        raise ValueError("The fixed architecture/literature scope changed")
    rows = state.get("observations", [])
    if {r.get("id") for r in rows} != {"C01", "C02", "C03", "C04"} or len(rows) != 4:
        raise ValueError("Exactly four final dispositions are required")
    for row in rows:
        if not row.get("management_status", "").startswith("closed_") or row.get("active_corrective_tasks") != []:
            raise ValueError("Unresolved management disposition")
        if row.get("full_guide_acceptance_established") is not False:
            raise ValueError("Closing a task cannot approve a missing acceptance criterion")
    return state

def verify(root=ROOT):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        rel = PurePosixPath(record["path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError("Unsafe manifest path")
        path = root / str(rel)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or nonregular input: " + str(rel))
        raw = path.read_bytes()
        if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Changed package input: " + str(rel))
    state = validate_state(json.loads((root / "decisions.json").read_text(encoding="utf-8")))
    prior = json.loads((root / "evidence/appendix_B_previous.json").read_text(encoding="utf-8"))
    current = json.loads((root / "appendix_B_final.json").read_text(encoding="utf-8"))
    if {r["id"]: r["status"] for r in prior["items"]} != {r["id"]: r["status"] for r in current["items"]}:
        raise ValueError("Documentary closure must preserve scientific Appendix B statuses")
    inputs = root / "evidence"
    negative = json.loads((inputs / "negative_controls.json").read_text(encoding="utf-8"))
    hpo = json.loads((inputs / "hpo_design.json").read_text(encoding="utf-8"))
    inference = json.loads((inputs / "conditional_inference.json").read_text(encoding="utf-8"))
    integrated = json.loads((inputs / "integrated_reanalysis.json").read_text(encoding="utf-8"))
    if negative["completed_fits"] != 6 or negative["original13_model_audit_accepted"] is not False:
        raise ValueError("Negative-control evidence does not support the disposition")
    if hpo["isolated_hyperparameter_importance_established"] is not False or hpo["completed_fold_outcomes"] != 231:
        raise ValueError("HPO evidence changed")
    if inference["independent_post_selection_validation_established"] is not False:
        raise ValueError("Independent validation was not established by this comparison")
    if integrated.get("status") != "passed" or integrated.get("full_historical_campaign_retrained") is not False:
        raise ValueError("Integrated replay must pass its own limited scope")
    return {"status": "passed", "management_closed": True, "observations": 4, "active_corrective_tasks": [], "scientific_acceptance_changed": False, "full_guide_compliance_certified": False, "checked_files": len(manifest["files"]), "new_fits": 0, "new_forwards": 0, "new_bootstrap_draws": 0, "network_calls": 0}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("verification_result.json"))
    args = parser.parse_args()
    result = verify()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
