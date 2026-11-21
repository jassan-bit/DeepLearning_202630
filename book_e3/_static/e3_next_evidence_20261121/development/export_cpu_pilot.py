"""Export an explicit metadata/derived-results allowlist, never NPZ or weights.

The raw input and all new checkpoints remain local to the owned private case.
This standard-library helper hashes checkpoints without unpickling them. It
does not train, infer, grant licenses or certify an external human execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cpu_pilot_driver as driver

ROOT_FILES = ("cpu_pilot_plan.json", "admission.json", "prepared_receipt.json",
              "execution_anchor.json", "execution_started.json", "failures.json",
              "assessment_predictions.csv", "per_trajectory_metrics.csv",
              "recipe_summary.csv", "pilot_result.json", "assessment_worker.log")
JOB_FILES = ("job.json", "worker.log", "supervised/request.json",
             "supervised/phase_started.json", "supervised/receipt.json",
             "supervised/curve.json", "supervised/segments.log")
MAX_FILE_BYTES = 2_000_000
MAX_TOTAL_BYTES = 20_000_000


def regular(path):
    path = driver.no_symlink_path(path)
    driver.require(path.is_file() and path.stat().st_nlink == 1,
                   "Only owned regular nonaliased metadata is exportable")
    return path


def export_case(source_root, case_root, output, *, require_completed=False):
    source_root = driver.no_symlink_path(source_root)
    case_root = driver.no_symlink_path(case_root)
    output = driver.no_symlink_path(output)
    driver.require(case_root.is_dir() and case_root.name.startswith("v4-cpu-dev-"),
                   "Owned new CPU development case required")
    driver.require(not output.exists() and not output.is_relative_to(case_root),
                   "Fresh public destination outside the private case required")
    plan_path = regular(case_root / "cpu_pilot_plan.json")
    plan = driver.read_json(plan_path); driver.check_plan(plan)
    source_hash = driver.verify_sources(source_root, source_root / "pilot_source_manifest.json")
    driver.require(source_hash == plan["source_manifest_sha256"], "Pilot source identity differs")
    prepared = driver.read_json(regular(case_root / "prepared_receipt.json"))
    driver.require(prepared.get("plan_sha256") == driver.digest(plan)
                   and prepared.get("plan_file_sha256") == driver.sha(plan_path),
                   "Original prepared plan bytes/semantics differ")
    admission = regular(case_root / "admission.json")
    driver.require(driver.digest(driver.read_json(admission)) == plan["admission_sha256"],
                   "Prepared admission differs")
    selected = []
    for name in ROOT_FILES:
        path = case_root / name
        if path.exists() or path.is_symlink():
            selected.append((name, regular(path)))
    inventory = []
    completed = 0
    for job in plan["jobs"]:
        directory = driver.job_directory(case_root, job)
        record = driver.completed_job(case_root, plan, job)
        if record is not None:
            completed += 1
        checkpoint = directory / "supervised/checkpoint.pt"
        entry = {"job": job, "attempt_marker_present": (directory / "supervised/phase_started.json").is_file(),
                 "validated_complete_job": record is not None, "checkpoint_payload_included": False,
                 "checkpoint_publication_license_inferred": False}
        if checkpoint.exists() or checkpoint.is_symlink():
            checkpoint = regular(checkpoint)
            driver.require(checkpoint.stat().st_size <= 10_000_000, "Unexpected large private checkpoint")
            entry.update(local_private_relative_path=checkpoint.relative_to(case_root).as_posix(),
                         bytes=checkpoint.stat().st_size, sha256=driver.sha(checkpoint),
                         checkpoint_state="validated_by_completed_job_receipt" if record else
                                          "partial_or_initial_state_not_deserialized_by_exporter")
        inventory.append(entry)
        for suffix in JOB_FILES:
            path = directory / suffix
            if path.exists() or path.is_symlink():
                selected.append((path.relative_to(case_root).as_posix(), regular(path)))
    final_path = case_root / "pilot_result.json"
    complete = final_path.is_file()
    if complete:
        result = driver.read_json(regular(final_path))
        driver.require(completed == 20 and result.get("plan_sha256") == driver.digest(plan)
                       and result.get("status") == "completed_new_exposed_CPU_development_pilot"
                       and result.get("new_real_data_fits") == 20
                       and result.get("new_assessment_predictions") == 1210,
                       "Final result is not the frozen complete pilot")
        for name, wanted in result["files"].items():
            driver.require(name in ROOT_FILES and name.endswith(".csv"), "Unexpected final output name")
            path = regular(case_root / name)
            driver.require(path.stat().st_size == wanted["bytes"] and driver.sha(path) == wanted["sha256"],
                           "Final derived output differs")
    driver.require(not require_completed or complete, "Complete pilot required; partial evidence preserved locally")
    names = [name for name, _ in selected]
    driver.require(len(names) == len(set(names)) and not any(name.endswith((".npz", ".pt", ".npy"))
                   or name.startswith("inputs/") for name in names), "Raw data/weight export refused")
    driver.require(all(path.stat().st_size <= MAX_FILE_BYTES for _, path in selected)
                   and sum(path.stat().st_size for _, path in selected) <= MAX_TOTAL_BYTES,
                   "Metadata/derived export exceeds its fixed byte bound")
    output.mkdir(parents=True)
    for name, path in selected:
        target = output / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        driver.require(driver.sha(target) == driver.sha(path), "Copied metadata differs")
    driver.write_json(output / "LOCAL_CHECKPOINT_HASHES_ONLY.json", {
        "scope": "local newly produced states; hashes only, not uploaded checkpoint payload",
        "case_id": plan["case_id"], "source_plan_sha256": driver.digest(plan),
        "checkpoints": inventory, "input_NPZ_included": False,
        "checkpoint_payloads_included": False, "new_checkpoint_publication_license_inferred": False,
        "human_reproduction_claimed": False})
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        files.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                      "sha256": driver.sha(path)})
    manifest = {"schema_version": 1, "kind": "CPU_development_metadata_and_derived_output_export",
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": plan["case_id"], "plan_sha256": driver.digest(plan),
                "source_manifest_sha256": source_hash, "complete_pilot_result_present": complete,
                "validated_completed_jobs": completed,
                "planned_neural_jobs": 20, "files": files,
                "raw_NPZ_or_array_payload_included": False, "checkpoint_payload_included": False,
                "checkpoint_payloads_retained_in_original_private_case": True,
                "checkpoint_publication_license_inferred": False,
                "model_or_checkpoint_deserialized_by_exporter": False,
                "new_model_fits_by_exporter": 0, "new_predictions_by_exporter": 0,
                "R3_accredited": False, "AI_assistance_declared": True}
    driver.write_json(output / "EXPORT_MANIFEST.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--require-completed", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_case(args.source_root, args.case_root, args.output,
                                 require_completed=args.require_completed), indent=2))


if __name__ == "__main__":
    main()
