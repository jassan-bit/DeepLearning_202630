"""Check the documentary follow-up without market data, model code or network.

AI assistance declared. Optional --upstream checks the selected immutable files
of the preceding review packet; this follow-up does not rewrite that packet.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(directory, upstream=None):
    directory = Path(directory).resolve()
    manifest = read_json(directory / "manifest.json")
    for entry in manifest["payloads"]:
        path = directory / entry["path"]
        require(path.is_file() and not path.is_symlink(), "Missing or nonregular payload")
        require(path.stat().st_size == entry["bytes"], "Payload size changed: " + entry["path"])
        require(sha(path) == entry["sha256"], "Payload hash changed: " + entry["path"])

    declaration = read_json(directory / "contribution_declaration.json")
    require(declaration["AI_assistance_declared"] is True, "AI use must remain explicit")
    for key in ("R1_accredited", "R3_accredited", "novelty_absence_proved",
                "bad_pilot_refutes_novelty", "old_796_jobs_relabelled_as_V3",
                "old_packets_modified", "R2_student_defense_observed"):
        require(declaration[key] is False, "Unsupported claim: " + key)
    for key in ("new_model_fits_executed_by_this_consolidation", "new_model_inference",
                "new_hypothesis_tests"):
        require(declaration[key] == 0, "New execution improperly attributed: " + key)
    pilot = declaration["pilot"]
    require(pilot["completed_fits"] == pilot["HPO_fits"] + pilot["final_fits"] == 30,
            "Pilot fit geometry changed")
    require(pilot["final_seeds_per_recipe"] == 3 and pilot["assessment_rows"] == 55,
            "Development assessment geometry changed")
    require(pilot["scope"] == "exposed_development_only", "Confirmation scope drift")
    require(all(pilot[k] == 0 for k in ("test_arrays_opened", "calibration_arrays_opened",
                                       "additional_datasets_used")), "Reserved-data claim drift")
    require(pilot["continuation_heuristic_passed"] is False and
            pilot["p_values_computed"] is False and
            pilot["automatic_promotion_of_best_ablation"] is False,
            "Adverse verdict changed after results")
    scores = {row["variant"]: row for row in pilot["scores"]}
    require(set(scores) == {"additive_positive_head", "atom_crps", "atom_mse_only",
                            "fixed_probs", "atom_plugin"}, "Recipe omitted")
    full = scores["atom_crps"]["mean_RMSE"]
    best = min((name for name in scores if name != "atom_crps"),
               key=lambda name: scores[name]["mean_RMSE"])
    require(best == pilot["best_control"] == "atom_mse_only", "Best control mismatch")
    relative_gain = 1 - full / scores[best]["mean_RMSE"]
    require(math.isclose(relative_gain, pilot["full_relative_gain_vs_best_control"],
                         rel_tol=1e-13, abs_tol=1e-15), "Relative-gain arithmetic mismatch")
    require(math.isclose(-100 * relative_gain, 2.704311988801411,
                         rel_tol=1e-12, abs_tol=1e-12), "Adverse magnitude mismatch")
    require(relative_gain < 0 < pilot["continuation_threshold"] == .02,
            "Predeclared continuation threshold changed")
    require(scores["fixed_probs"]["parameters"] == 1236 and
            scores["atom_crps"]["parameters"] == scores["atom_mse_only"]["parameters"] ==
            scores["atom_plugin"]["parameters"] == 1262,
            "Capacity distinction hidden")
    literature = read_csv(directory / "literature_crosswalk.csv")
    require(len(literature) == len({row["source_id"] for row in literature}) == 15,
            "Literature records duplicated or omitted")
    require(all(row["primary_url"] and row["access_scope_reused"] and
                row["record_source"] for row in literature), "Unscoped source")
    require(len(read_csv(directory / "mechanism_loss_training_application.csv")) == 7,
            "Mechanism/training/application crosswalk incomplete")
    require(len(read_csv(directory / "r1_evidence_matrix.csv")) == 9,
            "Requirement evidence matrix incomplete")
    reading = declaration["source_reading_scope"]
    require(sum(reading[k] for k in ("old_complete_abstracts", "old_partial_abstracts",
                                     "old_listing_only", "old_unavailable_abstracts")) ==
            reading["old_abstract_queue"] == 50, "Abstract coverage inflated")
    require(reading["new_primary_reads_claimed"] == 0, "Invented fresh reading")
    checked_upstream = 0
    if upstream:
        upstream = Path(upstream).resolve()
        for entry in manifest["immutable_prior_review_files"]:
            path = upstream / entry["path"]
            require(path.is_file() and sha(path) == entry["sha256"],
                    "Previous review packet changed: " + entry["path"])
            checked_upstream += 1
    return {"status": "passed_documentary_consistency_and_integrity",
            "payload_files_verified": len(manifest["payloads"]),
            "immutable_prior_files_verified": checked_upstream,
            "literature_records": len(literature),
            "new_primary_reads": 0, "new_model_fits": 0, "new_model_inference": 0,
            "new_hypothesis_tests": 0, "historical_pilot_fits": 30,
            "continuation_heuristic_passed": False,
            "R1_accredited": False, "R3_accredited": False,
            "AI_assistance_declared": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.upstream), sort_keys=True))
