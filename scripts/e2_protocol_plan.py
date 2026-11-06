#!/usr/bin/env python3
"""Validate/register the prospective E2 policy; never train or score a model.

Registration verifies Git bytes, not scientific correctness. Phase checks require
separate hashed execution receipts; the existence of this policy is insufficient.
Numerical demonstrations/tests use synthetic inputs only.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import random
from statistics import NormalDist
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = Path("experiments/e2_remediation_20261106")
PLAN_PATHS = [str(DIRECTORY / name) for name in
              ("hpo.json", "preregistration.json", "architecture_decisions.json")]
WATCHED_PATHS = [PLAN_PATHS[0], PLAN_PATHS[1], PLAN_PATHS[2],
                 "scripts/e2_protocol_plan.py", "tests/test_e2_protocol_plan.py"]
TOP10 = ["TSMixer", "TiDE", "TimeMixer", "TimesNet", "PatchTST", "Informer",
         "Autoformer", "iTransformer", "Crossformer", "S-Mamba"]
TUNABLES = TOP10 + ["GRU-E2", "LSTM-E2", "TCN-E2", "Ridge", "ARIMA"]
ANCHORS = ["GRU/D7-c03", "LSTM/D7-c02", "TCN/D7-c02"]
CATALOGUE = TOP10 + ["GRU-E2", "LSTM-E2", "TCN-E2"] + ANCHORS + [
    "Ridge", "Persistence", "SeasonalNaive", "ARIMA"]
DELTA = 0.00025772842664981106
GUIDE_HASH = "93499dcfda6a36d9621524a81821470017ba114ca6b08241a35de5a54f74a654"
PHASE_REQUIREMENTS = {
    "P0": ["environment", "data_audit", "P0_adapter_tests", "P0_recipe_freeze"],
    "OOF": ["environment", "data_audit", "P0_adapter_tests", "P0_recipe_freeze",
            "OOF_indices_audit"],
    "HPO": ["environment", "data_audit", "final_architecture_review", "all_adapter_tests", "candidate_registry",
            "compute_profile", "fair_budget"],
    "NAS_stage1": ["environment", "data_audit", "NAS_noise_pilot", "NAS_control_HPO",
                   "NAS_registry", "NAS_compute_profile"],
    "NAS_stage2": ["NAS_stage1_complete", "NAS_finalists_frozen"],
    "P1": ["environment", "data_audit", "final_architecture_review", "HPO_complete", "final_recipe_freeze"],
    "final_fit": ["environment", "data_audit", "final_architecture_review", "HPO_complete", "final_recipe_freeze",
                  "P1_complete", "power_selection"],
    "score": ["final_architecture_review", "final_checkpoint_freeze", "full_pipeline_audit", "analysis_tests",
              "calibration_plan", "test_cohort_frozen"],
}


def json_data(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_documents(root=ROOT):
    return tuple(json.loads((Path(root) / p).read_text()) for p in PLAN_PATHS)


def validate_documents(hpo, prereg, architecture, *, root=ROOT):
    """Fail on scientific/policy changes that require a new explicit review."""
    for document in (hpo, prereg, architecture):
        require(document["guide_sha256"] == GUIDE_HASH, "Different full-guide source")
        require(document["status"] == "specified_pending_git_registration_and_preflight",
                "A policy flag cannot assert registration or execution")
        require(document["no_training_or_new_Bitcoin_results"] is True,
                "This protocol record must not invent execution")
    require(prereg["registration"]["tracked_paths"] == WATCHED_PATHS,
            "All five protocol/script/test files must be registered together")
    contract = prereg["inherited_contract"]
    reference = json.loads((Path(root) / "book_e2/_static/e1_reference.json").read_text())
    require(prereg["source_E1_commit"] == reference["source_commit"], "E1 source changed")
    require(contract["partition"] == reference["partition"], "Inherited partition changed")
    require(contract["history_days"] == 7 and contract["horizon"] == 1 and
            contract["input_features"] == ["Open", "High", "Low", "Close", "Volume"] and
            contract["absolute_delta"] == DELTA and contract["baseline"] == ANCHORS[0] and
            contract["primary_metric"] == "mean_seed_RMSE_original_scale" and
            contract["new_unseen_test_assigned"] is False and
            contract["independent_market_series"] == 1, "Inherited D7 scientific contract changed")
    require(hpo["catalogue"] == CATALOGUE, "Expected twenty-model catalogue")
    models = hpo["tunable_models"]
    require([m["id"] for m in models] == TUNABLES, "Fifteen tunable recipes required")
    require(hpo["contextual_fixed_anchors"] == ANCHORS and
            hpo["deterministic_references"] == ["Persistence", "SeasonalNaive"],
            "Fixed historical and deterministic contexts must remain separate")
    require(all(m["trials"] == 12 and m["assigned_compute_seconds"] == 172800 for m in models),
            "Equal assigned T12/compute cap required for fifteen tunables")
    for m in models:
        if m["id"] not in ("Ridge", "ARIMA"):
            require(set(m["space"]) == {"learning_rate", "weight_decay", "dropout",
                                        "batch_size", "depth", "width"},
                    "Neural minimum HPO axes must be accounted for")
            require(m["space"]["learning_rate"]["distribution"] == "log_uniform" and
                    m["space"]["weight_decay"]["distribution"] == "log_uniform",
                    "Learning rate and weight decay need registered log ranges")
            require(all(m["space"][axis]["mode"] == "fixed_defined_architecture"
                        for axis in ("depth", "width")),
                    "Structural changes need architecture review/NAS; not silent HPO")
    entries = architecture["models"]
    registry = architecture["literature_registry"]
    require(digest(Path(root)/registry["path"]) == registry["sha256"],
            "Architecture source extraction registry changed")
    require([m["id"] for m in entries] == TUNABLES, "Architecture cases must cover all tunables")
    for m in entries:
        require(bool(m["evidence"]) and bool(m["fixed_structural_parameters"]),
                "Architecture decisions need source evidence and exact task structure")
        require(m["nas_required"] == (not m["architecture_defined"]),
                "NAS is conditional on undefined architecture, not availability of DOI")
        require(m["runtime_validated_by_this_record"] is False,
                "Specifications cannot certify adapter runtime")
        require(m["source_architecture_defined"] is True and
                type(m["final_architecture_exception_verified"]) is bool,
                "Separate source structure from final adapter/NAS exception review")
        if m["id"] in TOP10:
            require(m["pilot_recipe_is_exact_original"] is False and
                    bool(m["fixed_source_recipe"]),
                    "Tiny P0 recipes cannot replace published final source recipes")
        for evidence in m["evidence"]:
            if evidence["kind"] == "explicit_local_architecture":
                require(digest(Path(root)/evidence["path"]) == evidence["sha256"],
                        "Local architecture source changed: " + evidence["path"])
    require(architecture["nas_executed"] is False, "No NAS execution occurred")
    nas = architecture["nas_protocol"]
    require(nas["depth"] == 3 and nas["widths"] == [16, 32, 64, 128, 256] and
            nas["dropout_probability"] == .2 and nas["cardinality"] == 1000 and
            nas["stage1"]["no_pruning"] is True and nas["stage1"]["epoch_fraction"] == .25,
            "Conditional NAS must retain the complete declared space")
    stage2_seeds = nas["stage2"]["seed_ids"]
    require(nas["stage2"]["minimum_finalists"] >= 10 and
            len(stage2_seeds) >= 5 and len(set(stage2_seeds)) == len(stage2_seeds) and
            nas["stage1"]["seed"] not in stage2_seeds and
            not set(stage2_seeds) & set(nas["noise_pilot"]["seeds"]),
            "Invalid NAS phase2 seeds/finalists: independent disjoint seeds required")
    families = prereg["families"]
    require([f["id"] for f in families] == ["historical_anchor", "fair_retuned"],
            "Historical and fair-tuned comparisons are separate families")
    for f, ref in zip(families, [ANCHORS[0], "GRU-E2"]):
        expected = {(m, rq, ref) for m in TOP10 for rq in ("RQ1", "RQ2", "RQ3")}
        actual = {(x["candidate"], x["question"], x["reference"]) for x in f["hypotheses"]}
        require(f["alpha"] == .02 and f["control"] == "Holm_FWER" and
                f["hypothesis_count"] == 30 and len(f["hypotheses"]) == 30 and
                actual == expected, "Family must contain the exact thirty declared hypotheses")
    diagnostic = prereg["diagnostic_family"]
    require(diagnostic["alpha"] == .01 and diagnostic["hypothesis_count"] == 199 and
            len({x["id"] for x in diagnostic["hypotheses"]}) == 199,
            "Diagnostic family cannot shrink after failed tests")
    require(math.isclose(sum(f["alpha"] for f in families) + diagnostic["alpha"], .05),
            "Global registered error allocation exceeds or differs from .05")
    require(prereg["global_error_budget"]["allocations"] == {
        "historical_anchor": .02, "fair_retuned": .02, "diagnostics": .01},
        "Reported alpha allocation differs from actual families")
    require(prereg["generalization_gap"]["delta_g"] == DELTA,
            "Prospective operational gap tolerance changed")
    alt = prereg["alternatives_for_planning"]
    require(alt["RQ1_gain"] == 2 * DELTA and alt["RQ2_erosion"] == 2 * DELTA and
            alt["excess_above_null_boundary"] == DELTA and
            alt["RQ3_latency_relative_reduction"] == .2,
            "Power alternatives must exceed the null margin, not equal it")
    power = prereg["power"]
    require(power["target"] == .8 and power["maximum_R"] == 200 and
            power["mc_repetitions"] == 10000 and power["candidate_R"][-1] == 200 and
            power["alpha_for_conservative_planning"] == .02 / 30,
            "Registered power target/cap/adjustment changed")
    p0 = prereg["pilots"]["P0"]; p1 = prereg["pilots"]["P1"]
    require(p0["models"] == ["TimeMixer", "iTransformer", "PatchTST", ANCHORS[0]] and
            p0["seed_ids"] == list(range(660101, 660106)) and
            p0["not_final_HPO_recipe_power"] is True and
            p0["not_model_or_architecture_selection"] is True,
            "P0 is a fresh fixed-recipe feasibility/variability pilot, not final power")
    require(p0["development_gap_family"]["primary_dataset"] == "OOF_2024-01-01_to_2024-10-31" and
            p0["development_gap_family"]["alpha"] == .01 and
            p0["development_gap_family"]["size"] == 3 and
            p0["development_gap_family"]["secondary_NovDec_inferential_family"] is False,
            "Only OOF is the primary P0 gap family; do not choose a favorable calendar after results")
    seed_sets = [set(p0["seed_ids"]), set(p1["seed_ids"]),
                 set(hpo["selection"]["seed_ids"]), set(power["final_seed_ids_at_cap"])]
    require(all(not a & b for i, a in enumerate(seed_sets) for b in seed_sets[i+1:]),
            "Pilot/HPO/final randomizations must be disjoint")
    require(len(seed_sets[-1]) == 200 and len(seed_sets[1]) >= 5 and
            p1["recipe_manifest_frozen_before_pilot"] is True, "P1/final seed policy invalid")
    require(all(x == 0 for x in prereg["observed_execution"].values()),
            "Prospective policy cannot declare new experimental results")
    return {"policy_valid": True, "registered_by_this_validation": False,
            "ready_to_train_by_this_validation": False, "catalogue_size": 20,
            "tunable_recipes": 15, "trials_each": 12, "new_Bitcoin_results": 0,
            "primary_family_size": 30, "fair_family_size": 30,
            "diagnostic_family_size": 199, "global_allocated_alpha": .05}


def candidate_design(hpo, architecture):
    """Deterministic pre-result design; this does not execute HPO."""
    structures = {m["id"]: m["fixed_structural_parameters"] for m in architecture["models"]}
    rows = []
    for model in hpo["tunable_models"]:
        rng = random.Random(int.from_bytes(sha256((str(model["candidate_generation_seed"]) +
                                                  model["id"]).encode()).digest()[:8], "big"))
        orders = list(itertools.product([0, 1, 2], [0, 1], [0, 1]))
        for index in range(model["trials"]):
            parameters = {}
            if model["id"] == "ARIMA":
                p, d, q = orders[index]
                parameters = {"order": [p, d, q], "trend": "c" if d == 0 else "n"}
            else:
                for key, axis in model["space"].items():
                    kind = axis.get("distribution")
                    if kind == "log_uniform":
                        parameters[key] = math.exp(rng.uniform(math.log(axis["min"]), math.log(axis["max"])))
                    elif kind == "uniform":
                        parameters[key] = rng.uniform(axis["min"], axis["max"])
                    elif kind == "categorical":
                        parameters[key] = rng.choice(axis["values"])
            rows.append({"model": model["id"], "candidate_id": f"{model['id']}-c{index+1:02d}",
                         "parameters": parameters, "fixed_structure": structures[model["id"]],
                         "assigned_compute_seconds": model["assigned_compute_seconds"],
                         "status": "prospective_candidate_not_evaluated"})
    return rows


def nas_candidates():
    """Enumerate 1000 conditional cases, never claim that they were trained."""
    return [{"architecture_id": f"w{a:03d}-{b:03d}-{c:03d}-b{x}{y}{z}",
             "widths": [a, b, c], "dropout_bits": [x, y, z], "dropout_probability": .2,
             "status": "conditional_catalogue_not_executed"}
            for a, b, c in itertools.product([16, 32, 64, 128, 256], repeat=3)
            for x, y, z in itertools.product([0, 1], repeat=3)]


def choose_nas_k(within_sd, between_sd):
    require(math.isfinite(within_sd) and math.isfinite(between_sd) and
            within_sd >= 0 and between_sd >= 0, "Finite nonnegative pilot SDs required")
    return 40 if between_sd == 0 or within_sd / between_sd > 1 else 20


def wilson_lower(successes, repetitions, alpha=.05):
    require(type(successes) is int and type(repetitions) is int and
            0 <= successes <= repetitions and repetitions > 0 and 0 < alpha < 1,
            "Valid Monte Carlo success counts required")
    z = NormalDist().inv_cdf(1 - alpha); p = successes / repetitions
    center = p + z*z/(2*repetitions)
    radius = z * math.sqrt(p*(1-p)/repetitions + z*z/(4*repetitions*repetitions))
    return max(0., (center - radius)/(1 + z*z/repetitions))


def approximate_paired_R(sd_difference, excess=DELTA, alpha=.02/30, power=.8):
    """Conservative bilateral normal reference only; not temporal-power evidence."""
    require(math.isfinite(sd_difference) and sd_difference > 0 and math.isfinite(excess)
            and excess > 0 and 0 < alpha < 1 and 0 < power < 1, "Invalid planning inputs")
    z = NormalDist().inv_cdf(1-alpha/2) + NormalDist().inv_cdf(power)
    return max(5, math.ceil((z*sd_difference/excess)**2))


def select_final_R(curves, prereg):
    """Validate complete MC planning curves; cap honestly if target is unmet.

    Input is a future simulation receipt, never evidence of an executed pilot.
    A cell must be supplied for every registered hypothesis and both uncertainty
    sources at every candidate R, including proxy assumptions for unpiloted models.
    """
    policy = prereg["power"]
    ids = [h["id"] for f in prereg["families"] for h in f["hypotheses"]]
    expected = {(r, h, source) for r in policy["candidate_R"] for h in ids
                for source in policy["sources"]}
    observed = {}; bounds = {}
    for row in curves:
        key = (row["R"], row["hypothesis_id"], row["source"])
        require(key in expected and key not in observed, "Unknown/duplicate MC planning cell")
        require(row["repetitions"] == policy["mc_repetitions"] and
                row["mc_seed"] == policy["mc_seed"] and
                row["planning_kind"] in ("empirical_recipe_pilot", "conservative_representative_proxy"),
                "MC settings/proxy scope must be declared, not inferred from outcomes")
        require(row.get("pilot_receipt_sha256") and len(row["pilot_receipt_sha256"]) == 64,
                "MC planning requires an identified P1 pilot receipt")
        observed[key] = row
        bounds[key] = wilson_lower(row["successes"], row["repetitions"])
    require(set(observed) == expected, "Incomplete planning cells; cannot certify missing power")
    feasible = [r for r in policy["candidate_R"] if
                min(v for k, v in bounds.items() if k[0] == r) >= policy["target"]]
    chosen = min(feasible) if feasible else policy["maximum_R"]
    return {"selected_R": chosen, "target_met_in_planning_simulation": bool(feasible),
            "status": "planning_target_met" if feasible else "cap_selected_target_not_met",
            "minimum_power_Wilson_lower": min(v for k, v in bounds.items() if k[0] == chosen),
            "unpiloted_architectures_empirical_power_certified": False,
            "temporal_market_sample_size_increased_by_R": False,
            "MDE_report_required": not bool(feasible)}


def git_bytes(root, commit, relative):
    require(relative in json.loads((Path(root) / PLAN_PATHS[1]).read_text())["registration"]["tracked_paths"],
            "Unregistered path in registration receipt")
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "Full Git SHA required")
    result = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=root,
                            capture_output=True, check=False)
    require(result.returncode == 0, "Registration commit lacks required protocol bytes")
    return result.stdout


def registration_receipt(root=ROOT):
    hpo, prereg, architecture = read_documents(root)
    validate_documents(hpo, prereg, architecture, root=root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    hashes = {}
    for relative in prereg["registration"]["tracked_paths"]:
        current = (Path(root) / relative).read_bytes()
        require(git_bytes(root, commit, relative) == current,
                "Commit the exact protocol/script/tests before registration and before any fit")
        hashes[relative] = sha256(current).hexdigest()
    return {"kind": "Git_protocol_registration", "commit": commit,
            "files": hashes, "prospective_only_for_outputs_created_later": True,
            "no_fits_or_scores_performed": True,
            "scope": "Git identity only; pilot adapter/design and execution evidence must also be frozen"}


def check_registration(receipt, *, root=ROOT):
    expected = json.loads((Path(root) / PLAN_PATHS[1]).read_text())["registration"]["tracked_paths"]
    require(receipt.get("kind") == "Git_protocol_registration" and
            set(receipt.get("files", {})) == set(expected), "Incomplete registration receipt")
    for relative, claimed in receipt["files"].items():
        current = (Path(root) / relative).read_bytes()
        require(sha256(current).hexdigest() == claimed and
                sha256(git_bytes(root, receipt["commit"], relative)).hexdigest() == claimed,
                "Protocol was modified after registered commit; new registration required")
    return True


def require_phase(phase, registration, receipts, *, root=ROOT):
    require(phase in PHASE_REQUIREMENTS, "Unknown phase; this tool has no training entry point")
    hpo, prereg, architecture = read_documents(root)
    validate_documents(hpo, prereg, architecture, root=root)
    check_registration(registration, root=root)
    if phase in ("HPO", "P1", "final_fit", "score"):
        unresolved = [m["id"] for m in architecture["models"]
                      if not m["nas_required"] and not m["final_architecture_exception_verified"]]
        require(not unresolved, "Final architecture review is not verified: " + ", ".join(unresolved))
    if phase.startswith("NAS"):
        require(any(m["nas_required"] for m in architecture["models"]),
                "No registered current model requires NAS; do not apply it indiscriminately")
    expected_hash = digest(Path(root) / PLAN_PATHS[1])
    required = list(PHASE_REQUIREMENTS[phase])
    if phase in ("P1", "final_fit", "score") and any(m["nas_required"] for m in architecture["models"]):
        required += ["NAS_stage2_complete", "NAS_winner_freeze"]
    for key in required:
        r = receipts.get(key)
        require(isinstance(r, dict) and r.get("status") == "passed" and
                r.get("preregistration_sha256") == expected_hash, "Missing/stale preflight receipt: " + key)
        evidence = Path(r.get("evidence_path", ""))
        require(evidence.is_file() and r.get("evidence_sha256") == digest(evidence),
                "Missing or changed execution evidence: " + key)
    if phase == "score":
        require(receipts.get("prior_E2_scoring_events") == 0,
                "Only one registered E2 scoring phase; prior E1 knowledge remains disclosed")
    if phase == "final_fit":
        selection = receipts["power_selection"].get("selection")
        require(isinstance(selection, dict) and selection.get("selected_R") in prereg["power"]["candidate_R"] and
                type(selection.get("target_met_in_planning_simulation")) is bool,
                "Final fits need a concrete registered R and honest power status")
    return {"phase": phase, "receipt_checks_passed": True, "fit_or_score_performed": False,
            "scope": "Hash-bound receipt gate only; not a substitute for semantic audit of evidence"}


def export(root=ROOT, output=None):
    root = Path(root); hpo, prereg, architecture = read_documents(root)
    report = validate_documents(hpo, prereg, architecture, root=root)
    out = Path(output) if output is not None else root / "book_e2/_static/remediation_protocol"
    out.mkdir(parents=True, exist_ok=True)
    contents = {"validation.json": json_data(report),
                "candidate_registry.json": json_data(candidate_design(hpo, architecture)),
                "conditional_NAS_1000_not_executed.json": json_data(nas_candidates())}
    contents.update({Path(p).name: (root/p).read_bytes() for p in PLAN_PATHS})
    contents["provenance.json"] = json_data({
        "generator_sha256": digest(__file__), "input_hashes": {p: digest(root/p) for p in PLAN_PATHS},
        "output_hashes": {k: sha256(v).hexdigest() for k, v in contents.items()},
        "training_fits": 0, "test_scores": 0, "market_or_prediction_files_opened": 0})
    for name, payload in contents.items():
        (out/name).write_bytes(payload)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "export", "register", "phase"], nargs="?", default="validate")
    parser.add_argument("--output-dir")
    parser.add_argument("--phase", choices=list(PHASE_REQUIREMENTS))
    parser.add_argument("--registration")
    parser.add_argument("--receipts")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            report = validate_documents(*read_documents())
        elif args.command == "export":
            report = export(output=args.output_dir)
        elif args.command == "register":
            require(args.output_dir, "Explicit registration receipt output directory required")
            report = registration_receipt()
            out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
            (out/"registration.json").write_bytes(json_data(report))
        else:
            require(args.phase and args.registration and args.receipts,
                    "Phase gate needs phase, registration and hashed execution receipts")
            report = require_phase(args.phase, json.loads(Path(args.registration).read_text()),
                                   json.loads(Path(args.receipts).read_text()))
        print(json.dumps(report, ensure_ascii=False)); return 0
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
