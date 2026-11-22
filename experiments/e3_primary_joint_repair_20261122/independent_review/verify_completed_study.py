"""Independently verify a completed frozen study, then replay three fixed IDs.

This script never appends to the study. Replays repeat existing IDs and are
verification costs, not additional independent calibration evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time

# Match the frozen CLI before NumPy/SciPy load their BLAS backends.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_COMMIT = "db5a75bb66a0b28b1d05dacc41b66daf161f9120"
EXPECTED_PROTOCOL_SHA256 = "55dc350ce1e5252f95231113112b02eababaa6976be1ae288ba09407441b4bd9"
# Fixed by the parent before completion; all indices are zero-based.
REPLAY_PAIRS = ((2, 0), (3, 256), (8, 511))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def check_close(actual, expected, name):
    if actual is None or expected is None:
        require(actual is None and expected is None, name + ": missing bound mismatch")
    else:
        require(math.isclose(float(actual), float(expected), rel_tol=2e-13, abs_tol=2e-14), name + ": numerical mismatch")


def lower(count, trials, error):
    return 0.0 if count == 0 else float(beta.ppf(error, count, trials - count + 1))


def upper(count, trials, error):
    return 1.0 if count == trials else float(beta.ppf(1 - error, count + 1, trials - count))


def pointwise_interval(count, trials):
    return [0.0 if count == 0 else float(beta.ppf(0.025, count, trials - count + 1)),
            1.0 if count == trials else float(beta.ppf(0.975, count + 1, trials - count))]


def decode80(value):
    payload = bytes.fromhex(value)
    require(len(payload) == 10, "Exactly ten bytes are required for an 80-position decision")
    return np.array([bool((payload[i // 8] >> (i % 8)) & 1) for i in range(80)], dtype=bool).reshape(4, 20)


def joint_rule(flags, adverse=False):
    require(flags.shape == (4, 20), "Complete four-panel/twenty-rival geometry required")
    return bool(np.all(flags[0]) and np.all(np.count_nonzero(flags[1:], axis=0) >= 2) and not adverse)


def chain(path, genesis):
    raw = Path(path).read_bytes() if Path(path).exists() else b""
    require(not raw or raw.endswith(b"\n"), "Truncated chain: " + str(path))
    result = []
    previous = genesis
    for line in raw.splitlines():
        row = json.loads(line)
        claimed = row.pop("sha256", None)
        require(row.get("previous_sha256") == previous, "Hash-chain predecessor mismatch")
        require(claimed == digest(row), "Hash-chain record mismatch")
        row["sha256"] = claimed
        result.append(row)
        previous = claimed
    return result, previous


def identities_and_ledger(case):
    require(not (case / "SESSION_OPEN.json").exists() and not (case / ".run.lock").exists(),
            "Study is still open; wait for the completed final checkpoint")
    identity = load(case / "source_identity.json")
    claimed = identity.pop("identity_sha256", None)
    require(claimed == digest(identity), "Source identity hash mismatch")
    identity["identity_sha256"] = claimed
    require(identity["source_commit"] == EXPECTED_SOURCE_COMMIT, "Study uses another source anchor")
    require(identity["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256, "Study uses another protocol anchor")
    require(sha(case / "protocol.json") == EXPECTED_PROTOCOL_SHA256, "Case protocol bytes differ")
    require(sha(ROOT / "joint_validation/protocol.json") == EXPECTED_PROTOCOL_SHA256, "Installed protocol bytes differ")
    require(digest(identity["files_sha256"]) == identity["code_id"], "Scientific code-map identity differs")
    for name, expected in identity["files_sha256"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source identity member")
        require(sha(ROOT / relative) == expected, "Installed source bytes differ: " + name)
    installed = {path.relative_to(ROOT).as_posix(): sha(path)
                 for folder in ("methods", "joint_validation") for path in sorted((ROOT / folder).glob("*"))
                 if path.is_file() and path.suffix in (".py", ".cpp")}
    require(installed == identity["files_sha256"], "Installed scientific file inventory differs")
    plan = load(case / "protocol.json")
    require(plan["replicates_per_scenario"] == 512 and len(plan["scenarios"]) == 9, "Unexpected study dimensions")
    require(plan["criteria"]["MC_total_claims"] == 107, "Expected exactly 107 fixed confidence claims")
    require(plan["cell"] == dict(R=10, T=768, B=8191, block_length=32, hac_lags=48, horizon=1, reference_reps=49999),
            "Unexpected validated statistical cell")
    rows, last = chain(case / "ledger.jsonl", claimed)
    events, event_last = chain(case / "events.jsonl", claimed)
    expected = [(scenario, replica) for replica in range(512) for scenario in range(9)]
    require([(row["scenario_index"], row["replicate"]) for row in rows] == expected,
            "Study is incomplete, duplicated or not in the frozen round-robin order")
    for row in rows:
        i, rep = row["scenario_index"], row["replicate"]
        require(row["scenario"] == plan["scenarios"][i]["id"], "Scenario ID mismatch")
        seed = int(np.random.SeedSequence([int(plan["data_seed"]), i, rep]).generate_state(1, dtype=np.uint64)[0])
        require(row["data_seed"] == seed, "Outer seed derivation mismatch")
        truth = joint_rule(decode80(row["true_contrasts"]))
        observed = joint_rule(decode80(row["contrast_pass"]), row["additional_inferiority"])
        require(row["true_global"] == truth, "Population truth topology mismatch")
        require(row["joint_acceptance"] == observed, "Observed decision topology mismatch")
        require(row["global_false_acceptance"] == bool(observed and not truth), "False-acceptance flag mismatch")
        require(truth == (plan["scenarios"][i]["kind"] == "alternative"), "Scenario kind and population truth disagree")
    state = load(case / "state.json")
    require(state["status"] == "completed" and state["completed_replicates"] == len(rows), "Final summary is incomplete")
    require(state["source_commit"] == EXPECTED_SOURCE_COMMIT and state["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256,
            "Final summary source/protocol identity differs")
    require(state["code_id"] == identity["code_id"], "Final summary code-map identity differs")
    return plan, identity, rows, events, state, last, event_last


def verify_bounds(plan, rows, events, state):
    criteria = plan["criteria"]
    confidence_error = criteria["MC_family_confidence_error"] / 107
    claims = []
    all_passed = True
    scenario_checks = []
    for i, scenario in enumerate(plan["scenarios"]):
        records = [row for row in rows if row["scenario_index"] == i]
        n = len(records)
        require(n == 512, "Incomplete scenario")
        saved = state["scenarios"][i]
        require(saved["id"] == scenario["id"] and saved["kind"] == scenario["kind"] and saved["n"] == n,
                "Scenario summary identity/count mismatch")
        false = sum(int(row["global_false_acceptance"]) for row in records)
        accepted = sum(int(row["joint_acceptance"]) for row in records)
        require(saved["global_false_acceptances"] == false and saved["joint_acceptances"] == accepted,
                "Scenario summary event counts mismatch")
        check_close(saved["global_false_acceptance_rate"], false / n, "False-acceptance rate")
        check_close(saved["joint_acceptance_rate"], accepted / n, "Joint-acceptance rate")
        if scenario["kind"] == "null":
            bound = upper(false, n, confidence_error)
            check_close(saved["global_type_I_upper_simultaneous_MC"], bound, "Global type-I bound")
            global_passed = bound <= criteria["global_type_I_ceiling"]
            claims.append(dict(id=scenario["id"] + "/global_type_I", count=false, n=n,
                               direction="upper", bound=bound, threshold=criteria["global_type_I_ceiling"], passed=global_passed))
            require(saved["strong_joint_power_lower_simultaneous_MC"] is None and saved["strong_minimum_80_contrast_power_lower_simultaneous_MC"] is None,
                    "Null scenario mislabeled as an alternative-power claim")
        else:
            bound = lower(accepted, n, confidence_error)
            check_close(saved["strong_joint_power_lower_simultaneous_MC"], bound, "Strong joint-power bound")
            joint_passed = bound >= criteria["strong_joint_power_floor"]
            claims.append(dict(id=scenario["id"] + "/joint_power", count=accepted, n=n,
                               direction="lower", bound=bound, threshold=criteria["strong_joint_power_floor"], passed=joint_passed))
            counts = np.sum(np.stack([decode80(row["contrast_pass"]) for row in records]), axis=0).reshape(-1)
            lower_bounds = []
            for j, count in enumerate(counts):
                bound = lower(int(count), n, confidence_error)
                lower_bounds.append(bound)
                claims.append(dict(id=scenario["id"] + "/contrast_power/" + str(j), count=int(count), n=n,
                                   direction="lower", bound=bound, threshold=criteria["strong_per_contrast_power_floor"],
                                   passed=bound >= criteria["strong_per_contrast_power_floor"]))
            check_close(saved["strong_minimum_80_contrast_power_lower_simultaneous_MC"], min(lower_bounds), "Minimum per-contrast power bound")
            require(saved["global_type_I_upper_simultaneous_MC"] is None, "Alternative mislabeled as a null error claim")
            global_passed = joint_passed and min(lower_bounds) >= criteria["strong_per_contrast_power_floor"]
        coverage_passed = True
        for level in (95, 99):
            for kind in ("rmse_difference", "relative_rmse_gain"):
                label = "CI" + str(level) + "_" + kind
                count = sum(int(row["witness"]["coverage"][label]) for row in records)
                saved_coverage = saved["coverage"][label]
                require(saved_coverage["covered"] == count and saved_coverage["n"] == n, "Coverage counts mismatch")
                check_close(saved_coverage["rate"], count / n, "Coverage rate")
                for actual, expected in zip(saved_coverage["pointwise_CP95"], pointwise_interval(count, n)):
                    check_close(actual, expected, "Diagnostic pointwise CP95")
                if level == 99:
                    bound = lower(count, n, confidence_error)
                    check_close(saved_coverage["lower_simultaneous_MC"], bound, "CI99 coverage bound")
                    passed = bound >= criteria["CI99_coverage_floor"]
                    coverage_passed = coverage_passed and passed
                    claims.append(dict(id=scenario["id"] + "/" + label, count=count, n=n,
                                       direction="lower", bound=bound, threshold=criteria["CI99_coverage_floor"], passed=passed))
        passed = global_passed and coverage_passed
        require(saved["global_criterion_pass"] == global_passed and saved["CI99_floor_criterion_pass"] == coverage_passed and saved["criterion_pass"] == passed,
                "Final scenario criterion decision mismatch")
        all_passed = all_passed and passed
        scenario_checks.append(dict(id=scenario["id"], global_criterion_pass=global_passed,
                                    CI99_floor_criterion_pass=coverage_passed, criterion_pass=passed))
    require(len(claims) == 107 and len({claim["id"] for claim in claims}) == 107, "Wrong simultaneous confidence-claim inventory")
    require(state["accepted"] == all_passed, "Final accepted decision disagrees with independently recalculated criteria")
    charged = sum(float(row["CPU_seconds"]) for row in rows) + sum(float(event.get("CPU_seconds_charged", 0)) for event in events)
    check_close(state["charged_CPU_seconds"], charged, "Charged study CPU cost")
    require(state["CPU_cost_uncertain"] == any(event.get("cost_uncertain", False) for event in events), "Cost uncertainty flag mismatch")
    return claims, scenario_checks, all_passed


def replay_three(plan, identity, rows):
    observed_cohort = dict(python_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
                           numpy=importlib.metadata.version("numpy"), scipy=importlib.metadata.version("scipy"),
                           system=platform.system(), machine=platform.machine(), byteorder=sys.byteorder)
    require(observed_cohort == identity["numerical_cohort"], "Exact replays require the original numerical cohort")
    sys.path.insert(0, str(ROOT))
    from joint_validation.validation import one_replica
    expected = {(row["scenario_index"], row["replicate"]): row for row in rows}
    checks = []
    ignored = {"CPU_seconds", "wall_seconds", "previous_sha256", "sha256"}
    for scenario, replicate in REPLAY_PAIRS:
        recorded = expected[(scenario, replicate)]
        reproduced = one_replica(plan, scenario, replicate)
        scientific_record = {key: value for key, value in recorded.items() if key not in ignored}
        scientific_replay = {key: value for key, value in reproduced.items() if key not in ignored}
        require(scientific_record == scientific_replay, "Exact scientific replay mismatch: " + repr((scenario, replicate)))
        checks.append(dict(scenario_index=scenario, replicate=replicate, scientific_record_equal=True,
                           full_generated_data_sha256=reproduced["full_generated_data_sha256"],
                           full_statistical_reports_sha256=reproduced["full_statistical_reports_sha256"],
                           contrast_pass=reproduced["contrast_pass"], joint_acceptance=reproduced["joint_acceptance"],
                           verification_replay_CPU_seconds=reproduced["CPU_seconds"],
                           verification_replay_wall_seconds=reproduced["wall_seconds"]))
    return checks, observed_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve an existing verification result; select a new output path")
    started = datetime.now(timezone.utc).isoformat()
    wall = time.perf_counter()
    cpu = time.process_time()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    before = {name: sha(args.case / name) for name in ("protocol.json", "source_identity.json", "ledger.jsonl", "events.jsonl", "state.json")}
    plan, identity, rows, events, state, last, event_last = identities_and_ledger(args.case)
    claims, scenarios, accepted = verify_bounds(plan, rows, events, state)
    replays, numerical_cohort = replay_three(plan, identity, rows)
    after = {name: sha(args.case / name) for name in before}
    require(after == before, "Verification altered study files or study changed concurrently")
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(schema_version=1, status="VERIFIED", started_utc=started,
                  finished_utc=datetime.now(timezone.utc).isoformat(), source_commit=EXPECTED_SOURCE_COMMIT,
                  protocol_sha256=EXPECTED_PROTOCOL_SHA256, verifier_sha256=sha(__file__),
                  complete_study_replicates=len(rows), independently_recalculated_confidence_bounds=len(claims),
                  simultaneous_MC_error=plan["criteria"]["MC_family_confidence_error"], total_fixed_claims=107,
                  accepted_matches=True, accepted=accepted, scenario_checks=scenarios, confidence_claims=claims,
                  ledger_last_sha256=last, events_last_sha256=event_last, study_file_sha256=before,
                  study_files_unchanged=True, numerical_cohort=numerical_cohort,
                  prefixed_replay_pairs_zero_based=[list(pair) for pair in REPLAY_PAIRS],
                  verification_replays=replays, additional_unique_study_replicates=0,
                  verification_wall_seconds=time.perf_counter() - wall,
                  verification_parent_CPU_seconds=time.process_time() - cpu,
                  verification_children_CPU_seconds=child_end.ru_utime + child_end.ru_stime - children.ru_utime - children.ru_stime,
                  cost_scope="Verification and the three repeated existing IDs; separate from the frozen study budget and not additional calibration evidence",
                  scientific_scope="Independent arithmetic/integrity check and exact verification replays of a finite-DGP synthetic study; no market power or superiority claim",
                  market_fits=0, market_forwards=0, reserved_market_test_opened=False, R3_accredited=False,
                  AI_assistance_declared=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], complete_study_replicates=len(rows),
                          confidence_bounds=len(claims), accepted=accepted, exact_replays=len(replays),
                          study_files_unchanged=True, output=str(args.output))))


if __name__ == "__main__":
    main()
