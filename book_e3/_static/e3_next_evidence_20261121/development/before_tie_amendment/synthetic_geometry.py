"""Exact finite-law score/gradient diagnostic, fixed before execution.

This is a CPU synthetic mechanism diagnostic. It creates no market forecast,
optimizer or fitted network and cannot decide predictive utility or R1/R3.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from outcome_crps_v4 import (C, VERSION, VARIANTS, outcome_loss_components,
                             verify_base_source)

PLAN_SHA256 = "afb7362ec81b09e86a1e0fd29f3a222d4c2d9bcf7b7e464a918520824989c4d5"
MIN_INCREMENT = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_head(atoms, probabilities, u_scale):
    atoms = torch.as_tensor(atoms, dtype=torch.float64)
    increments = torch.cat((atoms[:1], atoms[1:] - atoms[:-1])) / u_scale - MIN_INCREMENT
    if not bool((increments > 0).all()):
        raise ValueError("Synthetic atom increments violate the fixed architecture")
    logits = torch.tensor([math.log(probabilities[i] / probabilities[2])
                           for i in range(2)], dtype=torch.float64)
    return torch.cat((torch.log(torch.expm1(increments)), logits)).requires_grad_()


def outputs_from_head(theta, A, u_scale, samples):
    atoms = u_scale * (MIN_INCREMENT + F.softplus(theta[:3])).cumsum(-1)
    probabilities = torch.softmax(torch.cat((theta[3:], theta.new_zeros(1))), -1)
    sigma_atoms = torch.hypot(theta.new_full((3,), math.sqrt(A)), math.sqrt(C) * atoms)
    prediction = (probabilities * sigma_atoms).sum()
    return {"atoms": atoms.expand(samples, -1),
            "probabilities": probabilities.expand(samples, -1),
            "sigma_atoms": sigma_atoms.expand(samples, -1),
            "prediction": prediction.expand(samples)}


def finite_targets(law, A):
    # Both fixed probability vectors are exact multiples of 1/20. These are
    # deterministic quadrature weights encoded as repeated rows, never draws.
    counts = [round(20 * p) for p in law["p"]]
    if sum(counts) != 20 or any(abs(n / 20 - p) > 1e-14 for n, p in zip(counts, law["p"])):
        raise ValueError("The fixed finite-law exact quadrature changed")
    U = torch.tensor([u for u, count in zip(law["atoms"], counts)
                      for _ in range(count)], dtype=torch.float64)
    Y = torch.hypot(torch.full_like(U, math.sqrt(A)), math.sqrt(C) * U)
    return U, Y, float(U.mean()), float(Y.mean())


def cosine(left, right, zero=1e-12):
    l, r = float(torch.linalg.vector_norm(left)), float(torch.linalg.vector_norm(right))
    if l <= zero or r <= zero:
        return None
    return max(-1.0, min(1.0, float(torch.dot(left, right)) / (l * r)))


def evaluate_case(A, law, atoms, probabilities):
    U, Y, us, ys = finite_targets(law, A)
    theta = raw_head(atoms, probabilities, us)
    output = outputs_from_head(theta, A, us, len(U))
    pieces = outcome_loss_components(output, Y, U, y_scale=ys, u_scale=us)
    keys = ("mse_scaled", "crps_u", "crps_sigma", "crps_sigma_a0matched")
    grads = {key: torch.autograd.grad(pieces[key], theta, retain_graph=True)[0] for key in keys}
    point = grads["mse_scaled"]
    row = {"A": A, "law": law["id"], "model_atoms": json.dumps(atoms),
           "model_probabilities": json.dumps(probabilities), "u_scale": us, "y_scale": ys,
           "sigma_default_to_A0matched_aux_gradient_norm_ratio_expected": math.sqrt(C) * us / ys}
    row.update({key: float(pieces[key].detach()) for key in keys})
    row["mse_gradient_norm"] = float(torch.linalg.vector_norm(point))
    for suffix, key in (("U", "crps_u"), ("sigma", "crps_sigma"),
                        ("sigma_a0matched", "crps_sigma_a0matched")):
        auxiliary = grads[key]
        joint = point + 0.1 * auxiliary
        row[suffix + "_aux_gradient_norm"] = float(torch.linalg.vector_norm(auxiliary))
        row[suffix + "_aux_cosine_with_point"] = cosine(auxiliary, point)
        row[suffix + "_joint_cosine_with_point"] = cosine(joint, point)
        row[suffix + "_point_risk_slope_along_negative_aux_gradient"] = -float(torch.dot(point, auxiliary))
        row[suffix + "_point_risk_slope_along_negative_joint_gradient"] = -float(torch.dot(point, joint))
    row["U_vs_sigma_aux_direction_cosine"] = cosine(grads["crps_u"], grads["crps_sigma"])
    row["sigma_vs_A0matched_aux_direction_cosine"] = cosine(grads["crps_sigma"], grads["crps_sigma_a0matched"])
    row["U_sigma_aux_gradient_max_abs_difference"] = float((grads["crps_u"] - grads["crps_sigma"]).abs().max())
    row["U_sigma_aux_score_abs_difference"] = abs(row["crps_u"] - row["crps_sigma"])
    return row


def summarize(rows, tolerance):
    result = {"scenarios": len(rows), "by_A": {}}
    for A in sorted({row["A"] for row in rows}):
        group = [row for row in rows if row["A"] == A]
        valid = [row for row in group if row["U_aux_cosine_with_point"] is not None
                 and row["sigma_aux_cosine_with_point"] is not None]
        result["by_A"][str(A)] = {
            "scenarios": len(group), "defined_point_auxiliary_alignment": len(valid),
            "zero_gradient_cases_excluded_from_alignment": len(group) - len(valid),
            "V4_auxiliary_alignment_better": sum(row["sigma_aux_cosine_with_point"] - row["U_aux_cosine_with_point"] > tolerance for row in valid),
            "V4_auxiliary_alignment_worse": sum(row["sigma_aux_cosine_with_point"] - row["U_aux_cosine_with_point"] < -tolerance for row in valid),
            "V4_auxiliary_alignment_equal_within_tolerance": sum(abs(row["sigma_aux_cosine_with_point"] - row["U_aux_cosine_with_point"]) <= tolerance for row in valid),
            "noncollinear_U_vs_sigma_auxiliary_directions": sum(row["U_vs_sigma_aux_direction_cosine"] is not None and 1 - row["U_vs_sigma_aux_direction_cosine"] > tolerance for row in group),
            "normalization_only_direction_differences": sum(row["sigma_vs_A0matched_aux_direction_cosine"] is not None and 1 - row["sigma_vs_A0matched_aux_direction_cosine"] > tolerance for row in group),
            "U_joint_locally_increases_point_risk": sum(row["U_joint_cosine_with_point"] is not None and row["U_joint_cosine_with_point"] < -tolerance for row in group),
            "V4_joint_locally_increases_point_risk": sum(row["sigma_joint_cosine_with_point"] is not None and row["sigma_joint_cosine_with_point"] < -tolerance for row in group),
            "max_auxiliary_gradient_abs_difference": max(row["U_sigma_aux_gradient_max_abs_difference"] for row in group),
            "max_auxiliary_score_abs_difference": max(row["U_sigma_aux_score_abs_difference"] for row in group),
        }
    result["any_nonzero_A_direction_difference"] = any(row["A"] > 0
        and row["U_vs_sigma_aux_direction_cosine"] is not None
        and 1 - row["U_vs_sigma_aux_direction_cosine"] > tolerance for row in rows)
    result["normalization_only_directions_collinear"] = all(
        row["sigma_vs_A0matched_aux_direction_cosine"] is None
        or 1 - row["sigma_vs_A0matched_aux_direction_cosine"] <= tolerance for row in rows)
    result["global_better_alignment_cases"] = sum(x["V4_auxiliary_alignment_better"] for x in result["by_A"].values())
    result["global_worse_alignment_cases"] = sum(x["V4_auxiliary_alignment_worse"] for x in result["by_A"].values())
    result["global_equal_alignment_cases"] = sum(x["V4_auxiliary_alignment_equal_within_tolerance"] for x in result["by_A"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("plan_before_results.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    if sha(args.plan) != PLAN_SHA256:
        raise ValueError("Prior fixed synthetic plan changed")
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        raise ValueError("Existing diagnostic output preserved; use a fresh directory")
    args.output.mkdir(parents=True)
    verify_base_source()
    torch.set_num_threads(plan["synthetic_runtime"]["max_threads"])
    grid = plan["fixed_synthetic_grid"]
    rows = []
    for A in grid["A"]:
        for law in grid["true_laws"]:
            for atoms in grid["model_atoms"]:
                for probabilities in grid["model_p"]:
                    if time.monotonic() - clock > plan["synthetic_runtime"]["maximum_seconds"]:
                        raise RuntimeError("Fixed CPU diagnostic budget exceeded; no model fitting occurred")
                    row = evaluate_case(A, law, atoms, probabilities)
                    row = {"scenario_id": "S" + str(len(rows) + 1).zfill(3), **row}
                    rows.append(row)
    if len(rows) != grid["scenarios"]:
        raise ValueError("Fixed diagnostic scenario count changed")
    with (args.output / "all_scenarios.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows, grid["cosine_tolerance"])
    a0 = summary["by_A"]["0.0"]
    if a0["max_auxiliary_gradient_abs_difference"] > 1e-12 or a0["max_auxiliary_score_abs_difference"] > 1e-12:
        raise AssertionError("The declared A0 normalized-score identity failed")
    result = {"status": "executed_fixed_finite_law_CPU_gradient_diagnostic",
              "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "version": VERSION, "prior_plan_SHA256": PLAN_SHA256,
              "plan_preceded_execution": plan["created_at_utc"] < started,
              "script_sha256": sha(__file__), "module_sha256": sha(Path(__file__).with_name("outcome_crps_v4.py")),
              "base_source_sha256": verify_base_source(),
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU", "threads": torch.get_num_threads()},
              "wall_seconds": time.monotonic() - clock, "summary": summary,
              "interpretation": "The fixed synthetic grid can establish local score/gradient geometry differences and adverse cases. Neither a favorable count nor the A0 identity selects a market recipe or proves utility.",
              "future_market_pilot_authorized_by_this_result": False,
              "original_V3_NO_GO_preserved": True, "R1_inherited": False, "R3_accredited": False,
              "optimizer_steps": 0, "neural_networks_fitted": 0, "market_model_fits": 0,
              "market_predictions": 0, "market_inputs_opened": 0, "random_simulation_draws": 0,
              "synthetic_gradient_scenarios": len(rows), "AI_assistance_declared": True,
              "outputs": {"all_scenarios.csv": {"bytes": (args.output / "all_scenarios.csv").stat().st_size,
                                                      "sha256": sha(args.output / "all_scenarios.csv")}}}
    (args.output / "diagnostic_receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
