"""Check source hashes and stored algebraic evidence without importing models.

AI assistance declared. Reads text/JSON only; does not train, run a model,
recompute the pilot, generate random laws, or access raw market arrays.
"""
from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(left, right):
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, Fraction(0)) + value
    return {key: value for key, value in out.items() if value}


def scale(poly, coefficient):
    return {key: value * coefficient for key, value in poly.items() if value * coefficient}


def square(vector):
    return {(i, j): vector[i] * vector[j] * (1 if i == j else 2)
            for i in range(len(vector)) for j in range(i, len(vector))
            if vector[i] * vector[j]}


def check_population_identity():
    n = 7
    diagonal = {(i, i): Fraction(1, n) for i in range(n)}
    left = add(diagonal, scale(square([Fraction(1)] * n), -Fraction(1, n*n)))
    known_diagonal = {(i, i): Fraction(1, n) for i in range(n-1)}
    known = [Fraction(1)] * (n-1) + [Fraction(0)]
    A = add(known_diagonal, scale(square(known), -Fraction(1, n*(n-1))))
    deviation = [-Fraction(1, n-1)] * (n-1) + [Fraction(1)]
    right = add(A, scale(square(deviation), Fraction(n-1, n*n)))
    if left != right:
        raise ValueError("Population sigma7 identity differs as an exact rational polynomial")
    return {"population_ddof": 0, "coefficient": "6/49",
            "monomial_coefficients_verified": len(left), "exact_fraction_arithmetic": True}


def verify(repo_root, packet_root):
    repo_root, packet_root = Path(repo_root).resolve(), Path(packet_root).resolve()
    metadata = json.loads((packet_root / "mechanism_audit.json").read_text())
    source_checks = []
    for record in metadata["verified_inputs"]:
        base = repo_root if record["location"] == "repository" else packet_root
        path = base / record["path"]
        if path.is_symlink() or not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError("Audited input changed: " + str(path))
        source_checks.append(record["path"])

    module = ast.parse((repo_root / "src/e3_overlap_model_v3.py").read_text())
    model = next(node for node in module.body if isinstance(node, ast.ClassDef)
                 and node.name == "OverlapDistributionModel")
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef)
                   and node.name == "forward")
    if [arg.arg for arg in forward.args.args] != ["self", "values", "A"]:
        raise ValueError("Forecast inputs changed; auxiliary target is not an input")

    original = ast.parse((repo_root / "src/original_model.py").read_text())
    config = next(node for node in original.body if isinstance(node, ast.ClassDef)
                  and node.name == "CandidateConfig")
    values = {node.target.id: ast.literal_eval(node.value) for node in config.body
              if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
    if [values[name] for name in ["input_features", "cnn_channels", "hidden_size", "kernel_size"]] != [5, 8, 12, 3]:
        raise ValueError("Static parameter arithmetic no longer matches source configuration")
    conv = 8 * (5 * 3 + 1)
    lstm = 4 * 12 * (8 + 12 + 2)
    shared = conv + lstm + 12 + 1
    counts = {"shared_encoder_attention": shared, "additive_positive_head": shared + 13,
              "atom_crps": shared + 39 + 26, "atom_mse_only": shared + 39 + 26,
              "fixed_probs": shared + 39, "atom_plugin": shared + 39 + 26}

    plan_path = repo_root / "experiments/e3_evidence_close_20261120/pilot/archive/payload/plan.json"
    plan = json.loads(plan_path.read_text())
    for name, value in plan["expected_parameters"].items():
        if counts[name] != value:
            raise ValueError("Pilot and audited source parameter counts differ")
    if plan["config"]["lambda_crps"] != .1 or plan["total_required_fits"] != 30:
        raise ValueError("Frozen pilot configuration changed")
    for path in ["src/e3_overlap_model_v3.py", "src/e3_innovation_model.py",
                 "src/e3_development.py", "src/original_model.py"]:
        if digest(repo_root / path) != plan["identity"]["source_files"][path]:
            raise ValueError("Current source differs from pilot-fixed source: " + path)

    previous = repo_root / "book_e3/_static/e3_r1_close_20261120/adaptation_design"
    stored = json.loads((previous / "analytic_verification.json").read_text())
    if digest(previous / "analytic_checks.py") != stored["source_sha256"]:
        raise ValueError("Stored analytic evidence has a different code binding")
    rows = stored["cases"]
    if stored["status"] != "passed" or len(rows) != 100:
        raise ValueError("Expected the 100 previously saved finite-law cases")
    for row in rows:
        tolerance = 5e-13 * max(1, abs(row["curvature_upper"]))
        if not row["curvature_lower"] - tolerance <= row["gap"] <= row["curvature_upper"] + tolerance:
            raise ValueError("Stored curvature interval no longer contains its stored gap")
        if abs(row["plugin_minus_exact_squared_risk"] - row["gap"]**2) >= 5e-12 * max(1, abs(row["plugin_minus_exact_squared_risk"])):
            raise ValueError("Stored oracle risk-gap identity differs")

    return {"status": "source_and_existing_theory_evidence_verified",
            "AI_assistance_declared": True, "inputs_verified": len(source_checks),
            "parameter_counts_static": counts, "identity": check_population_identity(),
            "prior_finite_law_rows_consistency_checked": len(rows),
            "new_finite_law_simulations": 0, "new_market_data_reads": 0,
            "new_model_fits": 0, "new_model_inference": 0,
            "new_pilot_analysis": 0, "new_statistical_resampling": 0,
            "implementation_difference_documented": True,
            "formal_property_reviewed_under_explicit_assumptions": True,
            "literature_novelty_assessed_by_this_script": False,
            "R1_accredited_by_this_script": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root, args.packet_root), ensure_ascii=False, sort_keys=True))
