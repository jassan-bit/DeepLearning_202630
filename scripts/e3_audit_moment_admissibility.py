"""Two scalar counterexamples for the moment-head admissibility review.

This audit uses known artificial laws and one gradient step per stated step size.
It instantiates no neural network, reads no market arrays/checkpoints and performs
no optimizer search. Existing scalar head functions are reused without changing
the previous proposal or its consistency theorem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e3_verify_moment_head import C, moment_bounds, moment_head, quasi_scores

OUTPUT = ROOT / "experiments/e3_dictamen_momentos_20261116/counterexamples.json"
LAMBDA = 0.1
COORDINATES = ("a", "b", "d")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def report(coordinates):
    """A=0 and s_u=1: three scalar readouts, not a neural module."""
    a, b, d = coordinates
    return moment_head(0.0, softplus(a), b, d)


def mean(coordinates):
    return report(coordinates)["prediction"]


def auxiliary(coordinates):
    head = report(coordinates)
    scores = quasi_scores(1.0, head["m"], head["q"])
    return LAMBDA * (scores["m"] + scores["q"])


def mse(coordinates):
    # V=1 almost surely, so the true mean is sqrt(c) and its variance is zero.
    # s_y=E[Y]=sqrt(c), exactly the proposed fit-only scale in this synthetic law.
    return ((mean(coordinates) - math.sqrt(C)) / math.sqrt(C)) ** 2


def finite_difference(function, coordinates, step=1e-6):
    result = []
    for index in range(3):
        lower, upper = list(coordinates), list(coordinates)
        lower[index] -= step
        upper[index] += step
        result.append((function(upper) - function(lower)) / (2 * step))
    return result


def mapping(values):
    return dict(zip(COORDINATES, values))


def fixed_wrong_moments():
    # The SAME incorrect moment reports are held fixed in both alternatives.
    # This refutes an automatic MSE-improvement claim from estimated bounds;
    # it does not assert that a trained model must produce these wrong moments.
    m, q = 4.0, 32.0
    bounds = moment_bounds(0.0, m, q)
    lower_head = moment_head(0.0, m, 1.0, 0.0)
    target = direct_mean = math.sqrt(C)
    constrained_mean = lower_head["prediction"]
    scores_direct = quasi_scores(1.0, m, q)
    scores_envelope = quasi_scores(1.0, lower_head["m"], lower_head["q"])
    require(scores_direct == scores_envelope, "The two alternatives must have identical moment scores")
    require(math.isclose(constrained_mean, math.sqrt(2 * C), rel_tol=2e-14),
            "The lower endpoint must equal sqrt(2c)")
    require(constrained_mean > target, "The estimated mean region must exclude the true mean")
    excess = (constrained_mean - target) ** 2
    require(math.isclose(excess, C * (math.sqrt(2.0) - 1.0) ** 2, rel_tol=2e-14),
            "The exact positive excess MSE did not reproduce")
    return {
        "law": "V=1 almost surely; U=1; A=0; c=6/49",
        "true_moments": {"m0": 1.0, "q0": 1.0},
        "true_mean_Y": target,
        "fixed_estimated_moments_both_controls": {"m": m, "q": q},
        "estimated_envelope": {"L": bounds["L"], "H_unattained_supremum": bounds["H"]},
        "direct_mean": direct_mean,
        "best_envelope_mean_with_these_reports": constrained_mean,
        "moment_scores_direct": scores_direct,
        "moment_scores_envelope": scores_envelope,
        "identical_auxiliary_score": True,
        "direct_raw_expected_MSE": 0.0,
        "envelope_minimum_raw_expected_MSE": excess,
        "exact_excess_MSE_formula": "c*(sqrt(2)-1)^2 > 0",
        "direct_normalized_expected_MSE": 0.0,
        "envelope_minimum_normalized_expected_MSE": excess / C,
        "interpretation": "Estimated moment bounds can exclude the true mean; keeping the SAME moment scores does not guarantee lower MSE.",
        "limit": "The incorrect reports are held fixed to isolate the output constraint; this is not an optimization result for a trained neural network."
    }


def same_head_gradient_step():
    # This is precisely the proposed constant initialization for this law:
    # fit E[V]=1, s_u=E[U]=1, b=sqrt(8), d=1, a=inverse_softplus(1).
    initial = (math.log(math.expm1(1.0)), math.sqrt(8.0), 1.0)
    head = report(initial)
    g = 1.0 - math.exp(-1.0)
    root_c = math.sqrt(C)
    mean_gradient = (root_c * g / 3.0, -root_c * math.sqrt(8.0) / 54.0, root_c / 3.0)
    aux_gradient = (16.0 * LAMBDA * g / 9.0, 16.0 * math.sqrt(8.0) * LAMBDA / 81.0, 0.0)
    mse_gradient = tuple(-2.0 * value / (3.0 * root_c) for value in mean_gradient)
    joint_gradient = tuple(first + second for first, second in zip(mse_gradient, aux_gradient))
    require(math.isclose(head["m"], 1.0, rel_tol=2e-14)
            and math.isclose(head["q"], 9.0, rel_tol=2e-14)
            and math.isclose(head["prediction"], 2.0 * root_c / 3.0, rel_tol=2e-14),
            "The declared initialization did not reproduce")

    derivative_checks = {}
    for name, function, analytic in (("mean_Y", mean, mean_gradient),
                                     ("auxiliary_score", auxiliary, aux_gradient),
                                     ("normalized_MSE", mse, mse_gradient)):
        numerical = finite_difference(function, initial)
        require(all(math.isclose(a, n, rel_tol=2e-7, abs_tol=2e-10)
                    for a, n in zip(analytic, numerical)), "Central derivatives disagree: " + name)
        derivative_checks[name] = {
            "analytic": mapping(analytic), "central_difference": mapping(numerical),
            "step": 1e-6, "maximum_absolute_error": max(abs(a - n) for a, n in zip(analytic, numerical))
        }

    coefficient = 16.0 * g * g / 27.0 - 64.0 / 2187.0
    directional = sum(a * b for a, b in zip(mean_gradient, aux_gradient))
    require(coefficient > 0.0 and math.isclose(directional, LAMBDA * root_c * coefficient, rel_tol=2e-14),
            "The exact directional derivative must be positive")
    updates = []
    for eta in (1e-6, 1e-5, 1e-4):
        mean_only = tuple(value - eta * gradient for value, gradient in zip(initial, mse_gradient))
        joint = tuple(value - eta * gradient for value, gradient in zip(initial, joint_gradient))
        prediction_only, prediction_joint = mean(mean_only), mean(joint)
        mse_only, mse_joint = mse(mean_only), mse(joint)
        difference = prediction_joint - prediction_only
        leading = -eta * directional
        require(prediction_joint < prediction_only < root_c,
                "The joint scalar step should remain further below the true mean")
        require(mse_joint > mse_only, "Joint scalar-step MSE should exceed same-head mean-only MSE")
        require(math.isclose(difference / eta, -directional, rel_tol=3e-4, abs_tol=2e-9),
                "The one-step prediction difference disagrees with its first-order expansion")
        updates.append({
            "eta": eta,
            "reset_to_same_initial_coordinates": True,
            "mean_only_coordinates_after_one_step": mapping(mean_only),
            "joint_coordinates_after_one_step": mapping(joint),
            "mean_only_prediction": prediction_only, "joint_prediction": prediction_joint,
            "mean_only_normalized_MSE": mse_only, "joint_normalized_MSE": mse_joint,
            "joint_minus_mean_only_prediction": difference,
            "first_order_prediction_difference": leading,
            "joint_minus_mean_only_normalized_MSE": mse_joint - mse_only,
            "joint_MSE_is_larger": True
        })
    return {
        "law": "V=1 almost surely; U=1; A=0; c=6/49",
        "true_mean_Y": root_c,
        "normalization": {"s_u": 1.0, "s_y": root_c, "source": "exact fit-population means for this artificial law"},
        "same_head_controls": ["moment head + normalized MSE only", "SAME moment head + normalized MSE + two moment scores"],
        "initial_coordinates_both_controls": mapping(initial),
        "initial_m": head["m"], "initial_q": head["q"], "initial_rho": head["rho"],
        "initial_prediction": head["prediction"], "initial_normalized_MSE": mse(initial),
        "lambda_m": LAMBDA, "lambda_q": LAMBDA,
        "derivative_checks": derivative_checks,
        "exact_aux_directional_derivative_formula": "D=lambda*sqrt(c)*[(16/27)*(1-exp(-1))^2-64/2187] > 0",
        "coefficient_without_lambda_sqrt_c": coefficient,
        "aux_directional_derivative_of_prediction": directional,
        "first_order_difference_formula": "mu_joint_after_SGD - mu_MSE_after_SGD = -eta*D + O(eta^2)",
        "independent_one_step_comparisons": updates,
        "encoder": "fixed; only the three scalar readout coordinates a,b,d are updated",
        "optimizer": "plain scalar SGD; one step from identical initialization for each eta and each objective",
        "interpretation": "Auxiliary gradients change ancestors of the mean. Detaching them only from an encoder cannot guarantee the same mean trajectory as this same head trained on MSE alone.",
        "limits": [
            "These are six independent scalar updates, not a multi-step neural training or an Adam campaign.",
            "A first idealized Adam step can coincide when it depends only on coordinate signs; no Adam trajectory was simulated or inferred.",
            "The example refutes an automatic optimization-fidelity or universal one-step MSE-improvement claim, not conditional-risk consistency in a sufficiently rich class.",
            "The artificial deterministic law has zero irreducible conditional variance. This is not a claim about financial data or the source of the v2 pilot result."
        ]
    }


def generate():
    source = ROOT / "scripts/e3_verify_moment_head.py"
    return {
        "schema": "e3-moment-admissibility-scalar-counterexamples-v1",
        "date_local": "2026-11-16", "timezone": "America/Bogota",
        "source_verifier": {"path": "scripts/e3_verify_moment_head.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "modified_by_this_audit": False},
        "operations": {"artificial_laws": 1, "scalar_SGD_updates": 6,
                       "independent_step_sizes": 3, "steps_per_update": 1,
                       "neural_networks_instantiated": 0, "neural_networks_trained": 0,
                       "market_rows_read": 0, "market_fits": 0, "market_inferences": 0,
                       "NPZ_or_checkpoint_opened": False, "Adam_campaign": False,
                       "optimizer_minimizations": 0, "multi_step_optimizer_search": False},
        "fixed_wrong_moments": fixed_wrong_moments(),
        "same_head_gradient_step": same_head_gradient_step(),
        "claims": {"previous_consistency_theorem_invalidated": False,
                   "market_failure_explained": False, "novelty_decided_by_counterexample": False,
                   "general_MSE_improvement_guaranteed": False,
                   "Faithful_optimization_theorem_inherited": False},
        "check_tolerance": {"float_rtol": 2e-9, "float_atol": 2e-12}
    }


def compare(actual, expected, path="counterexamples"):
    if type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=2e-9, abs_tol=2e-12), path + ": numeric value changed")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), path + ": keys changed")
        for key, value in expected.items():
            compare(actual[key], value, path + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ": list length changed")
        for index, value in enumerate(expected):
            compare(actual[index], value, path + f"[{index}]")
    else:
        require(type(actual) is type(expected) and actual == expected, path + ": metadata changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        expected = generate()
        if args.check:
            compare(json.loads(args.output.read_text()), expected)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(expected, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"status": "verified" if args.check else "generated", "scalar_SGD_updates": 6,
                          "neural_networks_instantiated": 0, "market_rows_read": 0, "market_fits": 0,
                          "market_inferences": 0, "Adam_campaign": False, "artifact": str(args.output.relative_to(ROOT))
                          if args.output.is_relative_to(ROOT) else str(args.output)}))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, "Moment admissibility audit failed: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
