"""Synthetic scalar verification of the moment-envelope proposal; no NN or market.

Population risks are integrated and minimized as scalar mathematical examples.
They are not neural training, evidence of novelty, or an empirical benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import scipy
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/e3_mecanismo_momentos_20261116/verification"
C = 6.0 / 49.0
MEAN_ABS_T5 = 4 * math.sqrt(5) / (3 * math.pi)
LOG_HALF_T5_CONSTANT = math.log(16 / (3 * math.pi * math.sqrt(5)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(value, name, positive=False):
    require(type(value) in (float, int) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), name + " must be finite and " + ("positive" if positive else "nonnegative"))
    return float(value)


def moment_bounds(A, m, q):
    A, m, q = scalar(A, "A"), scalar(m, "m"), scalar(q, "q")
    require(q >= m * m and (q > 0 if m > 0 else q == 0),
            "Require q>=m^2 and representably positive q for m>0; m=0 is possible only with q=0")
    if m == 0:
        return {"L": math.sqrt(A), "H": math.sqrt(A), "degenerate_zero": True}
    a, t = math.sqrt(A), math.sqrt(A + C * (q / m))
    lower = a + C * m / (t + a)
    upper = math.sqrt(A + C * m)
    require(math.isfinite(lower) and math.isfinite(upper), "Unrepresentable moment bounds")
    return {"L": lower, "H": upper, "degenerate_zero": False}


def moment_head(A, m, b, d):
    A, m = scalar(A, "A"), scalar(m, "m", positive=True)
    require(type(b) in (float, int) and type(d) in (float, int) and math.isfinite(b) and math.isfinite(d),
            "b and d must be finite real numbers")
    q = m * m * (1 + b * b)
    require(math.isfinite(q), "Moment parameterization overflow")
    square = d * d
    # Algebraically identical rational form avoids inf/inf. In floating point,
    # a large finite d can still round rho to 1; this is exposed, not clipped.
    rho = square / (1 + square) if math.isfinite(square) else 1 / (1 + (1 / d) ** 2)
    bounds = moment_bounds(A, m, q)
    prediction = bounds["L"] + rho * (bounds["H"] - bounds["L"])
    return {**bounds, "m": m, "q": q, "rho": rho, "prediction": prediction,
            "rho_rounded_to_one": rho == 1.0}


def head_gradients(A, m, b, d):
    """Analytic derivatives in coordinates (A,m,b,d); A=0 may be singular."""
    result = moment_head(A, m, b, d)
    a, t, H, rho = math.sqrt(A), math.sqrt(A + C * m * (1 + b * b)), result["H"], result["rho"]
    gap = result["H"] - result["L"]
    p = 1 / (1 + b * b)
    lower_A = (1 - p) / (2 * a) + p / (2 * t) if A > 0 else (math.inf if b != 0 else 1 / (2 * t))
    lower_b = -b / t * (C * m / (t + a)) ** 2
    if abs(d) <= 1:
        denominator = 1 + d * d
        rho_derivative = 2 * d / denominator / denominator
    else:
        inverse = 1 / d
        rho_derivative = 2 * inverse * inverse * inverse / (1 + inverse * inverse) ** 2
    # With finite real d, 1-rho remains strictly positive even when its
    # floating representation rounds to zero. Preserve the true singularity
    # at A=0,b!=0 instead of evaluating the indeterminate float product 0*inf.
    derivative_A = math.inf if A == 0 and b != 0 else (1 - rho) * lower_A + rho / (2 * H)
    return {"A": derivative_A,
            "m": (1 - rho) * C / (2 * t) + rho * C / (2 * H),
            "b": (1 - rho) * lower_b, "d": rho_derivative * gap}


def quasi_scores(V, m, q):
    V, m, q = scalar(V, "V"), scalar(m, "m", positive=True), scalar(q, "q", positive=True)
    result = {"m": math.log(m) + V / m, "q": math.log(q) + V * V / q}
    require(all(math.isfinite(value) for value in result.values()), "Unrepresentable quasi score")
    return result


def moment_excess_risk(prediction, m, q, *, mean_y, true_m, true_q, y_scale=1., lambda_m=.1, lambda_q=.1):
    """Expected excess relative to the true moments/mean; not raw MSE."""
    prediction, m, q = scalar(prediction, "prediction"), scalar(m, "m", True), scalar(q, "q", True)
    true_m, true_q = scalar(true_m, "true_m", True), scalar(true_q, "true_q", True)
    mean_y, y_scale = scalar(mean_y, "mean_y"), scalar(y_scale, "y_scale", True)
    lambda_m, lambda_q = scalar(lambda_m, "lambda_m", True), scalar(lambda_q, "lambda_q", True)
    point = ((prediction - mean_y) / y_scale) ** 2
    # log(z)+1/z-1 is the expected score difference, independent of unit constants.
    dm = math.log(m / true_m) + true_m / m - 1
    dq = math.log(q / true_q) + true_q / q - 1
    return {"point_excess_MSE_scaled": point, "m_score_excess": dm, "q_score_excess": dq,
            "total_excess_risk": point + lambda_m * dm + lambda_q * dq}


def upper_approach(A, m, q, R):
    bounds = moment_bounds(A, m, q)
    m, R = scalar(m, "m", True), scalar(R, "R", True)
    require(q > m * m and R >= q / m, "The nondegenerate extremal sequence needs q>m^2 and R>=q/m")
    variance = q - m * m
    a = m - variance / (R - m)
    probability = variance / ((R - m) ** 2 + variance)
    predicted = (1 - probability) * math.sqrt(A + C * a) + probability * math.sqrt(A + C * R)
    return {"a_R": a, "R": R, "probability_R": probability,
        "mean_V": a + probability * (R - a),
        "second_moment_V": a * a + probability * (R * R - a * a),
        "mean_Y": predicted, "gap_to_H": bounds["H"] - predicted}


@dataclass(frozen=True)
class SyntheticLaw:
    name: str
    mean_u: float
    m: float
    q: float
    eighth_u_finite: bool
    atoms: tuple = ()
    family: str = ""

    def expectation(self, function):
        if self.atoms:
            return math.fsum(probability * function(value) for value, probability in self.atoms)
        if self.family == "uniform":
            result, error = quad(function, 0, 1, epsabs=2e-12, epsrel=2e-12, limit=300)
        elif self.family == "exponential":
            result, error = quad(lambda u: function(u) * math.exp(-u), 0, math.inf,
                                 epsabs=2e-12, epsrel=2e-12, limit=300)
        elif self.family == "lognormal":
            def integrand(u):
                if u == 0:
                    return 0.
                z = (math.log(u) + .5) / .5
                return function(u) * math.exp(-.5 * z * z) / (u * .5 * math.sqrt(2 * math.pi))
            result, error = quad(integrand, 0, math.inf, epsabs=2e-12, epsrel=2e-12, limit=300)
        elif self.family == "pareto":
            result, error = quad(lambda u: function(u) * 5 * u ** -6, 1, math.inf,
                                 epsabs=2e-12, epsrel=2e-12, limit=300)
        else:
            raise ValueError("Unknown synthetic distribution")
        require(math.isfinite(result) and error <= 2e-9 * max(1., abs(result)), "Independent integral did not converge")
        return result


LAWS = (SyntheticLaw("Uniforme U[0,1]", .5, 1 / 3, 1 / 5, True, family="uniform"),
        SyntheticLaw("Exponencial U(1)", 1., 2., 24., True, family="exponential"),
        SyntheticLaw("U en {0,2}; P(2)=1/4", .5, 1., 4., True, atoms=((0., .75), (2., .25))),
        SyntheticLaw("Lognormal U(-0.5,0.5)", math.exp(-.375), math.exp(-.5), 1., True, family="lognormal"),
        SyntheticLaw("Pareto U: mínimo 1, alfa 5", 1.25, 5 / 3, 5., False, family="pareto"))


def expected_normalized_t5_nll(law, s):
    s = scalar(s, "s", True)
    return math.log(s / law.mean_u) - LOG_HALF_T5_CONSTANT + 3 * law.expectation(
        lambda u: math.log1p(u * u / (5 * s * s)))


def expected_t5_log_scale_derivative(law, s):
    s = scalar(s, "s", True)
    return 1 - 6 * law.expectation(lambda u: u * u / (5 * s * s + u * u))


def fixed_t_conflict(law, lambda_nll=.1):
    """A=0; fixed unit means match V2 normalization; only scalar risk minimized."""
    point_s = law.mean_u / MEAN_ABS_T5
    mean_y, y_scale, u_scale = math.sqrt(C) * law.mean_u, math.sqrt(C) * law.mean_u, law.mean_u
    variance_y = C * (law.m - law.mean_u ** 2)

    def risk_at(log_s, auxiliary_weight):
        s = math.exp(log_s)
        excess_mse_scaled = (MEAN_ABS_T5 * s / law.mean_u - 1) ** 2
        return excess_mse_scaled + auxiliary_weight * expected_normalized_t5_nll(law, s)

    joint = minimize_scalar(lambda x: risk_at(x, lambda_nll), bounds=(-8., 8.), method="bounded",
                            options={"xatol": 1e-13, "maxiter": 500})
    auxiliary = minimize_scalar(lambda x: expected_normalized_t5_nll(law, math.exp(x)),
                                bounds=(-8., 8.), method="bounded", options={"xatol": 1e-13, "maxiter": 500})
    require(joint.success and auxiliary.success, "Synthetic scalar population-risk optimization failed")
    joint_s = math.exp(joint.x)
    excess = (MEAN_ABS_T5 * joint_s / law.mean_u - 1) ** 2
    return {"distribution": law.name, "A": 0., "c": C, "lambda_nll": lambda_nll,
        "u_scale_fixed_mean": u_scale, "y_scale_fixed_mean": y_scale,
        "true_mean_sigma": mean_y, "point_scale": point_s, "point_scale_squared": point_s ** 2,
        "auxiliary_only_scale": math.exp(auxiliary.x), "auxiliary_only_scale_squared": math.exp(2 * auxiliary.x),
        "joint_scale": joint_s, "joint_scale_squared": joint_s ** 2,
        "NLL_log_scale_derivative_at_point": expected_t5_log_scale_derivative(law, point_s),
        "joint_log_scale_derivative_at_point": lambda_nll * expected_t5_log_scale_derivative(law, point_s),
        "minimum_raw_expected_MSE_original": variance_y,
        "minimum_raw_expected_MSE_scaled": variance_y / y_scale ** 2,
        "joint_excess_MSE_scaled": excess, "joint_raw_expected_MSE_scaled": variance_y / y_scale ** 2 + excess,
        "point_expected_NLL_normalized": expected_normalized_t5_nll(law, point_s),
        "joint_expected_NLL_normalized": expected_normalized_t5_nll(law, joint_s),
        "point_joint_risk_minus_constant_variance": risk_at(math.log(point_s), lambda_nll),
        "optimized_joint_risk_minus_constant_variance": float(joint.fun),
        "interpretation": "expected NLL contrast under misspecification; not KL between an atomic law and a continuous density" if law.atoms else
                          "continuous positive distribution: expected NLL contrast under misspecification",
        "optimizer": "SciPy bounded scalar optimization of synthetic integrated population risk"}


def _finite_difference(function, values, index):
    step = 1e-5 * max(1., abs(values[index]))
    first, second = list(values), list(values)
    first[index] -= step
    second[index] += step
    return (function(*second) - function(*first)) / (2 * step)


def verify_all():
    bounds_reports, homogeneity, gradients, consistency = [], [], [], []
    for law in LAWS:
        numerical_m = law.expectation(lambda u: u * u)
        numerical_q = law.expectation(lambda u: u ** 4)
        require(math.isclose(numerical_m, law.m, rel_tol=2e-10) and math.isclose(numerical_q, law.q, rel_tol=2e-10),
                "Independent integrated moments disagree with their analytic values")
        for A in (0., .01, .5, 4.):
            bounds = moment_bounds(A, law.m, law.q)
            mean = law.expectation(lambda u: math.sqrt(A + C * u * u))
            require(bounds["L"] - 2e-11 <= mean <= bounds["H"] + 2e-11, "Independent law violates the moment bounds")
            gap = bounds["H"] - bounds["L"]
            rho = (mean - bounds["L"]) / gap
            # A few last-bit integration errors at an attained lower endpoint
            # are reported as tolerance, never silently used to alter a score.
            require(-2e-11 <= rho < 1 + 2e-11, "True mean is outside the representable envelope")
            report = {"distribution": law.name, "A": A, "m": law.m, "q": law.q,
                "L": bounds["L"], "independent_mean_Y": mean, "H": bounds["H"],
                "rho_true_numeric": rho, "E_U4_finite": True, "E_U8_finite": law.eighth_u_finite,
                "integral_absolute_tolerance": 2e-11}
            bounds_reports.append(report)
            truth = moment_excess_risk(mean, law.m, law.q, mean_y=mean, true_m=law.m, true_q=law.q, y_scale=mean)
            require(truth["total_excess_risk"] == 0., "Moment-truth excess risk is not zero")
            deviations = []
            for factor in (.5, .8, 1.2, 2.):
                m, q = law.m * factor, law.q * factor ** 2
                head = moment_bounds(A, m, q)
                pred = head["L"] + .5 * (head["H"] - head["L"])
                excess = moment_excess_risk(pred, m, q, mean_y=mean, true_m=law.m, true_q=law.q, y_scale=mean)
                require(excess["total_excess_risk"] > 0., "Nontruth moment report did not increase expected risk")
                deviations.append({"moment_scale_factor": factor, **excess})
            consistency.append({"distribution": law.name, "A": A, "truth_excess_risk": truth,
                                "perturbations": deviations})
    for A, m, b, d in ((.2, 1., math.sqrt(8), 1.), (2., .4, .8, -1.3), (.01, 2., -1.4, .7)):
        head = moment_head(A, m, b, d)
        analytic = head_gradients(A, m, b, d)
        numerical = {key: _finite_difference(lambda *x: moment_head(*x)["prediction"], (A, m, b, d), i)
                     for i, key in enumerate(("A", "m", "b", "d"))}
        require(all(math.isclose(analytic[k], numerical[k], rel_tol=2e-6, abs_tol=2e-8) for k in analytic),
                "Analytic/finite-difference head gradients disagree")
        gradients.append({"A": A, "m": m, "b": b, "d": d, "analytic": analytic,
                          "central_finite_difference": numerical, "maximum_absolute_error": max(abs(analytic[k] - numerical[k]) for k in analytic)})
        for k in (.01, .2, 5., 100.):
            scaled_bounds = moment_bounds(k * k * A, k * k * m, k ** 4 * head["q"])
            scaled_head = moment_head(k * k * A, k * k * m, b, d)
            for key in ("L", "H"):
                require(math.isclose(scaled_bounds[key], k * head[key], rel_tol=2e-13), "Bound homogeneity failed")
            require(math.isclose(scaled_head["prediction"], k * head["prediction"], rel_tol=2e-13), "Head homogeneity failed")
            homogeneity.append({"A": A, "m": m, "b": b, "d": d, "k": k,
                                 "head_homogeneity_absolute_error": abs(scaled_head["prediction"] - k * head["prediction"])})
    sequence = [upper_approach(.2, 1., 4., 4. * k) for k in (1., 2., 10., 100., 1e3, 1e4, 1e6)]
    lower_exact = math.sqrt(.2) * .75 + math.sqrt(.2 + C * 4) * .25
    require(math.isclose(lower_exact, moment_bounds(.2, 1., 4.)["L"], rel_tol=2e-14), "Lower extremizer failed")
    require(all(a["gap_to_H"] > b["gap_to_H"] > 0 for a, b in zip(sequence, sequence[1:])), "Upper sequence does not approach the supremum")
    conflicts = [fixed_t_conflict(LAWS[2]), fixed_t_conflict(LAWS[1])]
    atom = conflicts[0]
    require(math.isclose(atom["point_scale_squared"], 9 * math.pi ** 2 / 320, rel_tol=2e-14), "Point-optimal t5 scale is wrong")
    require(math.isclose(atom["auxiliary_only_scale_squared"], .4, rel_tol=2e-7), "Auxiliary-optimal t5 scale is wrong")
    for conflict in conflicts:
        require(abs(conflict["NLL_log_scale_derivative_at_point"]) > 1e-4
                and conflict["joint_excess_MSE_scaled"] > 0
                and conflict["optimized_joint_risk_minus_constant_variance"] < conflict["point_joint_risk_minus_constant_variance"],
                "The synthetic misspecification conflict did not reproduce")
    return {"version": "e3-synthetic-moment-envelope-verification-v1", "c": C,
        "operations": {"synthetic_population_risk_optimizations": True, "neural_models_implemented": False,
            "synthetic_population_risk_optimization_count": 4, "neural_models_implemented_count": 0,
            "neural_models_trained": 0, "market_data_read": False, "market_data_rows_read": 0,
            "market_fits": 0, "market_inferences": 0, "NPZ_or_checkpoint_opened": False},
        "independent_law_checks": bounds_reports, "moment_truth_consistency": consistency,
        "gradient_checks": gradients, "homogeneity_checks": homogeneity,
        "extremizers": {"A": .2, "m": 1., "q": 4., "lower_exact_mean": lower_exact,
            "upper_sequence": sequence, "upper_is_unattained_supremum_for_q_gt_m_squared": True},
        "fixed_t5_misspecification_counterexamples": conflicts,
        "degenerate_boundary": {"A": .2, "m": 0., "q": 0., "algebraic_head": math.sqrt(.2),
            "log_scores_defined": False, "finite_positive_moment_score_minimizer_exists": False},
        "parameterization_limitations": {"b_zero_gradient_trap": head_gradients(.2, 1., 0., 1.)["b"],
            "d_zero_gradient_trap": head_gradients(.2, 1., 1., 0.)["d"],
            "suggested_nonzero_initialization": {"b": math.sqrt(8), "q_over_m_squared": 9., "d": 1., "rho": .5},
            "large_d_rounding_example": {"d": 1e150, "rho": moment_head(.2, 1., 1., 1e150)["rho"],
                "real_arithmetic_rho_is_strictly_below_one": True, "floating_point_strictness_guaranteed": False},
            "dL_dA_can_diverge_at_A_zero": True},
        "interpretation": ["Known moment inequalities and quasi-Gamma mean scores are not claimed as inventions.",
            "Truth moments and a rich report class jointly minimize expected component risks; no neural optimization guarantee.",
            "Expected score for q requires finite E[U^4]; Pareto alpha5 has infinite E[U^8] and may have infinite gradient variance.",
            "The computed envelope is not a prediction interval, confidence interval, quantile model or Student distribution.",
            "The algebraic known floor does not prove irreducible prediction error.",
            "Excess MSE can be zero while raw expected MSE is positive; constants are explicitly retained in the counterexamples."],
        "reproduction": {"python": sys.version.split()[0], "scipy": scipy.__version__,
            "python_patch_comparison": "same major/minor required; original generation patch retained",
            "integration": "independent scipy.integrate.quad over each synthetic U law", "float_tolerance_rtol": 2e-11, "float_tolerance_atol": 2e-12}}


def figure_svg(metrics):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "e3_moment_verification_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    with matplotlib.rc_context({"svg.hashsalt": "e3-moment-envelope-synthetic", "font.family": "DejaVu Sans", "font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(11., 4.6), layout="constrained")
        # The common A=.2 is evaluated independently for this view, without
        # adding a fitted model or selecting a favorable sample.
        rows = [{**moment_bounds(.2, law.m, law.q), "mean": law.expectation(lambda u: math.sqrt(.2 + C * u * u)),
                 "label": law.name} for law in LAWS]
        for index, row in enumerate(rows):
            axes[0].plot([index, index], [row["L"], row["H"]], color="#467799", linewidth=4, alpha=.5)
            axes[0].scatter(index, row["L"], marker="_", color="#222222", zorder=3)
            axes[0].scatter(index, row["H"], marker="o", facecolors="none", edgecolors="#222222", zorder=3)
            axes[0].scatter(index, row["mean"], color="#16834a", s=25, zorder=4)
        axes[0].plot([], [], color="#467799", linewidth=4, alpha=.5, label="Envolvente [L,H); H sólo es supremo")
        axes[0].scatter([], [], color="#16834a", s=25, label="Media integrada independiente")
        axes[0].set_xticks(range(len(rows)), ["Uniforme", "Exponencial", "Dos puntos", "Lognormal", "Pareto α=5"], rotation=20)
        axes[0].set(title="Cotas y medias de leyes sintéticas (A=0,2)", ylabel="Media de Y = √(A + cU²)")
        axes[0].legend(fontsize=8)
        for name, color in (("point_scale", "#16834a"), ("joint_scale", "#bd3345"), ("auxiliary_only_scale", "#c36b18")):
            label = {"point_scale": "Sólo error cuadrático", "joint_scale": "Criterio conjunto (λ=0,1)", "auxiliary_only_scale": "Sólo contraste NLL"}[name]
            axes[1].scatter([0, 1], [r[name] for r in metrics["fixed_t5_misspecification_counterexamples"]], color=color, label=label)
        axes[1].set_xticks([0, 1], ["U∈{0,2}", "U exponencial(1)"])
        axes[1].set(title="Conflicto poblacional de una escala t₅ fija (A=0)", ylabel="Escala física s", xlabel="Ejemplos sintéticos; ninguna red entrenada")
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.grid(alpha=.2)
        fig.suptitle("Verificación matemática sintética: momentos y contraste bajo mala especificación", fontsize=11)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": "e3_verify_moment_head.py"})
        plt.close(fig)
    return buffer.getvalue()


def artifacts():
    metrics = verify_all()
    return {"metrics.json": json.dumps(metrics, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n",
            "moment_envelope.svg": figure_svg(metrics)}


def _compare(actual, expected, path="metrics"):
    if path == "metrics.reproduction.python":
        require(isinstance(actual, str) and isinstance(expected, str)
                and len(actual.split(".")) == len(expected.split(".")) == 3
                and all(part.isdigit() for part in actual.split(".") + expected.split("."))
                and actual.split(".")[:2] == expected.split(".")[:2],
                path + ": Python major/minor changed")
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-12), path + ": numeric output changed")
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), path + ": keys changed")
        for key in expected:
            _compare(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), path + ": list changed")
        for i, value in enumerate(expected):
            _compare(actual[i], value, path + f"[{i}]")
    else:
        require(type(actual) is type(expected) and actual == expected, path + ": metadata changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = artifacts()
        if args.check:
            for name, value in result.items():
                old = (args.output_dir / name).read_bytes()
                if name.endswith(".json"):
                    _compare(json.loads(old), json.loads(value))
                else:
                    require(old == value, "Deterministic SVG changed")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for name, value in result.items():
                (args.output_dir / name).write_bytes(value)
        print(json.dumps({"status": "verified" if args.check else "generated", "synthetic_population_risk_optimizations": True,
                          "market_fits": 0, "market_inferences": 0, "artifacts": list(result)}))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Moment-head verification failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
