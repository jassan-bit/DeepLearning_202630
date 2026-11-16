"""Synthetic mathematical checks only; no networks, market arrays or checkpoints."""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import e3_verify_moment_head as design


@pytest.fixture(scope="module")
def metrics():
    return design.verify_all()


@pytest.mark.parametrize("law", design.LAWS, ids=lambda law: law.family or "two_point")
def test_independent_integrals_and_analytic_moments_agree_and_obey_envelope(law):
    assert law.expectation(lambda u: u ** 2) == pytest.approx(law.m, rel=2e-10)
    assert law.expectation(lambda u: u ** 4) == pytest.approx(law.q, rel=2e-10)
    for A in (0., .07, 3.):
        lower, upper = (design.moment_bounds(A, law.m, law.q)[key] for key in ("L", "H"))
        mean = law.expectation(lambda u: math.sqrt(A + design.C * u * u))
        assert lower - 2e-11 <= mean <= upper + 2e-11


def test_lower_bound_is_attained_by_zero_q_over_m_extremizer():
    for A in (0., .2, 5.):
        m, q = 1.7, 8.
        probability = m * m / q
        exact = (1 - probability) * math.sqrt(A) + probability * math.sqrt(A + design.C * q / m)
        assert design.moment_bounds(A, m, q)["L"] == pytest.approx(exact, rel=2e-14)


def test_upper_is_strict_for_nondegenerate_laws_and_only_approached_with_fixed_moments():
    A, m, q = .2, 1., 4.
    H = design.moment_bounds(A, m, q)["H"]
    rows = [design.upper_approach(A, m, q, R) for R in (4., 8., 40., 400., 40000., 4e6)]
    for row in rows:
        assert row["a_R"] >= 0 and 0 < row["probability_R"] < 1
        assert row["mean_V"] == pytest.approx(m, rel=2e-14)
        assert row["second_moment_V"] == pytest.approx(q, rel=2e-14)
        assert row["mean_Y"] < H
    assert [row["mean_Y"] for row in rows] == sorted(row["mean_Y"] for row in rows)
    assert rows[-1]["gap_to_H"] < 1e-6


def test_equal_moment_boundary_collapses_to_a_degenerate_V():
    for A, m in ((0., 1.), (.3, .7), (4., 3.)):
        result = design.moment_bounds(A, m, m * m)
        assert result["L"] == pytest.approx(result["H"], rel=2e-14)
        assert result["H"] == math.sqrt(A + design.C * m)
        assert design.moment_head(A, m, 0., 2.)["prediction"] == pytest.approx(result["H"], rel=2e-14)


def test_zero_moment_has_only_algebraic_limit_and_no_finite_log_score_minimum():
    assert design.moment_bounds(.3, 0., 0.)["L"] == math.sqrt(.3)
    assert design.moment_bounds(0., 0., 0.)["H"] == 0
    with pytest.raises(ValueError):
        design.quasi_scores(0., 0., 0.)
    scores = [sum(design.quasi_scores(0., m, m * m).values()) for m in (1., .1, .001)]
    assert scores[0] > scores[1] > scores[2]
    with pytest.raises(ValueError):
        design.moment_head(0., 0., 0., 1.)


@pytest.mark.parametrize("values", [(-1., 1., 1.), (1., -1., 1.), (1., 1., .99),
    (1., 0., 1.), (math.inf, 1., 1.), (1., math.nan, 1.), (True, 1., 1.)])
def test_invalid_moment_reports_are_rejected(values):
    with pytest.raises(ValueError):
        design.moment_bounds(*values)


def test_unsafe_derived_parameters_and_scores_fail_clearly():
    for b, d in ((math.inf, 1.), (1., math.nan), (1e300, 1.)):
        with pytest.raises(ValueError):
            design.moment_head(.2, 1., b, d)
    with pytest.raises(ValueError):
        design.quasi_scores(1e300, 1., 1.)
    with pytest.raises(ValueError):
        design.upper_approach(.2, 1., 4., 3.99)


def test_head_parameterization_includes_lower_endpoint_and_is_strict_inside_in_representable_domain():
    for A in (0., .2, 3.):
        for b in (.1, 1., math.sqrt(8)):
            for d in (-3., -.5, 0., 1., 6.):
                result = design.moment_head(A, .7, b, d)
                assert result["q"] >= .7 ** 2
                assert 0 <= result["rho"] < 1
                assert result["L"] <= result["prediction"] < result["H"]
                if d == 0:
                    assert result["prediction"] == result["L"]


def test_large_d_exposes_rounding_to_one_without_hidden_clipping():
    result = design.moment_head(.2, 1., 1., 1e150)
    assert result["rho"] == 1.
    assert result["rho_rounded_to_one"] is True
    assert result["prediction"] == result["H"]


def test_large_finite_d_has_stable_derivatives_and_preserves_the_real_A_zero_singularity():
    for d in (1e150, 1e308, -1e308):
        boundary = design.head_gradients(0., 1., 1., d)
        assert boundary["A"] == math.inf
        assert all(math.isfinite(boundary[k]) for k in ("m", "b", "d"))
        interior = design.head_gradients(.2, 1., 1., d)
        assert all(math.isfinite(value) for value in interior.values())
        # The nonzero real derivative underflows to zero, without inf/inf or
        # clipping the head coordinate. Its scale is proportional to |d|^-3.
        assert interior["d"] == 0


def test_homogeneity_has_correct_units_for_all_three_moments_and_mean():
    A, m, b, d = .13, 1.7, 1.3, .7
    original = design.moment_head(A, m, b, d)
    for k in (.001, .03, 2., 1000.):
        explicit = design.moment_bounds(k ** 2 * A, k ** 2 * m, k ** 4 * original["q"])
        scaled = design.moment_head(k ** 2 * A, k ** 2 * m, b, d)
        assert scaled["prediction"] == pytest.approx(k * original["prediction"], rel=2e-13)
        assert scaled["q"] == pytest.approx(k ** 4 * original["q"], rel=2e-13)
        assert explicit["L"] == pytest.approx(k * original["L"], rel=2e-13)
        assert explicit["H"] == pytest.approx(k * original["H"], rel=2e-13)


def test_analytic_gradients_match_finite_differences_away_from_singular_A_boundary(metrics):
    for report in metrics["gradient_checks"]:
        for key in ("A", "m", "b", "d"):
            assert math.isfinite(report["analytic"][key])
            assert report["analytic"][key] == pytest.approx(report["central_finite_difference"][key], rel=2e-6, abs=2e-8)


def test_A_zero_derivative_is_not_claimed_globally_finite():
    gradient = design.head_gradients(0., 1., 1., 1.)
    assert math.isinf(gradient["A"])
    assert all(math.isfinite(gradient[k]) for k in ("m", "b", "d"))
    assert math.isfinite(design.head_gradients(0., 1., 0., 1.)["A"])


def test_zero_b_and_d_are_real_gradient_traps_and_proposed_initialization_avoids_them():
    assert design.head_gradients(.2, 1., 0., 1.)["b"] == 0
    assert design.head_gradients(.2, 1., 1., 0.)["d"] == 0
    result = design.moment_head(.2, 1., math.sqrt(8), 1.)
    gradients = design.head_gradients(.2, 1., math.sqrt(8), 1.)
    assert result["q"] == pytest.approx(9.) and result["rho"] == .5
    assert gradients["b"] != 0 and gradients["d"] != 0


def test_moment_scores_handle_zero_observations_and_their_expected_gradients_vanish_at_truth():
    assert design.quasi_scores(0., 2., 3.) == {"m": math.log(2.), "q": math.log(3.)}
    for law in design.LAWS:
        score_m = law.expectation(lambda u: design.quasi_scores(u * u, law.m, law.q)["m"])
        score_q = law.expectation(lambda u: design.quasi_scores(u * u, law.m, law.q)["q"])
        assert score_m == pytest.approx(math.log(law.m) + 1, rel=2e-10)
        assert score_q == pytest.approx(math.log(law.q) + 1, rel=2e-10)
        assert law.expectation(lambda u: 1 / law.m - u * u / law.m ** 2) == pytest.approx(0., abs=2e-11)
        assert law.expectation(lambda u: 1 / law.q - u ** 4 / law.q ** 2) == pytest.approx(0., abs=2e-11)


def test_joint_component_truth_minimizes_population_excess_without_claiming_optimizer_convergence(metrics):
    for report in metrics["moment_truth_consistency"]:
        assert report["truth_excess_risk"]["total_excess_risk"] == 0
        assert all(r["total_excess_risk"] > 0 for r in report["perturbations"])
    assert any("no neural optimization guarantee" in x for x in metrics["interpretation"])


def test_pareto_finite_fourth_does_not_imply_finite_eighth_or_gradient_variance(metrics):
    law = design.LAWS[-1]
    assert law.q == 5 and law.eighth_u_finite is False
    report = [r for r in metrics["independent_law_checks"] if r["distribution"] == law.name][0]
    assert report["E_U4_finite"] is True and report["E_U8_finite"] is False


def test_atomic_t5_counterexample_has_correct_analytic_point_and_auxiliary_optima(metrics):
    report = metrics["fixed_t5_misspecification_counterexamples"][0]
    assert report["point_scale_squared"] == pytest.approx(9 * math.pi ** 2 / 320, rel=2e-14)
    assert report["auxiliary_only_scale_squared"] == pytest.approx(.4, rel=2e-7)
    analytic_derivative = 1 - 6 / (5 * report["point_scale_squared"] + 4)
    assert report["NLL_log_scale_derivative_at_point"] == pytest.approx(analytic_derivative, abs=1e-14)
    assert analytic_derivative != 0
    assert report["minimum_raw_expected_MSE_scaled"] == pytest.approx(3.)
    assert report["joint_raw_expected_MSE_scaled"] > 3
    assert report["joint_excess_MSE_scaled"] > 0
    assert "not KL" in report["interpretation"]


def test_continuous_positive_exponential_also_shifts_point_mean_under_misspecified_auxiliary(metrics):
    report = metrics["fixed_t5_misspecification_counterexamples"][1]
    assert report["point_scale"] < report["joint_scale"] < report["auxiliary_only_scale"]
    assert abs(report["NLL_log_scale_derivative_at_point"]) > .001
    assert report["joint_excess_MSE_scaled"] > 0
    assert report["minimum_raw_expected_MSE_scaled"] == pytest.approx(1.)
    assert "continuous positive" in report["interpretation"]


def test_joint_scalar_optimizer_is_stationary_for_the_recorded_normalized_v2_risk(metrics):
    for law, report in zip((design.LAWS[2], design.LAWS[1]), metrics["fixed_t5_misspecification_counterexamples"]):
        ratio = design.MEAN_ABS_T5 * report["joint_scale"] / law.mean_u
        derivative = 2 * ratio * (ratio - 1) + .1 * design.expected_t5_log_scale_derivative(law, report["joint_scale"])
        assert abs(derivative) < 2e-7
        assert report["optimized_joint_risk_minus_constant_variance"] < report["point_joint_risk_minus_constant_variance"]


def test_reproduction_is_deterministic_and_changed_numerics_or_claims_are_rejected(metrics):
    first, second = design.figure_svg(metrics), design.figure_svg(metrics)
    assert first == second
    ET.fromstring(first)
    assert metrics["operations"]["synthetic_population_risk_optimizations"] is True
    assert metrics["operations"]["neural_models_implemented"] is False
    assert metrics["operations"]["neural_models_implemented_count"] == 0
    assert metrics["operations"]["neural_models_trained"] == 0
    assert metrics["operations"]["synthetic_population_risk_optimization_count"] == 4
    assert metrics["operations"]["market_data_read"] is False
    assert metrics["operations"]["market_data_rows_read"] == 0
    expected = {"risk": .123, "market_fits": 0}
    design._compare({"risk": .123 + 1e-13, "market_fits": 0}, expected)
    with pytest.raises(ValueError):
        design._compare({"risk": .124, "market_fits": 0}, expected)
    with pytest.raises(ValueError):
        design._compare({"risk": .123, "market_fits": 1}, expected)


def test_real_metrics_keep_generation_python_patch_but_require_same_major_minor_and_scipy(metrics):
    actual = json.loads(json.dumps(metrics))
    major, minor, patch = metrics["reproduction"]["python"].split(".")
    actual["reproduction"]["python"] = f"{major}.{minor}.{int(patch) + 17}"
    design._compare(actual, metrics)
    assert actual["reproduction"]["python"] != metrics["reproduction"]["python"]
    actual["reproduction"]["python"] = f"{major}.{int(minor) + 1}.{patch}"
    with pytest.raises(ValueError, match="major/minor"):
        design._compare(actual, metrics)
    actual["reproduction"]["python"] = metrics["reproduction"]["python"]
    actual["reproduction"]["scipy"] = "99.0.0"
    with pytest.raises(ValueError, match="metadata changed"):
        design._compare(actual, metrics)
