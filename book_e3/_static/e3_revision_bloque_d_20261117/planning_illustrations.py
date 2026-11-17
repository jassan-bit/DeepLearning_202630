"""Mathematical planning illustrations for guide section 8; no study data.

These normal approximations and exact rank-test resolution calculations do not
estimate current power, choose an E3 sample size, or authorize new experiments.
Only the Python standard library is used; no predictions or model are loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist


ALPHA = 0.05
NORMAL = NormalDist()


def quantile_sum(family_size: int, power: float, *, two_sided: bool = True) -> float:
    if type(family_size) is not int or family_size < 1:
        raise ValueError("A positive integer family size is required")
    if not 0.5 < power < 1:
        raise ValueError("Illustrative power must lie strictly between .5 and 1")
    adjusted_alpha = ALPHA / family_size
    return NORMAL.inv_cdf(1 - adjusted_alpha / (2 if two_sided else 1)) + NORMAL.inv_cdf(power)


def approximate_pairs(family_size: int, power: float, standardized_separation: float,
                      *, two_sided: bool = True) -> float:
    if not math.isfinite(standardized_separation) or standardized_separation <= 0:
        raise ValueError("The planned separation from the null boundary must be positive")
    return (quantile_sum(family_size, power, two_sided=two_sided) / standardized_separation) ** 2


def standardized_mde(family_size: int, power: float, pairs: int,
                     *, two_sided: bool = True) -> float:
    if type(pairs) is not int or pairs < 2:
        raise ValueError("At least two independent pairs are needed for this illustration")
    return quantile_sum(family_size, power, two_sided=two_sided) / math.sqrt(pairs)


def signed_rank_minimum_p(nonzero_pairs: int, *, two_sided: bool = True) -> float:
    if type(nonzero_pairs) is not int or nonzero_pairs < 1:
        raise ValueError("A positive integer count of nonzero pairs is required")
    # All signs equal: two extreme sign assignments among 2**n, or one for a
    # one-sided test. This is the exact, tie-free signed-rank reference.
    return 2.0 ** ((1 if two_sided else 0) - nonzero_pairs)


def first_holm_resolution(family_size: int, *, two_sided: bool = True) -> int:
    if type(family_size) is not int or family_size < 1:
        raise ValueError("A positive integer family size is required")
    n = 1
    while signed_rank_minimum_p(n, two_sided=two_sided) >= ALPHA / family_size:
        n += 1
    return n


def self_checks() -> list[str]:
    checks = []
    for n in (5, 9, 10, 11):
        assert signed_rank_minimum_p(n + 1) == signed_rank_minimum_p(n) / 2
    checks.append("Exact rank-test p resolution halves with each additional nonzero pair")
    assert signed_rank_minimum_p(5) == 0.0625
    assert [first_holm_resolution(m) for m in (12, 16, 48)] == [9, 10, 11]
    for m in (12, 16, 48):
        n = first_holm_resolution(m)
        assert signed_rank_minimum_p(n) < ALPHA / m <= signed_rank_minimum_p(n - 1)
    checks.append("First-step Holm rank resolution is checked on both sides of its threshold")
    assert math.isclose(approximate_pairs(12, .8, .5), 4 * approximate_pairs(12, .8, 1))
    checks.append("Halving planned standardized separation quadruples approximate sample size")
    assert approximate_pairs(48, .9, 1) > approximate_pairs(16, .9, 1) > approximate_pairs(12, .9, 1)
    assert approximate_pairs(12, .9, 1) > approximate_pairs(12, .8, 1)
    checks.append("Larger families and higher target power do not reduce the normal approximation")
    assert math.isclose(standardized_mde(12, .8, 40), standardized_mde(12, .8, 10) / 2)
    checks.append("Four times as many independent pairs halves illustrative standardized MDE")
    assert approximate_pairs(48, .9, 1) > approximate_pairs(48, .9, 1, two_sided=False)
    checks.append("Bilateral and unilateral reference quantiles are kept distinct")
    assert math.ceil(approximate_pairs(12, .8, 1)) == 14
    assert math.ceil(approximate_pairs(12, .8, .5)) == 55
    checks.append("Guide illustrative bilateral reference gives ceilings 14 and 55")
    assert math.isclose(.9 ** 12, .2824295364810001)
    assert max(0.0, 1 - 12 * (1 - .95)) <= .95 ** 12 <= .95
    checks.append("Independent-product example respects the arbitrary-dependence union lower bound")
    for function, args in ((approximate_pairs, (12, .8, 0)), (first_holm_resolution, (0,)),
                           (signed_rank_minimum_p, (0,)), (standardized_mde, (12, .8, 1))):
        try:
            function(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid planning input was accepted")
    checks.append("Invalid family sizes, pair counts and zero separation are rejected")
    return checks


def run() -> dict:
    checks = self_checks()
    planning = []
    mde = []
    resolution = []
    joint = []
    for m in (12, 16, 48):
        for two_sided in (True, False):
            side = "bilateral_guide_reference" if two_sided else "unilateral_margin_reference"
            resolution.append({"family_size": m, "reference": side,
                "first_Holm_threshold": ALPHA / m,
                "minimum_nonzero_pairs_for_possible_first_step_rejection": first_holm_resolution(m, two_sided=two_sided),
                "not_a_statement_about_all_later_Holm_steps": True})
            for power in (.8, .9):
                for separation in (1.0, .5):
                    raw = approximate_pairs(m, power, separation, two_sided=two_sided)
                    planning.append({"family_size": m, "reference": side, "target_marginal_power": power,
                        "planned_separation_over_sigma": separation, "normal_approximation_R": raw,
                        "ceiling_R": math.ceil(raw), "is_actual_E3_sample_size_recommendation": False})
                for pairs in (10, 20):
                    mde.append({"family_size": m, "reference": side, "target_marginal_power": power,
                        "independent_pairs": pairs,
                        "MDE_separation_over_sigma": standardized_mde(m, power, pairs, two_sided=two_sided),
                        "not_an_estimate_of_E3_MDE": True})
        for power in (.9, .95):
            joint.append({"number_of_decisions": m, "common_marginal_power": power,
                "product_if_independent_only": power ** m,
                "union_lower_bound_under_arbitrary_dependence": max(0.0, 1 - m * (1 - power)),
                "actual_joint_power_estimated": False})
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone_for_document_date": "America/Bogota",
        "source_review_base_commit": "8fccb216f36a3e4fc5cb38ecdec769b7f5379397",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "alpha": ALPHA,
        "status": "mathematical_illustrations_only_not_a_frozen_experimental_plan",
        "observed_test_effect_used": False,
        "model_predictions_loaded": False,
        "market_fitting_or_inference": False,
        "normal_approximation_assumptions": [
            "Independent paired experimental units with a prespecified gain estimand and finite variance",
            "Known planning variance and normal reference; a real design must account for estimated variance and dependence",
            "Bonferroni alpha/m is a conservative planning reference, not a computed power for Holm or a three-test conjunction",
            "The unilateral margin reference uses planned gain g1 above null boundary delta, with separation g1-delta",
        ],
        "rank_resolution_assumptions": [
            "Exact sign distribution for n nonzero tie-free paired differences; symmetry assumptions remain necessary",
            "The first Holm threshold is alpha/m; thresholds can increase after earlier ordered rejections",
            "Sidedness and procedures must be fixed before observing results, not changed to obtain significance",
        ],
        "guide_reference_formula": "R approx ((z_(1-alpha/(2m)) + z_power) * sigma_d / separation)^2",
        "practical_margin_formula": "If H0: gain <= delta and planned g1 > delta, separation = g1-delta; not delta again",
        "normal_sample_size_illustrations": planning,
        "normal_MDE_illustrations": mde,
        "signed_rank_exact_minimum_p_examples": [{"nonzero_pairs": n,
            "bilateral_minimum_p": signed_rank_minimum_p(n),
            "unilateral_minimum_p": signed_rank_minimum_p(n, two_sided=False)} for n in (5, 9, 10, 11)],
        "signed_rank_first_Holm_resolution": resolution,
        "joint_power_illustrations": joint,
        "self_checks": {"passed": True, "checks": checks},
        "not_estimated": ["E3 prospective marginal power", "E3 joint power", "E3 actual MDE",
                          "a final required number of seeds", "an actual family success probability"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("planning_illustrations.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "self_checks": len(result["self_checks"]["checks"]),
                      "output": str(args.output)}, ensure_ascii=False))
