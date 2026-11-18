"""Check scalar identities on artificial numbers, with no model imports or data."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import random
from pathlib import Path


def verify():
    rng = random.Random(271801)
    link_errors, variance_errors, projection_increases = [], [], []
    for _ in range(1000):
        z = rng.uniform(-80.0, 80.0)
        a = 10.0 ** rng.uniform(-6.0, 0.0)
        b = rng.uniform(0.0, 0.2)
        positive = math.log1p(math.exp(z))
        excess = a * positive
        transformed_excess = math.exp(math.log(a) + math.log(positive))
        link_errors.append(abs(transformed_excess - excess) / excess)
        direct_head = b + a * (1e-4 + positive)
        transformed_head = (b + a * 1e-4) + transformed_excess
        assert math.isclose(direct_head, transformed_head, rel_tol=1e-13, abs_tol=1e-15)
        known = [rng.uniform(-0.1, 0.1) for _ in range(6)]
        future = rng.uniform(-0.1, 0.1)
        mean6 = math.fsum(known) / 6.0
        A = math.fsum((r - mean6) ** 2 for r in known) / 7.0
        mean7 = math.fsum(known + [future]) / 7.0
        direct_variance = math.fsum((r - mean7) ** 2 for r in known + [future]) / 7.0
        decomposed_variance = A + 6.0 / 49.0 * (future - mean6) ** 2
        variance_errors.append(abs(direct_variance - decomposed_variance))
        assert math.isclose(direct_variance, decomposed_variance, rel_tol=1e-13, abs_tol=1e-17)
        target = b + rng.uniform(0.0, 0.2)
        prediction = rng.uniform(-0.2, 0.4)
        projected = max(prediction, b)
        for before, after in [((prediction-target)**2, (projected-target)**2),
                              (abs(prediction-target), abs(projected-target))]:
            projection_increases.append(after-before)
            assert after <= before + 1e-15
    assert max(link_errors) < 1e-13
    return {"status": "passed_artificial_scalar_identities_only", "seed": 271801,
            "artificial_cases_per_identity": 1000,
            "maximum_relative_positive_link_difference": max(link_errors),
            "maximum_absolute_variance_identity_difference": max(variance_errors),
            "maximum_projection_loss_increase": max(projection_increases),
            "new_market_data_used": False, "new_model_fits": 0,
            "new_model_forwards": 0, "new_hypothesis_tests": 0,
            "novelty_certified": False, "human_external_review": False,
            "scope": "Scalar equality and algebraic sanity checks; no equivalence of finite neural classes or training.",
            "python": platform.python_version(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify()
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise SystemExit("Refusing to overwrite an existing receipt")
        args.output.write_text(text)
    print(text, end="")
