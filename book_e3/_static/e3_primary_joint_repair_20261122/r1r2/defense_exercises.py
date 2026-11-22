"""Synthetic, CPU-only R2 teaching checks; never trains or changes the candidate.

The author can explain and modify the isolated `didactic_*` readout below in a
live defence. A recorded execution by an AI agent is not a student's defence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def didactic_naive_readout(A, atoms, probabilities):
    """Real-arithmetic equivalent to the frozen readout; can overflow on squaring."""
    import torch
    return (probabilities * torch.sqrt(A[:, None] + (6 / 49) * atoms.square())).sum(-1)


def didactic_stable_readout(A, atoms, probabilities):
    """Reasoned local replacement, already present in the unchanged candidate."""
    import math
    import torch
    return (probabilities * torch.hypot(A.sqrt()[:, None], math.sqrt(6 / 49) * atoms)).sum(-1)


def state_sha(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(candidate_root):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    identity = json.loads((candidate_root / "CANDIDATE_IDENTITY.json").read_text())
    assert identity["candidate_id"] == "input-only-overlap-mse-1"
    before = {rel: sha(candidate_root / rel) for rel in identity["runtime_source_sha256"]}
    assert before == identity["runtime_source_sha256"], "Candidate source hash mismatch"
    sys.path.insert(0, str(candidate_root))
    c = importlib.import_module("runtime.candidate")
    assert Path(c.__file__).resolve() == candidate_root / "runtime/candidate.py"

    # Independent, generated OHLCV fixture. No real-market array/weight is read.
    rng = np.random.default_rng(908221)
    close = 100 * np.exp(np.cumsum(rng.normal(.0002, .012, (12, 7)), axis=1))
    opening = close * np.exp(rng.normal(0, .003, (12, 7)))
    high = np.maximum(opening, close) * np.exp(rng.uniform(.001, .006, (12, 7)))
    low = np.minimum(opening, close) * np.exp(-rng.uniform(.001, .006, (12, 7)))
    raw = np.stack((opening, high, low, close, rng.uniform(1000, 100000, (12, 7))), -1)
    A, *_ = c.data.known_geometry(raw)
    U = rng.uniform(.001, .04, 12)
    Y = np.sqrt(A + c.data.C * U ** 2)
    fitted = c.fit_inputs(raw, Y)  # Algebraic scales on generated fixture, not a forecasting fit.
    model = c.make_model(23, fitted).eval()
    model_state_before = state_sha(model)
    checks = []
    with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("Training prohibited")), \
            patch.object(torch.optim.SGD, "step", side_effect=AssertionError("Training prohibited")), \
            torch.no_grad():
        args = c.tensors(raw, fitted)[:3]
        out = model.forward_components(*args)
        assert sum(p.numel() for p in model.parameters()) == 1262
        assert out["atoms"].shape == out["probabilities"].shape == (12, 3)
        torch.testing.assert_close(out["probabilities"].sum(-1), torch.ones(12, dtype=torch.float64))
        assert torch.all(out["atoms"][:, 1:] > out["atoms"][:, :-1])
        checks.append({"id": "D01", "name": "Frozen identity, capacity, ordered atoms and simplex", "pass": True})

        # Perturb already-transformed future positions: states at the first four
        # positions must be identical. Raw normalization is origin-causal only.
        altered_X = args[0].clone()
        altered_X[:, 4:] += 10
        h1, h2 = model.encoder(args[0]), model.encoder(altered_X)
        assert torch.equal(h1[:, :4], h2[:, :4])
        checks.append({"id": "D02", "name": "Prefix causality of encoder on fixed transformed inputs", "pass": True,
                       "scope": "Not prefix causality of within-window raw normalization"})

        # The didactic refactor changes the evaluation expression, not the law.
        naive = didactic_naive_readout(args[1], out["atoms"], out["probabilities"])
        stable = didactic_stable_readout(args[1], out["atoms"], out["probabilities"])
        torch.testing.assert_close(naive, stable, atol=1e-15, rtol=1e-13)
        torch.testing.assert_close(stable, out["prediction"], atol=1e-15, rtol=1e-13)
        assert (stable >= args[1].sqrt()).all()
        checks.append({"id": "D03", "name": "Equivalent refactor, exact finite-law expectation and support", "pass": True,
                       "max_absolute_difference": float((naive - stable).abs().max())})

        huge_atoms = torch.tensor([[1e200, 2e200, 3e200]], dtype=torch.float64)
        huge_A = torch.tensor([1.0], dtype=torch.float64)
        huge_p = torch.full((1, 3), 1 / 3, dtype=torch.float64)
        huge_naive = didactic_naive_readout(huge_A, huge_atoms, huge_p)
        huge_stable = didactic_stable_readout(huge_A, huge_atoms, huge_p)
        assert torch.isinf(huge_naive).all() and torch.isfinite(huge_stable).all()
        checks.append({"id": "D04", "name": "Concrete overflow counterexample and verified stable correction", "pass": True,
                       "naive_overflows": True, "stable_result": float(huge_stable[0]),
                       "scope": "Extreme synthetic FP64 fixture; not an observed market failure"})

        changed_scale = model.forward_components(args[0], args[1], args[2] * 3)
        assert torch.equal(out["prediction"], changed_scale["prediction"])
        scaled = raw.copy()
        k = 1.7
        scaled[..., :4] = np.exp(.4 + k * np.log(scaled[..., :4]))
        scaled_args = c.tensors(scaled, fitted)
        assert not c.tensors(raw, fitted)[3].any() and not scaled_args[3].any()
        torch.testing.assert_close(args[0], scaled_args[0], rtol=1e-5, atol=1e-6)
        transformed = model.forward_components(*scaled_args[:3])
        formula = didactic_stable_readout(k*k*args[1], out["atoms"], out["probabilities"])
        torch.testing.assert_close(formula, transformed["prediction"], rtol=1e-5, atol=1e-10)
        assert (transformed["prediction"] < k*out["prediction"]).all()
        checks.append({"id": "D05", "name": "Input invariance under stated assumptions; output equivariance counterexample", "pass": True,
                       "local_output_scale_argument_ignored": True,
                       "max_equivariance_gap": float((k*out["prediction"]-transformed["prediction"]).max())})

        # Two distinct latent laws with equal transformed mean at A=0.
        aa = torch.tensor([[.5, 1, 1.5], [.25, .75, 2]], dtype=torch.float64)
        pp = torch.full((2, 3), 1 / 3, dtype=torch.float64)
        means = didactic_stable_readout(torch.zeros(2, dtype=torch.float64), aa, pp)
        torch.testing.assert_close(means[0], means[1], atol=1e-15, rtol=1e-13)
        variances = (pp * (aa - (aa*pp).sum(-1)[:, None]).square()).sum(-1)
        assert variances[0] != variances[1]
        checks.append({"id": "D06", "name": "MSE does not identify the latent atomic distribution", "pass": True,
                       "equal_transformed_means": means.tolist(), "different_U_variances": variances.tolist()})

    assert model_state_before == state_sha(model)
    after = {rel: sha(candidate_root / rel) for rel in before}
    assert before == after
    return {"candidate_id": identity["candidate_id"], "candidate_root": str(candidate_root),
            "candidate_identity_sha256": sha(candidate_root / "CANDIDATE_IDENTITY.json"),
            "candidate_sources_verified": before, "candidate_sources_unchanged": True,
            "model_state_sha256_before_and_after": model_state_before,
            "status": "PASS", "checks": checks, "new_checks": len(checks),
            "training_optimizer_steps": 0, "new_forecasting_fits": 0,
            "market_arrays_loaded": 0, "market_predictions_computed": 0,
            "generated_windows": 12, "synthetic_model_forward_calls": 3,
            "additional_synthetic_encoder_only_calls": 2,
            "student_defense_observed": False, "teacher_acceptance_claimed": False,
            "R1_technical_verdict_changed": False, "R3_accredited": False,
            "novel_designs_added": 0, "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
            "AI_assistance": "AI prepared and executed the teaching checks; no human act is represented"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "e3_input_only_candidate_20261121")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("EXECUTION_RECEIPT.json"))
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    wall, cpu = time.perf_counter(), time.process_time()
    result = run(args.candidate_root.resolve())
    result.update(started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                  process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  script_sha256=sha(__file__), executable=sys.executable)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "new_checks", "cpu_seconds", "wall_seconds", "new_forecasting_fits", "market_predictions_computed")}))


if __name__ == "__main__":
    main()
