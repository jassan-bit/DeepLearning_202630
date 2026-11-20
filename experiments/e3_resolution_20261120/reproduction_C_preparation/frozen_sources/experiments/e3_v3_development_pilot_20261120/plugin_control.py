"""Second-moment plug-in ablation for the existing atomic V3 prototype.

AI assistance declared. Only the scalar aggregation is replaced; the inherited
encoder, atomic locations, masses, initialization and normalized CRPS_U remain
unchanged. This is a development control, not an accredited R1/R3 result.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from src.e3_overlap_model_v3 import C, VERSION as BASE_VERSION
from src.e3_overlap_model_v3 import OverlapDistributionModel


VERSION = "e3-overlap-atomic-v3-plugin-control-v1"
VARIANT = "atom_plugin"


def second_moment_prediction(A, atoms, probabilities):
    """Return sqrt(A + C * sum(p*u**2)) without first squaring large atoms.

    Scaling the nonnegative weighted norm avoids first squaring large atoms and
    does not differentiate sqrt(p), which is singular at p=0. A detected loss
    of every positive contribution to underflow is rejected, not returned as
    a zero forecast. The zero-norm branch gives its zero subgradient.
    """
    if not isinstance(atoms, torch.Tensor) or atoms.ndim != 2 or atoms.shape[1] < 1:
        raise ValueError("Atoms must be a batch-by-K tensor")
    atoms = atoms.to(dtype=torch.float64)
    p = torch.as_tensor(probabilities, dtype=torch.float64, device=atoms.device)
    variance_floor = torch.as_tensor(A, dtype=torch.float64, device=atoms.device)
    if p.shape != atoms.shape or variance_floor.shape != atoms.shape[:1]:
        raise ValueError("Aligned atoms, probabilities and one variance floor per sample required")
    if (not bool(torch.isfinite(atoms).all()) or not bool((atoms >= 0).all()) or
            not bool(torch.isfinite(p).all()) or not bool((p >= 0).all()) or
            not bool(torch.isfinite(variance_floor).all()) or not bool((variance_floor >= 0).all()) or
            not torch.allclose(p.sum(-1), torch.ones_like(variance_floor), atol=1e-12, rtol=1e-12)):
        raise ValueError("Finite nonnegative atomic probability laws and floors required")
    floor = torch.sqrt(variance_floor)
    root_c_atoms = math.sqrt(C) * atoms
    scale = torch.maximum(floor, root_c_atoms.amax(dim=-1))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = ((floor / safe_scale).square() +
                  (p * (root_c_atoms / safe_scale[:, None]).square()).sum(-1))
    if bool(((scale > 0) & (normalized == 0)).any()):
        raise ValueError("Atomic dynamic range caused plug-in norm underflow")
    safe_normalized = torch.where(scale > 0, normalized, torch.ones_like(normalized))
    prediction = scale * torch.sqrt(safe_normalized)
    prediction = torch.maximum(prediction, floor)
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("Nonfinite plug-in prediction; no clipping or replacement performed")
    return prediction


class PluginSecondMomentModel(OverlapDistributionModel):
    """Capacity-matched V3 control with learned atoms/masses and CRPS_U.

    Bare state_dict bytes share their schema with atom_crps by design. Resume
    must also compare configuration; use load_control_checkpoint or the pilot's
    stronger source/data/job identity checks before loading a checkpoint.
    """
    def __init__(self, *, u_scale, y_scale, seed=20261118, lambda_crps=.1):
        super().__init__("atom_crps", u_scale=u_scale, y_scale=y_scale,
                         seed=seed, lambda_crps=lambda_crps)
        self.variant = self.name = self.model_name = VARIANT

    def forward_components(self, values, A):
        outputs = super().forward_components(values, A)
        outputs["finite_law_mean_prediction"] = outputs["prediction"]
        outputs["prediction"] = second_moment_prediction(
            A, outputs["atoms"], outputs["probabilities"])
        return outputs

    def configuration(self):
        configuration = super().configuration()
        configuration.update(
            version=VERSION,
            base_atomic_law_version=BASE_VERSION,
            aggregation="sqrt_A_plus_c_expected_U_squared",
            exact_mean_under_finite_atomic_law=False,
            same_atomic_law_as_atom_crps_at_equal_parameters=True,
            scope="Development second-moment plug-in control; not confirmation or R1/R3 accreditation",
        )
        return configuration


def load_control_checkpoint(model, checkpoint):
    """Check configuration and tensor schema before mutating the model.

    This narrow helper does not authenticate a file, deserialize bytes or check
    data/source identities. The enclosing pilot must bind those separately.
    """
    if not isinstance(model, PluginSecondMomentModel) or not isinstance(checkpoint, Mapping):
        raise ValueError("A plug-in model and structured checkpoint are required")
    if checkpoint.get("configuration") != model.configuration():
        raise ValueError("Checkpoint configuration differs; old weights remain unchanged")
    state = checkpoint.get("state_dict")
    expected = model.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("Checkpoint state schema differs")
    for name, target in expected.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or
                value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
            raise ValueError("Invalid checkpoint tensor: " + name)
    return model.load_state_dict(state, strict=True)
