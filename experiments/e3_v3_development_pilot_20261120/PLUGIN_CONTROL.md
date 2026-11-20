# Second-moment plug-in control

This separate development module changes the scalar aggregation of the existing
V3 atomic prototype. It does not alter the historical source or the completed
796-job audit. AI assistance was used to implement and test this control.

For the same predicted nonnegative atoms `u`, probabilities `p`, and observed
variance floor `A`, the full model predicts `sum(p * sqrt(A + (6/49)*u**2))`.
The control predicts `sqrt(A + (6/49)*sum(p*u**2))`. Both use the learned ordered
atoms, learned probabilities, encoder, initialization and normalized CRPS_U
weight 0.1. Their trainable parameter count is 1262. The control's configuration
is distinct: variant `atom_plugin`, version
`e3-overlap-atomic-v3-plugin-control-v1`, and
`exact_mean_under_finite_atomic_law=False`.

The control value is at least the full model's value **under a fixed common
atomic law**, by Jensen's inequality. Equality holds when all positively weighted
atoms give the same transformed value. This mathematical ordering does not
establish comparative forecasting performance after separate optimization.

The public interface matches `OverlapDistributionModel`:

```python
from plugin_control import PluginSecondMomentModel

model = PluginSecondMomentModel(
    u_scale=u_scale, y_scale=y_scale, seed=seed, lambda_crps=0.1,
)
components = model.forward_components(values, A)
losses = model.loss_components(components, Y, U)
prediction = model(values, A)
configuration = model.configuration()
```

Put the immutable scientific source root and this directory on `sys.path`.
`forward_components` also returns `finite_law_mean_prediction`, computed under
the same atoms and probabilities before replacing `prediction`. The inherited
`forward` and `loss_components` use the replacement. No parameters are added.

Checkpoint resumes must compare the complete model configuration plus the
pilot's source, job, data and optimizer identities before loading. Bare
`state_dict` keys intentionally match the full model for a capacity-matched
ablation. `load_control_checkpoint` checks configuration and tensor keys,
shapes, dtypes and finiteness before loading; it does not authenticate files or
replace the pilot's broader identity checks.

The standalone readout uses a scaled weighted norm in float64 to avoid directly
squaring very large atoms. It preserves the floor and rejects nonfinite inputs,
invalid probability laws and detected complete positive-contribution underflow.
The zero norm uses a zero subgradient for atoms and probabilities. `A` is an
observed external input; gradients with respect to `A` at zero are not claimed.

Verification used only synthetic tensors on CPU Torch 2.8.0. All 14 tests passed:
law and initialization identity; parameter count; Jensen order; degenerate
equality; correct point loss and unchanged CRPS; finite gradients; floor, dtype,
overflow/underflow behavior; checkpoint roundtrip and mismatch rejection. These
checks performed no market reads, real-data fitting or market predictions and
do not accredit R1 or R3.

Inside the repository, tests discover the source root among their parent
directories. For a staged standalone directory, supply the frozen source root
explicitly (`E3_SCIENTIFIC_SOURCE` remains a legacy alias):

```bash
E3_PILOT_SOURCE_ROOT=/path/to/immutable/source \
python -m unittest discover -s /path/to/model -p test_plugin_control.py -v
```
