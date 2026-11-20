# Controlled V3 development pilot engine

Prepared with AI assistance. Importing these modules does not load data or fit a
model. This engine performs no network requests; the launcher owns remote saves.

The five neural recipes are the 1210-parameter additive reference, `atom_crps`,
`atom_mse_only`, `fixed_probs` (1236 parameters), and the capacity-matched
1262-parameter `atom_plugin` control. The last control preserves the full V3 law
and loss but substitutes `sqrt(A + c E[U²])` for `E[sqrt(A + c U²)]`. The two
analytical controls are the known floor and floor plus the **fit-only** mean
excess. A successful development comparison does not establish substantive
novelty, R1 or R3.

Only three members of the byte-hash-pinned NPZ are indexed: `training_raw`,
`training_y_original`, `training_dates`. The original loader fits normalization
on 532 historical fit windows. The 116 exposed historical validation windows
are split prospectively for this pilot into 54 selection, seven purged, and 55
assessment windows. Calibration/test members and additional datasets are never
indexed. This separation does not erase prior exposure of these calendars.

Each recipe receives the same three HPO learning rates with seed 20261117 and
three final seeds 20261118–20261120: 30 fits in total. Learning-rate selection and
early stopping use selection RMSE. Every final recipe/seed must finish before
assessment predictions are produced. The continuation heuristic compares the
candidate's mean seed RMSE to the best neural control and to the best of all
controls; the required 2% improvement is exploratory, without p-values or a
confirmatory effect margin. Raw-unit CRPS_U is reported only for law models.

Each bounded optimizer phase has 40 epochs, patience eight, batch 32, AdamW with
zero weight decay, norm-one gradient clipping, and one immutable 120-second wall
opportunity from its phase-start marker. Model construction and tensor setup
precede that marker and are reported separately in each job receipt.
The frozen phase saves model, optimizer, RNG, curve and request identity.
Restart downtime, checkpoint I/O and durability waits consume the original
deadline. A batch cannot be preempted; any overrun is reported. A fit with no
validated trained checkpoint fails; it never substitutes random weights.

The launcher must write `hardware_profile.json`, `launcher_identity.json`
(`master_run_id`) and `launcher_lock.json` before `--prepare-only`. Preparation
seals the plan, admitted training-only geometry, source hashes and GPU/software
cohort. The launcher publishes these records and provides a publication receipt
binding `plan_sha256`, `admission_sha256`, `source_commit`, `master_run_id`, and a
40-character `git_commit`. `--run` verifies these values before fitting.

With `--durability-dir`, the engine emits `durability_barrier` events at
`before_fit`, `first_validated_checkpoint`, and `after_job`. The first checkpoint
barrier occurs after both optimizer state and validated curve are durable. The
launcher must remotely verify the save, then atomically write
`<barrier_id>.json` containing that `barrier_id` and `remote_verified: true`.
Acknowledgements may take up to 600 seconds, but do not extend the 120-second fit
deadline. A slow publication can therefore exhaust a fit. An interruption before
the first remotely saved trained checkpoint can remain incomplete; unsaved
optimizer steps cannot be recovered. Never reset the deadline to hide this.

Synthetic tests include poisoned calibration/test object arrays, interrupted
optimizer/RNG restoration, completed-checkpoint reuse, first validated checkpoint
barriers, changed-request/cohort rejection and refusal of incomplete comparisons.
Run in an environment with Torch and NumPy:

```bash
E3_PILOT_SOURCE_ROOT=/path/to/frozen/source python test_pilot_engine.py
```

The environment variable is needed only when the tests are staged outside the
repository. No market data, market fits or market inference are used by tests.
