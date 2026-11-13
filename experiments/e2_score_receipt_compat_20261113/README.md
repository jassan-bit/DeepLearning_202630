# Terminal backup receipt for the preserved acquisition failure

The source-recovery launcher at `24344bd6db20f0546b26167eda4c3076a38c0f4b` reached the scoring stage but stopped before a new worker reservation, downloads, decoding or inference. Its metadata disposition had `status: pre_payload_failure_reconciled`; the real GitHub backup API accepts only terminal receipt states. It therefore raised `terminal_result_receipt_required`.

The disposition itself had already been snapshotted to GitHub. This repair keeps its exact bytes and SHA256 `2fd9f653963277dd0b6e5c0fc98a305da0f5d56cb59ff0e3726612eed8c1dc55`, including the original source-recovery proof, failed job, charge and unknown optimizer-step evidence. The complete `24344bd6…` runtime, source policy and registration remain authoritative and unchanged.

A narrow callback adapter archives that exact metadata file with a terminal backup receipt whose status is `failed`. It explicitly records the original reconciliation status, `scientific_result: false` and `benchmark_complete: false`. It creates no scientific `result.json` and reports no completed benchmark. Every other receipt, including the eventual completed final-score receipt, passes through unchanged.

The new launcher/controller wrap the original source-recovery parent solely to install this callback. The same 81 frozen predictors, original raw-data loader, twenty registered archive identities, ARIMA correction and existing 80-hour consent remain in effect. Only the scoring stage is requested. No model, pilot, comparator, calibration or scientific evaluation is repeated by the adapter. GitHub remains the only persistent destination.

Tests cover the real backup receipt validator and artifact packaging using an in-memory Git transport, reproduce the original rejection and verify a failed metadata manifest, preserve immutable disposition bytes, exercise the original nested launcher chain, and pass completed scientific receipts through without alteration. No production market data, training or final evaluation runs in the development workspace.
