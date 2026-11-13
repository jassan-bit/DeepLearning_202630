# Resume the existing 80-hour E2 campaign

The failed resumption re-enumerated historical cache witnesses even though the
T4/A100 transition was already sealed. New cached results can change the first
witness in sorted order, and the immutable writer correctly refused to replace
`hardware_transition/historical_t4_profile.json`.

This runtime repair validates and reuses the original profile, cache audit and
transition receipt. It checks their recorded provenance, original completed
budget charges, current strict A100 execution policy, and all saved final-fit
identities and artifacts. A fresh compatible runtime observation gets a separate
resume admission; the three original files remain unchanged.

The funding authority stays at `41da88cf600aadc0319214465c94a85335e2b1b7`.
The launcher verifies the existing 80-hour anchor and original consent against
GitHub and reuses the same receipt and timestamp. It neither asks for another
increase nor grants further credit. E/H/P/Q and the 80-hour funding sources and
policies remain unchanged.

A complete local mirror can skip replaying historical metadata ZIPs after its
metadata and every backed-up artifact, including weights, have been checked
against the current GitHub state and pinned manifests. Missing files use the
original full restoration; divergent artifacts, unresolved leases and concurrent
control changes stop admission.

The unchanged final pipeline continues with final fits, pilots, classical
comparators, predictor freeze and final scoring. Completed HPO and final fits
are retained. Publishing this repair does not execute training.
