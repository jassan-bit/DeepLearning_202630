# Verificación independiente de la inferencia final

Estado: **PASS**.

No project analyzer/helper, torch or training module imported; manual HAC and Ljung–Box formulas; stored forecasts only.

- Registered source files, output hashes and the unchanged pre-test MLP/D7 selection verified.
- Six paired t/Shapiro comparisons, six exact signed-rank sensitivities, their Holm families, six seed-bootstrap effect intervals and six retrospective MDE scenarios reproduced.
- Eighteen DM values reconstructed with manual Bartlett HAC/HLN; nine Ljung–Box values reconstructed from centered residual autocorrelations.
- All36 RMSE/MAE paired circular-date intervals and relative-gain intervals reproduced with the registered random schedule.
- Twenty-four descriptive differences retain references fixed from CV; no model training, raw archive or checkpoint forward call executed.

| Comprobación | Cantidad |
|---|---:|
| source_hashes_checked | 10 |
| output_hashes_checked | 10 |
| numeric_values_checked | 37362 |
| paired_tests_checked | 6 |
| seed_effect_bootstrap_intervals_checked | 6 |
| wilcoxon_sensitivities_checked | 6 |
| retrospective_MDE_rows_checked | 6 |
| DM_HAC_HLN_rows_checked | 18 |
| manual_LjungBox_rows_checked | 9 |
| date_bootstrap_intervals_checked | 36 |
| relative_difference_rows_checked | 24 |

This confirms computation/integrity of an explicitly post-hoc supplement, not prospective confirmation, achieved power or independent corroboration of the several estimands.
