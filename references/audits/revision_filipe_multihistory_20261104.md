# Auditoría independiente: cuatro entradas y test reservado

Estado: **PASS**. Fase: **cv**.

No project analysis/training helpers imported; no raw archives, fitting or inference executed.

- All saved CV cohorts, scaler fits, seeds, metrics, checkpoints and fold hashes match the four-history causal contract.

| Control | Cantidad |
|---|---:|
| artifact_hashes_checked | 12868 |
| numeric_values_checked | 170615 |
| independently_reconstructed_volatility_days | 731 |
| prediction_metrics_recomputed | 87840 |
| training_preprocessors_checked | 2880 |
| cv_monthly_model_segments_checked | 1440 |
| cv_trajectories_checked | 120 |
| analysis_metric_rows_checked | 1612 |
| seed_summary_rows_checked | 1248 |
| bds_rows_recomputed | 2880 |
| bds_descriptive_rows_checked | 552 |

Los promedios de p-valores BDS son descriptivos; los meses cortos limitan la aproximación asintótica. Los intervalos entre semillas no estiman variabilidad entre mercados independientes.
