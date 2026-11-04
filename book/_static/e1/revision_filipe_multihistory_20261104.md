# Auditoría independiente: cuatro entradas y test reservado

Estado: **PASS**. Fase: **test**.

No project analysis/training helpers imported; no raw archives, fitting or inference executed.

- All saved CV cohorts, scaler fits, seeds, metrics, checkpoints and fold hashes match the four-history causal contract.
- Configuration selection and all120 epoch counts were independently recovered from development CV, before test access.
- All120 fixed final models precede first raw test access; twenty monthly test blocks contain no retraining, and all metrics/BDS diagnostics reproduce from saved predictions.

| Control | Cantidad |
|---|---:|
| artifact_hashes_checked | 13871 |
| numeric_values_checked | 453840 |
| independently_reconstructed_volatility_days | 1339 |
| prediction_metrics_recomputed | 160800 |
| training_preprocessors_checked | 3000 |
| cv_monthly_model_segments_checked | 1440 |
| cv_trajectories_checked | 120 |
| analysis_metric_rows_checked | 4216 |
| seed_summary_rows_checked | 3264 |
| bds_rows_recomputed | 7680 |
| bds_descriptive_rows_checked | 1296 |
| final_models_checked | 120 |
| test_checkpoint_outputs_checked | 120 |

Los promedios de p-valores BDS son descriptivos; los meses cortos limitan la aproximación asintótica. Los intervalos entre semillas no estiman variabilidad entre mercados independientes.
