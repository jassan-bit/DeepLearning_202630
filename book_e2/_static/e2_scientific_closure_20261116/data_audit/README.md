# Auditoría correctiva de datos actuales

Verificación posterior al benchmark, sin entrenamiento o inferencia de modelos.

| Control | Estado | Alcance | Límite |
| --- | --- | --- | --- |
| D01 | PASS | current protocol and declared source identities |  |
| D02 | PASS | actual development CSV calendar and OHLCV |  |
| D03 | PASS | actual train/calibration target construction and IDs |  |
| D04 | PASS | actual608 evaluation targets, inputs and calendar |  |
| D05 | PASS | outcome IDs, embargo and rolling raw supports |  |
| D06 | PASS | all available actual D7 input windows |  |
| D07 | PASS | actual common final preprocessing, reconstructed only from655train |  |
| D08 | PASS | common pipeline metamorphic check on copies of actual data | This checks common preparation, not all author-port behavior |
| D09 | PASS | current78 historical final-fit input signatures |  |
| D10 | PASS | exact numerical D7 raw window fingerprints | No claim of universal absence of leakage |
| D11 | PASS | actual available fixed-threshold near-window diagnostic | Review alerts if present; the threshold is descriptive and does not certify absence of leakage |
| D12 | PASS | all81 physical frozen scaler records and current training cutoff | The recovered checkpoint files belong to the documented reproduction; this audit only reads JSON scalers |
| D13 | PARTIAL | model-port behavior beyond common preprocessing | Port-specific learned behavior/negative controls require a separately recorded audit; this result is not an all-port leakage certificate |
| D14 | PARTIAL | real permuted-target training negative control | Requires a separately registered bounded training control |
| D15 | PARTIAL | adversarial train/test domain classifier | A classifier fit is a separate diagnostic of distribution difference, not proof of leakage |
| D16 | NA | inter-annotator agreement/classification label-noise tests | External measurement noise and a Bayes floor remain unidentified; arithmetic reconstruction is not their estimate |

Los controles PASS se limitan a los datos y funciones indicados. PARTIAL no significa un fallo observado.
Las cinco barras de entrada y seis cierres objetivo compartidos entre calibración y test son historia causal legítima, sin claves de objetivo compartidas.
Los vecinos son un diagnóstico descriptivo con umbral heredado 0,01; no seleccionan transformaciones ni eliminan casos.
current_cohort_arrays.npz contiene datos reales reconstruidos y la conversión float32 común, no nuevas predicciones.
Los controles sintéticos unitarios comprueban el programa; la evidencia de mercado está en los resultados de esta ejecución real.
