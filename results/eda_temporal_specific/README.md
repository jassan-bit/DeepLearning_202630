# EDA específico descriptivo — regresión temporal (bloque C)

Se usan únicamente CSV de desarrollo 2023–2024 existentes. La cuadrícula UTC conserva 731 días y el 24/03/2023 ausente; no se imputan datos ni se abren ZIP o archivos de test.

`covariate_structure.csv`, `availability_contract.csv` y `availability_by_lag.csv` describen las cinco covariables y su disponibilidad. La barra completa de d−k es observable al comienzo de d−k+1, por lo que k≥1 es causal al comienzo de d. El objetivo que incluye el retorno d y la correlación retrospectiva etiquetada d no se incorporan como entradas de ese origen.

`lag_correlations_common_train.csv` contiene 140 coeficientes Pearson/Spearman con los mismos 308 días de 2023. Se exige que los 28 rezagos de todas las covariables y sigma7 sean válidos. `cohort_comparison.csv` conserva el contraste con los análisis marginales previos, que tenían 328–350 pares. `lag_peak_summary.csv` registra máximos absolutos descriptivos; no selecciona un rezago óptimo, una ventana D o un modelo. No se calculan p-values ni se interpreta causalidad o capacidad de pronóstico.

`rolling_relationship_summary.csv` resume por año las correlaciones de 60 días, con mínimo 50 pares, calculadas previamente. Son ventanas retrospectivas solapadas; los cambios de signo no son rupturas de régimen demostradas. Se verificaron veinte reconstrucciones por fecha y covariable sin reutilizar el cálculo móvil de pandas.

Reproducir desde el repositorio: `/workspace/.venvs/deeplearning-eda/bin/python scripts/eda_temporal_specific.py`. `summary.json` conserva hashes, convenciones y verificaciones independientes. No se ajusta ninguna referencia lineal, red u otro predictor; el diagnóstico de residuos no se presenta como ejecutado por estas correlaciones.

Datos: Binance Vision / Binance Spot BTCUSDT. Procesamiento: Jassan Arteta. Derivados y figura: CC BY-NC-SA 4.0 y condiciones aplicables del proveedor. No modifica los derechos de los trabajos citados ni de figuras ajenas.
