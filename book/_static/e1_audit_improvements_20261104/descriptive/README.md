# Complemento descriptivo posterior a la evaluación

Se reutilizan los pronósticos guardados. No hay nuevos entrenamientos, pasadas de modelos ni lecturas de ZIP raw. Este complemento se prepara después de observar el test: no es una confirmación prospectiva ni autoriza cambiar configuraciones.

`annual_*` separa 2025 (365 fechas) de enero–agosto de 2026 (243); primero calcula las métricas por semilla y periodo, y después media/DE/IC marginal t95 entre las diez semillas. Persistencia aparece una vez por D: no tiene DE ni IC por semillas. R² se calcula con el promedio del objetivo del periodo correspondiente; no se promedian R² mensuales.

`variance_components.csv` aplica la identidad descriptiva de varianza poblacional a un panel balanceado de RMSE mensuales. Se descomponen efectos principales por arquitectura, semilla y mes; el resto son interacciones. Las fracciones no constituyen ANOVA inferencial, efectos causales o mercados independientes. Se añade el panel semilla×mes dentro de cada red. La varianza de RMSE entre celdas no es la MSE de los pronósticos.

Los límites de `severity_*` y `residual_bins_*` se calculan solo con σ7 de desarrollo hasta el 03/12/2024, último resultado admitido del entrenamiento final con D28. Son límites comunes, anteriores al test; la elegibilidad exige los 28 días OHLCV previos completos, sin cerrar huecos; los segmentos no alteran el pipeline. Bajo/medio/alto usa q25/q75 del objetivo; los cuatro intervalos del pronóstico usan q25/q50/q75. Las celdas con menos de diez fechas suprimen sus métricas; se conserva su conteo. Los momentos residuales identifican patrones de dispersión/sesgo, sin declarar una prueba formal de heterocedasticidad o una mejora futura alcanzable.

`monthly_prediction_aggregates.csv` y las figuras muestran únicamente medias mensuales y dispersión entre semillas. Cada punto resume 28–31 fechas. `difficult_months.csv` selecciona los tres meses con mayor RMSE media por D/modelo; es un relato retrospectivo de errores, no una política de selección. No se exportan vectores diarios, precios diarios, residuos individuales ni pesos.

`distribution_by_period.csv` describe objetivo y covariables de 2023,2024,2025 y enero–agosto2026. Distancia de Wasserstein/DE de entrenamiento y desplazamiento de media estandarizada son índices descriptivos, no nuevos p-valores ni evidencia causal. Los niveles de precios de periodos distintos pueden cambiar estructuralmente; observarlos tras cerrar test no permite reajustar el modelo y volver a llamar independiente a la misma evaluación.
