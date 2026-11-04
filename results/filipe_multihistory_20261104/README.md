# Cuatro entradas, BDS por fold y test final

Este documento describe los archivos completos conservados **localmente**. La versión pública incluye tablas y figuras agregadas, BDS por mes/semilla, código, protocolo, hashes y auditorías. Los vectores de predicciones, las series diarias, los pesos y los cinco paquetes completos no se incorporan a esta publicación pública.

Se completó la CV de Filipe para entradas de **7, 14, 21 y 28 días**, con objetivo fijo **σ7** (desviación estándar poblacional de siete retornos logarítmicos diarios, sin anualizar) y horizonte **h=1 día**. La fuente original es Binance Spot BTCUSDT de un minuto; las redes usan agregados diarios OHLCV, sin rezagos explícitos del objetivo.

El desarrollo conserva 2023 como historia inicial y doce bloques mensuales de validación forward en 2024. La entrada D determina el embargo, la elegibilidad y la purga de la validación interna; por ello, sus diferencias no aíslan exclusivamente el efecto de la longitud. La representación principal es `relative_window`, con escalas ajustadas únicamente al entrenamiento admitido. Las cuatro entradas usan diez semillas, las mismas configuraciones por familia y el mismo límite de épocas.

## Ejecución y cierre

| Parte | Ejecución comprobada |
|---|---:|
| CV D7 reutilizada, representación principal | 30 trayectorias; 360 segmentos mensuales; 720 ajustes |
| Nueva CV D14/21/28 | 90 trayectorias; 1.080 segmentos mensuales; 2.160 ajustes |
| CV principal conjunta | 43.920 predicciones, 366 fechas por trayectoria |
| Ajuste final exclusivamente con desarrollo | 120 modelos |
| Test enero de 2025–agosto de 2026 | 72.960 predicciones sobre 608 fechas comunes |
| Bloques mensuales de reporte del test | 20; cero reajustes durante test |
| BDS por fold, semilla y dimensión | 2.880 contrastes en CV y 4.800 en test |

La selección se cerró **antes de abrir el test** en `experiments/filipe_multihistory_20261104/freeze.json`. Las tres familias seleccionaron D7 por RMSE medio de CV; la referencia neuronal elegida fue MLP/D7. Las épocas finales de cada configuración y semilla proceden de su validación interna de diciembre de 2024. El ajuste final conserva la purga D antes del comienzo del test y guarda escalas y pesos antes de leer sus veinte archivos.

Los commits `8959d06` (configuración cerrada) y `47db98a` (identidades de los 120 modelos finales) se publicaron antes del primer acceso al test. `final_test_events.jsonl` documenta la secuencia. Cada checkpoint realizó una sola pasada sobre la cohorte común; los meses del test son intervalos de reporte de los mismos pesos, no nuevos folds con reentrenamiento. Las entradas de cada día pueden utilizar OHLCV de días anteriores ya observados. Septiembre y octubre de 2026 siguen excluidos.

## Resultados y límites

Las tablas completas están en `analysis_cv/` y `test/analysis/`: MAPE, MAE, RMSE, MSE, R² y MASE por modelo, entrada, semilla y mes. Los resúmenes principales calculan media y desviación estándar muestral entre semillas sobre el calendario completo, con IC95 marginales condicionados a esas fechas. La dispersión entre meses se identifica como descriptiva y no sustituye las diez repeticiones.

En test, CNN1D/D7 obtuvo el menor RMSE neuronal observado, pero ese resultado **no cambia retrospectivamente** la selección previa MLP/D7. La persistencia tuvo menor RMSE medio que las doce configuraciones neuronales. Estas comparaciones nuevas son descriptivas; la inferencia Holm12/DM6 del suplemento anterior pertenece únicamente a la CV inicial D7/2024.

BDS se aplica por mes y semilla en dimensiones 2 y 3, con radio de 1,5 desviaciones estándar muestrales del residuo y corrección Holm por fase. Sus 28–31 observaciones mensuales hacen que la referencia asintótica sea exploratoria. Las medias de p-valores son resúmenes descriptivos, no pruebas combinadas; rechazar iid no demuestra por sí solo no linealidad aprovechable. MAPE conserva su máscara de objetivos positivos, sin epsilon. MASE usa exclusivamente una escala histórica de entrenamiento; los denominadores pueden diferir entre D por la distinta cohorte y purga. La persistencia requiere ocho cierres previos para calcular σ7(d−1).

## Archivos reproducibles

- `protocol.json`: protocolo registrado antes de las nuevas corridas.
- `code_manifest_before_fitting.json`: hashes y entorno de entrenamiento.
- `D14/`, `D21/`, `D28/`: índices generados mediante llamadas reales a `timeseries-cv0.1.5`.
- `predictions.csv`, `runs.csv`, `fold_metrics.csv`: nueva CV; D7 conserva sus archivos originales en `results/dl_r2_cv_20261104/`.
- `final_models_manifest.json`, `final_fit_summary.csv`: modelos finales y sus identidades previas al test.
- `test/raw_test_audit.json`, `test/cohort_manifest.json`, `test/eligibility.csv`: integridad, calidad y cohorte común.
- `run_artifacts_manifest.json` y cinco archivos `.tar.gz`: los 10.950 archivos de las nuevas corridas, incluidos 1.200 checkpoints. La CV D7 original conserva su propio archivo de artefactos.
- `references/audits/revision_filipe_multihistory_20261104.json/.md`: auditoría independiente de métricas, escalas, selección, hashes y cronología.

Los folds firmados ya están versionados: no deben sobrescribirse al verificar la librería, pues sus metadatos de generación también forman parte de los hashes. Para reconstruir sus índices, use `prepare_filipe_multihistory.py --output-dir /tmp/filipe-fold-check` en el entorno dedicado de CV. Para verificar los resultados, extraiga los archivos de artefactos en sus directorios de resultados correspondientes y ejecute en el entorno de entrenamiento:

```bash
python scripts/close_filipe_multihistory.py --validate
python scripts/verify_filipe_multihistory.py --phase cv
python scripts/verify_filipe_multihistory.py --phase test
python scripts/package_filipe_multihistory.py
```

Una ejecución nueva requiere registrar un experimento separado antes de entrenar. El ejecutor final bloquea reajustes después del primer acceso al test y reutiliza resultados firmados al reanudar. Este test ya fue evaluado; no puede volver a presentarse como una reserva desconocida para justificar cambios posteriores.
