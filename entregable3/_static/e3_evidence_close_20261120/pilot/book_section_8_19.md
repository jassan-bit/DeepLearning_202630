(e3-v3-pilot-completed-20261120)=
## 8.19 Piloto V3 terminado: ablaciones y resultado adverso de desarrollo

**El piloto completó sus 30 ajustes y no superó la heurística de continuación.** El [resultado inmutable](https://github.com/jassan-bit/DeepLearning_202630/tree/8eaa581bc34b615a8507c9e384a3e4be29084720/runs/e3_v3_development_pilot/20261120T031424176339Z) corresponde al código `8f65eac34aaf49293924ae8328c235cedbdd2866`, ejecución `20261120T031424176339Z`, snapshot 93. Se conservan sus {download}`recibo final <../_static/e3_evidence_close_20261120/pilot/archive/payload/pilot_receipt.json>`, {download}`plan <../_static/e3_evidence_close_20261120/pilot/archive/payload/plan.json>` y {download}`veredicto original <../_static/e3_evidence_close_20261120/pilot/archive/payload/development_verdict.json>`. Los 796 trabajos anteriores pertenecen a la referencia aditiva y no se convierten en entrenamientos V3.

### Qué se ejecutó y qué se verificó después

El estudio original realizó 15 ajustes HPO y 15 finales: cinco recetas neuronales, tres tasas con una semilla HPO y tres semillas finales por receta; además evaluó los dos controles analíticos. Las 532 ventanas de ajuste y las 54 de selección conservan su función. Tras siete fechas purgadas, la evaluación usa **55 fechas, del 7 de septiembre al 31 de octubre de 2024**, ya expuestas históricamente como desarrollo. Los recibos declaran cero miembros de calibración/prueba abiertos y ningún conjunto D01–D03 usado en este piloto.

El {download}`control de integridad <../_static/e3_evidence_close_20261120/pilot/generated/archive_verification.json>` comprueba **19 archivos originales**, cinco hashes de artefactos y los enlaces del plan, admisión, launcher y estado. De las predicciones guardadas se recalculan **17 trayectorias y 935 filas**, con los mismos objetivos y fechas. Las 935 filas no son 935 observaciones independientes. Los 30 recibos de fase registran estados entrenados y terminados, sin exceso de plazo declarado en este piloto; ese dato no se extrapola a los recursos de la auditoría histórica. No se descargaron ni reprodujeron checkpoints, ni los 259 archivos completos del snapshot.

### Resultado de las siete recetas y decisión original

La métrica principal permanece fijada como **media de las RMSE individuales**, no RMSE de un ensamble medio ni una métrica elegida después del resultado. La {download}`recomputación completa <../_static/e3_evidence_close_20261120/pilot/generated/metrics_recomputed.csv>` y las {download}`comparaciones de ablación <../_static/e3_evidence_close_20261120/pilot/generated/ablation_comparisons.csv>` conservan todas las recetas:

| Receta | Trayectorias | RMSE media individual | DE entre semillas | MAE media |
|---|---:|---:|---:|---:|
| `atom_mse_only` | 3 | 0,002073833 | 0,000023659 | 0,001644275 |
| `fixed_probs` | 3 | 0,002083649 | 0,000040853 | 0,001557355 |
| `atom_plugin` | 3 | 0,002094286 | 0,000031909 | 0,001549696 |
| `floor_plus_mean_excess` | 1 analítica | 0,002114231 | No aplica | 0,001813483 |
| `atom_crps` —V3 completa— | 3 | 0,002129916 | 0,000013719 | 0,001591685 |
| Aditiva | 3 | 0,002157281 | 0,000022490 | 0,001811966 |
| `floor_only` | 1 analítica | 0,002790341 | No aplica | 0,001831766 |

La V3 completa tiene **2,70 % más RMSE que `atom_mse_only`**, el mejor control neuronal y global, y aproximadamente 0,74 % más que el control de piso más exceso. Mejora un 1,27 % frente a la aditiva, pero eso no satisface la regla frente al mejor de los controles completos. Para superar la heurística del 2 %, su RMSE debía ser como máximo **0,002032356**. El veredicto permanece **`continuation_heuristic_passed=false`**. La completa pierde en RMSE frente a las tres ablaciones neuronales en las tres semillas emparejadas; estos conteos no son una prueba estadística ni establecen un efecto causal general de cada componente.

Retirar CRPS, fijar masas y sustituir la lectura exacta ya tienen evidencia empírica **de desarrollo**. La variante de masas fijas tiene 1236 parámetros y las de masas aprendidas 1262; no se disimula esa diferencia de capacidad. La menor MAE de otro control y los CRPS_U guardados se conservan como resultados secundarios, sin reemplazar RMSE para rescatar la decisión. **La variante sin CRPS no se promueve automáticamente a modelo final.** El resultado adverso no constituye una sanción de integridad ni demuestra por sí mismo ausencia de novedad; muestra que este piloto no aporta la utilidad exigida por su regla original.

```{figure} ../_static/e3_evidence_close_20261120/pilot/generated/pilot_rmse_and_dependence.png
:name: fig-e3-v3-pilot-saved-results
:width: 100%

Resultados de desarrollo guardados. Los puntos muestran semillas, las barras su media de RMSE individuales y los rombos controles analíticos. Las correlaciones de residuos son promedios descriptivos entre semillas, sin intervalos ni pruebas. La dispersión entre semillas no mide incertidumbre entre mercados o calendarios. [SVG descargable](../_static/e3_evidence_close_20261120/pilot/generated/pilot_rmse_and_dependence.svg).
```

### Adecuación posible con las predicciones disponibles

El {download}`reanálisis reproducible <../_static/e3_evidence_close_20261120/pilot/analyze_saved_pilot.py>` añade {download}`errores y correlaciones por trayectoria <../_static/e3_evidence_close_20261120/pilot/generated/residual_diagnostics.csv>` y {download}`cortes mensuales completos <../_static/e3_evidence_close_20261120/pilot/generated/monthly_diagnostics.csv>`. Son diagnósticos posteriores al resultado, sin remuestreos ni valores p. En la completa, el sesgo medio `predicción − objetivo` es −0,00026448 y la correlación descriptiva media del residuo a siete días es −0,320. Una correlación pequeña en otro desfase no prueba independencia de ventanas solapadas.

Los dos meses se conservan juntos: en septiembre —24 fechas—, las RMSE medias son 0,002012253 para la completa y 0,002022917 sin CRPS; en octubre —31—, 0,002216532 y 0,002112206. No se selecciona septiembre para declarar éxito: la decisión usa las 55 fechas y la regla previa. Las cinco recetas seleccionaron la tasa máxima de la cuadrícula, 0,003; esto delimita la sensibilidad estudiada, sin autorizar una búsqueda nueva guiada por esta evaluación.

La {download}`matriz de alcance de adecuación <../_static/e3_evidence_close_20261120/pilot/generated/adequacy_scope_matrix.csv>` identifica límites que no pueden cerrarse con estos archivos. Faltan predicciones de ajuste y riesgos comparables para completar las Tablas A/B y la brecha de sobreajuste. Los CRPS_U originales y su agregación sí están disponibles, pero faltan leyes atómicas por observación e intervalos guardados para recalcular independientemente esa puntuación o certificar calibración. Tampoco se reconstruyen retrospectivamente margen formal, potencia prospectiva, tres pruebas apropiadas por contraste o una familia confirmatoria. Los tres conjuntos adicionales de 8.18 se mantienen como otra auditoría; no fueron evaluados por V3 aquí.

El {download}`verificador aritmético independiente <../_static/e3_evidence_close_20261120/pilot/verify_saved_pilot.py>` y su {download}`recibo <../_static/e3_evidence_close_20261120/pilot/generated/independent_arithmetic_verification.json>` usan una segunda formulación con la biblioteca estándar. Esta actualización declara asistencia de IA y ejecuta **cero entrenamientos, inferencias o remuestreos nuevos**. Se mantienen **`R1_accredited=false` y `R3_accredited=false`**; cualquier modificación posterior deberá registrarse como diseño nuevo, preservando este resultado, su protocolo y el conjunto de ensayos.
