# Auditoría empírica retrospectiva de input_only_mse

El CSV confirma una señal exploratoria favorable para estudiar `input_only_mse`: RMSE medio 0.0020515608242855487, mejora 2.759522465% respecto de `global_mse`, cuatro victorias de cinco semillas. Esto no es una confirmación independiente ni una autorización para promover una ablación. La decisión original de V5 completo sigue siendo **ABANDON_CURRENT_SEOI_RECIPE**. La suficiencia de la contribución y el dictamen global CONTINUAR/DESCARTAR requieren la auditoría técnica y la guía; este informe no los sustituye.

## Fuentes, integridad y alcance ejecutado

- ZIP local: `/workspace/e3_v5_pilot_results_readout_20261121/public_results.zip`; SHA-256 `beabedeb63ff10324b310ba8b86105505c3be783216e38770a092a7be438a17c`; 110 miembros.
- Código original declarado: `c4ca86f1431afeaa958ff0b544b453ab17a4d7b8`. Resultados declarados: `05c1a637f13356dd1df6b8c5aa8f4676dda61c12`, `runs/e3_seoi_v5/run-01`. La vinculación remota del commit debe comprobarse en el expediente principal; este script comprueba el ZIP fijado y su consistencia interna.
- 1035 comprobaciones internas: **PASS**; los siete archivos del manifiesto de resultados coinciden en bytes y hash. SHA de protocolo, plan serializado y plan/admisión canónicos verificados. Las métricas se recalculan desde `predictions.csv`; los JSON previos se usan solo para contrastar después los resultados.
- 30 ajustes históricos del piloto, seis recetas × cinco semillas. 1.760 predicciones guardadas: 1.650 neurales y 110 analíticas, sobre las mismas 55 fechas (2024-09-07 a 2024-10-31). Sin filas duplicadas, faltantes, valores no finitos ni objetivos discordantes. Los identificadores enteros de semilla no son fechas de calendario.
- Ejecutado en esta auditoría: lectura, aritmética, validación de recibos y escritura de tablas. **0 entrenamientos, 0 forwards, 0 predicciones nuevas, 0 bootstrap y 0 pruebas de significancia**. El script usa solo la biblioteca estándar.

`protocol.json` conserva `execution_status=prepared_no_training_executed` y `new_real_data_fits=0`, como propuesta fuente anterior a la ejecución. `prepared.json` y `fit_status.assessment_started_by_run=false` son recibos de fases distintas. La evidencia de cierre es `pilot_result`, los 30 `job.json`, las curvas y los CSV; esos campos históricos no se reinterpretan como estado final. El protocolo denomina el control `floor_plus_fit_mean_excess` y el plan/CSV `floor_plus_mean_excess`; se conserva la identidad efectiva del CSV y se registra la diferencia de nombre.

## Métricas y comparaciones emparejadas

La métrica primaria conservada es la media de RMSE individuales por semilla. No es la raíz del MSE combinado ni el error de un ensamble. MAE se promedia con el mismo criterio. Para los controles analíticos hay una sola trayectoria, sin SD entre semillas.

| Rango | Receta | RMSE medio | SD muestral RMSE entre semillas | MAE medio |
|---:|---|---:|---:|---:|
| 1 | input_only_mse | 0.00205156082428555 | 7.24269349297086e-05 | 0.00154849623028018 |
| 2 | global_mse | 0.00210978069656621 | 9.959767437735801e-06 | 0.0016536738452131 |
| 3 | floor_plus_mean_excess | 0.00211423094065699 | — | 0.00181348280779959 |
| 4 | global_crps | 0.00213614124756936 | 2.672367576183707e-05 | 0.00156449438084174 |
| 5 | seoi_mse | 0.00216096042003114 | 3.1428465535941026e-05 | 0.00170206158812624 |
| 6 | additive_mse | 0.00217139425537327 | 1.6936235906435556e-05 | 0.00183781396385986 |
| 7 | output_only_mse | 0.00224236869842473 | 7.50948679027169e-06 | 0.0017718685032642 |
| 8 | floor_only | 0.00279034061936247 | — | 0.0018317657439326 |

Cada diferencia compara las mismas fechas y, para dos recetas neurales, la misma semilla. Ganancia positiva favorece `input_only_mse`; diferencias negativas de error también la favorecen.

| Control | Δ RMSE medio | Ganancia RMSE | Victorias neurales /5 | Δ MAE medio |
|---|---:|---:|---:|---:|
| seoi_mse | -0.000109399596 | 5.062545% | 5 | -0.000153565358 |
| output_only_mse | -0.000190807874 | 8.509211% | 5 | -0.000223372273 |
| global_mse | -5.82198723e-05 | 2.759522% | 4 | -0.000105177615 |
| global_crps | -8.45804233e-05 | 3.959496% | 5 | -1.59981506e-05 |
| additive_mse | -0.000119833431 | 5.518732% | 4 | -0.000289317734 |
| floor_only | -0.000738779795 | 26.476330% | no aplica | -0.000283269514 |
| floor_plus_mean_excess | -6.26701164e-05 | 2.964204% | no aplica | -0.000264986578 |

`metrics_recomputed.csv` contiene RMSE y MAE de cada trayectoria. `all_pairwise_seed_differences.csv` y `all_pairwise_summary.csv` cubren las 56 comparaciones dirigidas entre las ocho recetas/controles, sin excluir rivales. Cuando hay un control analítico se reutiliza su única trayectoria: las cinco diferencias no convierten ese control en cinco entrenamientos.

La SD de RMSE de `input_only_mse` es 7.271950 veces la de `global_mse`. Esa variabilidad expresa inicialización/optimización condicionada al único panel de fechas; no estima por sí sola la incertidumbre temporal. La media de diferencias frente a global es -5.82198722807e-05; su SD entre semillas es 7.1100810508e-05. La ventaja no es uniforme: una semilla pierde frente a global y una frente a aditiva.

## Selección frente a evaluación expuesta

La selección comprende 54 fechas de 2024-07-08 a 2024-08-30; assessment comprende otras 55 fechas, ya expuestas antes del piloto. Se excluyeron siete fechas entre fit y selección, y siete entre selección y assessment (14 en total). Los 532 registros de fit y las purgas constan en admisión; sus arrays no están en el ZIP. El RMSE de selección se recupera de `sqrt(min(selection_mse_scaled))*y_scale` con desempate por época más temprana. Coincide con cada `job.json`; no se reconstruyen predicciones de selección ausentes.

| Receta | RMSE medio selección | Rango selección | RMSE assessment | Rango assessment neural | Épocas seleccionadas |
|---|---:|---:|---:|---:|---:|
| seoi_mse | 0.00384978598284 | 1 | 0.00216096042003 | 4 | 15–40 |
| additive_mse | 0.00387449625199 | 2 | 0.00217139425537 | 5 | 13–40 |
| output_only_mse | 0.0039996842934 | 3 | 0.00224236869842 | 6 | 3–30 |
| global_mse | 0.00414536693107 | 4 | 0.00210978069657 | 2 | 39–40 |
| global_crps | 0.00418483598762 | 5 | 0.00213614124757 | 3 | 39–40 |
| input_only_mse | 0.00430490508367 | 6 | 0.00205156082429 | 1 | 2–39 |

**`input_only_mse` pasa del último puesto de selección al primero de assessment.** Todas las recetas tienen menor error en assessment, un cambio de dificultad o régimen compatible con las fechas distintas; no demuestra que la selección generalice mejor. Las curvas y las diferencias se documentan en `selection_assessment_recomputed.csv`, `selection_assessment_summary.csv` y `selection_assessment_input_only_pairs.csv`. La candidata se eligió después de examinar el desarrollo; este análisis posterior no es un preregistro ni elimina el sesgo de selección.

## Sensibilidad descriptiva a fechas y volatilidad realizada

Particiones posteriores al resultado: meses; primeras 27/últimas 28 fechas; ocho bloques consecutivos de siete días (último de seis); cuartiles del objetivo realizado `sigma7`. Los puntos de corte de los cuartiles (interpolación lineal) son [0.015187003999136462, 0.017914339862395136, 0.021630795383108012]. Se usan las mismas fechas de cada partición para todas las recetas y se promedian RMSE individuales, preservando la definición de la métrica. `date_regimes_posthoc.csv` permite reconstruir cada subconjunto.

| Partición | Subconjunto | Fechas | Ganancia de input_only frente a global | Victorias /5 |
|---|---|---:|---:|---:|
| month | 2024-09 | 24 | 2.909961% | 5 |
| half | first_27 | 27 | 4.456942% | 4 |
| realized_target_quartile | Q4 | 14 | 3.210139% | 4 |
| realized_target_quartile | Q2 | 14 | 1.275427% | 4 |
| realized_target_quartile | Q3 | 13 | 6.895579% | 4 |
| realized_target_quartile | Q1 | 14 | -0.590997% | 3 |
| month | 2024-10 | 31 | 2.653678% | 4 |
| half | last_28 | 28 | 1.154519% | 3 |

La ventaja es más pequeña en la segunda mitad y se invierte en el cuartil de menor volatilidad realizada. Las particiones contienen pocas fechas y no son resultados independientes. El cuartil utiliza el objetivo futuro realizado: **es retrospectivo y no define una regla operativa causal**. La escala de la ventana conocida no está en las predicciones; no se inventan regímenes ex ante. Todos los bloques y controles están en `subset_*`.

Al quitar una sola fecha, la ganancia media RMSE frente a global oscila entre 2.132821% y 5.347518%; al quitar un bloque consecutivo, entre 2.307808% y 4.059890%. Se recalcula la métrica tras la exclusión: no es una nueva selección ni una validación cruzada. Véase `leave_out_sensitivity.csv`.

Los 275 pares semilla-fecha comparten objetivos y ventanas temporales. La autocorrelación Pearson de lag 1 del objetivo es 0.767205; los diferenciales diarios promediados sobre semillas también están en `descriptive_dependence.csv`. Son descripciones, sin tamaño efectivo inferido. No se tratan 275 observaciones como independientes, no se calcula potencia confirmatoria con cinco semillas y un panel corto, ni se añaden p-valores o intervalos convencionales que ignoren selección y dependencia. `date_loss_differences.csv` conserva la contribución diaria de cada control.

## Fallos, presupuesto y recuperación

Los 30 trabajos presentan `completed_validated_fit`, una entrada de intento por trabajo y 0 épocas interrumpidas; `failures.json` está vacío. Los motivos de parada fueron {'early_stopping': 17, 'max_epochs': 13}. Se verifican el número de épocas, época seleccionada, pasos Adam y concordancia del inventario de hashes de checkpoint con cada trabajo. Los pesos no están exportados: no se afirma haber verificado sus bytes ni reejecutado los modelos.

- Épocas totales: 965; pasos de optimizador comprometidos: 16405.
- Actividad registrada de ajuste: 76.023488776 s; IO de checkpoint registrado: 10.004634499 s. Máximo por ajuste: 3.598746846 s, frente a 120 s nominales; techo nominal de 30 ajustes: 3600 s. No equivale al coste total de VM ni incluye instalación, preparación, transporte y respaldo.
- `remote_restore.json` declara restauración `3badbb2e6c71bff6b765cf8cd6535e1fa92f7766` en `2026-11-21T22:03:31.974859+00:00`, con sesión previa cerrada, identidad/semilla conservadas, protocolo/curvas/pesos sin modificación y sin cómputo remoto no observado reportado. `github_session.json` termina cerrado en `assessment_finished`. `fit_status` y resultado dicen `budget_comparability_verified=true`; la contabilidad disponible es consistente con esa declaración.
- El propio recibo advierte que una pérdida abrupta puede ocultar cómputo no subido. `unuploaded_progress_may_be_lost=true` es una posibilidad general; `last_saved_session_open=false` indica que esta recuperación no documenta rollback de una sesión abierta. No se infiere trabajo físico desconocido ni se convierte el respaldo en un libro externo completo. Tampoco se reconstruye el número de forwards por duplicados o por filas únicas.

`resources_by_recipe.csv`, `jobs_audited.csv` y `HASH_AND_RECOVERY_AUDIT.json` contienen el detalle. Los tiempos cortos son los registrados en CPU, no una estimación transportable para el futuro protocolo R3.

## Soporte, evidencia histórica y límites de lo verificable

La predicción `floor_only` permite contrastar el piso publicado en cada fecha. Hay 0 medias puntuales por debajo del piso con tolerancia 1e-12 (`point_support_audit.csv`). Esto no verifica la distribución atómica ni su calibración: no se exportaron masas, posiciones, CRPS, PIT, cobertura o intervalos. Una pérdida MSE por sí sola tampoco acreditaría calibración.

Las declaraciones históricas preservadas en `protocol.prior_evidence` son 796 trabajos aditivos, V3 con 30 ajustes, perjuicio 2,7043119888% y NO_GO, y V4 con 20 ajustes y ABANDON_CURRENT_V4_RECIPE. Sus resultados originales no están en este ZIP; esta auditoría no los reetiqueta ni afirma verificarlos de nuevo. El piloto V5 conserva su candidato original `seoi_mse`, mejora negativa frente a `input_only_mse`, 0/5 victorias y MAE por encima del máximo; las tres condiciones de continuación originales fracasan. No se aplican esas reglas retrospectivamente a la ablación ganadora.

La coincidencia numérica con `RESULT_READOUT.json` e `INPUT_ONLY_COMPARISON.json` es completa a tolerancia de coma flotante. No aparecen discrepancias numéricas materiales. La diferencia `0.0020515608242855483`/`0.0020515608242855487` en promedios obedece al orden/suma de coma flotante y está por debajo de 1e-18. Identificadores de commit, protocolo y hashes son trazabilidad del piloto, **no preregistro de la selección posterior de input_only**.

Reproducción: `python analysis/audit_saved_results.py --zip /ruta/public_results.zip --out analysis`. El script se niega a aceptar un ZIP con SHA distinto y termina con error ante discordancias comprobables. `ARTIFACT_HASHES.json` fija los archivos generados. El dictamen global debe conservar estos límites; una decisión de continuar solo abriría preparación de una candidata independiente y confirmación prospectiva, con R3 todavía no acreditada.
