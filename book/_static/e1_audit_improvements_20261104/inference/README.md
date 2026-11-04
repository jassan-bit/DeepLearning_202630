# Inferencia complementaria del test final: D7

El plan se registró **después de observar los resultados del test**, antes de estos cálculos. Es un suplemento post-hoc/exploratorio, no una confirmación prospectiva. Se reutilizan las predicciones de MLP/LSTM/CNN1D con D7 y diez semillas, seleccionadas previamente por CV, y la persistencia determinista. No se cambian el baseline MLP/D7, los modelos, el periodo ni sus pesos.

`paired_comparisons.csv` contiene seis comparaciones bilaterales de RMSE por semillas, Holm-6, IC t marginales, d_z e IC percentile de d_z por remuestreo pareado de semillas. Las seis diferencias están condicionadas al mismo calendario. La persistencia se resta como puntuación fija; no representa diez entrenamientos independientes. Shapiro y Q–Q no certifican normalidad con diez pares.

`wilcoxon_sensitivity.csv` conserva seis sensibilidades de localización signed-rank con Holm-6 separado. Solo usa la distribución exacta cuando las diferencias son distintas de cero y sus valores absolutos no tienen empates; de otro modo conserva estado no definido. Requiere simetría para interpretar un parámetro de localización y no contrasta la misma media que t. No se elige la prueba que produzca rechazo; ninguna de estas referencias constituye confirmación independiente de las demás.

`dm_tests.csv` compara pérdidas cuadráticas por fecha, promediadas sobre semillas; usa Bartlett/HAC14, corrección HLN de h=1 y Holm-6 para los seis pares principales. HAC7/28 son sensibilidades descriptivas y no se selecciona la variante favorable. La pérdida media por fecha no es el error de una predicción ensemble ni la RMSE media por semilla.

`ljung_box.csv` diagnostica los residuos de la semilla designada4101 de las tres redes, lags7/14/28, model_df0 y Holm-9. No se resta el número de parámetros de la red como grados de libertad ARMA. Las etiquetas sigma7 solapadas, la deriva y la aproximación asintótica limitan la calibración.

`date_bootstrap.csv` conserva 2.000 muestras de bloques circulares pareados de fechas, L14 principal y L7/28 de sensibilidad, para ΔRMSE/ΔMAE y reducciones relativas. Recalcula primero cada métrica por semilla y después su media. No remuestrea semillas, reajusta modelos ni identifica incertidumbre entre mercados. Los intervalos de esta tabla son marginales, no simultáneos por llevar p-valores Holm en otra tabla.

`relative_differences.csv` fija MLP como mejor base y LSTM como peor base según la métrica primaria de **CV**, sin elegir referentes por el test; también informa persistencia. Sus reducciones son100×(error medio de referencia−error medio del modelo)/error medio de referencia. `design_sensitivity.csv` calcula una MDE frente a cero desde la dispersión observada, α de planificación0,05/6 y potencia hipotética0,80. No es piloto prospectivo, potencia lograda ni dimensión para demostrar una mejora superior a un margen práctico.

Todos los archivos son agregados. No se exportan predicciones individuales, precios diarios, residuos por fecha, pesos ni paquetes completos. Rechazar autocorrelación/iid y demostrar precisión comparativa son preguntas diferentes. No se ejecuta TOST/ROPE ni se concluye equivalencia por ausencia de significancia.
