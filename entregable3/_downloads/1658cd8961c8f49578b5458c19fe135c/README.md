# Auditoría exploratoria del límite observable de σ₇

Se midió una limitación concreta del benchmark conservado: **20.007 de 86.944 salidas predictor–fecha (23,01 %) quedan por debajo del mínimo de σ₇ compatible con la historia conocida**. La proyección física mejora los errores de varios rivales sin modificar pesos. La ventaja RMSE descriptiva de E3 frente al mejor rival disponible pasa de **24,10 % a 10,33 %**, y aparece un resultado adverso: predecir únicamente el piso observable obtiene menor MAE que E3.

Este análisis se ejecutó el 18 de noviembre de 2026, después de conocer resultados y calendario. Es un **nuevo diagnóstico y comparador exploratorio posterior**, sin preregistro, sin nuevos ajustes, sin nuevos forwards y sin contrastes de hipótesis. No modifica las métricas históricas, los 48 contrastes, las recetas, el catálogo del dashboard o la familia científica original. La identidad algebraica y la proyección descritas aquí no se presentan como una contribución original ni acreditan R1/R2/R3.

## Cantidad observable y propiedad

Para una fecha objetivo `t`, la entrada contiene siete cierres diarios completos `C[t−7],…,C[t−1]`. Sus seis retornos logarítmicos conocidos son `r[t−6],…,r[t−1]`, con media `μ₆`. El séptimo retorno `r[t]` todavía no se conoce. Para el objetivo poblacional de siete retornos, sin anualización:

\[
A_t=\frac{1}{7}\sum_{i=t-6}^{t-1}(r_i-\mu_6)^2,
\qquad
\sigma_{7,t}^2=A_t+\frac{6}{49}(r_t-\mu_6)^2.
\]

Por tanto, `b[t]=sqrt(A[t])` es un piso alcanzable cuando el retorno desconocido coincide con `μ₆`. El denominador de `A` es **siete**, aunque se utilicen seis retornos. `A` tiene unidades de varianza; el predictor de σ₇ se compara y proyecta contra **sqrt(A)**. La implementación científica previa está en `src/e3_innovation_model.py::load_innovation_data`, identificada por commit y SHA256 en el registro de admisión.

La transformación exploratoria es `p⁺[t]=max(p[t],b[t])`. No usa el objetivo realizado, el retorno futuro, una estadística ajustada o hiperparámetros seleccionados. Si `y[t]≥b[t]` y `p[t]<b[t]`, entonces

\[
(p_t-y_t)^2-(b_t-y_t)^2=(b_t-p_t)(2y_t-b_t-p_t)\ge0,
\qquad
|p_t-y_t|-|b_t-y_t|=b_t-p_t\ge0.
\]

Cuando `p[t]≥b[t]`, la salida queda idéntica. Esta propiedad demuestra que la proyección no puede aumentar el error cuadrático o absoluto **por observación** cuando la premisa es válida. No garantiza calibración, generalización, superioridad estadística o novedad. Una predicción inferior al piso señala incoherencia con este objetivo y esta historia; no prueba por sí sola fuga de información o un defecto general del algoritmo.

## Resultados de todos los sistemas

Cada red conserva sus diez semillas y cada clásico una trayectoria. RMSE es la **media de las RMSE individuales**, en unidades originales de σ₇: no la RMSE de un ensamble ni la raíz de un MSE agrupado entre semillas. MAE también se promedia por predictor. La tasa usa todas las parejas predictor–fecha de cada sistema; no son observaciones independientes. Se cuentan violaciones mayores que `1e−12` en σ₇ y se conserva también el conteo estricto. La proyección aplica el máximo exacto, sin esa tolerancia.

| Sistema | R | Salidas bajo piso | Tasa | RMSE antes | RMSE proyectada | MAE antes | MAE proyectada |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN-LSTM aditiva E3 | 10 | 0 | 0,00 % | 0,0029394324 | 0,0029394324 | 0,0018928273 | 0,0018928273 |
| FCN-1D | 10 | 1.624 | 26,71 % | 0,0045743280 | 0,0035379610 | 0,0030598548 | 0,0023239460 |
| ResNet-1D | 10 | 1.349 | 22,19 % | 0,0044769077 | 0,0037138665 | 0,0031438411 | 0,0025468371 |
| Inception-1D | 10 | 1.318 | 21,68 % | 0,0043002191 | 0,0036861101 | 0,0030926720 | 0,0025621682 |
| LSTM-autoencoder con predictor | 10 | 981 | 16,13 % | 0,0038728895 | 0,0035990849 | 0,0028608472 | 0,0025607201 |
| TimesNet | 10 | 1.281 | 21,07 % | 0,0046109645 | 0,0040809603 | 0,0033265135 | 0,0027771081 |
| TSMixer | 10 | 2.704 | 44,47 % | 0,0048215434 | 0,0032780684 | 0,0034346292 | 0,0020128791 |
| TiDE | 10 | 2.268 | 37,30 % | 0,0054397475 | 0,0039233521 | 0,0038649290 | 0,0025057351 |
| TimeMixer | 10 | 1.003 | 16,50 % | 0,0044233272 | 0,0039824080 | 0,0033945414 | 0,0029705000 |
| PatchTST | 10 | 1.343 | 22,09 % | 0,0100930508 | 0,0073159469 | 0,0075694456 | 0,0056749799 |
| iTransformer | 10 | 1.684 | 27,70 % | 0,0052506796 | 0,0039857038 | 0,0039807373 | 0,0028316520 |
| MLP16 | 10 | 1.763 | 29,00 % | 0,0045228734 | 0,0037776951 | 0,0034953219 | 0,0026439879 |
| LSTM32 | 10 | 1.218 | 20,03 % | 0,0046438265 | 0,0042752241 | 0,0036854745 | 0,0031877554 |
| GRU32 | 10 | 1.166 | 19,18 % | 0,0039499752 | 0,0037284146 | 0,0030048199 | 0,0026550257 |
| Persistence | 1 | 0 | 0,00 % | 0,0040855959 | 0,0040855959 | 0,0023531639 | 0,0023531639 |
| SeasonalNaive | 1 | 248 | 40,79 % | 0,0121485679 | 0,0090815936 | 0,0085205790 | 0,0052311792 |
| ARIMA | 1 | 57 | 9,38 % | 0,0039216248 | 0,0038396367 | 0,0024997096 | 0,0023717782 |

Las trece redes E2 tienen violaciones; E3 y Persistence no tienen ninguna y sus salidas quedan idénticas. En los 86.944 pares, el máximo aumento observado de pérdida cuadrática y absoluta fue cero. Las magnitudes de incumplimiento por semilla y sistema están en los CSV; la mayor fue `0,0517706791` en σ₇ para PatchTST.

El rival con menor RMSE bruta es el autoencoder. Tras aplicar la misma proyección a **todos** los sistemas, el rival con menor media es TSMixer: `0,0032780684`, frente a `0,0029394324` de E3. La brecha contra el mejor rival disponible disminuye de `0,0009334571` a `0,0003386360`: una reducción descriptiva del **63,72 %**. Este porcentaje cambia de rival entre las dos comparaciones; no es una fracción causal de mejora atribuible a una pieza de E3. Los cambios de rango completos permanecen en `per_model.csv`.

El control adicional **sólo piso** predice `sqrt(A)` sin aprendizaje: RMSE `0,0035275805`, un 20,01 % mayor que E3, pero MAE `0,0018227552`, un **3,70 % menor** que E3. Se conserva este resultado adverso junto al favorable de RMSE. Ese control está separado del catálogo de diecisiete sistemas y no se añade retrospectivamente a sus contrastes. La diferencia de comportamiento entre RMSE y MAE exige examinar errores y colas en una evaluación futura; este cálculo no identifica su causa.

El diagnóstico cuantifica una limitación específica de las salidas históricas y demuestra que una regla algebraica simple puede aprovechar parte de la ventaja sin cambiar las redes rivales. Sigue siendo necesario aislar el mecanismo defendido, controlar políticas de ajuste y presupuesto, dimensionar potencia y evaluar conjuntos adicionales. El entrenamiento E2 de 655 ejemplos frente a 532 de E3 y el calendario conocido mantienen sus límites después de proyectar.

## Admisión y separación temporal

El ejecutor reutiliza `experiments/e3_estadistica_posterior_20261117/inputs.py`, con SHA256 fijado, para verificar once archivos inmutables y sus relaciones: 133 predictores E2, diez E3, las 608 fechas exactas, objetivos idénticos, métricas guardadas y las 81 trayectorias históricas preservadas. La cohorte NPZ se verifica contra SHA256 `186d473fdfdf5cd4413eb1200dfcc8fea65ba9f6e7519b722eb9b1ae9f56a683` y sólo se abren `evaluation_raw`, `evaluation_y_original` y `evaluation_dates`.

Se comprueban geometría OHLCV, fechas consecutivas y solapamiento cronológico exacto de seis barras entre ventanas. Los 608 objetivos coinciden exactamente con el panel fijado y ninguno es inferior al piso; la menor distancia objetivo–piso es `4,1088e−10`. El piso coincide exactamente con el guardado por E3. Una fórmula independiente basada en diferencias por pares de los seis retornos verifica `A` con discrepancia máxima `8,6736e−19` en varianza.

En un bloque **separado de la construcción del piso**, se toma el cierre de la siguiente ventana para reconstruir 607 etiquetas ya congeladas a partir de ocho cierres; la diferencia máxima es `1,2577e−16` en σ₇. También se verifica la identidad de varianza con su retorno realizado. Ese cierre futuro sólo audita etiquetas y orientación: nunca entra en `observable_variance_floor` o `project`. No existe una ventana siguiente para reconstruir de esta forma el último objetivo, `2026−08−31`; su admisión depende del NPZ y panel fijados. No se reconstruye un cierre futuro a partir de la etiqueta.

La asignación de los siete cierres a `t−7…t−1` hereda el contrato y la auditoría de la cohorte inmutable. Estas comprobaciones no vuelven a descargar datos del exchange ni cierran los tres controles parciales del cierre anterior. Tampoco convierten días solapados o inicializaciones en mercados independientes.

## Reproducción y archivos

Desde la raíz del repositorio, en Python con NumPy 2.3.5 y con los insumos fijados ya disponibles:

```bash
python experiments/e3_remediacion_20261118/constraint_audit/test_constraint_audit.py \
  --receipt experiments/e3_remediacion_20261118/constraint_audit/test_receipt.json
python experiments/e3_remediacion_20261118/constraint_audit/run_audit.py \
  --data-dir /workspace/e3_statistical_audit/data \
  --cohort-npz /workspace/e3_frozen_eval/inputs/current_cohort_arrays.npz
```

Las rutas de entrada se pueden cambiar conservando sus SHA256. El ejecutor no descarga ni modifica insumos. En otro equipo, los localizadores inmutables y hashes de los once archivos y del NPZ están en `results/sources_and_admission.json`. El código de admisión versionado contiene además los localizadores originales de predicciones. No se requiere Torch, pesos de modelos o hardware de entrenamiento.

- `results/per_predictor.csv`: 143 filas; incumplimiento, magnitud y errores antes/después de cada predictor.
- `results/per_model.csv`: 17 filas; medias, DE entre semillas cuando existen, tasas, rangos y brechas descriptivas frente a E3. Una trayectoria determinista no recibe DE ficticia.
- `results/floor_by_date.csv`: las 608 fechas, soportes pasados, objetivo, varianza observable y piso de σ₇.
- `results/summary.json`: lectura de resultados y declaraciones explícitas de alcance.
- `results/sources_and_admission.json`: fuentes, hashes, soporte temporal y comprobaciones de etiquetas.
- `results/execution.json` y `results/manifest.json`: entorno, comando y huellas de productos.
- `test_receipt.json`: nueve pruebas numéricas sintéticas aprobadas; distinguen divisor, unidades, forma, orientación, futuro excluido, estimando, proyección y premisa del teorema. Son pruebas de software, sin p-valores.

Los datos crudos y las trayectorias fuente no se duplican aquí. Este paquete local no añade permisos de redistribución o una licencia a sus fuentes. No existe una reproducción humana independiente que atribuir a esta ejecución automatizada.
