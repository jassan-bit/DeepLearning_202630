# 5. Benchmark de los Modelos Base y Definición del Baseline

El benchmark comparará conjuntamente los **tres modelos de Deep Learning seleccionados desde la literatura**, bajo condiciones reproducibles, justas y comparables. Su resultado será identificar una referencia mínima de desempeño que un modelo avanzado deberá superar con evidencia estadística y relevancia práctica.

```{important}
Este capítulo establece el protocolo y las tablas requeridas. Todavía no hay tres modelos seleccionados y entrenados ni resultados de test: no existe un ganador o baseline experimental medido. La persistencia es una referencia contextual y no sustituye uno de los tres modelos obligatorios.
```

## 5.1. Comparación Tabular

### 5.1.1. Alcance y unidad de comparación

Se elaborará una tabla por configuración comparable: ventana de entrada $D$, ventana de volatilidad $n$, horizonte, periodo y conjunto de evaluación. Se mantendrán las mismas muestras, exclusiones, semillas y definición de métricas entre modelos.

Las ventanas de volatilidad definen objetivos diferentes. No se elegirá un supuesto mejor modelo comparando directamente la RMSE de una volatilidad de siete días con la RMSE de otra de veintiocho días. El ranking y las diferencias se calcularán dentro de cada comparación bien definida.

El horizonte principal es un día; se reportarán 2025 y 2026 por separado en el test previsto. Los valores de la literatura servirán como contexto documentado, sin mezclarlos con mediciones propias sobre otros datos o escalas.

### 5.1.2. Tabla obligatoria de desempeño y costo

La tabla siguiente es una **plantilla pendiente de resultados**, no una comparación ejecutada. M1, M2 y M3 son identificadores provisionales, no arquitecturas seleccionadas:

| Modelo base | Ejecuciones válidas | RMSE, media ± DE | MAE, media ± DE | Otras métricas | IC 95 % y método | Tiempo de entrenamiento | Tiempo de inferencia | Parámetros |
|---|---|---|---|---|---|---|---|---|
| M1 — por seleccionar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |
| M2 — por seleccionar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |
| M3 — por seleccionar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |

Se conservará un formato largo con un intervalo por cada métrica que lo requiera; la columna resumida de intervalos no se interpretará como un único IC aplicable a todas las métricas. Las métricas adicionales incluyen MAPE, MSE, R² y MASE, con los casos no definidos y la cobertura documentados según §4.4.

Los tiempos indicarán unidad, hardware, tamaño de lote, calentamiento y si incluyen preprocesamiento. El número de parámetros distinguirá entrenables y no entrenables. No se sustituirá una medición por una estimación teórica no identificada como tal.

### 5.1.3. Incertidumbre y trazabilidad

Cada fila se vinculará a configuraciones, código, manifiestos, logs y predicciones. Se reportarán media, desviación estándar **muestral**, número de ejecuciones e IC del 95 %, con método y supuestos explícitos.

Se diferenciará incertidumbre entre semillas sobre el mismo conjunto de evaluación de incertidumbre temporal por bloques. Los 366 folds no se tratarán como semillas independientes. Para RMSE y R² se usará la agregación de predicciones de §4.1.4 y §4.4; no un promedio inadecuado de métricas de folds unitarios.

## 5.2. Ranking de Modelos Base

### 5.2.1. Criterio principal y cierre de selección

El criterio propuesto es **menor RMSE media**, con métricas secundarias y costo como información complementaria. Se fijarán antes del test las reglas de selección y desempate; no se elegirá una regla después de observar qué modelo favorece.

El modelo de referencia operativo se seleccionará con validación de desarrollo y se congelará antes de test. La tabla final puede ordenar de forma descriptiva el desempeño de los tres modelos en test, pero ese orden no autoriza reabrir la selección. Si difiere del orden de validación, se reportarán ambos y esa discrepancia, sin reemplazar silenciosamente la referencia congelada.

Todavía no se asignan posiciones: los modelos no han sido entrenados. Un intervalo solapado o una diferencia pequeña no define por sí solo un empate formal; se requiere la comparación pareada y la interpretación de relevancia práctica.

### 5.2.2. Precisión, costo y complejidad

Se examinarán dos relaciones: error frente a costo computacional y error frente a complejidad. Se mostrarán RMSE contra tiempos y número de parámetros con sus condiciones de medición, conservando incertidumbre cuando corresponda.

Una menor RMSE no se presentará automáticamente como la mejor decisión operacional si exige un costo o complejidad desproporcionados. Se identificarán compromisos y, cuando los datos lo permitan, soluciones no dominadas: un modelo está dominado si otro obtiene menor o igual error con menor o igual costo y mejora estricta en al menos una dimensión.

La complejidad y el tiempo no se confunden: menos parámetros no garantiza menor latencia. No se dibujarán curvas o diagramas de Pareto ficticios mientras falten mediciones.

## 5.3. Definición del Baseline y Margen de Mejora Disponible

### 5.3.1. Baseline de referencia

Se identificará explícitamente el modelo seleccionado mediante desarrollo, su configuración, alcance y desempeño final con **media, desviación estándar e IC del 95 %**. Se documentará también el orden descriptivo de los tres modelos en la evaluación final y cualquier diferencia respecto a la selección inicial.

El baseline no es solo el nombre de una familia: incluye datos, objetivo, horizonte, pipeline, restricciones, configuración y protocolo de evaluación. Con los resultados actuales no puede afirmarse que MLP, RNN, LSTM, GRU o CNN sea el baseline ganador.

### 5.3.2. Diferencias relativas de RMSE y MAE

Para un error de referencia positivo se calculará:

$$
\Delta_{\mathrm{RMSE}}(m;\mathrm{ref})=
\frac{\mathrm{RMSE}_{\mathrm{ref}}-\mathrm{RMSE}_{m}}
{\mathrm{RMSE}_{\mathrm{ref}}}\times100\%.
$$

Se usará una expresión equivalente para MAE. Un valor positivo indica reducción del error frente a la referencia, cero igualdad y un valor negativo mayor error. Si el error de referencia es cero, el porcentaje no está definido y no se reemplazará su denominador por un epsilon.

El benchmark informará las diferencias respecto al **peor** y al **mejor** modelo de la misma tabla comparable. Frente al mejor error, los otros modelos tendrán mejoras negativas o nulas: no se cambiará el signo para aparentar ganancias. Esas referencias descriptivas se distinguirán del baseline operativo congelado antes de test.

Se declarará si el porcentaje se calcula sobre errores agregados o como media de porcentajes pareados; son operaciones diferentes. Las comparaciones estadísticas utilizarán las métricas por corrida y las predicciones correspondientes, no solo cifras redondeadas de una tabla.

### 5.3.3. Tamaño de efecto

Cada diferencia relevante se acompañará de una medida de efecto apropiada y su definición. Para diferencias pareadas por ejecución $d_s=m_{A,s}-m_{B,s}$ puede informarse:

$$
d_z=\frac{\bar d}{s_d},
$$

con signo, denominador y unidad de emparejamiento identificados. Si $s_d=0$, el cociente no está definido. No se sustituirá la desviación de las diferencias por la desviación de un solo modelo. Si se usa una medida basada en rangos, se especificará el cálculo y el tratamiento de empates.

El efecto entre semillas mide esa variabilidad condicionada a los mismos datos. Para efectos sobre errores temporalmente dependientes se justificará el método de bloques o diseño correspondiente. No se atribuirá independencia a todos los días por disponer de muchas predicciones.

### 5.3.4. Inferencia sobre diferencias y multiplicidad

Se realizarán comparaciones pareadas entre los tres modelos, con contrastes fijados previamente. Se reportarán **diferencia media, IC del 95 %, estadístico y p-value numérico**, tanto sin ajustar como ajustado por comparaciones múltiples cuando proceda.

Se declarará la familia de comparaciones y la corrección, por ejemplo **Holm o Bonferroni**, antes de realizar afirmaciones confirmatorias. Tres modelos producen tres pares por configuración; si se extienden las conclusiones a varios objetivos, entradas u horizontes, se declarará esa ampliación y su tratamiento, sin presentar cada prueba como si fuera la única.

El test empleado y la construcción de intervalos respetarán el emparejamiento y la dependencia pertinente. Un p-value obtenido con una aproximación se identificará como tal; “p-value exacto” en la tabla no se usará para afirmar que una distribución asintótica es una prueba exacta. Se evitará mostrar solo `p < 0,05`; se conservará el valor calculado con precisión suficiente y sus límites numéricos.

No se ha ejecutado inferencia entre modelos: no hay predicciones ni repeticiones disponibles. Los p-values del EDA no constituyen evidencia de diferencias entre los tres modelos.

### 5.3.5. Umbral para futuros modelos avanzados

Se distinguirán dos cantidades:

- **Mejora mínima detectable:** depende del diseño, la variabilidad pareada y la potencia; se estima con el procedimiento de §4.1, respetando sus supuestos.
- **Margen de relevancia práctica:** cambio que importa para la aplicación, justificado y fijado sin acomodarlo al resultado del test. El $\delta$ del proyecto todavía debe cerrarse.

La desviación entre corridas o la anchura de un IC por sí solas no bastan para declarar una mejora detectable: se necesitan contraste, potencia, emparejamiento y variabilidad de las diferencias. Si la mejora detectable supera $\delta$, se ampliará el diseño o se explicitará esa limitación.

Si se comunica el margen como reducción relativa del error, se declarará la referencia y cómo se convierte a las unidades de RMSE. La conversión se fijará con información de desarrollo o una referencia congelada, no se escogerá después para favorecer al modelo avanzado.

Los futuros modelos deberán compararse con el baseline bajo condiciones equivalentes y contrastes predefinidos. No basta con superar una cifra puntual redondeada ni con obtener un p-value pequeño.

### 5.3.6. Cota de referencia e interpretación

Cuando exista una cota de referencia verificable del problema, se discutirá junto con sus supuestos y su comparabilidad. Actualmente no se ha estimado ruido irreducible ni una cota de error para esta volatilidad financiera. La persistencia aporta contexto, pero **no es una cota teórica de error mínimo**; un resultado publicado en otro dataset tampoco lo es.

La conclusión final deberá distinguir significación estadística de relevancia práctica y valorar si el aumento de complejidad, parámetros y costo se traduce en una mejora útil. Se explicitará el baseline, su incertidumbre, el margen exigido y las limitaciones de generalización a otros periodos, pares o mercados.

### 5.3.7. Estado del benchmark

| Producto | Estado |
|---|---|
| Tabla comparable y reglas de reporte | Definidas; sin valores de modelos |
| Tres modelos y repeticiones ejecutadas | Pendientes |
| Ranking y modelo de referencia seleccionado | Pendientes de validación |
| Métricas, tiempos y parámetros medidos | Pendientes |
| Diferencias relativas y tamaños de efecto | Pendientes de resultados |
| Inferencia pareada, intervalos y corrección | No ejecutada |
| Margen práctico y mejora detectable | Pendientes de cierre del protocolo y piloto |
| Baseline experimental definitivo | No establecido |

Este capítulo deja preparado el producto final exigido, sin presentar el protocolo como un benchmark ya realizado.
