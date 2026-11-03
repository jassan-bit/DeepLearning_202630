# 5. Criterios del benchmark y baseline previstos para E2

El benchmark comparará **exactamente tres arquitecturas DL una vez se cierre la selección de §3**. Las fichas LSTM, TCN y encoder Transformer siguen como candidatas y GRU continúa evaluable; no fijan todavía los tres puestos experimentales. La comparación usará configuraciones respaldadas y el protocolo común de §4. Su resultado será identificar una referencia mínima de desempeño que un modelo avanzado deberá superar con evidencia estadística y relevancia práctica.

## 5.1. Comparación Tabular

### 5.1.1. Alcance y unidad de comparación

Se elaborará una tabla por configuración comparable: ventana de entrada $D$, ventana de volatilidad $n$, horizonte, periodo y conjunto de evaluación. Se mantendrán las mismas muestras, exclusiones, semillas y definición de métricas entre modelos.

Las ventanas de volatilidad definen objetivos diferentes. No se elegirá un supuesto mejor modelo comparando directamente la RMSE de una volatilidad de siete días con la RMSE de otra de veintiocho días. El ranking y las diferencias se calcularán dentro de cada comparación bien definida.

El horizonte principal es un día; se reportarán 2025 y 2026 por separado en el test previsto. Los valores de la literatura servirán como contexto documentado, sin mezclarlos con mediciones propias sobre otros datos o escalas.

### 5.1.2. Tabla obligatoria de desempeño y costo

La tabla siguiente es una **plantilla pendiente de resultados**, no una comparación ejecutada. M1, M2 y M3 son puestos experimentales sin arquitectura final asignada, con configuraciones por cerrar en E2:

| Modelo base | Ejecuciones válidas | RMSE, media ± DE | MAE, media ± DE | Otras métricas | IC 95 % y método | Tiempo de entrenamiento | Tiempo de inferencia | Parámetros |
|---|---|---|---|---|---|---|---|---|
| M1 — arquitectura por cerrar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |
| M2 — arquitectura por cerrar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |
| M3 — arquitectura por cerrar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |

Se conservará un formato largo con un intervalo por cada métrica que lo requiera; la columna resumida de intervalos no se interpretará como un único IC aplicable a todas las métricas. Las métricas adicionales incluyen MAPE, MSE, R² y MASE, con los casos no definidos y la cobertura documentados según §4.4.

Los tiempos indicarán unidad, hardware, tamaño de lote, calentamiento y si incluyen preprocesamiento. El número de parámetros distinguirá entrenables y no entrenables. No se sustituirá una medición por una estimación teórica no identificada como tal.

### 5.1.3. Incertidumbre y trazabilidad

Cada fila se vinculará a configuraciones, código, manifiestos, logs y predicciones. Se reportarán media, desviación estándar **muestral**, número de ejecuciones e IC del 95 %, con método y supuestos explícitos.

Se diferenciará incertidumbre entre semillas sobre el mismo conjunto de evaluación de incertidumbre temporal por bloques. Los 366 folds no se tratarán como semillas independientes. Para RMSE y R² se usará la agregación de predicciones de §4.1.4 y §4.4; no un promedio inadecuado de métricas de folds unitarios.

## 5.2. Ranking de Modelos Base

### 5.2.1. Criterio principal y cierre de selección

El criterio principal fijado es **menor RMSE media**, con métricas secundarias y costo como información complementaria. Se fijarán antes del test las reglas de selección y desempate; no se elegirá una regla después de observar qué modelo favorece.

El modelo de referencia operativo se seleccionará con validación de desarrollo y se congelará antes de test. La tabla final puede ordenar de forma descriptiva el desempeño de los tres modelos en test, pero ese orden no autoriza reabrir la selección. Si difiere del orden de validación, se reportarán ambos y esa discrepancia, sin reemplazar silenciosamente la referencia congelada.

Todavía no se asignan posiciones: los modelos no han sido entrenados. Un intervalo solapado o una diferencia pequeña no define por sí solo un empate formal; se requiere la comparación pareada y la interpretación de relevancia práctica.

### 5.2.2. Precisión, costo y complejidad

Se examinarán dos relaciones: error frente a costo computacional y error frente a complejidad. Se mostrarán RMSE contra tiempos y número de parámetros con sus condiciones de medición, conservando incertidumbre cuando corresponda.

Una menor RMSE no se presentará automáticamente como la mejor decisión operacional si exige un costo o complejidad desproporcionados. Se identificarán compromisos y, cuando los datos lo permitan, soluciones no dominadas: un modelo está dominado si otro obtiene menor o igual error con menor o igual costo y mejora estricta en al menos una dimensión.

La complejidad y el tiempo no se confunden: menos parámetros no garantiza menor latencia. No se dibujarán curvas o diagramas de Pareto ficticios mientras falten mediciones.

## 5.3. Definición del Baseline y Margen de Mejora Disponible

### 5.3.1. Baseline de referencia

Se identificará explícitamente el modelo seleccionado mediante desarrollo, su configuración, alcance y desempeño final con **media, desviación estándar e IC del 95 %**. Se documentará también el orden descriptivo de los tres modelos en la evaluación final y cualquier diferencia respecto a la selección inicial.

El baseline no es solo el nombre de una familia: incluye datos, objetivo, horizonte, pipeline, restricciones, configuración y protocolo de evaluación. Todavía no existen resultados propios que permitan ordenar LSTM, TCN y encoder Transformer o identificar un baseline ganador. La selección documental no satisface por sí sola la exigencia de desempeño medido.

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

Se declarará la familia de comparaciones y su corrección —por ejemplo **Holm**— antes de realizar afirmaciones confirmatorias; Bonferroni u otro procedimiento requieren justificación previa. Tres modelos producen tres pares por configuración; si se extienden las conclusiones a varios objetivos, entradas u horizontes, se declarará esa ampliación y su tratamiento, sin presentar cada prueba como si fuera la única.

El test empleado y la construcción de intervalos respetarán el emparejamiento y la dependencia pertinente. Se identificará si los IC del 95 % son marginales para cada diferencia o simultáneos para una familia; un ajuste de p-values no convierte por sí solo los intervalos marginales en intervalos simultáneos. Un p-value obtenido con una aproximación se identificará como tal; “p-value exacto” en la tabla no se usará para afirmar que una distribución asintótica es una prueba exacta. Se evitará mostrar solo `p < 0,05`; se conservará el valor calculado con precisión suficiente y sus límites numéricos.

No se ha ejecutado inferencia entre modelos: no hay predicciones ni repeticiones disponibles. Los p-values del EDA no constituyen evidencia de diferencias entre los tres modelos.

### 5.3.5. Umbral para futuros modelos avanzados

Se distinguirán dos cantidades:

- **Mejora mínima detectable:** depende del diseño, la variabilidad pareada y la potencia; se estima con el procedimiento de §4.1, respetando sus supuestos.
- **Margen de relevancia práctica:** cambio que importa para la aplicación, justificado y fijado sin acomodarlo al resultado del test. La regla aceptada para este estudio es reducción relativa del 5 % frente a persistencia; su conversión a unidades de planificación se calculará en E2, sin redefinir el porcentaje. Este margen no se traslada automáticamente a una comparación entre dos redes ni a un modelo avanzado frente al baseline DL.

La desviación entre corridas o la anchura de un IC por sí solas no bastan para declarar una mejora detectable: se necesitan contraste, potencia, emparejamiento y variabilidad de las diferencias. Si la mejora detectable supera el margen registrado para el contraste correspondiente, se ampliará el diseño o se explicitará esa limitación.

Si se comunica el margen como reducción relativa del error, se declarará la referencia y cómo se convierte a las unidades de RMSE. La conversión se fijará con información de desarrollo o una referencia congelada, no se escogerá después para favorecer al modelo avanzado.

Se distinguirá alcanzar el 5 % en la estimación puntual de aportar evidencia de que la mejora alcanza ese margen. La regla inferencial sobre el margen y su incertidumbre se registrará antes de evaluar; un contraste de igualdad de errores no contrasta automáticamente la superación del 5 %.

Antes de comparar un modelo avanzado con el baseline DL se deberá registrar un margen referido a ese baseline, con sentido aplicado y escala clara. El actual 5 % frente a persistencia no fija ese margen adicional. Los futuros modelos deberán compararse bajo condiciones equivalentes y contrastes predefinidos. No basta con superar una cifra puntual redondeada ni con obtener un p-value pequeño.

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
| Margen práctico relativo | Fijado en 5 % de mejora de RMSE frente a persistencia |
| Mejora mínima detectable | Dimensionamiento futuro de E2 con el piloto |
| Baseline experimental definitivo | No establecido |

Este capítulo deja preparado el producto final exigido, sin presentar el protocolo como un benchmark ya realizado.

## 5.4. Análisis Estadístico y Pruebas de Hipótesis

Se requieren contrastes formales para evaluar diferencias entre los tres modelos: **no bastan las métricas promedio**. El análisis distinguirá resultados entre semillas, diferencias de pérdidas por fecha y diagnósticos de residuos, porque corresponden a preguntas y unidades estadísticas distintas.

### 5.4.1. Hipótesis, unidad de análisis y supuestos

Antes de evaluar se registrarán métrica principal, configuraciones, contrastes, dirección bilateral o unilateral justificada, nivel de significación **$\alpha=0{,}05$** y familia de comparaciones. Las diferencias pareadas deberán corresponder a **las mismas semillas, fechas, particiones o bloques**, según la pregunta planteada.

| Pregunta | Unidad de análisis que debe declararse |
|---|---|
| Variación de desempeño por entrenamiento | Métrica agregada por ejecución y diferencia entre modelos sobre las mismas semillas |
| Precisión predictiva a lo largo del tiempo | Diferencia de pérdidas en las mismas fechas y horizonte |
| Estructura remanente de errores | Residuos fuera de entrenamiento, ordenados temporalmente |

Se examinarán normalidad de las **diferencias pareadas** mediante Shapiro–Wilk y gráficos Q–Q, dependencia, valores extremos y supuestos de varianza o covarianza pertinentes al contraste. Con pocas semillas, no rechazar normalidad no demuestra que sea cierta. La homogeneidad de varianzas entre grupos independientes no se impondrá como requisito genérico de todo contraste pareado; se verificará lo que realmente exige el diseño elegido.

Las 366 fechas y las 5.856 filas configuración–fold no se tratarán como réplicas i.i.d. Las semillas sobre un mismo test representan variabilidad del entrenamiento condicionada a ese mercado y no nuevos mercados independientes. Los supuestos y limitaciones deberán acompañar cada conclusión.

### 5.4.2. Comparaciones pareadas: t y Wilcoxon

Para diferencias de métricas por ejecución se considerará **t-test pareado** cuando el diseño y la distribución de diferencias lo justifiquen. Se reportarán hipótesis, número de pares, diferencia media, estadístico, grados de libertad, IC y p-value.

La alternativa **Wilcoxon signed-rank** requiere justificar su uso y declarar el tratamiento de ceros y empates. No es una prueba sin supuestos: para interpretaciones de desplazamiento se considerará la simetría de las diferencias y su unidad de independencia. Se distinguirá el cálculo exacto del aproximado cuando haya empates o limitaciones del software.

La elección entre pruebas se justificará desde el diseño y los diagnósticos; no se escogerá la que produzca el p-value más pequeño. Si se usan rangos se respetará el mínimo de seis ejecuciones de la guía, o el número mayor exigido por potencia. Con cinco pares sin ceros ni empates, el menor p-value bilateral exacto de Wilcoxon es $2/2^5=0{,}0625$; con seis es $0{,}03125$, todavía mayor que el primer umbral de Holm para tres pares ($0{,}05/3$). Cumplir el mínimo de semillas no garantiza detectar una diferencia.

### 5.4.3. Comparación conjunta de tres modelos y post-hoc

Se considerará **ANOVA de medidas repetidas** si se justifican sus supuestos, incluida la estructura de covarianza o esfericidad cuando corresponda y sus correcciones. **Friedman** es la alternativa recomendada por la guía para bloques emparejados, con la advertencia sobre potencia de su sección 7.

Los bloques deberán representar unidades defendibles del diseño. Repetir el mismo test con varias semillas no permite asumir automáticamente independencia de mercados. Si se usan bloques temporales, su definición y dependencia residual se documentarán.

Los análisis post-hoc podrán incluir Nemenyi u otras comparaciones pareadas con **Holm o Bonferroni**, según el plan registrado. Se declararán procedimiento, familia de hipótesis y ajustes aplicados; no se combinarán correcciones sin explicar qué controla cada una. La familia incluirá las configuraciones u horizontes sobre los que se realicen afirmaciones, no solo los tres pares de una tabla aislada.

Un omnibus no significativo no prueba igualdad entre modelos, y un análisis post-hoc no convierte un diseño de baja potencia en evidencia concluyente. Se interpretará junto con efectos, intervalos y MDE.

### 5.4.4. Intervalos de confianza, tamaños de efecto y potencia

Las diferencias se acompañarán de **IC del 95 %** y tamaños de efecto. Un bootstrap pareado conservará la correspondencia entre los modelos; si la unidad es temporal, se emplearán bloques u otro procedimiento que respete la dependencia. No se remuestrearán por separado los errores de cada modelo destruyendo el emparejamiento.

Para diferencias entre ejecuciones podrá usarse $d_z$ de §5.3.3, u otra medida justificada como una medida basada en rangos. Se identificarán signo, denominador, empates, límites e incertidumbre. No se sustituirá una medida pareada por una de grupos independientes sin justificación.

Cada contraste se acompañará de la potencia planificada o de la **diferencia mínima detectable del diseño**, con los supuestos de §4.1. No se utilizará una potencia retrospectiva calculada solo a partir del efecto observado como prueba de que el estudio fue suficiente. La muestra pequeña y los objetivos móviles solapados deben reconocerse.

### 5.4.5. Diebold–Mariano por horizonte

La precisión predictiva temporal se comparará con **Diebold–Mariano**, por horizonte y sobre fechas comunes. Como RMSE es la métrica primaria fijada, se usará una pérdida cuadrática coherente:

$$
e_{A,t}=Y_t-\widehat Y_{A,t},\qquad
e_{B,t}=Y_t-\widehat Y_{B,t},\qquad
d_t=e_{A,t}^2-e_{B,t}^2.
$$

La hipótesis de igualdad de precisión es $H_0:\mathbb E[d_t]=0$. Con este signo, una media negativa favorece a $A$ en pérdida cuadrática. La prueba se aplica a **diferencias de pérdidas por observación**, no a una lista de RMSE globales presentada como si fuera una serie diaria de errores.

Se declararán estimación de varianza de largo plazo, kernel, rezagos y tamaño de muestra, y se utilizará la **corrección de Harvey–Leybourne–Newbold para muestras pequeñas** solicitada por la guía, documentando su implementación y alcance. El horizonte principal es un día, pero las etiquetas de volatilidad comparten retornos entre fechas: **$h=1$ no garantiza ausencia de autocorrelación en $d_t$**. Los controles de dependencia no se reducirán automáticamente a cero rezagos por ese motivo.

El ajuste de muestra pequeña no corrige por sí solo todas las formas de no estacionariedad, selección de modelos o dependencia. Los huecos, etiquetas solapadas y posibles cambios de régimen se considerarán al elegir tramos y métodos. No se concatenarán segmentos discontinuos como si no hubiese interrupciones.

Las semillas se tratarán según un plan previo: no se concatenarán todas las fechas de todas las semillas como una única serie temporal más larga. Si se comparan predicciones agregadas entre semillas, esa agregación deberá registrarse como parte del método, no inventarse después de ver los resultados.

### 5.4.6. BDS sobre residuos

Se aplicará **BDS —Brock–Dechert–Scheinkman—** a residuos fuera de entrenamiento, incluido el primer horizonte exigido en el material recibido. Se documentarán tramo, número de observaciones, dimensiones de inmersión, radio o regla de escala, tratamiento de valores ausentes y versión del software.

BDS contrasta una hipótesis de comportamiento i.i.d. bajo sus condiciones. Un rechazo puede reflejar dependencia u otras desviaciones y **no demuestra por sí solo no linealidad estructural**. Un no rechazo tampoco acredita independencia absoluta ni que se haya agotado toda capacidad predictiva.

Los residuos se ordenarán y se evaluarán sobre tramos temporalmente válidos; no se sustituirán por retornos crudos del EDA. Las pruebas de diferentes dimensiones, modelos o configuraciones se incluirán en la política de multiplicidad cuando sustenten afirmaciones confirmatorias.

### 5.4.7. Ljung–Box e interpretación de residuos

Se usará **Ljung–Box** para estudiar autocorrelación residual en varios rezagos predefinidos. Se registrarán tamaño de muestra, rezagos, hipótesis, estadístico, p-values y tratamiento de grados de libertad.

La corrección de grados de libertad se justificará según el modelo y los parámetros de la prueba: **no se restará automáticamente el número de pesos de una red neuronal** como si fueran los parámetros ARMA de una fórmula clásica. También podrá examinarse la dependencia de residuos al cuadrado con una finalidad explícita, diferenciándola de la autocorrelación lineal.

Sesgo, autocorrelación o estructura no explicada pueden motivar investigar mejoras, pero no prueban que un modelo avanzado concreto vaya a aprovecharlas. La interpretación incorporará heterocedasticidad, cambios de régimen, especificación del objetivo y los límites de cada contraste.

### 5.4.8. Ausencia de significación y equivalencia

**No rechazar una diferencia no demuestra equivalencia.** Si se desea afirmar equivalencia entre modelos, se planteará un análisis separado, por ejemplo **TOST** con margen práctico predefinido. Una alternativa bayesiana con **ROPE** deberá declarar modelo, distribución posterior y regla de decisión; no se presentará como si fuera la misma prueba frecuentista.

El margen de equivalencia debe fijarse con sentido aplicado, escala, referencia y unidades claras. El 5 % de mejora frente a persistencia es un criterio de superioridad práctica y no define automáticamente una banda de equivalencia entre dos modelos. No se ensanchará después para conseguir equivalencia. Se distinguirán ausencia de evidencia, evidencia de diferencia y evidencia de equivalencia bajo el diseño correspondiente.

### 5.4.9. Reporte y buenas prácticas estadísticas

Cada contraste tendrá un registro con:

| Campo | Contenido obligatorio |
|---|---|
| Comparación | Modelos, configuración, horizonte y periodo |
| Hipótesis | Nula, alternativa y dirección fijadas previamente |
| Unidad y tamaño | Semillas, fechas o bloques; cantidad de pares válidos y exclusiones |
| Supuestos | Diagnósticos, limitaciones y justificación del método |
| Método | Prueba, versión, estadístico, grados de libertad o parámetros pertinentes |
| Evidencia | Diferencia o efecto, IC del 95 %, p-value calculado y tamaño de efecto |
| Multiplicidad | Familia, corrección y p-value ajustado y sin ajustar |
| Potencia | Potencia planificada o MDE y su relación con $\delta$ |
| Interpretación | Significación estadística, relevancia práctica y restricciones de generalización |

Los p-values se reportarán con precisión numérica suficiente, no solo como `p < 0,05`. Se distinguirá un procedimiento exacto de una aproximación y se explicitarán límites numéricos si el software devuelve cero por redondeo o subdesbordamiento. No se seleccionarán únicamente los contrastes favorables ni se multiplicarán análisis sin identificarlos como exploratorios.

Las pruebas de DeLong, McNemar, segmentación o análisis de lenguaje del catálogo general no corresponden a este objetivo continuo de una serie financiera. Su ausencia no se reemplaza con métricas ajenas al problema.

### 5.4.10. Estado del análisis estadístico

| Producto | Estado |
|---|---|
| Protocolo general y pruebas temporales | Documentados |
| Tres modelos DL, predicciones y residuos del benchmark | Pendientes de implementación y entrenamiento |
| Familia de contrastes y parámetros definitivos | Pendientes de congelación |
| Shapiro–Wilk, Q–Q y verificación de supuestos del benchmark | No ejecutados |
| t/Wilcoxon, ANOVA/Friedman y post-hoc | No ejecutados |
| Diebold–Mariano con corrección, BDS y Ljung–Box del benchmark | No ejecutados; el Ljung–Box exploratorio de la referencia EDA está en §2.2.4 |
| Intervalos, efectos, p-values y conclusiones de comparación | No disponibles todavía |
| Equivalencia, si se plantea | Análisis separado y preregistrado; no ejecutado |

El capítulo no atribuye significación ni equivalencia a modelos aún no evaluados. Las conclusiones requerirán predicciones y registros verificables bajo el protocolo congelado.

## 5.5. Discusión Crítica

La discusión responderá a las preguntas del entregable mediante resultados del benchmark, literatura verificable e inferencia adecuada. **No se aceptan conclusiones basadas únicamente en descripciones o promedios:** cada afirmación debe indicar su evidencia, alcance e incertidumbre.

### 5.5.1. Desempeño y explicación de las diferencias

Para cada configuración comparable se identificará qué modelo obtiene menor error y qué diferencias están respaldadas por intervalos, efectos y contrastes. Se distinguirán el modelo seleccionado con desarrollo y el orden descriptivo de la evaluación final, sin reabrir la selección al observar test.

La interpretación relacionará los resultados con **sesgo inductivo, capacidad y regularización** de las arquitecturas realmente implementadas. Una explicación causal sobre por qué un componente mejora el resultado necesitará evidencia adicional, por ejemplo una comparación controlada; no basta con atribuir a posteriori una ventaja a una propiedad conocida de la familia.

No se afirmará que una red recurrente, convolucional o MLP es superior por su nombre, por una sola semilla o por un artículo con datos y objetivos distintos. Las explicaciones se contrastarán con errores por periodo, curvas, sensibilidad y resultados repetidos.

### 5.5.2. Generalización y estabilidad

Se estudiará qué modelo mantiene mejor desempeño entre desarrollo y test, entre 2025 y el tramo evaluado de 2026 y entre los segmentos predefinidos. Se mostrarán cantidades de casos, incertidumbre y posibles cambios de distribución, no solo un valor agregado.

El test de 2026 cubre hasta agosto dentro del calendario actual, no el año completo. No se compararán periodos de distinta duración como si representaran las mismas condiciones sin discutirlo. La información descargada de septiembre–octubre no se incorporará retrospectivamente para mejorar una conclusión.

La estabilidad temporal en BTCUSDT de Binance Spot no demuestra generalización a otros exchanges, pares, activos o mercados. Tampoco garantiza estabilidad futura en el mismo activo. Las semillas adicionales aportan evidencia sobre entrenamiento y no reemplazan una validación externa.

### 5.5.3. Significación estadística y relevancia práctica

La discusión presentará juntas **diferencia estimada, IC, tamaño de efecto, p-value ajustado y relación con $\delta$**. Una diferencia pequeña puede ser estadísticamente distinguible sin aportar utilidad práctica; una diferencia no significativa puede deberse a incertidumbre o baja potencia.

La falta de significación no se interpretará como igualdad o equivalencia. Si se plantea equivalencia, se seguirá el análisis separado de §5.4.8. Se conservarán resultados desfavorables, contrastes no significativos y limitaciones, evitando seleccionar solo las comparaciones que favorezcan a un modelo.

### 5.5.4. Desempeño frente a costo y complejidad

Se discutirá si el cambio de error justifica el incremento de parámetros, tiempo de ajuste, latencia de inferencia y recursos. Los costos se compararán bajo condiciones de medición explícitas y en relación con la frecuencia diaria del uso previsto, no con exigencias operativas inventadas.

Un modelo más simple podrá resultar preferible bajo una regla previamente declarada si la mejora de otro no compensa su costo; esa decisión no equivale automáticamente a demostrar equivalencia estadística. Se explicarán los compromisos y las soluciones no dominadas sin convertir un único indicador en una conclusión universal.

### 5.5.5. Baseline y umbral para un modelo avanzado

La conclusión final deberá nombrar el **modelo de referencia, configuración, objetivo, horizonte, alcance y desempeño con incertidumbre**. También deberá indicar qué mejora detectable y qué margen práctico tendría que superar una propuesta avanzada bajo condiciones comparables.

Se usará la distinción de §5.3.5 entre sensibilidad del diseño y relevancia aplicada. El umbral no será simplemente el límite de un IC del baseline ni un porcentaje elegido después de observar el resultado. Si la incertidumbre no permite discriminar modelos, se declarará la insuficiencia y su consecuencia para el baseline, sin proclamar un ganador definitivo.

Actualmente ese baseline DL y el umbral para comparaciones avanzadas **no están establecidos**. El margen operativo del 5 % frente a persistencia sí está definido, pero responde a otra comparación; los valores de literatura no sustituyen la evaluación propia.

### 5.5.6. Brecha con la literatura y margen de mejora potencial

La brecha entre modelos avanzados publicados y las referencias básicas se examinará con los estudios incluidos en la revisión. Se comprobarán definición de volatilidad, escala, frecuencia, horizonte, activo, proveedor, periodos, particiones, selección y número de repeticiones.

Solo bajo comparabilidad suficiente se cuantificará una diferencia directa. Si los estudios usan otros datos o protocolos, se presentarán como contexto y se explicará por qué no permiten calcular un margen numérico trasladable al experimento propio. No se restará una RMSE publicada de la RMSE local sin verificar primero unidades y condiciones.

Se discutirá la credibilidad de la brecha considerando intervalos, variabilidad, sesgos de publicación y calidad metodológica. Una mejora publicada no constituye una garantía de que otra arquitectura la reproduzca en BTCUSDT bajo este calendario.

### 5.5.7. Amenazas a la validez

| Dimensión | Riesgo pertinente | Control o evidencia requerida | Límite actual |
|---|---|---|---|
| Interna | Fuga temporal por etiquetas, escalado o parada | Fechas, disponibilidad, transformaciones solo en train y validación interna pasada | Integración con modelos pendiente |
| Interna | Optimización desigual o selección de semillas | Presupuesto común, espacios justificados, logs completos y protocolo previo | Búsqueda y corridas no ejecutadas |
| Interna | Cambios de cobertura entre modelos | Mismos orígenes y exclusiones, con recuentos verificables | Datos/folds registrados; comparación pendiente |
| Externa | Un activo, un proveedor, una cotización | Delimitar BTCUSDT Spot y evitar afirmaciones sobre otros mercados | Sin evaluación externa de otros datasets |
| Externa | Regímenes y periodos de duración diferente | Resultados por periodo y segmentos fijados, con incertidumbre | Test todavía no evaluado |
| De construcción | Volatilidad como proxy y definición de ventanas | Fórmula, escala y sensibilidad a las cuatro ventanas documentadas | No mide volatilidad latente sin error |
| De construcción | Métricas sensibles a ceros o colas | Política previa de MAPE/MASE y métricas complementarias | Políticas de MAPE/MASE fijadas; evaluación posterior en E2 |
| De conclusión estadística | Dependencia, pocos modelos y pocas semillas | Efectos, IC, multiplicidad, potencia/MDE y unidades justificadas | No hay inferencia entre modelos todavía |

Los controles no se describirán como garantía absoluta: cada afirmación de validez deberá identificar qué se verificó realmente y qué permanece como supuesto o limitación.

### 5.5.8. Limitaciones y recomendaciones futuras

Las limitaciones observadas incluyen el alcance de un solo mercado, un día incompleto de desarrollo, objetivos móviles con retornos compartidos y la distinción entre variabilidad de semillas y variabilidad temporal. Se conservaron los brutos; excluir un día de los derivados no demuestra que no existan otros problemas económicos de calidad.

El estado actual añade limitaciones de ejecución: cobertura bibliográfica limitada y respaldo de atención en dos manuscritos grises; configuraciones, entrenamiento y evaluación propios reservados para E2. Las familias de trabajo, RMSE y el margen relativo del 5 % ya están documentados. Estas son tareas pendientes y no hallazgos de desempeño.

Las recomendaciones se derivarán de esas limitaciones y de los resultados que se obtengan: completar evidencia bibliográfica, cerrar el protocolo, ejecutar el piloto, dimensionar corridas y evaluar de manera comparable. Extensiones a noticias, otros mercados o arquitecturas avanzadas requerirán nueva justificación, fuentes y validación; no se presentan como mejoras garantizadas ni como experimentos ya ejecutados.

### 5.5.9. Matriz de conclusiones y evidencia pendiente

| Pregunta de la discusión | Evidencia necesaria | Respuesta actual |
|---|---|---|
| ¿Qué modelo funciona mejor y por qué? | Métricas, incertidumbre, contrastes y arquitectura | No determinable sin entrenamientos |
| ¿Cuál generaliza mejor? | Desarrollo/test y periodos comparables con errores segmentados | Pendiente |
| ¿La diferencia es significativa y relevante? | Comparación pareada, corrección, IC, efecto y $\delta$ | Pendiente |
| ¿Compensa el costo o la complejidad? | Medidas de error, tiempos, hardware y parámetros | Pendiente |
| ¿Cuál es el baseline y su umbral de mejora? | Selección cerrada, evaluación y dimensionamiento | No establecido |
| ¿Qué margen sugiere la literatura? | Revisión de estudios comparables y variabilidad reportada | No cuantificado |
| ¿Qué limita las conclusiones? | Registro de validez y alcance de la evidencia | Limitaciones de datos/diseño identificadas; validación experimental pendiente |

La discusión final reemplazará estas respuestas pendientes por conclusiones sustentadas, sin convertir una descripción del protocolo en evidencia de superioridad.
