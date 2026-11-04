# 5. Benchmark de los Modelos Base y Definición del Baseline

El benchmark debe ser reproducible, justo y comparable, e incluir conjuntamente los tres modelos base. El benchmark comparará **exactamente tres arquitecturas DL una vez se cierre la selección de §3**. Las fichas LSTM, TCN y encoder Transformer siguen como candidatas y GRU continúa evaluable; no fijan todavía los tres puestos experimentales. La comparación usará configuraciones respaldadas y el protocolo común de §4. Su resultado será identificar una referencia mínima de desempeño que un modelo avanzado deberá superar con evidencia estadística y relevancia práctica.

La guía exige un benchmark medido en E1; el alcance acordado mantiene su ejecución para E2. Este capítulo revisa los apartados 5.1–5.5 y distingue el protocolo de sus resultados aún inexistentes. Las cuatro Ridge del EDA y los valores de autores no constituyen ese benchmark ni permiten elegir un ganador DL.

La revisión está en el {download}`informe de benchmark y baseline <../_static/e1/revision_benchmark_baseline_20261103.md>` y su [matriz de requisitos](../_static/e1/revision_benchmark_baseline_20261103.json). El [contrato documental del benchmark](../_static/experiments/benchmark_documental_20261103.json) complementa la especificación de §4; conserva reglas de reporte y decisiones abiertas, sin ser una configuración ejecutable o congelada.

## 5.1. Comparación Tabular

### 5.1.1. Tabla de comparación de los tres modelos base

Se elaborará una tabla por configuración comparable: ventana de entrada $D$, ventana de volatilidad $n$, horizonte, periodo y conjunto de evaluación. Se mantendrán las mismas muestras, exclusiones, semillas y definición de métricas entre modelos.

La configuración $c$ identifica $(n,D,h)$ y versiones de datos/protocolo. Cada tabla añade fase, periodo y cohorte elegible. Para esa tabla, con orígenes comunes $\mathcal V_c$ y repeticiones $\mathcal S_c$, la métrica de una corrida y la puntuación principal serán:

$$
m_{m,s,c}=\sqrt{\frac1{N_c}\sum_{t\in\mathcal V_c}
(Y_t-\widehat Y_{m,s,t})^2},\qquad
S_{m,c}=\frac1{R_c}\sum_{s\in\mathcal S_c}m_{m,s,c}.
$$

Se ordena por **media de RMSE calculadas por corrida**, con DE muestral entre ellas. No equivale a calcular una única RMSE sobre todas las filas semilla×fecha ni a usar el promedio de predicciones: este último define un ensemble adicional. Si hay fallas, se registrarán intentos, repeticiones completas y cobertura conforme a la regla previa; no se elegirán corridas válidas por favorecer a un modelo ni se declarará completo un diseño que no cumpla su regla de comparación.

Las ventanas de volatilidad definen objetivos diferentes. No se elegirá un supuesto mejor modelo comparando directamente la RMSE de una volatilidad de siete días con la RMSE de otra de veintiocho días. El ranking y las diferencias se calcularán dentro de cada comparación bien definida.

El horizonte principal es un día; se reportarán 2025 y 2026 por separado en el test previsto. Los valores de la literatura servirán como contexto documentado, sin mezclarlos con mediciones propias sobre otros datos o escalas.

#### 5.1.1.1. Modelo base

La tabla siguiente es una **plantilla pendiente de resultados**, no una comparación ejecutada. M1, M2 y M3 son puestos experimentales sin arquitectura final asignada, con configuraciones por cerrar en E2:

| Modelo base | Ejecuciones válidas | RMSE, media ± DE | MAE, media ± DE | Otras métricas | IC 95 % y método | Tiempo de entrenamiento | Tiempo de inferencia | Parámetros |
|---|---|---|---|---|---|---|---|---|
| M1 — arquitectura por cerrar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |
| M2 — arquitectura por cerrar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |
| M3 — arquitectura por cerrar | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente | No medido | No medido | No implementado |

Las plantillas de [tabla de resultados](../_static/experiments/tabla_benchmark_plantilla.csv) y [registro de contrastes](../_static/experiments/contrastes_benchmark_plantilla.csv) contienen **solo encabezados**. No son filas de mediciones realizadas. Cada resultado futuro conservará fase —selección o test—, periodo, modelo, configuración, versión, máscara de casos y alcance del intervalo.

#### 5.1.1.2. Métricas: media ± desviación estándar

Cada fila se vinculará a configuraciones, código, manifiestos, logs y predicciones. Se reportarán media, desviación estándar **muestral**, número de ejecuciones e IC del 95 %, con método y supuestos explícitos.

#### 5.1.1.3. Intervalos de confianza del 95 % y método de construcción

Se conservará un formato largo con un intervalo por cada métrica que lo requiera; la columna resumida de intervalos no se interpretará como un único IC aplicable a todas las métricas. Las métricas adicionales incluyen MAPE, MSE, R² y MASE, con los casos no definidos y la cobertura documentados según §4.4.

Se diferenciará incertidumbre entre semillas sobre el mismo conjunto de evaluación de incertidumbre temporal por bloques. Los 366 folds no se tratarán como semillas independientes. Para RMSE y R² se usará la agregación de predicciones de §4.1.3 y §4.4; no un promedio inadecuado de métricas de folds unitarios.

Se declarará el método de construcción de cada intervalo: por ejemplo, t de Student si se justifican sus condiciones, o bootstrap adecuado a la unidad y dependencia registradas. El método definitivo no se seleccionará para favorecer el resultado.

#### 5.1.1.4. Tiempo de entrenamiento y de inferencia

Los tiempos indicarán unidad, hardware, tamaño de lote, calentamiento y si incluyen preprocesamiento.

Se separarán duración por ajuste, costo total de la corrida y costo de búsqueda de hiperparámetros, además de latencia y throughput de inferencia cuando procedan. No se comparará tiempo de un único ajuste con el total de cientos de reajustes. El uso de GPU, precisión, sincronización y condiciones de medición seguirán §4.3.4.

#### 5.1.1.5. Número de parámetros

El número de parámetros distinguirá entrenables y no entrenables. No se sustituirá una medición por una estimación teórica no identificada como tal.

## 5.2. Ranking de Modelos Base

### 5.2.1. Orden de los tres modelos según la métrica principal

El criterio principal fijado es **menor $S_{m,c}$**, con métricas secundarias y costo como información complementaria. Las reglas de selección, viabilidad y desempate se cerrarán con desarrollo antes de test, usando valores sin redondear. Un empate exacto conservará sus identificadores hasta aplicar esa regla; una diferencia pequeña o dos IC solapados no definen equivalencia.

El modelo de referencia operativo se seleccionará con validación de desarrollo y se congelará antes de test. La tabla final puede ordenar de forma descriptiva el desempeño de los tres modelos en test, pero ese orden no autoriza reabrir la selección. Si difiere del orden de validación, se reportarán ambos y esa discrepancia, sin reemplazar silenciosamente la referencia congelada.

Como no hay una D principal elegida, se definirá una referencia **$B_c$ por configuración**. Obtener una única referencia entre historias requeriría una regla global de D y una población comparable cerradas en desarrollo; no se introduce esa regla promediando filas después de evaluar. El mismo identificador seleccionado se conserva al reportar 2025 y 2026, junto con su política de actualización: no se elige retrospectivamente un baseline distinto en cada año.

Todavía no se asignan posiciones: los modelos no han sido entrenados. Un intervalo solapado o una diferencia pequeña no define por sí solo un empate formal; se requiere la comparación pareada y la interpretación de relevancia práctica.

La selección con desarrollo puede identificar una referencia operativa aunque no se demuestre superioridad estadística sobre todas las alternativas. Se comunica entonces la regla, su incertidumbre y el resultado inconcluso; no se exige obtener significación para poder nombrar una referencia. Los extremos elegidos después de ver test son dependientes de esos datos: sus comparaciones no se presentan como contrastes de referencia fijada a priori sin tratar esa selección.

### 5.2.2. Análisis del trade-off

Se examinarán dos relaciones: error frente a costo computacional y error frente a complejidad. Se mostrarán RMSE contra tiempos y número de parámetros con sus condiciones de medición, conservando incertidumbre cuando corresponda.

#### 5.2.2.1. Precisión frente a costo computacional

Una menor RMSE no se presentará automáticamente como la mejor decisión operacional si exige un costo o complejidad desproporcionados. Se identificarán compromisos y, cuando los datos lo permitan, soluciones no dominadas: un modelo está dominado si otro obtiene menor o igual error con menor o igual costo y mejora estricta en al menos una dimensión.

#### 5.2.2.2. Precisión frente a complejidad del modelo

La complejidad y el tiempo no se confunden: menos parámetros no garantiza menor latencia. No se dibujarán curvas o diagramas de Pareto ficticios mientras falten mediciones.

Se distinguirán los frentes error–tiempo y error–parámetros, o se declarará una comparación multidimensional. La dominancia sobre medias es **descriptiva de esas estimaciones**; no demuestra dominancia poblacional ante incertidumbre. Una recomendación de despliegue por costo puede diferir del líder de RMSE, pero se identificará como otra decisión. No sustituye el baseline del criterio principal salvo que la restricción de recursos y su regla hubieran quedado fijadas antes del test.

## 5.3. Definición del Baseline y Margen de Mejora Disponible

El objetivo central es identificar el mejor modelo base bajo el protocolo definido y cuantificar el nivel de error y el margen que deberá superar una propuesta avanzada para ser estadísticamente distinguible y prácticamente relevante.

### 5.3.1. Selección del baseline de referencia

#### 5.3.1.1. Identificación del mejor modelo base

La referencia se seleccionará por menor media de RMSE por corrida, bajo la regla y el cierre de desarrollo de §5.2.1.

El baseline no es solo el nombre de una familia: incluye datos, objetivo, horizonte, pipeline, restricciones, configuración y protocolo de evaluación. Todavía no existen resultados propios que permitan ordenar LSTM, TCN y encoder Transformer o identificar un baseline ganador. La selección documental no satisface por sí sola la exigencia de desempeño medido.

Es una **referencia empírica bajo condiciones definidas**, no una cota universal de error mínimo. La congelación identifica arquitectura, hiperparámetros, transformaciones y reglas de entrenamiento/actualización; los checkpoints y predicciones quedan ligados a su corrida y calendario. Si en una futura comparación se usan fechas nuevas, la referencia seguirá el procedimiento congelado aplicable a esas fechas, sin volver a seleccionar la arquitectura.

#### 5.3.1.2. Desempeño con media, desviación e IC del 95 %

Se identificará explícitamente el modelo seleccionado mediante desarrollo, su configuración, alcance y desempeño final con **media, desviación estándar e IC del 95 %**. Se documentará también el orden descriptivo de los tres modelos en la evaluación final y cualquier diferencia respecto a la selección inicial.

### 5.3.2. Cuantificación de las diferencias entre modelos base

#### 5.3.2.1. Regresión y series de tiempo: diferencias respecto al peor y al mejor modelo

Para la métrica de error k se usarán sus medias por corrida $\bar M_{m,k,c}$ sobre repeticiones y fechas comparables. Los extremos descriptivos se identifican **por métrica**:

$$
r^{\mathrm{mejor}}_{k,c}=\arg\min_m\bar M_{m,k,c},\qquad
r^{\mathrm{peor}}_{k,c}=\arg\max_m\bar M_{m,k,c}.
$$

El mejor o peor por RMSE no tiene por qué serlo por MAE. Las tablas identificarán qué referente se usa para cada porcentaje. Si se calcula MAE frente a los identificadores elegidos por RMSE como análisis complementario, se rotulará así y no se los llamará extremos MAE. Los empates conservarán sus identificadores según §5.2.1.

La reducción relativa descriptiva se calculará **sobre las medias por corrida**:

$$
\Delta_k(m;r,c)=100\,
\frac{\bar M_{r,k,c}-\bar M_{m,k,c}}{\bar M_{r,k,c}},
\qquad \bar M_{r,k,c}>0.
$$

Un valor positivo favorece al modelo m frente a r; negativo indica mayor error. Respecto al mejor error de esa métrica, las otras filas tendrán reducciones no positivas. Si el referente vale cero, el porcentaje es indefinido, sin epsilon. Se publicarán también diferencias en unidades originales, con signo y referentes identificados.

La razón de medias no equivale a la media de porcentajes por semilla. Esta última sería otro estimando; no se mezcla con el principal de la tabla. Las comparaciones pareadas usan las métricas individuales y las predicciones, sin sustituirlas por cifras redondeadas. Para un IC de la razón se recalculan numerador y denominador conjuntamente bajo la unidad de inferencia registrada.

Los extremos observados en test se mantienen como referencias descriptivas, separadas de $B_c$ congelado con desarrollo. La selección del mínimo o máximo en ese mismo test debe tratarse si se desean afirmaciones inferenciales sobre ellos; aplicar después un IC de referencia fija no elimina el efecto de haberlos escogido.

#### 5.3.2.2. Clasificación, etiquetado secuencial y segmentación

Estas modalidades **no aplican al proyecto actual**:

- **Clasificación:** se predice volatilidad continua, sin clases ni probabilidades de clase.
- **Etiquetado secuencial:** no se predicen etiquetas por token ni entidades delimitadas.
- **Segmentación:** no hay imágenes de entrada, máscaras ni etiquetas por píxel.

Para documentar la guía, cuando la métrica principal sea F1-score, AUC-ROC, F1 de entidad, mIoU o Dice y **un valor mayor sea mejor**, sean $\bar S_{m,k,c}$ y $\bar S_{r,k,c}$ sus medias por corrida bajo la misma evaluación. La diferencia en unidades de la métrica y la mejora relativa frente a la referencia r serían:

$$
\Delta_{\mathrm{abs},k}(m;r,c)=\bar S_{m,k,c}-\bar S_{r,k,c}.
$$

$$
\Delta_{\mathrm{rel},k}(m;r,c)=100\,
\frac{\bar S_{m,k,c}-\bar S_{r,k,c}}{\bar S_{r,k,c}},
\qquad \bar S_{r,k,c}>0.
$$

«Diferencia absoluta» designa aquí el cambio en unidades de la métrica, **con signo**; no el valor absoluto matemático. Un valor positivo favorece a m. Se declararán la escala, el promedio y el referente: si la métrica se expresa en porcentaje, la diferencia de esos valores se comunica en puntos porcentuales, mientras la segunda expresión es un cambio porcentual relativo. El mejor y el peor se identifican por la mayor y la menor puntuación de esa métrica, respectivamente, conservando los empates. Con referencia cero, la diferencia relativa queda indefinida y se informa el motivo, sin añadir epsilon; la diferencia en unidades puede seguir definida.

Se conservarán las mismas muestras y repeticiones emparejadas. La razón entre medias no se confundirá con la media de razones por semilla, y la incertidumbre de una razón recalculará conjuntamente numerador y denominador. Estas definiciones no aportan resultados de clasificación, etiquetado o segmentación del proyecto.

#### 5.3.2.3. Secuencia a secuencia

**No aplica al proyecto actual:** la salida es una volatilidad escalar y no una secuencia generada. Para una tarea de generación se documentarían las diferencias en unidades y relativas de su métrica principal —por ejemplo, BLEU, ROUGE, chrF o exactitud de secuencia completa— con la convención de §5.3.2.2 cuando un valor mayor sea mejor.

Se declararán la escala, tokenización, versión, referencias, parámetros y agregación de la métrica antes de comparar puntuaciones. Se identificarán los referentes mejor y peor de esa métrica y se mantendrán el emparejamiento y el tratamiento de referencias cero. No se mezclarán escalas o configuraciones de cálculo distintas ni se presentarán esas diferencias como mediciones propias de volatilidad.

#### 5.3.2.4. Tamaño de efecto de cada diferencia

En el proyecto, las cantidades m siguientes son métricas de error, como RMSE. Para puntuaciones donde un valor mayor sea mejor se declarará el sentido del signo correspondiente. Una alternativa basada en rangos es la correlación rango-biserial pareada, con su convención de ceros y empates explícita.

Cada diferencia relevante se acompañará de una medida de efecto apropiada y su definición. Para diferencias pareadas por ejecución $d_s=m_{A,s}-m_{B,s}$ puede informarse:

$$
d_z=\frac{\bar d}{s_d},
$$

con signo, denominador y unidad de emparejamiento identificados. Si $s_d=0$, el cociente no está definido. No se sustituirá la desviación de las diferencias por la desviación de un solo modelo. Si se usa una medida basada en rangos, se especificará el cálculo y el tratamiento de empates.

El efecto entre semillas mide esa variabilidad condicionada a los mismos datos. Para efectos sobre errores temporalmente dependientes se justificará el método de bloques o diseño correspondiente. No se atribuirá independencia a todos los días por disponer de muchas predicciones.

Con $d_s=m_{A,s}-m_{B,s}$, un $d_z$ negativo favorece a A; la reducción porcentual anterior usa el signo opuesto, positiva cuando mejora m. Se rotularán ambas convenciones. El d_z pareado no se calcula con una desviación marginal o combinada de grupos independientes. Tampoco se sustituye automáticamente por el delta de Cliff basado en todos los cruces A×B: eso perdería el emparejamiento. El efecto de rangos elegido deberá conservar las parejas y su convención de ceros/empates. Los umbrales convencionales de efecto pequeño o grande no reemplazan el margen práctico.

### 5.3.3. Inferencia sobre las diferencias

#### 5.3.3.1. Comparaciones pareadas y corrección por multiplicidad

Se declarará la familia de comparaciones y su corrección —por ejemplo **Holm**— antes de realizar afirmaciones confirmatorias; Bonferroni u otro procedimiento requieren justificación previa. Tres modelos producen tres pares por configuración; si se extienden las conclusiones a varios objetivos, entradas u horizontes, se declarará esa ampliación y su tratamiento, sin presentar cada prueba como si fuera la única.

Las métricas de 2024 utilizadas para seleccionar hiperparámetros, arquitectura o D son evidencia de selección. Un contraste posterior sobre esos mismos errores no adquiere nivel confirmatorio nominal por añadir Holm. Una evaluación externa del procedimiento requeriría selección anidada en su pasado o una evaluación reservada no usada para diseñarlo. La familia y la fase del contraste se declararán antes de estudiar sus resultados.

#### 5.3.3.2. Diferencia media, IC del 95 % y p-value ajustado y sin ajustar

Se realizarán comparaciones pareadas entre los tres modelos, con contrastes fijados previamente. Se reportarán **diferencia media, IC del 95 %, estadístico y p-value numérico**, tanto sin ajustar como ajustado por comparaciones múltiples cuando proceda.

El test empleado y la construcción de intervalos respetarán el emparejamiento y la dependencia pertinente. Se identificará si los IC del 95 % son marginales para cada diferencia o simultáneos para una familia; un ajuste de p-values no convierte por sí solo los intervalos marginales en intervalos simultáneos. Un p-value obtenido con una aproximación se identificará como tal; “p-value exacto” en la tabla no se usará para afirmar que una distribución asintótica es una prueba exacta. Se evitará mostrar solo `p < 0,05`; se conservará el valor calculado con precisión suficiente y sus límites numéricos.

No se ha ejecutado inferencia entre modelos: no hay predicciones ni repeticiones disponibles. Los p-values del EDA no constituyen evidencia de diferencias entre los tres modelos.

### 5.3.4. Umbral de mejora para modelos avanzados

Se distinguirán dos cantidades:

#### 5.3.4.1. Mejora mínima detectable frente al baseline

- **Mejora mínima detectable:** depende del diseño, la variabilidad pareada y la potencia; se estima con el procedimiento de §4.1, respetando sus supuestos.

La desviación entre corridas o la anchura de un IC por sí solas no bastan para declarar una mejora detectable: se necesitan contraste, potencia, emparejamiento y variabilidad de las diferencias. Si la mejora detectable supera el margen registrado para el contraste correspondiente, se ampliará el diseño o se explicitará esa limitación.

Para una propuesta avanzada A frente a un baseline B, en la misma unidad pareada:

$$
\operatorname{Var}(m_A-m_B)=\operatorname{Var}(m_A)
+\operatorname{Var}(m_B)-2\operatorname{Cov}(m_A,m_B).
$$

La DE o el IC de B aislado no identifica esa varianza ni una MDE verificable de A−B. Se necesitan estimando, política de entrenamiento o referencia fija, emparejamiento y datos de planificación o supuestos justificados sobre ambas cantidades. No existe aún una MDE numérica para propuestas avanzadas.

Se mantendrá la distinción de §4.1.5–§4.1.7: detectar un efecto frente a cero es distinto de demostrar que supera un margen práctico. La MDE por semillas es condicional al calendario y no acredita por sí sola potencia temporal ni potencia conjunta de toda una familia. El umbral avanzado no se define sumando una DE al baseline o tomando un extremo de su IC.

#### 5.3.4.2. Margen de relevancia práctica

- **Margen de relevancia práctica:** cambio que importa para la aplicación, justificado y fijado sin acomodarlo al resultado del test. La regla aceptada para este estudio es reducción relativa del 5 % frente a persistencia; su conversión a unidades de planificación se calculará en E2, sin redefinir el porcentaje. Este margen no se traslada automáticamente a una comparación entre dos redes ni a un modelo avanzado frente al baseline DL.

Si se comunica el margen como reducción relativa del error, se declarará la referencia y cómo se convierte a las unidades de RMSE. La conversión se fijará con información de desarrollo o una referencia congelada, no se escogerá después para favorecer al modelo avanzado.

Se distinguirá alcanzar el 5 % en la estimación puntual de aportar evidencia de que la mejora alcanza ese margen. La regla inferencial sobre el margen y su incertidumbre se registrará antes de evaluar; un contraste de igualdad de errores no contrasta automáticamente la superación del 5 %.

Antes de comparar un modelo avanzado con el baseline DL se deberá registrar un margen referido a ese baseline, con sentido aplicado y escala clara. El actual 5 % frente a persistencia no fija ese margen adicional. Los futuros modelos deberán compararse bajo condiciones equivalentes y contrastes predefinidos. No basta con superar una cifra puntual redondeada ni con obtener un p-value pequeño.

#### 5.3.4.3. Cota de referencia del problema

Cuando exista una cota de referencia verificable del problema, se discutirá junto con sus supuestos y su comparabilidad. Actualmente no se ha estimado ruido irreducible ni una cota de error para esta volatilidad financiera. La persistencia aporta contexto, pero **no es una cota teórica de error mínimo**; un resultado publicado en otro dataset tampoco lo es.

El acuerdo entre anotadores no aplica porque la etiqueta se construye mediante una fórmula sobre retornos, sin anotación humana. No se dispone de una medición de desempeño humano comparable que sirva como cota.

### 5.3.5. Interpretación

#### 5.3.5.1. Significación estadística y relevancia práctica

La conclusión final deberá distinguir significación estadística de relevancia práctica y valorar si el aumento de complejidad, parámetros y costo se traduce en una mejora útil. Se explicitará el baseline, su incertidumbre, el margen exigido y las limitaciones de generalización a otros periodos, pares o mercados.

Una diferencia estadísticamente distinguible no acredita por sí sola que alcance el margen práctico registrado. Un resultado inconcluso tampoco demuestra equivalencia entre modelos.

#### 5.3.5.2. Desempeño frente a complejidad, parámetros y costo

Se valorará si la mejora de la métrica principal compensa el incremento de complejidad, parámetros y costo con los datos y las condiciones documentados en §5.2.2. La conclusión distinguirá el criterio principal de desempeño de una eventual recomendación operacional por recursos; las mediciones todavía están pendientes.

#### 5.3.5.3. Conclusión explícita del baseline y margen disponible

La conclusión final identificará $B_c$, su configuración, desempeño e incertidumbre, diferencias entre modelos y margen exigido a una propuesta avanzada. Se declararán la MDE, el margen práctico, sus referencias y las limitaciones, o los campos que no pudieron estimarse. Actualmente no hay un baseline experimental ganador ni un margen avanzado medido; se completarán a partir de las ejecuciones reservadas para E2, sin sustituirlos por resultados de literatura o del EDA.

#### 5.3.5.4. Estado del benchmark

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

Cada análisis identificará su **estimando y fase**: media de métricas entre entrenamientos, pérdida temporal, razón de errores o diagnóstico residual. La evidencia confirmatoria principal, las sensibilidades y los diagnósticos exploratorios se distinguirán desde el plan. Varias pruebas sobre las mismas predicciones no son confirmaciones independientes; tampoco se dividirán familias después de observar sus p-values.

### 5.4.2. Comparaciones pareadas: t y Wilcoxon

Para diferencias de métricas por ejecución se considerará **t-test pareado** cuando el diseño y la distribución de diferencias lo justifiquen. Se reportarán hipótesis, número de pares, diferencia media, estadístico, grados de libertad, IC y p-value.

La t pareada contrasta la **media de las diferencias**. Sus condiciones corresponden a esas diferencias y a la independencia entre pares bajo el alcance registrado, no a normalidad de las métricas crudas de cada red ni a igualdad de sus varianzas marginales.

La alternativa **Wilcoxon signed-rank** requiere justificar su uso y declarar el tratamiento de ceros y empates. No es una prueba sin supuestos: para interpretaciones de desplazamiento se considerará la simetría de las diferencias y su unidad de independencia. Se distinguirá el cálculo exacto del aproximado cuando haya empates o limitaciones del software.

Wilcoxon no reemplaza automáticamente una prueba de la misma media cuando falta normalidad: su interpretación de desplazamiento y sus condiciones deben corresponder a la pregunta. Shapiro–Wilk y Q–Q son diagnósticos; no se adoptará después una regla mecánica de elegir t o Wilcoxon según qué p-value resulte favorable. El análisis principal y las sensibilidades se definirán previamente, conservando el estimando o admitiendo una conclusión limitada cuando sus supuestos no sean defendibles.

La elección entre pruebas se justificará desde el diseño y los diagnósticos; no se escogerá la que produzca el p-value más pequeño. Si se usan rangos se respetará el mínimo de seis ejecuciones de la guía, o el número mayor exigido por potencia. Con cinco pares sin ceros ni empates, el menor p-value bilateral exacto de Wilcoxon es $2/2^5=0{,}0625$; con seis es $0{,}03125$, todavía mayor que el primer umbral de Holm para tres pares ($0{,}05/3$). Cumplir el mínimo de semillas no garantiza detectar una diferencia.

### 5.4.3. Comparación conjunta de tres modelos y post-hoc

Se considerará **ANOVA de medidas repetidas** si se justifican sus supuestos, incluida la estructura de covarianza o esfericidad cuando corresponda y sus correcciones. **Friedman** es la alternativa recomendada por la guía para bloques emparejados, con la advertencia sobre potencia de su sección 7.

Los bloques deberán representar unidades defendibles del diseño. Repetir el mismo test con varias semillas no permite asumir automáticamente independencia de mercados. Si se usan bloques temporales, su definición y dependencia residual se documentarán.

Un bloque completo reúne los tres resultados bajo la misma unidad. Las configuraciones n/D no se convierten en bloques intercambiables y los faltantes o fallas se reportan sin imputar resultados o eliminar pares silenciosamente. Si los bloques son semillas, la afirmación es condicional al calendario; si son temporales, agrupar días no demuestra independencia. Friedman requiere condiciones de diseño y tratamiento de empates, aunque no sea paramétrico.

Los análisis post-hoc podrán incluir Nemenyi u otras comparaciones pareadas con **Holm o Bonferroni**, según el plan registrado. Se declararán procedimiento, familia de hipótesis y ajustes aplicados; no se combinarán correcciones sin explicar qué controla cada una. La familia incluirá las configuraciones u horizontes sobre los que se realicen afirmaciones, no solo los tres pares de una tabla aislada.

Se registrará si el omnibus actúa como requisito previo para los post-hoc o si existen contrastes pareados principales ya planificados. Nemenyi incorpora control simultáneo para su familia; no se le añade automáticamente Holm/Bonferroni como una segunda corrección equivalente. Un control adicional debe responder a otra familia o nivel justificado. Con muestras pequeñas se declarará también si la referencia utilizada es exacta, aproximada o de permutación y qué condiciones la sostienen.

Un omnibus no significativo no prueba igualdad entre modelos, y un análisis post-hoc no convierte un diseño de baja potencia en evidencia concluyente. Se interpretará junto con efectos, intervalos y MDE.

### 5.4.4. Intervalos de confianza, tamaños de efecto y potencia

Las diferencias se acompañarán de **IC del 95 %** y tamaños de efecto. Un bootstrap pareado conservará la correspondencia entre los modelos; si la unidad es temporal, se emplearán bloques u otro procedimiento que respete la dependencia. No se remuestrearán por separado los errores de cada modelo destruyendo el emparejamiento.

Para diferencias entre ejecuciones podrá usarse $d_z$ de §5.3.2.4, u otra medida justificada como una medida basada en rangos. Se identificarán signo, denominador, empates, límites e incertidumbre. No se sustituirá una medida pareada por una de grupos independientes sin justificación.

Cada contraste se acompañará de la potencia planificada o de la **diferencia mínima detectable del diseño**, con los supuestos de §4.1. No se utilizará una potencia retrospectiva calculada solo a partir del efecto observado como prueba de que el estudio fue suficiente. La muestra pequeña y los objetivos móviles solapados deben reconocerse.

Un remuestreo de predicciones guardadas mide variación de evaluación condicionada a los entrenamientos realizados. Incorporar además incertidumbre del entrenamiento exige declarar cómo se combinan ambos niveles; no basta agrupar semillas×fechas como casos intercambiables. En cada réplica se conservarán conjuntamente etiquetas y predicciones de los métodos y se recalculará la diferencia o razón de RMSE. El IC de diferencias de pérdidas cuadráticas no se reutiliza directamente como IC de diferencias de RMSE.

### 5.4.5. Diebold–Mariano: pérdida, semillas y dependencia

La comparación temporal se realizará sobre las mismas fechas y horizonte, con pérdida cuadrática coherente con RMSE. Para dos procedimientos A y B, una corrida y errores $e_{A,t}=Y_t-\widehat Y_{A,t}$ y $e_{B,t}=Y_t-\widehat Y_{B,t}$:

$$
d_t=e_{A,t}^2-e_{B,t}^2,\qquad H_0:\mathbb E[d_t]=0.
$$

Una media negativa favorece A en pérdida cuadrática. Para la misma corrida y calendario, menor MSE equivale a menor RMSE, pero la prueba no toma como observaciones las RMSE globales. Tampoco contrasta directamente la **media de RMSE entre semillas** usada para el ranking: promediar raíces y tomar raíz de una pérdida media son operaciones diferentes. El resultado se identificará como evidencia sobre la cantidad realmente contrastada, junto con las métricas y sus intervalos correspondientes.

Antes de evaluar se fijará una de las políticas de semillas y su relación con la afirmación:

| Política | Serie temporal contrastada | Alcance |
|---|---|---|
| Contraste por corrida preespecificada | $d_{s,t}=e_{A,s,t}^2-e_{B,s,t}^2$ | Condicionado a esa corrida; si se informan varias semillas confirmatoriamente, se registra su familia y agregación de conclusiones |
| Pérdidas promediadas entre semillas | $\bar d_t=R^{-1}\sum_s(e_{A,s,t}^2-e_{B,s,t}^2)$ | Promedio de pérdidas de los entrenamientos observados, con variabilidad temporal; no sustituye el análisis entre semillas |
| Ensemble de predicciones registrado | Pérdidas del pronóstico agregado definido previamente | Método agregado adicional; no es la pérdida media de las corridas individuales |

No se elegirá una semilla tras ver su error o p-value ni se concatenarán todas las semillas como una serie N×R. Los residuos y predicciones de cada corrida conservarán sus identificadores. El análisis de semillas del capítulo 4 permanece diferenciado de esta inferencia temporal.

Se registrará el estimador de varianza de largo plazo de la serie de diferencias, su kernel, ancho de banda/regla de selección y tamaño efectivo. **h=1 no implica independencia**: σ7 comparte seis retornos entre etiquetas consecutivas y puede existir dependencia adicional. Seis tampoco es un límite superior universal del ancho de banda. La regla y su sensibilidad se fijarán con desarrollo o mediante un procedimiento justificado antes de interpretar los resultados, sin buscar el rezago que minimice el p-value. Una varianza de largo plazo nula o no admisible no se transforma en evidencia favorable: se registra la prueba como no calculable y su causa.

Se documentará la corrección **Harvey–Leybourne–Newbold** requerida por la guía, usando el horizonte real h. No se sustituirá h por la ventana n del objetivo ni por la longitud D de entrada para representar dependencia adicional. El ajuste clásico de muestra pequeña no es una calibración general para cualquier HAC, cambio de régimen o proceso de estimación: se explicarán la implementación, sus condiciones y, cuando proceda, un análisis temporal de sensibilidad justificado. Los p-values se identificarán como aproximados cuando lo sean.

Para T fechas consecutivas y varianza de largo plazo admisible $\widehat\Omega_d>0$, las expresiones usuales son:

$$
DM=\frac{\sqrt T\,\bar d}{\sqrt{\widehat\Omega_d}},\qquad
DM_{\mathrm{HLN}}=DM\sqrt{\frac{T+1-2h+h(h-1)/T}{T}}.
$$

Con h=1, el factor HLN es $\sqrt{(T-1)/T}$; no representa la dependencia adicional por σ7. Se conservarán el estimador HAC y la distribución de referencia justificados para el diseño. Escribir este factor no convierte el p-value resultante en exacto ni garantiza calibración fuera de sus condiciones.

La interpretación requiere condiciones defendibles del proceso de pérdidas, incluidos momentos y dependencia adecuados para estimar su varianza de largo plazo. Se revisarán la actualización recursiva, los pares anidados y la selección previa de modelos: la distribución estándar no se presume válida en todos esos diseños. Los tramos y periodos principales se fijarán antes de estudiar las diferencias; no se concatenarán huecos ni se seleccionarán segmentos por producir significación. Si deriva o falta de observaciones impide sostener una calibración, se declarará la limitación o el método alternativo registrado.

Rechazar igualdad de pérdidas no demuestra que la mejora supere el **5 % frente a persistencia**, y no establece un margen para un modelo avanzado frente al baseline DL. Esas afirmaciones necesitan su frontera nula, efecto y procedimiento propios, conforme a §4.1.5–§4.1.7 y §5.3.4.

### 5.4.6. BDS sobre residuos

Se aplicará **BDS —Brock–Dechert–Scheinkman—** a residuos fuera de entrenamiento, incluido el primer horizonte exigido en el material recibido. Se documentarán tramo, número de observaciones, dimensiones de inmersión, radio o regla de escala, tratamiento de valores ausentes y versión del software.

BDS contrasta una hipótesis de comportamiento i.i.d. bajo sus condiciones. Un rechazo puede reflejar dependencia u otras desviaciones y **no demuestra por sí solo no linealidad estructural**. Un no rechazo tampoco acredita independencia absoluta ni que se haya agotado toda capacidad predictiva.

Los residuos se ordenarán y se evaluarán sobre tramos temporalmente válidos; no se sustituirán por retornos crudos del EDA. Las pruebas de diferentes dimensiones, modelos o configuraciones se incluirán en la política de multiplicidad cuando sustenten afirmaciones confirmatorias.

Se registrará cuántos vectores de inmersión quedan disponibles para cada dimensión y cómo se fija el radio respecto a la escala. Un prefiltrado o estandarización define otra serie: se declararán transformación, datos de ajuste e hipótesis sobre esa nueva serie. Modelos estimados, deriva, heterocedasticidad y etiquetas solapadas afectan la interpretación o calibración; no se agrupan semillas ni se omiten parametrizaciones desfavorables.

### 5.4.7. Ljung–Box e interpretación de residuos

Se usará **Ljung–Box** para estudiar autocorrelación residual en varios rezagos predefinidos. Se registrarán tamaño de muestra, rezagos, hipótesis, estadístico, p-values y tratamiento de grados de libertad.

La corrección de grados de libertad se justificará según el modelo y los parámetros de la prueba: **no se restará automáticamente el número de pesos de una red neuronal** como si fueran los parámetros ARMA de una fórmula clásica. También podrá examinarse la dependencia de residuos al cuadrado con una finalidad explícita, diferenciándola de la autocorrelación lineal.

Sesgo, autocorrelación o estructura no explicada pueden motivar investigar mejoras, pero no prueban que un modelo avanzado concreto vaya a aprovecharlas. La interpretación incorporará heterocedasticidad, cambios de régimen, especificación del objetivo y los límites de cada contraste.

El p-value se identificará como referencia asintótica bajo sus condiciones. Se registrará `model_df` y se comprobarán grados de libertad positivos para los rezagos elegidos; el valor usado en las Ridge del EDA no se hereda automáticamente a las redes. Holm controla multiplicidad de p-values válidos, pero no repara una calibración inválida ni convierte diagnósticos correlacionados en confirmaciones independientes.

### 5.4.8. Ausencia de significación y equivalencia

**No rechazar una diferencia no demuestra equivalencia.** Si se desea afirmar equivalencia entre modelos, se planteará un análisis separado, por ejemplo **TOST** con margen práctico predefinido. Una alternativa bayesiana con **ROPE** deberá declarar modelo, distribución posterior y regla de decisión; no se presentará como si fuera la misma prueba frecuentista.

El margen de equivalencia debe fijarse con sentido aplicado, escala, referencia y unidades claras. El 5 % de mejora frente a persistencia es un criterio de superioridad práctica y no define automáticamente una banda de equivalencia entre dos modelos. No se ensanchará después para conseguir equivalencia. Se distinguirán ausencia de evidencia, evidencia de diferencia y evidencia de equivalencia bajo el diseño correspondiente.

En un TOST al nivel 0,05, bajo el método y la calibración aplicables, ambas pruebas unilaterales deben rechazar; su regla usual corresponde al **IC del 90 % completamente dentro de la banda**. Se distinguirá ese intervalo de decisión del IC del 95 % requerido para el reporte general. Si cambia el nivel por comparación debido a multiplicidad, el intervalo de decisión debe corresponder al procedimiento realmente utilizado.

Las dos unilaterales de un TOST forman una prueba de intersección–unión; no se les aplica por defecto Bonferroni como si fueran dos afirmaciones independientes. Las equivalencias entre varios pares o configuraciones sí requieren su tratamiento de familia. Para ROPE se registrarán también dependencia, priors, posterior y criterio de decisión, sin inventar ahora sus valores.

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

El registro conservará además estimando y agregación de semillas, hash de la cohorte, fase de selección/evaluación, parámetros HAC/HLN o de los diagnósticos, política de estadísticos indefinidos y naturaleza exacta o aproximada del p-value. Las familias de desempeño y de diagnóstico se declararán según las afirmaciones previstas; no se dividirán después para reducir la corrección. Los IC marginales y simultáneos mantendrán su identificación.

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
| Diebold–Mariano con corrección, BDS y Ljung–Box del benchmark | No ejecutados; el Ljung–Box exploratorio de la referencia EDA está en §2.2.3.2.3 |
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

Las conclusiones se limitan a los orígenes con entradas, etiquetas e historia elegibles. No se extrapolan automáticamente a días excluidos; el mecanismo de faltantes no identificado y las máscaras pueden afectar representatividad. La brecha entre selección de 2024 y test puede mezclar optimismo de selección, deriva y diferencias de historia o reajuste: no identifica por sí sola sobreajuste de una arquitectura.

### 5.5.3. Significación estadística y relevancia práctica

La discusión presentará juntas **diferencia estimada, IC, tamaño de efecto, p-value ajustado y relación con $\delta$**. Una diferencia pequeña puede ser estadísticamente distinguible sin aportar utilidad práctica; una diferencia no significativa puede deberse a incertidumbre o baja potencia.

La falta de significación no se interpretará como igualdad o equivalencia. Si se plantea equivalencia, se seguirá el análisis separado de §5.4.8. Se conservarán resultados desfavorables, contrastes no significativos y limitaciones, evitando seleccionar solo las comparaciones que favorezcan a un modelo.

### 5.5.4. Desempeño frente a costo y complejidad

Se discutirá si el cambio de error justifica el incremento de parámetros, tiempo de ajuste, latencia de inferencia y recursos. Los costos se compararán bajo condiciones de medición explícitas y en relación con la frecuencia diaria del uso previsto, no con exigencias operativas inventadas.

Un modelo más simple podrá resultar preferible bajo una regla previamente declarada si la mejora de otro no compensa su costo; esa decisión no equivale automáticamente a demostrar equivalencia estadística. Se explicarán los compromisos y las soluciones no dominadas sin convertir un único indicador en una conclusión universal.

### 5.5.5. Baseline y umbral para un modelo avanzado

La conclusión final deberá nombrar el **modelo de referencia, configuración, objetivo, horizonte, alcance y desempeño con incertidumbre**. También deberá indicar qué mejora detectable y qué margen práctico tendría que superar una propuesta avanzada bajo condiciones comparables.

Se usará la distinción de §5.3.4 entre sensibilidad del diseño y relevancia aplicada. El umbral no será simplemente el límite de un IC del baseline ni un porcentaje elegido después de observar el resultado. Si la incertidumbre no permite discriminar modelos, se declarará la insuficiencia y su consecuencia para el baseline, sin proclamar un ganador definitivo.

Actualmente ese baseline DL y el umbral para comparaciones avanzadas **no están establecidos**. El margen operativo del 5 % frente a persistencia sí está definido, pero responde a otra comparación; los valores de literatura no sustituyen la evaluación propia.

Si los resultados de test del benchmark orientan el diseño o selección de una propuesta avanzada, ese test deja de ser su confirmación independiente. Se necesitará evaluación no utilizada o un esquema externo que hubiera aislado previamente todo el procedimiento de selección; una anidación retrospectiva no elimina la exposición. El baseline congelado deberá evaluarse también en las mismas nuevas fechas. Esta condición no reasigna hoy septiembre–octubre de 2026 ni otro periodo como nuevo test. Si avanzado y contraste ya estuvieran fijados antes de ver test, se declararía ese caso y su plan de inferencia.

### 5.5.6. Brecha con la literatura y margen de mejora potencial

La brecha entre modelos avanzados publicados y las referencias básicas se examinará con los estudios incluidos en la revisión. Se comprobarán definición de volatilidad, escala, frecuencia, horizonte, activo, proveedor, periodos, particiones, selección y número de repeticiones.

Solo bajo comparabilidad suficiente se cuantificará una diferencia directa. Si los estudios usan otros datos o protocolos, se presentarán como contexto y se explicará por qué no permiten calcular un margen numérico trasladable al experimento propio. No se restará una RMSE publicada de la RMSE local sin verificar primero unidades y condiciones.

Se discutirá la credibilidad de la brecha considerando intervalos, variabilidad, sesgos de publicación y calidad metodológica. Una mejora publicada no constituye una garantía de que otra arquitectura la reproduzca en BTCUSDT bajo este calendario.

En §3.4.3 ya se cuantificaron las reducciones puntuales de RMSE de D-TCN frente a TCN en S1: **User 8,53 %, Tweet 5,04 %, VADER −4,65 % y Count −26,74 %**. Proceden de medias redondeadas; cambian arquitectura e información social, con cuarenta reinicios frente a veinte del comparador. Son contexto bibliográfico, no un margen numérico transferible a OHLCV diario, n7/h1 o al baseline propio todavía no medido.

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
| ¿Qué margen sugiere la literatura? | Revisión de estudios, información, protocolo y variabilidad | Brechas publicadas cuantificadas en §3.4.3; margen numérico transferible al proyecto aún no establecido |
| ¿Qué limita las conclusiones? | Registro de validez y alcance de la evidencia | Limitaciones de datos/diseño identificadas; validación experimental pendiente |

La discusión final reemplazará estas respuestas pendientes por conclusiones sustentadas, sin convertir una descripción del protocolo en evidencia de superioridad.
