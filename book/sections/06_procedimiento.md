# 6. Guía de Procedimiento por Tipo de Problema

La formulación actual corresponde al **tipo 4: regresión temporal**, con la ruta del **apartado 6.4 de la guía del Entregable 1**. Las redes reciben ventanas de covariables OHLCV y no rezagos explícitos de la volatilidad objetivo. La clasificación describe el contrato de entrada vigente; el diagnóstico de cuatro referencias Ridge ejecutado en el bloque C encuentra dependencia residual, sin cambiar automáticamente ese contrato. El orden cronológico y la dependencia entre fechas se conservan en la partición y la inferencia.

La numeración de este capítulo es continua dentro del libro. La referencia 6.4 identifica la ruta de la guía original; no obliga a reproducir las rutas de otras modalidades.

La guía contiene **once rutas numeradas, de 6.1 a 6.11**. Su lista inicial agrupa etiquetado secuencial y secuencia a secuencia bajo el ítem 10; aquí se conservan esos rótulos sin convertirlos en una nueva enumeración de tipos. Para el contrato vigente se desarrolla únicamente la ruta 6.4:

| Ruta de la guía | Tipo de tarea | Aplicabilidad al proyecto |
|---|---|---|
| 6.1 | Clasificación tabular | No: el objetivo es una magnitud continua, sin clases. |
| 6.2 | Regresión tabular | No es la ruta principal: que OHLCV sea numérico no elimina el reloj de información ni permite barajar fechas. |
| 6.3 | Series de tiempo | No es la ruta principal mientras no se incluyan rezagos explícitos del objetivo; se revisará si cambia ese contrato. |
| **6.4** | **Regresión temporal** | **Aplicable: OHLCV rezagadas, salida continua, disponibilidad causal y partición cronológica con embargo.** |
| 6.5 | Regresión espacial | No hay coordenadas, vecindad o unidades espaciales de predicción. |
| 6.6 | Regresión espacio-temporal | No se predicen ubicaciones ni se construye un tensor o grafo espacial. |
| 6.7 | Clasificación de imágenes | Las entradas no son imágenes y no hay clases visuales. |
| 6.8 | Segmentación de imágenes | No existen máscaras ni objetivos por píxel. |
| 6.9 | Clasificación de texto | No se incorporan documentos o etiquetas textuales; citar estudios con Twitter no añade esa modalidad. |
| 6.10 | Etiquetado secuencial | No se asigna una etiqueta a cada elemento de la ventana de entrada. |
| 6.11 | Aprendizaje secuencia a secuencia | La salida es una volatilidad escalar de un día, no una secuencia de salida. |

Usar LSTM o Transformer y una entrada D×5 no cambia por sí solo el tipo de tarea. La clasificación como tipo 4 tampoco afirma independencia o exogeneidad de OHLCV respecto a la volatilidad: los precios intervienen en construir retornos y pueden aportar información sobre volatilidad pasada. La distinción operativa de la guía se aplica a las entradas explícitas registradas.

El procedimiento principal utiliza validación forward chaining de `timeseries-cv`, con coautoría de Filipe R. Ramos, y las mismas tres redes MLP/LSTM/CNN1D para D=7/14/21/28. Los doce bloques mensuales de 2024 generan 120 trayectorias primarias y 43.920 predicciones; D7 se reutiliza de su registro relativo inicial. El [método vigente](../_static/filipe_multihistory_20261104/method.json) distingue esos ajustes de los 120 modelos finales y de su evaluación posterior sobre test, sin reabrir decisiones con sus resultados.

## 6.1. Aplicación de la ruta 6.4 de la guía: regresión temporal

### 6.1.1. Partición cronológica y origen móvil

Se respeta **entrenamiento < validación < prueba** por disponibilidad de información. La historia inicial es 2023; los orígenes de validación son los días de 2024; el test final comprende 01/01/2025–31/08/2026. Cada entrada termina antes del día de resultado y cada etiqueta usada para ajustar debe estar completamente observada al emitir el pronóstico evaluado.

En notación de calendario, para el resultado del día d se emite al inicio UTC de d, se usa OHLCV de d−D,…,d−1 y se predice σ7(d), conocida al terminar d. Corresponde a $Y_t=\sigma_{t+1}$ de §1.1.1 cuando t identifica ese día de resultado. El OHLCV completo de d no es entrada disponible al comenzar d. Origen, día de resultado y disponibilidad de etiqueta se conservan como campos distintos.

Se mantiene **2023 como entrenamiento inicial, 2024 como validación forward chaining y 2023–2024 para el ajuste final tras cerrar las configuraciones**. El test evaluado comprende enero de 2025–agosto de 2026; septiembre–octubre de 2026 permanece sin asignación automática.

Se utiliza la función real `split_train_val_forwardChaining(np.arange(731),D,1,1)` de `timeseries-cv` 0.1.5 para D=7/14/21/28. Sus orígenes diarios se reúnen en **doce bloques mensuales**: cada red, historia y semilla se ajusta al comienzo del mes con índices de la función filtrados por embargo y calidad. Los pesos quedan fijos dentro del mes y cada pronóstico h=1 usa OHLCV hasta el día anterior. La agrupación mensual es una adaptación explícita, no una API nativa ni 366 reajustes.

Las cohortes crecen con el pasado permitido de 2024. Los manifiestos por historia y las tablas de CV conservan los índices realmente usados. Se reúnen las 366 predicciones por trayectoria para RMSE/R² anual; no se promedian R² de orígenes unitarios. La selección de historia y familia usa ese desarrollo ya expuesto: no es validación anidada independiente de todo el procedimiento.

La ruta 6.4 exige un **embargo igual al máximo rezago de las covariables**. Para $X_t=(z_{t-D},\ldots,z_{t-1})^\top$, los rezagos son de uno a $D$ días de calendario y se fija $E_D=D$. Se define embargo como el número de días de resultado inmediatamente anteriores a $t$ que se omiten del ajuste: $t-D,\ldots,t-1$. Así, los candidatos de entrenamiento deben cumplir $u\le t-D-1$, además de tener entradas válidas y etiquetas conocidas al emitir el pronóstico. Esta convención deja verificable la diferencia entre días omitidos y distancia entre dos fechas.

| Historia de covariables $D$ | Máximo rezago y embargo $E_D$ | Último día de resultado admisible en el ajuste del origen $t$ |
|---|---|---|
| 7 días | 7 días de calendario omitidos | $t-8$ |
| 14 días | 14 días de calendario omitidos | $t-15$ |
| 21 días | 21 días de calendario omitidos | $t-22$ |
| 28 días | 28 días de calendario omitidos | $t-29$ |

Como ejemplo de frontera para **t=01/01/2024 y D=7**, los resultados de 25–31/12/2023 se omiten del ajuste y el último resultado admisible es 24/12/2023. Las covariables completas de 25–31/12 sí forman la entrada conocida del pronóstico de 01/01. Su etiqueta se conoce al terminar 01/01, al inicio UTC de 02/01. Es una ilustración de índices y disponibilidad, no una corrida nueva ni una decisión de reajuste diario.

**Preparación corregida.** La versión 2 de `results/cv_forward_chaining/folds.csv` aplica esta convención y conserva **5.856 filas: 366 orígenes por cada una de las 16 configuraciones**. El último día de resultado del ajuste es $t-D-1$ en todas las filas. Se retiró una muestra de ajuste adicional por fila respecto de la versión anterior, que terminaba en $t-D$ y omitía $D-1$ días intermedios. Los índices, fechas y hashes se comprobaron mediante una reconstrucción independiente; los CSV de desarrollo y la exclusión del 24/03/2023 permanecen intactos. El [manifiesto vigente](../_static/cv_forward_chaining/folds.csv) y su [resumen de verificación](../_static/cv_forward_chaining/summary.json) documentan la regla, el hash y los recuentos corregidos. Esta preparación no entrena modelos ni abre archivos de test.

La historia observable anterior a validación o test puede formar parte de sus entradas, incluidos los días cuyo resultado se omite del ajuste por embargo. Se revisan por separado el soporte temporal y la disponibilidad de entradas y etiquetas. El embargo de covariables no elimina automáticamente el solapamiento entre volatilidades móviles: con la convención actual, si $n>D+1$, la última etiqueta de ajuste y la de validación comparten $n-D-1$ retornos. Esto ocurre en **2.196 filas de seis configuraciones de sensibilidad** del manifiesto; se registra como dependencia, no como consulta del retorno futuro. Esa dependencia debe conservarse en el diseño inferencial. Incorporar una etiqueta pasada de 2024, cuando ya se conoce y cumple los filtros del entrenamiento permitido, es diferente de consultar la etiqueta futura que se evalúa.

### 6.1.2. Ventanas y preprocesamiento causal

La adquisición es de un minuto y el modelado es **diario UTC**. La CV y el test comparan cinco covariables OHLCV con **D=7/14/21/28 días**, manteniendo la desviación poblacional de **n=7 retornos diarios** y horizonte h=1. Las volatilidades n=14/21/28 son sensibilidad del objetivo explorada en EDA, no objetivos entrenados del benchmark.

La ruta de regresión temporal requiere estudiar frecuencia, desfases y disponibilidad de covariables, su relación con el objetivo, cambios de régimen y autocorrelación de los errores de una referencia. Los diagnósticos de ACF/PACF y estacionariedad ya presentes en el EDA son información temporal complementaria; no constituyen por sí solos una selección de ventana o un resultado de modelos. La correspondencia de los requisitos específicos y la evidencia debe mantenerse explícita en el capítulo 2.

El bloque C de la guía exige diagnosticar residuos de una referencia y reconsiderar el planteamiento si queda autocorrelación importante. La sección 2.2 aporta el contrato de disponibilidad, correlaciones cruzadas de las cinco covariables con 28 rezagos sobre las mismas 308 fechas elegibles de 2023, relaciones móviles de 60 días con al menos 50 pares y residuos fuera de entrenamiento de la referencia definida en §6.1.5. Los máximos de correlación y las variaciones móviles son descriptivos: no seleccionan una historia de entrada ni demuestran cambios de régimen.

Los errores de las cuatro referencias muestran autocorrelación. Las etiquetas consecutivas de volatilidad de siete retornos comparten seis retornos; la especificación lineal y la ausencia de adaptación de esa referencia también limitan la interpretación. Esta evidencia exige conservar la dependencia en la evaluación y revisar la especificación, pero no establece por sí sola que el problema necesite autorregresión del objetivo. Si se introducen rezagos de la volatilidad como entradas, el proyecto pasaría al tipo 3 y habría que actualizar su ruta metodológica.

La reconsideración se documenta en esta revisión del **3 de noviembre de 2026**: se mantiene el tipo 4 porque continúa el contrato OHLCV sin historia explícita de σ como entrada. Se reconoce la dependencia residual y no se atribuye a una causa única. Si se decide añadir σ rezagada, antes de ejecutarlo se versionarán entrada, formulación, disponibilidad, partición y comparación bajo la ruta 6.3; el diagnóstico actual no demuestra que ese cambio sea necesario ni que ya haya ocurrido.

La representación principal `relative_window` resta a los cuatro logaritmos OHLC el logaritmo del último cierre de la propia ventana, conocido en d−1; conserva log1p del volumen y aprende escalas con posiciones de ventanas de entrenamiento. El control `absolute_log` se conserva solo en el ciclo inicial D7. El objetivo se divide por su media positiva de entrenamiento y se invierte para evaluar; no se transfieren estadísticas ni PCA globales del EDA.

En la parada interna del origen t, el bloque de validación termina como máximo en t−D−1. Si v es su primer resultado, el subentrenamiento usa u≤v−D−1. Escaladores y escala de Y se ajustan **solo al subentrenamiento**; la cohorte interna reciente de 90 días deja **90−D etiquetas**, es decir 83/76/69/62. Después de elegir duración se descartan los pesos internos y se reajusta una red nueva con todo el entrenamiento mensual permitido. La etapa final toma la duración ya seleccionada en el último corte de CV, usa desarrollo permitido y congela sus pesos antes de abrir test.

Si las métricas de 2024 intervienen en elegir hiperparámetros o D, son **resultados de selección**. Una evaluación externa de todo el procedimiento requeriría selección anidada en el pasado de cada origen; una parada causal por sí sola no corrige la selección hecha con todo ese año. La evaluación reservada no elegirá las reglas después de observar sus resultados.

La política vigente preserva los ZIP y el calendario completo: un día sin 1.440 minutos elegibles queda ausente, no se calculan retornos a través de huecos y se excluyen ventanas afectadas. Los 80 minutos ausentes y la fila de cierre no estándar del 24/03/2023 permanecen documentados. En esta revisión no se introduce una recuperación de otra granularidad ni una imputación nueva.

### 6.1.3. Familias de referencia y selección documental

La guía propone para regresión temporal MLP sobre covariables rezagadas, LSTM o GRU sobre secuencias de covariables y CNN1D poco profunda. Estas familias son orientativas, pero la selección final debe provenir de la frecuencia observada en la literatura revisada y justificar simplicidad y propósito comparativo. También debe identificar la arquitectura reproducible y los supuestos de cada alternativa.

El trío implementado es **MLP sobre covariables rezagadas, LSTM sobre secuencias de covariables y CNN1D poco profunda**. GRU sigue como alternativa documental. El [registro de propuesta](../_static/literature_e1/guide_aligned_proposal_20261104.json) distingue respaldo curricular, bibliografía y ejecución. MLP usa ancho 16 y 80D+33 parámetros; LSTM ancho 32 y 5.025; CNN1D dos capas de ancho 16 y 1.057. H y learning rate se mantienen al ampliar la historia.

Las fichas de la [sección 3](03_estado_arte.md) documentan las implementaciones propias **MLP, LSTM y CNN1D**, con referencias, ecuaciones, flujo, datos, resultados y complejidad. Los perfiles de TCN y encoder se conservan como antecedentes en el [archivo documental de fuentes](https://github.com/jassan-bit/DeepLearning_202630/blob/main/references/literature_e1/source_profiles_archive_20261104.md). No acreditan una CNN1D simple ni completan el respaldo aplicado de MLP. Una red lineal o una cabeza densa interna tampoco sustituyen una MLP no lineal independiente; los conteos de los cinco textos permanecen separados de la evidencia propia.

| Orientación de la guía 6.4 | Evidencia disponible | Decisión metodológica |
|---|---|---|
| MLP sobre covariables con rezagos | No hay MLP independiente no lineal acreditada en los cinco textos incluidos | P1 implementada: una capa oculta, ancho 16; completar respaldo aplicado. La densa lineal de S5 y la cabeza de S7 no lo sustituyen. |
| LSTM o GRU sobre covariables | Familia LSTM 4/5, unidireccional 3/5 y GRU 2/5, con variantes distintas | P2 implementada: LSTM unidireccional de una capa, ancho 32; GRU continúa como alternativa. La frecuencia no demuestra superioridad. |
| CNN1D poco profunda | TCN aparece en S1, pero su ganadora es ancha y de seis niveles; no hay CNN1D simple independiente acreditada | P3 implementada: dos convoluciones causales ordinarias de ancho 16, sin dilatación ni conexiones residuales; completar evidencia aplicada independiente. |
| Encoder Transformer, fuera de la lista típica | Dos apoyos grises, S2/S6 | Antecedente documental; no ocupa un puesto de la propuesta actual de tres modelos base. |

Las configuraciones están documentadas antes de la ejecución CV. La recuperación posterior de Miura et al. (2019), descrita en §3.2, verifica MLP y CNN independientes en un corpus ampliado de seis textos. No reescribe la selección previa ni acredita una réplica exacta: la dimensionalidad de la CNN fuente no se reporta y las recetas propias son adaptaciones.

Sea cual sea el trío respaldado, el diseño propio conservará las mismas covariables, etiquetas y orígenes entre modelos. Aplanar una ventana para una MLP o conservar su eje temporal para otra red no permite cambiar el conjunto de información disponible.

### 6.1.4. Pérdida y estrategia de pronóstico

Se conservan **MSE como pérdida y RMSE como métrica primaria** en el ciclo de desarrollo ejecutado. La referencia lineal de EDA minimiza una suma de errores cuadrados con penalización Ridge para estudiar residuos; no se utiliza para elegir una configuración ganadora. Cualquier cambio posterior de pérdida o criterio se registrará antes de una **nueva evaluación independiente**, con presupuesto comparable entre modelos. El test de 2025–agosto de 2026 ya fue observado y no vuelve a ser una reserva desconocida para diseñar cambios.

La ruta 6.4 enumera error cuadrático, absoluto o Huber como opciones; no exige probarlas todas. Se conserva MSE ya acordada. La implementación usa Y dividida por la media de entrenamiento y salida Softplus; todas las métricas se calculan después de invertir la escala. Son decisiones propias registradas, sin atribuirlas a una fuente que no las usa.

La salida principal es una volatilidad escalar no negativa de **un día adelante**. Se emite al inicio del día de resultado, con OHLCV hasta el día previo, y se observa al cierre. Es el siguiente valor de una medida histórica móvil; sus ventanas comparten retornos y no representan siete días enteramente futuros.

El ejemplo sintético 4/3/2 de `timeseries-cv` ilustra una función de partición y no cambia el horizonte de Bitcoin. Cualquier ampliación de horizontes requerirá definir entradas, salidas y estrategia directa, recursiva o multi-salida como otro alcance.

### 6.1.5. Métricas y referencias de contexto

Se reportarán RMSE, MAE, MSE, R², MAPE y MASE sobre predicciones agrupadas por ejecución, con escala y cobertura explícitas. MAPE usa solo etiquetas positivas, sin epsilon e informando los ceros; MASE usa diferencias de etiquetas separadas por un día de calendario y completamente conocidas en el entrenamiento permitido. Escalas nulas o sin pares suficientes producen una métrica no definida, no un denominador arbitrario.

Para MASE, la máscara común $\mathcal I_M$ contiene orígenes con escala train definida y positiva; se promedian sus errores escalados con $N_M=|\mathcal I_M|$ y se informa cobertura $N_M/N$. La máscara proviene del entrenamiento, no del error observado. Si cambia la escala por origen, no se divide una MAE global por una escala elegida después. R² se calcula sobre varios resultados con denominador admisible; por origen unitario es indefinido. Las definiciones completas están en §4.4.

El margen del estudio ejecutado es **5 % de reducción relativa de RMSE frente a persistencia**, con el mismo objetivo y fechas. Detectar frente a cero un efecto de ese tamaño y demostrar una ganancia superior al margen son preguntas distintas. La CV utiliza diez semillas como presupuesto exploratorio; la sensibilidad retrospectiva no la convierte en un piloto previo ni acredita potencia 0,80.

Para un modelo avanzado futuro se fija un **criterio académico superior al 5 % frente a MLP/D7 y frente a persistencia**, con incertidumbre y multiplicidad declaradas. La elección operativa consta después de observar el test actual, antes de entrenar una propuesta nueva. Los [27 escenarios conjuntos](../_static/e1_audit_improvements_20261104/planning/academic_joint_power_scenarios.csv) usan α unilateral 0,025 y potencia marginal 0,90 por referencia, alternativas comunes de 7,5/10/15 % por debajo de persistencia y dispersión/correlación hipotéticas. La cota conjunta 0,80 depende de esos supuestos; faltan el piloto y la evaluación independiente reales. No se afirma utilidad financiera ni se reasigna un periodo de test.

La **referencia de contexto de la ruta 6.4 es una regresión lineal regularizada con las mismas covariables rezagadas**. El diagnóstico autorizado ajustó cuatro Ridge con intercepto libre, penalización fija $\alpha=1$ y $5D$ entradas OHLCV estandarizadas exclusivamente con la media y desviación poblacional de su ajuste permitido de 2023. El objetivo es $n=7$, $h=1$, sin transformar su escala. Se minimiza $\sum_i(y_i-b-z_i^\top\beta)^2+\alpha\lVert\beta\rVert_2^2$, sin dividir por el tamaño muestral y sin penalizar el intercepto; la solución determinista se obtuvo con SciPy. Se conservaron las predicciones sin truncarlas y no hubo predicciones negativas.

Cada referencia Ridge del EDA se ajustó una vez con muestras admisibles de 2023: 343, 322, 301 y 280 para $D=7,14,21,28$, respectivamente. Respeta $u\le t-D-1$ en el primer origen y aporta 366 predicciones de 2024, **1.464 en total**. El [resumen y controles](../_static/eda_temporal_reference/summary.json) y las [predicciones con residuos](../_static/eda_temporal_reference/validation_predictions_residuals.csv) permiten auditar ese diagnóstico. No hubo tuning, elección de D o entrenamiento DL en dicho módulo; sus residuos no sustituyen los de la CV de las redes.

Persistencia se conserva como referencia adicional para el margen operativo aceptado del 5 % y no ocupa uno de los tres puestos DL. Para $n=7$ resume ocho cierres, mientras una ventana de siete días OHLCV no incluye explícitamente esa misma estadística: la comparación contextual debe conservar esa diferencia de información. Un ingenuo estacional necesita justificar y declarar su periodo antes de evaluarlo.

Con un origen de una sola salida, promediar su RMSE entre días equivale a promediar errores absolutos; no produce la RMSE global. Para ordenar modelos se usa la **media de RMSE calculadas por corrida**, no una RMSE sobre semillas×fechas ni la de un ensemble. El [reporte anual](../_static/e1_audit_improvements_20261104/descriptive/annual_metrics_summary.csv) separa 2025, con 365 fechas, y enero–agosto de 2026, con 243. Cada RMSE se recalcula sobre su periodo y después se resumen semillas; el valor global no se obtiene promediando RMSE anuales.

Las cuatro historias y el baseline elegido con desarrollo se reportan en §5. La selección previa al test quedó cerrada como **MLP con D=7**; no se reemplaza por la combinación que tenga mejor puntuación en test. Persistencia se reporta como referencia determinista, con ocho cierres para sigma7 previa: ese soporte difiere de siete días OHLCV y de las ventanas mayores.

### 6.1.6. Inferencia y diagnóstico: pregunta, unidad y estado

El [suplemento de la evaluación final](../_static/e1_audit_improvements_20261104/inference/summary.json) se registró **después de inspeccionar el test y antes de calcular sus nuevas comparaciones**. Reutiliza los 608 pronósticos por semilla de las tres configuraciones D7 elegidas con CV, más persistencia contextual; no entrena, abre ZIP ni ejecuta nuevos pronósticos. Conserva MLP/D7 como selección previa.

| Producto del suplemento final | Unidad y regla | Evidencia |
|---|---|---|
| Seis contrastes pareados de RMSE | Diez pares de semillas; tres pares neuronales y tres frente a persistencia; t e IC marginales bajo supuestos, Holm-6 y sensibilidad Wilcoxon separada | [Diferencias, IC y d_z](../_static/e1_audit_improvements_20261104/inference/paired_comparisons.csv) |
| Seis DM principales | Media por fecha de pérdidas cuadradas de las diez semillas, h=1, HAC14/HLN; sensibilidades HAC7/28 no son nuevas confirmaciones | [DM y familias](../_static/e1_audit_improvements_20261104/inference/dm_tests.csv) |
| Nueve Ljung–Box | Residuos de semilla designada 4101, rezagos 7/14/28, model_df=0 y Holm-9 | [Diagnóstico residual](../_static/e1_audit_improvements_20261104/inference/ljung_box.csv) |
| Bootstrap temporal pareado | 2.000 remuestras circulares de fechas por L=14 principal y 7/28 sensibilidad; recalcula RMSE/MAE por semilla y su media | [Diferencias e IC](../_static/e1_audit_improvements_20261104/inference/date_bootstrap.csv) |
| Magnitud y planificación | ΔRMSE/ΔMAE absolutas y relativas y MDE retrospectiva; sin declarar potencia lograda; criterio académico futuro separado del test ya observado | [Diferencias relativas](../_static/e1_audit_improvements_20261104/inference/relative_differences.csv) |

Bajo las referencias condicionales declaradas, **cinco de seis** pares por semillas rechazan con Holm-6 y **tres de seis** DM principales rechazan con Holm-6; los nueve Ljung–Box rechazan bajo su referencia asintótica aproximada. Son preguntas distintas: aleatoriedad del ajuste frente a pérdidas en el calendario. La normalidad tiene una bandera Shapiro y se conserva la sensibilidad no paramétrica sin elegir el contraste más favorable. Los IC marginales, incluidos los bootstrap, no son simultáneos solo por ajustar los p-valores. Los detalles e interpretación de cada par están en §5.

**Antecedente separado: CV inicial D7/2024.** El bloque que sigue conserva el análisis de desarrollo anterior y sus familias propias; no se suman sus rechazos ni p-values a los del test final.


El [complemento estadístico inicial D7](../_static/filipe_statistics_20261104/summary.json) se calculó sobre las predicciones guardadas de la CV, sin nuevos ajustes ni acceso a test. Su [enmienda de análisis](../_static/filipe_statistics_20261104/protocol.json) se registró **después de examinar los resultados descriptivos de 2024 y antes de estos cálculos**: conserva alcance exploratorio, sin atribuirle registro previo a toda observación.

En el ciclo inicial **D7**, la familia global de **doce contrastes por semillas** reúne tres pares entre redes relativas, tres comparaciones de redes frente a persistencia, tres pruebas unilaterales de mejora relativa superior al 5 % y tres controles relativa–absoluta. Se reportan diferencia media, IC t marginal, tamaño de efecto, p sin ajustar y **Holm-12** en [global_seed_tests.csv](../_static/filipe_statistics_20261104/global_seed_tests.csv). Las familias de nueve contrastes del complemento y tres controles se conservan como trazabilidad; no son pruebas independientes de corroboración.

| Herramienta ejecutada | Unidad, especificación y alcance |
|---|---|
| t pareada y Holm-12 | Diez pares de semillas por contraste, condicionados a fechas y configuración. La comparación de ganancia superior al 5 % responde una hipótesis distinta de diferencia frente a cero. |
| Friedman exploratorio | Tres redes relativas en diez bloques-semilla; referencia χ² aproximada. Resultado χ²=4,2 y p=0,122456; no se usó como puerta para los contrastes pareados registrados ni se ejecutó Nemenyi. |
| Diebold–Mariano y HLN | Pérdida cuadrática media de las diez semillas en cada fecha; seis pares entre redes y persistencia. Bartlett HAC14 principal, sensibilidad 7/28 y HLN con h=1; Holm-6 principal. |
| Ljung–Box | Residuos de la semilla designada 4101, 366 fechas continuas, rezagos 7/14/28 y `model_df=0`, sin restar los pesos de una red como parámetros ARMA. |
| BDS | Residuos de la misma semilla, dimensiones 2/3 y radio 1,5 veces su desviación muestral. Holm sobre los quince diagnósticos conjuntos. Contrasta i.i.d., no no linealidad aprovechable exclusivamente. |
| Bootstrap temporal | Dos mil réplicas de fechas emparejadas en bloques circulares; longitud principal 14 y sensibilidades 7/28. Recalcula la RMSE por semilla y luego su media, sin remuestrear semillas ni reentrenar. |
| Superioridad y sensibilidad | Fracción de semillas con menor RMSE y su IC binomial condicionado; MDE frente a cero y R orientativo derivados retrospectivamente de la dispersión. No certifican generalización o potencia. |

Los [contrastes DM](../_static/filipe_statistics_20261104/temporal_dm.csv), [diagnósticos](../_static/filipe_statistics_20261104/diagnostics.csv), [intervalos temporales](../_static/filipe_statistics_20261104/block_bootstrap.csv) y [sensibilidad](../_static/filipe_statistics_20261104/design_sensitivity.csv) responden preguntas diferentes. Las σ7 consecutivas comparten seis retornos y los modelos cambian por mes. HAC, HLN y los diagnósticos tienen referencias aproximadas ante esa dependencia y posibles cambios de régimen; el bootstrap requiere una representación defendible de dependencia y estabilidad local.

Ninguna comparación DM principal rechaza al 5 % tras Holm-6, mientras los contrastes condicionales entre semillas sí distinguen las tres redes de persistencia en dirección de mayor error. Los IC bootstrap marginales de esas diferencias son positivos. DM compara medias de pérdidas cuadradas y el bootstrap compara medias de RMSE por semilla; además difieren sus referencias y ajustes. Esta combinación se comunica con cada estimando, sin elegir el procedimiento que produzca la conclusión preferida ni interpretar falta de rechazo como equivalencia. Los IC marginales no son simultáneos por ajustar p-valores.

Los quince diagnósticos iniciales D7 rechazan sus hipótesis bajo las referencias usadas. La estructura residual puede proceder del solapamiento del objetivo, la actualización mensual, deriva o capacidad limitada; no cuantifica automáticamente una mejora futura. Las [curvas de parada interna de diciembre, semilla 4101](../_static/filipe_statistics_20261104/figures/internal_epoch_curves_december_seed4101.png) provienen de registros reales, no son curvas de test ni de tamaño de muestra.

La ampliación calcula BDS sobre los residuos de **cada mes y cada semilla** para las cuatro historias y tres redes. Son **2.880 hipótesis en CV** y **4.800 en test**, dimensiones 2/3 y radio 1,5 DE muestral, con Holm en familias separadas por fase. Los bloques de test son reportes de pesos finales fijos, no reajustes. Los valores no definidos se conservan con su estado; media y DE de p-valores son descriptivas y no constituyen un contraste combinado. Los meses de 28–31 resultados son muestras pequeñas para la referencia asintótica; no se atribuye a BDS no linealidad exclusivamente aprovechable.

El EDA conserva por separado los 112 valores ACF y doce Ljung–Box de las cuatro Ridge, con Holm en su propia familia. Sus rechazos no se mezclan con los diagnósticos de redes ni prueban causalidad. Ni las fechas, ni los doce folds, ni las semillas crean mercados independientes. No se afirma equivalencia: TOST/ROPE y su margen solo se requieren si se propone esa afirmación. El criterio académico avanzado ya se define para una comparación futura frente a ambas referencias; su dimensionamiento real y la nueva evaluación siguen pendientes. Esta definición posterior no cambia el margen histórico contra persistencia ni el registro del test completado.

## 6.2. Correspondencia entre procedimiento y evidencia


Esta matriz distingue reglas, ejecuciones y límites. La comparación de cuatro historias mantiene un mismo objetivo; sus resultados no completan por sí solos los requisitos bibliográficos o de potencia.

| Componente | Evidencia vigente | Límite |
|---|---|---|
| Tipo y frecuencia | Tipo 4, OHLCV diaria UTC desde fuente de un minuto | No son modelos entrenados por minuto. |
| Objetivo y entrada | n=7, h=1; D=7/14/21/28 | n=14/21/28 no entrenados. |
| CV de Filipe | Doce meses reales, entrenamiento creciente y u≤origen−D−1 | 2024 expuesto; selección exploratoria, sin anidamiento independiente. |
| Validación interna | 90−D etiquetas y embargo propio; escalado solo train | Las épocas se eligen con desarrollo, no con test. |
| Ejecución principal | 120 trayectorias, 1.440 segmentos, 2.880 ajustes y 43.920 predicciones | Incluye D7 relativo reutilizado; control absoluto solo D7. |
| Cierre final | Congelación con hashes, selección y 120 épocas por corrida; 120 modelos firmados | Las últimas D etiquetas de desarrollo se omiten por embargo. |
| Prueba | 608 fechas comunes del 01/01/2025 al 31/08/2026; 72960 predicciones | Sin reajustes, reescalado o selección durante test; veinte meses de reporte. |
| Métricas | MAPE/MAE/RMSE/MSE/R²/MASE por D, modelo, semilla y periodo, con coberturas | IC entre semillas condicionados al calendario. |
| BDS por bloque | CV y test, todas las semillas, dimensiones 2/3 y familias Holm separadas | Aproximación mensual de muestra pequeña; promedio p solo descriptivo. |
| Inferencia inicial D7 | Holm-12, DM-6, LB/BDS-15 y bootstrap de CV2024 | Se mantiene como antecedente separado. |
| Suplemento final post-hoc | Seis pares RMSE, DM-6, LB-9 y bootstrap temporal de RMSE/MAE | Registrado después de ver test; no confirma prospectivamente ni reselecciona el baseline. |
| Mejoras descriptivas | Tablas anuales, distribución, descomposición de variación y diagnóstico por segmentos | Agregados calculados sobre predicciones guardadas; sin nuevos pronósticos. |
| Referencia regularizada | Cuatro Ridge diagnósticas de EDA y 1.464 residuos de 2024 | No son un cuarto modelo DL ni nuevos competidores del test. |
| Literatura y potencia | Seis textos, fuente aplicada MLP/CNN y diez semillas propias | Dimensionalidad CNN y recetas fuente incompletas, recurrencia limitada; piloto/potencia prospectivos pendientes. |

El [método vigente](../_static/filipe_multihistory_20261104/method.json), la [congelación](../_static/filipe_multihistory_20261104/freeze.json), la [auditoría de test](../_static/filipe_multihistory_20261104/test/raw_test_audit.json) y el [manifiesto de cohorte](../_static/filipe_multihistory_20261104/test/cohort_manifest.json) vinculan esas etapas.

## 6.3. Secuencia de cierre y ejecución posterior


El registro de noviembre de 2026 es posterior a los periodos de mercado y al EDA de desarrollo; la evaluación es un **backtest histórico**, no predicciones emitidas prospectivamente en 2025–2026.

1. Se conservaron las trayectorias D7 relativas y se ejecutaron D14/21/28 bajo los índices reales de Filipe y las configuraciones comunes.
2. Las tablas y BDS por fold de 2024 se calcularon desde las predicciones guardadas. El criterio RMSE de desarrollo eligió una historia por familia y un baseline neuronal.
3. Antes de abrir test se cerraron las doce combinaciones que se reportarían, la selección, las épocas finales, las semillas, el preprocesamiento, el embargo, la política de calidad y la ausencia de actualización durante test.
4. Se ajustaron y firmaron los 120 modelos finales con datos permitidos de 2023–2024. La cronología conserva las firmas previas a la primera lectura de los veinte ZIP reservados.
5. Se agregaron y auditaron esos ZIP y se construyó la cohorte común por reglas cerradas, sin usar los errores para excluir fechas. Cada modelo produjo su trayectoria final una vez; los análisis posteriores reutilizan sus predicciones.
6. La mejora posterior a la auditoría materializó las métricas por año —2025 y enero–agosto de 2026—, además de las globales y mensuales; documentó variación, segmentos y residuos, y calculó el suplemento inferencial final explícitamente post-hoc. No se reabrió la selección, no se volvió a puntuar el modelo ni se incorporó septiembre–octubre de 2026.

La [congelación](../_static/filipe_multihistory_20261104/freeze.json) conserva el baseline **MLP con D=7** elegido con CV. La fuente aplicada MLP/CNN se recuperó después y conserva los límites documentales de §3; el test no acredita piloto ni potencia prospectivos. El criterio académico avanzado se define por separado para ese futuro; la equivalencia es condicional a afirmar que dos procedimientos son equivalentes. Si se usa este test para diseñar una propuesta posterior, se necesitará otra evaluación independiente.

La guía E1 anuncia que el benchmark de diez modelos se desarrolla en E2. Sus documentos completos y los de E3 no se han aportado; no se inventa una selección de diez arquitecturas o presupuesto a partir de esa mención.
