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

La revisión punto por punto se conserva en el {download}`informe del procedimiento <../_static/e1/revision_procedimiento_20261103.md>` y su [matriz de aplicabilidad y evidencia](../_static/e1/revision_procedimiento_20261103.json). Este capítulo consolida las reglas de las secciones 1–5 y distingue decisiones, evidencia ejecutada y productos pendientes.

## 6.1. Aplicación de la ruta 6.4 de la guía: regresión temporal

### 6.1.1. Partición cronológica y origen móvil

Se respeta **entrenamiento < validación < prueba** por disponibilidad de información. La historia inicial es 2023; los orígenes de validación son los días de 2024; el test previsto comprende 01/01/2025–31/08/2026. Cada entrada termina antes del día de resultado y cada etiqueta usada para ajustar debe estar completamente observada al emitir el pronóstico evaluado.

En notación de calendario, para el resultado del día d se emite al inicio UTC de d, se usa OHLCV de d−D,…,d−1 y se predice σ7(d), conocida al terminar d. Corresponde a $Y_t=\sigma_{t+1}$ de §1.1.1 cuando t identifica ese día de resultado. El OHLCV completo de d no es entrada disponible al comenzar d. Origen, día de resultado y disponibilidad de etiqueta se conservan como campos distintos.

Los orígenes de validación se materializaron con `split_train_val_forwardChaining` y los controles descritos en la [sección 2](02_eda.md). Son calendarios e índices de preparación, no corridas de entrenamiento. Su existencia no fija si las redes se reajustarán diariamente, mensualmente o con otra cadencia; esa política se declarará en E2 antes de ejecutarla.

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

La adquisición es de un minuto y el modelado es **diario UTC**. La entrada conserva cinco variables OHLCV de $D=7,14,21,28$ días. El objetivo principal es la desviación poblacional de **$n=7$ retornos diarios**, pronosticada un día adelante. Las volatilidades de 14/21/28 retornos constituyen sensibilidad del objetivo; sus RMSE no se ordenan para elegir una supuesta mejor definición.

La ruta de regresión temporal requiere estudiar frecuencia, desfases y disponibilidad de covariables, su relación con el objetivo, cambios de régimen y autocorrelación de los errores de una referencia. Los diagnósticos de ACF/PACF y estacionariedad ya presentes en el EDA son información temporal complementaria; no constituyen por sí solos una selección de ventana o un resultado de modelos. La correspondencia de los requisitos específicos y la evidencia debe mantenerse explícita en el capítulo 2.

El bloque C de la guía exige diagnosticar residuos de una referencia y reconsiderar el planteamiento si queda autocorrelación importante. La sección 2.2 aporta el contrato de disponibilidad, correlaciones cruzadas de las cinco covariables con 28 rezagos sobre las mismas 308 fechas elegibles de 2023, relaciones móviles de 60 días con al menos 50 pares y residuos fuera de entrenamiento de la referencia definida en §6.1.5. Los máximos de correlación y las variaciones móviles son descriptivos: no seleccionan una historia de entrada ni demuestran cambios de régimen.

Los errores de las cuatro referencias muestran autocorrelación. Las etiquetas consecutivas de volatilidad de siete retornos comparten seis retornos; la especificación lineal y el ajuste estático también limitan la interpretación. Esta evidencia exige conservar la dependencia en la evaluación y revisar la especificación, pero no establece por sí sola que el problema necesite autorregresión del objetivo. Si se introducen rezagos de la volatilidad como entradas, el proyecto pasaría al tipo 3 y habría que actualizar su ruta metodológica.

La reconsideración se documenta en esta revisión del **3 de noviembre de 2026**: se mantiene el tipo 4 porque continúa el contrato OHLCV sin historia explícita de σ como entrada. Se reconoce la dependencia residual y no se atribuye a una causa única. Si se decide añadir σ rezagada, antes de ejecutarlo se versionarán entrada, formulación, disponibilidad, partición y comparación bajo la ruta 6.3; el diagnóstico actual no demuestra que ese cambio sea necesario ni que ya haya ocurrido.

Las transformaciones aprendidas se ajustarán con entrenamiento permitido de cada fold. La etiqueta futura del origen evaluado no servirá para detener su propio ajuste. Las medias y componentes de PCA del EDA no se transferirán automáticamente a un modelo sin volver a respetar el corte temporal.

En la parada interna del origen t, el bloque de validación interna termina como máximo en t−D−1. Si v es su primer día de resultado, el subentrenamiento usa u≤v−D−1, además de calidad y disponibilidad. Escaladores, selección de atributos y transformaciones de Y se ajustan **solo al subentrenamiento**, no a su unión con la validación de parada. El tamaño de ese bloque y la regla de reajuste final permanecen abiertos en §4.2.3.

Si las métricas de 2024 intervienen en elegir hiperparámetros o D, son **resultados de selección**. Una evaluación externa de todo el procedimiento requeriría selección anidada en el pasado de cada origen; una parada causal por sí sola no corrige la selección hecha con todo ese año. La evaluación reservada no elegirá las reglas después de observar sus resultados.

La política vigente preserva los ZIP y el calendario completo: un día sin 1.440 minutos elegibles queda ausente, no se calculan retornos a través de huecos y se excluyen ventanas afectadas. Los 80 minutos ausentes y la fila de cierre no estándar del 24/03/2023 permanecen documentados. En esta revisión no se introduce una recuperación de otra granularidad ni una imputación nueva.

### 6.1.3. Familias de referencia y selección documental

La guía propone para regresión temporal MLP sobre covariables rezagadas, LSTM o GRU sobre secuencias de covariables y CNN1D poco profunda. Estas familias son orientativas, pero la selección final debe provenir de la frecuencia observada en la literatura revisada y justificar simplicidad y propósito comparativo. También debe identificar la arquitectura reproducible y los supuestos de cada alternativa.

El registro de evidencia, las tres fichas y el estado de la selección se mantienen en la [sección 3](03_estado_arte.md). El trío documental LSTM–TCN–encoder Transformer prioriza diversidad y sigue condicionado a su evidencia: GRU y encoder Transformer aparecen en 2/5 y TCN en 1/5; los dos apoyos de atención son literatura gris. Los ganadores de S1 están publicados en su apéndice y su TCN no acredita una configuración pequeña. Esa elección de trabajo no acredita por sí sola el requisito de los tres modelos básicos más empleados. Una referencia general de arquitectura tampoco demuestra uso frecuente en volatilidad de Bitcoin; TCN no se equipara automáticamente con una CNN1D simple ni una red lineal con una MLP de capas ocultas.

| Orientación de la guía 6.4 | Evidencia disponible | Decisión metodológica |
|---|---|---|
| MLP sobre covariables con rezagos | No hay MLP independiente no lineal acreditada en los cinco textos incluidos | Ampliar evidencia si se propone; la densa lineal de S5 y la cabeza de S7 no la sustituyen. |
| LSTM o GRU sobre covariables | Familia LSTM 4/5, unidireccional 3/5 y GRU 2/5, con variantes distintas | Cerrar variante y configuración; GRU continúa evaluable. |
| CNN1D poco profunda | TCN aparece en S1, pero su ganadora es ancha y de seis niveles; no hay CNN1D simple independiente acreditada | Una simplificación necesita justificación; no se presenta como réplica de una CNN pequeña. |
| Encoder Transformer, fuera de la lista típica | Dos apoyos grises, S2/S6 | La lista orientativa no lo prohíbe ni demuestra que sea básico; deben acreditarse sencillez y suficiencia de evidencia. |

Los tres puestos experimentales siguen sin identidad final. Estas familias orientativas y fichas candidatas no cierran la selección por recurrencia, complejidad y no redundancia de §3.2.

Sea cual sea el trío respaldado, el diseño propio conservará las mismas covariables, etiquetas y orígenes entre modelos. Aplanar una ventana para una MLP o conservar su eje temporal para otra red no permite cambiar el conjunto de información disponible.

### 6.1.4. Pérdida y estrategia de pronóstico

Se fijan **MSE como pérdida del diseño y RMSE como métrica primaria** para la comparación de redes de E2. La referencia lineal de EDA minimiza una suma de errores cuadrados con penalización Ridge para estudiar residuos; no se utiliza para elegir una configuración ganadora. Cualquier cambio posterior de pérdida o criterio de selección quedará registrado antes del uso del test, con presupuesto comparable entre modelos.

La ruta 6.4 enumera error cuadrático, absoluto o Huber como opciones; no exige probarlas todas. Se conserva MSE ya acordada. La escala de entrenamiento, transformación de Y y mecanismo de salida no negativa se cerrarán con la implementación conforme a §4, sin atribuir a una fuente una configuración que no usa.

La salida principal es una volatilidad escalar no negativa de **un día adelante**. Se emite al inicio del día de resultado, con OHLCV hasta el día previo, y se observa al cierre. Es el siguiente valor de una medida histórica móvil; sus ventanas comparten retornos y no representan siete días enteramente futuros.

El ejemplo sintético 4/3/2 de `timeseries-cv` ilustra una función de partición y no cambia el horizonte de Bitcoin. Cualquier ampliación de horizontes requerirá definir entradas, salidas y estrategia directa, recursiva o multi-salida como otro alcance.

### 6.1.5. Métricas y referencias de contexto

Se reportarán RMSE, MAE, MSE, R², MAPE y MASE sobre predicciones agrupadas por ejecución, con escala y cobertura explícitas. MAPE usa solo etiquetas positivas, sin epsilon e informando los ceros; MASE usa diferencias de etiquetas separadas por un día de calendario y completamente conocidas en el entrenamiento permitido. Escalas nulas o sin pares suficientes producen una métrica no definida, no un denominador arbitrario.

Para MASE, la máscara común $\mathcal I_M$ contiene orígenes con escala train definida y positiva; se promedian sus errores escalados con $N_M=|\mathcal I_M|$ y se informa cobertura $N_M/N$. La máscara proviene del entrenamiento, no del error observado. Si cambia la escala por origen, no se divide una MAE global por una escala elegida después. R² se calcula sobre varios resultados con denominador admisible; por origen unitario es indefinido. Las definiciones completas están en §4.4.

El margen aceptado es **5 % de reducción relativa de RMSE frente a persistencia**, con el mismo objetivo y fechas. Se registró como criterio operativo del estudio y no como utilidad financiera. La relevancia y el contraste estadístico se interpretarán por separado en E2.

La **referencia de contexto de la ruta 6.4 es una regresión lineal regularizada con las mismas covariables rezagadas**. El diagnóstico autorizado ajustó cuatro Ridge con intercepto libre, penalización fija $\alpha=1$ y $5D$ entradas OHLCV estandarizadas exclusivamente con la media y desviación poblacional de su ajuste permitido de 2023. El objetivo es $n=7$, $h=1$, sin transformar su escala. Se minimiza $\sum_i(y_i-b-z_i^\top\beta)^2+\alpha\lVert\beta\rVert_2^2$, sin dividir por el tamaño muestral y sin penalizar el intercepto; la solución determinista se obtuvo con SciPy. Se conservaron las predicciones sin truncarlas y no hubo predicciones negativas.

Cada $D$ utiliza un ajuste estático, sin incorporar etiquetas de 2024: 343, 322, 301 y 280 muestras para $D=7,14,21,28$, respectivamente. En el primer origen de validación se respeta $u\le t-D-1$, y el corte permanece válido en los posteriores. Cada referencia aporta 366 predicciones de 2024, **1.464 en total**. El [resumen y controles](../_static/eda_temporal_reference/summary.json) y las [predicciones con residuos](../_static/eda_temporal_reference/validation_predictions_residuals.csv) permiten auditar el diagnóstico. No hubo tuning, selección de una D, entrenamiento de redes, piloto, benchmark ni acceso a test. Esta elección diagnóstica no congela la penalización, implementación o cadencia de reajuste del experimento de E2.

Persistencia se conserva como referencia adicional para el margen operativo aceptado del 5 % y no ocupa uno de los tres puestos DL. Para $n=7$ resume ocho cierres, mientras una ventana de siete días OHLCV no incluye explícitamente esa misma estadística: la comparación contextual debe conservar esa diferencia de información. Un ingenuo estacional necesita justificar y declarar su periodo antes de evaluarlo.

Con un origen de una sola salida, promediar su RMSE entre días equivale a promediar errores absolutos; no produce la RMSE global. Para ordenar modelos se usa la **media de RMSE calculadas por corrida**, no una RMSE sobre semillas×fechas ni la de un ensemble. El reporte de 2025 y 2026 será separado, aclarando que 2026 termina el 31 de agosto. Cualquier agregado adicional requiere una regla de cobertura y ponderación registrada y se recalculará desde predicciones; no será un promedio de RMSE anuales.

El baseline se seleccionará con desarrollo por configuración y se congelará antes de test. La referencia mantiene su identificador al reportar ambos periodos; el ranking observado en test es descriptivo y no reabre la selección. Sin D principal ni una regla global cerrada, no se declara un único ganador para todas las historias. Ningún baseline DL está medido todavía.

### 6.1.6. Inferencia y diagnóstico: pregunta, unidad y estado

El procedimiento inferencial se aplicará a la pregunta y fase declaradas; sus detalles y campos abiertos están en los capítulos 4–5 y sus contratos documentales. Ninguno está congelado ni acredita ejecución de las tres redes.

| Herramienta | Pregunta, unidad y condición de aplicación |
|---|---|
| Diebold–Mariano y HLN | Diferencia de pérdida cuadrática por las mismas fechas y horizonte; política de semillas, varianza de largo plazo y calibración registradas. No contrasta automáticamente la media de RMSE entre semillas ni superar el 5 %. |
| Ljung–Box | Autocorrelación de residuos fuera de entrenamiento, por modelo/configuración/semilla y tramo válido; rezagos y `model_df` justificados, sin restar todos los pesos neuronales. |
| t pareada / Wilcoxon | Diferencias de métricas por ejecuciones emparejadas, con alcance condicionado al calendario. La t se refiere a una media; Wilcoxon requiere sus propias condiciones y no es un reemplazo automático del mismo estimando. |
| Friedman / post-hoc | Comparación de tres modelos en bloques completos defendibles, con tratamiento de empates, dependencia, multiplicidad y potencia. Configuraciones n/D no crean réplicas intercambiables. |
| BDS, complemento de §5.4 de la guía | Hipótesis i.i.d. en residuos del horizonte evaluado; dimensiones, radios, escala y tamaño útil registrados. Un rechazo no identifica exclusivamente no linealidad aprovechable. |

Los orígenes son las observaciones del problema predictivo; la **unidad de inferencia** puede ser la corrida por semilla, una diferencia de pérdidas por fecha o un bloque temporal, según la pregunta. No se mezclan esas unidades ni se concatenan semillas o huecos como una serie más larga. Las semillas describen variación del entrenamiento condicionada a los mismos datos; no son nuevos mercados.

Las σ7 consecutivas comparten seis retornos. Ni h=1 obliga a HAC de rezago cero ni ese solapamiento fija un máximo universal de seis; la regla y sensibilidad se justifican conforme a §5.4.5. HLN utiliza h real y no convierte la aproximación en exacta ante cualquier deriva, dependencia o política de actualización.

Se registrarán hipótesis, estimando, fase, dirección, familia y α nominal 0,05 antes de analizar resultados, con IC del 95 %, efecto pareado y p-values ajustados y sin ajustar bajo la calibración declarada. La potencia por semillas y la MDE respecto de cero tienen el alcance de §4.1.5–§4.1.7; no prueban por sí solas ganancia superior al 5 %, potencia temporal o equivalencia. El 5 % aceptado corresponde exclusivamente a persistencia.

En el bloque C ya se calcularon 112 valores ACF y 12 diagnósticos Ljung–Box sobre las cuatro Ridge, con Holm en la familia completa. Bajo la referencia asintótica utilizada, los 12 rechazan a α=0,05. Son evidencia exploratoria de dependencia en una referencia limitada, no contrastes del benchmark DL ni prueba de causalidad, no linealidad exclusiva o mejora avanzada. Los diagnósticos de retornos tampoco sustituyen residuos predictivos.

BDS procede del apartado 5.4 común a series de tiempo/regresión temporal y del material específico; no se atribuye a la lista textual de 6.4. Su ejecución con las redes y los demás contrastes comparativos permanece pendiente.

## 6.2. Correspondencia entre procedimiento y evidencia

Esta matriz distingue **reglas fijadas**, **evidencia ejecutada** y **productos pendientes**. Un índice de validación, una fórmula de inferencia o una ficha de arquitectura no constituye una corrida de red.

| Componente de la ruta | Evidencia o decisión actual | Pendiente para la ejecución |
|---|---|---|
| Tipo y reloj | Tipo 4; OHLCV diaria, origen al inicio UTC y etiqueta al terminar el día; §1 y §2.2.3.1 | Conservar identificadores y disponibilidad en las muestras definitivas. |
| Calidad y faltantes | Exclusión estricta del 24/03/2023 y ventanas afectadas, sin comprimir calendario | Mantener máscaras y cobertura comunes entre métodos. |
| Objetivo e historias | n7/h1; D7/14/21/28 sin D principal; otros n como sensibilidad | Regla global de D solo si se plantea y se registra en desarrollo. |
| Partición y embargo | Manifiesto v2, E_D=D, 5.856 filas y soporte compartido identificado | Cadencia y política de actualización; el test supervisado no está congelado. |
| Corte interno y escalado | Separación y embargo internos de §4.2.3; transformaciones solo en subentrenamiento | Longitud/posición interna, escaladores, parada y tratamiento de checkpoints. |
| Arquitecturas | Familias típicas de la guía y fichas candidatas de §3; trío abierto | Cerrar exactamente tres identidades y sus configuraciones respaldadas. |
| Pérdida y salida | MSE, salida escalar no negativa y RMSE primaria | Implementación de salida, escala y entrenamiento concretos. |
| Métricas | RMSE por corrida y media entre semillas; MAPE/MASE con cobertura y casos indefinidos | Predicciones, métricas e intervalos medidos por periodo. |
| Referencia regularizada | Cuatro Ridge diagnósticas, α=1 y 1.464 predicciones fuera del ajuste | Referencia contextual del benchmark y su política, sin heredar automáticamente la configuración de EDA. |
| Revisión ante autocorrelación | Se conserva tipo 4 tras revisar residuos y fuentes compatibles de dependencia | Un cambio a rezagos del objetivo exigiría contrato nuevo y revisión de ruta antes de ejecutarse. |
| Inferencia | Pregunta y unidad separadas; α nominal 0,05, IC/efectos y familias en §4–§5 | Agregación de semillas, método, HAC/bloques, supuestos y multiplicidad. |
| Margen práctico | 5 % relativo solo frente a persistencia; diferencia contra cero distinta de superar umbral | Estimar dispersión y R/MDE bajo la pregunta registrada; márgenes DL/avanzado/equivalencia siguen abiertos. |
| Baseline y periodos | Regla de selección con desarrollo por configuración, congelada antes de test | Elegir referencia con resultados de desarrollo y mantenerla en 2025/2026; ranking de test descriptivo. |
| Documentos actuales | [Diseño](../_static/experiments/protocolo_e2_documental_20261103.json) y [benchmark](../_static/experiments/benchmark_documental_20261103.json) documentales, incompletos y sin congelar | Cerrar cada fase y conservar versiones; las plantillas no contienen resultados. |

## 6.3. Secuencia de cierre y ejecución posterior

Las dependencias siguientes ordenan el procedimiento; sus resultados siguen pendientes. La definición ya acordada de n7/h1/D, OHLCV, MSE/RMSE y exclusión estricta se conserva.

| Hito | Qué debe quedar cerrado o documentado | Estado actual |
|---|---|---|
| Diseño previo a búsqueda o piloto | Trío respaldado; arquitectura/entorno, espacios y presupuesto, cortes internos, escalado, parada, cadencia, comparaciones y márgenes | Selección final y campos operativos abiertos. |
| Selección con desarrollo | Aplicar el procedimiento registrado; identificar qué resultados de 2024 se usan para elegir configuración y, si procede, D | No ejecutada con las tres redes; esos resultados serían de selección salvo evaluación verdaderamente anidada. |
| Piloto de variabilidad | Configuraciones y reglas declaradas; al menos cinco repeticiones por modelo y registro de media/DE/CV/diferencias | No ejecutado; no hay dispersión piloto. |
| Dimensionamiento posterior al piloto | R y MDE coherentes con el estimando, la alternativa, los mínimos y la multiplicidad | Sin resultados numéricos. La reutilización del piloto no es automática. |
| Plan definitivo y test | Congelar protocolo antes de ejecución definitiva; referencia seleccionada en desarrollo, manifiestos, actualización, inferencia y reporte antes de consultar test | No congelados ni ejecutados; no se elegirá el baseline por el ganador posterior en test. |
| Informe | Resultados por 2025 y tramo 2026, incertidumbre y límites; agregado adicional solo con regla registrada | Pendiente de predicciones verificables. |

El piloto puede servir como calibración preliminar, pero su dispersión no se transfiere automáticamente si después cambian arquitectura, hiperparámetros, cohortes o política de ajuste. El dimensionamiento definitivo debe corresponder al procedimiento que se evaluará, con revisión y versiones documentadas. La misma cautela se aplica al uso de resultados de test para diseñar una propuesta avanzada, conforme a §5.5.5.

La evidencia propia de **ajuste y predicción** incorporada hasta ahora corresponde a las cuatro Ridge del EDA. Adquisición, calidad, análisis exploratorio e índices tienen sus evidencias independientes. No hay comparación DL, piloto o test ejecutado. Este capítulo consolida la ruta; no sustituye esos productos ni modifica lo que exige literalmente la guía.

La guía E1 anuncia que la ruta completa del benchmark de diez modelos se desarrolla en E2. Los documentos completos de E2/E3 no se han aportado en esta revisión. El detalle de ese alcance se definirá con sus guías y el protocolo correspondiente; la mención de diez modelos no constituye una selección de arquitecturas, configuraciones o presupuesto ya realizada.
