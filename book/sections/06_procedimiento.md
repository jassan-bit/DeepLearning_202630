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

La revisión documental del 03/11 se conserva en el {download}`informe del procedimiento <../_static/e1/revision_procedimiento_20261103.md>` y su [matriz histórica](../_static/e1/revision_procedimiento_20261103.json). La actualización del **04/11/2026**, autorizada por el usuario, incorpora el primer ciclo real de desarrollo de MLP, LSTM y CNN1D: **96 corridas de búsqueda, 15 de piloto y 90 de comparación; 201 corridas y 402 ajustes**, porque cada corrida selecciona épocas internamente y realiza un reajuste desde cero. El [resumen de ejecución](../_static/dl_benchmark_20261104/execution_summary.json) registra cero archivos de test abiertos.

## 6.1. Aplicación de la ruta 6.4 de la guía: regresión temporal

### 6.1.1. Partición cronológica y origen móvil

Se respeta **entrenamiento < validación < prueba** por disponibilidad de información. La historia inicial es 2023; los orígenes de validación son los días de 2024; el test previsto comprende 01/01/2025–31/08/2026. Cada entrada termina antes del día de resultado y cada etiqueta usada para ajustar debe estar completamente observada al emitir el pronóstico evaluado.

En notación de calendario, para el resultado del día d se emite al inicio UTC de d, se usa OHLCV de d−D,…,d−1 y se predice σ7(d), conocida al terminar d. Corresponde a $Y_t=\sigma_{t+1}$ de §1.1.1 cuando t identifica ese día de resultado. El OHLCV completo de d no es entrada disponible al comenzar d. Origen, día de resultado y disponibilidad de etiqueta se conservan como campos distintos.

Los orígenes de validación se materializaron con `split_train_val_forwardChaining` y los controles descritos en la [sección 2](02_eda.md). Esos índices no equivalen a entrenamientos. El primer ciclo DL utiliza un **ajuste estático con datos permitidos de 2023**, sin incorporar etiquetas de 2024 ni reajustarse en sus 366 orígenes. Enero–junio de 2024 se usa para selección; julio–diciembre, para piloto y comparación con semillas nuevas. Este último tramo es desarrollo/calibración expuesta, no test independiente.

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

En el ciclo ejecutado, OHLC se transforma con logaritmo y volumen con log1p; la media y desviación de cada canal se ajustan solo con días de entrada únicos cubiertos por el entrenamiento permitido. Y se divide por su media positiva de entrenamiento, sin centrar, y se invierte la transformación para evaluar. No se transfieren medias ni componentes de PCA del EDA. La etiqueta evaluada no interviene en la parada interna.

En la parada interna del origen t, el bloque de validación interna termina como máximo en t−D−1. Si v es su primer día de resultado, el subentrenamiento usa u≤v−D−1, además de calidad y disponibilidad. Escaladores, selección de atributos y transformaciones de Y se ajustan **solo al subentrenamiento**, no a su unión con la validación de parada. En este ciclo, la validación interna empieza el 01/10/2023 y termina en el corte externo permitido de 2023; se elige la duración por su MSE. Después se descartan los pesos internos y se entrena un modelo nuevo con todo el ajuste admisible de 2023 durante ese número de épocas. Los cortes, escaladores y configuraciones están en el [registro de datos](../_static/dl_benchmark_20261104/data_audit.json) y los artefactos descritos en §4.2.

Si las métricas de 2024 intervienen en elegir hiperparámetros o D, son **resultados de selección**. Una evaluación externa de todo el procedimiento requeriría selección anidada en el pasado de cada origen; una parada causal por sí sola no corrige la selección hecha con todo ese año. La evaluación reservada no elegirá las reglas después de observar sus resultados.

La política vigente preserva los ZIP y el calendario completo: un día sin 1.440 minutos elegibles queda ausente, no se calculan retornos a través de huecos y se excluyen ventanas afectadas. Los 80 minutos ausentes y la fila de cierre no estándar del 24/03/2023 permanecen documentados. En esta revisión no se introduce una recuperación de otra granularidad ni una imputación nueva.

### 6.1.3. Familias de referencia y selección documental

La guía propone para regresión temporal MLP sobre covariables rezagadas, LSTM o GRU sobre secuencias de covariables y CNN1D poco profunda. Estas familias son orientativas, pero la selección final debe provenir de la frecuencia observada en la literatura revisada y justificar simplicidad y propósito comparativo. También debe identificar la arquitectura reproducible y los supuestos de cada alternativa.

La propuesta de diseño del proyecto es **MLP sobre covariables rezagadas, LSTM sobre secuencias de covariables y CNN1D poco profunda**. GRU es alternativa para el puesto recurrente. El [registro de propuesta](../_static/literature_e1/guide_aligned_proposal_20261104.json) distingue estas familias curriculares de su respaldo aplicado. El [protocolo ejecutado](../_static/dl_benchmark_20261104/protocol_initial.json) especifica las tres implementaciones propias; la [selección de desarrollo](../_static/dl_benchmark_20261104/selected_configuration.json) fija D=7 común, MLP de ancho 16, LSTM de ancho 32 y CNN1D de ancho 16, con 593, 5.025 y 1.057 parámetros respectivamente.

Los perfiles LSTM–TCN–encoder de la [sección 3](03_estado_arte.md) son fuentes documentales recuperadas, no el trío propuesto. La TCN ganadora de S1 es ancha y de seis niveles; no acredita una CNN1D simple. El encoder queda fuera de la propuesta de tres básicos. Los conteos de los cinco textos se conservan sin inventar apoyo para MLP/CNN1D simple. Una red lineal o una cabeza densa interna tampoco sustituyen una MLP no lineal independiente.

| Orientación de la guía 6.4 | Evidencia disponible | Decisión metodológica |
|---|---|---|
| MLP sobre covariables con rezagos | No hay MLP independiente no lineal acreditada en los cinco textos incluidos | P1 implementada: una capa oculta, ancho 16; completar respaldo aplicado. La densa lineal de S5 y la cabeza de S7 no lo sustituyen. |
| LSTM o GRU sobre covariables | Familia LSTM 4/5, unidireccional 3/5 y GRU 2/5, con variantes distintas | P2 implementada: LSTM unidireccional de una capa, ancho 32; GRU continúa como alternativa. La frecuencia no demuestra superioridad. |
| CNN1D poco profunda | TCN aparece en S1, pero su ganadora es ancha y de seis niveles; no hay CNN1D simple independiente acreditada | P3 implementada: dos convoluciones causales ordinarias de ancho 16, sin dilatación ni conexiones residuales; completar evidencia aplicada independiente. |
| Encoder Transformer, fuera de la lista típica | Dos apoyos grises, S2/S6 | Antecedente documental; no ocupa un puesto de la propuesta actual de tres modelos base. |

Las configuraciones de este primer ciclo sí quedaron congeladas antes de la comparación con semillas nuevas. Eso **no acredita la extracción aplicada de §3.2**: los apoyos MLP/CNN1D simple siguen pendientes en el corpus de cinco textos y no se reemplazan por perfiles de otras arquitecturas ni por haber escrito código propio.

Sea cual sea el trío respaldado, el diseño propio conservará las mismas covariables, etiquetas y orígenes entre modelos. Aplanar una ventana para una MLP o conservar su eje temporal para otra red no permite cambiar el conjunto de información disponible.

### 6.1.4. Pérdida y estrategia de pronóstico

Se conservan **MSE como pérdida y RMSE como métrica primaria** en el ciclo de desarrollo ejecutado. La referencia lineal de EDA minimiza una suma de errores cuadrados con penalización Ridge para estudiar residuos; no se utiliza para elegir una configuración ganadora. Cualquier cambio posterior de pérdida o criterio de selección quedará registrado antes del uso del test, con presupuesto comparable entre modelos.

La ruta 6.4 enumera error cuadrático, absoluto o Huber como opciones; no exige probarlas todas. Se conserva MSE ya acordada. La implementación usa Y dividida por la media de entrenamiento y salida Softplus; todas las métricas se calculan después de invertir la escala. Son decisiones propias registradas, sin atribuirlas a una fuente que no las usa.

La salida principal es una volatilidad escalar no negativa de **un día adelante**. Se emite al inicio del día de resultado, con OHLCV hasta el día previo, y se observa al cierre. Es el siguiente valor de una medida histórica móvil; sus ventanas comparten retornos y no representan siete días enteramente futuros.

El ejemplo sintético 4/3/2 de `timeseries-cv` ilustra una función de partición y no cambia el horizonte de Bitcoin. Cualquier ampliación de horizontes requerirá definir entradas, salidas y estrategia directa, recursiva o multi-salida como otro alcance.

### 6.1.5. Métricas y referencias de contexto

Se reportarán RMSE, MAE, MSE, R², MAPE y MASE sobre predicciones agrupadas por ejecución, con escala y cobertura explícitas. MAPE usa solo etiquetas positivas, sin epsilon e informando los ceros; MASE usa diferencias de etiquetas separadas por un día de calendario y completamente conocidas en el entrenamiento permitido. Escalas nulas o sin pares suficientes producen una métrica no definida, no un denominador arbitrario.

Para MASE, la máscara común $\mathcal I_M$ contiene orígenes con escala train definida y positiva; se promedian sus errores escalados con $N_M=|\mathcal I_M|$ y se informa cobertura $N_M/N$. La máscara proviene del entrenamiento, no del error observado. Si cambia la escala por origen, no se divide una MAE global por una escala elegida después. R² se calcula sobre varios resultados con denominador admisible; por origen unitario es indefinido. Las definiciones completas están en §4.4.

El margen aceptado es **5 % de reducción relativa de RMSE frente a persistencia**, con el mismo objetivo y fechas. Se registró como criterio operativo del estudio y no como utilidad financiera. La relevancia y el contraste estadístico se interpretan por separado en §5. El piloto usa 5 % del RMSE de persistencia como efecto alternativo para detectar una diferencia frente a cero; ese dimensionamiento no demuestra una ganancia superior al 5 %.

La **referencia de contexto de la ruta 6.4 es una regresión lineal regularizada con las mismas covariables rezagadas**. El diagnóstico autorizado ajustó cuatro Ridge con intercepto libre, penalización fija $\alpha=1$ y $5D$ entradas OHLCV estandarizadas exclusivamente con la media y desviación poblacional de su ajuste permitido de 2023. El objetivo es $n=7$, $h=1$, sin transformar su escala. Se minimiza $\sum_i(y_i-b-z_i^\top\beta)^2+\alpha\lVert\beta\rVert_2^2$, sin dividir por el tamaño muestral y sin penalizar el intercepto; la solución determinista se obtuvo con SciPy. Se conservaron las predicciones sin truncarlas y no hubo predicciones negativas.

Cada $D$ utiliza un ajuste estático, sin incorporar etiquetas de 2024: 343, 322, 301 y 280 muestras para $D=7,14,21,28$, respectivamente. En el primer origen de validación se respeta $u\le t-D-1$, y el corte permanece válido en los posteriores. Cada referencia aporta 366 predicciones de 2024, **1.464 en total**. El [resumen y controles](../_static/eda_temporal_reference/summary.json) y las [predicciones con residuos](../_static/eda_temporal_reference/validation_predictions_residuals.csv) permiten auditar el diagnóstico. Ese módulo histórico no hizo tuning, selección de D ni entrenamiento DL. El ciclo DL posterior tiene registros separados y conserva el diagnóstico Ridge sin modificarlo; no convierte automáticamente su penalización en configuración contextual del benchmark.

Persistencia se conserva como referencia adicional para el margen operativo aceptado del 5 % y no ocupa uno de los tres puestos DL. Para $n=7$ resume ocho cierres, mientras una ventana de siete días OHLCV no incluye explícitamente esa misma estadística: la comparación contextual debe conservar esa diferencia de información. Un ingenuo estacional necesita justificar y declarar su periodo antes de evaluarlo.

Con un origen de una sola salida, promediar su RMSE entre días equivale a promediar errores absolutos; no produce la RMSE global. Para ordenar modelos se usa la **media de RMSE calculadas por corrida**, no una RMSE sobre semillas×fechas ni la de un ensemble. El reporte de 2025 y 2026 será separado, aclarando que 2026 termina el 31 de agosto. Cualquier agregado adicional requiere una regla de cobertura y ponderación registrada y se recalculará desde predicciones; no será un promedio de RMSE anuales.

La búsqueda de enero–junio seleccionó **D=7 y MLP como referencia operacional de este ciclo**; su identidad se conservó después del piloto. En julio–diciembre, LSTM obtiene el menor RMSE medio entre las redes, pero **persistencia supera a las tres**: RMSE 0,00482248 frente a 0,00890346 de LSTM, 0,01599031 de MLP y 0,01758669 de CNN1D. Este ranking es descriptivo de desarrollo y no reemplaza retrospectivamente a MLP ni constituye un baseline final de test. Los [resultados completos](../_static/dl_benchmark_20261104/analysis/metrics_summary.csv) separan esa fase, su cobertura e incertidumbre.

### 6.1.6. Inferencia y diagnóstico: pregunta, unidad y estado

El [protocolo inicial](../_static/dl_benchmark_20261104/protocol_initial.json) registró el análisis antes del primer ajuste DL y el [protocolo de comparación](../_static/dl_benchmark_20261104/protocol_final.json) fijó configuraciones y semillas después del piloto. La inferencia ejecutada corresponde a desarrollo: IC t entre 30 semillas por red condicionados al calendario, nueve contrastes pareados con Holm, seis DM principales con Holm y 15 diagnósticos residuales con Holm. No confirma generalización en test.

| Herramienta | Pregunta, unidad y condición de aplicación |
|---|---|
| Diebold–Mariano y HLN | Diferencia de pérdida cuadrática por las mismas fechas y horizonte; política de semillas, varianza de largo plazo y calibración registradas. No contrasta automáticamente la media de RMSE entre semillas ni superar el 5 %. |
| Ljung–Box | Autocorrelación de residuos fuera de entrenamiento, por modelo/configuración/semilla y tramo válido; rezagos y `model_df` justificados, sin restar todos los pesos neuronales. |
| t pareada / Wilcoxon | Diferencias de métricas por ejecuciones emparejadas, con alcance condicionado al calendario. La t se refiere a una media; Wilcoxon requiere sus propias condiciones y no es un reemplazo automático del mismo estimando. |
| Friedman / post-hoc | Comparación de tres modelos en bloques completos defendibles, con tratamiento de empates, dependencia, multiplicidad y potencia. Configuraciones n/D no crean réplicas intercambiables. |
| BDS, complemento de §5.4 de la guía | Hipótesis i.i.d. en residuos del horizonte evaluado; dimensiones, radios, escala y tamaño útil registrados. Un rechazo no identifica exclusivamente no linealidad aprovechable. |

Los orígenes son las observaciones del problema predictivo; la **unidad de inferencia** puede ser la corrida por semilla, una diferencia de pérdidas por fecha o un bloque temporal, según la pregunta. No se mezclan esas unidades ni se concatenan semillas o huecos como una serie más larga. Las semillas describen variación del entrenamiento condicionada a los mismos datos; no son nuevos mercados.

Las σ7 consecutivas comparten seis retornos. Ni h=1 obliga a HAC de rezago cero ni ese solapamiento fija un máximo universal de seis; la regla y sensibilidad se justifican conforme a §5.4.9. HLN utiliza h real y no convierte la aproximación en exacta ante cualquier deriva, dependencia o política de actualización.

En el protocolo del ciclo se registraron hipótesis, estimando, fase, dirección, familia y α nominal 0,05 antes de los ajustes DL; se reportan IC del 95 %, efecto pareado y p-values ajustados y sin ajustar bajo la calibración declarada. La potencia por semillas y la MDE respecto de cero tienen el alcance de §4.1.5–§4.1.7; no prueban por sí solas ganancia superior al 5 %, potencia temporal o equivalencia. El 5 % aceptado corresponde exclusivamente a persistencia.

En el bloque C ya se calcularon 112 valores ACF y 12 diagnósticos Ljung–Box sobre las cuatro Ridge, con Holm en la familia completa. Bajo la referencia asintótica utilizada, los 12 rechazan a α=0,05. Son evidencia exploratoria de dependencia en una referencia limitada, no contrastes del benchmark DL ni prueba de causalidad, no linealidad exclusiva o mejora avanzada. Los diagnósticos de retornos tampoco sustituyen residuos predictivos.

BDS procede del apartado 5.4 común a series de tiempo/regresión temporal y del material específico; no se atribuye a la lista textual de 6.4. Se ejecutó en los residuos de la semilla designada 1001, junto con Ljung–Box, sin seleccionar una semilla favorable. DM usa pérdidas cuadráticas promediadas entre semillas por fecha, HAC Bartlett de 14 días y HLN con h=1; 7/28 son sensibilidades. El bootstrap circular pareado usa 2.000 réplicas y bloques de 14 días, con sensibilidad 7/28: remuestrea fechas de predicciones guardadas, sin reentrenar ni remuestrear semillas. Los resultados y límites están en el [resumen de análisis](../_static/dl_benchmark_20261104/analysis/summary.json).

## 6.2. Correspondencia entre procedimiento y evidencia

Esta matriz distingue **reglas fijadas**, **evidencia ejecutada** y **productos pendientes**. Un índice de validación, una fórmula de inferencia o una ficha de arquitectura no constituye una corrida de red.

| Componente de la ruta | Evidencia o decisión actual | Pendiente o límite |
|---|---|---|
| Tipo y reloj | Tipo 4; OHLCV diaria, origen al inicio UTC y etiqueta al terminar el día; fechas guardadas por predicción | Conservar el contrato en una futura evaluación reservada. |
| Calidad y faltantes | Exclusión estricta del 24/03/2023 y ventanas afectadas, sin comprimir calendario | Mantener máscaras y cobertura comunes; mecanismo de faltantes no identificado. |
| Objetivo e historias | n7/h1; búsqueda D7/14/21/28 y selección común D7 | Sensibilidades del objetivo n14/21/28 no entrenadas en este ciclo. |
| Partición y embargo | Ajuste estático permitido de 2023; corte externo u≤01/01/2024−D−1; búsqueda y calibración separadas dentro de 2024 | El manifiesto de test supervisado y su actualización no están congelados. |
| Corte interno y escalado | Validación interna desde 01/10/2023, embargo D; escalado del subentrenamiento y reajuste nuevo con 2023 permitido | No confundir el ajuste interno con una evaluación externa anidada. |
| Arquitecturas | MLP–LSTM–CNN1D implementadas y configuraciones seleccionadas por el mismo presupuesto | Completar respaldo aplicado independiente para MLP/CNN1D simple. |
| Pérdida y salida | MSE escalada, Softplus e inversión de escala para métricas | No atribuir decisiones propias a las fichas de fuentes. |
| Métricas | RMSE media por corrida; MAE/MSE/R²/MAPE/MASE y cobertura en CSV | Son 184 fechas de calibración, no mediciones de test. |
| Referencia regularizada | Cuatro Ridge diagnósticas, α=1 y 1.464 predicciones de 2024, conservadas | Su comparación bajo el protocolo DL no se ha ejecutado; no mezclar cohortes automáticamente. |
| Revisión ante autocorrelación | Se conserva tipo 4 y se ejecutaron diagnósticos DL de la semilla 1001 | Introducir rezagos del objetivo requeriría contrato nuevo. |
| Inferencia | t/Holm9 por semillas; DM/Holm6 temporal; diagnósticos/Holm15; bootstrap L14, sensibilidades 7/28 | Calibraciones aproximadas, datos expuestos y una sola serie limitan la conclusión. |
| Margen práctico | 5 % frente a persistencia; piloto calculó R requerido máximo 1.212 y MDE | Se ejecutaron 30 semillas por red por el límite registrado; no se alcanza la potencia planificada para ese efecto pequeño. Márgenes DL/avanzado/equivalencia abiertos. |
| Baseline y periodos | MLP seleccionada en búsqueda; LSTM líder descriptiva entre redes en julio–diciembre; persistencia con menor error que todas | No se ha medido el baseline final de 2025/2026. |
| Documentos actuales | [Protocolo inicial](../_static/dl_benchmark_20261104/protocol_initial.json), [plan posterior al piloto](../_static/dl_benchmark_20261104/protocol_final.json), [corridas](../_static/dl_benchmark_20261104/runs.csv) y [ejecución](../_static/dl_benchmark_20261104/execution_summary.json) | Los borradores del 03/11 se conservan como históricos, sin sustituir estos resultados. |

## 6.3. Secuencia de cierre y ejecución posterior

El primer ciclo de desarrollo se ejecutó el 04/11/2026 con n7/h1, OHLCV, MSE/RMSE y exclusión estricta. El protocolo se registró después del EDA, antes del primer ajuste DL; no es un registro previo a observar todos los datos históricos.

| Hito | Evidencia del ciclo de desarrollo | Estado o siguiente dependencia |
|---|---|---|
| Diseño previo a búsqueda o piloto | Tres implementaciones, entorno CPU determinista, cortes y escalado, rejilla y presupuesto registrados | Ejecutado; respaldo aplicado MLP/CNN1D aún parcial. |
| Selección con desarrollo | 96 corridas: cuatro configuraciones por familia/D y dos semillas; enero–junio de 2024 | Seleccionó D7 común y MLP operacional; no se reabrió tras ver julio–diciembre. |
| Piloto de variabilidad | Cinco semillas distintas por modelo, 15 corridas, julio–diciembre de 2024 | Media/DE/CV y dimensionamiento en el [piloto](../_static/dl_benchmark_20261104/pilot_summary.json). |
| Dimensionamiento posterior al piloto | Requerimientos aproximados 1.212/172/724 para MLP/LSTM/CNN1D | Máximo registrado de 30 por red; límite de potencia explícito, no cumplimiento de 0,80. |
| Comparación de desarrollo | 90 corridas nuevas, semillas 1001–1030, sin reutilizar las del piloto | Medidas y análisis publicados; mismas fechas del piloto, por tanto calibración expuesta. |
| Plan definitivo y test | Reserva bruta intacta y cero archivos de test abiertos por el ciclo DL | Protocolo final supervisado y evaluación 2025/2026 pendientes; este plan de desarrollo no los autoriza. |
| Informe | Resultados reales de julio–diciembre, incertidumbre, costos y límites en §4–§5 | No hay resultados reservados de 2025/2026 ni margen avanzado definitivo. |

El piloto puede servir como calibración preliminar, pero su dispersión no se transfiere automáticamente si después cambian arquitectura, hiperparámetros, cohortes o política de ajuste. El dimensionamiento definitivo debe corresponder al procedimiento que se evaluará, con revisión y versiones documentadas. La misma cautela se aplica al uso de resultados de test para diseñar una propuesta avanzada, conforme a §5.5.5.

La evidencia propia incluye las cuatro Ridge del EDA y el primer ciclo DL de desarrollo. Adquisición, EDA e índices mantienen sus registros independientes. El test no se ha consultado. La ejecución no cierra por sí sola la extracción bibliográfica, el límite de potencia o el margen para modelos avanzados que exige justificar la guía.

La guía E1 anuncia que la ruta completa del benchmark de diez modelos se desarrolla en E2. Los documentos completos de E2/E3 no se han aportado en esta revisión. El detalle de ese alcance se definirá con sus guías y el protocolo correspondiente; la mención de diez modelos no constituye una selección de arquitecturas, configuraciones o presupuesto ya realizada.
