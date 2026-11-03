# 6. Guía de Procedimiento por Tipo de Problema

La formulación actual corresponde al **tipo 4: regresión temporal**, con la ruta del **apartado 6.4 de la guía del Entregable 1**. Las redes reciben ventanas de covariables OHLCV y no rezagos explícitos de la volatilidad objetivo. La clasificación describe el contrato de entrada vigente; el diagnóstico de cuatro referencias Ridge ejecutado en el bloque C encuentra dependencia residual, sin cambiar automáticamente ese contrato. El orden cronológico y la dependencia entre fechas se conservan en la partición y la inferencia.

La numeración de este capítulo es continua dentro del libro. La referencia 6.4 identifica la ruta de la guía original; no obliga a reproducir las rutas de otras modalidades.

## 6.1. Ruta aplicable: regresión temporal

### 6.1.1. Partición cronológica y origen móvil

Se respeta **entrenamiento < validación < prueba** por disponibilidad de información. La historia inicial es 2023; los orígenes de validación son los días de 2024; el test previsto comprende 01/01/2025–31/08/2026. Cada entrada termina antes del día de resultado y cada etiqueta usada para ajustar debe estar completamente observada al emitir el pronóstico evaluado.

Los orígenes de validación se materializaron con `split_train_val_forwardChaining` y los controles descritos en la [sección 2](02_eda.md). Son calendarios e índices de preparación, no corridas de entrenamiento. Su existencia no fija si las redes se reajustarán diariamente, mensualmente o con otra cadencia; esa política se declarará en E2 antes de ejecutarla.

La ruta 6.4 exige un **embargo igual al máximo rezago de las covariables**. Para $X_t=(z_{t-D},\ldots,z_{t-1})^\top$, los rezagos son de uno a $D$ días de calendario y se fija $E_D=D$. Se define embargo como el número de días de resultado inmediatamente anteriores a $t$ que se omiten del ajuste: $t-D,\ldots,t-1$. Así, los candidatos de entrenamiento deben cumplir $u\le t-D-1$, además de tener entradas válidas y etiquetas conocidas al emitir el pronóstico. Esta convención deja verificable la diferencia entre días omitidos y distancia entre dos fechas.

| Historia de covariables $D$ | Máximo rezago y embargo $E_D$ | Último día de resultado admisible en el ajuste del origen $t$ |
|---|---|---|
| 7 días | 7 días de calendario omitidos | $t-8$ |
| 14 días | 14 días de calendario omitidos | $t-15$ |
| 21 días | 21 días de calendario omitidos | $t-22$ |
| 28 días | 28 días de calendario omitidos | $t-29$ |

**Preparación corregida.** La versión 2 de `results/cv_forward_chaining/folds.csv` aplica esta convención y conserva **5.856 filas: 366 orígenes por cada una de las 16 configuraciones**. El último día de resultado del ajuste es $t-D-1$ en todas las filas. Se retiró una muestra de ajuste adicional por fila respecto de la versión anterior, que terminaba en $t-D$ y omitía $D-1$ días intermedios. Los índices, fechas y hashes se comprobaron mediante una reconstrucción independiente; los CSV de desarrollo y la exclusión del 24/03/2023 permanecen intactos. El [manifiesto vigente](../_static/cv_forward_chaining/folds.csv) y su [resumen de verificación](../_static/cv_forward_chaining/summary.json) documentan la regla, el hash y los recuentos corregidos. Esta preparación no entrena modelos ni abre archivos de test.

La historia observable anterior a validación o test puede formar parte de sus entradas, incluidos los días cuyo resultado se omite del ajuste por embargo. Se revisan por separado el soporte temporal y la disponibilidad de entradas y etiquetas. El embargo de covariables no elimina automáticamente el solapamiento entre volatilidades móviles: con la convención actual, si $n>D+1$, la última etiqueta de ajuste y la de validación comparten $n-D-1$ retornos. Esto ocurre en **2.196 filas de seis configuraciones de sensibilidad** del manifiesto; se registra como dependencia, no como consulta del retorno futuro. Esa dependencia debe conservarse en el diseño inferencial. Incorporar una etiqueta pasada de 2024, cuando ya se conoce y cumple los filtros del entrenamiento permitido, es diferente de consultar la etiqueta futura que se evalúa.

### 6.1.2. Ventanas y preprocesamiento causal

La adquisición es de un minuto y el modelado es **diario UTC**. La entrada conserva cinco variables OHLCV de $D=7,14,21,28$ días. El objetivo principal es la desviación poblacional de **$n=7$ retornos diarios**, pronosticada un día adelante. Las volatilidades de 14/21/28 retornos constituyen sensibilidad del objetivo; sus RMSE no se ordenan para elegir una supuesta mejor definición.

La ruta de regresión temporal requiere estudiar frecuencia, desfases y disponibilidad de covariables, su relación con el objetivo, cambios de régimen y autocorrelación de los errores de una referencia. Los diagnósticos de ACF/PACF y estacionariedad ya presentes en el EDA son información temporal complementaria; no constituyen por sí solos una selección de ventana o un resultado de modelos. La correspondencia de los requisitos específicos y la evidencia debe mantenerse explícita en el capítulo 2.

El bloque C de la guía exige diagnosticar residuos de una referencia y reconsiderar el planteamiento si queda autocorrelación importante. La sección 2.2 aporta el contrato de disponibilidad, correlaciones cruzadas de las cinco covariables con 28 rezagos sobre las mismas 308 fechas elegibles de 2023, relaciones móviles de 60 días con al menos 50 pares y residuos fuera de entrenamiento de la referencia definida en §6.1.5. Los máximos de correlación y las variaciones móviles son descriptivos: no seleccionan una historia de entrada ni demuestran cambios de régimen.

Los errores de las cuatro referencias muestran autocorrelación. Las etiquetas consecutivas de volatilidad de siete retornos comparten seis retornos; la especificación lineal y el ajuste estático también limitan la interpretación. Esta evidencia exige conservar la dependencia en la evaluación y revisar la especificación, pero no establece por sí sola que el problema necesite autorregresión del objetivo. Si se introducen rezagos de la volatilidad como entradas, el proyecto pasaría al tipo 3 y habría que actualizar su ruta metodológica.

Las transformaciones aprendidas se ajustarán con entrenamiento permitido de cada fold. La validación externa no servirá para detener el ajuste sobre su propia etiqueta futura. Las medias y componentes de PCA del EDA no se transferirán automáticamente a un modelo sin volver a respetar el corte temporal.

La política vigente preserva los ZIP y el calendario completo: un día sin 1.440 minutos elegibles queda ausente, no se calculan retornos a través de huecos y se excluyen ventanas afectadas. Los 80 minutos ausentes y la fila de cierre no estándar del 24/03/2023 permanecen documentados. En esta revisión no se introduce una recuperación de otra granularidad ni una imputación nueva.

### 6.1.3. Familias de referencia y selección documental

La guía propone para regresión temporal MLP sobre covariables rezagadas, LSTM o GRU sobre secuencias de covariables y CNN1D poco profunda. Estas familias son orientativas, pero la selección final debe provenir de la frecuencia observada en la literatura revisada y justificar simplicidad y propósito comparativo. También debe identificar la arquitectura reproducible y los supuestos de cada alternativa.

El registro de evidencia, las tres fichas y el estado de la selección se mantienen en la [sección 3](03_estado_arte.md). El trío documental LSTM–TCN–encoder Transformer prioriza diversidad y sigue condicionado a su evidencia: GRU es más frecuente que TCN y Transformer en ese corpus, y la atención depende de una fuente gris. Esa elección de trabajo no acredita por sí sola el requisito de los tres modelos básicos más empleados. Una referencia general de arquitectura tampoco demuestra uso frecuente en volatilidad de Bitcoin; TCN no se equipara automáticamente con una CNN1D simple ni una red lineal con una MLP de capas ocultas.

Sea cual sea el trío respaldado, el diseño propio conservará las mismas covariables, etiquetas y orígenes entre modelos. Aplanar una ventana para una MLP o conservar su eje temporal para otra red no permite cambiar el conjunto de información disponible.

### 6.1.4. Pérdida y estrategia de pronóstico

Se fijan **MSE como pérdida del diseño y RMSE como métrica primaria** para la comparación de redes de E2. La referencia lineal de EDA minimiza una suma de errores cuadrados con penalización Ridge para estudiar residuos; no se utiliza para elegir una configuración ganadora. Cualquier cambio posterior de pérdida o criterio de selección quedará registrado antes del uso del test, con presupuesto comparable entre modelos.

La salida principal es una volatilidad escalar no negativa de **un día adelante**. Se emite al inicio del día de resultado, con OHLCV hasta el día previo, y se observa al cierre. Es el siguiente valor de una medida histórica móvil; sus ventanas comparten retornos y no representan siete días enteramente futuros.

El ejemplo sintético 4/3/2 de `timeseries-cv` ilustra una función de partición y no cambia el horizonte de Bitcoin. Cualquier ampliación de horizontes requerirá definir entradas, salidas y estrategia directa, recursiva o multi-salida como otro alcance.

### 6.1.5. Métricas y referencias de contexto

Se reportarán RMSE, MAE, MSE, R², MAPE y MASE sobre predicciones agrupadas por ejecución, con escala y cobertura explícitas. MAPE usa solo etiquetas positivas, sin epsilon e informando los ceros; MASE usa diferencias de etiquetas separadas por un día de calendario y completamente conocidas en el entrenamiento permitido. Escalas nulas o sin pares suficientes producen una métrica no definida, no un denominador arbitrario.

El margen aceptado es **5 % de reducción relativa de RMSE frente a persistencia**, con el mismo objetivo y fechas. Se registró como criterio operativo del estudio y no como utilidad financiera. La relevancia y el contraste estadístico se interpretarán por separado en E2.

La **referencia de contexto de la ruta 6.4 es una regresión lineal regularizada con las mismas covariables rezagadas**. El diagnóstico autorizado ajustó cuatro Ridge con intercepto libre, penalización fija $\alpha=1$ y $5D$ entradas OHLCV estandarizadas exclusivamente con la media y desviación poblacional de su ajuste permitido de 2023. El objetivo es $n=7$, $h=1$, sin transformar su escala. Se minimiza $\sum_i(y_i-b-z_i^\top\beta)^2+\alpha\lVert\beta\rVert_2^2$, sin dividir por el tamaño muestral y sin penalizar el intercepto; la solución determinista se obtuvo con SciPy. Se conservaron las predicciones sin truncarlas y no hubo predicciones negativas.

Cada $D$ utiliza un ajuste estático, sin incorporar etiquetas de 2024: 343, 322, 301 y 280 muestras para $D=7,14,21,28$, respectivamente. En el primer origen de validación se respeta $u\le t-D-1$, y el corte permanece válido en los posteriores. Cada referencia aporta 366 predicciones de 2024, **1.464 en total**. El [resumen y controles](../_static/eda_temporal_reference/summary.json) y las [predicciones con residuos](../_static/eda_temporal_reference/validation_predictions_residuals.csv) permiten auditar el diagnóstico. No hubo tuning, selección de una D, entrenamiento de redes, piloto, benchmark ni acceso a test. Esta elección diagnóstica no congela la penalización, implementación o cadencia de reajuste del experimento de E2.

Persistencia se conserva como referencia adicional para el margen operativo aceptado del 5 % y no ocupa uno de los tres puestos DL. Para $n=7$ resume ocho cierres, mientras una ventana de siete días OHLCV no incluye explícitamente esa misma estadística: la comparación contextual debe conservar esa diferencia de información. Un ingenuo estacional necesita justificar y declarar su periodo antes de evaluarlo.

Con un origen de una sola salida, promediar su RMSE entre días equivale a promediar errores absolutos; no produce la RMSE global. R² por origen unitario no está definido. El reporte de 2025 y 2026 será separado y agregado, aclarando que 2026 termina el 31 de agosto.

### 6.1.6. Inferencia y diagnóstico previstos

| Herramienta | Pregunta y unidad pertinente |
|---|---|
| Diebold–Mariano | Diferencia de precisión mediante pérdidas por las mismas fechas y horizonte, coherentes con RMSE, con varianza de largo plazo y corrección Harvey–Leybourne–Newbold justificadas |
| Ljung–Box | Autocorrelación de residuos en tramos con tamaño suficiente y rezagos documentados |
| t pareado o Wilcoxon | Diferencias entre ejecuciones bajo emparejamiento y supuestos declarados |
| Friedman y post-hoc | Comparación conjunta con bloques defendibles, multiplicidad y limitaciones de potencia |
| BDS, exigido en §5.4 de la guía y en el material específico | Hipótesis i.i.d. en residuos del primer horizonte, sin interpretar rechazo como prueba exclusiva de no linealidad |

La guía de regresión temporal y el material específico recibido motivan estos procedimientos. El [capítulo 5](05_benchmark.md) conserva su planteamiento para E2. En el bloque C ya se calcularon ACF de los errores de validación y 12 diagnósticos Ljung–Box: rezagos 7/14/28 para cada una de las cuatro referencias, con ajuste Holm conjunto. Los 12 rechazan bajo la referencia asintótica utilizada. Las bandas de ACF son aproximaciones puntuales de ruido blanco; Holm controla la multiplicidad de esa familia, sin corregir una calibración inadecuada. Estos resultados señalan dependencia en los errores de una referencia limitada y no prueban causalidad, no linealidad exclusiva o superioridad de una red. BDS y los contrastes comparativos de modelos siguen previstos para E2. Un diagnóstico sobre retornos crudos no sustituye una prueba sobre errores predictivos.

Las etiquetas móviles de validación consecutivas comparten $n-1$ retornos aunque el horizonte sea un día. La ausencia de retornos compartidos entre ajuste y validación tampoco demuestra independencia entre fechas. La unidad observacional es el origen temporal; la inferencia deberá conservar dependencia mediante bloques u otro diseño justificado. La longitud de bloque y su sensibilidad se fijarán antes de contrastar, sin importar decisiones de corridas ajenas a esta revisión. Semillas, fechas y folds no se suman como observaciones independientes del mismo tipo.

## 6.2. Correspondencia entre procedimiento y evidencia

| Componente | Evidencia o decisión en E1 | Ejecución posterior |
|---|---|---|
| Fuente, frecuencia y reloj | Secciones 1.2–1.5 y 2.3 | Conservar procedencia y comprobar disponibilidad por muestra |
| Calidad y faltantes | EDA 2023–2024, exclusión estricta y calendario conservado | Mantener reglas al formar muestras y registrar enmiendas |
| Objetivo y ventanas | $n=7$, $h=1$, cuatro historias D; otras n como sensibilidad | Sin seleccionar objetivos mediante RMSE de test |
| Partición y origen móvil | Manifiesto versión 2 con $E_D=D$ aplicado; 5.856 filas verificadas y hashes actualizados | Cerrar política de reajuste, test supervisado y tratamiento inferencial del soporte compartido |
| Arquitecturas | Evidencia, selección y fichas documentales en §3 | Implementar configuraciones justificadas en E2 |
| Pérdida y métricas | MSE/RMSE, MAPE y MASE definidas en §1.4 | Calcular predicciones y métricas sin consultar test para seleccionar |
| Margen de relevancia | Regla relativa del 5 % frente a persistencia | Separar relevancia, incertidumbre y significación |
| Referencia lineal y clasificación | Cuatro Ridge diagnósticos con $\alpha=1$; 1.464 predicciones de 2024 y autocorrelación residual documentada; tipo 4 vigente por contrato | Definir la referencia del benchmark y su reajuste; revisar el tipo si se añaden rezagos del objetivo |
| Inferencia | Unidades, supuestos y riesgo de dependencia descritos | Dimensionar, ejecutar y reportar IC/efectos/contrastes en E2 |

## 6.3. Condiciones para la ejecución posterior

Antes de E2 deben revisarse el cierre bibliográfico y los detalles arquitectónicos que permanezcan sin reportar; cerrar transformaciones, test supervisado y cadencia de actualización; definir la familia de comparaciones, tratar la dependencia de etiquetas y dimensionar el piloto. El embargo $E_D=D$ ya está aplicado y verificado en los índices de desarrollo. El diagnóstico lineal de EDA está ejecutado; su dependencia residual informa estos pendientes, sin resolver el benchmark ni fijar el reajuste de las redes. Las decisiones ya aceptadas —objetivo de siete retornos, horizonte de un día, RMSE y margen del 5 %— no figuran como elecciones nuevamente pendientes.

Los resultados de literatura se identifican como publicados por otros autores. La evidencia propia incorporada aquí se limita al diagnóstico lineal de EDA; no hay resultados comparativos de las tres redes ni mediciones del piloto o test. La documentación del procedimiento permite revisar la fase experimental posterior y conserva la diferencia entre el alcance pedido por el usuario y la distribución literal de requisitos del PDF.
