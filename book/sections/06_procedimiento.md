# 6. Guía de Procedimiento por Tipo de Problema

La formulación actual sigue la ruta de **Regresión Temporal del apartado 6.4 de la guía del Entregable 1**. Las redes reciben ventanas de covariables OHLCV y no rezagos explícitos de la volatilidad objetivo. Se conserva una tarea de pronóstico cronológico: esta clasificación no convierte las fechas en observaciones independientes ni permite barajar el tiempo.

La numeración de este capítulo es continua dentro del libro. La referencia 6.4 identifica la ruta de la guía original; no obliga a reproducir las rutas de otras modalidades.

## 6.1. Ruta aplicable: regresión temporal

### 6.1.1. Partición cronológica y origen móvil

Se respeta **entrenamiento < validación < prueba** por disponibilidad de información. La historia inicial es 2023; los orígenes de validación son los días de 2024; el test previsto comprende 01/01/2025–31/08/2026. Cada entrada termina antes del día de resultado y cada etiqueta usada para ajustar debe estar completamente observada al emitir el pronóstico evaluado.

Los orígenes de validación se materializaron con `split_train_val_forwardChaining` y los controles descritos en la [sección 2](02_eda.md). Son calendarios e índices de preparación, no corridas de entrenamiento. Su existencia no fija si las redes se reajustarán diariamente, mensualmente o con otra cadencia; esa política se declarará en E2 antes de ejecutarla.

La historia observable anterior a validación o test puede formar parte de sus entradas. Incorporar una etiqueta pasada de 2024, cuando ya se conoce, es diferente de consultar la etiqueta futura que se evalúa. El protocolo posterior deberá explicitar el embargo indicado en la ruta 6.4 y verificar por separado los intervalos de disponibilidad de entradas y etiquetas; un máximo rezago, por sí solo, no acredita todos esos controles.

### 6.1.2. Ventanas y preprocesamiento causal

La adquisición es de un minuto y el modelado es **diario UTC**. La entrada conserva cinco variables OHLCV de $D=7,14,21,28$ días. El objetivo principal es la desviación poblacional de **$n=7$ retornos diarios**, pronosticada un día adelante. Las volatilidades de 14/21/28 retornos constituyen sensibilidad del objetivo; sus RMSE no se ordenan para elegir una supuesta mejor definición.

La ruta de regresión temporal requiere estudiar frecuencia, desfases y disponibilidad de covariables, su relación con el objetivo, cambios de régimen y autocorrelación de los errores de una referencia. Los diagnósticos de ACF/PACF y estacionariedad ya presentes en el EDA son información temporal complementaria; no constituyen por sí solos una selección de ventana o un resultado de modelos. La correspondencia de los requisitos específicos y la evidencia debe mantenerse explícita en el capítulo 2.

Las transformaciones aprendidas se ajustarán con entrenamiento permitido de cada fold. La validación externa no servirá para detener el ajuste sobre su propia etiqueta futura. Las medias y componentes de PCA del EDA no se transferirán automáticamente a un modelo sin volver a respetar el corte temporal.

La política vigente preserva los ZIP y el calendario completo: un día sin 1.440 minutos elegibles queda ausente, no se calculan retornos a través de huecos y se excluyen ventanas afectadas. Los 80 minutos ausentes y la fila de cierre no estándar del 24/03/2023 permanecen documentados. En esta revisión no se introduce una recuperación de otra granularidad ni una imputación nueva.

### 6.1.3. Familias de referencia y selección documental

La guía propone para regresión temporal MLP sobre covariables rezagadas, LSTM o GRU sobre secuencias de covariables y CNN1D poco profunda. Estas familias son orientativas: la selección documental debe identificar el papel de comparación que cada estudio les asigna, su frecuencia, arquitectura reproducible y diversidad de sesgos inductivos.

El registro de evidencia, las tres fichas y el estado de la selección se mantienen en la [sección 3](03_estado_arte.md). Una referencia general de arquitectura no se cuenta como prueba de uso frecuente en volatilidad de Bitcoin. Tampoco se equipara automáticamente una TCN con una CNN1D simple o una red lineal con una MLP de capas ocultas.

Sea cual sea el trío respaldado, el diseño propio conservará las mismas covariables, etiquetas y orígenes entre modelos. Aplanar una ventana para una MLP o conservar su eje temporal para otra red no permite cambiar el conjunto de información disponible.

### 6.1.4. Pérdida y estrategia de pronóstico

Se fijan **MSE como pérdida del diseño y RMSE como métrica primaria**. Su ejecución se reserva a E2. Cualquier cambio posterior de pérdida o criterio de selección quedará registrado antes del uso del test, con presupuesto comparable entre modelos.

La salida principal es una volatilidad escalar no negativa de **un día adelante**. Se emite al inicio del día de resultado, con OHLCV hasta el día previo, y se observa al cierre. Es el siguiente valor de una medida histórica móvil; sus ventanas comparten retornos y no representan siete días enteramente futuros.

El ejemplo sintético 4/3/2 de `timeseries-cv` ilustra una función de partición y no cambia el horizonte de Bitcoin. Cualquier ampliación de horizontes requerirá definir entradas, salidas y estrategia directa, recursiva o multi-salida como otro alcance.

### 6.1.5. Métricas y referencias de contexto

Se reportarán RMSE, MAE, MSE, R², MAPE y MASE sobre predicciones agrupadas por ejecución, con escala y cobertura explícitas. MAPE usa solo etiquetas positivas, sin epsilon e informando los ceros; MASE usa diferencias de etiquetas separadas por un día de calendario y completamente conocidas en el entrenamiento permitido. Escalas nulas o sin pares suficientes producen una métrica no definida, no un denominador arbitrario.

El margen aceptado es **5 % de reducción relativa de RMSE frente a persistencia**, con el mismo objetivo y fechas. Se registró como criterio operativo del estudio y no como utilidad financiera. La relevancia y el contraste estadístico se interpretarán por separado en E2.

Persistencia no ocupa uno de los tres puestos DL. Para $n=7$ resume ocho cierres, mientras una ventana de siete días OHLCV no incluye explícitamente esa misma estadística: la comparación contextual debe conservar esa diferencia de información. La ruta de regresión temporal también propone una referencia lineal regularizada con las mismas covariables; su ajuste y resultados pertenecen a la ejecución posterior. Un ingenuo estacional necesita justificar y declarar su periodo antes de evaluarlo.

Con un origen de una sola salida, promediar su RMSE entre días equivale a promediar errores absolutos; no produce la RMSE global. R² por origen unitario no está definido. El reporte de 2025 y 2026 será separado y agregado, aclarando que 2026 termina el 31 de agosto.

### 6.1.6. Inferencia y diagnóstico previstos

| Herramienta | Pregunta y unidad pertinente |
|---|---|
| Diebold–Mariano | Diferencia de precisión mediante pérdidas por las mismas fechas, con varianza de largo plazo y corrección de muestra pequeña justificadas |
| Ljung–Box | Autocorrelación de residuos en tramos con tamaño suficiente y rezagos documentados |
| t pareado o Wilcoxon | Diferencias entre ejecuciones bajo emparejamiento y supuestos declarados |
| Friedman y post-hoc | Comparación conjunta con bloques defendibles, multiplicidad y limitaciones de potencia |
| BDS, complemento del material específico | Hipótesis i.i.d. en residuos del primer horizonte, sin interpretar rechazo como prueba exclusiva de no linealidad |

La guía de regresión temporal y el material específico recibido motivan estos procedimientos. El [capítulo 5](05_benchmark.md) conserva su planteamiento; la ejecución y sus resultados se reservan a E2. Un diagnóstico sobre retornos del EDA no sustituye una prueba sobre errores predictivos.

Las etiquetas móviles se solapan aunque el horizonte sea un día. La unidad observacional es el origen temporal; la inferencia deberá conservar dependencia mediante bloques u otro diseño justificado. La longitud de bloque y su sensibilidad se fijarán antes de contrastar, sin importar decisiones de corridas ajenas a esta revisión. Semillas, fechas y folds no se suman como observaciones independientes del mismo tipo.

## 6.2. Correspondencia entre procedimiento y evidencia

| Componente | Evidencia o decisión en E1 documental | Ejecución posterior |
|---|---|---|
| Fuente, frecuencia y reloj | Secciones 1.2–1.5 y 2.3 | Conservar procedencia y comprobar disponibilidad por muestra |
| Calidad y faltantes | EDA 2023–2024, exclusión estricta y calendario conservado | Mantener reglas al formar muestras y registrar enmiendas |
| Objetivo y ventanas | $n=7$, $h=1$, cuatro historias D; otras n como sensibilidad | Sin seleccionar objetivos mediante RMSE de test |
| Partición y origen móvil | Calendario y manifiestos existentes de `timeseries-cv` | Cerrar muestras definitivas, embargo y cadencia de reajuste |
| Arquitecturas | Evidencia, selección y fichas documentales en §3 | Implementar configuraciones justificadas en E2 |
| Pérdida y métricas | MSE/RMSE, MAPE y MASE definidas en §1.4 | Calcular predicciones y métricas sin consultar test para seleccionar |
| Margen de relevancia | Regla relativa del 5 % frente a persistencia | Separar relevancia, incertidumbre y significación |
| Inferencia | Unidades, supuestos y riesgo de dependencia descritos | Dimensionar, ejecutar y reportar IC/efectos/contrastes en E2 |

## 6.3. Condiciones para la ejecución posterior

Antes de E2 deben revisarse el cierre bibliográfico y los detalles arquitectónicos que permanezcan sin reportar; las muestras finales, transformaciones, embargo y cadencia de actualización; la familia de comparaciones y el dimensionamiento del piloto. Las decisiones ya aceptadas —objetivo de siete retornos, horizonte de un día, RMSE y margen del 5 %— no figuran como elecciones nuevamente pendientes.

Los resultados de literatura se identifican como publicados por otros autores. Este capítulo no aporta entrenamientos, tiempos o métricas propios. La documentación del procedimiento permite revisarlo antes de la fase experimental y conserva la diferencia entre el alcance pedido por el usuario y la distribución literal de requisitos del PDF.
