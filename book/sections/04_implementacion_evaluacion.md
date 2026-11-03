# 4. Implementación y evaluación: diseño y estado de ejecución

Este capítulo contrasta el diseño con los apartados 4.1–4.4 de la guía. **La selección definitiva de tres arquitecturas todavía está abierta**: LSTM, TCN y encoder Transformer tienen fichas candidatas, y GRU continúa en evaluación según §3.2. No se asignan esos nombres a tres puestos experimentales ya congelados.

La guía exige implementación, piloto y evaluación dentro de E1. El alcance de trabajo acordado reserva la ejecución de las tres redes y el test para E2; esta revisión corrige su documentación y mantiene visible lo que falta. Las cuatro Ridge del bloque C son diagnósticos de EDA, no los tres modelos DL ni un piloto de este capítulo.

La revisión punto por punto está en el {download}`informe de implementación y evaluación <../_static/e1/revision_implementacion_evaluacion_20261103.md>` y su [matriz de requisitos](../_static/e1/revision_implementacion_evaluacion_20261103.json). La [especificación documental actual](../_static/experiments/protocolo_e2_documental_20261103.json) reúne decisiones aceptadas, campos abiertos y productos no ejecutados; no es una configuración experimental congelada.

## 4.1. Diseño Experimental

### 4.1.1. Variables de entrada y representación

La unidad de modelado es un **día UTC de BTCUSDT**, construido con las velas oficiales de un minuto. Las covariables de partida son OHLCV diario; el cierre es la variable central para calcular retornos y volatilidad. No se incorporan noticias sin una fuente y disponibilidad temporal verificadas.

Para una historia de $D\in\{7,14,21,28\}$ días, la representación inicial es $X\in\mathbb R^{N\times D\times5}$. Una MLP puede utilizar una vista aplanada de $5D$ componentes; una red secuencial conserva los $D$ pasos. Cambiar la disposición del tensor no permite dar a un modelo información adicional. La representación y cualquier selección de atributos se fijarán antes de comparar los modelos.

Las ventanas no se forman atravesando días incompletos. Los ZIP originales permanecen intactos y las exclusiones se reproducen mediante el código y los manifiestos de las secciones 1 y 2.

### 4.1.2. Variable objetivo y horizonte

La definición principal fijada es volatilidad histórica poblacional de **siete retornos diarios** ($n=7$), sin anualizar. Las ventanas $n=14,21,28$ se conservan como sensibilidad de objetivos distintos. La longitud de entrada sigue siendo $D=7,14,21,28$, sin elegir una D principal.

Se mantiene el horizonte principal de **un día adelante** y la convención de §1.1.1: el pronóstico se emite al inicio del día $t$, usando información hasta $t-1$, para una volatilidad que incorpora el retorno todavía desconocido del día $t$. La salida es escalar y no negativa. El usuario fijó n=7; esa decisión no procede de elegir un ganador por los gráficos ni autoriza horizontes adicionales.

Para evitar confundir el subíndice de la volatilidad con su fecha, cada muestra conservará **día de resultado, origen de pronóstico y disponibilidad de la etiqueta**. En notación de calendario, para el resultado del día $d$: $X_d=(z_{d-D},\ldots,z_{d-1})$, $z=(O,H,L,C,V)$ y $y_d=\sigma_7(d)$, conocida al terminar $d$. Corresponde a $Y_t=\sigma_{t+1}$ de §1.1.1 cuando $t$ designa ese día de resultado; no incorpora siete días enteramente futuros.

El diseño fija **MSE como pérdida** y **RMSE como métrica primaria**. La ruta 6.4 requiere RMSE, MAE y R²; el reporte conserva además MSE, MAPE y MASE como métricas complementarias aceptadas para este estudio. MAPE usa objetivos positivos con cobertura explícita; MASE emplea rezago de un día de calendario y escala ajustada solo con entrenamiento. Estas políticas ya están definidas sin consultar test.

### 4.1.3. Partición, proporciones y validación

El calendario experimental conserva entrenamiento inicial de 2023, validación creciente durante 2024 y test previsto de 2025–agosto de 2026. Se diferencian proporciones de **calendario** de proporciones de muestras supervisadas:

| Etapa | Días de calendario UTC | Proporción del calendario experimental | Estado de elegibilidad |
|---|---:|---:|---|
| Historia inicial de entrenamiento, 2023 | 365 | 27,26 % | 364 días OHLCV completos; muestras según ventana |
| Validación, 2024 | 366 | 27,33 % | 366 días completos y un fold diario por configuración |
| Test previsto, 01/01/2025–31/08/2026 | 608 | 45,41 % | Muestras supervisadas y congelación final pendientes |

Las proporciones suman el calendario de 1.339 días; no son un reparto aleatorio ni los pesos efectivos de entrenamiento. Los días de septiembre–octubre de 2026 descargados quedan fuera de ese calendario, sin asignación automática al test.

La [reserva de archivos brutos de test](../_static/data/test_raw_reserve_20250101_20260831.json) identifica los 20 ZIP mensuales del periodo mediante sus metadatos y hashes de adquisición. Se construyó sin abrir sus contenidos en esta revisión; los 875.520 registros son los reportados por el manifiesto de adquisición, no un nuevo análisis. Esta reserva de identidad y calendario no equivale a congelar las muestras supervisadas, máscaras o transformaciones de E2.

Se ejecutó `split_train_val_forwardChaining` sobre índices diarios. El manifiesto contiene **366 folds por configuración**, para 16 combinaciones de $D$ y $n$. Las etiquetas de entrenamiento se comprueban contra su disponibilidad en cada origen. Al avanzar la validación, pueden incorporarse etiquetas anteriores de 2024 ya observadas, según los límites que produce la función; nunca etiquetas futuras.

La ruta 6.4 exige un embargo igual al máximo rezago de las covariables. Con la indexación de §1.1.1, el rezago máximo de la entrada es $D$ días. Se fija la convención de **$D$ días de resultado omitidos** antes del origen $t$: se excluyen $t-D,\ldots,t-1$ y se admiten únicamente etiquetas de entrenamiento con día $u\le t-D-1$, además de sus controles de disponibilidad. El manifiesto se corrige con este filtro explícito tras la partición de la biblioteca. La separación anterior dejaba $D-1$ días omitidos; no satisfacía esta convención. La corrección modifica los índices de entrenamiento, no entrena modelos ni cambia el número de orígenes de validación.

El embargo por covariables no demuestra independencia de las etiquetas de volatilidad. Bajo esta convención, el soporte de retornos de la última etiqueta de entrenamiento y el de validación se solapan cuando $n>D+1$: el manifiesto registra esta situación en 2.196 filas de seis configuraciones. La ausencia de retornos compartidos tampoco demuestra independencia del mercado. Esta dependencia se conserva como condición del análisis posterior; no se resuelve contando filas o semillas como nuevas series independientes.

La cronología es determinista y no necesita semilla para barajar. Las semillas sí se registrarán para inicialización, optimización y otras operaciones estocásticas. La selección de hiperparámetros utilizará exclusivamente desarrollo y el mismo calendario, presupuesto y criterio entre modelos; no se ha ejecutado aún esa búsqueda. Si se informa desempeño de 2024 después de usarlo para seleccionar configuraciones, se identificará como desempeño de selección, no como estimación de test independiente.

Los orígenes de **2024 son validación de selección** si intervienen en elegir hiperparámetros o D. Una parada temprana con pasado permitido no elimina el optimismo de seleccionar con todo ese año. Para presentar un origen de 2024 como evaluación externa de un procedimiento de selección, toda la búsqueda y la parada tendrían que estar anidadas en su pasado. La estrategia definitiva —selección con 2024 y test reservado, o evaluación temporal anidada— y los cortes internos se registrarán antes del piloto. Las proporciones de calendario anteriores no fijan esos cortes ni obligan a reajustar diariamente.

### 4.1.4. Pipeline común y controles de fuga

1. **Construcción de datos:** comprobar hashes y disponibilidad; agregar días completos; formar entradas y etiquetas alineadas; aplicar las mismas exclusiones para los tres modelos dentro de cada configuración.
2. **Preprocesamiento:** conservar la exclusión estricta de días incompletos, sin imputar; ajustar escalado y selección de variables solo con el entrenamiento admitido de cada ajuste. Los parámetros exploratorios de PCA no se transfieren automáticamente al experimento.
3. **Entrenamiento:** inicialización nueva por repetición; arquitectura, optimizador, pérdida, batch size, épocas máximas y regla de parada registrados antes de comparar resultados. Los reajustes dentro de una repetición seguirán su política causal declarada, sin presumir reinicios diarios o entrenamiento incremental.
4. **Early stopping:** utilizar datos pasados de ajuste interno, sin consultar el objetivo futuro del fold evaluado. La etiqueta externa futura y los resultados de test no seleccionarán la regla de parada ni los hiperparámetros. Un eventual ajuste secuencial seguirá solo el pasado permitido y la política registrada de §4.3.7.
5. **Validación:** conservar predicciones, objetivos, fechas, configuración, semilla y duración por fold. Calcular las métricas con la agregación predefinida.
6. **Evaluación definitiva:** congelar selección y manifiestos antes de test; aplicar el procedimiento una vez fijado y reportar 2025 y 2026 por separado.

Los folds tienen una salida diaria. Por ello, para cada semilla y configuración se concatenarán las predicciones fuera de entrenamiento y se calculará:

$$
\operatorname{RMSE}_{m,s}=\sqrt{\frac1{N_{\mathrm{val}}}
\sum_{t\in\mathcal V}(Y_t-\widehat Y_{m,s,t})^2}.
$$

**No se promediarán las RMSE de folds unitarios como si fueran la RMSE global:** ese promedio equivale a MAE. El reporte por fold se conservará por trazabilidad, separado de la métrica agregada de cada ejecución.

### 4.1.5. Estudio piloto: al menos cinco ejecuciones por modelo

Antes del experimento definitivo se ejecutarán **al menos cinco semillas distintas de cada uno de los tres modelos**, sobre el mismo protocolo de validación. El borrador histórico enumera las semillas comunes `11, 23, 37, 53, 71`; esa lista no está congelada ni representa ejecuciones realizadas. La lista operativa y su alcance se registrarán al cerrar el protocolo de E2. Cada repetición parte de una inicialización nueva y produce la métrica primaria agregada de su configuración.

Para valores $m_1,\ldots,m_R$ de RMSE por ejecución se reportarán:

$$
\bar m=\frac1R\sum_{s=1}^R m_s,\qquad
s_m=\sqrt{\frac1{R-1}\sum_{s=1}^R(m_s-\bar m)^2},\qquad
\mathrm{CV}=\frac{s_m}{\bar m}.
$$

CV puede mostrarse como porcentaje multiplicando por 100; si la media es cero, no está definido. Se conservarán ejecuciones fallidas y su diagnóstico; no se descartarán semillas por obtener resultados desfavorables.

La selección de configuraciones y orígenes del piloto se cerrará antes de ejecutarlo. La tabla muestra una contabilidad algebraica para tres modelos y cinco semillas; **ninguna de estas cadencias está elegida o ejecutada**:

| Alcance hipotético | Configuraciones | Un ajuste fijo inicial por configuración y semilla | Reinicio y ajuste en cada uno de 366 orígenes |
|---|---:|---:|---:|
| Objetivo principal n=7, cuatro historias D | 4 | 60 | 21.960 |
| Principal y sensibilidad: cuatro n × cuatro D | 16 | 240 | 87.840 |

Se usa $3\times5\times C\times F$, con $F=1$ o $366$. No se incluyen búsqueda de hiperparámetros, posibles reajustes posteriores a la selección interna, fallas o repeticiones técnicas. Son escenarios de cantidad de ajustes, no tiempos, memoria o presupuesto adoptado. Las doce configuraciones con n≠7 responden a otros objetivos y no se promedian con la tarea principal.

Las semillas describen variabilidad del entrenamiento sobre un mercado observado. **Las 366 fechas y las 5.856 filas configuración–fold no son $R$ repeticiones independientes**, y distintas semillas tampoco constituyen mercados independientes.

La media de métricas entre semillas describe el entrenamiento aleatorio bajo ese calendario; no es la RMSE de un ensemble de sus predicciones. Antes del piloto se declarará si sus corridas se usan solo para calibración o si podrán integrarse al análisis definitivo, con justificación de cualquier reestimación del tamaño. No se reutilizarán automáticamente tras adaptar R ni se añadirán semillas hasta conseguir significación.

### 4.1.6. Margen práctico, contraste y unidad de repetición

El criterio aceptado es una **reducción relativa del 5 % de RMSE frente a persistencia**, sobre las mismas fechas. Para una configuración y calendario comunes, sean $b$ la RMSE de persistencia y $m_{A,s}$ la de una red en la semilla $s$:

$$
G_s=b-m_{A,s},\qquad Q_s=1-\frac{m_{A,s}}{b},\quad b>0.
$$

Una ganancia positiva favorece a la red. Si $b=0$, la reducción relativa es indefinida; no se introduce un denominador artificial. Para planificar en unidades absolutas se conserva la regla $\delta_{\mathrm{plan}}=0{,}05\,b_{\mathrm{desarrollo}}$, con fechas y cobertura fijadas antes del piloto. Esa referencia todavía no se ha medido para el piloto: la conversión y su valor numérico siguen pendientes de E2.

| Pregunta | Referencia nula | Cantidad necesaria para planificar |
|---|---|---|
| Detectar una diferencia de tamaño práctico frente a cero | Ausencia de diferencia | Efecto alternativo de magnitud $\delta_{AB}$, o $\delta_{\mathrm{plan}}$ frente a persistencia |
| Aportar evidencia de que se supera el umbral práctico | Ganancia relativa ≤0,05 | Ganancia alternativa mayor que 0,05 y su distancia a esa frontera |

**Detectar una mejora de tamaño 5 % frente a cero no demuestra que exceda el 5 %.** La conversión de desarrollo tampoco convierte la RMSE de persistencia en una constante poblacional conocida. El criterio relativo se volverá a expresar con la referencia del periodo realmente evaluado y la incertidumbre correspondiente.

Entre dos redes se utilizarán diferencias pareadas $d_s=m_{A,s}-m_{B,s}$ y un margen $\delta_{AB}$ específico, todavía por justificar. El 5 % frente a persistencia no se traslada automáticamente a esos pares, a equivalencia o a una propuesta avanzada frente al baseline DL. La dispersión de planificación es:

$$
\widehat\sigma_d=\sqrt{\frac1{R-1}\sum_{s=1}^R(d_s-\bar d)^2},
\qquad \text{efecto mínimo estandarizado}=\frac{\delta_{AB}}{\widehat\sigma_d}.
$$

No se sustituye por la desviación de una red aislada ni por la dispersión de días tratados como independientes. Cada contraste registrará escala, dirección, familia de hipótesis y unidad. Aquí $R$ cuenta **semillas bajo datos y procedimiento fijados**; no cuenta mercados, folds o configuraciones.

### 4.1.7. Dimensionamiento y alcance de la potencia

La guía requiere potencia $1-\beta\ge0{,}80$ y nivel $\alpha=0{,}05$. Para **detectar una diferencia respecto de cero**, la aproximación normal bilateral de la guía es:

$$
R_{\mathrm{semillas}}\approx
\left\lceil\frac{(z_{1-\alpha/2}+z_{1-\beta})^2\widehat\sigma_d^2}
{\delta_{AB}^2}\right\rceil.
$$

El factor de cuantiles es aproximadamente **7,85**, sin ajuste por multiplicidad. Esta fórmula dimensiona variación del entrenamiento condicionada al calendario y a la configuración; no dimensiona fechas de test ni acredita potencia temporal sobre nuevos periodos. Con pocas semillas, estimar la dispersión tiene incertidumbre considerable: se justificará la aproximación y el tamaño definitivo, mediante un método adecuado al diseño cuando sea necesario. No rechazar normalidad con un piloto pequeño no demuestra ese supuesto.

Si la pregunta es **superar el umbral práctico**, se puede planificar directamente con $Q_s$. Para $H_0:\mu_Q\le0{,}05$ frente a $H_1:\mu_Q>0{,}05$, hace falta una alternativa $q_{\mathrm{alt}}>0{,}05$ fijada antes del dimensionamiento:

$$
R_{\mathrm{semillas}}\approx
\left\lceil\frac{(z_{1-\alpha_*}+z_{1-\beta})^2\widehat\sigma_Q^2}
{(q_{\mathrm{alt}}-0{,}05)^2}\right\rceil.
$$

Esta es una aproximación **unilateral contra una frontera distinta de cero**, condicionada al mismo calendario. La ganancia alternativa, dispersión y nivel de planificación están pendientes; no se inventa una mejora esperada para obtener un número de semillas. En escala absoluta la distancia equivalente sería $g_{\mathrm{alt}}-0{,}05b_{\mathrm{desarrollo}}$, con su dispersión en las mismas unidades.

El nivel $\alpha_*$ deberá derivarse del plan de multiplicidad. No se asigna automáticamente 0,05 a cada contraste ni se interpreta Holm como un único nivel fijo por comparación. Se distinguirá potencia marginal del contraste principal de potencia conjunta de toda una familia; obtener 0,80 en cada cálculo marginal no demuestra 0,80 conjunta. La familia debe contemplar las afirmaciones frente a persistencia, entre redes y entre configuraciones que finalmente se declaren confirmatorias.

El tamaño definitivo respetará el mayor requerimiento de los contrastes registrados y los mínimos aplicables: **cinco ejecuciones por modelo y seis si se usan pruebas por rangos**. Seis tampoco garantiza significación tras correcciones ni potencia suficiente, como explica §7.1. Si las diferencias no satisfacen los supuestos, se justificará la alternativa sin escogerla por los resultados de test.

Una dispersión estimada como cero exige revisar redondeo, determinismo o un piloto insuficiente; no prueba que sobren ejecuciones. **No existe todavía un R calculado**, ni un estudio de potencia ejecutado. La inferencia temporal de una ganancia relativa deberá conservar el emparejamiento y recalcular conjuntamente ambas RMSE, incluido el denominador de persistencia, al aplicar el método temporal justificado.

### 4.1.8. Diferencia mínima detectable

Para las ejecuciones realmente disponibles, la aproximación bilateral es:

$$
\mathrm{MDE}_0(R)\approx
\frac{(z_{1-\alpha/2}+z_{1-\beta})\widehat\sigma_d}{\sqrt R}.
$$

El subíndice cero explicita la referencia: es una diferencia detectable **respecto de cero**, en unidades del contraste. Si supera su margen $\delta_{AB}$, se ampliará el diseño o se declarará la limitación. Para demostrar superación de un umbral se necesita, en cambio, distancia detectable **respecto de ese umbral**, conforme a §4.1.7; no basta comparar MDE₀ con 5 % y afirmar que se ha demostrado una mejora superior al criterio.

Se reportarán contraste, configuración, R, dispersión, supuestos y multiplicidad. El estimador piloto será un punto de partida para E2, no un valor definitivo. No se utilizará potencia retrospectiva calculada con el efecto observado como certificación de suficiencia. R y MDE numéricos permanecen pendientes de ejecución.

### 4.1.9. Preregistro fechado y trazabilidad

La [especificación documental del 03/11/2026](../_static/experiments/protocolo_e2_documental_20261103.json) es la referencia actual de preparación: conserva n=7, h=1, las cuatro historias, métricas, exclusión estricta y hash del manifiesto corregido. Distingue tres identidades finales sin cerrar, candidatas, decisiones de ejecución pendientes y cantidades de potencia aún desconocidas. **No es ejecutable ni está congelada**: faltan decisiones metodológicas y configuraciones de entrenamiento.

Se conserva como antecedente para E2 el [borrador de preregistro](../_static/experiments/preregistro_piloto.json), conservado en `experiments/preregistro_piloto.json`, fechado el **2 de noviembre de 2026**, con configuración conocida, semillas previstas y campos por cerrar. Su estado es **borrador no congelado**, no un preregistro experimental completo. La [plantilla CSV de resultados](../_static/experiments/piloto_resultados_plantilla.csv) contiene solo encabezados y no representa ejecuciones realizadas.

El [registro de decisiones documentales de E1](../_static/e1/decisiones.json) contiene el criterio vigente; el borrador histórico no lo reemplaza. Los campos de métrica, márgenes y manifiesto de ese borrador histórico preceden a las decisiones y correcciones actuales, incluido el embargo explícito; deberán sincronizarse antes del piloto y no se usarán como configuración operativa vigente. Antes del piloto de E2 se cerrará su protocolo experimental inicial. Después se incorporarán sus resultados y el dimensionamiento en una versión fechada y congelada **antes de la ejecución definitiva**, conservando el historial de cambios. El registro debe identificar los modelos, datos y hashes, código, folds, objetivos, comparaciones, transformaciones, reglas de parada, $\delta$, métrica, supuestos y número final de ejecuciones.

Este plan se documenta después de los periodos de mercado adquiridos y del EDA. Que preceda a las redes y al piloto no lo convierte en un registro anterior a la observación de todos los datos del estudio. Las fechas de cambios, su motivo y los resultados ya conocidos permanecerán identificados.

### 4.1.10. Estado del diseño y del piloto

| Producto | Estado |
|---|---|
| Datos y EDA de desarrollo | Ejecutados y documentados |
| Folds de validación de 2024 | Materializados con `timeseries-cv` y filtro explícito de embargo $D$; validaciones de disponibilidad conservadas |
| Selección bibliográfica de tres arquitecturas | Pendiente; tres fichas candidatas y GRU aún evaluable, según §3.2 |
| Métrica principal, margen y políticas de métricas | RMSE, mejora relativa del 5 %, MAPE sobre objetivos positivos y MASE con rezago de un día fijados |
| Configuraciones y protocolo final de contrastes | Pendientes de cierre en E2 |
| Preregistro | Especificación documental actual y antecedente histórico diferenciados; ambos sin congelación experimental |
| Cinco ejecuciones piloto por modelo | No ejecutadas |
| Media, desviación, CV y diferencias pareadas | Sin resultados todavía |
| $R$ requerido y MDE | Cálculo futuro de E2; margen relativo ya fijado |
| Entrenamiento definitivo y test | No ejecutados |

La presencia de esta sección no sustituye las implementaciones y evaluaciones reservadas para E2 en el alcance solicitado.

## 4.2. Implementación de los Tres Modelos Base

La implementación comenzará una vez se cierre la selección de **exactamente tres arquitecturas**, con configuración publicada o una variante mínima respaldada. **LSTM, TCN y encoder Transformer tienen fichas candidatas; GRU no está descartada y el trío final no está congelado.** Sus implementaciones y ejecuciones quedan para E2. Estos apartados definen condiciones verificables y no acreditan que ya exista código de las redes.

### 4.2.1. Arquitecturas y contrato común

Cada modelo tendrá un identificador, referencias y una descripción reproducible de capas, unidades, activaciones, conexiones y salida no negativa. La arquitectura se enlazará con las ecuaciones y el análisis de complejidad de §3.3. Si la literatura no ofrece detalles suficientes, se seguirá la regla de selección y ampliación de evidencia de §3.2.5; no se completarán por preferencia personal.

| Componente | Condición común para los tres modelos |
|---|---|
| Datos | Mismos derivados diarios de BTCUSDT y mismos hashes |
| Entradas | Mismas covariables disponibles y días de historia dentro de cada configuración |
| Representación | Tensor diario o vista aplanada según la arquitectura, sin añadir información a un modelo |
| Objetivo | Misma definición de volatilidad y horizonte de un día dentro de cada comparación |
| Partición | Mismos orígenes y exclusiones del manifiesto de forward chaining |
| Preprocesamiento | Misma política causal y ajuste limitado al entrenamiento de cada fold |
| Semillas | Mismo conjunto predefinido por fase; inicialización nueva por repetición |
| Selección | Misma métrica primaria, regla de agregación y esfuerzo de búsqueda |

Se conservarán identificadores de las muestras para comprobar la igualdad de los conjuntos evaluados. Un modelo que falle o pierda muestras no se comparará ocultando ese cambio de cobertura.

### 4.2.2. Hiperparámetros y configuración seleccionada

Para cada arquitectura se registrarán el espacio de búsqueda declarado antes de entrenar y el valor finalmente seleccionado. Las fichas documentales ya están en §3.3; la tabla siguiente corresponde a las configuraciones de implementación propias, todavía pendientes:

| Campo obligatorio | Espacio por declarar | Valor seleccionado |
|---|---|---|
| Learning rate inicial | Pendiente | No seleccionado |
| Planificador de learning rate y sus parámetros, o ausencia explícita | Pendiente | No seleccionado |
| Número de capas y unidades o filtros por capa | Según evidencia y reglas de §3.2 | No seleccionado |
| Activaciones y configuración de la salida | Según arquitectura y salida no negativa | No seleccionado |
| Batch size | Pendiente | No seleccionado |
| Optimizador y sus parámetros | Pendiente | No seleccionado |
| Dropout y ubicación, si aplica | Pendiente; declarar también si no se usa | No seleccionado |
| Decaimiento de pesos u otra regularización | Pendiente | No seleccionado |
| Función de pérdida | MSE fijada documentalmente en §4.1 | Implementación y escala de entrenamiento pendientes |
| Épocas máximas y criterio de parada | Pendiente | No seleccionado |

Los campos no se rellenarán con valores supuestamente “habituales” atribuidos a artículos que no los reportan. Los ganadores de S1 sí constan en sus Tablas 5–6 y el SELF-ATT de S2 tiene configuración descrita: recuperarlos no obliga a adoptarlos para OHLCV diario ni cierra el trío. La versión de `keras-tcn` y los detalles incompletos de S6 se mantienen diferenciados. Cualquier desviación respecto a una fuente se justificará y versionará antes del experimento pertinente.

Las transformaciones aprendidas de X o Y, si se adoptan, se ajustarán solo con el subentrenamiento correspondiente. Estandarizar Y de forma afín requiere invertir esa transformación antes de evaluar; aplicar logaritmos a Y cambia la pérdida que se optimiza en su escala original y no se introduce como si fuera la misma MSE ya acordada. El mecanismo de salida no negativa deberá declararse sin atribuir a los artículos una activación que no utilizan.

### 4.2.3. Estrategia de entrenamiento y early stopping

Cada repetición comenzará con pesos nuevos y su semilla registrada. Se fijarán el número máximo de épocas, orden de lotes, tamaño de lote, precisión numérica y fuentes estocásticas. Si la política elegida reinicia en cada origen, cada reajuste vuelve a inicializarse; un eventual warm start utilizará únicamente un checkpoint anterior permitido y una regla registrada. No se presupone ninguna de las dos políticas. El entrenamiento conservará la relación causal entre muestras y etiquetas aunque se utilicen lotes para optimizar.

El early stopping deberá registrar métrica monitoreada, dirección de mejora, `patience`, `min_delta`, frecuencia de evaluación y política de restauración de pesos. Su conjunto de validación será **interno y anterior al origen evaluado**, con separación suficiente para que sus etiquetas estén disponibles. El único objetivo futuro de cada fold externo no se utilizará para decidir cuándo detener su propio entrenamiento.

Para un origen de resultado t, el bloque interno termina como máximo en t−D−1. Si v es su primer día de validación interna, el subentrenamiento usa resultados u≤v−D−1, además de calidad y disponibilidad: se aplica el embargo D también al corte interno. El tamaño del bloque y la regla para situarlo siguen pendientes. **Escaladores, selección de atributos y transformaciones de Y se ajustan al subentrenamiento**, no a su unión con los días usados para decidir la parada. Se conservan identificadores y hashes de ambos subconjuntos.

Después de elegir una duración mediante esa parada, cualquier reajuste sobre más historia permitida tendrá una regla de épocas y preprocesamiento ya fijada; no podrá consultar la etiqueta externa. La validación de selección de 2024 comparará configuraciones bajo el protocolo de §4.1.3. El procedimiento de ajuste final se cerrará con desarrollo antes de test. Sus resultados no modificarán la arquitectura, los hiperparámetros ni la regla de parada. El uso de etiquetas pasadas ya disponibles dentro de una eventual actualización, incluida su parada registrada, deberá quedar resuelto en el protocolo de §4.3.7.

### 4.2.4. Semillas, reinicios y reproducibilidad

El antecedente histórico propone una lista de semillas, pero no existe lista operativa congelada ni corridas ejecutadas. La especificación actual mantiene abiertas las listas por fase y su posible reutilización. El conjunto definitivo se dimensionará con el piloto y se compartirá entre los tres modelos finalmente elegidos. Se registrarán semillas del generador, framework y operaciones estocásticas, junto con las limitaciones de determinismo de hardware y kernels.

Las transformaciones y el modelo se ajustarán en los orígenes previstos por la cadencia registrada, utilizando solo el entrenamiento permitido en cada ajuste. El calendario de folds no obliga por sí mismo a reentrenar todos los días. No se reutilizarán pesos de un origen posterior ni escaladores ajustados con toda la validación. Cualquier entrenamiento incremental o warm start requerirá una regla causal predefinida y comparable; no se presupone en este diseño.

Si se conserva un checkpoint entre orígenes, su identificador y el de su escalador permanecerán vinculados. Reajustar un escalador cambia la representación recibida por los pesos; la política deberá explicar su compatibilidad con el estado conservado, en lugar de mezclar automáticamente pesos antiguos con una escala nueva.

Las fallas, reintentos y corridas interrumpidas quedarán registrados. No se reemplazará una semilla por otra para mejorar el resultado. Si se repite una corrida por un problema técnico, se conservarán la causa y su relación con el intento original.

### 4.2.5. Registros de cada implementación y corrida

Una **corrida** identifica modelo, configuración, fase, repetición y versión de protocolo, y reúne su secuencia completa de pronósticos. Cada **ajuste** dentro de ella conserva además origen, partición interna, checkpoint y número de intento. Cinco semillas son cinco corridas por modelo y configuración; los posibles cientos de ajustes de una corrida no aumentan ese número de repeticiones.

| Registro | Contenido mínimo |
|---|---|
| Configuración | Modelo, referencias, arquitectura, hiperparámetros, fase, semilla y versión de código |
| Datos y partición | Hashes, configuración de entrada/volatilidad, fold y fechas elegibles |
| Curvas | Pérdida y métricas de entrenamiento y validación interna por época, con conjuntos identificados |
| Parada | Época de parada, mejor época, motivo y checkpoint restaurado |
| Parámetros | Total de parámetros entrenables y no entrenables; contraste con la expresión analítica de §3.3 |
| Predicciones | Fecha de emisión, día objetivo, valor real, predicción y horizonte |
| Costo | Duración de ajuste y de inferencia, unidad de medición, batch size y calentamiento declarado |
| Entorno | Sistema, versión de Python, framework, bibliotecas, CPU/GPU, memoria y precisión numérica |
| Incidencias | Fallas, reintentos, advertencias y muestras no evaluadas |

Las curvas gráficas se generarán a partir de logs reales, identificando modelo, configuración, fold y semilla. No se dibujan curvas ilustrativas como si fueran entrenamientos ejecutados. Para RMSE por repetición se mantiene la agregación de predicciones fuera de entrenamiento de §4.1.4.

### 4.2.6. Equidad en la búsqueda de hiperparámetros

Los tres modelos recibirán el **mismo método de búsqueda**, el **mismo presupuesto de evaluaciones** y espacios comparables en amplitud y esfuerzo. Comparable no significa obligar a arquitecturas diferentes a tener exactamente los mismos hiperparámetros: cada espacio deberá justificarse por su familia y complejidad.

Antes de buscar se declararán número de candidatos o trials, semillas de búsqueda, configuraciones y folds por evaluar, regla de selección, desempates y tratamiento de fallas. El presupuesto se contabilizará con una unidad explícita; no se comparará un trial que evalúa un fold con otro que recorre todos los folds como si costaran lo mismo.

Se registrarán todos los candidatos evaluados, sus métricas, costo y estado, no solo el ganador. Cualquier ampliación de presupuesto deberá documentarse y respetar la equidad entre modelos. Las ventanas de entrada y volatilidad también forman parte del alcance de selección y no se explorarán con un esfuerzo mayor para favorecer a una arquitectura.

El costo real puede variar entre modelos aun con el mismo número de evaluaciones; se reportará como parte del resultado. No se fijan tiempos de entrenamiento antes de medir las implementaciones en el hardware disponible.

### 4.2.7. Bibliotecas y versiones

Se podrá implementar desde cero o utilizar un framework justificado, conservando versión, procedencia y configuración. El cierre del trío y sus configuraciones sigue pendiente según §3; el framework y el entorno de entrenamiento se fijarán antes de ejecutar E2. Las fichas candidatas no acreditan una implementación validada.

Actualmente están verificados el entorno de EDA de `requirements-eda.lock.txt` y el entorno de partición de `requirements-cv.txt`. `timeseries-cv` genera particiones y **no entrena las redes**. Su dependencia de NumPy menor que 2 se mantiene aislada del entorno del EDA, según §2.4.1.

Antes del piloto se fijará el entorno de entrenamiento con versiones compatibles y se comprobará una ejecución funcional de cada modelo con la misma interfaz de datos. No se declarará reproducibilidad solo por disponer de una lista de dependencias.

### 4.2.8. Estado de implementación

| Producto | Modelo 1 | Modelo 2 | Modelo 3 |
|---|---|---|---|
| Identidad final | Pendiente | Pendiente | Pendiente |
| Configuración de implementación propia | Pendiente en E2 | Pendiente en E2 | Pendiente en E2 |
| Código y prueba funcional | Pendiente | Pendiente | Pendiente |
| Espacio de búsqueda y presupuesto comparable | Pendiente | Pendiente | Pendiente |
| Hiperparámetros elegidos y regla de parada | Pendiente | Pendiente | Pendiente |
| Curvas, parámetros, tiempos y entorno medidos | No ejecutado | No ejecutado | No ejecutado |
| Piloto y repeticiones definitivas | No ejecutado | No ejecutado | No ejecutado |

Los tres puestos experimentales permanecen sin asignación definitiva. Las fichas LSTM/TCN/encoder de §3.3 y la evaluación de GRU en §3.2 son antecedentes para cerrar esa decisión. Los datos y las particiones disponibles son prerrequisitos completados; no sustituyen la implementación ni la evaluación.

## 4.3. Requisitos Experimentales

Estos requisitos se aplicarán conjuntamente a los tres modelos, respetando el diseño temporal y el dimensionamiento de §4.1. La ejecución del EDA y la generación de folds no acreditan que el entrenamiento, los intervalos de confianza o la evaluación de test ya estén realizados.

### 4.3.1. Control de semillas y determinismo

Cada corrida conservará un identificador y las semillas de los generadores utilizados: Python, NumPy, framework de Deep Learning y operaciones del dispositivo que correspondan. Se registrarán también el orden de lotes, los workers de carga y cualquier otra fuente estocástica del pipeline.

Se declarará si se exige **determinismo estricto**, qué opciones del framework se activan y qué operaciones o condiciones de hardware pueden impedirlo. Fijar una sola semilla no demuestra reproducibilidad exacta. El framework y las opciones concretas se fijarán antes del piloto; no se atribuyen ahora configuraciones a un software de entrenamiento aún no elegido.

La lista operativa de semillas será común entre modelos. Una misma etiqueta de semilla permite registrar repeticiones y controles compartidos, pero no produce los mismos pesos o trayectorias aleatorias en arquitecturas distintas ni garantiza reducción de varianza. Cada repetición parte de una inicialización nueva; una repetición idéntica usada para comprobar determinismo no cuenta como otra semilla independiente.

### 4.3.2. Repetición de experimentos y tamaño definitivo

Cada modelo tendrá al menos **cinco ejecuciones independientes y documentadas**, o **seis si se usan pruebas por rangos**, y el número aumentará cuando lo exija el análisis de potencia. El tamaño definitivo no se fija por conveniencia ni por haber completado el mínimo.

Cuando sea posible se combinarán semillas distintas con orígenes de validación distintos, manteniendo el orden temporal. No se introducirán particiones aleatorias de la serie para aumentar artificialmente el número de resultados. Todos los modelos deberán compartir las fechas y configuraciones comparadas.

Se distinguirán explícitamente:

| Unidad | Qué representa | Qué no representa |
|---|---|---|
| Semilla de una ejecución | Variabilidad de inicialización y entrenamiento bajo el protocolo fijado | Un mercado nuevo o una nueva realización independiente de la serie |
| Fold diario | Un origen temporal de validación | Una repetición independiente para dimensionar el número de semillas |
| Configuración | Una combinación de entrada, objetivo e hiperparámetros | Otra réplica del mismo experimento |

Los 366 folds y las 5.856 filas del manifiesto no se contarán como ejecuciones independientes. Si se presentan resultados con varios niveles —semillas, configuraciones y fechas—, se declarará su estructura y no se agruparán todas las filas como observaciones i.i.d.

### 4.3.3. Media, desviación estándar e intervalos de confianza

Por modelo y configuración se reportarán número de ejecuciones válidas, media, **desviación estándar muestral** e **intervalo de confianza del 95 %** de las métricas pertinentes. La métrica de una ejecución se calcula sobre sus predicciones agregadas según §4.1.4, no promediando incorrectamente las RMSE de folds unitarios.

Cada intervalo indicará método, unidad de remuestreo o repetición, tamaño muestral, nivel y supuestos. Para un intervalo de la media entre semillas puede emplearse una aproximación t de Student si sus condiciones se justifican, o un procedimiento de remuestreo adecuado. Un intervalo entre semillas sobre el mismo test mide incertidumbre del entrenamiento condicionada a ese conjunto; no cuantifica por sí solo toda la incertidumbre temporal del mercado.

Para incertidumbre asociada a fechas dependientes se justificará un bootstrap por bloques u otro método temporal, con longitud de bloque y sensibilidad documentadas. Los métodos exploratorios de bloques usados en el EDA no se convierten automáticamente en el método definitivo de inferencia del benchmark.

Las comparaciones entre modelos usarán diferencias pareadas en las mismas fechas y semillas, con los controles de multiplicidad preregistrados. Un p-value no sustituye el intervalo de confianza, el tamaño del efecto ni su relevancia frente a $\delta$.

### 4.3.4. Uso de GPU y registro del entorno

Si se utiliza GPU, se documentarán modelo, memoria, driver, runtime y versiones relevantes del framework, precisión numérica y opciones de determinismo. También se registrarán CPU, memoria, número de hilos y sistema operativo, aunque no se use acelerador.

Los tiempos de entrenamiento e inferencia se medirán con unidades y procedimiento claros; en GPU se considerará la ejecución asíncrona para no reportar solo el tiempo de lanzamiento de operaciones. Se declararán calentamiento, tamaño de lote y si se incluye preprocesamiento.

Los modelos se compararán bajo condiciones de hardware consistentes o se explicarán las diferencias. **No se han ejecutado entrenamientos de las redes ni mediciones de GPU**, y no se asignan cifras de tiempo o memoria no observadas.

### 4.3.5. Empaquetado reproducible de extremo a extremo

El paquete final deberá permitir reconstruir datos derivados, particiones, entrenamiento y evaluación a partir de instrucciones y versiones fijadas. Conservará:

1. Dependencias y versiones del entorno de entrenamiento, además de los entornos ya registrados para EDA y particiones.
2. Scripts o cuadernos ejecutables con directorios de trabajo y parámetros completos.
3. Manifiestos de datos, muestras, folds y hashes, junto con reglas de exclusión.
4. Configuración de cada corrida, semillas, arquitectura, hiperparámetros, código y hardware.
5. Logs, curvas, checkpoints, predicciones, métricas, incidencias y vínculo entre sus identificadores.
6. Preregistro fechado, historial de cambios y procedimiento de generación de tablas y figuras.

Actualmente se pueden reproducir la adquisición oficial, el EDA y los folds mediante los scripts del repositorio. **La reproducción de extremo a extremo del benchmark todavía está pendiente**, porque no existen implementaciones entrenadas de los tres modelos. Los ZIP brutos no se incluyen en el sitio: sus URLs y hashes permiten adquirirlos y verificarlos respetando las condiciones del proveedor.

### 4.3.6. Prevención de data leakage

| Control obligatorio | Aplicación en BTCUSDT | Estado o evidencia |
|---|---|---|
| Separación de entrenamiento, validación y test | Fechas de desarrollo y test identificadas; ninguna etiqueta futura puede entrar en el entrenamiento del origen evaluado | Folds de desarrollo materializados; test supervisado final pendiente de congelación |
| Transformaciones solo en train | Escalado y selección se ajustan dentro del entrenamiento permitido; faltantes por exclusión estricta, sin imputación | Regla documentada; integración con modelos pendiente |
| Orden temporal | No se barajan periodos; se comprueba disponibilidad de etiquetas y continuidad de las ventanas | Controles ejecutados en el manifiesto de forward chaining |
| Embargo de ruta 6.4 | Se omiten $D$ días de resultado antes de $t$; último día permitido $t-D-1$ | Filtro explícito del manifiesto corregido; separado del soporte de las etiquetas móviles |
| Mismo protocolo entre modelos | Mismas muestras, horizonte, covariables disponibles, semillas y criterio de selección | Requisito fijado; comparación todavía no ejecutada |
| Early stopping sin consultar la salida futura evaluada | Validación interna pasada, distinta del día externo que se pretende pronosticar | Regla documentada; implementación pendiente |
| Test reservado | No participa en EDA de desarrollo, piloto ni hiperparámetros | Scripts de EDA y partición no abren archivos de test |
| Integridad y trazabilidad | Brutos intactos, hashes y exclusiones documentadas | Adquisición y derivados de desarrollo registrados |

La separación se refiere al papel de cada objetivo y a su disponibilidad temporal. Una entrada de validación puede reutilizar historia pasada ya observable, como requiere forward chaining; esto no autoriza compartir etiquetas futuras ni ajustar transformaciones con validación o test.

Los controles espaciales, semánticos de NLP o de agrupamiento de imágenes de la guía general no corresponden al alcance actual de una serie financiera única sin esas modalidades. Si se incorporan noticias u otros datos agrupados, deberán añadirse sus controles de disponibilidad, duplicados y dependencia antes de utilizarlos.

### 4.3.7. Evaluación única del test por corrida

El protocolo de prueba realizará **una sola pasada cronológica de evaluación por corrida**, con arquitectura, hiperparámetros, reglas de actualización y decisiones ya fijados. El preregistro identificará cuándo se considera cerrada una corrida y qué artefactos se conservan. Si el procedimiento elegido actualiza pesos durante test, cada pronóstico se emitirá antes de observar su etiqueta y cada ajuste usará únicamente pasado disponible con su embargo. Esta evaluación secuencial debe declararse previamente y se distingue de evaluar pesos estáticos; ninguna de las dos políticas está elegida aquí.

Las predicciones guardadas permitirán calcular métricas, residuos y figuras sin volver a consultar test para elegir otra configuración. No se cambiará la regla de reentrenamiento ni se seleccionará una semilla según su desempeño en test. Si ocurre una falla técnica, se registrará el incidente y el tratamiento previsto; no se ocultará una evaluación desfavorable como si nunca hubiera ocurrido.

Se reportarán 2025 y 2026 por separado dentro del calendario fijado. Disponer de más archivos descargados no amplía automáticamente el conjunto de prueba ni valida una nueva selección después de observar resultados.

### 4.3.8. Estado de cumplimiento

| Requisito | Estado actual |
|---|---|
| Hashes, datos derivados y exclusiones de desarrollo | Registrados |
| Folds temporales, embargo y comprobación de disponibilidad | Índices corregidos con $D$ días omitidos y controles registrados; sin entrenamientos |
| Dependencias y scripts de EDA/partición | Disponibles y verificados |
| Configuración de semillas y determinismo del entrenamiento | Prevista; framework y opciones pendientes |
| Repeticiones, media, desviación e intervalos de modelos | No ejecutados |
| GPU, tiempos y hardware de entrenamiento medidos | No ejecutados |
| Paquete de entrenamiento y evaluación de extremo a extremo | Pendiente |
| Reserva de identidad bruta de test | Manifiesto de 20 ZIP con hashes de adquisición, construido sin abrir sus contenidos en esta revisión |
| Test supervisado congelado y evaluación final | Pendientes |

Esta lista distingue requisitos establecidos de controles ejecutados. No se presenta el benchmark como reproducible o completo hasta verificar también las implementaciones y sus resultados.

## 4.4. Evaluación

El proyecto corresponde a **regresión temporal con objetivo continuo, ruta 6.4 de la guía**, porque las entradas OHLCV no incluyen rezagos explícitos del objetivo. Se aplican las métricas de regresión y los diagnósticos temporales pertinentes. Las métricas de clasificación, segmentación, texto o localización espacial no se incorporan a este problema de volatilidad de Bitcoin.

### 4.4.1. Métrica principal y métricas secundarias

Se fija **RMSE como métrica principal**, coherente con la pérdida MSE prevista, y se reportarán **MAE, MSE, R², MAPE y MASE**, atendiendo tanto a las indicaciones específicas recibidas como a la guía general. sMAPE es una alternativa complementaria si se registra antes de evaluar; no se elegirá después para favorecer un resultado.

Para objetivos $Y_t$, predicciones $\widehat Y_t$ y $N$ pares elegibles del mismo modelo, configuración, semilla y horizonte, se define $e_t=Y_t-\widehat Y_t$:

$$
\mathrm{MSE}=\frac1N\sum_t e_t^2,\qquad
\mathrm{RMSE}=\sqrt{\frac1N\sum_t e_t^2},\qquad
\mathrm{MAE}=\frac1N\sum_t|e_t|.
$$

$$
R^2=1-\frac{\sum_t e_t^2}{\sum_t(Y_t-\bar Y)^2}.
$$

R² puede ser negativo. Si el objetivo evaluado es constante, su denominador es cero y se reportará como no definido, junto con el contexto; no se sustituirá silenciosamente por un valor favorable. Con un único resultado por fold, R² por fold no está definido: se calculará sobre la colección de predicciones fuera de entrenamiento, no promediando R² diarios.

La media de Y del periodo evaluado aparece solo en la **definición de R²**. No autoriza usarla para entrenar ni como pronóstico disponible en el origen. Un R² frente a media train, si se informa, tendrá nombre distinto del R² convencional.

Las métricas se calcularán en la **escala original de la volatilidad**, después de invertir transformaciones de la etiqueta. Se declarará si se muestran unidades de retorno logarítmico o puntos porcentuales. Multiplicar la etiqueta por 100 multiplica RMSE y MAE por 100 y MSE por 10.000; los resultados de distintas escalas no se mezclarán.

### 4.4.2. Errores relativos y escalados

La política de MAPE ya está fijada: se utiliza exclusivamente el subconjunto de objetivos positivos, $\mathcal I_+=\{t:Y_t>0\}$, con $N_+=|\mathcal I_+|$:

$$
\mathrm{MAPE}_+=\frac{100}{N_+}\sum_{t\in\mathcal I_+}\left|\frac{e_t}{Y_t}\right|,\qquad N_+>0.
$$

Se informarán el número de ceros, los casos incluidos y la cobertura $N_+/N$. Si no hay objetivos positivos, MAPE se marca no definida. No se añadirán epsilons arbitrarios ni se ocultarán casos excluidos. Valores positivos muy pequeños pueden dominar MAPE y requieren interpretación conjunta con MAE y RMSE.

Si se adopta sMAPE, se registrará su convención exacta, incluido el factor de escala y el tratamiento del caso $Y_t=\widehat Y_t=0$. Que sea simétrica no elimina todos los problemas cerca de cero.

Para MASE se fija el rezago de referencia $s=1$ día de calendario y se calculará, en el entrenamiento permitido de cada fold:

$$
a_{\mathrm{train},s}
=\frac1{|\mathcal J_s|}\sum_{j\in\mathcal J_s}|Y_j-Y_{j-s}|,
\qquad |\mathcal J_s|>0.
$$

$\mathcal J_s$ contiene pares de etiquetas válidas separados por **$s$ días de calendario**, ambos pertenecientes a la historia de entrenamiento permitida. No se comprimen huecos. La escala se calcula en unidades originales del objetivo y es común a los métodos comparados en ese origen, con identificador de su conjunto de referencia.

Se define la máscara de escalas admisibles y su tamaño:

$$
\mathcal I_M=\{t\in\mathcal V:a_{\mathrm{train}(t),1}>0\text{ y escala definida}\},
\qquad N_M=|\mathcal I_M|,
$$

$$
\mathrm{MASE}=\frac1{N_M}\sum_{t\in\mathcal I_M}
\frac{|Y_t-\widehat Y_t|}{a_{\mathrm{train}(t),1}},\qquad N_M>0.
$$

Si no hay pares válidos o la escala es cero, se conserva el caso como no definido y su motivo. Se reportan $N_M$, el total esperado y cobertura $N_M/N$; si $N_M=0$, la métrica completa es indefinida. La máscara proviene del entrenamiento y es común entre métodos, sin excluir fechas por su error. Una predicción no finita sigue la política de incidentes, no se oculta como exclusión de MASE.

Si la escala cambia con el entrenamiento creciente, se promedian **errores escalados por origen**; no se divide la MAE global por una escala elegida después. MASE<1 compara esos errores con la variación ingenua estimada en entrenamiento, no demuestra por sí sola que el modelo supere a persistencia durante la evaluación. El rezago s=1 permanece fijado sin mirar test.

### 4.4.3. Evaluación por horizonte y referencias ingenuas

El horizonte principal confirmado es **un día**. Las métricas se reportarán para cada combinación de longitud de entrada $D$ y ventana de volatilidad $n$, manteniéndolas separadas porque definen configuraciones y objetivos distintos. Si se añaden horizontes, se registrarán y se informarán métricas por horizonte antes de cualquier promedio global.

La ruta 6.4 incluye como referencia de contexto una **regresión lineal regularizada con las mismas covariables rezagadas**. Su representación aplanada conservará los $5D$ valores disponibles para las redes y su ajuste respetará el mismo protocolo temporal; no ocupa un puesto del trío DL. La versión diagnóstica estática ya ejecutada se documenta en §2.2.3.2.3; su configuración no se transfiere automáticamente al benchmark.

Se conserva además **persistencia de la volatilidad histórica disponible** al pronosticar como referencia del margen operativo del 5 %. Se evaluará también un ingenuo estacional cuando se justifique un periodo $s$ con desarrollo y su información sea observable. Por ejemplo, el periodo semanal debe justificarse y fijarse; no se adopta automáticamente porque el EDA muestre un STL con periodo siete.

Las referencias se evaluarán en las mismas fechas, escala y etiquetas que los modelos. Si una referencia necesita más historia, se declarará la cobertura común y la exclusión correspondiente. Ningún método ingenuo ocupa uno de los tres puestos obligatorios de modelos de Deep Learning.

Se conservarán cobertura disponible por método e intersección elegible para la comparación pareada, con fechas y soportes históricos identificados. Una falla técnica de predicción no se convierte automáticamente en una nueva exclusión por calidad de datos ni se utiliza para eliminar días difíciles del conjunto común.

En los folds diarios actuales, RMSE se calcula agrupando las predicciones de una ejecución según §4.1.4. El promedio de errores RMSE de folds con un solo punto equivale a MAE y no se presentará como RMSE global.

### 4.4.4. Comparación justa entre los tres modelos

Antes de comparar se comprobará igualdad de datos, particiones, orígenes, etiquetas, preprocesamiento permitido, semillas y configuración de evaluación. Las arquitecturas pueden organizar la misma información de forma diferente, pero no recibir atributos o fechas adicionales sin declarar otro experimento.

Se mantendrán las mismas definiciones de métricas y reglas sobre ceros, inversas de transformaciones, valores no finitos y fallas. Una predicción no finita no se omitirá silenciosamente del promedio. Se conservarán el número previsto de casos, el evaluado y el motivo de cualquier diferencia de cobertura.

Se usará el mismo presupuesto de búsqueda documentado en §4.2.6. Las comparaciones no mezclarán el mejor resultado de una única semilla para un modelo con la media de todas las semillas de otro.

La ganancia relativa será $Q=1-\mathrm{RMSE}_{\mathrm{modelo}}/\mathrm{RMSE}_{\mathrm{persistencia}}$ sobre el mismo calendario, cuando el denominador sea positivo. Un valor puntual $Q\ge0{,}05$ alcanza el criterio puntual; sostener que se **supera** requiere el intervalo o contraste correspondiente respecto a 0,05, además del alcance de inferencia declarado en §4.1.7. Rechazar ausencia de mejora frente a cero no basta. Para incertidumbre temporal, el remuestreo pareado recalculará ambas RMSE y la razón sobre los mismos bloques; no tratará el denominador estimado como exacto.

### 4.4.5. Análisis de resultados y variabilidad

Se reportarán **media, desviación estándar muestral e intervalo de confianza** por modelo y configuración, con el número de ejecuciones y el método de construcción. Se distinguirán métricas de selección en validación de la evaluación final del test, y se conservarán las fechas de cada predicción.

La variabilidad por semilla se examinará dentro de cada modelo y configuración. Las diferencias entre modelos se examinarán sobre las mismas semillas y fechas. Se presentarán distribuciones de métricas por corrida y diferencias pareadas; si se realiza una descomposición formal de varianza, se declararán el modelo estadístico y sus supuestos, sin mezclar configuraciones o días dependientes como réplicas independientes.

Las arquitecturas son **niveles elegidos**, no una muestra aleatoria de todas las redes DL. La varianza de tres medias no estima una variabilidad universal entre arquitecturas. Una descomposición formal declarará efectos fijos o aleatorios, interacciones, emparejamiento y covarianza. Se conservarán métricas por corrida: promediar predicciones entre semillas define un método agregado adicional, que no sustituye la media y DE de las métricas individuales.

La variabilidad temporal se analizará por bloques o segmentos predefinidos y con métodos adecuados a su dependencia. Los intervalos entre semillas sobre el mismo mercado no se interpretarán como toda la incertidumbre de generalización. Las pruebas y tamaños de efecto se interpretarán junto con $\delta$, no solo por significación estadística.

### 4.4.6. Robustez, sobreajuste y segmentos relevantes

La robustez se estudiará con segmentos y, si se incorporan, perturbaciones controladas fijadas antes de la evaluación correspondiente. En desarrollo pueden definirse niveles de volatilidad con umbrales estimados solo en entrenamiento; esos umbrales se congelarán antes de trasladarlos a validación o test.

Las fechas de bloques y los umbrales se registrarán antes de examinar sus errores, indicando si los umbrales quedan fijos desde el entrenamiento inicial o se actualizan causalmente por origen. Los rangos del objetivo observado sirven para análisis **retrospectivo** de errores, no como covariables disponibles antes de conocer ese objetivo. Se reportarán cantidades y cobertura de cada segmento; los pequeños se identificarán con sus límites de precisión.

| Análisis | Evidencia que deberá conservarse | Cautela |
|---|---|---|
| Periodos y regímenes | Métricas por bloque temporal y cantidad de casos | No seleccionar retrospectivamente solo los periodos favorables |
| Magnitud de volatilidad | Errores por rango del objetivo con umbrales documentados | No ajustar umbrales con test |
| Sensibilidad a ruido o faltantes simulados | Tipo, intensidad, semilla y momento de la perturbación | No contaminar etiquetas ni cambiar inadvertidamente el problema |
| Tamaño de entrada | Resultados separados para 7/14/21/28 días | No confundir historia de entrada con ventana de volatilidad |
| Curvas y brecha train–validation | Pérdida y métricas por época y conjunto, con escalas comparables | Una brecha pequeña por sí sola no descarta underfitting |

Si se simula **ruido de observación**, se perturban solo covariables históricas disponibles y se conserva la etiqueta original. La ley, intensidad y semilla se registrarán con desarrollo. Deben mantenerse precios positivos, volumen no negativo, $H\ge\max(O,C)$, $L\le\min(O,C)$ y $H\ge L$; ruido independiente sin esos controles puede fabricar velas imposibles. No se elige aquí una intensidad o distribución.

Modificar la trayectoria que define Y y recalcular su volatilidad constituye otra simulación o tarea, que se reportaría separadamente. Los faltantes simulados siguen la exclusión estricta, sin imputar ni comprimir fechas, con máscara de perturbación común y ventanas perdidas documentadas. Una mejora aparente tras retirar días difíciles no acredita por sí sola robustez.

Las curvas de aprendizaje y las brechas entre entrenamiento y validación se usarán para diagnosticar overfitting o underfitting, considerando también el nivel de error y el comportamiento de la referencia ingenua. No se dibujarán curvas sintéticas como si fueran resultados observados. Las evaluaciones de robustez de test deberán estar predefinidas y reutilizar las predicciones guardadas cuando corresponda, sin reabrir selección de modelos.

Las curvas usadas para detener se identificarán como **validación interna de parada**, con conjuntos y escala comparables a su entrenamiento. La etiqueta externa no se consulta por época. Una curva de selección con 2024 se denomina de selección, no test independiente; una brecha pequeña tampoco descarta un error absoluto alto.

### 4.4.7. Análisis cualitativo de predicciones y residuos

Se generarán gráficas de valores reales frente a predicciones en el tiempo y diagramas real–predicho, identificando modelo, semilla, configuración, horizonte y periodo. Los casos difíciles y errores sistemáticos se seleccionarán con una regla trazable y se contextualizarán sin atribuir causas económicas no verificadas.

Los residuos conservarán fecha y signo. Con $e_t=Y_t-\widehat Y_t$, una media positiva indica subestimación de volatilidad. Se examinarán sesgo, distribución, relación con nivel pronosticado, estructura temporal y heterocedasticidad; se mostrarán ACF de residuos y de sus cuadrados cuando correspondan.

Para esta regresión temporal se documentarán **Ljung–Box** y el diagnóstico **BDS** previsto en el apartado específico de series de tiempo y regresión temporal de la guía, con rezagos, parámetros, tamaño muestral, supuestos y multiplicidad. El BDS requerido sobre residuos del primer horizonte se ejecutará cuando existan predicciones de las redes: una prueba sobre retornos crudos del EDA no lo sustituye. Un rechazo o no rechazo no demuestra por sí solo capacidad predictiva ni independencia absoluta.

Los huecos y el solapamiento de objetivos móviles se considerarán al elegir tramos y métodos. No se concatenarán fechas separadas como si fueran observaciones consecutivas ni se usarán residuos de entrenamiento para aparentar un diagnóstico fuera de muestra.

Los residuos se ordenarán por calendario **dentro de cada modelo, configuración y semilla**; no se concatenan semillas o tramos separados para crear una serie ficticia. σ7 consecutivas comparten seis retornos, por lo que h=1 no implica errores independientes. Se distinguirán ACF de errores y de errores al cuadrado, y no se restará automáticamente el número de pesos de la red a los grados de libertad de Ljung–Box. Antes de afirmaciones confirmatorias se registrarán tramos, rezagos, dimensiones/radios de BDS, software, supuestos y multiplicidad; un ajuste de p-values no repara una referencia estadística mal calibrada.

### 4.4.8. Productos y estado de ejecución

| Producto de evaluación | Estado |
|---|---|
| Métricas, escalas y agregación documentadas | Criterios y políticas de MAPE/MASE fijados; ejecución futura en E2 |
| Predicciones de los tres modelos por fecha y semilla | No generadas |
| Tablas por configuración, horizonte y referencia ingenua | Sin resultados todavía |
| Media, desviación, intervalos y comparación pareada | Pendientes de corridas |
| Robustez, curvas y análisis por segmentos | Pendientes de modelos entrenados |
| Predicciones frente a reales y diagnóstico de residuos de las tres redes | Pendientes; las referencias lineales del EDA están en §2.2.3.2.3 |
| Evaluación definitiva de test | No ejecutada |

Las tablas y gráficas de este apartado se incorporarán desde predicciones verificables, sin reutilizar estadísticas descriptivas del EDA como si fueran métricas de pronóstico.
