# 4. Implementación y Evaluación de los Tres Modelos Base

Los tres modelos extraídos de la literatura deberán implementarse, compararse bajo el **mismo entorno experimental** y evaluarse con un protocolo común. Esta sección documenta el diseño y los requisitos del piloto; no presenta entrenamientos ni resultados que aún no existen.

```{important}
Los datos y el EDA de desarrollo están disponibles y los folds de validación se materializaron con `timeseries-cv`. Todavía faltan la selección bibliográfica de los tres modelos, el margen práctico y el cierre de configuraciones. El piloto, el dimensionamiento y el benchmark no se han ejecutado.
```

## 4.1. Diseño Experimental

### 4.1.1. Variables de entrada y representación

La unidad de modelado es un **día UTC de BTCUSDT**, construido con las velas oficiales de un minuto. Las covariables de partida son OHLCV diario; el cierre es la variable central para calcular retornos y volatilidad. No se incorporan noticias sin una fuente y disponibilidad temporal verificadas.

Para una historia de $D\in\{7,14,21,28\}$ días, la representación inicial es $X\in\mathbb R^{N\times D\times5}$. Una MLP puede utilizar una vista aplanada de $5D$ componentes; una red secuencial conserva los $D$ pasos. Cambiar la disposición del tensor no permite dar a un modelo información adicional. La representación y cualquier selección de atributos se fijarán antes de comparar los modelos.

Las ventanas no se forman atravesando días incompletos. Los ZIP originales permanecen intactos y las exclusiones se reproducen mediante el código y los manifiestos de las secciones 1 y 2.

### 4.1.2. Variable objetivo y horizonte

Se exploran cuatro definiciones de volatilidad histórica poblacional de **retornos diarios**, con $n=7,14,21,28$ y divisor $n$, sin anualizar. Esta ventana es distinta de $D$, la longitud de entrada.

Se mantiene el horizonte principal de **un día adelante** y la convención de §1.3: el pronóstico se emite al inicio del día $t$, usando información hasta $t-1$, para una volatilidad que incorpora el retorno todavía desconocido del día $t$. La salida es escalar y no negativa. Los análisis de EDA no han seleccionado una definición ganadora ni autorizado horizontes adicionales.

La configuración inicial propone **MSE como pérdida** y **RMSE como métrica primaria**. El reporte incluirá MAPE, MAE, RMSE y MSE, con MASE adicional según la guía de series de tiempo. Antes del entrenamiento deben cerrarse la política de ceros de MAPE y el rezago del denominador MASE; no se usarán resultados de test para elegirlos.

### 4.1.3. Partición, proporciones y validación

El calendario experimental conserva entrenamiento inicial de 2023, validación creciente durante 2024 y test previsto de 2025–agosto de 2026. Se diferencian proporciones de **calendario** de proporciones de muestras supervisadas:

| Etapa | Días de calendario UTC | Proporción del calendario experimental | Estado de elegibilidad |
|---|---:|---:|---|
| Historia inicial de entrenamiento, 2023 | 365 | 27,26 % | 364 días OHLCV completos; muestras según ventana |
| Validación, 2024 | 366 | 27,33 % | 366 días completos y un fold diario por configuración |
| Test previsto, 01/01/2025–31/08/2026 | 608 | 45,41 % | Muestras supervisadas y congelación final pendientes |

Las proporciones suman el calendario de 1.339 días; no son un reparto aleatorio ni los pesos efectivos de entrenamiento. Los días de septiembre–octubre de 2026 descargados quedan fuera de ese calendario, sin asignación automática al test.

Se ejecutó `split_train_val_forwardChaining` sobre índices diarios. El manifiesto contiene **366 folds por configuración**, para 16 combinaciones de $D$ y $n$. Las etiquetas de entrenamiento se comprueban contra su disponibilidad en cada origen. Al avanzar la validación, pueden incorporarse etiquetas anteriores de 2024 ya observadas, según los límites que produce la función; nunca etiquetas futuras.

La cronología es determinista y no necesita semilla para barajar. Las semillas sí se registrarán para inicialización, optimización y otras operaciones estocásticas. La selección de hiperparámetros utilizará exclusivamente desarrollo y el mismo calendario, presupuesto y criterio entre modelos; no se ha ejecutado aún esa búsqueda. Si se informa desempeño de 2024 después de usarlo para seleccionar configuraciones, se identificará como desempeño de selección, no como estimación de test independiente.

### 4.1.4. Pipeline común y controles de fuga

1. **Construcción de datos:** comprobar hashes y disponibilidad; agregar días completos; formar entradas y etiquetas alineadas; aplicar las mismas exclusiones para los tres modelos dentro de cada configuración.
2. **Preprocesamiento:** ajustar escalado, imputación permitida y selección de variables solo con el entrenamiento admitido de cada fold. Los parámetros exploratorios de PCA no se transfieren automáticamente al experimento.
3. **Entrenamiento:** inicialización nueva por semilla; arquitectura, optimizador, pérdida, batch size, épocas máximas y regla de parada registrados antes de comparar resultados.
4. **Early stopping:** utilizar datos pasados de ajuste interno, sin consultar el objetivo futuro del fold evaluado. El test no puede intervenir en la parada ni en decisiones de entrenamiento.
5. **Validación:** conservar predicciones, objetivos, fechas, configuración, semilla y duración por fold. Calcular las métricas con la agregación predefinida.
6. **Evaluación definitiva:** congelar selección y manifiestos antes de test; aplicar el procedimiento una vez fijado y reportar 2025 y 2026 por separado.

Los folds tienen una salida diaria. Por ello, para cada semilla y configuración se concatenarán las predicciones fuera de entrenamiento y se calculará:

$$
\operatorname{RMSE}_{m,s}=\sqrt{\frac1{N_{\mathrm{val}}}
\sum_{t\in\mathcal V}(Y_t-\widehat Y_{m,s,t})^2}.
$$

**No se promediarán las RMSE de folds unitarios como si fueran la RMSE global:** ese promedio equivale a MAE. El reporte por fold se conservará por trazabilidad, separado de la métrica agregada de cada ejecución.

### 4.1.5. Estudio piloto: al menos cinco ejecuciones por modelo

Antes del experimento definitivo se ejecutarán **al menos cinco semillas distintas de cada uno de los tres modelos**, sobre el mismo protocolo de validación. Se prevén las semillas comunes `11, 23, 37, 53, 71`; no representan ejecuciones ya realizadas. Cada repetición parte de una inicialización nueva y produce la métrica primaria agregada de su configuración.

Para valores $m_1,\ldots,m_R$ de RMSE por ejecución se reportarán:

$$
\bar m=\frac1R\sum_{s=1}^R m_s,\qquad
s_m=\sqrt{\frac1{R-1}\sum_{s=1}^R(m_s-\bar m)^2},\qquad
\mathrm{CV}=\frac{s_m}{\bar m}.
$$

CV puede mostrarse como porcentaje multiplicando por 100; si la media es cero, no está definido. Se conservarán ejecuciones fallidas y su diagnóstico; no se descartarán semillas por obtener resultados desfavorables.

La selección de configuraciones y folds incluidos en el piloto se cerrará antes de ejecutarlo. Cubrir las 16 configuraciones con tres modelos, cinco semillas y 366 folds supondría **87.840 ajustes de modelo**; es un conteo de diseño, no trabajo ejecutado ni estimación de tiempo. El alcance y los recursos deben fijarse antes de lanzar esa carga, sin reducirlos retrospectivamente para favorecer un resultado.

Las semillas describen variabilidad del entrenamiento sobre un mercado observado. **Las 366 fechas y las 5.856 filas configuración–fold no son $R$ repeticiones independientes**, y distintas semillas tampoco constituyen mercados independientes.

### 4.1.6. Diferencia mínima de interés y diferencias pareadas

El margen práctico $\delta>0$ debe fijarse antes del piloto, sin elegirlo para acomodar sus resultados, con justificación en las unidades de la RMSE y coherencia con §1.4. Sigue pendiente; no se inventa un valor para producir un tamaño muestral.

Para un contraste predefinido entre modelos $A$ y $B$, sobre las mismas fechas y semillas:

$$
d_s=m_{A,s}-m_{B,s},\qquad
\widehat\sigma_d=
\sqrt{\frac1{R-1}\sum_{s=1}^R(d_s-\bar d)^2},\qquad
\text{efecto estandarizado}=\frac{\delta}{\widehat\sigma_d}.
$$

La dispersión relevante es la de las **diferencias pareadas**, no la desviación de un modelo aislado ni el error entre días tratado como independiente. Se fijarán los contrastes principales y el tratamiento de multiplicidad antes del dimensionamiento.

### 4.1.7. Dimensionamiento de ejecuciones y potencia

La guía requiere potencia $1-\beta\ge0{,}80$ y nivel $\alpha=0{,}05$. Bajo la aproximación normal bilateral para las diferencias pareadas:

$$
R_{\mathrm{req}}\approx
\left\lceil
\frac{\left(z_{1-\alpha/2}+z_{1-\beta}\right)^2\widehat\sigma_d^2}{\delta^2}
\right\rceil.
$$

El factor de cuantiles es aproximadamente **7,85** para esos valores. El número definitivo respetará el mínimo aplicable y el mayor requerimiento de los contrastes preespecificados. Si se utilizan pruebas por rangos, se atenderá además el mínimo de **seis ejecuciones** indicado en §4.3 de la guía; cinco es el mínimo del piloto, no un tamaño definitivo garantizado.

Se documentarán los supuestos y la incertidumbre de estimar dispersión con pocas repeticiones. La aproximación no resuelve por sí sola dependencia temporal o multiplicidad; si se planifican varias afirmaciones, se registrará el ajuste y se revisará la potencia correspondiente. Si la normalidad no es defendible, se justificará una alternativa o simulación de potencia que respete el diseño, sin escogerla según los resultados de test.

Una dispersión piloto estimada como cero requiere investigar redondeo, determinismo o piloto insuficiente; no justifica concluir que no hacen falta ejecuciones. **No hay un valor calculado de $R$ todavía**, porque faltan $\delta$ y las diferencias observadas del piloto.

### 4.1.8. Diferencia mínima detectable

Para las ejecuciones efectivamente disponibles se reportará la aproximación:

$$
\mathrm{MDE}(R)\approx
\frac{\left(z_{1-\alpha/2}+z_{1-\beta}\right)\widehat\sigma_d}{\sqrt R}.
$$

Si MDE es mayor que $\delta$, se ampliará el diseño o se declarará esa limitación explícitamente. No se afirmará potencia suficiente por haber completado cinco semillas. El valor $\widehat\sigma_d$ servirá como punto de partida para el dimensionamiento del Entregable 2, sin considerarlo definitivo.

### 4.1.9. Preregistro fechado y trazabilidad

Se creó el [borrador de preregistro](../_static/experiments/preregistro_piloto.json), conservado en `experiments/preregistro_piloto.json`, fechado el **2 de noviembre de 2026**, con configuración conocida, semillas previstas y campos por cerrar. Su estado es **borrador no congelado**, no un preregistro experimental completo. La [plantilla CSV de resultados](../_static/experiments/piloto_resultados_plantilla.csv) contiene solo encabezados y no representa ejecuciones realizadas.

Antes del piloto se cerrará el protocolo inicial. Después se incorporarán sus resultados y el dimensionamiento en una versión fechada y congelada **antes de la ejecución definitiva**, conservando el historial de cambios. El registro debe identificar los modelos, datos y hashes, código, folds, objetivos, comparaciones, transformaciones, reglas de parada, $\delta$, métrica, supuestos y número final de ejecuciones.

### 4.1.10. Estado del diseño y del piloto

| Producto | Estado |
|---|---|
| Datos y EDA de desarrollo | Ejecutados y documentados |
| Folds de validación de 2024 | Materializados con `timeseries-cv` |
| Selección bibliográfica de tres modelos | Pendiente |
| Configuraciones, contrastes, margen práctico y política de métricas | Pendientes de cierre |
| Preregistro | Borrador fechado; no congelado |
| Cinco ejecuciones piloto por modelo | No ejecutadas |
| Media, desviación, CV y diferencias pareadas | Sin resultados todavía |
| $R$ requerido y MDE | Pendientes de piloto y $\delta$ |
| Entrenamiento definitivo y test | No ejecutados |

La presencia de esta sección no sustituye las implementaciones y evaluaciones posteriores del Entregable 1.

## 4.2. Implementación de los Tres Modelos Base

Los modelos deberán ser exactamente los tres seleccionados en §3.2, con la configuración reportada en la literatura o una configuración mínima justificada mediante evidencia. **Todavía no se han seleccionado ni implementado esas tres arquitecturas.** Los apartados siguientes definen las condiciones verificables que debe cumplir su implementación; no son resultados de entrenamiento.

### 4.2.1. Arquitecturas y contrato común

Cada modelo tendrá un identificador, referencias y una descripción reproducible de capas, unidades, activaciones, conexiones y salida no negativa. La arquitectura se enlazará con las ecuaciones y el análisis de complejidad de §3.3. Si la literatura no ofrece detalles suficientes, se seguirá la regla de búsqueda de arquitectura de §3.2.7; no se completarán por preferencia personal.

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

Para cada arquitectura se registrarán el espacio de búsqueda declarado antes de entrenar y el valor finalmente seleccionado. Esta ficha está pendiente de completar con las arquitecturas reales:

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
| Función de pérdida | MSE propuesta en §4.1; cierre antes de comparar | No congelada |
| Épocas máximas y criterio de parada | Pendiente | No seleccionado |

Los campos no se rellenarán con valores supuestamente “habituales” atribuidos a artículos que no los reportan. Cualquier desviación respecto a una fuente se justificará y versionará antes del experimento pertinente.

### 4.2.3. Estrategia de entrenamiento y early stopping

Cada ajuste comenzará con pesos nuevos y su semilla registrada. Se fijarán el número máximo de épocas, orden de lotes, tamaño de lote, precisión numérica y cualquier configuración estocástica relevante. El entrenamiento conservará la relación causal entre muestras y etiquetas aunque se utilicen lotes para optimizar.

El early stopping deberá registrar métrica monitoreada, dirección de mejora, `patience`, `min_delta`, frecuencia de evaluación y política de restauración de pesos. Su conjunto de validación será **interno y anterior al origen evaluado**, con separación suficiente para que sus etiquetas estén disponibles. El único objetivo futuro de cada fold externo no se utilizará para decidir cuándo detener su propio entrenamiento.

La validación externa de 2024 servirá para comparar configuraciones bajo el protocolo predefinido. No se usarán etiquetas de test para seleccionar épocas, arquitectura, transformaciones ni hiperparámetros. El procedimiento de ajuste final y su parada se fijarán con desarrollo antes de evaluar test.

### 4.2.4. Semillas, reinicios y reproducibilidad

Las semillas previstas para el piloto constan en el preregistro; todavía no representan corridas ejecutadas. El conjunto definitivo se dimensionará con el piloto y se compartirá entre los tres modelos. Se registrarán semillas del generador, framework y operaciones estocásticas que correspondan, además de las limitaciones de determinismo de hardware y kernels.

Las transformaciones y el modelo se ajustarán de nuevo en cada fold según su entrenamiento permitido. No se reutilizarán pesos de un fold posterior ni escaladores ajustados con toda la validación. Cualquier entrenamiento incremental o warm start requerirá una regla causal predefinida y comparable; no se presupone en este diseño.

Las fallas, reintentos y corridas interrumpidas quedarán registrados. No se reemplazará una semilla por otra para mejorar el resultado. Si se repite una corrida por un problema técnico, se conservarán la causa y su relación con el intento original.

### 4.2.5. Registros de cada implementación y corrida

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

Se podrá implementar desde cero o utilizar un framework justificado, conservando versión, procedencia y configuración. La elección del framework de Deep Learning sigue pendiente junto con las arquitecturas; no se afirma que ya exista una implementación validada.

Actualmente están verificados el entorno de EDA de `requirements-eda.lock.txt` y el entorno de partición de `requirements-cv.txt`. `timeseries-cv` genera particiones y **no entrena las redes**. Su dependencia de NumPy menor que 2 se mantiene aislada del entorno del EDA, según §2.4.1.

Antes del piloto se fijará el entorno de entrenamiento con versiones compatibles y se comprobará una ejecución funcional de cada modelo con la misma interfaz de datos. No se declarará reproducibilidad solo por disponer de una lista de dependencias.

### 4.2.8. Estado de implementación

| Producto | Modelo 1 | Modelo 2 | Modelo 3 |
|---|---|---|---|
| Identidad y arquitectura extraídas de literatura | Pendiente | Pendiente | Pendiente |
| Código y prueba funcional | Pendiente | Pendiente | Pendiente |
| Espacio de búsqueda y presupuesto comparable | Pendiente | Pendiente | Pendiente |
| Hiperparámetros elegidos y regla de parada | Pendiente | Pendiente | Pendiente |
| Curvas, parámetros, tiempos y entorno medidos | No ejecutado | No ejecutado | No ejecutado |
| Piloto y repeticiones definitivas | No ejecutado | No ejecutado | No ejecutado |

Los datos y las particiones disponibles son prerrequisitos completados; no sustituyen la implementación ni la evaluación de estos tres modelos.

## 4.3. Requisitos Experimentales

Estos requisitos se aplicarán conjuntamente a los tres modelos, respetando el diseño temporal y el dimensionamiento de §4.1. La ejecución del EDA y la generación de folds no acreditan que el entrenamiento, los intervalos de confianza o la evaluación de test ya estén realizados.

### 4.3.1. Control de semillas y determinismo

Cada corrida conservará un identificador y las semillas de los generadores utilizados: Python, NumPy, framework de Deep Learning y operaciones del dispositivo que correspondan. Se registrarán también el orden de lotes, los workers de carga y cualquier otra fuente estocástica del pipeline.

Se declarará si se exige **determinismo estricto**, qué opciones del framework se activan y qué operaciones o condiciones de hardware pueden impedirlo. Fijar una sola semilla no demuestra reproducibilidad exacta. El framework y las opciones concretas se fijarán antes del piloto; no se atribuyen ahora configuraciones a un software de entrenamiento aún no elegido.

Las semillas previstas del piloto son comunes entre modelos. Cada repetición parte de una inicialización nueva; una repetición idéntica usada para comprobar determinismo no cuenta como otra semilla independiente.

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
| Transformaciones solo en train | Escalado, imputación permitida y selección se ajustan dentro del entrenamiento admitido de cada fold | Regla documentada; integración con modelos pendiente |
| Orden temporal | No se barajan periodos; se comprueba disponibilidad de etiquetas y continuidad de las ventanas | Controles ejecutados en el manifiesto de forward chaining |
| Mismo protocolo entre modelos | Mismas muestras, horizonte, covariables disponibles, semillas y criterio de selección | Requisito fijado; comparación todavía no ejecutada |
| Early stopping sin consultar la salida futura evaluada | Validación interna pasada, distinta del día externo que se pretende pronosticar | Regla documentada; implementación pendiente |
| Test reservado | No participa en EDA de desarrollo, piloto ni hiperparámetros | Scripts de EDA y partición no abren archivos de test |
| Integridad y trazabilidad | Brutos intactos, hashes y exclusiones documentadas | Adquisición y derivados de desarrollo registrados |

La separación se refiere al papel de cada objetivo y a su disponibilidad temporal. Una entrada de validación puede reutilizar historia pasada ya observable, como requiere forward chaining; esto no autoriza compartir etiquetas futuras ni ajustar transformaciones con validación o test.

Los controles espaciales, semánticos de NLP o de agrupamiento de imágenes de la guía general no corresponden al alcance actual de una serie financiera única sin esas modalidades. Si se incorporan noticias u otros datos agrupados, deberán añadirse sus controles de disponibilidad, duplicados y dependencia antes de utilizarlos.

### 4.3.7. Evaluación única del test por corrida

El conjunto de prueba se evaluará **una sola vez por corrida**, con modelo, pesos, hiperparámetros y decisiones ya fijados. El preregistro identificará cuándo se considera cerrada una corrida y qué artefactos se conservan.

Las predicciones guardadas permitirán calcular métricas, residuos y figuras sin volver a consultar test para elegir otra configuración. No se reentrenará ni se seleccionará una semilla según su desempeño en test. Si ocurre una falla técnica, se registrará el incidente y el tratamiento previsto; no se ocultará una evaluación desfavorable como si nunca hubiera ocurrido.

Se reportarán 2025 y 2026 por separado dentro del calendario fijado. Disponer de más archivos descargados no amplía automáticamente el conjunto de prueba ni valida una nueva selección después de observar resultados.

### 4.3.8. Estado de cumplimiento

| Requisito | Estado actual |
|---|---|
| Hashes, datos derivados y exclusiones de desarrollo | Registrados |
| Folds temporales y comprobación de disponibilidad | Ejecutados |
| Dependencias y scripts de EDA/partición | Disponibles y verificados |
| Configuración de semillas y determinismo del entrenamiento | Prevista; framework y opciones pendientes |
| Repeticiones, media, desviación e intervalos de modelos | No ejecutados |
| GPU, tiempos y hardware de entrenamiento medidos | No ejecutados |
| Paquete de entrenamiento y evaluación de extremo a extremo | Pendiente |
| Test supervisado congelado y evaluación final | Pendientes |

Esta lista distingue requisitos establecidos de controles ejecutados. No se presenta el benchmark como reproducible o completo hasta verificar también las implementaciones y sus resultados.

## 4.4. Evaluación

El proyecto corresponde a **series de tiempo con objetivo continuo**. Se aplican las métricas de regresión y los diagnósticos temporales de la guía. Las métricas de clasificación, segmentación, texto o localización espacial no se incorporan a este problema de volatilidad de Bitcoin.

```{important}
Esta sección define la evaluación que se ejecutará. Todavía no hay predicciones de los tres modelos ni resultados de test; no se presentan valores de desempeño, curvas de redes o p-values de residuos inexistentes.
```

### 4.4.1. Métrica principal y métricas secundarias

Se propone **RMSE como métrica principal**, coherente con la pérdida MSE, y se reportarán **MAE, MSE, R², MAPE y MASE**, atendiendo tanto a las indicaciones específicas recibidas como a la guía general. sMAPE es una alternativa complementaria si se registra antes de evaluar; no se elegirá después para favorecer un resultado.

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

Las métricas se calcularán en la **escala original de la volatilidad**, después de invertir transformaciones de la etiqueta. Se declarará si se muestran unidades de retorno logarítmico o puntos porcentuales. Multiplicar la etiqueta por 100 multiplica RMSE y MAE por 100 y MSE por 10.000; los resultados de distintas escalas no se mezclarán.

### 4.4.2. Errores relativos y escalados

Para MAPE se requiere una política previa sobre ceros y valores cercanos a cero:

$$
\mathrm{MAPE}=\frac{100}{N}\sum_t\left|\frac{e_t}{Y_t}\right|.
$$

Si hay $Y_t=0$, esa expresión no está definida sobre el conjunto completo. Se informará la cantidad de casos y la política preregistrada; si se reporta una versión sobre un subconjunto válido, se identificará su denominador y cobertura. No se añadirán epsilons arbitrarios ni se ocultarán casos excluidos. Valores positivos muy pequeños pueden dominar MAPE y requieren interpretación conjunta con MAE y RMSE.

Si se adopta sMAPE, se registrará su convención exacta, incluido el factor de escala y el tratamiento del caso $Y_t=\widehat Y_t=0$. Que sea simétrica no elimina todos los problemas cerca de cero.

Para MASE se fijará un rezago de referencia $s$ usando exclusivamente desarrollo y se calculará, en el entrenamiento permitido de cada fold:

$$
a_{\mathrm{train},s}
=\frac1{|\mathcal J_s|}\sum_{j\in\mathcal J_s}|Y_j-Y_{j-s}|,
\qquad
q_t=\frac{|Y_t-\widehat Y_t|}{a_{\mathrm{train}(t),s}},
\qquad
\mathrm{MASE}=\frac1N\sum_t q_t.
$$

$\mathcal J_s$ contiene pares de etiquetas válidas separados por **$s$ días de calendario**, completamente observables en ese entrenamiento. No se comprimen huecos para fingir consecutividad. El denominador es común a los modelos comparados dentro de cada fold; puede variar al crecer el entrenamiento. Si vale cero o no hay pares suficientes, MASE se marca no definida para esos casos y se reporta su cobertura. El valor de $s$ sigue pendiente de cierre, sin escogerlo mirando test.

### 4.4.3. Evaluación por horizonte y referencias ingenuas

El horizonte principal confirmado es **un día**. Las métricas se reportarán para cada combinación de longitud de entrada $D$ y ventana de volatilidad $n$, manteniéndolas separadas porque definen configuraciones y objetivos distintos. Si se añaden horizontes, se registrarán y se informarán métricas por horizonte antes de cualquier promedio global.

La referencia contextual principal será **persistencia de la volatilidad histórica disponible** al pronosticar. Se evaluará también un ingenuo estacional cuando se justifique un periodo $s$ con desarrollo y su información sea observable. Por ejemplo, el periodo semanal debe justificarse y fijarse; no se adopta automáticamente porque el EDA muestre un STL con periodo siete.

Las referencias se evaluarán en las mismas fechas, escala y etiquetas que los modelos. Si una referencia necesita más historia, se declarará la cobertura común y la exclusión correspondiente. Ningún método ingenuo ocupa uno de los tres puestos obligatorios de modelos de Deep Learning.

En los folds diarios actuales, RMSE se calcula agrupando las predicciones de una ejecución según §4.1.4. El promedio de errores RMSE de folds con un solo punto equivale a MAE y no se presentará como RMSE global.

### 4.4.4. Comparación justa entre los tres modelos

Antes de comparar se comprobará igualdad de datos, particiones, orígenes, etiquetas, preprocesamiento permitido, semillas y configuración de evaluación. Las arquitecturas pueden organizar la misma información de forma diferente, pero no recibir atributos o fechas adicionales sin declarar otro experimento.

Se mantendrán las mismas definiciones de métricas y reglas sobre ceros, inversas de transformaciones, valores no finitos y fallas. Una predicción no finita no se omitirá silenciosamente del promedio. Se conservarán el número previsto de casos, el evaluado y el motivo de cualquier diferencia de cobertura.

Se usará el mismo presupuesto de búsqueda documentado en §4.2.6. Las comparaciones no mezclarán el mejor resultado de una única semilla para un modelo con la media de todas las semillas de otro.

### 4.4.5. Análisis de resultados y variabilidad

Se reportarán **media, desviación estándar muestral e intervalo de confianza** por modelo y configuración, con el número de ejecuciones y el método de construcción. Se distinguirán métricas de selección en validación de la evaluación final del test, y se conservarán las fechas de cada predicción.

La variabilidad por semilla se examinará dentro de cada modelo y configuración. Las diferencias entre modelos se examinarán sobre las mismas semillas y fechas. Se presentarán distribuciones de métricas por corrida y diferencias pareadas; si se realiza una descomposición formal de varianza, se declararán el modelo estadístico y sus supuestos, sin mezclar configuraciones o días dependientes como réplicas independientes.

La variabilidad temporal se analizará por bloques o segmentos predefinidos y con métodos adecuados a su dependencia. Los intervalos entre semillas sobre el mismo mercado no se interpretarán como toda la incertidumbre de generalización. Las pruebas y tamaños de efecto se interpretarán junto con $\delta$, no solo por significación estadística.

### 4.4.6. Robustez, sobreajuste y segmentos relevantes

La robustez se estudiará con segmentos y, si se incorporan, perturbaciones controladas fijadas antes de la evaluación correspondiente. En desarrollo pueden definirse niveles de volatilidad con umbrales estimados solo en entrenamiento; esos umbrales se congelarán antes de trasladarlos a validación o test.

| Análisis | Evidencia que deberá conservarse | Cautela |
|---|---|---|
| Periodos y regímenes | Métricas por bloque temporal y cantidad de casos | No seleccionar retrospectivamente solo los periodos favorables |
| Magnitud de volatilidad | Errores por rango del objetivo con umbrales documentados | No ajustar umbrales con test |
| Sensibilidad a ruido o faltantes simulados | Tipo, intensidad, semilla y momento de la perturbación | No contaminar etiquetas ni cambiar inadvertidamente el problema |
| Tamaño de entrada | Resultados separados para 7/14/21/28 días | No confundir historia de entrada con ventana de volatilidad |
| Curvas y brecha train–validation | Pérdida y métricas por época y conjunto, con escalas comparables | Una brecha pequeña por sí sola no descarta underfitting |

Las curvas de aprendizaje y las brechas entre entrenamiento y validación se usarán para diagnosticar overfitting o underfitting, considerando también el nivel de error y el comportamiento de la referencia ingenua. No se dibujarán curvas sintéticas como si fueran resultados observados. Las evaluaciones de robustez de test deberán estar predefinidas y reutilizar las predicciones guardadas cuando corresponda, sin reabrir selección de modelos.

### 4.4.7. Análisis cualitativo de predicciones y residuos

Se generarán gráficas de valores reales frente a predicciones en el tiempo y diagramas real–predicho, identificando modelo, semilla, configuración, horizonte y periodo. Los casos difíciles y errores sistemáticos se seleccionarán con una regla trazable y se contextualizarán sin atribuir causas económicas no verificadas.

Los residuos conservarán fecha y signo. Con $e_t=Y_t-\widehat Y_t$, una media positiva indica subestimación de volatilidad. Se examinarán sesgo, distribución, relación con nivel pronosticado, estructura temporal y heterocedasticidad; se mostrarán ACF de residuos y de sus cuadrados cuando correspondan.

Para series de tiempo se documentarán diagnósticos como **Ljung–Box** y **BDS**, con rezagos, parámetros, tamaño muestral, supuestos y multiplicidad. El BDS requerido sobre residuos del primer horizonte se ejecutará cuando existan predicciones: una prueba sobre retornos crudos del EDA no lo sustituye. Un rechazo o no rechazo no demuestra por sí solo capacidad predictiva ni independencia absoluta.

Los huecos y el solapamiento de objetivos móviles se considerarán al elegir tramos y métodos. No se concatenarán fechas separadas como si fueran observaciones consecutivas ni se usarán residuos de entrenamiento para aparentar un diagnóstico fuera de muestra.

### 4.4.8. Productos y estado de ejecución

| Producto de evaluación | Estado |
|---|---|
| Métricas, escalas y agregación documentadas | Protocolo disponible; políticas de MAPE/MASE pendientes de cierre |
| Predicciones de los tres modelos por fecha y semilla | No generadas |
| Tablas por configuración, horizonte y referencia ingenua | Sin resultados todavía |
| Media, desviación, intervalos y comparación pareada | Pendientes de corridas |
| Robustez, curvas y análisis por segmentos | Pendientes de modelos entrenados |
| Predicciones frente a reales y diagnóstico de residuos | Pendientes |
| Evaluación definitiva de test | No ejecutada |

Las tablas y gráficas de este apartado se incorporarán desde predicciones verificables, sin reutilizar estadísticas descriptivas del EDA como si fueran métricas de pronóstico.
