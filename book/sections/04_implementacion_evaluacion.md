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
