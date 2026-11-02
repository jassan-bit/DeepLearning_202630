# 3. Revisión del Estado del Arte y Extracción de los Tres Modelos Base

El resultado central de esta revisión será identificar **exactamente tres modelos simples de Deep Learning** utilizados como referencias comparativas en estudios sobre pronóstico de volatilidad de Bitcoin. Su selección se sustentará en artículos y en la descripción reproducible de sus arquitecturas y protocolos.

```{important}
Se presenta la estrategia de revisión propuesta. Las consultas bibliográficas sistemáticas, sus conteos, el cribado y la selección definitiva de tres modelos aún no se han ejecutado. El antecedente ya citado no constituye por sí solo una revisión del estado del arte.
```

## 3.1. Estrategia de búsqueda

### 3.1.1. Palabras clave y cadenas booleanas

Se buscarán términos de activo, objetivo y método. Las siguientes cadenas son propuestas completas en sintaxis booleana genérica; al ejecutarlas se conservará también la sintaxis exacta adaptada al campo y plataforma de cada base.

**Cadena principal:**

```text
("Bitcoin" OR "BTC")
AND ("volatility forecasting" OR "volatility prediction" OR "realized volatility")
AND ("deep learning" OR "neural network" OR "neural networks")
```

**Cadena complementaria para referencias comparativas:**

```text
("Bitcoin" OR "BTC")
AND ("volatility forecasting" OR "volatility prediction")
AND ("MLP" OR "multilayer perceptron" OR "RNN" OR "recurrent neural network"
     OR "LSTM" OR "GRU" OR "CNN" OR "convolutional neural network")
AND ("baseline" OR "benchmark" OR "comparison" OR "comparative")
```

La cadena principal evita depender de que los autores utilicen explícitamente la palabra baseline. Las búsquedas complementarias y por referencias citadas se registrarán por separado para no ocultar duplicados ni presentar una búsqueda dirigida como evidencia de exhaustividad.

### 3.1.2. Bases de datos y periodo

Se consultarán **Web of Science, Scopus e IEEE Xplore**, con arXiv como fuente complementaria de tendencias recientes. Cada consulta registrará base, interfaz, campos, cadena exacta, filtros, fecha de ejecución y número de resultados. Una consulta no ejecutada o bloqueada se identificará como tal: no se sustituirá su conteo por cero.

El periodo propuesto comprende **2019–2026**, con cierre en la fecha real de ejecución de la búsqueda: ocho años calendario, incluyendo el año parcial 2026. Se priorizará literatura científica reciente y, cuando pueda verificarse, publicaciones Q1. El cuartil se registrará por año, categoría y sistema de clasificación; no se inferirá de la reputación de la revista ni se asignará a preprints.

| Base | Cadena y filtros | Fecha de ejecución | Registros recuperados | Estado |
|---|---|---|---|---|
| Web of Science | Por registrar al ejecutar | Pendiente | No disponible | No ejecutada |
| Scopus | Por registrar al ejecutar | Pendiente | No disponible | No ejecutada |
| IEEE Xplore | Por registrar al ejecutar | Pendiente | No disponible | No ejecutada |
| arXiv, complementaria | Por registrar al ejecutar | Pendiente | No disponible | No ejecutada |

### 3.1.3. Criterios de inclusión y exclusión

| Dimensión | Inclusión | Exclusión o tratamiento |
|---|---|---|
| Problema | Pronóstico de volatilidad de Bitcoin con definición del objetivo verificable | Predicción exclusivamente de precio, dirección o rentabilidad sin evaluación de volatilidad |
| Método | Uso de Deep Learning y comparación con una referencia identificable | Solo modelos clásicos o referencias comparativas no identificables |
| Evidencia | Métricas verificables y protocolo de evaluación descrito | Resultados sin definición de métricas o sin protocolo suficiente |
| Arquitectura | Descripción suficiente para extraer una arquitectura base reproducible | Referencia mencionada sin arquitectura ni información suficiente para reconstruirla |
| Periodo y documento | Publicaciones del periodo definido; preprints identificados como tales | Fuera del periodo, salvo antecedente justificado registrado por separado |
| Acceso | Texto suficiente para verificar los criterios | Un resumen por sí solo no acredita detalles experimentales; registrar acceso pendiente |

Se distinguirá relevancia temática de comparabilidad experimental: una diferencia de exchange, horizonte, frecuencia, definición de volatilidad o partición impide trasladar directamente cifras al experimento propio. El uso exacto de Binance Spot BTCUSDT se documentará cuando exista evidencia, sin asumirlo por el título del artículo.

### 3.1.4. Trazabilidad y flujo de selección

Cada registro conservará DOI o identificador, referencia, base de origen y decisión. Se depurarán duplicados por DOI y, cuando falte, por título, autores y año. Versiones preprint y publicadas se vincularán evitando contar dos veces el mismo estudio.

El flujo tipo PRISMA registrará las etapas siguientes, con conteos reconciliables:

| Etapa | Conteo | Evidencia requerida |
|---|---|---|
| Registros identificados por base y otras vías | Pendiente | Exportaciones y registros de consulta |
| Duplicados retirados | Pendiente | Identificadores y regla de deduplicación |
| Registros cribados por título y resumen | Pendiente | Decisiones de cribado |
| Textos completos buscados y no recuperados | Pendiente | Estado de acceso |
| Textos completos evaluados | Pendiente | Ficha por estudio |
| Textos excluidos, con razones | Pendiente | Motivo explícito por registro |
| Estudios finalmente incluidos | Pendiente | Matriz de extracción final |

La selección y su revisión serán documentadas por Jassan Arteta. No se afirma que haya habido dos revisores independientes. Las dudas se registrarán y resolverán consultando el texto completo y, si procede, mediante revisión académica; se identificará quién resolvió cada caso y la justificación. El diagrama final se generará con los conteos reales, no con valores ilustrativos.

### 3.1.5. Evaluación de calidad de los estudios

Para cada estudio incluido se evaluarán los siguientes aspectos con estados **documentado, parcial, no documentado o no aplicable**, acompañados de una cita de página, tabla o sección:

- Claridad de datos, objetivo, horizonte, particiones y selección de hiperparámetros.
- Disponibilidad de información y ausencia aparente de fuga temporal, incluido el ajuste de transformaciones y la separación de etiquetas solapadas.
- Número de repeticiones, semillas y tratamiento de la incertidumbre.
- Pruebas estadísticas, supuestos, correcciones por comparaciones múltiples y tamaños de efecto.
- Suficiencia de la descripción de las arquitecturas de referencia para implementarlas.

La ausencia aparente de fuga no demuestra su ausencia real; la falta de información se reportará como incertidumbre. La calidad se tendrá en cuenta al justificar las tres referencias y no se reemplazará por el cuartil de la revista.

## 3.2. Identificación de los Tres Modelos Base de Deep Learning

Los tres modelos se extraerán de las referencias comparativas empleadas por los estudios incluidos. Su función será establecer un nivel mínimo de desempeño para el pronóstico de volatilidad de Bitcoin y permitir evaluar mejoras de modelos posteriores. **Todavía no hay tres modelos seleccionados:** las familias candidatas no constituyen una selección final.

### 3.2.1. Procedimiento de extracción

1. Identificar en cada estudio incluido cuáles modelos de Deep Learning se usan realmente como baseline, distinguiéndolos del modelo novedoso propuesto por sus autores.
2. Registrar arquitectura, profundidad, ancho, regularización, configuración de entrenamiento y ubicación exacta de la evidencia.
3. Construir una tabla de frecuencia por modelo y variante arquitectónica. Un mismo estudio cuenta como máximo una vez para cada candidato; versiones duplicadas de un estudio no suman evidencia independiente.
4. Comparar frecuencia, profundidad y número de parámetros. Si falta información para calcular parámetros, registrar **no reportado**, sin inventar dimensiones.
5. Seleccionar exactamente tres referencias recurrentes y de baja complejidad, justificando los compromisos entre frecuencia, reproducibilidad y simplicidad.
6. Verificar diversidad del sesgo inductivo y capacidad de utilizar el mismo pipeline, calendario, objetivo y muestras elegibles.

La selección no se resolverá por preferencia personal ni solo por el mejor resultado publicado. Los conteos y las decisiones deberán poder reconstruirse desde las fichas de los estudios.

### 3.2.2. Ficha de arquitectura por estudio

| Campo | Información que se extraerá |
|---|---|
| Identificación | Referencia, DOI o identificador y página, tabla o sección |
| Papel experimental | Evidencia de uso como baseline o referencia comparativa |
| Entrada y salida | Variables, frecuencia, ventana, horizonte, dimensiones y estrategia de pronóstico |
| Arquitectura | Familia, capas, unidades o filtros, kernels, dilataciones, conexiones y activaciones |
| Complejidad | Profundidad y número de parámetros reportados o calculados con dimensiones explícitas |
| Regularización | Dropout, penalizaciones, restricciones y otros mecanismos declarados |
| Entrenamiento | Pérdida, optimizador, learning rate, batch size, épocas y early stopping |
| Reproducibilidad | Código, semillas, repeticiones e información faltante |
| Evaluación | Particiones, métricas, valores y tratamiento de incertidumbre |

Un valor calculado por nosotros se distinguirá de uno reportado por el artículo. La ausencia de detalles no se interpretará como ausencia de regularización ni como una configuración predeterminada.

### 3.2.3. Tabla de síntesis obligatoria

La tabla final tendrá las columnas exigidas por el Entregable 1:

| Referencia | Problema | Dataset | Modelos de Deep Learning básicos utilizados como referencia | Métricas reportadas | Valores obtenidos |
|---|---|---|---|---|---|
| Pendiente de revisión y extracción | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente |

Esta fila indica el estado del producto y **no cuenta como un estudio incluido**. Cada valor se acompañará de unidades, horizonte, partición y localización en el artículo. No se compararán directamente cifras de objetivos, escalas o datasets diferentes ni se presentarán como resultados propios.

El antecedente de Huang, Sangiorgi y Urquhart (2024), listado en [Referencias](referencias.md), deberá pasar los criterios de inclusión y la evaluación de calidad antes de formar parte de esta tabla.

### 3.2.4. Frecuencia y complejidad de candidatos

Para series de tiempo, la guía propone las familias siguientes como orientación. Su presencia aquí no acredita frecuencia observada ni selecciona modelos.

| Familia candidata | Estudios que la usan como referencia | Frecuencia | Profundidad y parámetros de variantes documentadas | Estado |
|---|---|---|---|---|
| MLP sobre ventana de rezagos | Pendiente | No calculada | Por extraer | Candidata |
| RNN simple | Pendiente | No calculada | Por extraer | Candidata |
| LSTM de una capa | Pendiente | No calculada | Por extraer | Candidata |
| GRU de una capa | Pendiente | No calculada | Por extraer | Candidata |
| CNN 1D o TCN poco profunda | Pendiente | No calculada | Por extraer | Candidata |

CNN y TCN, así como LSTM y GRU, se desagregarán cuando las variantes documentadas lo requieran; no se fusionarán conteos de forma arbitraria. Se definirá el conjunto de estudios elegibles que sirve como denominador de las frecuencias relativas.

La tabla ordenada y la justificación final mostrarán cómo se equilibran recurrencia y complejidad. Si empatan candidatos, se documentará la decisión considerando la calidad de la evidencia, la reproducibilidad y la diversidad de sesgos inductivos.

### 3.2.5. Criterios de selección y no redundancia

Cada modelo seleccionado deberá ser de Deep Learning, tener arquitectura simple y reproducible, aparecer como comparación en literatura pertinente y poder entrenarse bajo las mismas condiciones de evaluación que los otros dos.

La diversidad se justificará por los mecanismos que incorporan los modelos: una MLP procesa una ventana explícita de rezagos; una red recurrente incorpora un estado compartido en el tiempo; una convolucional incorpora filtros locales y compartición de pesos. Estas diferencias ilustran el criterio, **sin fijar de antemano ese trío**. Elegir dos variantes recurrentes requeriría justificar por qué su diferencia aporta diversidad suficiente para el problema.

Persistencia y otros modelos clásicos solo podrán actuar como referencias de contexto: no ocuparán ninguno de los tres puestos obligatorios.

### 3.2.6. Equivalencia con las referencias clásicas

Para este objetivo continuo, la comparación conceptual pertinente es el papel de una SVR con núcleo lineal como referencia sencilla en regresión. La equivalencia requerida es de **propósito comparativo**, no de identidad matemática ni de capacidad de representación.

La justificación de cada modelo final explicará su capacidad, sus supuestos, su sesgo inductivo y su complejidad, y por qué constituye una referencia mínima razonable frente a arquitecturas avanzadas. Se distinguirán los supuestos de linealidad de una referencia clásica de la representación no lineal y temporal de las redes seleccionadas. Esta justificación se completará después de seleccionar las arquitecturas concretas.

### 3.2.7. Arquitecturas insuficientemente descritas

Si un estudio reporta únicamente una familia, por ejemplo «MLP», no se elegirá una arquitectura por criterio personal. Conforme al Entregable 1, se aplicará el procedimiento de búsqueda de arquitectura de la sección 14 del Entregable 2, declarando el espacio antes de entrenar.

La sección 14 debe consultarse antes de ejecutar ese procedimiento; aquí no se inventan sus reglas. Se registrarán la información faltante, el espacio predefinido, el criterio de selección y su evaluación exclusivamente en desarrollo.

### 3.2.8. Estado de cierre

| Producto | Estado |
|---|---|
| Estudios incluidos y fichas de extracción | Pendientes |
| Tabla de síntesis con métricas y valores verificables | Pendiente |
| Frecuencia, profundidad y parámetros de candidatos | Pendientes |
| Selección de exactamente tres modelos no redundantes | Pendiente |
| Justificación comparativa y arquitecturas reproducibles | Pendientes |

Esta sección documenta el procedimiento exigido; no afirma haber ejecutado la revisión, seleccionado los modelos ni entrenado redes.
