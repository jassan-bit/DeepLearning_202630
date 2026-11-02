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

## 3.2. Estado de la selección de modelos

**Todavía no hay tres modelos seleccionados.** Las familias candidatas mencionadas en el libro no constituyen una selección definitiva. Después de la revisión se incorporará una matriz estudio–modelo con arquitectura, papel como baseline, datos, objetivo, métricas, protocolo, evidencia bibliográfica y limitaciones.

El antecedente de Huang, Sangiorgi y Urquhart (2024), listado en [Referencias](referencias.md), deberá pasar los mismos criterios y evaluación de calidad antes de incluirse en el conjunto final de estudios.
