# Repetición de dos consultas nativas con IA declarada

**Revisor: agente de IA de ChatGPT/Codex. No hubo revisión humana ni certificación de R1.** La sustitución se documenta conforme a la indicación del profesor comunicada por el autor y aplicada a la modalidad del revisor de R1; los formularios históricos se conservan.

Las dos cadenas canónicas se enviaron sin alterarlas a la interfaz nativa de arXiv mediante la herramienta web. Ambas produjeron contenido bibliográfico observable entre **2026-11-18 21:05:55 y 21:05:56 UTC**. La herramienta indicó «Crawled: today»; no expuso una hora de transacción del servidor de arXiv ni un estado HTTP verificable de forma independiente. Se acredita una nueva solicitud y observación mediante esa herramienta, con esa limitación de frescura.

| ID | Cadena exacta | Total nativo observado | Página anunciada | Pares título/resumen examinados |
|---|---|---:|---|---:|
| Q2 | `CNN LSTM attention cryptocurrency` | 2 | 1–2 | 2 |
| Q5 | `neural network output constraints` | 963 | 1–50 | 4 |

Los enlaces solicitados fueron [Q2 nativa](https://arxiv.org/search/?query=CNN+LSTM+attention+cryptocurrency&searchtype=all&abstracts=show&order=-announced_date_first&size=50) y [Q5 nativa](https://arxiv.org/search/?query=neural+network+output+constraints&searchtype=all&abstracts=show&order=-announced_date_first&size=50). Ambas usan campo `all`, resúmenes visibles, orden descendente de anuncio y tamaño de página 50. La herramienta mostró los dos resúmenes completos de Q2 y el inicio de Q5; se cribaron deliberadamente los cuatro primeros resultados de Q5. No se atribuye lectura a todos los 50 resultados anunciados.

El archivo [evidence.json](evidence.json) conserva URL, encabezado nativo exacto, conteos, localizadores de líneas, identificadores, títulos, paráfrasis breves de los resúmenes realmente examinados y decisiones de selección. Son seis registros curados; no es una exportación completa de la página ni un archivo del HTML original. **No se leyeron cuerpos de artículos en este control.** El resultado de Q5 en la posición 4 se conserva como antecedente transferible que requiere leer su texto completo; ninguna coincidencia o ausencia parcial decide la novedad del modelo.

## La interfaz y la API histórica tienen sintaxis diferente

Se repitieron las dos cadenas léxicas exactas en una interfaz nativa. No se volvió a ejecutar la solicitud histórica de la API. Las respuestas congeladas de Colab anteponen `all:` y sus títulos de feed muestran explícitamente `OR` entre los conceptos. Sus totales son 133166 para Q2 y 620884 para Q5, con diez entradas cada una. No pueden compararse con los totales de la interfaz como si representaran altas o bajas de artículos.

Los dos XML históricos se leyeron en el commit fijado `edf6fe312987fd7299f6ca850a20a67e4f1c3cbc` mediante el complemento de GitHub y se analizaron nuevamente desde sus bytes locales. Los SHA de blob Git y tamaños coincidieron con los leídos mediante el complemento. Sus SHA-256, títulos de feed, fechas originales y enlaces están en `frozen_api_comparison` del JSON. Esta comprobación **es una relectura de evidencia congelada y no una repetición de red de la API**; se incluye para documentar la diferencia de sintaxis, no para sustituir la observación nativa.

## Alcance del estado

`two_native_ui_lexical_query_repetitions_observed=true`: las dos consultas nativas devolvieron resultados observables. `identical_historical_api_request_repeated=false`, `human_external_review_completed=false`, `full_screening_complete=false` y `R1_certified=false` permanecen explícitos. La modalidad de IA declarada se aplica conforme a esa indicación; la valoración sustantiva de R1 corresponde al evaluador. Una eventual entrega posterior a esta fecha necesita considerar su propio corte de actualización.

Este directorio es nuevo y no modifica la revisión previa, el formulario humano, el candidato, los entrenamientos, los pronósticos o los contrastes. No se realizaron mutaciones remotas en GitHub.
