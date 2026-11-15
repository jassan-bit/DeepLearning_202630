# Revisión preliminar de antecedentes del Bloque C

La consulta y la lectura realizadas permiten identificar trabajos cercanos, pero **no acreditar R1**. El candidato conserva CNN causal, LSTM y atención, con una penalización de discordancia respecto a un promedio causal y una saturación radial de los valores. La novedad deberá evaluarse sobre esa regla exacta; no sobre la combinación convencional de bloques, la palabra Lipschitz o el cambio de datos a Bitcoin.

## Alcance realizado

- Se registraron seis cadenas bilingües y 18 solicitudes a OpenAlex, Crossref y arXiv. Las 18 fueron denegadas por el proxy antes de ejecutar una búsqueda en esas bases: el número de resultados es desconocido, no cero. No se consultaron nuevamente Scopus o Web of Science.
- Se ejecutaron ocho consultas de repositorios de GitHub. Sus resultados son descubrimiento de software, no una revisión científica exhaustiva ni una cuarta base que subsane las solicitudes bloqueadas.
- Se examinaron por palabras del título tres exportaciones recibidas en E2. Sus 38 ocurrencias candidatas no son 38 estudios nuevos ni únicos: los archivos se solapan y no se realizó aquí una adjudicación completa de elegibilidad.
- Se leyeron focalmente dos artículos originales completos desde el repositorio editorial oficial de PMLR en GitHub. Los PDF permanecen en el área temporal de trabajo; los entregables conservan la referencia, versión, hash, localizadores y reseña crítica propia.
- La matriz reúne siete trabajos cercanos con niveles de lectura explícitos. Las lecturas previas documentadas por E2 no se presentan como nuevas lecturas del original.

## Hallazgos que afectan el diseño

**Kim, Papamakarios y Mnih (2021)** prueban que la atención estándar por productos entre entradas no es globalmente Lipschitz en dominio no acotado y proponen atención L2. Su §3.2, ecuación 8 y teorema 3.2 usan distancias entre pares y requieren, en su construcción, mapas de consulta y clave ligados. Este resultado no debe trasladarse como prueba de que toda atención o el híbrido de E2 sea inestable. La consulta de pooling paramétrica y la regla candidata son otras funciones.

**Dasoulas, Scaman y Virmaux (2021)** ya estudian una consulta aprendida como parámetro y normalizan los scores usando normas de consultas y entradas. Sus ecuaciones 15 y 17 y el teorema 3 demuestran propiedades Lipschitz. Por tanto, una consulta aprendida o demostrar una cota no constituyen por sí mismas una contribución nueva. La diferencia candidata consiste en penalizar discordancia causal y saturar los valores, en vez de reproducir su normalización de scores. Esa diferencia estructural todavía debe cotejarse con otros trabajos, equivalencias y usos previos.

**Peng et al. (2024)** aparece en la exportación Scopus recibida con el título *Attention-based CNN–LSTM for high-frequency multiple cryptocurrency trend prediction*. Esa evidencia bibliográfica impide presentar la familia CNN–LSTM–atención para criptomonedas como una idea inédita. Su título apunta a tendencia; la tarea exacta y la regla de atención requieren lectura íntegra antes de decidir su comparabilidad con σ₇.

**TimeMixer** aporta procesamiento multiescala previo, y **HARNet** aporta estructura causal para volatilidad. Ninguna de esas ideas genéricas sustenta novedad por sí sola. La limitación de calendario medida en ETTh1 por E2 tampoco demuestra una limitación predictiva equivalente en BTC/D7. El trabajo CEEMDAN–RF–LSTM mantiene una definición de varianza realizada distinta del objetivo σ₇; no permite promediar métricas entre tareas.

La saturación tampoco se reclama como nueva: $T_B(h)=Bh/\sqrt{B^2+\|h\|^2}$ coincide con el gradiente de $\phi_B(h)=B^2(\sqrt{1+\|h\|^2/B^2}-1)$, una penalización de forma pseudo-Huber radial. Es una identidad algebraica explícita; en esta revisión no se verificó una fuente histórica primaria específica para ese nombre. La búsqueda debe evaluar la combinación exacta, no atribuir originalidad a la saturación aislada.

## Archivos

| Archivo | Contenido |
|---|---|
| `search_plan.json` | Seis cadenas bilingües previstas y su texto de descubrimiento para APIs. |
| `search_attempts.json` y `.csv` | URL, consulta enviada, sintaxis, fecha UTC, denegación y conteos desconocidos. |
| `github_discovery.json` | Ocho consultas realmente ejecutadas y repositorios devueltos, sin inferencia de ausencia científica. |
| `corpus_scan.json` y `corpus_title_candidates.csv` | Fuentes recibidas en E2, hashes, recuentos por archivo y candidatos sólo por título. |
| `review_matrix.json` y `.csv` | Siete trabajos, lectura, tarea, mecanismo, diferencia candidata y acciones pendientes. |
| `sources_manifest.json` | Versiones Git, URL, bytes, SHA-256 y alcance de lectura. |
| `references.bib` | Referencias de los siete trabajos. |
| `summary.json` | Alcance verificable, límites y R1 no acreditado. |

`record_search_attempts.py` permite repetir las solicitudes públicas y guardar una nueva bitácora en otra carpeta mediante `--output`. La sintaxis de OpenAlex y Crossref es texto libre: las cadenas booleanas bilingües son planificación y no se presentan como filtros booleanos ejecutados. El script nunca convierte una denegación en cero resultados o en una declaración de novedad.

## Pendientes para R1

Completar la búsqueda efectiva en al menos tres bases, inspeccionar los originales y trabajos derivados más cercanos, cerrar el recorrido de citas desde tres a cinco trabajos y revisar implementaciones pertinentes. Después se debe redactar una declaración de novedad acotada y repetir la búsqueda antes de entregar. Una persona ajena al diseño debe repetir al menos dos cadenas y emitir su concepto. Otra ejecución o subagente de IA no satisface esa verificación externa.

La revisión presente no demuestra que la regla exacta esté ausente de la literatura, no verifica superioridad y no constituye un protocolo experimental congelado.
