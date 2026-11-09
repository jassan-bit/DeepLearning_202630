# Bitácora nativa PROVISIONAL y editable · E1

Este registro completa una hoja de trabajo provisional a partir de evidencia guardada. **Sus cotas, fechas de referencia y cadenas propuestas quedan pendientes de actualización y no acreditan totales nativos, consultas históricas ni nuevas ejecuciones.** El registro se compila el 07/11/2026 (UTC); esa fecha tampoco es una fecha de búsqueda.

| Base | Familia | Fecha de referencia | Filas exportadas exactas | Cota compatible con el export | Estado |
| --- | --- | --- | ---: | --- | --- |
| Web of Science | TCN | 06/11/2026 (referencia) | 8 | ≥8 | provisional_lower_bound |
| Web of Science | GRU/LSTM | 06/11/2026 (referencia) | 59 | ≥59 | provisional_lower_bound |
| Scopus | TCN | 06/11/2026 (referencia) | 19 | ≥19 | provisional_lower_bound |
| Scopus | GRU/LSTM | 06/11/2026 (referencia) | 158 | ≥158 | provisional_lower_bound |
| IEEE Xplore | TCN/GRU/LSTM | No ejecutada | Desconocido | Sin base para estimar | planned_not_executed |

Las cotas ≥8, ≥59, ≥19 y ≥158 se apoyan en IDs nativos distintos dentro de cada archivo, contados por `source_key`; en estos cuatro exports los IDs son únicos y ninguno está vacío. Cada cota expresa únicamente el mínimo compatible con ese export, **no un estimador del total nativo ni un total visible real**. No se suman como total de resultados o estudios. Las 244 ocurrencias exportadas y los 182 grupos candidatos conservan su significado y valores exactos del registro anterior; los grupos siguen sujetos a revisión de identidad.

Para Scopus, «06/11/2026 (referencia)» procede del nombre del archivo: la zona horaria y la fecha/hora real de ejecución no están establecidas. Para WoS, la misma etiqueta corresponde a recepción y confirmación contextual del usuario el 06/11/2026 (America/Bogota), no a la ejecución de la consulta. No se reconstruyen horas. Las familias TCN y GRU/LSTM y el filtro 2019–2026 están confirmados por el usuario, sin verificación independiente del filtro en el export; se conservan también las cinco ocurrencias nominales de 2027 del registro original.

`provisional_log.csv` contiene las cinco filas y sus cadenas propuestas completas. Las propuestas se derivan de `search_recovery_20261106/query_plan.csv`: TCN conserva la cadena del plan; GRU/LSTM une ambos bloques con OR dentro del mismo campo; IEEE reúne TCN/GRU/LSTM en All Metadata. WoS solicita el filtro de años aparte; Scopus conserva PUBYEAR > 2018 AND PUBYEAR < 2027; IEEE solicita el filtro aparte. **Ninguna propuesta reconstruye la cadena histórica ni se atribuye a una ejecución observada.** La interfaz y el campo describen la propuesta, no una interfaz histórica recuperada.

En JSON, `native_total_verified`, `native_query_literal_verified`, `native_search_date_verified` y `native_export_range_verified` permanecen `null`, y `native_query_literal_recovered=false`; en CSV los valores `null` son celdas vacías. IEEE está `planned_not_executed`, no ejecutada/no acreditada: sus conteos son `null` (desconocidos), nunca cero. Las cotas, las referencias de fecha y las propuestas llevan `exclude_from_verified_native_search_totals=true` y `exclude_from_prisma_totals=true`; no se incorporan como evidencia verificada de identificación/PRISMA.

Hay **0 nuevas ejecuciones nativas**, 0 totales verificados añadidos, 0 fechas nativas verificadas añadidas y 0 cadenas históricas recuperadas. Se conservan N7=7, las frecuencias históricas y todos los experimentos. La actualización futura deberá aportar captura nativa de cadena literal, fecha/hora y zona, colección/interfaz, filtros, total y rango exportado antes de cambiar el estado de verificación.

`provenance.json` fija hashes y tamaños de las fuentes de metadatos anteriores; este compilador no lee ni publica exports originales, PDF, textos completos, resúmenes nativos o contactos. `manifest.json` usa una lista pública cerrada (`e1_public_metadata_allowlist_v1`): sólo README, CSV provisional, resumen y procedencia, con SHA256 y bytes, más el manifiesto. El espejo público conserva exactamente los mismos bytes.

Reproducción: `python scripts/build_e1_provisional_native_log.py`. Verificación sin escritura: `python scripts/build_e1_provisional_native_log.py --verify`. No construye HTML, no crea ZIP, no hace commits y no altera las fuentes originales.
