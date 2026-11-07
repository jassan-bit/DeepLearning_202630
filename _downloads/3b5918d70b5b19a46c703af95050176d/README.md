# Nueva ronda de búsqueda nativa: evidencia e ingesta

La ronda real recibida el 7 de noviembre de 2026 en UTC documenta consultas
nuevas, sin reconstruir las cuatro búsquedas históricas. WoS entregó **271
filas/271 UT** y Scopus **538 filas/538 EID**, coincidentes con sus totales
presenciados visualmente en la conversación. IEEE entregó **25 citas**; el
usuario transcribió el encabezado nativo «Showing 1-25 of 32 results». Ese total
**32 es una transcripción del usuario**, no un número visible en la captura
recortada. La exportación IEEE recibida es parcial: faltan siete resultados.

El día **2026-11-06, Colombia**, fue confirmado expresamente por el usuario para
las tres bases; para IEEE respondió «Sí, también hoy 6 de noviembre». El nombre
de la exportación IEEE aporta contexto adicional y no se usa como hora exacta
de búsqueda. Las horas de recepción y registro de evidencia no sustituyen la
fecha de ejecución.

Las consultas literales y cada procedencia quedan en `execution.json`. WoS
muestra Web of Science Core Collection; no se conocen sus ediciones ni se
acredita un filtro 2019–2026. Sus filas incluyen un año 2017 y tres años 2018.
Scopus contiene años nominales 2019–2026 y el literal suministrado limita PUBYEAR
al intervalo. IEEE muestra OPEN ACCESS ONLY seleccionado y controles de año
2018/2026; el registro distingue controles visibles de la confirmación de un
intervalo efectivamente aplicado. Los alcances reales no se presentan como una
búsqueda perfectamente armonizada entre plataformas.

## Alcance del flujo

Los totales nativos suman **841 ocurrencias de base**, no estudios únicos. Se
recibieron **834 filas/citas**: 271 + 538 + 25. Sobre estas exportaciones, la
agrupación por DOI/identificador produce **630 unidades**; diez enlaces
provisionales por título producen **620 grupos candidatos provisionales**. Los
enlaces por título afectan ocho grupos y requieren cautela de identidad.

La unión con las 244 ocurrencias históricas contiene **1.078 ocurrencias y 676
grupos candidatos**. De los 182 grupos históricos, 126 coinciden por DOI o
identificador con la ronda nueva; 56 quedan solo en el histórico y hay 494 grupos
sin solapamiento histórico. No se fusionaron grupos históricos entre sí. Los
identificadores históricos y sus archivos permanecen intactos.

Esta es una **re-ejecución amplia de auditoría**, no una nueva revisión
sistemática completada. No se leyó ni incluyó automáticamente cada resultado;
las cantidades evaluadas para nueva selección e incluidas quedan desconocidas
para esta rama. No se crean 620 obligaciones nuevas de lectura. Los corpus y
registros de selección anteriores, incluido el denominador histórico de siete
textos completos, mantienen su alcance y procedencia.

## Entradas privadas y procedencia

El contrato de entrada es `schema.json`. La ingesta requiere un manifiesto
privado con `schema_version: "1.0"`, `campaign_id`, `evidence` y entre una y tres
`executions` de bases distintas. Los originales, imágenes y el manifiesto de
entrada se conservan fuera del checkout. Los caminos a archivos son absolutos;
sus SHA-256 se verifican antes de generar resultados. No se solicitan cookies,
credenciales ni registros privados de otras sesiones.

Una imagen presenciada en conversación usa `kind: "conversation_image"`, una
referencia descriptiva `conversation_ref`, `path: null` y `sha256: null`. Una
transcripción nativa usa `kind: "conversation_transcription"`. Ambas conservan
bytes/hash desconocidos: no se fabrica un hash ni se simula que se descargó la
imagen. `review_provenance` identifica quién inspeccionó o comunicó el contenido.

Los campos `verified` enlazan valores revisados con sus `evidence_ids`; los
valores solo comunicados permanecen en `user_declared`. En IEEE, el total
transcrito se enlaza mediante `reported_native_total_evidence_ids` y sigue
separado del total visualmente verificado. Un hash solo acredita los bytes del
archivo, no el contenido de una captura. La persona que prepara el manifiesto
debe revisar la evidencia y enumerar únicamente los campos comprobados.

Las ocho claves auxiliares de `verified` son `executed_at`, `query_literal`,
`interface`, `field_scope`, `collection`, `editions`, `filters` y
`displayed_total`. Lo desconocido queda `null`. Una fecha de día puede guardarse
en `user_declared.executed_on`, con su zona cuando se conoce. No se inventan
segundos ni offsets. `query_literal_received` conserva por separado caracteres
de formato recibidos; la herramienta no reescribe automáticamente consultas.

## Exportaciones e integridad

Se admiten WoS Plain Text con etiquetas (`wos_tagged`), Scopus CSV (`scopus_csv`),
IEEE CSV (`ieee_csv`) e IEEE citas Plain Text (`ieee_plain_text`), en UTF-8. En
IEEE, `Document Identifier` es un tipo documental; un identificador único solo
se toma del número de artículo/enlace. Las citas sin ese número mantienen DOI
como identificador sin inventar un ID nativo.

Cada exportación requiere `export_id` nuevo, `path`, `sha256`, `format`,
`range_start`, `range_end` y `range_evidence_ids`. Si no se conoce el rango,
ambos extremos son `null` y la lista de evidencias está vacía. El número de filas
no se convierte en un rango observado. Cuando se conoce, su cardinalidad debe
coincidir con las filas; se rechazan solapamientos y rangos superiores al total.
La confirmación de exportación total puede registrarse por separado en
`user_declared.exported_all_results`.

Un error HTTP, SSO o captcha se registra como `blocked`, con evidencia revisada
para `access_error`, sin total ni exportaciones. Un bloqueo nunca se convierte
en cero resultados. Un cero solo se acredita como ejecución cuando también se
conocen la consulta y la fecha.

## Generación reproducible

```bash
python scripts/ingest_e1_native_execution.py \
  --manifest /tmp/e1-native-private/execution_manifest.json \
  --output-dir /tmp/e1-native-review
```

El destino debe estar vacío. La orden no busca en internet, publica, empaqueta,
modifica libros ni entrena modelos. Genera `execution.json`, `summary.json`,
`occurrences.csv`, `records.csv`, `deduplication.csv`, este contrato/esquema y un
manifiesto cerrado de hashes. Solo se publican metadatos normalizados y hashes;
no los originales, imágenes, rutas privadas, resúmenes, autores ni contactos.

La deduplicación conserva cada ocurrencia y su `historical_group_id`; el nuevo
`group_id` identifica la unión sin sobrescribir los E1N antiguos. Los conflictos
de DOI quedan separados; el título normalizado solo propone una identidad. Se
comprueban los archivos históricos antes y después de la operación.

`guide_3_1_search_log_complete` significa exclusivamente que se documentaron las
tres bases, sus consultas, fechas y totales con procedencia explícita. Admite el
literal y día confirmados por el usuario y distingue el encabezado transcrito
de IEEE. **No significa que todo §3.1, el cribado exhaustivo o la selección final
estén cerrados.** La hora exacta, las ediciones y el rango ordinal son metadatos
auxiliares; su ausencia no invalida una consulta real ni su total. Las banderas
`all_three_native_metadata_verified` y `all_three_native_exports_complete`
describen esa evidencia auxiliar más exhaustiva, no requisitos añadidos a la guía.
