# Recolección bibliográfica R1 en CPU

Asistencia de IA declarada. Este paquete no entrena modelos ni predice mercados.

## Estado real del 18 de noviembre de 2026

`access_probe_receipt.json` conserva cuatro solicitudes API directas y trece sondeos mediante la herramienta web: tres API y diez interfaces públicas nativas, incluidas IDEAS/RePEc, EconPapers, Google Scholar y BASE. Todos fallaron antes de recuperar resultados bibliográficos. IDEAS y EconPapers pertenecen a la misma base RePEc; no se cuentan como dos bases. No se cambió el proxy ni se sortearon restricciones de acceso.

Los conteos de esos accesos son desconocidos, no cero. El recibo conserva los errores observados y distingue la hora de ensamblado de la ausencia de tiempos individuales exactos. La ejecución anterior con resultados nativos de arXiv permanece intacta en `experiments/e3_remediacion_20261118/literature/search_log.json`. Este nuevo sondeo no completa las tres bases ni repite ficticiamente la búsqueda histórica.

## Ejecución preparada para Colab

Usar un entorno normal de **CPU**. `launch_colab.py` espera la variable `R1_SOURCE_COMMIT` con el SHA completo de la versión publicada que contiene estos archivos. Descarga ese código público, verifica el SHA del checkout y ejecuta el recolector con la biblioteca estándar de Python. El cuaderno preparado por el proyecto fija y muestra ese SHA antes de ejecutar.

El recolector utiliza las seis cadenas bilingües ya registradas, sin cambiar sus palabras, en Crossref, DBLP, arXiv, OpenAlex y Semantic Scholar: **30 solicitudes previstas**, a diez registros por solicitud. Conserva la URL efectiva, fecha UTC, estado de transporte, conteo nativo cuando existe, títulos, año, DOI, URL y respuesta nativa con bytes y SHA-256. Las reglas de búsqueda y ordenación difieren entre bases; arXiv conserva la expresión histórica `all:`. La recogida es de la primera página, no un examen exhaustivo de todos los resultados.

Cada base tiene como máximo una solicitud simultánea, separación de 3,3 segundos y tiempo máximo de 15 segundos por solicitud. Un error, límite de servicio o requerimiento de clave se registra sin inventar resultados ni incluir credenciales bibliográficas. No hay reintentos automáticos ilimitados.

Las credenciales GitHub se obtienen del secreto Colab `GITHUB_TOKEN`, de la variable de entorno del mismo nombre o mediante entrada oculta. La publicación necesita `Contents: Read and write` sobre el repositorio. El token se mantiene en memoria, sólo se envía a la API fija de GitHub y no forma parte de URLs, recibos ni código guardado.

La celda guarda el resultado en una rama nueva `e3-r1-search-<run_id>` y en `runs/r1_search/<run_id>/`, con manifiesto de tamaños, SHA-256 y blobs Git. Verifica todos los archivos anteriores del árbol antes de crear esa rama y no modifica `main`. Si el guardado falla, conservar la sesión y reejecutar la misma celda: reutiliza `_E3_R1_SEARCH_CONTEXT` y reintenta guardar las consultas ya realizadas. Una sesión reiniciada pierde ese contexto local; la persistencia se acredita sólo cuando se imprime la URL GitHub verificada.

## Qué queda después de recolectar

`search_receipt.json` informa las bases con seis respuestas analizadas correctamente. `three_base_response_coverage=true` sólo significa cobertura de respuestas; **no acredita cribado, lectura, novedad ni revisión humana**. Un cero nativo válido sigue siendo evidencia limitada a esa consulta y base. Los errores conservan `null`.

`screening.csv` empieza con cada título en `not_screened`. Deben registrarse selección/exclusión y razón, leerse resúmenes y originales pertinentes, localizar las ecuaciones comparadas y seguir citas desde tres a cinco trabajos cercanos. Los duplicados se concilian por DOI/título sin borrar la procedencia de cada consulta. La consulta final debe repetirse antes de entregar. La valoración sustantiva y el concepto de una persona ajena al diseño se documentan aparte. El recolector fija siempre `R1_accredited=false`, `human_external_review_completed=false` y `screening_complete=false`.

## Comprobación local, sin red

```bash
python -B -m unittest discover -s experiments/e3_r1_review_20261118/search -p test_search.py -v
```

Los contratos distinguen cero real de acceso fallido, verifican parsers de cinco bases, integridad de respuestas, secreto y alcance científico, conservación del árbol fuente, reintento idéntico y confirmación perdida al guardar. No constituyen búsquedas bibliográficas ni una defensa humana.
