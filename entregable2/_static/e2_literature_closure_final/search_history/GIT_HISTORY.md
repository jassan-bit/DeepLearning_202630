# Historia recuperada de las búsquedas

**Las dos cadenas WoS ya estaban guardadas en GitHub desde el primer commit de literatura E2. No deben solicitarse otra vez.** Se retiró la limitación shallow del clon y se recuperaron las 52 referencias remotas publicadas, incluidas `entregables`, `gh-pages` y el backup de Pages. El historial alcanzable comprende 9.547 commits. No se modificó ningún archivo del checkout.

## Prueba del expediente inicial

El commit [f8fbbe6af3e91202f2eb6971d60102624ef38446](https://github.com/jassan-bit/DeepLearning_202630/commit/f8fbbe6af3e91202f2eb6971d60102624ef38446) incorporó la revisión E2. Su [capítulo de literatura](https://github.com/jassan-bit/DeepLearning_202630/blob/f8fbbe6af3e91202f2eb6971d60102624ef38446/book_e2/sections/03_literatura.md) contiene Q1 y Q2 completas, la tabla WoS de 302/500/500 registros y Scopus 7.500 recibidos frente a 7.874 mostrados. El déficit de 374 ya se declaraba allí; no lo generó la auditoría actual.

El mismo capítulo explica que los originales se preservaban fuera de los archivos públicos y que la asociación consulta–exportación y los rangos no estaban confirmados. Por tanto, **historia de búsquedas guardada y exportaciones originales versionadas son hechos diferentes**. El manifiesto Scopus inicial conserva nombre, tamaño y SHA-256 del CSV de 7.500, pero señala que el archivo no incluye la cadena nativa completa. Los manifiestos WoS identifican los tres TXT y sus hashes; no acreditan el total de Q2 ni los rangos de cada lote.

`git_query_history_ledger.csv` y `.json` conservan las cadenas recuperadas, campos conocidos/desconocidos, commit y ruta. `initial_E2_query_evidence.txt` preserva el fragmento inicial íntegro de procedencia, sin abstracts. La hora del commit y el campo DA no se presentan como hora de ejecución de consulta.

## Ronda enfocada y arXiv

El commit [5a38a93b0bf41c5a399002474617f090c1bf313f](https://github.com/jassan-bit/DeepLearning_202630/commit/5a38a93b0bf41c5a399002474617f090c1bf313f) añadió la ronda WoS/Scopus/IEEE de 340/828/466 ocurrencias, 1.078 grupos, junto con las cuatro consultas arXiv y sus respuestas paginadas identificadas por hash. Su manifiesto arXiv sí conserva cadenas, totales y recuperación completa: 181/391/605/22 apariciones antes de deduplicar. No es necesario volver a pedir esas cuatro cadenas.

Las cadenas nativas exactas de la ronda enfocada y su correspondencia con totales/rangos no se han recuperado de los documentos revisados. Se conserva el estado desconocido: los conteos recibidos no se transforman en total nativo mostrado.

## Alcance de la recuperación de originales

Los logs de cambios y rutas históricos revisados no localizaron los nombres de los siete exports privados ni los CSV de decisiones 59/42/295/40. Tampoco los textos de procedencia iniciales afirman haberlos versionado: explicitan la separación privada. La enumeración general de objetos se detuvo al agotar el alcance temporal de esta revisión: el clon parcial estaba recuperando árboles históricos uno a uno y ese recorrido no constituye una búsqueda del contenido de archivos comprimidos. El examen paralelo de esos archivos debe concluir antes de afirmar ausencia de cualquier copia renombrada. **No se infiere inexistencia del material a partir del working tree ni se presenta la enumeración parcial como búsqueda exhaustiva de contenido.**

`remote_refs_checked.txt` fija los hashes de las ramas consultadas. `git_objects_paths.txt` es un registro técnico de enumeración, no una prueba de lectura de resúmenes ni de elegibilidad. Los manifiestos, el capítulo y los metadatos recuperados permiten reutilizar las búsquedas originales sin repetirlas; la disponibilidad de los cuerpos privados se mantiene separada.
