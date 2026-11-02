# Evidencia documental para el Entregable 1

Este directorio incorpora exclusivamente registros bibliográficos y resultados publicados por terceros, conservados de las consultas reales del 2 de noviembre de 2026. No se importaron entrenamientos, predicciones ni métricas propias del trabajo separado para E2. `provenance.json` documenta origen y hashes y distingue la importación de una nueva búsqueda.

## Conteos y unidades

Diez consultas complementarias produjeron 118 ocurrencias, 12 duplicadas y 106 repositorios únicos. Se inspeccionaron 14 repositorios; 92 no se priorizaron tras el cribado de metadatos, lo que no demuestra ausencia de artículos. Se evaluaron cinco textos completos y se incluyeron tres. Los dos textos excluidos tienen un motivo en `screening.csv`. Un repositorio no es un artículo: el flujo identifica ese cambio de unidad. Las treinta referencias generales del libro son un catálogo diferente y no constituyen treinta estudios elegibles adicionales.

WoS, Scopus, IEEE y arXiv no se cuentan como consultados cuando su interfaz quedó bloqueada. Los fallos tienen estado propio, no cero resultados de una búsqueda. El alcance es una revisión complementaria de fuentes accesibles; no acredita exhaustividad ni prevalencia global.

## Productos

- `search_log.json`, `access_checks.json`, `discovery_records.json`, `screening.csv` y `flow.svg`: consultas, acceso, registros, decisiones y flujo.
- `access_checks_20261102_update.json` y `access_routes.csv`: nueva comprobación de conexión, enlaces oficiales, modalidades de acceso y consultas preparadas. No añaden estudios ni consultas ejecutadas al flujo anterior; los fallos del túnel se distinguen de una denegación del proveedor.
- `studies.json`, `study_synthesis.csv` y `evidence_matrix.csv`: cinco fuentes identificadas y matriz de los tres textos incluidos. S3 y S4 son contexto con acceso pendiente y no suman al denominador.
- `reported_metrics.csv`: 42 valores publicados por los autores, con ubicación. El campo `own_experiment` es falso en todas las filas. No se recalculan resultados ni se traducen sus escalas a las del proyecto.
- `quality_assessment.csv` y `.json`: valoración por dimensión con evidencia y limitaciones, sin una puntuación agregada inventada.
- `family_frequencies.csv`: frecuencias de las familias en el corpus de tres textos, con variantes y clasificación lineal explícitas.
- `selection_evidence.json`: regla y compromisos de la selección documental de trabajo. LSTM, TCN y Transformer priorizan diversidad; no representan el ranking de frecuencia pura ni una prueba de menor costo. La selección está acotada al corpus accesible y el respaldo aplicado del Transformer procede de literatura gris.
- `source_manifest.json`: URLs por commit y hashes de 25 fuentes. Los originales de autores no se redistribuyen.

S1, S2 y S5 responden a diferentes objetivos, mercados, escalas y particiones. No debe construirse un ranking cruzando sus RMSE o R² ni presentarse sus valores como resultados propios.
