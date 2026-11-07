# Evidencia documental del Entregable 1

## Actualización vigente del 7 de noviembre de 2026

La bibliografía canónica reúne **70 referencias**. La [ampliación gratuita TCN/MLP](free_model_expansion_20261107/README.md) documenta 13 obras seleccionadas: tres ya citadas y diez incorporaciones. Sus [autores, DOI y rutas gratuitas](free_model_expansion_20261107/references.csv), [BibTeX](free_model_expansion_20261107/references.bib) y [lecturas críticas](free_model_expansion_20261107/readings.json) distinguen publicaciones, versiones de autor, un preprint y el texto editorial indexado parcial de Vo.

La [síntesis acumulada](homogeneous_expansion_20261107/README.md) incorpora las nuevas identidades al registro anterior, cuenta todas las familias de referencia verificadas y conserva separados los objetivos cripto, la transferencia financiera y los componentes híbridos. La cuota dirigida de siete fuentes TCN y seis MLP no es una frecuencia poblacional. **GRU, LSTM y TCN son la terna experimental; MLP es una alternativa bibliográfica y un antecedente histórico.**

La [bitácora nativa acreditada](native_execution_20261107/README.md) conserva las consultas del 06/11/2026 en Colombia: WoS 271 resultados, Scopus 538 e IEEE 32; se recibieron 271/538/25 registros. Los 620 grupos de esa ronda son candidatos de recuperación, sin afirmación de cribado completo. Esta búsqueda web complementaria no reescribe consultas ni conteos nativos.

Los directorios anteriores conservan sus cortes y manifiestos. El corte homogéneo anterior está en `homogeneous_synthesis_20261107/`; la síntesis vigente se distribuye como una etapa nueva y no modifica sus archivos históricos.

## Corte histórico del 6 de noviembre de 2026

`native_search_20261106/` conserva cuatro exportaciones recibidas (244 ocurrencias, 182 grupos provisionales), deduplicación, primera pasada y cola de revisión. `primary_reviews_20261106/` contiene cinco fichas de acceso: dos originales completos adicionales, dos resúmenes y una lectura primaria parcial. CVI implícita y redes dentro de híbridos GARCH permanecen en estratos distintos de los siete textos del corte del 05/11. Ese complemento reunió 42 referencias; su cantidad no equivale a estudios elegibles. Los manifiestos identifican evidencia y límites, sin declarar un PRISMA global cerrado.

`editorial_screening_20261106/` añade decisiones individuales de título/resumen sobre los grupos nuevos. Distingue exclusiones del núcleo, candidatos que requieren original y límites de fecha o deduplicación. Es una revisión editorial asistida posterior a la ingesta; no reemplaza sus archivos ni certifica inclusión por resumen. `selection_synthesis_20261106/` compara recurrencia, simplicidad y mecanismos de las familias candidatas sin ampliar artificialmente el denominador.

`crypto_extension_20261105/` conserva los siete textos y las frecuencias históricas; `gru_lstm_tcn_selection_20261105/` identifica la terna experimental. Sus resultados, HPO y piloto están en los capítulos 4, 5 y 9 y no cambian con esta ampliación. En aquel corte seguían pendientes la consulta acreditada de IEEE y elementos del historial nativo. La ronda documentada en `native_execution_20261107/` acredita después las tres bases; la recuperación histórica no se afirma.

## Archivo histórico del 3 de noviembre de 2026

Los apartados siguientes describen exclusivamente ese corte. Sus expresiones de estado no sustituyen la actualización anterior.

La versión del 03/11/2026 conserva las consultas originales del 02/11/2026 y añade recuperación dirigida, correcciones de cribado y extracción de los apéndices de S1. No ejecuta redes ni utiliza datos de mercado o test del proyecto.

## Conteos y cortes

Las diez consultas complementarias contienen 18 intentos, 14 páginas recuperadas y 118 ocurrencias, deduplicadas a 106 repositorios. El corte inicial inspeccionó 14 y evaluó cinco textos, con tres incluidos. La revisión corrigió motivos que excedían los metadatos y examinó seis repositorios adicionales: ahora son **20 inspeccionados y 86 sin inspección; siete textos evaluados, cinco incluidos y dos excluidos**. No hubo nuevas cadenas el 03/11/2026.

El corpus exploratorio es S1/S2/S5/S6/S7. S2/S6/S7 son literatura gris y S6 no acredita fecha editorial. La inclusión no certifica la validez de sus resultados: calidad y recuperabilidad se documentan separadamente. El catálogo de **35 referencias**, con 11 aplicaciones, ocho fundamentos neuronales y 16 metodológicos, no constituye 35 estudios elegibles. Los 33 registros anteriores se mantienen intactos en su catálogo histórico.

WoS, Scopus, IEEE y arXiv no se presentan como bases consultadas. Sus conteos son desconocidos, no cero. Tampoco se acredita Q1 por tener DOI, un manuscrito o una copia en GitHub.

## Productos del corte histórico 03/11

- `search_log.json`, `query_summary_20261103.csv`, `discovery_records.json`: cadenas, fechas, páginas, conteos, límites y hashes originales.
- `screening.csv`, `screening_revision_20261103.json`, `screening_recovery_stage_20261103.json` y `flow.svg`: motivos corregidos, nueva recuperación y flujo acumulado con unidades diferenciadas.
- `history_20261102/`, `history_20261103_metadata_revision/` y `screening_20261102_original.csv`: cortes históricos. No se reescribe la fecha de la búsqueda al revisar una decisión.
- `directed_recovery_20261103/`: manifiestos de fuentes nuevas, informes de inspección y transcripción de Tabla 4 de S6. Se conservan enlaces por commit, hashes y fechas; los originales de autores no se redistribuyen aquí.
- `source_manifest.json`: fuentes del corte inicial; `primary_recheck_20261103.json`: comprobación nueva de las tres copias de autores iniciales, sin alterar sus hashes ni sumar estudios.
- `references_access_35.json/csv`: catálogo vigente. `references_access_33.json/csv` conserva el anterior; `additional_references_20261103.json` identifica las dos incorporaciones.
- `studies.json`, `study_synthesis.csv`, `evidence_matrix.csv`: cinco fuentes completas y dos antecedentes con texto pendiente. `reported_metrics.csv` conserva **81 cifras de autores**, 42 anteriores y 39 nuevas, todas con `own_experiment=False`; no son resultados propios.
- `quality_assessment.csv/json`: cinco estudios valorados por dimensiones y evidencia, sin puntuación agregada. Se mantienen contradicciones y riesgos aparentes; no se corrigen suponiendo resultados.
- `family_frequencies.csv`, `model_extraction_review_20261103.csv`, `model_complexity_review_20261103.csv`, `model_frequency_review_20261103.json`: variantes, roles, frecuencias y parámetros analíticos. El script `scripts/audit_literature_models.py --check` reproduce esos productos sin frameworks ni modelos.
- `selection_evidence.json`: selección final pendiente, con fichas candidatas separadas. La frecuencia ampliada es LSTM4/5, GRU2/5, TCN1/5 y encoder2/5; las sensibilidades conservan el corte original, el corpus sin S6 y el corpus sin las fuentes explícitamente grises.
- `fichas_arquitecturas.md/json`: fichas, diagramas y fuentes. S1 sí publica los ganadores de LSTM/GRU/TCN en apéndice C, Tablas5–6. La versión de `keras-tcn`, la especificación parcial de S6 y las configuraciones propias siguen abiertas.

Las fuentes difieren en objetivo, datos, horizonte, escala y protocolo. No se construye un ranking mezclando sus RMSE/R² ni se considera que diversidad, frecuencia aislada o cantidad de citas cierre la selección definitiva de tres modelos básicos. En ese corte los experimentos figuraban pendientes. Las ejecuciones posteriores de E1 se documentan por separado en los capítulos 4, 5 y 9, con su propia cronología.


## Recuperación por internet posterior al cribado del 06/11

El corte de §3.7 reunió 50 entradas de fundamentos, aplicaciones y metodología; no es un recuento de inclusiones. `internet_closure_20261106/` contiene fichas de los 22 casos inicialmente pendientes: cuatro originales completos, una transcripción completa, cinco decisiones estrictas cerradas por exclusión/contexto y 17 pendientes de original o suplemento. Los candidatos E1X001–004 se conservan aparte y no aumentan el denominador histórico N7. `search_recovery_20261106/` registra cuatro intentos de acceso nativo bloqueados y doce consultas propuestas no ejecutadas, además de verificar las cuatro exportaciones recibidas. Los conteos nativos no recuperados son nulos, no cero resultados. La consulta efectiva de IEEE y el historial original siguen pendientes.


## Sustitución de respaldo por acceso gratuito · corte 07/11/2026 UTC

La bibliografía canónica actual tiene 60 entradas, sin equivalencia con el número de estudios incluidos. `open_access_replacement_20261107/` conserva la revisión de los tres PDF suministrados, las alternativas públicas y su matriz por tema/función. La disposición actual de la cola de 22 es ocho decisiones de exclusión/contexto y 14 informes no recuperados al corte, sin borrar los estados de §3.7. No se infiere mala calidad por pago ni se exige una sustitución uno-a-uno. La licencia y el acceso se distinguen de la lectura íntegra y la extracción de tablas. El denominador histórico N7, los experimentos y sus métricas permanecen intactos.


## Bitácora provisional para completar con evidencia nativa

`provisional_native_log_20261107/` publica valores de trabajo conservadores, procedencia y campos de actualización. Las cotas por archivo no equivalen a totales nativos observados ni se suman como estudios únicos. Las fechas son referencias del nombre de archivo o del contexto, y las cadenas se identifican como propuestas. IEEE no tiene una base verificable para estimar su total. Se conservan las cifras exactas de 244 ocurrencias y 182 grupos, sin nuevas consultas acreditadas o cambios experimentales.
