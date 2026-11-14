# Borrador de reconciliación del corpus recibido

Estado operativo: cerrado al snapshot.

11.635 ocurrencias − 1.720 duplicados internos − 356 enlaces = 9.559 grupos cribados. Seis dirigidos quedan aparte; faltan 374 ocurrencias Scopus mostradas y no recibidas; no están confirmados todos los totales/rangos nativos WoS. No se declara cobertura histórica completa ni preregistro bibliográfico.

El CSV conserva propuestas de cribado y dictamen final en columnas distintas. Se reutilizan 30 decisiones focales previas: 25 evaluaciones de secciones primarias localizadas (17 incluidas, 6 excluidas, 2 contextos) y 5 exclusiones por título/resumen. Una evaluación primaria focal no afirma lectura íntegra de cada PDF. Los contenedores y alias no añaden estudios. Un informe no recuperado no es exclusión; la incertidumbre científica tras lectura primaria tampoco supone una tarea técnica nueva. Los candidatos contextuales por título/resumen no son fuentes primarias incluidas.

full_reports_compact.jsonl limita la razón propia y los 12 valores a 70 palabras por fila. Los textos con … están recortados para publicación; las notas completas quedan privadas, con hash de trazabilidad. Los estados y localizadores se preservan. No reportado, no verificado y no indexado son categorías diferentes.

Los conteos de estudios usan identificadores exactos y alias explícitos, sin unión difusa. El flujo conserva grupos históricos; por eso grupos, informes y estudios tienen denominadores distintos. Las versiones mantienen sus extracciones separadas; se prefiere la ficha editorial focal previa para síntesis, sin importar fórmulas del preprint. No se permite pooling numérico.

`study_id` es una clave sustituta determinista para un componente de identidad bibliográfica; 9.555 componentes no equivalen a 9.555 estudios empíricos verificados. La inclusión aplicada se adjudica aparte. `quality_applied_studies.csv` contiene 504 filas: doce dimensiones para los 42 estudios incluidos, usando una sola versión preferida por estudio. `quality_12_dimensions.csv` abarca los 306 dictámenes generales, con otro denominador.

Los campos previos usan `summary` como alternativa a `value` y enlazan cada dimensión al índice público de calidad de 17 estudios. Si un recorte pudiera eliminar una negación o limitación, se publica una referencia abreviada neutral a la fuente, sin una afirmación positiva recortada. `final_reason_ref=screening_reason_brief` indica reutilización exacta de esa razón y evita duplicarla.

La adjudicación primaria adicional de Sentiment-IGARCH conserva target, orden temporal y vínculo neural identificados, pero lo reclasifica como contexto: la evidencia examinada verifica estimación/significación y trayectoria de varianza, sin establecer evaluación empírica del forecast de dispersión. La comparación de retornos no sustituye esa evaluación; no se añade OOS como regla. La inclusión de Lehrer/Xie/Yi se conserva por su evaluación predictiva de BTC-RV. El historial de los 52 dictámenes originales permanece íntegro y ambas verificaciones están registradas por separado.
