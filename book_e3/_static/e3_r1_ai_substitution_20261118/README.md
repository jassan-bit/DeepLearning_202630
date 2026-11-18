# Adopción de la revisión de R1 mediante IA declarada

El 18 de noviembre de 2026 el autor comunicó la indicación del profesor: se puede sustituir la revisión humana por IA con la condición de declarar su uso. Este registro aplica esa modalidad a la revisión externa de R1 que se estaba discutiendo. La comunicación se conserva como procedencia de la instrucción; no se atribuye al profesor un dictamen de novedad sobre el modelo.

**El componente de revisión externa se documenta como completado mediante IA declarada. R1 global permanece pendiente de acreditación.** El dictamen técnico disponible no establece todavía una diferencia algorítmica sustantiva. La modalidad del revisor y la conclusión científica son condiciones distintas.

## Evidencias adoptadas

1. El dictamen técnico del commit `e5fcfdd1143037da7e5961f31c3d0d4d7ef9a6d4`, en `experiments/e3_r1_search_review_20261118/ai_review/`, preparado por un agente de IA con contexto separado de los dictámenes anteriores. Contiene comparación de la referencia de 1210 parámetros, cinco lecturas focales de secciones del cuerpo y una fuente consultada sólo por resumen. Su conclusión y sus limitaciones no se cambian al admitir la modalidad IA.
2. La repetición por IA de Q2 y Q5 en la interfaz nativa de arXiv, documentada en `query_check/`: dos cadenas léxicas exactas, dos resultados y 963 resultados respectivamente. Se examinaron seis pares de título/resumen y ningún cuerpo en ese control. La interfaz nativa y la API histórica tienen sintaxis distinta y sus conteos no se comparan como evolución del índice.
3. La declaración específica `ai_use_declaration.md`, visible también en el libro. Identifica la asistencia de IA, las tareas y el alcance de las conclusiones.

`review_adoption.json` vincula esos informes por ruta, commit y SHA-256 y registra la modalidad autorizada, las dos consultas observadas y el estado del componente. `instructor_instruction.json` conserva la fuente y el ámbito de la indicación.

## Alcance y continuidad

No hubo una revisión humana: los registros históricos que lo indican siguen siendo verdaderos y se conservan. La revisión por IA deja de ser un bloqueo de modalidad para este requisito bajo la indicación docente comunicada. No se inventa una persona, firma, aval humano o aprobación académica del mecanismo.

La declaración se refiere al revisor externo de R1. La defensa del autor, sus responsabilidades y declaraciones personales y la reproducción independiente del alcance final conservan sus estados propios. El registro no modifica arquitectura, entrenamiento, pronósticos, contrastes, pesos de rúbrica o nota.

El refinamiento y la actualización final de las búsquedas, las lecturas pertinentes y la justificación de una diferencia sustantiva siguen formando parte del cierre científico. La revisión puede ser favorable, desfavorable o no concluyente; realizarla no garantiza una conclusión favorable.
