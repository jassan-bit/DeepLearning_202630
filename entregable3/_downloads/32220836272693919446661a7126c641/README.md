# Paquete para revisión humana independiente de R1

**Estado: preparado; revisión humana no realizada.** Este paquete se preparó con asistencia de IA el 18 de noviembre de 2026. No identifica a una persona revisora, no contiene un concepto emitido y no acredita R1. La guía exige una persona ajena al diseño que repita al menos dos cadenas y anexe su concepto; otro agente de IA no sustituye esa actividad.

## Objeto fijo de la revisión

Se revisa exclusivamente `InnovationModel("additive_positive_head")`, la **CNN causal → LSTM unidireccional → atención por contenido → cabeza positiva aditiva de 1210 parámetros**, del commit `76b2397d99027457b922225cec8a4f0aab1cc04e`. Su salida es `sqrt(A) + y_scale * (1e-4 + softplus(z))` y la pérdida es MSE normalizada. `A` usa los seis retornos conocidos de la ventana; no utiliza el retorno futuro en inferencia.

La propuesta de atención penalizada/saturada de 1211 parámetros y la cabeza de momentos de 1236 son antecedentes de desarrollo, **no el candidato evaluado por este formulario**. Las diez semillas y los 48 contrastes retrospectivos no establecen por sí solos novedad. Este paquete no cambia arquitectura, pérdida, predicciones o datos y no ejecuta entrenamientos.

`code_identity.json` identifica el árbol, las rutas y los objetos Git exactos, junto con las relaciones entre los módulos. Un identificador de blob Git no se presenta como checksum SHA-256 ni como prueba de una nueva ejecución.

## Instrucciones para la persona revisora

1. Registrar identidad académica y relación con el proyecto en `REVIEW_FORM.md`. Declarar si participó en el diseño, la selección del candidato o su implementación. Si hubo participación, describirla para que el evaluador determine la independencia; no autocompletar una declaración favorable.
2. Leer la formulación vigente, el análisis de diferencias de `../mechanism/` y los trabajos más próximos. Revisar las fórmulas y sus supuestos, no sólo títulos o resúmenes. La ausencia de una coincidencia exacta no prueba novedad.
3. Repetir al menos **dos cadenas distintas** de la tabla siguiente mediante interfaces o API nativas. Conservar fecha, base, URL, cadena exacta, conteo, resultados exportados y trabajos examinados. Un bloqueo se registra con conteo desconocido; una respuesta real con cero resultados sólo describe esa consulta.
4. Comparar la contribución con mecanismos de atención y restricciones por muestra, así como con pronóstico de volatilidad. Examinar citas hacia atrás/adelante e implementaciones pertinentes. La revisión general del proyecto requiere al menos cinco cadenas bilingües en tres bases efectivamente consultadas; las dos cadenas de la persona externa no sustituyen toda esa cobertura.
5. Emitir un concepto razonado usando el formulario, indicando antecedentes que anticipen la diferencia, límites del acceso y cualquier incertidumbre. Adjuntar únicamente evidencia bibliográfica redistribuible o sus enlaces; no copiar cuerpos completos protegidos.
6. Entregar el formulario y sus evidencias al autor para incorporarlos con su procedencia y fecha real. La aceptación académica de R1 corresponde al evaluador. El archivo `pending_review.json` permanece pendiente mientras no exista esa revisión efectiva.

## Cadenas históricas verificadas

Las seis cadenas se conservan literalmente del [registro histórico fijado](https://github.com/jassan-bit/DeepLearning_202630/blob/76b2397d99027457b922225cec8a4f0aab1cc04e/experiments/e3_remediacion_20261118/literature/search_log.json). Se recomiendan **Q2 y Q5** por cubrir arquitectura/aplicación y mecanismo de restricción; Q3 es una alternativa en español. Registrar cualquier expansión o sinónimo como consulta adicional, sin sustituir la cadena repetida.

| ID | Idioma | Cadena exacta | Papel |
|---|---|---|---|
| Q1 | inglés | `cryptocurrency volatility` | Contexto y tarea |
| Q2 | inglés | `CNN LSTM attention cryptocurrency` | Repetición recomendada: familia arquitectónica |
| Q3 | español | `volatilidad bitcoin atención` | Tarea y sinónimo en español |
| Q4 | español | `volatilidad piso positivo` | Restricción de salida y tarea |
| Q5 | inglés | `neural network output constraints` | Repetición recomendada: mecanismo transferible |
| Q6 | inglés | `rolling variance conditional forecast` | Objetivo móvil y estructura observable |

## Fuentes próximas que deben contrastarse

| Fuente primaria | Pasaje de referencia | Pregunta para el concepto |
|---|---|---|
| [ConstraintNet, AAAI 2021](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf) | p. 6816, guarda por muestra sobre umbral | ¿Derivar el umbral de σ₇ y usar softplus cambia sustantivamente el mecanismo conocido? |
| [Dasoulas et al., 2021](https://proceedings.mlr.press/v139/dasoulas21a/dasoulas21a.pdf) | §3.3 ec. (2), §5 ec. (17)–(18) | ¿Qué diferencia subsiste frente a pooling softmax y consulta aprendida? |
| [Kim et al., 2021](https://proceedings.mlr.press/v139/kim21i/kim21i.pdf) | §6 | ¿La cota condicional es análisis particular de una propiedad conocida o un aporte distinto? |
| [HARNet](https://arxiv.org/abs/2205.07719) | §4 | ¿Cómo se diferencia incorporar estructura conocida del objetivo frente a un sesgo inductivo de volatilidad existente? |
| [TimeMixer](https://arxiv.org/abs/2405.14616) | §3 y apéndice J | ¿Se documenta una limitación concreta pertinente para esta tarea, además de usar otra arquitectura? |

Las referencias anteriores no prueban copia o influencia histórica. La procedencia real y la semejanza técnica se examinan por separado. La [matriz histórica](https://github.com/jassan-bit/DeepLearning_202630/blob/76b2397d99027457b922225cec8a4f0aab1cc04e/experiments/e3_remediacion_20261118/literature/closest_work_matrix.csv) contiene once trabajos y distingue lecturas focales de acceso sólo a metadatos o resúmenes.

## Archivos del paquete

- `REVIEW_FORM.md`: formulario en blanco para la revisión efectiva.
- `pending_review.json`: estado estructurado sin identidad, resultados ni concepto humanos inventados.
- `review_checklist.csv`: criterios de revisión y evidencia exigible, sin casillas aprobadas.
- `code_identity.json`: candidato y vínculos inmutables a las fuentes.

No se ha contactado a una persona externa ni se ha atribuido una opinión humana a este paquete. La preparación de los documentos no cierra la verificación externa.
