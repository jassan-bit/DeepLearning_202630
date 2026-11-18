# Revisión focal de novedad de la referencia aditiva

**Resultado del 18 de noviembre de 2026: la novedad R1 no queda acreditada.** Esta revisión aporta evidencia nueva de antecedentes y diferencias, con búsquedas efectivas en arXiv y lectura focal de textos completos. No es una declaración de inexistencia de antecedentes ni un concepto humano independiente. Se preparó con asistencia de IA sobre la referencia identificada en el corte fuente `73d8958c50a69fbea46deb828f03f89385318789`; no se entrenaron modelos ni se cambiaron hipótesis o código del modelo.

## Objeto exacto de la revisión

La receta de 1210 parámetros es CNN causal, LSTM unidireccional y pooling por atención de contenido, con consulta aprendida y salida `sqrt(A) + a_tr*(10^-4 + softplus(z))`. La pérdida es MSE normalizada. No usa la penalización/saturación de v1 ni la cabeza de momentos. `A` procede de los seis retornos conocidos; no se reclama como invención la identidad de varianza que separa la contribución del retorno futuro.

## Resultado por tipo de diferencia

| Elemento | Evidencia primaria y clasificación | Qué puede sostenerse |
|---|---|---|
| Consulta aprendida y pooling softmax | Dasoulas et al., §3.3 ec.(2) y §5 ec.(17)–(18), ya describen score lineal y consulta paramétrica. **Mecanismo conocido.** | La restricción de parámetros de E3 no se convierte por sí sola en otra contribución sustantiva. E3 no implementa LipschitzNorm. |
| Atención sobre estados recurrentes | Luong et al., §3.1 ec.(7)–(8), aporta el antecedente de familia. **Mecanismo conocido y cambio de aplicación.** | El vector paramétrico de E3 difiere de la consulta del decoder; la familia no es inédita. |
| Piso dependiente de la muestra más exceso positivo | ConstraintNet, p.6816, propone explícitamente una salida sobre umbral mediante `exp(z)+b`. **Guarda de salida conocida y parametrización de tarea.** | La sustitución del enlace positivo por softplus y el umbral derivado de sigma7 requieren demostrar adaptación sustantiva; no bastan para afirmar novedad. No se identificó aquí una publicación con la composición exacta de E3, lo que tampoco demuestra su ausencia. |
| Cota con estados latentes acotados | Kim et al., §6 p.8, trata explícitamente el dominio compacto. **Análisis de mecanismo conocido.** | La constante y verificación particulares de E3 son evidencia técnica, no novedad o garantía predictiva por sí solas. |
| CNN-LSTM y volatilidad BTC | Los antecedentes de Peng, Huang y la tesis UCLA cubren familias y tareas cercanas. **Antecedente de arquitectura/aplicación.** | No homologar tendencia, varianza realizada y desviación estándar poblacional centrada. |
| MSE, optimizador y parada | El código y capítulos6–8 identifican procedimientos convencionales. **Entrenamiento conocido.** | No se ha demostrado una pérdida o regla de entrenamiento nueva para la receta aditiva. |

Los textos completos de [Dasoulas et al.](https://proceedings.mlr.press/v139/dasoulas21a/dasoulas21a.pdf), [Luong et al.](https://aclanthology.org/D15-1166.pdf), [ConstraintNet, versión AAAI2021](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf) y [Kim et al.](https://proceedings.mlr.press/v139/kim21i/kim21i.pdf) permiten verificar esos pasajes. Son antecedentes, no prueba de que el autor copiara o transfiriera históricamente esos métodos.

La tesis de [Xiaomeng Li (UCLA,2025)](https://escholarship.org/content/qt073763g4/qt073763g4.pdf), §2.3 y §3.2.4, usa una **suma de siete retornos cuadrados**, con CNN-LSTM: su etiqueta de siete días no la hace idéntica a sigma7 centrada. [HARNet](https://arxiv.org/html/2205.07719v1), §4, aporta convoluciones causales y estructura HAR, sin anticipar por identidad esta cabeza. El [preprint de Alex Y.Li](https://arxiv.org/html/2410.09062v1), §3.1/§3.5, ya aplica TimeMixer a criptomonedas, pero usa desviación móvil21d anualizada. El [TimeMixer original](https://arxiv.org/html/2405.14616v1), §3 y apéndiceJ, no establece que E3 corrija un defecto concreto de sigma7 BTC.

## Búsqueda ejecutada y límites observados

Se ejecutaron seis cadenas distintas en inglés y español, además de cuatro búsquedas por título en la **interfaz nativa de arXiv**. Los encabezados devolvieron respectivamente251,2,0,0,963 y4 resultados para las seis cadenas; estos son conteos observados, con cribado parcial. Los dos ceros corresponden sólo a las cadenas españolas y al índice consultado; no prueban ausencia de antecedentes. Las consultas por título localizaron ConstraintNet, HARNet, Kim y cuatro resultados TimeMixer. La matriz contiene once trabajos; ocho textos completos estuvieron disponibles para lectura focal.

También se ejecutaron **24 solicitudes API nativas**, seis cadenas en cada una de OpenAlex, Crossref, Semantic Scholar y arXiv. Todas fallaron antes de recibir resultados por `Tunnel connection failed: 403 Forbidden`. Los intentos de sus interfaces web no recuperaron resultados salvo arXiv. No se modificó el proxy ni se sortearon sus restricciones. Los conteos bloqueados permanecen desconocidos: **la cobertura efectiva de tres bases no se ha completado**. Las quince consultas de descubrimiento web están registradas por separado y no se contabilizan como búsquedas en esas bases.

**Comprobación de alternativas públicas del mismo día:** se ejecutaron las mismas seis cadenas mediante API nativa de DBLP y API pública de OpenReview: **doce solicitudes adicionales**, también sin respuesta bibliográfica por el mismo bloqueo403. Hubo además once comprobaciones pertinentes mediante la herramienta web sobre sus API e interfaces HTML, incluida la búsqueda pública de IEEE Xplore y una consulta específica de pooling/restricciones de salida en DBLP; ninguna recuperó resultados. Los recibos distinguen el error de transporte directo del mensaje de inaccesibilidad de la herramienta. El total directo es **36 solicitudes API en seis bases**, con cero respuestas bibliográficas accesibles; eso no equivale a cero antecedentes. Sólo arXiv aporta aquí resultados nativos observados. Los sondeos biomédicos no se incluyen como cobertura pertinente ni se usan para completar una cantidad de bases.

Los siete antecedentes iniciales se contrastaron por identidad: cinco cuentan ahora con lectura focal del texto completo; Peng y Wang/CEEMDAN conservaron acceso únicamente a extractos editoriales indexados porque el cuerpo respondió403. El PDF institucional de Huang tampoco estuvo accesible. Se distinguen explícitamente de lecturas completas; no se infieren sus ecuaciones ni resultados no observados. La matriz enlaza sus fuentes originales.

La búsqueda nativa por título TimeMixer localizó además un trabajo posterior sobre volatilidad; se leyó su formulación del objetivo. Ese seguimiento hacia adelante es parcial. Se revisaron relaciones hacia antecedentes en los textos leídos, pero no se completó el seguimiento sistemático hacia atrás y adelante desde tres a cinco fuentes ni una búsqueda exhaustiva de implementaciones y tablas. Esos cierres, la repetición final de búsquedas y la revisión por una persona ajena al diseño permanecen pendientes. No existe una firma humana que atribuir a estos archivos.

## Dictamen delimitado

La evidencia nueva **debilita una pretensión amplia** de novedad basada en CNN-LSTM-atención, consulta aprendida, cota sobre latentes acotados o salida superior a un piso por muestra. La posible contribución queda reducida a la adaptación concreta del soporte de sigma7 y su integración; falta demostrar que excede una aplicación o parametrización conocida y que produce un beneficio atribuible bajo controles comparables. No se selecciona otra arquitectura ni se presenta una hipótesis posterior como preregistro.

El estado correcto es **R1 pendiente de acreditación**, con antecedentes concretos ya documentados. No se declara R1=sí por una búsqueda focal sin coincidencia exacta; tampoco R1=no para toda posible contribución sin completar el examen sustantivo. La defensa del estudiante y la verificación humana externa siguen siendo necesarias. Ninguna conclusión de esta revisión concede R2 o R3.

## Archivos verificables

- `search_log.json` y `search_log.csv`: consultas, modalidad, fecha, URL, conteos conocidos/desconocidos y alcance del cribado.
- `native_api_attempts.json`: recibos reales de las24 solicitudes; `run_native_queries.py` permite repetirlas sin garantizar acceso futuro.
- `alternative_api_attempts.json` y `alternative_access_probes.json`: doce solicitudes DBLP/OpenReview y once comprobaciones web pertinentes; `run_alternative_queries.py` conserva consultas y tratamiento de errores. Los tres sondeos biomédicos descartados se registran aparte, sin crédito de cobertura.
- `closest_work_matrix.csv` y `.md`: once fuentes, localizadores, acceso, diferencias y consecuencias delimitadas.
- `supplemental_observations.json`: observaciones nativas y descubrimiento general separados; `build_review_log.py` recompone el registro integrado sin entrenar.

Estos archivos documentan trabajo realizado y límites efectivos. No son plantillas de requisitos marcadas como cumplidas.

**Revisión de citas:** se usan paráfrasis, títulos bibliográficos y fórmulas breves. No se reproduce ningún pasaje largo de las fuentes; no se anexan sus textos completos. Esta revisión no asigna puntos de rúbrica por el número de consultas o fuentes.
