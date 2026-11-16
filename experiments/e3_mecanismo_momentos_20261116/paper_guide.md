# Relación de la propuesta por momentos con el paper guía

**Fecha de revisión: 16 de noviembre de 2026. Estado: propuesta de mecanismo, sin nuevos ajustes de mercado ni declaración de R1.** Se conserva la base CNN causal → LSTM unidireccional → atención temporal por contenido. La modificación estudiada pertenece a la cabeza de salida y a sus objetivos de aprendizaje; no se presenta como una extensión implementada de TimeMixer.

## Fuentes primarias y alcance de lectura

El paper guía es Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y. y Zhou, J. (2024), *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*, ICLR 2024. Se reabrieron la [ficha arXiv de la versión 2405.14616v1](https://arxiv.org/abs/2405.14616v1), su [HTML](https://arxiv.org/html/2405.14616v1) y el [PDF de esa versión](https://arxiv.org/pdf/2405.14616v1). La ficha registra el 23 de mayo de 2024; la portada del PDF acredita ICLR 2024. El [DOI 10.48550/arXiv.2405.14616](https://doi.org/10.48550/arXiv.2405.14616) identifica el preprint, no un DOI editorial de revista.

La reapertura fue **focal**, sobre la portada, §3, apéndice A y apéndice J; no se declara una nueva lectura íntegra de las 28 páginas. Localizadores: §3.1–3.3 y ecuaciones 1–6, páginas físicas 3–5; tabla 7, página 14; tabla 13, página 17; apéndice J, página 25. El mecanismo PDM/FMM mezcla representaciones de varias escalas; la tabla 7 usa MSE en ETTh1. J señala crecimiento de parámetros con la historia, mezcla sólo temporal y análisis teórico pendiente. Estos puntos no demuestran el conflicto de objetivos de V2 ni un fallo sobre volatilidad de Bitcoin.

El [registro de selección de E2](../../book_e2/_static/paper_guide_evidence/paper_selection.json) conserva la versión leída y su SHA-256 histórico `a599b338e44af70d8e9c87be3c5417bde7864b2c92074e1346703f3e2b641e3d`. Ese hash es el recibo heredado; esta reapertura web no se presenta como una nueva verificación de los bytes descargados. El código de referencia de E2 es `kwuking/TimeMixer`, commit `e24610583b36fdd8c76cc17a8df4e65759a5f460`, distinto de otros ports y de sus defaults.

El acceso a la [ficha OpenReview](https://openreview.net/forum?id=7oLshfEIC2) devolvió una comprobación de navegador. No se atribuye a esa consulta lectura del manuscrito, reseñas ni respuestas de autores. Las afirmaciones anteriores proceden de arXiv y del PDF identificado.

## Qué estableció E2 y qué no se puede trasladar a Bitcoin

La [sección 5 de E2](../../book_e2/sections/05_paper_guia.md), especialmente §§5.3–5.5, separa la réplica externa ETTh1 de la adaptación Bitcoin D7. Los números siguientes son evidencia **heredada**, no resultados generados en esta propuesta.

| Evidencia de E2 | Resultado y alcance | Consecuencia para esta propuesta |
|---|---|---|
| Réplica parcial ETTh1, L96/H96 y siete variables | Cinco semillas; MSE media 0,377838 frente a 0,375 de la tabla 13. Se cumplió el criterio registrado de compatibilidad numérica de E2. | Acredita esa réplica parcial; no acredita rendimiento en BTC ni una limitación de su cabeza de volatilidad. |
| L1, composición del batch con calendario histórico | Cambio absoluto medio B1/B128 de TimeMixer 0,041588, IC95 % temporal [0,035654; 0,048342]. La intervención sin calendario produjo cambio cero con los mismos pesos en las cinco semillas. | Limitación de la implementación fijada y esos marcadores. La adaptación BTC omite calendario: L1 no prueba ese defecto en el predictor BTC. Corregir la alineación es higiene experimental, no novedad suficiente. |
| L2, alta variabilidad observable de entrada en ETTh1 | Diferencia RMSE alta−resto de TimeMixer 0,178216, IC95 % [−0,035298; 0,395833], p Holm 0,1192 con bloque principal 192 h. | Resultado inconcluso: no demuestra fragilidad, equivalencia ni ausencia de efecto. No se transforma en una limitación predictiva BTC demostrada. |

El estrato L2 de ETTh1 usa variabilidad **observable de entrada**. No es el régimen de alta volatilidad **realizada del objetivo** utilizado después para describir V2. Tampoco son intercambiables sus frecuencias, horizontes, escalas o unidades de error. Las sensibilidades de bloque conservadas en E2 no sustituyen el análisis principal por su resultado más favorable.

## Permiso metodológico comprobado en la guía original

Se leyó directamente, mediante extracción de texto por página, el archivo proporcionado `Entregable3DeepLearningProject.pdf`: **60 páginas, 922.729 bytes, SHA-256 `d6048aebd7b3db85a59231a5a89b01bc3f8ad8cb581a96b462e341d21ff1f364`**. Se mantuvo en su ubicación de entrada y no se copia ni redistribuye aquí. Los localizadores siguientes corresponden a sus páginas físicas, coincidentes con la numeración impresa examinada.

- **§6.1, página 8:** extender el paper guía es el origen recomendado. El apartado «Origen alternativo permitido» admite otra limitación observada en un piloto o documentada en otra fuente, con **evidencia equivalente** y discusión del paper guía como trabajo cercano. Si se transfiere un mecanismo, exige fuente precisa y adaptación algorítmica sustantiva.
- **§7.1, D1, páginas 16–17:** pide evidencia y cuantificación de una limitación de métodos existentes; dice **preferiblemente** la del paper guía. No obliga a atribuir a TimeMixer una limitación distinta de las estudiadas.
- **§7.1, D2–D5, páginas 17–18:** exige explicitar la fuente, una hipótesis falsable, predicciones secundarias previas y criterios de éxito/fracaso anteriores a la evaluación final.
- **§7.2, página 18:** exige bitácora de diseños, separación entre desarrollo y confirmación, datasets adicionales independientes y congelamiento real antes del protocolo final.

Por ello se mantiene TimeMixer como guía y referencia multiescala, pero se propone un **origen alternativo** centrado en la cabeza de V2. El permiso de la guía no acredita que la evidencia equivalente o la novedad de esta propuesta ya estén completas.

## Nueva limitación candidata: restricción de forma en V2

V2 utiliza la identidad del objetivo heredado

$$
Y=\sigma_7=\sqrt{A+cU^2},\qquad c=6/49,
$$

con \(A\) conocido en el origen y \(U\) como etiqueta de innovación, nunca como entrada futura. Su familia half-Student con \(\nu=5\) y escala \(s\) vincula, para \(V=U^2\),

$$
m=\mathbb E[V]=\tfrac53s^2,\qquad
q=\mathbb E[V^2]=25s^4=9m^2,
$$

y vincula la media de \(Y\) al mismo \(s\) mediante la media aproximada \(g_{128}(A,s)\). Las esperanzas de este modelo son condicionales a la información de pronóstico. Es una restricción matemática de la familia elegida, no una observación nueva sobre TimeMixer o sobre toda distribución BTC.

La limitación propuesta es que, **bajo mala especificación de esa familia**, la mejor escala para el objetivo puntual y para la NLL auxiliar puede diferir. La pérdida compartida puede imponer un compromiso. El alcance y las condiciones de esa posibilidad deben justificarse formalmente y con verificaciones sintéticas; todavía no identifican la causa del resultado de mercado.

La nueva formulación estudia predicciones separadas de \(m\), \(q\) y una posición \(\rho\) para la media de \(Y\) dentro de cotas dependientes de \(A,m,q\), con objetivos puntuales y de momentos. Retiene el encoder conocido y cambia la regla de salida y aprendizaje. Las cotas clásicas de momentos, Jensen, una salida positiva y las pérdidas cuasi-Gamma no se reclaman aisladamente como invenciones. La diferencia exacta frente a cabezas y pérdidas existentes sigue sometida a revisión de equivalencias.

## Resultado adverso de V2 y límites de la explicación

El [cierre de V2](../e3_cierre_v2_20261116/README.md) y su [resumen inmutable](https://github.com/jassan-bit/DeepLearning_202630/blob/1ca029e869a517cb8de704e573c78ca1e0cbad3c/runs/innovacion_v2/summary.json) conservan una semilla, tres recetas neuronales completas de 40 épocas y 116 fechas de validación de desarrollo reutilizada para seleccionar época. La RMSE de `innovation_density_mean` fue 0,003198296979 frente a 0,003144614784 de `additive_positive_head`: **1,707 % mayor**; no superó la regla exploratoria del 2 % de mejora. También fue mayor que la de `innovation_mean_only`, 0,003173926388.

Ese resultado motiva investigar la restricción, pero **no demuestra que la forma Student o el conflicto de objetivos causaran la diferencia**. No separa por sí solo optimización, representación, variación entre semillas, ponderación de objetivos o especificación de la distribución. Tampoco prueba que MSE sea incapaz de aprender una media condicional: bajo sus condiciones usuales, la media condicional minimiza el riesgo cuadrático. Esta propuesta no busca corregir una supuesta inconsistencia universal de MSE.

Se conserva el resultado adverso y la trazabilidad de V1/V2. No se renombra el control aditivo como original por su menor RMSE, no se borra el esfuerzo previo para reiniciar \(N_d\), ni se usa esta formulación para autorizar más épocas, semillas o ajustes de mercado. Una nueva regla no convierte la validación ya observada en confirmación independiente.

## R1 y R2 siguen sujetos a evidencia

| Condición de la guía original | Exigencia aplicable y estado de esta propuesta |
|---|---|
| **R1, §6, página 7** | Diferencia sustantiva de mecanismo, pérdida o entrenamiento para el mismo fenómeno y tarea. Mantener CNN–LSTM–atención o cambiar activación/datos no la acredita. La novedad específica de la cabeza por momentos permanece pendiente. |
| **Comparación cercana, §6.2, página 10; búsqueda, §6.3, páginas 11–12** | Comparar cinco a diez métodos afines; consultar al menos tres bases, usar al menos cinco cadenas bilingües y buscar también en otras aplicaciones; documentar matriz, fuentes y búsqueda por citas desde tres a cinco cercanos, incluidos TimeMixer y los orígenes matemáticos. No se declara que este documento haya completado esa revisión. |
| **Verificación externa, §6.3, página 12** | Otra persona ajena al diseño debe repetir al menos dos cadenas y anexar su concepto. Una revisión por otro agente de IA no sustituye esa verificación. |
| **R2, páginas 7–8; implementación, §6.5, páginas 13–14** | Implementar el núcleo desde ecuaciones con primitivas del framework, documentar derivadas, pseudocódigo y pruebas, y poder explicarlo/modificarlo en la defensa. La formulación asistida por IA y las verificaciones de software no acreditan por sí solas comprensión y autoría responsable del estudiante. |

La propuesta se declara **propia asistida por IA, con antecedentes y novedad bajo revisión**. Una propiedad formal de la cabeza deberá especificar sus supuestos; no demuestra por sí sola que los momentos estimados sean verdaderos, que la familia describa BTC, que exista una cota de toda la red ni que mejore RMSE. R3 requerirá el protocolo y la familia completa de comparaciones: no se desprende de esta justificación ni de una verificación sintética.
