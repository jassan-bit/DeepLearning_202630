# Revisión focal de la cabeza de momentos para CNN–LSTM con atención

Fecha: 16 de noviembre de 2026. Alcance: 12 consultas web reales, lectura focal de fuentes primarias y contraste conceptual. No es una revisión sistemática ni acredita las búsquedas nativas en tres bases, la búsqueda repetida antes de entrega o la verificación humana independiente exigidas para R1. No se entrenó, importó ni copió código/modelos de terceros.

**Dictamen provisional: R1 pendiente.** La candidata es una adaptación algorítmica en estudio de ingredientes conocidos. La desigualdad inferior tiene un antecedente matemático directo; tampoco son inéditos los scores de media, las cabezas de momentos, la propagación de momentos ni las guardas de salida. No se identificó la composición exacta en esta búsqueda focal. Esa observación no demuestra su ausencia en la literatura ni basta para declarar una contribución sustantiva. La propuesta no queda aprobada como modelo original ni como modelo superior.

## Objeto exacto revisado

Se conserva CNN causal → LSTM unidireccional → atención temporal. La identidad conocida del objetivo es `Y = sqrt(A + c V)`, con `c=6/49`, `A` calculado a partir de seis retornos observados y `V=U²≥0`. La cabeza estima `m=E[V|X]`, `q=E[V²|X]` y una coordenada `rho` de la media transformada:

\[
L=\sqrt A+\frac{cm}{\sqrt{A+cq/m}+\sqrt A},\quad
H=\sqrt{A+cm},\quad \mu=L+\rho(H-L).
\]

Para `m>0` y `q>m²`, la región exacta de medias realizables es `[L,H)`: el mínimo se alcanza con `V` en `{0,q/m}`, mientras que `H` es un supremo no alcanzado. Con `q=m²`, `V=m` casi seguramente y `L=H`. La parametrización propuesta `q=m²(1+b²)` admite igualdad y `rho=d²/(1+d²)` incluye cero y excluye uno. Inicializar `b≠0` evita iniciar exactamente en el punto de derivada nula del exceso de segundo momento. Estas parametrizaciones son decisiones de diseño; no se reivindican como nuevos principios matemáticos ni aseguran convergencia.

El criterio poblacional propuesto suma error cuadrático de `Y` y los scores `lambda_m(log m+V/m)` y `lambda_q(log q+V²/q)`, con pesos positivos. No supone una familia Student ni produce una densidad o un intervalo de predicción.

## Matriz de antecedentes cercanos

La columna de diferencia es una comparación con la propuesta escrita, no una afirmación de que todo el artículo haya sido leído o de que nadie haya publicado esa combinación.

| Fuente primaria y localizador leído | Parte ya establecida | Diferencia que debería justificarse | Consecuencia para la novedad |
| --- | --- | --- | --- |
| [Merhav (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10288939/), §2, ecs. (3)–(8); ejemplo 2, ec. (15) | Jensen con distribución reponderada proporciona cotas por momentos. Su ec. (8) produce exactamente `L` al especializar las funciones. | Emplear esa cota como parte de una cabeza condicional entrenada para `σ7`, con momentos aprendidos y posición de la media. | Antecedente directo de la desigualdad: **no presentarla como nueva**. |
| [Gneiting (preprint v2 de 2010; artículo de 2011)](https://arxiv.org/html/0912.0902), §3.1, teorema 3.1 y ecs. (18), (20) | Los scores Bregman elicitan la media; la ec. (20), `b=0`, contiene QLIKE. `log m+V/m` difiere por un término de observación. | Dos scores auxiliares para momentos de una innovación y un score del objetivo transformado con restricción conjunta. | No atribuir novedad al score, a la elicitación de momentos ni a la consistencia de la media por sí solos. |
| [Stirn et al. (2023), Faithful](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf), §3, ec. (5), teorema 1; discusión de Student en p. 3 | La NLL puede deteriorar la estimación de la media. Reescala el gradiente de media y bloquea gradientes de covarianza al tronco; el objetivo equivalente combina MSE y NLL con `stop-gradient`. | La propuesta restringe la media transformada mediante dos momentos, sin especificar una densidad. Su consistencia funcional no garantiza los mismos pasos de entrenamiento que una red de media sola. | **Control obligatorio del diseño futuro.** No afirmar ser el primer método que protege la media frente a una pérdida probabilística ni heredar su garantía de optimización. |
| [Goulet Coulombe, Frenette y Klieber (2024, HNN v2)](https://arxiv.org/html/2311.16333), §2, ingredientes 1–4, en particular restricción de énfasis de volatilidad | Una red con tronco común y hemisferios de media/varianza aprende momentos condicionales; una restricción y validación por bloques atienden su indeterminación y sobreajuste. | Aquí los momentos son los de `V`, y la salida es la media de una transformación conocida, acotada sin ley de innovaciones. | Las cabezas conjuntas y las restricciones para momentos ya tienen antecedentes cercanos en pronóstico de volatilidad. |
| [Ebeigbe et al. (2021, GenUT v2)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8043458/), §§II–III y V; metadatos de versión contrastados con [arXiv](https://arxiv.org/abs/2104.01958) | Puntos sigma propagan momentos por funciones no lineales, capturan momentos superiores y admiten restricciones de soporte. | Se busca una envolvente exacta de la media por dos momentos y una coordenada aprendida, en lugar de aproximar una expectativa mediante puntos sigma. | No es nuevo usar momentos para una transformación no lineal. GenUT es comparador conceptual; no certifica nuestras cotas. |
| [Li, Ling y Ye (2026)](https://www.mdpi.com/1911-8074/19/9/703), §5.1, ecs. (31)–(32) y descripción de tres modelos | Modelos de ML pronostican momentos y los convierten en distribución/riesgo mediante un mapa cerrado de una familia half-logistic. | Su mapa es específico de una familia y usa modelos de boosting; nuestra cabeza conserva CNN–LSTM–atención y no convierte momentos en una densidad ni reclama VaR/ES. | La idea amplia de aprender momentos y decodificarlos analíticamente para riesgo financiero ya está publicada. |
| [Brosowsky et al. (2021), ConstraintNet](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf), pp. 6814–6816, ecs. (4)–(5), (7); guarda sobre umbral en p. 6816 | Una guarda parametriza por muestra la región de salida, incluidos intervalos y pisos. El artículo explica límites de accesibilidad de fronteras de softmax. | La región depende de la identidad del objetivo y de dos momentos aprendidos, cuya exactitud no se conoce en aplicación. | Una guarda de intervalo dependiente de la entrada no constituye novedad suficiente. |
| [Bertsimas y Popescu (2005)](https://web.mit.edu/~dbertsim/www/papers/ioana/Optimal-inequalities-in-probability-theory-A-convex-optimization-approach-SIAM15.pdf), §2, problemas (2.1)–(2.2), teorema 2.1, pp. 784–785 | Optimización sobre leyes con momentos prescritos y dualidad para cotas óptimas. Distingue cotas alcanzadas de aproximables. | Esta especialización tiene forma cerrada para raíz cuadrada y se propone integrarla en una red. | No atribuir novedad al marco de cotas óptimas de momentos. No se verificó allí nuestra fórmula literal. |
| [Ben-Tal y Hochman (1972)](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/more-bounds-on-the-expectation-of-a-convex-function-of-a-random-variable/198DE610197C1E94644B4FB4A058E5B8), metadatos y resumen | Antecedente histórico de refinamientos de expectativas; el resumen usa desviación absoluta media. | No se leyó el cuerpo ni se verificó la fórmula `m,q`. | Contexto histórico, no evidencia de que este artículo contenga nuestra cota exacta. |

## Atribución matemática directa de la cota inferior

Merhav establece en su ec. (8), para `f` convexa y `g≥0`,

\[
E[f(V)g(V)]\ge E[g(V)] f\!\left(\frac{E[Vg(V)]}{E[g(V)]}\right).
\]

Tomando `g(v)=v` y `f(v)=c/(sqrt(A+cv)+sqrt(A))`, convexa para `A>0`, se obtiene `E[Y]≥L`. La convexidad también se ve en `f(v)=integral_0^c [2 sqrt(A+tv)]^{-1} dt`. Para `A=0`, el ejemplo 2, ec. (15), con `s=1/2`, contiene `E sqrt(V)≥m^(3/2)/sqrt(q)`. El argumento condicionado por `X` y la verificación de igualdad son una especialización de principios conocidos. Pueden demostrarse para el entregable y verificarse numéricamente, atribuyendo su procedencia; demostrar una propiedad no implica haber inventado esa propiedad.

## Qué limitación sostiene la adaptación

La cabeza V2 de escala half-t5 fija relaciones entre `m`, `q` y la media de `Y`; en particular `q=9m²`. Bajo mala especificación, una misma escala puede tener óptimos incompatibles para NLL y MSE del objetivo transformado. Un contraejemplo poblacional puede establecer esa posibilidad, pero el RMSE peor de un piloto no identifica la causa y la escala libre sí puede representar muchas medias. No debe afirmarse que Student carece universalmente de capacidad para la media ni que ya se diagnosticó la distribución real de BTC. El problema general de interferencia de NLL con la media ya está documentado por Faithful; la limitación específica que se estudia es el acoplamiento de momentos de la escala V2.

El posible cambio sustantivo es sustituir esa restricción de familia por momentos auxiliares consistentes y una media transformada situada en su región admisible. Para defenderlo falta demostrar qué aporta ese acoplamiento respecto a opciones más simples y a Faithful. No basta cambiar de distribución, añadir cabezas o combinar una cota conocida con una red.

## Límites de las garantías

La consistencia condicional es una afirmación de minimización de riesgo poblacional en una clase capaz de representar los momentos y la media verdaderos, con `m0>0` y `E[U⁴|X]<∞`. El score del cuarto momento requiere esa esperanza; no requiere `E[U⁸|X]` para definirla, como sí haría el error cuadrático de `U⁴`. Esto no garantiza varianza finita del gradiente, estabilidad numérica, convergencia ni superioridad finita de muestra. Para integrar riesgos sobre `X` deben explicitarse además las condiciones de integrabilidad. Si `m0=0`, el caso degenerado necesita una rama separada o una exclusión explícita: el score logarítmico no admite reportar cero y puede tender a menos infinito.

La envolvente certifica la media poblacional **con los momentos verdaderos**. Sustituirlos por estimaciones neuronales garantiza pertenencia a una región calculada, no que esa región contenga la media real. `[L,H)` no es un intervalo de predicción, confianza o cobertura, ni justifica probabilidades de eventos futuros. La consistencia funcional propuesta tampoco es la fidelidad de optimización de Faithful: los scores auxiliares pueden modificar el tronco compartido y sus actualizaciones.

## Comparación futura necesaria y puertas abiertas

Sin entrenar ni modificar la familia confirmatoria congelada de E2, el diseño de desarrollo debe incluir como controles la misma CNN–LSTM–atención con media directa MSE, los mismos scores auxiliares sin envolvente, el plug-in superior de Jensen, un objetivo Faithful adaptado de forma explícita y la escala V2. Faithful es obligatorio para la motivación de desacoplar media y objetivos auxiliares. Las ablaciones deben aislar el momento `q`, la envolvente y el entrenamiento auxiliar, con capacidad y presupuesto comparables. Estos controles futuros no se convierten automáticamente en nuevos rivales confirmatorios ni autorizan reajustar los benchmarks de E2.

La matriz debe ampliarse con búsquedas nativas reproducibles en las tres bases exigidas, seguimiento de citas de Merhav/Faithful/ConstraintNet y comprobación humana independiente de las diferencias. Si aparece un antecedente con la misma cabeza, objetivo y uso de momentos que impida una adaptación sustantiva, deberá reclasificarse o descartarse la pretensión de originalidad. Hasta completar esa revisión y la evidencia correspondiente: `novelty_verified=false`, R1 pendiente y superioridad no verificada.
