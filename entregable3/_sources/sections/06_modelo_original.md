# 6 Bloque C El modelo original

**Estado actual de R1:** {ref}`6.18 <e3-r1-substantive-verdict-20261120>` aporta el dictamen técnico por IA favorable y acotado para la adaptación V3. La referencia aditiva sigue sin acreditar diferencia sustantiva; los resultados de ambos objetos se mantienen separados.

La referencia evaluada es **CNN causal → LSTM unidireccional → atención temporal por contenido, con salida aditiva positiva**, de **1210 parámetros**. Sus diez semillas congeladas obtuvieron una RMSE media individual de **0,0029394324** sobre las 608 fechas históricas comunes. La [comparación estadística posterior](08_verificacion_estadistica.md) ya está calculada frente a los 16 rivales de E2; sus resultados no acreditan por sí solos R1, R2 o R3.

El {ref}`cierre de 8.18 <e3-final-extended-audit>` añade una auditoría exploratoria común de 19 sistemas en D00–D03, con 796 trabajos completos, análisis y figuras. Sus resultados se distinguen de las diez réplicas históricas y no acreditan por sí solos R1/R2/R3.

Los pilotos v1, v2 y por momentos se conservan como historia de desarrollo. La RMSE **0,0031446147843954175** corresponde a la validación de una semilla aditiva de v2, no a la media de prueba de las diez réplicas. La receta half-t5 con MSE y NLL y el posterior piloto por momentos no superaron sus heurísticas de continuación. La cabeza de momentos no es la arquitectura de la comparación actual ni se ha seleccionado como aporte original. La sección 6.5 conserva la formulación del primer mecanismo y 6.7 la propuesta por momentos; 6.8 identifica la referencia efectivamente evaluada.

Las evidencias completas se encuentran en los anexos: [revisión del piloto v1, B](17_revision_post_piloto.md), [ficha v2, C](18_ficha_cnn_lstm_atencion_v2.md), [ejecución v2, D](19_piloto_innovacion_v2.md), [cierre v2, E](20_cierre_v2.md), [formulación por momentos, F](21_propuesta_momentos.md) y [dictamen de admisibilidad, G](22_dictamen_momentos.md).

## 6.1 Relación con el paper guía de E2

La continuidad elegida conserva D7×5 OHLCV, horizonte de un día y objetivo $\sigma_7$ poblacional sin anualizar. El primer mecanismo v1 añadió penalización de discordancia respecto a un promedio causal y saturación radial de los valores agregados. Ese diseño tiene formulación, código y una propiedad latente demostrada, pero su piloto no mejoró al control M00.

El paper guía sigue siendo TimeMixer. Su limitación de calendario medida en ETTh1 no se traslada automáticamente al adaptador BTC sin esos marcadores. El estrés artificial de E2 motivó investigar sensibilidad, pero puede romper coherencia OHLCV y no prueba que la regla nueva resuelva un defecto económico real. La [sección 7](07_diseno_riguroso.md) distingue esa hipótesis inicial de los resultados posteriores y del diseño aún pendiente de confirmación.

La propuesta por momentos de la sección 6.7 estudia otra limitación: una escala paramétrica compartida puede tener óptimos distintos para la NLL auxiliar y para la media de la transformación objetivo bajo mala especificación. Los contraejemplos artificiales demuestran esa posibilidad, sin atribuirla como causa ya identificada del resultado de BTC ni como un defecto nuevo de TimeMixer.

## 6.2 Condiciones de originalidad y admisibilidad

| Condición | Evidencia actual | Estado que debe conservarse |
|---|---|---|
| R1 Novedad para el fenómeno y la tarea | La {ref}`comparación de cierre de 6.15 <e3-r1-comparison-close>` examina diez trabajos cercanos de la referencia aditiva y distingue antecedentes, accesos y diferencias. La {ref}`decisión de 6.16 <e3-r1-adaptation-decision>` conserva V3 como adaptación candidata de estructura y entrenamiento. | La aditiva no acredita diferencia sustantiva; V3 tiene un dictamen técnico favorable y acotado en 6.18. La utilidad empírica es otra pregunta: el piloto completo falló su heurística. La revisión no acredita R2/R3, una nota o la aprobación del docente. |
| R2 Núcleo algorítmico propio y defendible | El modelo aditivo evaluado se construye con los módulos locales `InnovationModel`, `DevelopmentModel` y `OriginalModel`, usando primitivas de PyTorch. Hay pruebas específicas de su salida y de los gradientes, además del código histórico de las otras variantes; la asistencia de IA está declarada. | La implementación y sus pruebas no acreditan por sí solas comprensión, autoría responsable y defensa del estudiante; esa verificación humana sigue pendiente. |
| R3 Superioridad estadísticamente verificada | Pilotos de desarrollo y comparación posterior completa frente a 16 rivales: los 48 contrastes primarios rechazan tras Holm, con ámbitos temporal y de inicialización distintos. | No se acredita R3. El corte histórico conserva calendario conocido y tratamientos distintos; el {ref}`cierre de 8.18 <e3-final-extended-audit>` añade comparación común en cuatro conjuntos y la menor RMSE media descriptiva del candidato en los cuatro. Esa auditoría no calcula la nueva batería de tres pruebas con Holm ni certifica potencia prospectiva o independencia entre mercados. |

CNN, LSTM, atención por contenido, consultas aprendidas y saturación no se reclaman como invenciones. Tampoco se reclaman nuevas las desigualdades de momentos, los scores de media o las guardas de salida. La contribución candidata debe diferir sustantivamente en mecanismo, pérdida o entrenamiento; cambiar datos o hiperparámetros no basta. Las variantes funcionales y los resultados de desarrollo se conservan sin convertirlos en innovaciones acreditadas.

## 6.3 Revisión de literatura y evidencia de novedad


**Antecedentes del primer mecanismo de atención.** En el corte inicial se examinaron focalmente los originales de [Kim, Papamakarios y Mnih (2021)](https://proceedings.mlr.press/v139/kim21i.html) y [Dasoulas, Scaman y Virmaux (2021)](https://proceedings.mlr.press/v139/dasoulas21a.html). Ambos son antecedentes directos de atención y propiedades Lipschitz. Una consulta aprendida y una cota, por separado, ya tienen antecedentes. La matriz siguiente conserva el alcance histórico de ese primer corte; la ampliación del **18 de noviembre**, con ocho lecturas focales de texto completo, se documenta en **6.10**.

En ese corte, las seis cadenas bilingües generaron 18 solicitudes a OpenAlex, Crossref y arXiv. **Las 18 fueron denegadas por la política de red antes de ejecutar las búsquedas.** El número de resultados es desconocido, no cero. Las ocho consultas GitHub y el examen de tres exportaciones heredadas no sustituyeron la búsqueda exigida en tres bases. Tampoco se presenta un subagente de IA como la persona externa requerida.

| Trabajo cercano | Relación que debe distinguirse |
|---|---|
| Kim et al., *The Lipschitz Constant of Self-Attention* | Atención L2 entre pares y estabilidad: no toda atención se vuelve una contribución nueva por tener cota. |
| Dasoulas et al., *Lipschitz Normalization* | Consulta aprendida y normalización de scores; el candidato acota valores y añade discordancia, pero su novedad exacta está por verificar. |
| TimeMixer | El procesamiento multiescala ya tiene antecedentes. La diferencia debe ser de mecanismo. |
| Luong et al., atención en traducción | La atención por contenido es un componente convencional. |
| Peng et al., CNN-LSTM con atención para tendencia cripto | La familia híbrida y su uso en criptomonedas no se pueden presentar como inéditos. Su objetivo no se equipara a σ₇ sin revisar el original. |
| CEEMDAN-RF-LSTM para volatilidad realizada | Separar mecanismo y definición del objetivo; no comparar métricas de tareas distintas como si fueran homogéneas. |
| HARNet | La causalidad y los inductive biases para volatilidad ya tienen precedentes. |

La saturación $T_B$ es también el gradiente de $\phi_B(h)=B^2(\sqrt{1+\|h\|^2/B^2}-1)$, de forma pseudo-Huber radial. Esta identidad impide presentarla aislada como una función inventada aquí. No se atribuye una fuente histórica primaria no verificada. El [registro completo](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_bloque_c_20261115/literature) distingue lecturas nuevas, heredadas y sólo metadatos. **No se emite aún una declaración de novedad.**

**Revisión posterior de la cabeza de momentos.** Los [anexos F](21_propuesta_momentos.md) y [G](22_dictamen_momentos.md) integran los antecedentes y el dictamen específicos de esa adaptación:

| Fuente y localizador comprobado | Consecuencia para la propuesta |
|---|---|
| [Merhav (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10288939/), §2, ec. (8), ejemplo 2 ec. (15). | La cota inferior se especializa de Jensen reponderado; no es una desigualdad inventada aquí. |
| [Gneiting (2011; preprint v2)](https://arxiv.org/html/0912.0902v2), §3.1, teorema 3.1 y ec. (20). | Los scores de media de forma QLIKE/Bregman tienen antecedentes. |
| [ConstraintNet, Brosowsky et al. (2021)](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf), pp. 6813–6817, ec. (7) en p. 6815. | Una guarda admite restricciones por muestra, incluso de otra red; aprender las fronteras no basta como diferencia sustantiva. La guarda sobre umbral está en p. 6816 y la aplicación a intervalos en p. 6817. |
| [Faithful, Stirn et al. (2023)](https://proceedings.mlr.press/v206/stirn23a/stirn23a.pdf), **§2 Methods**, páginas físicas 2–4; ec. (5) y teorema 1 en p. 4. | Su garantía de actualizaciones no se hereda por consistencia del riesgo. El control de media sola conserva todos los ancestros de la media, incluidas las lecturas de momentos de nuestra cabeza. |

Las revisiones posteriores registran doce consultas web iniciales y ocho adicionales; no equivalen a búsquedas nativas completas. Los tres intentos API de esa etapa no recibieron respuesta bibliográfica y sus conteos son desconocidos. Al cierre de esa etapa seguían pendientes la cobertura exigida en §6.3 de la guía, seguimiento por citas, repetición antes de entrega y verificación externa. Los avances posteriores y la modalidad de IA declarada se documentan en 6.12–6.13. El [paquete de revisión del anexo G](22_dictamen_momentos.md) está preparado, sin diligenciar ni firmar. No identificar la composición exacta en una revisión focal no prueba su ausencia.

El cierre de la revisión exige al menos cinco cadenas en inglés y español en **tres bases**, combinando mecanismo, tarea y sinónimos e incluyendo su posible origen en otras aplicaciones. La ampliación de 6.10 sólo obtuvo resultados nativos observables en arXiv; ese corte histórico no completó las tres bases. La ejecución Colab posterior de **6.12** sí añade respuestas verificadas de Crossref, arXiv y OpenAlex, con límites de sintaxis y cribado explícitos. Cada consulta debe registrar fecha, cadena, resultados, títulos/resúmenes examinados, textos completos revisados y trabajos relevantes; un acceso bloqueado se registra con conteo desconocido. Debe completarse el seguimiento de citas hacia atrás y adelante desde tres a cinco trabajos cercanos —incluidos TimeMixer y la fuente del mecanismo, cuando corresponda— y la búsqueda de implementaciones y tablas de resultados. La matriz debe clasificar cada diferencia como mecanismo, pérdida/entrenamiento o sólo aplicación, y repetirse antes de la entrega. La guía original prevé que **una persona ajena al diseño repita al menos dos cadenas y anexe su concepto**. Para este proyecto se aplica la indicación docente comunicada por el autor: ese revisor puede sustituirse por IA si se declara su uso. El dictamen y las dos repeticiones por IA se documentan en 6.13; no se presentan como una revisión humana.

## 6.4 Ciclo de desarrollo

| Etapa | Evidencia disponible | Condición siguiente |
|---|---|---|
| Revisión y hueco | Primer mecanismo de atención y propuesta posterior por momentos, con antecedentes y límites documentados. | Completar revisión y verificación externa antes de una declaración de novedad. |
| Formulación | Cota latente del primer módulo y teoría condicional de momentos con sus supuestos; contraejemplos a garantías no demostradas. | Justificar una contribución sustantiva y su alcance sin convertir la prueba en promesa de desempeño. |
| Implementación y desarrollo | Cinco recetas entrenadas entre v1/v2 y tres recetas completadas en el piloto por momentos; diez réplicas posteriores de la receta aditiva congelada. | Repetir semillas de una receta no equivale a proponer diseños nuevos ni completa la defensa académica. |
| Pruebas y auditoría | Pruebas de software y auditorías reproducibles, incluidas las comprobaciones artificiales del nuevo modelo, reanudación y guardado. Las 35 pruebas del primer módulo conservan su alcance histórico. | El software y los cálculos artificiales no sustituyen evaluación confirmatoria o autoría defendida; la revisión externa por IA de 6.13 es un producto distinto. |
| Selección de desarrollo | La cabeza aditiva v2 obtiene la menor RMSE global del piloto; half-t5 con auxiliares y el piloto por momentos no superan sus heurísticas. La aditiva se replica sin nueva selección. | Conservarla como referencia, sin promoverla a aporte original o ganador único. |
| Diseño y confirmación | Resultados históricos, diez réplicas aditivas y análisis posterior completo de la familia conservados; no hay preregistro confirmatorio de estos resultados. | La comparación común exploratoria y los tres conjuntos adicionales ya tienen resultados en el {ref}`cierre de 8.18 <e3-final-extended-audit>`; siguen pendientes potencia prospectiva, batería formal y diseño confirmatorio apropiado. |

La [sección 7](07_diseno_riguroso.md) integra método, pilotos, propiedades y controles. Las cinco recetas de v1/v2 y las tres del piloto por momentos forman un subtotal histórico de ocho; las diez semillas aditivas repiten una receta. La {ref}`conciliación de 7.14 <e3-design-inventory-reconciled>` identifica 15 recetas neuronales implementadas y separa controles, propuestas, configuraciones y réplicas dentro del corpus registrado; el subtotal histórico de ocho no se usa como Nd global. D01–D03 ya fueron seleccionados, admitidos y evaluados junto con D00 en el {ref}`cierre de 8.18 <e3-final-extended-audit>`. Sus resultados conocidos no constituyen confirmación ciega de una adaptación posterior.

## 6.5 Primer mecanismo implementado: formulación y reproducción

Esta sección conserva la formulación histórica de penalización y saturación del primer módulo. Sus constantes, conteo de parámetros, pruebas y descargas pertenecen a ese código; la implementación posterior de momentos se distingue en 6.7 y la receta de referencia v2 en el apartado 7.5.

El encoder usa convolución con relleno sólo histórico, activación tanh y LSTM unidireccional. Recibe ventanas ya estandarizadas y produce estados $H=(h_1,\ldots,h_7)$. Una matriz fija $P$, independiente de $H$, calcula promedios causales. La cabeza implementa

$$
d_j=\sqrt{\|h_j-(PH)_j\|_2^2+\varepsilon^2},\qquad
a=\operatorname{softmax}\bigl((Hq-\lambda d)/\tau\bigr),
$$ (eq-e3-v1-discordance-attention)

$$
v_j=\frac{Bh_j}{\sqrt{B^2+\|h_j\|_2^2}},\qquad
f(H)=\operatorname{softplus}\left(w^\top\sum_j a_jv_j+b\right).
$$ (eq-e3-v1-bounded-readout)

Los parámetros efectivos satisfacen cotas mediante normalizaciones suaves; no se presume que cualquier peso libre esté acotado sin esa parametrización. La salida es positiva en unidades $\sigma_7/a_{\rm tr}$ y el pipeline invierte la escala del objetivo fuera del módulo. Los detalles, derivadas y algoritmo completos están en la demostración descargable y en la [sección 7.3](07_diseno_riguroso.md).

La configuración inicial tiene **1.211 parámetros**, ocho canales CNN y doce estados LSTM. B=1, Q=1, W=4, Λ=1 y τmín=0,25 fueron decisiones iniciales de software, no ganadores de HPO. Las tres variantes funcionales eliminan penalización, saturación o contenido. La base M00 sin ambas modificaciones **ya fue implementada y entrenada en v1**; su contraste con el modelo completo se conserva en el [anexo B](17_revision_post_piloto.md). Esto no completa automáticamente el factorial estadístico de las ablaciones.

### Pruebas y límites de exportación

El primer módulo aprobó **35 casos pytest**: formas y entradas inválidas, semilla reproducible sin alterar el RNG global, causalidad del encoder, independencia entre muestras, gradientes activos, `gradcheck`, restricciones suaves, ablaciones aisladas y serialización. Este conteo identifica ese módulo histórico; las ampliaciones de software conservan sus propios ámbitos de comprobación. El sobreajuste diminuto usa ocho ejemplos artificiales y 350 pasos de Adam; no es entrenamiento en Bitcoin.

La ida y vuelta TorchScript se probó con lotes 1, 3 y 4 y tolerancia declarada. El trazado LSTM produce advertencias de comprobaciones de forma tratadas como constantes: hay que aplicar `validate_input` antes de usar una traza. No se entrega ese ensayo como artefacto final entrenado en mercado.

La cabeza usa una multiplicación densa por $P$: su término dominante es $O(n^2d)$, además del encoder. No se declara coste lineal de una implementación de promedio móvil que todavía no se usa. La complejidad asintótica no sustituye FLOPs, memoria y latencias medidos.

### Uso y reproducción

```python
from src.original_model import CandidateConfig, OriginalModel

model = OriginalModel(CandidateConfig(), seed=0)
# x contiene ventanas ya transformadas y estandarizadas de forma lote×7×5.
scaled_prediction = model(x)
```

```bash
python -m pip install -r requirements-e3-original-cpu.lock.txt
python -m pytest tests/test_original_model.py -q
python scripts/e3_verify_original.py --check
```

El workflow CPU conserva esas comprobaciones. Los ejecutores de desarrollo v1/v2 existen y sus resultados están auditados; eso no acredita una integración confirmatoria completa con todos los rivales de E2. El autor deberá comprender y modificar el código y la demostración; la asistencia de IA está declarada en las fuentes y la procedencia.

- {download}`Código del candidato <../_static/e3_bloque_c_20261115/original_model.py>`.
- {download}`Pruebas del módulo <../_static/e3_bloque_c_20261115/test_original_model.py>`.
- {download}`Derivación y pseudocódigo completos <../_static/e3_bloque_c_20261115/theory.md>`.
- {download}`Informe de verificación <../_static/e3_bloque_c_20261115/software_verification.json>`.
- {download}`Matriz de antecedentes <../_static/e3_bloque_c_20261115/review_matrix.csv>`.
- {download}`Bitácora de consultas <../_static/e3_bloque_c_20261115/search_attempts.csv>`.
- {download}`Referencias BibTeX <../_static/e3_bloque_c_20261115/references.bib>`.

## 6.6 Validez del modelo y regla de evaluación

La guía distingue: si R1 o R2 fallan, tope 3,0; si R1/R2 y la propiedad se acreditan pero no R3, tope 4,0; la ruta al 5,0 exige cumplir la familia completa y los demás requisitos de evaluación. Es una regla condicionada a evidencia y evaluación académica, no una calificación calculada para este avance.

```{figure} ../_static/e3_diseno_riguroso_20261117/decision_tree.svg
:name: e3_arbol_topes
:alt: Árbol condicional de los topes: incumplir R1 o R2 conduce a 3,0; R1 y R2 acreditadas sin R3 a 4,0; las tres acreditadas a 5,0, sujeto a los demás requisitos.

Árbol de decisión adaptado de la Figura 3 de la guía. Las ramas «No» describen incumplimientos establecidos; un estado pendiente de revisión no se convierte automáticamente en incumplimiento. La propiedad formal y los demás requisitos siguen siendo exigibles. El árbol no asigna una nota al proyecto.
```

Aquí las condiciones siguen abiertas. Una demostración de la cabeza y pruebas de software no convierten R1 o R3 en aprobadas ni certifican la defensa académica del autor. Las limitaciones y resultados adversos deberán conservarse.

## 6.7 Propuesta posterior: cabeza de momentos y dictamen vigente

La propuesta conserva el mismo codificador y atención por contenido. Con los seis retornos conocidos al origen, define

$$
A=\frac17\sum_{i=t-6}^{t-1}(r_i-\bar r_6)^2,\qquad
U=|r_t-\bar r_6|,\quad V=U^2,\quad
Y=\sigma_7(t)=\sqrt{A+cV},\quad c=6/49.
$$ (eq-e3-moment-target-decomposition)

`A` se calcula con historia observable; `U,V,V²` son etiquetas de entrenamiento, nunca entradas disponibles al pronosticar. Tres lecturas escalares del contexto producirían

$$
m=s_u^2\operatorname{softplus}(a),\qquad
q=m^2(1+b^2),\qquad \rho=\frac{d^2}{1+d^2},
$$ (eq-e3-moment-parameterization)

$$
L=\sqrt A+\frac{cm}{\sqrt{A+cq/m}+\sqrt A},\qquad
H=\sqrt{A+cm},\qquad
\widehat\mu=L+\rho(H-L).
$$ (eq-e3-moment-interpolated-mean)

Aquí `m,q` representan momentos de `V`; este `q` no es la consulta de atención de la sección 6.5. Con momentos verdaderos y positivos, la región realizable de la media es `[L,H)` cuando `q>m²`; `L` se alcanza y `H` es supremo no alcanzado. Si `q=m²`, ambas cotas coinciden. Con momentos estimados, la región calculada puede no contener la media verdadera: no es un intervalo de predicción ni garantiza menor MSE.

El criterio propuesto suma MSE de $Y$ y scores conocidos de media para `V,V²`, con escalas calculadas sólo en ajuste. La demostración de consistencia condicional exige `E[V|X]>0`, `E[V²|X]<∞` y una clase que represente conjuntamente los valores verdaderos; no demuestra convergencia de esta red finita, fidelidad de trayectoria o superioridad predictiva. El caso verdadero de momentos nulos queda fuera del teorema logarítmico. La formulación, bordes y verificación artificial completos están en el [anexo F](21_propuesta_momentos.md).

El [anexo G](22_dictamen_momentos.md) demuestra dos límites: momentos estimados incorrectos pueden excluir la media verdadera, y los auxiliares pueden empeorar un paso escalar SGD frente a la misma cabeza con MSE sola. Son cálculos artificiales, no una simulación del Adam futuro ni evidencia de fracaso en BTC. El control que faltaba en el primer diseño —**la misma cabeza por momentos sin pérdidas auxiliares**— se incluyó en el piloto completado; una media directa con igual número de parámetros responde otra pregunta.

**Cierre posterior al dictamen:** se conserva la adaptación experimental sin seleccionarla como aporte original. El piloto solicitado para comprobar su funcionamiento ya completó las tres recetas; su [resumen inmutable](https://github.com/jassan-bit/DeepLearning_202630/blob/4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4/runs/momentos_v1/summary.json) registra RMSE **0,0032589201** para `moment_joint`, frente a **0,0031446148** de la referencia aditiva congelada, sobre la validación de desarrollo. No superó la heurística del 2 %. Ese piloto no cerró la revisión sustantiva del mecanismo por momentos; su resultado no demuestra ausencia de novedad ni explica por sí solo el mecanismo.

La implementación en [src/e3_moment_model.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/src/e3_moment_model.py) tiene **1236 parámetros**, contados sobre el módulo: 1197 del codificador y atención, más tres lecturas activas de 13 parámetros. Se compararon `moment_joint`, `moment_mean_only` y `direct_moment_aux`; la referencia aditiva v2 se reutilizó mediante su CSV congelado, con 1210 parámetros. El apartado 7.8 actualiza el cierre; el anexo F.6 conserva las instrucciones históricas. Estas tres recetas no forman parte de las diez réplicas aditivas de la comparación actual.

## 6.8 Referencia evaluada y alcance de la evidencia posterior

La denominación de los archivos, `CNN-LSTM-atencion-aditiva E3`, identifica la salida aditiva positiva de **1210 parámetros**, con atención temporal por contenido. No identifica la penalización y saturación de 6.5 ni la cabeza de momentos de 1236 parámetros. Se mantienen las diez semillas `271801`–`271810`; la semilla histórica de desarrollo `20261116` queda fuera de esa media y no se elige una réplica por su error de prueba.

### Formulación de la referencia efectivamente evaluada

Para una ventana estandarizada $X\in\mathbb R^{7\times5}$, el encoder aplica una convolución causal de kernel 3 y ocho canales, `tanh` y una LSTM unidireccional de doce estados. Si $H=(h_1,\ldots,h_7)$ es su salida, la atención usa

$$
q=\frac{q_{\rm raw}}{\sqrt{1+\|q_{\rm raw}\|^2}},\quad
\tau=0.25+\operatorname{softplus}(\tau_{\rm raw}),\quad
\alpha_j=\frac{\exp(h_j^\top q/\tau)}{\sum_k\exp(h_k^\top q/\tau)},\quad
c_H=\sum_j\alpha_jh_j.
$$ (eq-e3-additive-content-attention)

No se penalizan los scores ni se saturan los valores. La consulta no es otro modelo entrenado por separado: es un vector de parámetros de esta misma red. La lectura es

$$
w=\frac{4w_{\rm raw}}{\sqrt{1+\|w_{\rm raw}\|^2}},\qquad
z=w^\top c_H+b,\qquad
\widehat y=\sqrt A+a_{\rm tr}\,[10^{-4}+\operatorname{softplus}(z)].
$$ (eq-e3-additive-output)

$A$ es el piso observable definido en la ecuación {eq}`eq-e3-moment-target-decomposition` de 6.7, calculado a partir de los seis retornos de la ventana; $a_{\rm tr}>0$ es la media del objetivo en los 532 ejemplos de ajuste. El encoder y la atención usan float32; la lectura y la salida usan float64. La pérdida de esta receta es únicamente $\operatorname{mean}[(\widehat y-y)^2/a_{\rm tr}^2]$: **no usa NLL, cuadratura para predecir, una escala Student ni intervalos calibrados**. La palabra «aditiva» se refiere a sumar un exceso positivo al piso, no a una segunda clase de atención.

La propiedad de soporte se verifica directamente: para $A\ge0$, $a_{\rm tr}>0$ y $z$ real finito, `softplus(z)>0`, por lo que $\widehat y>\sqrt A+10^{-4}a_{\rm tr}$ en aritmética exacta. Además, con $A$ fijo, $\partial\widehat y/\partial z=a_{\rm tr}\operatorname{sigmoid}(z)\in(0,a_{\rm tr})$. Es una propiedad conocida de esta parametrización escalar; **no es una cota global del mapa OHLCV, una prueba de novedad ni una garantía de menor error**. Los casos sintéticos de salida comprueban la fórmula y el respeto del piso con tolerancia numérica. La cota latente del módulo de 1211 parámetros no se hereda al retirar la saturación.

El apartado [7.3](07_diseno_riguroso.md) aporta una derivación independiente para la cabeza aditiva, expresada en la cota {eq}`eq-e3-additive-lipschitz-bound`: con el piso y los parámetros fijos y las filas latentes acotadas por $\sqrt{12}$, obtiene $K_{\rm ad}=77{,}32121111929344\,a_{\rm tr}$. Su verificación usa estados sintéticos y no pesos entrenados. Es una cota condicional conocida de atención sobre estados acotados; tampoco acredita originalidad o explica por sí sola la mejora histórica.

El conteo auditable es CNN 128 + LSTM 1056 + consulta, lectura, sesgo y temperatura 26 = **1210**. Para longitud $n$, $f$ variables, $c$ canales, kernel $k$ y estado $h$, el coste por muestra incluye $O(nkfc+n(ch+h^2)+n^2h)$. El último término sigue presente en el código eager heredado porque calcula la referencia densa $PH$ aunque su penalización esté desactivada; no se declara un coste lineal de una implementación optimizada inexistente. Faltan los FLOPs completos y la curva empírica de escalabilidad de esta receta.

### Pseudocódigo descriptivo de la ejecución cerrada

```text
Entradas: datos y código fijados por hash; escalador de 532 fechas;
          semillas 271801–271810; plan y presupuesto ya registrados.
Para cada semilla:
  Construir la red de 1210 parámetros y ajustar el sesgo inicial
  por 64 pasos de bisección usando sólo el exceso medio de ajuste.
  Optimizar MSE con AdamW(lr=0.001, weight_decay=0), lotes de 32
  y recorte de norma del gradiente a 1; conservar optimizador y RNG.
  Tras cada época, medir RMSE en las 116 fechas de validación.
  Conservar la menor RMSE; los empates conservan la época anterior.
  Parar a las 40 épocas o tras 8 épocas sin mejora, sujeto al presupuesto.
Confirmar guardado remoto de los diez entrenamientos terminados.
Cargar los mejores pesos de cada semilla y predecir las 608 fechas fijadas.
Salida: diez trayectorias y sus métricas individuales; conservar todas.
```

Este pseudocódigo explica la campaña histórica; no inicia ni amplía entrenamiento. El [plan inmutable de estabilidad](https://github.com/jassan-bit/DeepLearning_202630/blob/2d25989dffd8e4d447bfa21d57c255b9efdcb465/notebooks/e3_estabilidad_aditiva_v1/plan.json) y su código fijan los detalles de reanudación y cómputo del consumo.

E3 conserva **532 ejemplos de ajuste y 116 de validación**, sin usar las 53 fechas de calibración de E2. Los 13 rivales neuronales de E2 tienen diez semillas cada uno; Persistence, SeasonalNaive y ARIMA conservan una trayectoria determinista por sistema: **133 predictores de E2**, además de las diez réplicas de E3. Todos se puntúan sobre las mismas **608 fechas**, del 1 de enero de 2025 al 31 de agosto de 2026. E2 se ajustó con 655 ejemplos y calibró con 53; no se afirma igualdad de tratamiento o presupuesto.

Las [métricas descargables](../_static/e3_estadistica_posterior_20261117/metrics.csv) y los [48 contrastes primarios](../_static/e3_estadistica_posterior_20261117/primary_contrasts.csv) respaldan el resultado posterior: los tres procedimientos rechazan para cada uno de los 16 rivales tras Holm en la familia de 48. Dos procedimientos estudian incertidumbre temporal condicionada a trayectorias ajustadas y otro la inicialización condicionada al calendario; no se presentan como tres confirmaciones temporales equivalentes. El MCS auxiliar conserva diez sistemas, incluida la aditiva, y no identifica un ganador único. La [sección 8](08_verificacion_estadistica.md) expone supuestos, sensibilidades y límites; ese análisis de la aditiva no acredita R1, R2 o R3. El dictamen posterior específico de V3 se separa en 6.18.

## 6.9 Revisión del cumplimiento para esta referencia

La revisión histórica del **17 de noviembre de 2026, a las 20:47 UTC**, distingue los requisitos de la guía de la evidencia que correspondía entonces a la receta aditiva. Se volvieron a ejecutar en CPU las pruebas sintéticas existentes del modelo y del entrenamiento, sin cargar cohortes de mercado ni repetir ajustes del benchmark. El {download}`informe de comprobación <../_static/e3_revision_bc_20261117/c_software_review.json>` conserva resultados, entorno y hashes de las fuentes. Los casos del modelo cubren también las otras dos cabezas de v2; su total no se atribuye exclusivamente a la aditiva.

Ese recibo precede a la exportación ONNX verificada el **18 de noviembre a las 00:08 UTC**: la referencia aditiva de la semilla 271801 ya tiene paridad frente a memoria CPU y pronósticos congelados sobre las **608 fechas comunes**, con tolerancia absoluta de **10⁻⁶ en σ₇**. Las {ref}`secciones 5.2 <exportaciones-onnx-verificadas>` y [9](09_dashboard_predictivo.md) presentan ese artefacto y los otros tres ONNX del prototipo. El pendiente de exportación del informe histórico conserva su fecha y no describe esta etapa posterior; el funcionamiento público, la política común de ajuste y los demás requisitos siguen teniendo sus propios pendientes.

| Requisito de la guía | Evidencia aplicable a la receta de 1210 parámetros | Cierre pendiente |
|---|---|---|
| Módulo, formulación y pseudocódigo | Código local, forward y pérdida identificados en 6.8; conteo 1210 verificado. | Diferencia algorítmica sustantiva y defensa humana de la contribución candidata. |
| Propiedad con supuestos y verificación | Piso y exceso positivo, más cota condicional de la cabeza aditiva en 7.3 con 1728 pares sintéticos y 96 gradientes de referencia. La cota de saturación pertenece a v1. | El dominio latente y el piso fijo no cubren el pipeline OHLCV. Faltan evidencia de utilidad predictiva y una contribución admisible; no se declara nueva esta propiedad. |
| Pruebas del modelo | Inicialización reproducible, admisión de datos, salida, precisión mixta y gradientes finitos no nulos; reanudación sintética de v2 con pesos, optimizador y RNG. La etapa posterior añade el ONNX aditivo de la semilla 271801 y su ida y vuelta en 608 fechas. | Sobreajuste diminuto específico de esta receta y `gradcheck` pertinente. El `gradcheck` de cuadratura y el caso aislado de copia de `state_dict` prueban cabezas half-t; no se atribuyen a toda la aditiva. La paridad de una exportación no completa esas pruebas o la política final común. |
| Protocolo común y HPO equivalente | Se conserva la receta elegida por validación y sus diez semillas; los resultados históricos no se alteran. | E2 y E3 no comparten política final de ajuste, tamaño de entrenamiento o presupuesto demostrados como equivalentes. |
| Ablación de al menos tres componentes | Hay controles de desarrollo y pruebas funcionales de otras variantes. | Ablación estadística de la receta defendida, aislando componentes, con particiones, semillas y tratamiento comparables; los controles de v1 no la completan. |
| Interpretabilidad, fidelidad y estabilidad | Se puede inspeccionar la atención; los errores por régimen están documentados. | Comparación con el mejor benchmark, fidelidad por borrado/inserción y estabilidad entre semillas. Los pesos de atención no bastan como explicación validada. |
| Complejidad y eficiencia | Conteo y coste asintótico identificados; resultados y código conservados. | FLOPs completos, escalabilidad y recursos medidos en condiciones comunes; ver capítulo 5. |
| Literatura y R1 | Lecturas focales, consultas históricas y respuestas posteriores verificadas de tres bases en 6.12. | Afinar consultas y completar cribado, lecturas, seguimiento de citas y diferencia sustantiva. El dictamen externo y dos repeticiones de cadenas por IA declarada se aportan en 6.13. |
| R3 y evaluación amplia | Diez semillas y análisis posterior frente a 16 rivales, con límites explícitos. | Potencia por contraste y conjunta, protocolo confirmatorio justo y al menos tres conjuntos adicionales elegidos antes de sus resultados; ver capítulos 7 y 8. |

La {download}`matriz de revisión de requisitos <../_static/e3_revision_bc_20261117/c_requirements_review.csv>` conserva el mismo corte histórico de revisión; el avance posterior de exportación se documenta arriba y en las secciones 5.2 y 9. La política común y los tres conjuntos adicionales ya tienen resultados en el {ref}`cierre de 8.18 <e3-final-extended-audit>`; las filas históricas de pendientes no describen ese estado posterior. La revisión no sustituye la model card, la declaración de datos, la bitácora de diseños ni la declaración de IA. La {ref}`decisión de 6.16 <e3-r1-adaptation-decision>` identifica V3 como candidato distinto; una mejor RMSE de la aditiva no demuestra su novedad ni se transfiere a V3.

(revision-focal-ampliada-20261118)=
## 6.10 Revisión focal ampliada del 18 de noviembre

La revisión final de este corte reúne **75 registros de consulta o acceso, once trabajos comparados y ocho lecturas focales de texto completo**. Los registros no son 75 artículos ni una medida de cumplimiento. La {download}`bitácora JSON <../_static/e3_remediacion_20261118/literature/search_log.json>` y su {download}`versión CSV <../_static/e3_remediacion_20261118/literature/search_log.csv>` distinguen búsquedas nativas, descubrimiento web y accesos bloqueados; la {download}`matriz de trabajos cercanos <../_static/e3_remediacion_20261118/literature/closest_work_matrix.csv>` conserva tarea, localizador, diferencia y alcance de cada lectura.

Se ejecutaron **seis cadenas en inglés y español y cuatro búsquedas por título en la interfaz nativa de arXiv**. Las 36 solicitudes API —seis cadenas en OpenAlex, Crossref, Semantic Scholar, arXiv, DBLP y OpenReview— no recibieron resultados bibliográficos por bloqueo de transporte. Los once sondeos alternativos mediante la herramienta web sobre DBLP/OpenReview/IEEE tampoco recuperaron resultados. Sus conteos permanecen desconocidos; las quince consultas de buscador general no se contabilizan como otras bases. **Sólo arXiv aportó resultados nativos observados: no se completó la cobertura efectiva de tres bases.**

| Antecedente primario y pasaje leído | Consecuencia para la referencia aditiva |
|---|---|
| [Dasoulas et al. (2021)](https://proceedings.mlr.press/v139/dasoulas21a/dasoulas21a.pdf), §3.3 ec.(2), §5 ec.(17)–(18); [Luong et al. (2015)](https://aclanthology.org/D15-1166.pdf), §3.1 ec.(7)–(8). | Pooling softmax, consulta aprendida y atención sobre estados recurrentes tienen antecedentes. E3 no implementa LipschitzNorm; cambiar la aplicación no acredita un mecanismo nuevo. |
| [Kim et al. (2021)](https://proceedings.mlr.press/v139/kim21i/kim21i.pdf), §6 p.8. | La atención diferenciable restringida a un dominio compacto ya se considera Lipschitz. La constante y verificación particulares de 7.3 son sustento técnico, sin acreditar novedad por sí solas. |
| [ConstraintNet, versión AAAI2021](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf), p.6816. | La guarda sobre un umbral por muestra mediante `exp(z)+b` anticipa el mecanismo general de piso más exceso positivo. E3 usa softplus y el piso específico de σ₇; falta demostrar que esa adaptación es sustantiva. No se atribuye a esta fuente una influencia histórica no comprobada. |

De los siete antecedentes iniciales, cinco cuentan con lectura focal del cuerpo; Peng y CEEMDAN conservan sólo extractos editoriales indexados por bloqueo del texto completo. Entre los trabajos añadidos, la [tesis UCLA de Xiaomeng Li (2025)](https://escholarship.org/content/qt073763g4/qt073763g4.pdf), §2.3/§3.2.4, aplica CNN-LSTM a una suma de siete retornos cuadrados: **no es la σ₇ poblacional centrada**. El [preprint de Alex Y. Li (2024)](https://arxiv.org/html/2410.09062v1), §3.1/§3.5, aplica TimeMixer a volatilidad cripto móvil de 21 días anualizada. Esas diferencias impiden tratar sus métricas o tareas como idénticas.

El {download}`dictamen delimitado <../_static/e3_remediacion_20261118/literature/precise_novelty_assessment.md>` clasifica los antecedentes como mecanismo conocido, análisis, entrenamiento o aplicación y reduce la posible contribución a la adaptación concreta del soporte de σ₇ y su integración. No localizar la composición exacta en esta revisión parcial no demuestra su ausencia. **R1 sigue pendiente de acreditación**: siguen sin cerrarse la diferencia sustantiva defendible, el seguimiento bibliográfico completo y la repetición final. Ese corte no disponía del dictamen externo que se adopta por la modalidad de IA declarada en 6.13. La adopción no acredita R2/R3. El {ref}`control posterior de soporte y proyección <control-posterior-del-soporte-causal-y-proyeccion>` aporta evidencia adicional de utilidad y límites, sin convertirla en originalidad o atribución causal.

## 6.11 Expediente de revisión y verificación externa de R1

El primer expediente de revisión del 18 de noviembre fija el objeto en `InnovationModel("additive_positive_head")`, de **1210 parámetros**, con identidad de código y procedencia verificables. Reúne la comparación formal del mecanismo, seguimiento focal de citas e implementaciones, consultas reproducibles y un paquete humano sin diligenciar. El {download}`expediente completo <../_static/e3_r1_review_20261118/expediente_r1.zip>`, su {download}`estado estructurado <../_static/e3_r1_review_20261118/status.json>` y el {download}`manifiesto SHA-256 <../_static/e3_r1_review_20261118/manifest.json>` conservan aquel corte; **6.12** documenta los resultados posteriores de Colab sin reescribir esos archivos. La integridad de archivos no acredita novedad.

La comparación delimita un antecedente especialmente cercano: ConstraintNet ya describe una guarda por muestra sobre un umbral. Para un piso y una escala positivos fijos, el exceso de E3 puede expresarse mediante una transformación escalar del enlace exponencial. Esto identifica el mismo tipo de región admisible por muestra; **no demuestra igualdad de clases neuronales con lecturas afines, gradientes, optimización o aritmética finita**. La sustitución de `exp` por `softplus` y el umbral específico de σ₇ siguen necesitando una justificación de adaptación sustantiva. El análisis separa semejanza matemática de influencia histórica o copia.

| Condición | Evidencia del expediente | Estado en el corte histórico de 6.11 |
|---|---|---|
| Objeto preciso y diferencia frente a los antecedentes. | Identidad del modelo, descomposición del objetivo, comparación de fórmulas y límites de la posible contribución. | Comparación disponible; diferencia sustantiva sin acreditar. |
| Tres bases bibliográficas efectivas y cribado verificable. | Se conservan los resultados históricos de arXiv y los nuevos fallos de acceso con conteos desconocidos. El recolector registra respuestas originales, metadatos e integridad para su ejecución fuera de este entorno. | Cobertura y cribado pendientes; un recolector preparado no añade bases consultadas. |
| Seguimiento de fuentes próximas, citas e implementaciones. | Seguimiento focal con localizadores, relaciones y versiones de código observadas; los límites de lectura se conservan. | Evidencia parcial que debe completarse y repetirse antes de entregar. |
| Concepto de una persona ajena al diseño. | Instrucciones, dos cadenas recomendadas, formulario vacío y registro `NOT_PERFORMED`. | Revisión humana no realizada; no hay identidad, firma o concepto que atribuir. |

El [cuaderno de revisión bibliográfica para Colab](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/e3_r1_revision_bibliografica_colab.ipynb) usa CPU y conserva los resultados en una rama nueva de GitHub. Una ejecución satisfactoria recupera evidencia bibliográfica; no aprueba automáticamente el cribado, la revisión humana o R1. No entrena modelos, ejecuta predicciones de mercado ni modifica los resultados históricos.

El paquete inicial conserva las {download}`instrucciones <../_static/e3_r1_review_20261118/reviewer/README.md>` y el {download}`formulario de revisión <../_static/e3_r1_review_20261118/reviewer/REVIEW_FORM.md>` sin diligenciar. La modalidad vigente para este requisito es la revisión por IA declarada de 6.13, con dictamen y dos cadenas repetidas documentados. La aceptación de R1 corresponde a la evaluación académica. **R1 permanece pendiente de acreditación**: el expediente puede sustentar un concepto favorable, desfavorable o no concluyente; no se transforma la preparación asistida por IA en una revisión humana ni en una declaración de originalidad.

## 6.12 Resultados Colab de tres bases y revisión técnica adicional

El {download}`paquete de revisión de esta ejecución <../_static/e3_r1_search_review_20261118/review_bundle.zip>` reúne recibos, auditoría, cribado, fuentes y dictamen de IA. Su {download}`manifiesto de integridad <../_static/e3_r1_search_review_20261118/manifest.json>` distingue la conservación de archivos de la valoración científica y humana.

La ejecución `20261118T202323797581Z` consultó el código `314f6e7e888b0748c8790cf3c0d9f4fc71bd6a0b` y guardó sus resultados en el [commit inmutable `edf6fe3`](https://github.com/jassan-bit/DeepLearning_202630/tree/edf6fe312987fd7299f6ca850a20a67e4f1c3cbc/runs/r1_search/20261118T202323797581Z). La auditoría verificó **27 archivos y 1.626.702 bytes**, incluidos manifiesto, respuestas y CSV inicial. Sus tamaños, SHA-256 y blobs Git coinciden; los 10.966 blobs anteriores permanecen idénticos. La {download}`auditoría de bytes y metadatos <../_static/e3_r1_search_review_20261118/byte_audit.json>` y la {download}`copia empaquetada de los recibos <../_static/e3_r1_search_review_20261118/colab_receipts.zip>` conservan esas evidencias. El ZIP es un empaquetado posterior de los archivos originales, sin modificarlos.

| Base | Resultado de las seis cadenas | Alcance |
|---|---|---|
| Crossref | Seis respuestas bibliográficas válidas. | Primera página, diez registros por consulta; incluye componentes de figuras y búsquedas por relevancia amplias. |
| arXiv | Seis respuestas Atom válidas. | El feed expande palabras no entrecomilladas con OR; los resultados incluyen áreas ajenas al mecanismo. |
| OpenAlex | Seis respuestas bibliográficas válidas. | Primera página, diez registros por consulta; recuperar metadatos no equivale a leer sus fuentes. |
| DBLP | Seis respuestas HTTP 202 con cuerpo HTML no analizable como bibliografía. | Fallos conservados; conteos desconocidos y sin crédito de base efectiva. |
| Semantic Scholar | Seis respuestas HTTP 429. | Límite de servicio; conteos desconocidos y sin crédito de base efectiva. |

Las **18 respuestas válidas de tres bases** contienen **180 ocurrencias de metadatos**. Esto acredita la cobertura de respuestas de esta ejecución, no una búsqueda exhaustiva o ausencia de antecedentes. Los conteos no son comparables entre bases ni equivalen a artículos pertinentes. El CSV original mantiene `not_screened`; el {download}`cribado posterior de los títulos <../_static/e3_r1_search_review_20261118/screening/title_screening.csv>` documenta decisiones nuevas, duplicados y candidatos para lectura sin alterar ese recibo. El cribado de la primera página no completa la lectura de resúmenes, textos o todas las páginas.

Se solicitó además una revisión técnica por IA con contexto inicial separado y sin acceso a los dictámenes previos. El {download}`informe de esa revisión <../_static/e3_r1_search_review_20261118/ai_review/read_report.md>` identifica fuentes realmente consultadas, formulación, coincidencias, diferencias y limitaciones. Esa separación de contexto no prueba independencia humana; no hubo una persona revisora. El dictamen se adopta para el requisito por la modalidad de IA declarada autorizada en 6.13. El {download}`estado de la ampliación <../_static/e3_r1_search_review_20261118/status.json>` conserva el corte anterior a esa indicación docente.

La lectura adicional identifica un antecedente financiero concreto: [Opschoor y van der Wel, publicado en línea en 2024 y en JBES 43(2), 298–311 en 2025](https://doi.org/10.1080/07350015.2024.2365779), §2.2, ecuación (6), utiliza límite inferior más softplus. Su problema es la curva de rendimientos, con una cota prefijada y factores Nelson–Siegel; no implementa la ventana, el piso observable de σ₇ o la CNN–LSTM de E3. Este antecedente limita una pretensión amplia sobre la transformación de salida, sin probar identidad de todo el modelo ni ausencia de originalidad de la adaptación específica. Las {download}`fuentes y localizadores <../_static/e3_r1_search_review_20261118/ai_review/sources.json>` distinguen cinco lecturas focales de secciones del cuerpo y una fuente consultada sólo por resumen.

La ampliación del expediente no ejecuta entrenamientos, pronósticos de mercado o contrastes nuevos. **R1 no queda acreditada por consultar tres bases ni por disponer de otro dictamen de IA.** Deben justificarse la diferencia sustantiva, las lecturas pertinentes y el refinamiento y repetición final de búsquedas. La revisión externa por IA declarada se documenta en 6.13; el formulario humano continúa sin diligenciar como hecho histórico, sin ser el bloqueo de modalidad de ese requisito. La aceptación científica corresponde al evaluador.

## 6.13 Revisión externa por IA declarada: indicación docente aplicada

El autor comunicó el **18 de noviembre de 2026** que el profesor permite sustituir al revisor humano por IA, con la condición de mencionar su uso. Se aplica esa indicación a la revisión externa de R1 que se estaba discutiendo. Su {download}`registro de procedencia y alcance <../_static/e3_r1_ai_substitution_20261118/instructor_instruction.json>` conserva la comunicación; no atribuye al profesor una aprobación científica del mecanismo.

El {download}`expediente de adopción de la modalidad <../_static/e3_r1_ai_substitution_20261118/reviewer_modality_bundle.zip>` reúne declaración, registro de adopción y repeticiones de consultas, con manifiestos SHA-256. Los informes técnicos enlazados conservan sus archivos y conclusiones originales.

**Declaración de uso:** se utilizó un asistente de inteligencia artificial de OpenAI, en este entorno de ChatGPT/Codex, para búsqueda bibliográfica, cribado, lectura focal de fuentes, comparación del mecanismo y redacción del dictamen. La revisión técnica se realizó con contexto separado de los dictámenes previos. Se empleó además IA para repetir dos cadenas exactas en la interfaz nativa de arXiv. La {download}`declaración específica <../_static/e3_r1_ai_substitution_20261118/ai_use_declaration.md>` identifica las tareas y limitaciones; no se presenta una persona o firma humana.

| Evidencia del componente de revisión externa | Resultado documentado |
|---|---|
| Dictamen de la referencia aditiva de 1210 parámetros. | Informe de IA de 6.12, con cinco lecturas focales de secciones del cuerpo y una fuente sólo por resumen. Se vincula por commit, ruta y SHA-256; sus conclusiones críticas se conservan. |
| Repetición de Q2: `CNN LSTM attention cryptocurrency`. | La interfaz nativa mostró dos resultados; se examinaron sus dos pares de título/resumen. |
| Repetición de Q5: `neural network output constraints`. | La interfaz nativa mostró 963 resultados y anunció una primera página de 50; se examinaron deliberadamente sus cuatro primeros pares de título/resumen. |
| Uso de IA mencionado y alcance identificado. | Declaración visible en este apartado, capítulo 15 y anexo final, con evidencias y procedencia descargables. |
| Estado del componente de revisión externa. | **Completado mediante IA declarada conforme a la indicación docente comunicada.** No hubo revisión humana. |

La {download}`evidencia de las dos consultas <../_static/e3_r1_ai_substitution_20261118/query_check/evidence.json>` conserva URLs, cadenas, encabezados, conteos y decisiones. Fueron nuevas solicitudes y observaciones mediante la herramienta web; ésta indicó una captura del día, sin exponer una hora de transacción del servidor. Las cadenas léxicas son las mismas, pero la interfaz y la API histórica con expansión OR no representan una solicitud idéntica: sus conteos no se comparan como cambios del índice. El control de consultas no añadió lecturas de cuerpos a las del dictamen técnico.

El {download}`registro de adopción del dictamen <../_static/e3_r1_ai_substitution_20261118/review_adoption.json>` distingue la modalidad autorizada y completada del resultado científico. **R1 global sigue pendiente de acreditación:** la diferencia algorítmica sustantiva y el cierre bibliográfico no quedan demostrados por cambiar la modalidad del revisor. La defensa del autor y los requisitos de R2/R3 conservan sus estados. Los informes históricos, el formulario humano vacío, los modelos, los entrenamientos y las predicciones se conservan sin cambios.

## 6.14 Candidato V3: ley atómica del retorno desconocido y transformación exacta

La nueva propuesta aborda una limitación concreta de la cabeza aditiva: imponer el piso observable garantiza soporte, pero no modela la distribución del retorno que falta para completar la etiqueta. **Ese corte inicial preparó V3 sin resultados de mercado. El piloto posterior ya completó 30 ajustes de desarrollo**, con veredicto adverso de la receta completa; véase el {ref}`cierre de 8.19 <e3-v3-pilot-completed-20261120>`. No sustituye la referencia aditiva de 1210 parámetros en las tablas históricas o en el servicio, ni hereda su RMSE.

Sean los seis retornos ya conocidos $r_1,\ldots,r_6$, su media $\bar r_6$ y $A=\frac{1}{7}\sum_{i=1}^{6}(r_i-\bar r_6)^2$. Si $U=|r_{\mathrm{futuro}}-\bar r_6|$, la identidad algebraica de la etiqueta poblacional es

```{math}
:label: eq-e3-v3-pushforward
\sigma_7=\sqrt{A+\frac{6}{49}U^2},\qquad
\widehat\sigma_7=\sum_{k=1}^{3}p_k(X)
\sqrt{A+\frac{6}{49}u_k(X)^2}.
```

La red aprende **tres átomos positivos ordenados** $u_k$ y sus probabilidades $p_k$. Predice la esperanza exacta bajo esa ley finita, no la raíz de una varianza esperada. Entrena con MSE normalizado de σ₇ y CRPS normalizado de $U$, con peso 0,1 fijado antes del piloto. La pérdida CRPS se calcula exactamente como $\sum_k p_k|u_k-U|-\frac12\sum_{k,l}p_kp_l|u_k-u_l|$; es una regla conocida, no una invención de este proyecto.

El codificador y la atención se reutilizan de M00. La cabeza añade átomos y masas aprendidas: **1262 parámetros** en `atom_crps` y `atom_mse_only`, y **1236** en el control `fixed_probs`. Las tres variantes comparten átomos y predicciones iniciales; el control de probabilidades fijas tiene menor capacidad y se declara así, sin añadir parámetros inactivos para aparentar igualdad. El plan histórico del cierre integral incluía la aditiva y M00 proyectado. El piloto posterior, fijado en 7.13, ejecutó cinco recetas neuronales y dos controles analíticos; su composición y resultado se conservan en {ref}`8.19 <e3-v3-pilot-completed-20261120>`. No se confunden los dos planes.

La {download}`demostración y sus supuestos <../_static/e3_cierre_integral_20261118/mechanism/theory.md>` acredita soporte, orden, la identidad de transformación y propiedades respecto de $A$ **manteniendo fija la ley**. No demuestra monotonicidad de todo el pipeline cuando la red también cambia sus átomos con $X$. La interpretación poblacional requiere que la verdadera ley condicional pertenezca a la clase finita; tres átomos pueden estar mal especificados. No se afirma consistencia universal, calibración de mercado o superioridad.

El {download}`recibo de siete pruebas sintéticas <../_static/e3_cierre_integral_20261118/mechanism/verification.json>` registra Python 3.12.14 y Torch 2.8.0+cpu: conteos y estados, CRPS por enumeración, gradientes, propiedades, escala y serialización. Hubo **cero ajustes y pronósticos de mercado**. El [módulo fuente](https://github.com/jassan-bit/DeepLearning_202630/blob/main/src/e3_overlap_model_v3.py), las {download}`comparaciones de afirmaciones <../_static/e3_cierre_integral_20261118/mechanism/claim_comparison.csv>` y la {download}`procedencia y declaración de IA <../_static/e3_cierre_integral_20261118/mechanism/provenance.json>` hacen revisable la adaptación.

**La novedad de V3 sigue sin acreditar.** Gneiting y Raftery (2007), §4.2 y ecuación (21), sustentan CRPS; Bishop (1994) anticipa la predicción neuronal de leyes condicionales; Bellemare, Dabney y Munos (2017) constituyen un antecedente de aproximaciones categóricas en otro contexto. El expediente original distingue lectura focal del primero y consulta de resúmenes/metadatos de los otros dos. La {ref}`comparación posterior de 6.16 <e3-r1-adaptation-decision>` amplía los antecedentes y explicita qué cambia en V3. La revisión de la aditiva en 6.13 no se transfiere automáticamente a esta propuesta: la prioridad de la composición sigue abierta. El piloto de {ref}`8.19 <e3-v3-pilot-completed-20261120>` ya aporta tres ablaciones comparables de desarrollo y un resultado desfavorable de la receta completa; no constituye ablación confirmatoria ni acreditación de R1. Se declara la asistencia de IA en formulación, código y verificación, conforme a la modalidad docente comunicada por el autor.

(e3-r1-comparison-close)=
## 6.15 Comparación bibliográfica de cierre de la referencia aditiva

**Revisión del 19 de noviembre de 2026 en Colombia, 20 de noviembre en UTC; asistencia de IA declarada.** El objeto evaluado es exactamente `InnovationModel('additive_positive_head')`, de **1210 parámetros**, en la auditoría de 796 trabajos de 8.18. Se distingue de v1, la cabeza de momentos y V3. La guía permite una transferencia citada con modificación sustantiva; no convierte un cambio de aplicación, enlace positivo o hiperparámetros en R1.

La comparación delimitada está cerrada como **dictamen sobre las fuentes examinadas**. No equivale a una búsqueda exhaustiva mundial, a demostrar que toda posible contribución carece de novedad o a acreditar R1. La {download}`identidad y procedencia de los modelos <../_static/e3_r1_close_20261120/model_identity.json>` registra los hashes comprobados frente al commit del libro `f666a215`, la fuente científica `5a07d615` y las dos clases distintas. No se modificaron los modelos, pesos o resultados.

### Alcance de búsqueda, cribado y lectura

El recolector Colab anterior conservó **180 ocurrencias** de seis cadenas bilingües en Crossref, arXiv y OpenAlex, con diez registros por cadena y base. Tras deduplicar identificadores quedaron 166 registros canónicos y 14 repeticiones; identificadores diferentes no se consideran automáticamente trabajos diferentes o versiones idénticas. El cierre conserva aquel registro de tres bases y sus primeras páginas, sin inventar cobertura adicional.

La cola de 48 títulos prioritarios y dos inciertos tuvo **50 decisiones documentadas**. Se leyeron **41 resúmenes completos**: 32 de respuestas nativas guardadas y nueve recuperados en fuentes primarias. Otros tres sólo ofrecieron extractos parciales, uno una ficha editorial sin resumen y cinco carecieron de resumen primario verificable. Registrar una decisión de acceso no significa haber leído ese resumen. El {download}`cribado final por identificador <../_static/e3_r1_close_20261120/search_review/abstract_screening_all_50.csv>` y su {download}`resumen de alcance <../_static/e3_r1_close_20261120/search_review/abstract_screening_summary.json>` preservan esas diferencias.

La repetición actual de seis cadenas en las tres API produjo **18 fallos de transporte antes de recibir bibliografía**, todos con túnel 403. Sus conteos son desconocidos, no cero: véase el {download}`recibo de repetición <../_static/e3_r1_close_20261120/search_review/fresh_native_search_receipt.json>`. Se añadieron accesos primarios y seguimiento focal; no se contabilizan como otra ejecución exitosa de las tres bases. La modalidad externa por IA ya adoptada en 6.13 permanece completada según la indicación docente comunicada por el autor; no se inventa una firma humana.

### Diez trabajos cercanos y diferencia que permiten sostener

La matriz completa distingue tarea, fórmula, localizador, acceso, coincidencias, diferencias y consecuencia para R1. **Ocho cuerpos estuvieron accesibles para lectura de secciones seleccionadas; dos fuentes se limitaron a resumen o extracto primario.** No son diez lecturas completas de principio a fin.

| Fuente primaria | Lectura efectivamente observada | Diferencia frente a la referencia |
|---|---|---|
| [Brosowsky, Keck, Dünkel and Zöllner (2021)](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf) | Cuerpo accesible; secciones seleccionadas | Piso de σ₇ y enlace softplus frente a una guarda por muestra con exponencial; la familia de soporte tiene antecedente directo. |
| [Dasoulas, Scaman and Virmaux (2021)](https://proceedings.mlr.press/v139/dasoulas21a/dasoulas21a.pdf) | Cuerpo accesible; secciones seleccionadas | Mantiene consulta aprendida y pooling softmax; no implementa el denominador de LipschitzNorm. |
| [Luong, Pham and Manning (2015)](https://aclanthology.org/D15-1166.pdf) | Cuerpo accesible; secciones seleccionadas | Consulta paramétrica de pooling frente a consulta del decodificador; la atención sobre estados recurrentes es conocida. |
| [Kim, Papamakarios and Mnih (2021)](https://proceedings.mlr.press/v139/kim21i/kim21i.pdf) | Cuerpo accesible; secciones seleccionadas | Cota condicionada a estados acotados; no se transfiere una garantía global de robustez desde OHLCV. |
| [Wang et al. (2024)](https://arxiv.org/html/2405.14616v1) | Cuerpo accesible; secciones seleccionadas | Cabeza aditiva CNN–LSTM frente a mezcla multiescala; la diferencia no prueba una corrección de un defecto del paper guía. |
| [Reisenhofer, Bayer and Hautsch (2022)](https://arxiv.org/html/2205.07719v1) | Cuerpo accesible; secciones seleccionadas | σ₇ poblacional centrada de siete retornos frente a la tarea de varianza realizada y estructura HAR de la fuente. |
| [Huang, Sangiorgi and Urquhart (2024)](https://research.birmingham.ac.uk/en/publications/forecasting-bitcoin-volatility-using-machine-learning-techniques/) | Resumen/extracto primario; cuerpo no leído | Arquitectura y definición del objetivo requieren distinguirse; el resumen no permite atribuir ecuaciones del cuerpo no leído. |
| [Peng et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0957417423020225) | Resumen/extracto primario; cuerpo no leído | Pronóstico de σ₇ frente a tendencia de múltiples criptomonedas; la familia CNN–LSTM–atención tiene antecedente. |
| [Dudek, Orzeszko and Fiszeder (2025)](https://arxiv.org/html/2508.15922v1) | Cuerpo accesible; secciones seleccionadas | Media puntual de σ₇ centrada frente a cuantiles de varianza diaria realizada a partir de retornos intradía; el pronóstico probabilístico cripto ya existe. |
| [Yang, Lorch, Graule, Lakkaraju and Doshi-Velez (2020)](https://arxiv.org/html/2010.10969v1) | Cuerpo accesible; secciones seleccionadas | Guarda determinista frente a restricciones probabilísticas mediante priors bayesianos; las restricciones interpretables tienen antecedentes. |

Se conservan la {download}`matriz CSV <../_static/e3_r1_close_20261120/primary_review/comparison_matrix.csv>`, su {download}`versión legible <../_static/e3_r1_close_20261120/primary_review/comparison_matrix.md>` y las {download}`fuentes con localizadores <../_static/e3_r1_close_20261120/primary_review/sources.json>`. La dirección de las citas se volvió a comprobar desde tres semillas —ConstraintNet, Dasoulas y TimeMixer—, con **seis relaciones observadas**, una anterior y una posterior para cada una. El {download}`seguimiento de citas <../_static/e3_r1_close_20261120/primary_review/citation_followup.json>` no se presenta como un censo exhaustivo de todos sus citantes.

También se consultaron los README de los repositorios oficiales [TimeMixer](https://github.com/kwuking/TimeMixer/tree/e24610583b36fdd8c76cc17a8df4e65759a5f460) y [HARNet](https://github.com/mdsunivie/HARNet/tree/3b2832f163ec52cab2435163e82f2e816ae348c9), sus tareas y contexto de resultados. El {download}`registro de repositorios <../_static/e3_r1_close_20261120/code_review/repository_observations.json>` fija esos commits. Fue una inspección delimitada de documentación, no una auditoría de todos sus archivos; no se copió su implementación al proyecto. Las tablas de otros objetivos —por ejemplo varianza realizada intradía— no se comparan numéricamente con σ₇ centrada como si midieran la misma tarea.

### Diferencia escalar precisa frente a la guarda conocida

La referencia usa `b=√A`, escala positiva `s` e incremento mínimo `ε=10⁻⁴`:

```{math}
:label: eq-e3-r1-guard-comparison
\widehat y=b+s[\varepsilon+\operatorname{softplus}(z)]
=(b+s\varepsilon)+\exp\{\log[s\operatorname{softplus}(z)]\}.
```

ConstraintNet describe `exp(v)+b` como guarda sobre un umbral por muestra en la página impresa 6816. La igualdad anterior es una reparametrización escalar en aritmética real; explica por qué la propiedad «piso más exceso positivo» tiene antecedente directo. **No demuestra equivalencia entre redes de capacidad fija, gradientes, optimizadores o resultados de entrenamiento.** Esos efectos necesitarían sus controles; no se afirman por la identidad. La verificación escalar histórica y sus 1000 casos artificiales conservan el alcance descrito en el expediente de 6.11.

La consulta aprendida y el pooling softmax también tienen antecedentes explícitos. La cota con estados acotados y la identidad de varianza aportan sustento técnico, pero no se reclaman como descubrimientos matemáticos. El resultado de mercado de la referencia sigue siendo válido con su alcance experimental; su rendimiento no establece causalmente la utilidad de cada componente ni la novedad.

### Declaración y relación con el paper guía

**Declaración que sí sostiene esta revisión:** se implementó y evaluó una referencia CNN–LSTM con atención de contenido y soporte explícito para σ₇, y se examinó su comparación común con 18 rivales en cuatro conjuntos. En el repertorio consultado se identificaron antecedentes de sus mecanismos genéricos. La revisión no establece una adaptación algorítmica sustantiva de la aditiva que permita acreditar R1; tampoco prueba la inexistencia mundial de cualquier composición relacionada.

TimeMixer conserva su condición de paper guía y trabajo cercano. La limitación de calendario de ETTh1 y la corrupción artificial antigua no demuestran que este modelo corrija un defecto de TimeMixer en BTC. El soporte observable de la etiqueta y el deterioro con ruido son limitaciones medibles del problema; no una explicación causal ya probada. La {ref}`decisión siguiente <e3-r1-adaptation-decision>` se apoya en una modificación explícita del modelo de salida y de su entrenamiento, con antecedentes atribuidos.

(e3-r1-adaptation-decision)=
## 6.16 Adaptación candidata que puede defenderse: V3 existente

**Decisión de diseño:** conservar la aditiva como referencia evaluada y estudiar **V3**, ya implementada en 6.14, como única adaptación candidata. No se selecciona otra arquitectura ni se cambian ahora K=3 o λ=0,1 usando el resultado de prueba. V3 muestra diferencias verificables de **estructura y entrenamiento**; su prioridad para el fenómeno y tarea, utilidad y cumplimiento global de R1 siguen sin establecerse.

| Elemento | Aditiva evaluada | V3 de desarrollo |
|---|---|---|
| Salida aprendida | Un exceso escalar positivo. | Tres posiciones positivas ordenadas y sus masas aprendidas para la magnitud futura U. |
| Pronóstico σ₇ | `√A + sY(ε+softplus(z))`. | `Σ pₖ √(A + (6/49)uₖ²)`: transformar cada átomo antes de promediar. |
| Entrenamiento | MSE normalizado de σ₇. | MSE normalizado de σ₇ más 0,1 CRPS normalizado de U. |
| Parámetros | 1210. | 1262 en `atom_crps` y `atom_mse_only`; 1236 con masas fijas. |
| Evidencia de mercado | Auditoría completa de 8.18 y resultados históricos separados. | El expediente bibliográfico inicial no ejecutó ajustes. El piloto posterior completó **30 ajustes de desarrollo**, con heurística fallida, en {ref}`8.19 <e3-v3-pilot-completed-20261120>`. No hereda los 796 trabajos. |

Con seis retornos conocidos y su media m, `A=Σ(rᵢ−m)²/7` es observable; `U=|r_futuro−m|` sólo es una etiqueta al ajustar. La modificación preserva la identidad `Y=√(A+(6/49)U²)` y aprende una ley finita `Gₓ=Σpₖδᵤₖ`. La media que calcula es exacta **bajo esa ley programada**, no bajo una ley de mercado conocida. Las escalas de U y de Y se estiman únicamente en ajuste. La formulación y el código preexistentes están en 6.14; el {download}`dictamen detallado de adaptación <../_static/e3_r1_close_20261120/adaptation_design/decision.md>` fija su objeto, fuentes y controles.

### Procedencia y alcance de la diferencia

La predicción neuronal de distribuciones condicionales es conocida desde Bishop y sus aplicaciones a volatilidad financiera aparecen en Schittenkopf, Dorffner y Dockner; CRPS pertenece a la literatura de puntuación propia. C51 es un antecedente categórico en aprendizaje por refuerzo, con soporte fijo y actualizaciones Bellman/KL: V3 no implementa ese algoritmo ni hereda sus garantías.

| Antecedente | Alcance de acceso | Qué obliga a reconocer |
|---|---|---|
| [Bishop (1994)](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/) | Resumen institucional; extracción del PDF ilegible. | La predicción neuronal de una ley condicional no es nueva. |
| [Schittenkopf, Dorffner y Dockner (1998)](https://research.wu.ac.at/ws/portalfiles/portal/18954278/document.pdf) | Lectura focal del cuerpo, §2 ec. (1)–(9), §3 caso ATX. | Las mezclas neuronales ya se aplicaron a volatilidad financiera; su objetivo no se equipara a σ₇ centrada. |
| [Bellemare, Dabney y Munos (2017)](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf) | Lectura focal, §4.1–4.2. | Soporte categórico fijo y Bellman/KL en otro problema; V3 aprende posiciones y usa CRPS supervisado. |
| [Gneiting y Raftery (2007)](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf) | Lectura focal anterior reutilizada, §4.2 ec. (21). | CRPS y su propiedad estricta son conocidos; la normalización y los supuestos se declaran. |
| [Feng, Li y Zhang (2023)](https://aimspress.com/article/doi/10.3934/era.2023194) | Resumen editorial y metadatos; no lectura del cuerpo. | Es otro antecedente de previsión probabilística de volatilidad, no prueba de la composición exacta de V3. |

La candidata adapta la **composición específica**: ley del único componente futuro no observado de una volatilidad móvil solapada, transformación dependiente de la historia antes de tomar la media y señal auxiliar propia de U. Una descripción amplia como «modelo probabilístico con soporte para volatilidad» no sería una afirmación de novedad admisible. La {download}`decisión estructurada <../_static/e3_r1_close_20261120/adaptation_design/decision.json>` distingue esa modificación concreta de los componentes conocidos.

### Limitación precisa y prueba que todavía falta

Conocer sólo `E[U²|X]` no determina `E[√(A+cU²)|X]`, por lo que un control que sustituye la ley por su segundo momento puede perder información relevante. **Eso no demuestra una incapacidad de la aditiva:** una cabeza escalar flexible entrenada con MSE puede representar la misma media puntual que V3. La justificación investigable es una restricción inductiva y señal auxiliar; su beneficio de muestra finita puede resultar favorable o adverso.

El {ref}`análisis de 7.12 <e3-r1-v3-design-protocol>` aporta cotas delimitadas, contraejemplos y una hipótesis futura refutable. No se afirma que una ley de tres átomos corrija el ruido o la infracobertura observados en 8.18. Tampoco se deriva calibración real de usar CRPS. El resultado analítico favorable compara exclusivamente la esperanza no lineal con el control de sustitución por segundo momento bajo la ley correcta, no con la red aditiva.

Los controles propuestos aíslan la pérdida (`atom_mse_only`), las masas (`fixed_probs`, con menor capacidad declarada) y la agregación exacta frente a la sustitución por segundo momento (`atom_plugin`). El expediente analítico precede a la implementación de este último control. El seguimiento de {ref}`7.13 <e3-v3-controlled-development-pilot>` fijó el código y la política común antes de los ajustes. Los tres controles se ejecutaron en el piloto posterior de {ref}`8.19 <e3-v3-pilot-completed-20261120>`, con tres semillas finales por receta. Es evidencia de ablación de desarrollo, con menor capacidad declarada en `fixed_probs`; no completa la batería estadística confirmatoria exigida.

### Frontera entre diseño y confirmación

La comparación actual tiene **19 sistemas**, incluidos la aditiva, `floor_only` y `floor_plus_mean_excess`. Añadir V3 supone 20 sistemas; el planificador histórico de 18 no dimensiona automáticamente esa familia. El piloto de 7.13 fijó presupuesto, variantes, semillas, regla de continuación y fallos antes de su ejecución de 8.19. Su regla no se modifica después del resultado. Una eventual confirmación necesita su propia versión/tag, datos pertinentes, adecuación temporal, potencia por contraste y conjunta y las semillas suficientes.

Los resultados de D00–D03 ya están expuestos. Si guían decisiones sobre V3, esos conjuntos se reportan como desarrollo o evaluación exploratoria para la adaptación, sin recuperar cegamiento ni convertir los 796 trabajos en entrenamientos de V3.

### Expediente descargable y estado del componente

El {download}`paquete de cierre <../_static/e3_r1_close_20261120/review_bundle.zip>` reúne comparación, cribado por identificador, fuentes, citas, procedencia, decisión, comprobación analítica y manifiesto. El {download}`control de integridad <../_static/e3_r1_close_20261120/packet_manifest.json>` fija los archivos; las respuestas y paquetes históricos mantienen sus propios hashes. No se redistribuyen cuerpos completos de artículos ni los cachés íntegros de resúmenes.

Este expediente inicial mejoró la evidencia de revisión y diseño: objeto, antecedentes, diferencia, limitación, hipótesis, matemáticas y controles. El seguimiento de 6.17 incorpora el resultado posterior sin reescribir ese corte. **La aditiva continúa como referencia experimental; la receta completa V3 falló su heurística de desarrollo y su prioridad específica sigue sin establecerse. R1 y R3 no se marcan acreditadas y no se asigna una nota automática.**

(e3-contribution-evidence-close-20261120)=
## 6.17 Contribución delimitada y procedencia tras el piloto

**Corte anterior a la revisión específica de 6.18.** Sus archivos, dictamen y alcances se conservan sin reescribirlos.

El {ref}`cierre de 8.19 <e3-v3-pilot-completed-20261120>` incorpora el piloto `20261120T031424176339Z`: 30 ajustes, tres semillas finales por receta y 55 fechas de evaluación en desarrollo BTC previamente expuesto. La V3 completa obtuvo RMSE media 0,00212991576846 frente a 0,00207383285786 de `atom_mse_only`: **2,70 % mayor error que el mejor control**. Su mejora de 1,27 % frente a la aditiva tampoco alcanza el umbral del 2 %. La receta completa no superó la regla registrada; se conservan todos los controles y no se cambia el umbral ni se promueve automáticamente la ablación ganadora.

**Declaración de contribución admisible:** se implementó, fundamentó y evaluó una referencia con soporte explícito y una adaptación candidata que aprende una ley finita del único componente desconocido de la etiqueta solapada. Se aislaron pérdida, masas y agregación mediante un piloto común, con diferencias de capacidad declaradas, y se conservó el resultado adverso. No se reivindican como invenciones CNN/LSTM, atención, salidas distribucionales, CRPS, la identidad de varianza, Jensen o la descomposición de riesgo.

La {download}`declaración precisa <../_static/e3_evidence_close_20261120/contribution/contribution_declaration.md>`, su {download}`versión estructurada <../_static/e3_evidence_close_20261120/contribution/contribution_declaration.json>` y la {download}`comparación de mecanismo, pérdida, entrenamiento y aplicación <../_static/e3_evidence_close_20261120/contribution/mechanism_loss_training_application.csv>` separan la diferencia programada, las propiedades conocidas y el resultado empírico. La {download}`matriz bibliográfica consolidada <../_static/e3_evidence_close_20261120/contribution/literature_crosswalk.csv>` reúne los diez registros de la aditiva y cinco antecedentes de V3 conservando sus accesos anteriores; no anuncia nuevas búsquedas, cuerpos leídos o prueba de exhaustividad.

La [procedencia canónica](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/procedencia.md) distingue v1, momentos, referencia aditiva y V3, fija sus fuentes y reconoce la asistencia de IA en revisión, formulación, código, análisis y redacción. El inventario reconciliado de 7.14 separa diseños, parametrizaciones, propuestas y ejecuciones; semillas y dominios no se cuentan como nuevos mecanismos. Es una consolidación retrospectiva y no recupera prerregistro o cegamiento.

La {download}`matriz de requisitos y evidencia <../_static/e3_evidence_close_20261120/contribution/r1_evidence_matrix.csv>` deja cerrada esta consolidación documental y mantiene abiertas las conclusiones que requieren otras evidencias. La modalidad externa de IA autorizada de 6.13 permanece completada, sin fingir una firma humana. La comprensión y defensa del estudiante para R2 siguen pendientes de observación. **La hipótesis empírica de la receta completa no superó su piloto; ello no prueba inexistencia de novedad. En aquel corte R1 seguía sin acreditar, al igual que R3; el dictamen posterior específico de V3 se incorpora en 6.18.**


(e3-r1-substantive-verdict-20261120)=
## 6.18 Revisión específica de R1: adaptación de lectura y pérdida de V3

**Actualización del 20 de noviembre de 2026:** la revisión independiente por IA
emite un **dictamen técnico favorable y acotado para R1 de V3**, no para la
referencia aditiva. Se aplica la modalidad autorizada comunicada por el autor.
El concepto sostiene documentalmente una adaptación para el problema bajo la
definición de la guía; no es una aprobación del profesor, una nota, una defensa
observada de R2 o una acreditación de R3. Los estados de 6.15–6.17 corresponden
a los cortes anteriores a esta revisión específica.

### Contribución concreta y contraste con los antecedentes

Frente a las mezclas neuronales financieras {cite:p}`r1v3_schittenkopf1998identifying`, la
regresión de proxies móviles {cite:p}`r1v3_destefani2017`, la ley de posiciones
y masas aprendidas de FQF {cite:p}`r1v3_yang2019fqf` y el modelo mezcla-GARCH
{cite:p}`r1v3_feng2023`, V3 **aprende una ley del componente aún desconocido,
transforma sus átomos con la parte ya observada y sólo después promedia**.
El ajuste combina error de la media de volatilidad con CRPS de la magnitud
futura. Cambia la lectura y la pérdida respecto a esos métodos; no se reclama
novedad por cambiar el instrumento, la longitud de ventana o un hiperparámetro.

Con los seis retornos disponibles al origen,

$$
m_6=\frac{1}{6}\sum_{i=1}^{6}r_i,\quad
A=\frac{1}{7}\sum_{i=1}^{6}(r_i-m_6)^2,\quad
U=|r_7-m_6|,\quad T_A(u)=\sqrt{A+\frac{6}{49}u^2},
$$ (eq-e3-r1-overlap-identity)

la etiqueta es $Y=\sigma_7=T_A(U)$, **desviación poblacional, `ddof=0`**,
diaria y sin anualizar. U se reconstruye de Y y A sólo como etiqueta; no es
información futura independiente ni una entrada del forward. La red aprende
tres átomos ordenados $a_k$ y masas $p_k$, y predice

$$
\mu_\theta=\sum_{k=1}^{3}p_kT_A(a_k),\qquad
\mathcal L=\mathbb E_{\rm ajuste}\left[
\frac{(\mu_\theta-Y)^2}{s_Y^2}
+\frac{0{,}1}{s_U}\operatorname{CRPS}(G_\theta,U)\right].
$$ (eq-e3-r1-joint-objective)

Las escalas proceden del ajuste interno y el peso 0,1 estaba fijado en el plan.
La {download}`correspondencia con el código <../_static/e3_r1_substantive_20261120/code_theory/mechanism_crosswalk.csv>`
vincula catorce elementos a archivos y líneas del código congelado. Los
1262 parámetros de V3, 1236 de masas fijas y 1210 de la aditiva no se declaran
igualados en capacidad. Los tres controles de componentes ya fueron entrenados
en el piloto conservado; esta revisión no repite sus ajustes.

| Antecedente examinado | Qué ya existe | Diferencia delimitada de V3 |
|---|---|---|
| MDN financiero, Schittenkopf et al. (1998) | Densidad condicional de mezcla y estimación de heterocedasticidad financiera. | Magnitud del componente faltante y mapa de la etiqueta solapada; objetivo supervisado conjunto. |
| De Stefani et al. (2017) | Proxies de desviación móvil y redes NAR/NARX. | Una etiqueta móvil por sí sola no es nueva: V3 modifica la lectura y su actualización distribucional. |
| FQF, Yang et al. (2019) | Ley discreta con posiciones y masas aprendidas. | V3 no reclama esa parametrización; usa U supervisada, mapa por A y MSE/CRPS en vez de Bellman/cuantiles. |
| Feng et al. (2023) | Mezcla neuronal para volatilidad condicional GARCH y objetivo de verosimilitud. | No se equipara su variable a sigma7 solapada; V3 promedia el mapa exacto bajo su ley finita y usa otra pérdida. |
| CRPS y puntuaciones ponderadas | Reglas conocidas; la transformación de variable puede producir un CRPS ponderado. | La adaptación no es un score inédito: fija la variable a puntuar y su acoplamiento con la lectura puntual. |

Las {download}`matrices de distribución <../_static/e3_r1_substantive_20261120/literature_distribution/antecedent_matrix.csv>`,
de {download}`aplicación <../_static/e3_r1_substantive_20261120/literature_application/application_comparison.csv>`
y de {download}`dos primarios recientes <../_static/e3_r1_substantive_20261120/independent_review/recent_primary_comparison.csv>`
declaran las páginas, ecuaciones o metadatos efectivamente inspeccionados.
FQF es un antecedente más cercano que C51 para posiciones y masas aprendidas.
Feng 2023 tiene ahora una lectura focal de su cuerpo, distinta del resumen
registrado en el corte anterior. No se presenta una lectura focal como examen
íntegro de un artículo ni un PDF ilegible como texto leído.

### Objeción de composición conocida y alcance del dictamen

La objeción principal es válida: distribución discreta, transformaciones,
soporte y CRPS son conocidos. La {ref}`sección 7.15 <e3-r1-formal-scope-20261120>`
expone además la equivalencia con CRPS ponderado. La respuesta no consiste
en inventar otro nombre: la guía admite una adaptación sustantiva de lectura,
pérdida o entrenamiento para el problema. La comparación identifica ese
acoplamiento concreto frente a los trabajos cercanos examinados. No demuestra
una clase puntual más expresiva que cualquier aditiva, una regla de scoring
nueva o la ausencia de cualquier antecedente mundial.

La revisión acumula la cobertura histórica efectiva de tres bases y sus
cadenas documentadas, las consultas focales actuales, seguimiento desde cinco
trabajos semilla y dos repositorios inspeccionados en metadata/README/licencia.
Las nuevas llamadas nativas bloqueadas antes de origen no cuentan como
respuestas bibliográficas ni como otra revisión efectiva de tres bases.
El revisor IA repitió dos cadenas específicas de V3 y registró sus resultados.
La {download}`declaración de contribución <../_static/e3_r1_substantive_20261120/contribution_declaration.md>`,
la {download}`matriz final por requisito <../_static/e3_r1_substantive_20261120/r1_evidence_matrix.csv>`
y el {download}`dictamen independiente <../_static/e3_r1_substantive_20261120/independent_review/technical_verdict.md>`
permiten evaluar la recomendación favorable con sus límites y objeciones.

La {download}`procedencia revisada de V3 <../_static/e3_r1_substantive_20261120/procedencia_v3.md>`
actualiza el documento histórico sin alterar sus bytes. Corrige una identidad:
el PDF WU del DOI `10.57938/a6057c38-dbad-42dc-bebc-bd2c71fe537f` es
*Identifying Stochastic Processes with Mixture Density Networks*, WP11/1998,
no el distinto *Volatility Prediction…*, WP15/1998. No se atribuye al segundo
una lectura de ecuaciones del primero. Los antecedentes descubiertos después
no se inventan como lecturas que el autor realizó antes de programar.

### Decisiones que no cambian

El dictamen favorable de novedad es **de V3**. La aditiva conserva su falta de
acreditación sustantiva y sus 796 trabajos; V3 conserva sus treinta ajustes de
desarrollo. `atom_crps` fue **2,70 % peor que `atom_mse_only`** y falló la
heurística del 2 %. No se cambia el umbral, ni se escoge una métrica favorable,
ni se promueve esa ablación, ni se inicia confirmación por este dictamen.
La propiedad formal no exige un teorema inédito o vencer a la aditiva; R3 sí
mantiene sus requisitos propios de comparación. La comprensión del estudiante
se evalúa por separado. El {download}`material del autor <../_static/e3_r1_substantive_20261120/AUTHOR_ACTIONS.md>`
deja preparada la defensa y la aclaración del bosquejo Faithful, sin inventar
un registro histórico ni volver a contar semillas como diseños.
