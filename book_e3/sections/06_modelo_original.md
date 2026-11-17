# 6 Bloque C El modelo original

La referencia evaluada es **CNN causal → LSTM unidireccional → atención temporal por contenido, con salida aditiva positiva**, de **1210 parámetros**. Sus diez semillas congeladas obtuvieron una RMSE media individual de **0,0029394324** sobre las 608 fechas históricas comunes. La [comparación estadística posterior](08_verificacion_estadistica.md) ya está calculada frente a los 16 rivales de E2; sus resultados no acreditan por sí solos R1, R2 o R3.

Los pilotos v1, v2 y por momentos se conservan como historia de desarrollo. La RMSE **0,0031446147843954175** corresponde a la validación de una semilla aditiva de v2, no a la media de prueba de las diez réplicas. La receta half-t5 con MSE y NLL y el posterior piloto por momentos no superaron sus heurísticas de continuación. La cabeza de momentos no es la arquitectura de la comparación actual ni se ha seleccionado como aporte original. La sección 6.5 conserva la formulación del primer mecanismo y 6.7 la propuesta por momentos; 6.8 identifica la referencia efectivamente evaluada.

Las evidencias completas se encuentran en los anexos: [revisión del piloto v1, B](17_revision_post_piloto.md), [ficha v2, C](18_ficha_cnn_lstm_atencion_v2.md), [ejecución v2, D](19_piloto_innovacion_v2.md), [cierre v2, E](20_cierre_v2.md), [formulación por momentos, F](21_propuesta_momentos.md) y [dictamen de admisibilidad, G](22_dictamen_momentos.md).

## 6.1 Relación con el paper guía de E2

La continuidad elegida conserva D7×5 OHLCV, horizonte de un día y objetivo $\sigma_7$ poblacional sin anualizar. El primer mecanismo v1 añadió penalización de discordancia respecto a un promedio causal y saturación radial de los valores agregados. Ese diseño tiene formulación, código y una propiedad latente demostrada, pero su piloto no mejoró al control M00.

El paper guía sigue siendo TimeMixer. Su limitación de calendario medida en ETTh1 no se traslada automáticamente al adaptador BTC sin esos marcadores. El estrés artificial de E2 motivó investigar sensibilidad, pero puede romper coherencia OHLCV y no prueba que la regla nueva resuelva un defecto económico real. La [sección 7](07_diseno_riguroso.md) distingue esa hipótesis inicial de los resultados posteriores y del diseño aún pendiente de confirmación.

La propuesta por momentos de la sección 6.7 estudia otra limitación: una escala paramétrica compartida puede tener óptimos distintos para la NLL auxiliar y para la media de la transformación objetivo bajo mala especificación. Los contraejemplos artificiales demuestran esa posibilidad, sin atribuirla como causa ya identificada del resultado de BTC ni como un defecto nuevo de TimeMixer.

## 6.2 Condiciones de originalidad y admisibilidad

| Condición | Evidencia actual | Estado que debe conservarse |
|---|---|---|
| R1 Novedad para el fenómeno y la tarea | Revisiones focales del mecanismo de atención y de la cabeza de momentos; sus componentes conocidos se atribuyen. El dictamen no selecciona todavía la adaptación por momentos como aporte original. | Pendientes la revisión bibliográfica completa, la diferencia sustantiva y la verificación humana externa; no se declara novedad ni ausencia de novedad demostradas. |
| R2 Núcleo algorítmico propio y defendible | Código del primer mecanismo, ejecutores v1/v2 y cabeza neuronal de momentos desde la formulación, con pruebas artificiales y asistencia de IA declarada. | La implementación y sus pruebas no acreditan por sí solas comprensión, autoría responsable y defensa del estudiante; esa verificación humana sigue pendiente. |
| R3 Superioridad estadísticamente verificada | Pilotos de desarrollo y comparación posterior completa frente a 16 rivales: los 48 contrastes primarios rechazan tras Holm, con ámbitos temporal y de inicialización distintos. | No se acredita R3: calendario conocido, ajuste de E2 con 655 ejemplos frente a 532 en E3, potencia no dimensionada y conjuntos adicionales pendientes; la pertinencia de las pruebas y sus supuestos debe justificarse para la afirmación F2a. |

CNN, LSTM, atención por contenido, consultas aprendidas y saturación no se reclaman como invenciones. Tampoco se reclaman nuevas las desigualdades de momentos, los scores de media o las guardas de salida. La contribución candidata debe diferir sustantivamente en mecanismo, pérdida o entrenamiento; cambiar datos o hiperparámetros no basta. Las variantes funcionales y los resultados de desarrollo se conservan sin convertirlos en innovaciones acreditadas.

## 6.3 Revisión de literatura y evidencia de novedad


**Antecedentes del primer mecanismo de atención.** Se examinaron focalmente los originales de [Kim, Papamakarios y Mnih (2021)](https://proceedings.mlr.press/v139/kim21i.html) y [Dasoulas, Scaman y Virmaux (2021)](https://proceedings.mlr.press/v139/dasoulas21a.html). Ambos son antecedentes directos de atención y propiedades Lipschitz. Una consulta aprendida y una cota, por separado, ya tienen antecedentes. La matriz siguiente conserva el alcance histórico de ese primer corte.

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

Las revisiones posteriores registran doce consultas web iniciales y ocho adicionales; no equivalen a búsquedas nativas completas. Los tres intentos API de esa etapa no recibieron respuesta bibliográfica y sus conteos son desconocidos. Siguen pendientes la cobertura exigida en §6.3 de la guía, seguimiento por citas, repetición antes de entrega y verificación humana externa. El [paquete de revisión del anexo G](22_dictamen_momentos.md) está preparado, sin diligenciar ni firmar. No identificar la composición exacta en una revisión focal no prueba su ausencia.

## 6.4 Ciclo de desarrollo

| Etapa | Evidencia disponible | Condición siguiente |
|---|---|---|
| Revisión y hueco | Primer mecanismo de atención y propuesta posterior por momentos, con antecedentes y límites documentados. | Completar revisión y verificación externa antes de una declaración de novedad. |
| Formulación | Cota latente del primer módulo y teoría condicional de momentos con sus supuestos; contraejemplos a garantías no demostradas. | Justificar una contribución sustantiva y su alcance sin convertir la prueba en promesa de desempeño. |
| Implementación y desarrollo | Cinco recetas entrenadas entre v1/v2 y tres recetas completadas en el piloto por momentos; diez réplicas posteriores de la receta aditiva congelada. | Repetir semillas de una receta no equivale a proponer diseños nuevos ni completa la defensa académica. |
| Pruebas y auditoría | Pruebas de software y auditorías reproducibles, incluidas las comprobaciones artificiales del nuevo modelo, reanudación y guardado. Las 35 pruebas del primer módulo conservan su alcance histórico. | El software y los cálculos artificiales no sustituyen evaluación confirmatoria, autoría defendida o revisión humana. |
| Selección de desarrollo | La cabeza aditiva v2 obtiene la menor RMSE global del piloto; half-t5 con auxiliares y el piloto por momentos no superan sus heurísticas. La aditiva se replica sin nueva selección. | Conservarla como referencia, sin promoverla a aporte original o ganador único. |
| Diseño y confirmación | Resultados históricos, diez réplicas aditivas y análisis posterior completo de la familia conservados; no hay preregistro confirmatorio de estos resultados. | Cerrar comparación justa, potencia y conjuntos adicionales para una confirmación futura. |

La [sección 7](07_diseno_riguroso.md) integra método, pilotos, propiedades y controles. Las cinco recetas de v1/v2 y las tres del piloto por momentos forman un subtotal histórico de ocho; las diez semillas aditivas repiten una receta. El **$N_d$ global sigue pendiente de conciliación** de propuestas, parametrizaciones, variantes y ensayos y no se reduce a ese subtotal. Los conjuntos adicionales todavía no se han elegido y el análisis posterior no se presenta como preregistro confirmatorio.

## 6.5 Primer mecanismo implementado: formulación y reproducción

Esta sección conserva la formulación histórica de penalización y saturación del primer módulo. Sus constantes, conteo de parámetros, pruebas y descargas pertenecen a ese código; la implementación posterior de momentos se distingue en 6.7 y la receta de referencia v2 en el apartado 7.5.

El encoder usa convolución con relleno sólo histórico, activación tanh y LSTM unidireccional. Recibe ventanas ya estandarizadas y produce estados $H=(h_1,\ldots,h_7)$. Una matriz fija $P$, independiente de $H$, calcula promedios causales. La cabeza implementa

$$
d_j=\sqrt{\|h_j-(PH)_j\|_2^2+\varepsilon^2},\qquad
a=\operatorname{softmax}\bigl((Hq-\lambda d)/\tau\bigr),
$$

$$
v_j=\frac{Bh_j}{\sqrt{B^2+\|h_j\|_2^2}},\qquad
f(H)=\operatorname{softplus}\left(w^\top\sum_j a_jv_j+b\right).
$$

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

Aquí las condiciones siguen abiertas. Una demostración de la cabeza y pruebas de software no convierten R1 o R3 en aprobadas ni certifican la defensa académica del autor. Las limitaciones y resultados adversos deberán conservarse.

## 6.7 Propuesta posterior: cabeza de momentos y dictamen vigente

La propuesta conserva el mismo codificador y atención por contenido. Con los seis retornos conocidos al origen, define

$$
A=\frac17\sum_{i=t-6}^{t-1}(r_i-\bar r_6)^2,\qquad
U=|r_t-\bar r_6|,\quad V=U^2,\quad
Y=\sigma_7(t)=\sqrt{A+cV},\quad c=6/49.
$$

`A` se calcula con historia observable; `U,V,V²` son etiquetas de entrenamiento, nunca entradas disponibles al pronosticar. Tres lecturas escalares del contexto producirían

$$
m=s_u^2\operatorname{softplus}(a),\qquad
q=m^2(1+b^2),\qquad \rho=\frac{d^2}{1+d^2},
$$

$$
L=\sqrt A+\frac{cm}{\sqrt{A+cq/m}+\sqrt A},\qquad
H=\sqrt{A+cm},\qquad
\widehat\mu=L+\rho(H-L).
$$

Aquí `m,q` representan momentos de `V`; este `q` no es la consulta de atención de la sección 6.5. Con momentos verdaderos y positivos, la región realizable de la media es `[L,H)` cuando `q>m²`; `L` se alcanza y `H` es supremo no alcanzado. Si `q=m²`, ambas cotas coinciden. Con momentos estimados, la región calculada puede no contener la media verdadera: no es un intervalo de predicción ni garantiza menor MSE.

El criterio propuesto suma MSE de $Y$ y scores conocidos de media para `V,V²`, con escalas calculadas sólo en ajuste. La demostración de consistencia condicional exige `E[V|X]>0`, `E[V²|X]<∞` y una clase que represente conjuntamente los valores verdaderos; no demuestra convergencia de esta red finita, fidelidad de trayectoria o superioridad predictiva. El caso verdadero de momentos nulos queda fuera del teorema logarítmico. La formulación, bordes y verificación artificial completos están en el [anexo F](21_propuesta_momentos.md).

El [anexo G](22_dictamen_momentos.md) demuestra dos límites: momentos estimados incorrectos pueden excluir la media verdadera, y los auxiliares pueden empeorar un paso escalar SGD frente a la misma cabeza con MSE sola. Son cálculos artificiales, no una simulación del Adam futuro ni evidencia de fracaso en BTC. El control que faltaba en el primer diseño —**la misma cabeza por momentos sin pérdidas auxiliares**— se incluyó en el piloto completado; una media directa con igual número de parámetros responde otra pregunta.

**Cierre posterior al dictamen:** se conserva la adaptación experimental sin seleccionarla como aporte original. El piloto solicitado para comprobar su funcionamiento ya completó las tres recetas; su [resumen inmutable](https://github.com/jassan-bit/DeepLearning_202630/blob/4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4/runs/momentos_v1/summary.json) registra RMSE **0,0032589201** para `moment_joint`, frente a **0,0031446148** de la referencia aditiva congelada, sobre la validación de desarrollo. No superó la heurística del 2 %. La revisión sustantiva y humana sigue pendiente; este resultado no demuestra ausencia de novedad ni explica por sí solo el mecanismo.

La implementación en [src/e3_moment_model.py](https://github.com/jassan-bit/DeepLearning_202630/blob/main/src/e3_moment_model.py) tiene **1236 parámetros**, contados sobre el módulo: 1197 del codificador y atención, más tres lecturas activas de 13 parámetros. Se compararon `moment_joint`, `moment_mean_only` y `direct_moment_aux`; la referencia aditiva v2 se reutilizó mediante su CSV congelado, con 1210 parámetros. El apartado 7.8 actualiza el cierre; el anexo F.6 conserva las instrucciones históricas. Estas tres recetas no forman parte de las diez réplicas aditivas de la comparación actual.

## 6.8 Referencia evaluada y alcance de la evidencia posterior

La denominación de los archivos, `CNN-LSTM-atencion-aditiva E3`, identifica la salida aditiva positiva de **1210 parámetros**, con atención temporal por contenido. No identifica la penalización y saturación de 6.5 ni la cabeza de momentos de 1236 parámetros. Se mantienen las diez semillas `271801`–`271810`; la semilla histórica de desarrollo `20261116` queda fuera de esa media y no se elige una réplica por su error de prueba.

E3 conserva **532 ejemplos de ajuste y 116 de validación**, sin usar las 53 fechas de calibración de E2. Los 13 rivales neuronales de E2 tienen diez semillas cada uno; Persistence, SeasonalNaive y ARIMA conservan una trayectoria determinista por sistema: **133 predictores de E2**, además de las diez réplicas de E3. Todos se puntúan sobre las mismas **608 fechas**, del 1 de enero de 2025 al 31 de agosto de 2026. E2 se ajustó con 655 ejemplos y calibró con 53; no se afirma igualdad de tratamiento o presupuesto.

Las [métricas descargables](../_static/e3_estadistica_posterior_20261117/metrics.csv) y los [48 contrastes primarios](../_static/e3_estadistica_posterior_20261117/primary_contrasts.csv) respaldan el resultado posterior: los tres procedimientos rechazan para cada uno de los 16 rivales tras Holm en la familia de 48. Dos procedimientos estudian incertidumbre temporal condicionada a trayectorias ajustadas y otro la inicialización condicionada al calendario; no se presentan como tres confirmaciones temporales equivalentes. El MCS auxiliar conserva diez sistemas, incluida la aditiva, y no identifica un ganador único. La [sección 8](08_verificacion_estadistica.md) expone supuestos, sensibilidades y límites; R1, R2 y R3 siguen sin acreditarse.
