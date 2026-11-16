# 7 Diseño riguroso del modelo original

El mecanismo candidato y su implementación se describen en el [Bloque C](06_modelo_original.md). Este capítulo organiza su diseño según la Sección 7 de la guía: limitación, hipótesis, disciplina experimental, propiedad teórica y ablaciones. La implementación y los ensayos de software existen; todavía no se han ejecutado HPO, entrenamiento o comparaciones del candidato sobre datos de mercado. R1 y R3 siguen abiertos, y la comprensión y defensa académica de R2 deben verificarse con el autor.

## 7.1 Del hueco de investigación a la hipótesis contrastable

**D1. Limitación identificada y evidencia disponible.** El [estudio de robustez de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/54_robustez_interpretabilidad.html) ejecutó un estrés que desplaza el 10 % de las celdas de entrada en ±5 desviaciones estándar. Bajo esa condición, la RMSE aumentó un 585,40 % en el SAE, un 513,98 % en GRU32 y un 1.552,52 % en TimeMixer. Es evidencia de sensibilidad bajo una corrupción artificial concreta, observada retrospectivamente en el benchmark. No demuestra fragilidad ante cualquier error de mercado: las ventanas perturbadas pueden incumplir la coherencia OHLCV. La limitación operativa de calendario de TimeMixer en ETTh1 tampoco se traslada al adaptador BTC sin marcadores de calendario.

**D2. Fuente y diferencia candidata del mecanismo.** Se conserva el codificador CNN causal → LSTM unidireccional y se estudia una regla de atención que combina penalización de discordancia respecto a una referencia causal fija con saturación radial de valores. La consulta por contenido, las CNN, las LSTM y la estabilidad Lipschitz tienen antecedentes. Además, la saturación usada es el gradiente de una función pseudo-Huber radial. La [formulación y la revisión focal](06_modelo_original.md) distinguen esos componentes conocidos de la combinación que se investiga. Las dos modificaciones candidatas no se presentan como una invención acreditada: faltan cerrar la revisión de novedad, las equivalencias con trabajos cercanos y su verificación externa.

**D3. Hipótesis del mecanismo, propuesta para desarrollo.** Si la penalización y la saturación reducen la influencia de estados discordantes por el mecanismo previsto, el modelo completo deberá reducir la RMSE frente al mismo codificador con atención convencional de contenido en un conjunto de estrés $S$ definido antes de evaluar, sin un deterioro limpio superior al margen heredado. Para BTCUSDT se conserva

$$
\delta=0{,}00025772842664981106
$$

en unidades originales de $\sigma_7$; no se recalcula con el resultado del candidato. La propuesta exige una ganancia de al menos $\delta$ en $S$. Quedaría refutada si no se observa ese beneficio bajo el diseño fijado o si el deterioro limpio excede el margen. Mejorar sólo la robustez no satisface la comparación principal de R3 frente al Top 10 y todos los baselines fuertes.

**D4. Predicciones secundarias que deben contrastarse.** Se proponen antes de los experimentos finales:

- Al retirar la saturación, debería aumentar la sensibilidad a perturbaciones que producen estados latentes grandes. El ejemplo analítico del [anexo técnico](09_modelo_original.md) muestra ese efecto en un caso artificial, no en BTC.
- Al retirar la penalización, debería reducirse la ventaja atribuida específicamente a estados discordantes. Si el efecto permanece igual, debe revisarse la explicación del papel de ese componente.
- La mejora debería mantenerse bajo alteraciones plausibles que preserven la coherencia OHLCV, además del estrés artificial ya conocido. Los tipos, severidades y subconjuntos se fijarán antes de medirlos.
- Los saltos genuinos y la cobertura en la cola alta no deberían degradarse de manera incompatible con la hipótesis. Una ganancia media acompañada de supresión de extremos obliga a revisar el diseño y sus límites.

**D5. Éxito y fracaso antes de la evaluación definitiva.** Como criterio de desarrollo propuesto, sean $\Delta_S=\operatorname{RMSE}_{\rm base}(S)-\operatorname{RMSE}_{\rm completo}(S)$ y $D_0=\operatorname{RMSE}_{\rm completo}(\text{limpio})-\operatorname{RMSE}_{\rm base}(\text{limpio})$. El criterio a someter a cierre previo es que el límite inferior del IC95 % de $\Delta_S$ supere $\delta$ y que el límite superior del IC95 % de $D_0$ quede por debajo de $\delta$, con inferencia temporal y control familiar definidos en el protocolo. Los umbrales de los análisis de cola también deben fijarse antes de conocer sus resultados.

No cumplir esas condiciones impide declarar éxito de esta hipótesis; un intervalo amplio se reportaría como no concluyente, sin confundirlo con equivalencia. La hipótesis, los IC anteriores y los criterios secundarios son **propuestas sin ejecución ni preregistro**. Antes de confirmar se deben cerrar el estimando, el remuestreo, las familias y las reglas de decisión. R3 requerirá además sus condiciones completas de relevancia, adecuación y superioridad frente a cada rival; no se deduce de una comparación favorable con el modelo base.

## 7.2 Disciplina de diseño: evitar el sobreajuste al conjunto de validación

La [bitácora de desarrollo](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/design_log.md) conserva la familia completa y tres intervenciones funcionales de software: sin penalización, sin saturación y sin contenido. Las cuatro variantes se examinaron con entradas artificiales; no se eligió una receta por su RMSE en datos de mercado. El corte informa **$N_{d,\rm mercado}=0$**, que no equivale a ausencia de esfuerzo de diseño. Hay **cuatro variantes funcionales especificadas y comprobadas**. Antes del protocolo se deberá conciliar y declarar el $N_d$ total, incluyendo variantes descartadas, intentos y fallos, con su fecha, evidencia, métricas disponibles y decisión.

El ajuste diminuto de una prueba unitaria realizó 350 pasos de Adam sobre ocho ejemplos artificiales. Comprueba capacidad de ajuste del software; no constituye entrenamiento de BTC, validación del mecanismo económico ni selección de hiperparámetros con los datos de E1/E2.

La separación de datos para las siguientes fases debe conservar estos roles:

| Uso | Decisiones permitidas | Límite |
|---|---|---|
| Desarrollo | Formular el mecanismo, probar variantes y fijar espacios, hipótesis y criterios. | Registrar todo el esfuerzo y el sesgo de selección; no presentar este conjunto como confirmación independiente. |
| Confirmación | Aplicar el protocolo final con arquitectura, pérdida, ajustes y reglas ya cerrados. | No cambiar el diseño después de observar sus resultados. |
| Tres conjuntos adicionales | Evaluar la transferencia de la misma tarea bajo el procedimiento prefijado. | **Todavía no se han elegido.** Deben seleccionarse antes de conocer el desempeño y no usarse para rediseñar el mecanismo. |

Las 608 fechas de enero de 2025 a agosto de 2026 ya se evaluaron en E1/E2; la reserva de septiembre–octubre de 2026 también tiene antecedentes publicados. Repetir esos periodos no recupera cegamiento. El [contrato heredado](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) conserva su procedencia; la confirmación de E3 requiere documentar la exposición previa y una separación real de las decisiones de diseño.

Antes de la campaña definitiva deben quedar cerrados la arquitectura y la pérdida, la revisión del mecanismo, las particiones y fuentes, los tres datasets adicionales, el espacio y presupuesto de ajuste comparable, el piloto de potencia, las semillas y el número de ejecuciones, las hipótesis y familias, el $N_d$ y la bitácora. El número final será al menos diez semillas o el mayor requerido por potencia; esta organización del informe no lo dimensiona ni ejecuta.

El congelamiento debe registrarse con fecha y tag Git reales anteriores a los experimentos definitivos. Un cambio posterior de arquitectura, pérdida o espacio de búsqueda reinicia el protocolo con nueva versión y conserva el historial. El esfuerzo de diseño del candidato se reportará separado del presupuesto de HPO de cada rival, sin ocultar una asimetría bajo el mismo número de configuraciones.

## 7.3 Sustento teórico: qué demostrar y cómo verificarlo

La propiedad disponible es una cota Lipschitz de la **cabeza sobre estados latentes**. La [formulación del Bloque C](06_modelo_original.md) y la {download}`demostración íntegra con derivadas y pseudocódigo <../_static/e3_bloque_c_20261115/theory.md>` conservan su desarrollo. La demostración siguiente es autocontenida; no extiende la garantía al preprocesamiento, a la CNN o a la LSTM.

Sean $H,H'\in\mathbb R^{n\times d}$, con $n=7$ en el problema principal. Se usan norma euclídea, norma de Frobenius y norma espectral para vectores, estados y operadores, respectivamente. Sean $P$ una matriz causal fija e independiente de $H$, $A=I-P$ y parámetros de inferencia fijos tales que

$$
B>0,\quad\varepsilon>0,\quad\|q\|_2\le Q,\quad\|w\|_2\le W,
\quad0\le\lambda\le\Lambda,\quad\tau\ge\tau_{\min}>0.
$$

El mecanismo es

$$
r_j=(AH)_j,\quad d_j=\sqrt{\|r_j\|_2^2+\varepsilon^2},\quad
s_j=q^\top h_j-\lambda d_j,\quad a=\operatorname{softmax}(s/\tau),
$$

$$
v_j=\frac{B h_j}{\sqrt{B^2+\|h_j\|_2^2}},\qquad
c=\sum_{j=1}^n a_jv_j,\qquad
f(H)=\operatorname{softplus}(w^\top c+b).
$$

En el código, las restricciones se cumplen mediante $q=Qu_q/\sqrt{1+\|u_q\|_2^2}$, $w=Wu_w/\sqrt{1+\|u_w\|_2^2}$, $\lambda=\Lambda\operatorname{sigmoid}(\theta_\lambda)$ y $\tau=\tau_{\min}+\operatorname{softplus}(\theta_\tau)$. $B$ y $P$ son fijos. Las constantes se guardan en la configuración; no son supuestos de norma que se atribuyan a pesos sin controlar.

**Lema 1: valores acotados y transformación 1-Lipschitz.** Para $T_B(h)=Bh/u$, $u=\sqrt{B^2+\|h\|_2^2}$,

$$
J_{T_B}(h)=\frac B u I-\frac B{u^3}hh^\top.
$$

Sus autovalores son $B/u$ en direcciones ortogonales a $h$ y $B^3/u^3$ en la dirección de $h$; en $h=0$ el jacobiano es $I$. Están en $(0,1]$. Integrando sobre el segmento entre dos puntos, $\|T_B(h)-T_B(h')\|_2\le\|h-h'\|_2$. Además, $\|T_B(h)\|_2\le B$. Para la matriz $V$ de filas $v_j^\top$,

$$
\|V(H)-V(H')\|_F\le\|H-H'\|_F,
\qquad\|V(H)\|_2\le\|V(H)\|_F\le B\sqrt n.
$$

**Lema 2: sensibilidad de los puntajes.** La igualdad $d_j(H)=\|(r_j,\varepsilon)\|_2$ y la desigualdad triangular inversa implican $|d_j(H)-d_j(H')|\le\|r_j-r'_j\|_2$. Sumando cuadrados por filas,

$$
\|d(H)-d(H')\|_2\le\|A(H-H')\|_F
\le\|A\|_2\|H-H'\|_F.
$$

La parte de contenido cumple $\|(H-H')q\|_2\le Q\|H-H'\|_F$. Por tanto,

$$
\|s(H)-s(H')\|_2
\le\bigl(Q+\Lambda\|A\|_2\bigr)\|H-H'\|_F.
$$

**Lema 3: softmax es $1/2$-Lipschitz en norma euclídea.** Su jacobiano es $J=\operatorname{diag}(a)-aa^\top$. La suma absoluta de la fila $j$ es $2a_j(1-a_j)\le1/2$. Como $J$ es simétrico, $\|J\|_2\le\|J\|_\infty\le1/2$. Integrando el jacobiano y aplicando el Lema 2,

$$
\|a(H)-a(H')\|_2
\le\frac{Q+\Lambda\|A\|_2}{2\tau_{\min}}\|H-H'\|_F.
$$

**Teorema.** Bajo los supuestos anteriores, para cualesquiera estados finitos $H,H'$ y los mismos parámetros de inferencia,

$$
|f(H)-f(H')|\le C_{\rm head}\|H-H'\|_F,
\qquad
C_{\rm head}=W\left[1+\frac{B\sqrt n}{2\tau_{\min}}
\left(Q+\Lambda\|I-P\|_2\right)\right].
$$

**Demostración.** Al escribir $c(H)=V(H)^\top a(H)$,

$$
c(H)-c(H')=V(H)^\top\bigl(a(H)-a(H')\bigr)
+\bigl(V(H)-V(H')\bigr)^\top a(H').
$$

Como $a(H')$ es una distribución, $\|a(H')\|_2\le1$. Los lemas dan

$$
\begin{aligned}
\|c(H)-c(H')\|_2
&\le B\sqrt n\,\|a(H)-a(H')\|_2+\|V(H)-V(H')\|_F\\
&\le\left[1+\frac{B\sqrt n}{2\tau_{\min}}
\left(Q+\Lambda\|I-P\|_2\right)\right]\|H-H'\|_F.
\end{aligned}
$$

Softplus es 1-Lipschitz porque su derivada es sigmoid, entre cero y uno. La proyección tiene norma $\|w\|_2\le W$ y el sesgo fijo se cancela al tomar diferencias. La composición demuestra la cota. $\square$

La prueba usa aritmética real y puede ser conservadora. En unidades originales de $\sigma_7$, la constante se multiplica por la media de entrenamiento $a_{\rm tr}>0$. Obtener una cota desde OHLCV crudo exigiría demostrar también las constantes del codificador y de sus transformaciones. Esas constantes no se han cuantificado aquí; tampoco se demuestra mejor RMSE, calibración o R3.

Como $c$ es combinación convexa de vectores de norma a lo sumo $B$, también se obtiene

$$
\operatorname{softplus}(b-WB)\le f(H)\le\operatorname{softplus}(b+WB).
$$

Ese techo de salida puede perjudicar extremos genuinos y debe evaluarse. El promedio causal actual incluye el estado presente: en la primera posición la discordancia es $\varepsilon$, otra asimetría del diseño que debe examinarse. La atención normalizada usa toda la ventana observada y no es causal por prefijos de sus pesos; la predicción sólo es causal respecto al día objetivo si el pipeline respeta la disponibilidad temporal.

**Verificación ejecutada.** El {download}`informe guardado <../_static/e3_bloque_c_20261115/software_verification.json>` y el [anexo técnico](09_modelo_original.md) conservan código, configuración, ambiente y hashes. Los resultados corresponden a estados y ventanas artificiales:

| Comprobación | Resultado |
|---|---:|
| Casos pytest del módulo | 35 aprobados, ninguno omitido |
| Pares latentes examinados, float64 | 256 |
| Escalas de estados | 0,001 a 1.000 |
| Cota $C_{\rm head}$ de la configuración registrada | 46,15606644 |
| Mayor cociente de diferencias observado | 0,11516301 |
| Violaciones observadas de la cota | 0 |
| Error máximo de derivada manual frente a autograd | $6{,}939\times10^{-18}$ |

La ausencia de violaciones en 256 pares apoya la implementación; la afirmación universal procede de la demostración. El caso analítico del anexo ilustra una consecuencia observable de acotar valores, pero **todavía falta comprobar un beneficio predictivo en datos reales de desarrollo**. Las pruebas de gradientes, formas, causalidad del encoder, sobreajuste diminuto y serialización tampoco sustituyen esa evidencia.

## 7.4 Diseño de la ablación

Hay dos modificaciones algorítmicas candidatas: penalización de discordancia y saturación radial. Se propone un factorial completo $2\times2$ para separar sus aportes e interacción, con el mismo codificador y atención de contenido. Esta organización del informe no agrega código, entrenamientos o resultados.

| Modelo $M_{uv}$ | Penalización $u$ | Saturación $v$ | Relación con el código disponible | Estado experimental |
|---|---:|---:|---|---|
| $M_{00}$, base convencional | 0 | 0 | Requiere desactivar ambos componentes simultáneamente; **todavía no está implementado** como variante. | Propuesto; sin métricas de mercado. |
| $M_{10}$, sólo penalización | 1 | 0 | `without_saturation`. | Intervención funcional comprobada con datos artificiales. |
| $M_{01}$, sólo saturación | 0 | 1 | `without_penalty`. | Intervención funcional comprobada con datos artificiales. |
| $M_{11}$, completo | 1 | 1 | `full`. | Software y propiedad latente verificados; sin entrenamiento de mercado. |
| Control adicional sin contenido | 1 | 1 | `without_content` fuerza $q=0$. | Intervención funcional comprobada; no forma parte del factorial de los dos cambios. |

El control sin contenido estudia un componente convencional y **no es una tercera innovación**. Las tres ablaciones funcionales existentes no equivalen a haber completado el factorial, ni a la ablación estadística exigida. El modelo base debe implementarse y verificarse en una revisión posterior antes de ejecutar ese diseño, conservando su versión y trazabilidad.

Para errores $E_{uv}$ sobre unidades comparables, la interacción puede expresarse como

$$
I=E_{11}-E_{10}-E_{01}+E_{00}.
$$

Un $I<0$ indicaría una reducción de error conjunta mayor que la suma aditiva de sus reducciones, bajo esta codificación; no se ha estimado $I$. El análisis deberá reportar efectos principales e interacciones con IC, mediante un modelo o remuestreo compatible con la dependencia temporal y las repeticiones. Los contrastes contra el modelo completo usarán las pruebas adecuadas de la Sección 8 de la guía y Holm dentro de una familia de ablación separada de la comparación principal.

Se mantendrán particiones, preprocesamiento, semillas pareadas, política de ajuste, épocas y presupuesto comparables; el esfuerzo real y los fallos se registrarán. Las semillas no se cuentan como mercados independientes. Se reportarán parámetros totales y activos: retirar una penalización puede eliminar su parámetro, y retirar contenido elimina $q$. Cuando cambie el número de parámetros se incluirá una variante de tamaño igualado para separar el efecto del mecanismo del efecto de la capacidad, y se discutirá esa comparación.

Las ablaciones pueden apoyar que un componente contribuye bajo el diseño evaluado; no prueban por sí solas el mecanismo causal. Sus resultados deberán contrastarse con las predicciones D4 y conservar también los efectos nulos, adversos y la posible pérdida de desempeño en extremos.
