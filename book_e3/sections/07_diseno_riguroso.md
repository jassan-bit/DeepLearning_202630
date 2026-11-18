# 7 Diseño riguroso del modelo original

El mecanismo y su estado de admisibilidad se describen en el [Bloque C](06_modelo_original.md). Este capítulo integra el diseño inicial de atención, los pilotos de desarrollo completados y la posterior réplica de la **salida aditiva positiva de 1210 parámetros** en diez semillas. La [sección 8](08_verificacion_estadistica.md) presenta la comparación estadística posterior ya calculada. R1 y R3 siguen abiertos; la comprensión y defensa académica de R2 deben verificarse con el autor. Ese análisis no convierte el diseño en un protocolo confirmatorio.

Este capítulo aplica la sección 7 de la guía: identificar una limitación, formular una explicación refutable, registrar las decisiones y contrastar teoría y ablaciones. Los apartados 7.1–7.4 distinguen los diseños históricos de la referencia evaluada; 7.5–7.8 conservan los pilotos y su cierre, 7.9 identifica la comparación posterior vigente y 7.10 reúne la revisión de requisitos. Los anexos [B](17_revision_post_piloto.md) a [G](22_dictamen_momentos.md) conservan el detalle técnico y las fechas de esas decisiones.

| Arquitectura | Pregunta de diseño documentada | Alcance de la evidencia |
|---|---|---|
| Primer módulo, 1211 parámetros | Penalizar discordancia y saturar los valores de atención para limitar su sensibilidad. | D1–D5 y cota latente históricas; el piloto global fue desfavorable frente a M00. |
| Referencia aditiva, 1210 parámetros | Sumar un exceso positivo al piso observable del objetivo, conservando atención por contenido. | Diez réplicas y análisis posterior completos. Su buen rendimiento no demuestra que el cambio de salida sea una contribución nueva ni que explique por sí solo la ganancia. |
| Cabeza de momentos, 1236 parámetros | Relacionar momentos de una innovación con la media de la transformación objetivo. | Teoría condicional y tres recetas de desarrollo completadas; no superó la heurística de continuación. |

## 7.1 Del hueco de investigación a la hipótesis contrastable

**Origen del primer diseño de atención.** Las hipótesis D1–D5 siguientes motivaron penalización y saturación. Los pilotos posteriores no las convierten en un protocolo confirmatorio ni en una declaración de superioridad. Su resultado de desarrollo se integra en 7.5.

**D1. Limitación identificada y evidencia disponible.** El [estudio de robustez de E2](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/54_robustez_interpretabilidad.html) ejecutó un estrés que desplaza el 10 % de las celdas de entrada en ±5 desviaciones estándar. Bajo esa condición, la RMSE aumentó un 585,40 % en el SAE, un 513,98 % en GRU32 y un 1.552,52 % en TimeMixer. Es evidencia de sensibilidad bajo una corrupción artificial concreta, observada retrospectivamente en el benchmark. No demuestra fragilidad ante cualquier error de mercado: las ventanas perturbadas pueden incumplir la coherencia OHLCV. La limitación operativa de calendario de TimeMixer en ETTh1 tampoco se traslada al adaptador BTC sin marcadores de calendario.

**D2. Fuente y diferencia candidata del mecanismo.** Se conserva el codificador CNN causal → LSTM unidireccional y se estudia una regla de atención que combina penalización de discordancia respecto a una referencia causal fija con saturación radial de valores. La consulta por contenido, las CNN, las LSTM y la estabilidad Lipschitz tienen antecedentes. Además, la saturación usada es el gradiente de una función pseudo-Huber radial. La [formulación y la revisión focal](06_modelo_original.md) distinguen esos componentes conocidos de la combinación que se investiga. Las dos modificaciones candidatas no se presentan como una invención acreditada: faltan cerrar la revisión de novedad, las equivalencias con trabajos cercanos y su verificación externa.

No se ha cerrado una transferencia desde una aplicación de origen concreta. Citar antecedentes de atención y estabilidad no demuestra por sí solo esa transferencia ni su adaptación sustantiva. Si se defiende una transferencia, deberán identificarse la fuente exacta, la propiedad o estructura compartida que permite trasladarla y el cambio algorítmico requerido en BTC. Si se defiende una combinación propia, deberá demostrarse su diferencia frente a los trabajos cercanos. La revisión pendiente no se completa asignando una procedencia retrospectiva por conveniencia.

**D3. Hipótesis del mecanismo, propuesta para desarrollo.** Si la penalización y la saturación reducen la influencia de estados discordantes por el mecanismo previsto, el modelo completo deberá reducir la RMSE frente al mismo codificador con atención convencional de contenido en un conjunto de estrés $S$ definido antes de evaluar, sin un deterioro limpio superior al margen heredado. Para BTCUSDT se conserva

$$
\delta=0{,}00025772842664981106
$$ (eq-e3-design-practical-margin)

en unidades originales de $\sigma_7$; no se recalcula con el resultado del candidato. La propuesta exige una ganancia de al menos $\delta$ en $S$. Quedaría refutada si no se observa ese beneficio bajo el diseño fijado o si el deterioro limpio excede el margen. Mejorar sólo la robustez no satisface la comparación principal de R3 frente al Top 10 y todos los baselines fuertes.

La explicación causal propuesta es que la penalización reduzca el peso de estados discordantes y la saturación limite la amplitud de los valores que éstos pueden aportar. La cota demuestra una restricción matemática de sensibilidad, pero no demuestra que la discordancia identifique errores en vez de saltos genuinos. Esa diferencia es precisamente lo que deben contrastar el estrés plausible, la ablación y los análisis de cola; no se da por cierta a partir de una RMSE global.

**D4. Predicciones secundarias que deben contrastarse.** Se proponen antes de los experimentos finales:

- Al retirar la saturación, debería aumentar la sensibilidad a perturbaciones que producen estados latentes grandes. El ejemplo analítico del [anexo técnico](09_modelo_original.md) muestra ese efecto en un caso artificial, no en BTC.
- Al retirar la penalización, debería reducirse la ventaja atribuida específicamente a estados discordantes. Si el efecto permanece igual, debe revisarse la explicación del papel de ese componente.
- La mejora debería mantenerse bajo alteraciones plausibles que preserven la coherencia OHLCV, además del estrés artificial ya conocido. Los tipos, severidades y subconjuntos se fijarán antes de medirlos.
- Los saltos genuinos y la cobertura en la cola alta no deberían degradarse de manera incompatible con la hipótesis. Una ganancia media acompañada de supresión de extremos obliga a revisar el diseño y sus límites.

**D5. Éxito y fracaso antes de la evaluación definitiva.** Como criterio de desarrollo propuesto, sean $\Delta_S=\operatorname{RMSE}_{\rm base}(S)-\operatorname{RMSE}_{\rm completo}(S)$ y $D_0=\operatorname{RMSE}_{\rm completo}(\text{limpio})-\operatorname{RMSE}_{\rm base}(\text{limpio})$. El criterio a someter a cierre previo es que el límite inferior del IC95 % de $\Delta_S$ supere $\delta$ y que el límite superior del IC95 % de $D_0$ quede por debajo de $\delta$, con inferencia temporal y control familiar definidos en el protocolo. Los umbrales de los análisis de cola también deben fijarse antes de conocer sus resultados.

No cumplir esas condiciones impide declarar éxito de esta hipótesis; un intervalo amplio se reportaría como no concluyente, sin confundirlo con equivalencia. El contraste de estrés y sus criterios secundarios siguen siendo **propuestas sin ejecución ni preregistro**, aunque M00 y el modelo completo sí se entrenaron en el piloto global de 7.5. Antes de confirmar se deben cerrar el estimando, el remuestreo, las familias y las reglas de decisión. R3 requerirá además sus condiciones completas de relevancia, adecuación y superioridad frente a cada rival; no se deduce de una comparación favorable con el modelo base.

**Aplicación a la referencia actualmente evaluada.** El diagnóstico posterior del piso y la elección de la cabeza aditiva se conservan como decisiones de desarrollo. No heredan la hipótesis de penalización/saturación: la referencia de 1210 parámetros carece de ambas modificaciones. Sus diez semillas y los 48 contrastes describen la receta ejecutada, sin aislar causalmente la cabeza, la atención o el entrenamiento. Para defenderla como aporte debe fijarse una limitación específica y una hipótesis propia, con subconjunto observable al pronosticar, magnitud de efecto y predicciones secundarias refutables. Esas decisiones futuras no se presentarán como anteriores a los resultados ya conocidos.

## 7.2 Disciplina de diseño: evitar el sobreajuste al conjunto de validación

La [bitácora inicial](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/design_log.md) registra cuatro variantes de software comprobadas con entradas artificiales: completa, sin penalización, sin saturación y sin contenido. Su conteo de cero recetas entrenadas en mercado corresponde al corte anterior a los pilotos. **V1/v2 reunieron cinco recetas neuronales y el piloto por momentos completó otras tres.** Las diez réplicas aditivas posteriores conservan una receta, sin diez diseños nuevos. El $N_d$ global sigue pendiente de conciliación e incluye también propuestas, parametrizaciones y ensayos descartados; no se reduce al subtotal de ocho recetas. Antes de confirmar debe declararse el inventario completo con fecha, evidencia, métricas y decisión.

La [bitácora canónica de consolidación](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md), fechada el 17 de noviembre de 2026, enlaza el {download}`registro posterior de diseño <../_static/e3_diseno_riguroso_20261117/design_log.md>` y el {download}`inventario de variantes y ejecuciones <../_static/e3_diseno_riguroso_20261117/design_inventory.csv>`. La consolidación conserva las fuentes históricas y los campos desconocidos; no es un registro previo de aquellos experimentos ni un cierre del $N_d$ global. Las fechas de un plan o del nombre de una carpeta no se convierten en fechas verificadas de ejecución.

El ajuste diminuto de una prueba unitaria realizó 350 pasos de Adam sobre ocho ejemplos artificiales. Comprueba capacidad de ajuste del software; no constituye entrenamiento de BTC, validación del mecanismo económico ni selección de hiperparámetros con los datos de E1/E2.

La separación de datos para las siguientes fases debe conservar estos roles:

| Uso | Decisiones permitidas | Límite |
|---|---|---|
| Desarrollo | Formular el mecanismo, probar variantes y fijar espacios, hipótesis y criterios. | Registrar todo el esfuerzo y el sesgo de selección; no presentar este conjunto como confirmación independiente. |
| Confirmación | Aplicar el protocolo final con arquitectura, pérdida, ajustes y reglas ya cerrados. | No cambiar el diseño después de observar sus resultados. |
| Tres conjuntos adicionales | Evaluar la transferencia de la misma tarea bajo el procedimiento prefijado. | **Todavía no se han elegido.** Deben seleccionarse antes de conocer el desempeño y no usarse para rediseñar el mecanismo. |

En la evidencia actual, las 116 fechas de julio–octubre de 2024 son **validación de desarrollo**: guiaron las mejores épocas y la elección entre recetas. Los 532 ejemplos anteriores se usaron para ajuste y estadísticas de transformación. Las 53 fechas de calibración de E2 no se abrieron en el procedimiento aditivo congelado. Las 608 fechas de prueba conservan su etiqueta histórica y permiten la comparación posterior, pero no se convierten en una reserva confirmatoria nueva. Renombrar una cohorte no deshace su exposición previa.

Las 608 fechas de enero de 2025 a agosto de 2026 ya se evaluaron en E1/E2; la reserva de septiembre–octubre de 2026 también tiene antecedentes publicados. Repetir esos periodos no recupera cegamiento. El [contrato heredado](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) conserva su procedencia; la confirmación de E3 requiere documentar la exposición previa y una separación real de las decisiones de diseño.

Antes de la campaña definitiva deben quedar cerrados la arquitectura y la pérdida, la revisión del mecanismo, las particiones y fuentes, los tres datasets adicionales, el espacio y presupuesto de ajuste comparable, el piloto de potencia, las semillas y el número de ejecuciones, las hipótesis y familias, el $N_d$ y la bitácora. La comparación posterior ya dispone de diez semillas por red; ese número observado no acredita que se haya alcanzado la potencia exigida para una confirmación futura.

El congelamiento debe registrarse con fecha y tag Git reales anteriores a los experimentos definitivos. Un cambio posterior de arquitectura, pérdida o espacio de búsqueda reinicia el protocolo con nueva versión y conserva el historial. El esfuerzo de diseño del candidato se reportará separado del presupuesto de HPO de cada rival, sin ocultar una asimetría bajo el mismo número de configuraciones.

El número de semillas, el tiempo de entrenamiento y el número de diseños describen aspectos diferentes del esfuerzo. Repetir diez veces una receta no elimina la selección previa entre recetas; el registro de $N_d$ tampoco produce por sí solo una corrección numérica del sesgo del ganador. La comparación actual mantiene 532 ejemplos de ajuste y 116 de desarrollo en E3, frente a 655 de ajuste y 53 de calibración en E2. No se declara equivalencia de HPO, política de ajuste final o cómputo a partir de compartir ventanas D7×5.

## 7.3 Sustento teórico: qué demostrar y cómo verificarlo

### Propiedad histórica del módulo con penalización y saturación

La propiedad del primer diseño es una cota Lipschitz de la **cabeza sobre estados latentes**. La [formulación del Bloque C](06_modelo_original.md) y la {download}`demostración íntegra con derivadas y pseudocódigo <../_static/e3_bloque_c_20261115/theory.md>` conservan su desarrollo. La demostración siguiente es autocontenida; no extiende la garantía al preprocesamiento, a la CNN o a la LSTM. La propiedad de la propuesta posterior por momentos se integra en 7.6.

Sean $H,H'\in\mathbb R^{n\times d}$, con $n=7$ en el problema principal. Se usan norma euclídea, norma de Frobenius y norma espectral para vectores, estados y operadores, respectivamente. Sean $P$ una matriz causal fija e independiente de $H$, $A=I-P$ y parámetros de inferencia fijos tales que

$$
B>0,\quad\varepsilon>0,\quad\|q\|_2\le Q,\quad\|w\|_2\le W,
\quad0\le\lambda\le\Lambda,\quad\tau\ge\tau_{\min}>0.
$$ (eq-e3-v1-head-assumptions)

El mecanismo es

$$
r_j=(AH)_j,\quad d_j=\sqrt{\|r_j\|_2^2+\varepsilon^2},\quad
s_j=q^\top h_j-\lambda d_j,\quad a=\operatorname{softmax}(s/\tau),
$$ (eq-e3-v1-head-scores)

$$
v_j=\frac{B h_j}{\sqrt{B^2+\|h_j\|_2^2}},\qquad
c=\sum_{j=1}^n a_jv_j,\qquad
f(H)=\operatorname{softplus}(w^\top c+b).
$$ (eq-e3-v1-head-values)

En el código, las restricciones se cumplen mediante $q=Qu_q/\sqrt{1+\|u_q\|_2^2}$, $w=Wu_w/\sqrt{1+\|u_w\|_2^2}$, $\lambda=\Lambda\operatorname{sigmoid}(\theta_\lambda)$ y $\tau=\tau_{\min}+\operatorname{softplus}(\theta_\tau)$. $B$ y $P$ son fijos. Las constantes se guardan en la configuración; no son supuestos de norma que se atribuyan a pesos sin controlar.

**Lema 1: valores acotados y transformación 1-Lipschitz.** Para $T_B(h)=Bh/u$, $u=\sqrt{B^2+\|h\|_2^2}$,

$$
J_{T_B}(h)=\frac B u I-\frac B{u^3}hh^\top.
$$ (eq-e3-saturation-jacobian)

Sus autovalores son $B/u$ en direcciones ortogonales a $h$ y $B^3/u^3$ en la dirección de $h$; en $h=0$ el jacobiano es $I$. Están en $(0,1]$. Integrando sobre el segmento entre dos puntos, $\|T_B(h)-T_B(h')\|_2\le\|h-h'\|_2$. Además, $\|T_B(h)\|_2\le B$. Para la matriz $V$ de filas $v_j^\top$,

$$
\|V(H)-V(H')\|_F\le\|H-H'\|_F,
\qquad\|V(H)\|_2\le\|V(H)\|_F\le B\sqrt n.
$$ (eq-e3-saturation-matrix-bounds)

**Lema 2: sensibilidad de los puntajes.** La igualdad $d_j(H)=\|(r_j,\varepsilon)\|_2$ y la desigualdad triangular inversa implican $|d_j(H)-d_j(H')|\le\|r_j-r'_j\|_2$. Sumando cuadrados por filas,

$$
\|d(H)-d(H')\|_2\le\|A(H-H')\|_F
\le\|A\|_2\|H-H'\|_F.
$$ (eq-e3-discordance-sensitivity)

La parte de contenido cumple $\|(H-H')q\|_2\le Q\|H-H'\|_F$. Por tanto,

$$
\|s(H)-s(H')\|_2
\le\bigl(Q+\Lambda\|A\|_2\bigr)\|H-H'\|_F.
$$ (eq-e3-score-sensitivity)

**Lema 3: softmax es $1/2$-Lipschitz en norma euclídea.** Su jacobiano es $J=\operatorname{diag}(a)-aa^\top$. La suma absoluta de la fila $j$ es $2a_j(1-a_j)\le1/2$. Como $J$ es simétrico, $\|J\|_2\le\|J\|_\infty\le1/2$. Integrando el jacobiano y aplicando el Lema 2,

$$
\|a(H)-a(H')\|_2
\le\frac{Q+\Lambda\|A\|_2}{2\tau_{\min}}\|H-H'\|_F.
$$ (eq-e3-attention-sensitivity)

**Teorema.** Bajo los supuestos anteriores, para cualesquiera estados finitos $H,H'$ y los mismos parámetros de inferencia,

$$
|f(H)-f(H')|\le C_{\rm head}\|H-H'\|_F,
\qquad
C_{\rm head}=W\left[1+\frac{B\sqrt n}{2\tau_{\min}}
\left(Q+\Lambda\|I-P\|_2\right)\right].
$$ (eq-e3-v1-head-lipschitz-bound)

**Demostración.** Al escribir $c(H)=V(H)^\top a(H)$,

$$
c(H)-c(H')=V(H)^\top\bigl(a(H)-a(H')\bigr)
+\bigl(V(H)-V(H')\bigr)^\top a(H').
$$ (eq-e3-context-difference)

Como $a(H')$ es una distribución, $\|a(H')\|_2\le1$. Los lemas dan

$$
\begin{aligned}
\|c(H)-c(H')\|_2
&\le B\sqrt n\,\|a(H)-a(H')\|_2+\|V(H)-V(H')\|_F\\
&\le\left[1+\frac{B\sqrt n}{2\tau_{\min}}
\left(Q+\Lambda\|I-P\|_2\right)\right]\|H-H'\|_F.
\end{aligned}
$$ (eq-e3-context-lipschitz-bound)

Softplus es 1-Lipschitz porque su derivada es sigmoid, entre cero y uno. La proyección tiene norma $\|w\|_2\le W$ y el sesgo fijo se cancela al tomar diferencias. La composición demuestra la cota. $\square$

La prueba usa aritmética real y puede ser conservadora. En unidades originales de $\sigma_7$, la constante se multiplica por la media de entrenamiento $a_{\rm tr}>0$. Obtener una cota desde OHLCV crudo exigiría demostrar también las constantes del codificador y de sus transformaciones. Esas constantes no se han cuantificado aquí; tampoco se demuestra mejor RMSE, calibración o R3.

Como $c$ es combinación convexa de vectores de norma a lo sumo $B$, también se obtiene

$$
\operatorname{softplus}(b-WB)\le f(H)\le\operatorname{softplus}(b+WB).
$$ (eq-e3-v1-head-range)

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

La ausencia de violaciones en 256 pares apoya la implementación; la afirmación universal procede de la demostración. El caso analítico del anexo ilustra una consecuencia observable de acotar valores, pero **todavía falta comprobar un beneficio predictivo atribuible a esa propiedad en datos reales de desarrollo**. El piloto global de v1 fue desfavorable al módulo completo; no respalda una promesa general de mejora. Tampoco ejecutó el contraste de estrés D3–D5, de modo que no identifica por sí solo la causa de ese resultado. Las pruebas de gradientes, formas, causalidad del encoder, sobreajuste diminuto y serialización mantienen su alcance de software.

### Cota condicional de la referencia aditiva de 1210 parámetros

La referencia actual conserva atención convencional por contenido y retira penalización y saturación. Por ello se demuestra por separado una propiedad conocida sobre **estados acotados**, sin trasladarle el teorema anterior. Sean $H,H'\in\mathbb R^{n\times d}$ con filas de norma a lo sumo $R$, y parámetros de inferencia fijos con $\|q\|_2\le Q$, $\|w\|_2\le W$ y $\tau\ge\tau_{\min}>0$. Sean también $A\ge0$ y $a_{\rm tr}>0$ fijos. La cabeza definida por las ecuaciones {eq}`eq-e3-additive-content-attention` y {eq}`eq-e3-additive-output` de [6.8](06_modelo_original.md) es

$$
\alpha(H)=\operatorname{softmax}(Hq/\tau),\qquad
c(H)=H^\top\alpha(H),\qquad
f_A(H)=\sqrt A+a_{\rm tr}[10^{-4}+\operatorname{softplus}(w^\top c(H)+b)].
$$ (eq-e3-additive-head-map)

El mismo límite del jacobiano de softmax da $\|\alpha(H)-\alpha(H')\|_2\le Q\|H-H'\|_F/(2\tau_{\min})$. Como $\|H\|_2\le\sqrt nR$ y $\|\alpha(H')\|_2\le1$, descomponer la diferencia del contexto y componer la lectura y softplus demuestra

$$
|f_A(H)-f_A(H')|\le K_{\rm ad}\|H-H'\|_F,\qquad
K_{\rm ad}=a_{\rm tr}W\left(1+\frac{\sqrt n RQ}{2\tau_{\min}}\right).
$$ (eq-e3-additive-lipschitz-bound)

Cada estado oculto de la LSTM estándar tiene coordenadas $h_j=o_j\odot\tanh(C_j)$, de magnitud a lo sumo uno en aritmética real; se puede usar $R=\sqrt d$. Esto identifica un dominio que contiene los estados alcanzables, sin afirmar que todo tensor de ese dominio sea alcanzable. Para $n=7$, $d=12$, $Q=1$, $W=4$ y $\tau_{\min}=0{,}25$, resulta **$K_{\rm ad}/a_{\rm tr}=77{,}32121111929344$**. No se ha acotado la sensibilidad del encoder respecto de su entrada.

Si cambia el piso, sólo se añade el término $|\sqrt A-\sqrt{A'}|\le\sqrt{|A-A'|}$. La raíz no es uniformemente Lipschitz respecto de $A$ cerca de cero. Esto impide extender esta cota condicional a una garantía global en $(H,A)$ con esas normas, y no demuestra inestabilidad desde retornos u OHLCV. Tampoco acredita novedad, calibración o menor RMSE.

**Verificación ejecutada para esta fórmula.** El 17 de noviembre de 2026 se comprobaron estados artificiales en CPU, sin cargar datos ni pesos entrenados de mercado. Se fijaron $a_{\rm tr}=0{,}02$ y $A=0{,}0001$ como constantes sintéticas; el primer valor no es una estadística estimada del conjunto real. Se examinaron inicialización sembrada, parámetros cercanos a sus límites y atención uniforme, con coordenadas en $[-1,1]$, tres amplitudes y tres escalas de perturbación. Dentro de cada perfil los parámetros permanecieron fijos.

| Comprobación de la referencia aditiva | Resultado |
|---|---:|
| Pares latentes sintéticos | 1728 |
| Normas de gradiente de la fórmula de referencia float64 | 96 |
| Cota con la escala sintética fijada | 1,5464242224 |
| Mayor cociente de diferencias observado, precisión mixta | 0,0256703577 |
| Mayor cociente dividido por la escala sintética | 1,2835178841 |
| Violaciones observadas con la tolerancia registrada | 0 |
| Mayor diferencia absoluta entre vía original mixta y referencia float64 | $1{,}591\times10^{-8}$ |

La {download}`demostración y sus supuestos <../_static/e3_diseno_riguroso_20261117/theory_reference.md>`, el {download}`script de reproducción <../_static/e3_diseno_riguroso_20261117/verify_theory.py>` y el {download}`informe numérico con entorno y hashes <../_static/e3_diseno_riguroso_20261117/theory_verification.json>` conservan el procedimiento, semilla, distancias y tolerancia. Las tres fuentes del módulo coinciden con las de la campaña congelada. La comprobación finita apoya el código; la afirmación universal procede de la demostración en aritmética real. El redondeo float32 no hereda literalmente una garantía continua para diferencias arbitrariamente pequeñas. Estos ensayos no prueban una cota del pipeline OHLCV ni que esta propiedad explique la ganancia histórica de predicción.

El [workflow de comprobación CPU](https://github.com/jassan-bit/DeepLearning_202630/blob/main/.github/workflows/verify-e3-design-theory.yml) reproduce estos ensayos y verifica las copias publicadas de la bitácora y la demostración. Guarda su propio informe sin sobrescribir los resultados versionados; no ejecuta entrenamiento o inferencia de mercado ni sustituye una reproducción humana independiente.

### Qué exige la guía además de una desigualdad

La propiedad elegida debe tener supuestos explícitos, demostración y verificación numérica, y debe mostrarse si importa para el problema. Una simulación artificial puede comprobar la implementación o una consecuencia matemática; no sustituye el beneficio predictivo que se atribuya al mecanismo. Las discrepancias entre supuestos y datos deben declararse, conservando resultados adversos.

| Línea de diseño | Lo demostrado o comprobado | Lo que no se ha establecido |
|---|---|---|
| Penalización y saturación, 1211 parámetros | Cota latente anterior, derivadas y verificación artificial. | Beneficio predictivo atribuible a la cota y robustez de mercado bajo D3–D5. |
| Salida aditiva, 1210 parámetros | Soporte de la salida y cota condicional anterior sobre estados acotados, con comprobación numérica propia. | Novedad del mecanismo, beneficio causal de cada componente y una garantía desde OHLCV crudo. La cota de saturación de v1 no se hereda. |
| Momentos, 1236 parámetros | Consistencia condicional del riesgo bajo capacidad y momentos verdaderos; contraejemplos con momentos erróneos y auxiliares. | Convergencia de la red finita o mejora de RMSE con momentos estimados; el piloto no superó su heurística. |

Las otras propiedades admisibles de la guía —invariancia/equivarianza, descenso de pérdida, expresividad, generalización o propiedades de la pérdida— son alternativas que requieren su propia formulación, supuestos y verificación. No se afirma haberlas demostrado por enumerarlas. Una propiedad conocida de una parametrización tampoco acredita por sí sola la contribución original.

## 7.4 Diseño de la ablación

En el primer diseño hay dos modificaciones algorítmicas candidatas: penalización de discordancia y saturación radial. Se propone un factorial completo $2\times2$ para separar sus aportes e interacción, con el mismo codificador y atención de contenido. Este plan pertenece al módulo de 1211 parámetros; no completa la ablación de la referencia aditiva ni agrega entrenamientos o resultados.

| Modelo $M_{uv}$ | Penalización $u$ | Saturación $v$ | Relación con el código disponible | Estado experimental |
|---|---:|---:|---|---|
| $M_{00}$, base convencional | 0 | 0 | Ambos componentes desactivados; implementado en el ejecutor de desarrollo v1. | Entrenado en v1; RMSE de desarrollo 0,00537677. |
| $M_{10}$, sólo penalización | 1 | 0 | `without_saturation`. | Intervención funcional comprobada con datos artificiales. |
| $M_{01}$, sólo saturación | 0 | 1 | `without_penalty`. | Intervención funcional comprobada con datos artificiales. |
| $M_{11}$, completo | 1 | 1 | `full`. | Entrenado en v1; RMSE de desarrollo 0,00604531. |
| Control adicional sin contenido | 1 | 1 | `without_content` fuerza $q=0$. | Intervención funcional comprobada; no forma parte del factorial de los dos cambios. |

El control sin contenido estudia un componente convencional y **no es una tercera innovación**. Las tres ablaciones funcionales existentes no equivalen a haber completado el factorial, ni a la ablación estadística exigida. El piloto v1 ya implementó y comparó $M_{00}$ y $M_{11}$; las dos intervenciones aisladas siguen sin esa evaluación de mercado. El resultado de v1 fue desfavorable al modelo completo en la métrica global.

Las cuatro celdas cubren las variantes mínimas de la guía para esos dos componentes: $M_{11}$ es el completo; $M_{01}$ retira penalización y añade sólo saturación a la base; $M_{10}$ retira saturación y añade sólo penalización; $M_{00}$ carece de ambas. Con dos factores, el factorial completo permite identificar la interacción sin fraccionarlo. Si un diseño posterior incorpora más factores, un factorial fraccionado deberá justificar qué interacciones permite estimar y cuáles quedan confundidas, antes de observar los resultados.

La exigencia general de aislar al menos tres componentes debe justificarse sobre el modelo que finalmente se defienda, sin contar tres innovaciones por tres intervenciones ni trasladar las de otro núcleo. Los controles convencionales pueden responder preguntas de dependencia arquitectónica, pero no demuestran que esos bloques sean nuevos. Para la referencia aditiva sigue pendiente el plan de ablación correspondiente al aporte que se presente: los pilotos v1, v2 y por momentos comparan preguntas y recetas distintas.

Para errores $E_{uv}$ sobre unidades comparables, la interacción puede expresarse como

$$
I=E_{11}-E_{10}-E_{01}+E_{00}.
$$ (eq-e3-ablation-interaction)

Un $I<0$ indicaría una reducción de error conjunta mayor que la suma aditiva de sus reducciones, bajo esta codificación; no se ha estimado $I$. El análisis deberá reportar efectos principales e interacciones con IC, mediante un modelo o remuestreo compatible con la dependencia temporal y las repeticiones. Los contrastes contra el modelo completo usarán las pruebas adecuadas de la Sección 8 de la guía y Holm dentro de una familia de ablación separada de la comparación principal.

Se mantendrán particiones, preprocesamiento, semillas pareadas, política de ajuste, épocas y presupuesto comparables; el esfuerzo real y los fallos se registrarán. Las semillas no se cuentan como mercados independientes. Se reportarán parámetros totales y activos: retirar una penalización puede eliminar su parámetro, y retirar contenido elimina $q$. Cuando cambie el número de parámetros se incluirá una variante de tamaño igualado para separar el efecto del mecanismo del efecto de la capacidad, y se discutirá esa comparación.

Las ablaciones pueden apoyar que un componente contribuye bajo el diseño evaluado; no prueban por sí solas el mecanismo causal. Sus resultados deberán contrastarse con las predicciones D4 y conservar también los efectos nulos, adversos y la posible pérdida de desempeño en extremos.

## 7.5 Pilotos de desarrollo completados y decisiones

Los dos pilotos usan la misma base CNN-LSTM-atención. V1 comparó la atención modificada con $M_{00}$ y obtuvo RMSE **0,00604531 frente a 0,00537677**: el mecanismo completo no mejoró al control. La revisión posterior del [anexo B](17_revision_post_piloto.md) detectó predicciones incompatibles con el piso observable del objetivo; su diagnóstico es posterior y no se presenta como un preregistro.

V2 comparó tres recetas con el mismo codificador y cuarenta épocas por receta, una semilla y 116 fechas de validación de desarrollo. Las mejores épocas se seleccionaron con esa validación, ya expuesta a decisiones de diseño. Sus resultados en unidades originales de σ₇ son:

| Receta v2 | RMSE de desarrollo | Decisión |
|---|---:|---|
| Cabeza positiva aditiva | **0,0031446148** | Referencia predictiva de desarrollo. |
| Half-t5: sólo media | 0,0031739264 | Control sin NLL auxiliar. |
| Half-t5: media y NLL | 0,0031982970 | No supera la heurística exploratoria del 2 %. |

La receta con NLL quedó 1,707 % peor que la aditiva. La referencia aditiva también presentó una RMSE 22,50 % mayor que $M_{00}$ conservado y proyectado en las 18 fechas de objetivo realizado alto. Ese grupo sólo sirve para diagnóstico retrospectivo: no es información disponible al pronosticar. El [anexo E](20_cierre_v2.md) conserva cohortes, cifras por régimen, errores individuales y reproducción sin nuevos ajustes.

La comparación con $M_{00}$ conservado enfrenta recetas completas, no aísla por sí sola cada cambio de salida o entrenamiento. Estos resultados de desarrollo no prueban superioridad frente al Top 10, no completan R3 y no permiten promover automáticamente la aditiva como aporte original.

## 7.6 Propuesta posterior: momentos y media transformada

La identidad conocida del objetivo es $Y=\sqrt{A+cV}$, con $c=6/49$, $A$ observable al origen y $V=U^2$ una etiqueta futura. Aquí $A$ es un escalar de la varianza histórica, distinto del operador latente del apartado 7.3. La [formulación del capítulo 6](06_modelo_original.md) propone estimar los escalares $m\approx E[V\mid X]$, $q\approx E[V^2\mid X]$ y una posición $\rho$ de la media entre sus cotas $L,H$.

Con escalas positivas obtenidas sólo de ajuste, $\widetilde V=V/s_u^2$, $\widetilde m=m/s_u^2$, $\widetilde q=q/s_u^4$, el criterio propuesto es

$$
\mathcal L=((Y-\widehat\mu)/s_y)^2
+\lambda_m[\log\widetilde m+\widetilde V/\widetilde m]
+\lambda_q[\log\widetilde q+\widetilde V^2/\widetilde q].
$$ (eq-e3-moment-training-loss)

Con $m_0>0$, $q_0=E[U^4\mid X]<\infty$ y una clase capaz de representar conjuntamente momentos y media verdaderos, el exceso de riesgo condicional es

$$
\frac{(\widehat\mu-\mu_0)^2}{s_y^2}
+\lambda_m[\log(m/m_0)+m_0/m-1]
+\lambda_q[\log(q/q_0)+q_0/q-1]\ge0.
$$ (eq-e3-moment-excess-risk)

La demostración y la verificación con cinco leyes artificiales están en el [anexo F](21_propuesta_momentos.md). La cota y los scores tienen antecedentes atribuidos. El teorema no garantiza convergencia de la red, trayectoria de Faithful o mejora de RMSE; excluye el borde verdadero $m_0=q_0=0$ del score logarítmico. Con momentos estimados la región calculada puede no contener la media verdadera y **no es un intervalo de predicción**.

El [anexo G](22_dictamen_momentos.md) delimita además dos contraejemplos: momentos incorrectos pueden excluir la media verdadera, y los auxiliares pueden empeorar un paso escalar de SGD respecto de la misma cabeza con MSE sola. Son seis actualizaciones artificiales independientes, no entrenamiento neuronal o una trayectoria Adam. La implementación neuronal posterior se evaluó en el piloto de 7.8, que no superó su heurística de continuación; ese resultado no identifica causalmente cuál de esos límites explicó su error.

## 7.7 Controles de comparación y decisión de desarrollo

Los dos primeros controles de la tabla se evaluaron en el piloto de 7.8. El control inspirado en Faithful y las ablaciones complementarias continúan previstos; no se presentan como comparaciones completadas.

| Control conceptual | Pregunta que permite contrastar |
|---|---|
| Misma cabeza por momentos con y sin auxiliares; 1236 parámetros | Efecto de las pérdidas auxiliares conservando todos los ancestros de la media. |
| Cabeza por momentos frente a media directa con las mismas dos cabezas auxiliares; 1236 parámetros | Efecto del vínculo entre media y momentos, con tres lecturas activas. |
| Media directa con auxiliares separados y gradiente detenido al contexto, frente a su propia media sola | Comparación inspirada en Faithful, sujeta a comprobar condiciones de actualización. |
| Referencia aditiva v2 y ablaciones Jensen / $q=9m^2$ | Utilidad frente al control observado y necesidad de cada componente; se declaran capacidades diferentes. |

Igualar parámetros no demuestra igual clase de funciones, coste o trayectoria. El dictamen del corte anterior conservó la propuesta como adaptación experimental sin seleccionar como aporte original. El posterior piloto exploratorio de 7.8 ya terminó; no cerró la revisión externa ni el protocolo confirmatorio. Los rivales congelados y el margen heredado se conservan.

## 7.8 Piloto exploratorio por momentos: condiciones históricas y cierre

El usuario solicitó ejecutar la CNN-LSTM con atención y cabeza de momentos para comprobar su funcionamiento antes de completar los requisitos finales. El [plan de ejecución](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_piloto_momentos_20261116/plan.json) fijó datos, pérdidas, controles y presupuesto; no fue un preregistro confirmatorio del curso. El piloto ya terminó y su [resumen inmutable](https://github.com/jassan-bit/DeepLearning_202630/blob/4cd455d90fb72fe78beeb885dc5dadc4ea9edeb4/runs/momentos_v1/summary.json) conserva los resultados de desarrollo, con la exposición histórica real.

| Receta nueva | Pérdida y salida | Parámetros |
|---|---|---:|
| `moment_joint` | Cabeza de momentos; MSE y ambos scores auxiliares, con pesos 0,1. | 1236 |
| `moment_mean_only` | La misma cabeza y sus ancestros de media; MSE sola. Sus salidas auxiliares no se presentan como momentos identificados. | 1236 |
| `direct_moment_aux` | Media positiva directa y las mismas dos cabezas de momentos con scores auxiliares. | 1236 |

Se conservaron como controles adicionales el piso observable, el piso más incremento medio de ajuste y las predicciones congeladas de la aditiva v2. Esta última fue una referencia de 1210 parámetros, no un control de capacidad igualada, comprobada mediante texto y hashes sin cargar sus pesos neuronales.

La receta usó una semilla, `20261116`, hasta **40 épocas por modelo**, AdamW, lote 32 y paciencia 8; se intercalaron épocas entre las tres variantes. Sólo se admitieron las tres matrices `training_*`: 532 ejemplos de ajuste, siete purgados y 116 fechas de validación de desarrollo ya expuesta a decisiones. Los escaladores, momentos e igualación del exceso medio inicial se calcularon sólo con ajuste. No se usaron calibración ni prueba para esta evaluación.

El piloto dispuso de **600 segundos adicionales**, manteniendo el consumo previo de **556,2126593619992 segundos** y el límite global de 1200. Consumió **94,1646841730 segundos**, hasta un acumulado de **650,3773435350**, sin reiniciar el saldo. Instalación, descarga y publicación quedaron fuera del contador activo. Los registros remotos, checkpoints y costes se conservan en el cierre de esa campaña.

El [notebook de ese piloto](https://colab.research.google.com/github/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable3_momentos_colab.ipynb) y el anexo F.6 conservan las instrucciones históricas; no representan una corrida pendiente ni autorizan repetirla para escoger otro resultado.

Las tres recetas terminaron. `moment_joint` obtuvo RMSE **0,0032589201**, frente a **0,0031446148** de la aditiva congelada: quedó un **3,635 % peor** y no alcanzó la mejora exploratoria mínima del 2 % frente al mejor control. El cierre conserva también `moment_mean_only` y `direct_moment_aux`; no se sustituyó la primaria por una métrica más favorable. Este resultado no acredita ni refuta R1 por sí solo, y no satisface R3.

## 7.9 Réplica aditiva y comparación posterior vigente

La comparación actual conserva la receta **CNN causal → LSTM → atención temporal por contenido y salida aditiva positiva**, de **1210 parámetros**, en diez semillas `271801`–`271810`. No vuelve a seleccionar arquitectura o época con las 608 fechas de prueba conocidas. E3 usa 532 ejemplos de ajuste, 116 de validación y ninguno de calibración; E2 mantiene 655 de ajuste y 53 de calibración. Las trece redes de E2 ya tienen diez semillas cada una, mientras sus tres clásicos deterministas conservan una trayectoria: 133 predictores de E2 frente a las diez réplicas aditivas.

El código del análisis se fija en [0213266](https://github.com/jassan-bit/DeepLearning_202630/tree/0213266404a0aa3af985d7306dce6b13892c71ec) y el plan en [4a21b03](https://github.com/jassan-bit/DeepLearning_202630/tree/4a21b0329b99e864ec64fc2df49260aa874141bf). Ese orden fija el cálculo posterior, no retrofecha una confirmación anterior a los entrenamientos o a la exposición al calendario. La [sección 8](08_verificacion_estadistica.md) y sus [tablas de contrastes](../_static/e3_estadistica_posterior_20261117/primary_contrasts.csv) presentan los 48 rechazos tras Holm con sus distintos ámbitos de incertidumbre. No se afirma igualdad de entrenamiento, potencia suficiente ni superioridad R3; el MCS auxiliar conserva diez modelos, incluida la aditiva.

## 7.10 Revisión del diseño frente a la guía

La revisión del 17 de noviembre de 2026 conserva las decisiones y los resultados con su fecha y arquitectura. La {download}`matriz de requisitos y evidencia <../_static/e3_diseno_riguroso_20261117/requirements_review.csv>` permite seguir los pendientes sin presentar un plan como una ejecución.

| Exigencia | Evidencia actual | Para cerrar |
|---|---|---|
| D1: limitación concreta y cuantificada | Estrés artificial de E2, diagnóstico posterior del piso y resultados por régimen documentados. | Limitación específica que la contribución finalmente defendida pretende corregir, sin trasladar automáticamente el calendario de ETTh1 ni una corrupción artificial a BTC. |
| D2: fuente propia o transferida y adaptación | Componentes conocidos y fuentes atribuidos; tres líneas de diseño diferenciadas. | Resolver la diferencia sustantiva y, si hay transferencia, aplicación de origen, motivo de transferibilidad y modificación desde la formulación. |
| D3–D5: hipótesis, predicciones secundarias y éxito/fracaso previos | Propuesta histórica para penalización/saturación; heurísticas de desarrollo y resultados favorables y adversos conservados. | Hipótesis específica del mecanismo defendido, subconjuntos observables, magnitudes y reglas cerradas antes de una nueva evaluación confirmatoria. No retrofecharlas. |
| Registro de iteraciones y $N_d$ | Bitácora parcial consolidada; ocho recetas neuronales históricas y diez réplicas posteriores de una de ellas. | Conciliar todas las propuestas, parametrizaciones, intentos y fallos; cuantificar y discutir selección sin tomar ocho como $N_d$ global. |
| Separación entre desarrollo y confirmación | Cohortes y exposición histórica declaradas; las 116 fechas siguen siendo desarrollo. | Separación real para decisiones futuras, sin usar confirmación para rediseñar o recuperar cegamiento por cambiar nombres. |
| Tres conjuntos adicionales independientes del diseño | Ninguno elegido ni evaluado en E3. | Seleccionarlos antes de conocer su desempeño, documentar tarea, diferencias y admisión; si guían el diseño, declararlos desarrollo. |
| Congelamiento y esfuerzo comparable | Recetas, planes y consumos guardados; comparación posterior identificada. | Arquitectura, pérdida, espacio de búsqueda, recursos y protocolo con fecha/tag anteriores a futuros experimentos; HPO equivalente y esfuerzo de diseño separado. Un cambio posterior exige nueva versión. |
| Propiedad formal y verificación numérica | Pruebas delimitadas por arquitectura: cota histórica de v1, cota condicional aditiva con 1728 pares/96 gradientes y teoría condicional de momentos. | Justificar la propiedad del aporte defendido y sus supuestos; no convertir una desigualdad conocida en prueba de novedad o garantía sobre todo el pipeline. |
| Mostrar que la propiedad importa | Ejemplos artificiales y resultados de desarrollo conservados, incluido el deterioro de v1. | Beneficio predictivo atribuible al mecanismo bajo el diseño previsto; la RMSE favorable de otra receta no lo demuestra. |
| Ablaciones mínimas e interacción | M00 y completo entrenados en v1; intervenciones aisladas artificiales y controles v2/momentos con preguntas distintas. | Modelo completo, cada componente retirado o sustituido, base y cada componente añadido solo; al menos tres componentes aislados justificados del modelo defendido, interacción e IC. |
| Inferencia, capacidad y conclusión prudente | Regla de familia de ablación separada y controles de capacidad documentados. | Mismas pruebas pertinentes, Holm dentro de esa familia, presupuestos y semillas comparables, variantes de tamaño igualado cuando cambie capacidad; contribución observada no equivale a mecanismo causal demostrado. |

Esta revisión no selecciona un mecanismo nuevo, abre entrenamiento de mercado o aprueba un protocolo confirmatorio. La siguiente decisión de investigación debe resolver qué aporte se sostendrá con evidencia y comprensión del autor, conservando la referencia aditiva como resultado reproducible dentro de su alcance.
