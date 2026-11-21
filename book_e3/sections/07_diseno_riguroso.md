# 7 Diseño riguroso del modelo original

El mecanismo y su estado de admisibilidad se describen en el [Bloque C](06_modelo_original.md). Este capítulo integra el diseño inicial de atención, los pilotos de desarrollo completados y la posterior réplica de la **salida aditiva positiva de 1210 parámetros** en diez semillas. La [sección 8](08_verificacion_estadistica.md) presenta la comparación estadística posterior ya calculada. La referencia aditiva conserva su estado de R1; la revisión específica de V3 tiene un dictamen técnico favorable y acotado en {ref}`6.18 <e3-r1-substantive-verdict-20261120>`. R3 y la comprensión/defensa académica de R2 mantienen sus requisitos. Ese análisis no convierte el diseño en un protocolo confirmatorio.

Este capítulo aplica la sección 7 de la guía: identificar una limitación, formular una explicación refutable, registrar las decisiones y contrastar teoría y ablaciones. Los apartados 7.1–7.4 distinguen los diseños históricos de la referencia evaluada; 7.5–7.8 conservan los pilotos y su cierre, 7.9 identifica la comparación posterior vigente, 7.10 reúne la revisión de requisitos y 7.11 añade el control posterior de soporte del 18 de noviembre. Los anexos [B](17_revision_post_piloto.md) a [G](22_dictamen_momentos.md) conservan el detalle técnico y las fechas de esas decisiones.

| Arquitectura | Pregunta de diseño documentada | Alcance de la evidencia |
|---|---|---|
| Primer módulo, 1211 parámetros | Penalizar discordancia y saturar los valores de atención para limitar su sensibilidad. | D1–D5 y cota latente históricas; el piloto global fue desfavorable frente a M00. |
| Referencia aditiva, 1210 parámetros | Sumar un exceso positivo al piso observable del objetivo, conservando atención por contenido. | Diez réplicas y análisis posterior completos. Su buen rendimiento no demuestra que el cambio de salida sea una contribución nueva ni que explique por sí solo la ganancia. |
| Cabeza de momentos, 1236 parámetros | Relacionar momentos de una innovación con la media de la transformación objetivo. | Teoría condicional y tres recetas de desarrollo completadas; no superó la heurística de continuación. |
| V3, 1262 parámetros | Aprender una ley atómica de U y promediar la transformación exacta de σ₇, con auxiliar CRPS_U. | Piloto de treinta ajustes adverso y NO-GO; dictamen R1 favorable acotado, sin confirmación. |
| V4 de desarrollo, 1262 parámetros | Conservar la lectura V3 y puntuar los átomos transformados en la coordenada de σ₇, con su normalización. | Diagnóstico geométrico y piloto CPU de veinte ajustes; V4 tuvo 1,443261 % más RMSE que el mejor control y la receta se abandonó. No hereda R1 ni acredita R3. |

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

La [bitácora inicial](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/e3_bloque_c_20261115/design_log.md) registra cuatro variantes de software comprobadas con entradas artificiales: completa, sin penalización, sin saturación y sin contenido. Su conteo de cero recetas entrenadas en mercado corresponde al corte anterior a los pilotos. **V1/v2 reunieron cinco recetas neuronales y el piloto por momentos completó otras tres.** Las diez réplicas aditivas posteriores conservan una receta, sin diez diseños nuevos. La {ref}`conciliación documental de 7.14 <e3-design-inventory-reconciled>` identifica variantes, controles, propuestas, parametrizaciones e intentos y deduplica alias. Separa las 15 recetas neuronales implementadas de las réplicas y las 183 oportunidades HPO observadas. Sus límites de corpus y una referencia conceptual ambigua permanecen explícitos; la consolidación posterior no es un preregistro ni acredita todas las decisiones informales.

La [bitácora canónica de consolidación](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md), fechada el 17 de noviembre de 2026, enlaza el {download}`registro posterior de diseño <../_static/e3_diseno_riguroso_20261117/design_log.md>` y el {download}`inventario de variantes y ejecuciones <../_static/e3_diseno_riguroso_20261117/design_inventory.csv>`. La consolidación conserva las fuentes históricas y los campos desconocidos; no es un registro previo de aquellos experimentos ni un cierre del $N_d$ global. Las fechas de un plan o del nombre de una carpeta no se convierten en fechas verificadas de ejecución.

El ajuste diminuto de una prueba unitaria realizó 350 pasos de Adam sobre ocho ejemplos artificiales. Comprueba capacidad de ajuste del software; no constituye entrenamiento de BTC, validación del mecanismo económico ni selección de hiperparámetros con los datos de E1/E2.

La separación de datos para las siguientes fases debe conservar estos roles:

| Uso | Decisiones permitidas | Límite |
|---|---|---|
| Desarrollo | Formular el mecanismo, probar variantes y fijar espacios, hipótesis y criterios. | Registrar todo el esfuerzo y el sesgo de selección; no presentar este conjunto como confirmación independiente. |
| Confirmación | Aplicar el protocolo final con arquitectura, pérdida, ajustes y reglas ya cerrados. | No cambiar el diseño después de observar sus resultados. |
| Tres conjuntos adicionales | Evaluar la transferencia de la misma tarea bajo el procedimiento prefijado. | D01 WIKI/AAPL RAW, D02 ETHUSDT Spot y D03 BTCUSD_PERP COIN-M tienen admisión física y evaluación de los **19 sistemas** terminadas. La {ref}`auditoría exploratoria cerrada en 8.18 <e3-final-extended-audit>` incluye también D00, conserva sus dependencias y aporta análisis y figuras. Sus resultados no se usan para rediseñar el mecanismo y presentarlos después como confirmación independiente. |

En la evidencia actual, las 116 fechas de julio–octubre de 2024 son **validación de desarrollo**: guiaron las mejores épocas y la elección entre recetas. Los 532 ejemplos anteriores se usaron para ajuste y estadísticas de transformación. Las 53 fechas de calibración de E2 no se abrieron en el procedimiento aditivo congelado. Las 608 fechas de prueba conservan su etiqueta histórica y permiten la comparación posterior, pero no se convierten en una reserva confirmatoria nueva. Renombrar una cohorte no deshace su exposición previa.

Las 608 fechas de enero de 2025 a agosto de 2026 ya se evaluaron en E1/E2; la reserva de septiembre–octubre de 2026 también tiene antecedentes publicados. Repetir esos periodos no recupera cegamiento. El [contrato heredado](https://jassan-bit.github.io/DeepLearning_202630/entregable2/sections/17_contrato_sincronizado.html) conserva su procedencia; la confirmación de E3 requiere documentar la exposición previa y una separación real de las decisiones de diseño.

Antes de la campaña definitiva deben quedar cerrados la arquitectura y la pérdida, la revisión del mecanismo, las particiones y fuentes, los tres datasets adicionales, el espacio y presupuesto de ajuste comparable, el piloto de potencia, las semillas y el número de ejecuciones, las hipótesis y familias, el $N_d$ y la bitácora. La comparación posterior ya dispone de diez semillas por red; ese número observado no acredita que se haya alcanzado la potencia exigida para una confirmación futura.

El congelamiento debe registrarse con fecha y tag Git reales anteriores a los experimentos definitivos. Un cambio posterior de arquitectura, pérdida o espacio de búsqueda reinicia el protocolo con nueva versión y conserva el historial. El esfuerzo de diseño del candidato se reportará separado del presupuesto de HPO de cada rival, sin ocultar una asimetría bajo el mismo número de configuraciones.

El número de semillas, el tiempo de entrenamiento y el número de diseños describen aspectos diferentes del esfuerzo. Repetir diez veces una receta no elimina la selección previa entre recetas; el registro de $N_d$ tampoco produce por sí solo una corrección numérica del sesgo del ganador. La comparación histórica mantiene 532 ejemplos de ajuste y 116 de desarrollo en E3, frente a 655 de ajuste y 53 de calibración en E2. No se declara equivalencia de HPO, política de ajuste final o cómputo a partir de compartir ventanas D7×5.

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

La comparación histórica conserva la receta **CNN causal → LSTM → atención temporal por contenido y salida aditiva positiva**, de **1210 parámetros**, en diez semillas `271801`–`271810`. No vuelve a seleccionar arquitectura o época con las 608 fechas de prueba conocidas. E3 usa 532 ejemplos de ajuste, 116 de validación y ninguno de calibración; E2 mantiene 655 de ajuste y 53 de calibración. Las trece redes de E2 ya tienen diez semillas cada una, mientras sus tres clásicos deterministas conservan una trayectoria: 133 predictores de E2 frente a las diez réplicas aditivas.

El código del análisis se fija en [0213266](https://github.com/jassan-bit/DeepLearning_202630/tree/0213266404a0aa3af985d7306dce6b13892c71ec) y el plan en [4a21b03](https://github.com/jassan-bit/DeepLearning_202630/tree/4a21b0329b99e864ec64fc2df49260aa874141bf). Ese orden fija el cálculo posterior, no retrofecha una confirmación anterior a los entrenamientos o a la exposición al calendario. La [sección 8](08_verificacion_estadistica.md) y sus [tablas de contrastes](../_static/e3_estadistica_posterior_20261117/primary_contrasts.csv) presentan los 48 rechazos tras Holm con sus distintos ámbitos de incertidumbre. No se afirma igualdad de entrenamiento, potencia suficiente ni superioridad R3; el MCS auxiliar conserva diez modelos, incluida la aditiva.

## 7.10 Revisión del diseño frente a la guía

La revisión histórica del 17 de noviembre de 2026 conserva las decisiones y los resultados con su fecha y arquitectura. La {download}`matriz de requisitos y evidencia <../_static/e3_diseno_riguroso_20261117/requirements_review.csv>` permite seguir los pendientes sin presentar un plan como una ejecución; el control posterior del 18 de noviembre se identifica por separado en 7.11 y en la primera fila siguiente.

| Exigencia | Evidencia actual | Para cerrar |
|---|---|---|
| D1: limitación concreta y cuantificada | Estrés artificial de E2, diagnóstico posterior del piso y resultados por régimen documentados; el control posterior de 7.11 cuantifica el efecto de respetar el soporte en los pronósticos congelados. | Conservar el carácter retrospectivo y resolver la limitación que sostendrá la contribución, sin atribuir causalmente la ganancia a la arquitectura ni trasladar automáticamente el calendario de ETTh1 o una corrupción artificial a BTC. |
| D2: fuente propia o transferida y adaptación | Componentes conocidos y fuentes atribuidos; tres líneas de diseño diferenciadas. | Resolver la diferencia sustantiva y, si hay transferencia, aplicación de origen, motivo de transferibilidad y modificación desde la formulación. |
| D3–D5: hipótesis, predicciones secundarias y éxito/fracaso previos | Propuesta histórica para penalización/saturación; heurísticas de desarrollo y resultados favorables y adversos conservados. | Hipótesis específica del mecanismo defendido, subconjuntos observables, magnitudes y reglas cerradas antes de una nueva evaluación confirmatoria. No retrofecharlas. |
| Registro de iteraciones y $N_d$ | Bitácora parcial consolidada; ocho recetas neuronales históricas y diez réplicas posteriores de una de ellas. | Conciliar todas las propuestas, parametrizaciones, intentos y fallos; cuantificar y discutir selección sin tomar ocho como $N_d$ global. |
| Separación entre desarrollo y confirmación | Cohortes y exposición histórica declaradas; las 116 fechas siguen siendo desarrollo. | Separación real para decisiones futuras, sin usar confirmación para rediseñar o recuperar cegamiento por cambiar nombres. |
| Tres conjuntos adicionales independientes del diseño | Tres archivos admitidos, persistidos y evaluados por los 19 sistemas; el análisis de D00–D03 está terminado en {ref}`8.18 <e3-final-extended-audit>`. Las dependencias entre mercados y la exposición histórica siguen declaradas. | Justificar el alcance de independencia y generalización; si los resultados guían un nuevo diseño, declararlos desarrollo. La finalización no convierte los cuatro conjuntos en cuatro mercados independientes. |
| Congelamiento y esfuerzo comparable | La auditoría exploratoria tiene fuente, hashes, GPU, presupuesto y tag previos a sus ajustes; completó familia y análisis. {ref}`8.18 <e3-final-extended-audit>` documenta la corrección de lectura D00 declarada y las recuperaciones, conservando el protocolo. Las comparaciones históricas mantienen su alcance posterior. | Dimensionar la potencia pendiente para R3, completar recursos comparables y conservar el esfuerzo de diseño separado. Una modificación científica futura requiere su propia versión y declaración. |
| Propiedad formal y verificación numérica | Pruebas delimitadas por arquitectura: cota histórica de v1, cota condicional aditiva con 1728 pares/96 gradientes y teoría condicional de momentos. | Justificar la propiedad del aporte defendido y sus supuestos; no convertir una desigualdad conocida en prueba de novedad o garantía sobre todo el pipeline. |
| Mostrar que la propiedad importa | Ejemplos artificiales y resultados de desarrollo conservados, incluido el deterioro de v1. | Beneficio predictivo atribuible al mecanismo bajo el diseño previsto; la RMSE favorable de otra receta no lo demuestra. |
| Ablaciones mínimas e interacción | M00 y completo entrenados en v1; intervenciones aisladas artificiales y controles v2/momentos con preguntas distintas. | Modelo completo, cada componente retirado o sustituido, base y cada componente añadido solo; al menos tres componentes aislados justificados del modelo defendido, interacción e IC. |
| Inferencia, capacidad y conclusión prudente | Regla de familia de ablación separada y controles de capacidad documentados. | Mismas pruebas pertinentes, Holm dentro de esa familia, presupuestos y semillas comparables, variantes de tamaño igualado cuando cambie capacidad; contribución observada no equivale a mecanismo causal demostrado. |

Esta revisión no selecciona un mecanismo nuevo, abre entrenamiento de mercado o aprueba un protocolo confirmatorio. La siguiente decisión de investigación debe resolver qué aporte se sostendrá con evidencia y comprensión del autor, conservando la referencia aditiva como resultado reproducible dentro de su alcance.

(control-posterior-del-soporte-causal-y-proyeccion)=
## 7.11 Control posterior del soporte causal y proyección

La auditoría del **18 de noviembre de 2026** comprueba una limitación específica: varios pronósticos guardados no respetan el piso observable de la σ₇ objetivo. Es un control exploratorio posterior sobre la cronología conocida, separado de los pilotos de 7.5/7.8 y de los 48 contrastes originales. El {download}`resumen reproducible <../_static/e3_remediacion_20261118/constraint_audit/results/summary.json>` conserva definiciones, métricas y alcance; no hubo nuevos ajustes, forwards de modelos, calibraciones, conjuntos ni pruebas de hipótesis.

### Identidad conocida y garantía de la proyección

Sean $r_i=\log(C_i/C_{i-1})$ y $\bar r_6$ la media de los seis retornos conocidos al origen. Para la desviación estándar **poblacional, diaria y sin anualizar** de siete retornos,

$$
\bar r_6=\frac16\sum_{i=t-6}^{t-1}r_i,\qquad
A_t=\frac17\sum_{i=t-6}^{t-1}(r_i-\bar r_6)^2,\qquad
Y_t^2=A_t+\frac6{49}(r_t-\bar r_6)^2.
$$ (eq-e3-support-audit-identity)

Es la identidad de 6.7 expresada para este control, sin una nueva pretensión de originalidad. $A_t$ sólo requiere los siete cierres históricos $C_{t-7},\ldots,C_{t-1}$; el retorno futuro no interviene en el piso $L_t=\sqrt{A_t}$. Como $Y_t\ge L_t$, proyectar cualquier pronóstico puntual $p$ mediante

$$
\pi_t(p)=\max(p,L_t),\qquad
|Y_t-\pi_t(p)|\le |Y_t-p|
$$ (eq-e3-support-projection)

no aumenta la pérdida absoluta ni la cuadrática: si $p\ge L_t$ no cambia; si $p<L_t\le Y_t$, se acerca al objetivo. La garantía supone objetivo y piso correctamente definidos y disponibles causalmente. No garantiza mejor calibración, eficiencia ni comportamiento fuera de ese soporte; tampoco atribuye un efecto al entrenamiento o a la atención.

### Resultado sobre las trayectorias congeladas

Se aplicó esa transformación a **143 predictores de los 17 sistemas**, en las mismas 608 fechas: **86.944 pares predictor–fecha**, sin eliminar filas. La métrica sigue siendo la media de las RMSE individuales. La aditiva ya respeta el piso y permanece igual. El mejor rival descriptivo cambia al proyectar toda la familia; por eso las dos ventajas siguientes no son una comparación con un único rival fijo.

| Resultado descriptivo | Pronósticos originales | Pronósticos proyectados |
|---|---:|---:|
| RMSE media E3 | 0,0029394324 | 0,0029394324 |
| Mejor rival por RMSE | LSTM-autoencoder: 0,0038728895 | TSMixer: 0,0032780684 |
| Ventaja relativa de E3 frente al mejor rival disponible | **24,10 %** | **10,33 %** |
| Diferencia absoluta de RMSE frente al mejor rival disponible | 0,0009334571 | 0,0003386360 |

La reducción descriptiva de esa diferencia es **63,72 %**, pero **no es una fracción causal explicada por el piso**: cambian el posprocesamiento y el rival mejor situado, mientras persisten diferencias de entrenamiento, selección y presupuesto. El resultado evidencia la importancia de incluir este control físico común; no demuestra que la arquitectura cause la ventaja restante. Las pérdidas absoluta y cuadrática no aumentaron en ningún par auditado.

El control determinista **sólo piso**, independiente de la familia original de 17 sistemas, obtuvo RMSE **0,0035275805** y MAE **0,0018227552**. Su MAE fue **3,70 % menor** que el MAE medio de E3, **0,0018928273**: es un resultado adverso a una afirmación de dominio general por métricas. E3 conserva mejor RMSE que ese control; ninguno de estos valores es una prueba de significancia o una explicación causal.

Los pronósticos y contrastes históricos permanecen identificados por su tratamiento original: no se sustituyen sus 48 p-valores por esta proyección ni se incorpora retrospectivamente el control al protocolo previo. Faltan comparación justa, ablaciones de la receta defendida, adecuación, potencia y confirmación independiente. La {ref}`revisión ampliada de antecedentes <revision-focal-ampliada-20261118>` delimita además qué partes del mecanismo son conocidas. **Este control no es un nuevo preregistro ni acredita R1, R2 o R3.**

(e3-r1-v3-design-protocol)=
## 7.12 Diseño delimitado de la adaptación V3 y controles futuros

Este apartado desarrolla la {ref}`decisión de 6.16 <e3-r1-adaptation-decision>`, con fecha posterior a los resultados ya conocidos. **Es diseño de investigación, no un protocolo confirmatorio congelado ni evidencia de rendimiento V3.** Conserva la referencia aditiva y el candidato V3 existente, sin cambiar sus fuentes.

### Limitación, hipótesis y explicación que pueden refutarse

La identidad `Y=√(A+cU²)`, con `c=6/49`, separa una parte observable de la ventana y una magnitud futura. La limitación matemática específica es que conocer sólo `E[U²|X]` no determina la media de Y. V3 intenta usar una ley condicional y una señal auxiliar para aprender esa media. **La aditiva entrenada directamente con MSE también puede representarla**: no se afirma una imposibilidad de su arquitectura ni una mejora universal de expresividad.

**Hipótesis candidata de desarrollo:** bajo particiones, selección y presupuesto comunes, la ley aprendida de U y su transformación antes de promediar producen una mejora práctica al menos δ_dev frente a la aditiva y aportes identificables frente a las ablaciones. El valor de δ_dev, las semillas, los subgrupos y la regla conjunta de éxito deben justificarse y registrarse antes del piloto; no se completan retrospectivamente a partir de 8.18. El informe actual no congela esos campos.

Predicciones secundarias y resultados que debilitarían esa explicación:

1. Con una misma ley finita fijada, la diferencia entre el pronóstico exacto y el control de segundo momento debe coincidir con el gap de Jensen y respetar las cotas siguientes. Incumplirlo identifica un error de cálculo.
2. Una mejora atribuida a la ley debe examinarse junto al CRPS_U fuera del ajuste y por subgrupos de gap **predicho al origen**, con cortes fijados mediante desarrollo. No se forman grupos con el retorno futuro para presentarlos como una regla operativa.
3. Retirar CRPS, fijar masas y cambiar la agregación deben examinarse por separado. Si un componente no muestra un aporte bajo la regla y potencia fijadas, su utilidad no queda acreditada; la falta de rechazo tampoco prueba equivalencia.
4. La cobertura y el error bajo ruido se reportan, incluidos los resultados adversos. El deterioro de la aditiva en 8.18 no prueba que V3 lo repare.

### Propiedad delimitada: transformación de la ley y sustitución por segundo momento

Sean `V=U²`, una ley finita fijada G con masas válidas y `A>0`. Para `f(v)=√(A+cv)`,

```{math}
:label: eq-e3-v3-jensen-curvature
f''(v)=-\frac{c^2}{4(A+cv)^{3/2}}.
```

Si `V∈[v_min,v_max]`, Taylor alrededor de `E_G[V]`, junto a la esperanza nula del término lineal, da

```{math}
:label: eq-e3-v3-jensen-gap-bounds
\frac{c^2\operatorname{Var}_G(V)}{8(A+cv_{\max})^{3/2}}
\leq \sqrt{A+cE_G[V]}-E_G[\sqrt{A+cV}]
\leq\frac{c^2\operatorname{Var}_G(V)}{8(A+cv_{\min})^{3/2}}.
```

Es una aplicación de la cota estándar de curvatura y Jensen, **no un teorema nuevo**. La concavidad explica el signo; si A=0 y el soporte incluye cero, no se promete la misma cota superior finita uniforme.

Si además G es la verdadera ley condicional, su media transformada `μ_G` minimiza el riesgo cuadrático. Para el control `b=√(A+cE_G[U²])`,

```{math}
:label: eq-e3-v3-plugin-risk-gap
E[(Y-b)^2\mid X]-E[(Y-\mu_G)^2\mid X]=(b-\mu_G)^2\geq0.
```

Esta identidad compara con **ese control de sustitución por segundo momento**, no con la red aditiva ni con cualquier predictor. Bajo mala especificación no hay una ventaja garantizada de la ley aproximada. La propiedad de soporte y el mínimo poblacional del riesgo MSE+CRPS, bajo representabilidad y momentos suficientes, conservan los supuestos restrictivos de 6.14; no son una garantía de optimización o calibración de mercado.

### Verificación analítica y contraejemplos

El {download}`código Python estándar <../_static/e3_r1_close_20261120/adaptation_design/analytic_checks.py>` comprueba **100 leyes finitas artificiales**, las cotas y la identidad de riesgo. Su {download}`recibo con todos los casos <../_static/e3_r1_close_20261120/adaptation_design/analytic_verification.json>` incluye además:

| Comprobación | Resultado y significado |
|---|---|
| Dos leyes con el mismo segundo momento | Con A=0,04 y V uniforme en {0,2; 1,8} o en {0,8; 1,2}, ambas tienen E[V]=1. La sustitución por segundo momento da 0,4030495994 en ambos casos, pero las medias exactas son 0,3821252217 y 0,4018963707. El segundo momento no determina la media transformada. |
| Cabeza aditiva que reproduce la media | La inversión del enlace softplus reconstruye el mismo punto del ejemplo cuando el exceso supera su incremento mínimo. No hay una imposibilidad general de representación de la aditiva. |
| Pérdida auxiliar bajo aproximación | Entre dos leyes aproximadas de tres átomos, añadir CRPS con λ=0,1 cambia el orden del riesgo conjunto y empeora el riesgo puntual. No son óptimos globales ni un experimento de entrenamiento; es un contraejemplo a una garantía incondicional de mejora puntual. |

Estas comprobaciones no importan PyTorch, cargan precios, optimizan modelos o generan pronósticos de mercado. Los números artificiales no se interpretan como rentabilidades plausibles o resultados del benchmark. La demostración y la verificación sostienen propiedades delimitadas. La utilidad se evaluó después en el piloto de 8.19 y la receta completa no superó su heurística; novedad y utilidad se juzgan por separado en 6.18.

### Ablaciones, capacidad y objeto de la comparación

| Variante o control | Qué permite investigar | Estado y límite |
|---|---|---|
| V3 `atom_crps` | Ley, transformación y entrenamiento propuestos. | 1262 parámetros; el corte analítico inicial no aportó mercado. El piloto posterior de {ref}`8.19 <e3-v3-pilot-completed-20261120>` completó sus ajustes y falló la heurística. |
| `atom_mse_only` | Efecto de retirar CRPS. | Mismo inicio y 1262 parámetros; tres semillas finales de desarrollo ejecutadas. Tuvo la menor RMSE media del piloto; ese orden no es significancia ni lo promueve automáticamente a candidata. |
| `fixed_probs` | Efecto de aprender masas frente a uniformes. | 1236 parámetros y tres semillas finales de desarrollo ejecutadas; diferencia de capacidad declarada, sin parámetros inactivos para aparentar igualdad. |
| Sustitución por segundo momento | `√(A+cΣpₖuₖ²)` frente a `Σpₖ√(A+cuₖ²)`. | `atom_plugin`,1262 parámetros; política de 7.13 y tres semillas finales ejecutadas en 8.19. La comparación de redes entrenadas no mantiene necesariamente la misma ley final. |
| Aditiva y familia actual | Referencia de 1210 parámetros y los 18 rivales restantes, incluidos ambos controles de piso. | Se conservan. Añadir V3 da 20 sistemas; una ventaja frente a un único control no completa R3. |

La cabeza de V3 requiere O(K) para transformar y promediar átomos; el CRPS de la implementación por pares es O(K²), con K=3. Esto no reemplaza el perfilado de toda la red, memoria, FLOPs o energía. Las ablaciones deben conservar particiones, escalado, selección, semillas y presupuesto, y reportar el consumo efectivo.

### Algoritmo propuesto y frontera temporal

```text
Entradas: versión y hashes del modelo; datos de desarrollo y particiones;
          K=3, lambda=0.1, escalas de ajuste, variantes y semillas;
          presupuesto común, delta_dev y criterios de éxito/fracaso fechados.
Verificar las fuentes y que la confirmación no haya orientado el diseño.
Para cada receta/semilla predefinida:
  estimar escalas y transformaciones sólo con ajuste;
  por lote, calcular A de la historia y U,Y como etiquetas de ajuste;
  codificar X, producir átomos ordenados y masas;
  transformar cada átomo, promediar y calcular MSE+CRPS_U;
  actualizar sólo con ajuste y seleccionar/parar usando validación;
  conservar estados, fallos, consumo y selección antes del scoring.
Comparar todos los controles, incluidos resultados adversos;
  conservar o descartar V3 por la regla previa, sin acreditar R1/R3.
Sólo después, si procede, registrar la versión y un protocolo confirmatorio
  nuevo con datos reservados, márgenes, familia y potencia suficientes.
```

El algoritmo describe la propuesta; no la ejecuta. Los cuatro conjuntos D00–D03 tienen resultados ya expuestos y no recuperan cegamiento si se usan para diseñar V3. Antes de confirmar deben quedar fijados los datos realmente reservados, la batería apropiada, el ajuste familiar, potencia por contraste y conjunta y al menos diez semillas o el número mayor que requiera el dimensionamiento. El plan histórico de 18 sistemas no se reutiliza como si ya cubriera la familia ampliada.

(e3-v3-controlled-development-pilot)=
## 7.13 Piloto controlado de desarrollo de V3

El plan siguiente fijó un piloto acotado antes de sus ajustes. **Su ejecución posterior ya terminó** y se incorpora en {ref}`8.19 <e3-v3-pilot-completed-20261120>`, conservando esta especificación y su regla anterior al resultado. Su [código y controles](https://github.com/jassan-bit/DeepLearning_202630/tree/main/experiments/e3_v3_development_pilot_20261120) se publican por separado del controlador de los 796 trabajos. El [cuaderno Colab](https://github.com/jassan-bit/DeepLearning_202630/blob/main/notebooks/entregable3_v3_controlled_pilot_colab.ipynb) prepara un intérprete aislado, registra la GPU realmente asignada y guarda el plan en una rama nueva de GitHub antes del primer ajuste. El corte de preparación no aportaba resultados de entrenamiento; el recibo posterior de 8.19 sí acredita 30 ajustes de desarrollo. Ni el código ni ese recibo acreditan automáticamente R1.

| Decisión previa | Especificación |
|---|---|
| Datos | Sólo los tres miembros `training_raw`, `training_y_original` y `training_dates` del NPZ histórico, comprobado por SHA-256. No se materializan los miembros de calibración ni prueba; D01–D03 quedan fuera de este piloto. |
| Ajuste y selección | Las 532 ventanas originales de ajuste conservan el escalado. De las 116 ventanas de validación, las primeras 54 seleccionan y paran; se purgan siete y las últimas 55 se puntúan después de completar todos los ajustes finales. Este calendario ya estuvo expuesto y sigue siendo desarrollo. |
| Recetas neuronales | Aditiva (1210 parámetros), `atom_crps` (1262), `atom_mse_only` (1262), `fixed_probs` (1236) y `atom_plugin` (1262). Las tres últimas aíslan pérdida, masas y agregación, respectivamente. |
| Controles analíticos | `floor_only` y `floor_plus_mean_excess`, con exceso estimado sólo en ajuste. |
| HPO común | Tasas 0,0003; 0,001; 0,003, semilla 20261117: 15 ajustes. Selección por RMSE de σ₇ en las 54 ventanas y desempate por menor tasa. |
| Réplicas finales | Semillas 20261118, 20261119 y 20261120 para las cinco recetas: 15 ajustes adicionales. Son tres réplicas de desarrollo, no las diez o más exigibles para confirmar. |
| Presupuesto y fallos | Una oportunidad nominal de 120 segundos por fase de optimización, hasta 40 épocas, paciencia ocho, lote 32, AdamW sin decaimiento y recorte de gradiente a uno. La construcción y preparación de tensores se mide aparte. El plazo de la fase incluye guardado, esperas y desconexiones, y se conserva al reanudar; se reportan pasos, consumo y posibles excesos. Una receta incompleta bloquea la decisión de continuación. |
| Regla de continuación | V3 debe reducir al menos 2 % la RMSE media de las tres semillas frente al mejor control neuronal y al mejor de todos los controles en las 55 ventanas. Es una heurística de desarrollo, sin pruebas de significancia ni margen confirmatorio. Se conservan todos los resultados, favorables o adversos. |

El control `atom_plugin` conserva la ley atómica y su inicialización, la pérdida CRPS y el número de parámetros de `atom_crps`; reemplaza únicamente su lectura por `√(A+cΣpₖuₖ²)`. Su configuración y versión son distintas para impedir que una reanudación mezcle variantes. Las pruebas sintéticas contrastan igualdad de la ley, desigualdad de Jensen, gradientes, extremos numéricos y restauración del estado. Las diferencias observadas al entrenar ambos controles no serían una prueba de que la aditiva carece de capacidad para representar la media.

El guardado utiliza objetos Git empaquetados y verifica cada publicación. Publica el plan antes del primer ajuste y, por receta, el inicio, el primer checkpoint validado y el cierre. Conserva modelo, optimizador, generadores aleatorios, selección, plazo e identidad del software; una reanudación exige el mismo perfil registrado y recupera únicamente lo guardado. Si se pierde el estado local antes del primer checkpoint remoto validado y vence el plazo, el ajuste queda incompleto: no se renueva su presupuesto. Los resultados nuevos se guardan en `runs/e3_v3_development_pilot/`, sin sobrescribir el master anterior. El recibo final de `20261120T031424176339Z` ya distingue los 30 ajustes completados y mantiene la declaración de IA; sus hashes, tablas y veredicto se incorporan en {ref}`8.19 <e3-v3-pilot-completed-20261120>`. Este plan no se convierte retrospectivamente en una confirmación.

La heurística **falló** en la ejecución posterior: la candidata completa fue 2,70 % peor que el mejor control. Se reporta el resultado adverso y se conserva la regla. Cualquier modificación o elección futura debe anotarse como decisión nueva, sin promover automáticamente la ablación ganadora. La revisión de prioridad y un eventual protocolo confirmatorio con datos adecuados, familia y potencia mantienen sus requisitos propios. El inventario retrospectivo se consolida en 7.14; no recupera prerregistro o independencia temporal.

(e3-design-inventory-reconciled)=
## 7.14 Conciliación documental de diseños, selecciones e intentos

La [bitácora canónica actualizada](https://github.com/jassan-bit/DeepLearning_202630/blob/main/experiments/design_log.md) incorpora una conciliación retrospectiva posterior a los resultados conocidos. El {download}`inventario de identidades <../_static/e3_evidence_close_20261120/inventory/design_inventory.csv>`, la {download}`tabla de alias <../_static/e3_evidence_close_20261120/inventory/design_aliases.csv>` y el {download}`resumen estructurado <../_static/e3_evidence_close_20261120/inventory/reconciliation.json>` fijan los conteos y sus unidades. Los registros históricos conservan sus bytes y fechas.

| Unidad documental | Conteo reconciliado |
|---|---:|
| Variantes neuronales funcionales implementadas | **15**: cinco iniciales, tres v2, tres por momentos y cuatro atómicas. |
| Variantes neuronales distintas con ajuste de mercado | **12**; las tres intervenciones iniciales restantes sólo tienen evidencia artificial. |
| Controles analíticos o de transformación examinados | **4**, incluido full proyectado, omitido en el subtotal previo de tres. |
| Propuestas neuronales sin implementación acreditada | **3**. |
| Entradas de diseño documentadas con identidad propia | **22**; un bosquejo de referencia Faithful sigue sin especificación suficiente para decidir si sería una entrada adicional. |
| Ensayos HPO conservados | **183**: 168 en la auditoría y 15 en el piloto V3. |
| Trabajos completos en los alcances conciliados | **844**: 18 neuronales históricos, los 796 de auditoría y 30 neuronales V3. |

Los 22 registros son un conteo **del corpus disponible**, no de todas las ideas informales o decisiones sin archivo. El intervalo 22–23 sólo representa la ambigüedad identificada de aquel bosquejo dentro del corpus; no constituye un límite superior de todas las oportunidades de selección. La conciliación cierra las identidades observables y deja una ausencia concreta, en lugar de equiparar ocho recetas, diez semillas o 796 trabajos al Nd global.

La completa inicial y full/M11 se deduplican; M00 y la aditiva conservan sus identidades. Las 93 corridas de la aditiva —una v2, diez de estabilidad, 76 de auditoría y seis del nuevo piloto— son repeticiones de una estructura. Las pérdidas distintas sí definen recetas. El {download}`registro de selección <../_static/e3_evidence_close_20261120/inventory/selection_opportunities.csv>` separa tasas, inicialización y validaciones; las decisiones de cuadratura 64/128/256 o pruebas artificiales no se ocultan ni se cuentan como arquitecturas entrenadas en mercado.

El {download}`ledger comprimido de intentos <../_static/e3_evidence_close_20261120/inventory/attempt_ledger.csv.gz>` contiene 824 ajustes neuronales, cuatro ARIMA y 16 trabajos sin optimizador. Los 900 renglones de recursos históricos corresponden a fases de 796 trabajos únicos; el autoencoder tiene tres fases que comparten presupuesto. El {download}`registro de incidentes <../_static/e3_evidence_close_20261120/inventory/incidents.json>` conserva interrupciones de admisión, transporte y trabajador. Tres fases históricas mantienen cómputo perdido desconocido; los sobrepasos de plazo que incluyen desconexiones no se presentan como horas GPU medidas.

El {download}`verificador reproducible <../_static/e3_evidence_close_20261120/inventory/verify_inventory.py>`, las {download}`fuentes propias congeladas <../_static/e3_evidence_close_20261120/inventory/sources.zip>` y su {download}`manifiesto <../_static/e3_evidence_close_20261120/inventory/source_manifest.json>` permiten reconstruir la conciliación sin PyTorch, NPZ, pesos o entrenamiento. Se declara IA; los 21 blobs fuente se cotejaron con sus commits públicos. El conteo no corrige por sí solo el sesgo de selección, restaura cegamiento o acredita R1/R3. La V3 completa conserva su resultado adverso y la heurística incumplida, sin promocionar retrospectivamente al control ganador.


(e3-r1-formal-scope-20261120)=
## 7.15 Propiedad revisada y equivalencias de la adaptación V3

La {ref}`revisión específica de 6.18 <e3-r1-substantive-verdict-20261120>`
apoya la adaptación de lectura y pérdida para esta tarea. La propiedad formal
sigue siendo una propiedad correcta de un mecanismo que utiliza herramientas
conocidas: no se exige convertirla en teorema inédito para cumplir la guía.

**Soporte y esperanza finita.** Si $A\geq0$, $a_k\geq0$, $p_k\geq0$ y
$\sum p_k=1$, cada $T_A(a_k)\geq\sqrt A$; por tanto,
$\mu_G=\sum p_kT_A(a_k)\geq\sqrt A$. La suma es exacta para la ley de tres
átomos aprendida, no para la ley desconocida del mercado. Con $V=U^2$,
$v\mapsto\sqrt{A+cv}$ es cóncava, de modo que
$\mu_G\leq\sqrt{A+c\mathbb E_G[V]}$. Si $A>0$ y V tiene soporte finito,
la cota por curvatura de 7.12 es válida. No se extiende sin condiciones a
$A=0$ con soporte en cero ni a una garantía sobre todo el pipeline.

**Riesgo condicionado al oráculo.** Sólo si G es la ley verdadera condicional,
la media transformada minimiza el riesgo cuadrático y la diferencia frente
al plug-in es el cuadrado de la diferencia entre ambas lecturas. Una ley
aprendida de tres átomos puede estar mal especificada; la identidad no prueba
mejor RMSE que una cabeza aditiva flexible ni convergencia del ajuste.

**CRPS_U no es una nueva regla.** Para $A>0$, el mapa
$y=T_A(u)=\sqrt{A+cu^2}$ es monótono en $u\geq0$ y su inversa es
$h_A(y)=\sqrt{(y^2-A)/c}$. Si $F_Y$ es la ley transformada, cambiar la variable
en la representación integral de CRPS da

$$
\operatorname{CRPS}(G_U,U)
=\int_{\sqrt A}^{\infty}
\bigl(F_Y(y)-\mathbf1\{Y\leq y\}\bigr)^2
\frac{y}{\sqrt c\sqrt{y^2-A}}\,dy.
$$ (eq-e3-r1-weighted-crps)

La expresión se entiende por el integral impropio en la frontera; para las
leyes finitas de la revisión es integrable. En $A=0$ el peso se reduce a
$1/\sqrt c$ fuera del punto frontera. Dividir por $s_U$ proporciona la
normalización del código. Por tanto, CRPS_U equivale a un CRPS ponderado en
Y; en general **no** es el CRPS sin ponderar de sigma7. El carácter propio
del score no demuestra calibración empírica de la ley finita ajustada.

La {download}`nota de cambio de variable <../_static/e3_r1_substantive_20261120/literature_distribution/change_of_variable_note.md>`
atribuye CRPS y su familia ponderada. La
{download}`auditoría formal <../_static/e3_r1_substantive_20261120/code_theory/theory_audit.md>`
expone supuestos, derivadas parciales y objeciones. El
{download}`verificador estático <../_static/e3_r1_substantive_20261120/code_theory/verify_static_audit.py>`
comprobó diez insumos, los 28 coeficientes racionales de la identidad de
sigma7 poblacional, los conteos 1210/1262/1236 y la coherencia de los cien
casos finitos **ya guardados**. No importó Torch, ejecutó redes o repitió el
piloto. El {download}`recibo <../_static/e3_r1_substantive_20261120/code_theory/static_verification.json>`
no se presenta como certificación automática de novedad.

U deriva de la misma etiqueta Y y A; no agrega información futura independiente.
La diferencia defendida es la elección de ley, lectura y objetivo. No se
reclama que un predictor aditivo suficientemente flexible no pueda producir
esa misma media. Esta delimitación permite defender la adaptación sin
atribuirle nuevas reglas de scoring o promesas de superioridad.


(e3-next-v4-design-20261121)=
## 7.16 Diseño V4: coordenada del score, normalización y empates

La revisión de desarrollo V4 conserva $T_A(u)=\sqrt{A+cu^2}$, $c=6/49$, las tres masas/posiciones aprendidas y la media $\mu=\sum_kp_kT_A(a_k)$. A sigue dependiendo sólo de la historia conocida. La arquitectura y sus **1262 parámetros** no cambian; las etiquetas Y/U sólo se utilizan en la pérdida de entrenamiento. La {download}`implementación local <../_static/e3_next_evidence_20261121/development/outcome_crps_v4.py>` hereda el forward de V3 y verifica sus fuentes fijadas.

Para $z_k=T_A(a_k)$, la pérdida nueva es

$$
\mathcal L_{\mathrm{V4}}=
\frac{(\mu-Y)^2}{s_Y^2}
+0.1\,\frac{\operatorname{CRPS}(\sum_kp_k\delta_{z_k},Y)}{s_Y},
\qquad
\operatorname{CRPS}=\sum_kp_k|z_k-Y|
-\frac12\sum_{k,\ell}p_kp_\ell|z_k-z_\ell|.
$$ (eq-e3-v4-outcome-score)

$s_U$ y $s_Y$ son escalas de entrenamiento fijadas, no parámetros aprendidos o coeficientes elegidos por el resultado. El diagnóstico sintético utiliza sus medias exactas bajo cada ley finita, como definió el plan; no estima una escala con objetivos de prueba. V3 utilizaba `CRPS_U/sU`, de modo que **coordenada y escala se modifican a la vez**. No se atribuye toda diferencia a una sola de ellas.

### Qué justifica la transformación y qué no demuestra

La ecuación {eq}`eq-e3-r1-weighted-crps` expresa CRPS_U como un CRPS en Y ponderado por el jacobiano inverso $dU/dY=Y/(cU)$. Para A>0, el peso tiene una singularidad integrable junto al piso $\sqrt A$. En las leyes del diagnóstico con **A fijo igual a cero**, el mapa es lineal y sus medias satisfacen `sY=sqrt(c)*sU`, por lo que `CRPS_U/sU = CRPS_sigma/sY`. No se extiende esa identidad a escalas globales de un entrenamiento con A variable por observar un único caso A=0. Ambas reglas siguen siendo propias para leyes no restringidas y la transformación es inyectiva. **No se deriva de ello un óptimo poblacional incorrecto de V3.** La cuestión nueva es el comportamiento de optimización y la interacción con MSE en una ley de tres átomos posiblemente mal especificada.

El {download}`plan previo y sus argumentos <../_static/e3_next_evidence_20261121/development/METHOD_AND_PRIOR_PLAN.md>` fijó cuatro controles de pérdida, todos con la misma MSE:

| Receta | Auxiliar y función del control |
|---|---|
| U-CRPS original | `0.1*CRPS_U/sU`, referencia de la pérdida V3. |
| V4 | `0.1*CRPS_sigma/sY`, coordenada de la etiqueta y su escala. |
| σ₇ con escala emparejada en A=0 | `0.1*CRPS_sigma/(sqrt(c)*sU)`, control que separa el cambio de norma por normalización. |
| MSE-only | Sin auxiliar; comparador de diagnóstico, sin promoción automática. |

Una escala positiva sólo cambia la norma de un gradiente auxiliar, no su dirección. El control emparejado no crea un modelo puntual más expresivo ni más parámetros. Si una revisión de mecanismo o regla posterior alterase estas condiciones, sería otra versión registrada y evaluada primero en desarrollo; no una modificación del piloto V3 terminado.

### Diagnóstico fijo con todos los casos conservados

La cuadrícula previa combinó seis valores de A —incluidos cero, casi piso y 100—, dos leyes verdaderas finitas, cuatro tríos de átomos y tres vectores de probabilidades: **144 casos**. Se calcularon expectativas y gradientes de cinco parámetros de cabeza sin ajustar encoder u optimizador. La {download}`regla previa de empates <../_static/e3_next_evidence_20261121/development/TIE_HANDLING.md>` reconoce que los valores absolutos no tienen un gradiente único en ciertas coincidencias. La regla fijada antes de los resultados marcó 72 casos sensibles durante el diagnóstico, que permanecen en el CSV; el subgradiente automático no se presenta como una dirección diferenciable única.

Los 72 restantes tienen doce casos diferenciables por A. En A=0, los errores máximos fueron **2,78×10⁻¹⁶ en scores normalizados** y **1,39×10⁻¹⁶ en gradientes**. En A>0, 50 de 60 direcciones U/σ₇ no fueron colineales. La alineación auxiliar con MSE mejoró en 49 casos, empeoró en **11** y quedó igual en 12. Las comparaciones por A y las pendientes puntuales se conservan completas en {ref}`6.19 <e3-next-v4-development-20261121>`. Esto sostiene sólo una diferencia geométrica acotada; no demuestra mejor optimización durante entrenamiento, generalización, superioridad de mercado o R1 de V4.

Las {download}`catorce pruebas de software finales <../_static/e3_next_evidence_20261121/development/software_validation.json>` verificaron fuente, inicialización y forward idénticos, soporte, escalas fijas, gradientes y manejo de empates. El fallo previo de una fixture float32 y el código anterior permanecen conservados; pasar esas guardas no equivale a entrenar el modelo. El paso posterior al piloto real exigió otro plan de desarrollo previo, presupuesto y controles comunes, registros de fallos y criterio de avance/abandono, documentados a continuación. **La receta V3 mantiene NO-GO y la regla original; V4 no hereda su dictamen R1 ni la auditoría de la aditiva.**


### Plan separado del piloto real CPU y ejecución completada

El {download}`plan del nuevo piloto CPU <../_static/e3_next_evidence_20261121/development/cpu_pilot_public/cpu_pilot_plan.json>` fijó **cuatro recetas×cinco semillas: veinte ajustes nuevos de desarrollo**, sin HPO. Las recetas entrenables son V4, U-CRPS original, MSE-only y aditiva. Las tres primeras conservan 1262 parámetros e inicialización común; la aditiva tiene 1210 y comparte encoder/atención, sin declarar capacidad total idéntica. Los controles `floor_only` y `floor_plus_mean_excess` se evalúan una sola vez, sin fingir cinco semillas deterministas. El control de escala emparejada del diagnóstico sintético no se convierte en una quinta receta entrenada ni se afirma una ablación factorial completa.

La partición expuesta conserva **532 filas de ajuste, 54 de parada/selección, siete purgadas y 55 de evaluación de desarrollo**. Las semillas nuevas se fijan en `20261121–20261125`; tasa común `0.003`, batch 32, máximo 40 épocas y paciencia ocho. La cohorte efectivamente usada fue CPU, Python 3.12.14/Torch 2.8.0+cpu/NumPy 2.3.5, dos hilos de cómputo y algoritmos deterministas. Se entrenaron y validaron los veinte checkpoints antes de abrir la evaluación de 55 fechas. No se reutilizan pesos GPU antiguos para llamar justa a una comparación nueva ni se presenta este alcance como reproducción de aquella sesión.

Cada ajuste dispone de 120 s nominales de fase y 150 s de proceso, con límite global de 3300 s; no son consumo medido o coste energético/FLOPs. La regla nueva de desarrollo exige **al menos 2 % de mejora RMSE media frente al mejor control neuronal y frente al mejor de los cinco controles**, conservando ambos dictámenes y todos los resultados. Fallar implica abandonar la receta V4 actual, sin promover automáticamente un control ni cambiar la regla V3 anterior. El plan y la fuente se [anclaron públicamente antes del primer ajuste](https://github.com/jassan-bit/DeepLearning_202630/commit/5ee55723e7d1ce63dbe4a94dca8941cb85a45fee). El {download}`recibo de preparación conservado <../_static/e3_next_evidence_20261121/development/cpu_pilot_public/prepared_receipt.json>` mantiene su instante previo y sus ceros, separado del {download}`recibo ejecutado posterior <../_static/e3_next_evidence_20261121/development/cpu_pilot_results/pilot_result.json>`: **20 ajustes, 22 trayectorias y 1.210 predicciones de desarrollo**. V4 fue **1,443261 % peor** que MSE-only y falló ambas puertas del 2 %; se registra **abandono de la receta V4 actual**, sin promover un control o modificar el piloto V3. Sus 90,096 s totales/54,926 s de fases no se sustituyen por los 2.400 s nominales. No se evaluó la familia completa ni datos nuevos de confirmación.
