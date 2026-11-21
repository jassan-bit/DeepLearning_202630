# Dictamen propio R1 de `input_only_mse`

Fecha: 21 de noviembre de 2026 (UTC). Objeto: la receta `input_only_mse` del código piloto fijado en `c4ca86f1431afeaa958ff0b544b453ab17a4d7b8`, examinada como candidata distinta de la receta completa V5. Revisión documental y algebraica con asistencia de IA; cero entrenamientos, pasos de optimización, predicciones nuevas o remuestreos en este subexpediente.

## Decisión y alcance

**CONTINUAR como candidata propia para preparar una futura confirmación R3.** El algoritmo completo tiene una adaptación estructural defendible bajo R1: representa mediante una ley finita positiva el único componente todavía desconocido de una etiqueta de volatilidad móvil parcialmente observada y aplica la transformación exacta dependiente de la historia **antes** de promediar. En esta variante, la ley latente se condiciona a una forma OHLC normalizada causalmente y conserva una escala física global de ajuste. Esa lectura permanece implementada y diferenciada de los antecedentes próximos de normalización, regresión directa y densidad financiera.

La conclusión **no se fundamenta en que dividir por una escala local, retirar CRPS o combinar piezas conocidas sea novedoso**. La normalización de entrada no constituye, por sí sola, una modificación sustantiva frente a DAIN/RevIN; frente al control interno `global_mse`, la diferencia es exactamente el preprocesamiento. La contribución defendida incluye el núcleo propio procedente de V3 y se evalúa frente a literatura externa para el mismo fenómeno/tarea. La guía no exige reinventar el núcleo frente a cada variante interna anterior. Se vuelve a examinar aquí lo que sobrevive de ese núcleo; no se hereda automáticamente el dictamen favorable de V3 ni su propiedad de identificación con CRPS.

Este dictamen técnico respalda una pretensión académica acotada de R1, revisable si aparece un antecedente equivalente. **No es aceptación del docente, firma humana, acreditación institucional de R1/R2/R3, autorización de entrenamiento en esta tarea ni promesa de obtener R3.** El historial de selección posterior a resultados y el abandono de V5 siguen siendo hechos adversos separados de la originalidad del mecanismo.

## Regla exacta de la guía

La guía original, §6, pp.7–8, exige dos elementos simultáneos: no haber usado el modelo para el mismo fenómeno y tarea, y una diferencia de mecanismo, pérdida o entrenamiento; una transferencia debe tener fuente precisa y modificación sustantiva. §6.1(iii), p.8, excluye una transferencia que sólo cambie preprocesamiento/hiperparámetros. §6.2, pp.9–10, admite módulos, adaptaciones algorítmicas y variantes con núcleo propio, con hipótesis y comparación de métodos próximos. §6.3, pp.11–12, exige revisión trazable y una declaración limitada a la búsqueda; no una demostración de ausencia mundial.

La lectura aplicada es estricta en el punto pertinente: si se presentara `input_only_mse` exclusivamente como «una nueva normalización de `global_mse`», **ese argumento no cumpliría R1**. Tampoco se describe como «RevIN reversible adaptado», porque no restaura escala local a la salida. El objeto defendido es el modelo estructurado completo, cuyo cambio de propagación frente a regresores directos/MDN no desaparece al retirar CRPS. La pérdida novedosa no es necesaria si existe una modificación de mecanismo sustantiva; R1 usa esas categorías como alternativas.

R2 (implementación propia y defensa), la propiedad formal/verificación y R3 (superioridad) se distinguen de R1. Las casillas conjuntas de §14.5.1, pp.43–44, no quedan todas acreditadas por una comparación bibliográfica. La guía permite IA declarada (§15, p.48) y exige que el estudiante comprenda y defienda cada línea. El expediente previo `requirements/guide_interpretation.md` documenta la modalidad de tercero mediante IA autorizada según el autor; aquí se emite un concepto específico de esta candidata sin pedir una firma humana nueva.

## Ecuaciones derivadas del código efectivo

### Información temporal y representación

Una ventana termina en el instante de emisión, después de observar sus siete filas OHLCV completas. Escribimos sus cierres como $C_0,\ldots,C_6$, sus seis retornos conocidos como $r_i=\log(C_i/C_{i-1})$, $i=1,\ldots,6$, y el siguiente retorno no observado como $r_7$. Precios positivos, volumen no negativo y geometría high/low válida son condiciones de admisión.

`data.py:41–59` calcula

\[
m_6=\tfrac16\sum_{i=1}^6r_i,\quad
A=\tfrac17\sum_{i=1}^6(r_i-m_6)^2,\quad
q=\sqrt{\tfrac16\sum_{i=1}^6r_i^2}.
\]

El ajuste estima una sola referencia $r_{\rm fit}=\sqrt{\operatorname{mean}_{\rm fit,i}(r_i^2)}>0$. La escala de cada ventana es $s=\max(q,10^{-6}r_{\rm fit})$. Para cada fila $j$ y canal de precio $d\in\{O,H,L,C\}$,

\[
F_{j,d}=\frac{\log P_{j,d}-\log C_6}{s},\qquad
F_{j,V}=\log(1+V_j),\qquad
X_{j,d}=\frac{F_{j,d}-\mu_{d,\rm fit}^{\rm local}}
{\sigma_{d,\rm fit}^{\rm local}}.
\]

Las medias y desviaciones se agrupan sobre filas/ventanas **de ajuste**, `ddof=0`; una desviación exactamente nula se sustituye por uno. No hay centramiento de precio por la media local: el ancla es el último cierre conocido. Tampoco hay RMS de volúmenes ni normalización de volumen por volatilidad. La posterior estandarización tiene constantes locales/globales distintas según representación, estimadas en el mismo ajuste. $X$ se entrega como FP32; geometría y lectura son FP64.

El argumento `scale=s/r_fit` llega a `forward_components`, pero `input_only_mse` lo ignora en su lectura: no es una característica del codificador ni una restitución implícita de unidades. $A$ entra separadamente en la fórmula de lectura; no se entrega al codificador.

### Identidad de la etiqueta solapada

La desviación poblacional (`ddof=0`) de los siete retornos $r_1,\ldots,r_7$ satisface, por expansión de la suma de cuadrados alrededor de su media,

\[
\sum_{i=1}^7(r_i-m_7)^2
=\sum_{i=1}^6(r_i-m_6)^2+\tfrac67(r_7-m_6)^2.
\]

Con $U=|r_7-m_6|$, $c=6/49$, la etiqueta es

\[
Y=T_A(U)=\sqrt{A+cU^2}.
\]

No se presenta esta identidad de varianza conocida como un teorema nuevo. Durante preparación se reconstruye $U=\sqrt{\max(Y^2-A,0)/c}$, con tolerancia de admisión $10^{-12}$ para el piso. Eso usa la etiqueta sólo en el ajuste/evaluación correspondientes. `input_only_mse` no usa $U$ en su objetivo, aunque la interfaz común lo valide; no dispone del retorno futuro al emitir un pronóstico.

### Codificador, ley latente y lectura

`model.py:88–103` reutiliza CNN causal de kernel3/8 canales y LSTM unidireccional/12 estados del backbone propio anterior. Sobre los estados $h_j$,

\[
q_\theta=\frac{q_{\rm raw}}{\sqrt{1+\|q_{\rm raw}\|^2}},\quad
\tau=.25+\operatorname{softplus}(\tau_{\rm raw}),\quad
\alpha_j=\operatorname{softmax}_j(h_j^\top q_\theta/\tau),\quad
h=\sum_j\alpha_jh_j.
\]

No se usan aquí penalización por discrepancia ni saturación radial del primer modelo aditivo. No se transfiere su cota Lipschitz al mapa actual.

Sean $z_k=(W_ah+b_a)_k$, $\delta=10^{-4}$, $s_U=\operatorname{mean}_{\rm fit}U>0$, $s_Y=\operatorname{mean}_{\rm fit}Y>0$. Para $K=3$,

\[
d_k=\delta+\operatorname{softplus}(z_k),\quad
u_k=s_U\sum_{i=1}^kd_i,\quad
(p_1,p_2,p_3)=\operatorname{softmax}(W_ph+b_p,0),
\]

\[
G_X=\sum_{k=1}^3p_k\delta_{u_k},\quad
\widehat Y=\mu_G(X,A)=\sum_{k=1}^3p_kT_A(u_k),\qquad
L=\frac1n\sum_t\left(\frac{\widehat Y_t-Y_t}{s_Y}\right)^2.
\]

`torch.hypot(sqrt(A),sqrt(c)*u)` implementa $T_A$ de forma estable. Un `maximum` final protege el piso ante redondeo. Hay 1262 parámetros entrenables, frente a 1210 de la lectura aditiva. No se usa CRPS ni se cambia el número de átomos por resultado. $s_U$ es común a las recetas y es **global y fijo durante inferencia**.

### Diferencias efectivas, sin heredar resultados entre cohortes

| Objeto | Entrada de precios | Átomos en unidades de U | Objetivo | Diferencia pertinente |
|---|---|---|---|---|
| `input_only_mse` | Relativa al último cierre, dividida por $s_t$; estandarización fit local | $s_Ua_k(X_{\rm local})$ | MSE de sigma7 | Candidata examinada: forma de entrada y lectura estructurada, escala global. |
| `global_mse` | Relativa, dividida por $r_{\rm fit}$; estandarización fit global | $s_Ua_k(X_{\rm global})$ | Mismo MSE | Sólo cambia la representación de entrada y sus estadísticas; misma estructura/capacidad y política de escala. |
| V3 `atom_mse_only` | Representación global relativa anterior | Átomos positivos con escala global | MSE sin CRPS | Antecedente interno de la misma lectura atómica; el control V5 global es algebraicamente equivalente según auditoría previa. Pesos, cohortes y resultados no se transfieren. |
| V5 completo `seoi_mse` | Misma entrada local que input-only | $s_U(s_t/r_{\rm fit})a_k$ | Mismo MSE | Sí restituye escala local dentro de $T_A$; sustenta equivariancia de salida bajo supuestos. Esa propiedad se pierde aquí. |
| `global_crps` / V3 `atom_crps` | Global | Global | MSE+0,1CRPS_U | Añade puntuación de la ley de U; input-only no puede heredar su resultado de identificación poblacional. |
| Aditiva | Global | Sin ley atómica | MSE | Predice $\sqrt A+s_Y[\delta+\operatorname{softplus}(w^\top h+b)]$; mantiene piso pero no transforma una ley antes de promediar. |

La división global constante $1/r_{\rm fit}$ se cancela algebraicamente con su media/desviación de ajuste en canales no degenerados. Por eso «global» no significa precios OHLC crudos sin transformar. Se conserva la auditoría previa `qa/THEORY_CODE_REVIEW.json`; no se reprodujeron aquí sus inferencias sintéticas.

## Propiedades correctas y pérdidas de garantía

### P1. Causalidad en el instante de emisión

Con preprocesador y pesos congelados, (X,A,s) son funciones sólo de la ventana observada; el futuro $Y,U,r_7$ no es argumento de `transform`/`forward_components`. Las estadísticas y escalas supervisadas usan índices de ajuste únicamente. El código admite sólo los tres miembros `training_raw`, `training_y_original`, `training_dates`; no deserializa calibration/evaluation. Hashear el NPZ completo no equivale a leer sus otras matrices.

La afirmación depende de que la fuente alinee correctamente ventana y etiqueta. La revisión de estas funciones demuestra el flujo de información, no reconstruye por sí sola toda la procedencia externa del NPZ. Además, normalizar toda la ventana con $C_6,s_t$ hace que un estado intermedio utilice estadísticas de filas posteriores **ya observadas al emitir**. Se demuestra causalidad del pronóstico al final de la ventana; no una representación incremental por prefijos ni un pronóstico intradía antes de completar OHLCV.

El calendario de desarrollo está expuesto: ajuste532 hasta2024-06-30, selección54 desde2024-07-08, evaluación de desarrollo55 desde2024-09-07 hasta2024-10-31 y dos purgas de siete fechas. Separación operativa no convierte esas55 fechas en prueba reservada. Tampoco protege contra el sesgo de escoger input-only después de ver resultados.

### P2. Soporte y esperanza bajo el modelo finito

Por softplus y acumulación, $0<u_1<u_2<u_3$, $p_k>0$ y $\sum p_k=1$, con parámetros finitos. Cada átomo transformado cumple $T_A(u_k)\ge\sqrt A$. Así,

\[
\widehat Y\ge\sqrt A,\qquad
\widehat Y=\mathbb E_{U\sim G_X}[T_A(U)].
\]

Es suma exacta bajo la ley finita implementada, sin cuadratura o Monte Carlo; no significa esperanza exacta bajo la ley de mercado. En aritmética real la desigualdad es estricta con átomos positivos. El `maximum` sólo añade guarda numérica. Esta propiedad heredada sigue siendo válida sin CRPS y sirve como propiedad formal delimitada, sin exigir que sea inédita.

### P3. Invariancia de la representación

Fijar pesos, $r_{\rm fit},s_U,s_Y,\mu_{\rm fit},\sigma_{\rm fit}$, volumen y convenciones de datos. Transformar **las siete filas** de log-precios por $\ell'_{j,d}=b+k\ell_{j,d}$, $k>0$. Entonces $r'=kr$, $A'=k^2A$, $q'=kq$. Si el piso no actúa antes ni después, $s'=ks$, y

\[
\frac{\ell'_{j,d}-\ell'_{6,C}}{s'}
=\frac{\ell_{j,d}-\ell_{6,C}}s.
\]

Con volumen igual, $X'=X$, y por tanto $p'=p$, $u'=u$, en aritmética real. La verificación de implementación debe admitir tolerancias por logaritmos, redondeo y casteo FP32. Es invariancia de la representación/ley latente ante transformación log-afín local, bajo condiciones; no ante cambios arbitrarios de régimen. Si actúa el piso físico fijo, en general $s'\ne ks$ y no vale esa demostración. Una ventana totalmente plana puede tener invariancia trivial, sin que ello repare el caso general. Para $k=1$, multiplicar todos los precios por la misma constante deja $X,A,\widehat Y$ invariantes incluso con piso; esta invariancia al nivel ya existe también en los controles relativos.

### P4. La salida NO conserva la equivariancia de V5 completo

Bajo los supuestos de P3, input-only conserva $p,u$, pero cambia $A$:

\[
\widehat Y'=\sum_kp_k\sqrt{k^2A+cu_k^2},\qquad
k\widehat Y=\sum_kp_k\sqrt{k^2A+k^2cu_k^2}.
\]

Para $k>1$, átomos positivos implican **$\widehat Y'<k\widehat Y$**. Para $0<k<1$, la desigualdad se invierte. Con $A=0$, la predicción queda idéntica al cambiar escala; es un contraejemplo directo a equivariancia. Para $A\ge0,k>1$, también $\widehat Y'\ge\widehat Y$: el efecto de escala pasa exclusivamente por la parte observada. V5 completo sí cambia $u'=ku$ por el factor local y obtiene $\widehat Y'=k\widehat Y$. Cotransformar las referencias de ajuste no rescata una propiedad distintiva de inferencia con estado fijo.

La subhomogeneidad puede ser una limitación importante si la magnitud futura crece con el régimen de escala; la menor RMSE de desarrollo no permite ignorarla. Tampoco prueba que deba imponerse equivariancia: se evalúa como sesgo inductivo cuya utilidad queda abierta a confirmación.

### P5. MSE no identifica ni calibra la ley atómica

Condicionando por la información observada $\mathcal I$, suponer $\mathbb E[Y^2\mid\mathcal I]<\infty$ y una predicción finita $\mu$ medible respecto de $\mathcal I$. Entonces

\[
\mathbb E[(Y-\mu)^2\mid\mathcal I]
=\operatorname{Var}(Y\mid\mathcal I)
+(\mu-\mathbb E[Y\mid\mathcal I])^2.
\]

El riesgo depende de $G$ sólo mediante el escalar $\mu_G$. Dos leyes distintas con la misma media transformada tienen exactamente el mismo MSE esperado. Ejemplo algebraico: con $A=0$ y unidades $s_U=1$, leyes de masas iguales sobre $\{.5,1,1.5\}$ y $\{.25,.75,2\}$ tienen $\mathbb E U=1$ y la misma $\mathbb E T_0(U)=\sqrt c$, pero distinta varianza y CDF. Ambas admiten los incrementos positivos requeridos. No se ejecutó este ejemplo como predicción nueva.

No hay término estrictamente propio que seleccione una ley única. La ley finita es una **parametrización latente del predictor puntual**; no se la vende como distribución predictiva calibrada ni se heredan intervalos, cobertura, PIT o propiedades de consistencia de V3 con CRPS. Ni siquiera un score propio garantizaría calibración empírica de una familia mal especificada de tres átomos. Si se evalúan distribuciones exploratoriamente, se reporta su estado sin prometer calibración.

## Literatura próxima y adaptación específica

Se contrastan nueve antecedentes externos y un antecedente interno. Las fuentes nuevas abiertas el 21 de noviembre incluyen textos primarios DAIN y RevIN, Non-stationary Transformers y CryptoL; las comparaciones especializadas MDN/MIDAS/FQF/CRPS/HARNet se apoyan además en cuerpos/recibos primarios preservados en el expediente R1 de V3, cuyo alcance se mantiene. Las referencias encontradas después no se inventan como inspiración histórica previa del autor.

| Antecedente verificable | Qué ya existe y diferencia relevante de input-only | Categoría y límite |
|---|---|---|
| Passalis et al. (2020), **DAIN**, TNNLS31(9):3760–3765, DOI[10.1109/TNNLS.2019.2944933](https://doi.org/10.1109/TNNLS.2019.2944933); [preprint1902.07892v2](https://arxiv.org/pdf/1902.07892), §II, pp.2–3. | Desplazamiento, escalado y gating aprendidos de la entrada; probado en libros de órdenes financieros/carga. Input-only usa escala RMS analítica de retornos conocidos y no aprende una capa DAIN. La diferencia sustantiva defendida está en la ley de la magnitud faltante y lectura por A, ausentes de ese método. | La normalización financiera ya existe. Sólo cambiar DAIN por RMS no acredita R1. |
| Kim et al. (2022), **RevIN**, ICLR2022; [registro](https://openreview.net/forum?id=cGDAkQo1C0p), [PDF primario](https://openreview.net/references/pdf?id=SRgrKlAhbc), §3.1, ecs.1–3, pp.3–5. | Normaliza estadísticas por instancia y las restaura simétricamente con transformación afín aprendida. Input-only no invierte estadísticas locales a la salida y pronostica otra variable: sigma7 desde una ley U y A observado. | No es RevIN reversible nuevo; diferencia de lectura para la etiqueta, con pérdida de equivariancia explícita. |
| Liu et al. (2022), **Non-stationary Transformers**, NeurIPS35; [texto primario v4](https://arxiv.org/html/2205.14415v4), §§3.1–3.2. | Stationarization y atención que reincorpora información eliminada. Input-only no implementa esa atención; fuerza p/u iguales ante ventanas de la misma forma normalizada. | Antecedente y objeción: puede perder señal de régimen. No se afirma haber resuelto over-stationarization. |
| Taheri et al. (2026), **CryptoL**, [arXiv2609.11206v1](https://arxiv.org/html/2609.11206v1), §§3.1–3.4, resumen. | Integra normalización de OHLC cripto, epsilon dinámico, MSE en espacio normalizado y restricciones suaves de velas para pronosticar OHLC multiactivo. Input-only puntúa sigma7 en unidades físicas globales y aplica soporte por identidad de etiqueta a una ley U. | Nuevo antecedente cercano: normalizar OHLC cripto y usar restricciones ya es conocido. No se describe allí esta etiqueta/lectura en las secciones examinadas; no se extrapola ausencia a todo el campo. |
| Schittenkopf, Dorffner y Dockner (1998), **Identifying Stochastic Processes with Mixture Density Networks**, WP11; [PDF institucional](https://research.wu.ac.at/ws/portalfiles/portal/18954278/document.pdf), §2 ecs.1–9 y §3. | Estimación financiera mediante mezclas gaussianas condicionadas y verosimilitud. Input-only aprende átomos positivos de U con MSE de su esperanza transformada por A, no NLL de una mezcla sobre retornos. | Mecanismo de lectura para etiqueta parcialmente observada. Se conserva la fe de erratas: este PDF no es el distinto WP15 *Volatility Prediction with Mixture Density Networks*. |
| De Stefani et al. (2017), **Machine Learning for Multi-step Ahead Forecasting of Volatility Proxies**, [CEUR/MIDAS](https://ceur-ws.org/Vol-1941/MIDAS2017_paper3.pdf), §2 y §3 ecs.17–22. | Regresión directa NAR/NARX/MLP de proxies de desviación móvil; volatilidad móvil financiera no es nueva. Input-only separa la parte ya observada y sólo parametriza el grado futuro faltante antes de la esperanza. | Cambio de propagación respecto regresión directa; cambiar ventana/ddof/activo por sí solo no sería diferencia sustantiva. |
| Yang et al. (2019), **FQF**, NeurIPS32; [fuente primaria](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f471223d1a1614b58a7dc45c9d01df19-Abstract.html). | Parametriza fracciones y valores de una distribución en RL. Input-only no usa Bellman ni actualizaciones de cuantiles: usa ley positiva latente, mapa dependiente de A y supervisión puntual de sigma7. | Aprender posiciones/masas no se reclama nuevo. Se transfieren ideas generales de representación, no las garantías de RL. |
| Gneiting y Raftery (2007), **Strictly Proper Scoring Rules, Prediction, and Estimation**, JASA102:359–378, [DOI](https://doi.org/10.1198/016214506000001437), §§4.2 y6.4; comparación primaria preservada en expediente V3. | CRPS y su fórmula de esperanzas son conocidas; reglas propias identifican distribuciones bajo condiciones. Input-only retira ese término y sólo elicita la media de Y. | Diferencia de pérdida conocida, no innovación; obliga a retirar la garantía distributiva, no la lectura estructurada. |
| Reisenhofer, Bayer y Hautsch (2022), **HARNet**, [fuente primaria](https://arxiv.org/html/2205.07719v1), §§2,4.1–4.3; lectura previa preservada. | CNN causal con estructura HAR para varianza realizada y comparación de pérdidas. Input-only usa identidad del objetivo solapado en lectura, no filtros/inicialización HAR. | Los sesgos inductivos financieros y CNN ya existen. No hereda rendimiento/garantías HARNet; se discute como paper guía/trabajo cercano. |
| **V3 propio**, `e3_overlap_model_v3.py`, expediente `e3_r1_substantive_20261120`. | Ya implementaba átomos ordenados, masas, transformación antes de esperanza y ablación MSE. Aquí se conserva ese núcleo sin CRPS y se cambia condicionamiento a X local. | Antecedente interno íntegramente atribuido. No nueva invención del núcleo ni nueva fecha de diseño; su dictamen bibliográfico se usa como evidencia, no acreditación automática. |

### Objeción decisiva y respuesta

**Objeción:** al retirar CRPS, sólo queda una mezcla de tres bases no lineales con un preprocesamiento conocido; podría ser una composición ordinaria, incluso menos identificada que V3.

**Respuesta:** es correcto que no existe identificación distributiva y que la mejora input-only/global sólo aísla preprocesamiento. Sin embargo, persiste una modificación de estructura de propagación frente a las lecturas externas examinadas: salida positiva latente U, dependencia determinista en A observado y promedio del push-forward exacto de una etiqueta parcialmente conocida. No es una cabeza genérica que deba reaprender todo el proxy ni una inversión afín de precio. La forma de la etiqueta decide dónde entra A y qué función de los átomos se optimiza. Esto basta para sostener la **categoría de adaptación algorítmica sustantiva** de un modelo completo bajo la guía, sin afirmar una clase universal de funciones puntuales mayor ni un score nuevo.

La combinación es transferible a pronóstico de desviación móvil poblacional de ventana $n\ge2$ con $n-1$ retornos conocidos: $A=\sum_{1}^{n-1}(r_i-m_{n-1})^2/n$, $c=(n-1)/n^2$, $U=|r_n-m_{n-1}|$. No se implementa otra arquitectura aquí ni se propone cambiar n tras resultados; la fórmula explica por qué el mecanismo supera una coincidencia de ticker/ventana7. Si sólo se aceptara como contribución lo añadido después de V3, el incremento disponible sería preprocesamiento y la respuesta a ese objeto más estrecho sería negativa. No es el objeto de R1 que fija la guía.

## Evidencia bibliográfica acumulada y límites

El expediente histórico conserva18 respuestas nativas válidas (seis por Crossref/arXiv/OpenAlex), seis cadenas bilingües y seguimiento de citas/repositorios, con límites de primeras páginas y semántica de consulta. Esa búsqueda histórica era más amplia y no se renombra como nueva búsqueda nativa de input-only. La revisión sustantiva V3 agrega comparaciones de aplicación/distribución y concepto IA específico. Para este objeto se repitieron siete consultas focales web el 21 de noviembre; sus cadenas, fuentes primarias y límites están en `R1_SEARCH_RECEIPT.json`. Un agente crítico separado repitió dos cadenas y emitió concepto sobre **esta** receta $`R1_INDEPENDENT_AI_REVIEW.json`$. No son tres bases nativas nuevas ni una persona humana.

La búsqueda actual devuelve RevIN y normalización financiera/cripto; éstos se discuten como antecedentes adversos. No se usa «cero resultados» como prueba. Los cuerpos focales revisados no describen la misma lectura de ley U y sigma7 parcialmente observada. La declaración defendible es: **según el corpus acumulado y la actualización focal descritos, no se identificó en el material examinado un modelo equivalente para pronóstico de esta volatilidad móvil solapada; se propone una adaptación propia de lectura estructurada condicionada a forma causal normalizada**. No se afirma prioridad mundial, exhaustividad o ausencia de antecedentes inaccesibles.

No queda un impedimento sustantivo documental que obligue a descartar ahora la candidata por R1. Sí quedan obligaciones separadas antes de acreditar todo el modelo original o lanzar una confirmación: la prueba numérica propia de las propiedades y limitaciones, defensa del estudiante y protocolo R3 completo. Identificarlas no posterga el dictamen R1 ni convierte este documento en un listado de tareas: la decisión actual es CONTINUAR y el fundamento es el mecanismo íntegro, con la pretensión acotada anterior.

## Resultado histórico que este dictamen no cambia

Los796 trabajos corresponden a la arquitectura aditiva. V3 conserva el2,7043% adverso de `atom_crps` frente a `atom_mse_only`; V4 conserva abandono. V5 completo conserva `ABANDON_CURRENT_SEOI_RECIPE`; su regla no se aplica de nuevo a una ablación para proclamar éxito. La selección de input-only se produjo **después** de revisar las30 ejecuciones/seis recetas/cinco semillas y las55 fechas expuestas. La auditoría cuantitativa general del cierre contrasta predicciones originales y hashes; este subexpediente no sustituye esa recalculación ni verifica cifras mediante nuevos pronósticos.

Continuar significa congelar una identidad propia y preparar un protocolo revisable con todos los rivales/condiciones de la guía. No significa congelar como «suficiente» un protocolo sin potencia/datos reservados resueltos. Ni los cinco inicios aleatorios ni las55 fechas se convierten en275 unidades independientes. La decisión favorable R1 no se puede usar para vender un resultado confirmatorio de RMSE, calibración o generalización.

## Procedencia, licencias y uso de IA

Se consultaron artículos y metadatos por enlace; no se copia ni incorpora código externo de DAIN/RevIN/FQF/HARNet/CryptoL. La implementación revisada usa primitivas PyTorch y código propio previo del proyecto; esta revisión no certifica la autoría/defensa humana. RevIN oficial se documentó previamente con MIT; en la consulta actual una ruta de licencia `master/LICENSE` devolvió404, por lo que no se usa esa respuesta como nueva verificación. El repositorio DAIN abierto no mostró archivo de licencia en su listado raíz; no se infiere permiso de copia por ser público. CryptoL HTML indica CC BY4.0; Non-stationary Transformers v4 indica CC BY-NC-ND4.0. No se redistribuyen aquí sus cuerpos ni se hereda una licencia para el paquete propio. Los artículos restantes se citan bajo los derechos de sus autores/editoriales; las ecuaciones particulares de esta adaptación se derivan del código del proyecto.

La IA asistió en lectura de código/guía, búsqueda focal, álgebra, revisión crítica y redacción. El agente crítico no participó en entrenar o seleccionar esta variante y su independencia es funcional dentro de la misma asistencia IA, no independencia humana ni institucional. El autor debe comprender y poder modificar en vivo cada fórmula y línea entregada. La modalidad IA aceptada según el registro previo no permite afirmar que ya ocurrió esa defensa.

Los hashes de fuentes efectivamente leídas y de los documentos derivados quedan en `R1_SOURCE_MANIFEST.json`. No se modificaron archivos de V3/V4/V5 ni el libro en este trabajo.
