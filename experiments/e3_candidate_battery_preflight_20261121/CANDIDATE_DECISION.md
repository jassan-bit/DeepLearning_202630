# Decisión de identidad y continuidad de la candidata

**Resultado: `NO_CONFIRMATORY_CANDIDATE`. `candidate_id=null`; `eligible_for_confirmation=false`.** Es una decisión sobre el expediente actual, no una prueba de imposibilidad de futuras propuestas. Se conserva V3 completa como objeto documental de la contribución revisada, sin seleccionarla para confirmar.

Fuente base: [`5f29ab5`](https://github.com/jassan-bit/DeepLearning_202630/tree/5f29ab51e1bb3e79d972377b6d682d1785bdc298). Asistencia de IA declarada. No se ejecutó ningún modelo o estudio nuevo.

## Identidades que no pueden combinarse

| Versión | Objeto y evidencia propia | Decisión |
|---|---|---|
| Aditiva de 1210 parámetros | CNN–LSTM con atención y exceso positivo sobre piso observable; 796 trabajos. Hay resultados retrospectivos favorables y límites documentados. | Benchmark histórico. La revisión no acredita adaptación sustantiva R1; tampoco la declara refutada automáticamente. Su rendimiento no completa R1 de V3. |
| V3 `atom_crps`, 1262 parámetros | Ley finita de la magnitud desconocida, transformación por historia antes de la esperanza y pérdida MSE/CRPS_U; 30 ajustes, tres semillas finales por receta y 55 fechas. | Dictamen técnico R1 favorable y acotado. **NO-GO**: 2,7043119888 % peor en RMSE que `atom_mse_only`, sin superar la regla interna de mejora del 2 %. R2 no observada y R3 no acreditada. |
| V3 `atom_mse_only` | Ablación sin el auxiliar CRPS_U, mejor control del piloto V3. | Control, no candidata promovida. Retirar una parte de la contribución cambia el objeto al que se refería el dictamen. Requeriría justificación y revisión propias. |
| V4 `sigma_crps_v4`, 1262 parámetros | Cambia simultáneamente la coordenada del CRPS y su escala; conserva encoder y lectura V3. Piloto CPU de 20 ajustes totales entre cuatro recetas. | **ABANDON_CURRENT_V4_RECIPE**: 1,4432610847 % peor que el mejor control CPU. No hereda R1 de V3. |
| Control CPU `mse_only_control` | Otra cohorte y comparación, con su propia identidad y resultados. | No se agrupa con V3 GPU ni se promueve por ranking. |
| Cabeza de momentos y bosquejo Faithful | Antecedentes de desarrollo distintos; el bosquejo no tiene identidad histórica completa suficiente. | No se rescatan resultados ni se completan retrospectivamente diseños o ejecuciones. |

Fuentes: [modelo, §§6.15–6.19](https://github.com/jassan-bit/DeepLearning_202630/blob/5f29ab51e1bb3e79d972377b6d682d1785bdc298/book_e3/sections/06_modelo_original.md); [diseño, §§7.8, 7.14–7.16](https://github.com/jassan-bit/DeepLearning_202630/blob/5f29ab51e1bb3e79d972377b6d682d1785bdc298/book_e3/sections/07_diseno_riguroso.md); tablas V3 y recibo V4 vinculados en INPUT_MANIFEST.json.

## Contribución de V3 que sí puede describirse

Con los seis retornos conocidos se definen su media m6 y el término observable A. Para el retorno siguiente se considera U=|r7−m6| y

\[
T_A(u)=\sqrt{A+(6/49)u^2},\qquad
G=\sum_{k=1}^{3}p_k\delta_{a_k},\qquad
\widehat y=\sum_k p_k T_A(a_k).
\]

La red produce masas p_k y átomos a_k no negativos. La pérdida fijada es L=MSE(mu,Y)/sY²+0,1·CRPS(G,U)/sU, con escalas determinadas en ajuste. La transformación de cada átomo precede a la esperanza. No se afirma que la distribución real sea de tres átomos.

| Antecedente próximo | Elemento conocido | Diferencia específica documentada y límite |
|---|---|---|
| MDN financiero, Schittenkopf et al. (1998) | Densidad condicional mediante mezclas neuronales. | V3 supervisa U y compone una lectura dependiente de la historia de la etiqueta solapada. No inventa las mezclas. |
| De Stefani et al. (2017) | Regresión neuronal de proxies de volatilidad móviles. | La etiqueta móvil ya tiene antecedentes; se distingue la representación y composición específica de V3. |
| FQF, Yang et al. (2019) | Posiciones y masas aprendidas de una distribución. | No se atribuye novedad a aprender átomos/masas. V3 tiene variable supervisada, transformación y pérdidas diferentes del ámbito Bellman/cuantiles. |
| Feng et al. (2023) | Mezclas neuronales de volatilidad GARCH y verosimilitud. | No es la misma variable objetivo ni lectura; la utilidad adicional sigue sin confirmarse. |
| CRPS, Jensen y puntuación ponderada | Reglas y resultados matemáticos conocidos. | Se atribuyen; la contribución propuesta es una adaptación concreta de representación, lectura y entrenamiento. |

Esta tabla reutiliza la revisión focal de §6.18 y sus límites de consulta; no afirma nuevas lecturas primarias, búsqueda exhaustiva o ausencia mundial de antecedentes. Su dictamen no se traslada a la aditiva, V4 o MSE-only.

## Propiedad formal y alcance

Con A,a_k,p_k≥0 y suma p_k=1, la lectura respeta \(\widehat y\ge\sqrt A\) y calcula exactamente la esperanza bajo G. Jensen determina la dirección frente al plug-in del segundo momento. La optimalidad MSE de la media transformada necesita que G sea la ley condicional verdadera; no demuestra que una familia finita la aprenda. La propiedad no acredita calibración, convergencia, dominancia expresiva ni menor RMSE de mercado. Véase §7.15 y su verificación previa.

La hipótesis del piloto V3 era que la composición completa ofrecería la mejora mínima prefijada frente a los controles. El resultado no superó esa regla. Sus controles examinan lectura/agregación, pérdida y masas mediante `atom_plugin`, `atom_mse_only` y `fixed_probs`; la heurística de continuación es una regla de decisión, no un componente ablado. Tres semillas finales y un calendario expuesto no establecen una atribución causal general del fracaso.

## Por qué no se formula V5 en este cierre

El expediente no identifica una causa empírica única que fundamente una nueva adaptación sustantiva concreta. Cambiar K, lambda, semillas o umbral sería otra decisión de desarrollo; el ranking adverso no demuestra que CRPS sea impropio. V4 ya mostró diferencias geométricas sin beneficio predictivo. Una cabeza inspirada en Faithful requeriría una definición y revisión nuevas, y conservar exactamente los gradientes de una media MSE no demostraría ganancia puntual.

Por ello no se inventa una propuesta nueva para llenar una casilla. MSE-only queda como control y V3 como referencia del mecanismo, ambos sin promoción. El 2 % es una regla interna de desarrollo conservada; no es el margen universal de R3.

## Qué necesitaría una propuesta posterior

Antes de un piloto separado: identidad única y versión, limitación con evidencia, mecanismo sustantivo contrastado con antecedentes, hipótesis falsable, métrica y regla de abandono anteriores a resultados, controles que aíslen el mecanismo, datos de desarrollo admitidos y presupuesto concreto. No se exige que ese piloto tenga ya la potencia final que pretende ayudar a dimensionar.

Si un piloto futuro satisface su regla, ello sólo justificaría estudiar continuidad. Harían falta revisión R1 propia, comprensión/defensa R2, datos confirmatorios adecuados, métodos admitidos y diseño de potencia. Ninguna propuesta hereda los 796 trabajos o la opinión R1 de otra identidad.

## Decisión inmediata preparada

La alternativa de menor coste y defendible con el expediente es un informe de benchmark, alcance de la adaptación y límites observados, preservando los resultados adversos; la guía §14.5.9, p.47, permite definir un hallazgo alternativo. Adoptarlo no acredita R3 ni determina automáticamente una nota o tope.

El autor tiene preparado [AUTHOR_REVIEW.md](AUTHOR_REVIEW.md) para aceptar esa ruta o aportar una propuesta sustantiva separada. No se inventa su elección, firma o comprensión. La investigación estadística puede continuar sin seleccionar una receta, dentro de [STATISTICAL_BATTERY.md](STATISTICAL_BATTERY.md); eso no habilita una campaña.
