# V5 propuesta: coordenadas causales de escala y reconstrucción de la innovación solapada

Fecha de preparación: 21 de noviembre de 2026. Asistencia de IA declarada. Documento de diseño nuevo; no es un preregistro de campañas anteriores ni contiene entrenamientos, predicciones de mercado nuevas o una acreditación docente.

## Decisión concreta y motivación delimitada

Preparar `seoi_mse`, una adaptación del principio de normalización reversible a una salida de volatilidad cuya etiqueta contiene seis retornos ya observados y una innovación desconocida. La modificación conjunta se hace dentro del mapa de predicción: convertir OHLC a geometría logarítmica relativa, quitar una escala causal de los seis retornos observados, y reintroducirla en las magnitudes latentes antes de aplicar la transformación exacta de la etiqueta. No se añade la media del precio al pronóstico de volatilidad.

El objetivo inmediato es probar una hipótesis de sesgo inductivo temporal, no corregir una causa supuestamente identificada. El expediente observó inversión entre selección y evaluación de desarrollo y fracasos de V3/V4. Eso permite preguntar si una representación dependiente de unidades/global-estadísticas generaliza peor al variar la escala de retornos. No demuestra que la normalización cause las pérdidas, ni que esta propuesta mejorará.

Se prefiere esta propuesta a una cabeza meramente faithful: con arquitectura puntual y ejecución idénticas, preservar exactamente los gradientes de MSE reproduce su trayectoria, de modo que exigirle vencer a su propia ablación MSE sería contradictorio. Se prefiere también a presentar el centrado por el piso como novedad de una mezcla Hedge: restar el mismo piso al objetivo y a todos los expertos deja algebraicamente idéntica la MSE.

## Datos y fórmula del objetivo

Una ventana termina al cierre t y tiene siete filas OHLCV positivas en precios. Los seis retornos conocidos son r_i=log(C_i/C_(i-1)), i=1,...,6. Definir

    m = mean(r_1,...,r_6)
    A = (1/7) * sum_i (r_i-m)^2
    c = 6/49
    U = abs(r_future-m)
    Y = sqrt(A+c*U^2).

Sólo las primeras tres cantidades son entradas. U y Y se usan exclusivamente como etiquetas. Ni el retorno futuro ni estadísticas de stop/assessment definen la escala.

Definir r_fit>0 como RMS de retornos admitidos del segmento de ajuste, calculado una sola vez; epsilon=1e-6; y

    R_t = sqrt(mean_i r_i^2)
    s_t = max(R_t, epsilon*r_fit).

El RMS sin centrar mantiene sensibilidad al desplazamiento de los retornos conocidos; el piso garantiza que una ventana plana sea computable. No usar R_t como canal adicional sin normalizar: rompería la propiedad local que se quiere imponer.

Para cada una de las cuatro columnas de precio Q y cada fila j:

    Z_(j,Q) = (log(Q_j)-log(C_t))/d_in(t).

La quinta columna será log1p(V) estandarizado con media/desviación sólo de ajuste, idéntico en todas las recetas. La transformación de precio considerada en el teorema no cambia V. No hace falta introducir otra normalización de volumen para esta hipótesis.

El encoder causal CNN-LSTM-atención heredado procesa Z en float32. Una lectura en float64 obtiene tres incrementos positivos ordenados y dos logits libres de masas:

    b_k = sum_(j<=k) [1e-4+softplus(h_j(context))]
    p = softmax(l_1,l_2,0).

Sean u_fit=mean_fit(U)>0 y y_fit=mean_fit(Y)>0, estimados sólo en ajuste y comunes a todas las recetas; se rechaza una escala degenerada. Definir gamma=u_fit/r_fit. No se usa mean_fit(U/s_t), ni se recalibran estos factores en stop o assessment. La lectura será

    a_k = gamma*d_out(t)*b_k = u_fit*(d_out(t)/r_fit)*b_k
    z_k = hypot(sqrt(A),sqrt(c)*a_k)
    mu = sum_k p_k*z_k.

En la interfaz del código, `scale=s_t/r_fit` es adimensional y `u_scale=u_fit` tiene unidades de retorno. Las cuatro columnas de precio y la columna de volumen se estandarizan después con medias/desviaciones calculadas sólo en ajuste, por separado para las coordenadas locales y globales. El divisor constante r_fit de las coordenadas globales cancela contra esas estadísticas: reproduce algebraicamente la transformación global de `src/e3_innovation_model.py:133–144`, salvo redondeo. Se conserva esa atribución interna.

La candidata usa d_in=d_out=s_t. No usa CRPS auxiliar. Esta es una identidad nueva de desarrollo; no se promueve `atom_mse_only` histórico.

Pérdida primaria:

    L_MSE = mean[((mu-Y)/y_fit)^2],

donde y_fit es una escala positiva fijada en ajuste. Así se optimiza la misma métrica absoluta evaluada: no se cambia silenciosamente a error relativo por ventana. Una sensibilidad CRPS, si se registra, usa exactamente las mismas unidades y coeficiente en sus controles; no debe convertirse en candidata por ranking.

## Factorial mínimo que identifica el mecanismo

| Receta | d_in | d_out | Pérdida | Papel |
|---|---|---|---|---|
| `seoi_mse` | s_t | s_t | MSE | Única candidata predeterminada |
| `input_only_mse` | s_t | r_fit | MSE | Ablación de reconstrucción |
| `output_only_mse` | r_fit | s_t | MSE | Ablación de entrada |
| `global_mse` | r_fit | r_fit | MSE | Control estático factorial |
| `global_crps` | r_fit | r_fit | MSE+CRPS_U | Sensibilidad al auxiliar, no candidata |
| `additive_mse` | entrada global declarada | exceso positivo sobre sqrt(A) | MSE | Referencia aditiva reentrenada en esta cohorte |

Cinco semillas son cinco repeticiones por receta, no treinta diseños. Se registra un mecanismo propuesto, tres variantes funcionales nuevas respecto de la base global (conjunta, sólo entrada y sólo salida), y seis recetas del piloto incluyendo controles de mecanismos existentes. No se inventa un nuevo Nd global sumando semillas o aliases; los controles reentrenados no sustituyen las cifras históricas de 796/30/20. La elección completa debe fijarse antes del piloto. Comparar sólo entrada y salida globales sin los dos controles parciales no aislaría la contribución de cada transformación.

## Propiedades formales y demostración

### P1. Soporte y lectura exacta bajo la ley finita

Si A>=0, a_k>=0, p_k>=0 y sum p_k=1, cada z_k>=sqrt(A), por tanto mu>=sqrt(A). Además mu es exactamente E_G sqrt(A+cU^2) para G=sum p_k delta_(a_k). No se concluye que G sea la verdadera ley condicional ni que esté calibrada.

### P2. Equivarianza local de amplitud con referencias de ajuste fijas

Aplicar a las siete filas de precios de una ventana la transformación

    log(Q'_j)=b+k*log(Q_j), k>0,

sin cambiar V, pesos, gamma o r_fit. Supóngase que R_t>epsilon*r_fit y k*R_t>epsilon*r_fit; es decir, no actúa el piso ni antes ni después. Entonces r'_i=k*r_i, m'=k*m, A'=k^2*A y s'_t=k*s_t. Para la candidata, Z'=(k*(log Q-log C_t))/(k*s_t)=Z. El encoder y las masas e incrementos son iguales. Los átomos satisfacen a'_k=k*a_k y

    z'_k = sqrt(k^2*A+c*k^2*a_k^2)=k*z_k,
    mu'=k*mu.

La constante b desaparece exactamente de la entrada relativa y no se añade al objetivo de volatilidad. Las masas son invariantes y la ley latente se transforma por push-forward u->k*u. La lectura no calcula sqrt(E U^2) en lugar de E sqrt(...).

Un modelo `global_mse` con r_fit fijo no tiene esta garantía local: sus entradas cambian por k y sus magnitudes de salida no reciben un factor k por construcción. Los pesos podrían aprenderla aproximadamente, pero no se la impone.

### P3. Caso plano y transformación conjunta de referencias

Con R_t=0, la escala positiva proviene del piso de ajuste. No es correcto afirmar equivarianza local exacta manteniendo r_fit fijo: el piso no se escala. Si también se transforma la referencia r'_fit=k*r_fit, entonces s'_t=k*s_t para todas las ventanas, incluso planas y de transición. Si las escalas de entrenamiento de la pérdida también se cotransforman, la pérdida normalizada es invariante.

Esta propiedad conjunta NO diferencia a la candidata de todos los controles, porque un control global también puede ser equivariante bajo cotransformación de sus referencias. La propiedad que distingue la candidata es P2, con referencias fijas y piso no activo. Se debe informar la fracción de ventanas donde el piso actúa, sin convertir su ausencia en criterio escogido después del resultado.

Las pruebas numéricas deben verificar P2 en ventanas sintéticas no degeneradas, P3 en planas, invariancia al nivel de precio, soporte, serialización y que cambiar un dato futuro no cambia un pronóstico anterior. Son pruebas del algoritmo, no entrenamiento ni evidencia de superioridad.

## Antecedentes primarios comprobados y diferencia acotada

1. Kim et al. (ICLR 2022), *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*. PDF primario de 25 páginas abierto por la referencia de OpenReview: https://openreview.net/references/pdf?id=SRgrKlAhbc . Lectura focal §3.1, ecuaciones (1)–(3), páginas impresas 4–5, y apéndice A.8. Se elimina/restaura estadística por instancia en predicción de las mismas variables; esa idea no es nueva aquí. RevIN oficial exige correspondencia de dimensiones para su normalización inversa. La adaptación V5 cambia el objeto: log-OHLCV -> distribución de magnitud futura -> volatilidad solapada; la salida no debe recibir el desplazamiento del precio ni una inversión variable-a-variable estándar. Se implementa la reconstrucción de unidades de U antes de la lectura con A. Fuente principal: https://openreview.net/forum?id=cGDAkQo1C0p (interfaz bloqueada por challenge en esta consulta); página oficial https://seharanul17.github.io/RevIN/ confirmó año 2022 y autores. Repositorio oficial MIT https://github.com/ts-kim/RevIN consultado; no se copia código de RevIN.

2. Passalis et al. (2019 preprint, publicación TNNLS 2020), *Deep Adaptive Input Normalization for Time Series Forecasting*, arXiv:1902.07892v2. PDF primario de 7 páginas abierto: https://arxiv.org/pdf/1902.07892 . Lectura focal introducción y §II: capas aprendidas de desplazamiento, escala y gating para series financieras. Impide reclamar que normalizar dinámicamente finanzas sea novedoso. V5 fija una escala causal analítica de retornos y modifica reconstrucción de una variable distinta; no atribuye como invención las capas adaptativas.

3. Liu, Wu, Wang y Long, *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting*, arXiv:2205.14415v4 (NeurIPS 2022). Texto primario HTML abierto: https://arxiv.org/html/2205.14415v4 . Lectura focal §3.1–3.2: normalización/reconstrucción y pérdida posible de información no estacionaria al normalizar. Es una amenaza concreta para V5: imponer invariancia puede borrar información útil del régimen. V5 no contiene De-stationary Attention, no importa el código del Transformer y no afirma resolver ese problema por demostrar P2.

4. Stirn et al. (AISTATS 2023), *Faithful Heteroscedastic Regression with Neural Networks*, PMLR206:5593–5613. Fuente primaria y resumen confirmados: https://proceedings.mlr.press/v206/stirn23a.html . No se implementa en la candidata elegida. Se documenta para justificar por qué preservar exactamente una trayectoria puntual no es una hipótesis de ganancia puntual.

5. V3 local `src/e3_overlap_model_v3.py` ya contiene la ley finita, CRPS discreto, incrementos ordenados y T_A antes de la esperanza. Es antecedente interno explícito; esos componentes no se reclaman nuevos en V5.

6. Escalas condicionales en GARCH y modelos de innovación normalizada son antecedentes históricos directos. La proposición de homogeneidad por sí sola no es una invención. Una revisión focal posterior puede ampliar esta fila con la formulación primaria exacta sin imputar aquí una lectura de texto completo que no se hizo.

**Declaración defendible:** implementación propia de una adaptación de normalización reversible, con coordenadas logarítmicas causales y reconstrucción analítica de una ley de innovaciones para una etiqueta de volatilidad solapada. El mecanismo, las ablaciones y el teorema son revisables. La consulta focal no demuestra prioridad mundial ni concede automáticamente R1. La propiedad no demuestra menor RMSE, estabilidad temporal, calibración o R3.

## Hipótesis y límites de decisión

Hipótesis primaria falsable: el mecanismo conjunto mejora la RMSE media por semilla frente al mejor control independiente predeterminado por al menos el margen interno fijado antes del piloto (por ejemplo 2%), y la mejora es consistente en al menos cuatro de cinco semillas. El margen de 2% pertenece al nuevo piloto; no es el margen universal de la guía para R3.

Los controles de entrada/salida se usan para atribución y evitar elegir a posteriori otro objeto. Si sólo gana una ablación, no se cambia de candidata automáticamente: se conserva el resultado y se diseña otra decisión futura con identidad propia. Si la variante conjunta no satisface su regla, se abandona esta receta. Los resultados no se mezclan con los calendarios o GPUs de V3/V4.

El piloto utiliza sólo desarrollo expuesto y no puede producir confirmación independiente. Una señal favorable justificaría preparar continuidad: R1 propia, datos confirmatorios adecuados, batería válida, potencia y presupuesto de toda la familia. No convierte el costo de preparación en una acreditación de R3.


## Correspondencia final con el paquete implementado

Objeto: `/workspace/e3_seoi_v5_pilot_20261121/seoi_v5/model.py` y `data.py`. El MSE global, CRPS global y la lectura aditiva son controles reimplementados desde las fuentes internas y ejecutados de nuevo cuando se corra el piloto; no se reutilizan pesos ni se imputan resultados históricos. La candidata y sus dos ablaciones comparten exactamente parámetros/capacidad iniciales, escalas físicas de ajuste, familia y presupuesto. La representación global se comprobó contra el código anterior y no se interpreta como raw-OHLC sin transformar. El protocolo fija MAE <=1.05 veces la MAE del control neuronal de menor RMSE, no el control de menor MAE.
