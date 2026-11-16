# Identidad del objetivo móvil y alcance de una proyección de coherencia

Fecha de esta revisión: 16 de noviembre de 2026. Este documento se redacta después del primer ensayo de desarrollo de E3. Conserva una identidad matemática del objetivo heredado y una propiedad de una proyección estándar; no define un nuevo modelo original, no acredita R1 y no convierte un análisis posterior en confirmación. No se modifican aquí pesos, predicciones, presupuestos o resultados originales del ensayo.

## 1. Definición verificada y disponibilidad de la información

La definición se conserva en el [contrato E2](https://github.com/jassan-bit/DeepLearning_202630/blob/0da14d2fb5f0a3f97754610d2d3b3c20c1aa14fb/book_e2/sections/17_contrato_sincronizado.md). La [admisión de desarrollo](https://github.com/jassan-bit/DeepLearning_202630/blob/0da14d2fb5f0a3f97754610d2d3b3c20c1aa14fb/scripts/e2_development_data.py) verifica las etiquetas contra `diff(log(close)).rolling(7).std(ddof=0)`. La [reconstrucción del panel](https://github.com/jassan-bit/DeepLearning_202630/blob/0da14d2fb5f0a3f97754610d2d3b3c20c1aa14fb/scripts/e2_reduced_final_data.py) y la [auditoría independiente de ocho cierres](https://github.com/jassan-bit/DeepLearning_202630/blob/0da14d2fb5f0a3f97754610d2d3b3c20c1aa14fb/scripts/e2_current_data_closure.py) preservan esa convención.

Sean $C_s>0$ los cierres diarios UTC y

$$
r_s=\log C_s-\log C_{s-1},\qquad
\bar r_t=\frac17\sum_{j=0}^{6}r_{t-j},\qquad
\sigma_7(t)=\sqrt{\frac17\sum_{j=0}^{6}(r_{t-j}-\bar r_t)^2}.
$$

Es desviación poblacional, `ddof=0`, sin anualización. Se pronostica al comienzo del día $t$, usando únicamente las siete barras de $t-7$ a $t-1$. La etiqueta se completa al terminar $t$; el horizonte sigue siendo un día.

Los siete cierres de entrada $C_{t-7},\ldots,C_{t-1}$ determinan seis retornos:

$$
u_i=\log C_{t-7+i}-\log C_{t-8+i}=r_{t-7+i},\qquad i=1,\ldots,6.
$$

Así, $u_1=r_{t-6}$ y $u_6=r_{t-1}$. El único retorno no observado del objetivo es $x=r_t=\log C_t-\log C_{t-1}$. Siete barras no contienen siete retornos observados: haría falta un cierre adicional. El piso que se deriva abajo usa exclusivamente los seis retornos conocidos y no requiere $C_t$.

La transformación de precios `log(P)-log(C[t-1])` no elimina estos seis retornos: al restar dos cierres relativos consecutivos se cancela el ancla. La estandarización congelada es invertible cuando su escala es positiva. Para una comprobación numérica debe conservarse la precisión y procedencia de la ventana admitida, sin estimar nuevos escaladores con validación.

## 2. Descomposición exacta de la desviación de siete retornos

Defínanse cantidades conocidas al emitir el pronóstico:

$$
m_6=\frac16\sum_{i=1}^{6}u_i,\qquad
S_6=\sum_{i=1}^{6}(u_i-m_6)^2,\qquad A=\frac{S_6}{7}.
$$

La media de los siete retornos es $\bar r_t=(6m_6+x)/7$. Por tanto,

$$
m_6-\bar r_t=-\frac{x-m_6}{7},\qquad
x-\bar r_t=\frac67(x-m_6).
$$

Como $\sum_{i=1}^{6}(u_i-m_6)=0$, los términos cruzados se cancelan y

$$
\begin{aligned}
\sum_{i=1}^{6}(u_i-\bar r_t)^2+(x-\bar r_t)^2
&=S_6+6\frac{(x-m_6)^2}{49}+\frac{36(x-m_6)^2}{49}\\
&=S_6+\frac67(x-m_6)^2.
\end{aligned}
$$

Al dividir entre siete se obtiene la identidad

$$
\boxed{\sigma_7(t)^2=A+\frac6{49}(r_t-m_6)^2.}
$$

No es una aproximación que suponga retorno medio cero o normalidad. Es válida para cualquier colección finita de siete retornos reales, bajo la definición indicada. Tampoco supone que el siguiente retorno sea igual a la media conocida.

El piso algebraico es

$$
\boxed{\ell_t=\sqrt A\le\sigma_7(t).}
$$

Si $\sigma_6=\sqrt{S_6/6}$ usa `ddof=0`, entonces $\ell_t=\sqrt{6/7}\,\sigma_6$. No debe sustituirse el piso por $\sigma_6$ sin ese factor. La igualdad $\sigma_7(t)=\ell_t$ ocurre si $r_t=m_6$. Sin restringir el próximo retorno, el intervalo factible es exactamente $[\ell_t,\infty)$: cualquier $y\ge\ell_t$ corresponde a

$$
r_t=m_6\mathbin{\pm}\sqrt{\frac{49}{6}(y^2-A)}.
$$

Esta identidad es un caso de la actualización conocida de suma de cuadrados al agregar una observación. [Chan, Golub y LeVeque (1983)](https://doi.org/10.1080/00031305.1983.10483115), p.242, ecuaciones 1.3a–b, presentan esa actualización; con siete observaciones y denominador poblacional siete se obtiene el coeficiente $6/49$. No se reclama una nueva identidad matemática.

## 3. Proyección y errores sobre la escala de sigma

Para una predicción finita $a_t$ en las mismas unidades originales de $\sigma_7$, considérese

$$
\Pi_t(a_t)=\max(a_t,\ell_t).
$$

Es la proyección euclídea escalar sobre el intervalo convexo $[\ell_t,\infty)$. Se estudia aquí como comprobación de coherencia, posterior al ensayo; no se atribuye a una arquitectura propia.

**Proposición.** Si la etiqueta $y_t$ es exactamente la desviación definida y está correctamente alineada, para cada fecha se cumplen

$$
|\Pi_t(a_t)-y_t|\le|a_t-y_t|,\qquad
(\Pi_t(a_t)-y_t)^2\le(a_t-y_t)^2.
$$

**Demostración.** Si $a_t\ge\ell_t$, la predicción no cambia. Si $a_t<\ell_t$, la identidad anterior garantiza $y_t\ge\ell_t>a_t$. En consecuencia,

$$
|a_t-y_t|-|\ell_t-y_t|=\ell_t-a_t>0,
$$

$$
(a_t-y_t)^2-(\ell_t-y_t)^2
=(\ell_t-a_t)(2y_t-a_t-\ell_t)\ge0.
$$

Esto prueba ambas desigualdades. Al promediar sobre las mismas fechas y con los mismos pesos no negativos, no empeoran MAE ni MSE. Como la raíz cuadrada es creciente, tampoco empeora RMSE. La afirmación se refiere al error de **$\sigma_7$**, no a un contraste estadístico de superioridad entre algoritmos. $\square$

La mejora es estricta en las fechas proyectadas bajo esos supuestos. Si ninguna predicción viola el piso, la proyección deja todos los errores iguales. No se deriva un margen de mejora positivo uniforme ni se garantiza un orden entre dos modelos después de proyectar ambos.

La escala del objetivo del ensayo es $y/a_{\rm fit}$, donde $a_{\rm fit}>0$ es la media del objetivo del ajuste interno. En esa escala, el piso correspondiente es $\ell_t/a_{\rm fit}$. Aplicar un piso en unidades originales a una salida todavía escalada sería incorrecto; la salida debe desescalarse una sola vez.

## 4. Estimando condicional: media de sigma y raíz de su segundo momento

Sea $\mathcal F_{t-1}$ la información disponible al emitir el pronóstico. $A$ y $m_6$ son medibles respecto de ella. Si $r_t$ tiene segundo momento condicional finito, escribiendo $\mu_t=E[r_t\mid\mathcal F_{t-1}]$ y $v_t=\operatorname{Var}(r_t\mid\mathcal F_{t-1})$ se obtiene

$$
E[\sigma_7(t)^2\mid\mathcal F_{t-1}]
=A+\frac6{49}\left\{v_t+(\mu_t-m_6)^2\right\}.
$$

No basta con introducir una varianza condicional y omitir el término de diferencia de medias. Además, el óptimo condicional para pérdida cuadrática sobre $\sigma_7$ es

$$
a_t^*=E[\sigma_7(t)\mid\mathcal F_{t-1}]
=E\left[\sqrt{A+\frac6{49}(r_t-m_6)^2}\,\middle|\,\mathcal F_{t-1}\right].
$$

La igualdad se deduce de la descomposición usual del riesgo cuadrático en varianza condicional más $(a_t-E[\sigma_7\mid\mathcal F_{t-1}])^2$. En cambio, $E[\sigma_7^2\mid\mathcal F_{t-1}]$ es el óptimo para pérdida cuadrática sobre la **varianza móvil**, un objetivo diferente.

Por concavidad de la raíz,

$$
E[\sigma_7\mid\mathcal F_{t-1}]
\le\sqrt{E[\sigma_7^2\mid\mathcal F_{t-1}]}.
$$

La desigualdad es estricta cuando $\sigma_7^2$ no es condicionalmente constante. Por ello, reemplazar sigma por la raíz de su segundo momento no produce en general el predictor óptimo para RMSE sobre sigma. Para pérdida absoluta, los óptimos son las medianas condicionales de sigma; tampoco se justifican automáticamente por una media condicional de varianza. Estas distinciones no exigen fijar una distribución normal para los retornos.

## 5. Supuestos y alcance de cualquier comprobación posterior

- **Calendario y causalidad:** cierres finitos y positivos, barras diarias UTC consecutivas y ordenadas, sin comprimir huecos ni desplazar etiquetas. El piso sólo puede usar $C_{t-7},\ldots,C_{t-1}$, nunca el cierre objetivo $C_t$.
- **Definición:** siete retornos logarítmicos, centrado por su media de siete y denominador siete. Un objetivo de retornos cuadrados sin centrar, una desviación muestral o una serie anualizada exige otra fórmula o conversión explícita.
- **Etiquetas y aritmética:** la garantía es exacta en aritmética real. El código heredado comprueba las etiquetas con tolerancias; redondeo, almacenamiento, datos corregidos o una transformación mal invertida deben comprobarse antes de atribuir la garantía a un archivo concreto. Una tolerancia no convierte por sí misma la igualdad numérica en exactitud matemática.
- **Comparación:** conservar predicciones y métricas originales, identificar por separado cualquier proyección posterior y usar la misma cohorte. No reasignar sus resultados al modelo originalmente entrenado ni presentarlos como un experimento pre-registrado.
- **Independencia:** la propiedad punto a punto no vuelve independientes las ventanas solapadas ni convierte la validación usada para parada en prueba confirmatoria. Tampoco corrige por sí sola sesgos de selección, presupuestos o pruebas múltiples.

## 6. Antecedentes originales y lo que no permiten afirmar

Las siguientes son lecturas focales de fuentes originales; no se presentan como lectura íntegra de todos sus artículos ni como revisión sistemática de novedad terminada.

| Fuente y localización de lectura | Relación comprobada | Diferencia y límite |
|---|---|---|
| Tony F. Chan, Gene H. Golub y Randall J. LeVeque (1983), *Algorithms for Computing the Sample Variance: Analysis and Recommendations*, *The American Statistician* 37(3), 242–247. [DOI editorial](https://doi.org/10.1080/00031305.1983.10483115); [scan del original](https://math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf), p.242, ecuaciones 1.3a–b, texto y comprobación visual. | Actualización de media y suma de cuadrados al incorporar una observación; implica la descomposición mostrada. | Es álgebra estadística conocida; aplicarla con seis más un retorno no acredita una nueva propiedad de una arquitectura. |
| Mathis Brosowsky, Florian Keck, Olaf Dünkel y Marius Zöllner (2021), *Sample-Specific Output Constraints for Neural Networks*, AAAI 35(8), 6812–6819. [PDF editorial original](https://cdn.aaai.org/ojs/16841/16841-13-20335-1-2-20210518.pdf), pp.6812–6816, ecuaciones 4–5 y ejemplo de umbral escalar en p.6816. | ConstraintNet parametriza dominios de salida específicos de cada muestra. El ejemplo escalar usa $\exp(z)+b$ para imponer un umbral inferior $b$. | No estudia este objetivo financiero, pero anticipa la idea de una capa que suma un piso a una salida positiva. Cambiar esa positividad por Softplus no basta para sostener R1. |
| Cristi Spulbar y Cezar Cătălin Ene (2026), *Residual Deep Learning for Realized Volatility Using a Hybrid HAR-LSTM Model with ARIMA Features and SHAP Interpretability*, *Mathematics* 14(20), 3661. [Original editorial](https://www.mdpi.com/2227-7390/14/20/3661), [DOI](https://doi.org/10.3390/math14203661), secciones 3.1 y 4.3. | Distingue un objetivo móvil de cinco retornos cuadrados con cuatro retornos ya observados y un GARCH alineado que incorpora esa parte conocida; contrasta con un horizonte sin ese solapamiento. | Su objetivo no es nuestra desviación centrada de siete retornos. Es un antecedente directo de componente conocido más componente futuro, no la fuente de una afirmación de novedad para el piso/proyección de sigma. |
| Mateusz Buczynski y Marcin Chlebus, *GARCHNet: Value-at-Risk Forecasting with GARCH Models Based on Neural Networks*, publicación online 22/05/2023, *Computational Economics* 63, 1949–1979 (2024). [Original editorial](https://link.springer.com/article/10.1007/s10614-023-10390-7), [DOI](https://doi.org/10.1007/s10614-023-10390-7), sección 2.3, ecuaciones 19–23. | Combina LSTM y distribución condicional de retornos; utiliza activaciones para imponer positividad y restricciones de parámetros. | No deriva este piso ni pronostica la misma desviación móvil. Restringir una salida positiva o modelar innovación condicional son antecedentes, no novedades demostradas aquí. |

La combinación de una identidad conocida con una proyección o parametrización conocida puede ser útil como referencia analítica y control de coherencia. Su utilidad no equivale a una modificación algorítmica sustantiva acreditada. No se afirma que estos cuatro trabajos agoten los antecedentes ni que la ausencia de un artículo idéntico pruebe novedad. El candidato previo conserva sus propios resultados y su procedencia asistida por IA; este documento no le atribuye retrospectivamente otra justificación ni acredita R1, la defensa académica de R2 o R3.
