# Cabeza de media delimitada por dos momentos: formulación y alcance

Fecha: 16 de noviembre de 2026. Es una propuesta matemática de desarrollo dentro de la misma **CNN causal → LSTM unidireccional → atención temporal por contenido**. No se entrenaron modelos para este documento. Las desigualdades y los scores utilizados tienen antecedentes; su especialización no se declara nueva ni acredita R1/R3.

## 1. Variables, unidades y dominio factible

Al emitir el pronóstico se conocen los seis retornos $r_{t-6},\ldots,r_{t-1}$ derivados de las siete barras anteriores. Sean $\bar r_6$ su media y
$$
A=\frac17\sum_{i=t-6}^{t-1}(r_i-\bar r_6)^2\ge0,\qquad c=\frac6{49},\qquad
U=|r_t-\bar r_6|,\qquad V=U^2\ge0.
$$
El objetivo sigue siendo $Y=\sigma_7(t)=\sqrt{A+cV}$, desviación poblacional de siete retornos logarítmicos, con horizonte de un día y sin anualización. $A$ se calcula sólo con historia disponible. $U,V$ y $V^2$ son etiquetas transformadas, no predictores observados al origen.

En lo sucesivo se condiciona en la información $X$ disponible al pronosticar, por lo que $A$ es fijo. Se supone $A<\infty$ y se escriben los momentos verdaderos
$$
m_0=E[V\mid X]>0,\qquad q_0=E[V^2\mid X]<\infty,\qquad
\mu_0=E[Y\mid X].
$$
Para candidatos de momentos se exige $m>0$, $m^2\le q<\infty$. Si $m=0$ fuese el momento verdadero, $V=0$ casi seguramente y necesariamente $q=0$; la pareja $m=0,q>0$ es imposible. La rama degenerada $m=q=0$ se separa de la pérdida logarítmica de la sección 5.

## 2. Límites de la media para dos momentos dados

Para cualquier ley de $V\ge0$ con $E[V]=m>0$ y $E[V^2]=q<\infty$, defínanse
$$
H(A,m)=\sqrt{A+cm},
$$
$$
L(A,m,q)=\sqrt A+\frac{m^2}{q}\left(\sqrt{A+\frac{cq}{m}}-\sqrt A\right)
=\sqrt A+\frac{cm}{\sqrt{A+cq/m}+\sqrt A}.
$$
**Teorema.** Se cumple $L\le E[\sqrt{A+cV}]\le H$. La cota inferior se alcanza. Si $q>m^2$, la superior es un supremo sharp que no se alcanza; si $q=m^2$, ambas coinciden y se alcanzan.

**Prueba de la cota superior.** La raíz es estrictamente cóncava, y Jensen da
$$
E[\sqrt{A+cV}]\le\sqrt{A+cE[V]}=H.
$$
Para $c>0$, la igualdad exige que $V$ sea constante casi seguramente; equivale a $q=m^2$. Por ello no puede declararse un máximo alcanzado en $H$ cuando $q>m^2$.

**Prueba de la cota inferior.** Considérese la ley sesgada por tamaño $dP^*(v)=(v/m)\,dP(v)$. Es una probabilidad y $E^*[V]=q/m$. Para
$$
h(v)=\frac1{\sqrt{A+cv}+\sqrt A},
$$
la identidad racionalizada da
$$
E[\sqrt{A+cV}]=\sqrt A+cm\,E^*[h(V)].
$$
Escribiendo $a=\sqrt A$ y $t=\sqrt{A+cv}$, para $t>0$ se tiene
$$
h''(v)=\frac{c^2(3t+a)}{4t^3(t+a)^3}>0.
$$
Así, Jensen aplicado bajo $P^*$ proporciona
$$
E^*[h(V)]\ge h(E^*[V])=\frac1{\sqrt{A+cq/m}+\sqrt A},
$$
lo que prueba $L$. Si $A=0$, $h(0)$ es infinito, pero $P^*(V=0)=0$; se aplica la desigualdad en $v>0$. La identidad sigue siendo válida, pues el término original correspondiente a $V=0$ es cero y la esperanza de la raíz es finita.

**Sharpness inferior.** La ley
$$
P(V=q/m)=m^2/q,\qquad P(V=0)=1-m^2/q
$$
tiene los momentos exigidos. Bajo su ley sesgada, $V=q/m$ casi seguramente, de modo que alcanza $L$. No se postula que los datos financieros sigan esa ley; es una distribución extremal del problema de momentos.

**Sharpness superior.** Sea $\Delta=q-m^2>0$. Para $R\ge q/m$ se define
$$
a_R=m-\frac{\Delta}{R-m}\ge0,\qquad
p_R=\frac{\Delta}{(R-m)^2+\Delta}.
$$
La ley con masa $p_R$ en $R$ y masa $1-p_R$ en $a_R$ conserva exactamente $m,q$. Cuando $R\to\infty$, $a_R\to m$, $p_R\to0$ y
$$
(1-p_R)\sqrt{A+ca_R}+p_R\sqrt{A+cR}\longrightarrow\sqrt{A+cm}.
$$
El segundo término tiende a cero porque $p_R=O(R^{-2})$ y la raíz crece como $O(R^{1/2})$. Esto prueba que $H$ es el supremo, sin contradecir su no alcanzabilidad. $\square$

El rango realizable de la media, cuando $q>m^2$, es exactamente **$[L,H)$**. Para cualquier punto interior se mezcla la ley extremal inferior con una ley de la secuencia anterior cuya media lo supere; la mezcla conserva ambos momentos e interpola linealmente la media de $Y$.

## 3. Bordes e interpretación de las cotas

- Si $A=0$, $L=\sqrt c\,m^{3/2}/\sqrt q$ y $H=\sqrt{cm}$. Son límites de momentos conocidos, no una nueva desigualdad para volatilidad.
- Si $q=m^2>0$, $V=m$ casi seguramente y $L=H=\sqrt{A+cm}$. Un parámetro interpolador de la media deja de ser identificable.
- Si $m=q=0$, la media es $\sqrt A$. No se evalúa la fórmula $q/m$ ni los logaritmos de cero.
- Las cotas se refieren a **la media condicional**, no al soporte de cada observación. $Y$ puede quedar fuera de $[L,H]$, y $H$ no es un límite superior de las realizaciones ni un intervalo de predicción.

Con momentos estimados, el intervalo calculado es el correspondiente a esos valores de salida. No se garantiza que contenga la media verdadera si los momentos están mal estimados. Tampoco se obtiene una garantía punto a punto de reducir MSE al proyectar en ese intervalo; esa propiedad no debe confundirse con la proyección sobre el piso algebraico observable $\sqrt A$.

## 4. Parametrización propuesta en el mismo backbone

Sean $a,b,d$ tres lecturas escalares del contexto de la CNN–LSTM–atención y $s_u>0$ una escala fija estimada únicamente en ajuste. Se propone
$$
m=s_u^2\operatorname{softplus}(a)>0,\qquad
q=m^2(1+b^2)\ge m^2,\qquad
\rho=\frac{d^2}{1+d^2}\in[0,1),
$$
$$
\widehat\mu=L(A,m,q)+\rho\,[H(A,m)-L(A,m,q)].
$$
Para $q>m^2$, la media producida es realizable por alguna ley con esos momentos. Para $q=m^2$ se produce la única media posible, con independencia de $d$. La cabeza no especifica una distribución única, un signo de retorno o cuantiles calibrados.

Las transformaciones cubren todos los momentos positivos factibles y todas sus medias realizables: dado $m,q$, puede tomarse $b=\pm\sqrt{q/m^2-1}$; para $\rho\in[0,1)$, $d=\pm\sqrt{\rho/(1-\rho)}$. Los signos de $b,d$ no son identificables. Permitir $\rho=0$ es necesario para representar las leyes que alcanzan $L$; una sigmoid estricta no lo permitiría. Excluir $\rho=1$ evita atribuir realizabilidad al supremo no alcanzado.

Esto es una afirmación sobre el operador, no sobre capacidad universal de una red finita. No prueba Lipschitz global del predictor, estabilidad de todos sus gradientes o convergencia de entrenamiento. Hay puntos planos: $\partial q/\partial b=2m^2b$ y $\partial\rho/\partial d=2d/(1+d^2)^2$ se anulan en cero. Si $b=0$, además $H=L$ y la rama de $d$ no recibe señal.

Se declara como **inicialización propuesta, sin ejecución de mercado**, $b=\sqrt8$ y $d=1$. Así $q/m^2=9$ y $\rho=1/2$. El valor 9 es un ancla matemática de la familia t5 utilizada en el diseño anterior: $E[T_5^2]=5/3$, $E[T_5^4]=25$ y $E[T_5^4]/E[T_5^2]^2=9$. No es un valor seleccionado por ajuste o búsqueda sobre resultados de mercado, ni impone que la nueva cabeza mantenga esa familia. Evitar esos dos ceros iniciales no garantiza convergencia, buena estimación de momentos o superioridad.

Las afirmaciones de realizabilidad son de aritmética real. Con precisión finita, un $d$ grande puede redondear $\rho$ a uno y la predicción a $H$; el verificador registra esa limitación. No se certifica automáticamente la desigualdad estricta en todos los números representables ni se encubre el redondeo como una nueva ley factible.

## 5. Scores de momentos y consistencia condicional

Con escalas de ajuste $s_u>0$, $s_y>0$ congeladas, sean $\widetilde V=V/s_u^2$, $\widetilde m=m/s_u^2$ y $\widetilde q=q/s_u^4$. Para pesos fijos $\lambda_m,\lambda_q>0$ se estudia
$$
\mathcal L=\left(\frac{Y-\widehat\mu}{s_y}\right)^2
+\lambda_m\left(\log\widetilde m+\frac{\widetilde V}{\widetilde m}\right)
+\lambda_q\left(\log\widetilde q+\frac{\widetilde V^2}{\widetilde q}\right).
$$
Son scores de forma quasi-Gamma para medias positivas. No se afirma que $V$ o $V^2$ tengan distribución Gamma ni que sumar ambos scores sea la log-verosimilitud de una distribución conjunta.

**Teorema de consistencia condicional del criterio.** Fijado $X$, supónganse $m_0>0$, $q_0<\infty$ y una clase capaz de representar conjuntamente $m_0,q_0,\mu_0$ mediante la cabeza anterior. Entonces el riesgo condicional se minimiza en esos tres valores, aunque los parámetros $b,d$ puedan no ser únicos.

**Demostración.** La esperanza de cada score es $\log m+m_0/m$ o $\log q+q_0/q$, más constantes de unidades. La pérdida cuadrática satisface la descomposición en varianza condicional más error de la media. El exceso de riesgo respecto de los valores verdaderos es exactamente
$$
\frac{(\widehat\mu-\mu_0)^2}{s_y^2}
+\lambda_m\left[\log\frac m{m_0}+\frac{m_0}{m}-1\right]
+\lambda_q\left[\log\frac q{q_0}+\frac{q_0}{q}-1\right]\ge0.
$$
Para $z>0$, $\log z+z^{-1}-1\ge0$, con igualdad sólo en $z=1$. Por tanto la igualdad total requiere $m=m_0$, $q=q_0$ y $\widehat\mu=\mu_0$. Los momentos verdaderos son factibles y la sección 2 garantiza que su media tiene un $\rho$ representable, incluida la igualdad inferior. Esto prueba la afirmación. Si $q_0=m_0^2$, $L=H$ y $\rho$ es irrelevante. $\square$

La condición de clase rica es una hipótesis, no una propiedad demostrada de este backbone concreto. Integrar el resultado sobre $X$ requiere las condiciones de integrabilidad correspondientes; se formula primero como exceso de riesgo condicional para no restar dos esperanzas infinitas.

Para que exista la esperanza del score de $q$ basta $E[V^2\mid X]=E[U^4\mid X]<\infty$; no se exige $E[U^8]$ para esa elicitación. Eso no asegura varianza finita del gradiente. De hecho,
$$
\frac{\partial(\log m+V/m)}{\partial\log m}=1-V/m,\qquad
\frac{\partial(\log q+V^2/q)}{\partial\log q}=1-V^2/q,
$$
son scores no acotados por abajo. La varianza del segundo puede requerir $E[V^4]=E[U^8]$. No se deducen robustez a extremos, normalidad asintótica o garantías para SGD sólo de la finitud de cuarto momento de $U$.

Si $m_0=0$, entonces $V=0$ y $q_0=0$. En ese caso, llevando $m\to0$ con $b$ fijo, la parte logarítmica tiende a $-\infty$ mientras la media converge a $\sqrt A$. No existe el mínimo finito invocado arriba. Debe excluirse ese caso por un supuesto explícito o diseñarse una rama degenerada separada; un piso arbitrario de parámetros modificaría el dominio y la afirmación de consistencia.

## 6. Contraejemplo para el criterio compartido half-t5 de V2

Se muestra un conflicto posible entre objetivos bajo mala especificación, no una explicación demostrada de los resultados de BTC. Tómese $A=0$ y la magnitud verdadera
$$
P(U=0)=\frac34,\qquad P(U=2)=\frac14.
$$
Entonces $m_0=E[U^2]=1$, $q_0=E[U^4]=4$ y $\mu_0=E[Y]=\sqrt c/2$. La cota inferior se alcanza y puede representarse con $\rho=0$.

Para una half-t5 con escala compartida $s$, sea $k_5=E|T_5|=4\sqrt5/(3\pi)$. La media del modelo continuo es $g(0,s)=\sqrt c\,k_5s$, así que el óptimo del MSE sobre sigma exige
$$
s_{\rm MSE}=\frac1{2k_5}=\frac{3\pi}{8\sqrt5},\qquad s_{\rm MSE}^2=\frac{9\pi^2}{320}.
$$
La esperanza de la fórmula auxiliar half-t NLL, omitiendo constantes independientes de $s$, es
$$
R_{\rm aux}(s)=\log s+\frac34\log\left(1+\frac4{5s^2}\right),\qquad
\frac{dR_{\rm aux}}{d\log s}=1-\frac6{5s^2+4}.
$$
Su mínimo exige $s_{\rm aux}^2=2/5$. En $s_{\rm MSE}$ la derivada auxiliar es negativa, mientras la derivada de MSE se anula. Para cualquier peso auxiliar positivo, ese $s_{\rm MSE}$ deja de ser estacionario del objetivo mixto; el mínimo se desplaza entre $s_{\rm MSE}$ y $\sqrt{2/5}$ y la media correcta se altera. Las normalizaciones fijas de unidades no eliminan la discrepancia.

Como la ley del ejemplo es atómica y la half-t es continua, aquí se evalúa el riesgo de la **fórmula auxiliar** usada por el algoritmo; no se presenta una divergencia KL finita entre ambas leyes. El fenómeno no depende de los átomos: sustituyendo las masas por uniformes en $[0,\varepsilon]$ y $[2-\varepsilon,2+\varepsilon]$, $0<\varepsilon<1$, las medias y la derivada auxiliar en el óptimo puntual convergen a las anteriores. Esa derivada es una función acotada y continua para escalas alejadas de cero, por lo que su signo negativo persiste para un suavizado suficientemente pequeño.

Para la aproximación finita real de V2, $g_{128}(0,s)=\sqrt c\,k_{128}s$, donde $k_{128}=\sum_i a_i z_i$. Su óptimo puntual sería $1/(2k_{128})$. El criterio numérico tampoco adquiere consistencia exacta por aproximar $k_5$; el argumento continuo basta para refutar una garantía universal y no sustituye una comprobación numérica del operador concreto.

El contraejemplo no contradice que, con una familia distributiva correctamente especificada y su media exacta, MSE y NLL puedan compartir mínimo. No prueba que la half-t sea incorrecta en BTC ni que otra cabeza supere a TimeMixer o a la familia de E2. La pérdida de momentos ofrece una afirmación distinta bajo sus propios supuestos, no una garantía de desempeño finito.

## 7. Antecedentes y alcance de la adaptación

La cota inferior es una especialización de Jensen ponderada, documentada en [Neri Merhav (2023), *Some Families of Jensen-like Inequalities with Application to Information Theory*](https://pmc.ncbi.nlm.nih.gov/articles/PMC10288939/), sección 2, ecuación 8; el caso $A=0$ se relaciona con su ejemplo 2, ecuación 15. No se atribuye originalidad a esa desigualdad, a size-biasing, a Jensen o a la construcción extremal de momentos.

Los scores consistentes para la media y su relación con divergencias de Bregman tienen antecedentes en [Tilmann Gneiting (2011), *Making and Evaluating Point Forecasts*](https://arxiv.org/html/0912.0902v2), sección 3.1, teorema 3.1 y ecuaciones 18 y 20. Aquí se prueba directamente la versión del criterio definida para observaciones no negativas y parámetros estrictamente positivos; no se necesita tomar logaritmos de las observaciones cero.

La adaptación a investigar es usar conjuntamente dos momentos de la innovación recuperable y una media realizable de sigma7 en la cabeza del backbone preservado. La autoría asistida por IA, los antecedentes, las diferencias de mecanismo y la revisión de novedad deben mantenerse explícitos. Estos teoremas no son nuevos por especializarlos al objetivo; no acreditan novedad, comprensión académica defendida, rendimiento de mercado ni superioridad estadística.
