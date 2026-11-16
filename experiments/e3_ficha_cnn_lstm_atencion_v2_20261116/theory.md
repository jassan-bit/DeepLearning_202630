# Cabeza de magnitud condicional: formulación y propiedades delimitadas

Esta ficha del 16/11/2026 es una propuesta posterior al primer ensayo. Conserva la base **CNN causal → LSTM unidireccional → atención convencional M00**, sin penalización de discordancia ni saturación radial. Su salida escalar parametriza una escala positiva; la distribución y la integración que siguen no sustituyen esa base por una arquitectura flow. No se acredita novedad R1 ni superioridad R3.

## 1. Objetivo, información y supuesto distributivo

Al comienzo de $t$ se conocen siete cierres $C_{t-7},\ldots,C_{t-1}>0$ y seis retornos $u_i=r_{t-7+i}$, $i=1,\ldots,6$, donde $r_j=\log C_j-\log C_{j-1}$. Sean
$$
m_6=\frac16\sum_{i=1}^6u_i,\quad A=\frac17\sum_{i=1}^6(u_i-m_6)^2,\quad c=\frac6{49},\quad U=|r_t-m_6|.
$$
El objetivo heredado es la desviación poblacional de siete retornos, sin anualización. Su identidad exacta es $Y=\sigma_7(t)=\sqrt{A+cU^2}$. $A$ se calcula sólo con historia observada; $U$ se obtiene durante aprendizaje de la etiqueta alineada mediante $U=\sqrt{(Y^2-A)/c}$. La [derivación y condiciones del objetivo](../e3_revision_post_piloto_20261116/theory.md) conservan los índices y su antecedente estadístico.

Se propone **directamente** $U\mid X\sim s_\theta(X)|T_5|$, con una única escala y cinco grados de libertad fijos. No se estima un signo ni una localización libre; no se pronostica un retorno firmado ni se afirma haber demostrado simetría de sus signos. La familia de magnitudes es un supuesto contrastable, no una consecuencia de la identidad de $Y$.

En el ajuste interno se fijan $u_{\rm scale}=\operatorname{media}_{\rm fit}(U)>0$ y $y_{\rm scale}=\operatorname{media}_{\rm fit}(Y)>0$. Una muestra degenerada con escala cero requiere detener y revisar el diseño. Para salida escalar $z_\theta$ se define
$$
s=u_{\rm scale}\,[10^{-4}+\operatorname{softplus}(z_\theta)],\qquad s_{\min}=10^{-4}u_{\rm scale}>0.
$$
Las escalas se congelan sin usar validación ni prueba. $s$ es escala Student, no desviación estándar de $U$ ni de $Y$.

## 2. Densidad, cambio de variable y borde

Escribiendo $f_5,F_5$ para densidad y CDF Student estándar, la densidad de magnitud es
$$
q_U(u;s)=\frac2s f_5(u/s)=\frac{16}{3\pi\sqrt5\,s}\left(1+\frac{u^2}{5s^2}\right)^{-3},\qquad u\ge0.
$$
El factor dos por plegado ya está incluido una vez. Para $y>\sqrt A$, la inversa $u(y)=\sqrt{(y^2-A)/c}>0$ es monótona y $du/dy=y/(cu)$. Por tanto,
$$
p_Y(y\mid A,s)=q_U(u(y);s)\frac{y}{cu(y)}.
$$
Es cero por debajo del piso. Al cambiar variable de $y$ a $u$ su integral es uno; no se añade otro factor dos. En el interior,
$$
-\log p_Y(y\mid A,s)=-\log q_U(u(y);s)-\log\frac{y}{cu(y)}.
$$
El jacobiano depende de etiqueta y $A$ observados, no de $\theta$: puede omitirse para optimizar parámetros interiores manteniendo ambos fijos. No se puede omitir al reportar el logscore completo de $Y$.

Si $A>0$, al aproximarse al piso se tiene $u\to0$ y $p_Y\to\infty$; una densidad no necesita ser finita en ese punto de probabilidad cero. NLL de $U=0$ es finita para cada $s>0$ y se usa como término auxiliar, sin afirmar un logscore de sigma finito en ese borde. Si $A=0$, en cambio, $Y=\sqrt c\,U$ es lineal y el límite es $p_Y(0)=q_U(0)/\sqrt c$, finito. La expresión $y/(cu)$ debe reemplazarse por ese límite, evitando $0/0$.

Para $y\ge\sqrt A$ y $0<p<1$,
$$
F_Y(y\mid A,s)=2F_5\!\left(\frac{\sqrt{(y^2-A)/c}}s\right)-1,\qquad
Q_Y(p)=\sqrt{A+cs^2\left[F_5^{-1}\!\left(\frac{1+p}{2}\right)\right]^2}.
$$
Debajo del piso la CDF es cero. Un intervalo central nominal del 90 % usa $Q_Y(0.05)$ y $Q_Y(0.95)$. Es nominal bajo la familia propuesta; no se denomina calibrado ni garantiza cobertura empírica o individual.

El plegado conserva una distribución continua: no introduce una masa puntual en $U=0$ ni en $Y=\sqrt A$. Para $s>0$, el límite de la CDF en el piso es cero y en infinito es uno. Un mecanismo de redondeo o censura que genere empates en el piso requeriría distinguir probabilidades de intervalos de una densidad evaluada en un punto.

## 3. Media continua y aproximación positiva de 128 nodos

La predicción deseada para MSE sobre sigma es $g(A,s)=E[\sqrt{A+cs^2T_5^2}]$, no $\sqrt{E[Y^2]}$. Integrando la densidad Student se obtiene
$$
E[U]=s\frac{4\sqrt5}{3\pi},\quad E[U^2]=\frac53s^2,\quad
\operatorname{Var}(U)=s^2\left(\frac53-\frac{80}{9\pi^2}\right),\quad E[Y^2]=A+\frac{10}{49}s^2.
$$
Así, $5s^2/3$ no es la varianza de la half-t. Por concavidad, $g(A,s)\le\sqrt{E[Y^2]}$, con desigualdad estricta para la distribución no degenerada aquí propuesta.

Para cualquier predicción $b$ y bajo la distribución condicional especificada,
$$
E[(b-Y)^2\mid X]=\operatorname{Var}(Y\mid X)+(b-E[Y\mid X])^2.
$$
De ahí procede usar la media de sigma: la raíz del segundo momento no minimiza ese riesgo. El momento finito necesario existe para cinco grados de libertad.

La variable $T_5^2/(5+T_5^2)$ sigue Beta$(1/2,5/2)$. Absorbiendo el factor $(1-v)^{-1/2}$ del integrando se obtiene exactamente
$$
g(A,s)=K_\beta E_{V\sim\operatorname{Beta}(1/2,2)}\sqrt{A(1-V)+5cs^2V},\quad
K_\beta=\frac{B(1/2,2)}{B(1/2,5/2)}=\frac{32}{9\pi}.
$$
La constante recíproca es $D_*=E[\sqrt{1-V}]=9\pi/32$. Se proponen nodos $v_i\in(0,1)$ y pesos positivos $w_i$ de Gauss–Jacobi, $N=128$, con convención $(1-x)^\alpha(1+x)^\beta$ en $[-1,1]$, $\alpha=1$, $\beta=-1/2$, $v=(x+1)/2$. Una normalización común de los pesos se cancela en
$$
D_N=\sum_iw_i\sqrt{1-v_i}>0,\qquad
g_N(A,s)=\frac{\sum_iw_i\sqrt{A(1-v_i)+5cs^2v_i}}{D_N}.
$$
Defínanse constantes $a_i=w_i\sqrt{1-v_i}/D_N$ y $z_i=\sqrt{5v_i/(1-v_i)}$. Entonces $a_i>0$, $\sum_i a_i=1$ y, exactamente para la regla discreta,
$$
g_N(A,s)=\sum_i a_i\sqrt{A+cs^2z_i^2}.
$$
La razón calibra algebraicamente el límite $s\to0$ al piso. Sigue siendo una aproximación: en $A=0$, $g_N=\sqrt c\,s\sum_i a_i z_i$, mientras $g=\sqrt c\,s\,4\sqrt5/(3\pi)$. No se supone igualdad entre esos coeficientes ni exactitud por elegir 128 nodos.

## 4. Propiedades exactas de la función discreta

**Proposición (soporte).** Para $A\ge0$ y $s\ge0$, $g_N(A,s)\ge\sqrt A$. Cada radical de su mezcla es al menos $\sqrt A$ y sus pesos positivos suman uno. Además, $g_N(A,0)=\sqrt A$ como extensión continua; el modelo entrenable conserva $s>s_{\min}$.

**Proposición (homogeneidad).** Para $k>0$,
$$
g_N(k^2A,ks)=k\,g_N(A,s).
$$
Se factoriza $k$ dentro de cada radical. Es una propiedad de $g_N$, no equivarianza demostrada de toda la red. En una conversión de unidades deben escalarse también $u_{\rm scale}$, $y_{\rm scale}$ y $s_{\min}$; mantener un epsilon físico numéricamente fijo invalidaría esa afirmación global.

**Proposición (Lipschitz respecto de escala).** Para $A\ge0$ fijo y $s,s'\ge0$,
$$
|g_N(A,s)-g_N(A,s')|\le L_N|s-s'|,\qquad L_N=\sqrt c\sum_i a_i z_i
=\sqrt{5c}\,\frac{\sum_iw_i\sqrt{v_i}}{D_N}.
$$
En cada término se aplica la desigualdad triangular inversa a los vectores $(\sqrt A,\sqrt c\,s z_i)$ y $(\sqrt A,\sqrt c\,s' z_i)$; después se suman los pesos. La misma prueba por integración da para la media continua $L_*=\sqrt c\,4\sqrt5/(3\pi)$. No se reemplaza $L_N$ por $L_*$ sin demostrar su relación.

La cota es sólo respecto de $s$. No cubre $A$, las entradas crudas o CNN/LSTM/atención. De hecho, para $s>0$ la derivada de la media continua respecto de $A$ diverge en $A=0$, por la densidad positiva de $U$ cerca de cero. La regularidad de una regla finita no autoriza atribuirle al promedio continuo esa cota adicional.

## 5. Objetivo propuesto y score auxiliar

Para $\widetilde U=U/u_{\rm scale}$ y $\widetilde s=s/u_{\rm scale}$ se propone, sobre ejemplos del ajuste,
$$
\mathcal L=\operatorname{media}\left[\left(\frac{g_{128}(A,s)-Y}{y_{\rm scale}}\right)^2\right]
+0.1\,\operatorname{media}\bigl[-\log q_{\widetilde U}(\widetilde U;\widetilde s)\bigr].
$$
Se evalúa la NLL de la variable normalizada; no se divide el valor de una NLL por una escala física. El jacobiano de esa normalización también es constante respecto de $\theta$. Los valores $\nu=5$, $N=128$, $0.1$ y $10^{-4}$ son decisiones propuestas, no hiperparámetros ganadores ni recomendaciones de las fuentes.

En particular, la transformación de unidades cumple
$$
q_{\widetilde U}(\widetilde u;\widetilde s)=u_{\rm scale}\,q_U(u;s),\qquad
-\log q_{\widetilde U}=-\log q_U-\log u_{\rm scale}.
$$

**Proposición (score auxiliar acotado).** Para $u\ge0$ fijo y $s>0$, derivando $\log s+3\log(1+u^2/(5s^2))$ más constantes,
$$
\frac{\partial[-\log q_U(u;s)]}{\partial\log s}=1-\frac{6u^2}{5s^2+u^2}\in[-5,1].
$$
La normalización de $U$ no cambia esa derivada. El límite inferior se aproxima en la cola y el superior se alcanza en $u=0$. Esta cota no acota el gradiente del MSE, el gradiente de toda la red ni demuestra robustez o calibración. Sin un piso positivo, NLL$(0;s)$ tendería a $-\infty$ cuando $s\to0$.

La regla de la cadena mantiene explícito el factor no controlado por esa proposición:
$$
\nabla_\theta[-\log q_U]=\left(1-\frac{6u^2}{5s^2+u^2}\right)\nabla_\theta\log s.
$$

La NLL continua y la media aproximada corresponden a objetos distintos: $g_{128}$ es una mezcla discreta de integración y no la media exacta de la densidad usada en la NLL. Por ello no se promete propiedad exacta del objetivo compuesto, ni calibración por minimizarlo. El error numérico, la adecuación distributiva y la cobertura requieren verificaciones separadas.

## 6. Supuestos, procedencia y límites académicos

La identidad requiere etiquetas exactas, fechas y unidades alineadas, ocho cierres para construir cada etiqueta y sólo siete barras pasadas como entrada. Una etiqueta que incumpla $Y^2\ge A$ fuera de una tolerancia numérica declarada debe rechazarse e investigarse; no se corrige silenciosamente mediante recorte. El borde $U=0$, las transformaciones y las tolerancias se registran por separado. Esta ficha no consume conjuntos reservados ni convierte validación ya usada en confirmación.

El cambio de variable y el plegado por valor absoluto tienen antecedentes matemáticos en [SurVAE Flows (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/9578a63fbe545bd82cc5bbe749636af1-Paper.pdf), sección 2, ecuación 1, y sección 3.1, tabla 2; el [apéndice D, ecuaciones 13–22](https://arxiv.org/pdf/2007.02731) desarrolla el valor absoluto y el signo. Se citan esos principios; no se adopta una arquitectura flow ni se atribuye al artículo esta cabeza condicional de sigma.

La descomposición de varianza, positividad y restricciones por muestra ya cuentan con antecedentes, recogidos en la [revisión posterior al piloto](../e3_revision_post_piloto_20261116/theory.md). Los teoremas anteriores son especializaciones de normas, homogeneidad y mezcla positiva conocidas. Una adaptación propuesta y una prueba correcta no acreditan por sí mismas R1, comprensión defendida de R2 o superioridad R3. No se atribuye retrospectivamente esta ficha al candidato previo ni se sustituyen sus resultados adversos.
