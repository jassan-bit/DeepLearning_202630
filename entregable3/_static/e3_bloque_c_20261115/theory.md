# Formulación y demostración del mecanismo candidato

**Estado:** desarrollo teórico del candidato de E3. La propiedad demostrada corresponde a la cabeza que recibe estados latentes, con parámetros de inferencia fijos. No demuestra novedad bibliográfica, superioridad estadística ni una cota completa desde OHLCV crudo. La implementación asistida por IA y su comprensión deben declararse por separado.

## 1 Notación y dominio

Una ventana contiene $n$ barras anteriores al día objetivo, con $m=5$ canales. En el problema principal, $n=7$. El codificador CNN causal y LSTM unidireccional produce

$$
H=\begin{bmatrix}h_1^\top\\\vdots\\h_n^\top\end{bmatrix}
\in\mathbb R^{n\times d}.
$$

La formulación siguiente usa cualquier $H$ finito de ese espacio. Se emplea la norma euclídea para vectores, la norma de Frobenius para matrices de estados y la norma espectral para operadores. Los vectores $q,w\in\mathbb R^d$ y el sesgo $b\in\mathbb R$ son parámetros efectivos de inferencia.

Sea $P\in\mathbb R^{n\times n}$ una matriz de promedio causal **fija e independiente de $H$**. Para un promedio de anchura $k$ que incluye el estado presente,

$$
P_{j\ell}=
\begin{cases}
1/\min(k,j), & \max(1,j-k+1)\le\ell\le j,\\
0, & \text{en otro caso}.
\end{cases}
$$

El prototipo puede usar $k=3$ como configuración de desarrollo; no es una receta seleccionada experimentalmente ni un preregistro. Definimos $A=I-P$ y $r_j=(AH)_j$. Los hiperparámetros satisfacen

$$
B>0,\quad\varepsilon>0,\quad\tau\ge\tau_{\min}>0,\quad
0\le\lambda\le\Lambda,\quad\|q\|_2\le Q,\quad\|w\|_2\le W.
$$

$B$ es un radio escalar de saturación. $\varepsilon$ suaviza la norma en la discordancia y evita división por cero en sus derivadas. La temperatura $\tau$ y la intensidad $\lambda$ son escalares independientes de la entrada durante inferencia. No se aplica este teorema a una temperatura, radio u operador de promedio que cambie con $H$ sin volver a derivar la cota.

## 2 Mecanismo y salida

La discordancia local, los puntajes y las ponderaciones son

$$
d_j(H)=\sqrt{\|r_j\|_2^2+\varepsilon^2},\qquad
s_j(H)=q^\top h_j-\lambda d_j(H),\qquad
a(H)=\operatorname{softmax}\bigl(s(H)/\tau\bigr).
$$

Cada valor se transforma mediante la saturación radial

$$
v_j(H)=T_B(h_j)=\frac{B h_j}{\sqrt{B^2+\|h_j\|_2^2}}.
$$

La agregación y la predicción escalada son

$$
c(H)=\sum_{j=1}^n a_j(H)v_j(H),\qquad
z(H)=w^\top c(H)+b,\qquad
f(H)=\operatorname{softplus}(z(H)).
$$

El término de contenido $q^\top h_j$ es convencional. Las modificaciones candidatas son la penalización por discordancia y la saturación radial de los valores. Su combinación requiere revisión de antecedentes; escribir estas ecuaciones o programarlas no acredita por sí mismo R1.

La salida del modelo está en unidades $\sigma_7/a_{\rm tr}$, donde $a_{\rm tr}>0$ es la media del objetivo del entrenamiento permitido. La predicción en unidades originales es $\widehat\sigma_7=a_{\rm tr}f(H)$; no se vuelve a estimar ese factor con validación o prueba.

## 3 Restricciones suaves de los parámetros

Una implementación diferenciable que cumple las restricciones anteriores usa parámetros libres $u_q,u_w,\theta_\lambda,\theta_\tau$:

$$
q=\frac{Q u_q}{\sqrt{1+\|u_q\|_2^2}},\qquad
w=\frac{W u_w}{\sqrt{1+\|u_w\|_2^2}},
$$

$$
\lambda=\Lambda\operatorname{sigmoid}(\theta_\lambda),\qquad
\tau=\tau_{\min}+\operatorname{softplus}(\theta_\tau).
$$

Para parámetros libres finitos, $\|q\|_2<Q$, $\|w\|_2<W$, $0<\lambda<\Lambda$ y $\tau>\tau_{\min}$. La demostración permite también los extremos no estrictos, usados por las ablaciones. Las constantes $Q,W,B,\Lambda,\tau_{\min}$ y $\varepsilon$ forman parte de la configuración y deben guardarse con el estado del modelo; no son cotas que se presuman sin comprobar el código efectivo.

## 4 Lemas

### Lema 1 La saturación radial es 1 Lipschitz y acota los valores

Para $u=\sqrt{B^2+\|h\|_2^2}$,

$$
J_{T_B}(h)=\frac B u I-\frac B{u^3}hh^\top.
$$

En la dirección de $h$ su autovalor es $B^3/u^3$; en las direcciones ortogonales es $B/u$. En $h=0$ el jacobiano es $I$. Todos los autovalores pertenecen a $(0,1]$, de modo que $\|J_{T_B}(h)\|_2\le1$ para cualquier $h$. Integrando el jacobiano sobre el segmento entre dos puntos,

$$
\|T_B(h)-T_B(h')\|_2\le\|h-h'\|_2.
$$

Además,

$$
\|T_B(h)\|_2=\frac{B\|h\|_2}{\sqrt{B^2+\|h\|_2^2}}\le B.
$$

Si $V(H)$ es la matriz cuyas filas son $v_j(H)^\top$, se sigue que

$$
\|V(H)-V(H')\|_F\le\|H-H'\|_F,\qquad
\|V(H)\|_2\le\|V(H)\|_F\le B\sqrt n.
$$

### Lema 2 La discordancia es Lipschitz

Puede escribirse $d_j(H)=\|(r_j,\varepsilon)\|_2$. La desigualdad triangular inversa da

$$
|d_j(H)-d_j(H')|\le\|r_j-r_j'\|_2.
$$

Al elevar al cuadrado y sumar por filas,

$$
\|d(H)-d(H')\|_2\le\|A(H-H')\|_F
\le\|A\|_2\|H-H'\|_F.
$$

Por tanto,

$$
\|s(H)-s(H')\|_2
\le\bigl(Q+\Lambda\|A\|_2\bigr)\|H-H'\|_F.
$$

$\varepsilon>0$ asegura diferenciabilidad, pero no empeora esta cota.

### Lema 3 Softmax tiene constante euclídea a lo sumo 1/2

Para una distribución $a=\operatorname{softmax}(u)$,

$$
J_{\rm softmax}(u)=\operatorname{diag}(a)-aa^\top.
$$

La suma absoluta de la fila $j$ es

$$
a_j(1-a_j)+\sum_{i\ne j}a_ja_i
=2a_j(1-a_j)\le\tfrac12.
$$

El jacobiano es simétrico, por lo que su norma espectral está acotada por su norma infinita. Así, $\|J_{\rm softmax}(u)\|_2\le1/2$. La integración sobre segmentos demuestra que softmax es $1/2$ Lipschitz en norma euclídea. Aplicando el escalado por temperatura y el Lema 2,

$$
\|a(H)-a(H')\|_2
\le\frac{Q+\Lambda\|A\|_2}{2\tau_{\min}}\|H-H'\|_F.
$$

La prueba de este lema y las demás son autocontenidas; no se apela a una afirmación de estabilidad sin derivarla.

## 5 Teorema de estabilidad de la cabeza latente

**Teorema.** Bajo los supuestos de las secciones 1 y 3, para cualesquiera $H,H'\in\mathbb R^{n\times d}$ finitos y los mismos parámetros de inferencia,

$$
|f(H)-f(H')|\le C_{\rm head}\|H-H'\|_F,
$$

donde

$$
\boxed{
C_{\rm head}=W\left[
1+\frac{B\sqrt n}{2\tau_{\min}}
\left(Q+\Lambda\|I-P\|_2\right)\right].
}
$$

**Demostración.** Escribimos $c(H)=V(H)^\top a(H)$ y descomponemos

$$
c(H)-c(H')
=V(H)^\top\bigl(a(H)-a(H')\bigr)
+\bigl(V(H)-V(H')\bigr)^\top a(H').
$$

Como $a(H')$ es una distribución, $\|a(H')\|_2\le1$. Aplicando los tres lemas y las desigualdades de normas,

$$
\begin{aligned}
\|c(H)-c(H')\|_2
&\le B\sqrt n\,\|a(H)-a(H')\|_2
+\|V(H)-V(H')\|_F\|a(H')\|_2\\
&\le\left[1+
\frac{B\sqrt n}{2\tau_{\min}}
\left(Q+\Lambda\|I-P\|_2\right)\right]\|H-H'\|_F.
\end{aligned}
$$

La derivada de Softplus es sigmoid, perteneciente a $(0,1)$; por ello Softplus es 1 Lipschitz. La proyección escalar $w^\top c+b$ tiene constante $\|w\|_2\le W$. Componiendo ambas funciones se obtiene la cota enunciada. El sesgo fijo $b$ se cancela al tomar diferencias. $\square$

Para el problema principal basta sustituir $n=7$. La cota puede ser conservadora; no se afirma que sea la constante Lipschitz óptima. La derivación usa aritmética real. Los ensayos de coma flotante requieren rangos y tolerancias declarados; no prometen ausencia de overflow para cualquier flotante representable.

### Corolario de rango de salida

Como $c$ es una combinación convexa de vectores de norma a lo sumo $B$, se tiene $\|c\|_2\le B$. Por tanto,

$$
\operatorname{softplus}(b-WB)\le f(H)
\le\operatorname{softplus}(b+WB).
$$

Este rango finito es una propiedad del modelo con pesos fijos, no una garantía de calibración. También muestra una limitación: radios o cotas demasiado restrictivos pueden impedir representar volatilidad extrema y agravar la infracobertura. El conjunto de desarrollo deberá evaluar esa posibilidad antes de congelar el diseño.

### Escala original y límite de la afirmación

En unidades de $\sigma_7$, la constante es $a_{\rm tr}C_{\rm head}$. Para deducir una cota respecto a la entrada preprocesada $X$ sería necesario demostrar una constante $L_E$ del codificador $E$ y usar $a_{\rm tr}C_{\rm head}L_E$. Para OHLCV crudo haría falta además controlar la transformación $T$, sus rangos y su constante $L_T$.

Aquí no se han cuantificado $L_E$ ni $L_T$. En particular, logaritmos, normalización y anclaje por el último cierre requieren rangos y escalas válidos; no se obtiene una cota global sobre todo OHLCV crudo por multiplicar constantes no demostradas. Que la cabeza tenga una cota no demuestra que el predictor completo sea más robusto que otro modelo, ni que reduzca la RMSE.

## 6 Derivadas y regla de actualización

Para una observación sea $\mathcal L=\ell(f(H),y^*)$ y $\delta=\partial\mathcal L/\partial z$. Con la pérdida MSE de un lote de $N$ observaciones,

$$
\mathcal L=\frac1N\sum_{i=1}^N\bigl(f(H_i)-y_i^*\bigr)^2,
\qquad y_i^*=y_i/a_{\rm tr},
$$

la contribución de la observación $i$ satisface

$$
\delta_i=\frac2N\bigl(f(H_i)-y_i^*\bigr)
\operatorname{sigmoid}(z_i).
$$

El escalado objetivo es fijo por entrenamiento: multiplicar esta MSE por $a_{\rm tr}^2$ recupera la MSE en unidades originales sin cambiar su minimizador.

Para una observación definimos $g=\delta w$, $\kappa_j=g^\top v_j$ y

$$
\eta_j=\frac{a_j}{\tau}
\left(\kappa_j-\sum_{i=1}^n a_i\kappa_i\right)
=\frac{\partial\mathcal L}{\partial s_j}.
$$

Los jacobianos de las partes centrales son

$$
J_{T_B}(h_j)=\frac B{u_j}I-\frac B{u_j^3}h_jh_j^\top,
\qquad u_j=\sqrt{B^2+\|h_j\|_2^2},
$$

$$
\frac{\partial d_i}{\partial h_j}
=A_{ij}\frac{r_i}{d_i},\qquad
\frac{\partial s_i}{\partial h_j}
=\mathbf1_{i=j}q-\lambda A_{ij}\frac{r_i}{d_i}.
$$

Aquí los gradientes de los escalares respecto a $h_j$ se expresan como vectores columna. En consecuencia,

$$
\boxed{
\frac{\partial\mathcal L}{\partial h_j}
=a_jJ_{T_B}(h_j)^\top g+\eta_jq
-\lambda\sum_{i=1}^n A_{ij}\eta_i\frac{r_i}{d_i}.
}
$$

En forma matricial, si la fila $i$ de $D$ es $\eta_i r_i^\top/d_i$, el último término es la fila $j$ de $-\lambda A^\top D$. Esta forma muestra el acoplamiento del promedio causal: no basta derivar cada $d_j$ como si dependiera sólo de $h_j$.

Para los parámetros efectivos,

$$
\frac{\partial\mathcal L}{\partial w}=\delta c,\qquad
\frac{\partial\mathcal L}{\partial b}=\delta,\qquad
\frac{\partial\mathcal L}{\partial q}=H^\top\eta,
$$

$$
\frac{\partial\mathcal L}{\partial\lambda}=-d^\top\eta,
\qquad
\frac{\partial\mathcal L}{\partial\tau}
=-\frac{s^\top\eta}{\tau}.
$$

Las restricciones suaves requieren la regla de la cadena. Para $G_K(u)=Ku/\sqrt{1+\|u\|_2^2}$ y $v=\sqrt{1+\|u\|_2^2}$,

$$
J_{G_K}(u)=\frac K v I-\frac K{v^3}uu^\top.
$$

Se aplica este jacobiano a los gradientes de $q$ y $w$. Además,

$$
\frac{\partial\lambda}{\partial\theta_\lambda}
=\Lambda\sigma(\theta_\lambda)\bigl(1-\sigma(\theta_\lambda)\bigr),\qquad
\frac{\partial\tau}{\partial\theta_\tau}=\sigma(\theta_\tau).
$$

Los gradientes se propagan desde $H$ a la LSTM y a la CNN mediante diferenciación automática. Una actualización SGD ilustrativa es $\theta\leftarrow\theta-\alpha\nabla_\theta\mathcal L$; si se emplea AdamW, se guardarán su configuración y estado, y no se confundirá su decaimiento desacoplado con la derivada de esta MSE. La elección de optimizador, presupuesto y parada deberá ser comparable con los rivales en el protocolo definitivo.

## 7 Pseudocódigo

### Algoritmo 1 Predicción de una ventana

**Entradas:** ventana $X\in\mathbb R^{n\times5}$ estandarizada con estadísticas de entrenamiento, configuración, pesos del codificador y parámetros libres de la cabeza, escala $a_{\rm tr}>0$. **Salida:** $\widehat\sigma_7$.

1. Validar forma, tipo y finitud de $X$; rechazar una ventana inválida antes de inferir.
2. Aplicar la CNN con relleno sólo por el lado histórico y producir sus $n$ representaciones.
3. Recorrer la LSTM unidireccional de $j=1$ a $n$; conservar sus estados $H$.
4. Construir los parámetros efectivos $q,w,\lambda,\tau$ mediante las restricciones suaves.
5. Calcular $R=(I-P)H$ y $d_j=\sqrt{\|r_j\|_2^2+\varepsilon^2}$.
6. Calcular $s=Hq-\lambda d$ y $a=\operatorname{softmax}(s/\tau)$ con estabilización numérica por el máximo.
7. Calcular cada $v_j=B h_j/\sqrt{B^2+\|h_j\|_2^2}$ y $c=\sum_j a_jv_j$.
8. Devolver $\widehat\sigma_7=a_{\rm tr}\operatorname{softplus}(w^\top c+b)$.

El algoritmo termina tras recorrer $n$ estados y evaluar la cabeza. No recibe el objetivo del día pronosticado ni ejecuta optimización durante inferencia.

### Algoritmo 2 Desarrollo y entrenamiento de una variante

**Entradas:** particiones de entrenamiento y validación de desarrollo, configuración candidata, semilla, optimizador, máximo de épocas $E$, paciencia $p$ y criterio de mejora definidos antes de la ejecución. **Salida:** estado elegido por validación y registro completo de la variante.

1. Fijar generadores y configuración de determinismo sin alterar silenciosamente otros experimentos; inicializar los pesos en un contexto de RNG controlado.
2. Estimar las transformaciones exclusivamente en el entrenamiento permitido y guardar sus estadísticas.
3. Para cada época, optimizar la MSE sobre lotes de entrenamiento usando el Algoritmo 1 en escala de objetivo y las derivadas anteriores.
4. Evaluar en validación, sin actualizar pesos o transformaciones; conservar el estado con mejor criterio de validación y registrar las épocas.
5. Detenerse al alcanzar $E$ o $p$ épocas sin mejora, según la regla prefijada.
6. Guardar pesos, configuración, estado del optimizador cuando corresponda, semilla, curvas, tiempo y desenlace; registrar la variante en la bitácora y conservar fallos.

Este algoritmo describe el desarrollo. No fija una campaña confirmatoria ni autoriza usar los conjuntos adicionales para cambiar el mecanismo. La confirmación requiere diseño congelado, presupuesto común y preregistro previo; ninguna tabla de pruebas sintéticas lo sustituye.

## 8 Causalidad y complejidad

La CNN y la LSTM unidireccional no necesitan posiciones posteriores para calcular un estado de un prefijo. $P$ usa sólo estados hasta $j$ para su promedio. Sin embargo, el denominador de softmax reúne todos los $n$ estados: un peso $a_j$ puede cambiar si cambia una posición posterior de la misma ventana. No se afirma causalidad por prefijo de los pesos de atención.

La causalidad por prefijo del encoder se refiere al tensor ya preprocesado. `relative_window` ancla todos los precios al último cierre conocido de la ventana; esa transformación no es invariante a modificar un sufijo de la ventana cruda. Esto no introduce el objetivo del día siguiente si toda la ventana termina antes de emitir el pronóstico, pero impide atribuir al pipeline crudo la propiedad por prefijo probada para el encoder.

La predicción del día $t$ sigue siendo causal si las $n$ barras de entrada pertenecen a días estrictamente anteriores a $t$, las transformaciones están congeladas y ninguna variable de entrada contiene el objetivo aún no disponible. Esa disponibilidad temporal debe auditarse en el pipeline real, no deducirse únicamente de la arquitectura.

Para una CNN con $L$ capas, anchuras $c_0=5,c_1,\ldots,c_L$ y kernels $k_\ell$, el coste por ventana es

$$
O\left(n\sum_{\ell=1}^{L}k_\ell c_{\ell-1}c_\ell\right).
$$

Una LSTM con entrada $c_L$ y estado $d$ cuesta $O\bigl(nd(c_L+d)\bigr)$. La cabeza con multiplicación **densa** por $P$ cuesta $O(n^2d)$, más $O(nd)$ por puntajes, saturación y suma. Su memoria incluye $O(n^2)$ para el operador y $O(nd)$ para estados/valores. Con un promedio móvil realmente implementado mediante sumas acumuladas o bandas puede reducirse ese término; no se atribuye esa optimización a una implementación que materialice y multiplique la matriz densa.

Con $B,Q,W,\Lambda,\tau_{\min},\varepsilon$ y $P$ fijos, la cabeza completa tiene $2d+3$ parámetros entrenables: $u_q,u_w,b,\theta_\lambda,\theta_\tau$. El número total y los parámetros activos de cada ablación deben contarse desde el código. La retropropagación tiene el mismo orden asintótico que el forward para esta arquitectura de profundidad fija, con activaciones adicionales guardadas. FLOPs, memoria y tiempo empíricos deberán medirse; la notación $O(\cdot)$ no los reemplaza.

## 9 Ablaciones y verificación numérica

Las tres intervenciones funcionales propuestas mantienen el mismo codificador:

| Variante | Intervención aislada | Qué permite examinar | Alcance de la cota |
|---|---|---|---|
| Sin penalización | Forzar $\lambda=0$ | Aporte de la discordancia al reparto de pesos. | Teorema válido con $\Lambda_{\rm efectiva}=0$. |
| Sin saturación | Sustituir $v_j$ por $h_j$ | Aporte del acotamiento de valores. | No se aplica el teorema global enunciado con radio $B$; no devolver esa constante como si siguiera demostrada. |
| Sin contenido | Forzar $q=0$ y conservar la penalización | Interacción entre contenido convencional y discordancia. | Teorema válido con $Q_{\rm efectivo}=0$. |

Son tres componentes aislados para estudiar el mecanismo. La tercera intervención examina un componente convencional; no constituye una tercera innovación. La comparación con una cabeza convencional de contenido lineal que retire simultáneamente las dos modificaciones es un control adicional, no otra ablación aislada. La admisibilidad y suficiencia del diseño frente a la guía deberán evaluarse con la revisión y el protocolo completos.

La verificación numérica debe registrar semilla, dtype, configuración y versión del código y comprobar al menos:

1. Gradcheck de doble precisión sobre la cabeza y comparación de gradientes analíticos/autodiferenciados en casos no degenerados.
2. Restricciones efectivas $\|q\|_2\le Q$, $\|w\|_2\le W$, $\lambda\le\Lambda$ y $\tau\ge\tau_{\min}$, incluso cerca de sus límites.
3. Cocientes $|f(H)-f(H')|/\|H-H'\|_F$ para pares distintos en varias escalas y perturbaciones pequeñas, comparados con $C_{\rm head}$; reportar máximo y cantidad de violaciones.
4. Normas del jacobiano en estados seleccionados y casos especiales, incluido $H=0$, donde $\varepsilon$ mantiene derivadas definidas.
5. Predicciones antes/después de guardar, cargar y exportar, con tolerancia explícita, además de pruebas de entrada inválida y gradientes de todos los parámetros activos.

Los ensayos finitos complementan la demostración; no demuestran por muestreo una propiedad universal ni sustituyen las comparaciones de RMSE contra todo el benchmark. Sus resultados se incorporarán desde ejecuciones verificadas, sin rellenar valores anticipados.
