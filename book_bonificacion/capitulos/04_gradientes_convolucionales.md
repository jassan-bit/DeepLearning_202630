# Gradientes de una capa convolucional

Una misma entrada puede intervenir en varias salidas, y un mismo coeficiente del filtro se utiliza en todas las posiciones de la imagen. Por esta razón, sus derivadas se obtienen **sumando todas las contribuciones** que llegan desde la función de pérdida.

## Proposición 3.1 del documento

Sea

$$
O(i,j)=\sum_{t=1}^{m}\sum_{r=1}^{m}
h(t,r)I(i+t-1,j+r-1)+b,
$$

con $I\in\mathbb R^{l\times l}$, un filtro $h\in\mathbb R^{m\times m}$, stride $1$ y sin padding. Se supone $1\le m\le l$. Entonces

$$
O\in\mathbb R^{k\times k},\qquad k=l-m+1.
$$

Sea $\mathcal L$ una pérdida diferenciable que depende de $I$, $h$ y $b$ a través de $O$, y definamos

$$
\delta(i,j)=\frac{\partial\mathcal L}{\partial O(i,j)},
\qquad 1\le i,j\le k.
$$

Extendemos $\delta$ con ceros fuera de $\{1,\ldots,k\}^2$. Las identidades que debemos demostrar son

```{math}
:label: eq-conv-gradiente-filtro
\frac{\partial\mathcal L}{\partial h(t,r)}
=\sum_{i=1}^{k}\sum_{j=1}^{k}
\delta(i,j)I(i+t-1,j+r-1),
\qquad 1\le t,r\le m,
```

```{math}
:label: eq-conv-gradiente-entrada
\frac{\partial\mathcal L}{\partial I(u,v)}
=\sum_{t=1}^{m}\sum_{r=1}^{m}
h(t,r)\delta(u-t+1,v-r+1),
\qquad 1\le u,v\le l,
```

```{math}
:label: eq-conv-gradiente-sesgo
\frac{\partial\mathcal L}{\partial b}
=\sum_{i=1}^{k}\sum_{j=1}^{k}\delta(i,j).
```

**Convención de la operación.** La fórmula del enunciado utiliza el filtro sin invertirlo. Técnicamente es una correlación cruzada, aunque en redes neuronales se denomina habitualmente convolución. Todas las derivadas siguientes corresponden exactamente a esa fórmula. Si la pérdida contiene además un término que depende directamente de $h$, $I$ o $b$, habrá que sumar la derivada de ese término.

## Demostración del gradiente respecto al filtro

### Paso 1. Identificar la dependencia

Fijemos $t,r\in\{1,\ldots,m\}$. El coeficiente $h(t,r)$ participa en cada salida $O(i,j)$ multiplicando a

$$
I(i+t-1,j+r-1).
$$

Como $O(i,j)$ es lineal en los coeficientes del filtro,

$$
\frac{\partial O(i,j)}{\partial h(t,r)}
=I(i+t-1,j+r-1).
$$

Los demás términos de la suma no dependen del coeficiente que estamos derivando, y el término $b$ tampoco.

### Paso 2. Aplicar la regla de la cadena

La pérdida puede depender de $h(t,r)$ a través de cualquiera de las $k^2$ salidas. Por tanto,

$$
\frac{\partial\mathcal L}{\partial h(t,r)}
=\sum_{i=1}^{k}\sum_{j=1}^{k}
\frac{\partial\mathcal L}{\partial O(i,j)}
\frac{\partial O(i,j)}{\partial h(t,r)}.
$$

### Paso 3. Sustituir las derivadas

Usando la definición de $\delta(i,j)$ y el resultado del paso anterior,

$$
\boxed{
\frac{\partial\mathcal L}{\partial h(t,r)}
=\sum_{i=1}^{k}\sum_{j=1}^{k}
\delta(i,j)I(i+t-1,j+r-1)
}.
$$

La suma aparece porque el parámetro $h(t,r)$ se comparte entre todas las posiciones de salida.

## Demostración del gradiente respecto a la entrada

### Paso 1. Determinar cuándo aparece la entrada

Fijemos una posición $(u,v)$ de la entrada. En la salida $O(i,j)$ aparece $I(u,v)$ si existe una posición $(t,r)$ del filtro tal que

$$
u=i+t-1,\qquad v=j+r-1.
$$

Cuando esto ocurre, el coeficiente que multiplica a $I(u,v)$ es $h(t,r)$. Si la entrada no pertenece a la ventana de esa salida, la derivada correspondiente es cero. Por ello,

$$
\frac{\partial O(i,j)}{\partial I(u,v)}
=
\sum_{\substack{1\le t,r\le m\\
u=i+t-1,\ v=j+r-1}}h(t,r).
$$

Esta suma contiene un único término cuando la entrada está dentro de la ventana, y ningún término cuando está fuera.

### Paso 2. Aplicar la regla de la cadena

Sumamos las contribuciones de todas las salidas:

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial I(u,v)}
&=\sum_{i=1}^{k}\sum_{j=1}^{k}
\frac{\partial\mathcal L}{\partial O(i,j)}
\frac{\partial O(i,j)}{\partial I(u,v)}\\
&=\sum_{i=1}^{k}\sum_{j=1}^{k}
\delta(i,j)
\sum_{\substack{1\le t,r\le m\\
u=i+t-1,\ v=j+r-1}}h(t,r).
\end{aligned}
$$

### Paso 3. Despejar los índices de salida

Las igualdades que indican dónde aparece $I(u,v)$ equivalen a

$$
i=u-t+1,\qquad j=v-r+1.
$$

Como todas las sumas son finitas, podemos agrupar primero por la posición del filtro:

$$
\frac{\partial\mathcal L}{\partial I(u,v)}
=\sum_{t=1}^{m}\sum_{r=1}^{m}h(t,r)
\sum_{\substack{1\le i,j\le k\\
i=u-t+1,\ j=v-r+1}}\delta(i,j).
$$

Si $(u-t+1,v-r+1)$ es una posición válida de salida, la suma interior vale $\delta(u-t+1,v-r+1)$. En caso contrario, vale cero. La extensión de $\delta$ con ceros permite escribir ambos casos en una sola expresión:

$$
\boxed{
\frac{\partial\mathcal L}{\partial I(u,v)}
=\sum_{t=1}^{m}\sum_{r=1}^{m}
h(t,r)\delta(u-t+1,v-r+1)
}.
$$

**Por qué aparecen signos negativos en los índices.** No se introducen por una regla arbitraria de inversión del filtro: surgen al despejar $i$ y $j$ en las relaciones $u=i+t-1$ y $v=j+r-1$.

## Demostración del gradiente respecto al sesgo

El mismo escalar $b$ se suma una vez en cada salida. En consecuencia,

$$
\frac{\partial O(i,j)}{\partial b}=1.
$$

Aplicamos la regla de la cadena y sustituimos:

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial b}
&=\sum_{i=1}^{k}\sum_{j=1}^{k}
\frac{\partial\mathcal L}{\partial O(i,j)}
\frac{\partial O(i,j)}{\partial b}\\
&=\sum_{i=1}^{k}\sum_{j=1}^{k}\delta(i,j)\cdot1\\
&=\boxed{\sum_{i=1}^{k}\sum_{j=1}^{k}\delta(i,j)}.
\end{aligned}
$$

Quedan demostradas las tres identidades.

## Ejemplo de comprobación

Este ejemplo ilustra las fórmulas anteriores. Tomemos

$$
I=\begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix},
\qquad
h=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad b=1.
$$

Aquí $l=3$, $m=2$ y $k=2$. Por ejemplo,

$$
O(1,1)=1\cdot1+0\cdot2+0\cdot4-1\cdot5+1=-3.
$$

Las otras tres salidas también valen $-3$. Si elegimos, solo para esta comprobación, la pérdida

$$
\mathcal L=\sum_{i=1}^{2}\sum_{j=1}^{2}O(i,j),
$$

entonces $\delta(i,j)=1$ en las cuatro posiciones. Los gradientes son

$$
\nabla_h\mathcal L=
\begin{pmatrix}
1+2+4+5&2+3+5+6\\
4+5+7+8&5+6+8+9
\end{pmatrix}
=\begin{pmatrix}12&16\\24&28\end{pmatrix},
$$

$$
\frac{\partial\mathcal L}{\partial b}=4,
\qquad
\nabla_I\mathcal L=
\begin{pmatrix}
1&1&0\\
1&0&-1\\
0&-1&-1
\end{pmatrix}.
$$

La entrada central $I(2,2)$ participa en cuatro ventanas, pero sus coeficientes suman $-1+0+0+1=0$. De ahí que su derivada sea cero en este ejemplo.

## Lectura del resultado

El gradiente del filtro acumula los productos entre el error de cada salida y la entrada que multiplicaba a ese coeficiente. El gradiente de la entrada acumula las contribuciones de todas las ventanas que la utilizaron. El gradiente del sesgo suma todos los errores de salida porque ese mismo sesgo afecta a todas las posiciones.

