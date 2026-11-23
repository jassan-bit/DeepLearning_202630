# Campo receptivo de capas convolucionales

El campo receptivo describe qué parte de la entrada puede influir en una unidad de una capa posterior. Su tamaño depende del ancho de cada filtro y de la separación entre las posiciones que se seleccionan al avanzar por la red.

## Lema 3.1 del documento

Consideremos $L$ capas convolucionales sucesivas con filtros cuadrados de lados $m_1,\ldots,m_L$ y strides $s_1,\ldots,s_L$. El lado del campo receptivo teórico de una unidad de la capa $L$, medido en coordenadas de la entrada, es

```{math}
:label: eq-campo-receptivo
r_L=1+\sum_{k=1}^{L}(m_k-1)\prod_{j=1}^{k-1}s_j.
```

Se supone que no hay dilatación y que cada capa aplica el mismo stride en las dos direcciones espaciales. El producto vacío, que aparece cuando $k=1$, se define como

$$
\prod_{j=1}^{0}s_j=1.
$$

Aquí el lado del campo receptivo significa la extensión desde la primera posición hasta la última posición que puede influir, contando ambas. Es una propiedad de la arquitectura. No afirma que todas las entradas de ese intervalo tengan un efecto distinto de cero para unos pesos y unas activaciones particulares.

## Paso 1. Considerar una sola dirección espacial

Basta demostrar la fórmula en una dimensión. En una imagen, el mismo argumento se aplica por separado a las filas y a las columnas. Como los filtros son cuadrados y el stride es igual en ambas direcciones, el tamaño resultante es el mismo en cada una.

En la entrada, una unidad corresponde a una posición. Por tanto, definimos

$$
r_0=1.
$$

En la primera capa, un filtro de ancho $m_1$ utiliza $m_1$ posiciones consecutivas. Así,

$$
r_1=m_1=1+(m_1-1).
$$

Este resultado coincide con la fórmula del lema para $L=1$.

## Paso 2. Calcular la separación entre unidades consecutivas

Dos unidades consecutivas de la primera capa se obtienen desplazando el filtro $s_1$ posiciones en la entrada. Por eso sus campos receptivos están desplazados $s_1$ posiciones entre sí.

Dos unidades consecutivas de la segunda capa se separan $s_2$ posiciones en la primera capa. Cada una de esas posiciones representa un desplazamiento de $s_1$ en la entrada. Por tanto, su separación, medida en la entrada, es $s_1s_2$.

Continuando este razonamiento, dos unidades consecutivas de la capa $k-1$ corresponden a campos receptivos desplazados

$$
\prod_{j=1}^{k-1}s_j
$$

posiciones en la entrada. Para $k=1$, esta separación es $1$, de acuerdo con la convención del producto vacío.

## Paso 3. Obtener la recurrencia del tamaño

Una unidad de la capa $k$ combina $m_k$ unidades consecutivas de la capa $k-1$.

El campo receptivo de la primera de esas unidades tiene extensión $r_{k-1}$. Desde la primera hasta la última de las $m_k$ unidades hay $m_k-1$ desplazamientos. Cada desplazamiento equivale, en coordenadas de la entrada, a

$$
\prod_{j=1}^{k-1}s_j.
$$

Por tanto, la extensión total es la del primer campo más el desplazamiento hasta el último:

```{math}
:label: eq-campo-receptivo-recurrencia
r_k=r_{k-1}+(m_k-1)\prod_{j=1}^{k-1}s_j.
```

**Por qué no aparece $s_k$ en el incremento de $r_k$.** El filtro de la capa $k$ utiliza $m_k$ unidades de la capa anterior, cuya separación ya está determinada por $s_1,\ldots,s_{k-1}$. El stride $s_k$ decide la separación entre dos salidas de la capa $k$; su efecto sobre el tamaño de un campo receptivo aparecerá al agregar capas posteriores.

## Paso 4. Desarrollar la recurrencia

Para las primeras capas,

$$
\begin{aligned}
r_1&=1+(m_1-1),\\
r_2&=r_1+(m_2-1)s_1\\
&=1+(m_1-1)+(m_2-1)s_1,\\
r_3&=r_2+(m_3-1)s_1s_2\\
&=1+(m_1-1)+(m_2-1)s_1+(m_3-1)s_1s_2.
\end{aligned}
$$

Al llegar a la capa $L$,

$$
\begin{aligned}
r_L
&=1+(m_1-1)+(m_2-1)s_1+\cdots\\
&\quad +(m_L-1)s_1s_2\cdots s_{L-1}\\
&=\boxed{1+\sum_{k=1}^{L}(m_k-1)\prod_{j=1}^{k-1}s_j}.
\end{aligned}
$$

La misma conclusión puede formalizarse por inducción: la fórmula vale para $L=1$; si vale para $L-1$, al sumar el incremento de la recurrencia para la capa $L$ se obtiene exactamente la suma hasta $L$. Esto completa la demostración.

## Ejemplo de tres capas

Consideremos filtros y strides

$$
(m_1,m_2,m_3)=(3,3,3),
\qquad
(s_1,s_2,s_3)=(1,2,1).
$$

Aplicando la recurrencia,

$$
r_1=1+(3-1)=3,
$$

$$
r_2=3+(3-1)\cdot1=5,
$$

$$
r_3=5+(3-1)\cdot1\cdot2=9.
$$

La fórmula cerrada da el mismo resultado:

$$
r_3=1+2\cdot1+2\cdot1+2\cdot(1\cdot2)=9.
$$

| Capa | Lado del filtro | Stride | Separación entre salidas en la entrada | Lado del campo receptivo |
|---|---:|---:|---:|---:|
| Entrada | — | — | $1$ | $1$ |
| $1$ | $3$ | $1$ | $1$ | $3$ |
| $2$ | $3$ | $2$ | $2$ | $5$ |
| $3$ | $3$ | $1$ | $2$ | $9$ |

Una unidad interior de la tercera capa tiene un campo receptivo teórico de $9\times9$ posiciones. Dos unidades vecinas de esa capa tienen campos receptivos desplazados $2$ posiciones en la entrada.

## Alcance de la fórmula

### Extensión y posiciones utilizadas

Cuando los strides son grandes, pueden quedar huecos entre las posiciones utilizadas. La fórmula sigue dando la extensión entre extremos, pero esa extensión puede contener posiciones que no participan en el cálculo.

Por ejemplo, si $m_1=1$, $s_1=2$ y $m_2=2$, una unidad de la segunda capa puede utilizar las posiciones $1$ y $3$ de la entrada. La extensión es $3$, aunque solo intervienen dos posiciones:

$$
r_2=1+(1-1)+(2-1)\cdot2=3.
$$

### Bordes y padding

Sin padding, la fórmula describe las unidades que existen dentro del dominio válido de las convoluciones. Con padding, también describe la extensión teórica sobre la cuadrícula ampliada, pero cerca de los bordes parte de ese campo corresponde a valores añadidos. La cantidad de posiciones de la imagen original que influyen puede ser menor.

### Dependencia real y campo teórico

Un peso igual a cero o una activación que anule una trayectoria puede hacer que algunas entradas no influyan para un caso concreto. El lema describe las conexiones posibles de la arquitectura, no el patrón de derivadas efectivamente no nulas para cada ejemplo.

### Dilatación

El enunciado no incluye convoluciones dilatadas. Cuando hay dilatación, la separación entre los elementos de cada filtro también debe incorporarse; no se debe aplicar sin ajustes la fórmula escrita aquí.

## Lectura del resultado

Cada capa agrega $(m_k-1)$ veces la separación acumulada de la capa anterior. Por eso un mismo filtro puede aumentar mucho más el campo receptivo cuando se coloca después de capas con stride mayor que uno. La profundidad acumula estos incrementos, mientras que los strides multiplican la escala espacial a la que operan las capas posteriores.
