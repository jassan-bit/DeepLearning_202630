# Retropropagación y regla de la cadena

La retropropagación calcula las derivadas de una pérdida recorriendo la red desde su salida hacia su entrada. En cada capa reutiliza la derivada que ya se ha calculado en la capa siguiente. El resultado es una aplicación organizada de la regla de la cadena.

## Teorema 3.3 del documento

Sea

$$
\mathcal L(\theta)=\ell(a_L),\qquad a_0=x,
\qquad a_k=f_k(a_{k-1};\theta_k)\in\mathbb R^{n_k},
$$

donde cada $f_k$ es de clase $C^1$ y $\ell:\mathbb R^{n_L}\to\mathbb R$ es de clase $C^1$. Definimos, desde la última capa hacia la primera,

```{math}
:label: eq-backprop-recursion
\delta_L=\nabla\ell(a_L),\qquad
\delta_{k-1}=J_k^T\delta_k,\qquad
J_k=\frac{\partial f_k}{\partial a_{k-1}}(a_{k-1};\theta_k).
```

Entonces, para cada capa,

```{math}
:label: eq-backprop-gradientes
\nabla_{a_k}\mathcal L=\delta_k,
\qquad
\nabla_{\theta_k}\mathcal L
=\left(\frac{\partial f_k}{\partial\theta_k}\right)^T\delta_k.
```

La primera identidad se extiende a $k=0$. La segunda se aplica a los bloques de parámetros $\theta_k$ de las capas $k=1,\ldots,L$.

Si los productos $J_k^T\delta_k$ y $(\partial f_k/\partial\theta_k)^T\delta_k$ se pueden evaluar a un costo acotado por una misma constante multiplicada por el costo de evaluar $f_k$, el paso hacia atrás tiene un costo de ese mismo orden que el paso hacia adelante. Los productos pueden calcularse sin construir las matrices jacobianas completas.

**Hipótesis sobre los parámetros.** La expresión por capa supone que $\theta_k$ es un bloque independiente y que la pérdida no tiene una dependencia adicional directa de ese bloque. Si un parámetro se comparte entre varias capas, sus contribuciones se suman. Si existe una penalización explícita en la pérdida, también se suma su gradiente.

## Convención de gradientes y dimensiones

Escribimos los gradientes como vectores columna. Como $f_k$ transforma una entrada de dimensión $n_{k-1}$ en una salida de dimensión $n_k$,

$$
J_k\in\mathbb R^{n_k\times n_{k-1}},
\qquad
\delta_k\in\mathbb R^{n_k}.
$$

Por tanto,

$$
J_k^T\delta_k\in\mathbb R^{n_{k-1}}.
$$

La transpuesta hace que el resultado tenga la dimensión de la entrada de la capa. La derivada $\partial f_k/\partial\theta_k$ tiene una fila por componente de $f_k$ y una columna por componente de $\theta_k$; su transpuesta convierte $\delta_k$ en un gradiente con una componente por parámetro.

## Demostración del gradiente respecto a las activaciones

### Paso 1. Comenzar en la última capa

Por la definición de la pérdida,

$$
\mathcal L=\ell(a_L).
$$

Al derivar respecto a $a_L$ obtenemos directamente

$$
\nabla_{a_L}\mathcal L=\nabla\ell(a_L)=\delta_L.
$$

Esta es la base de la inducción hacia atrás.

### Paso 2. Aplicar la regla de la cadena a una capa

Supongamos que ya sabemos que

$$
\nabla_{a_k}\mathcal L=\delta_k.
$$

Al variar $a_{k-1}$ y mantener fijos los parámetros de la capa,

$$
\mathrm da_k=J_k\,\mathrm da_{k-1}.
$$

El diferencial de la pérdida es

$$
\begin{aligned}
\mathrm d\mathcal L
&=(\nabla_{a_k}\mathcal L)^T\mathrm da_k\\
&=\delta_k^T J_k\,\mathrm da_{k-1}\\
&=(J_k^T\delta_k)^T\mathrm da_{k-1}.
\end{aligned}
$$

Pero la definición del gradiente respecto a $a_{k-1}$ también da

$$
\mathrm d\mathcal L
=(\nabla_{a_{k-1}}\mathcal L)^T\mathrm da_{k-1}.
$$

Como la igualdad se cumple para cualquier variación $\mathrm da_{k-1}$,

$$
\nabla_{a_{k-1}}\mathcal L=J_k^T\delta_k=\delta_{k-1}.
$$

### Paso 3. Repetir hasta la entrada

La identidad es cierta para $k=L$, y el paso anterior demuestra que su validez en una capa implica su validez en la capa anterior. Por inducción inversa,

$$
\boxed{\nabla_{a_k}\mathcal L=\delta_k},
\qquad k=0,1,\ldots,L.
$$

En particular, $\delta_0=\nabla_x\mathcal L$ es la sensibilidad de la pérdida respecto a la entrada de la red.

## Demostración del gradiente respecto a los parámetros

### Paso 1. Identificar por dónde influye el parámetro

Al modificar $\theta_k$, manteniendo fijos la entrada y los demás bloques de parámetros, las activaciones anteriores a la capa $k$ permanecen fijas. La variación en la salida de esta capa es

$$
\mathrm da_k
=\frac{\partial f_k}{\partial\theta_k}\,\mathrm d\theta_k.
$$

Las capas posteriores ya están incorporadas en $\delta_k=\nabla_{a_k}\mathcal L$.

### Paso 2. Calcular el diferencial de la pérdida

$$
\begin{aligned}
\mathrm d\mathcal L
&=\delta_k^T\mathrm da_k\\
&=\delta_k^T\frac{\partial f_k}{\partial\theta_k}
\,\mathrm d\theta_k\\
&=\left[
\left(\frac{\partial f_k}{\partial\theta_k}\right)^T\delta_k
\right]^T\mathrm d\theta_k.
\end{aligned}
$$

Comparando con

$$
\mathrm d\mathcal L
=(\nabla_{\theta_k}\mathcal L)^T\mathrm d\theta_k,
$$

concluimos que

$$
\boxed{
\nabla_{\theta_k}\mathcal L
=\left(\frac{\partial f_k}{\partial\theta_k}\right)^T\delta_k
}.
$$

## Por qué basta un solo recorrido hacia atrás

Primero se calculan y conservan las activaciones $a_0,a_1,\ldots,a_L$ durante el recorrido hacia adelante. Después se calcula $\delta_L$ a partir de la pérdida. En la capa $k$, el valor de $\delta_k$ permite obtener tanto el gradiente de sus parámetros como $\delta_{k-1}$. Una vez calculados, se continúa con la capa anterior.

No hace falta volver a recorrer toda la red por cada parámetro: las dependencias posteriores ya están resumidas en $\delta_k$.

### Costo computacional

La hipótesis del teorema exige que exista una cota uniforme para el costo de los productos locales, relativa al costo del paso hacia adelante de la correspondiente capa. Al sumar sobre todas las capas, esa misma cota uniforme acota el costo total del recorrido hacia atrás por una constante multiplicada por el costo total de las capas en el recorrido hacia adelante.

El costo de evaluar $\ell$ y $\nabla\ell$ debe incluirse por separado, o tratarse como una última operación del grafo con una propiedad de costo análoga. La afirmación de complejidad **no se deduce solamente de que las funciones sean diferenciables**: utiliza la hipótesis adicional sobre la eficiencia de los productos locales.

### Evitar la construcción de jacobianos

El algoritmo necesita el producto de cada jacobiano transpuesto por $\delta_k$, no todas las entradas de ese jacobiano. Las reglas de derivación de cada operación pueden calcular directamente ese producto. Por eso la notación matricial de la demostración no obliga a almacenar matrices jacobianas completas en una implementación.

## Ejemplo escalar con dos capas

Para comprobar las identidades, consideremos el ejemplo

$$
a_0=x,\qquad a_1=f_1(a_0;\theta_1)=\theta_1a_0,
\qquad a_2=f_2(a_1;\theta_2)=\theta_2a_1,
$$

con $\ell(a_2)=\tfrac12a_2^2$. Entonces

$$
\mathcal L(\theta)=\frac12(\theta_2\theta_1x)^2.
$$

En la última capa,

$$
\delta_2=\ell'(a_2)=a_2,
\qquad
\frac{\partial\mathcal L}{\partial\theta_2}
=a_1\delta_2=a_1a_2.
$$

Propagamos hacia la primera capa:

$$
J_2=\theta_2,\qquad
\delta_1=\theta_2\delta_2=\theta_2a_2,
$$

$$
\frac{\partial\mathcal L}{\partial\theta_1}
=a_0\delta_1=x\theta_2a_2.
$$

Finalmente,

$$
\delta_0=J_1\delta_1=\theta_1\theta_2a_2.
$$

Si $x=2$, $\theta_1=3$ y $\theta_2=4$, obtenemos

$$
a_1=6,\qquad a_2=24,\qquad\mathcal L=288,
$$

$$
\frac{\partial\mathcal L}{\partial\theta_2}=144,
\qquad
\frac{\partial\mathcal L}{\partial\theta_1}=192,
\qquad
\frac{\partial\mathcal L}{\partial x}=288.
$$

Se obtienen los mismos valores derivando directamente $\tfrac12(\theta_2\theta_1x)^2$ respecto a cada variable.

## Lectura del resultado

$\delta_k$ mide cómo cambia la pérdida cuando cambia la activación de la capa $k$. Para llevar esa sensibilidad a la capa anterior, se aplica $J_k^T$. Para llevarla a los parámetros de la capa actual, se aplica $(\partial f_k/\partial\theta_k)^T$. Ambos cálculos utilizan la misma sensibilidad que llega desde la salida.

El enunciado supone funciones $C^1$. La función ReLU no cumple esa hipótesis en cero: fuera de ese punto puede aplicarse el cálculo usual, y en cero una implementación adopta una regla de derivación que debe especificarse. Esta salvedad no altera la demostración bajo las hipótesis formuladas.

