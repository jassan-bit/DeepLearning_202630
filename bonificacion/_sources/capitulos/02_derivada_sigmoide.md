# Derivada de la función sigmoidea

## Enunciado

La sección **2.9. Cálculo de $\delta^r_{nj}$** del material solicita verificar que, para la función sigmoidea logística,

```{math}
:label: eq-sigmoide-derivada
f'(z)=a f(z)\bigl(1-f(z)\bigr).
```

Consideramos la sigmoide con parámetro constante $a$,

$$
f(z)=\frac{1}{1+e^{-az}},\qquad z\in\mathbb{R}.
$$

La elección habitual $a>0$ produce una función creciente. La identidad de la derivada también es válida si $a=0$ o $a<0$.

## Demostración paso a paso

### Paso 1 Escribir el cociente como una potencia

La función puede escribirse como

$$
f(z)=\left(1+e^{-az}\right)^{-1}.
$$

Esta forma permite aplicar la regla de la cadena: la función exterior eleva a la potencia $-1$, y la interior es $1+e^{-az}$.

### Paso 2 Derivar la función exterior y la interior

Por la regla de la cadena,

$$
f'(z)
=-\left(1+e^{-az}\right)^{-2}
\frac{d}{dz}\left(1+e^{-az}\right).
$$

Como $a$ es constante,

$$
\frac{d}{dz}\left(1+e^{-az}\right)
=e^{-az}\frac{d}{dz}(-az)
=-a e^{-az}.
$$

Al sustituir y simplificar los signos,

$$
\begin{aligned}
f'(z)
&=-\left(1+e^{-az}\right)^{-2}\left(-a e^{-az}\right)\\
&=a e^{-az}\left(1+e^{-az}\right)^{-2}\\
&=\frac{a e^{-az}}{\left(1+e^{-az}\right)^2}.
\end{aligned}
$$

### Paso 3 Expresar el resultado mediante la propia función

De la definición de $f$,

$$
\begin{aligned}
1-f(z)
&=1-\frac{1}{1+e^{-az}}\\
&=\frac{1+e^{-az}-1}{1+e^{-az}}\\
&=\frac{e^{-az}}{1+e^{-az}}.
\end{aligned}
$$

Por tanto,

$$
\begin{aligned}
a f(z)\bigl(1-f(z)\bigr)
&=a\left(\frac{1}{1+e^{-az}}\right)
\left(\frac{e^{-az}}{1+e^{-az}}\right)\\
&=\frac{a e^{-az}}{\left(1+e^{-az}\right)^2}\\
&=f'(z).
\end{aligned}
$$

Con esto queda demostrada la identidad {eq}`eq-sigmoide-derivada`.

## Comprobación en un punto

Si $a=1$ y $z=0$, entonces

$$
f(0)=\frac{1}{1+e^0}=\frac12.
$$

La fórmula demostrada da

$$
f'(0)=1\cdot\frac12\left(1-\frac12\right)=\frac14.
$$

La expresión obtenida antes de factorizar produce el mismo resultado:

$$
\frac{e^0}{(1+e^0)^2}=\frac{1}{4}.
$$

## Relación con la retropropagación

En una neurona con salida $f(z)$, si una pérdida $\mathcal L$ depende de $z$ a través de esa salida, la regla de la cadena da

$$
\frac{\partial\mathcal L}{\partial z}
=\frac{\partial\mathcal L}{\partial f(z)}\,f'(z)
=\frac{\partial\mathcal L}{\partial f(z)}\,
a f(z)\bigl(1-f(z)\bigr).
$$

El gradiente que llega desde las operaciones posteriores se multiplica por la derivada local de la activación. Si $f(z)$ ya se calculó en el paso hacia adelante, este factor se obtiene directamente a partir de ese valor.

El recorte original anuncia el cálculo de $\delta^r_{nj}$, pero no incluye la definición de sus índices ni de esa cantidad. La demostración anterior establece el factor local solicitado; la regla general que lo incorpora se desarrolla en el capítulo de [retropropagación](05_retropropagacion.md).

## Consecuencia sobre la magnitud de la derivada

Como $0<f(z)<1$ y

$$
f(z)\bigl(1-f(z)\bigr)
=\frac14-\left(f(z)-\frac12\right)^2,
$$

se obtiene

$$
|f'(z)|\leq\frac{|a|}{4}.
$$

Para $a>0$, la derivada alcanza $a/4$ en $z=0$ y tiende a cero cuando $z$ tiende a cualquiera de los dos extremos infinitos. Así, una activación muy cercana a $0$ o a $1$ aporta un factor local pequeño al producto de derivadas de la retropropagación. Esta observación describe ese factor; la magnitud del gradiente completo también depende de los pesos y de las demás operaciones de la red.


## Comprobación gráfica

```{figure} ../figuras/sigmoide.png
:alt: La sigmoide logística y su derivada para a = 0.5, 1 y 2.
:width: 900px

La sigmoide logística y su derivada para $a=0.5$, $a=1$ y $a=2$. El código y las comprobaciones se encuentran en el [cuaderno](../notebooks/comprobaciones.ipynb).
```
