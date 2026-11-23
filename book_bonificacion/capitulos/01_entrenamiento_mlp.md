# Entrenamiento de MLP

## Enunciado de la sección 2.7

Realizar cinco iteraciones de descenso de gradiente sobre la red del ejemplo de clase, con tasa de aprendizaje $\alpha=0.01$ y función de pérdida

$$
L=(y-\hat y)^2.
$$

Se deben calcular los valores intermedios $p_1,\ldots,p_{10}$, la predicción, la pérdida, los seis gradientes y la actualización de $\omega_1,\ldots,\omega_6$. El ejercicio pide registrar los resultados, graficar la pérdida, explicar el tamaño de los cambios y estudiar una tolerancia de $10^{-6}$.

**Datos que faltan en la captura.** El documento no incluye el ejemplo previo: no aparecen las ecuaciones que definen $p_1,\ldots,p_{10}$, las entradas, el valor de $y$, las activaciones ni los seis pesos iniciales. Por tanto, no existe una tabla numérica única determinada por este archivo. La captura original puede consultarse en el {ref}`anexo <mlp-original>`.

## Derivada de la pérdida

El valor observado $y$ se mantiene fijo al derivar con respecto a los pesos. Aplicando la regla de la cadena,

$$
\begin{aligned}
\frac{\partial L}{\partial\omega_j}
&=\frac{\partial (y-\hat y)^2}{\partial\hat y}
  \frac{\partial\hat y}{\partial\omega_j}\\
&=2(y-\hat y)(-1)\frac{\partial\hat y}{\partial\omega_j}\\
&=2(\hat y-y)\frac{\partial\hat y}{\partial\omega_j},
\qquad j=1,\ldots,6.
\end{aligned}
$$

El factor $2$ debe conservarse, porque el ejercicio define $L=(y-\hat y)^2$ sin el factor $1/2$. La derivada $\partial\hat y/\partial\omega_j$ se obtiene recorriendo las dependencias reales de $p_1,\ldots,p_{10}$. Si un peso influye en la salida por varias rutas, deben sumarse las contribuciones de todas ellas.

## Cinco actualizaciones

La fila $t=0$ representa el estado inicial. Las cinco actualizaciones llevan sucesivamente a las filas $t=1,2,3,4,5$. Para cada $t=0,\ldots,4$, se realiza lo siguiente.

### Paso 1 Calcular la propagación hacia adelante

Con los seis pesos de la fila $t$, se evalúan $p_1^{(t)},\ldots,p_{10}^{(t)}$ en el orden de dependencias de la red. De este cálculo resulta $\hat y^{(t)}$.

### Paso 2 Calcular la pérdida de la misma fila

$$
L^{(t)}=\left(y-\hat y^{(t)}\right)^2.
$$

### Paso 3 Calcular los seis gradientes

$$
\left.\frac{\partial L}{\partial\omega_j}\right|_{\omega=\omega^{(t)}}
=2\left(\hat y^{(t)}-y\right)
\left.\frac{\partial\hat y}{\partial\omega_j}\right|_{\omega=\omega^{(t)}},
\qquad j=1,\ldots,6.
$$

Todos los gradientes se calculan con los pesos anteriores a la actualización. Usar un peso recién actualizado para calcular otro gradiente produciría un algoritmo distinto.

### Paso 4 Actualizar simultáneamente los pesos

```{math}
:label: eq-mlp-update
\omega_j^{(t+1)}
=\omega_j^{(t)}
-0.01\left.\frac{\partial L}{\partial\omega_j}\right|_{\omega=\omega^{(t)}},
\qquad j=1,\ldots,6.
```

Equivalentemente,

$$
\omega_j^{(t+1)}
=\omega_j^{(t)}-0.02\left(\hat y^{(t)}-y\right)
\left.\frac{\partial\hat y}{\partial\omega_j}\right|_{\omega=\omega^{(t)}}.
$$

### Paso 5 Evaluar el nuevo estado

Después de actualizar los seis pesos se vuelve a hacer la propagación hacia adelante. Así, $\hat y^{(t+1)}$ y $L^{(t+1)}$ corresponden a los pesos de la fila $t+1$.

No se debe registrar en una misma fila la pérdida anterior y los pesos nuevos. Aunque el ejercicio pide al menos cuatro cifras decimales, es preferible conservar toda la precisión durante los cálculos y redondear únicamente al presentar la tabla.

## Tabla de resultados

Las celdas quedan pendientes de los datos del ejemplo de clase; el guion no representa un cero.

| $t$ | $\omega_1$ | $\omega_2$ | $\omega_3$ | $\omega_4$ | $\omega_5$ | $\omega_6$ | $\hat y^{(t)}$ | $L^{(t)}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | — | — | — | — | — | — | — | — |
| 1 | — | — | — | — | — | — | — | — |
| 2 | — | — | — | — | — | — | — | — |
| 3 | — | — | — | — | — | — | — | — |
| 4 | — | — | — | — | — | — | — | — |
| 5 | — | — | — | — | — | — | — | — |

El código {download}`mlp_cinco_iteraciones.py <../codigo/mlp_cinco_iteraciones.py>` implementa el registro y las actualizaciones simultáneas. Recibe una función que devuelve la predicción, sus seis derivadas respecto de los pesos y los diez valores intermedios del ejemplo original. No asigna una red ni pesos ficticios.

## Respuestas a las preguntas

### ¿Disminuye la pérdida en cada iteración?

Se debe verificar comparando $L^{(t+1)}$ con $L^{(t)}$. La dirección del gradiente negativo es una dirección local de descenso cuando el gradiente no es cero. Un paso fijo de tamaño $0.01$ no garantiza por sí solo una disminución en cualquier red y en cualquier punto.

Para una pérdida diferenciable con gradiente localmente Lipschitz, los pasos suficientemente pequeños producen descenso. En puntos donde intervienen activaciones no diferenciables se necesitan las precisiones correspondientes. Sin las funciones del ejemplo no es posible decidir numéricamente qué ocurre en estas cinco iteraciones.

### ¿Por qué los cambios pueden ser pequeños con $\alpha=0.01$?

Por la regla de actualización,

$$
\left|\omega_j^{(t+1)}-\omega_j^{(t)}\right|
=0.01\left|\left.\frac{\partial L}{\partial\omega_j}\right|_{\omega=\omega^{(t)}}\right|.
$$

El tamaño del cambio depende tanto de $0.01$ como de la magnitud del gradiente. Si hay activaciones sigmoideas saturadas, sus derivadas pueden reducir aún más el gradiente. Un $\alpha$ mayor aumenta el desplazamiento para un mismo gradiente, pero también puede provocar oscilaciones o crecimiento de la pérdida. El [capítulo sobre tamaño de paso](03_tamano_paso.md) muestra esta relación de forma exacta en el caso cuadrático.

### ¿Cuántas iteraciones adicionales hacen falta para una tolerancia de $10^{-6}$?

La interpretación literal de «la pérdida cambie menos que una tolerancia» corresponde al criterio

$$
\left|L^{(t+1)}-L^{(t)}\right|<10^{-6}.
$$

No equivale a exigir $L^{(t)}<10^{-6}$ ni a exigir un gradiente menor que $10^{-6}$. El número de iteraciones adicionales depende de la red y de su trayectoria; no puede deducirse solamente de $\alpha=0.01$ o de la existencia de cinco iteraciones iniciales.

Una vez disponibles los datos, se continúa desde la fila $5$, se compara la pérdida entre estados consecutivos y se cuenta el número de actualizaciones adicionales hasta satisfacer el criterio. También se fija un máximo de iteraciones y se comprueba que los valores sean finitos. Un cambio pequeño de pérdida puede indicar estancamiento y, por sí solo, no certifica que se haya alcanzado un mínimo global.
