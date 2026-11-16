# Revisión matemática: qué añade la región de momentos

**16 de noviembre de 2026. Inferencias propias a partir de la formulación conocida; sin datos de mercado.** Se conserva el teorema de consistencia condicional del [primer corte](../e3_mecanismo_momentos_20261116/theory.md), con sus supuestos. Los ejemplos siguientes delimitan afirmaciones que ese teorema no contiene.

## 1. Región de reportes frente a clase de funciones neuronales

Para `A>=0,c>0`, sean

\[
\mathcal D_A=\{(\mu,m,q):\mu>\sqrt A,\ m>0,\ q\ge m^2\},
\]

\[
\mathcal C_A=\{(\mu,m,q)\in\mathcal D_A:
L(A,m,q)\le\mu<H(A,m)\ \text{si }q>m^2;\quad
\mu=H=L\ \text{si }q=m^2\}.
\]

La cabeza libre admite reportes en `D_A`; la propuesta sólo en `C_A`, un subconjunto propio. No son el mismo espacio de reportes conjuntos. Sin embargo, para cualquier `mu>sqrt(A)` se puede elegir

\[
m=(\mu^2-A)/c>0,\quad q=m^2,\quad b=0,
\]

de modo que `L=H=mu`. Ambas regiones proyectan sobre la misma semirrecta de medias. Por tanto, **con lecturas que sean funciones arbitrarias**, la propuesta no añade medias puntuales que una cabeza positiva libre no pueda reportar. Su diferencia reside en el vínculo con los momentos reportados.

Esto no establece inclusión o igualdad entre dos clases de funciones `X -> mu` implementadas por redes finitas de tres lecturas afines. Las composiciones no son convertibles necesariamente con los mismos pesos ni capacidad. Tampoco compartir clase puntual u óptimo Bayes prueba falta de novedad; descarta únicamente una justificación de mayor expresividad puntual en el límite funcional.

Si los momentos y la media verdaderos son representables y cumplen los supuestos, el triplete verdadero pertenece a `C_A`. Tanto la cabeza restringida como la libre minimizan el score conjunto allí. La consistencia del funcional no es una ventaja Bayes exclusiva. En una clase finita, la restricción y los scores sí pueden cambiar estimación, gradientes y error; su efecto debe contrastarse.

## 2. Momentos incorrectos: una restricción que perjudica la media

Sea `A=0`, `V=1` casi seguramente y `c=6/49`. Entonces `Y=mu0=sqrt(c)`, `m0=q0=1`. Fijemos reportes auxiliares válidos pero incorrectos `m=4,q=32`.

\[
L=\sqrt c\,m^{3/2}/\sqrt q=\sqrt{2c},\qquad H=2\sqrt c.
\]

Toda media de la envolvente satisface `mu>=sqrt(2c)>mu0`; su MSE mínima es

\[
c(\sqrt2-1)^2>0.
\]

Una cabeza libre puede reportar `mu=mu0` con **los mismos `m,q` y los mismos scores auxiliares**, obteniendo MSE cero en esta ley determinista. Aísla la posible desventaja de la guarda con momentos mal estimados. No demuestra que la propuesta aprenda esos valores en BTC, que su mínimo global sea incorrecto o que una cabeza libre siempre gane. Tampoco contradice el teorema, cuyo óptimo verdadero es representable en otra parte de la región.

## 3. Auxiliares y fidelidad: un paso escalar que no coincide

Se mantiene esa ley determinista. Escalas propuestas `s_u=1,s_y=sqrt(c)`, con `m=softplus(a)`. Fijado el contexto, se consideran sólo las tres coordenadas de lectura `(a,b,d)`. Tomemos

\[
a=\log(e-1),\quad m=1,\quad b=\sqrt8,\quad q=9,\quad d=1,
\quad\rho=1/2,\quad\mu=2\sqrt c/3.
\]

Con `g=1-exp(-1)` y pesos auxiliares iguales `lambda>0`, las derivadas en ese punto son

\[
\nabla\mu=\sqrt c\,(g/3,-\sqrt8/54,1/3),\qquad
\nabla\mathcal L_{aux}=\lambda(16g/9,16\sqrt8/81,0).
\]

Su producto es

\[
D=\nabla\mu\cdot\nabla\mathcal L_{aux}
=\lambda\sqrt c\left(\frac{16g^2}{27}-\frac{64}{2187}\right)>0.
\]

Desde el mismo punto, con SGD de paso pequeño `eta`,

\[
\mu_{conjunto}-\mu_{MSE}=-\eta D+O(\eta^2).
\]

Ambas medias permanecen por debajo de la verdadera para pasos suficientemente pequeños; la conjunta queda más lejos. Con `lambda=0,1` y `eta=1e-5`, la MSE normalizada después del paso es aproximadamente `0,1111105461` frente a `0,1111104078` con MSE sola. El [JSON reproducible](counterexamples.json) informa los pasos prefijados `1e-6,1e-5,1e-4`, las derivadas y tolerancias.

Es un ejemplo **algebraico de un paso SGD en tres coordenadas**, con codificador fijo, no el entrenamiento de una red ni la trayectoria del Adam que se pudiera elegir después. Refuta heredar una garantía general de igualdad de actualizaciones o de no perjudicar la media. No prueba fracaso de la CNN-LSTM en BTC ni contradice la consistencia de riesgo poblacional bajo clase rica.

## 4. Comparador correcto y decisión

Faithful define su comparador de media sola por todos los ancestros de la media. En nuestra salida `m,q` son ancestros. Por ello el control relevante para el efecto auxiliar es **esta misma cabeza con `lambda_m=lambda_q=0`**, no únicamente una cabeza directa con igual conteo de parámetros. Detener el gradiente al codificador deja que los scores cambien los readouts `m,q` y, por ellos, la media. La [comparación primaria](primary_comparison.md) distingue esa inferencia de la garantía original de Faithful.

La modificación verificable es una restricción conjunta y su sesgo inductivo; no una nueva clase puntual, una nueva elicitación o una garantía general de MSE. Su sustantividad permanece sin acreditar. Este corte justifica **no seleccionarla todavía como aporte original ni pasar a confirmación**, preservando la propuesta y sus pruebas para revisión externa.
