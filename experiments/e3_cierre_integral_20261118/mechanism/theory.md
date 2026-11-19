# Formulación y propiedades delimitadas de V3

## Identidad y mecanismo

Sean r1,…,r6 las seis rentabilidades conocidas de una ventana de siete cierres, m su media, u la rentabilidad futura y U=|u−m|. Para la desviación estándar **poblacional** de las siete rentabilidades:

```text
A = (1/7) sum_{i=1}^6 (r_i-m)^2
c = 6/49
σ7 = sqrt(A+c U²)
```

Para comprobarla, defínanse d=u−m y S=sum_i(r_i−m)². La media conjunta es m+d/7. Como sum_i(r_i−m)=0:

```text
sum_{i=1}^6 (r_i-m-d/7)² = S+6d²/49
(u-m-d/7)² = 36d²/49
σ7² = [S+(42/49)d²]/7 = S/7+(6/49)d².
```

Tomar la raíz no negativa demuestra la identidad. No se presenta como una identidad matemática nueva. A se calcula con historia; U y σ7 son etiquetas de entrenamiento, nunca entradas futuras.

Para el contexto h de la ventana, una lectura produce tres incrementos z1,z2,z3 y dos logits l1,l2. Con escala de ajuste sU>0, ε=10^-4:

```text
a_k = sU sum_{j=1}^k [ε+softplus(z_j)]
p = softmax(l1,l2,0)
G_h = sum_{k=1}^3 p_k δ_{a_k}
μ_G(A) = sum_k p_k sqrt(A+c a_k²)
```

La referencia de logit cero elimina una traslación redundante de los tres logits. En `fixed_probs`, p=(1/3,1/3,1/3) y sólo se programan las tres lecturas de incrementos. Los átomos de U tienen unidades de rentabilidad, no de σ7. No hay densidad continua, likelihood Student o cuadratura numérica.

## Proposición 1: soporte y esperanza finita

Con parámetros finitos, A≥0 y sU>0, los incrementos son positivos, los átomos están estrictamente ordenados en aritmética real y p es una ley de probabilidades. Cada `sqrt(A+c a_k²)` es mayor que sqrt(A), luego μG(A)>sqrt(A). La implementación protege el último bit mediante un máximo con sqrt(A); no se promete separación estricta representable para cualquier escala extrema de punto flotante.

La suma calcula exactamente E_G[σ7] para esa ley finita. Generalmente no coincide con `sqrt(A+c E_G[U²])`: la concavidad de la raíz da μG(A)≤sqrt(A+c E_G[U²]), con desigualdad estricta para átomos distintos de masa positiva. No se hereda por ello exactitud respecto de la ley real de U.

## Proposición 2: sensibilidad parcial con la ley fija

Si se mantienen átomos y masas fijos, entonces, para A≥0 con denominadores positivos:

```text
∂μG/∂A = (1/2) sum_k p_k / sqrt(A+c a_k²) > 0
∂²μG/∂A² = -(1/4) sum_k p_k / (A+c a_k²)^(3/2) < 0
∂μG/∂a_k = c p_k a_k / sqrt(A+c a_k²) <= sqrt(c) p_k
```

Así, el pronóstico es creciente y cóncavo respecto del piso de varianza **si la ley permanece fija**. Al cambiar la historia real cambian simultáneamente A y G_h: no es una garantía de monotonía respecto de OHLCV ni una cota global del codificador. La implementación con `hypot(sqrt(A),sqrt(c)*a)` estabiliza magnitudes; no se reclama diferenciabilidad automática respecto de una entrada A con gradiente en A=0.

Para el incremento z_j, la regla de la cadena y sigmoid(z_j)≤1 dan |∂μG/∂z_j|≤sU sqrt(c) sum_{k≥j}p_k≤sU sqrt(c). Esto acota una derivada parcial de la cabeza, no los gradientes de todos los parámetros del sistema.

## Proposición 3: riesgo poblacional bajo especificación correcta

El CRPS en orientación de pérdida es

```text
CRPS(G,U) = sum_k p_k |a_k-U|
            - (1/2) sum_{j,k} p_j p_k |a_j-a_k|.
L = [(μG(A)-σ7)/sY]^2 + λ CRPS(G/sU,U/sU), λ>0.
```

La representación y propiedad estricta son conocidas: Gneiting y Raftery (2007), §4.2, ecuación (21) y párrafo posterior, orientación de pérdida. Se usa sólo una normalización del CRPS: `CRPS(G/sU,U/sU)=CRPS(G,U)/sU`.

Condicionado a una historia x, supóngase que la verdadera ley F0 de U tiene segundo momento finito y pertenece exactamente a la clase de leyes representable, y que puede alcanzarse el mínimo global poblacional. Con A=A(x) fijo, la descomposición de riesgo cuadrático da

```text
R_MSE(G)-R_MSE(F0) = [μG(A)-μF0(A)]² / sY².
E CRPS(G,U)-E CRPS(F0,U) = integral [G(t)-F0(t)]² dt.
```

Ambos sumandos son no negativos y el CRPS es estrictamente propio para leyes de primer momento finito. Por tanto F0 minimiza el riesgo conjunto y su esperanza transformada es E[σ7|x]. Este resultado identifica una ley en el mínimo poblacional; **no garantiza convergencia del optimizador, consistencia estadística sin condiciones de estimación adicionales, pesos únicos o superioridad de muestra finita**.

La hipótesis de pertenencia es restrictiva: una ley continua real no pertenece a una familia de sólo tres átomos. Con K fijo, red de capacidad finita o restricciones de entrenamiento, la ley óptima es una aproximación y la pérdida auxiliar puede cambiar el mejor pronóstico puntual. No se reclama consistencia universal, calibración o mejora de RMSE en mercados.

## Procedencia y contribución por contrastar

Leyes discretas de salida, redes de mezcla, probabilidades softmax y CRPS son métodos establecidos. Bishop (1994) ya propuso que una red predijera una distribución condicional mediante un modelo de mezcla; V3 no reclama esa idea como propia ni implementa la densidad continua de aquella familia. Bellemare, Dabney y Munos (2017) es un antecedente de aproximaciones distribucionales categóricas, en aprendizaje por refuerzo; V3 no implementa su actualización Bellman ni su proyección C51. La transferencia candidata consiste en aprender posiciones y masas de U, empujar la ley por la identidad específica del objetivo solapado y elicitarla con CRPS además de la media puntual. Su carácter sustantivo y su novedad para el problema necesitan todavía literatura y controles; las proposiciones no los certifican.

Fuentes primarias consultadas: [Gneiting y Raftery, JASA 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf), §4.2 p.367, ecuación (21), incluida su orientación negativa; [Bishop, NCRG/94/004, 1994](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/), resumen primario y metadatos; [Bellemare et al., ICML 2017](https://proceedings.mlr.press/v70/bellemare17a.html), resumen primario y metadatos. De los últimos dos no se afirma una lectura completa del cuerpo en este corte. El PDF de Bishop se abrió, pero su extracción textual resultó ilegible; la atribución limitada procede del resumen alojado por la institución del autor.
