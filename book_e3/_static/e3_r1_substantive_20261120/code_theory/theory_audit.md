# Auditoría del mecanismo y de la propiedad formal de V3

Revisión documental y algebraica asistida por IA, 20 de noviembre de 2026
(America/Bogota). Corte de lectura: `29cedb5228ebcecce2a354a1d0ab89adf3936f12`;
fuente efectivamente fijada por el piloto: `8f65eac34aaf49293924ae8328c235cedbdd2866`.
No se entrenó, ejecutó un modelo, abrió OHLCV ni recalculó el piloto.
Este documento evalúa correspondencia entre ecuaciones y código; no sustituye
el contraste bibliográfico o el juicio académico sobre novedad para el problema.

## Objeto causal y definición exacta de la etiqueta

Al origen se observan siete cierres y por tanto seis retornos logarítmicos
`r1,...,r6`. Sean `m6=(r1+...+r6)/6`, `r7` el retorno aún desconocido y

```text
S6 = sum_{i=1}^6 (ri-m6)^2
A = S6/7
U = |r7-m6|
c = 6/49
Y = sigma7 = sqrt(A+c U^2).
```

La etiqueta es la desviación estándar **poblacional, diaria y sin anualizar** de
los siete retornos: `ddof=0`. Si `d=r7-m6`, la nueva media es `m7=m6+d/7`;
la suma de cuadrados pasa de `S6` a `S6+(6/7)d^2`, que se divide por siete.
La identidad es conocida; aquí determina una adaptación específica a una
ventana solapada con exactamente un retorno futuro. Con `ddof=1` no servirían
estos mismos coeficientes: la desviación muestral sería `sqrt(7/6)*Y`.

El cargador calcula los seis retornos con los cierres raw, y `A` antes de usar
la etiqueta. Construye `U=sqrt(max(Y^2-A,0)/c)` **sólo como etiqueta**. Rechaza
márgenes inferiores a `-1e-12`; la truncación de márgenes negativos tolerados
se registra como redondeo. `U` no aparece en la firma de `forward(values,A)`.
Este procedimiento no obtiene el signo de `r7-m6` ni una observación futura
independiente: `U` es una transformación de la etiqueta ya disponible.

Fuente: [cargador, líneas 85–168](https://github.com/jassan-bit/DeepLearning_202630/blob/8f65eac34aaf49293924ae8328c235cedbdd2866/src/e3_innovation_model.py#L85).
Sólo se indexan `training_raw`, `training_y_original` y `training_dates`;
los escaladores y las escalas `sY` y `sU` se estiman con el ajuste interno.
Las fechas ya expuestas conservan su carácter de desarrollo.

## Transformación implementada y parámetros

Se reutiliza el codificador M00: convolución causal y LSTM unidireccional,
con atención de contenido. M00 retira la penalización de discordancia y la
saturación radial del primer mecanismo; **no se hereda su certificado latente**.
Con estados `hi`, `q=qraw/sqrt(1+||qraw||²)` y `tau=.25+softplus(tauraw)`,
`alpha_i=softmax_i(hi·q/tau)` y `h=sum_i alpha_i hi`.

V3 produce tres preactivaciones `zj` y dos logits libres `ell1,ell2`:

```text
Delta_j = sU [1e-4+softplus(zj)]
a_k = sum_{j<=k} Delta_j
p = softmax(ell1,ell2,0)
G_theta(.|X) = sum_{k=1}^3 p_k delta_{a_k}
T_A(u) = sqrt(A+c u^2)
mu_theta(X,A) = sum_k p_k T_A(a_k).
```

Se aprende la ley de una **magnitud no negativa**, no la densidad del retorno
firmado. El cero de referencia elimina la traslación redundante de logits.
En aritmética real, los átomos son positivos y estrictamente ordenados y las
masas son positivas y suman uno. El incremento mínimo `1e-4*sU` impide
representar exactamente un átomo cero, átomos coincidentes o saltos menores;
softmax con logits finitos tampoco da masas exactamente nulas.
En coma flotante, el código comprueba finitud y protege el piso con `maximum`;
no promete separación estricta representable a cualquier escala extrema.

La esperanza es exacta **para la ley finita aprendida**: tres términos, sin
cuadratura. No es una esperanza exacta para la distribución real del mercado.
El codificador y atención usan float32 y la lectura atómica usa float64.

Conteo estático, sin construir redes: convolución `8*(5*3+1)=128`; LSTM
`4*12*(8+12+2)=1056`; consulta y temperatura `12+1=13`: **1197** compartidos.
La lectura atómica tiene `3*(12+1)=39`, y los logits libres `2*(12+1)=26`:
**1262** en V3 completa y sin CRPS; **1236** con masas uniformes. La aditiva
tiene los mismos 1197 más 13 de lectura: **1210**. No se simula igualdad de
capacidad mediante parámetros inactivos.

Fuente: [V3, líneas 83–184](https://github.com/jassan-bit/DeepLearning_202630/blob/8f65eac34aaf49293924ae8328c235cedbdd2866/src/e3_overlap_model_v3.py#L83).

## Pérdida, controles y diferencia de entrenamiento

```text
CRPS(G,u) = sum_k p_k |a_k-u| - .5 sum_jk p_j p_k |a_j-a_k|
L = mean[((mu-Y)/sY)^2] + lambda mean[CRPS(G/sU,U/sU)].
```

`lambda=.1` estaba fijado en el plan. La normalización del CRPS se efectúa una
sola vez: `CRPS(G/sU,U/sU)=CRPS(G,U)/sU`. Es CRPS de `U`, **no CRPS de sigma7**.
Para `A>0`, puntuar directamente la ley transformada de sigma7 introduciría el
peso `T'_A(u)=c*u/sqrt(A+c*u²)` en la representación integral del CRPS; no es
en general la misma pérdida. Este cambio de representación/objetivo no añade
datos nuevos, y no acredita calibración observada de sigma7.

La aditiva predice `sqrt(A)+sY*[1e-4+softplus(w_eff·h+b)]` y optimiza sólo MSE.
V3 usa dos lecturas y propaga gradientes tanto desde la media transformada
como desde CRPS a átomos, masas y codificador. No es la misma cabeza escalar
con otro nombre, otro activo o un nuevo valor de un hiperparámetro.
Sin embargo, sus operadores individuales y la idea de estimar una ley
condicional seguida de una transformación son conocidos: la diferencia de
código por sí sola no demuestra prioridad para este problema.

Controles efectivamente implementados y entrenados en el piloto conservado:

| Control | Cambio | Qué no permite concluir |
|---|---|---|
| `atom_mse_only` | Mismos 1262 parámetros e inicio; peso CRPS cero. | Una mejor RMSE no identifica una ley calibrada ni promociona la receta. |
| `fixed_probs` | `p=(1/3,1/3,1/3)`, sólo 1236 parámetros; CRPS permanece. | No es un control perfectamente igualado en capacidad. |
| `atom_plugin` | Mismos átomos/masas e inicio; `b=sqrt(A+c*sum p_k a_k²)`; CRPS permanece. | Después de entrenar, sus leyes pueden ser diferentes; el contraste empírico no mantiene una misma ley fija. |
| Aditiva | Cabeza escalar con soporte, 1210 parámetros, MSE. | La identidad de soporte no implica que V3 sea más expresiva en predicción puntual. |

El motor usa la misma MSE en unidades físicas para validación/selección;
CRPS auxiliar sólo interviene en las pérdidas de ajuste correspondientes.
Fuente: [motor, líneas 258–301](https://github.com/jassan-bit/DeepLearning_202630/blob/8f65eac34aaf49293924ae8328c235cedbdd2866/experiments/e3_v3_development_pilot_20261120/pilot_engine.py#L258),
[control de segundo momento, líneas 58–87](https://github.com/jassan-bit/DeepLearning_202630/blob/8f65eac34aaf49293924ae8328c235cedbdd2866/experiments/e3_v3_development_pilot_20261120/plugin_control.py#L58).

## Propiedades válidas y alcance

1. **Soporte y esperanza.** Para una ley válida, `mu>=sqrt(A)`; en reales con
   átomos positivos la desigualdad es estricta. No prueba calibración o mejora.
2. **Derivadas parciales con la ley fija.** `dmu/dA=.5 sum p_k/T_A(a_k)>0` y
   `d²mu/dA²=-.25 sum p_k/T_A(a_k)^3<0`. También
   `dmu/da_k=c*p_k*a_k/T_A(a_k)<=sqrt(c)*p_k`; por incrementos acumulados,
   `|dmu/dz_j|<=sU*sqrt(c)*sum_{k>=j}p_k<=sU*sqrt(c)`.
   Si cambia X cambian simultáneamente A, átomos y masas: no hay monotonicidad
   global de OHLCV ni cota de todos los gradientes del sistema. La composición
   `sqrt(A)` no reclama autodiferenciación respecto de una A externa en cero.
3. **Jensen frente al control definido.** Con `V=U²` y ley fija,
   `b=sqrt(A+c E_G V)>=mu_G`. Si `A>0` y soporte finito
   `[vmin,vmax]`, Taylor y curvatura dan
   `c² Var(V)/(8(A+c vmax)^(3/2)) <= b-mu_G <= c² Var(V)/(8(A+c vmin)^(3/2))`.
   Es una propiedad estándar aplicada a esta cabeza, no un teorema nuevo.
4. **Riesgo cuadrático oracle.** Sólo si G es la ley verdadera condicional,
   `R(b)-R(mu_G)=(b-mu_G)²>=0`. No compara con la aditiva aprendida ni
   garantiza una ventaja bajo mala especificación.
5. **Mínimo poblacional conjunto.** Si la verdadera ley de U está exactamente
   en la clase representable, tiene segundo momento finito, `lambda>0` y se
   alcanza el mínimo global poblacional, los excesos de riesgo son
   `(mu_G-mu_F0)²/sY² + (lambda/sU)*integral(G-F0)²`.
   Ambos son no negativos. Se identifica la ley verdadera mediante la regla
   estrictamente propia conocida; no se identifican pesos neuronales únicos.
   Una ley continua no pertenece a esta familia de tres átomos. El enunciado
   no da convergencia, consistencia universal ni superioridad de muestra finita.

No hace falta que una propiedad exigida por la guía sea un teorema inédito.
Tampoco demostrarla acredita automáticamente R1. La complejidad adicional de
la lectura es `O(K*d)`, media `O(K)` y CRPS enumerado `O(K²)`; con K=3 es
pequeña, pero no sustituye mediciones completas de FLOPs/energía.

## Verificaciones previas y contraejemplos que deben conservarse

El recibo antiguo contiene 100 leyes artificiales y no datos de mercado.
Comprobó las cotas y la identidad de riesgo, dos leyes con el mismo segundo
momento y distinta media transformada, un punto representable por la aditiva
y una inversión del orden de riesgo al añadir CRPS bajo aproximación.
Esta auditoría verifica hashes y la coherencia aritmética de los valores ya
guardados; no genera otras leyes, predicciones o remuestreos.

El punto aditivo del contraejemplo es representable si su exceso es mayor que
`1e-4*sY`. No debe expandirse ese ejemplo a una equivalencia de todas las
funciones de ambas arquitecturas, ni a una imposibilidad general de la aditiva.

## Dictamen técnico y declaración defendible

**Diferencia implementada: acreditada documentalmente.** Se cambia de una
lectura escalar de exceso a una ley latente de magnitud con átomos y masas
aprendidos, su transformación por la identidad de la etiqueta solapada y una
pérdida auxiliar de distribución. Es una modificación de estructura y
entrenamiento verificable, no sólo un cambio de aplicación.

**Propiedad formal delimitada: correcta bajo los supuestos explícitos.** Las
pruebas existentes verifican ejemplos numéricos y software; no son una prueba
experimental de novedad o rendimiento.

**Novedad para el problema: requiere dictamen bibliográfico independiente.**
La objeción más fuerte sigue siendo la composición directa de distribución
condicional, transformación determinista y scoring conocido. Para responder
debe contrastarse la cadena exacta (magnitud faltante, ventana solapada,
átomos aprendidos, pushforward y doble objetivo) con los antecedentes cercanos,
sin exigir invención mundial ni confundir diferencia programada con prioridad.

La candidata completa falló la heurística del piloto y fue 2,70 % peor que
`atom_mse_only`; esto refuta su continuación bajo esa regla y alcance, no
demuestra ausencia de novedad. Ni las propiedades ni el resultado de la aditiva
se transfieren a R3 de V3. La receta queda documentada sin cambiar la regla,
promover una ablación o abrir otra campaña.
