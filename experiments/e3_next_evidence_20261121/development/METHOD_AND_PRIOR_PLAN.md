# V4: hipótesis de puntuación en la coordenada de la etiqueta

Preparado con asistencia de IA. Esta es una revisión nueva de desarrollo; la
receta V3 original conserva sus 30 ajustes, su resultado adverso y su NO-GO.
El dictamen acotado de R1 de V3 no se hereda a esta modificación. No se promueve
la ablación MSE-only ni se declara una propiedad predictiva mejorada.

## Hipótesis precisa

La lectura de V3 se conserva: `mu = sum(p_k * T_A(a_k))`, con
`T_A(u) = sqrt(A + (6/49)*u^2)` y una ley de tres átomos de U. Se mantienen el
encoder, las masas aprendidas, la inicialización y los 1262 parámetros.

La revisión cambia el auxiliar `0.1*CRPS_U/sU` por
`0.1*CRPS_sigma/sY`, donde los átomos de sigma son `T_A(a_k)` y el objetivo es Y.
La MSE puntual se mantiene dividida por `sY^2`. Las escalas son medias de
entrenamiento fijadas, no parámetros aprendidos ni coeficientes elegidos por
el resultado. **Cambian tanto la variable puntuada como su normalización.**

Para A fijo y U no negativo, el cambio de variable de la representación
integral del CRPS muestra:

`CRPS_U(G,u) = integral[(F_sigma(y)-1{Y<=y})^2 * (dU/dY) dy]`,

donde `dU/dY = Y/(c*U)`, `c=6/49`. Si A>0, el peso diverge en el piso
`sqrt(A)` como una constante por `1/sqrt(Y-sqrt(A))`; es una singularidad
integrable. Para A=0 el mapa es lineal y la normalización por medias da
`CRPS_U/sU = CRPS_sigma/sY`.

Ambos scores son propios para leyes no restringidas y la transformación es
inyectiva. Este argumento **no** prueba que el score original tenga un óptimo
poblacional incorrecto. En una familia finita misspecificada y durante la
optimización, los gradientes y el compromiso con MSE pueden diferir. El nuevo
score también puede perjudicar la MSE. No se afirma dominancia, calibración,
una clase más expresiva o novedad mundial por usar CRPS de una transformación.

## Controles fijados antes del diagnóstico

1. U-CRPS original y su escala sU, con la MSE común.
2. Sigma-CRPS con escala sY: nueva revisión V4.
3. Sigma-CRPS con denominador `sqrt(c)*sU`: control de normalización que iguala
   exactamente el original para A=0.
4. MSE-only: comparador de diagnóstico; no se convierte en candidata ganadora.

Un cambio positivo de escala sólo cambia la norma del gradiente auxiliar,
no su dirección. El diagnóstico informa separadamente colinealidad entre los
dos scores sigma y diferencia de dirección entre U/sigma. Tampoco interpreta
un conteo de casos favorables como evidencia de utilidad de mercado.

## Ejecución permitida y criterio de decisión

`plan_before_results.json`, SHA256
`afb7362ec81b09e86a1e0fd29f3a222d4c2d9bcf7b7e464a918520824989c4d5`,
fija los 144 casos: seis A (incluido cero, casi piso y 100), dos leyes verdaderas
finitas, cuatro tríos de átomos de modelo y tres vectores de probabilidades.
Las leyes tienen tres átomos representables o cinco átomos misspecificados.
Las expectativas se calculan exactamente con pesos finitos; no hay sorteos
Monte Carlo, optimizador, entrenamiento o archivos de mercado.

Se informan todos los riesgos, normas, cosenos y pendientes locales de la MSE,
incluidos casos adversos y gradientes cero. Si no se observa diferencia de
dirección no lineal en A>0, se abandona ese fundamento en vez de inventarlo.
Las propiedades de software y las identidades deben pasar antes de plantear
un piloto de mercado. Un resultado geométrico favorable sólo sostiene que
existe una diferencia comprobable; no selecciona esta receta ni abre confirmación.

Un eventual piloto de desarrollo requiere un plan distinto previo, admisión,
presupuesto y tratamiento comunes, controles y recursos disponibles. No se
reutiliza un calendario expuesto como confirmación ni se cambia la regla del
piloto V3. R3 y la revisión de R1 de V4 permanecen pendientes.
