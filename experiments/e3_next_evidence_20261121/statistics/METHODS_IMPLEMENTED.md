# Batería experimental implementada: dos contrastes y un complemento global

Preparación con asistencia de IA. La receta V3 mantiene NO-GO. Este trabajo
implementa inferencia sobre errores ya disponibles; no entrena, predice, accede
al mercado o habilita una confirmación. El plan controlado se fijó antes de sus
resultados en el commit `550b6d26cbd1b82f40a347c24b180320776e5ca3`.

## Objeto y alcance admisible

La entrada es `q[semilla,fecha,sistema]`, errores cuadrados pareados. El
estimador es la media de RMSE individuales por semilla, no RMSE agrupada ni de
ensemble. Se contrasta `theta_j <= delta_j` contra ventaja práctica positiva.
Se requieren al menos diez semillas y 32 fechas para ejecutar estas extensiones;
esas cantidades son guardas computacionales, no suficiencia asintótica o potencia.

El Cuadro 4 F2 de la guía nombra bootstrap estacionario, DM-HLN y MCS, entre
otras opciones. Aquí los tres algoritmos son adaptaciones propuestas del
funcional RMSE y del muestreo cruzado: su nombre no acredita automáticamente
su admisibilidad. **Sólo dos producen p-valores direccionales por rival.** MCS
es global y no ocupa el tercer contraste exigido por R3. Subsampling del
borrador anterior no se contó como método nombrado por ese cuadro.

## P1: funcional completo y bootstrap estacionario cruzado

Cada réplica usa una muestra de filas de semillas completas y un calendario
de índices estacionarios común a todos los modelos. Recalcula la raíz de la
MSE de cada semilla antes de promediar. El bloque medio es 12 fechas, fijado en
el plan; no se escogió por sus p-valores. Las S×T celdas no se tratan como IID.
Se usan raíces bootstrap centradas en el estimador observado para p unilateral
y CI básicos. No se afirma una studentización anidada que no esté implementada.

El remuestreo producto o pigeonhole puede sobrecontar la interacción en muestra
finita. Su validez asintótica requiere un proceso cruzado no degenerado,
independencia de las inicializaciones como unidades, dependencia temporal débil
y regularidad del funcional con MSE positiva. Los casos con raíces cercanas a
cero se rechazan; fluctuaciones de redondeo no se llaman variabilidad real.

## P2: linealización cruzada con HAC y factor HLN

Para cada rival y semilla, la influencia temporal es
`psi_st=(q_jst-mu_js)/(2 sqrt(mu_js))-(q_cst-mu_cs)/(2 sqrt(mu_cs))`.
Se estima `Vs=var_seed(RMSE_js-RMSE_cs)/S` y
`Vt=HAC_t(mean_seed psi_st)/T`, con Bartlett y lag 8 prefijado.

En una linealización cruzada aditiva `a_s+b_t+c_st`, Vs contiene
`sigma_a²/S + sigma_c²/(S T)`, mientras Vt contiene
`sigma_b²/T + sigma_c²/(S T)`. Su suma cuenta dos veces la interacción positiva.
Se conserva esa suma como varianza superior dentro de esa aproximación y se
reporta separadamente un diagnóstico de interacción por HAC de
`psi_st-mean_seed psi_st`, con divisor `(S-1)T`. La versión corregida se informa,
pero no se elige por dar menor p ni se recorta para aumentar potencia.

Esta identificación no es una demostración exacta para cualquier funcional
no lineal. Se requieren momentos, mezcla, muestreos y tasas S/T apropiados;
la corrección no resuelve por sí sola sesgo de raíz o deriva estructural.
El factor HLN de horizonte h=1 es `sqrt((T-1)/T)` y se aplica sólo al estadístico
con su inversión en el CI. El horizonte de un día no elimina el solapamiento de
sigma7. Se usa referencia t con S−1 grados de libertad como propuesta conservadora,
no como teorema de DM clásico para este diseño cruzado.

## P3: MCS global del funcional

Se calcula el estadístico de rango máximo `T_R`, studentizado por la dispersión
bootstrap de diferencias de riesgos funcionales. Las réplicas están centradas
bajo igualdad de capacidad predictiva; se elimina iterativamente el peor sistema
con p de exclusión monótono hasta el conjunto de confianza al 90 %.

Su hipótesis es igualdad de riesgos en el conjunto actual. Un rival excluido no
recibe por ello un p unilateral de `theta_j <= delta_j`; no se inventa una
prueba práctica pareada a partir de un ranking. Se documenta además que el uso
de riesgos RMSE no aditivos y bootstrap cruzado es una extensión propuesta del
MCS habitual de pérdidas temporales, pendiente de revisión de admisibilidad.

## Familias y potencia

La validación controlada registra 38 p de los dos procedimientos direccionales
frente a 19 rivales. Holm38 se identifica como diagnóstico de software/DGP,
distinto del borrador anterior de 57 componentes. También se reporta una
reserva conservadora Holm57 con 19 p=1 no implementados: esas casillas no son
una tercera prueba. Ninguna opción completa la batería ni computa potencia
de tres métodos. Se conserva el mínimo de los requisitos reales y el gate false.

La validación usa procesos de errores sintéticos con verdad analítica conocida,
no varianzas de los diecinueve rivales reales. Los intervalos Monte Carlo son
individuales, no simultáneos sobre todos los escenarios/rivales. Los resultados
pueden detectar problemas; no certifican potencia final o validez universal.
El caso t5 tiene segundo momento unitario exacto, pero su distribución estacionaria
completa sólo se aproxima mediante burn-in de 128 puntos. Esa distinción se
preserva incluso si sus resultados parecen favorables.

Se necesitan aún el tercer procedimiento pertinente por contraste, revisión
metodológica de las extensiones, entradas de planificación reales admitidas y
dimensionamiento final antes de cualquier protocolo confirmatorio.

La guarda numérica usa `64*eps*max(1, escala)`; por ello no es universalmente
invariante a un cambio de unidades. Unidades extremadamente pequeñas pueden
clasificarse como no soportadas aunque una reescala sea positiva. El protocolo
futuro debe fijar unidades y revisar esa guarda antes de usar datos reales; no
se ajusta aquí después de observar resultados para producir un rechazo.
