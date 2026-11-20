# Batería temporal propuesta: revisión y límites

**Estado: borrador futuro, sin implementación estadística validada; NO-GO vigente.**
Esta revisión nueva no cambia el piloto ni sus reglas. Registra una alternativa
metodológica a la propuesta GW del borrador anterior, antes de cualquier dato
confirmatorio nuevo. La revisión fue asistida por IA y no es una firma humana.

## Estimando y unidades

La propuesta futura es theta_j = E_seed[sqrt(E_P(error_j²)) −
sqrt(E_P(error_V3²))], con proceso temporal P, distribución de inicializaciones
y margen práctico delta_j predefinidos. Su estimador conserva la **media de
las diferencias de RMSE individuales**, calculadas sobre las mismas fechas.
No es RMSE de un ensemble ni raíz de MSE agrupado entre semillas. El calendario
fijo de 55 fechas ya expuesto no se renombra como una muestra de P no observada.

El objetivo futuro implica dos fuentes de incertidumbre: semillas y calendario.
Un remuestreo de fechas solamente no cubre E_seed; un t-test entre semillas
solamente condiciona al calendario observado. Las S×T celdas no son independientes.
La ley conjunta y el esquema de muestreo deben aprobarse antes de fijar pruebas.

## Tres propuestas para el mismo nulo unilateral

El nulo de superioridad práctica es **theta_j <= delta_j**; la alternativa de
planificación debe especificar theta_alt > delta_j. Simular exactamente en delta
corresponde a la frontera nula, no a una alternativa con potencia 0,90. La fórmula
de detección de una diferencia delta contra cero es otra hipótesis y no certifica
este gate completo.

1. **P1, bootstrap pareado por bloques del funcional RMSE.** Recalcular el
   funcional completo en cada réplica; usar los mismos bloques temporales y
   emparejamiento de semillas para todos los modelos. Requiere justificar la
   variación cruzada semilla/fecha, studentización y calibración bajo el nulo.
   Copiar un bootstrap bidimensional genérico no establece esa validez.
2. **P2, expansión delta y HAC.** Para cada semilla, la influencia temporal es
   (q_j,t−mu_j)/(2 sqrt(mu_j)) − (q_V3,t−mu_V3)/(2 sqrt(mu_V3)), q=error².
   Se requiere derivar el componente entre semillas, la covarianza temporal y
   su interacción; no sumar varianzas sin identificación. Comprobar momentos,
   denominadores positivos, dependencia débil y sesgo de la raíz. DM sobre MSE
   no es automáticamente una prueba de esta diferencia RMSE. HLN es una
   corrección candidata que necesita revisión específica para el funcional.
3. **P3, subsampling pareado del mismo funcional.** Es una alternativa futura al
   GW omnibus, con tamaños de submuestra, unidades, tasas y normalización
   fijados de antemano. Debe tratar tanto semillas como calendario y justificar
   su distribución límite. Compartir estimador con P1 no basta para validarlo.

El horizonte de pronóstico es un día. El soporte de sigma7 usa ocho cierres y
genera solapamiento de etiquetas; h=1 no elimina la autocovarianza. HAC y bloques
deben justificar su longitud y sensibilidad; lag 7 no demuestra independencia.

GW puede quedar como diagnóstico condicional auxiliar: su rechazo omnibus no
demuestra dirección favorable ni el mismo margen RMSE. Los instrumentos deben
ser causales; {1, low, high} es redundante si low/high son exhaustivos. Clark-West
no se aplica indiscriminadamente a rivales no anidados. La propuesta P3 no se
declara aprobada ni se escoge por producir un p favorable.

## Control familiar y validación antes de la potencia

Para la familia principal, ajustar **los 57 p-valores** mediante Holm a 0,05.
Cada rival se supera sólo si sus tres p ajustados son <0,05 y se cumple la
dirección práctica prefijada. Es una elección conservadora explícita; la
planificación debe usar exactamente esa multiplicidad y esa intersección.
Holm tolera dependencia, pero no corrige p-valores marginales inválidos.

Hay que validar tamaño y FWER en la frontera nula, nulos interiores y
configuraciones parcialmente nulas, con shocks comunes, colas, heterogeneidad
entre semillas y dependencia/solapamiento temporal. Registrar tolerancias y
límites binomiales Monte Carlo; la resolución de las colas debe permitir evaluar
el umbral inicial 0,05/57. El diseño de las familias adicionales y de la decisión
de consistencia entre dominios debe quedar aprobado antes de congelar.

Sólo después se simula potencia bajo alternativas justificadas del diseño
completo. Se cuenta la intersección de las tres decisiones por rival y la
intersección de toda la familia en **la misma réplica**, conservando dependencia.
No se multiplican potencias marginales. La guía exige >=0,90 por contraste y
reportar la conjunta; no exige que la conjunta sea >=0,90. Los límites Monte
Carlo describen precisión computacional condicional al DGP, no validez externa.

`power_contract.py summarize-simulation` deja implementada esa agregación y
rechaza familias incompletas, presupuestos mezclados y resultados observados
etiquetados como confirmación. No implementa ni valida P1/P2/P3; requiere sus
p-valores de una simulación de planificación futura. Por ello no certifica
potencia real ni habilita una campaña.

Los tres seeds piloto, las 55 fechas expuestas y las 16 varianzas de rivales
ausentes impiden aprobar hoy calibración, dimensionamiento o potencia final.
