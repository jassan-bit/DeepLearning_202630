# Batería propuesta: especificación y alcance aún no acreditado

**Estado: `PRIMARY_BATTERY_NOT_ADMITTED`.** Existen dos familias pareadas propuestas y una ruta global condicionada. No hay tres procedimientos primarios actualmente acreditados. Este documento no convierte el diseño en resultados.

Fuente: corte 5f29ab5; [§8.27 y antecedentes](https://github.com/jassan-bit/DeepLearning_202630/blob/5f29ab51e1bb3e79d972377b6d682d1785bdc298/book_e3/sections/08_verificacion_estadistica.md). Asistencia de IA declarada.

## 1. Pregunta, métrica y muestreo

Sea q_m(s,t) el error cuadrático de la ejecución con semilla s del sistema m en fecha t, manteniendo fijos los datos y procedimiento de entrenamiento que se declaren. El blanco poblacional propuesto es

\[
\mu_m(s)=E_Z[q_m(s,Z)\mid s],\quad
\Psi_m=E_S\sqrt{\mu_m(S)},\quad
\theta_j=\Psi_j-\Psi_c,
\]

con estimador observado

\[
\widehat\Psi_m=R^{-1}\sum_s\sqrt{T^{-1}\sum_t q_m(s,t)},\qquad
\widehat\theta_j=\widehat\Psi_j-\widehat\Psi_c.
\]

La métrica reportada sigue siendo la **media de RMSE individuales**, en unidades originales de sigma7. No es media de MSE, RMSE de ensemble ni esperanza redefinida para coincidir con el calendario observado. La incertidumbre por muestreo del entrenamiento completo no se incluye silenciosamente: si se reivindica, necesita diseño y aproximación propios.

La igualdad primaria es H0j:theta_j=0 frente a diferencia bilateral; la dirección favorable es theta_j>0. La ganancia práctica delta_j, alternativa de planificación, efecto estandarizado, IC y adecuación son condiciones separadas. La guía no prescribe tres hipótesis idénticas theta≤delta. El margen histórico no se adopta automáticamente para otra identidad o escala.

Las filas representan ejecuciones de entrenamiento emparejadas y las columnas una cronología común dependiente. Deben conservarse el emparejamiento real entre modelos y la dependencia temporal, entre modelos y entre semillas inducida por el calendario. R×T no son observaciones IID. Un control determinista tiene una trayectoria, no R réplicas inventadas.

## 2. Especificación de las tres familias propuestas

### P1 — Bootstrap-t del funcional cruzado

- **Pregunta:** igualdad de Psi candidata–rival; ámbitos de semillas y calendario explícitos.
- **Muestreo propuesto:** filas de semillas emparejadas y bloques temporales comunes a todos los modelos. Cada sorteo recalcula RMSE, contraste y studentizador; la validez de este producto de remuestreos todavía debe derivarse para la estructura serial y su interacción.
- **Estadístico histórico:** t_j=thetahat_j/SEhat_j; pivote remuestreado t*_j=(thetahat*_j−thetahat_j)/SEhat*_j.
- **p histórico:** (1+conteo(|t*|≥|t|))/(B+1), bilateral, empates incluidos y rechazo estricto. El IC cerrado usa el estadístico de orden del pivote absoluto; con B999 y alpha.05 el rango951, sin cuantiles interpolados.
- **Supuestos pendientes:** estacionariedad/momentos/mezcla pertinentes; semillas según su diseño; riesgos positivos, control del sesgo y resto no lineal; régimen R/T; reglas de bloque y studentización justificadas. El genérico «bootstrap por bloques» no suministra estas garantías.
- **Degeneraciones:** no borrar rivales o sorteos. SE nulo/no soportado implica resultado no soportado, p1 e IC ausente, sin victoria ni intervalos de certeza fabricados. Una igualdad estructural conocida se registra como tal, no como tamaño efectivo adicional.
- **Estado:** familia pertinente como propuesta; no calibrada para todo el alcance requerido.

### P2 — Aproximación delta-HAC cruzada

- **Pregunta:** misma igualdad primaria, por aproximación de las proyecciones de semilla y tiempo del funcional.
- **Estadístico histórico conservado:** z_j=thetahat_j·sqrt((T−1)/T)/SEhat_j, con p bilateral de una referencia t(R−1). El IC Wald correspondiente usa SEhat/sqrt((T−1)/T).
- **Covarianza propuesta:** combina proyecciones de semilla y tiempo con covarianza temporal HAC. Deben justificarse interacción, posible doble conteo, grados de libertad y resto. La referencia t(R−1) y factor histórico no están acreditados por invocar DM-HLN.
- **Supuestos pendientes:** regularidad anterior, influencia de la raíz, ancho de banda apropiado y crecimiento, consistencia conjunta para todos los modelos/contrastes. Horizonte uno no elimina solapamiento sigma7 o autocorrelación.
- **Degeneraciones:** SE/covarianza no soportados, n insuficiente o falta de regularidad conservan su posición y resultado inconcluso; no se sustituye por p0 o se elimina el rival.
- **Estado:** segunda familia propuesta; teoría y calibración finitas aún no justificadas. Compartir una covarianza con P1 no demuestra independencia de sus decisiones, ni la guía la exige.

### P3 — Conjunto de confianza de modelos, ruta global estricta

Sea M* el conjunto de **todos** los minimizadores de Psi dentro de candidata y familia completa. La pregunta global es H0G: existe algún rival j con Psi_j≤Psi_c. El criterio favorable propuesto es exclusivamente C_alpha={candidata}.

Si un conjunto C_alpha cubre todos los óptimos con error≤alpha, una falsa decisión singleton candidata implica M* no contenido en C_alpha. Por tanto su probabilidad queda acotada por el error de cobertura, con exactamente el alcance —finito/asintótico, puntual/uniforme— que se haya demostrado. Es una implicación lógica condicionada: no demuestra la cobertura del algoritmo actual.

La implementación histórica usa máximo de contrastes estandarizados en cada subconjunto y eliminación secuencial. Necesita la aproximación conjunta válida en **todos los subconjuntos visitados**, nulos parciales y empates; no basta controlar el rechazo de igualdad global inicial. Una covarianza/studentización no soportada conserva todos los modelos y marca falta de soporte.

La pertenencia o exclusión no produce un tercer p por rival. El singleton tampoco demuestra delta ni sustituye adecuación. El nivel0.10 fue de desarrollo y no es una obligación universal; el nivel de una futura regla debe fijarse y justificarse. Para el estudio propuesto en VALIDATION_PLAN se considera0.05 como elección metodológica nueva, sin trasladarla retrospectivamente ni fijar aún una campaña.

**Puerta docente/metodológica:** resolver si dos procedimientos pareados más este criterio global simultáneo satisfacen la correspondencia de la guía. Si se exige literalmente un tercer p por contraste, continúa faltando un procedimiento. No se reinterpretará el resultado del MCS como diecinueve p fabricados.

## 3. Multiplicidad y decisión completa

Para m rivales fijados antes de resultados, una construcción conservadora propuesta aplica Holm a las 2m posiciones de P1/P2 y exige ambos p ajustados<.05 para cada rival. MCS se informa aparte, con su nivel y garantía propios. La guía admite una prueba global/control múltiple; no obliga universalmente a aplanar3m si existe otra construcción compatible y defendida. No se rellenan posiciones inexistentes del tercer procedimiento.

La futura decisión R3 sería conjunción de: R1/R2 evaluadas para la misma identidad; familia completa y comparación justa; media favorable, pruebas pertinentes favorables, diferencia práctica y efecto/IC; consistencia adicional; adecuación y robustez. Potencia del rechazo EPA inicial, singleton o dos métodos aislados no es potencia de esa conjunción.

Para el p bootstrap con corrección+1 y comparación estricta con alpha/K se necesita 1/(B+1)<alpha/K. B999 permite alcanzar el primer umbral de Holm38, pero no el de Holm57. Se recalcula B cuando se congelen familia y estructura; alcanzar el umbral no prueba buena resolución de cola ni calibración. El diseño real m y los B finales continúan sin fijar.

## 4. Resultados previos que se conservan

| Endpoint previo, estudio 8.27 | Resultado y alcance |
|---|---|
| P1 nuevo/P2 con Holm38, sólo semillas | 6/256 falsas decisiones familiares frente a31/256 del control sobre las mismas muestras; mejora localizada. |
| Dependencia temporal fuerte | 12/256; límite superior CP95=.08044903>.08: **inconcluso según su regla previa**. No es por sí solo una demostración de inflación por encima de.05. |
| Cobertura puntual | Nueve señales descriptivas de infracobertura; no una nueva prueba simultánea. |
| MCS, igualdad parcial | Todos los óptimos retenidos en226/256=88,28%, IC95[83,69%;91,95%]; no certifica90%. El rechazo EPA inicial allí no es error porque también hay alternativas. |
| Alternativa sintética | Intersección P1/P2:20/128; con dirección/IC práctico16/128; singleton35/128. Eventos distintos, sin tercera prueba plenamente acreditada ni adecuación de mercado. |

El cribado .08 y el MCS .13 fueron tolerancias internas exploratorias, no exigencias literales de la guía. No se ensanchan ni redondean ahora para aprobar. Los datos y decisiones históricos permanecen intactos.

## 5. Reparación propuesta y alternativa de alcance

La brecha de Jensen introduce sesgo por calendario finito: distintas dependencias temporales pueden dar distintas esperanzas de la RMSE muestral incluso a igual Psi. Aumentar semillas con T fijo no lo elimina. [BIAS_BOUND_DERIVATION.md](BIAS_BOUND_DERIVATION.md) obtiene una identidad exacta y una cota condicionada a insumos defendibles; conserva el ranking observado y propone ampliar IC, sin declarar esa cota estimada o útil en los datos reales.

Antes de código nuevo se debe escoger y justificar UNA construcción de inferencia: cota de sesgo y raíz centrada con cobertura conjunta, o una corrección/jackknife claramente etiquetada con su resto y covarianza. No se compararán reparaciones y elegirán después por el p favorable. Las reglas de bloque/HAC y régimen R/T deben derivarse; bloque12/HAC8 fijos no constituyen por sí mismos teoría asintótica.

Un análisis condicionado al calendario puede usar Theta_j,T(Z)=E_S[RMSE_T,j−RMSE_T,c | Z]; elimina una parte del objetivo temporal por definición. Es una sensibilidad legítima con esa etiqueta, **no** una reparación silenciosa de Psi ni confirmación independiente. Una prueba de signos, rangos o permutación exige además sus supuestos y puede probar otro parámetro; no llena automáticamente el tercer casillero.

No se recomiendan GW con instrumentos para fingir el mismo nulo marginal, Clark–West sin anidamiento, Friedman sobre días correlacionados o tres variantes del bootstrap como tres familias. Los ámbitos pueden diferir si su correspondencia se defiende, sin imponer una homogeneidad no literal de la guía.

## 6. Literatura y límites de lectura

Se reutiliza [PRIMARY_SOURCES.json](https://github.com/jassan-bit/DeepLearning_202630/blob/5f29ab51e1bb3e79d972377b6d682d1785bdc298/experiments/e3_r3_route_20261121/methods_admissibility/PRIMARY_SOURCES.json): Hansen–Lunde–Nason2011 para MCS; Owen–Eckles2012 y Menzel2021 para dependencia cruzada con sus límites; Giacomini–White2006 para nulos condicionales/HAC; Clark–West2007 para anidamiento. No se atribuye a estas fuentes un teorema para nuestra extensión serial de media de raíces. Los accesos parciales y artículos sólo identificados por metadatos mantienen esa etiqueta.

**Decisión:** suficiente especificación para continuar derivación y revisión metodológica; insuficiente para ejecutar una validación aleatoria completa con código aún no fijado, y con mayor razón para una campaña confirmatoria. Véanse VALIDATION_PLAN.json, GATE_REPORT.json y NEXT_ACTIONS.md.
