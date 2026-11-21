# Admisibilidad de tres procedimientos para R3

Preparado con asistencia de IA el 2026-11-21. Este componente de investigación documental y derivación matemática ejecutó cero sorteos, remuestreos, Monte Carlo, entrenamientos o predicciones nuevas. Estos contadores se limitan a este componente y no describen las ejecuciones de revisión estadística del trabajo principal. Los intentos anteriores permanecen intactos.

**Dictamen actual: no existe una batería de tres procedimientos confirmatorios acreditada.** Hay dos familias pareadas propuestas —bootstrap del funcional y delta-HAC— y una ruta global MCS que requiere justificar cobertura para el mismo funcional. Las variantes de bootstrap y sus ajustes no añaden familias por sí solas. El resultado adverso anterior de Holm38, 14/128 falsos rechazos en seed-only, permanece vigente; sustituir P1 o descartar sus rechazos sería desarrollo nuevo.

El objetivo se conserva como `rho_m = E_S[sqrt(E_Z[q_m(S,Z)])]`, `q=error²`, para una población declarada de semillas de entrenamiento y un proceso temporal común. Para cada rival `j`, `theta_j=rho_j-rho_c`: igualdad `theta_j=0`, dirección favorable `theta_j>0`, margen práctico `delta_j>0` y alternativa de planificación `theta_plan,j` son objetos diferentes. La guía exige igualdad, dirección y relevancia práctica; no impone universalmente tres nulos `theta<=delta`. Los 19 rivales y 20 modelos corresponden al estudio sintético anterior. La familia prospectiva completa y su tamaño quedan por declarar; este documento no fija un roster futuro ni incorpora una candidata abandonada.

## Qué puede contarse

| Familia | Hipótesis y unidades reales | Dictamen para el objetivo primario |
|---|---|---|
| P1: bootstrap estacionario cruzado, recalculando RMSE por semilla | Igualdad de `rho_c,rho_j`; semillas emparejadas entre modelos y bloques temporales conjuntos | Candidata pertinente. Studentizar P1 es una reparación de esa familia. Faltan prueba de aproximación, sesgo, colas y validación nueva. |
| P2: delta-Wald con covarianza semilla + temporal HAC | Igualdad de `rho_c,rho_j`, usando la influencia del funcional | Segunda familia candidata. No es automáticamente el teorema DM-HLN; `t(R-1)` y el factor h=1 no se justifican por sus nombres. |
| P3: MCS global del mismo vector `rho` | EPA en los subconjuntos y cobertura del conjunto de óptimos | Tercera familia global posible, no acreditada. Exclusión de un rival no contrasta por sí sola igualdad candidata-rival. Véase la regla singleton abajo. |
| DM-HLN sobre MSE diaria media entre semillas | Media temporal de diferencias de pérdidas aditivas | Cambia el estimando. Una adaptación mediante influencias necesita derivación; como reetiquetado de P2 no añade otra familia. |
| GW con instrumentos no constantes | Igualdad predictiva condicional, comprobada mediante momentos instrumentados | Igualdad incondicional de `rho` no implica ese nulo más fuerte. Rechazarlo no demuestra diferencia de `rho`. Con instrumento constante se vuelve a la prueba incondicional; el original identifica esa forma con DM. |
| Clark-West | MSPE ajustada para modelos realmente anidados | No es genérico para arquitecturas heterogéneas; ni distinto tamaño ni número de parámetros prueba anidamiento. La pérdida ajustada tampoco es la media individual de RMSE. |
| Friedman sobre fechas, semillas u orígenes arbitrarios | Comparación de rangos sobre bloques justificables | No transforma un calendario correlacionado en bloques independientes. Igualdad de medias no implica igualdad de rangos/distribuciones; no satisface automáticamente el nulo primario. |
| Signos, permutaciones, Wilcoxon | Simetría/intercambiabilidad o nulos de distribución/rango | No derivados de igualdad de medias. No se añade una tercera prueba bajo supuestos no verificados. |

Fuentes primarias y alcance de consulta: [Hansen, Lunde y Nason (2011)](https://doi.org/10.3982/ECTA5771), [Giacomini y White (2006)](https://doi.org/10.1111/j.1468-0262.2006.00718.x), [Clark y West (2007)](https://doi.org/10.1016/j.jeconom.2006.05.023). Los apartados concretos inspeccionados, las copias consultadas y las limitaciones de lectura figuran en `PRIMARY_SOURCES.json`; no se declara lectura integral de esos artículos.

## Por qué condicionar no repara el objetivo

Fijar la cohorte de semillas conduce a `rho_m,cohort = R^-1 sum_r sqrt(E_Z q_m(S_r,Z))`. Es un parámetro condicional distinto de `rho_m`, aunque su estimador tenga idéntico aspecto numérico. Fijar el calendario y remuestrear sólo semillas aproxima otro objeto: `E_S sqrt(T^-1 sum_t q_m(S,Z_t))`. El sesgo de la raíz puede diferir entre modelos con igual riesgo poblacional y distinta dependencia. Aumentar semillas no elimina ese sesgo. El ejemplo determinista independiente está en `../qa/FINITE_CALENDAR_RMSE_BIAS_AUDIT.json`.

Para explicitar la dificultad, bajo un diseño independiente entre semillas y proceso temporal, escriba `mu_m(s)=E_Z q_m(s,Z)` y `r_m(s)=sqrt(mu_m(s))`. La expansión que debe justificarse es

`rhohat_m-rho_m = R^-1 sum_r [r_m(S_r)-rho_m] + T^-1 sum_t b_m(Z_t) + remainder`,

`b_m(z)=E_S[(q_m(S,z)-mu_m(S))/(2 r_m(S))]`.

Esta es una derivación de trabajo propia, no un teorema demostrado por citar bootstrap cruzado. Requiere controlar la interacción semilla × tiempo, el resto de Taylor, momentos, mezcla, RMSE alejada de cero y regímenes relativos de R y T. Si la primera proyección se degenera, no se conserva automáticamente una referencia normal. Como diagnóstico de segundo orden, bajo condiciones para intercambiar esperanza y Taylor, el sesgo temporal por semilla es aproximadamente `-LRV(q_m(s,Z))/(8 r_m(s)^3 T)`. Hay que controlar la diferencia de esos sesgos respecto de la escala del contraste. Si domina la variación entre semillas, una condición suficiente orientativa para este término es `sqrt(R)/T -> 0`; no es un teorema general para el código actual.

[Owen y Eckles (2012)](https://doi.org/10.1214/12-AOAS547) estudian medias de arreglos de efectos cruzados y discuten extensiones suaves; no demuestran esta raíz anidada con columna temporal serial. [Menzel (2021)](https://doi.org/10.3982/ECTA15383) trata dimensiones muestreadas independientemente y deja la dependencia temporal débil como extensión fuera de su desarrollo. Estas referencias ayudan a formular la brecha, sin cerrarla. GW §3.4 exige una secuencia creciente del ancho HAC para su consistencia asintótica incondicional; un lag fijo 8 necesita justificación finita específica.

## Una tercera ruta global concreta: singleton MCS

Defina `M*=argmin_m rho_m` y un MCS `C_alpha` que cubra **todos** los óptimos. La fuente de la garantía abstracta es Hansen–Lunde–Nason, definición y Teorema 1, pp. 458–460; el siguiente uso para la candidata es una deducción propia:

`H0_global: existe j!=c con rho_j <= rho_c`.

Si `C_alpha={c}` bajo ese nulo, entonces `M*` no está contenido en `C_alpha`: si la candidata está empatada en el mínimo falta otro óptimo; si no es óptima falta todo óptimo. Por tanto, si la cobertura es válida, el evento singleton falso está acotado por su error de cobertura. La conclusión es dominancia estricta **global** de la candidata sobre todos los rivales de la familia completa previamente declarada. Con cobertura meramente asintótica el resultado también es meramente asintótico y puntual; la uniformidad exige otro argumento.

Esta regla exige que sólo quede la candidata. Si quedan candidata y otro modelo, no acredita el tercer criterio; que un rival concreto haya salido antes tampoco lo acredita. No se fabrican p-valores pareados ni se prueba el margen práctico. Los p MCS de exclusión conservan su interpretación de pertenencia al conjunto óptimo.

La Tabla 4 de la guía menciona MCS; no obliga a confianza .90 ni a `alpha_MCS=.10`. El .10 de los cribados anteriores fue una elección del proyecto. El nivel prospectivo del MCS queda sin fijar: debe congelarse y justificarse junto con el nivel y la multiplicidad de la regla principal. Si se pretende una afirmación global respaldada sólo por MCS al .05, se necesita la cobertura correspondiente o una construcción combinada que justifique ese control. Un cribado a .10 no demuestra control a .05 ni se transfiere automáticamente a R3.

**La ruta resuelve la interpretación estadística global, no demuestra por sí sola cumplimiento literal de tres procedimientos por contraste con Holm.** Si se utiliza, el protocolo debe declarar expresamente un criterio global compartido adicional a las dos familias pareadas, justificar su correspondencia con la guía y mantener los resultados globales en columnas propias. Si la interpretación exige tres p-valores pareados distintos, sigue faltando el tercero. Construir cierre múltiple o max-t a partir de P1 no cuenta automáticamente como una familia nueva.

El código anterior calcula una extensión MCS y sólo cribó rechazo EPA inicial en tres nulos. Eso no acredita cobertura de `M*` con múltiples óptimos ni bajo nulos parciales. Para evaluar esta ruta hacen falta: una teoría o alcance asintótico explícito para `rho`; una construcción conjunta válida; nulos parciales con óptimos múltiples, empates candidata-rival subóptimos y dependencia heterogénea; error de cobertura y falsas decisiones singleton, además de potencia de la regla completa. `THIRD_PROCEDURE_PLAN.json` fija esas tareas sin autorizar ejecución.

La multiplicidad debe congelarse antes de nuevos resultados. Holm38 fue una elección del proyecto anterior. La guía no obliga universalmente a aplanar todo en Holm57: otras construcciones de intersección y control por rival requieren su propia justificación. Ninguna elección corrige p-valores inválidos. La puerta confirmatoria permanece cerrada.
