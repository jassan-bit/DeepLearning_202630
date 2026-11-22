# Interpretación independiente de la validación V1 cerrada

**Resultado: estudio completo; batería V1 no admitida por el criterio previo de cobertura.**
Esta lectura se escribió después de terminar las 4.608 réplicas y cerrar la
sesión. No se alteraron código, protocolo, resultados ni criterios por las
observaciones parciales.

## Procedencia, integridad y coste

El código y el protocolo se publicaron **antes de las réplicas formales** en el
commit `db5a75bb66a0b28b1d05dacc41b66daf161f9120`. Protocolo SHA256:
`55dc350ce1e5252f95231113112b02eababaa6976be1ae288ba09407441b4bd9`.
El caso está en `../results/validation-v1/` y su copia de fuentes anterior a
cualquier enmienda se conserva en `../versions/mean_rmse_v1/`.

Se verificaron las **4.608 filas**, su orden round-robin y toda la cadena SHA256.
No faltan réplicas ni se descartaron resultados adversos. El evento final
registra **4.410,37298 s CPU observados**: 4.375,99475 s en réplicas y
34,37823 s adicionales de preparación/pool/contabilidad. Quedaron
**989,62702 s** del presupuesto declarado de 5.400 s. No hubo coste incierto.
La medida corresponde al tramo documentado de ejecución; no es duración de
pared ni incluye instalación/exportación/verificación exterior al tramo.
La cohorte fue Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Linux x86_64.

Hubo **cero entrenamientos, inferencias de redes, predicciones financieras,
descargas de mercado o aperturas de reservas**. Las pérdidas son sintéticas.
Las semillas y los 20 rivales etiquetados tienen la estructura de dependencia y
compresión restringida declarada antes de simular; no representan 20
algoritmos estadísticamente independientes ni una población de entrenamientos.

## Resultados con los criterios previamente fijados

Cada escenario tiene 512 réplicas. Los límites son unilaterales Clopper–Pearson,
con Bonferroni sobre una única familia de **107 afirmaciones** y confianza
Monte Carlo global 95 %. La cobertura operativa de CI99 debía tener límite
inferior ≥ 0,95; no se pretendía certificar cobertura exacta 99 % mediante ese piso.

| Escenario | Falsas aceptaciones conjuntas | Límite superior simultáneo del error | CI99 diferencia cubre | CI99 ganancia cubre | Menor límite inferior simultáneo de cobertura | Criterio completo |
|---|---:|---:|---:|---:|---:|---|
| `complete_epa_iid` | 0/512 | 0.01486605 | 504/512 | 504/512 | 0.95701416 | Pasa |
| `complete_epa_ar_shared` | 0/512 | 0.01486605 | 507/512 | 507/512 | 0.96622949 | Pasa |
| `main_single_practical_boundary` | 4/512 | 0.03050290 | 501/512 | 501/512 | 0.94836605 | **No pasa** |
| `two_null_additional_boundaries` | 4/512 | 0.03050290 | 505/512 | 505/512 | 0.96000779 | Pasa |
| `two_null_additional_negative_coupling` | 6/512 | 0.03692582 | 505/512 | 506/512 | 0.96000779 | Pasa |
| `conditional_cancellation` | 0/512 | 0.01486605 | 508/512 | 508/512 | 0.96949710 | Pasa |
| `variance_break_adverse` | 0/512 | 0.01486605 | 509/512 | 509/512 | 0.97291316 | Pasa |
| `positive_risk_ranking_reversal` | 0/512 | 0.01486605 | 504/512 | 505/512 | 0.95701416 | Pasa |
| `strong_positive_control` | control positivo | — | 505/512 | 504/512 | 0.95701416 | Pasa |

**Control de la decisión:** los ocho escenarios nulos pasan el techo 0,05. La
mayor tasa observada es **6/512=0,01171875**, con límite superior simultáneo
**0,03692582**, bajo dependencia temporal fuerte y acoplamiento negativo de
adicionales. No se obtiene ese límite multiplicando cotas marginales
insuficientes ni suponiendo independencia: se evaluó la decisión completa de
cuatro paneles, 20 rivales, tres procedimientos y dos nulos, manteniendo
Holm120, la pantalla180, dirección, efecto e intervalos.

**No vacuidad:** la alternativa fuerte prefijada del 25 % fue aceptada en
**512/512** réplicas. La cota inferior simultánea de potencia conjunta es
**0,98513395** y la menor de los 80 contrastes también es **0,98513395**. Por
tanto, la protección no consiste simplemente en que el programa siempre se
abstenga. Es potencia bajo ese DGP y efecto fuerte; **no es potencia del diseño
real, MDE o evidencia de mercado**.

**Cobertura:** sólo `main_single_practical_boundary` falla el criterio
completo. Ambos CI99 cubren en **501/512=0,978515625** casos y su límite
inferior simultáneo es **0,94836605**, menor que 0,95. Son once faltas cuando la
capacidad calculada antes de simular permitía como máximo diez. Se conserva
`accepted=false`: no se eliminan el escenario o una afirmación de la familia,
no se baja el piso y no se añaden réplicas de V1 hasta obtener un resultado
favorable.

Este fallo **no prueba que la cobertura poblacional sea menor que 95 %**:
el intervalo puntual CP95 para 501/512 es [0,96188423;0,98922745]. Lo que
incumple es la certificación simultánea prefijada. Al mismo tiempo, no conviene
atribuir todo a una coincidencia Monte Carlo: el extremo superior puntual es
menor que el nominal 99 %, y los CI95 del mismo escenario cubren 472/512
(ganancia) y 473/512 (diferencia), con extremos superiores puntuales 0,94360220 y
0,94527630, inferiores a 0,95. Hay señales diagnósticas de infracobertura
nominal también en el acoplamiento negativo y en la inversión de ranking.
Son diagnósticos puntuales posteriores, no nuevas reglas de aceptación.

## Diagnóstico del fallo y corrección concreta propuesta

En la frontera principal de ruido bajo, las once faltas de ambos CI99 se
reparten en **cinco** casos con verdad por debajo del extremo inferior y
**seis** por encima del superior. A nivel 95 %, la diferencia tiene 16 faltas
por debajo y 23 por encima; la ganancia 17 y 23. No aparece sólo una cola o una
violación de la definición de la primaria: los puntos y el bootstrap
recalculan la media de las raíces por semilla. El problema se localiza en la
calibración de intervalos operativos de la versión V1.

Una corrección matemáticamente motivada, única y prospectiva es invertir el
**test escalar delta/HAC con referencia fixed-b ya presente** para construir
los CI operativos de la diferencia de medias-RMSE y del logaritmo del cociente.
Para este último, se invierten los extremos por la transformación monótona
`1-exp(logratio)`. El valor crítico procede de la referencia FCLT prefijada y
se invierte con la misma corrección HLN que el test; no se elige un factor de
anchura para conseguir el criterio. El bootstrap-t puede conservarse como
procedimiento contrastivo y diagnóstico de intervalos, con su alcance separado.

La motivación es que, con b=L/T finito, el estadístico studentizado tiene una
ley no estándar; regenerar una distribución bootstrap-t no garantiza
aproximarla suficientemente en muestra finita. La distribución Gaussiana fixed-b
que ya usa el test permite una inversión coherente bajo sus supuestos. Este
argumento **no demuestra anticipadamente** que los intervalos V2 vayan a pasar,
ni identifica de forma causal una única razón para todas las faltas observadas.
Debe verificarse el álgebra de inversión y validarse una versión nueva.

La nueva evaluación requiere un nuevo anclaje público anterior al cálculo y
una semilla externa independiente. Debe conservar los mismos nueve DGP,
N = 512, 107 afirmaciones, niveles, margen, regla principal+dos-de-tres y criterios.
No se vuelve a ejecutar o reinterpretar V1 como aprobada. Si la corrección
única también falla, corresponde reportar el bloqueo en lugar de buscar otra
configuración hasta que pase.

## Alcance del dictamen

La implementación y prueba de la primaria exacta y el control conjunto
observado aportan evidencia nueva respecto a las dos brechas anteriores.
Sin embargo, **la batería completa V1 sigue bloqueada** por su criterio de
intervalos. Tampoco una eventual validación V2 acreditaría R3: faltan datos y
potencia reales, protocolo/reserva adecuados, ejecución y superioridad de
mercado, junto con los restantes requisitos científicos.

Esta interpretación usa asistencia de IA declarada y no atribuye una firma,
defensa, revisión humana independiente o nota oficial.

## Identificadores de los resultados V1

- `state.json` SHA256: `dba1f3fe9d9d08b5e89c4264213122aa29d1aa1c7a5fcf0b6bc27a937c74f186`.
- `ledger.jsonl` SHA256: `febe123ef44781b8e05bffbb6632d607eebbf21a0201da8a7cec71379dad70b2`.
