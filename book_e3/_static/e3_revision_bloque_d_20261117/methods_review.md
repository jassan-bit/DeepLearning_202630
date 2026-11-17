# Revisión metodológica del Bloque D frente a la guía

Fecha: **17 de noviembre de 2026**, America/Bogota. Fuente revisada:
`8fccb216f36a3e4fc5cb38ecdec769b7f5379397`. Es una revisión posterior de
resultados conocidos, no un protocolo preregistrado, un nuevo cálculo de
p-valores o una acreditación de R3. Preparación y revisión asistidas por IA;
no constituye revisión humana independiente.

## Procedimientos ejecutados y relación con la Tabla 4

La guía exige al menos tres pruebas **distintas y apropiadas** de la familia
del problema; al menos una debe ser global o controlar comparaciones múltiples.
No exige literalmente que todas representen el mismo ámbito de incertidumbre.
La adecuación de cada procedimiento y de su combinación debe justificarse,
además de cumplir los restantes requisitos de R3. El rechazo no sustituye esa
justificación.

| Procedimiento guardado | Lo que contrasta y condiciona | Relación que debe conservarse |
|---|---|---|
| Bootstrap circular pareado, bloques fijos de 14 días | Recalcula la media de RMSE individuales en fechas remuestreadas de forma sincronizada; centra el ruido para contrastar ganancia mayor que δ. Las trayectorias ya ajustadas quedan fijas. | Es remuestreo temporal por bloques. No es literalmente el bootstrap estacionario de longitudes geométricas nombrado en la Tabla 4. Su pertinencia y supuestos de dependencia/estabilidad deben justificarse; no se lo renombra. |
| Influencia de la media de RMSE con Bartlett-HAC | Aproximación delta y varianza de largo plazo, con 14 rezagos; la derivada de cada raíz se calcula antes de promediar. Condiciona las trayectorias ajustadas. | No es el DM con corrección HLN registrado en la tabla. No se sustituye el margen RMSE por su cuadrado ni se presenta como DM-HLN por usar HAC. |
| t pareada entre diez semillas | Ganancias por identificador de inicialización; 9 grados de libertad, bajo independencia de inicializaciones y aproximación normal. Condiciona el calendario conocido. | No es otra prueba temporal de los errores diarios. Para un clásico fijo, la variación procede de las diez réplicas E3; no se inventan diez ajustes clásicos. |
| MCS auxiliar, Tmax al 90 % | Igualdad de capacidad predictiva sobre MSE diario medio entre semillas, con remuestreo temporal; conserva diez de diecisiete sistemas. | Es global y pertenece a la familia temporal de la Tabla 4, pero usa otro estimando y no demuestra ganancia de RMSE mayor que δ contra cada rival. Pertenencia no significa ganador único o equivalencia. |

Los 48 contrastes guardados son una evidencia retrospectiva bajo sus supuestos,
**no una batería F2 de la Tabla 4 acreditada automáticamente por sumar tres
procedimientos**. Clark–West requiere justificar la estructura anidada y su
ajuste; Giacomini–White requiere definir la capacidad predictiva condicional,
la información y el diseño de estimación. Friedman sobre series u orígenes
exige una unidad de análisis y dependencia defendibles: diez semillas de una
sola cronología no son diez conjuntos independientes. No se añaden pruebas
después de ver los resultados para conservar sólo la favorable.

El docstring y el plan numérico históricos contienen expresiones sobre una
batería «homogénea». Sus bytes permanecen intactos como registro de aquel
corte; esas expresiones no añaden un requisito literal a la guía. Las fuentes
actuales deben distinguir esa redacción histórica de la interpretación aquí
revisada.

## Supuestos, efectos y multiplicidad

La admisión de fechas, objetivos, máscaras y semillas prueba compatibilidad de
los insumos; no demuestra estacionariedad, independencia temporal o
normalidad. Los Shapiro entre semillas son diagnósticos con diez observaciones,
no validadores universales. La dependencia débil y la estabilidad aproximada
necesarias para los procedimientos temporales siguen requiriendo una
justificación pertinente para este proceso. Ningún ajuste de multiplicidad
corrige el diseño desigual, la selección previa o un p-valor inválido.

La ganancia absoluta de RMSE, la reducción relativa y sus IC conservan las
unidades y el calendario. El Cohen dz guardado es la media de las ganancias
pareadas dividida por su DE entre semillas, referido a ganancia cero; **no es
un efecto entre mercados ni el efecto normalizado respecto de la frontera δ**.
El umbral orientativo de Cohen sólo puede discutirse con ese alcance, no como
un sello de relevancia económica. La magnitud de Cliff requiere definir su
estimando y estructura de dependencia: no se aplica automáticamente un
estadístico de muestras independientes a pares o días correlacionados.

La guía pide un efecto no despreciable cuando el indicador corresponda,
IC95 % de la diferencia por encima de cero e idealmente de δ, y discusión del
coste y utilidad. Los IC nominales de este análisis son marginales, no
simultáneos. La cota Bonferroni48 de FCN que no supera δ no se oculta, aunque
Holm rechace: son construcciones de multiplicidad diferentes. Estos hechos
no completan adecuación, calibración, consistencia adicional o R3.

En la familia principal guardada se usó Holm sobre **48** p-valores, no una
selección de dieciséis después de observarlos. Para p ordenados y tamaño K,

$$
p^{\rm Holm}_{(i)}=\min\left(1,\max_{j\le i}(K-j+1)p_{(j)}\right).
$$

Cada rival conserva el máximo de sus tres p ajustados como resumen de la
regla conjunta. Superar a toda la familia requiere todas las decisiones
previstas, además de los requisitos sustantivos. Las familias secundarias
de MAE, conjuntos adicionales, ablación, subgrupos u horizontes deben
identificarse y controlarse como fueron definidas; no se mezclan para
justificar un α más permisivo. El MCS auxiliar no reemplaza la nula primaria.

## Potencia y número de pares: ilustraciones, no resultado del proyecto

La guía solicita un piloto de al menos cinco semillas para estimar la DE de
diferencias pareadas y exige potencia de al menos 0,90 por contraste, con
potencia conjunta reportada. Cinco pares pueden dar una estimación muy
imprecisa. Diez semillas es un mínimo, no una demostración de potencia.
Las unidades temporales, las inicializaciones y la selección de entrenamiento
aportan incertidumbres distintas; los 6080 pares semilla-fecha no son iid.

Como referencia normal bilateral de la guía, con α′=0,05/m,

$$
R\approx\frac{(z_{1-\alpha'/2}+z_{1-\beta})^2\sigma_d^2}{d^2}.
$$

Aquí d es la **separación alternativa respecto de la frontera nula**. Para
comparar con igualdad cero y detectar una diferencia δ, d=δ. Si la nula ya es
ganancia≤δ, se necesita una alternativa planificada g1>δ y **d=g1−δ**. Volver
a colocar δ en el denominador confundiría el margen práctico con esa
separación. La referencia unilateral usa z(1−α′), si esa dirección fue fijada
antes de evaluar. No se cambia unilateralidad después para lograr rechazo.

Ejemplos sin datos del proyecto, usando la referencia **bilateral**:

| Tamaño m | Potencia marginal objetivo | d/σd=1: R aproximado, techo | d/σd=0,5: R aproximado, techo |
|---|---:|---:|---:|
| 12, ejemplo de la guía | 0,80 | 13,74097; 14 | 54,96388; 55 |
| 12 | 0,90 | 17,19605; 18 | 68,78419; 69 |
| 16 | 0,90 | 17,94978; 18 | 71,79913; 72 |
| 48 | 0,90 | 20,79885; 21 | 83,19541; 84 |

Son aproximaciones con varianza de planificación conocida, normalidad y
pares independientes. No incluyen incertidumbre al estimar σd, dependencia
temporal, las tres pruebas simultáneas o el procedimiento completo de Holm.
Por tanto **no son tamaños recomendados para E3**. Bajo el ejemplo m=12,
potencia 0,80, la MDE estandarizada de esa referencia es 1,17222σd con
R=10 y 0,82888σd con R=20; no se ha estimado la MDE real de este proyecto.

Para el cierre futuro se necesita una alternativa y una unidad independiente
defendibles, un piloto pertinente y una simulación o dimensionamiento que
represente calendario, semillas, todas las hipótesis y la decisión conjunta.
Agregar semillas no crea calendarios nuevos ni elimina la exposición previa.
No se calcula potencia observada con la misma diferencia favorable del test.

### Resolución de las pruebas exactas de rangos

Con n pares no nulos, sin empates, el menor p bilateral exacto de Wilcoxon
por rangos con signo es 2^(1−n). Con cinco pares es **0,0625**: ni siquiera
alcanza α=0,05 sin ajuste. Para superar el **primer** umbral Holm 0,05/m:

| m | Primer umbral | Mínimo n para que ese primer rechazo sea posible | Menor p bilateral con ese n |
|---:|---:|---:|---:|
| 12 | 0,00416667 | 9 | 0,00390625 |
| 16 | 0,00312500 | 10 | 0,001953125 |
| 48 | 0,00104167 | 11 | 0,0009765625 |

Esta resolución no es potencia y tampoco afirma imposibilidad en **todos**
los pasos posteriores de Holm: otras hipótesis pueden rechazarse primero y
relajar el umbral del contraste considerado. Los ceros reducen el n efectivo,
los empates afectan las convenciones exactas y los supuestos de simetría y
dependencia aún deben justificarse. Un p unilateral tiene otra resolución,
que no se adopta como rescate posterior.

### Potencia conjunta

Los ejemplos 0,90^12≈0,28243 y 0,95^12≈0,54036 sólo valen si las doce
decisiones son independientes y tienen esas potencias marginales. **No
estiman la potencia conjunta real** de esta familia. Con potencias marginales
al menos π y dependencia arbitraria, una cota conservadora es
máx(0,1−m(1−π)); no identifica la potencia real ni sustituye una simulación
de la regla completa. El control FWER de Holm y la probabilidad de rechazar
todas las alternativas son conceptos diferentes.

`planning_illustrations.py` reproduce estos cálculos con la biblioteca estándar
de Python y nueve comprobaciones algebraicas/de admisión. Su JSON preserva
inputs, fórmulas, bilateralidad, supuestos y hash del script. **No carga
predicciones, no calcula nuevos p-valores del proyecto, no estima potencia
real ni recomienda semillas.**

```bash
python experiments/e3_revision_bloque_d_20261117/planning_illustrations.py \
  --output /tmp/e3-planning-illustrations.json
```

## Qué falta para cerrar la afirmación

Debe congelarse un protocolo futuro real antes de ejecutar sus experimentos,
con las tres pruebas apropiadas, sus supuestos, familias, alternativa y
potencia; comparación justa; conjuntos adicionales elegidos sin consultar
su desempeño; adecuación, calibración, robustez, eficiencia e interpretación
del propio modelo. Se mantienen los resultados históricos y sus 48 rechazos
condicionados sin transformarlos en un preregistro, una estimación de potencia
o una certificación de R3.
