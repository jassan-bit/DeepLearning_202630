# Validación de la decisión principal y dos de tres adicionales

Este paquete usa **sólo pérdidas sintéticas con verdad conocida**. No entrena
pronosticadores, descarga precios, abre reservas o estima potencia de mercado.
El funcional es la media de RMSE de los modelos ajustados por semilla. El
protocolo está en `protocol.json`; ejecutar el programa no permite cambiar sus
escenarios, niveles, tamaños o reglas.

## Fundamento de la composición

Sea S[p,r] la afirmación poblacional de superioridad práctica sobre el rival r
en el panel p. La parte positiva de la regla requiere todas las afirmaciones
del panel principal y, **para cada rival**, al menos dos afirmaciones verdaderas
entre los tres adicionales. No exige que los mismos dos paneles ganen frente a
todos los rivales. La decisión observada añade las tres pruebas, dos nulos,
Holm120 por panel, dirección, efecto, intervalos y pantalla adversa Holm180.

Si esa afirmación conjunta es falsa, existen testigos fijados **por la verdad
poblacional**, sin mirar los p observados:

1. Si hay una afirmación falsa en el principal, aceptar la decisión requiere
   un falso positivo para esa afirmación dentro del panel principal.
2. En otro caso, algún rival tiene superioridad práctica en como máximo un
   adicional. Se fijan dos adicionales donde su afirmación es falsa. Aceptar
   requiere un falso positivo en al menos uno de esos dos paneles.

Si los p escalares pertinentes son válidos y Holm controla la familia de sus
nulos verdaderos a q[p], resulta la cota
`max(q[0], max(q[i]+q[j] para 1<=i<j<=3))`. Con q[p]=0,025 es **0,05**, sin
independencia entre paneles, rivales o procedimientos. Las condiciones
adicionales sólo restringen el evento de aceptar.

La igualdad o frontera del funcional escalar ofrece un testigo delta/HAC;
el argumento no supone que el nulo condicional vectorial de GMM sea idéntico
al nulo de media-RMSE. La prueba de Holm para el subconjunto de nulos escalares
válidos tampoco requiere que el resto de p corresponda al mismo nulo:
antes del primer rechazo de un nulo escalar verdadero quedan al menos todos
los nulos escalares verdaderos en la familia. Esto no excusa errores en otros
tests ni demuestra validez de sus supuestos.

`logic.py` comprueba exhaustivamente **48 pares** locales de verdad falsa y
decisión aceptada. Las pruebas de software verifican sus **960 inserciones** en
la familia de 20 rivales. Esta implicación booleana es exacta; la superuniformidad
de los p temporales es una condición estadística, no un resultado de ese
enumerador. Las cotas marginales CP95 antiguas menores que 0,05 **no** se
reinterpretan como q<=0,025. La validación nueva mide directamente el falso
reclamo de la decisión completa en los DGP prefijados.

El alcance de la prueba es la conjunción estadística de dirección y margen.
No acredita todas las condiciones científicas de R3, ausencia poblacional de
inferioridad, adecuación, robustez, novedad, custodia o una reserva real.

## Protocolo fijado antes de las réplicas formales

- Cuatro paneles y **21 sistemas/20 rivales**, tres métodos y dos nulos:
  **120 posiciones Holm por panel**, más la pantalla adversa adicional de 180.
- Celda **R10/T768/B8191**, bloque estacionario32, HAC48, horizonte1 y referencia
  fixed-b de 49.999 sorteos. Alfa por panel **0,025** y margen práctico **2 %**
  de la media de RMSE individuales; no se sustituye por RMSE agregado.
- **512 réplicas por escenario, nueve escenarios: 4.608 réplicas externas**.
- La semilla interna bootstrap `2026112203` es común y está fijada antes de los
  datos en todos los paneles y réplicas. La semilla de referencia es
  `2026112202`. El muestreo externo de datos utiliza una semilla independiente
  y una correspondencia reproducible entre escenario e índice de réplica.
  El alcance es condicional a este algoritmo de aleatorización interna fijo.
- Presupuesto declarado **5.400 s CPU**, como máximo cuatro trabajadores y
  BLAS de un hilo por proceso. Los chequeos entre lotes son suaves: el último
  lote y su contabilidad pueden superar el instante de corte. El consumo
  observado y cualquier incertidumbre se reportan, sin fingir un corte duro.

| Escenario prefijado | Papel |
|---|---|
| EPA completa IID | Nulo de igualdad en toda la familia |
| EPA completa AR(0,5), factor común entre paneles | Dependencia temporal y entre paneles |
| Un rival en la frontera práctica del principal, ruido bajo | Frontera no bloqueada trivialmente por la guarda de efecto |
| Dos adicionales en la frontera práctica, ruido bajo | Estructura parcial que ejercita la regla dos-de-tres |
| Dos fronteras adicionales, AR(0,85) y acoplamiento negativo | Estrés de dependencia fuerte y entre paneles |
| Cancelación condicional con dos grupos heterogéneos de semillas | Media escalar nula, nulo vectorial condicional distinto |
| Quiebre de varianza | Adverso fuera de la hipótesis de estacionariedad, identificado antes de resultados |
| Inversión de ordenación de riesgos positivos | Media-RMSE y raíz del MSE agregado tienen órdenes opuestos |
| Alternativa positiva fuerte del25 % | Control de no vacuidad, no potencia de mercado |

El caso de quiebre de varianza **no se elimina** si da un resultado adverso:
se exige que satisfaga los mismos criterios numéricos fijados, sin convertir
eso en un teorema de estacionariedad bajo un quiebre.

Las pérdidas lognormales tienen medias poblacionales conocidas exactamente.
Hay tres clases estocásticas de reglas en los casos homogéneos, conservando
todos los nombres y posiciones Holm. Las trayectorias de 19 rivales no-testigo
se repiten; las semillas son perfectamente dependientes en siete casos y forman
dos grupos heterogéneos en los otros dos. La compresión por canales idénticos
ahorra cómputo, **sin borrar hipótesis**. No se presentan estos escenarios como
validación de 20 reglas independientes o de toda heterogeneidad posible.

## Criterios e incertidumbre Monte Carlo

Se usa una **única familia de 107 afirmaciones**, con error de confianza
Monte Carlo global0,05 y Bonferroni0,05/107 por límite unilateral exacto CP:

* Ocho cotas del error de aceptar falsamente la decisión completa: cada cota
  superior debe ser <=0,05.
* Dieciocho coberturas de CI99 —dos intervalos, diferencia y ganancia, de un
  testigo poblacional fijado por escenario—: cada cota inferior debe ser
  >=0,95. Es un piso para los CI99, **no una certificación de cobertura exacta
  nominal99 %**. Las coberturas CI95 se reportan como diagnóstico.
* Ochenta potencias por contraste bajo la alternativa fuerte: cotas inferiores
  >=0,90, más la potencia conjunta con cota inferior>=0,80.

N512 permite hasta diez faltas de cobertura en cada una de esas afirmaciones
de CI99 para certificar el piso95 %. La capacidad binomial se calcula
analíticamente antes de observar resultados. Las cotas simultáneas se refieren
a esas 107 afirmaciones y a estos escenarios, no a procesos financieros
arbitrarios. Una ejecución incompleta nunca recibe aprobación.

Los diagnósticos de tamaño de pruebas bilaterales cuentan nulos de **igualdad o
frontera** realmente verdaderos. Un rechazo que indica correctamente que la
candidata es peor no se etiqueta como error tipoI. Para el nulo compuesto de
superioridad falsa se exige además la dirección favorable observada. El caso
de cancelación no trata el rechazo GMM bajo su nulo vectorial falso como un
error de tamaño para ese nulo.

## Ejecución, reanudación y exportación

Después de publicar el código y el protocolo fijados, con NumPy2.3.5,
SciPy1.17.0, Python y compilador CPU compatibles:

```bash
python -m joint_validation.cli freeze --output /ruta/caso \
  --protocol joint_validation/protocol.json --source-commit COMMIT_PUBLICADO
python -m joint_validation.cli run --case /ruta/caso --resume \
  --max-replicates 4608 --seconds-budget 5400 --workers 4
python -m joint_validation.cli status --case /ruta/caso
python -m joint_validation.cli export --case /ruta/caso --output respaldo.zip
python -m joint_validation.cli restore --archive respaldo.zip \
  --case /otra/ruta/caso --source-commit COMMIT_PUBLICADO
```

`freeze`, instalación, `status`, `export` y `restore` no ejecutan réplicas.
`run` es la activación explícita. Un chunk limita cómputo, no cambia N ni los
criterios. El orden del ledger es round-robin: escenarios dentro del índice
externo de réplica. Cada réplica se guarda antes de completar el siguiente
lote. Un bloqueo impide dos escritores concurrentes. Se rechazan cambios de
protocolo/código, duplicados, cadenas inválidas o líneas truncadas sin borrar
la evidencia original.

Cada fila conserva semilla, verdad, 80 decisiones, p crudos y ajustados del
testigo, sus intervalos/coberturas, hashes de datos/resultados completos
regenerables y costes. No guarda miles de informes completos redundantes.
Las réplicas adversas no se filtran. El ZIP conserva inventario y hashes, y la
restauración valida el contenido antes de instalarlo en una carpeta nueva.
La importación es portable; añadir nuevas réplicas exige la misma cohorte
numérica de Python menor, NumPy, SciPy y plataforma. Una nueva cohorte no se
mezcla silenciosamente con la original.

La contabilidad de cada chunk mide el tramo de ejecución de réplicas y ciclo
del pool, incluyendo CPU del padre, trabajadores, preparación de sus módulos
y compilador hijo mediante `RUSAGE_CHILDREN`. Instalación, verificación previa
del caso, exportación y resumen final quedan fuera de ese tramo y se distinguen
del cómputo de Monte Carlo. Una sesión sin cierre conserva incertidumbre y una
cota inferior de consumo; no se inventa el tiempo perdido.

Las comprobaciones de software se ejecutan con:

```bash
python -m unittest -v joint_validation.test_joint_validation
```

Son fixtures y pruebas de integridad, no réplicas de calibración. La asistencia
de IA se declara en la implementación, revisión y protocolo.
