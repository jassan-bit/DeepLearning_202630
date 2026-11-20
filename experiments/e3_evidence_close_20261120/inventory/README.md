# Conciliación documental de diseños, selecciones e intentos de E3

Consolidación **retrospectiva del 19 de noviembre de 2026, America/Bogota**,
sobre `main` `29f9a26f1fd05484c6581ea6538ad7cb5a4781cb` y los resultados
V3 `8eaa581bc34b615a8507c9e384a3e4be29084720`. El sufijo del directorio
usa la fecha UTC. Se declara asistencia de IA. Esta auditoría sólo lee
documentos, metadatos y CSV guardados: cero ajustes, pronósticos o aperturas
de NPZ/checkpoints. No retrofecha planes, reinicia presupuestos, acredita
R1/R3 o calcula una corrección del sesgo de selección.

## Qué queda conciliado

La expresión histórica «Nd global pendiente» se sustituye por un inventario
explícito **dentro del corpus documentado**, con unidades distintas:

| Unidad | Conteo documentado | Interpretación |
|---|---:|---|
| Variantes neuronales funcionales implementadas | **15** | Cinco estructuras iniciales de atención, tres cabezas v2, tres por momentos y cuatro atómicas. |
| Variantes neuronales distintas entrenadas en mercado | **12** | Dos de v1, tres de v2, tres por momentos y cuatro nuevas atómicas. |
| Variantes neuronales sólo comprobadas artificialmente | **3** | Retirar penalización, saturación o contenido por separado. |
| Controles analíticos/de transformación examinados | **4** | Piso, piso más exceso medio de ajuste, M00 proyectado y full proyectado. |
| Propuestas neuronales descritas pero no implementadas | **3** | Auxiliares separados inspirados en Faithful, plug-in Jensen por momentos y q=9m² fijo. |
| Entradas documentadas con identidad propia | **22** | 15 + 4 + 3; no son 22 redes entrenadas. |
| Referencia conceptual con identidad aún ambigua | **1** | «Su propia media sola» del comparador Faithful: no tiene una especificación congelada que permita deduplicarla con la aditiva. |
| Ensayos HPO observados | **183** | 168 de la auditoría y 15 del piloto V3; no son 183 arquitecturas. |
| Registros de trabajo completos en los alcances declarados | **844** | 18 neuronales históricos + 796 trabajos de auditoría + 30 neuronales V3. |
| Ajustes neuronales completos dentro de esos alcances | **824** | 18 + 776 + 30. Se conservan además cuatro ARIMA y 16 trabajos sin optimizador. |

El intervalo **22–23** expresa únicamente la posible identidad adicional de
esa referencia incompletamente descrita dentro del corpus. **No es un límite
superior de todas las ideas informales que pudo considerar el equipo.** Se
cierran las identidades observables; no se inventa un historial externo o sin
registro. El inventario de comparadores de E2 se ofrece aparte y no se mezcla
con las variantes del candidato E3.

## Deducciones y equivalencias

`full`/M11 se cuenta una vez como N01; M00 es N05. Los M01/M10 son los
controles sin penalización/sin saturación. Las cuatro intervenciones iniciales
no se suman otra vez a la completa entrenada en v1. La referencia aditiva N06
se repite en v2, estabilidad, auditoría y piloto V3: conserva una identidad
funcional. Sus **93 ajustes** observados incluyen una corrida v2, diez de
estabilidad, 76 de auditoría y seis del piloto nuevo; no representan 93 diseños.
Las cuatro recetas atómicas agregan 24 ajustes del piloto, seis por receta.

`floor_plus_fit_mean` y `floor_plus_mean_excess` nombran la misma regla. La
media se estima sólo con ajuste; es un ajuste estadístico aunque no sea un
entrenamiento neuronal. Los 1210/1211 parámetros de M00/full proyectados son
**heredados**: la proyección añade cero pesos. El control full proyectado,
documentado en la continuación de v1, faltaba en el subtotal antiguo de tres
controles; se incorpora sin editar aquella fuente.

Se conservan las diferencias de pérdida como recetas: λ_NLL o λ_CRPS igual a
cero no se oculta como si fuera la misma receta conjunta. La menor capacidad
de `fixed_probs` —1236 frente a 1262— sigue explícita. El plug-in atómico N15
no es el plug-in Jensen de una cabeza de momentos P02: tienen operadores,
capacidad y objetivos distintos.

## Selección, parametrizaciones y decisiones numéricas

`selection_opportunities.csv` conserva los **183 ensayos HPO reales** y sus
ganadores por validación. El subconjunto de la aditiva en la auditoría tiene
12 ensayos —tres tasas en cuatro conjuntos—; el piloto V3 tiene 15 —cinco
recetas por tres tasas—. Los 61 grupos de selección incluyen los 56 pares
conjunto/red de la auditoría y las cinco recetas del piloto. No se vuelve a
elegir con el error de prueba o con el resultado de las 55 fechas finales.

No se condensan todos los grados de libertad en ese 183. También se documentan:

- Las tasas 0,0003, 0,001 y 0,003 y sus ámbitos; repetir una tasa en otro
  conjunto es otra oportunidad de selección, no otro mecanismo.
- El igualado de sesgo inicial de v2 a la media de exceso de ajuste, frente
  al constructor nativo usado por la auditoría y el piloto V3. No se supone
  que las curvas históricas y nuevas compartan toda la receta de inicialización.
- Las 116 fechas históricas de validación, la validación por conjunto de la
  auditoría y el nuevo bloque de selección de 54 fechas; no se intercambian.
- La cuadratura artificial de órdenes 64/128/256 que llevó a N=128: selección
  de precisión numérica sin datos de mercado, no tres arquitecturas nuevas.
- Las configuraciones artificiales de prueba con CNN8/LSTM12, CNN8/LSTM4 y
  CNN6/LSTM8; los errores de configuración/input rechazados son casos de
  software, no nuevos candidatos de mercado.
- Los seis pasos SGD escalares y las 100 leyes finitas de los contraejemplos:
  análisis matemático artificial, no ajustes neuronales de mercado.

El número total de oportunidades humanas/IA, criterios y parametrizaciones
informales no puede reconstruirse a partir de un repositorio si nunca se
registró. No se declara que 12, 15, 22 o 183 agoten ese número ni produzcan
una estimación numérica del sesgo del ganador.

## Intentos, fases, interrupciones y desenlaces adversos

`attempt_ledger.csv.gz` contiene 844 registros con identidad y fuente. Los
900 renglones históricos de recursos se agrupan en **796 trabajos**, porque
el autoencoder tiene tres fases que comparten un presupuesto. No se cuentan
900 ajustes. Los 796 se descomponen en 168 HPO, 560 finales, 48 curvas y
20 clásicos/analíticos; estos últimos contienen cuatro ajustes ARIMA y
16 trabajos sin optimizador. Las 76 corridas de la aditiva dentro de la
auditoría son 12 HPO, 40 finales y 24 curvas de tamaño reducido.

Los tres pilotos de mecanismos conjuntos conservaron sus desenlaces
desfavorables: full pierde frente a M00; la receta half-t con NLL no supera
la heurística; la conjunta por momentos tampoco. La nueva V3 `atom_crps`
queda **2,7043 % peor que `atom_mse_only`**, y su regla del 2 % sigue no
superada. El mejor control no se convierte retroactivamente en el candidato
primario ni en un resultado confirmatorio.

`incidents.json` conserva seis clases de incidentes de admisión, transporte
y trabajador, cuatro de ellas con extractos terminales de consolas aportadas
por el autor y checksum de su archivo. El registro final de fallos de modelo
de la auditoría es vacío porque acabó completando sus trabajos; esto no borra
esas interrupciones. No puede deducirse de cuatro capturas el número total
de invocaciones físicas o reinicios.

La tabla de recursos histórica retiene **tres fases con cómputo perdido
desconocido en segmentos de caída** y un máximo de sobrepaso de plazo de
**35283,630231 s que incluye tiempo de desconexión**. No son 9,8 horas
medidas de cómputo GPU. No se reemplazan esas ausencias por cero. El piloto
V3 conserva por separado sus 30 recibos completos y sus plazos originales.

## Fuentes y reproducción

`sources.zip` conserva 21 fuentes propias del proyecto —bitácoras, planes,
configuraciones, pruebas, recibos y CSV—. No contiene artículos completos,
resúmenes bibliográficos íntegros, README ajenos, pesos o arreglos de mercado.
`source_manifest.json` fija las URL por commit, tamaños, SHA-256 y SHA de blob
Git. Los 21 blobs se cotejaron con GitHub por lectura; las fuentes originales
siguen intactas.

```bash
python build_inventory.py
python verify_inventory.py
```

Ambos comandos usan únicamente la biblioteca estándar. El verificador
reconstruye las salidas en un directorio temporal, coteja los bytes, las
identidades, las fases/ajustes y la selección HPO declarada. La verificación
no importa PyTorch ni realiza ajuste, inferencia o remuestreo de mercado.

La consolidación documental está terminada con las fronteras expuestas. Para
una confirmación futura deberán preservarse estos registros y añadirse
nuevas decisiones con fecha real; el conteo no restaura cegamiento ni
sustituye potencia, preregistro, procedencia o defensa del estudiante.
