# Matriz fija de extracción, calidad y brecha: catorce inclusiones focales

La matriz actualizada contiene **14 inclusiones aplicadas dentro de los mismos 30 objetivos focales**, no 31 objetivos. Conserva las trece filas previas y añade E2-0345 tras la nueva lectura primaria documentada por el adjudicador pair_c. Esta ampliación extrae ese recibo; su autor no afirma haber releído el artículo. Los cuatro archivos originales de trece estudios se conservaron byte a byte. La ampliación usa `quality_extraction_matrix_14.csv/.json` y `source_hashes_14.json`.

Los archivos CSV y JSON conservan DOI, versión, target, datos, horizonte, partición, métricas, modelos, repeticiones, HPO, pruebas, código, riesgo de fuga y equidad de comparadores, con estados y localizadores. `conocido` identifica un dato/declaración respaldado por la ficha, con sus límites; `no_verificado` significa desconocido en la evidencia consultada; `no_evaluado` identifica un campo que la extracción disponible no estudió. No se convierte un vacío en ausencia. No se construye un score agregado de calidad.

## Repeticiones: qué sostienen las fuentes

**Seis de catorce declaran conteos de nuevas ejecuciones o inicializaciones del entrenamiento.** No equivale a demostrar independencia estadística, reproducirlas ni disponer de sus enteros de semilla.

| Fila | Declaración documentada | Localizador y límite |
|---|---|---|
| E2-0109 | 10 ejecuciones aleatorias por configuración/ventana | E1F-M01; Deep Learning Models/Ec.24 y protocolo. No son nueve inicializaciones por tener nueve activos. |
| E2-0139 | 30 reejecuciones por modelo/criptomoneda | Ficha primaria E1, §4 y líneas1361–1379. Su t pareada entre 27 activos es una unidad distinta. |
| E2-0238 | 20 ejecuciones de LSTM, GRU y PSO-GRU | §3.4 p.2024/PDF14. No queda reconstruido el reinicio por horizonte ni si cada ejecución repite PSO. |
| E2-0671 | 30 inicializaciones aleatorias de test | p.170 §3 y tabla1 p.171. Separarlas de los 100 runs de CV; la dispersión tabulada es varianza. |
| E2-0867 | 20 corridas por red base, con media/DE | §5.3/tabla2; S1. Las 40 del D-TCN avanzado tienen otro alcance. |
| E2-0345 | 10 semillas finales por activo para el modelo propuesto | §3.3; recibo pair_c. Repeticiones de cada comparador desconocidas; no extenderles automáticamente las diez semillas. |

En las otras ocho filas no hay verificación de repeticiones suficiente en las fichas consultadas. E2-0064 y E2-0593 no las evaluaron después de su exclusión estricta previa por target; E2-0253 tampoco las describe en su recibo limitado. **No es defendible escribir que ocho artículos carecen de repeticiones.** Tampoco son equivalentes folds, ensayos HPO, días, activos y nuevas inicializaciones.

## Cuatro brechas sustentadas y respuesta concreta de E2

**1. Falta de un contrato comparable, no falta general de estudios de volatilidad.** E2-0011 emplea varianza desde retornos de diez minutos; E2-0139, desviación de treinta días; E2-0671, varianza de tres horas; E2-0064, volatilidad implícita; E2-0593, desviación condicional. E2-0284 usa ventanas de siete días sobre retornos de treinta minutos: compartir el número siete no iguala el objetivo. E2 responde con σ7, D7×5, h1 y soporte temporal común (§48). La brecha permite justificar una evaluación bajo información y etiqueta comunes; **no permite normalizar sus errores externos como si midieran el mismo efecto** ni afirmar que ningún artículo estudia el problema.

**2. Separar selección, aleatoriedad de entrenamiento y evidencia temporal.** Las seis filas anteriores documentan repetición con alcances diferentes; E2-0345 no acredita el mismo presupuesto de semillas para cada comparador. E2-0113 reporta rolling/validación pero no semillas verificadas; E2-0284 tiene cinco folds, que no son cinco inicializaciones. E2 distingue 18 oportunidades HPO y dos semillas de selección de las seis ejecuciones finales y sus fechas comunes (§48). Esa separación hace evaluable la incertidumbre de entrenamiento. **R=6 sigue siendo una decisión operativa:** el piloto condicional alcanzó aproximadamente **12,3%** de potencia para la ganancia planificada, inferior a 80%; no significancia no prueba equivalencia (§48, líneas136–138).

**3. Definir el ámbito temporal de todo el pipeline y el esfuerzo de los comparadores.** E2-0238 deja sin reconstruir el ajuste de RobustScaler/selección de variables y las recetas base; E2-0109 y E2-0284 presentan interpolación y alcance del escalado no completamente establecidos. Son riesgos de reporte/implementación, **no demostraciones de fuga ejecutada**. E2 usa validación temporal, purga/embargo, transformaciones ajustadas en entrenamiento y predicciones/checkpoints conservados (§48). E2-0867 sí documenta 250 ensayos por red base: es un contraejemplo que evita declarar ausente el control del ajuste en toda la literatura. Igual número de ensayos tampoco garantiza igual cómputo o información.

**4. Convertir diferencias puntuales en afirmaciones proporcionales a la evidencia.** E2-0113 conserva casos donde LSTM no domina MLP/HAR; E2-0253 sólo acredita un promedio de pérdidas heterogéneas, que no es RMSE; E2-0238 conserva inconsistencias entre tablas. E2-0345 aporta un contraejemplo adicional: tabla7 informa BTC MSE −2,88% y MAE +4,82% respecto a GARCH para el propuesto; mejora una pérdida y empeora la otra. Son cambios porcentuales, no errores absolutos. Sus marcas DM corresponden a QLIKE, no a significancia separada de MSE o MAE. La tabla6 fue leída por el adjudicador primario, pero sus celdas no están transcritas en el recibo: la matriz no inventa esos valores. E2 reporta pérdidas sobre casos comunes, comparadores simples, inferencia pareada, multiplicidad y bootstrap temporal con 20.000 sorteos (§56). Estos controles mejoran trazabilidad y resolución numérica, pero **el periodo de prueba ya era conocido desde E1, δg no se definió previamente y la potencia fue insuficiente** (§61). La contribución defendible es un contraste retrospectivo auditable; no una confirmación prospectiva de superioridad práctica.

## Código y reproducibilidad

La ficha S1/E2-0867 identifica código público del autor y un commit. E2-0345 añade un enlace editorial a código y un commit histórico del autor; la correspondencia de la versión actual con el artículo y la diferencia entre salida log-varianza previa y softplus de la prosa actual siguen sin conciliar. E2-0238 declara código a solicitud, que no se recibió ni inspeccionó. En las demás filas se conserva `no_verificado` o `no_evaluado`, según el alcance real del recibo. **No se concluye que no haya código público.** En particular, una receta no transcrita por esta revisión no demuestra que el artículo no la publique, y un enlace a un PDF no acredita una implementación pública.

## Alcance del producto

El hash de cada insumo consultado figura en `source_hashes_14.json`; identifica fichas/documentos y el recibo de nueva lectura primaria pair_c, no un binario original materializado. Las métricas permanecen dentro de su fuente, versión, población, escala y horizonte. No se combinan las catorce inclusiones en un efecto, no se convierten sus diferencias en una selección anterior al catálogo y no se altera el protocolo histórico estricto de E1. Esta matriz ofrece un producto revisable del requisito de calidad; no cierra la elegibilidad global ni garantiza una nota concreta.
