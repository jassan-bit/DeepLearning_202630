# Seguimiento de los controles negativos

Este suplemento conserva intacta la auditoría original y los resultados del
benchmark. Investiga TSMixer e iTransformer porque sus controles originales
retuvieron asociación con el objetivo. Esa selección se hizo después de conocer
los resultados y se declara como diagnóstico posterior.

La revisión de código documentada en `trace_review.json` no encontró una ruta
explícita desde las etiquetas de prueba hacia la fase de entrenamiento. El
optimizador recibe las 655 ventanas de entrenamiento y las etiquetas de
entrenamiento permutadas; conserva receta, épocas, inicialización y escalado.
Los dos modelos reciben solamente la ventana D7 y estadísticas de esa ventana.
Esta revisión estática no prueba ausencia de fuga ni equivalencia al azar.

El archivo `plan.json` se escribió y se vinculó al hash del ejecutor antes de
iniciar seis ajustes adicionales: tres permutaciones de bloques prefijadas por
dos modelos. No se descartan permutaciones por filas fijas ni por sus métricas.
No se seleccionan otras recetas, semillas o épocas con los datos de prueba.

## Alcance de la comparación con la media de entrenamiento

La referencia es una predicción constante igual a la media del objetivo de
entrenamiento, que se mantiene invariante al permutar las etiquetas. Una ganancia
de RMSE positiva favorece al control entrenado. El margen es el delta de RMSE
heredado del protocolo: 0.00025772842664981106.

El resumen registra ganancia e intervalo por ajuste. La comparación de
equivalencia, si es calculable, corresponde a la media de los tres nuevos
controles fijos de cada modelo, con bootstrap por bloques de 14 días y corrección
de Holm sobre dos extremos. Sus intervalos describen la incertidumbre de
calendario condicionada a esas funciones entrenadas. No representan la
variabilidad de todas las permutaciones posibles, de la selección de
hiperparámetros ni de nuevas particiones de entrenamiento. La correlación se
reporta como diagnóstico descriptivo; no se inventa un margen de equivalencia
para correlación.

El control tiene dos limitaciones relevantes. La arquitectura, hiperparámetros
y épocas ya habían sido seleccionados usando etiquetas reales de desarrollo;
estas repeticiones no permutan y repiten el HPO completo. Además, sigma7(t)
contiene siete retornos de los cierres t-7 a t: seis de esos retornos se conocen
en la ventana legítima que termina en t-1. La asociación residual puede reflejar
sesgo inductivo, la relación parcial conocida o un problema que requiera más
investigación; por sí sola no demuestra cuál de esas explicaciones corresponde.

Por tanto, no se acepta integralmente la familia original de trece modelos
basándose en este seguimiento de dos. Un resultado adverso o inconcluso permanece
registrado; la ausencia de rechazo tampoco se convierte en equivalencia.

## Resultados ejecutados

Se completaron los seis ajustes previstos y el resumen, sin errores ni cambios
de permutación. La equivalencia de la media fija de tres controles no se
estableció en ninguna red: ambos valores p de TOST ajustados por Holm fueron 1.

| Modelo | Ganancia media de RMSE | IC90 % condicional por calendario | Margen |
|---|---:|---:|---:|
| TSMixer | 0.000606372 | [0.000410671, 0.000845935] | ±0.000257728 |
| iTransformer | 0.000627953 | [0.000506498, 0.000752111] | ±0.000257728 |

Cinco de los seis intervalos descriptivos de correlación estuvieron por encima
de cero. TSMixer con la tercera permutación tuvo rho=0.0173 e intervalo que
incluye cero. Se conserva el resultado adverso a la aceptación de rendimiento
cercano a la constante; no se cambia el margen ni se ejecutan semillas hasta
obtener aceptación. Estos intervalos describen las funciones ya entrenadas,
no un muestreo suficiente de la distribución completa de permutaciones.

Un segundo plan, congelado después de completar lo anterior y antes de sus
propias inferencias, evaluó las dos funciones de inicialización fija sin
entrenarlas. Verificó el hash exacto de los estados iniciales originales y la
invariancia de parámetros y buffers durante la evaluación. TSMixer sin
entrenamiento obtuvo RMSE=0.0542958 y rho=0.2827; iTransformer obtuvo
RMSE=0.0117881 y rho=0.0100. Ambas funciones fueron peores que la constante en
RMSE. La correlación inicial de TSMixer apoya investigar sesgos de la función,
pero estas observaciones no explican íntegramente la señal posterior a las
permutaciones, especialmente en iTransformer, ni cambian el criterio de
aceptación. El diagnóstico no demuestra fuga ni permite certificar su ausencia.

El libro acumulativo registra 337.626356 segundos para los seis ajustes y su
resumen, y 6.895862 segundos para las dos funciones iniciales, sin nuevos
ajustes en esta segunda etapa. Se mantienen los recibos, predicciones y fuentes
con sus hashes en `results/`, `predictions/`, `summary.json` e `initialization/`.

El subpaquete `reanalysis_portable/` reproduce el resumen estadístico sin
repositorio, PyTorch ni pesos. Su ejecución independiente fuera del repositorio
recalculó los 10.000 remuestreos y comparó 80 valores numéricos, además de todos
los campos restantes, con diferencia máxima de 0.0. El recibo de esa prueba está
en `portable_qa_result.json`; consumió 9.715501 segundos adicionales, sin ajustes
ni inferencias de modelos. Este alcance acredita reanálisis portátil de los
pronósticos incluidos y conserva la aceptación integral como pendiente.

## Ejecución

El código está en `scripts/e2_negative_followup.py`. La preparación sólo revisa
fuentes y congela el plan. La ejecución científica exige una reserva activa en
el libro acumulativo de 84 horas y conserva fallos e intentos; no realiza
reintentos implícitos. Los pesos adicionales se conservan bajo
`work/e2-negative-followup-20261116`; los pronósticos, hashes, registros de fase y
resumen quedan junto a este archivo. Ninguno reemplaza los controles originales.
