# Resultado ejecutado: interacción local heterogénea, sin candidata nueva

La medición V2 terminó el21 de noviembre de 2026 a las 09:33:53 de Bogotá —14:33:53 UTC—. Completó150forwards por lote,300autograd y4500evaluaciones puntuales internas en2,243s, con286.863.360bytes de RSS máximo. **Cero pasos de optimizador, entrenamientos o pronósticos nuevos fuera de muestra.** Los parámetros y archivos de entrada permanecieron intactos.

El [anclaje V2 `d3b34a4`](https://github.com/jassan-bit/DeepLearning_202630/tree/d3b34a4829b54c3d7c6ba48b5f24bce83ef4b816/experiments/e3_fixed_state_gradients_20261121) se leyó públicamente a las14:33:50 UTC, antes de iniciar esa ejecución. Registra desarrollo nuevo, no un preregistro de los pilotos anteriores o una campaña confirmatoria. La asistencia de IA está declarada; la revisión técnica por agentes IA no sustituye evaluación humana o docente.

## 1. Respuesta a la hipótesis planteada

**En estos checkpoints y lotes no se fundamenta una explicación de perjuicio dominado por conflicto global frecuente de gradientes.** Predomina la alineación positiva. Sí existen conflictos locales y diferencias por partición; tampoco se demuestra su ausencia universal o durante toda la trayectoria histórica.

| Receta CPU | Snapshot | Cosenos efectivos negativos, vector completo | Mediana de coseno | Mediana de norma auxiliar efectiva / norma MSE |
|---|---|---:|---:|---:|
| V4 sigma-CRPS | best | 2/25 | .796 | .282 |
| V4 sigma-CRPS | latest | 3/25 | .717 | .268 |
| U-CRPS de esta cohorte CPU | best | 2/25 | .709 | 1.471 |
| U-CRPS de esta cohorte CPU | latest | 1/25 | .771 | 1.287 |

Los25casos son cinco semillas por cinco lotes fijos; no son ensayos independientes. Best/latest repiten tensores en4/15 checkpoints; se conservaron ambas etiquetas como estaba previsto. Las450filas son150mediciones por tres particiones, no 450 experimentos.

MSE-only tiene coeficiente auxiliar real0. Su coseno efectivo se registra ausente; la geometría del CRPS calculada sobre sus pesos es contrafactual y no una fuerza que hubiera intervenido en su entrenamiento. Una mayor norma auxiliar en U-CRPS tampoco demuestra por sí sola perjuicio.

## 2. Desplazamientos hipotéticos de Adam

Se utilizó únicamente `latest` con sus propios momentos y contador. No existe Adam de `best` cuando corresponde a otra época. Full y MSE contrafactual parten de la misma memoria histórica; **no reconstruyen cómo habría entrenado una receta sin auxiliar**.

Para d=theta_propuesta−theta_actual, el signo g_M·d>0 indica ascenso infinitesimal de primer orden de la MSE normalizada del lote. No se aplicó d ni se observó una pérdida posterior a una actualización.

| Receta, vector completo/latest | Ambos descendentes | Ambos ascendentes | Full descendente / MSE contrafactual ascendente | Full ascendente / MSE contrafactual descendente |
|---|---:|---:|---:|---:|
| V4 sigma-CRPS | 22 | 2 | 1 | **0** |
| U-CRPS CPU | 19 | 0 | 6 | **0** |
| MSE-only | 23 | 2 | 0 | 0 |

No apareció un caso global medido en el que el contrafactual MSE descendiera y la dirección completa ascendiera. Esto no demuestra beneficio predictivo: son pendientes locales, con momentos reales de la receta ya entrenada y sobre un lote propuesto distinto del paso histórico.

La pendiente completa es mayor —menos favorable localmente— que la contrafactual en3/25casos V4 y1/25U. Ese recuento tampoco equivale a cambiar de descenso a ascenso. En los 75 casos latest/global, el coeficiente de clipping fue1: no hubo clipping **en esta medición**, sin afirmar nada sobre todas las épocas históricas.

## 3. Las particiones sí importan

En latest, V4 tiene 5/25 cosenos negativos en los parámetros compartidos y 4/25 en la cabeza, frente a3/25global. U tiene1/25 y 2/25, frente a1/25global. Sus contribuciones pueden compensarse.

Por ejemplo, U no tiene ascensos completos globales, pero sí 2/25 en shared y 3/25 en head. No se generaliza «sin ascensos» a todas las particiones ni se convierten esas particiones en componentes causales independientes. Los productos por partición suman el global.

## 4. Qué explica la memoria local de Adam

Para el contrafactual MSE, con wd0, coeficiente de clipping c>0, denominadores Adam D_i>0 y correcciones por parámetro bc_{1,i}=1−beta1^(t_i+1)>0, la regla teórica en el único grupo de hiperparámetros guardado da, en aritmética real,

\[
g_M^T d_M=-lr\left[\beta_1\sum_i\frac{g_{M,i}m_i^{old}}{bc_{1,i}D_i}
+(1-\beta_1)c\sum_i\frac{g_{M,i}^2}{bc_{1,i}D_i}\right].
\]

El segundo término es no negativo. Un producto positivo exige que el primer momento almacenado contribuya adversamente y domine al gradiente actual en ese lote/estado. Un precondicionador diagonal positivo aplicado sólo al gradiente MSE actual no produciría ese ascenso.

Es una explicación algebraica de la dirección hipotética observada, **no** una identificación de por qué la RMSE histórica fue peor. Los momentos almacenados no están separados en sus contribuciones MSE/auxiliar a lo largo del entrenamiento; faltan estados intermedios para esa atribución.

## 5. Decisión después de medir

1. Se conserva la interacción local/escala/memoria como diagnóstico heterogéneo. No se declara que CRPS sea impropio o la causa del deterioro RMSE.
2. **No se propone una candidata de proyección/gating sólo con este resultado.** El conflicto global observado es limitado, el auxiliar no es uniformemente perjudicial en las pendientes y los mecanismos de protección tienen antecedentes conocidos.
3. No se promueve U-CRPS o MSE-only por una pendiente local. V3 GPU conserva su NO-GO, V4 su abandono y la aditiva sus 796 trabajos. Los gradientes U-CRPS CPU no se transfieren a la cohorte GPU original.
4. No se inicia entrenamiento o confirmación por finalizar este instrumento. R1/R2/R3 y los requisitos de potencia/datos conservan su alcance.

**Siguiente cuestión científica concreta:** distinguir límites de representación/objetivo de límites de selección temporal. El expediente ya documenta inversión entre stop-MSE y assessment para la aditiva, pero los checkpoints finales no identifican por sí solos su causa. Antes de otra propuesta debe plantearse una hipótesis falsable con variables observables al emitir el pronóstico, antecedentes y un control que pueda distinguir esas explicaciones. No corresponde aumentar épocas, cambiar lambda o elegir fechas usando este assessment.

El resultado útil inmediato es evitar invertir por defecto en una corrección de conflicto de gradientes sin fundamento de utilidad y novedad. Si no se obtiene una propuesta sustantiva defendible, la alternativa preparada de benchmark/límites sigue disponible; no equivale a R3.

## 6. Fallo técnico V1 preservado y gasto real

El primer intento, anclado en `a40f498`, se detuvo por promoción FP32/F64 de un escalar al calcular la dirección hipotética. Hizo6forwards,12autograd y182evaluaciones internas, sin pasos de optimizador. Se preservan recibo,15filas parciales y comprobación posterior de archivos. V2 corrigió sólo el clipping in-place y añadió una prueba específica; se volvió a anclar antes de ejecutarse.

El gasto total real de ambos intentos es **156forwards,312autograd y4682evaluaciones internas**, no150/300/4500. Los resultados completos de este informe proceden exclusivamente de V2. Las15filas parciales V1 coinciden exactamente con el prefijo correspondiente V2; no se mezclan como réplicas adicionales.

## Archivos verificables

- [Datos completos](results/measurements.csv), [JSON completo](results/MEASUREMENTS.json), [resumen por grupo](results/summary_by_group.csv).
- [Recibo V2](results/EXECUTION_RECEIPT.json), [admisión y estados](results/DATA_AND_STATE_PREFLIGHT.json), [recibo acumulado](results/CUMULATIVE_EXECUTION.json).
- [Intento V1](attempt_v1/EXECUTION_RECEIPT.json), [filas parciales](attempt_v1/PARTIAL_MEASUREMENTS.json), [preservación posterior al fallo](attempt_v1/PRESERVATION_CHECK.json).
- [Código de medición](measure_fixed_states.py), [código del resumen](summarize_measurements.py), [plan V2](PLAN.json), [anclaje V2](ANCHOR_RECEIPT_V2.json).

No se cambiaron parámetros, datos fuente, campañas previas, servicio, DOI, calificación o libro publicado. No se redistribuyen PT, NPZ, estados del optimizador o valores de pesos.
