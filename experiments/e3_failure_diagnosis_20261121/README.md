# Primer paso hacia una candidata: diagnóstico de los fallos guardados

Fecha: 21 de noviembre de 2026, America/Bogota. **Diagnóstico de desarrollo completado; no nueva candidata, entrenamiento o confirmación.** Asistencia de IA declarada.

Se analizaron las predicciones ya existentes de V3 GPU y V4 CPU por separado, y veinte curvas CPU. Las fuentes pertenecen al corte [`5f29ab5`](https://github.com/jassan-bit/DeepLearning_202630/tree/5f29ab51e1bb3e79d972377b6d682d1785bdc298); la decisión previa de candidata está en el [paquete de preparación](https://github.com/jassan-bit/DeepLearning_202630/tree/4695589c98bf50a04e68846c986d626faf2470f6/experiments/e3_candidate_battery_preflight_20261121).

## Resultado observado

| Cohorte y comparación | RMSE media candidata/control | Cambio RMSE | Semillas con peor RMSE | MAE candidata/control |
|---|---:|---:|---:|---:|
| V3 GPU:atom_crps / atom_mse_only | .002129915768 / .002073832858 | +2,7043% | 3/3 | .001591685259 / .001644274600 |
| V4 CPU:sigma_crps_v4 / mse_only_control | .002087524306 / .002057824526 | +1,4433% | 5/5 | .001621356164 / .001608359945 |

Ambas evaluaciones cubren55 fechas de desarrollo ya expuestas, del 7 de septiembre al 31 de octubre de 2024. Compartir fechas no permite combinar las cohortes: cambian semillas y contexto de ejecución. Las tres y cinco semillas no son ocho réplicas de una única receta.

**V3 reduce MAE pero aumenta RMSE.** Tiene mayor error cuadrático medio en25/55 fechas y menor en30. Las cinco fechas con mayor perjuicio suman179,7% del exceso MSE neto, porque otras lo compensan. No es179,7% del RMSE ni una probabilidad. Su error medio observado es−.000264484, frente a−.000056299 del control. El patrón es compatible con un compromiso entre errores frecuentes y errores grandes, pero no identifica su causa.

**V4 empeora ambas métricas.** Su error cuadrático medio por fecha es mayor en37/55 fechas. En una partición algebraica balanceada de los errores, la interacción semilla×fecha es el mayor componente positivo de su diferencia MSE. No se deduce de ello una varianza poblacional, significancia o causa de inestabilidad.

## Qué aporta la descomposición

Para e_st=predicción−objetivo se separan media global, componente por semilla, por fecha e interacción. En la cuadrícula balanceada:

\[
\overline{e^2}=\mu^2+\overline{a_s^2}+\overline{b_t^2}+\overline{c_{st}^2}.
\]

Es una identidad descriptiva de la **MSE secundaria** en sigma7², no un ANOVA inferencial. La métrica primaria sigue siendo la media de RMSE individuales; no se reemplaza por la raíz de una MSE agregada. Se conserva también la partición MSE=bias²+varianza centrada dentro de cada trayectoria. Ambas responden a particiones distintas y no se confunden.

Los ocho cortes de calendario/mediana de objetivo están etiquetados como posteriores sobre desarrollo. No se convierten en reglas de despliegue o pesos para mejorar estas55 fechas.

## Qué dicen realmente las curvas CPU

- 16/20 fases llegaron al máximo de40 épocas y4 terminaron por parada temprana. **Cero terminaron por presupuesto temporal y cero tuvieron reanudación.** Duraron1,19–3,54 s frente a120 s nominales por fase. El límite de épocas sigue siendo otra condición; esto no demuestra convergencia.
- En la semilla 20261125, V4 tuvo mejor época 3 y terminó en11; U-CRPS tuvo mejor época 2 y terminó en10; MSE-only llegó a40 y eligió40. Es una diferencia observada de trayectoria/selección, no permiso para cambiar retrospectivamente paciencia o épocas.
- V4 y U-CRPS pierden el mejor stop-MSE frente a MSE-only en5/5 semillas. La aditiva tiene mejor stop-MSE que MSE-only en5/5, pero peor RMSE media en assessment: existe una inversión de ranking entre esos dos calendarios, sin causa identificada.
- `train_loss` de los auxiliares incluye MSE+CRPS; la validación guarda MSE puntual. Compararlos como brecha de ajuste sin descomponer sería incorrecto. Las escalas sY y sU son medias del ajuste, no desviaciones estándar; no se dividen dos veces.

## Hipótesis priorizada y siguiente acción

**H_interacción:** la señal auxiliar puede modificar la actualización de la media —por dirección, magnitud, clipping, estado de Adam o selección de checkpoint— y no compensar ese cambio en RMSE. El diagnóstico motiva examinarla; no la demuestra.

La acción concreta siguiente es **instrumentar componentes y gradientes sobre estados de desarrollo**, usando los checkpoints existentes verificados, sin cambiar sus pesos. Hay que registrar MSE y CRPS por separado, gradientes en parámetros compartidos y cabeza, norma/coseno y un desplazamiento hipotético calculado con la regla y el estado coherente de Adam. No se observa o reproduce un paso histórico real. No sirve repetir el ranking del assessment.

Ese estudio requiere una ejecución nueva de cálculo de pérdidas/gradientes; **no se ejecutó aquí**, no cambia el NO-GO y no necesita empezar entrenando otra red. Su diseño y límites están en [MECHANISM_MEASUREMENT_PLAN.json](MECHANISM_MEASUREMENT_PLAN.json). Ya se verificaron los hashes de veinte checkpoints y del NPZ original, sin cargarlos. Falta inspeccionar el esquema seguro real y admitir el acceso limitado a las entradas previstas. Si eso falla, se informa el bloqueo; no se recrean silenciosamente las campañas antiguas.

Sólo si se demuestra un mecanismo relevante, se evalúa una adaptación que lo controle y tenga una diferencia sustantiva defendible. [MECHANISM_HYPOTHESIS.md](MECHANISM_HYPOTHESIS.md) contrasta esa posibilidad con antecedentes conocidos y explicita su regla de descarte. No se selecciona V5 ni se acredita R1 por llamar nueva a una proyección de gradientes.

## Archivos y reproducción de este diagnóstico

- [Código determinista](diagnose_saved_predictions.py), sin imports de redes o RNG.
- [Plan de análisis y44 fuentes](results/ANALYSIS_PLAN.json); es un registro del cálculo actual, no preregistro del desarrollo ya expuesto.
- [Resultados](results/DIAGNOSTIC_RESULTS.json), [descomposición por semilla](results/paired_seed_decomposition.csv), [contribuciones por fecha](results/date_error_contributions.csv), [cortes descriptivos](results/development_slices.csv), [curvas](results/training_curve_diagnostics.csv).
- [Recibo de ejecución](results/EXECUTION_RECEIPT.json) y [resumen visual](excess_mse_by_date.svg).

Desde un checkout que contenga los datos fuente fijados, ejecutar en un directorio nuevo:

```bash
python experiments/e3_failure_diagnosis_20261121/diagnose_saved_predictions.py \
  --source-root . --output /tmp/e3-diagnostic-case-new
```

Los CSV originales conservan su hash. La mayor discrepancia al reconstruir RMSE/MAE frente a las tablas guardadas es 8,67×10⁻¹⁹. Son reagregaciones nuevas sobre resultados existentes; nuevos entrenamientos, forwards, predicciones de mercado, p-valores, remuestreos y réplicas Monte Carlo: cero. R3 sigue pendiente y este diagnóstico no cambia la calificación.
