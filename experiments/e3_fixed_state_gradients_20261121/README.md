# Medición CPU de pérdidas y gradientes en estados fijos

Fecha: 21 de noviembre de 2026, America/Bogota. Asistencia de IA declarada. **Estado actual: medición V2 completada.** Véanse [resultados e interpretación](RESULTS.md) y [contabilidad acumulada](results/CUMULATIVE_EXECUTION.json). La especificación siguiente conserva el diseño y sus límites; el plan/código previos están fijados en el commit `d3b34a4`. No se acredita R3.

El usuario pidió continuar tras el [diagnóstico de errores guardados](https://github.com/jassan-bit/DeepLearning_202630/tree/754237f8d73fd622f0b806a412954c5207b63b0f/experiments/e3_failure_diagnosis_20261121). Se ejecutará su medición delimitada en los quince checkpoints CPU atómicos existentes: V4 sigma-CRPS, U-CRPS y MSE-only, cinco semillas cada uno. No se entrena, reanuda o modifica ninguna red.

**Desviación técnica V1 preservada.** La ejecución del código anclado en `a40f498` se detuvo tras6forwards,182evaluaciones internas y12autograd por promoción de dtype de un escalar al calcular Adam hipotético. No hubo pasos de optimizador o cambios de archivos de pesos. V2 corrige sólo el producto de clipping para conservar el dtype nativo y añade un test mixto. Se conserva V1 y se vuelve a anclar V2 antes de una ejecución completa sobre los mismos estados/lotes. Los acumulados máximos son156forwards,4682evaluaciones internas y312autograd; las filas parciales V1 no se mezclan con V2.

## Diseño fijado

- Dos snapshots etiquetados por checkpoint: `best` para gradientes del predictor evaluado y `latest` para gradientes y desplazamientos hipotéticos con el Adam de esa misma época. No se deduplican; aunque cuatro pares sean iguales, se mantienen ambas etiquetas.
- Cinco lotes por snapshot: tres lotes fit de32ejemplos y dos lotes selection de27. Los índices y fechas están fijados; no se usan las 55 fechas de assessment para seleccionar o medir.
- Máximo y ejecución completa prevista: **150 forwards por lote,4500evaluaciones puntuales internas y300autograd**. No se guardan pronósticos por caso ni se generan pronósticos nuevos fuera de muestra. Se publican métricas agregadas de pérdidas, gradientes y ley atómica.
- Cero entrenamientos, `optimizer.step()`, cambios de pesos, Monte Carlo, bootstrap o p-valores. No se invocan `phase()`, `worker()` o `assess()`.
- CPU, dos hilos y uno inter-op; máximo 300 s y 2 GiB RSS. Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5, con campos de cohorte registrados iguales a los originales; dependencias auxiliares observadas se informan por separado, sin afirmar una instalación histórica binaria idéntica.

## Qué se mide

El gradiente g_A corresponde al auxiliar normalizado **sin lambda**. La receta usa g_M+lambda_actual·g_A, con lambda.1 en sigma/U-CRPS y0 en MSE-only. La geometría del CRPS evaluada sobre MSE-only se etiqueta contrafactual; su influencia auxiliar real es cero.

Se guardan normas, productos y cosenos por parámetros compartidos 1197, cabeza 65 y global 1262; MSE/CRPS separados y diagnósticos agregados de átomos/masas. Los cosenos con norma cero son ausentes, no valores fabricados con epsilon.

En `latest` se calculan dos desplazamientos analíticos hipotéticos desde los **mismos momentos antiguos**: receta completa y contrafactual MSE. La fórmula respeta máscaras `None`, contador por parámetro, clipping global, betas, epsilon y corrección de sesgo. Usa clones y no llama a `optimizer.step()` ni aplica el desplazamiento. No se promete igualdad bit a bit con un backward/paso histórico realizado sobre otro lote y RNG.

Un coseno negativo es conflicto geométrico local; g_M·d>0 indica ascenso infinitesimal de primer orden en ese lote/estado. Ninguno demuestra daño finito, causalidad del resultado del piloto, generalización o superioridad de una adaptación. Si no se observa conflicto en los estados medidos, eso no prueba ausencia universal.

## Guardas y evidencia previa

El [preflight](PREFLIGHT.json) cargó de forma restringida quince PT autenticados y verificó sus esquemas, parámetros, estados, fuentes y admisión; **no ejecutó forwards o gradientes**. Se usó `weights_only=True` y una allowlist NumPy estática después de verificar hashes. No existen `best_optimizer` o `best_rng`; nunca se mezclan con `latest`.

Un guard de acceso observa que sólo se deserializan `training_raw`, `training_y_original` y `training_dates`, con `allow_pickle=False`. El loader reconstruye el desarrollo histórico completo para verificar la admisión; sólo fit 532 y selection 54 entran en los lotes. No se cargan familias de arrays calibration/evaluation.

Los archivos privados de pesos y NPZ se leen y verifican, **no se redistribuyen**. Los hashes se comprueban antes y después. La salida de ejecución debe ser un directorio nuevo; los fallos conservan su recibo parcial y no disparan cambio de receta, lote, tolerancia o entrenamiento.

Las [diez pruebas de software](SOFTWARE_VALIDATION.json) verifican propiedades matemáticas y guardas sobre tensores sintéticos, no modelos de mercado. No son calibración estadística ni evidencia R3.

## Archivos

- [Plan congelado](PLAN.json).
- [Código](measure_fixed_states.py) y [pruebas](test_measurement_math.py).
- [Preflight](PREFLIGHT.json) y [revisión técnica](REVIEW_RECEIPT.json).

La autorización de este alcance proviene de la instrucción del usuario de continuar el trabajo. El archivo no habilita entrenamiento o campañas. La medición requiere además lectura pública real del commit de plan/código anterior a los gradientes. Ese anclaje registra desarrollo nuevo; no preregistra retrospectivamente los pilotos expuestos.

R3 permanece pendiente; V3 conserva NO-GO, V4 su abandono y MSE-only su papel de control. No se selecciona candidata, cambia la nota o modifica el servicio.
