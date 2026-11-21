# Continuación condicionada de SE-OI V5 hacia la evaluación R3

**Fecha:**21de noviembre de2026, America/Bogota. **Candidata:**`seoi_mse`. **Alcance presente:** código y piloto preparados; este archivo no registra entrenamientos ejecutados, resultados favorables o una autorización automática de campaña. Asistencia de IA declarada.

## 1. Primera decisión: aplicar la regla del piloto sin cambiarla

El piloto fija seis recetas por cinco semillas:30ajustes nuevos CPU y dos controles analíticos evaluados una vez. Tras completar y verificar los30ajustes, se abre el assessment de desarrollo ya expuesto. Se aplica `protocol.json`: ganancia de RMSE medio de al menos2% frente al mejor control —incluidos los analíticos—, al menos4/5victorias emparejadas frente al control neuronal con mejor RMSE medio y MAE de la candidata no mayor de1,05veces el MAE de ese mismo control neuronal. La receta primaria no se sustituye por la ablación ganadora.

Si cumple, la decisión es **estudiar continuidad de la misma V5**. Si falla, es incompleto o hay errores, se conserva el resultado y no se interpreta como selección de otra candidata. El2% pertenece a esta regla de desarrollo; no es un margen universal de la guía ni una prueba significativa.

## 2. Evidencia concreta que producirá la ejecución

| Archivo previsto por el controlador | Uso posterior y límite |
|---|---|
| `metrics.csv` y `recipe_summary.csv` | RMSE/MAE por receta y semilla, media y dispersión. La métrica primaria es media de RMSE individuales, no raíz de un MSE agrupado. |
| `predictions.csv` | Fecha, objetivo, receta, semilla y predicción: permitirá reconstruir diferencias y su dependencia temporal/cruzada. Se esperan1650filas neuronales y110analíticas;1760filas únicas no equivalen necesariamente a todos los forwards físicos si ocurrió una caída. |
| `jobs/.../curve.json`, `job.json`, `attempts.json` | Épocas, selección, parada, segmentos, interrupciones y evidencia de presupuesto; permiten auditar consistencia selección–assessment y fallos sin ocultar intentos. |
| `resources.csv`, `checkpoint_inventory.json` | Tiempo cobrado, parámetros, pasos, I/O y hashes de estados. Los bytes privados de checkpoints no se incluyen por defecto en la exportación pública. |
| `pilot_result.json`, `failures.json`, `fit_status.json` | Decisión literal, completitud y fallos; no se reemplazan por una narración favorable. |
| `plan.json`, `protocol.json`, `prepared.json`, `admission.json`, `cohort.json`, `source_identity.json`, `PUBLIC_SOURCE_ANCHOR.json`, `EXPORT_MANIFEST.json` | Identidad, procedencia, configuración, fechas reales y hashes. Un plan local o una publicación posterior no se convierten en un preregistro anterior a resultados. |

Estos archivos aportarán cinco pares por comparación neuronal del piloto y observaciones temporales correlacionadas. **No proporcionarán varianzas/covarianzas frente a toda la familia Top10/baselines ni frente a los tres conjuntos adicionales.** Para dimensionar esa familia hacen falta insumos de desarrollo compatibles que cubran sus contrastes, con población y unidades de independencia explícitas.

## 3. Preparar la confirmación sin convertir planificación en resultados

Antes de los experimentos definitivos debe fijarse un paquete concreto:

1. **Identidad y comparadores.** Mantener algoritmo, representación y regla de selección de la misma candidata; documentar su justificación R1 y comprensión/implementación R2. Enumerar todo el Top10 y los baselines fuertes, presupuestos y preprocesamiento comparables. No reutilizar los796trabajos aditivos como si validaran V5.
2. **Datos.** Admitir el panel principal y al menos tres adicionales de la misma tarea, diversos según la guía, elegidos sin conocer sus resultados; registrar fuentes, licencias, versiones, hashes y exposición. En cada panel separar desarrollo de una prueba reservada. No es obligatorio que las fechas sean futuras por calendario: sí debe ser auténtica la reserva y respetarse la causalidad/purga. Los55días ya examinados no recuperan esa condición.
3. **Batería.** Justificar al menos tres procedimientos distintos y pertinentes por contraste, incluidos un procedimiento global o control múltiple, los supuestos y la unidad de análisis. Declarar Holm para la familia y el nivel de un global cuando corresponda. No sustituir el blanco “media de RMSE por semilla” por MSE temporal sin explicarlo; un MCS global tampoco produce automáticamente una victoria direccional frente a cada rival.
4. **Dimensionamiento y coste.** Utilizar un piloto compatible para estimar variabilidad y dependencia frente a la familia; fijar efecto/margen práctico, longitud temporal y número de semillas. Exigir potencia≥0,90 por contraste y **reportar** la conjunta; no imponerle por defecto≥0,90. La evaluación final requiere al menos diez semillas o más si lo exige la potencia. Un presupuesto concreto se calcula entonces: los340ajustes de otro borrador no se heredan.
5. **Registro anterior a definitivos.** Publicar código, manifiesto, familia, paneles/particiones, métodos, márgenes, potencia, selección, presupuesto, persistencia y reglas de fallo con un commit/tag y fecha verificables; comprobar su lectura antes de ejecutar. Los cambios posteriores forman una versión nueva y conservan el plan anterior.

El desarrollo y la validación controlada de métodos estadísticos pueden avanzar sin que R1 haya sido aceptada o existan todos los paneles reales. Tampoco se exige conocer la potencia final antes del piloto que debe aportar sus insumos. Esas tareas metodológicas se distinguen de abrir la prueba reservada y ejecutar una campaña definitiva.

## 4. Qué deberá mostrar la confirmación

La guía§8.4 exige simultáneamente: media favorable frente a cada rival; tres procedimientos pertinentes con control familiar y diferencia superior al margen práctico; efecto no despreciable e IC95favorable; mejora media en al menos dos de tres conjuntos adicionales sin inferioridad significativa frente a ningún rival en ninguno; adecuación —baseline trivial, brecha, calibración/residuos y consistencia—; y robustez de la conclusión. Deben conservarse diferencias, intervalos, efectos, pcrudos/ajustados, potencia y resultados adversos completos.

La publicación del protocolo, diez semillas o una victoria del piloto no garantizan ese desenlace. Esta entrega deja **implementado el experimento que puede decidir el siguiente paso**, con una ruta de confirmación definida y sin atribuirle resultados que no se hayan medido.
